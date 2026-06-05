import os
import re
import json
import requests
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
IN_DIR = Path("datasets")  # Put ALL your CSVs (baselines, outputs, etc.) in here
OUT_DIR = Path("eval_output_loramoe")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:27b-it-qat"

# -----------------------------
# Deterministic parsing & cleaning utilities
# -----------------------------
BULLET_RE = re.compile(r"^\s*([-*?]|(\d+[\).\]]))\s+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
PIPE_SEP_RE = re.compile(r"\s*\|\s*")

def normalize_text(x: str) -> str:
    return "" if pd.isna(x) else str(x).strip()

def clean_text(text: str) -> str:
    """Cleans spaces, quotes, and converts non-breaking hyphens to standard hyphens."""
    t = normalize_text(text)
    t = re.sub(r'\s+', ' ', t)
    t = t.replace('""', '"')
    t = t.replace('‑', '-').replace('–', '-') 
    return t.strip()

def split_recommendations(text: str) -> List[str]:
    t = clean_text(text)
    if not t:
        return []

    if "|" in t:
        parts = [p.strip() for p in PIPE_SEP_RE.split(t) if p.strip()]
        if len(parts) >= 2:
            return parts

    lines = [ln.rstrip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return []

    items = []
    current = []
    saw_bullet = False

    for ln in lines:
        if BULLET_RE.match(ln):
            saw_bullet = True
            if current:
                items.append(" ".join(current).strip())
                current = []
            ln2 = BULLET_RE.sub("", ln).strip()
            current.append(ln2)
        else:
            if saw_bullet:
                current.append(ln.strip())
            else:
                if current:
                    items.append(" ".join(current).strip())
                current = [ln.strip()]

    if current:
        items.append(" ".join(current).strip())

    return [it for it in items if it]

def count_sentences(s: str) -> int:
    s = normalize_text(s)
    parts = [p for p in SENT_SPLIT_RE.split(s) if p.strip()]
    return max(1, len(parts)) if parts else 0

def list_structure_ok(text: str, n_items: int) -> bool:
    t = clean_text(text)
    if n_items < 2:
        return False
    if "|" in t:
        return True
    return any(BULLET_RE.match(ln) for ln in t.splitlines()) or len([ln for ln in t.splitlines() if ln.strip()]) >= n_items

def format_score_0_to_30(items: List[str], raw_text: str) -> Tuple[int, Dict]:
    n = len(items)
    a1 = 10 if 3 <= n <= 5 else 0

    if n >= 2 and list_structure_ok(raw_text, n):
        a2 = 10
    elif n >= 2:
        a2 = 5
    else:
        a2 = 0

    if n == 0:
        a3 = 0
    else:
        frac_ok = sum(1 for it in items if count_sentences(it) in (1, 2)) / n
        if frac_ok >= 0.8:
            a3 = 10
        elif frac_ok >= 0.5:
            a3 = 5
        else:
            a3 = 0

    score = a1 + a2 + a3
    return score, {}

# -----------------------------
# LLM Judge setup (Ruthless COO Persona)
# -----------------------------
SINGLE_JUDGE_TEMPLATE = """SYSTEM:
You are a ruthless, budget-conscious Chief Operating Officer evaluating customer-support remediation recommendations.
You do not give out high scores easily. You actively look for flaws in logic, cost, and effort.
Judge only what is written. Do not assume missing details.

USER:
Evaluate the candidate response against the task requirements.

Task requirements:
- Provide 3-5 specific and actionable recommendations.
- Each recommendation must reference a concrete tool, process, system, or measurable action.
- Avoid generic advice; specify how.
- Each recommendation should be 1-2 sentences.

When scoring, use this EXPLICIT RUBRIC. Do NOT default to a score of 4. You must justify every 4 or 5.

1. RELEVANCE (1-5):
- 5: Perfectly targets the root cause of the specific customer issue and theme.
- 3: Addresses the theme generally, but misses the specific nuance of the customer's exact issue.
- 1: Completely off-topic or misaligned.

2. ACTIONABILITY (1-5):
- 5: Crystal clear, step-by-step. A frontline worker could execute this immediately without asking questions.
- 4: Clear steps, but requires minor interpretation.
- 3: Tells the user *what* to do, but fails to explain *how* to do it.
- 2: Mostly generic buzzwords (e.g., "improve communication", "be proactive").
- 1: Utterly unactionable.

3. CONCRETENESS (1-5):
- 5: Explicitly names real-world tools (e.g., Jira, Zendesk), specific KPIs, or exact metrics.
- 3: Mentions processes but uses vague placeholders instead of specific tools/metrics.
- 1: Entirely conceptual with zero real-world grounding.

4. FEASIBILITY (1-5) - *BE STRICT HERE*:
- 5: Trivial to implement. Costs zero dollars, uses existing tools, can be rolled out today.
- 4: Low effort. Requires minor configuration changes or small training tweaks.
- 3: Moderate effort. Requires cross-department coordination, moderate budget, or new software integration.
- 2: High effort. Very expensive, requires massive structural changes, or months of work.
- 1: Impossible, unsafe, or financially ruinous.

INPUT CONTEXT:
Theme: {theme}
Customer issue: {issue}

CANDIDATE RESPONSE:
{model_output}

Return ONLY valid JSON in this schema:
{{
  "format_checks": {{
    "num_recommendations": int,
    "count_ok_3_to_5": bool,
    "list_structure_ok": bool,
    "mostly_1_to_2_sentences": bool,
    "format_score_0_to_30": int
  }},
  "rubric_scores_1_to_5": {{
    "relevance": int,
    "actionability": int,
    "concreteness": int,
    "feasibility": int
  }},
  "content_score_0_to_70": int,
  "afq_score_0_to_100": int,
  "short_rationale": "1-3 sentences, mention why you deducted points based on the strict rubric."
}}
"""

def emergency_json_parse(text: str) -> dict:
    """Failsafe: Forcefully extracts scores from broken JSON text using regex."""
    rubric = {}
    for key in ["relevance", "actionability", "concreteness", "feasibility"]:
        match = re.search(fr'"{key}"\s*:\s*"?(\d)"?', text, re.IGNORECASE)
        rubric[key] = int(match.group(1)) if match else 1 

    rat_match = re.search(r'"short_rationale"\s*:\s*"(.*?)"\s*\}?\s*$', text, re.IGNORECASE | re.DOTALL)
    rationale = rat_match.group(1) if rat_match else "[Recovered via Regex due to broken JSON]"

    return {
        "rubric_scores_1_to_5": rubric,
        "short_rationale": rationale
    }

def evaluate_row_ollama(theme: str, issue: str, model_output: str, max_retries: int = 2) -> dict:
    prompt = SINGLE_JUDGE_TEMPLATE.format(theme=theme, issue=issue, model_output=model_output)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "num_ctx": 4096  # Limit context window to save VRAM
        }
    }
    
    last_error = None
    last_data = ""
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
            
            if resp.status_code != 200:
                print(f"      [Server Error] {resp.status_code}: {resp.text}")
                continue
                
            last_data = resp.json().get("response", "")
            match = re.search(r'\{.*\}', last_data, re.DOTALL)
            
            if match:
                json_str = match.group(0)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    last_error = e
                    print(f"      [Attempt {attempt+1}] Broken JSON detected. Retrying...")
                    
        except Exception as e:
            print(f"      [Attempt {attempt+1}] Connection failed: {e}")
            last_error = e

    if last_data:
        print(f"      [Failsafe Triggered] Force-extracting data via regex...")
        return emergency_json_parse(last_data)
        
    print(f"      [Total Failure] Could not evaluate row. Last Error: {last_error}")
    return None

def safe_extract_score(val, default=1) -> int:
    try:
        if isinstance(val, (int, float)):
            return max(1, min(5, int(val)))
        match = re.search(r'\d+', str(val))
        if match:
            return max(1, min(5, int(match.group(0))))
        return default
    except:
        return default

# -----------------------------
# Main Execution
# -----------------------------
def run_evaluation():
    if not IN_DIR.exists():
        print(f"Directory {IN_DIR} not found. Please create it and add your CSVs.")
        return

    # Automatically finds EVERY .csv file in the datasets/ folder
    csv_files = list(IN_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {IN_DIR}.")
        return

    for input_file in csv_files:
        out_csv = OUT_DIR / f"evaluated_{input_file.name}"
        
        # If you are restarting, this skips files that are already completed
        if out_csv.exists() and os.path.getsize(out_csv) > 0:
            print(f"Skipping {input_file.name}: Output already exists.")
            continue

        try:
            df = pd.read_csv(input_file)
        except Exception as e:
            print(f"Failed to read {input_file.name}: {e}")
            continue
            
        print(f"\nLoaded {len(df)} rows from {input_file.name}")
        
        # Dynamically map columns (handles lowercase, uppercase, and alternative headers)
        cols_lower = {c.lower(): c for c in df.columns}
        
        issue_col = cols_lower.get("issue")
        theme_col = cols_lower.get("theme")
        fixes_col = cols_lower.get("fixes", cols_lower.get("fix"))
        review_col = cols_lower.get("review", cols_lower.get("review_text"))
        
        if not (issue_col and theme_col and fixes_col):
            print(f"Skipping {input_file.name}: Missing base columns. Found: {list(df.columns)}")
            continue

        for idx, row in df.iterrows():
            print(f"[{input_file.stem}] Evaluating row {idx + 1}/{len(df)}...")
            
            # Text cleaning applied directly to dynamic columns
            review = clean_text(row.get(review_col, "")) if review_col else ""
            issue = clean_text(row.get(issue_col, ""))
            theme = clean_text(row.get(theme_col, ""))
            fixes_text = clean_text(row.get(fixes_col, ""))
            
            items = split_recommendations(fixes_text)
            fmt_score_det, _ = format_score_0_to_30(items, fixes_text)
            
            llm_result = evaluate_row_ollama(theme=theme, issue=issue, model_output=fixes_text)
            
            row_res = row.to_dict()
            if llm_result and isinstance(llm_result, dict):
                rubric = llm_result.get("rubric_scores_1_to_5", llm_result)
                if not isinstance(rubric, dict):
                    rubric = {}
                
                relevance = safe_extract_score(rubric.get("relevance", rubric.get("Relevance", 1)))
                actionability = safe_extract_score(rubric.get("actionability", rubric.get("Actionability", 1)))
                concreteness = safe_extract_score(rubric.get("concreteness", rubric.get("Concreteness", 1)))
                feas_val = rubric.get("feasibility", rubric.get("feasability", rubric.get("Feasibility", 1)))
                feasibility = safe_extract_score(feas_val)
                
                content_score = round((((relevance + actionability + concreteness + feasibility) / 4 - 1) / 4) * 70)
                afq_score = fmt_score_det + content_score
                rationale = llm_result.get("short_rationale", "")
                
                row_res.update({
                    "format_score_0_to_30": fmt_score_det,
                    "relevance": relevance,
                    "actionability": actionability,
                    "concreteness": concreteness,
                    "feasibility": feasibility,
                    "content_score_0_to_70": content_score,
                    "afq_score_0_to_100": afq_score,
                    "rationale": rationale
                })
            else:
                row_res.update({
                    "format_score_0_to_30": fmt_score_det,
                    "relevance": None, "actionability": None, "concreteness": None, "feasibility": None,
                    "content_score_0_to_70": None, "afq_score_0_to_100": None, "rationale": "LLM failed"
                })
            
            # Incremental Save
            single_row_df = pd.DataFrame([row_res])
            single_row_df.to_csv(out_csv, mode='a', header=not out_csv.exists(), index=False)

        print(f"Evaluation complete for {input_file.name}. Fully saved to {out_csv}\n")

if __name__ == "__main__":
    run_evaluation()