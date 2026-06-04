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
# Add or remove your CSV files here
CSV_PATHS = [
    "New_model_restaurant - Sheet1.csv",
    "hospitality_inference_results.csv",
    "Airline_inference_results.csv"
]
'''
CSV_PATHS = [
    "hospitality_inference_results.csv",
    "Airline_inference_results.csv"
]'''
OUT_DIR = Path("eval_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "tensortemplar/prometheus2:7b-fp16"

# -----------------------------
# Deterministic parsing utilities
# -----------------------------
BULLET_RE = re.compile(r"^\s*([-*?]|(\d+[\).\]]))\s+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
PIPE_SEP_RE = re.compile(r"\s*\|\s*")

def normalize_text(x: str) -> str:
    return "" if pd.isna(x) else str(x).strip()

def clean_text(text: str) -> str:
    t = normalize_text(text)
    t = re.sub(r'\s+', ' ', t)
    t = t.replace('""', '"')
    # Replace non-breaking hyphens and en-dashes with standard hyphens
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
# LLM Judge setup
# -----------------------------
SINGLE_JUDGE_TEMPLATE = """SYSTEM:
You are a strict evaluator of customer-support remediation recommendations.
Judge only what is written. Do not assume missing details.
Be consistent across items. Do not reward verbosity or buzzwords.

USER:
Evaluate the candidate response against the task requirements.

Task requirements:
- Provide 3-5 specific and actionable recommendations.
- Each recommendation must reference a concrete tool, process, system, or measurable action.
- Avoid generic advice; specify how.
- Each recommendation should be 1-2 sentences.
- Output should be a bulleted or numbered list (or clearly separated list items).

When scoring, reward high-quality responses that show:
- cross-functional coverage where relevant (e.g., operations + QA + monitoring + escalation)
- coherent sequencing of actions (detect -> decide -> act -> verify)
- measurable control loops (thresholds, alerts, audits, acceptance checks)
- integration across systems/processes instead of isolated tips
- realistic tradeoff handling and scoped rollout

Dimension guidance:
- Relevance: prioritize root-cause targeting and multi-driver alignment to the stated issue/theme.
- Actionability: prioritize implementable sequencing, clear operational steps, and verification.
- Concreteness: prioritize artifact-linked mechanisms (metrics, SOPs, alerts, tests, dashboards, runbooks).
- Feasibility: prioritize practical effort-impact balance, realistic dependencies, and safe rollout.

Penalty guidance:
- Penalize generic filler, repeated paraphrases, and style-only sophistication.
- If more than half of recommendations are generic, cap Actionability <= 2 and Concreteness <= 2.
- If any recommendation is egregiously unsafe/unrealistic/impossible, cap Feasibility <= 2.

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
  "short_rationale": "1-3 sentences, mention the biggest weaknesses"
}}

Scoring rules:
- content_score_0_to_70 = round((((relevance+actionability+concreteness+feasibility)/4 - 1) / 4) * 70)
- afq_score_0_to_100 = format_score_0_to_30 + content_score_0_to_70
"""

def emergency_json_parse(text: str) -> dict:
    """Failsafe: Forcefully extracts scores from broken JSON text using regex."""
    rubric = {}
    # Hunt down the specific numbers assigned to each dimension
    for key in ["relevance", "actionability", "concreteness", "feasibility"]:
        match = re.search(fr'"{key}"\s*:\s*"?(\d)"?', text, re.IGNORECASE)
        rubric[key] = int(match.group(1)) if match else 1  # Fallback to 1 if completely missing

    # Try to extract the rationale text, stopping at the end of the JSON
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
            "num_ctx": 4096  # Keeps VRAM usage low
        }
    }
    
    last_error = None
    last_data = ""
    
    # Layer 1: Auto-Retry Loop
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
                    # Try standard strict parsing
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    last_error = e
                    print(f"      [Attempt {attempt+1}] Broken JSON detected. Retrying...")
                    
        except Exception as e:
            print(f"      [Attempt {attempt+1}] Connection failed: {e}")
            last_error = e

    # Layer 2: Regex Recovery (If we exhausted our retries and still have broken text)
    if last_data:
        print(f"      [Failsafe Triggered] Force-extracting data via regex...")
        return emergency_json_parse(last_data)
        
    print(f"      [Total Failure] Could not evaluate row. Last Error: {last_error}")
    return None
# -----------------------------
# Main Execution
# -----------------------------
def run_evaluation():
    for csv_path in CSV_PATHS:
        input_file = Path(csv_path)
        if not input_file.exists():
            print(f"Skipping {csv_path}: File not found.")
            continue
            
        out_csv = OUT_DIR / f"evaluated_{input_file.name}"

        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} rows from {input_file.name}")
        
        fixes_col = "Fixes" if "Fixes" in df.columns else "Fix"
        review_col = "Review" if "Review" in df.columns else None
        
        required_cols = {"Issue", "Theme", fixes_col}
        if not required_cols.issubset(set(df.columns)):
            print(f"Skipping {input_file.name}: Missing base columns.")
            continue

        # If we already started this file and stopped, don't overwrite (optional safety)
        # We will write headers only if the file doesn't exist yet
        write_mode = 'a' if out_csv.exists() else 'w'
        
        results = []
        for idx, row in df.iterrows():
            print(f"[{input_file.stem}] Evaluating row {idx + 1}/{len(df)}...")
            
            review = clean_text(row.get(review_col, "")) if review_col else ""
            issue = clean_text(row.get("Issue", ""))
            theme = clean_text(row.get("Theme", ""))
            fixes_text = clean_text(row.get(fixes_col, ""))
            
            items = split_recommendations(fixes_text)
            fmt_score_det, _ = format_score_0_to_30(items, fixes_text)
            
            llm_result = evaluate_row_ollama(theme=theme, issue=issue, model_output=fixes_text)
            
            row_res = row.to_dict()
            if llm_result:
                rubric = llm_result.get("rubric_scores_1_to_5", {})
                relevance = rubric.get("relevance", 1)
                actionability = rubric.get("actionability", 1)
                concreteness = rubric.get("concreteness", 1)
                feasibility = rubric.get("feasibility", 1)
                
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
            
            # --- INCREMENTAL SAVE ---
            # Create a dataframe of just this one row and append it to the CSV immediately
            single_row_df = pd.DataFrame([row_res])
            single_row_df.to_csv(out_csv, mode='a', header=not out_csv.exists(), index=False)

        print(f"Evaluation complete for {input_file.name}. Fully saved to {out_csv}\n")

if __name__ == "__main__":
    run_evaluation()