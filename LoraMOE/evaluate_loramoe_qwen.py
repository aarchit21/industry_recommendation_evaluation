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
IN_DIR = Path("datasets")  
OUT_DIR = Path("eval_output_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/chat" 
OLLAMA_MODEL = "qwen3:30b" # Change this to whatever model you are running

# -----------------------------
# Text Cleaning & Deterministic Parsing
# -----------------------------
PIPE_SEP_RE = re.compile(r"\s*\|\s*")
BULLET_RE = re.compile(r"^\s*([-*?]|(\d+[\).\]]))\s+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def clean_text(text: str) -> str:
    if pd.isna(text): return ""
    t = str(text).strip()
    t = re.sub(r'\s+', ' ', t)
    return t.replace('""', '"').replace('‑', '-').replace('–', '-')

def split_recommendations(text: str) -> List[str]:
    t = clean_text(text)
    if not t: return []
    if "|" in t: return [p.strip() for p in PIPE_SEP_RE.split(t) if p.strip()]
    return [ln.strip() for ln in t.splitlines() if ln.strip()]

def count_sentences(s: str) -> int:
    parts = [p for p in SENT_SPLIT_RE.split(s) if p.strip()]
    return max(1, len(parts)) if parts else 0

def list_structure_ok(text: str, n_items: int) -> bool:
    t = clean_text(text)
    if n_items < 2: return False
    if "|" in t: return True
    return any(BULLET_RE.match(ln) for ln in t.splitlines()) or len([ln for ln in t.splitlines() if ln.strip()]) >= n_items

def format_score_0_to_30(items: List[str], raw_text: str) -> int:
    n = len(items)
    a1 = 10 if 3 <= n <= 5 else 0
    a2 = 10 if (n >= 2 and list_structure_ok(raw_text, n)) else (5 if n >= 2 else 0)
    
    if n == 0: 
        a3 = 0
    else:
        frac_ok = sum(1 for it in items if count_sentences(it) in (1, 2)) / n
        a3 = 10 if frac_ok >= 0.8 else (5 if frac_ok >= 0.5 else 0)
        
    return a1 + a2 + a3

# -----------------------------
# EXACT ORIGINAL PROMPT
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
}}"""

# -----------------------------
# LLM Execution
# -----------------------------
def emergency_json_parse(text: str) -> dict:
    rubric = {}
    for key in ["relevance", "actionability", "concreteness", "feasibility"]:
        match = re.search(fr'"{key}"\s*:\s*"?(\d)"?', text, re.IGNORECASE)
        rubric[key] = int(match.group(1)) if match else 1 

    rat_match = re.search(r'"short_rationale"\s*:\s*"(.*?)"\s*\}?\s*$', text, re.IGNORECASE | re.DOTALL)
    rationale = rat_match.group(1) if rat_match else "[Recovered via Regex]"
    
    return {"rubric_scores_1_to_5": rubric, "short_rationale": rationale}

def evaluate_row_ollama(theme: str, issue: str, model_output: str) -> dict:
    raw_prompt = SINGLE_JUDGE_TEMPLATE.format(theme=theme, issue=issue, model_output=model_output)
    
    # Split prompt into System and User
    system_content = ""
    user_content = raw_prompt
    if "SYSTEM:" in raw_prompt and "USER:" in raw_prompt:
        parts = raw_prompt.split("USER:")
        system_content = parts[0].replace("SYSTEM:", "").strip()
        user_content = parts[1].strip()

    # THE NUDGE: Force Qwen to start writing JSON immediately
    user_content += "\n\nOutput your JSON below:\n```json\n"

    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True  # <--- THE FIX: Streaming prevents silent OOM crashes on split models
    }
    
    # Use standard JSON mode for other models
    if "qwen" not in OLLAMA_MODEL.lower():
        payload["format"] = "json"

    last_content = ""
    for attempt in range(2): 
        try:
            # Note: stream=True in the requests call as well
            resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=300)
            
            if resp.status_code == 200:
                last_content = ""
                # Collect the streamed tokens one by one
                for line in resp.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        data = json.loads(decoded_line)
                        if "message" in data and "content" in data["message"]:
                            last_content += data["message"]["content"]
                            
                last_content = last_content.strip()
                
                if not last_content:
                    print(f"      [Attempt {attempt+1}] Model returned empty string. Retrying...")
                    continue

                # Clean up the output and extract the JSON
                match = re.search(r'\{.*\}', last_content, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except json.JSONDecodeError:
                        print(f"      [Attempt {attempt+1}] Minor JSON formatting error. Retrying...")
                else:
                    print(f"      [Attempt {attempt+1}] No JSON brackets found. Retrying...")
            else:
                print(f"      [Server Error] {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"      [Attempt {attempt+1} Failed] Connection or stream dropped.")

    # If it failed strict parsing but we got text, force extract the numbers
    if last_content:
        print(f"      [Failsafe Triggered] Extracting via regex...")
        return emergency_json_parse(last_content)

    return None

def safe_int(val, default=1) -> int:
    try: return max(1, min(5, int(val)))
    except: return default

# -----------------------------
# Main Execution Loop
# -----------------------------
def run_evaluation():
    if not IN_DIR.exists():
        print(f"Directory {IN_DIR} not found.")
        return

    for input_file in IN_DIR.glob("*.csv"):
        out_csv = OUT_DIR / f"evaluated_{input_file.name}"
        
        if out_csv.exists() and os.path.getsize(out_csv) > 0:
            print(f"Skipping {input_file.name} (Already done).")
            continue

        df = pd.read_csv(input_file)
        print(f"\nProcessing {len(df)} rows from {input_file.name}")
        
        cols_lower = {c.lower(): c for c in df.columns}
        issue_col = cols_lower.get("issue")
        theme_col = cols_lower.get("theme")
        fixes_col = cols_lower.get("fixes", cols_lower.get("fix"))
        
        if not (issue_col and theme_col and fixes_col):
            print(f"Skipping {input_file.name}: Missing required columns.")
            continue

        for idx, row in df.iterrows():
            print(f"[{input_file.stem}] Row {idx + 1}/{len(df)}...")
            
            issue = clean_text(row.get(issue_col, ""))
            theme = clean_text(row.get(theme_col, ""))
            fixes_text = clean_text(row.get(fixes_col, ""))
            
            items = split_recommendations(fixes_text)
            fmt_score = format_score_0_to_30(items, fixes_text)
            
            llm_result = evaluate_row_ollama(theme, issue, fixes_text)
            row_res = row.to_dict()
            
            if llm_result:
                rubric = llm_result.get("rubric_scores_1_to_5", llm_result)
                # Fallback if model puts the scores at the root instead of inside 'rubric_scores_1_to_5'
                if not isinstance(rubric, dict):
                    rubric = llm_result
                
                rel = safe_int(rubric.get("relevance", 1))
                act = safe_int(rubric.get("actionability", 1))
                con = safe_int(rubric.get("concreteness", 1))
                fea = safe_int(rubric.get("feasibility", 1))
                
                content_score = round((((rel + act + con + fea) / 4 - 1) / 4) * 70)
                
                row_res.update({
                    "format_score_0_to_30": fmt_score,
                    "relevance": rel, "actionability": act, "concreteness": con, "feasibility": fea,
                    "content_score_0_to_70": content_score,
                    "afq_score_0_to_100": fmt_score + content_score,
                    "rationale": llm_result.get("short_rationale", "")
                })
            else:
                row_res.update({
                    "format_score_0_to_30": fmt_score,
                    "relevance": None, "actionability": None, "concreteness": None, "feasibility": None,
                    "content_score_0_to_70": None, "afq_score_0_to_100": None, "rationale": "LLM failed"
                })
            
            pd.DataFrame([row_res]).to_csv(out_csv, mode='a', header=not out_csv.exists(), index=False)

        print(f"✅ Saved to {out_csv}")

if __name__ == "__main__":
    run_evaluation()