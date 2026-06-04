import os
import re
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
CSV_PATH = "New_model_restaurant - Sheet1.csv"
OUT_DIR = Path("eval_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "evaluated_restaurant.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:27b-it-qat"  # Room for changing models in Ollama

# -----------------------------
# Deterministic parsing utilities
# -----------------------------
BULLET_RE = re.compile(r"^\s*([-*?]|(\d+[\).\]]))\s+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
PIPE_SEP_RE = re.compile(r"\s*\|\s*")

def normalize_text(x: str) -> str:
    return "" if pd.isna(x) else str(x).strip()

def clean_text(text: str) -> str:
    """Basic cleaning to remove multiple spaces and normalize quotes before LLM pass."""
    t = normalize_text(text)
    t = re.sub(r'\s+', ' ', t)
    t = t.replace('""', '"')
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

def evaluate_row_ollama(theme: str, issue: str, model_output: str) -> dict:
    prompt = SINGLE_JUDGE_TEMPLATE.format(theme=theme, issue=issue, model_output=model_output)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json().get("response", "")
        # Extract JSON using regex in case of trailing characters
        match = re.search(r'\{.*\}', data, re.DOTALL)
        if match:
            data = match.group(0)
        return json.loads(data)
    except Exception as e:
        print(f"Ollama call failed: {e}")
        return None

# -----------------------------
# Main Execution
# -----------------------------
def run_evaluation():
    if not Path(CSV_PATH).exists():
        raise FileNotFoundError(f"Missing input CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    
    required_cols = {"Review", "Issue", "Theme", "Fixes"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    results = []
    
    for idx, row in df.iterrows():
        print(f"Evaluating row {idx + 1}/{len(df)}...")
        
        # 1. Clean inputs
        review = clean_text(row.get("Review", ""))
        issue = clean_text(row.get("Issue", ""))
        theme = clean_text(row.get("Theme", ""))
        fixes_text = clean_text(row.get("Fixes", ""))
        
        # 2. Deterministic Format Score
        items = split_recommendations(fixes_text)
        fmt_score_det, _ = format_score_0_to_30(items, fixes_text)
        
        # 3. Call Ollama Judge
        llm_result = evaluate_row_ollama(theme=theme, issue=issue, model_output=fixes_text)
        
        row_res = row.to_dict()
        if llm_result:
            rubric = llm_result.get("rubric_scores_1_to_5", {})
            relevance = rubric.get("relevance", 1)
            actionability = rubric.get("actionability", 1)
            concreteness = rubric.get("concreteness", 1)
            feasibility = rubric.get("feasibility", 1)
            
            # Recompute content explicitly to ensure correct math overrides LLM hallucination
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
            
        results.append(row_res)

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Evaluation complete. Saved to {OUT_CSV}")

if __name__ == "__main__":
    run_evaluation()