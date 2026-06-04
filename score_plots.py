import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Configuration
# -----------------------------
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300

INPUT_CSV = Path("eval_output/evaluated_restaurant.csv")
PLOTS_DIR = Path("plot_output")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def run_plots():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV {INPUT_CSV}. Run evaluation.py first.")
        
    df = pd.read_csv(INPUT_CSV)
    
    # Drop rows where LLM failed to return valid JSON
    df_valid = df.dropna(subset=["afq_score_0_to_100"]).copy()
    if df_valid.empty:
        print("No valid evaluation scores found to plot.")
        return
        
    # 1. Overall AFQ Score Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df_valid["afq_score_0_to_100"], bins=10, kde=True, color="blue")
    plt.title("Distribution of Overall AFQ Scores")
    plt.xlabel("AFQ Score (0-100)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_afq_distribution.png")
    plt.close()
    
    # 2. Format vs Content Scatter
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_valid, 
        x="format_score_0_to_30", 
        y="content_score_0_to_70",
        alpha=0.7, s=100
    )
    plt.title("Format vs Content Score")
    plt.xlabel("Format Score (0-30)")
    plt.ylabel("Content Score (0-70)")
    # Add heuristic reference bounds
    plt.axvline(30, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(70, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_format_vs_content.png")
    plt.close()
    
    # 3. Dimension Boxplots
    dimensions = ["relevance", "actionability", "concreteness", "feasibility"]
    melted = df_valid.melt(value_vars=dimensions, var_name="Dimension", value_name="Score (1-5)")
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=melted, x="Dimension", y="Score (1-5)", palette="Set2")
    sns.stripplot(data=melted, x="Dimension", y="Score (1-5)", color=".25", alpha=0.6, jitter=True)
    plt.title("Scores by Dimension")
    plt.ylim(0.5, 5.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_dimension_boxplots.png")
    plt.close()
    
    print(f"Plots successfully generated and saved to '{PLOTS_DIR}'")

if __name__ == "__main__":
    run_plots()