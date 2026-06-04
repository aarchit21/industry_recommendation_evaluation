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

EVAL_DIR = Path("eval_output")
PLOTS_DIR = Path("plot_output")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def run_plots():
    if not EVAL_DIR.exists():
        raise FileNotFoundError(f"Missing directory {EVAL_DIR}. Run evaluation.py first.")
        
    csv_files = list(EVAL_DIR.glob("evaluated_*.csv"))
    if not csv_files:
        print("No evaluated CSV files found to plot.")
        return

    # Load and combine all datasets
    dfs = []
    for file in csv_files:
        temp_df = pd.read_csv(file)
        # Extract base name to label the datasets (e.g. "Hospitality" or "Airline")
        dataset_name = file.stem.replace("evaluated_", "").replace("_inference_results", "").replace("New_model_", "").capitalize()
        temp_df["Dataset"] = dataset_name
        dfs.append(temp_df)
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Drop rows where LLM failed
    df_valid = df.dropna(subset=["afq_score_0_to_100"]).copy()
    if df_valid.empty:
        print("No valid evaluation scores found to plot.")
        return
        
    # 1. Overall AFQ Score Distribution Comparison
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_valid, x="afq_score_0_to_100", hue="Dataset", fill=True, common_norm=False, alpha=0.5)
    plt.title("Distribution of Overall AFQ Scores by Dataset")
    plt.xlabel("AFQ Score (0-100)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_afq_distribution_comparison.png")
    plt.close()
    
    # 2. Format vs Content Scatter Comparison
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_valid, 
        x="format_score_0_to_30", 
        y="content_score_0_to_70",
        hue="Dataset",
        alpha=0.7, s=100
    )
    plt.title("Format vs Content Score by Dataset")
    plt.xlabel("Format Score (0-30)")
    plt.ylabel("Content Score (0-70)")
    plt.axvline(30, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(70, color='gray', linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_format_vs_content_comparison.png")
    plt.close()
    
    # 3. Dimension Boxplots Comparison
    dimensions = ["relevance", "actionability", "concreteness", "feasibility"]
    melted = df_valid.melt(id_vars=["Dataset"], value_vars=dimensions, var_name="Dimension", value_name="Score (1-5)")
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x="Dimension", y="Score (1-5)", hue="Dataset", palette="Set2")
    plt.title("Scores by Dimension across Datasets")
    plt.ylim(0.5, 5.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_dimension_boxplots_comparison.png")
    plt.close()
    
    print(f"Plots successfully generated and saved to '{PLOTS_DIR}'")

if __name__ == "__main__":
    run_plots()