import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Configuration & Theme
# -----------------------------
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300

MODEL_DIRECTORIES = {
    "eval_output_gemma": "Gemma",
    "eval_output_prometheus": "Prometheus"
}

PASS_THRESHOLD = 80

PLOTS_DIR = Path("plot_output/comprehensive_comparison")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def get_clean_dataset_name(filename: str) -> str:
    lower = filename.lower()
    if "restaurant" in lower: return "Restaurant"
    if "airline" in lower: return "Airline"
    if "hospitality" in lower: return "Hospitality"
    return "Unknown"

def run_comprehensive_plots():
    print("🔍 Scanning folders for evaluated CSVs...")
    dfs = []
    
    for folder, model_label in MODEL_DIRECTORIES.items():
        dir_path = Path(folder)
        if not dir_path.exists(): continue
        for file in dir_path.glob("evaluated_*.csv"):
            temp_df = pd.read_csv(file)
            temp_df["Dataset"] = get_clean_dataset_name(file.name)
            temp_df["Model"] = model_label
            dfs.append(temp_df)

    if not dfs:
        print("❌ No data loaded. Check folders.")
        return

    df_valid = pd.concat(dfs, ignore_index=True).dropna(subset=["afq_score_0_to_100"]).copy()
    dimensions = ["relevance", "actionability", "concreteness", "feasibility"]
    for dim in dimensions:
        df_valid[dim] = pd.to_numeric(df_valid[dim], errors="coerce")

    print(f"✅ Loaded {len(df_valid)} valid rows. Generating 7 comprehensive plots...")
    model_palette = sns.color_palette("Set1", n_colors=df_valid["Model"].nunique())

    # ==========================================
    # Plot 1: Mean AFQ by Model
    # ==========================================
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_valid, x="Dataset", y="afq_score_0_to_100", hue="Model", palette=model_palette, edgecolor="black")
    plt.title("1. Mean Overall AFQ Score by Model")
    plt.ylim(0, 100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_model_mean_afq.png")
    plt.close()

    # ==========================================
    # Plot 2: Industry vs Model Heatmap
    # ==========================================
    pivot_afq = df_valid.pivot_table(index="Dataset", columns="Model", values="afq_score_0_to_100", aggfunc="mean")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot_afq, annot=True, fmt=".1f", cmap="Blues", cbar_kws={'label': 'Mean AFQ Score'}, linewidths=.5)
    plt.title("2. Heatmap: Mean AFQ by Industry & Model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_industry_model_heatmap.png")
    plt.close()

    # ==========================================
    # Plot 3: AFQ Distribution (Violin + Boxplot)
    # ==========================================
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df_valid, x="Dataset", y="afq_score_0_to_100", hue="Model", 
                   split=True, inner=None, palette=model_palette, alpha=0.5)
    sns.boxplot(data=df_valid, x="Dataset", y="afq_score_0_to_100", hue="Model",
                width=0.3, showfliers=False, boxprops={'facecolor':'none', 'zorder':10})
    plt.title("3. AFQ Score Distribution (Violin & Box)")
    plt.ylim(0, 110)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_afq_distribution.png")
    plt.close()

    # ==========================================
    # Plot 4: Dimension Boxplots (Overall Spread)
    # ==========================================
    melted = df_valid.melt(id_vars=["Model", "Dataset"], value_vars=dimensions, var_name="Dimension", value_name="Score")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x="Dimension", y="Score", hue="Model", palette=model_palette)
    plt.title("4. Overall Spread of Qualitative Dimensions")
    plt.ylim(0.5, 5.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_dimension_boxplots.png")
    plt.close()

    # ==========================================
    # Plot 5: Format vs Content Regression
    # ==========================================
    g = sns.lmplot(data=df_valid, x="format_score_0_to_30", y="content_score_0_to_70", 
                   hue="Model", col="Dataset", palette=model_palette, height=5, aspect=1, scatter_kws={"alpha":0.6})
    g.fig.suptitle("5. Format vs Content Trendline", y=1.05)
    for ax in g.axes.flat:
        ax.set_xlim(-2, 32); ax.set_ylim(-5, 75)
        ax.axvline(30, color='gray', linestyle='--', alpha=0.3)
        ax.axhline(70, color='gray', linestyle='--', alpha=0.3)
    plt.savefig(PLOTS_DIR / "05_format_vs_content.png")
    plt.close()

    # ==========================================
    # Plot 6: NEW - Dimension Scores Broken Down by Dataset
    # ==========================================
    # Calculate the exact mean for each dimension, separated by dataset and model
    dim_means = melted.groupby(["Dataset", "Model", "Dimension"])["Score"].mean().reset_index()

    g = sns.catplot(
        data=dim_means, x="Dimension", y="Score", hue="Model", col="Dataset",
        kind="bar", palette=model_palette, edgecolor="black", height=5, aspect=1.2
    )
    g.fig.suptitle("6. Average Dimension Scores by Dataset & Model", y=1.05)
    g.set_axis_labels("", "Mean Score (1-5)")
    g.set(ylim=(0, 5.8)) # Extra room for the text labels
    
    # Add values on top of the bars and rotate text for readability
    for ax in g.axes.flat:
        ax.tick_params(axis='x', rotation=45)
        for p in ax.patches:
            h = p.get_height()
            if not np.isnan(h) and h > 0:
                ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h),
                            ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', fontsize=10)

    # Use bbox_inches to ensure rotated text isn't cut off when saving
    plt.savefig(PLOTS_DIR / "06_dimension_scores_by_dataset.png", bbox_inches='tight')
    plt.close()

    # ==========================================
    # Plot 7: Pass Rates
    # ==========================================
    df_valid["Passed"] = (df_valid["afq_score_0_to_100"] >= PASS_THRESHOLD).astype(int)
    pass_rates = df_valid.groupby(["Dataset", "Model"])["Passed"].mean().reset_index()
    pass_rates["Pass_Pct"] = pass_rates["Passed"] * 100

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=pass_rates, x="Dataset", y="Pass_Pct", hue="Model", palette=model_palette, edgecolor="black")
    plt.title(f"7. Pass Rates (AFQ ≥ {PASS_THRESHOLD})")
    plt.ylim(0, 110)
    plt.ylabel("Pass Rate (%)")
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(f"{h:.0f}%", (p.get_x() + p.get_width() / 2., h), 
                        ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_pass_rates.png")
    plt.close()

    print(f"✅ All 7 targeted comparison plots successfully generated in '{PLOTS_DIR}'!")

if __name__ == "__main__":
    run_comprehensive_plots()