import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------------
# Configuration & Theme
# -----------------------------
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300

EVAL_DIR = Path("eval_output")
PLOTS_DIR = Path("plot_output")

INDIV_DIR = PLOTS_DIR / "individual"
COMP_DIR = PLOTS_DIR / "comparative"
INDIV_DIR.mkdir(parents=True, exist_ok=True)
COMP_DIR.mkdir(parents=True, exist_ok=True)

def get_clean_dataset_name(filename: str) -> str:
    lower_name = filename.lower()
    if "restaurant" in lower_name: return "Restaurant"
    if "airline" in lower_name: return "Airline"
    if "hospitality" in lower_name: return "Hospitality"
    return filename.replace("evaluated_", "").split(".")[0].capitalize()

def run_plots():
    if not EVAL_DIR.exists():
        raise FileNotFoundError(f"Missing directory {EVAL_DIR}. Run evaluation.py first.")
        
    csv_files = list(EVAL_DIR.glob("evaluated_*.csv"))
    if not csv_files:
        print("No evaluated CSV files found in 'eval_output' folder to plot.")
        return

    print(f"🔍 Found {len(csv_files)} files in {EVAL_DIR}:")
    for f in csv_files:
        print(f"   - {f.name}")

    # Load and combine all datasets
    dfs = []
    for file in csv_files:
        temp_df = pd.read_csv(file)
        dataset_name = get_clean_dataset_name(file.name)
        temp_df["Dataset"] = dataset_name
        dfs.append(temp_df)
        print(f"✅ Loaded '{dataset_name}' with {len(temp_df)} total rows.")
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Drop rows where LLM failed to return a score
    df_valid = df.dropna(subset=["afq_score_0_to_100"]).copy()
    if df_valid.empty:
        print("❌ No valid evaluation scores found to plot.")
        return
        
    print("\n📊 Rows available for plotting after removing LLM failures:")
    print(df_valid["Dataset"].value_counts().to_string())
    print("-" * 40)
        
    dimensions = ["relevance", "actionability", "concreteness", "feasibility"]

    # FIX: Forcefully convert all dimension columns to numeric to prevent Pandas from silently dropping them
    for dim in dimensions:
        df_valid[dim] = pd.to_numeric(df_valid[dim], errors="coerce")

    # ==========================================
    # 1. INDIVIDUAL PLOTS (Per Dataset)
    # ==========================================
    print("Generating Individual Dataset Plots...")
    for dataset_name, df_sub in df_valid.groupby("Dataset"):
        ds_dir = INDIV_DIR / dataset_name
        ds_dir.mkdir(exist_ok=True)
        
        # A. AFQ Histogram + KDE
        plt.figure(figsize=(8, 6))
        sns.histplot(df_sub["afq_score_0_to_100"], bins=10, kde=True, color="steelblue")
        plt.title(f"{dataset_name}: Overall AFQ Score Distribution")
        plt.xlabel("AFQ Score (0-100)")
        plt.ylabel("Count")
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.savefig(ds_dir / f"{dataset_name}_01_afq_distribution.png")
        plt.close()
        
        # B. Format vs Content Scatter
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df_sub, x="format_score_0_to_30", y="content_score_0_to_70",
            color="indigo", alpha=0.7, s=120
        )
        plt.title(f"{dataset_name}: Format vs Content Score")
        plt.xlabel("Format Score (0-30)")
        plt.ylabel("Content Score (0-70)")
        plt.axvline(30, color='gray', linestyle='--', alpha=0.5, label='Max Format (30)')
        plt.axhline(70, color='gray', linestyle='--', alpha=0.5, label='Max Content (70)')
        plt.xlim(-2, 32)
        plt.ylim(-5, 75)
        plt.legend()
        plt.tight_layout()
        plt.savefig(ds_dir / f"{dataset_name}_02_format_vs_content.png")
        plt.close()

        # C. Dimension Boxplots
        melted_sub = df_sub.melt(value_vars=dimensions, var_name="Dimension", value_name="Score (1-5)")
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=melted_sub, x="Dimension", y="Score (1-5)", color="lightseagreen", showfliers=False)
        sns.stripplot(data=melted_sub, x="Dimension", y="Score (1-5)", color=".25", alpha=0.6, jitter=True)
        plt.title(f"{dataset_name}: Scores by Qualitative Dimension")
        plt.ylim(0.5, 5.5)
        plt.tight_layout()
        plt.savefig(ds_dir / f"{dataset_name}_03_dimensions.png")
        plt.close()

        # D. Quality Tiers Pie Chart
        bins = [-1, 59.9, 79.9, 100]
        labels = ['Needs Improvement (<60)', 'Acceptable (60-79)', 'Excellent (80-100)']
        tiers = pd.cut(df_sub['afq_score_0_to_100'], bins=bins, labels=labels)
        tier_counts = tiers.value_counts().sort_index()
        
        plt.figure(figsize=(8, 8))
        colors = ['#ff9999','#ffcc99','#99ff99']
        plt.pie(tier_counts, labels=tier_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
        plt.title(f"{dataset_name}: Overall Quality Breakdown")
        plt.tight_layout()
        plt.savefig(ds_dir / f"{dataset_name}_04_quality_tiers.png")
        plt.close()

    # ==========================================
    # 2. COMPARATIVE PLOTS (Across Datasets)
    # ==========================================
    print("Generating Comparative Overlays...")
    
    # A. AFQ KDE Comparison
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_valid, x="afq_score_0_to_100", hue="Dataset", fill=True, common_norm=False, alpha=0.4, linewidth=2)
    plt.title("Comparative: Overall AFQ Score Density")
    plt.xlabel("AFQ Score (0-100)")
    plt.ylabel("Density")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(COMP_DIR / "01_comparative_afq_density.png")
    plt.close()

    # B. Format vs Content Scatter Comparison
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_valid, x="format_score_0_to_30", y="content_score_0_to_70",
        hue="Dataset", style="Dataset", alpha=0.7, s=100
    )
    plt.title("Comparative: Format vs Content Scores")
    plt.xlabel("Format Score (0-30)")
    plt.ylabel("Content Score (0-70)")
    plt.xlim(-2, 32)
    plt.ylim(-5, 75)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(COMP_DIR / "02_comparative_format_vs_content.png")
    plt.close()

    # C. Grouped Boxplots for Dimensions
    melted_all = df_valid.melt(id_vars=["Dataset"], value_vars=dimensions, var_name="Dimension", value_name="Score (1-5)")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted_all, x="Dimension", y="Score (1-5)", hue="Dataset", palette="Set2")
    plt.title("Comparative: Dimension Score Ranges")
    plt.ylim(0.5, 5.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(COMP_DIR / "03_comparative_dimensions_boxplot.png")
    plt.close()

    # D. Mean Score Bar Chart
    mean_scores = df_valid.groupby("Dataset")[dimensions].mean().reset_index()
    melted_means = mean_scores.melt(id_vars="Dataset", var_name="Dimension", value_name="Mean Score")
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted_means, x="Dimension", y="Mean Score", hue="Dataset", palette="muted")
    plt.title("Comparative: Average Dimension Scores")
    plt.ylim(0, 5.0)
    for p in plt.gca().patches:
        # Ignore NaNs to prevent plotting errors if a bar is totally empty
        if not np.isnan(p.get_height()):
            plt.gca().annotate(f"{p.get_height():.2f}", 
                               (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(COMP_DIR / "04_comparative_mean_scores_bar.png")
    plt.close()

    print(f"Plots successfully generated! Check the '{PLOTS_DIR}' folder.")

if __name__ == "__main__":
    run_plots()