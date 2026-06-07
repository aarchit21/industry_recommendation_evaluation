import os
import re
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

# Directories
PLOTS_DIR = Path("plot_output/paper_figures")
STATS_DIR = Path("plot_output/statistics")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

# Bootstrap Config
N_BOOT = 10000
SEED = 42

def parse_metadata(filepath: Path):
    """Extracts Judge, System, and Domain securely using strict regex anchors."""
    path_str = str(filepath).lower()
    filename = filepath.name.lower()
    
    if "gemma" in path_str: judge = "Gemma"
    elif "prometheus" in path_str: judge = "Prometheus"
    elif "qwen" in path_str: judge = "Qwen"
    else: return None, None, None
    
    domain_map = {"1": "Restaurant", "2": "Airline", "3": "Hospitality"}
    
    is_xlora = "xlora" in path_str or "inference_results" in filename or "sheet1" in filename
    
    if is_xlora:
        system = "XLORA"
        if "airline" in filename: domain = "Airline"
        elif "hospitality" in filename: domain = "Hospitality"
        elif "restaurant" in filename: domain = "Restaurant"
        else: domain = "Unknown"
        return system, domain, judge

    # Base Model (Base)
    m_base = re.search(r'base_model_(\d)\.csv', filename)
    if m_base: 
        return "Base", domain_map.get(m_base.group(1), "Unknown"), judge
        
    # Capacity Match (LoRAMoE-1E-R64)
    m_r64 = re.search(r'final_output_1_64_(\d)\.csv', filename)
    if m_r64: 
        return "LoRAMoE-1E-R64", domain_map.get(m_r64.group(1), "Unknown"), judge
        
    # Emergent LoRAMoE
    m_experts = re.search(r'final_output_(\d)_(\d)\.csv', filename)
    if m_experts:
        return f"LoRAMoE-{m_experts.group(1)}E", domain_map.get(m_experts.group(2), "Unknown"), judge
            
    return "Unknown", "Unknown", judge

# -----------------------------
# Statistical Bootstrap Functions
# -----------------------------
def stratified_bootstrap_mean(df: pd.DataFrame, score_col="afq_score_0_to_100"):
    rng = np.random.default_rng(SEED)
    boots = []

    for _ in range(N_BOOT):
        parts = []
        for _, g in df.groupby("Domain"):
            idx = rng.integers(0, len(g), len(g))
            parts.append(g.iloc[idx])
        sample = pd.concat(parts, ignore_index=True)
        boots.append(sample[score_col].mean())

    return {
        "Mean_AFQ": df[score_col].mean(),
        "CI_Low": np.percentile(boots, 2.5),
        "CI_High": np.percentile(boots, 97.5),
    }

def system_cis(per_example: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system, g in per_example.groupby("System"):
        out = stratified_bootstrap_mean(g)
        rows.append({
            "System": system,
            "N_Examples": len(g),
            "Mean_AFQ": round(out["Mean_AFQ"], 1),
            "95% CI": f"[{out['CI_Low']:.1f}, {out['CI_High']:.1f}]"
        })
    return pd.DataFrame(rows).sort_values("Mean_AFQ", ascending=False)

def paired_bootstrap_ci(per_example: pd.DataFrame, system_a: str, system_b: str):
    pivot = (
        per_example.pivot_table(
            index=["Domain", "Row_ID"],
            columns="System",
            values="afq_score_0_to_100",
            aggfunc="first",
        )
        .dropna(subset=[system_a, system_b])
        .reset_index()
    )

    if pivot.empty:
        return None

    pivot["diff"] = pivot[system_a] - pivot[system_b]
    rng = np.random.default_rng(SEED)
    boots = []

    for _ in range(N_BOOT):
        parts = []
        for _, g in pivot.groupby("Domain"):
            idx = rng.integers(0, len(g), len(g))
            parts.append(g.iloc[idx])
        sample = pd.concat(parts, ignore_index=True)
        boots.append(sample["diff"].mean())

    ci_low = np.percentile(boots, 2.5)
    ci_high = np.percentile(boots, 97.5)
    mean_diff = pivot["diff"].mean()
    
    excludes_zero = bool(ci_low > 0 or ci_high < 0)
    interpretation = "Statistically Significant" if excludes_zero else "Statistically Indistinguishable"

    return {
        "Comparison": f"{system_a} minus {system_b}",
        "Mean_AFQ_Diff": f"{mean_diff:+.1f}",
        "95% CI": f"[{ci_low:+.1f}, {ci_high:+.1f}]",
        "Interpretation": interpretation
    }

# -----------------------------
# Main Execution
# -----------------------------
def run_paper_plots(base_folder="Plots"):
    print("🔍 Scanning folders for evaluated CSVs...")
    dfs = []
    
    for filepath in Path(base_folder).rglob("*.csv"):
        system, domain, judge = parse_metadata(filepath)
        if system is None or system == "Unknown" or domain == "Unknown":
            continue
        try:
            temp_df = pd.read_csv(filepath)
        except Exception:
            continue
        if "afq_score_0_to_100" not in temp_df.columns:
            continue
            
        temp_df["Row_ID"] = temp_df.index
        temp_df["System"] = system
        temp_df["Domain"] = domain
        temp_df["Judge"] = judge
        dfs.append(temp_df)
        
    if not dfs:
        print("❌ No valid data loaded.")
        return
        
    df_raw = pd.concat(dfs, ignore_index=True)
    df_raw["afq_score_0_to_100"] = pd.to_numeric(df_raw["afq_score_0_to_100"], errors="coerce")
    df_valid = df_raw.dropna(subset=["afq_score_0_to_100"])
    
    print(f"✅ Loaded {len(df_valid)} valid rows. Processing 3-Judge protocol...")
    
    # -----------------------------
    # 1. Figure 7: Judge Agreement
    # -----------------------------
    pivot_judges = df_valid.pivot_table(index=["System", "Domain", "Row_ID"], columns="Judge", values="afq_score_0_to_100").dropna()
    if not pivot_judges.empty and len(pivot_judges.columns) > 1:
        corr = pivot_judges.corr()
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=1)
        plt.title("Fig 7: Judge AFQ Correlation")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.savefig(PLOTS_DIR / "fig7_judge_agreement.pdf", bbox_inches='tight')
        plt.close()
        
    # -----------------------------
    # 2. Judge Aggregation
    # -----------------------------
    agg_cols = ["afq_score_0_to_100", "relevance", "actionability", "concreteness", "feasibility", "format_score_0_to_30", "content_score_0_to_70"]
    agg_dict = {col: "median" for col in agg_cols if col in df_valid.columns}
    for g in [c for c in df_valid.columns if "gate_" in c]:
        df_valid[g] = pd.to_numeric(df_valid[g], errors="coerce")
        agg_dict[g] = "mean"
        
    df_agg = df_valid.groupby(["System", "Domain", "Row_ID"]).agg(agg_dict).reset_index()
    system_order = ["Base", "LoRAMoE-1E-R64", "LoRAMoE-1E", "LoRAMoE-2E", "LoRAMoE-3E", "LoRAMoE-4E", "XLORA"]
    available_systems = [s for s in system_order if s in df_agg["System"].unique()]
    
    # -----------------------------
    # 3. Statistical Analysis (Bootstrap 95% CIs)
    # -----------------------------
    print("📈 Running 10,000 stratified bootstrap resamples for Statistical Uncertainty...")
    
    # System Level CIs
    sys_ci_df = system_cis(df_agg)
    sys_ci_df.to_csv(STATS_DIR / "Table1_System_AFQ_Bootstrap_CI.csv", index=False)
    
    # Paired Comparisons
    comparisons_to_run = [
        ("XLORA", "LoRAMoE-3E"),
        ("XLORA", "LoRAMoE-4E"),
        ("XLORA", "LoRAMoE-1E-R64"),
        ("LoRAMoE-1E-R64", "LoRAMoE-4E"),
        ("LoRAMoE-1E", "Base"),
        ("XLORA", "Base"),
    ]
    
    pair_rows = []
    avail_set = set(df_agg["System"])
    for sysA, sysB in comparisons_to_run:
        if sysA in avail_set and sysB in avail_set:
            res = paired_bootstrap_ci(df_agg, sysA, sysB)
            if res: pair_rows.append(res)
            
    if pair_rows:
        pd.DataFrame(pair_rows).to_csv(STATS_DIR / "Table2_Paired_Differences_CI.csv", index=False)
        print("✅ Statistics saved successfully to plot_output/statistics/")
    
    # -----------------------------
    # 4. Figures Generation
    # -----------------------------
    print("🎨 Generating final Figures...")
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df_agg, x="System", y="afq_score_0_to_100", hue="Domain", order=available_systems, capsize=.05, edgecolor="black")
    plt.title("Fig 1: Main AFQ by System & Domain (Median Judge Score)", pad=15)
    plt.ylabel("AFQ Score (0-100)")
    plt.xlabel("")
    plt.ylim(0, 100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.savefig(PLOTS_DIR / "fig1_afq_by_system_domain.pdf", bbox_inches='tight')
    plt.close()

    pivot_all = df_agg.pivot_table(index="Domain", columns="System", values="afq_score_0_to_100", aggfunc="mean")
    ordered_cols = [s for s in system_order if s in pivot_all.columns]
    pivot_all = pivot_all[ordered_cols]
    plt.figure(figsize=(14, 5)) 
    sns.heatmap(pivot_all, annot=True, fmt=".1f", cmap="crest", cbar_kws={'label': 'Mean AFQ Score', 'shrink': 0.8}, linewidths=2, edgecolor='white', annot_kws={"size": 13, "weight": "bold"})
    plt.title("Figure 1b: Comprehensive AFQ Score by System & Domain", pad=20, fontsize=16, fontweight='bold')
    plt.xlabel("", fontsize=0)
    plt.ylabel("", fontsize=0)
    plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
    plt.yticks(rotation=0, fontsize=13, fontweight='bold')
    plt.savefig(PLOTS_DIR / "fig1b_afq_heatmap_all.pdf", bbox_inches='tight')
    plt.close()
    
    base_means = df_agg[df_agg["System"] == "Base"].groupby("Domain")["afq_score_0_to_100"].mean()
    def calc_delta(row): return row["afq_score_0_to_100"] - base_means[row["Domain"]] if row["Domain"] in base_means else np.nan
    df_agg["AFQ_Delta"] = df_agg.apply(calc_delta, axis=1)
    df_delta = df_agg[df_agg["System"] != "Base"]
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df_delta, x="System", y="AFQ_Delta", hue="Domain", order=[s for s in available_systems if s != "Base"], capsize=.05, edgecolor="black")
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
    plt.title("Fig 2: AFQ Delta relative to Base", pad=15)
    plt.ylabel("AFQ Improvement (+/-)")
    plt.xlabel("")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.savefig(PLOTS_DIR / "fig2_delta_vs_base.pdf", bbox_inches='tight')
    plt.close()
    
    loramoe_sys = [s for s in available_systems if "LoRAMoE" in s and "E" in s and "R64" not in s]
    if loramoe_sys:
        df_sweep = df_agg[df_agg["System"].isin(loramoe_sys)].copy()
        df_sweep["Expert_Count"] = df_sweep["System"].str.extract(r'(\d)E').astype(int)
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df_sweep, x="Expert_Count", y="afq_score_0_to_100", hue="Domain", marker="o", err_style="bars", markersize=8, linewidth=2)
        plt.title("Fig 3: Emergent Expert Count Sweep", pad=15)
        plt.xlabel("Number of LoRAMoE Experts")
        plt.ylabel("AFQ Score (0-100)")
        plt.xticks([1, 2, 3, 4])
        plt.ylim(0, 100)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(PLOTS_DIR / "fig3_loramoe_expert_sweep.pdf", bbox_inches='tight')
        plt.close()
        
    cap_sys = [s for s in ["LoRAMoE-1E-R64", "LoRAMoE-4E"] if s in available_systems]
    if len(cap_sys) == 2:
        df_cap = df_agg[df_agg["System"].isin(cap_sys)]
        plt.figure(figsize=(8, 6))
        sns.barplot(data=df_cap, x="Domain", y="afq_score_0_to_100", hue="System", capsize=.05, edgecolor="black")
        plt.title("Fig 4: Capacity (1E-R64) vs Routing (4E)", pad=15)
        plt.ylabel("AFQ Score (0-100)")
        plt.xlabel("Domain")
        plt.ylim(0, 100)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(PLOTS_DIR / "fig4_capacity_vs_routing.pdf", bbox_inches='tight')
        plt.close()
        
    df_xlora = df_agg[df_agg["System"] == "XLORA"]
    if not df_xlora.empty:
        gate_cols = [c for c in ["gate_restaurant", "gate_airline", "gate_hospitality"] if c in df_xlora.columns]
        if gate_cols:
            melt_gate = df_xlora.melt(id_vars=["Domain"], value_vars=gate_cols, var_name="Gate_Assigned", value_name="Probability")
            melt_gate["Gate_Assigned"] = melt_gate["Gate_Assigned"].str.replace("gate_", "").str.capitalize()
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=melt_gate, x="Domain", y="Probability", hue="Gate_Assigned", palette="Set2")
            plt.title("Fig 6: X-LoRA Gate Probability Separation", pad=15)
            plt.xlabel("True Domain (Dataset)")
            plt.ylabel("Assigned Gate Probability")
            plt.legend(title="Expert Gate", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.savefig(PLOTS_DIR / "fig6_xlora_gate_separation.pdf", bbox_inches='tight')
            plt.close()
            
    print(f"\n🎉 Workflow complete! Outputs saved to '{PLOTS_DIR}' and '{STATS_DIR}'")

if __name__ == "__main__":
    run_paper_plots(base_folder="Plots")