# Evaluation Results & Plots Guide

This directory contains the final evaluation visualizations for the **"Actionable Advice from Reviews via Mixture of LoRA Experts"** study. These plots are automatically generated using the `paper_evaluation_plots.py` script based on the outputs of our 3-judge LLM evaluation protocol.

All generated figures are saved as high-resolution PDFs in the `plot_output/paper_figures/` directory.

## Methodology Overview

* **The 3-Judge Protocol:** Every candidate response was independently scored by three distinct LLM judges (Gemma, Prometheus, and Qwen) on a strict 0-100 scale (AFQ-100).
* **Median Aggregation:** To prevent single-model hallucinations or biases from skewing the results, the final score for each row is the **median** of the three judges' scores. 
* **Domains Evaluated:** Restaurant, Airline, and Hospitality.
* **Systems Evaluated:**
  * `Base` (The zero-expert base model)
  * `LoRAMoE-1E-R64` (A single high-capacity expert to test parameter scaling; equivalent to rank 64)
  * `LoRAMoE-[1-4]E` (Emergent routing with 1, 2, 3, or 4 experts)
  * `XLORA` (Predefined, auditable routing)

---

## Guide to the Figures

### Figure 1: Main AFQ by System and Domain
**File:** `fig1_afq_by_system_domain.pdf`  
**Type:** Grouped Bar Plot  
**What it shows:** The primary quality comparison across all evaluated systems, broken down by domain. The Y-axis represents the median AFQ-100 score.  
**How to interpret:** This is the headline figure. Look for the highest bars to see which system ultimately generates the most relevant, actionable, concrete, and feasible advice for each industry.

### Figure 1b: Comprehensive AFQ Score Heatmap
**File:** `fig1b_afq_heatmap_all.pdf`  
**Type:** Heatmap  
**What it shows:** A dense, grid-based view of the exact mean AFQ scores for every combination of System and Domain.  
**How to interpret:** Darker/richer colors indicate higher scores. This allows for rapid cross-referencing (e.g., quickly looking up how XLORA performed specifically on the Airline dataset compared to LoRAMoE-4E).

### Figure 2: AFQ Delta Relative to Base
**File:** `fig2_delta_vs_base.pdf`  
**Type:** Diverging Bar Plot  
**What it shows:** The exact point improvement (or regression) each system provides over the `Base` model. The baseline is normalized to zero.  
**How to interpret:** This graph isolates the pure "gain" achieved by fine-tuning and routing. Any bar extending above the zero-line proves that the specific fine-tuning architecture successfully improved advice quality.

### Figure 3: Emergent Expert Count Sweep
**File:** `fig3_loramoe_expert_sweep.pdf`  
**Type:** Line Plot  
**What it shows:** The AFQ score progression as the number of emergent LoRAMoE experts increases from 1 to 4.  
**How to interpret:** This answers the core scaling question: *Do additional emergent experts actually improve quality, or does the performance saturate after a single adapter?* Look for upward trends to justify multi-expert deployments.

### Figure 4: Capacity vs. Routing
**File:** `fig4_capacity_vs_routing.pdf`  
**Type:** Paired Bar Plot  
**What it shows:** A direct head-to-head comparison between `LoRAMoE-1E-R64` (a single, high-capacity adapter) and `LoRAMoE-4E` (four standard-capacity routed adapters).  
**How to interpret:** This tests whether the gains of Mixture-of-Experts (MoE) come from *actual routing behavior* or simply from *having more trainable parameters*. If 4E beats the 1E-R64 model, routing is demonstrably effective.

### Figure 6: X-LoRA Gate Probability Separation
**File:** `fig6_xlora_gate_separation.pdf`  
**Type:** Box Plot  
**What it shows:** The internal routing confidence of the XLORA model. It maps the true domain of the prompt (X-axis) against the probability assigned to each expert gate (Y-axis).  
**How to interpret:** This tests the reliability of predefined routing. A successful plot will show clear separation—for example, when evaluating the "Restaurant" dataset, the "Restaurant Gate" probability should be consistently near 1.0, while the Airline and Hospitality gates remain near 0.0.

### Figure 7: Judge Agreement
**File:** `fig7_judge_agreement.pdf`  
**Type:** Correlation Heatmap  
**What it shows:** The Pearson correlation coefficient between the scores given by the three LLM judges (Gemma, Prometheus, Qwen).  
**How to interpret:** This is a statistical defense of the evaluation methodology. High correlation values (closer to 1.0) prove that the 3-judge protocol is reliable, objective, and that the LLMs largely agree on what constitutes high-quality actionable advice.

---

## Running the Plot Generation

To regenerate these plots, ensure your raw evaluated CSV files are located in the `Plots/` directory, and run the following command from the repository root:

```bash
python paper_evaluation_plots.py