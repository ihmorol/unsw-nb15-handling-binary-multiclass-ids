# Implementation Plan: Strategic Transformation to State-of-the-Art

**Goal:** Transform the current research into a publishable masterpiece by addressing statistical gaps and adding explanations ("Why it works").

## User Review Required
> [!IMPORTANT]
> **Computational Cost Warning**: Phase 1 requires running ~72 experiment seeds (Multiclass). Estimated time: 2-3 hours on improved hardward.
> **Approval Needed**: Confirm enabling `n_seeds=5` for Multiclass in `main.py`.

## Proposed Changes

### 1. Experiment Execution (The Foundation)
#### [MODIFY] [main.py](file:///e:/University/ML/ML_PAPER_REVIEW/main.py)
- Ensure the script correctly handles resuming or re-running specific seeds for `task=multi`.
- Verify `n_seeds` logic is robust.

#### [NEW] [scripts/run_missing_seeds.py](file:///e:/University/ML/ML_PAPER_REVIEW/scripts/run_missing_seeds.py)
- A specialized script to target *only* the missing seeds (43, 44, 45, 46) for Multiclass to avoid re-running everything.
- Why? `main.py` might be too broad.

### 2. Advanced Visualization (The "Wow" Factor)
#### [NEW] [scripts/generate_cd_diagram.py](file:///e:/University/ML/ML_PAPER_REVIEW/scripts/generate_cd_diagram.py)
- **Purpose**: Generate Critical Difference (CD) diagrams.
- **Dependencies**: `scikit-posthocs` or `autorank`.
- **Output**: `results/figures/critical_difference_diagram.png`.

#### [NEW] [scripts/generate_shap_analysis.py](file:///e:/University/ML/ML_PAPER_REVIEW/scripts/generate_shap_analysis.py)
- **Purpose**: Explain *why* the model detects Worms.
- **Output**: `results/figures/shap_summary_worms.png`.

### 3. Documentation & Paper
#### [MODIFY] [paper/4.results.tex](file:///e:/University/ML/ML_PAPER_REVIEW/paper/4.results.tex)
- Update tables with new N=5 mean/std.
- Add Statistical significance section.

## Verification Plan

### Automated Tests
1. **Statistics Check**:
   - Run `python scripts/generate_statistics.py`
   - Verify `results/tables/metric_confidence_intervals.csv` has `N_Seeds=5` for ALL rows.
   - Verify `results/tables/paired_significance_tests.csv` shows significant p-values for Multi.

2. **Visualization Check**:
   - Verify `results/figures/critical_difference_diagram.png` exists and is meant for inclusion in the paper.

### Manual Verification
- Review the CD diagram to ensure "S2A" is statistically separated from "S0".
- Review SHAP plots to ensure they make domain sense (e.g., Worms using specific ports/TTL).
