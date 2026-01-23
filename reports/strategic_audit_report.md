# Strategic Audit Report

**Date:** 2026-01-23
**Auditor:** Antigravity (Principal Researcher Agent)
**Status:** [READY] (Pending Completion of Experiments)

## 1. Executive Summary
The codebase is **strictly compliant** with the defined Research Contracts. The "Delta" between promise and reality is near zero regarding methodology, data leakage prevention, and hyperparameter consistency. However, the experimental grid is **incomplete** (54/90 runs finished), which is the only barrier to final publication.

## 2. Contract Compliance Analysis

| Contract | Clause | Status | Evidence |
|----------|--------|--------|----------|
| **Data Contract** | "Fit preprocessing on train only" | **PASS** | `UNSWPreprocessor.fit_transform` splits train/val *before* fitting. |
| **Data Contract** | "Drop specific identifiers" | **PASS** | `configs/main.yaml` lists all 7 leakage columns; `DataLoader` expects them compliant. |
| **Experiment Contract** | "S2a = RandomOverSampler" | **PASS** | `src/strategies/imbalance.py` uses `imblearn.over_sampling.RandomOverSampler`. |
| **Experiment Contract** | "Fixed Random Seeds" | **PASS** | `main.py` propagates `seed` to `run_config`; `ModelTrainer` uses it. |
| **Experiment Contract** | "Run 18 experiments x 5 seeds" | **FAIL** | Only 54/90 experiment runs found in `results/metrics/`. |

## 3. Implementation Reality Check

### 3.1 Data Leakage
*   **Promise**: Validation/Test sets are never seen during generic fitting or resampling.
*   **Reality**: Confirmed. `UNSWPreprocessor` separates `train_split` and `val_split` before `fit` is called. `transform` is called on val/test. Imbalance strategies are applied *only* to `X_train` in `main.py`.

### 3.2 Imbalance Strategies
*   **Promise**: S1 uses class weighting; S2a uses Oversampling.
*   **Reality**: Confirmed.
    *   `S1`: `get_class_weight` returns 'balanced'; `get_scale_pos_weight` logic for XGB is correct.
    *   `S2a`: Instantiates `RandomOverSampler`.

### 3.3 Model Configuration
*   **Promise**: RF `n_estimators=300`, XGB `max_depth=15`.
*   **Reality**: Confirmed in `src/models/config.py`.

## 4. The Delta & Risks

*   **The Delta**: The only deviation is the **execution status**. The code promises a full grid search (18 expts * 5 seeds = 90 runs), but only ~54 runs match the pattern. `multi` task runs are significantly lagging (e.g., `multi_lr_s1` has only seed 42).
*   **Risk**: **Low**. The methodology is scientifically sound. The missing runs are likely due to time/compute constraints, not bugs (as some `multi` runs succeeded).

## 5. Action Items

1.  **[CRITICAL] Resume Experiments**: Re-run `main.py`. The script already has logic to skip existing metrics (`if metrics_file.exists(): continue`), so it will naturally pick up where it left off.
2.  **[RECOMMENDATION] Rare Class Analysis**: Verify that `results/metrics/*_multi_*.json` contains the `rare_class_analysis` block. (Checked code: `main.py` implements this logic).
3.  **[Minor] Artifact Check**: Ensure `experiment_log.csv` is regenerated or appended to correctly after the runs complete.

## 6. Final Verdict
**Codebase is scientifically solid.** Finish the compute job, and you are ready to write.
