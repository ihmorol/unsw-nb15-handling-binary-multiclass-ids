# Scientific Review of "Handling Class Imbalance in UNSW-NB15"

**Reviewer:** Dr. Ayaan Rahman  
**Date:** 2026-01-22  
**Scope:** Repository Implementation, Methodology Contracts, and Paper Draft  

---

## 1. Summary ofContributions
*   **Systematic Imbalance Study**: Proposes a rigorous 18-experiment grid crossing 2 tasks (Binary/Multi), 3 models (LR, RF, XGB), and 3 strategies (S0, S1, S2) to isolate the effect of imbalance handling.
*   **Rare Class Focus**: Explicitly targets and quantifies performance on critically rare classes (Worms: 0.07%, Shellcode: 0.65%), moving beyond misleading aggregate metrics.
*   **Reproducible Baseline**: Defines a strict contract for data splitting, preprocessing, and fixed hyperparameters to ensure replicability.
*   **Metric Rigor**: Adopts G-Mean and Per-Class Recall as primary evaluation metrics instead of Accuracy/Weighted-F1 which mask minority failure.

## 2. Strengths
*   **Strong Experimental Design**: The Cartesian product of tasks/models/strategies (Grid of 18) allows for high-confidence causal attribution of performance gains.
*   **Appropriate Metric Selection**: The use of G-Mean and Macro-F1 (rather than Accuracy) properly aligns with the scientific goal of handling imbalance.
*   **Clear Contract Definitions**: The existence of `data_contract.md`, `experiment_contract.md`, and `methodology_contract.md` sets a high standard for research governance.
*   **Code Structure**: The modular design (`src/models`, `src/strategies`, `src/evaluation`) separates concerns effectively, making the logic auditable.
*   **Strategy Implementation**: The implementation of `S2a` (RandomOverSampler) and `S2b` (SMOTE with fallback) in `src/strategies/imbalance.py` is robust and handles edge cases (like k_neighbors failure) correctly.

## 3. Major Concerns (Critical for Publication)

### (1) Fatal Methodology Gap: Preprocessing Mismatch (Leakage & Crash Risk)
*   **Why it matters**: The Paper (Section 3.2.1) and Data Contract (Section 2.1) explicitly state that `srcip`, `dstip`, `sport`, `dsport`, `stime`, and `ltime` are dropped to prevent overfitting/leakage. However, the code (`src/data/preprocessing.py`) and config (`configs/main.yaml`) **DO NOT drop these columns**.
*   **Consequence**: 
    1.  **Pipeline Crash**: `StandardScaler` will likely fail when it encounters string values in `srcip`/`dstip` (since they are not in `categorical_columns`).
    2.  **Invalid Results**: If the code somehow runs, the model will learn from specific IPs/Ports (shortcuts), invalidating the "generalizable intrusion detection" claim.
*   **Concrete Fix**: Update `configs/main.yaml` to include the 6 missing columns in `drop_columns`.

### (2) Total Absence of Empirical Results
*   **Why it matters**: The repository contains no `results/` directory, no `experiment_log.csv`, and no figures. The `Methodology.tex` draft cites these as key evidence. Without these, the paper is a "proposal" not a "study".
*   **Consequence**: Impossible to verify if `S2` actually improves Worms detection or if the models converge.
*   **Concrete Fix**: Run the full experimental grid (18 runs) and confirm artifacts are generated in `results/`.

### (3) SMOTE Reliability for "Worms" Class
*   **Why it matters**: The "Worms" class has only 130 samples. While `src/strategies/imbalance.py` has a fallback, relying on SMOTE with $k=5$ for such a small, potentially disjoint manifold is risky.
*   **Concrete Fix**: Verify in the logs which strategy (SMOTE vs ROS) was actually applied for Worms. If SMOTE is rarely viable, acknowledge this limitation in the paper.

## 4. Minor Concerns
*   **Config Completeness**: The `main.yaml` is missing the definitions for the columns that need to be dropped, relying on defaults that don't match the contract.
*   **Hyperparameter Tuning**: The paper claims fixed hyperparameters (good for baseline), but `config.py` contains `TUNING_GRIDS`. Ensure the final run uses the *fixed* values, not the grids, to match the paper text.
*   **Visualization**: Code for generating figures (CM, ROC curves) mentioned in the paper is not obviously visible in the main execution path.

## 5. Suggested Experiments (Priority Order)
1.  **Fix & Run Baseline (S0)**: Correct the `drop_columns` in config and run the S0 (No Balancing) experiments to look for the "Crash" or "Leakage".
2.  **Run Full Grid (18 Exps)**: Execute the full matrix to populate `results/metrics/*.json`.
3.  **Verify Rare Class Recall**: Check if `Worms` recall > 0.20 in S2a/S2b. If not, the novel contribution claim weakens.
4.  **Ablation Study**: Test if retaining `sport`/`dsport` (as numeric) adds value vs noise, given the paper says they are dropped.

## 6. Rating & Confidence
*   **Rating**: **Reject** (Current State) / **Accept** (If Preprocessing Fixed & Results Added)
*   **Confidence**: **High** (The code audit reveals a deterministic failure mode in preprocessing).

## 7. Questions for Authors
1.  How did you generate the "System Flow Diagram" and results discussed in the draft if the preprocessing code fails to handle string IPs?
2.  Can you confirm the exact feature dimension count after one-hot encoding? (Contract says ~196, but keeping IPs would blow this up or crash).
3.  What is the standard deviation of the G-Mean across the 5 seeds? (Evidence of stability).
