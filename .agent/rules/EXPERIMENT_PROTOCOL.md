# Experiment Protocol

This project evaluates classical ML under extreme class imbalance using a fixed matrix of runs.

## Fixed Experiment Matrix

| Dimension | Values | Count |
|-----------|--------|-------|
| Tasks | binary, multi | 2 |
| Models | lr, rf, xgb | 3 |
| Strategies | s0, s1, s2a | 3 |
| **Total** | | **18** |

### Strategy Definitions

| ID | Name | Description |
|----|------|-------------|
| `s0` | None (Baseline) | No balancing applied |
| `s1` | Class Weight | Inverse frequency weights (`class_weight='balanced'`) |
| `s2a` | RandomOverSampler | Duplicate minority samples (imblearn) |
| `s2b` | SMOTE (Optional) | Synthetic minority samples (if enabled) |

## Run ID and Reproducibility

Each run must have:
- **experiment_id:** `{task}_{model}_{strategy}` (e.g., `binary_rf_s1`)
- **timestamp:** ISO format
- **random_state:** 42 (from `configs/main.yaml`)
- **config snapshot:** Referenced in JSON output

## Required Outputs Per Experiment

| Artifact | Path | Format |
|----------|------|--------|
| Metrics | `results/metrics/{exp_id}.json` | JSON |
| Confusion Matrix | `results/figures/cm_{exp_id}.png` | PNG |
| Run Log | `results/logs/run_*.log` | Text |

## Global Outputs (After All Runs)

| Artifact | Path |
|----------|------|
| Experiment Log | `results/experiment_log.csv` |
| Summary Table | `results/tables/final_summary_tables.csv` |
| Per-Class Metrics | `results/tables/per_class_metrics.csv` |
| Rare Class Report | `results/tables/rare_class_report.csv` |

## Metrics (Minimum Required)

### Overall Metrics
- accuracy
- macro_f1
- weighted_f1
- g_mean
- roc_auc (binary; OVR macro for multiclass)

### Per-Class Metrics
- precision, recall, f1, support for every class

### Rare Class Focus (Multiclass Only)
Explicit reporting for: **Worms, Shellcode, Backdoor, Analysis**

## Hyperparameters

Fixed hyperparameters are defined in `docs/contracts/experiment_contract.md` §3.3.
Do not change without updating the contract and re-running all experiments.
