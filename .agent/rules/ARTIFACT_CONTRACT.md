# Artifact Contract (What Must Exist)

Artifacts are the project's evidence. If an artifact is missing, the result does not exist.

## Directory Structure

```
results/
├── metrics/           # Per-experiment JSON files
├── figures/           # Confusion matrix PNGs
├── tables/            # Summary CSVs
├── logs/              # Run logs
├── models/            # Trained model files (optional)
└── processed/         # Preprocessing metadata
```

## Per-Experiment Artifacts

For each experiment ID `{task}_{model}_{strategy}`:

| Artifact | Path | Required |
|----------|------|----------|
| Metrics JSON | `results/metrics/{exp_id}.json` | ✅ Yes |
| Confusion Matrix | `results/figures/cm_{exp_id}.png` | ✅ Yes |
| Trained Model | `results/models/{exp_id}.joblib` | ⚠️ Optional |

### Metrics JSON Schema

```json
{
  "experiment_id": "binary_rf_s1",
  "task": "binary",
  "model": "rf",
  "strategy": "s1",
  "timestamp": "2026-01-17T12:00:00Z",
  "training_time_seconds": 120.5,
  "train_samples": 140273,
  "test_samples": 82332,
  "metrics": {
    "overall": {
      "accuracy": 0.89,
      "macro_f1": 0.87,
      "weighted_f1": 0.89,
      "g_mean": 0.88,
      "roc_auc": 0.92
    },
    "per_class": {...},
    "confusion_matrix": [[...], [...]]
  },
  "rare_class_analysis": null
}
```

## Global Artifacts

| Artifact | Path | Required |
|----------|------|----------|
| Experiment Log | `results/experiment_log.csv` | ✅ Yes |
| Summary Table | `results/tables/final_summary_tables.csv` | ✅ Yes |
| Per-Class Metrics | `results/tables/per_class_metrics.csv` | ✅ Yes |
| Rare Class Report | `results/tables/rare_class_report.csv` | ✅ Yes |
| Preprocessing Metadata | `results/processed/preprocessing_metadata.json` | ✅ Yes |
| Run Log | `results/logs/run_*.log` | ✅ Yes |

## Processed Data Artifacts (Optional)

These are kept in memory during execution. Saving to disk is optional:

| Artifact | Path | Status |
|----------|------|--------|
| X_train_enc | `results/processed/X_train_enc.csv` | ⚠️ Optional |
| X_val_enc | `results/processed/X_val_enc.csv` | ⚠️ Optional |
| X_test_enc | `results/processed/X_test_enc.csv` | ⚠️ Optional |
| y splits | `results/processed/y_*.csv` | ⚠️ Optional |

## Validation Rule

After a full 18-experiment run:
- `results/metrics/` should contain 18 JSON files
- `results/figures/` should contain 18 PNG files
- `results/tables/` should contain 3 CSV files
- `results/experiment_log.csv` should have 18 rows
