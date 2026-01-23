# Experiment Reports & Artifacts

This section documents the generated reports and how to interpret the output artifacts from the experiment pipeline.

---

## 📊 Available Reports

The `reports/` directory contains auto-generated analysis files:

| Report | Path | Description |
|--------|------|-------------|
| **Final Results** | `reports/final_results.md` | Executive summary with strategy rankings. |
| **Final Result Report** | `reports/final_result_report.md` | Comprehensive per-experiment breakdown. |
| **XGBoost Improvement** | `reports/XGBoost_Improvement_Report.md` | Deep dive into XGBoost performance gains. |
| **Rare Class Analysis** | `reports/deep_dive_rare_classes.md` | Focus on Worms, Shellcode, Backdoor. |
| **Strategic Audit** | `reports/strategic_audit_report.md` | Compliance check against contracts. |

---

## 📁 Artifact Schema

### Metrics JSON (`results/metrics/*.json`)

Each JSON file represents one complete experiment run.

```json
{
  "experiment_id": "multi_xgb_s1_s42",
  "task": "multi",
  "model": "xgb",
  "strategy": "s1",
  "seed": 42,
  "timestamp": "2026-01-23T12:34:56",
  "training_time_seconds": 142.5,
  "metrics": {
    "accuracy": 0.686,
    "macro_f1": 0.513,
    "weighted_f1": 0.740,
    "g_mean": 0.795,
    "roc_auc_ovr": 0.959
  },
  "per_class_report": {
    "Normal": {"precision": 0.98, "recall": 0.65, "f1": 0.78, "support": 37000},
    "Worms": {"precision": 0.12, "recall": 0.75, "f1": 0.21, "support": 44},
    // ... other classes
  },
  "confusion_matrix": [[...], [...], ...]
}
```

### Summary Tables (`results/tables/`)

| File | Content |
|------|---------|
| `experiment_log.csv` | Master log with all experiments and their metrics. |
| `per_class_metrics.csv` | Precision/Recall/F1 for each class, per experiment. |
| `rare_class_report.csv` | Focused report on Worms, Shellcode, Backdoor, Analysis. |
| `final_summary_tables.csv` | Aggregated summary by Task/Model/Strategy. |

### Figures (`results/figures/`)

| File Pattern | Description |
|--------------|-------------|
| `cm_*.png` | Confusion matrix heatmaps. |
| `radar_*.png` | Strategy comparison radar charts. |
| `rare_class_*.png` | Bar charts for rare class performance. |

---

## 📈 Interpreting Results

When analyzing results, focus on:

1.  **G-Mean over Accuracy**: High Accuracy with low G-Mean indicates majority class bias.
2.  **Rare Class Recall**: Check `per_class_report` for Worms, Shellcode specifically.
3.  **Strategy Comparison**: Compare S0 vs S1 vs S2a on the same model/task.

> [!TIP]
> Use the radar charts (`radar_*.png`) to quickly visualize the trade-offs between Accuracy, F1, and G-Mean.
