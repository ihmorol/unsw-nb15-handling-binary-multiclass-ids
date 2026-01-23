# Running Experiments

This guide provides a comprehensive reference for executing experiments locally or in the cloud.

---

## Overview

The project executes a **systematic grid** of experiments:

| Dimension | Values | Count |
|-----------|--------|-------|
| **Tasks** | Binary, Multiclass | 2 |
| **Models** | LR, RF, XGBoost | 3 |
| **Strategies** | S0, S1, S2a | 3 |
| **Total** | | **18** |

Each experiment produces a JSON file with performance metrics and a confusion matrix figure.

---

## Execution Methods

### Method 1: `main.py` (Recommended)

The primary entry point for running the full grid.

```bash
python main.py --config configs/main.yaml
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `configs/main.yaml` | Path to the YAML configuration file. |

**Key Configuration Options (`configs/main.yaml`):**

```yaml
experiments:
  n_seeds: 1          # Number of seeds (1-5 recommended)
  n_jobs: -1          # Parallelism (-1 = all cores, 1 = sequential)
  tasks:
    - binary
    - multi
  models:
    - lr
    - rf
    - xgb
  strategies:
    - s0              # Baseline (no balancing)
    - s1              # Class Weighting
    - s2a             # Random Oversampling
```

---

### Method 2: `colab_full_grid.py` (Cloud)

Optimized for Google Colab execution with Google Drive integration.

See the dedicated **[Colab Guide](colab.md)** for detailed instructions.

---

### Method 3: `runner.py` (Advanced)

A lower-level script for executing individual configurations. Useful for debugging or resuming specific experiments.

```bash
python runner.py --task binary --model xgb --strategy s1 --seed 42
```

---

## Output Structure

Every run generates artifacts in the `results/` directory:

```
results/
├── metrics/
│   ├── binary_lr_s0_s42.json
│   ├── binary_lr_s1_s42.json
│   └── ...
├── figures/
│   ├── cm_binary_lr_s0_s42.png      # Confusion Matrix
│   └── ...
├── tables/
│   ├── experiment_log.csv           # Master summary
│   ├── per_class_metrics.csv        # Rare class detail
│   └── final_summary_tables.csv
├── logs/
│   └── run_YYYYMMDD_HHMMSS.log
└── processed/
    └── preprocessing_metadata.json  # Feature names, scalers
```

---

## Understanding the Output Files

### `metrics/*.json`

Each JSON contains the complete result of one experiment:

| Field | Description |
|-------|-------------|
| `experiment_id` | Unique identifier (e.g., `binary_xgb_s1_s42`). |
| `task` | `binary` or `multi`. |
| `model` | `lr`, `rf`, or `xgb`. |
| `strategy` | `s0`, `s1`, or `s2a`. |
| `seed` | Random seed used. |
| `metrics` | Dict of Accuracy, Macro-F1, G-Mean, ROC-AUC, etc. |
| `confusion_matrix` | 2D array of confusion matrix values. |
| `per_class_report` | Dict of Precision/Recall/F1 per class. |

### `figures/*.png`

Visual representations of model performance:
-   **Confusion Matrices**: Heatmaps showing TP, FP, TN, FN.
-   **Radar Charts**: Strategy comparisons across multiple metrics.

---

## Monitoring Long Runs

For long runs (especially Multiclass with S2a), you can monitor progress:

1.  **Check `results/metrics/`**: JSON files appear as experiments complete.
2.  **Tail the log file**:
    ```bash
    tail -f results/logs/run_*.log
    ```
3.  **View `experiment_log.csv`**: Appended after each experiment.
