# Quickstart Guide

This guide will walk you through running a single experiment to verify your setup is correct.

---

## ⏱️ Estimated Time: 5 minutes

---

## Step 1: Activate Your Environment

If you haven't already, activate your Python virtual environment:

=== "Windows"
    ```bash
    .venv\Scripts\activate
    ```

=== "Linux/macOS"
    ```bash
    source .venv/bin/activate
    ```

---

## Step 2: Run a Single Experiment

Execute the main script. By default, it runs a full grid, but you can limit to a quick test with a single configuration:

```bash
python main.py --config configs/main.yaml
```

> [!NOTE]
> On first run, the script will load the dataset from `dataset/` and apply preprocessing. This may take 1-2 minutes.

---

## Step 3: Check the Output

After execution, you'll see output similar to this:

```
================================================================================
🚀 EXPERIMENT GRID EXECUTION
================================================================================
Running Experiment 1/18: binary_lr_s0_s42
   ✅ Training complete (32.4s)
   📊 Metrics: Accuracy=0.809, Macro-F1=0.795, G-Mean=0.791
   💾 Saved: results/metrics/binary_lr_s0_s42.json
...
```

---

## Step 4: Explore Results

Results are saved to the `results/` directory:

| Path | Content |
|------|---------|
| `results/metrics/*.json` | Raw performance metrics per experiment. |
| `results/figures/*.png` | Confusion matrices and radar charts. |
| `results/tables/*.csv` | Summary tables (per-class metrics, etc.). |
| `results/logs/*.log` | Detailed execution logs. |

**Example JSON (`results/metrics/binary_lr_s0_s42.json`):**
```json
{
  "experiment_id": "binary_lr_s0_s42",
  "task": "binary",
  "model": "lr",
  "strategy": "s0",
  "seed": 42,
  "metrics": {
    "accuracy": 0.809,
    "macro_f1": 0.795,
    "weighted_f1": 0.791,
    "g_mean": 0.791,
    "roc_auc": 0.956
  }
}
```

---

## 🎉 Success!

If you see the output above, your environment is correctly configured.

**Next Steps:**
-   **[Run the Full Grid](../experiments/running.md)**: Execute all 18 experiments.
-   **[Run on Colab](../experiments/colab.md)**: Use cloud resources for faster execution.
-   **[Understand the Methodology](../research/methodology.md)**: Learn about S0, S1, S2 strategies.
