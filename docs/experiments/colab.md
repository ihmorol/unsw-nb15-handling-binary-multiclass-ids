# Running Experiments in Google Colab

This guide explains how to execute the full 90-experiment grid in Google Colab using the provided `colab_full_grid.py` launcher. This approach is recommended for users who want to leverage cloud GPUs and ensure data persistence via Google Drive.

## 🚀 Overview

The `colab_full_grid.py` script automates the entire experiment pipeline:
1.  **Mounts Google Drive**: Ensures your results are saved persistently (no data loss if runtime disconnects).
2.  **Sets up the Environment**: Clones the repository and installs dependencies.
3.  **Optimizes Configuration**: Adjusts parallelism (`n_jobs=1`) to prevent Colab resource crashes.
4.  **Runs the Grid**: Executes all experiments (Binary/Multi × LR/RF/XGB × S0/S1/S2a).
5.  **Periodic Sync**: Backs up results to Drive every 60 seconds.

## 🛠️ Step-by-Step Guide

### 1. Open the Notebook
Open a new Google Colab notebook or upload the `UNSW_NB15_Full_Grid.ipynb` if you have it locally.

### 2. Copy the Launcher Script
If not using the notebook, paste the content of `colab_full_grid.py` into a code cell. 

### 3. Key Configurations
Look for the `CONFIGURATION` section in the script and adjust if necessary:

```python
# ==============================================================================
# CONFIGURATION - MODIFY THESE AS NEEDED
# ==============================================================================
REPO_URL = "https://github.com/StartDust/ML_PAPER_REVIEW.git" # Your Repo URL
BRANCH = "main"
PROJECT_DIR = "/content/ml_project"
DRIVE_BASE_DIR = "/content/drive/MyDrive/UNSW_Archive" # Where results save
```

### 4. Run the Cell
Execute the cell. You will be prompted to authorize Google Drive access.

### 5. Monitor Progress
The script will print progress logs directly in the output.
-   **Dependencies**: Installing... ✅
-   **Config**: Optimizing for Colab... ✅
-   **Execution**: Running Main...

### 6. Access Results
Once completed (or even during execution), navigate to your Google Drive folder defined in `DRIVE_BASE_DIR` (e.g., `UNSW_Archive/run_20260123_120000`).
You will find:
-   `results/metrics/`: JSON files for every experiment.
-   `results/figures/`: Generated plots (Radar charts, Heatmaps).
-   `results/experiment_log.csv`: Master summary log.

## ⚡ Colab Tips
-   **Runtime Type**: A Standard CPU runtime is often sufficient, but a GPU runtime will speed up XGBoost training significantly.
-   **Timeout**: Colab runtimes can disconnect. The script's periodic sync ensures you don't lose progress. If disconnected, simply restart the cell; you can modify `main.yaml` to resume from a specific point if needed, or use the `clean_results(force_clean=False)` option in the script to resume (requires minor code tweak).

## ⚠️ Troubleshooting
-   **Drive Mount Fails**: Ensure you are signed in and accept the permission popup.
-   **Resource Exhaustion**: The script sets `n_jobs=1` to avoid this. Do not increase `n_jobs` in `main.yaml` when running on Colab.
