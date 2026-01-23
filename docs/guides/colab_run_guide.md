# 🚀 Comprehensive Guide: Running Experiments on Google Colab

This guide explains how to execute the full 90-experiment grid on Google Colab, ensuring all results are safely synced to your Google Drive.

## ⏱️ Time Estimation

**Estimated Total Runtime:** ~4 to 6 Hours

| Phase | Description | Est. Time |
|-------|-------------|-----------|
| **Setup** | Cloning repo, installing deps | 2-3 mins |
| **Binary Task** | 45 runs (LR, RF, XGB) | ~45 mins |
| **Multiclass Task** | 45 runs (LR, RF, XGB) | ~3 - 4 hours |
| **Total** | | **~4 - 5 Hours** |

> **Why so long?**
> The **Multiclass S2a** (RandomOverSampler) experiments explode the training set size from **175k** to **~560k** samples to balance all 10 classes. Training XGBoost on this expanded dataset 5 times takes the majority of the time.

---

## 🛠️ Step-by-Step Execution Guide

### 1. Preparation (Local)
Before going to Colab, ensure your latest code is on GitHub.

1.  **Commit and Push** your changes:
    ```bash
    git add .
    git commit -m "feat: Ready for full Colab run"
    git push origin main
    ```

### 2. Open Google Colab
1.  Navigate to [colab.research.google.com](https://colab.research.google.com).
2.  Click **File > Open Notebook**.
3.  Select the **GitHub** tab.
4.  Enter your repository URL: `https://github.com/StartDust/ML_PAPER_REVIEW`.
5.  Select the `UNSW_NB15_Full_Grid.ipynb` file from the list.

### 3. Configure Runtime (Critical)
To speed up XGBoost and ensure stability:
1.  Click **Runtime** in the top menu.
2.  Select **Change runtime type**.
3.  **Hardware accelerator**: Select **T4 GPU**.
4.  **Runtime shape**: Standard is fine, "High-RAM" is better if available (Colab Pro).
5.  Click **Save**.

### 4. Run the Experiments
1.  Locate the cell titled **🚀 Run Full Experiment Grid (0-89)**.
2.  Verify the configuration form fields:
    - `REPO_URL`: Your GitHub URL.
    - `BRANCH`: `main` (usually).
    - `FORCE_FRESH_RUN`: `True` (checks `check` to wipe old partial runs).
    - `SYNC_INTERVAL_SECONDS`: `60`.
3.  **Start execution**:
    - Click **Runtime > Run all** (or press `Ctrl + F9`).
    - Or click the "Play" button on the cell.

### 5. Authentication
1.  The script will prompt: `Mounting Google Drive...`.
2.  A pop-up window will ask for permission to access your Google Drive.
3.  Click **Connect to Google Drive** and authorize it.
    - *Note: This is required to save your results to `UNSW_Archive` so they aren't lost if Colab disconnects.*

### 6. Monitor Progress
- The cell output will stream logs in real-time.
- **Do not close the tab**: You must keep the browser tab open (active or background) for the run to continue.
- **Prevent Disconnects**:
    - Colab might timeout if idle for 90 mins.
    - Since the cell is printing output (streaming), it usually stays active.
    - Check on it every hour if possible.

### 7. View Results (Real-Time)
You don't need to wait for it to finish!
1.  Open a new tab to **Google Drive**.
2.  Navigate to `My Drive > UNSW_Archive`.
3.  Find the folder `run_YYYYMMDD_HHMMSS` (matching the timestamp in the Colab output).
4.  You can watch JSONs and PNGs appear in `results/metrics` and `results/figures` as they complete.

---

## 🆘 Troubleshooting

### "Runtime Disconnected"
If Colab disconnects halfway:
1.  **Don't panic.** Your results up to the last 60 seconds are safe in Drive.
2.  Reconnect the runtime.
3.  **Uncheck** `FORCE_FRESH_RUN` in the notebook form.
4.  Re-run the cell.
    - The script identifies existing JSON files in your Drive folder (if you mounted it to the same place... actually, the script creates a *new* timestamp folder by default).
    - **To Resume specific folder**: You would need to manually set `RESULTS_DIR` in the script to the *old* timestamp folder.
    - *Easier Path*: Just accept the split results (partial run in Folder A, rest in Folder B) and merge them later using `scripts/consolidate_all.py`.

### "Out of Memory" (OOM)
If the session crashes during **Multiclass S2a**:
- This is rare on T4 GPU but possible with standard RAM.
- **Fix**: The script already uses `n_jobs=1` to minimize memory spikes. If it crashes, unfortunately, you might skip S2a or try Colab Pro.

### "Drive not syncing"
- Check the output logs for `rsync` errors.
- Ensure you have free space in Google Drive (Project is ~500MB - 1GB total).
