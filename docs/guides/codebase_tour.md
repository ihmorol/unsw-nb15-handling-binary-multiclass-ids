# Codebase Tour & Script Guide

This document provides a comprehensive explanation of the scripts and modules in this repository, designed to help you navigate, modify, and extend the codebase.

---

## 🏗️ Root Directory Scripts

These are the entry points for running experiments and pipelines.

### `main.py`
**The Master Orchestrator.**
-   **Role**: Coordinates the entire experimental pipeline.
-   **Key Functions**:
    -   `run_single_experiment()`: The atomic unit of work. Executes one (Task, Model, Strategy) combination.
    -   `main()`: Reads the config, creates the experiment queue (the "Grid"), and executes them (optionally in parallel).
    -   Aggregates results into `tables/aggregated_summary.csv`.
-   **Usage**: `python main.py` (uses default `configs/main.yaml`)

### `runner.py`
**The Scalpel (Single Run).**
-   **Role**: Runs exactly *one* specific experiment configuration.
-   **Key Feature**: Appends `_single` to the experiment ID (e.g., `binary_lr_s0_single`) to avoid overwriting your main grid results.
-   **Usage**: `python runner.py --task binary --model lr --strategy s0`
-   **Why use it**: Perfect for smoke testing, debugging a specific model crash, or testing a new strategy without waiting for the full grid.

### `run_full_grid.py`
**The Safe Runner (Colab/Low-Resource).**
-   **Role**: A wrapper that ensures stability on constrained hardware.
-   **Key Feature**: Forces `n_jobs=1` (sequential execution) regardless of the config file. This prevents RAM explosion or "fork bombs" on Google Colab.
-   **Usage**: `python run_full_grid.py`

---

## 📦 Source Modules (`src/`)

The core logic resides here.

### `src/data/`
-   **`loader.py`**:
    -   Handles reading the CSV files (`UNSW_NB15_training-set.csv`, etc.).
    -   Renames columns to standard snake_case.
    -   Fixes basic data types.
-   **`preprocessing.py`**:
    -   **Crucial Class**: `UNSWPreprocessor`.
    -   **Strict Logic**: Implements the "Fit on Train, Transform on Test" rule to guarantee **Zero Data Leakage**.
    -   Drops ID columns, handles imputing, one-hot encoding, and standard scaling.

### `src/models/`
-   **`trainer.py`**:
    -   **Class**: `ModelTrainer`.
    -   **Abstraction**: Provides a single `.train()` and `.predict()` method that works for Logistic Regression, Random Forest, and XGBoost.
    -   **Smart Handling**: Automatically detects if a model needs `class_weight='balanced'` (sklearn) or `scale_pos_weight` (XGBoost).
-   **`config.py`**:
    -   Stores the "Best Parameters" found during research.
    -   Ensures every model instance uses the exact same `random_state`.

### `src/strategies/`
-   **`imbalance.py`**:
    -   **Heart of the Study**: Defines the S0, S1, S2a, S2b strategies.
    -   **Factory**: `get_strategy('s2a')` returns the correct class instance.
    -   **S2a_RandomOverSampler**: Implements the logic to safety upsample *only* the training data.

### `src/evaluation/`
-   **`metrics.py`**: Calculates G-Mean, Macro-F1, and per-class metrics.
-   **`plots.py`**: Generates the Confusion Matrices, ROC curves, and Learning Curves found in `results/figures/`.

---

## 🛠️ Analysis Scripts (`scripts/`)

Post-processing and deeper analysis tools.

### Reporting & Visualization
-   **`generate_report.py`**:
    -   Scans `results/metrics/` for all JSON files.
    -   Compiles them into the master `final_summary_tables.csv`.
-   **`generate_publication_figures.py`**:
    -   Creates the complex "Radar Charts" and "Bar Comparisons" used in the final paper.
-   **`generate_statistics.py`**:
    -   Performs the Friedman Test and Nemenyi post-hoc analysis.
    -   Generates the Critical Difference (CD) diagrams.

### Maintenance & Audit
-   **`deep_audit.py`**:
    -   **Quality Assurance**: Checks if all 18 experiments exist, if seeds are consistent, and if data contracts were respected.
-   **`cleanup_stale_xgb.py`**:
    -   Utility to kill stuck XGBoost processes (useful on Windows).
-   **`generate_dashboard.py`**:
    -   Creates a simple HTML dashboard to view results results.

---

## 📄 Documentation (`docs/`)

-   **`research/methodology.md`**: The scientific theory behind the code.
-   **`contracts/*.md`**: The "Laws" that the code must follow (e.g., "Never split validation data dynamically").
