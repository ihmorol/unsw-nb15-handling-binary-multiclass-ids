# Research Methodology

This document details the experimental methodology used in this study, including mathematical definitions for metrics and strategies.

---

## 1. Problem Statement

Intrusion Detection Systems (IDS) face severe **class imbalance**: normal traffic vastly outweighs attack traffic in real-world network datasets. The UNSW-NB15 dataset exemplifies this:

| Class | Label | Count (Train) | Percentage |
|-------|-------|---------------|------------|
| Normal | 0 | 56,000 | 31.9% |
| Generic | 1 | 40,000 | 22.8% |
| Exploits | 2 | 33,000 | 18.8% |
| ... | ... | ... | ... |
| **Worms** | 9 | **130** | **0.07%** |
| **Shellcode** | 8 | **1,133** | **0.65%** |

**Consequence**: Standard classifiers (optimizing Accuracy) learn to ignore rare classes, achieving high Accuracy but near-zero Recall on critical attacks.

---

## 2. Strategies Evaluated

We evaluate three distinct strategies for handling class imbalance.

### 2.1 S0: Baseline (No Handling)

*   **Description**: Train on the original, imbalanced class distribution.
*   **Goal**: Establish a baseline for comparison.
*   **Expected Behavior**: High Accuracy, but poor G-Mean and Macro-F1 due to majority class bias.

---

### 2.2 S1: Cost-Sensitive Learning (Class Weighting)

*   **Description**: Assign higher misclassification costs to minority classes during training.
*   **Mechanism**: The learning algorithm's loss function is modified to penalize errors on minority classes more heavily.

**Mathematical Definition (Sklearn `balanced`):**

For each class $i$, the weight $w_i$ is calculated as:

$$w_i = \frac{N}{k \cdot n_i}$$

Where:
-   $N$ = Total number of samples in the training set.
-   $k$ = Number of unique classes.
-   $n_i$ = Number of samples in class $i$.

| Implementation | Parameter |
|----------------|-----------|
| Logistic Regression | `class_weight='balanced'` |
| Random Forest | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight` (for Binary) / `sample_weight` vector (for Multi) |

---

### 2.3 S2: Resampling

Resampling modifies the training data distribution *before* training.

> [!CAUTION]
> Resampling is applied **only to the training set**. Applying it to validation or test sets constitutes **data leakage** and invalidates results.

#### S2a: Random Oversampling (ROS)

*   **Description**: Duplicates minority class samples uniformly at random until all classes have the same count as the majority class.
*   **Pros**: Simple, fast, preserves original feature space.
*   **Cons**: Can lead to overfitting on duplicated samples.

#### S2b: SMOTE (Synthetic Minority Over-sampling Technique)

*   **Description**: Generates synthetic samples by interpolating between existing minority class samples and their k-nearest neighbors.
*   **Pros**: Creates new, diverse samples.
*   **Cons**: Can create noisy samples if classes overlap in feature space.

*Note: S2b (SMOTE) is considered optional in this study; S2a is the primary resampling strategy.*

---

## 3. Models

We evaluate three classical machine learning models with optimized hyperparameters (see [Experiment Contract](../contracts/experiment_contract.md) for full details).

| Model | Description | Key Hyperparameters |
|-------|-------------|---------------------|
| **Logistic Regression (LR)** | Linear model; serves as a baseline. | `solver='lbfgs'`, `max_iter=1000`, `C=1.0` |
| **Random Forest (RF)** | Ensemble of decision trees (bagging). | `n_estimators=300`, `max_depth=None`, `class_weight='balanced_subsample'` |
| **XGBoost (XGB)** | Gradient boosting method. | `n_estimators=150`, `max_depth=15`, `learning_rate=0.05`, `gamma=1.0` |

---

## 4. Metrics

We prioritize **macro-averaged** and **class-balanced** metrics to ensure fair evaluation across all classes, regardless of size.

### 4.1 Macro-F1 Score

The harmonic mean of Precision and Recall, averaged equally across all $k$ classes.

$$\text{Macro-F1} = \frac{1}{k} \sum_{i=1}^{k} F1_i$$

Where $F1_i = \frac{2 \cdot P_i \cdot R_i}{P_i + R_i}$ is the F1-Score for class $i$.

### 4.2 G-Mean (Geometric Mean)

Measures the balance between sensitivity (Recall) across all classes. It is the $k$-th root of the product of per-class Recalls.

$$G\text{-}Mean = \left( \prod_{i=1}^{k} R_i \right)^{1/k}$$

**Interpretation**: A high G-Mean indicates that the model performs well across *all* classes. If *any* class has zero Recall (e.g., Worms), G-Mean collapses to zero.

### 4.3 Per-Class Recall

We explicitly report Recall (Sensitivity/True Positive Rate) for critical rare categories:
-   **Worms** (0.07%)
-   **Shellcode** (0.65%)
-   **Backdoor** (1.2%)
-   **Analysis** (1.5%)

---

## 5. Experimental Protocol

1.  **Data Split**: Use official UNSW-NB15 training and testing sets. No re-splitting.
2.  **Preprocessing**:
    -   Drop identifiers (`id`, `srcip`, `dstip`, `sport`, `dsport`).
    -   Impute missing numerics with median (fit on train).
    -   One-Hot Encode categoricals (fit on train, `handle_unknown='ignore'`).
    -   Standard scale numerics (fit on train).
3.  **Strategy Application**: Apply S1 or S2 *only* to the training set.
4.  **Training**: Train model with fixed hyperparameters.
5.  **Evaluation**: Compute metrics on the **untouched official test set**.

---

## 6. Codebase Structure & Scripts

The implementation is modularized to ensure reproducibility and extensibility. Below is a guide to the key scripts and their roles in the methodology.

### 6.1 Core Execution Scripts (Root)

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `main.py` | **Orchestrator**: Runs the full 18-experiment grid (or subsets if configured). Manages parallel execution, logging, and result saving. | To run the full study reproduction. |
| `runner.py` | **Single Experiment Runner**: Runs an isolated experiment (e.g., `binary lr s0`) without affecting the main grid results. Uses `_single` suffix. | For debugging, smoke testing, or quick validation. |
| `run_full_grid.py` | **Optimized Grid Runner**: A wrapper around `main.py` that forces sequential execution (n_jobs=1) to prevent resource exhaustion on constrained environments (e.g., Colab). | When running on Colab or limited hardware. |

### 6.2 Source Modules (`src/`)

#### Data Handling (`src/data/`)
*   **`loader.py`**: Manages loading of CSVs, column naming, and basic cleaning.
*   **`preprocessing.py`**: Implements the strict `UNSWPreprocessor`. Handles splitting, imputation, one-hot encoding, and scaling. Enforces the "fit on train only" rule.

#### Modeling (`src/models/`)
*   **`trainer.py`**: Contains the `ModelTrainer` class. Unified interface for LR, RF, and XGB. Handles class weighting application and training loops.
*   **`config.py`**: Central repository for model hyperparameters (fixed random states, solver parameters, tree depths).

#### Strategies (`src/strategies/`)
*   **`imbalance.py`**: Implements the `ImbalanceStrategy` abstract base class and concrete strategies (`S0`, `S1`, `S2a`, `S2b`). Contains the logic for resampling and weight calculation.

#### Evaluation (`src/evaluation/`)
*   **`metrics.py`**: Calculates all performance metrics (Macro-F1, G-Mean, Recalls).
*   **`plots.py`**: Generates visual artifacts (Confusion Matrices, ROC/PR Curves, Learning Curves).

### 6.3 Analysis Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `generate_statistics.py` | Conducts statistical significance tests (Friedman, Nemenyi) and generates Critical Difference (CD) diagrams. |
| `generate_publication_figures.py` | Produces high-resolution, publication-ready composite figures (e.g., Radar Charts, Bar Plots). |
| `generate_report.py` | Compiles JSON metrics into readable CSV summary tables (`aggregated_summary.csv`, `rare_class_report.csv`). |
| `deep_audit.py` | Scans the codebase and results to verify compliance with contracts (Leakage checks, Seed verification). |

