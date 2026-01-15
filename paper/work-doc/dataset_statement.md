# Final Statement of Work: Dataset & Experimental Usage

## 1. Dataset Preparation
We have finalized the dataset preparation for the UNSW-NB15 experiments.

**Action Taken:**
- **Source**: Official `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`.
- **Feature Selection**:
    - **Removed**: `id` column (row identifier, non-predictive).
    - **Retained**: **All 42** original predictive features.
        - *Rationale*: We explicitly decided **not** to perform aggressive feature reduction (e.g., Top 30). Rare attack classes (Worms, Shellcode) are extremely sparse. Aggressive feature selection often prioritizes features that explain the *majority* variance (Normal vs. Attack), potentially discarding subtle signals required to distinguish rare attacks. Retaining all 42 features maximizes the information available to the models.
- **Preprocessing Pipeline** (Fit on Training only):
    - **Numeric**: Median Imputation + Standard Scaling.
    - **Categorical**: "Missing" Imputation + One-Hot Encoding.

## 2. Usage in 18-Experiment Grid
This exact processed dataset will be the **single consistent input** for all 18 experiments to ensuring rigorous comparability.

**The Grid (2 Tasks × 3 Models × 3 Strategies):**

| Task | Model | Strategy | Data Usage |
| :--- | :--- | :--- | :--- |
| **Binary** | LR, RF, XGB | **S0 (None)** | Raw Imbalanced Data (Preprocessed) |
| | | **S1 (Weighting)** | Raw Imbalanced Data + `class_weight` param |
| | | **S2 (SMOTE)** | **Resampled Training Data** (Synthetic minority samples added) |
| **Multiclass** | LR, RF, XGB | **S0 (None)** | Raw Imbalanced Data (Preprocessed) |
| | | **S1 (Weighting)** | Raw Imbalanced Data + `sample_weight` param |
| | | **S2 (SMOTE)** | **Resampled Training Data** (All minority classes upsampled) |

**Leakeage Prevention**:
- **Preprocessing**: Learned statistics (mean, scale, categories) are computed on **Training Data Only**.
- **Resampling (S2)**: SMOTE is applied **inside the cross-validation loop** or **only on the training split**, never touching the Validation or Test sets.
