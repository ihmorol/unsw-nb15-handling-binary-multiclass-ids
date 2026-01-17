# Experiment Contract: UNSW-NB15 Imbalance Study

> **Document Version:** 2.0  
> **Last Updated:** 2026-01-17  
> **Status:** APPROVED

---

## 1. Experiment Scope

This contract defines the complete scope of experiments to analyze the impact of class imbalance handling on the UNSW-NB15 dataset for intrusion detection.

### 1.1 Research Objectives

| Objective ID | Description | Addressed By |
|--------------|-------------|--------------|
| O1 | Build binary IDS classifiers | Task A experiments |
| O2 | Build multiclass IDS classifiers | Task B experiments |
| O3 | Compare imbalance handling strategies | S0/S1/S2 comparison |
| O4 | Analyze rare attack detection | Per-class metrics focus |
| O5 | Establish reproducible baseline | Fixed seeds, configs |

### 1.2 Research Questions Mapping

| RQ | Question | Experiments Needed |
|----|----------|-------------------|
| RQ1 | Imbalance effect on binary vs multiclass | All Task A vs Task B |
| RQ2 | Effectiveness of weighting/oversampling | S1/S2 vs S0 comparisons |
| RQ3 | Model response patterns to strategies | Cross-model analysis |
| RQ4 | Rare class improvement with S2 | Multi_*_S2 vs Multi_*_S0 |

---

## 2. Methodology Matrix

The experiment space is defined by the Cartesian product of the following dimensions:

### 2.1 Dimension A: Tasks

| Task ID | Name | Description | Classes | Label Column |
|---------|------|-------------|---------|--------------|
| **A** | Binary | Normal vs Attack | 2 | `label` |
| **B** | Multiclass | Normal + 9 attack types | 10 | `attack_cat` |

### 2.2 Dimension B: Models

| Model ID | Full Name | Family | scikit-learn Class |
|----------|-----------|--------|-------------------|
| **LR** | Logistic Regression | Linear | `LogisticRegression` |
| **RF** | Random Forest | Bagging | `RandomForestClassifier` |
| **XGB** | XGBoost | Boosting | `XGBClassifier` |

### 2.3 Dimension C: Imbalance Strategies

| Strategy ID | Name | Description | Data Modification |
|-------------|------|-------------|-------------------|
| **S0** | None (Baseline) | No balancing applied | None |
| **S1** | Class Weight | Inverse frequency weights | Model parameter |
| **S2a** | RandomOverSampler | Duplicate minority samples | Training data expanded |
| **S2b** | SMOTE (Optional) | Synthetic minority samples | Training data expanded |

### 2.4 Experiment Grid Summary

**Core Grid:**
```
Tasks (2) × Models (3) × Strategies (3) = 18 Core Experiments
```

**With Optional SMOTE:**
```
Tasks (2) × Models (3) × Strategies (4) = 24 Total Experiments
```

### 2.5 Complete Experiment List

| # | Experiment ID | Task | Model | Strategy |
|---|---------------|------|-------|----------|
| 1 | `binary_lr_s0` | Binary | LR | None |
| 2 | `binary_lr_s1` | Binary | LR | Class Weight |
| 3 | `binary_lr_s2a` | Binary | LR | RandomOverSampler |
| 4 | `binary_rf_s0` | Binary | RF | None |
| 5 | `binary_rf_s1` | Binary | RF | Class Weight |
| 6 | `binary_rf_s2a` | Binary | RF | RandomOverSampler |
| 7 | `binary_xgb_s0` | Binary | XGB | None |
| 8 | `binary_xgb_s1` | Binary | XGB | Class Weight |
| 9 | `binary_xgb_s2a` | Binary | XGB | RandomOverSampler |
| 10 | `multi_lr_s0` | Multiclass | LR | None |
| 11 | `multi_lr_s1` | Multiclass | LR | Class Weight |
| 12 | `multi_lr_s2a` | Multiclass | LR | RandomOverSampler |
| 13 | `multi_rf_s0` | Multiclass | RF | None |
| 14 | `multi_rf_s1` | Multiclass | RF | Class Weight |
| 15 | `multi_rf_s2a` | Multiclass | RF | RandomOverSampler |
| 16 | `multi_xgb_s0` | Multiclass | XGB | None |
| 17 | `multi_xgb_s1` | Multiclass | XGB | Class Weight |
| 18 | `multi_xgb_s2a` | Multiclass | XGB | RandomOverSampler |

**Optional SMOTE Experiments (S2b):**

| # | Experiment ID | Task | Model | Strategy |
|---|---------------|------|-------|----------|
| 19 | `binary_lr_s2b` | Binary | LR | SMOTE |
| 20 | `binary_rf_s2b` | Binary | RF | SMOTE |
| 21 | `binary_xgb_s2b` | Binary | XGB | SMOTE |
| 22 | `multi_lr_s2b` | Multiclass | LR | SMOTE |
| 23 | `multi_rf_s2b` | Multiclass | RF | SMOTE |
| 24 | `multi_xgb_s2b` | Multiclass | XGB | SMOTE |

---

## 3. Configuration Parameters

### 3.1 Data Splitting

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Training Set | Official UNSW-NB15 train | Benchmark compatibility |
| Validation Split | 20% of Training (stratified) | Hyperparameter tuning |
| Test Set | Official UNSW-NB15 test | Final evaluation only |
| Stratification | Yes (by target class) | Preserve class ratios |

### 3.2 Reproducibility Parameters

| Parameter | Value | Implementation |
|-----------|-------|----------------|
| Random Seed | 42 | All random operations |
| NumPy Seed | 42 | `np.random.seed(42)` |
| Python Hash Seed | 42 | `PYTHONHASHSEED=42` |

### 3.3 Hyperparameter Specifications

#### Logistic Regression (Binary)

```python
LogisticRegression(
    C=1.0,                    # Regularization strength
    solver='saga',            # Supports L1/L2
    max_iter=1000,            # Convergence iterations
    penalty='l2',             # Regularization type
    class_weight=None,        # S0: None, S1: 'balanced'
    n_jobs=-1,
    random_state=42
)
```

#### Logistic Regression (Multiclass)

```python
LogisticRegression(
    C=1.0,
    solver='saga',
    max_iter=1000,
    penalty='l2',
    multi_class='multinomial',  # Multiclass handling
    class_weight=None,          # S0: None, S1: 'balanced'
    n_jobs=-1,
    random_state=42
)
```

#### Random Forest (Both Tasks)

```python
RandomForestClassifier(
    n_estimators=200,         # Number of trees
    max_depth=20,             # Maximum tree depth
    min_samples_split=5,      # Min samples to split node
    min_samples_leaf=2,       # Min samples in leaf
    max_features='sqrt',      # Features per split
    class_weight=None,        # S0: None, S1: 'balanced'
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
```

#### XGBoost (Binary)

```python
XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=10,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=None,    # S0: None, S1: computed ratio
    use_label_encoder=False,
    eval_metric='logloss',
    objective='binary:logistic',
    n_jobs=-1,
    random_state=42,
    verbosity=0
)
```

#### XGBoost (Multiclass)

```python
XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=10,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    objective='multi:softprob',
    num_class=10,
    n_jobs=-1,
    random_state=42,
    verbosity=0
)
```

### 3.4 Optional Hyperparameter Tuning Grid

If time permits, limited tuning on validation set:

| Model | Parameter | Values |
|-------|-----------|--------|
| LR | C | [0.01, 0.1, 1.0, 10.0] |
| RF | n_estimators | [100, 200] |
| RF | max_depth | [10, 20, None] |
| XGB | n_estimators | [100, 200] |
| XGB | learning_rate | [0.01, 0.1] |
| XGB | max_depth | [5, 10] |

**Tuning Method:** 3-Fold Stratified Cross-Validation on Training Set  
**Scoring Metric:** Macro F1 (to weight all classes equally)

---

## 4. Imbalance Strategy Specifications

### 4.1 S0: No Balancing (Baseline)

**Purpose:** Establish baseline to show how models fail on minority classes.

```python
# No modification to training data
X_train_s0 = X_train
y_train_s0 = y_train

# No class weight
model.fit(X_train_s0, y_train_s0)
```

**Expected Results:**
- High overall accuracy (>85%)
- Very low recall for rare classes (Worms: 0-5%, Shellcode: 5-20%)
- This demonstrates the **severity of imbalance problem**

### 4.2 S1: Class Weighting

**Purpose:** Penalize misclassification of minority classes more heavily.

```python
# For sklearn models
model = RandomForestClassifier(class_weight='balanced', ...)

# For XGBoost Binary
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
model = XGBClassifier(scale_pos_weight=scale_pos_weight, ...)

# For XGBoost Multiclass (via sample_weight)
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_train)
model.fit(X_train, y_train, sample_weight=sample_weights)
```

**Computed Weights (Multiclass, Training Set):**

| Class | Count | Weight (approx.) |
|-------|-------|------------------|
| Normal | 56,000 | 0.31 |
| Generic | 40,000 | 0.44 |
| Exploits | 33,393 | 0.53 |
| Fuzzers | 18,184 | 0.96 |
| DoS | 12,264 | 1.43 |
| Reconnaissance | 10,491 | 1.67 |
| Analysis | 2,000 | 8.77 |
| Backdoor | 1,746 | 10.04 |
| Shellcode | 1,133 | 15.48 |
| Worms | 130 | 134.88 |

### 4.3 S2a: Random Oversampling

**Purpose:** Duplicate minority class samples to balance distribution.

```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(
    sampling_strategy='auto',  # Upsample all minorities to majority
    random_state=42
)
X_train_s2a, y_train_s2a = ros.fit_resample(X_train, y_train)
```

**Post-Resampling Size:**

| Task | Original Size | After ROS |
|------|---------------|-----------|
| Binary | 140,273 | ~224,000 (2×116,341 attacks) |
| Multiclass | 140,273 | ~560,000 (10×56,000 per class) |

### 4.4 S2b: SMOTE (Optional)

**Purpose:** Generate synthetic samples in feature space for minority classes.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(
    k_neighbors=5,            # Nearest neighbors for interpolation
    sampling_strategy='auto',
    random_state=42,
    n_jobs=-1
)

try:
    X_train_s2b, y_train_s2b = smote.fit_resample(X_train, y_train)
except ValueError:
    # Fallback for classes with < k_neighbors samples (e.g., Worms=130 < 5 is OK, but edge cases)
    X_train_s2b, y_train_s2b = RandomOverSampler(random_state=42).fit_resample(X_train, y_train)
```

> [!WARNING]
> **SMOTE Memory Requirements:**
> - Multiclass SMOTE can generate ~560K samples
> - Requires ~10GB RAM for full dataset
> - Use S2a if memory is constrained

---

## 5. Evaluation Metrics Specification

### 5.1 Overall Metrics (Per Experiment)

| Metric | Formula | scikit-learn | Purpose |
|--------|---------|--------------|---------|
| **Accuracy** | (TP+TN)/Total | `accuracy_score` | Baseline (misleading for imbalance) |
| **Macro F1** | Mean(F1_per_class) | `f1_score(average='macro')` | Equal weight all classes |
| **Weighted F1** | Weighted Mean(F1) | `f1_score(average='weighted')` | Reflects majority |
| **G-Mean** | √(Sensitivity×Specificity) | `geometric_mean_score` | **Primary metric** |
| **ROC-AUC** | Area under ROC | `roc_auc_score` | Threshold-independent |

**G-Mean Implementation:**

```python
from imblearn.metrics import geometric_mean_score

# Binary
g_mean = geometric_mean_score(y_true, y_pred)

# Multiclass
g_mean = geometric_mean_score(y_true, y_pred, average='macro')
```

**ROC-AUC Implementation:**

```python
from sklearn.metrics import roc_auc_score

# Binary
roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])

# Multiclass (One-vs-Rest)
roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
```

### 5.2 Per-Class Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Precision** | TP / (TP + FP) | False positive control |
| **Recall** | TP / (TP + FN) | **Critical for rare classes** |
| **F1-Score** | 2×(P×R)/(P+R) | Balance of P and R |
| **Support** | Count per class | Context for interpretation |

### 5.3 Confusion Matrix

| Task | Size | Focus Areas |
|------|------|-------------|
| Binary | 2×2 | False Negatives (missed attacks) |
| Multiclass | 10×10 | Rows for Worms, Shellcode, Backdoor, Analysis |

### 5.4 Rare Class Analysis

**Rare Classes (Training < 3% of data):**

| Class | Index | Training Samples | Target Recall |
|-------|-------|------------------|---------------|
| Worms | 9 | 130 | > 20% |
| Shellcode | 8 | 1,133 | > 40% |
| Backdoor | 3 | 1,746 | > 50% |
| Analysis | 2 | 2,000 | > 60% |

**Analysis Report Structure:**

```json
{
  "rare_class_analysis": {
    "Worms": {
      "s0_recall": 0.02,
      "s1_recall": 0.15,
      "s2a_recall": 0.25,
      "improvement_s2a_vs_s0": 0.23
    },
    "Shellcode": {...},
    "Backdoor": {...},
    "Analysis": {...}
  }
}
```

---

## 6. Output Artifacts

### 6.1 Per-Experiment Outputs

| Artifact | Format | Path | Contents |
|----------|--------|------|----------|
| Metrics JSON | JSON | `results/metrics/{exp_id}.json` | All metrics |
| Confusion Matrix | PNG | `results/figures/cm_{exp_id}.png` | Heatmap |
| Model Object | joblib | `results/models/{exp_id}.joblib` | Trained model |
| Predictions | NPY | `results/predictions/{exp_id}.npy` | Test set predictions |

### 6.2 Metrics JSON Schema

```json
{
  "experiment_id": "binary_rf_s1",
  "task": "binary",
  "model": "rf",
  "strategy": "s1",
  "timestamp": "2026-01-17T12:00:00Z",
  "training_time_seconds": 120.5,
  "overall": {
    "accuracy": 0.89,
    "macro_f1": 0.87,
    "weighted_f1": 0.89,
    "g_mean": 0.88,
    "roc_auc": 0.92
  },
  "per_class": {
    "0": {"precision": 0.85, "recall": 0.92, "f1": 0.88, "support": 37000},
    "1": {"precision": 0.91, "recall": 0.84, "f1": 0.87, "support": 45332}
  },
  "confusion_matrix": [[34040, 2960], [7253, 38079]],
  "rare_class_analysis": null
}
```

### 6.3 Aggregated Outputs

| Artifact | Path | Contents |
|----------|------|----------|
| Experiment Log | `results/experiment_log.csv` | Summary of all experiments |
| Comparison Tables | `results/tables/` | LaTeX-ready comparison tables |
| Final Report | `reports/final_results.md` | Narrative analysis |

### 6.4 Experiment Log Schema

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | String | Unique run identifier |
| `experiment_id` | String | Experiment name |
| `timestamp` | DateTime | Run timestamp |
| `task` | String | 'binary' or 'multi' |
| `model` | String | 'lr', 'rf', 'xgb' |
| `strategy` | String | 's0', 's1', 's2a', 's2b' |
| `accuracy` | Float | Accuracy score |
| `macro_f1` | Float | Macro F1 score |
| `weighted_f1` | Float | Weighted F1 score |
| `g_mean` | Float | Geometric mean score |
| `roc_auc` | Float | ROC-AUC score |
| `training_time` | Float | Seconds to train |
| `status` | String | 'success' or 'failed' |

---

## 7. Success Criteria

### 7.1 Execution Criteria

| Criterion | Requirement | Validation |
|-----------|-------------|------------|
| Completeness | All 18 core experiments completed | Check experiment_log.csv |
| Reproducibility | Same seed → identical results | Re-run and compare |
| No Crashes | All experiments succeed | status == 'success' |
| Reasonable Runtime | Total < 8 hours | Sum(training_time) |

### 7.2 Quality Criteria

| Criterion | Requirement | Validation |
|-----------|-------------|------------|
| Non-trivial Metrics | Accuracy not 0 or 1 | 0.5 < accuracy < 1.0 |
| Strategy Variation | Metrics differ across S0/S1/S2 | Compare JSONs |
| Model Variation | Metrics differ across LR/RF/XGB | Compare JSONs |
| Rare Class Focus | Worms/Shellcode metrics recorded | Check per_class in multi |

### 7.3 Scientific Validity Criteria

| Criterion | Requirement | Validation |
|-----------|-------------|------------|
| No Data Leakage | Test metrics from test set only | Code review |
| Proper Stratification | Val set class ratios match train | Distribution check |
| Correct Resampling | S2 only on training | Code review |
| Consistent Preprocessing | Same pipeline for all | Config-driven |

---

## 8. Experiment Execution Protocol

### 8.1 Execution Order

```
Phase 1: Preprocessing
├── Load raw data
├── Apply preprocessing pipeline (fit on train only)
├── Create train/val/test splits
└── Save processed data artifacts

Phase 2: Experiments (Loop)
├── For each task in [binary, multi]:
│   ├── For each model in [lr, rf, xgb]:
│   │   ├── For each strategy in [s0, s1, s2a]:
│   │   │   ├── Apply strategy to training data
│   │   │   ├── Train model
│   │   │   ├── Evaluate on test set
│   │   │   ├── Compute all metrics
│   │   │   ├── Save artifacts
│   │   │   └── Log to experiment_log

Phase 3: Analysis
├── Aggregate results
├── Generate comparison tables
├── Create visualizations
└── Compile final report
```

### 8.2 Error Handling

| Error Type | Handling | Recovery |
|------------|----------|----------|
| Memory Error (SMOTE) | Skip S2b, log warning | Continue with other experiments |
| Convergence Warning (LR) | Increase max_iter, retry | Log warning, proceed |
| File I/O Error | Retry 3 times | Skip and log error |
| Unexpected Exception | Log full traceback | Skip experiment, continue grid |

### 8.3 Checkpointing

- Save processed data after preprocessing
- Save metrics JSON immediately after each experiment
- Experiment log updated after each experiment
- Support resuming from failed point

---

## 9. Tracking & Monitoring

### 9.1 Progress Tracking

```
Experiment Progress: [████████████████████░░░░░░░░░░] 18/18 (100%)
Current: multi_xgb_s2a
Elapsed: 02:45:30
ETA: 00:15:00
```

### 9.2 Real-time Metrics Dashboard (Optional)

| Experiment | Status | Accuracy | Macro F1 | G-Mean |
|------------|--------|----------|----------|--------|
| binary_lr_s0 | ✅ Done | 0.87 | 0.85 | 0.86 |
| binary_lr_s1 | ✅ Done | 0.86 | 0.86 | 0.86 |
| ... | ... | ... | ... | ... |
| multi_xgb_s2a | 🔄 Running | - | - | - |

---

## 10. Documentation Requirements

### 10.1 Per-Experiment Documentation

Each experiment must include:
- Exact hyperparameters used
- Training data shape (after resampling if applicable)
- Execution time
- Any warnings or issues encountered

### 10.2 Final Documentation

- Complete methodology description
- Experiment grid summary
- Results tables (binary and multiclass)
- Rare class analysis section
- Limitations and future work

---

## Appendix A: Experiment ID Naming Convention

```
{task}_{model}_{strategy}
```

| Component | Values | Example |
|-----------|--------|---------|
| task | binary, multi | binary |
| model | lr, rf, xgb | rf |
| strategy | s0, s1, s2a, s2b | s1 |

**Examples:**
- `binary_lr_s0` - Binary Logistic Regression, No balancing
- `multi_rf_s1` - Multiclass Random Forest, Class weighting
- `binary_xgb_s2a` - Binary XGBoost, RandomOverSampler

---

## Appendix B: Expected Results Summary Template

### Binary Classification Results

| Model | Strategy | Accuracy | Macro F1 | G-Mean | ROC-AUC |
|-------|----------|----------|----------|--------|---------|
| LR | S0 | | | | |
| LR | S1 | | | | |
| LR | S2a | | | | |
| RF | S0 | | | | |
| RF | S1 | | | | |
| RF | S2a | | | | |
| XGB | S0 | | | | |
| XGB | S1 | | | | |
| XGB | S2a | | | | |

### Multiclass Rare Class Recall

| Model | Strategy | Worms | Shellcode | Backdoor | Analysis |
|-------|----------|-------|-----------|----------|----------|
| LR | S0 | | | | |
| LR | S1 | | | | |
| LR | S2a | | | | |
| RF | S0 | | | | |
| RF | S1 | | | | |
| RF | S2a | | | | |
| XGB | S0 | | | | |
| XGB | S1 | | | | |
| XGB | S2a | | | | |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-17 | Initial contract |
| 2.0 | 2026-01-17 | Complete enhancement with hyperparameters, metrics, and protocols |
