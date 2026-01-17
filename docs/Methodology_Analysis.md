# State of the Art Analysis: Intrusion Detection on UNSW-NB15

> **Document Version:** 2.0  
> **Last Updated:** 2026-01-17  
> **Status:** APPROVED FOR IMPLEMENTATION

---

## 1. Introduction

This document provides a comprehensive analysis of the proposed methodology for developing an Intrusion Detection System (IDS) using the UNSW-NB15 dataset. The analysis compares the proposed approach against current State of the Art (SOTA) research to ensure scientific rigor, relevance, and contribution significance.

### 1.1 Research Objectives

| Objective | Description | Priority |
|-----------|-------------|----------|
| **O1** | Build binary IDS models (Normal vs Attack) | HIGH |
| **O2** | Build multiclass IDS models (Normal + 9 attacks) | HIGH |
| **O3** | Systematic comparison of 3 imbalance strategies | HIGH |
| **O4** | Per-class analysis focusing on rare attacks | HIGH |
| **O5** | Reproducible baseline pipeline | MEDIUM |

### 1.2 Research Questions

- **RQ1:** How does class imbalance in UNSW-NB15 affect the performance of classical ML models on binary vs multiclass intrusion detection tasks?
- **RQ2:** To what extent do class weighting and oversampling improve detection of minority attack classes compared to raw imbalanced data?
- **RQ3:** Is there a consistent pattern in how different models (LR, RF, XGB) respond to imbalance-handling methods across binary and multiclass tasks?
- **RQ4:** For extremely rare classes (Worms, Shellcode), does oversampling significantly improve recall without degrading majority class performance?

---

## 2. Dataset Context: UNSW-NB15

### 2.1 Dataset Overview

The UNSW-NB15 dataset is a modern benchmark for Network Intrusion Detection Systems (NIDS), developed by the Australian Centre for Cyber Security (ACCS) using the IXIA PerfectStorm toolset. It supersedes legacy datasets (KDD99, NSL-KDD) by reflecting contemporary threat landscapes with nine distinct attack families.

**Key Statistics:**

| Metric | Training Set | Testing Set | Total |
|--------|--------------|-------------|-------|
| Total Records | 175,341 | 82,332 | 257,673 |
| Normal | 56,000 (31.9%) | 37,000 (44.9%) | 93,000 |
| Attack | 119,341 (68.1%) | 45,332 (55.1%) | 164,673 |
| Features | 42 (after ID removal) | 42 | 42 |

### 2.2 Class Distribution Analysis

#### 2.2.1 Multiclass Distribution (Attack Categories)

| Attack Category | Training Count | Testing Count | Training % | Imbalance Ratio |
|-----------------|----------------|---------------|------------|-----------------|
| Normal | 56,000 | 37,000 | 31.94% | 1.00 (baseline) |
| Generic | 40,000 | 18,871 | 22.82% | 0.71 |
| Exploits | 33,393 | 11,132 | 19.04% | 0.60 |
| Fuzzers | 18,184 | 6,062 | 10.37% | 0.32 |
| DoS | 12,264 | 4,089 | 6.99% | 0.22 |
| Reconnaissance | 10,491 | 3,496 | 5.98% | 0.19 |
| Analysis | 2,000 | 677 | 1.14% | 0.04 |
| Backdoor | 1,746 | 583 | 1.00% | 0.03 |
| Shellcode | 1,133 | 378 | 0.65% | 0.02 |
| **Worms** | **130** | **44** | **0.07%** | **0.002** |

> [!WARNING]
> **Extreme Imbalance Alert:** Worms class represents only 0.07% of training data (130 samples). Traditional ML models will likely achieve 0% recall on this class without intervention.

#### 2.2.2 Rare Class Identification

Classes requiring special analysis (< 3% of training data):

| Class | Training Samples | Category |
|-------|------------------|----------|
| Worms | 130 | **Critically Rare** |
| Shellcode | 1,133 | **Rare** |
| Backdoor | 1,746 | **Rare** |
| Analysis | 2,000 | **Moderately Rare** |

### 2.3 SOTA Alignment

- **Dataset Selection:** UNSW-NB15 is standard practice in recent literature (2019-2025). Older datasets are correctly discarded.
- **Challenge:** The dataset is effectively "solved" for binary classification (High Accuracy > 99%), but remains challenging for **multiclass classification** of rare events.

---

## 3. Methodology Assessment

### 3.1 Problem Formulation

**Proposed:** Dual-track approach (Binary vs. Multiclass) with specific focus on Class Imbalance.

**Analysis:**

| Aspect | Strength | Gap Addressed |
|--------|----------|---------------|
| Separation of tasks | Allows identifying where "Overall Accuracy" masks failures | Many papers only report high binary accuracy |
| Focus on reliability | More operationally relevant than marginal accuracy gains | SOTA papers chase 99.5% → 99.6% |
| Per-class analysis | Exposes detection failures on rare attacks | Most papers use Weighted F1 dominated by majority |

### 3.2 Data Preprocessing Pipeline

**Proposed Pipeline:**

```
Raw Data → Feature Cleaning → Missing Value Handling → Label Preparation
         → Encoding → Scaling → Train/Val/Test Split → Ready for Modeling
```

#### 3.2.1 Feature Cleaning and Selection

| Step | Action | Justification |
|------|--------|---------------|
| Drop `id` | Row identifier, non-predictive | Standard practice |
| Drop `srcip`, `dstip` | High cardinality, not generalizable | Prevents overfitting to specific IPs |
| Drop `sport`, `dsport` | Contains ports, can be kept as numeric but dropped for simplicity | Reduces noise |
| Drop `stime`, `ltime` | Timestamps, not generalizable | Prevents temporal leakage |
| **Retain all 42 features** | All predictive features kept | Maximizes information for rare class detection |

> [!NOTE]
> **Decision Rationale:** Aggressive feature selection often prioritizes features explaining majority variance, potentially discarding subtle signals for rare attacks. All 42 predictive features are retained.

#### 3.2.2 Missing Value Handling

| Feature Type | Strategy | Implementation |
|--------------|----------|----------------|
| Numerical | Median Imputation | `SimpleImputer(strategy='median')` |
| Categorical | "missing" token | `SimpleImputer(strategy='constant', fill_value='missing')` |

#### 3.2.3 Label Preparation

**Binary Classification:**
```python
y_binary = (attack_cat != 'Normal').astype(int)  # 0 = Normal, 1 = Attack
```

**Multiclass Classification:**
```python
label_encoder = LabelEncoder()
y_multi = label_encoder.fit_transform(attack_cat)  # 0-9 classes
```

#### 3.2.4 Encoding and Scaling

| Feature Type | Encoding | Implementation |
|--------------|----------|----------------|
| Categorical (`proto`, `state`, `service`) | One-Hot Encoding | `OneHotEncoder(handle_unknown='ignore', sparse=False)` |
| Numerical (all others) | StandardScaler | `StandardScaler()` |

**Expected Dimensionality After Encoding:**

| Stage | Feature Count |
|-------|---------------|
| Original (after drops) | 42 |
| After One-Hot | ~196 (varies based on cardinality) |

#### 3.2.5 Data Splitting Strategy

```
Official Train Set (175,341) ─┬─→ Training (80%): 140,273
                              └─→ Validation (20%): 35,068 (Stratified)

Official Test Set (82,332) ───→ Test (100%): 82,332 (UNTOUCHED until final evaluation)
```

> [!CAUTION]
> **Data Leakage Prevention:**
> 1. Preprocessing statistics (mean, std, categories) computed on TRAINING ONLY
> 2. Resampling (SMOTE/ROS) applied ONLY to training split
> 3. Test set NEVER used for any tuning or validation

---

### 3.3 Imbalance Handling Strategies

#### Strategy Matrix

| Strategy ID | Name | Description | When Applied |
|-------------|------|-------------|--------------|
| **S0** | No Balancing (Baseline) | Raw imbalanced data | Training |
| **S1** | Class Weighting | Inverse frequency weights | During model training |
| **S2a** | Random Oversampling | Duplicate minority samples | Training data only |
| **S2b** | SMOTE | Synthetic minority oversampling | Training data only (optional) |

#### 3.3.1 S0: No Balancing (Baseline)

- **Purpose:** Establish baseline performance showing how models fail on minority classes
- **Expected Outcome:** High accuracy, near-zero recall for Worms/Shellcode
- **Implementation:** No modification to training data

#### 3.3.2 S1: Class Weighting

**Binary Classification:**
```python
class_weight = 'balanced'  # Automatically computes n_samples / (n_classes * np.bincount(y))
```

**Multiclass Classification:**
```python
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
sample_weight = weights[y]  # For models without class_weight param
```

**XGBoost Binary Specific:**
```python
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
```

#### 3.3.3 S2: Resampling Strategies

**S2a: RandomOverSampler**
```python
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
```

**S2b: SMOTE (Synthetic Minority Over-sampling Technique)**
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(
    k_neighbors=5,           # Number of nearest neighbors
    random_state=42,
    n_jobs=-1
)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

> [!IMPORTANT]
> **SMOTE Constraint:** For classes with < 6 samples, SMOTE will fail due to k_neighbors requirement. Use RandomOverSampler as fallback.

**Memory Considerations:**
| Strategy | Training Size (Binary) | Training Size (10-class) |
|----------|------------------------|--------------------------|
| S0 | 140,273 | 140,273 |
| S1 | 140,273 | 140,273 |
| S2a/S2b | ~224,000 (≈1.6x) | ~560,000 (≈4x) |

---

### 3.4 Model Selection

#### 3.4.1 Model Families

| Model | Family | Strengths | Weaknesses |
|-------|--------|-----------|------------|
| **Logistic Regression** | Linear | Interpretable, fast, probabilistic | Limited by linear separability |
| **Random Forest** | Bagging | Robust, handles non-linearity | Memory intensive, slower |
| **XGBoost** | Boosting | SOTA for tabular data, efficient | Requires careful tuning |

#### 3.4.2 Hyperparameter Specifications

**Logistic Regression:**
```python
LogisticRegression(
    C=1.0,                    # Regularization (tune: [0.01, 0.1, 1, 10])
    solver='saga',            # Supports L1/L2/elastic-net
    max_iter=1000,            # Sufficient for convergence
    multi_class='multinomial',# For multiclass
    class_weight='balanced',  # For S1 strategy
    n_jobs=-1,
    random_state=42
)
```

**Random Forest:**
```python
RandomForestClassifier(
    n_estimators=200,         # Tune: [100, 200]
    max_depth=20,             # Tune: [10, 20, None]
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # For S1 strategy
    n_jobs=-1,
    random_state=42
)
```

**XGBoost:**
```python
XGBClassifier(
    n_estimators=200,         # Tune: [100, 200]
    learning_rate=0.1,        # Tune: [0.01, 0.1]
    max_depth=10,             # Tune: [5, 10]
    scale_pos_weight=None,    # Computed for S1 binary
    use_label_encoder=False,
    eval_metric='mlogloss',   # For multiclass
    n_jobs=-1,
    random_state=42
)
```

#### 3.4.3 Hyperparameter Tuning Protocol

| Aspect | Specification |
|--------|---------------|
| Method | Grid Search with Stratified K-Fold |
| Folds | 3 (for time efficiency) |
| Scoring | Macro F1 (to weight minority classes equally) |
| Data Used | Validation Set ONLY |
| Best Model Selection | Highest Macro F1 on validation |

---

### 3.5 Evaluation Metrics

#### 3.5.1 Overall Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Accuracy** | (TP + TN) / Total | Baseline; misleading for imbalanced data |
| **Macro F1** | Mean of per-class F1 | Equal weight to all classes |
| **Weighted F1** | Weighted mean by support | Reflects majority class performance |
| **G-Mean** | √(Sensitivity × Specificity) | **Primary metric** for imbalanced data |
| **ROC-AUC** | Area under ROC curve | Threshold-independent performance |

**G-Mean Calculation:**

For Binary:
```python
from imblearn.metrics import geometric_mean_score
g_mean = geometric_mean_score(y_true, y_pred)
```

For Multiclass:
```python
g_mean = geometric_mean_score(y_true, y_pred, average='macro')
```

#### 3.5.2 Per-Class Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| Precision | TP / (TP + FP) | False positive control |
| Recall | TP / (TP + FN) | **Critical for rare classes** |
| F1-Score | 2 × (P × R) / (P + R) | Harmonic mean |
| Support | Count per class | Context for interpretation |

#### 3.5.3 ROC-AUC Computation

**Binary Classification:**
```python
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_true, y_pred_proba)
```

**Multiclass Classification:**
```python
roc_auc = roc_auc_score(
    y_true, 
    y_pred_proba, 
    multi_class='ovr',      # One-vs-Rest
    average='macro'         # Equal weight per class
)
```

#### 3.5.4 Confusion Matrix Specifications

| Task | Size | Key Focus |
|------|------|-----------|
| Binary | 2×2 | False Negatives (missed attacks) |
| Multiclass | 10×10 | Rare class rows (Worms, Shellcode, Backdoor, Analysis) |

---

### 3.6 Rare-Class Analysis Protocol

> [!IMPORTANT]
> This is the **core differentiator** of the study. Special attention must be given to:
> - Worms (130 training samples)
> - Shellcode (1,133 samples)
> - Backdoor (1,746 samples)
> - Analysis (2,000 samples)

#### 3.6.1 Success Thresholds

| Class | Baseline (S0) Expected Recall | Target Recall (S2) | Improvement Threshold |
|-------|------------------------------|--------------------|-----------------------|
| Worms | 0% - 5% | > 20% | +15pp |
| Shellcode | 5% - 20% | > 40% | +20pp |
| Backdoor | 10% - 30% | > 50% | +20pp |
| Analysis | 20% - 40% | > 60% | +20pp |

#### 3.6.2 Analysis Methodology

1. **Baseline Documentation:** Record exact recall for each rare class under S0
2. **Strategy Comparison:** Create comparison tables showing:
   - Recall improvement (S1 vs S0, S2 vs S0)
   - Trade-off with majority class precision
3. **Statistical Significance:** McNemar's test for paired comparisons
4. **Visualization:** Per-class recall bar charts across strategies

---

## 4. Technology Stack

| Component | Tool | Version |
|-----------|------|---------|
| Programming Language | Python | 3.10+ |
| Core ML Library | scikit-learn | 1.3+ |
| Imbalanced Learning | imbalanced-learn | 0.11+ |
| Gradient Boosting | xgboost | 2.0+ |
| Data Manipulation | pandas, numpy | Latest |
| Visualization | matplotlib, seaborn | Latest |
| Configuration | PyYAML | Latest |
| Experiment Tracking | JSON/CSV logs | N/A |

---

## 5. Strategic Recommendations

### 5.1 Implementation Priorities

| Priority | Recommendation | Rationale |
|----------|---------------|-----------|
| 1 | Strict data isolation | Prevents leakage, ensures validity |
| 2 | Document S0 baseline thoroughly | Shows severity of imbalance problem |
| 3 | Reproducibility via config | Enables fair comparison |
| 4 | Focus on per-class metrics | Core contribution of the study |

### 5.2 Experimental Best Practices

1. **Seed Control:** All random operations use `random_state=42`
2. **Checkpoint Saving:** Save model after each experiment
3. **Metric Logging:** JSON output per experiment
4. **Version Control:** Git commit after each phase

---

## 6. Conclusion

**Status:** The methodology meets all requirements for a high-quality scientific study and is ready for implementation.

**Key Differentiators:**
- Systematic comparison of 3 strategies × 3 models × 2 tasks = 18 experiments
- Explicit focus on rare attack classes (Worms, Shellcode, Backdoor, Analysis)
- G-Mean as primary metric (not accuracy)
- Transparent, reproducible pipeline

---

## Appendix A: Complete Feature List

### A.1 Features to Drop (Identifiers)

| Feature | Type | Reason for Dropping |
|---------|------|---------------------|
| `id` | Integer | Row identifier |
| `srcip` | String | Source IP - high cardinality |
| `dstip` | String | Destination IP - high cardinality |
| `sport` | Integer | Source Port |
| `dsport` | Integer | Destination Port |
| `stime` | Timestamp | Start time - temporal leakage |
| `ltime` | Timestamp | Last time - temporal leakage |

### A.2 Categorical Features (One-Hot Encoded)

| Feature | Description | Cardinality |
|---------|-------------|-------------|
| `proto` | Protocol (tcp, udp, etc.) | ~130 |
| `state` | Connection state | ~15 |
| `service` | Service type | ~13 |

### A.3 Numerical Features (Scaled)

All remaining features are treated as numerical and scaled using StandardScaler. Total: 36 features.

### A.4 Target Labels

| Label | Type | Values |
|-------|------|--------|
| `label` | Binary | 0 (Normal), 1 (Attack) |
| `attack_cat` | Multiclass | Normal, Fuzzers, Analysis, Backdoor, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms |

---

## Appendix B: Experiment Naming Convention

```
{task}_{model}_{strategy}
```

**Examples:**
- `binary_lr_s0` - Binary Logistic Regression, No balancing
- `multi_rf_s1` - Multiclass Random Forest, Class weighting
- `binary_xgb_s2a` - Binary XGBoost, RandomOverSampler
- `multi_xgb_s2b` - Multiclass XGBoost, SMOTE

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-17 | Initial analysis |
| 2.0 | 2026-01-17 | Complete enhancement with specifications |
