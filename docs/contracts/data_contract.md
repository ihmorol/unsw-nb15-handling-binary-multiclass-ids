# Data Contract: UNSW-NB15

> **Document Version:** 2.0  
> **Last Updated:** 2026-01-17  
> **Status:** APPROVED

---

## 1. Source System

| Attribute | Value |
|-----------|-------|
| **Dataset** | UNSW-NB15 |
| **Provider** | Australian Centre for Cyber Security (ACCS) |
| **Creation Tool** | IXIA PerfectStorm |
| **Format** | CSV (comma-separated) |
| **Encoding** | UTF-8 |
| **Location** | Local Storage (`dataset/`) |

### 1.1 Source Files

| File | Path | Size | Records |
|------|------|------|---------|
| Training Set | `dataset/UNSW_NB15_training-set.csv` | ~32 MB | 175,341 |
| Testing Set | `dataset/UNSW_NB15_testing-set.csv` | ~15 MB | 82,332 |
| Feature List | `dataset/NUSW-NB15_features.csv` | ~4 KB | Feature descriptions |

---

## 2. Schema Definition

The dataset must contain the following features (after header normalization):

### 2.1 Identifiers (To Be Removed in Modeling)

| Feature | Type | Description | Removal Reason |
|---------|------|-------------|----------------|
| `id` | Integer | Row identifier | Non-predictive |
| `srcip` | String | Source IP Address | High cardinality, prevents generalization |
| `dstip` | String | Destination IP Address | High cardinality, prevents generalization |
| `sport` | Integer | Source Port | Configuration noise |
| `dsport` | Integer | Destination Port | Configuration noise |
| `stime` | Float | Start Timestamp | Temporal leakage |
| `ltime` | Float | Last Timestamp | Temporal leakage |

> [!NOTE]
> These 7 columns are **already removed** in the source dataset files used for this project (`dataset/UNSW_NB15_training-set.csv` and `testing-set.csv`). The code does not need to drop them explicitly as they are absent. The remaining 42 predictive features are retained.

### 2.2 Categorical Features (One-Hot Encoded)

| Feature | Description | Unique Values | Encoding |
|---------|-------------|---------------|----------|
| `proto` | Network Protocol | ~130 (tcp, udp, etc.) | OneHotEncoder |
| `state` | Connection State | ~15 (FIN, CON, INT, etc.) | OneHotEncoder |
| `service` | Application Service | ~13 (http, ftp, dns, etc.) | OneHotEncoder |

**Post-Encoding Dimensionality:**

| Feature | Original Columns | After One-Hot |
|---------|------------------|---------------|
| proto | 1 | ~130 |
| state | 1 | ~15 |
| service | 1 | ~13 |
| **Total Categorical** | 3 | **~158** |

### 2.3 Numerical Features (Scaled)

#### Flow Features
| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| `dur` | Duration of connection | Float | 0 - 60+ |
| `sbytes` | Source to destination bytes | Integer | 0 - 10M+ |
| `dbytes` | Destination to source bytes | Integer | 0 - 10M+ |
| `sttl` | Source to destination TTL | Integer | 0 - 255 |
| `dttl` | Destination to source TTL | Integer | 0 - 255 |
| `sloss` | Source packets retransmitted/dropped | Integer | 0 - 1000+ |
| `dloss` | Destination packets retransmitted/dropped | Integer | 0 - 1000+ |
| `Sload` | Source bits per second | Float | 0 - 1B+ |
| `Dload` | Destination bits per second | Float | 0 - 1B+ |
| `Spkts` | Source to destination packet count | Integer | 0 - 100K+ |
| `Dpkts` | Destination to source packet count | Integer | 0 - 100K+ |

#### Window Features
| Feature | Description | Type |
|---------|-------------|------|
| `swin` | Source TCP window advertisement | Integer |
| `dwin` | Destination TCP window advertisement | Integer |
| `stcpb` | Source TCP base sequence number | Integer |
| `dtcpb` | Destination TCP base sequence number | Integer |

#### Derived/Statistical Features
| Feature | Description | Type |
|---------|-------------|------|
| `smeansz` | Mean of packet size transmitted by src | Integer |
| `dmeansz` | Mean of packet size transmitted by dst | Integer |
| `trans_depth` | Connection depth (pipelined) | Integer |
| `res_bdy_len` | Response body length (HTTP) | Integer |
| `Sjit` | Source jitter | Float |
| `Djit` | Destination jitter | Float |
| `sintpkt` | Source interpacket arrival time | Float |
| `dintpkt` | Destination interpacket arrival time | Float |
| `tcprtt` | TCP connection round-trip time | Float |
| `synack` | Time between SYN and SYN-ACK | Float |
| `ackdat` | Time between SYN-ACK and ACK | Float |

#### Binary Indicator Features
| Feature | Description | Values |
|---------|-------------|--------|
| `is_sm_ips_ports` | If source/dest IPs and ports are equal | 0, 1 |
| `is_ftp_login` | If FTP session has login | 0, 1 |

#### Connection Count Features
| Feature | Description | Type |
|---------|-------------|------|
| `ct_state_ttl` | Count of same state/TTL connections | Integer |
| `ct_flw_http_mthd` | Count of HTTP methods | Integer |
| `ct_ftp_cmd` | Count of FTP commands | Integer |
| `ct_srv_src` | Count connections same service/src | Integer |
| `ct_srv_dst` | Count connections same service/dst | Integer |
| `ct_dst_ltm` | Count connections to dst in last 100 | Integer |
| `ct_src_ltm` | Count connections from src in last 100 | Integer |
| `ct_src_dport_ltm` | Count same src/dst port in last 100 | Integer |
| `ct_dst_sport_ltm` | Count same dst/src port in last 100 | Integer |
| `ct_dst_src_ltm` | Count same dst/src in last 100 | Integer |

**Total Numerical Features:** 36

---

### 2.4 Target Labels

| Label | Type | Description | Values |
|-------|------|-------------|--------|
| `label` | Binary | Attack presence | 0 = Normal, 1 = Attack |
| `attack_cat` | Multiclass | Attack category | 10 classes (see below) |

#### 2.4.1 Attack Category Mapping

| Index | Category | Description | Training Count | Test Count | Total % |
|-------|----------|-------------|----------------|------------|---------|
| 0 | Normal | Benign traffic | 56,000 | 37,000 | 36.10% |
| 1 | Fuzzers | Protocol fuzzing attacks | 18,184 | 6,062 | 9.41% |
| 2 | Analysis | Port scan, spam, HTML attacks | 2,000 | 677 | 1.04% |
| 3 | Backdoor | Persistent access channels | 1,746 | 583 | 0.90% |
| 4 | DoS | Denial of service | 12,264 | 4,089 | 6.35% |
| 5 | Exploits | Known vulnerability exploits | 33,393 | 11,132 | 17.28% |
| 6 | Generic | Generic attack patterns | 40,000 | 18,871 | 22.85% |
| 7 | Reconnaissance | Network probing | 10,491 | 3,496 | 5.43% |
| 8 | Shellcode | Executable code injection | 1,133 | 378 | 0.59% |
| 9 | Worms | Self-replicating malware | 130 | 44 | 0.07% |

---

## 3. Class Distribution Analysis

### 3.1 Binary Classification Distribution

| Class | Training | Testing | Total |
|-------|----------|---------|-------|
| Normal (0) | 56,000 (31.94%) | 37,000 (44.94%) | 93,000 (36.10%) |
| Attack (1) | 119,341 (68.06%) | 45,332 (55.06%) | 164,673 (63.90%) |
| **Total** | **175,341** | **82,332** | **257,673** |

**Imbalance Ratio (Binary):** Normal : Attack = 1 : 1.77

### 3.2 Multiclass Classification Distribution

#### Training Set Distribution

```
 Normal        ██████████████████████████████████  56,000 (31.94%)
 Generic       ████████████████████████████        40,000 (22.82%)
 Exploits      ██████████████████████              33,393 (19.04%)
 Fuzzers       ████████████                        18,184 (10.37%)
 DoS           ████████                            12,264 (6.99%)
 Reconnaissance██████                              10,491 (5.98%)
 Analysis      █                                    2,000 (1.14%)
 Backdoor      █                                    1,746 (1.00%)
 Shellcode     █                                    1,133 (0.65%)
 Worms         ▎                                      130 (0.07%)
```

#### Imbalance Ratios (Relative to Worms)

| Class | Count | Ratio to Worms | Category |
|-------|-------|----------------|----------|
| Normal | 56,000 | 430.8x | Majority |
| Generic | 40,000 | 307.7x | Majority |
| Exploits | 33,393 | 256.9x | Majority |
| Fuzzers | 18,184 | 139.9x | Majority |
| DoS | 12,264 | 94.3x | Moderate |
| Reconnaissance | 10,491 | 80.7x | Moderate |
| Analysis | 2,000 | 15.4x | **Rare** |
| Backdoor | 1,746 | 13.4x | **Rare** |
| Shellcode | 1,133 | 8.7x | **Rare** |
| **Worms** | **130** | **1.0x** | **Critically Rare** |

> [!WARNING]
> **Extreme Imbalance Warning:**  
> - Worms class has only 130 training samples (0.07%)
> - Without imbalance handling, models will likely predict 0% recall for Worms
> - Special attention required during SMOTE (k_neighbors > 5 may fail)

---

## 4. Data Quality Expectations

### 4.1 Missing Values

| Feature Type | Expected Missing | Handling Strategy |
|--------------|------------------|-------------------|
| Numerical | ≤ 1% | Median Imputation |
| Categorical | ≤ 0.1% | "missing" token |

**Implementation:**
```python
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
```

### 4.2 Data Types

| Column Group | Expected Type | Validation |
|--------------|---------------|------------|
| Identifiers | int/string | Drop before modeling |
| Categoricals | string | Check for novel categories |
| Numericals | float64/int64 | Check for infinity/NaN |
| Labels | int/string | Validate against known categories |

### 4.3 Split Integrity

| Constraint | Requirement |
|------------|-------------|
| Official Split | Training and Test sets must NOT be merged and re-split |
| Benchmark Compatibility | Official split enables comparison with published results |
| Validation Set | Created from Training Set (20%, stratified) |
| Test Set Isolation | NEVER touched during preprocessing, tuning, or balancing |

---

## 5. Derived Tables (Preprocessing Output)

### 5.1 Output Artifacts

| Artifact | Format | Contents |
|----------|--------|----------|
| `X_train_processed.npz` | NumPy sparse/dense | Preprocessed training features |
| `X_val_processed.npz` | NumPy sparse/dense | Preprocessed validation features |
| `X_test_processed.npz` | NumPy sparse/dense | Preprocessed test features |
| `y_train_binary.npy` | NumPy array | Binary labels (training) |
| `y_val_binary.npy` | NumPy array | Binary labels (validation) |
| `y_test_binary.npy` | NumPy array | Binary labels (test) |
| `y_train_multi.npy` | NumPy array | Multiclass labels (training) |
| `y_val_multi.npy` | NumPy array | Multiclass labels (validation) |
| `y_test_multi.npy` | NumPy array | Multiclass labels (test) |
| `feature_names.json` | JSON | List of feature names after encoding |
| `label_mapping.json` | JSON | Multiclass label to integer mapping |
| `preprocessing_metadata.json` | JSON | Scaler means, encoder categories, etc. |

### 5.2 Expected Dimensions

| Artifact | Rows | Columns (approx.) |
|----------|------|-------------------|
| X_train | 140,273 | ~196 |
| X_val | 35,068 | ~196 |
| X_test | 82,332 | ~196 |
| y_* | Same as X | 1 |

### 5.3 After Resampling (S2 Strategy)

| Strategy | Task | X_train Rows |
|----------|------|--------------|
| S0 | Both | 140,273 |
| S1 | Both | 140,273 |
| S2a (ROS) | Binary | ~224,000 |
| S2a (ROS) | Multi | ~560,000 (all classes = max class) |
| S2b (SMOTE) | Binary | ~224,000 |
| S2b (SMOTE) | Multi | ~560,000 |

---

## 6. Data Leakage Prevention Protocol

> [!CAUTION]
> **Critical Constraints - Violation Invalidates Results**

### 6.1 Preprocessing Leakage Prevention

| Step | Constraint | Validation |
|------|------------|------------|
| Imputation | Fit ONLY on X_train | Check imputer fitted before transform |
| Scaling | Fit ONLY on X_train | Check scaler.mean_ from train only |
| Encoding | Learn categories from X_train ONLY | Check encoder.categories_ source |
| Label Encoding | Fit on training labels | Same for all splits |

### 6.2 Resampling Leakage Prevention

| Constraint | Validation |
|------------|------------|
| Apply ONLY to X_train, y_train | Never call fit_resample on val/test |
| Synthetic samples ONLY in training | Val/test row counts unchanged |
| Cross-validation: apply inside fold | Use `Pipeline` with `imblearn` |

### 6.3 Validation Checklist

```python
# Run these assertions before training
assert X_val.shape[0] == ORIGINAL_VAL_SIZE, "Validation set modified!"
assert X_test.shape[0] == ORIGINAL_TEST_SIZE, "Test set modified!"
assert not np.isnan(X_train).any(), "NaN values in training!"
assert not np.isinf(X_train).any(), "Inf values in training!"
```

---

## 7. Ownership & SLAs

| Role | Responsibility |
|------|----------------|
| **Owner** | Data Engineering Agent (Preprocessing) |
| **Consumer** | Model Training Agent |
| **Validator** | QA Agent |

### 7.1 Change Management

| Change Type | Required Action |
|-------------|----------------|
| Schema Change (add/remove feature) | Version bump (major) |
| Imputation Strategy Change | Version bump (minor) |
| Split Ratio Change | Version bump (minor) |
| Bug Fix | Version bump (patch) |

### 7.2 Versioning Protocol

```
data_contract_v{major}.{minor}.{patch}.md
```

**Current Version:** 2.0.0

---

## 8. Usage in Experiment Grid

This exact processed dataset is the **single consistent input** for all 18 experiments:

| Task | Model | Strategy | Data Source |
|------|-------|----------|-------------|
| Binary | LR, RF, XGB | S0 | X_train (raw), y_train_binary |
| Binary | LR, RF, XGB | S1 | X_train + class_weight |
| Binary | LR, RF, XGB | S2 | X_train_resampled, y_train_resampled |
| Multi | LR, RF, XGB | S0 | X_train (raw), y_train_multi |
| Multi | LR, RF, XGB | S1 | X_train + sample_weight |
| Multi | LR, RF, XGB | S2 | X_train_resampled, y_train_resampled |

---

## Appendix A: Feature List Reference

### A.1 Complete Feature List After ID Removal (42 features)

```
1.  dur           22. ct_state_ttl
2.  proto         23. ct_flw_http_mthd
3.  state         24. is_ftp_login
4.  service       25. ct_ftp_cmd
5.  sbytes        26. ct_srv_src
6.  dbytes        27. ct_srv_dst
7.  sttl          28. ct_dst_ltm
8.  dttl          29. ct_src_ltm
9.  sloss         30. ct_src_dport_ltm
10. dloss         31. ct_dst_sport_ltm
11. Sload         32. ct_dst_src_ltm
12. Dload         33. swin
13. Spkts         34. dwin
14. Dpkts         35. stcpb
15. smeansz       36. dtcpb
16. dmeansz       37. Sjit
17. trans_depth   38. Djit
18. res_bdy_len   39. tcprtt
19. sintpkt       40. synack
20. dintpkt       41. ackdat
21. is_sm_ips_ports 42. (targets: label, attack_cat)
```

---

## 9. Feature Correlation Audit

> [!NOTE]
> All 42 predictive features are retained by design. This section documents the rationale.

### 9.1 Multicollinearity Assessment

| Feature Pair | Correlation | Action | Rationale |
|--------------|-------------|--------|-----------|
| `sbytes` ↔ `Sload` | 0.85+ | Retain Both | Capture different attack signatures |
| `sttl` ↔ `dttl` | 0.70+ | Retain Both | TTL asymmetry indicates spoofing |
| `sintpkt` ↔ `dintpkt` | 0.60+ | Retain Both | Jitter patterns differ by attack type |

### 9.2 Feature Retention Justification

**Why retain all 42 features?**
1. **Rare Class Signal Preservation:** Aggressive feature selection optimizes for majority variance, potentially discarding subtle signals critical for Worms (130 samples) and Shellcode (1,133 samples).
2. **Model Robustness:** Tree-based models (RF, XGB) naturally handle correlated features via feature importance.
3. **Reproducibility:** Consistent feature set across all 18 experiments ensures fair comparison.

---

## 10. Statistical Validation Requirements

### 10.1 Uncertainty Quantification

| Metric | Method | Parameters |
|--------|--------|------------|
| Macro-F1 | Bootstrap CI | 1000 iterations, 95% CI |
| G-Mean | Bootstrap CI | 1000 iterations, 95% CI |
| Per-Class Recall | Bootstrap CI | 1000 iterations, 95% CI |

### 10.2 Significance Testing

| Comparison Type | Test | Correction |
|-----------------|------|------------|
| Paired Models (same data) | McNemar's Test | Bonferroni (α = 0.05/18) |
| Strategy Comparison | Wilcoxon Signed-Rank | Bonferroni |
| Effect Size | Cohen's κ | Report with CI |

### 10.3 Required Statistical Artifacts

```yaml
statistical_outputs:
  - "results/tables/metric_confidence_intervals.csv"
  - "results/tables/paired_significance_tests.csv"
  - "results/tables/effect_sizes.csv"
  - "results/tables/rare_class_ci.csv"
```

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-17 | Initial contract |
| 2.0 | 2026-01-17 | Complete enhancement with class distributions, leakage protocols |
| 3.0 | 2026-01-18 | Added §9 Feature Correlation Audit, §10 Statistical Validation Requirements |
