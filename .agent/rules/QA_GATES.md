# QA Gates (Must Pass)

No run is accepted unless these gates pass.

## T001: Schema Validation

**Purpose:** Verify data schema matches contract.

**Checks:**
- [ ] Expected label columns exist (`label`, `attack_cat`)
- [ ] All 42 predictive features present after dropping IDs
- [ ] No unexpected NaN or Inf values

**Command:**
```python
assert set(['label', 'attack_cat']).issubset(df.columns)
assert df.drop(columns=drop_cols).shape[1] == 42 + 2  # +2 for labels
```

## T002: Split Integrity

**Purpose:** Ensure no data leakage between splits.

**Checks:**
- [ ] Train/val/test have zero row overlap
- [ ] Class distributions are stratified (within 2% of target ratio)

**Command:**
```python
train_ids = set(X_train.index)
val_ids = set(X_val.index)
test_ids = set(X_test.index)
assert len(train_ids & val_ids) == 0
assert len(train_ids & test_ids) == 0
```

## T003: Preprocessing Fit Scope

**Purpose:** Verify encoders/scalers fitted on train only.

**Checks:**
- [ ] `preprocessing_metadata.json` exists
- [ ] Metadata shows fit was on training samples only
- [ ] Val/test shapes unchanged from original

**Evidence:** `results/processed/preprocessing_metadata.json`

## T004: Resampling Scope

**Purpose:** Verify resampling applied only to training.

**Checks:**
- [ ] Val/test sample counts match original sizes
- [ ] For S2a: training samples increased (logged)
- [ ] For S0/S1: training samples unchanged

**Log verification:**
```
grep "resampled" results/logs/run_*.log
```

## T005: Metric Sanity

**Purpose:** Verify metrics computed correctly.

**Checks:**
- [ ] Macro-F1 is unweighted mean of per-class F1 scores
- [ ] Per-class support sums match test set size
- [ ] Confusion matrix row sums match actual class counts

**Command:**
```python
assert sum(metrics['per_class'][c]['support'] for c in classes) == len(y_test)
```

## T006: Artifact Contract

**Purpose:** Verify all required outputs exist.

**Checks:**
- [ ] 18 JSON files in `results/metrics/`
- [ ] 18 PNG files in `results/figures/`
- [ ] `experiment_log.csv` has 18 rows with no nulls
- [ ] 3 CSV files in `results/tables/`

**Command:**
```bash
ls results/metrics/*.json | wc -l  # Should be 18
ls results/figures/cm_*.png | wc -l  # Should be 18
```

## T007: Metric Consistency

**Purpose:** Cross-validate metrics between JSON and CSV.

**Checks:**
- [ ] `experiment_log.csv` metrics match individual JSON files
- [ ] `per_class_metrics.csv` F1 values match JSON per-class

## T008: Rare Class Coverage

**Purpose:** Verify rare class analysis exists for multiclass.

**Checks:**
- [ ] All 9 multiclass experiments have `rare_class_analysis` in JSON
- [ ] `rare_class_report.csv` contains Worms, Shellcode, Backdoor, Analysis
