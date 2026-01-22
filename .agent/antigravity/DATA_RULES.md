# Data Rules (Leakage & Integrity)

UNSW-NB15 is imbalanced; leakage can inflate scores and invalidate conclusions.

## Split Integrity

- Use **official UNSW-NB15 training split** and **official test split**.
- Create validation split (20%, stratified) **only from training**.
- Never mix train and test, then re-split.

## Dropped Columns (7 Total)

Remove before modeling:
- `id` - Row identifier
- `srcip`, `dstip` - IP addresses (high cardinality)
- `sport`, `dsport` - Ports (configuration noise)
- `stime`, `ltime` - Timestamps (temporal leakage)

**Result:** 42 predictive features retained.

## Preprocessing Integrity

| Step | Fit On | Transform On |
|------|--------|--------------|
| Imputation (median/missing) | Train only | Train, Val, Test |
| One-Hot Encoding | Train only | Train, Val, Test |
| StandardScaler | Train only | Train, Val, Test |

## Resampling Integrity

- SMOTE/RandomOverSampler applies **ONLY to training split**.
- Never resample validation or test.
- Never compute class weights using validation/test.

## Forbidden Actions

1. Combining train+test then re-splitting.
2. Feature engineering that uses labels (target leakage).
3. Using test performance for early stopping, tuning, or selection.
4. Peeking at test distribution before preprocessing is frozen.

## Required Metadata

Store in: `results/processed/preprocessing_metadata.json`

Contents:
- dropped columns
- categorical columns encoded
- scaler type and parameters
- random seed
- split sizes and class distributions

## Rare Classes

| Class | Training Samples | Percentage |
|-------|------------------|------------|
| Worms | 130 | 0.07% |
| Shellcode | 1,133 | 0.65% |
| Backdoor | 1,746 | 1.00% |
| Analysis | 2,000 | 1.14% |

> **Warning:** k_neighbors for SMOTE may fail for Worms. Use RandomOverSampler (S2a) as default.

## Authoritative Source

Full schema and distribution details: `docs/contracts/data_contract.md`
