# Antigravity Workspace Rules

These rules define how work is done in this repo so experiments remain comparable, leak-free, and reproducible.  
They apply to code, data handling, results, and writing.

## Non-Negotiables

1. **Split first; fit preprocessing on train only; resampling on train only; validation/test untouched.**
2. Official UNSW-NB15 test split is never used for tuning, selection, or early stopping.
3. **Mandatory metrics:** Accuracy, Macro-F1, Weighted-F1, G-Mean, ROC-AUC, per-class precision/recall/F1.
4. Every claim in `docs/` or `paper/` must map to an artifact (JSON/CSV/PNG) or a citation.
5. Rare classes (Worms, Shellcode, Backdoor, Analysis) must be explicitly reported.

## Scope Lock (Do Not Expand Without Approval)

| Dimension | Values |
|-----------|--------|
| **Tasks** | Binary (Normal vs Attack), Multiclass (10-class) |
| **Models** | LR, RF, XGB |
| **Strategies** | S0 (None), S1 (Class Weight), S2a (RandomOverSampler) |

**Expected grid:** 2 tasks × 3 models × 3 strategies = **18 experiments**.

> Optional extension: S2b (SMOTE) adds 6 more experiments if enabled.

## Experiment ID Convention

```
{task}_{model}_{strategy}
```

Examples: `binary_lr_s0`, `multi_rf_s1`, `binary_xgb_s2a`

## Definition of "Done"

A change is "done" only if:

1. It passes all QA gates in `QA_GATES.md`.
2. It produces required artifacts listed in `ARTIFACT_CONTRACT.md`.
3. It updates `CHANGELOG.md` (if methodology/experiments changed).
4. It updates `results/experiment_log.csv` (if experiments were re-run).

## References

- Authoritative contracts: `docs/contracts/`
- Visualization standards: `.agent/workflows/visualization_standards.md`
- Config source of truth: `configs/main.yaml`
