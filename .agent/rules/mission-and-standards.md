---
trigger: always_on
---

# Mission and Global Standards

**Source:** `persona/team.yaml`

## Mission
Build a clean, reproducible classical-ML baseline for binary and multiclass intrusion detection on UNSW-NB15, systematically evaluating imbalance strategies and rare attack detection using macro and per-class metrics.

## Global Standards (Non-Negotiable)
1.  **Evidence First:** No claims without an artifact (CSV/JSON) or a credible citation.
2.  **Leakage Zero Tolerance:** Split first; fit preprocessing on train only; resample/train-only; validation/test untouched; official test untouched until final.
3.  **Reproducibility:** Every run has config + seed + run_id + logs + metrics + tables.
4.  **Comparability:** Same split, same preprocessing contract, same metric definitions across runs.
5.  **Reporting:** Macro-F1 + per-class + confusion matrices; rare-class report mandatory.

## Glossary
- **Binary Task:** Normal vs Attack
- **Multiclass Task:** 10-class (Normal + 9 attack categories)
- **S0:** No imbalance handling
- **S1:** Class weighting
- **S2:** SMOTE/Resampling (train only)