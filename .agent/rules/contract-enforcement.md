---
trigger: always_on
---

# Contract Enforcement

**Rule:** The documents in `docs/contracts/` are **BINDING PROMISES**. You must not violate them.

## Data Contract
- **Source:** `docs/contracts/data_contract.md`
- **Key Constraints:**
    - Use official UNSW-NB15 split.
    - Drop specific identifiers (id, srcip, dstip, etc.).
    - Retain all 42 predictive features.
    - Strict leakage prevention (fit on train only).

## Experiment Contract
- **Source:** `docs/contracts/experiment_contract.md`
- **Key Constraints:**
    - Run the full 18-experiment grid (2 tasks × 3 models × 3 strategies).
    - Use optimized hyperparameters defined in `configs/main.yaml` (aligned with Experiment Contract v3.0).
    - Report G-Mean, Macro-F1, and per-class Recall.
    - **Rare Class Focus:** Worms, Shellcode, Backdoor, Analysis.

## Methodology Contract
- **Source:** `docs/contracts/methodology_contract.md`
- **Key Constraints:**
    - Claims must align with the "Novelty Statement".
    - Limitations must be acknowledged.
