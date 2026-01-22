# Changelog (Antigravity Rules)

Track changes to the `.agent/antigravity/` rules that affect experiments or methodology.

## Entry Format

```
## [YYYY-MM-DD] Title

**Change:** What was changed
**Why:** Rationale
**Impact:** What needs to be re-run or verified
**Evidence:** Related artifacts or run_ids
```

---

## [2026-01-22] Rules Alignment with Implementation

**Change:** Rewrote all 8 antigravity rule files to match actual repository implementation.

**Why:** Rules had drifted from implementation:
- S2 terminology was "SMOTE" but implementation uses "S2a (RandomOverSampler)"
- Artifact paths referenced `results/runs/<run_id>/` but implementation uses `results/metrics/{exp_id}.json`
- Processed data CSVs are optional (kept in memory)

**Impact:** No re-runs needed. This was documentation alignment only.

**Evidence:** Git commit for this change.

---

## Rules for Future Changes

1. Any change that affects experiments must include at least one new experiment run for validation.
2. Any change that affects preprocessing must bump `preprocessing_metadata.json` and re-run all experiments.
3. Strategy changes (S0/S1/S2a/S2b definitions) require updating:
   - `WORKSPACE_RULES.md`
   - `EXPERIMENT_PROTOCOL.md`
   - `configs/main.yaml`
   - `.agent/rules/member-rules.md`
