# Strategic Engineering Plan: The "Clean Slate" Doctrine (v2.0)

**Date:** 2026-01-23
**Version:** 2.0.0 (Multi-Persona Reviewed)
**Status:** PROPOSED → REVIEWED

---

## Executive Summary (Orchestrator Assessment)

**Overall Rating:** ⭐⭐⭐⭐ (4/5 - Solid, needs minor polish)

| Persona | Verdict | Key Point |
|---------|---------|-----------|
| **Orchestrator** | ✅ Approve | Strategy is sound, addresses user's "juggling" pain point |
| **Auditor** | ⚠️ Conditional | Missing formal acceptance tests |
| **Architect** | ✅ Pass | Existing `main.py` is robust enough; new script optional |
| **Reviewer** | ✅ Approve | N=5 seeds is statistically valid for CIs and t-tests |
| **Author** | ⚠️ Advice | Add "Rare Class Narrative" as explicit goal |

---

## 1. The Strategy: "Clean Slate & Full Grid"

**Objective:** Eliminate confusion, data skew, and environmental variance by executing a single, unified experiment grid from start to finish.

**Recommendation on Environment:**
- **Local (Preferred):** Run overnight. Simplest, avoids Colab artifacts.
- **Colab (Fallback):** Only if RAM < 16GB. Use strict "Zip & Download" workflow.

---

## 2. The "Golden Grid" Specification

| Dimension | Values | Count |
|-----------|--------|-------|
| Tasks | Binary, Multiclass | 2 |
| Models | LR, RF, XGB | 3 |
| Strategies | S0, S1, S2a | 3 |
| Seeds | 42, 43, 44, 45, 46 | 5 |
| **Total** | | **90** |

### Configuration Lock (`configs/main.yaml`)
```yaml
experiments:
  n_seeds: 5
  n_jobs: -1
  tasks: [binary, multi]
  models: [lr, rf, xgb]
  strategies: [s0, s1, s2a]
```

---

## 3. Execution Protocol

### Step 1: The Purge (Archive)
```bash
# Safety: Never delete, only archive
mv results results_archive_$(date +%Y%m%d_%H%M%S)
mkdir -p results/{metrics,figures,tables,logs,processed}
```

### Step 2: The Run
Option A (Simple): Use existing `main.py` with config (Recommended).
```bash
python main.py --config configs/main.yaml
```
Option B (Advanced): Create `scripts/run_golden_grid.py` with resume logic.

### Step 3: Verification (Auditor Requirement ⚠️)
**NEW:** Formal acceptance test script.
```bash
python scripts/verify_golden_grid.py
```
**Tests:**
- `T001`: 90 JSON files exist in `results/metrics/`.
- `T002`: Each JSON has keys: `experiment_id`, `task`, `model`, `strategy`, `seed`, `metrics`.
- `T003`: Preprocessing metadata checksum matches for all runs (same split).

---

## 4. The SOTA Analysis Layer

### 4.1 Statistical Validity (Critical)
- **Script:** `scripts/generate_statistics.py`
- **Outputs:** `metric_confidence_intervals.csv`, `paired_significance_tests.csv`
- **Success Criteria:** Multiclass S2a > S0 with $p < 0.05$.

### 4.2 Global Ranking (Wow Factor)
- **Script:** `scripts/generate_cd_diagram.py`
- **Outputs:** `results/figures/cd_diagram_macro_f1.png`
- **Method:** Friedman test + Nemenyi post-hoc.

### 4.3 Explainability (Differentiator)
- **Script:** `scripts/generate_shap.py`
- **Output:** `results/figures/shap_worms_summary.png`
- **Why:** Explains *why* S2a improves Worms detection.

### 4.4 Rare Class Narrative (Author Requirement ⚠️)
- **NEW:** Explicitly frame the paper around "Rare Class Detection".
- **Hook:** "Worms recall improves from 0% to >80% with simple resampling."

---

## 5. Acceptance Tests Checklist (Auditor Addition)

| Test ID | Assertion | Script |
|---------|-----------|--------|
| T001 | 90 metric JSONs exist | `verify_golden_grid.py` |
| T002 | All JSONs structurally valid | `verify_golden_grid.py` |
| T003 | Preprocessing consistent | Checksum on `preprocessing_metadata.json` |
| T004 | CIs computed for all 18 configs | `generate_statistics.py` |
| T005 | CD diagram generated | File exists check |
| T006 | SHAP plot for Worms exists | File exists check |

---

## 6. Roadmap & Timeline

| Phase | Task | Time | Owner |
|-------|------|------|-------|
| 1. Prep | Archive, verify config | 15 min | User |
| 2. Run | Execute 90 experiments | 3-6 hrs | main.py |
| 3. Verify | Acceptance tests | 5 min | verify_golden_grid.py |
| 4. Analyze | Stats, CD, SHAP | 30 min | Scripts |
| 5. Write | Update Results section | 1 hr | Author |

---

## 7. Risk Mitigation (New Section)

| Risk | Mitigation |
|------|------------|
| Run fails mid-way | `main.py` has resume logic (skips existing JSONs) |
| OOM on SMOTE | S2a (ROS) is used, not SMOTE; lower memory |
| Colab timeout | Run locally; or batch by task in Colab |
| Statistical insignificance | N=5 is sufficient for t-tests; if fails, result is still valid (honest reporting) |

---

## Recommendation
**Proceed with v2.0.** The plan is now auditor-approved with acceptance tests to catch errors early.

**Action Required:** Confirm "Purge & Run" to begin execution.
