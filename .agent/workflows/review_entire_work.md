---
description: Comprehensive Multi-Perspective Project Review (Author, Auditor, Architect, Statistician, Area Chair)
---

# Workflow: Review Entire Work

This workflow orchestrates a full "World-Class" review of the project by simulating five distinct expert personas.

## Phase 1: Implementation Audit (Role: Auditor)
**Reference Workflow:** `/strategic_audit`
**Focus:** Contracts & Standards
**Tasks:**
1.  **Contract Check**: Compare `docs/contracts/` against `src/`. Ensure absolute compliance.
2.  **Sanity Check**: Verify if `random_state` is fixed everywhere (Configs vs Code).
3.  **Artifact Check**: Check `results/` artifacts. Do they match the `experiment_contract.md`?

## Phase 1.5: Pipeline Integrity & Alignment (Role: Architect)
**Focus:** Code Structure & Data Flow
**Tasks:**
1.  **End-to-End Trace**: Follow the data journey from `dataset/` → `loader.py` → `preprocessing.py` → `trainer.py` → `results/`. Are there broken links or unused files?
2.  **Interface Consistency**: Do the outputs of the Preprocessor (e.g., sparse matrices, array shapes) strictly match what the Models expect?
3.  **Logic Alignment**: Does the actual python code in `src/pipelines` match the theoretical flow described in the documentation?
4.  **Hidden Side-Effects**: Look for "spaghetti code" or global state mutations that might cause invisible bugs (e.g., modifying dataframes in place unexpectedly).

## Phase 2: Statistical Rigor Check (Role: Reviewer)
**Reference Workflow:** `/reviewer`
**Focus:** Numbers & Validity
**Tasks:**
1.  Analyze `results/tables/aggregated_summary.csv` and `results/experiment_log.csv`.
2.  **Critical Check**: Are `Seed_count` values > 1? Are there Confidence Intervals or Standard Deviations?
3.  **Significance Check**: Do the "State-of-the-Art" claims hold up? If `std` is missing or high, flag it immediately.
4.  **Hard Truth**: Are the improvements statistically significant or just random noise?

## Phase 3: Paper Narrative Review (Role: Author)
**Reference Workflow:** `/author` (or `/paper_review`)
**Focus:** Storytelling & Claims
**Tasks:**
1.  Read the latest PDF/TeX draft in `paper/` or the current `task.md` / `README.md`.
2.  **Alignment Check**: Does the narrative align with the Phase 1, 1.5 & 2 findings? (e.g., claiming "Robust" when N=1).
3.  **Mitigation Strategy**: If results are weak, propose specific pivots for the paper's contribution.

## Phase 4: Meta-Review & Decision (Role: Area Chair)
**Focus:** Final Decision & Strategy
**Tasks:**
1.  **Synthesis**: Review the findings from Phases 1-3.
2.  **Critique the Reviewers**: Was the Auditor too pedantic? Did the Architect miss a practical workaround?
3.  **Final Verdict**: Provide a formal decision: **Strong Reject**, **Reject**, **Weak Accept**, **Accept**.
4.  **Action Plan**: Generate a prioritized, bulleted list of immediate actions to fix Critical Flaws and move the paper to "Accept".
