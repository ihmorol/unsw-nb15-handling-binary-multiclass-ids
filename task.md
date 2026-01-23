# Task Checklist: UNSW-NB15 Imbalance Study

## Phase 1: Strategic Reset & Planning
- [x] **Project Analysis** (`/review_entire_work`)
    - [x] Statistical Audit (Found N=1 gap in Multi-class)
    - [x] Codebase Review (Found solid infrastructure)
    - [x] Methodology Review (Found need for SOTA visualization)
- [x] **Strategic Engineering Plan** (`docs/plans/strategic_engineering_plan.md`)
    - [x] Define "Golden Grid" (Single consistent configuration)
    - [x] Define Execution Strategy (Local vs Colab vs Hybrid)
    - [x] Define SOTA Artifacts (CD Diagrams, SHAP)

## Phase 2: The "Golden Grid" Execution
- [x] **Environment Setup**
    - [x] Archive/Wipe `results/` (Keep `results_old/` for safety)
    - [x] Finalize `configs/main.yaml` (N=5, explicit seeds)
- [/] **Execution (Delegate to Colab)**
    - [/] Batch 1: Binary Tasks (Fast, High Confidence)
    - [/] Batch 2: Multiclass Baselines (S0/S1)
    - [/] Batch 3: Multiclass Resampling (S2a - Computationally Expensive)
- [ ] **Validation**
    - [ ] Verify `experiment_log.csv` completeness (90+ runs)
    - [ ] Verify reproducible checksums

## Phase 3: SOTA Transformation (Analysis)
- [ ] **Statistical Validation**
    - [ ] Generate CIs and P-Values (`scripts/generate_statistics.py`)
    - [ ] Confirm "Rare Class" hypothesis
- [ ] **Advanced Visualization**
    - [ ] Critical Difference (CD) Diagram (`scripts/generate_cd_diagram.py`)
    - [ ] SHAP Analysis for Worms/Shellcode (`scripts/generate_shap.py`)
- [ ] **Paper Integration**
    - [ ] Rewrite "Results" section with statistical rigour
    - [ ] Add "Discussion" on Cost-Sensitive tradeoffs

## Phase 4: Final Polish
- [ ] Abstract & Intro Refinement
- [ ] Code Cleanup for Repository Release
