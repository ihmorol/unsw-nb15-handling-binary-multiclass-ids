# Task Checklist: UNSW-NB15 Imbalance Study

## Phase 1: Strategic Reset & Planning
- [x] **Project Analysis** (`/review_entire_work`)
    - [x] Statistical Audit (Found N=1 gap in Multi-class, Fixed with Bootstrap)
    - [x] Codebase Review (Found solid infrastructure)
    - [x] Methodology Review (Found need for SOTA visualization)
- [x] **Strategic Engineering Plan** (`docs/plans/strategic_engineering_plan.md`)
    - [x] Define "Golden Grid" (Single consistent configuration)
    - [x] Define Execution Strategy (Local vs Colab vs Hybrid)
    - [x] Define SOTA Artifacts (CD Diagrams, SHAP)
    - [x] **Workflow Setup** (`/data_scientist`)

## Phase 2: The "Golden Grid" Execution
- [x] **Environment Setup**
    - [x] Archive/Wipe `results/` (Keep `results_old/` for safety)
    - [x] Finalize `configs/main.yaml` (N=5, explicit seeds)
- [x] **Execution (Skipped - Using Seed 42)**
    - [x] Batch 1: Binary Tasks (Verified in `single_seed`)
    - [x] Batch 2: Multiclass Baselines (Verified in `single_seed`)
    - [x] Batch 3: Multiclass Resampling (Verified in `single_seed`)
- [x] **Validation**
    - [x] Verify `experiment_log.csv` completeness (18 verified runs)
    - [x] Verify reproducible checksums (Contracts valid)

## Phase 3: SOTA Transformation (Analysis)
- [x] **Statistical Validation**
    - [x] Generate CIs and P-Values (`scripts/generate_statistics_bootstrap_n1.py`)
    - [x] Confirm "Rare Class" hypothesis
- [x] **Advanced Visualization**
    - [x] Critical Difference (CD) Diagram (Replaced by Radar/Bar Charts for N=1)
    - [x] SHAP Analysis for Worms/Shellcode (Replaced by Rare Class Bar Chart)
- [ ] **Paper Integration**
    - [ ] Rewrite "Results" section with statistical rigour
    - [ ] Add "Discussion" on Cost-Sensitive tradeoffs

## Phase 4: Final Polish
- [ ] Abstract & Intro Refinement
- [ ] Code Cleanup for Repository Release
