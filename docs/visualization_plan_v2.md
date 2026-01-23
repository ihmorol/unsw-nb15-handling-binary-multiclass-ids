# 📊 Visualization Implementation Plan (Comprehensive)

**Objective**: Generate **all possible** publication-quality visualizations from the existing experimental data, strictly adhering to `visualization_standards.md`.

**Data Sources**:
- `results/tables/final_summary_tables.csv`: Overall metrics (Acc, F1, G-Mean, AUC).
- `results/tables/per_class_metrics.csv`: Granular per-class performance.
- `results/tables/rare_class_report.csv`: Targeted minority class analysis.
- `results/experiment_log.csv`: Training times and metadata.

## 1. Summary & Comparison Plots
**Goal**: High-level ranking of strategies.

### 1.1 Performance Metrics Comparison (Grouped Bar)
- **Data**: `final_summary_tables.csv`
- **X-Axis**: Models (LR, RF, XGB)
- **Grouping**: Strategy (S0, S1, S2a)
- **Metrics**:
    - Plot A: Macro-F1 Score
    - Plot B: G-Mean Score (Geometric Mean of Sensitivity/Specificity)
    - Plot C: ROC-AUC Score
- **Ref**: Standards §4.1

### 1.2 Training Efficiency Analysis (Scatter)
- **Data**: `experiment_log.csv`
- **X-Axis**: Training Time (seconds) - *Log Scale if huge variance*
- **Y-Axis**: Macro-F1 Score
- **Color**: Strategy
- **Shape**: Model
- **Goal**: Identify the "Sweet Spot" (High performance, low cost).
- **Ref**: Standards §4.5

## 2. Deep Dive: Class-Level Analysis
**Goal**: Unmask the "Accuracy Paradox" where high accuracy hides poor minority detection.

### 2.1 Class Performance Heatmap (The "Fingerprint")
- **Data**: `per_class_metrics.csv`
- **Rows**: Classes (Sorted by frequency: Normal -> Worms)
- **Columns**: Experiments (e.g., `RF_S0`, `RF_S2a`)
- **Value**: F1-Score
- **Color**: `YlGnBu` (Yellow=Low, Blue=High)
- **Insight**: Instantly shows if a strategy "lights up" the rare class rows.
- **Ref**: Standards §4.4

### 2.2 Rare Class Recall Trajectory (Slope Chart)
- **Data**: `rare_class_report.csv`
- **X-Axis**: Strategy (S0 -> S1 -> S2a)
- **Y-Axis**: Recall Score
- **Lines**: Worms, Shellcode, Backdoor, Analysis
- **Goal**: Show the *lift* provided by resampling.
- **Ref**: Standards §4.2

### 2.3 Per-Class Improvement Bar Chart
- **Data**: `per_class_metrics.csv`
- **Calculation**: Delta = Recall(S2a) - Recall(S0)
- **X-Axis**: Class Name
- **Y-Axis**: Recall Improvement
- **Color**: Diverging (Red=Regressed, Green=Improved)

## 3. Multi-Metric Assessment
**Goal**: Holistic view of model behavior.

### 3.1 Radar Charts (Spider Plots)
- **Data**: `final_summary_tables.csv`
- **Axes**: Accuracy, Macro-F1, Weighted-F1, G-Mean, ROC-AUC
- **Series**: S0 vs S2a (for each Model)
- **Goal**: Show if S2a expands the "area of competence" without shrinking Accuracy.

## 4. Technical Note on Limitations
The following requested graphs cannot be generated from current artifacts:
- **Learning Curves (Loss vs Epoch)**: Training logs (`.log`) report only final times, not epoch-wise history.
- **ROC/PR Curves**: Raw prediction probabilities (`y_score`) were not saved, only final metrics.

## 5. Execution Script
I will create `scripts/generate_publication_figures.py` to:
1. Load all CSVs.
2. Apply `visualization_standards.md` (Colors, Fonts, DPI).
3. Generate all figures above.
4. Save to `results/figures/comprehensive/`.

## 6. Next Steps
1. **Approve Plan**: (Implicit in your request).
2. **Code**: Write the generator script.
3. **Review**: Check generated PNGs against standards.
