# 📊 Visualization Implementation Plan

**Objective**: Generate publication-quality visualizations for the UNSW-NB15 Class Imbalance Analysis project, strictly adhering to `visualization_standards.md`.

**Data Sources**:
- `results/tables/final_summary_tables.csv`: Overall performance metrics.
- `results/tables/rare_class_report.csv`: Specific metrics for minority classes.
- `results/experiment_log.csv`: Training times and metadata.

## 1. Planned Visualizations

### A. Performance Overview (Bar Charts)
**Goal**: Compare the efficacy of strategies (S0, S1, S2a) across all models.
- **Chart 1**: Macro-F1 Score Comparison
- **Chart 2**: G-Mean Score Comparison
- **Specs**:
  - **Type**: Grouped Vertical Bar Chart
  - **X-Axis**: Models (LR, RF, XGB)
  - **Grouping**: Strategies (Color coded using `STRATEGY_COLORS`)
  - **Y-Axis**: Metric Score (0.0 - 1.0)
  - **Ref**: Standards §4.1

### B. Efficiency Analysis (Scatter Plot)
**Goal**: Visualize the trade-off between training cost and model performance.
- **Chart**: Training Time (log scale if needed) vs. Macro-F1
- **Specs**:
  - **X-Axis**: Training Time (seconds)
  - **Y-Axis**: Macro-F1 Score
  - **Colors**: Mapped to Strategy (`STRATEGY_COLORS`)
  - **Markers**: Mapped to Model (LR=`o`, RF=`s`, XGB=`^`)
  - **Ref**: Standards §4.5

### C. Rare Class Impact (Line/Slope Charts)
**Goal**: Demonstrate how strategies specifically affect the detection of rare attacks.
- **Chart**: Rare Class Recall Trajectory
- **Specs**:
  - **X-Axis**: Strategy (S0, S1, S2a)
  - **Y-Axis**: Recall Score
  - **Lines**: One per rare class (Worms, Shellcode, Backdoor, Analysis)
  - **Colors**: `CLASS_COLORS` (Emphasis on warm colors for rare classes)
  - **Ref**: Standards §4.2

### D. Summary Heatmaps
**Goal**: Provide a dense summary of performance metrics.
- **Chart**: Model vs. Strategy Performance Matrix
- **Specs**:
  - **Rows**: Models
  - **Columns**: Strategies
  - **Values**: Macro-F1 Score (annotated)
  - **Colormap**: `YlGnBu`
  - **Ref**: Standards §4.4

## 2. Implementation Strategy

### Scripts
I will create a single robust Python script `scripts/generate_publication_figures.py` that:
1.  **Loads Standards**: Imports constants (colors, fonts) directly from a standards module or defines them exactly as specified.
2.  **Loads Data**: Reads the CSVs from `results/tables/` and `results/`.
3.  **Generates Figures**: Uses `matplotlib` and `seaborn` to create the plots.
4.  **Saves Output**: Exports 300 DPI PNGs to `results/figures/summary/` and `results/figures/rare_class_analysis/`.

### Directory Structure Update
```
results/figures/
├── summary/                # New: For high-level comparisons
│   ├── bar_macro_f1_comparison.png
│   ├── scatter_time_vs_f1.png
│   └── heatmap_performance_matrix.png
├── rare_class_analysis/    # New: For detailed rare class plots
│   └── line_rare_class_recall.png
└── ... (existing folders)
```

## 3. Compliance Checklist
- [ ] Font: DejaVu Sans (Size 10-16pt)
- [ ] Colors: S0=#6C757D, S1=#0077B6, S2a=#38B000
- [ ] Dimensions: 10x6 (Standard), 16x6 (Double)
- [ ] DPI: 300 (Publication)
- [ ] File Naming: `{category}_{descriptor}_{timestamp}.png`

## 4. Next Steps
1. Create `scripts/generate_publication_figures.py`.
2. Run script to generate artifacts.
3. Validate artifacts against `visualization_standards.md`.
