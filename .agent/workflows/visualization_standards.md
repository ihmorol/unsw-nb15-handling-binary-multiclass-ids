---
description: Visualization Ground Truth Standards for UNSW-NB15 ML/IDS Research
---

# 🎨 Visualization Ground Truth Standards

**Project**: UNSW-NB15 Class Imbalance Analysis for Intrusion Detection  
**Version**: 1.0  
**Last Updated**: 2026-01-18  
**Status**: CANONICAL REFERENCE

---

## 🎯 Purpose

This document defines the **binding standards** for all data visualizations, charts, graphs, and diagrams created for this research project. These standards ensure:

1. **Publication Quality**: All figures meet peer-review standards
2. **Reproducibility**: Consistent styling across all visualizations
3. **Accessibility**: Colorblind-safe palettes and readable fonts
4. **Brand Consistency**: Uniform look-and-feel for presentations/papers

---

## 📊 1. CANONICAL COLOR PALETTES

### 1.1 Task Colors (Binary vs Multiclass)
```python
TASK_COLORS = {
    'binary': '#2E86AB',    # Deep blue
    'multi': '#A23B72'      # Deep magenta
}
```

### 1.2 Model Colors
```python
MODEL_COLORS = {
    'lr': '#E63946',        # Red (Logistic Regression)
    'rf': '#06A77D',        # Green (Random Forest)
    'xgb': '#F77F00'        # Orange (XGBoost)
}
```

### 1.3 Strategy Colors
```python
STRATEGY_COLORS = {
    's0': '#6C757D',        # Gray (Baseline/None)
    's1': '#0077B6',        # Blue (Class Weight)
    's2a': '#38B000'        # Green (RandomOverSampler)
}
```

### 1.4 Class Colors (Multiclass - 10 Classes

)
```python
CLASS_COLORS = {
    'Normal': '#2A9D8F',           # Teal
    'Generic': '#E76F51',          # Coral
    'Exploits': '#F4A261',         # Sandy brown
    'Fuzzers': '#E9C46A',          # Mustard
    'DoS': '#8338EC',              # Purple
    'Reconnaissance': '#3A86FF',   # Bright blue
    'Analysis': '#FB5607',         # Bright orange (RARE)
    'Backdoor': '#FF006E',         # Hot pink (RARE)
    'Shellcode': '#FFBE0B',        # Golden yellow (RARE)
    'Worms': '#D62828'             # Dark red (RARE - most rare)
}
```

**Rare Class Emphasis**: Worms, Shellcode, Backdoor, Analysis always use bolder/warmer colors.

### 1.5 Confusion Matrix Colors
```python
# Sequential (for counts)
CM_CMAP_COUNTS = 'Blues'  # matplotlib colormap

# Diverging (for normalized confusion matrices)
CM_CMAP_NORMALIZED = 'RdYlGn_r'  # Red (bad) → Yellow → Green (good), reversed
```

### 1.6 Heatmap Colors (Performance Metrics)
```python
HEATMAP_CMAP = 'YlGnBu'  # Yellow (low) → Green → Blue (high)
```

### 1.7 Accessibility: Colorblind-Safe Palettes
For charts with >3 categories, use **Tableau 10** or **Okabe-Ito** palettes:
```python
import seaborn as sns
sns.color_palette("colorblind")  # Okabe-Ito safe palette
```

---

## 🖋️ 2. TYPOGRAPHY & TEXT STANDARDS

### 2.1 Fonts
```python
FONT_FAMILY = 'DejaVu Sans'  # Default matplotlib font (cross-platform)
# Alternative for papers: 'Times New Roman' or 'Arial'

FONT_SIZES = {
    'title': 16,           # Figure title
    'axis_label': 14,      # X/Y axis labels
    'tick_label': 12,      # Axis tick labels
    'legend': 11,          # Legend text
    'annotation': 10       # In-plot annotations
}
```

### 2.2 Title Format
- **Figure titles**: Title Case, Bold
- **Axis labels**: Sentence case with units in parentheses
- **Legend titles**: Title Case

**Example**:
```python
ax.set_title('Macro-F1 Score Comparison Across Strategies', fontsize=16, fontweight='bold')
ax.set_xlabel('Experiment ID', fontsize=14)
ax.set_ylabel('Macro-F1 Score (%)', fontsize=14)
```

---

## 📏 3. FIGURE DIMENSIONS & RESOLUTION

### 3.1 Standard Figure Sizes
```python
FIGURE_SIZES = {
    'single_plot': (10, 6),        # Single chart
    'double_plot': (16, 6),        # Side-by-side comparison
    'grid_2x2': (12, 10),          # 2×2 subplot grid
    'grid_3x3': (18, 14),          # 3×3 confusion matrix grid
    'wide_comparison': (14, 5),    # Wide horizontal bar chart
    'poster': (16, 12)             # Conference poster figure
}
```

### 3.2 Resolution Standards
```python
DPI_DRAFT = 100       # For quick iteration
DPI_PRESENTATION = 150  # For slides/reports
DPI_PUBLICATION = 300   # For journal submission
```

**Export Command**:
```python
fig.savefig('figure.png', dpi=300, bbox_inches='tight', facecolor='white')
```

---

## 🎨 4. CHART-SPECIFIC STANDARDS

### 4.1 Bar Charts

**Orientation**: Horizontal for >5 categories, Vertical otherwise

**Spacing**:
```python
bar_width = 0.8
group_spacing = 0.2  # Between grouped bars
```

**Gridlines**: Y-axis only (for horizontal bars), light gray (#E0E0E0)

**Example Template**:
```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(categories))
ax.bar(x, values, color=STRATEGY_COLORS['s1'], width=0.8, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.set_ylabel('Macro-F1 Score', fontsize=14)
ax.set_title('Strategy S1 Performance', fontsize=16, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines('right'].set_visible(False)
plt.tight_layout()
```

---

### 4.2 Line Charts

**Line Width**: 2.5px (thick enough for clarity)

**Markers**: Use for <10 data points per line
```python
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p']  # Circle, square, triangle, etc.
MARKER_SIZE = 8
```

**Legend Placement**: Best location (auto) or upper-right

**Example**:
```python
for i, rare_class in enumerate(['Worms', 'Shellcode', 'Backdoor', 'Analysis']):
    ax.plot(strategies, recall_values[i], marker='o', linewidth=2.5, 
            label=rare_class, color=CLASS_COLORS[rare_class])
ax.legend(loc='upper left', fontsize=11, frameon=True, fancybox=False)
```

---

### 4.3 Confusion Matrices

**Size Standards**:
- **Binary (2×2)**: 6×5 inches
- **Multiclass (10×10)**: 10×8 inches

**Annotations**:
```python
annot=True          # Show counts
fmt='d'             # Integer format for counts
fmt='.2f'           # Float format for normalized matrices
cbar=True           # Always include colorbar
```

**Cell Text Color**: Auto-adjust (white on dark cells, black on light cells)

**Example Template**:
```python
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Sample Count'}, ax=ax)
ax.set_xlabel('Predicted Label', fontsize=14)
ax.set_ylabel('True Label', fontsize=14)
ax.set_title(f'Confusion Matrix: {experiment_id}', fontsize=16, fontweight='bold')
plt.tight_layout()
```

---

### 4.4 Heatmaps (Performance Metrics)

**Aspect Ratio**: Equal for square matrices, otherwise auto

**Colorbar**: Always include with label

**Value Annotations**: Show for ≤100 cells

**Example**:
```python
# Performance heatmap (Models × Strategies)
data = np.array([[0.85, 0.87, 0.88],  # LR: S0, S1, S2a
                 [0.89, 0.91, 0.90],  # RF
                 [0.87, 0.90, 0.89]]) # XGB

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(data, annot=True, fmt='.2f', cmap='YlGnBu', 
            xticklabels=['S0', 'S1', 'S2a'],
            yticklabels=['LR', 'RF', 'XGB'],
            vmin=0.80, vmax=0.95,  # Fix scale for comparability
            cbar_kws={'label': 'Macro-F1 Score'}, ax=ax)
ax.set_title('Macro-F1 Performance Heatmap', fontsize=16, fontweight='bold')
```

---

### 4.5 Scatter Plots

**Point Size**: 100 (adjustable for emphasis)

**Alpha**: 0.7 (for overlapping points)

**Gridlines**: Both axes, light gray

**Example**:
```python
fig, ax = plt.subplots(figsize=(10, 6))
for strategy in ['s0', 's1', 's2a']:
    mask = df['strategy'] == strategy
    ax.scatter(df[mask]['training_time'], df[mask]['macro_f1'], 
               s=100, alpha=0.7, label=strategy.upper(), 
               color=STRATEGY_COLORS[strategy], edgecolors='black', linewidth=0.5)
ax.set_xlabel('Training Time (seconds)', fontsize=14)
ax.set_ylabel('Macro-F1 Score', fontsize=14)
ax.legend(title='Strategy', fontsize=11)
ax.grid(alpha=0.3, linestyle='--')
```

---

## 📁 5. FILE NAMING CONVENTION

All saved figures must follow this structure:

```
{category}_{descriptor}_{timestamp}.{ext}
```

**Examples**:
- `cm_binary_rf_s1_20260118.png` (Confusion matrix)
- `performance_macro_f1_comparison_   20260118.png` (Bar chart)
- `rare_class_recall_lines_20260118.png` (Line chart)
- `dist_class_distribution_original_20260118.png` (Distribution chart)

**Extensions**:
- `.png`: Default (high compatibility)
- `.pdf`: Vector format (for LaTeX papers)
- `.svg`: Editable vector (for further editing)

---

## 📂 6. DIRECTORY STRUCTURE

All visualizations must be saved to:

```
results/
└── figures/
    ├── confusion_matrices/
    │   ├── binary/
    │   └── multiclass/
    ├── performance_comparisons/
    ├── rare_class_analysis/
    ├── distributions/
    |-- training/
    └── summary/
```

**Create directories if they don't exist**:
```python
import os
os.makedirs('results/figures/confusion_matrices/binary', exist_ok=True)
```
---

## 📐 8. MERMAID DIAGRAM STANDARDS

For methodology flowcharts and architecture diagrams:

**Node Styling**:
```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#2E86AB', 'primaryTextColor':'#fff', 'primaryBorderColor':'#1a5276', 'lineColor':'#6C757D', 'secondaryColor':'#06A77D', 'tertiaryColor':'#E63946'}}}%%
```

**Arrow Types**:
- Solid arrow (`-->`) for sequential steps
- Dotted arrow (`-.->`) for optional paths
- Thick arrow (`==>`) for primary workflows

---

## ✅ 9. QUALITY CHECKLIST

Before saving any visualization, verify:

- [ ] **Color Palette**: Uses canonical colors from §1
- [ ] **Font Sizes**: Matches standards from §2
- [ ] **Resolution**: DPI ≥ 300 for final versions
- [ ] **Axis Labels**: Clear with units
- [ ] **Title**: Descriptive and properly formatted
- [ ] **Legend**: Present and well-positioned (if multi-series)
- [ ] **Gridlines**: Appropriate and not overwhelming
- [ ] **File Name**: Follows naming convention from §5
- [ ] **Saved Location**: Correct subdirectory under `results/figures/`
- [ ] **Accessibility**: Colorblind-safe (test with simulator if >3 colors)
- [ ] **White Space**: No excessive margins (use `bbox_inches='tight'`)

---

## 🚫 10. PROHIBITED PRACTICES

**Never**:
1. Use default matplotlib colors without customization
2. Save figures with generic names like `figure1.png`
3. Mix styling across figures in the same category
4. Use 3D plots (hard to interpret, not publication-standard)
5. Overlay too many lines (>6) on a single chart
6. Use pie charts for >5 categories (use bar charts instead)
7. Omit axis labels or units
8. Use Comic Sans or similar informal fonts
9. Save at <150 DPI for presentations or <300 DPI for papers
10. Create visualizations manually (always use code for reproducibility)

---

## 📚 11. REFERENCES

### Python Libraries
- **matplotlib**: https://matplotlib.org/
- **seaborn**: https://seaborn.pydata.org/
- **plotly** (if interactive needed): https://plotly.com/python/

### Color Resources
- **Colorbrewer**: https://colorbrewer2.org/
- **Coolors**: https://coolors.co/
- **Adobe Color**: https://color.adobe.com/

### Accessibility
- **Coblis Color Blindness Simulator**: https://www.color-blindness.com/coblis-color-blindness-simulator/
- **WebAIM Contrast Checker**: https://webaim.org/resources/contrastchecker/

---

## 📝 12. CHANGE LOG

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-18 | Initial ground truth standards document |

---

**END OF DOCUMENT**

**Enforcement**: All visualization scripts MUST import and adhere to these standards.  
**Audit**: Run `scripts/validate_figures.py` to check compliance before committing.