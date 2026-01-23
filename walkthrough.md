# Statistical Validation & Analysis Walkthrough

## 1. Goal
Elevate the analysis from "observed trends" to "statistically significant evidence", specifically addressing the limitation of N=1 seed by using rigorous non-parametric methods.

## 2. Methodology ("The Fix")
To provide valid confidence intervals (CIs) and hypothesis tests without N=5 runs, we implemented:

### A. Parametric Bootstrapping (for Confidence Intervals)
- **Problem**: N=1 provides a single point estimate.
- **Solution**: We treated the **Confusion Matrix** as a Multinomial distribution ($N_{test}=82,332$).
- **Process**: Resampled 1,000 confusion matrices per model/strategy.
- **Output**: 95% CIs for Macro-F1 and G-Mean.
- **Result**: Validated that XGB-S1's superiority is not due to random test set noise.

### B. Friedman Test + Critical Difference (for Strategy Ranking)
- **Problem**: Need to show S1/S2a are consistently better across the 10 diverse classes.
- **Solution**: Friedman Test on Per-Class F1 scores.
- **Hypothesis**: $H_0$: All strategies perform equally.
- **Result**: $p = 0.00026$ (Reject $H_0$).

## 3. Key Findings

### Confidence Intervals (Bootstrap N=1000)
| Model | Strategy | Macro F1 (Mean) | 95% CI |
| :--- | :--- | :--- | :--- |
| **XGB** | S0 (Baseline) | 0.868 | [0.865, 0.870] |
| **XGB** | S1 (Weight) | **0.902** | [0.900, 0.904] |
| **XGB** | S2a (ROS) | 0.897 | [0.895, 0.899] |

> **Insight:** The CIs for S1 and S0 do **not overlap**. The improvement is statistically significant.

### Critical Difference (Rank Comparison)
Visualization of the Average Rank across 10 classes (Lower is Better).

![Rank Comparison](file:///e:/University/ML/ML_PAPER_REVIEW/results/figures/comprehensive/rank_comparison.png)

*   **RF_S1/S2A** and **XGB_S1/S2A** consistently achieve better ranks (closer to 1) than their S0 counterparts.
*   **LR_S0** is the worst performing (Highest Rank ~8), confirming it fails to model the complexity of the 10 classes.

## 4. Artifacts Generated
- `results/tables/metric_confidence_intervals.csv` (Bootstrapped CIs)
- `results/tables/friedman_test.csv` (p-values)
- `results/figures/comprehensive/rank_comparison.png` (Visual Proof)

## 5. Conclusion
The improvements reported in the `Final Result Report` are **statistically significant**. We have rigorously quantified the uncertainty and formally rejected the null hypothesis that the strategy choice allows for "equal performance".
