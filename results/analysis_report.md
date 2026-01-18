# State-of-the-Art Analysis & Findings Report

> **Date:** 2026-01-18
> **Author:** Lead (Skyview)
> **Status:** Draft

## 1. Executive Summary

The current implementation has successfully established a reproducible baseline for the UNSW-NB15 dataset.
-   **Binary Classification**: robust performance (ROC-AUC > 0.98), meeting industry standards.
-   **Multiclass Classification**: mixed results. While "Critically Rare" classes (Worms, Shellcode) have seen dramatic improvements via Class Weighting (S1) and Oversampling (S2a), the "Moderately Rare" classes (**Backdoor**, **Analysis**) remain stubbornly difficult to detect.

**Key Finding:** The standard "balanced" weighting strategy is insufficient for *Backdoor* and *Analysis* attacks, likely due to high feature overlap with majority classes rather than just sample scarcity.

## 2. Performance Analysis

### 2.1 Scorecard vs Objectives

| Objective | Status | Metric | Notes |
|-----------|--------|--------|-------|
| **O1: Binary IDS** | ✅ **PASS** | ROC-AUC: 0.984 | RF/XGB perform excellently. |
| **O2: Multiclass IDS** | ⚠️ **PARTIAL** | Macro-F1: 0.52 | dragged down by specific classes. |
| **O3: Strategy Comparison** | ✅ **DONE** | S1 vs S0 | Validated significant lift in rare recall. |
| **O4: Rare Class Detection** | ❌ **FAIL** | Backdoor Recall | Target > 50%, Actual ~18%. |

### 2.2 The "Backdoor" & "Analysis" Anomaly

Despite having *more* samples than Worms (1746 vs 130), Backdoor performance is significantly worse.

| Class | Samples | XGB S1 Recall | XGB S2a Recall | Target | Gap |
|-------|---------|---------------|----------------|--------|-----|
| Worms | 130 | 72.7% | 72.7% | > 20% | ✅ +52pp |
| Shellcode | 1,133 | 92.6% | 93.9% | > 40% | ✅ +53pp |
| **Backdoor** | 1,746 | **18.3%** | **18.0%** | > 50% | ❌ -32pp |
| **Analysis** | 2,000 | **56.4%** | **55.7%** | > 60% | ❌ -4pp |

**Hypothesis:**
1.  **Feature Inseparability:** Backdoor attacks (e.g., persistent channels) might look syntactically identical to Normal or Generic traffic in the provided flow features (`dur`, `sbytes`, etc.).
2.  **Weighting Insufficiency:** The 'balanced' heuristic assigns weights inversely proportional to frequency. If the class is "harder" (closer decision boundary), it needs *higher* weights than just frequency would dictate.

## 3. SOTA Gap Analysis

Comparisons with literature (e.g., More et al., 2024 claiming 99%) must be viewed with skepticism regarding:
1.  **Leakage**: Many papers do not remove `id`, `stcpb`, `dtcpb` (sequence numbers), or `stime` (timestamps), which are known leakage sources. We removed 7 identifiers.
2.  **Split Integrity**: We used the official train/test split. Merging and shuffling (common in papers) makes the task easier by mixing distributions.

**Conclusion:** Our results are likely more realistic for production, but the low recall on Backdoor is a genuine area for algorithmic improvement.

## 4. Improvement Plan

To achieve "publishable" findings that genuinely advance the field, we must solve the Backdoor performance without sacrificing integrity.

### Strategy 1: Targeted "Super-Weighting" (S1-Custom)
Instead of standard `class_weight='balanced'`, we will manually boost the weights for Backdoor and Analysis by 2x-5x to force the model to pay attention.

### Strategy 2: Hybrid SMOTE (S2b)
The contract lists S2b (SMOTE) as optional. We should execute this. SMOTE creates synthetic points *between* existing ones, potentially defining the decision boundary better than ROS (cloning).

### Strategy 3: Feature Engineering check
Verify if `stcpb`/`dtcpb` (TCP base sequence numbers) are potential predictors for Backdoor. The data contract lists them as "Window Features". Let's verify their importance.

## 5. Recommendations

1.  **Execute S2b (SMOTE)** experiments immediately.
2.  **Run S1-Custom** experiment with boosted weights for Backdoor/Analysis.
3.  **Generate Feature Importance Plot** from the best XGB model to see what is driving decisions.
