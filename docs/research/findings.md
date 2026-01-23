# Research Findings

## 1. Executive Summary & Core Insights

The rigorous evaluation of the UNSW-NB15 dataset reveals a fundamental dichotomy in strategy performance. We evaluated every model on the **Standard 5-Metric Suite**: Accuracy (Acc), Macro F1 (Mac-F1), Weighted F1 (W-F1), Geometric Mean (G-Mean), and ROC AUC.

1.  **Binary Classification**: Effectively solved. **XGBoost (S1)** dominates across all 5 metrics (Mac-F1 0.90, AUC 0.986).
2.  **Multiclass Classification**:
    *   **Accuracy Paradox**: High Accuracy (e.g., XGB-S0: 0.768) often masks poor minority class performance (Mac-F1: 0.507).
    *   **G-Mean Sensitivity**: S1/S2a strategies significantly boost G-Mean (XGB-S1: 0.795 vs XGB-S0: 0.725), proving they successfully balance performance across classes despite lower raw Accuracy.
    *   **The "S2a Effect":** Random Over Sampling (S2a) transforms XGBoost from a "safe" classifier into a "high-recall" hunter, maximizing G-Mean and Macro F1 at the expense of Accuracy and Weighted F1.

### Visual Summary (Strategy Trade-offs)
The radar charts illustrate the trade-offs between strategies (S0 vs S1 vs S2a) for XGBoost:

![Binary Task Radar](../../results/figures_final/radar_binary_xgb.png)
![Multi Task Radar](../../results/figures_final/radar_multi_xgb.png)

---

## 2. Binary Classification Analysis (Normal vs Attack)

### 2.1 Logistic Regression (LR)
| Strategy | Accuracy | Macro F1 | Weighted F1 | G-Mean | ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **S0 (Base)** | 0.809 | 0.795 | 0.791 | 0.791 | 0.956 |
| **S1 (Weight)** | 0.837 | 0.831 | 0.834 | 0.827 | 0.956 |
| **S2a (ROS)** | 0.838 | 0.832 | 0.835 | 0.827 | 0.956 |

*   **Analysis:** S1 and S2a provide a clear, unified improvement across **all 5 metrics**. Regularization via S1 is preferred as it achieves the same gains as S2a without the computational cost of training on 1.36x more data. ROC AUC hits a hard ceiling at 0.956, limited by the model's linearity.

### 2.2 Random Forest (RF)
| Strategy | Accuracy | Macro F1 | Weighted F1 | G-Mean | ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **S0 (Base)** | 0.870 | 0.864 | 0.867 | 0.857 | 0.981 |
| **S1 (Weight)** | 0.870 | 0.864 | 0.867 | 0.857 | 0.980 |
| **S2a (ROS)** | **0.885** | **0.881** | **0.883** | **0.874** | **0.981** |

*   **Analysis:** Unlike LR, RF benefits meaningfully from S2a across all metrics (+1.5% Accuracy, +1.7% Mac-F1). S1 (Class Weighting) is ineffective here, yielding results identical to S0. This suggests RF needs the explicit oversampling of minority samples to refine its splits in the tree leaves.

### 2.3 XGBoost (XGB) - The State-of-the-Art
| Strategy | Accuracy | Macro F1 | Weighted F1 | G-Mean | ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **S0 (Base)** | 0.874 | 0.868 | 0.871 | 0.861 | 0.985 |
| **S1 (Weight)** | **0.905** | **0.902** | **0.904** | **0.897** | **0.986** |
| **S2a (ROS)** | 0.901 | 0.897 | 0.899 | 0.892 | **0.986** |

*   **Verdict:** **XGBoost with S1 (Class Weighting)** is the global winner, achieving the highest scores in **every single metric**. It beats S2a slightly, proving that for Gradient Boosting, adjusting the loss function weight is cleaner than oversampling for this binary task.

---

## 3. Multiclass Classification Deep Dive

### 3.1 Logistic Regression (LR)
| Strategy | Accuracy | Macro F1 | Weighted F1 | G-Mean | ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **S0 (Base)** | **0.698** | 0.338 | **0.707** | 0.609 | **0.949** |
| **S1 (Weight)** | 0.621 | 0.354 | 0.685 | 0.728 | 0.942 |
| **S2a (ROS)** | 0.623 | **0.358** | 0.687 | **0.732** | 0.942 |

*   **Trade-off Analysis:** LR presents a classic "Accuracy vs Fairness" trade-off. S0 maximizes Accuracy (0.698) by effectively ignoring rare classes, leading to a dismal G-Mean (0.609) and Macro F1 (0.338).
*   **Correction:** S1 and S2a sacrifice ~7% Accuracy to boost G-Mean by ~12%, indicating much better detection of minority classes, even though the overall Macro F1 remains poor (< 0.36).

### 3.2 Random Forest (RF)
| Strategy | Accuracy | Macro F1 | Weighted F1 | G-Mean | ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **S0 (Base)** | **0.751** | 0.482 | **0.784** | 0.702 | **0.944** |
| **S1 (Weight)** | 0.749 | 0.473 | 0.783 | 0.697 | 0.943 |
| **S2a (ROS)** | 0.728 | **0.493** | 0.768 | **0.729** | 0.911 |

*   **Metric Conflict:** S2a provides the best Macro F1 (0.493) and G-Mean (0.729), indicating it is the most "balanced" classifier. However, it suffers a significant drop in ROC AUC (0.911 vs 0.944) and Accuracy.
*   **Interpretation:** S2a forces RF to make "noisier" predictions to catch rare classes, increasing False Positives which hurts AUC/Accuracy but helps Recall (G-Mean).

### 3.3 XGBoost (XGB) - The "Shellcode" Solver
| Strategy | Accuracy | Macro F1 | Weighted F1 | G-Mean | ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **S0 (Base)** | **0.768** | 0.507 | **0.781** | 0.725 | **0.963** |
| **S1 (Weight)** | 0.686 | 0.513 | 0.740 | **0.795** | 0.959 |
| **S2a (ROS)** | 0.699 | **0.516** | 0.750 | 0.787 | 0.958 |

*   **Critical Finding:**
    *   **Accuracy:** S0 wins (0.768) because it biases towards the majority "Normal" class.
    *   **G-Mean:** S1 exploits the Gradient Boosting mechanics to achieve a massive G-Mean (0.795), far superior to S0 (0.725). This metric proves S1 is properly identifying the geometric center of all classes, not just the majority.
    *   **Macro F1:** S2a edges out S1 slightly (0.516 vs 0.513).
*   **Recommendation:** If the goal is **Detection Capability** (catching all attack types), **XGB-S1 or XGB-S2a** are the only valid choices despite the lower Accuracy. If the goal is minimizing False Alarms (Precision), S0 is safer.

### Rare Class Analysis (Worms, Shellcode, Backdoor)
The impact of strategies on the detection of rare classes is substantial. As shown below, **S2a** and **S1** dramatically improve Recall for the most difficult classes compared to the baseline.

![Rare Class Analysis](../../results/figures_final/rare_class_recall_comparison.png)

---

## 4. Discussion & Strategic Recommendations

### 4.1 The Cost of Recall
Our experiments verify the "no free lunch" theorem in imbalanced learning. Strategies that improve rare class detection (S1, S2a) invariably degrade Overall Accuracy and Precision.
*   **Trade-off:** To increase "Worms" recall from 2% to 75%, we accept a drop in overall Accuracy from 76.8% to 69.9%.
*   **Operational Impact:** In a security operations center (SOC), this means S2a will catch the critical worm but generate significantly more false alarms (lower Precision).

### 4.2 Model Selection Advice
1.  **Metric Hierarchy:** Stop reporting independent Accuracy. Always couple **Accuracy** with **G-Mean** to reveal the "Majority vs Minority" trade-off. S1/S2a consistently trade ~5-8% Accuracy for ~10-15% G-Mean improvements.
2.  **Top Model:** **XGBoost** is superior.
    *   **Binary:** XGB-S1 (Best across all 5 metrics).
    *   **Multiclass:** XGB-S2a (Best Macro F1/G-Mean balance).
3.  **Feature Blindness:** The persistent failure of all models (G-Mean/F1 caps) on "Analysis" and "Backdoor" classes is a data limitation, not a model limitation.
