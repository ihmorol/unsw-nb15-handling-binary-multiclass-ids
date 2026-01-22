# Methodology

1. Dataset
2. Data Prepossessing
    1. Feature Cleaning and Selection —> Abu bakr
    2. Missing values 
    3. Leveling (Binary and Multiclass)
    4. Data Scaling
        1. Splitting Into Train Validation and Test
3. System flow
4. Technology used
5. Evaluation matrix

# **4. Methodology**

This section presents the complete methodological framework followed in this study, ranging from dataset preparation to model evaluation. The process was designed to be systematic, reproducible, and aligned with best practices recommended in recent intrusion detection research.

---

## **4.1 Dataset**

All experiments were performed on the **UNSW-NB15** benchmark dataset, developed by the Australian Centre for Cyber Security using the IXIA PerfectStorm toolset. The dataset provides modern network traffic along with nine contemporary attack types, making it more representative of real-world scenarios compared to older datasets like KDD99 and NSL-KDD, which suffer from redundancy and outdated attack patterns. This viewpoint is supported by several studies, including Ring et al. (2019), Moustafa and Slay (2015, 2016), and Janarthanan and Zargari (2017), who highlight the dataset’s relevance, diversity, and statistical complexity.

The dataset is distributed with an **official training split** and a **testing split**, both of which were preserved to maintain comparability with prior published work.

---

## **4.2 Data Preprocessing**

A unified preprocessing pipeline was applied across all experiments to ensure a fair comparison between models and imbalance-handling techniques.

### **4.2.1 Feature Cleaning and Selection**

Several attributes in the raw dataset—such as connection identifiers, source and destination IP addresses, timestamps, and other meta-identifiers—were removed because they do not contribute meaningful predictive value. Prior work also shows that reducing redundant or non-informative features improves intrusion detection performance (Janarthanan & Zargari, 2017; Kasongo & Sun, 2020).

The remaining numerical and categorical attributes defined in the original UNSW-NB15 specification were retained.

### **4.2.2 Handling Missing Values**

To maintain data consistency:

- **Numerical features** were imputed using the **median**, which is robust to outliers.
- **Categorical features** were assigned to a special **“missing”** category.

This prevents data loss and ensures that all records can be used during model training.

### **4.2.3 Label Preparation (Binary and Multiclass)**

Two classification tasks were defined:

1. **Binary Classification**
    - Normal → 0
    - All attack types → 1
2. **Multiclass Classification**
    - Ten classes: Normal + nine attack categories

This dual-level labeling enables both broad anomaly detection and fine-grained attack classification.

### **4.2.4 Encoding and Scaling**

Categorical fields such as protocol, service, and connection state were transformed into numerical form using **one-hot encoding**.

All numerical features were normalized using **StandardScaler** (zero mean, unit variance). While tree-based algorithms do not require scaling, applying a uniform transformation across all models ensures consistency.

### **4.2.5 Train–Validation–Test Split**

The preprocessing steps were followed by splitting the dataset into:

- **Training set**
- **Validation set** (20% of training data, stratified by class)
- **Test set** (official UNSW split, untouched until final evaluation)

The validation set was used for parameter tuning and selecting suitable imbalance-handling strategies.

---

## **4.3 System Flow**

The overall workflow followed in this study is summarized below in a textual diagram. This linear representation illustrates every stage of the experimental pipeline.

---

### **System Flow: Textual Diagram**

```
Start
 │
 ▼
Load UNSW-NB15 dataset
 │
 ▼
Remove non-informative and identifier features
 │
 ▼
Handle missing values (median for numeric, “missing” for categorical)
 │
 ▼
Prepare labels (Binary / Multiclass)
 │
 ▼
Encode categorical features using one-hot encoding
 │
 ▼
Scale numerical features using StandardScaler
 │
 ▼
Split data into Training → Validation → Test
 │
 ▼
Apply imbalance-handling strategy on Training data only:
    - No balancing
    - Class weighting
    - Random Oversampling / SMOTE
 │
 ▼
Train classical ML models:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
 │
 ▼
Validate model performance
 │
 ▼
Select best configurations
 │
 ▼
Evaluate on Test set (held-out)
 │
 ▼
Analyze detection of rare attack categories
 │
 ▼
End

```

---

## **4.4 Technology Stack**

The experiments were implemented using widely adopted tools:

- **Python 3.x** for all code development
- **scikit-learn** for classical machine-learning models and preprocessing routines
- **imbalanced-learn** for resampling methods including SMOTE (Chawla et al., 2002)
- **pandas / NumPy** for dataset manipulation
- **matplotlib** and **seaborn** for plotting evaluation metrics

These tools are commonly used in intrusion detection research (Amin et al., 2021; Primartha & Tama, 2017).

---

## **4.5 Evaluation Metrics**

To evaluate each experiment, a combination of overall and per-class metrics was used. This supports a balanced assessment, especially given the highly skewed class distribution of UNSW-NB15 (Bagui & Li, 2021; Shanmugam et al., 2024).

### **Overall Metrics**

- **Accuracy**
- **Macro F1-score**
- **Weighted F1-score**
- **ROC-AUC** (binary and multiclass with one-vs-rest averaging)
- **G-Mean**

The **G-Mean** (Geometric Mean) is particularly important in imbalanced classification because it balances the true positive rate and true negative rate. A high G-Mean indicates that the model performs well across both majority and minority classes.

Formally:

G-Mean=Sensitivity×Specificity\text{G-Mean} = \sqrt{\text{Sensitivity} \times \text{Specificity}}

G-Mean=Sensitivity×Specificity

### **Per-Class Metrics**

- Precision
- Recall
- F1-score for each class

These are essential for understanding whether minority attacks such as Worms, Shellcode, Analysis, and Backdoor are detected successfully.

### **Confusion Matrices**

- **Binary:** 2×2
- **Multiclass:** 10×10

Confusion matrices provide detailed insights into misclassification patterns.

### **Rare-Class Focus Evaluation**

Since UNSW-NB15 contains highly underrepresented attacks, a dedicated analysis was performed to evaluate how different imbalance-handling techniques influence:

- Recall of minority classes
- F1-score of minority classes
- G-Mean across different strategies
- Improvement compared to no balancing

This follows recommendations from Bagui and Li (2021), Karatas et al. (2020), and Thai-Nghe et al. (2010), who emphasize the importance of specialized evaluation for minority classes.