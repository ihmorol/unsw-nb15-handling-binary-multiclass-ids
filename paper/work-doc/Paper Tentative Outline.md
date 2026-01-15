# Paper Tentative Outline

[Literature Review Tracker](https://www.notion.so/fb87a2efa86248209776e025d32cdde3?pvs=21)

### Tentative Title

> “Handling Class Imbalance in Binary and Multiclass Intrusion Detection on the UNSW-NB15 Dataset Using Classical Machine Learning”
> 

### One-paragraph abstract draft

> Intrusion Detection Systems (IDS) are commonly evaluated on benchmark datasets such as UNSW-NB15, but class imbalance and highly skewed attack distributions often lead to misleadingly high accuracy while rare attacks remain undetected. The UNSW-NB15 dataset contains modern normal traffic and nine attack categories (Fuzzers, Analysis, Backdoor, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms), with extreme imbalance between normal and minority attack types. UNSW Sites+1 In this work, we build simple, reproducible IDS baselines using Logistic Regression, Random Forest, and Gradient Boosting on both binary (normal vs. attack) and multiclass (normal + nine attacks) tasks. We systematically compare three imbalance-handling strategies—no balancing, class weighting, and oversampling (RandomOverSampler / SMOTE)—and evaluate performance using accuracy, macro/weighted F1-score, per-class recall, confusion matrices, and ROC-AUC. Our experiments show how model performance degrades when moving from binary to multiclass classification, and how different balancing techniques affect the detection of rare attacks such as Worms and Shellcode. We provide a transparent experimental protocol and openly share code and configuration so that future work, including deep learning and generative augmentation, can build on a clean classical-ML baseline.
> 

You can refine this after you have real results.

---

## 2. Problem Statement, Objectives and Research Questions

### 2.1 Problem statement

- **Dataset**: UNSW-NB15; modern network traffic with 10 classes: Normal + 9 attack types. [UNSW Sites+1](https://research.unsw.edu.au/projects/unsw-nb15-dataset?utm_source=chatgpt.com)
- **Issues**:
    - Strong **class imbalance** (Normal and Generic massively dominate; Worms, Shellcode, Backdoor are tiny). [staff.itee.uq.edu.au+1](https://staff.itee.uq.edu.au/marius/NIDS_datasets/?utm_source=chatgpt.com)
    - **Class overlap** between certain attack categories and normal traffic reported in prior analysis. [arXiv+1](https://arxiv.org/pdf/2101.05067?utm_source=chatgpt.com)
- Many recent IDS papers:
    - Either focus on **deep learning / ensembles + heavy feature selection**, or
    - Use classical ML but optimize for **overall accuracy** and hide per-class metrics. [MDPI+2SpringerLink+2](https://www.mdpi.com/1999-4893/17/2/64?utm_source=chatgpt.com)

Your supervisor wants:

- Attack vs Non-attack (binary)
- And also **“individual detective”** → i.e., **multiclass per attack type**

So the technical problem becomes:

> How do simple classical ML models behave on both binary and multiclass UNSW-NB15 IDS tasks under strong class imbalance, and how much can simple imbalance-handling strategies improve detection of rare attacks?
> 

### 2.2 Objectives

1. Build **binary** IDS models to distinguish Normal vs Attack on UNSW-NB15.
2. Build **multiclass** IDS models to classify Normal + 9 attack types.
3. Apply and compare **three imbalance strategies**:
    - No balancing (raw data)
    - Class weighting
    - Oversampling (RandomOverSampler, optionally SMOTE)
4. Analyze performance:
    - Overall metrics (accuracy, macro/weighted F1, ROC-AUC)
    - **Per-class recall / precision**, especially for **minority attack classes**.
5. Provide a **reproducible baseline pipeline** (code + hyperparameters + splits) that others can extend with deep learning, advanced resampling, or focal loss. [IIETA+2MDPI+2](https://www.iieta.org/download/file/fid/98121?utm_source=chatgpt.com)

### 2.3 Research Questions (RQs)

You can write them explicitly in the paper:

- **RQ1**: How does class imbalance in UNSW-NB15 affect the performance of classical ML models on **binary vs multiclass** intrusion detection tasks?
- **RQ2**: To what extent do **class weighting** and **oversampling** improve detection of **minority attack classes** compared to using the raw imbalanced data?
- **RQ3**: Is there a consistent pattern in how different models (Logistic Regression, Random Forest, Gradient Boosting) respond to imbalance-handling methods across binary and multiclass tasks?
- **RQ4 (optional)**: For extremely rare classes (e.g., Worms, Shellcode), does oversampling significantly improve recall without destroying performance on majority classes?

If the paper gets too long, RQ4 can be turned into a short sub-analysis instead of a formal research question.

---

## 3. Dataset Understanding and Technical Details (for beginners)

### 3.1 What exactly is UNSW-NB15?

- Captured at UNSW Canberra Cyber Range using IXIA PerfectStorm. [UNSW Sites](https://research.unsw.edu.au/projects/unsw-nb15-dataset?utm_source=chatgpt.com)
- Contains **2M+ network flows**, each with:
    - Basic features: source/destination IP/port, protocol, state…
    - Content features: bytes, packets, flags…
    - Time features: duration…
    - Derived features: flow statistics, etc.
- Each record is labeled as:
    - `normal`
    - Or one of 9 attack categories: `{ Fuzzers, Analysis, Backdoor, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms }`. [UNSW Sites+1](https://research.unsw.edu.au/projects/unsw-nb15-dataset?utm_source=chatgpt.com)

### 3.2 Class distribution (high level)

Approximate distribution (numbers vary slightly across published versions, but the pattern is clear): [staff.itee.uq.edu.au+1](https://staff.itee.uq.edu.au/marius/NIDS_datasets/?utm_source=chatgpt.com)

- Normal: ~1.5M+
- Generic, Exploits, Fuzzers, Reconnaissance: tens of thousands
- DoS: several thousands
- Backdoor, Analysis, Shellcode: hundreds
- Worms: a few dozen

Implications:

- Binary **Normal vs Attack** is already imbalanced (Normal >> all attacks combined).
- Multiclass is **extremely imbalanced** (Worms is basically “rare events”).

### 3.3 Files / splits on Kaggle

Typical Kaggle versions provide:

- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

You should:

- Respect the **out-of-the-box train/test separation** for the main comparisons.
- Optionally create a **validation split from the training set** (e.g., 70/15/15) for hyperparameter tuning.

---

## 4. Methodology Design (what you will actually implement)

You can structure this as your **Methods** section.

### 4.1 Tasks

Define **three main experimental tracks**:

1. **Task A – Binary classification**
    - Labels:
        - `0 = Normal`
        - `1 = Attack` (anything not Normal)
2. **Task B – Multiclass “10-class” classification**
    - Labels: `Normal` + 9 specific attack types
3. **Task C (optional, if time permits) – Rare-class focused analysis**
    - For very rare classes (e.g., Worms, Shellcode, Backdoor), do:
        - Either one-vs-rest binary detectors, or
        - Evaluate carefully their per-class metrics in Task B with extra oversampling.

If time is tight, keep Task C as a **short additional analysis**, not a full experiment grid.

### 4.2 Preprocessing pipeline

Same pipeline for all tasks (make it a single scikit-learn Pipeline):

1. **Feature selection**
    - Drop identifiers and non-useful fields:
        - IDs, timestamps if not used, IP addresses (or encode them if you want but simplest is drop).
    - Keep numeric and categorical features from the original UNSW spec.
2. **Handle missing values**
    - Numeric: impute with median
    - Categorical: impute with a special category (“missing”)
3. **Encode categorical features**
    - One-hot encode categorical columns (protocol, state, service, etc.).
4. **Scale numeric features**
    - StandardScaler or MinMaxScaler.
    - For tree-based models (RF, XGBoost), scaling is not strictly necessary, but for **Logistic Regression** it is important.
    - Easiest: scale for all models for consistency.
5. **Train / validation / test split**
    - Use UNSW official train/test as base, and inside training create a validation split (e.g., 80/20), stratified by class.
    - That gives:
        - Train
        - Validation (for hyperparameters)
        - Test (held-out, used only once at the end)

### 4.3 Models

Use three classical models:

1. **Logistic Regression (LR)**
    - Baseline linear model
    - Use regularization (C parameter)
2. **Random Forest (RF)**
    - Bagging of decision trees
    - Handles non-linearities and heterogeneous features
3. **Gradient Boosting**
    - Either **XGBoost** or **LightGBM** (choose one and stick to it)
    - Strong tabular baseline; already widely used on UNSW. [SpringerLink+1](https://link.springer.com/article/10.1007/s42979-024-03369-0?utm_source=chatgpt.com)

Do **light hyperparameter tuning** using validation set or simple cross-validation (not full grid search; stay realistic).

### 4.4 Imbalance-handling strategies

For each task (A and B) and each model, compare:

1. **S0: No balancing**
    - Train on raw imbalanced data.
2. **S1: Class weighting**
    - Use `class_weight="balanced"` (or manually computed weights).
3. **S2: Oversampling**
    - Use `RandomOverSampler` from `imbalanced-learn` on the training split.
    - Optional: compare `SMOTE` as S3, but only if time permits. [IIETA+2ResearchGate+2](https://www.iieta.org/download/file/fid/98121?utm_source=chatgpt.com)

So your **minimum experiment grid**:

- Tasks: 2 (binary + multiclass)
- Models: 3
- Strategies: 3

→ **2 × 3 × 3 = 18 experiments**, still manageable.

### 4.5 Evaluation metrics

For each experiment, compute:

- **Overall:**
    - Accuracy
    - Macro F1
    - Weighted F1
    - ROC-AUC (binary and one-vs-rest or macro for multiclass)
- **Per-class metrics:**
    - Precision, recall, F1 for each class
- **Confusion matrices:**
    - Binary: 2×2
    - Multiclass: 10×10
- **Focus analysis:**
    - Rare classes (Worms, Shellcode, Backdoor, Analysis):
        - Check recall and support (number of samples)
        - Observe how they change across S0/S1/S2

You then write a **Results** subsection per task:

- Binary: mostly to show global behavior.
- Multiclass: to show how severe the imbalance issue is.

---

---

---

---

---

---

---

## 5. Literature Review Plan (what to read and how to position your work)

You do NOT need a giant survey. You need a **targeted, critical review**.

### 5.1 Main themes

Organize your related work into 3–4 subsections:

1. **UNSW-NB15 dataset and issues**
    - Original dataset description: attacks, features, purpose. [UNSW Sites+1](https://research.unsw.edu.au/projects/unsw-nb15-dataset?utm_source=chatgpt.com)
    - Papers highlighting **class imbalance and overlap** in UNSW-NB15. [arXiv+1](https://arxiv.org/pdf/2101.05067?utm_source=chatgpt.com)
2. **Classical ML on UNSW-NB15**
    - Works using LR, RF, SVM, XGBoost, etc., often with some feature selection. [MDPI+2SpringerLink+2](https://www.mdpi.com/1999-4893/17/2/64?utm_source=chatgpt.com)
    - Note: many optimize accuracy and mention “high performance” but do not deeply analyze per-class metrics.
3. **Imbalance handling in IDS**
    - SMOTE, ADASYN, random oversampling/undersampling. [IIETA+2ResearchGate+2](https://www.iieta.org/download/file/fid/98121?utm_source=chatgpt.com)
    - Focal loss and cost-sensitive learning (mainly in deep learning). [MDPI+1](https://www.mdpi.com/2073-8994/13/1/4?utm_source=chatgpt.com)
4. **Multiclass IDS on UNSW-NB15**
    - Recent multiclass ML/ensemble/XAI papers using UNSW-NB15. [PeerJ+3Frontiers+3Nature+3](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1520741/pdf?utm_source=chatgpt.com)
    - Highlight that they often:
        - Use more complex models,
        - Focus on high overall metrics,
        - Sometimes do feature selection, but
        - Rarely provide a **clean, simple classical baseline** across both binary and multiclass with a systematic imbalance-strategy comparison.

### 5.2 How to explicitly state the gap

In your literature review conclusion, clearly say something like:

- Many works:
    - Use **UNSW-NB15**,
    - Evaluate multiple ML models,
    - Sometimes mention imbalance and apply SMOTE or class weights.
- But:
    - There is **no simple, unified baseline** that:
        - Compares binary vs multiclass on the same pipeline,
        - Uses simple classical models only,
        - Systematically compares **no balancing vs class weighting vs oversampling**,
        - Carefully reports **per-class** metrics for rare attacks.
- That is your niche.

---

## 6. Draft Paper Structure (section-wise skeleton)

You can follow this as your initial LaTeX/Word outline.

1. **Introduction**
    - Motivation: cyberattacks, need for IDS.
    - Benchmark datasets; UNSW-NB15 as modern baseline. [UNSW Sites+1](https://research.unsw.edu.au/projects/unsw-nb15-dataset?utm_source=chatgpt.com)
    - Problems: imbalance, misleading accuracy, rare attacks.
    - Supervisor-requested dual focus: binary + multiclass.
    - Contributions (bullet list):
        - C1: Build simple, reproducible baseline on UNSW-NB15 for both binary and multiclass tasks.
        - C2: Systematic comparison of three imbalance-handling strategies across three classical ML models.
        - C3: Detailed per-class analysis of minority attacks.
        - C4: Publicly available code/config.
2. **Related Work**
    - Organized as per Section 5.
3. **Dataset Description**
    - Data source, features, attack definitions.
    - Class distribution, imbalance ratios (with a table / barplot).
    - Note on data overlap and challenges.
4. **Methodology**
    - Tasks A & B (and optional C).
    - Preprocessing pipeline.
    - Models and hyperparameters.
    - Imbalance strategies S0/S1/S2.
    - Experimental setup (train/val/test, hardware, software).
5. **Results and Discussion**
    - Binary results (tables + confusion matrix).
    - Multiclass results.
    - Focused analysis for rare classes.
    - Comparison/discussion: what worked, what didn’t.
6. **Threats to Validity / Limitations**
    - Only classical ML, no deep learning.
    - No advanced feature engineering or generative augmentation.
    - Single dataset (UNSW-NB15).
    - Limited hyperparameter search.
7. **Conclusion and Future Work**
    - Summary of key findings.
    - Future: deep learning, focal loss, GAN-based augmentation, multi-dataset experiments, deployment aspects.

---

## 7. Beginner FAQ: Questions Your Team Will Have (and the answers)

I’ll list the questions and give short, direct answers.

### 7.1 Dataset / preprocessing questions

1. **Q: Do we use the Kaggle train/test split or create our own?**
    
    A: Use the **official UNSW training/testing files** as the outer split. Inside training, create a validation split (stratified). Do not mix test into training.
    
2. **Q: What do we do with IP addresses and ports?**
    
    A: For a first paper, **drop IP addresses** (too high cardinality, not generalizable). Keep ports if they are already in numeric columns. Do not overcomplicate.
    
3. **Q: Should we balance before or after splitting?**
    
    A: Always **split first** (train/val/test). Then apply resampling (**only on the training set**). Keep validation/test untouched to avoid information leakage.
    
4. **Q: What about feature selection (PCA, mutual information, etc.)?**
    
    A: Skip for this paper. Mention in future work. You already have enough contribution from imbalance handling and dual task design.
    
5. **Q: How to handle non-numeric categorical features?**
    
    A: Use **one-hot encoding** (e.g., `OneHotEncoder(handle_unknown='ignore')`).
    

### 7.2 Modeling questions

1. **Q: Should we try SVM, k-NN, deep learning, etc.?**
    
    A: No. You risk exploding your scope. Stick to **3 models** (LR, RF, XGBoost/LightGBM) and do them properly.
    
2. **Q: How many hyperparameters should we tune?**
    
    A: Few:
    
    - LR: C (regularization strength)
    - RF: n_estimators, max_depth
    - XGBoost/LightGBM: n_estimators, learning_rate, max_depth
        
        Use a **small grid** and validation set.
        
3. **Q: Is SMOTE mandatory?**
    
    A: No. At minimum, compare **RandomOverSampler vs no resampling vs class_weight**. Add SMOTE only if you have time and can implement it cleanly.
    
4. **Q: How do we handle multiclass ROC-AUC?**
    
    A: Use **macro or weighted one-vs-rest AUC** (scikit-learn supports this).
    

### 7.3 Evaluation / analysis questions

1. **Q: Which metric is “main” in the paper?**
    
    A: Use **macro F1** and **per-class recall** as the main story, especially to highlight minority classes. Accuracy is secondary.
    
2. **Q: What if some rare classes still have 0 recall?**
    
    A: That’s expected and itself is a **research finding**: some attacks are too rare for classical models without stronger techniques. Discuss this clearly.
    
3. **Q: How many tables/figures should we have?**
    
    A: Example:
    
    - Table 1: Dataset class distribution.
    - Table 2: Binary results summary (macro F1, accuracy, AUC).
    - Table 3: Multiclass results summary.
    - Table 4: Per-class recall for key configurations.
    - Figure 1–2: Confusion matrices (best binary, best multiclass).
    - Maybe a bar chart for rare class performance under different strategies.

### 7.4 Literature / writing questions

1. **Q: How many related work papers do we need?**
    
    A: Around **20–30**, focusing on:
    
    - UNSW-NB15 usage,
    - Imbalance handling in IDS,
    - Classical vs deep approaches.
2. **Q: How do we avoid looking “too simple” compared to deep-learning papers?**
    
    A: By:
    
    - Being **methodologically strict**,
    - Clearly showing **where others ignore imbalance / per-class metrics**,
    - Providing a **baseline** that is actually reproducible and transparent.
3. **Q: Should we include equations?**
    
    A: Yes, but minimal: definitions of accuracy, precision, recall, F1, maybe ROC-AUC. No need for heavy math.
    

### 7.5 Team / organization questions

1. **Q: How do we divide work among 4–5 members?**
    
    Example split:
    
    - Person A: Dataset preprocessing + EDA + splitting.
    - Person B: Modeling pipeline (binary).
    - Person C: Modeling pipeline (multiclass).
    - Person D: Evaluation scripts + plots + per-class analysis.
    - Person E: Literature review + writing + integration.
2. **Q: How do we keep the code professional?**
    - Use **Git & GitHub**.
    - Have a **single repo** with:
        - `/data/` (read-only paths / scripts, not raw data if it’s huge),
        - `/src/` (preprocessing, models, evaluation),
        - `/notebooks/` (exploratory, not final),
        - `/docs/` (paper drafts, notes).
    - Use `requirements.txt` or `environment.yml`.
3. **Q: How long will each phase take?**
    
    Roughly (for 1.5–2 months):
    
    - Week 1–2: Dataset understanding, preprocessing, simple baseline run.
    - Week 3–4: Implement full experiment grid (binary+multiclass × models × strategies).
    - Week 5: Analysis, plots, first draft of paper.
    - Week 6+: Refinement, supervisor feedback, polishing.

---

## 8. What you should do next (concrete immediate steps)

1. **Lock the scope**:
    - Tasks A and B as defined.
    - Models = LR + RF + XGBoost (or LightGBM).
    - Strategies = S0, S1, S2.
2. **Set up the repo and baseline**:
    - Load UNSW-NB15 training file.
    - Implement preprocessing pipeline.
    - Train one simple **binary LR** and **multiclass RF** without balancing, just to verify end-to-end.
3. **Start the literature spreadsheet**:
    - Columns: Author, Year, Dataset(s), Models, Imbalance handling, Metrics, Main findings, Limitations.
    - Fill from the key UNSW-NB15 and imbalance papers you find.
4. **Create your paper skeleton** in LaTeX/Word with the structure from Section 6 and start dropping bullet points under each heading as you progress.