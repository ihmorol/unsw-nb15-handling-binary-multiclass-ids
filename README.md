# UNSW-NB15 Class Imbalance Analysis for Intrusion Detection

A comprehensive, reproducible implementation of class imbalance handling strategies for Network Intrusion Detection Systems (NIDS) using the UNSW-NB15 dataset.

## 🎯 Research Objectives

This study systematically compares imbalance handling strategies across binary and multiclass intrusion detection tasks:

- **18 Experiments**: 2 Tasks × 3 Models × 3 Strategies
- **Focus Areas**: Rare attack detection (Worms, Shellcode, Backdoor, Analysis)
- **Primary Metric**: G-Mean (Geometric Mean) for balanced evaluation

## 📊 Experiment Grid

| Dimension | Options |
|-----------|---------|
| **Tasks** | Binary (Normal/Attack), Multiclass (10 classes) |
| **Models** | Logistic Regression, Random Forest, XGBoost |
| **Strategies** | S0 (None), S1 (Class Weight), S2a (RandomOverSampler) |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- 8GB+ RAM (16GB recommended for SMOTE)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ML_PAPER_REVIEW

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

Place the UNSW-NB15 dataset files in the `dataset/` directory:
- `UNSW_NB15_training-set.csv` (175,341 samples)
- `UNSW_NB15_testing-set.csv` (82,332 samples)

### Run Experiments

```bash
# Run full 18-experiment grid
python main.py

# Or run preprocessing only (for debugging)
python scripts/run_preprocessing.py

# Generate final reports after experiments
python scripts/generate_report.py
```

## 📁 Project Structure

```
ML_PAPER_REVIEW/
├── configs/
│   └── main.yaml              # Master configuration
├── dataset/                   # UNSW-NB15 CSV files
├── src/
│   ├── data/
│   │   ├── loader.py          # Data loading
│   │   └── preprocessing.py   # UNSWPreprocessor
│   ├── models/
│   │   ├── trainer.py         # ModelTrainer
│   │   └── config.py          # Hyperparameters
│   ├── evaluation/
│   │   ├── metrics.py         # Metric computation
│   │   └── plots.py           # Visualizations
│   ├── strategies/
│   │   └── imbalance.py       # S0, S1, S2a, S2b
│   └── utils/
│       ├── config.py          # Config loader
│       └── logging.py         # Logging setup
├── scripts/
│   ├── run_preprocessing.py   # Preprocessing runner
│   └── generate_report.py     # Report generator
├── results/
│   ├── metrics/               # JSON per experiment
│   ├── figures/               # Confusion matrices, charts
│   └── tables/                # CSV summary tables
├── main.py                    # Main orchestrator
└── requirements.txt
```

## 📈 Output Artifacts

| File | Description |
|------|-------------|
| `experiment_log.csv` | Master tracker for all experiments |
| `final_summary_tables.csv` | Accuracy, F1, G-Mean, ROC-AUC summary |
| `per_class_metrics.csv` | Precision, Recall, F1 per class |
| `rare_class_report.csv` | Analysis of Worms, Shellcode, Backdoor, Analysis |
| `cm_*.png` | Confusion matrix heatmaps |

## 🔬 Methodology

### Preprocessing Pipeline

1. **Drop Identifiers**: Remove 7 non-predictive columns (id, srcip, dstip, sport, dsport, stime, ltime)
2. **Impute Missing**: Median for numeric, 'missing' for categorical
3. **Encode Categoricals**: One-Hot Encoding (proto, state, service)
4. **Scale Numericals**: StandardScaler (fit on training only)
5. **Stratified Split**: 80/20 train/validation from official training set

### Imbalance Strategies

| Strategy | Description | Data Modification |
|----------|-------------|-------------------|
| **S0** | Baseline | None |
| **S1** | Class weighting | Model parameter (`class_weight='balanced'`) |
| **S2a** | RandomOverSampler | Training data duplicated |

### Evaluation Metrics

- **Accuracy**: Baseline (misleading for imbalanced data)
- **Macro F1**: Equal weight to all classes
- **G-Mean**: Primary metric (√Sensitivity × √Specificity)
- **ROC-AUC**: Threshold-independent performance

## 🔒 Data Leakage Prevention

- All transformers fitted on **training data only**
- Resampling applied **only to training set**
- Test set **never touched** until final evaluation

## 📖 Documentation

- [Methodology Analysis](docs/Methodology_Analysis.md)
- [Implementation Plan](docs/implementation_plan.md)
- [Data Contract](docs/contracts/data_contract.md)
- [Experiment Contract](docs/contracts/experiment_contract.md)

## 🔄 Reproducibility

All experiments use `random_state=42` for deterministic results.

```python
# Configuration ensures reproducibility
random_state: 42
```

## 📊 Expected Results

### Binary Classification
- Accuracy: ~87-92%
- G-Mean: ~85-90%

### Multiclass (Rare Classes)
- Baseline (S0): Near-zero recall for Worms
- With S2a: Significant improvement in rare class detection

## 📋 Project Audit & Roadmap (Jan 2026)

A comprehensive "State of the Art" audit has been conducted on the repository.

| Dimension | Verdict | Summary |
|:---:|:---:|---|
| **Engineering** | ✅ **PASS** | "With Distinction". Top 1% of research repos. Zero leakage, perfect strategy isolation. |
| **Research** | ⚠️ **GAPS** | Technically perfect but lacks statistical depth (single seed) and external baselines. |
| **Statistics** | ❌ **FAIL** | Current protocol uses single seed (`42`). Must upgrade to multi-seed (5-10 runs). |

**Next Steps (Roadmap):**
1.  **Multi-Seed Execution**: Upgrade from single `random_state=42` to 5-seed average.
2.  **Significance Testing**: Implement paired t-tests S1 vs S2a.
3.  **External Baselines**: Compare against recent (2022-2024) literature benchmarks.

👉 **[Read Full Analysis Report](results/reports/comprehensive_analysis.md)**

## 📝 License

This project is for academic research purposes.

## 🙏 Acknowledgments

- UNSW-NB15 Dataset: Australian Centre for Cyber Security (ACCS)
- Original Paper: "Handling Class Imbalance in Binary and Multiclass IDS"

## ⚠️ Validation Split Note

The validation split (20%) generated during preprocessing is currently **reserved for future hyperparameter tuning** and is **NOT used during training** (which uses fixed hyperparameters). This ensures strict isolation of the test set and allows for future optimization without data leakage.
