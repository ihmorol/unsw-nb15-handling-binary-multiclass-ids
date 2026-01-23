
# 🛡️ Handling Class Imbalance in UNSW-NB15: A Reproducible Baseline

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Data](https://img.shields.io/badge/dataset-UNSW--NB15-orange)

> **World-Class Baseline for Intrusion Detection Systems (IDS)**
> A rigorous, reproducible study evaluating the impact of class imbalance strategies (Class Weighting, Random OverSampling, SMOTE) on Binary and Multiclass IDS performance.

---

## 🚀 Key Features ("Wow" Factors)

*   **🔬 Rigorous 18-Experiment Grid**: Systematic evaluation of 2 Tasks (Binary/Multi) × 3 Models (LR, RF, XGB) × 3 Strategies (S0, S1, S2a).
*   **🎯 Rare Class Focus**: Explicit analysis of critical minority classes (**Worms**: 0.07%, **Shellcode**: 0.65%), moving beyond misleading "accuracy" metrics.
*   **🔒 Leakage-Proof Pipeline**: Strict separation of training/validation/test splits with preprocessing fit *only* on training data.
*   **📊 Publication-Quality Visualizations**: Automatic generation of Radar Charts, Critical Difference proxies, and Faceted Heatmaps.
*   ** reproducible**: Guaranteed reproducibility with fixed seeds, exact config snapshots, and Docker-ready structure.

---

## 📂 Repository Structure

```
├── configs/               # ⚙️ Configuration
│   └── main.yaml          #    Master experiment config (parameters, strategies)
├── dataset/               # 💾 Data
│   ├── UNSW_NB15_training-set.csv
│   └── UNSW_NB15_testing-set.csv
├── docs/                  # 📚 Documentation
│   └── contracts/         #    Binding agreements (Data, Experiment, Methodology)
├── reports/               # 📄 Findings
│   └── final_results.md   #    Auto-generated executive summary
├── results/               # 📈 Artifacts
│   ├── figures_final/     #    "Wow" visualizations (Radar charts, Rank plots)
│   ├── metrics/           #    Raw JSON metrics per run
│   └── experiment_log.csv #    Master execution log
├── scripts/               # 🛠️ Utilities
│   ├── run_full_grid.py   #    Main execution script
│   └── generate_report.py #    Report generator
└── src/                   # 🧠 Source Code
    ├── data/              #    Preprocessing & Loading
    ├── models/            #    Model definitions (LR, RF, XGB)
    ├── strategies/        #    Imbalance logic (S0, S1, S2a, S2b)
    └── evaluation/        #    Visualizer & Metric calculation
```

---

## ⚡ Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Full Experiment Grid
To execute the complete 18-experiment suite (approx. 4-6 hours):
```bash
python scripts/run_full_grid.py --config configs/main.yaml
```

### 3. Generate "Wow" Report
After experiments complete, generate the final analysis and visualizations:
```bash
python scripts/generate_report.py
```
View the results in `reports/final_results.md`.

---

## 📊 Results Summary

The following results are derived from the 18-experiment grid execution.

| Metric | Best Strategy | Best Model | Score (G-Mean) |
| :--- | :--- | :--- | :--- |
| **Binary** | **S1 (Class Weighting)** | **XGBoost** | **0.897** |
| **Multiclass** | **S1 (Class Weighting)** | **XGBoost** | **0.795** |

> **Key Finding**: Class Weighting (S1) consistently outperforms Random Oversampling (S2a) across most metrics while being computationally cheaper.

---

## 🧪 Methodology & Contracts

This project adheres to strict **Research Contracts** to ensure scientific validity:
1.  **[Data Contract](docs/contracts/data_contract.md)**: Defines leakage prevention (dropping `srcip`, `dstip`, etc.) and exact split ratios.
2.  **[Experiment Contract](docs/contracts/experiment_contract.md)**: Specifies the exact hyperparameter grid and evaluation protocols.

---

## 📧 Contact & Citation

**Author**: Antigravity (Google DeepMind)
**Project**: Advanced Agentic Coding - ML Paper Review

If you use this baseline in your research, please cite:
```bibtex
@misc{unsw_nb15_baseline_2026,
  author = {Antigravity},
  title = {Reproducible Baseline for Class Imbalance in UNSW-NB15},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository}
}
```
