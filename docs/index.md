# UNSW-NB15 IDS Research Workspace

<p align="center">
  <img src="https://img.shields.io/badge/Research-Class%20Imbalance-blueviolet" alt="Research Focus">
  <img src="https://img.shields.io/badge/Dataset-UNSW--NB15-orange" alt="Dataset">
  <img src="https://img.shields.io/badge/Framework-Scikit--Learn%20%7C%20XGBoost-green" alt="Framework">
  <img src="https://img.shields.io/badge/Documentation-MkDocs%20Material-blue" alt="Docs">
</p>

Welcome to the official documentation for the **Handling Class Imbalance in Binary and Multiclass Intrusion Detection on the UNSW-NB15 Dataset** research project.

---

## 🎯 Mission Statement

> The goal of this project is to provide a **rigorous, reproducible, and transparent baseline** for evaluating the impact of class imbalance strategies on Network Intrusion Detection Systems (IDS).

Traditional IDS models trained on the UNSW-NB15 dataset often achieve high overall accuracy by simply learning to predict the dominant class (Normal traffic). This creates a dangerous illusion of security, as critical minority attack classes like **Worms (0.07%)** and **Shellcode (0.65%)** are almost entirely missed.

This project directly addresses this problem by:

1.  **Systematically Evaluating** three imbalance handling strategies (Baseline, Class Weighting, Random Oversampling).
2.  **Focusing on Rare Class Detection** using metrics like G-Mean and Macro-F1, which treat all classes equally.
3.  **Enforcing Strict Data Contracts** to prevent data leakage and ensure scientific validity.

---

## ✨ Key Features

| Feature                    | Description                                                                   |
| :------------------------- | :---------------------------------------------------------------------------- |
| **🔬 18-Experiment Grid**  | Comprehensive evaluation: 2 Tasks × 3 Models × 3 Strategies.                  |
| **🎯 Rare Class Focus**    | Explicit analysis of Worms, Shellcode, Backdoor, and Analysis attack types.  |
| **🔒 Leakage-Proof**       | Strict contracts: preprocessing fitted *only* on training data.              |
| **📊 Automated Reports**   | Radar charts, heatmaps, and per-class metrics generated automatically.       |
| **☁️ Colab Ready**         | [Zero-setup cloud execution](experiments/colab.md) via Google Colab.         |

---

## 🗺️ Documentation Map

Navigate the documentation using the links below:

### Getting Started
*   **[Introduction](getting-started/index.md)**: Prerequisites and setup roadmap.
*   **[Installation](getting-started/installation.md)**: Local and Colab installation guides.
*   **[Quickstart](getting-started/quickstart.md)**: Run your first experiment in minutes.

### Research
*   **[Methodology](research/methodology.md)**: Deep dive into S0, S1, S2 strategies with mathematical definitions.
*   **[Findings](research/findings.md)**: Comprehensive analysis of results with tables and visualizations.
*   **[Data Contract](contracts/data_contract.md)**: Binding rules for data handling.
*   **[Experiment Contract](contracts/experiment_contract.md)**: Fixed hyperparameters and evaluation protocols.

### Experiments
*   **[Running Experiments](experiments/running.md)**: CLI reference and output structure.
*   **[Google Colab Guide](experiments/colab.md)**: Full tutorial for cloud execution.
*   **[Reproducibility](experiments/reproducibility.md)**: How we ensure deterministic results.

### API Reference
*   **[Data Module](api/data.md)**: `src.data` - Preprocessing, loading.
*   **[Models Module](api/models.md)**: `src.models` - LR, RF, XGB wrappers.
*   **[Evaluation Module](api/evaluation.md)**: `src.evaluation` - Metrics, visualizers.
*   **[Strategies Module](api/strategies.md)**: `src.strategies` - Imbalance handling logic.

---

## 📂 Repository Architecture

```
.
├── configs/
│   └── main.yaml          # Master configuration file
├── dataset/
│   ├── UNSW_NB15_training-set.csv
│   └── UNSW_NB15_testing-set.csv
├── docs/                  # <- You are here
├── reports/               # Auto-generated analysis reports
├── results/               # Experiment outputs (metrics, figures, logs)
├── scripts/               # Utility scripts (grid runners, visualizers)
├── src/                   # Core source code
│   ├── data/              # Data loading & preprocessing
│   ├── models/            # Model definitions
│   ├── evaluation/        # Metrics & visualization
│   └── strategies/        # Imbalance handling
└── tests/                 # Unit tests
```

---

## 🚀 Next Steps

Ready to dive in? Start with the **[Quickstart Guide](getting-started/quickstart.md)** to run your first experiment, or jump directly to **[Running in Colab](experiments/colab.md)** for zero-setup execution.
