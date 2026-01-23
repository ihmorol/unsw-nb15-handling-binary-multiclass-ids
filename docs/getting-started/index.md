# Getting Started

Welcome! This section guides you through setting up your environment and running your first experiment.

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python**  | 3.9+    | `python --version` to check. |
| **pip**     | Latest  | `pip install --upgrade pip` |
| **Git**     | Any     | For cloning the repository. |
| **RAM**     | 8GB+    | 16GB recommended for full grid runs. |

> [!TIP]
> **No local setup?** Run everything in the cloud using our [Google Colab Guide](../experiments/colab.md).

---

## 🗺️ Setup Roadmap

Follow these steps in order:

<div style="display: flex; gap: 20px;">

<div style="flex: 1; border: 1px solid #ccc; padding: 15px; border-radius: 8px;">
<h3>1️⃣ Installation</h3>
<p>Clone the repository and install dependencies.</p>
<a href="installation.md">→ Installation Guide</a>
</div>

<div style="flex: 1; border: 1px solid #ccc; padding: 15px; border-radius: 8px;">
<h3>2️⃣ Quickstart</h3>
<p>Run a single experiment to verify your setup.</p>
<a href="quickstart.md">→ Quickstart Guide</a>
</div>

<div style="flex: 1; border: 1px solid #ccc; padding: 15px; border-radius: 8px;">
<h3>3️⃣ Full Grid</h3>
<p>Execute the complete 18-experiment grid.</p>
<a href="../experiments/running.md">→ Running Experiments</a>
</div>

</div>

---

## 🤔 Need Help?

If you encounter issues:
1.  Check the [Troubleshooting section](installation.md#troubleshooting) in the Installation Guide.
2.  Review the [FAQ](#faq) below.
3.  Open an issue on [GitHub](https://github.com/ihmorol/unsw-nb15-handling-binary-multiclass-ids/issues).

---

## ❓ FAQ

**Q: Can I use my own dataset?**
A: The pipeline is designed for UNSW-NB15 with specific column names. Adapting to other datasets requires modifying `src/data/loader.py` and the configuration files.

**Q: How long does the full grid take?**
A: Approximately 3-6 hours depending on your hardware. Binary tasks complete in ~45 minutes; Multiclass with S2a (oversampling to ~560k samples) takes the longest.

**Q: Do I need a GPU?**
A: Not required. All models (LR, RF, XGB) are CPU-based. A GPU can speed up XGBoost slightly if configured with `tree_method='gpu_hist'`.
