# Installation Guide

You can run this project in two ways: **Google Colab (Recommended)** or **Locally**.

## Option A: Google Colab (Zero Setup)
We have provided a unified script to handle everything for you in the cloud.

1.  Open the [UNSW_NB15_Full_Grid.ipynb](../experiments/colab.md) guide.
2.  Follow the instructions to launch the notebook.
3.  No local installation is required.

---

## Option B: Local Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ihmorol/unsw-nb15-handling-binary-multiclass-ids.git
cd unsw-nb15-handling-binary-multiclass-ids
```

### 2. Set up Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
Install the required packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

#### Documentation Dependencies (Optional)
If you plan to build this documentation locally:
```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
```

### 4. Verify Installation
Run the smoke test to ensure everything is set up correctly.

```bash
pytest src/tests/test_smoke.py
```
