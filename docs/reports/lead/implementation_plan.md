# Implementation Plan: UNSW-NB15 Imbalance Analysis

> **Document Version:** 2.0  
> **Last Updated:** 2026-01-17  
> **Status:** READY FOR IMPLEMENTATION

---

## Goal

Implement a comprehensive, reproducible analysis of class imbalance handling strategies for Intrusion Detection on the UNSW-NB15 dataset, comparing binary and multiclass classification across three classical ML models and three imbalance strategies.

---

## User Review Required

> [!IMPORTANT]
> **Dataset Availability**: Ensure `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` are present in `dataset/` directory. The code will fail if these files are missing.

> [!WARNING]
> **Processing Power**: SMOTE on multiclass data can produce ~560K samples. Ensure the environment has:
> - Minimum: 8GB RAM
> - Recommended: 16GB RAM
> - Disk Space: 2GB for processed data and models

> [!NOTE]
> **Execution Time Estimates**:
> - Full 18-experiment grid: ~2-4 hours (depending on hardware)
> - Single experiment: ~5-15 minutes

---

## Proposed Changes

### Phase 1: Environment & Repository Structure

#### [NEW] `requirements.txt`

```
# Core Data Science
pandas>=2.0.0
numpy>=1.24.0

# Machine Learning
scikit-learn>=1.3.0
xgboost>=2.0.0

# Imbalanced Learning
imbalanced-learn>=0.11.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Configuration
pyyaml>=6.0

# Development
jupyter>=1.0.0
```

#### [MODIFY] Directory Structure

```
ML_PAPER_REVIEW/
├── configs/
│   └── main.yaml              # Master configuration
├── dataset/
│   ├── UNSW_NB15_training-set.csv
│   ├── UNSW_NB15_testing-set.csv
│   └── NUSW-NB15_features.csv
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # Data loading utilities
│   │   └── preprocessing.py   # UNSWPreprocessor class
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py         # ModelTrainer class
│   │   └── config.py          # Model hyperparameters
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py         # Metric calculation functions
│   │   └── plots.py           # Visualization functions
│   ├── strategies/
│   │   ├── __init__.py
│   │   └── imbalance.py       # S0, S1, S2 implementations
│   └── utils/
│       ├── __init__.py
│       ├── config.py          # Configuration loader
│       └── logging.py         # Experiment logging
├── scripts/
│   ├── run_preprocessing.py   # Phase 1 runner
│   ├── run_experiments.py     # Phase 2 runner
│   └── generate_report.py     # Phase 3 runner
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing_test.ipynb
│   └── 03_results_analysis.ipynb
├── results/
│   ├── metrics/               # JSON metrics per experiment
│   ├── models/                # Saved model objects
│   ├── figures/               # Generated plots
│   └── experiment_log.csv     # Master experiment tracker
├── docs/
│   ├── Methodology_Analysis.md
│   ├── implementation_plan.md # This document
│   └── contracts/
│       ├── data_contract.md
│       └── experiment_contract.md
└── main.py                    # Master orchestrator
```

---

### Phase 2: Data Pipeline Implementation

#### [NEW] `src/data/loader.py`

**Purpose:** Load raw data from CSV files.

```python
class DataLoader:
    """Load UNSW-NB15 dataset from CSV files."""
    
    def __init__(self, config: dict):
        self.train_path = config['data']['train_path']
        self.test_path = config['data']['test_path']
    
    def load_train(self) -> pd.DataFrame:
        """Load training set."""
        pass
    
    def load_test(self) -> pd.DataFrame:
        """Load test set (only for final evaluation)."""
        pass
```

#### [NEW] `src/data/preprocessing.py`

**Class: `UNSWPreprocessor`**

| Method | Description | Input | Output |
|--------|-------------|-------|--------|
| `__init__(config)` | Initialize with configuration | Config dict | None |
| `load_data()` | Read CSVs from configured paths | None | Raw DataFrames |
| `clean_data(df)` | Drop identifier columns | DataFrame | Cleaned DataFrame |
| `handle_missing(df)` | Impute missing values | DataFrame | Imputed DataFrame |
| `encode_categorical(df)` | One-Hot encode categoricals | DataFrame | Encoded DataFrame |
| `scale_numerical(df)` | StandardScaler for numerics | DataFrame | Scaled DataFrame |
| `prepare_labels(df, task)` | Create binary/multiclass labels | DataFrame, 'binary'\|'multi' | y array |
| `fit_transform(X_train)` | Fit on training, transform | DataFrame | np.ndarray |
| `transform(X)` | Transform only (for val/test) | DataFrame | np.ndarray |
| `get_splits()` | Return train/val/test splits | None | Tuple of arrays |

**Implementation Details:**

```python
class UNSWPreprocessor:
    """
    Unified preprocessing pipeline for UNSW-NB15.
    
    Ensures:
    - No data leakage (fit on train only)
    - Consistent encoding across splits
    - Stratified validation split
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.drop_columns = config['data']['drop_columns']
        self.target_binary = config['data']['target_binary']
        self.target_multi = config['data']['target_multiclass']
        self.val_size = config.get('validation_size', 0.20)
        self.random_state = config.get('random_state', 42)
        
        # Fitted transformers
        self.num_imputer = None
        self.cat_imputer = None
        self.scaler = None
        self.encoder = None
        self.label_encoder = None
        
        # Feature tracking
        self.numerical_cols = []
        self.categorical_cols = ['proto', 'state', 'service']
        self.feature_names = []
    
    def fit_transform(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """Fit all transformers on training data and transform."""
        # 1. Impute missing
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        
        # 2. Encode categoricals
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        # 3. Scale numericals
        self.scaler = StandardScaler()
        
        # Pipeline execution...
        pass
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform using fitted transformers (for val/test)."""
        pass
```

**Data Leakage Prevention Checklist:**

- [ ] Imputer statistics computed on training only
- [ ] Scaler mean/std computed on training only
- [ ] Encoder categories learned from training only
- [ ] Resampling NEVER applied to validation/test

---

### Phase 3: Imbalance Strategies Implementation

#### [NEW] `src/strategies/imbalance.py`

**Strategy Implementations:**

```python
class ImbalanceStrategy:
    """Base class for imbalance handling strategies."""
    
    def apply(self, X_train, y_train) -> Tuple[np.ndarray, np.ndarray]:
        """Apply strategy and return modified training data."""
        raise NotImplementedError


class S0_NoBalancing(ImbalanceStrategy):
    """Baseline: No modification to training data."""
    
    def apply(self, X_train, y_train):
        return X_train, y_train
    
    def get_sample_weight(self, y_train) -> Optional[np.ndarray]:
        return None
    
    def get_class_weight(self) -> Optional[dict]:
        return None


class S1_ClassWeight(ImbalanceStrategy):
    """Apply class weights during model training."""
    
    def apply(self, X_train, y_train):
        return X_train, y_train
    
    def get_class_weight(self) -> str:
        return 'balanced'
    
    def compute_sample_weights(self, y_train) -> np.ndarray:
        """For models that don't support class_weight parameter."""
        weights = compute_class_weight('balanced', 
                                        classes=np.unique(y_train), 
                                        y=y_train)
        return weights[y_train]


class S2a_RandomOverSampler(ImbalanceStrategy):
    """Random over-sampling of minority classes."""
    
    def __init__(self, random_state=42):
        self.sampler = RandomOverSampler(random_state=random_state)
    
    def apply(self, X_train, y_train):
        return self.sampler.fit_resample(X_train, y_train)


class S2b_SMOTE(ImbalanceStrategy):
    """Synthetic Minority Over-sampling Technique."""
    
    def __init__(self, k_neighbors=5, random_state=42):
        self.sampler = SMOTE(k_neighbors=k_neighbors, 
                             random_state=random_state,
                             n_jobs=-1)
    
    def apply(self, X_train, y_train):
        try:
            return self.sampler.fit_resample(X_train, y_train)
        except ValueError as e:
            # Fallback to ROS if k_neighbors > minority class samples
            print(f"SMOTE failed: {e}. Falling back to RandomOverSampler.")
            ros = RandomOverSampler(random_state=self.sampler.random_state)
            return ros.fit_resample(X_train, y_train)
```

---

### Phase 4: Model Training Framework

#### [NEW] `src/models/trainer.py`

**Class: `ModelTrainer`**

```python
class ModelTrainer:
    """Unified training interface for all models."""
    
    MODELS = {
        'lr': LogisticRegression,
        'rf': RandomForestClassifier,
        'xgb': XGBClassifier
    }
    
    def __init__(self, config: dict):
        self.config = config
        self.random_state = config.get('random_state', 42)
    
    def get_model(self, 
                  model_name: str, 
                  task: str,
                  class_weight: Optional[str] = None) -> BaseEstimator:
        """
        Get model instance with appropriate configuration.
        
        Args:
            model_name: 'lr', 'rf', or 'xgb'
            task: 'binary' or 'multi'
            class_weight: None, 'balanced', or dict
        """
        pass
    
    def train(self, 
              model: BaseEstimator,
              X_train: np.ndarray,
              y_train: np.ndarray,
              sample_weight: Optional[np.ndarray] = None) -> BaseEstimator:
        """Train model and return fitted instance."""
        pass
    
    def predict(self, model: BaseEstimator, X: np.ndarray) -> np.ndarray:
        """Get class predictions."""
        pass
    
    def predict_proba(self, model: BaseEstimator, X: np.ndarray) -> np.ndarray:
        """Get probability predictions for ROC-AUC."""
        pass
```

#### [NEW] `src/models/config.py`

**Hyperparameter Configurations:**

```python
MODEL_CONFIGS = {
    'lr': {
        'binary': {
            'C': 1.0,
            'solver': 'saga',
            'max_iter': 1000,
            'n_jobs': -1
        },
        'multi': {
            'C': 1.0,
            'solver': 'saga',
            'max_iter': 1000,
            'multi_class': 'multinomial',
            'n_jobs': -1
        }
    },
    'rf': {
        'binary': {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'n_jobs': -1
        },
        'multi': {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'n_jobs': -1
        }
    },
    'xgb': {
        'binary': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 10,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'n_jobs': -1
        },
        'multi': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 10,
            'use_label_encoder': False,
            'eval_metric': 'mlogloss',
            'objective': 'multi:softprob',
            'n_jobs': -1
        }
    }
}

# Hyperparameter tuning grid (for optional tuning)
TUNING_GRID = {
    'lr': {'C': [0.01, 0.1, 1, 10]},
    'rf': {'n_estimators': [100, 200], 'max_depth': [10, 20]},
    'xgb': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [5, 10]}
}
```

---

### Phase 5: Evaluation & Reporting

#### [NEW] `src/evaluation/metrics.py`

**Metric Calculation Functions:**

```python
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.metrics import geometric_mean_score


def compute_all_metrics(y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        y_pred_proba: np.ndarray,
                        task: str) -> dict:
    """
    Compute comprehensive metrics for a single experiment.
    
    Returns:
        Dictionary with overall and per-class metrics
    """
    metrics = {
        'overall': {},
        'per_class': {},
        'confusion_matrix': None
    }
    
    # Overall metrics
    metrics['overall']['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['overall']['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['overall']['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
    metrics['overall']['g_mean'] = geometric_mean_score(y_true, y_pred, average='macro')
    
    # ROC-AUC
    if task == 'binary':
        metrics['overall']['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
    else:
        try:
            metrics['overall']['roc_auc'] = roc_auc_score(
                y_true, y_pred_proba, 
                multi_class='ovr', 
                average='macro'
            )
        except ValueError:
            metrics['overall']['roc_auc'] = None
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    for class_name, class_metrics in report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics['per_class'][class_name] = {
                'precision': class_metrics['precision'],
                'recall': class_metrics['recall'],
                'f1': class_metrics['f1-score'],
                'support': class_metrics['support']
            }
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    return metrics


def compute_rare_class_analysis(metrics: dict, rare_classes: list) -> dict:
    """
    Extract performance specifically for rare attack classes.
    
    Args:
        metrics: Full metrics dictionary
        rare_classes: List of rare class names ['Worms', 'Shellcode', 'Backdoor', 'Analysis']
    
    Returns:
        Dictionary with rare class performance summary
    """
    rare_analysis = {}
    for cls in rare_classes:
        if str(cls) in metrics['per_class']:
            rare_analysis[cls] = metrics['per_class'][str(cls)]
    return rare_analysis
```

#### [NEW] `src/evaluation/plots.py`

**Visualization Functions:**

```python
def plot_confusion_matrix(cm: np.ndarray, 
                          labels: list,
                          save_path: str,
                          title: str = 'Confusion Matrix') -> None:
    """Generate and save confusion matrix heatmap."""
    pass


def plot_roc_curve(y_true: np.ndarray,
                   y_pred_proba: np.ndarray,
                   save_path: str) -> None:
    """Generate and save ROC curve."""
    pass


def plot_rare_class_comparison(results: dict,
                               save_path: str) -> None:
    """Bar chart comparing rare class recall across strategies."""
    pass


def plot_strategy_comparison(results: dict,
                             metric: str,
                             save_path: str) -> None:
    """Compare all strategies for a given metric."""
    pass
```

---

### Phase 6: Main Orchestrator

#### [NEW] `main.py`

**Experiment Orchestrator:**

```python
#!/usr/bin/env python3
"""
Main orchestrator for UNSW-NB15 Imbalance Study.
Runs the complete 18-experiment grid.
"""

import yaml
import json
import logging
from datetime import datetime
from pathlib import Path

from src.data.preprocessing import UNSWPreprocessor
from src.models.trainer import ModelTrainer
from src.strategies.imbalance import S0_NoBalancing, S1_ClassWeight, S2a_RandomOverSampler, S2b_SMOTE
from src.evaluation.metrics import compute_all_metrics, compute_rare_class_analysis
from src.evaluation.plots import plot_confusion_matrix

# Configuration
TASKS = ['binary', 'multi']
MODELS = ['lr', 'rf', 'xgb']
STRATEGIES = {
    's0': S0_NoBalancing(),
    's1': S1_ClassWeight(),
    's2a': S2a_RandomOverSampler(random_state=42),
    # 's2b': S2b_SMOTE(random_state=42)  # Optional
}

RARE_CLASSES = ['Worms', 'Shellcode', 'Backdoor', 'Analysis']


def run_experiment(task, model_name, strategy_name, 
                   X_train, y_train, X_val, y_val, X_test, y_test,
                   config, results_dir):
    """Run a single experiment and save results."""
    
    experiment_id = f"{task}_{model_name}_{strategy_name}"
    logging.info(f"Starting experiment: {experiment_id}")
    
    # Apply imbalance strategy
    strategy = STRATEGIES[strategy_name]
    X_train_bal, y_train_bal = strategy.apply(X_train, y_train)
    
    # Get class weight if applicable
    class_weight = None
    if hasattr(strategy, 'get_class_weight'):
        class_weight = strategy.get_class_weight()
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    model = trainer.get_model(model_name, task, class_weight)
    
    # Train
    trainer.train(model, X_train_bal, y_train_bal)
    
    # Evaluate on TEST set (final evaluation)
    y_pred = trainer.predict(model, X_test)
    y_pred_proba = trainer.predict_proba(model, X_test)
    
    # Compute metrics
    metrics = compute_all_metrics(y_test, y_pred, y_pred_proba, task)
    
    # Add rare class analysis for multiclass
    if task == 'multi':
        metrics['rare_class_analysis'] = compute_rare_class_analysis(metrics, RARE_CLASSES)
    
    # Save results
    results_path = results_dir / 'metrics' / f"{experiment_id}.json"
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save confusion matrix plot
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        labels=['Normal', 'Attack'] if task == 'binary' else None,
        save_path=str(results_dir / 'figures' / f"cm_{experiment_id}.png"),
        title=f"Confusion Matrix: {experiment_id}"
    )
    
    logging.info(f"Completed: {experiment_id} | Accuracy: {metrics['overall']['accuracy']:.4f}")
    
    return experiment_id, metrics


def main():
    """Main entry point."""
    # Load config
    with open('configs/main.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup results directory
    results_dir = Path(config['results_dir'])
    for subdir in ['metrics', 'models', 'figures']:
        (results_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = UNSWPreprocessor(config)
    preprocessor.load_data()
    
    # Prepare data for both tasks
    splits = {
        'binary': preprocessor.get_splits('binary'),
        'multi': preprocessor.get_splits('multi')
    }
    
    # Run all experiments
    all_results = []
    for task in TASKS:
        X_train, y_train, X_val, y_val, X_test, y_test = splits[task]
        
        for model_name in MODELS:
            for strategy_name in STRATEGIES.keys():
                exp_id, metrics = run_experiment(
                    task, model_name, strategy_name,
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    config, results_dir
                )
                all_results.append({
                    'experiment_id': exp_id,
                    'task': task,
                    'model': model_name,
                    'strategy': strategy_name,
                    'accuracy': metrics['overall']['accuracy'],
                    'macro_f1': metrics['overall']['macro_f1'],
                    'g_mean': metrics['overall']['g_mean'],
                    'roc_auc': metrics['overall']['roc_auc']
                })
    
    # Save experiment log
    import pandas as pd
    log_df = pd.DataFrame(all_results)
    log_df.to_csv(results_dir / 'experiment_log.csv', index=False)
    
    logging.info("All experiments completed successfully!")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
```

---

### Phase 7: Configuration Management

#### [MODIFY] `configs/main.yaml`

**Complete Configuration:**

```yaml
# ===================================
# UNSW-NB15 Experiment Configuration
# ===================================

# Data Paths
data:
  train_path: "dataset/UNSW_NB15_training-set.csv"
  test_path: "dataset/UNSW_NB15_testing-set.csv"
  
  # Columns to drop (identifiers)
  drop_columns:
    - "id"
    - "srcip"
    - "dstip"
    - "sport"
    - "dsport"
    - "stime"
    - "ltime"
  
  # Categorical columns for encoding
  categorical_columns:
    - "proto"
    - "state"
    - "service"
  
  # Target columns
  target_binary: "label"
  target_multiclass: "attack_cat"
  
  # Class ordering for multiclass
  multiclass_labels:
    - "Normal"
    - "Fuzzers"
    - "Analysis"
    - "Backdoor"
    - "DoS"
    - "Exploits"
    - "Generic"
    - "Reconnaissance"
    - "Shellcode"
    - "Worms"

# Reproducibility
random_state: 42

# Data Splitting
validation_size: 0.20  # 20% of training data for validation

# Experiment Grid
experiments:
  tasks:
    - binary
    - multi
  models:
    - lr
    - rf
    - xgb
  strategies:
    - s0
    - s1
    - s2a
    # - s2b  # Optional SMOTE

# Rare Class Analysis
rare_classes:
  - "Worms"
  - "Shellcode"
  - "Backdoor"
  - "Analysis"

# Output Paths
results_dir: "results"
reports_dir: "reports"

# Logging
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

---

## Verification Plan

### Automated Tests

#### Unit Tests (`tests/test_preprocessing.py`)

```python
def test_preprocessor_shapes():
    """Verify output shapes match expected dimensions."""
    pass

def test_no_data_leakage():
    """Ensure test data statistics are not used in fitting."""
    pass

def test_stratification():
    """Verify class distribution preserved in validation split."""
    pass
```

#### Integration Test (`tests/test_pipeline.py`)

```python
def test_end_to_end_pipeline():
    """Run mini-experiment with subset of data."""
    # Use first 1000 rows only
    # Verify pipeline completes without errors
    # Verify output files are created
    pass
```

### Manual Verification Checklist

| Check | Description | Status |
|-------|-------------|--------|
| Data Leakage | Resampling only on training split | [ ] |
| Metric Validity | Accuracy ≠ 0 and ≠ 1 | [ ] |
| File Outputs | All 18 experiment JSONs created | [ ] |
| Confusion Matrix | Plots generated for all experiments | [ ] |
| Rare Class Focus | Metrics for Worms, Shellcode, Backdoor, Analysis logged | [ ] |
| Reproducibility | Same seed produces identical results | [ ] |

---

## Execution Commands

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Preprocessing
```bash
python scripts/run_preprocessing.py
```

### 3. Run Full Experiment Grid
```bash
python main.py
```

### 4. Generate Report
```bash
python scripts/generate_report.py
```

---

## Output Artifacts Summary

| Artifact | Location | Description |
|----------|----------|-------------|
| Metrics JSON | `results/metrics/{exp_id}.json` | Per-experiment metrics |
| Confusion Matrix | `results/figures/cm_{exp_id}.png` | Heatmap visualization |
| Experiment Log | `results/experiment_log.csv` | Summary of all experiments |
| Trained Models | `results/models/{exp_id}.joblib` | Serialized models (optional) |
| Plots | `results/figures/*.png` | ROC curves, comparisons |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-17 | Initial plan |
| 2.0 | 2026-01-17 | Complete enhancement with code specifications |
