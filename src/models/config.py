"""
Model hyperparameter configurations.

This module contains the default hyperparameter configurations for
all three model families used in the experiments:
- Logistic Regression (LR)
- Random Forest (RF)
- XGBoost (XGB)

Configurations are provided for both binary and multiclass tasks.
"""

MODEL_CONFIGS = {
    'lr': {
        'binary': {
            'C': 1.0,
            'solver': 'saga',           # Supports L1/L2/ElasticNet
            'max_iter': 1000,
            'penalty': 'l2',
            'n_jobs': -1,
            'verbose': 1                # Show optimization progress
        },
        'multi': {
            'C': 1.0,
            'solver': 'saga',
            'max_iter': 1000,
            'penalty': 'l2',
            'multi_class': 'multinomial',  # Softmax regression
            'n_jobs': -1,
            'verbose': 1                # Show optimization progress
        }
    },
    'rf': {
        'binary': {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'n_jobs': -1,
            'verbose': 1                # Show tree-building progress
        },
        'multi': {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'n_jobs': -1,
            'verbose': 1                # Show tree-building progress
        }
    },
    'xgb': {
        'binary': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 10,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'objective': 'binary:logistic',
            'n_jobs': -1,
            'verbosity': 1              # Show training progress
        },
        'multi': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 10,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'use_label_encoder': False,
            'eval_metric': 'mlogloss',
            'objective': 'multi:softprob',
            'n_jobs': -1,
            'verbosity': 1              # Show training progress
        }
    }
}

# Optional: Hyperparameter tuning grids
TUNING_GRIDS = {
    'lr': {
        'C': [0.01, 0.1, 1.0, 10.0]
    },
    'rf': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None]
    },
    'xgb': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [5, 10]
    }
}
