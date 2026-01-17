"""
Model training framework for UNSW-NB15 experiments.

This module provides a unified interface for training all three model
families (LR, RF, XGB) with consistent handling of class weights and
sample weights across different imbalance strategies.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator
from typing import Optional, Dict, Any
import logging
import time

from .config import MODEL_CONFIGS

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Unified training interface for all classification models.
    
    Handles the complexity of applying class weights and sample weights
    appropriately for each model type and strategy combination.
    
    Attributes:
        config: Configuration dictionary
        random_state: Random seed for reproducibility
    """
    
    # Model class mapping
    MODELS = {
        'lr': LogisticRegression,
        'rf': RandomForestClassifier,
        'xgb': XGBClassifier
    }
    
    def __init__(self, config: dict):
        """
        Initialize ModelTrainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.random_state = config.get('random_state', 42)
    
    def get_model(self,
                  model_name: str,
                  task: str,
                  class_weight: Optional[str] = None,
                  scale_pos_weight: Optional[float] = None,
                  num_classes: int = 2) -> BaseEstimator:
        """
        Create and configure a model instance.
        
        Args:
            model_name: 'lr', 'rf', or 'xgb'
            task: 'binary' or 'multi'
            class_weight: 'balanced' or None (for LR/RF)
            scale_pos_weight: For XGBoost binary with S1
            num_classes: Number of classes for multiclass XGBoost
            
        Returns:
            Configured but unfitted model instance
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(self.MODELS.keys())}")
        
        # Get base configuration
        model_config = MODEL_CONFIGS[model_name][task].copy()
        model_config['random_state'] = self.random_state
        
        # Apply class weight for LR and RF
        if model_name in ['lr', 'rf'] and class_weight is not None:
            model_config['class_weight'] = class_weight
        
        # Apply scale_pos_weight for XGBoost binary
        if model_name == 'xgb' and task == 'binary' and scale_pos_weight is not None:
            model_config['scale_pos_weight'] = scale_pos_weight
        
        # Set num_class for XGBoost multiclass
        if model_name == 'xgb' and task == 'multi':
            model_config['num_class'] = num_classes
        
        # Create model instance
        model_class = self.MODELS[model_name]
        model = model_class(**model_config)
        
        logger.debug(f"Created {model_name.upper()} for {task} task")
        return model
    
    def train(self,
              model: BaseEstimator,
              X_train: np.ndarray,
              y_train: np.ndarray,
              sample_weight: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train a model and return training metadata.
        
        Args:
            model: Unfitted sklearn/xgboost model
            X_train: Training features
            y_train: Training labels
            sample_weight: Optional sample weights (for XGBoost multiclass S1)
            
        Returns:
            Dictionary with training metadata (time, etc.)
        """
        logger.info(f"Training {type(model).__name__} on {len(y_train):,} samples...")
        
        start_time = time.time()
        
        if sample_weight is not None:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        logger.info(f"Training completed in {training_time:.2f}s")
        
        return {
            'training_time_seconds': training_time,
            'n_samples': len(y_train),
            'n_features': X_train.shape[1]
        }
    
    def predict(self, model: BaseEstimator, X: np.ndarray) -> np.ndarray:
        """
        Get class predictions from trained model.
        
        Args:
            model: Trained model
            X: Features to predict
            
        Returns:
            Array of predicted class labels
        """
        return model.predict(X)
    
    def predict_proba(self, model: BaseEstimator, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions from trained model.
        
        Args:
            model: Trained model  
            X: Features to predict
            
        Returns:
            Array of class probabilities (n_samples, n_classes)
        """
        return model.predict_proba(X)
    
    def get_model_name(self, model_name: str) -> str:
        """Get human-readable model name."""
        names = {
            'lr': 'Logistic Regression',
            'rf': 'Random Forest',
            'xgb': 'XGBoost'
        }
        return names.get(model_name, model_name)
