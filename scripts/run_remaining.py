#!/usr/bin/env python3
"""
Custom Runner for Remaining Multiclass Experiments
==================================================
Runs only:
- Task: multi
- Strategies: s0, s1, s2a (NO S2b/SMOTE)
- Seeds: 43, 44, 45, 46
"""

import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from joblib import Parallel, delayed

from src.data import DataLoader, UNSWPreprocessor
from src.utils import load_config, setup_logging
from main import run_single_experiment, generate_summary_tables

logger = logging.getLogger(__name__)

def main():
    # Force load main config
    config = load_config("configs/main.yaml")
    
    # Override settings for this run
    n_seeds = 5  # We need seeds 43-46, but logic below handles ranges
    target_seeds = [43, 44, 45, 46]
    target_task = 'multi'
    target_strategies = ['s0', 's1', 's2a']
    
    # Configure results dir
    results_dir = Path(config['results_dir'])
    log_path = results_dir / 'logs' / f"run_remaining_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(level="INFO", log_file=str(log_path))
    
    logger.info("=" * 60)
    logger.info("RUNNING REMAINING MULTICLASS EXPERIMENTS")
    logger.info(f"Target Seeds: {target_seeds}")
    logger.info(f"Target Strategies: {target_strategies}")
    logger.info("=" * 60)

    # 1. Preprocessing (Must match main.py exactly)
    logger.info("Loading Data...")
    loader = DataLoader(config)
    train_df, test_df = loader.load_all()
    
    preprocessor = UNSWPreprocessor(config)
    preprocessor.fit_transform(train_df, test_df)
    
    # 2. Build Queue
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.get_splits(target_task)
    class_names = list(preprocessor.label_encoder.classes_)
    
    experiment_queue = []
    skipped_count = 0
    
    for model_name in config['experiments']['models']: # ['lr', 'rf', 'xgb']
        for strategy_name in target_strategies:
            for seed in target_seeds:
                experiment_id = f"{target_task}_{model_name}_{strategy_name}_s{seed}"
                
                # Check exist
                metrics_file = results_dir / 'metrics' / f"{experiment_id}.json"
                if metrics_file.exists():
                    logger.info(f"Skipping {experiment_id} (Done)")
                    skipped_count += 1
                    continue
                
                experiment_queue.append({
                    'experiment_id': experiment_id,
                    'task': target_task,
                    'model_name': model_name,
                    'strategy_name': strategy_name,
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': y_test,
                    'X_val': X_val,
                    'y_val': y_val,
                    'config': config,
                    'results_dir': results_dir,
                    'class_names': class_names,
                    'seed': seed
                })

    logger.info(f"Queued {len(experiment_queue)} experiments (Skipped {skipped_count})")
    
    if not experiment_queue:
        logger.info("Nothing to run!")
        return

    # 3. Execute
    n_jobs = config['experiments'].get('n_jobs', -1)
    logger.info(f"Starting execution with n_jobs={n_jobs}")
    
    Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_experiment)(**args) for args in experiment_queue
    )
    
    logger.info("Batch Complete.")

if __name__ == "__main__":
    main()
