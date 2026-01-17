#!/usr/bin/env python3
"""
Main Orchestrator for UNSW-NB15 Imbalance Study
================================================

This script runs the complete 18-experiment grid:
- 2 Tasks: Binary, Multiclass
- 3 Models: Logistic Regression, Random Forest, XGBoost
- 3 Strategies: S0 (None), S1 (Class Weight), S2a (RandomOverSampler)

Usage:
    python main.py

Output:
    - results/metrics/{experiment_id}.json - Per-experiment metrics
    - results/figures/cm_{experiment_id}.png - Confusion matrices
    - results/experiment_log.csv - Master experiment tracker
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

# Local imports
from src.data import DataLoader, UNSWPreprocessor
from src.models import ModelTrainer
from src.strategies import get_strategy
from src.evaluation import (
    compute_all_metrics,
    compute_rare_class_analysis,
    format_metrics_for_logging,
    plot_confusion_matrix
)
from src.utils import load_config, setup_logging

# Configure logging
logger = logging.getLogger(__name__)


def run_single_experiment(
    experiment_id: str,
    task: str,
    model_name: str,
    strategy_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
    results_dir: Path,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Execute a single experiment and save results.
    
    Args:
        experiment_id: Unique experiment identifier
        task: 'binary' or 'multi'
        model_name: 'lr', 'rf', or 'xgb'
        strategy_name: 's0', 's1', or 's2a'
        X_train, y_train: Training data
        X_test, y_test: Test data
        config: Configuration dictionary
        results_dir: Output directory path
        class_names: List of class names for reporting
        
    Returns:
        Experiment results dictionary
    """
    start_time = time.time()
    random_state = config.get('random_state', 42)
    
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT: {experiment_id}")
    logger.info(f"  Task:     {task}")
    logger.info(f"  Model:    {model_name.upper()}")
    logger.info(f"  Strategy: {strategy_name.upper()}")
    logger.info("=" * 60)
    
    # Step 1: Apply imbalance strategy
    strategy = get_strategy(strategy_name, random_state=random_state)
    X_train_balanced, y_train_balanced = strategy.apply(X_train, y_train)
    
    # Step 2: Configure model with appropriate weights
    trainer = ModelTrainer(config)
    
    # Get class weight and sample weight based on strategy
    class_weight = strategy.get_class_weight()
    sample_weight = None
    scale_pos_weight = None
    
    if strategy_name == 's1':
        if model_name == 'xgb' and task == 'binary':
            scale_pos_weight = strategy.get_scale_pos_weight(y_train)
            class_weight = None  # Use scale_pos_weight instead
        elif model_name == 'xgb' and task == 'multi':
            # XGBoost multiclass uses sample_weight
            sample_weight = strategy.get_sample_weight(y_train_balanced)
            class_weight = None
    
    # Create model instance
    num_classes = len(np.unique(y_train))
    model = trainer.get_model(
        model_name=model_name,
        task=task,
        class_weight=class_weight,
        scale_pos_weight=scale_pos_weight,
        num_classes=num_classes
    )
    
    # Step 3: Train model
    training_metadata = trainer.train(
        model=model,
        X_train=X_train_balanced,
        y_train=y_train_balanced,
        sample_weight=sample_weight
    )
    
    # Step 4: Predict on TEST set
    y_pred = trainer.predict(model, X_test)
    y_pred_proba = trainer.predict_proba(model, X_test)
    
    # Step 5: Compute metrics
    metrics = compute_all_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        task=task,
        class_names=class_names if task == 'multi' else ['Normal', 'Attack']
    )
    
    # Step 6: Rare class analysis for multiclass
    rare_class_analysis = None
    if task == 'multi':
        rare_classes = config.get('rare_classes', [])
        rare_class_analysis = compute_rare_class_analysis(
            metrics=metrics,
            rare_classes=rare_classes,
            label_mapping=None  # Use class names directly
        )
    
    # Log metrics
    logger.info("\nResults:")
    logger.info(format_metrics_for_logging(metrics, task))
    
    total_time = time.time() - start_time
    
    # Compile results
    result = {
        'experiment_id': experiment_id,
        'task': task,
        'model': model_name,
        'strategy': strategy_name,
        'timestamp': datetime.now().isoformat(),
        'training_time_seconds': training_metadata['training_time_seconds'],
        'total_time_seconds': total_time,
        'train_samples': len(y_train_balanced),
        'test_samples': len(y_test),
        'metrics': metrics,
        'rare_class_analysis': rare_class_analysis
    }
    
    # Save metrics JSON
    metrics_path = results_dir / 'metrics' / f"{experiment_id}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Save confusion matrix plot
    cm = np.array(metrics['confusion_matrix'])
    cm_labels = class_names if task == 'multi' else ['Normal', 'Attack']
    cm_path = results_dir / 'figures' / f"cm_{experiment_id}.png"
    plot_confusion_matrix(
        cm=cm,
        labels=cm_labels,
        save_path=str(cm_path),
        title=f"Confusion Matrix: {experiment_id.replace('_', ' ').title()}"
    )
    
    logger.info(f"Completed {experiment_id} in {total_time:.2f}s")
    logger.info("-" * 60)
    
    return result


def main():
    """
    Main entry point for the experiment pipeline.
    
    Runs all 18 experiments and generates summary reports.
    """
    # Load configuration
    config = load_config("configs/main.yaml")
    
    # Setup logging
    log_path = Path(config['results_dir']) / 'logs' / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(level="INFO", log_file=str(log_path))
    
    logger.info("=" * 70)
    logger.info("UNSW-NB15 CLASS IMBALANCE STUDY")
    logger.info("=" * 70)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    
    results_dir = Path(config['results_dir'])
    random_state = config.get('random_state', 42)
    
    # Create output directories
    for subdir in ['metrics', 'figures', 'models', 'tables', 'logs', 'processed']:
        (results_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load and preprocess data
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 1: DATA PREPROCESSING")
    logger.info("=" * 50)
    
    loader = DataLoader(config)
    train_df, test_df = loader.load_all()
    
    preprocessor = UNSWPreprocessor(config)
    preprocessor.fit_transform(train_df, test_df)
    
    # Save preprocessing metadata
    preprocessor.save_metadata(str(results_dir / 'processed' / 'preprocessing_metadata.json'))
    
    # Get class names for multiclass
    multiclass_labels = config['data'].get('multiclass_labels', 
                                           list(preprocessor.label_mapping.keys()))
    
    # Step 2: Run experiment grid
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 2: RUNNING EXPERIMENTS")
    logger.info("=" * 50)
    
    tasks = config['experiments']['tasks']
    models = config['experiments']['models']
    strategies = config['experiments']['strategies']
    
    total_experiments = len(tasks) * len(models) * len(strategies)
    logger.info(f"Total experiments to run: {total_experiments}")
    
    all_results = []
    experiment_count = 0
    
    for task in tasks:
        # Get data for this task
        X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.get_splits(task)
        
        # For multiclass, get class names
        if task == 'multi':
            class_names = multiclass_labels
        else:
            class_names = ['Normal', 'Attack']
        
        for model_name in models:
            for strategy_name in strategies:
                experiment_count += 1
                experiment_id = f"{task}_{model_name}_{strategy_name}"
                
                # Check if experiment already completed (resume capability)
                metrics_file = results_dir / 'metrics' / f"{experiment_id}.json"
                if metrics_file.exists():
                    logger.info(f"\n[{experiment_count}/{total_experiments}] SKIPPING {experiment_id} (already completed)")
                    with open(metrics_file, 'r') as f:
                        existing_result = json.load(f)
                    all_results.append(existing_result)
                    continue
                
                logger.info(f"\n[{experiment_count}/{total_experiments}] Running {experiment_id}")
                
                try:
                    result = run_single_experiment(
                        experiment_id=experiment_id,
                        task=task,
                        model_name=model_name,
                        strategy_name=strategy_name,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        config=config,
                        results_dir=results_dir,
                        class_names=class_names
                    )
                    all_results.append(result)
                    
                except Exception as e:
                    logger.error(f"FAILED: {experiment_id}")
                    logger.error(f"Error: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
                    # Record failure
                    all_results.append({
                        'experiment_id': experiment_id,
                        'task': task,
                        'model': model_name,
                        'strategy': strategy_name,
                        'status': 'failed',
                        'error': str(e)
                    })
    
    # Step 3: Generate summary reports
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 3: GENERATING REPORTS")
    logger.info("=" * 50)
    
    # Create experiment log CSV
    log_data = []
    for r in all_results:
        if 'metrics' in r:
            log_data.append({
                'experiment_id': r['experiment_id'],
                'task': r['task'],
                'model': r['model'],
                'strategy': r['strategy'],
                'accuracy': r['metrics']['overall']['accuracy'],
                'macro_f1': r['metrics']['overall']['macro_f1'],
                'weighted_f1': r['metrics']['overall']['weighted_f1'],
                'g_mean': r['metrics']['overall']['g_mean'],
                'roc_auc': r['metrics']['overall']['roc_auc'],
                'training_time': r['training_time_seconds'],
                'total_time': r['total_time_seconds'],
                'timestamp': r['timestamp']
            })
    
    log_df = pd.DataFrame(log_data)
    log_path = results_dir / 'experiment_log.csv'
    log_df.to_csv(log_path, index=False)
    logger.info(f"Saved experiment log to {log_path}")
    
    # Generate summary tables
    generate_summary_tables(all_results, results_dir, config)
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT PIPELINE COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Completed at: {datetime.now().isoformat()}")
    logger.info(f"Total experiments: {len(all_results)}")
    successful = len([r for r in all_results if 'metrics' in r])
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {len(all_results) - successful}")
    logger.info(f"\nResults saved to: {results_dir.absolute()}")


def generate_summary_tables(
    results: List[Dict[str, Any]],
    results_dir: Path,
    config: dict
) -> None:
    """Generate summary CSV tables from experiment results."""
    
    tables_dir = results_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Final Summary Table (Overall Metrics)
    summary_rows = []
    for r in results:
        if 'metrics' not in r:
            continue
        summary_rows.append({
            'Task': r['task'],
            'Model': r['model'].upper(),
            'Strategy': r['strategy'].upper(),
            'Accuracy': r['metrics']['overall']['accuracy'],
            'Macro_F1': r['metrics']['overall']['macro_f1'],
            'Weighted_F1': r['metrics']['overall']['weighted_f1'],
            'G_Mean': r['metrics']['overall']['g_mean'],
            'ROC_AUC': r['metrics']['overall']['roc_auc']
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(tables_dir / 'final_summary_tables.csv', index=False)
    logger.info("Generated final_summary_tables.csv")
    
    # 2. Per-Class Metrics Table (Multiclass only)
    per_class_rows = []
    for r in results:
        if r.get('task') != 'multi' or 'metrics' not in r:
            continue
        for cls_name, cls_metrics in r['metrics']['per_class'].items():
            per_class_rows.append({
                'Experiment': r['experiment_id'],
                'Model': r['model'].upper(),
                'Strategy': r['strategy'].upper(),
                'Class': cls_name,
                'Precision': cls_metrics['precision'],
                'Recall': cls_metrics['recall'],
                'F1': cls_metrics['f1'],
                'Support': cls_metrics['support']
            })
    
    if per_class_rows:
        per_class_df = pd.DataFrame(per_class_rows)
        per_class_df.to_csv(tables_dir / 'per_class_metrics.csv', index=False)
        logger.info("Generated per_class_metrics.csv")
    
    # 3. Rare Class Report
    rare_classes = config.get('rare_classes', [])
    rare_rows = []
    for r in results:
        if r.get('task') != 'multi' or 'rare_class_analysis' not in r or r['rare_class_analysis'] is None:
            continue
        for cls_name, cls_metrics in r['rare_class_analysis'].items():
            if cls_name in rare_classes:
                rare_rows.append({
                    'Experiment': r['experiment_id'],
                    'Model': r['model'].upper(),
                    'Strategy': r['strategy'].upper(),
                    'Rare_Class': cls_name,
                    'Precision': cls_metrics['precision'],
                    'Recall': cls_metrics['recall'],
                    'F1': cls_metrics['f1'],
                    'Support': cls_metrics['support']
                })
    
    if rare_rows:
        rare_df = pd.DataFrame(rare_rows)
        rare_df.to_csv(tables_dir / 'rare_class_report.csv', index=False)
        logger.info("Generated rare_class_report.csv")


if __name__ == '__main__':
    main()
