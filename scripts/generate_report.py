#!/usr/bin/env python3
"""
Report Generation Script for UNSW-NB15 Study
=============================================

This script generates final summary reports and visualizations
from completed experiment results.

Usage:
    python scripts/generate_report.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import numpy as np
import pandas as pd

from src.utils import load_config, setup_logging
from src.evaluation import (
    plot_strategy_comparison,
    plot_rare_class_recall
)

# Setup logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def load_all_results(metrics_dir: Path) -> list:
    """Load all experiment result JSONs."""
    results = []
    for json_file in sorted(metrics_dir.glob("*.json")):
        with open(json_file, 'r') as f:
            results.append(json.load(f))
    return results


def generate_comparison_figures(results: list, figures_dir: Path, config: dict):
    """Generate comparison visualizations."""
    
    # Group results for plotting
    metrics_to_plot = ['accuracy', 'macro_f1', 'g_mean', 'roc_auc']
    
    for task in ['binary', 'multi']:
        task_results = [r for r in results if r.get('task') == task and 'metrics' in r]
        
        for metric in metrics_to_plot:
            # Build data structure for plotting
            plot_data = {}
            for model in ['lr', 'rf', 'xgb']:
                plot_data[model] = {}
                for strategy in ['s0', 's1', 's2a']:
                    matching = [r for r in task_results 
                               if r['model'] == model and r['strategy'] == strategy]
                    if matching:
                        plot_data[model][strategy] = matching[0]['metrics']['overall'].get(metric, 0) or 0
            
            if any(plot_data[m] for m in plot_data):
                plot_strategy_comparison(
                    results=plot_data,
                    metric=metric,
                    save_path=str(figures_dir / f"{task}_{metric}_comparison.png"),
                    title=f"{task.title()} Classification: {metric.replace('_', ' ').title()}"
                )
    
    # Rare class recall heatmap for multiclass
    rare_classes = config.get('rare_classes', [])
    if rare_classes:
        multi_results = [r for r in results if r.get('task') == 'multi' and 'rare_class_analysis' in r]
        
        if multi_results:
            rare_data = {}
            for r in multi_results:
                exp_id = r['experiment_id']
                if r['rare_class_analysis']:
                    rare_data[exp_id] = r['rare_class_analysis']
            
            if rare_data:
                plot_rare_class_recall(
                    results=rare_data,
                    rare_classes=rare_classes,
                    save_path=str(figures_dir / 'rare_class_recall_heatmap.png')
                )


def generate_latex_tables(results: list, tables_dir: Path):
    """Generate LaTeX-formatted tables for papers."""
    
    # Binary results table
    binary_results = [r for r in results if r.get('task') == 'binary' and 'metrics' in r]
    if binary_results:
        rows = []
        for r in binary_results:
            rows.append({
                'Model': r['model'].upper(),
                'Strategy': r['strategy'].upper(),
                'Accuracy': f"{r['metrics']['overall']['accuracy']:.4f}",
                'Macro F1': f"{r['metrics']['overall']['macro_f1']:.4f}",
                'G-Mean': f"{r['metrics']['overall']['g_mean']:.4f}",
                'ROC-AUC': f"{r['metrics']['overall']['roc_auc']:.4f}" if r['metrics']['overall']['roc_auc'] else 'N/A'
            })
        
        df = pd.DataFrame(rows)
        latex = df.to_latex(index=False, caption='Binary Classification Results', label='tab:binary')
        with open(tables_dir / 'binary_results.tex', 'w') as f:
            f.write(latex)
        logger.info("Generated binary_results.tex")
    
    # Multiclass results table
    multi_results = [r for r in results if r.get('task') == 'multi' and 'metrics' in r]
    if multi_results:
        rows = []
        for r in multi_results:
            rows.append({
                'Model': r['model'].upper(),
                'Strategy': r['strategy'].upper(),
                'Accuracy': f"{r['metrics']['overall']['accuracy']:.4f}",
                'Macro F1': f"{r['metrics']['overall']['macro_f1']:.4f}",
                'G-Mean': f"{r['metrics']['overall']['g_mean']:.4f}",
                'ROC-AUC': f"{r['metrics']['overall']['roc_auc']:.4f}" if r['metrics']['overall']['roc_auc'] else 'N/A'
            })
        
        df = pd.DataFrame(rows)
        latex = df.to_latex(index=False, caption='Multiclass Classification Results', label='tab:multi')
        with open(tables_dir / 'multiclass_results.tex', 'w') as f:
            f.write(latex)
        logger.info("Generated multiclass_results.tex")


def main():
    """Generate all reports and visualizations."""
    logger.info("=" * 60)
    logger.info("GENERATING FINAL REPORTS")
    logger.info("=" * 60)
    
    config = load_config("configs/main.yaml")
    results_dir = Path(config['results_dir'])
    metrics_dir = results_dir / 'metrics'
    figures_dir = results_dir / 'figures'
    tables_dir = results_dir / 'tables'
    
    # Check for results
    if not metrics_dir.exists() or not list(metrics_dir.glob("*.json")):
        logger.error("No experiment results found! Run main.py first.")
        return
    
    # Load all results
    results = load_all_results(metrics_dir)
    logger.info(f"Loaded {len(results)} experiment results")
    
    # Generate comparison figures
    logger.info("\n--- Generating Comparison Figures ---")
    generate_comparison_figures(results, figures_dir, config)
    
    # Generate LaTeX tables
    logger.info("\n--- Generating LaTeX Tables ---")
    generate_latex_tables(results, tables_dir)
    
    # Print best results
    logger.info("\n" + "=" * 60)
    logger.info("BEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    successful = [r for r in results if 'metrics' in r]
    
    for task in ['binary', 'multi']:
        task_results = [r for r in successful if r['task'] == task]
        if task_results:
            # Best by G-Mean (primary metric)
            best = max(task_results, key=lambda x: x['metrics']['overall']['g_mean'])
            logger.info(f"\n{task.upper()} - Best by G-Mean:")
            logger.info(f"  Experiment: {best['experiment_id']}")
            logger.info(f"  G-Mean:     {best['metrics']['overall']['g_mean']:.4f}")
            logger.info(f"  Accuracy:   {best['metrics']['overall']['accuracy']:.4f}")
            logger.info(f"  Macro F1:   {best['metrics']['overall']['macro_f1']:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("REPORT GENERATION COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
