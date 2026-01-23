
"""
Training Visualization Module.

This module provides functions to plot learning curves and other training-related
visualizations to analyze model convergence and overfitting.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

def plot_learning_curves(results_dir: Path, experiment_id: str):
    """
    Plot learning curves for a specific experiment (XGBoost).
    
    Args:
        results_dir: Base results directory
        experiment_id: Experiment identifier
    """
    lc_path = results_dir / 'learning_curves' / f"{experiment_id}.json"
    
    if not lc_path.exists():
        logger.warning(f"No learning curve data found for {experiment_id}")
        return

    try:
        with open(lc_path, 'r') as f:
            history = json.load(f)
            
        # Structure of history: {'validation_0': {'logloss': [0.1, 0.05, ...]}}
        # Note: XGBoost output keys depend on metric used.
        
        metrics_data = []
        for dataset_name, metrics in history.items():
            for metric_name, values in metrics.items():
                for epoch, value in enumerate(values):
                    metrics_data.append({
                        'Epoch': epoch,
                        'Value': value,
                        'Metric': metric_name,
                        'Dataset': dataset_name
                    })
        
        df = pd.DataFrame(metrics_data)
        
        if df.empty:
            logger.warning("Empty learning curve data")
            return

        # Plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Epoch', y='Value', hue='Dataset', style='Metric')
        plt.title(f"Learning Curve: {experiment_id}")
        plt.grid(True, alpha=0.3)
        
        output_path = results_dir / 'figures' / f"lc_{experiment_id}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved learning curve to {output_path}")

    except Exception as e:
        logger.error(f"Failed to plot learning curve for {experiment_id}: {e}")

def generate_all_learning_curves(results_dir: Path):
    """Generate plots for all available learning curves."""
    lc_dir = results_dir / 'learning_curves'
    if not lc_dir.exists():
        logger.warning("No learning_curves directory found.")
        return
        
    for lc_file in lc_dir.glob("*.json"):
        experiment_id = lc_file.stem
        plot_learning_curves(results_dir, experiment_id)

if __name__ == "__main__":
    # Test execution
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1:
        generate_all_learning_curves(Path(sys.argv[1]))
    else:
        generate_all_learning_curves(Path("results"))
