
import os
import sys
import json
import logging
from pathlib import Path

# --- HARDCODED COMPLETED EXPERIMENTS (65/90) ---
# Paste this from the analysis step to avoid needing to upload files
# --- HARDCODED COMPLETED EXPERIMENTS (65/90) ---
COMPLETED_IDS = {
    'binary_lr_s0_s42', 'binary_lr_s0_s43', 'binary_lr_s0_s44', 'binary_lr_s0_s45', 'binary_lr_s0_s46',
    'binary_lr_s1_s42', 'binary_lr_s1_s43', 'binary_lr_s1_s44', 'binary_lr_s1_s45', 'binary_lr_s1_s46',
    'binary_lr_s2a_s42', 'binary_lr_s2a_s43', 'binary_lr_s2a_s44', 'binary_lr_s2a_s45', 'binary_lr_s2a_s46',
    'binary_rf_s0_s42', 'binary_rf_s0_s43', 'binary_rf_s0_s44', 'binary_rf_s0_s45', 'binary_rf_s0_s46',
    'binary_rf_s1_s42', 'binary_rf_s1_s43', 'binary_rf_s1_s44', 'binary_rf_s1_s45', 'binary_rf_s1_s46',
    'binary_rf_s2a_s42', 'binary_rf_s2a_s43', 'binary_rf_s2a_s44', 'binary_rf_s2a_s45', 'binary_rf_s2a_s46',
    'binary_xgb_s0_s42', 'binary_xgb_s0_s43', 'binary_xgb_s0_s44', 'binary_xgb_s0_s45', 'binary_xgb_s0_s46',
    'binary_xgb_s1_s42', 'binary_xgb_s1_s43', 'binary_xgb_s1_s44', 'binary_xgb_s1_s45', 'binary_xgb_s1_s46',
    'binary_xgb_s2a_s42', 'binary_xgb_s2a_s43', 'binary_xgb_s2a_s44', 'binary_xgb_s2a_s45', 'binary_xgb_s2a_s46',
    'multi_lr_s0_s42', 'multi_lr_s0_s43', 'multi_lr_s0_s44', 'multi_lr_s0_s45', 'multi_lr_s0_s46',
    'multi_lr_s1_s42', 'multi_lr_s1_s43', 'multi_lr_s1_s44', 'multi_lr_s1_s45', 'multi_lr_s1_s46',
    'multi_lr_s2a_s42',
    'multi_rf_s0_s42', 'multi_rf_s0_s43', 'multi_rf_s0_s44', 'multi_rf_s0_s45',
    'multi_rf_s1_s42',
    'multi_rf_s2a_s42',
    'multi_xgb_s0_s42',
    'multi_xgb_s1_s42',
    'multi_xgb_s2a_s42'
}

def run_rescue():
    import main
    from src.utils import load_config
    
    # Create fake marker files for completed experiments so main.py skips them
    results_dir = Path("results")
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- MARKING {len(COMPLETED_IDS)} EXPERIMENTS AS DONE ---")
    for exp_id in COMPLETED_IDS:
        dummy_file = metrics_dir / f"{exp_id}.json"
        if not dummy_file.exists():
            with open(dummy_file, 'w') as f:
                json.dump({"status": "skipped_by_rescue_script", "experiment_id": exp_id}, f)
    
    print("--- STARTING RESCUE RUN FOR REMAINING EXPERIMENTS ---")
    # Now run main.py as normal. It will see the dummy files and skip them.
    main.main()

if __name__ == "__main__":
    run_rescue()
