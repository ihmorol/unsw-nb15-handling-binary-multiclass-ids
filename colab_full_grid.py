#!/usr/bin/env python3
"""
🚀 Colab Full Grid Launcher
===========================

This script is designed to run in Google Colab and executes the complete
90-experiment grid (0-89):
- 2 Tasks (binary, multi)
- 3 Models (LR, RF, XGBoost)
- 3 Strategies (S0, S1, S2a)
- 5 Seeds (42-46)

Features:
- Mounts Google Drive for persistent result storage
- Clones/updates code from GitHub
- Optimizes execution for Colab (n_jobs=1 for pipeline, full power per model)
- Periodic sync to Drive (every 60 seconds)
- Generates unique timestamped run folder

Usage (in Colab cell):
    !python colab_full_grid.py

Or copy-paste this entire script into a Colab cell.
"""

import os
import sys
import subprocess
import time
import shutil
import yaml
from pathlib import Path
from datetime import datetime

# ==============================================================================
# CONFIGURATION - MODIFY THESE AS NEEDED
# ==============================================================================
REPO_URL = "https://github.com/StartDust/ML_PAPER_REVIEW.git"
BRANCH = "main"
PROJECT_DIR = "/content/ml_project"
DRIVE_BASE_DIR = "/content/drive/MyDrive/UNSW_Archive"

# ==============================================================================
# STEP 1: Check if running in Colab and mount Drive
# ==============================================================================
def is_colab():
    """Check if running in Google Colab environment."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def mount_drive():
    """Mount Google Drive for persistent result storage."""
    if is_colab():
        print("📁 Mounting Google Drive for Results...")
        from google.colab import drive
        drive.mount('/content/drive')
        return True
    else:
        print("⚠️  Not running in Colab, skipping Drive mount.")
        return False


# ==============================================================================
# STEP 2: Setup Unique Results Directory
# ==============================================================================
def setup_results_dir():
    """Create a unique timestamped directory in Drive for this run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{DRIVE_BASE_DIR}/run_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"✅ Results will be saved to: {results_dir}")
    return results_dir


# ==============================================================================
# STEP 3: Clone or Update Repository
# ==============================================================================
def setup_repository():
    """Clone repository from GitHub or pull latest changes."""
    print("\n📥 Setting up code from GitHub...")
    
    if os.path.exists(PROJECT_DIR):
        print(f"   Repository exists at {PROJECT_DIR}, pulling latest...")
        result = subprocess.run(
            ["git", "pull"],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"   ⚠️  Git pull warning: {result.stderr}")
        else:
            print(f"   ✅ Repository updated.")
    else:
        print(f"   Cloning from {REPO_URL}...")
        result = subprocess.run(
            ["git", "clone", "-b", BRANCH, REPO_URL, PROJECT_DIR],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"   ❌ Git clone failed: {result.stderr}")
            sys.exit(1)
        print(f"   ✅ Repository cloned to {PROJECT_DIR}")
    
    os.chdir(PROJECT_DIR)
    print(f"   📂 Working directory: {os.getcwd()}")


# ==============================================================================
# STEP 4: Install Dependencies
# ==============================================================================
def install_dependencies():
    """Install Python dependencies from requirements.txt."""
    print("\n📦 Installing Dependencies...")
    requirements_path = Path(PROJECT_DIR) / "requirements.txt"
    
    if requirements_path.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_path)],
            check=True
        )
        print("   ✅ Dependencies installed.")
    else:
        print("   ⚠️  No requirements.txt found, skipping dependency installation.")


# ==============================================================================
# STEP 5: Optimize Configuration for Colab
# ==============================================================================
def optimize_config():
    """
    Create an optimized config for Colab execution.
    
    Key optimization: Set n_jobs=1 for the experiment pipeline to prevent
    "fork bomb" thrashing on Colab (where 20+ parallel jobs fight for 2 cores).
    This allows each model to use 100% of available CPU/GPU.
    """
    print("\n⚙️  Optimizing Configuration for Colab...")
    
    config_path = Path(PROJECT_DIR) / "configs" / "main.yaml"
    temp_config_path = Path(PROJECT_DIR) / "configs" / "colab_optimized.yaml"
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"   ❌ Error reading config: {e}")
        sys.exit(1)
    
    original_jobs = config.get('experiments', {}).get('n_jobs', 'Unknown')
    print(f"   ℹ️  Original n_jobs: {original_jobs}")
    
    # Force sequential pipeline execution for Colab
    config['experiments']['n_jobs'] = 1
    
    # Verify experiment grid (should be exactly 90 experiments)
    n_seeds = config['experiments'].get('n_seeds', 5)
    tasks = config['experiments'].get('tasks', ['binary', 'multi'])
    models = config['experiments'].get('models', ['lr', 'rf', 'xgb'])
    strategies = config['experiments'].get('strategies', ['s0', 's1', 's2a'])
    
    total_experiments = len(tasks) * len(models) * len(strategies) * n_seeds
    
    print(f"   📊 Experiment Grid:")
    print(f"      - Tasks:      {tasks}")
    print(f"      - Models:     {models}")
    print(f"      - Strategies: {strategies}")
    print(f"      - Seeds:      {n_seeds} (42-{41 + n_seeds})")
    print(f"      - Total:      {total_experiments} experiments (0-{total_experiments - 1})")
    
    # Save optimized config
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"   ✅ Optimized config saved: {temp_config_path}")
    
    return str(temp_config_path), total_experiments


# ==============================================================================
# STEP 6: Clean Previous Results (Force Fresh Run)
# ==============================================================================
def clean_results(force_clean=True):
    """
    Clean previous results to force a fresh run from experiment 0.
    
    Args:
        force_clean: If True, removes metrics directory to re-run all experiments.
    """
    if not force_clean:
        print("\n🔄 Incremental mode: Keeping existing results.")
        return
    
    print("\n🧹 Cleaning previous results for fresh run...")
    
    results_dir = Path(PROJECT_DIR) / "results"
    metrics_dir = results_dir / "metrics"
    
    if metrics_dir.exists():
        try:
            shutil.rmtree(metrics_dir)
            print(f"   ✅ Cleared {metrics_dir}")
        except Exception as e:
            print(f"   ⚠️  Could not clear {metrics_dir}: {e}")
    else:
        print(f"   ℹ️  No previous metrics found at {metrics_dir}")


# ==============================================================================
# STEP 7: Sync Results to Drive
# ==============================================================================
def sync_to_drive(local_results_dir: str, drive_results_dir: str, full_sync: bool = False):
    """
    Sync results from local directory to Google Drive.
    
    Args:
        local_results_dir: Path to local results directory
        drive_results_dir: Path to Drive results directory
        full_sync: If True, sync all files. If False, sync only key artifacts.
    """
    if not is_colab():
        return
    
    try:
        if full_sync:
            # Full sync - copy everything
            cmd = [
                "rsync", "-av",
                f"{local_results_dir}/",
                f"{drive_results_dir}/"
            ]
        else:
            # Incremental sync - only key artifacts (JSON, PNG, CSV)
            cmd = [
                "rsync", "-avq",
                "--include=*/",
                "--include=*.json",
                "--include=*.png",
                "--include=*.csv",
                "--exclude=*",
                f"{local_results_dir}/",
                f"{drive_results_dir}/"
            ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"💾 Synced to Drive: {drive_results_dir}")
        else:
            print(f"⚠️  Sync warning: {result.stderr[:100] if result.stderr else 'Unknown'}")
    except Exception as e:
        print(f"⚠️  Sync error: {e}")


# ==============================================================================
# STEP 8: Run Full Experiment Grid
# ==============================================================================
def run_experiment_grid(config_path: str, drive_results_dir: str, sync_interval: int = 60):
    """
    Execute the full 90-experiment grid with periodic Drive sync.
    
    Args:
        config_path: Path to the optimized config file
        drive_results_dir: Path to Drive results directory for syncing
        sync_interval: Seconds between sync operations (default: 60)
    """
    print("\n" + "=" * 70)
    print("🚀 STARTING FULL EXPERIMENT GRID (0-89)")
    print("=" * 70)
    print(f"   Config:        {config_path}")
    print(f"   Sync Interval: {sync_interval}s")
    print(f"   Drive Target:  {drive_results_dir}")
    print("=" * 70 + "\n")
    
    local_results_dir = str(Path(PROJECT_DIR) / "results")
    
    # Start main.py as subprocess
    cmd = [sys.executable, "main.py", "--config", config_path]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    last_sync_time = time.time()
    
    try:
        # Stream output and sync periodically
        while True:
            # Check if process has finished
            return_code = proc.poll()
            
            # Read available output (non-blocking would be ideal but this works)
            if proc.stdout:
                line = proc.stdout.readline()
                if line:
                    print(line, end='')
            
            # Periodic sync to Drive
            current_time = time.time()
            if current_time - last_sync_time >= sync_interval:
                sync_to_drive(local_results_dir, drive_results_dir, full_sync=False)
                last_sync_time = current_time
            
            # Exit loop if process finished
            if return_code is not None:
                # Drain remaining output
                remaining_output = proc.stdout.read() if proc.stdout else ""
                if remaining_output:
                    print(remaining_output)
                break
            
            time.sleep(0.1)  # Small sleep to prevent CPU spinning
        
        if return_code != 0:
            print(f"\n❌ Pipeline failed with exit code {return_code}")
        else:
            print("\n✅ Pipeline completed successfully!")
        
        return return_code
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user. Terminating...")
        proc.terminate()
        proc.wait()
        return 130
    
    finally:
        # Final full sync to Drive
        print("\n📤 Performing final sync to Drive...")
        sync_to_drive(local_results_dir, drive_results_dir, full_sync=True)


# ==============================================================================
# STEP 9: Cleanup
# ==============================================================================
def cleanup(config_path: str):
    """Remove temporary config file."""
    try:
        if os.path.exists(config_path):
            os.remove(config_path)
            print(f"🗑️  Cleaned up temp config: {config_path}")
    except Exception as e:
        print(f"⚠️  Could not remove temp config: {e}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    """
    Main entry point for Colab full grid execution.
    
    Runs the complete 90-experiment grid:
    - Experiments 0-89
    - 2 tasks × 3 models × 3 strategies × 5 seeds
    """
    print("=" * 70)
    print("🧪 UNSW-NB15 FULL EXPERIMENT GRID - COLAB LAUNCHER")
    print("=" * 70)
    print(f"Started at: {datetime.now().isoformat()}")
    print()
    
    # Step 1: Mount Drive (if in Colab)
    drive_available = mount_drive()
    
    # Step 2: Setup unique results directory in Drive
    if drive_available:
        drive_results_dir = setup_results_dir()
    else:
        drive_results_dir = None
    
    # Step 3: Clone/update repository
    setup_repository()
    
    # Step 4: Install dependencies
    install_dependencies()
    
    # Step 5: Optimize config for Colab
    config_path, total_experiments = optimize_config()
    
    # Step 6: Clean previous results (force fresh run)
    clean_results(force_clean=True)
    
    # Step 7: Run experiment grid
    try:
        exit_code = run_experiment_grid(
            config_path=config_path,
            drive_results_dir=drive_results_dir if drive_available else "/tmp/results",
            sync_interval=60
        )
    finally:
        # Step 8: Cleanup
        cleanup(config_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 EXECUTION SUMMARY")
    print("=" * 70)
    print(f"   Total Experiments: {total_experiments}")
    print(f"   Exit Code:         {exit_code}")
    if drive_available:
        print(f"   Results Saved To:  {drive_results_dir}")
    print(f"   Completed at:      {datetime.now().isoformat()}")
    print("=" * 70)
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
