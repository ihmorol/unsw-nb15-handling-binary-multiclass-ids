# @title 🚀 Full Experiment Grid Launcher (GitHub + Drive) - Experiments 0-89
# @markdown **Run the complete 90-experiment grid with automatic Drive sync**
# @markdown
# @markdown This cell will:
# @markdown 1. Mount Google Drive for persistent storage
# @markdown 2. Clone/update code from GitHub
# @markdown 3. Run all 90 experiments (2 tasks × 3 models × 3 strategies × 5 seeds)
# @markdown 4. Sync results to Drive every 60 seconds
# @markdown
# @markdown ---

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
REPO_URL = "https://github.com/StartDust/ML_PAPER_REVIEW.git"  # @param {type:"string"}
BRANCH = "main"  # @param {type:"string"}
FORCE_FRESH_RUN = True  # @param {type:"boolean"}
SYNC_INTERVAL_SECONDS = 60  # @param {type:"integer"}

PROJECT_DIR = "/content/ml_project"
DRIVE_BASE_DIR = "/content/drive/MyDrive/UNSW_Archive"

# ==============================================================================
# STEP 1: Mount Google Drive
# ==============================================================================
print("📁 Mounting Google Drive for Results...")
from google.colab import drive
drive.mount('/content/drive')

# ==============================================================================
# STEP 2: Setup Unique Results Directory
# ==============================================================================
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = f"{DRIVE_BASE_DIR}/run_{TIMESTAMP}"
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"✅ Results will be saved to unique run folder: {RESULTS_DIR}")

# ==============================================================================
# STEP 3: Clone or Update Repository
# ==============================================================================
print("\n📥 Setting up code from GitHub...")

if os.path.exists(PROJECT_DIR):
    print(f"   Repository exists at {PROJECT_DIR}, pulling latest...")
    os.chdir(PROJECT_DIR)
    !git pull
else:
    print(f"   Cloning from {REPO_URL}...")
    !git clone -b {BRANCH} {REPO_URL} {PROJECT_DIR}
    os.chdir(PROJECT_DIR)

print(f"   📂 Working directory: {os.getcwd()}")

# ==============================================================================
# STEP 4: Install Dependencies
# ==============================================================================
print("\n📦 Installing Dependencies...")
!pip install -q -r requirements.txt

# ==============================================================================
# STEP 5: Optimize Configuration for Colab
# ==============================================================================
print("\n⚙️  Optimizing Configuration for Colab...")

config_path = Path(PROJECT_DIR) / "configs" / "main.yaml"
temp_config_path = Path(PROJECT_DIR) / "configs" / "colab_optimized.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

original_jobs = config.get('experiments', {}).get('n_jobs', 'Unknown')
print(f"   ℹ️  Original n_jobs: {original_jobs}")

# Force sequential pipeline execution for Colab
# This prevents "fork bomb" thrashing - each model gets 100% CPU
config['experiments']['n_jobs'] = 1

# Calculate total experiments
n_seeds = config['experiments'].get('n_seeds', 5)
tasks = config['experiments'].get('tasks', ['binary', 'multi'])
models = config['experiments'].get('models', ['lr', 'rf', 'xgb'])
strategies = config['experiments'].get('strategies', ['s0', 's1', 's2a'])

TOTAL_EXPERIMENTS = len(tasks) * len(models) * len(strategies) * n_seeds

print(f"   📊 Experiment Grid:")
print(f"      - Tasks:      {tasks}")
print(f"      - Models:     {models}")
print(f"      - Strategies: {strategies}")
print(f"      - Seeds:      {n_seeds} (42-{41 + n_seeds})")
print(f"      - Total:      {TOTAL_EXPERIMENTS} experiments (0-{TOTAL_EXPERIMENTS - 1})")

with open(temp_config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"   ✅ Optimized config saved: {temp_config_path}")

# ==============================================================================
# STEP 6: Clean Previous Results (Force Fresh Run)
# ==============================================================================
if FORCE_FRESH_RUN:
    print("\n🧹 Cleaning previous results for fresh run (0-89)...")
    metrics_dir = Path(PROJECT_DIR) / "results" / "metrics"
    
    if metrics_dir.exists():
        shutil.rmtree(metrics_dir)
        print(f"   ✅ Cleared {metrics_dir}")
    else:
        print(f"   ℹ️  No previous metrics found.")
else:
    print("\n🔄 Incremental mode: Keeping existing results.")

# ==============================================================================
# STEP 7: Run Full Experiment Grid with Sync
# ==============================================================================
print("\n" + "=" * 70)
print("🚀 STARTING FULL EXPERIMENT GRID (0-89)")
print("=" * 70)
print(f"   Config:        {temp_config_path}")
print(f"   Sync Interval: {SYNC_INTERVAL_SECONDS}s")
print(f"   Drive Target:  {RESULTS_DIR}")
print("=" * 70 + "\n")

LOCAL_RESULTS = str(Path(PROJECT_DIR) / "results")

# Run main.py in subprocess with output streaming
proc = subprocess.Popen(
    [sys.executable, "main.py", "--config", str(temp_config_path)],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

last_sync_time = time.time()
exit_code = 0

try:
    while True:
        # Check if process finished
        return_code = proc.poll()
        
        # Read and print output
        if proc.stdout:
            line = proc.stdout.readline()
            if line:
                print(line, end='')
        
        # Periodic sync to Drive
        current_time = time.time()
        if current_time - last_sync_time >= SYNC_INTERVAL_SECONDS:
            !rsync -avq --include='*/' --include='*.json' --include='*.png' --include='*.csv' --exclude='*' {LOCAL_RESULTS}/ "{RESULTS_DIR}/"
            print(f"💾 Synced artifacts to Drive...")
            last_sync_time = current_time
        
        # Exit if done
        if return_code is not None:
            remaining = proc.stdout.read() if proc.stdout else ""
            if remaining:
                print(remaining)
            exit_code = return_code
            break
        
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user. Saving progress...")
    proc.terminate()
    proc.wait()
    exit_code = 130

# ==============================================================================
# STEP 8: Final Sync and Cleanup
# ==============================================================================
print("\n📤 Performing final full sync to Drive...")
!rsync -av {LOCAL_RESULTS}/ "{RESULTS_DIR}/"

# Cleanup temp config
if os.path.exists(temp_config_path):
    os.remove(temp_config_path)
    print(f"🗑️  Cleaned up temp config.")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("📊 EXECUTION SUMMARY")
print("=" * 70)
print(f"   Total Experiments: {TOTAL_EXPERIMENTS} (0-{TOTAL_EXPERIMENTS - 1})")
print(f"   Exit Code:         {exit_code}")
print(f"   Results Saved To:  {RESULTS_DIR}")
print(f"   Completed at:      {datetime.now().isoformat()}")

if exit_code == 0:
    print("\n✅ ALL 90 EXPERIMENTS COMPLETED SUCCESSFULLY!")
else:
    print(f"\n⚠️  Run completed with exit code {exit_code}")

print("=" * 70)
