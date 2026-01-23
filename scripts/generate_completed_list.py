import os
from pathlib import Path

RESULTS_DIR = Path(r"e:\University\ML\ML_PAPER_REVIEW/results")
METRICS_DIR = RESULTS_DIR / "metrics"

def get_completed():
    if not METRICS_DIR.exists():
        return []
    
    # Get all json files
    completed = [f.stem for f in METRICS_DIR.glob("*.json")]
    return sorted(completed)

if __name__ == "__main__":
    completed_ids = get_completed()
    print(f"FOUND {len(completed_ids)} COMPLETED EXPERIMENTS:")
    print("="*60)
    print("completed_ids = [")
    for cid in completed_ids:
        print(f"    '{cid}',")
    print("]")
    print("="*60)
    
    # Redundancy Check
    colab_folder = RESULTS_DIR / "results-from-colab-drive"
    if colab_folder.exists():
        print(f"\nREDUNDANT FOLDER DETECTED: {colab_folder}")
        print("Recommendation: Delete this folder after confirming 'results/metrics' has everything.")
