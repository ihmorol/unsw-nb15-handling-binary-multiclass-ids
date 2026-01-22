
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cleanup")

def cleanup():
    metrics_dir = Path("results/metrics")
    figures_dir = Path("results/figures")
    
    # Pattern to match XGBoost results
    patterns = ["*xgb*", "*XGB*"]
    
    logger.info(f"Cleaning stale XGBoost results from {metrics_dir}...")
    
    count = 0
    # Clean Metrics
    for pattern in patterns:
        for f in metrics_dir.glob(pattern):
            f.unlink()
            logger.info(f"Deleted: {f.name}")
            count += 1
            
    # Clean Figures (Old format cm_*.png)
    for pattern in ["cm_*xgb*.png", "lc_*xgb*.png"]:
        for f in figures_dir.glob(pattern):
            f.unlink()
            logger.info(f"Deleted: {f.name}")
            count += 1
            
    logger.info(f"Cleanup complete. Removed {count} files.")
    logger.info("Ready to restart 'python scripts/run_full_grid.py' to regenerate XGBoost with learning curves.")

if __name__ == "__main__":
    cleanup()
