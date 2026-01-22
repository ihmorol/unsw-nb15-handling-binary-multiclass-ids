#!/usr/bin/env python3
"""
Orchestration script for valid 18-experiment grid execution.
Matches rules in docs/contracts/experiment_contract.md.
"""
import sys
import os
sys.path.append(os.getcwd())

import logging
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("results/logs/orchestrator.log")
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*60)
    logger.info("STARTING FULL EXPERIMENTAL GRID (18 Runs)")
    logger.info("="*60)

    # 1. Clean previous temp results (optional - safety first)
    # We rely on main.py skipping existing, but let's ensure we are clean
    
    # 2. Execute main.py
    # We run it directly as a subprocess to ensure clean state or just import?
    # Importing main is better for debugging, but subprocess handles memory better over long runs.
    # Given main.py uses joblib, we can call it directly.
    
    start_time = time.time()
    
    cmd = [sys.executable, "main.py"]
    logger.info(f"Executing: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output
        with process.stdout:
            for line in iter(process.stdout.readline, b''):
                if line:
                    print(line, end='')
                    
        return_code = process.wait()
        
        if return_code != 0:
            logger.error(f"Experiment execution failed with code {return_code}")
            sys.exit(return_code)
            
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user.")
        process.terminate()
        sys.exit(130)
        
    duration = time.time() - start_time
    logger.info(f"Grid execution completed in {duration/60:.2f} minutes.")

if __name__ == "__main__":
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    main()
