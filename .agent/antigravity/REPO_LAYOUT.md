# Repository Layout

Keep the repo structured so new members can find everything quickly.

## Directory Structure

```
ML_PAPER_REVIEW/
├── .agent/                     # AI agent configuration
│   ├── antigravity/            # Workspace rules (this directory)
│   ├── rules/                  # Persistent agent rules
│   └── workflows/              # Agent personas and workflows
│
├── configs/
│   └── main.yaml               # Single source of truth for config
│
├── dataset/
│   ├── UNSW_NB15_training-set.csv
│   ├── UNSW_NB15_testing-set.csv
│   └── NUSW-NB15_features.csv
│
├── docs/
│   ├── contracts/              # Authoritative contracts
│   │   ├── data_contract.md
│   │   ├── experiment_contract.md
│   │   └── methodology_contract.md
│   └── reports/                # Analysis reports
│
├── src/
│   ├── data/                   # DataLoader, Preprocessor
│   ├── models/                 # ModelTrainer
│   ├── strategies/             # S0, S1, S2a handlers
│   ├── evaluation/             # Metrics, plotting
│   └── utils/                  # Config, logging
│
├── scripts/                    # Utility scripts
│
├── results/                    # Generated outputs (gitignored)
│   ├── metrics/                # {exp_id}.json files
│   ├── figures/                # cm_{exp_id}.png files
│   ├── tables/                 # Summary CSVs
│   ├── logs/                   # Run logs
│   ├── models/                 # Trained models (optional)
│   └── processed/              # preprocessing_metadata.json
│
├── paper/                      # Research paper source
│
├── main.py                     # Experiment orchestrator
├── runner.py                   # Alternative runner
└── requirements.txt            # Python dependencies
```

## Editable vs Generated

| Directory | Type | Notes |
|-----------|------|-------|
| `src/` | ✏️ Editable | Core implementation |
| `configs/` | ✏️ Editable | Configuration only |
| `docs/` | ✏️ Editable | Contracts and reports |
| `paper/` | ✏️ Editable | Research paper |
| `scripts/` | ✏️ Editable | Utility scripts |
| `.agent/` | ✏️ Editable | Agent rules and workflows |
| `results/` | 🔒 Generated | Do not hand-edit |
| `dataset/` | 🔒 External | Do not modify |

## Key Files

| File | Purpose |
|------|---------|
| `configs/main.yaml` | All configuration (paths, params, grid) |
| `main.py` | Run all 18 experiments |
| `docs/contracts/*.md` | Authoritative methodology |
| `.agent/antigravity/*.md` | Quick-reference rules |
