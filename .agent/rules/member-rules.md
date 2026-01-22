---
trigger: always_on
---

## mission:
    Build a clean, reproducible classical-ML baseline for binary and multiclass intrusion detection on UNSW-NB15,
    systematically evaluating imbalance strategies and rare attack detection using macro and per-class metrics.

  # Canonical terminology used across docs/configs/code/results
  glossary:
    tasks:
      binary: "Normal vs Attack"
      multi: "10-class (Normal + 9 attack categories)"
    models:
      LR: "Logistic Regression"
      RF: "Random Forest"
      XGB: "XGBoost"
    strategies:
      S0_none: "No imbalance handling (raw training distribution)"
      S1_class_weight: "Cost-sensitive learning via class weights (train-time only)"
      S2a_ros: "RandomOverSampler applied to training split only (default)"
      S2b_smote: "SMOTE oversampling applied to training split only (optional)"
    primary_metrics:

      - "macro_f1"
      - "per_class_precision_recall_f1"
      - "confusion_matrix"
      - "rare_class_report"

  global_standards:
    evidence_first: "No claims without an artifact (CSV/JSON) or a credible citation."
    leakage_zero_tolerance: >
      Split first; fit preprocessing on train only; resample/train-only; validation/test untouched; official test untouched until final.
    reproducibility: "Every run has config + seed + run_id + logs + metrics + tables."
    comparability: "Same split, same preprocessing contract, same metric definitions across runs."
    reporting: "Macro-F1 + per-class + confusion matrices; rare-class report mandatory."

  # Single source of truth for "what must exist after a full run"
  artifact_contract:
    required_directories:
      - "results/"
      - "results/metrics/"
      - "results/figures/"
      - "results/tables/"
      - "results/logs/"
      - "results/processed/"
    per_experiment_required_files:
      - "results/metrics/{exp_id}.json"
      - "results/figures/cm_{exp_id}.png"
    global_required_files:
      - "results/experiment_log.csv"
      - "results/tables/final_summary_tables.csv"
      - "results/tables/per_class_metrics.csv"
      - "results/tables/rare_class_report.csv"
    processed_required_files:
      - "results/processed/preprocessing_metadata.json"


  # Non-negotiable experimental protocol (matches Methodology.md)
  protocol:
    data_split:
      outer_split: "Use official UNSW-NB15 train vs official test; never mix."
      inner_split:
        validation_fraction: 0.20
        stratified: true
        seed_from_config: true
    preprocessing:
      fit_on: "train only"
      transform_on: ["val", "test"]
      steps:
        - "Drop identifiers / leakage-prone columns (IDs, IPs, timestamps, etc.)."
        - "Impute numeric with median; categorical with 'missing'."
        - "One-hot encode categorical features (handle_unknown=ignore)."
        - "Scale numeric features with StandardScaler (for consistency across models)."
    imbalance_handling:
      allowed_strategies: ["S0_none", "S1_class_weight", "S2a_ros"]
      optional_strategies: ["S2b_smote"]
      rule: "Apply S1/S2 only on training split; never on validation/test."
    experiment_matrix:
      tasks: ["binary", "multi"]
      models: ["LR", "RF", "XGB"]
      strategies: ["S0_none", "S1_class_weight", "S2a_ros"]
      expected_total_runs: 18
