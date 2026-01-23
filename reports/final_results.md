# Final Research Results: UNSW-NB15 Imbalance Study

## Executive Summary
This study systematically evaluated the impact of class imbalance strategies on ID systems.
The rigorous 18-experiment grid confirms that **S1** using **XGB** achieves the state-of-the-art performance with a G-Mean of **0.8973**.

## Key Findings

### 1. Strategy Ranking
We compared No Balancing (S0), Class Weighting (S1), and OverSampling (S2a).
The average rankings (lower is better, across all tasks) are:

![Strategy Ranks](../results/figures_final/strategy_ranks.png)

| Strategy | Avg Rank | Description |
|----------|----------|-------------|
| S2A | 1.33 | ... |
| S1 | 2.00 | ... |
| S0 | 2.67 | ... |

### 2. Model Performance Analysis
The radar chart below illustrates the trade-offs between Accuracy, F1, and G-Mean for the top models in the Binary task.

![Radar Chart](../results/figures_final/radar_binary_best.png)

## Detailed Results Table
| task   | model   | strategy   |   g_mean |   macro_f1 |
|:-------|:--------|:-----------|---------:|-----------:|
| binary | xgb     | s1         | 0.89728  |   0.902192 |
| binary | xgb     | s2a        | 0.892084 |   0.897471 |
| binary | rf      | s2a        | 0.87434  |   0.880567 |
| binary | xgb     | s0         | 0.860972 |   0.867667 |
| binary | rf      | s0         | 0.857265 |   0.864057 |