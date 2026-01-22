# State of the "Art" Analysis: UNSW-NB15 Imbalance Strategies

**Date:** 2026-01-22
**Project:** UNSW-NB15 Class Imbalance Analysis
**Review Team:** Dr. Ayaan Rahman (Lead Reviewer), Statistical Reviewer, QA Auditor

---

## 1. Executive Summary

The repository represents a **technically excellent** engineering baseline but currently falls short of "State of the Art" (SOTA) research standards due to a lack of experimental execution and statistical depth. 

While the code quality, modularity, and leakage prevention mechanisms are virtually perfect (top 1% of research repos), the valid "research" component is currently non-existent as no results have been generated. To become publishable, the project must shift from "building the tool" to "generating robust evidence" with multiple seeds and external comparisons.

**Status:** `READY_FOR_EXECUTION`
**Publication Potential:** High (if Rare Class angle is exploited)

---

## 2. Implementation Audit (The Auditor)

**Verdict:** ✅ **PASS (With Distinction)**

The implementation defies the common "research code" stereotype. It is robust, modular, and strictly enforces the *Evaluation Contract*.

*   **Leakage Prevention:** The `UNSWPreprocessor` correctly fits *only* on the training split. Validation and Test splits are transformed using the frozen preprocessor. This is the #1 reason papers get rejected, and you have solved it perfectly.
*   **Split Safety:** The "Official Test Set" is respected and never touched during training or tuning.
*   **Strategy Isolation:** S2a (ROS) and S2b (SMOTE) are applied inside the training loop *after* splitting. This prevents "synthetic data leakage" where a synthetic sample in train is a neighbour of a real sample in test.
*   **Artifacts:** The code is set up to save everything (models, logs, metrics), ensuring full reproducibility.

**Minor Note:** The `requirements.txt` is minimal. Ensure versions are pinned (e.g., `scikit-learn==1.3.0`) for long-term reproducibility.

---

## 3. Research & Novelty Critique (Dr. Ayaan Rahman)

**Verdict:** ⚠️ **CRITICAL GAPS**

As a reviewer for NeurIPS/ICML/KDD, I would reject a paper based solely on this *plan* for the following reasons:

1.  **The "So What?" Problem:** Comparing S0 vs S1 vs S2a on standard classifiers is a well-trodden path (2015-2020 era). The "Novelty" must come from the **Rare Class Analysis** (Worms, Shellcode).
    *   *Action:* The `rare_class_report.csv` is your golden ticket. The paper shouldn't just say "S2a is better"; it must say "S2a improves Worm detection by 400% while maintaining <1% False Positive Rate increase".

2.  **The "Default Hyperparams" Trap:**
    *   You are using fixed hyperparameters (e.g., `RF max_depth=25`, `XGB learning_rate=0.1`) without justification.
    *   *Reviewer Comment:* "How do we know S2a is actually better? Maybe S0 just needed a deeper tree? Maybe S1 works best if you tune `C`?"
    *   *Fix:* You don't need massive AutoML. But you *must* show that you at least tried to tune the baseline (S0) on the validation set.

3.  **Missing External Baselines:**
    *   A SOTA analysis compares against *competitors*, not just yourself. You need to cite 2-3 recent papers (2022-2024) using UNSW-NB15 and list their reported F1/G-Mean in your final table.

---

## 4. Statistical Rigor (The Statistician)

**Verdict:** ❌ **FAIL (Current Protocol)**

1.  **Single Seed Syndrome:**
    *   Running with `random_state=42` only is insufficient.
    *   *Requirement:* You must run 5-10 seeds (e.g., 42, 43, ... 52).
    *   *Output:* Report `Mean ± Std Dev` (e.g., "Macro F1: 0.85 ± 0.02").
    *   *Why:* In high-imbalance scenarios, one "lucky" minority fold can skew the G-Mean massively.

2.  **Significance Testing:**
    *   Is S2a *significantly* better than S1? Or is it just noise?
    *   *Action:* Implement a paired t-test or Wilcoxon signed-rank test on the 5 seeds' results.

---

## 5. Roadmap to Publication (Bias-Free)

To transform this from a "School Project" to "Publishable Research", execute the following order:

### Phase 1: The "Lazy" Baseline (Now)
1.  **Execute the Grid:** Run `python main.py` as is.
2.  **Sanity Check:** Ensure the "Worms" class has >0 recall in at least one strategy. If not, the research is dead on arrival.

### Phase 2: The "SOTA" Upgrade (Next 24 Hours)
1.  **Multi-Seed Wrapper:** Create a `run_experiments.py` that loops `main.py` with seeds `[42, 43, 44, 45, 46]`.
2.  **Aggregator:** Create a script to average the 5 `metrics.json` files for each experiment.
3.  **Literature Comparison:** Find 3 papers. Add their numbers to `final_summary_tables.csv` manually as "Reference_Paper_1", etc.

### Phase 3: The Narrative
*   **Title:** "Beyond Accuracy: A Rigorous Evaluation of Resampling Strategies for Rare Attack Detection in NIDS"
*   **Abstract:** Focus on the *trade-off* between detecting Worms/Backdoors and false alarms. Do not claim "SOTA performance" unless you beat the references. Claim "SOTA Evaluation Methodology".

---

### Conclusion
You have built a Ferrari engine (the code) but haven't put gas in it (data/experiments) or driven it on the track (statistical rigor). **Fill the tank and drive.**
