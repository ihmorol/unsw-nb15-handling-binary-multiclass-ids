# Research Paper Transformation Plan

## Goal
Elevate the current undergraduate-level draft to a **World-Class, Publishable Research Paper** (NeurIPS/KDD standard).
**Focus:** Rigorous statistical evidence, deep forensic analysis, and professional, confident academic writing.

## Current State Analysis
- **Structure:** Good standard LaTeX structure exists (`0.0.title.tex` to `6.conclusion.tex`).
- **Content:** "Good" but descriptive. Lacks the *statistical bite* and *forensic depth* we uncovered.
- **Missing Elements:**
    -   Bootstrapped Confidence Intervals (claims are "point estimates").
    -   Friedman Test / Critical Difference analysis.
    -   The "Analysis-Backdoor Confound" explanation.
    -   Operational trade-off analysis (Precision vs Recall for Shellcode).

## Proposed Changes

### 1. Abstract (`0.1.abstract.tex`) [REWRITE]
-   **Current:** "validates improvements".
-   **New:** "Explicitly rejects the null hypothesis ($p < 0.001$) using Friedman tests."
-   **Add:** Mention the "Shellcode Sink" and "Worms Solvability" as key contributions.
-   **Tone:** Shift from "We tried X" to "We demonstrate Y with Z% confidence."

### 2. Implementation & Results (`4.implementation.tex`) [MAJOR OVERHAUL]
-   **Rename:** `4.results_and_analysis.tex` (Focus on Analysis, not just Implementation).
-   **New Section:** "Statistical Validation Protocol".
    -   Explain N=1 Bootstrap methodology (Methodological contribution).
    -   Present Friedman + Nemenyi results (Defense against "random seed" critiques).
-   **New Section:** "Forensic Rare-Class Analysis".
    -   Integrate `docs/reports/deep_dive_rare_classes.md`.
    -   Explain **why** Analysis fails (confound with Backdoor).
    -   Present the "Recall vs Precision" trade-off for Shellcode.
-   **Visuals:** Replace basic tables with references to the new `rank_comparison.png` and `heatmap_class_f1.png`.

### 3. Discussion (`5.sic.tex`) [REFINE]
-   **Focus:** Move from "Results summary" to "Implications for NIDS Design".
-   **Key Point:** The "No Free Lunch" theorem in Intrusion Detection (Recall cost).
-   **Key Point:** The failure of features for Analysis/Backdoor (Call for Feature Engineering, not Model Tuning).

### 4. Title (`0.0.title.tex`) [POLISH]
-   Refine title to be more punchy and precise if needed (e.g., adding "Statistical Validation" or "Rare-Class Forensics").

## Execution Steps
1.  **Rewrite Abstract**: Set the high bar immediately.
2.  **Rewrite Results**: Inject the hard math (CIs, Friedman) and deep forensics.
3.  **Refine Intro/Conclusion**: Ensure consistency with the new strong claims.
4.  **Verify**: Latex compilation check.
