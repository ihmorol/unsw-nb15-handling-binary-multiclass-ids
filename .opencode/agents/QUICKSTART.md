# Agent Workflows System - Quick Start Guide

## Overview

The Agent Workflows System provides a complete framework for coordinating specialized agents across your ML research project. Each agent has a defined persona, workflow, and set of prompts for consistent, high-quality execution.

---

## Key Components

### 1. **Personas** (`agents/personas/`)
Role definitions with expertise, strengths, and goals.
- **Auditor**: QA planning and acceptance testing
- **Author**: Research paper writing
- **Executor**: Implementation and experiment execution
- **Lead**: Project oversight and coordination
- **Paper Reviewer**: Peer review expertise
- **Reviewer**: Statistics and reproducibility
- **Visualizer**: Data visualization and diagrams
- And more...

### 2. **Workflows** (`agents/workflows/`)
Step-by-step procedures and protocols for common tasks.
- **Debug**: Systematic root cause analysis
- **Explain**: Technical education and concept breakdown
- **Docstring**: Documentation generation
- **Git Commit**: Version control messaging
- **Optimize**: Performance improvement
- **Unit Test**: Test strategy and implementation
- **Write Findings**: Results documentation
- **Reality Check**: Assumption validation
- **Strategic Audit**: High-level project assessment

### 3. **Prompts** (`agents/prompts/`)
Specialized instructions for complex tasks.
- **paper_review_prompt.md**: Complete ML paper review procedure
- **experiment_audit_prompt.md**: Rigorous experiment validation
- **implementation_planning_prompt.md**: Executable plan authoring

### 4. **Configuration** (`agents/config/`)
Default setups and integration rules.
- **default_agents.yaml**: Agent initialization parameters
- **rules/**: Shared standards and guardrails

---

## Quick Start Examples

### Example 1: Activate the Auditor for QA Planning

**Goal**: Create a comprehensive implementation plan

```
Agent: Auditor (QA Planner)
Workflow: Quality Assurance
Task: "Review the proposed methodology and create an executable implementation plan"

Sources to use:
- docs/Methodology_Analysis.md
- docs/contracts/data_contract.md
- configs/main.yaml

Expected outputs:
- docs/implementation_plan/00_overview.md through 09_acceptance_tests_checklist.md
- Acceptance test definitions
- Risk assessment document
```

**Result**: Complete, executable plan that the Executor can follow precisely.

---

### Example 2: Run the Executor

**Goal**: Implement the methodology and produce research-grade artifacts

```
Agent: Executor (Methodology Executor)
Workflow: Implementation
Task: "Implement the methodology from docs/implementation_plan/ exactly"

Inputs:
- docs/implementation_plan/ (complete reference)
- configs/main.yaml (parameters)
- data/raw/ (source data)

Expected outputs:
- results/metrics/{exp_id}.json (18 experiment results)
- results/tables/final_summary_tables.csv
- results/experiment_log.csv
- results/figures/cm_*.png (confusion matrices)
- results/logs/run_*.log (execution logs)

Quality gates:
- All 18 experiments complete without error
- All artifacts match expected schema
- Reproducible across runs
```

**Result**: Research-grade artifacts ready for analysis.

---

### Example 3: Statistical Validation

**Goal**: Ensure reported improvements are statistically sound

```
Agent: Reviewer (Statistics & Reproducibility)
Workflow: Statistical Validation
Prompt: prompts/experiment_audit_prompt.md
Task: "Validate the statistical soundness of all reported metrics"

Check:
- Confidence intervals on all metrics (95% CI minimum)
- Paired tests for strategy comparisons
- Effect sizes reported (not just p-values)
- Uncertainty handling for rare classes
- No test-set tuning or selection bias

Deliverables:
- results/tables/metric_confidence_intervals.csv
- results/tables/paired_significance_tests.csv
- docs/statistical_validation_notes.md
```

**Result**: Defense-ready statistical validation.

---

### Example 4: Peer Review of Paper

**Goal**: Rigorous peer review before submission

```
Agent: Paper Reviewer (Dr. Ayaan Rahman)
Prompt: prompts/paper_review_prompt.md
Task: "Provide a decision-quality peer review of the manuscript"

Review includes:
1. Summary of contributions
2. Strengths
3. Major concerns (with concrete fixes)
4. Minor concerns
5. Suggested experiments (prioritized)
6. Ethics/safety assessment
7. Rating + confidence
8. Questions for authors

Output format: Structured review document with actionable feedback
```

**Result**: Peer-quality feedback before journal submission.

---

### Example 5: Data Visualization

**Goal**: Create publication-ready figures

```
Agent: Visualizer
Task: "Create confusion matrices and performance comparison charts"

Inputs:
- results/tables/final_summary_tables.csv
- results/metrics/*.json

Outputs:
- figures/confusion_matrix_comparison.png
- figures/model_performance_by_strategy.png
- figures/roc_curves.png
- Mermaid diagrams for methodology flowchart

Quality:
- 300 DPI for publication
- Colorblind-friendly palettes
- Clear titles and labeled axes
- Proper figure captions
```

**Result**: Publication-ready visualizations.

---

## Integration Workflows

### Full Research Pipeline

```
1. LEAD (Project Overview)
   ↓
2. AUDITOR (QA Planning & Implementation Plan)
   ↓
3. EXECUTOR (Implementation & Experiments)
   ↓
4. REVIEWER (Statistical Validation)
   ↓
5. VISUALIZER (Figure Creation)
   ↓
6. AUTHOR (Paper Writing)
   ↓
7. REALITY_CHECKER (Assumption Validation)
   ↓
8. PAPER_REVIEWER (Peer Review)
   ↓
PUBLISHED!
```

### Code Development Cycle

```
1. EXPLAINER (Understand existing code)
   ↓
2. DEBUGGER (Identify issues)
   ↓
3. EXECUTOR (Implement fixes)
   ↓
4. UNIT_TEST (Verify correctness)
   ↓
5. OPTIMIZER (Improve performance)
   ↓
READY FOR PRODUCTION
```

### Quality Assurance Loop

```
1. REALITY_CHECKER (Validate assumptions)
   ↓
2. AUDITOR (Design QA plan)
   ↓
3. REVIEWER (Statistical soundness)
   ↓
4. STRATEGIC_AUDITOR (High-level assessment)
   ↓
QUALITY GATE PASSED
```

---

## Using Agents in Your Project

### Step 1: Understand the System
- Read `agents/README.md` for overview
- Review `agents/MANIFEST.md` for all agents
- Check `agents/config/default_agents.yaml` for configurations

### Step 2: Activate the Right Agent
Match your task to the appropriate agent:

| Task | Agent | Workflow |
|------|-------|----------|
| Create implementation plan | Auditor | QA Planning |
| Run experiments | Executor | Implementation |
| Validate statistics | Reviewer | Statistical Validation |
| Review peer submission | Paper Reviewer | Paper Review |
| Create figures | Visualizer | Visualization |
| Write paper | Author | Academic Writing |
| Fix code | Debugger | Debugging |
| Improve performance | Optimizer | Optimization |

### Step 3: Provide Context
Give the agent:
- **Task**: Clear objective
- **Sources**: Files and artifacts to reference
- **Constraints**: Success criteria and guardrails
- **Timeline**: If applicable

### Step 4: Use Outputs
- Review agent's deliverables
- Provide feedback if needed
- Move to next stage or agent
- Maintain artifact traceability

---

## Standards & Rules

All agents follow these standards (in `agents/config/rules/`):

1. **Mission & Standards**: Core project values and quality expectations
2. **Coding Standards**: Code style and structure guidelines
3. **QA Gates**: Mandatory checkpoints and acceptance criteria
4. **Data Rules**: Data handling, leakage prevention, contracts
5. **Experiment Protocol**: Experimental methodology standards
6. **Artifact Contract**: Output format and naming requirements

---

## Advanced Usage

### Custom Agent Creation

To create a new agent:

1. **Define persona**: Create `agents/personas/your_agent.md`
2. **Define workflow**: Create `agents/workflows/your_workflow.md`
3. **Create prompt**: Optional specialized prompt in `agents/prompts/`
4. **Register**: Add to `agents/MANIFEST.md`
5. **Configure**: Add entry to `agents/config/default_agents.yaml`

### Agent Specialization

Combine agents for specialized tasks:
- **Code Review**: EXPLAINER + DEBUGGER + UNIT_TEST
- **Feature Engineering**: EXECUTOR + REVIEWER + OPTIMIZER
- **Documentation**: AUTHOR + VISUALIZER + FINDINGS_WRITER

---

## Troubleshooting

### Agent Output Doesn't Match Expectations
- Check if the right agent was selected
- Verify task description was specific and clear
- Ensure all required context/artifacts were provided
- Review agent persona to confirm role match

### Missing or Incomplete Artifacts
- Check agent's workflow for expected outputs
- Verify success criteria were met
- Ask agent to clarify any assumptions
- Review source artifacts for completeness

### Quality Issues
- Use REALITY_CHECKER to validate assumptions
- Use REVIEWER for statistical concerns
- Use STRATEGIC_AUDITOR for high-level assessment

---

## Key Principles

1. **Evidence-Based**: All claims linked to artifacts
2. **Specific**: No vague instructions; exact parameters specified
3. **Traceable**: Every output traced to input and decision
4. **Reproducible**: All experiments use fixed seeds and documented configs
5. **Transparent**: Limitations and negative results reported
6. **Executable**: Plans can be followed by other engineers

---

## Directory Reference

```
agents/
├── README.md                    # System overview
├── MANIFEST.md                  # Central agent registry
├── QUICKSTART.md               # This file
├── personas/                    # Agent role definitions
├── workflows/                   # Procedural guidelines
├── prompts/                     # Specialized task prompts
├── config/                      # Configuration files
│   ├── default_agents.yaml      # Default setup
│   └── rules/                   # Shared standards
└── examples/                    # (Optional) Usage examples
```

---

## Getting Help

- **What agent for my task?** → See "Using Agents" section above or MANIFEST.md
- **How does an agent work?** → Read the agent's persona file and workflow
- **Custom instructions?** → Use prompt files in `agents/prompts/`
- **Modify behavior?** → Edit persona, workflow, or prompt files
- **Add new agent?** → Follow "Custom Agent Creation" above

---

## Next Steps

1. Start with **LEAD** (Project Manager) to understand scope
2. Use **AUDITOR** to create implementation plan
3. Run **EXECUTOR** to implement methodology
4. Validate with **REVIEWER** and **REALITY_CHECKER**
5. Document with **AUTHOR** and **VISUALIZER**
6. Get feedback with **PAPER_REVIEWER**

Good luck! 🚀

---

**Last Updated**: January 22, 2026  
**System Version**: 1.0
