# Agent Workflows System

This directory contains reusable agent personas, workflows, and prompts for coordinating ML research tasks.

## Directory Structure

```
agents/
├── README.md                          # This file
├── MANIFEST.md                        # Central registry of all agents
├── workflows/                         # Workflow protocols and procedures
│   ├── debug.md                      # Systematic Debugging Workflow
│   ├── explain.md                    # Technical Explanation Workflow
│   ├── docstring.md                  # Docstring Generation Workflow
│   ├── git_commit.md                 # Git Commit Message Workflow
│   ├── optimize.md                   # Code Optimization Workflow
│   ├── unit_test.md                  # Unit Testing Workflow
│   ├── write_findings.md             # Findings Report Workflow
│   └── [more workflows...]
├── prompts/                           # Specialized prompts for complex tasks
│   ├── paper_review_prompt.md        # Comprehensive paper review prompt
│   ├── experiment_audit_prompt.md    # Experiment validation prompt
│   └── [more prompts...]
└── config/                            # Configuration files
    ├── default_agents.yaml            # Default agent configurations
    └── rules/                         # Shared rules and standards
        ├── MISSION_AND_STANDARDS.md
        ├── CODING_STANDARDS.md
        ├── QA_GATES.md
        └── [more rules...]
```

## Usage

### Activating an Agent

Use the agent workflow system by specifying:
1. **Persona**: Who is performing the task?
2. **Workflow**: What process should they follow?
3. **Prompt**: What specific instruction do they need?

Example:
```
Agent: Auditor (QA Planner)
Workflow: Experiment Validation
Prompt: "Validate all metrics in experiment run X123"
```

### Creating New Agent Workflows

When creating a new agent:
1. Define the persona in `personas/`
2. Define workflows in `workflows/`
3. Create specialized prompts in `prompts/`
4. Register in `MANIFEST.md`

## Agent Categories

### Research & Analysis
- **Auditor**: QA and test planning
- **Reviewer**: Statistics and reproducibility validation
- **Paper Reviewer**: ML paper peer review
- **Lead**: Project oversight and coordination

### Development
- **Executor**: Code implementation and experiment execution
- **Debugger**: Root cause analysis
- **Explainer**: Code and concept explanation

### Communication
- **Author**: Research paper writing
- **Visualizer**: Data visualization and diagrams
- **Technical Writer**: Documentation and findings

## Integration with OpenCode

These workflows are designed to work with OpenCode's agent system:
- Personas provide role-specific expertise
- Workflows enforce consistent procedures
- Prompts guide task execution
- Rules ensure quality and reproducibility

## Best Practices

1. **Always reference sources of truth** - Link to data/artifact locations
2. **Be specific and measurable** - Clear inputs, outputs, acceptance criteria
3. **Enforce non-negotiables** - Quality gates and mandatory checks
4. **Document assumptions** - Be explicit about constraints and limitations
5. **Enable reproducibility** - Provide traceable artifact paths and configs

## Contributing

When extending this system:
- Follow the persona template structure
- Use consistent formatting (Markdown)
- Include specific examples and use cases
- Link to related rules and standards
- Update MANIFEST.md
