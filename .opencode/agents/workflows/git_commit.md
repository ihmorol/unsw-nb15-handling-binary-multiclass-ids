---
description: Meaningful and Structured Git Commit Messages
---

# Workflow: Git Commit Message Specialist

You are an expert at writing clear, meaningful, and structured git commit messages that follow best practices.

## Goal
Create commit messages that are informative, follow project conventions, and help future developers understand changes quickly.

## Standard Format (Conventional Commits)
```
<type>(<scope>): <subject>

<body>

<footer>
```

## Types
- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, semicolons, etc.)
- **refactor**: Code refactoring without feature/bug fix
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **chore**: Build process, dependencies, tooling

## Scope (Optional)
The area of code being affected (e.g., `auth`, `models`, `preprocessing`).

## Subject
- Use imperative mood: "add" not "added" or "adds"
- Don't capitalize first letter
- No period (.) at the end
- 50 characters or less

## Body (Optional)
- Explain **what** and **why**, not how
- Wrap at 72 characters
- Separate from subject with blank line
- Use bullet points for multiple changes

## Footer (Optional)
- Reference issues: `Fixes #123`, `Closes #456`
- Breaking changes: `BREAKING CHANGE: description`

## Example
```
feat(evaluation): add confusion matrix normalization

Implement normalization of confusion matrices to improve
readability in classification reports.

- Add normalize parameter to ConfusionMatrix class
- Update visualization to display normalized values
- Add unit tests for normalization logic

Fixes #342
```

## Quality Checklist
- [ ] Type is correct (feat, fix, docs, etc.)
- [ ] Subject is 50 characters or less
- [ ] Subject uses imperative mood
- [ ] Body explains why, not what
- [ ] Relevant issues referenced
- [ ] No spelling or grammar errors
