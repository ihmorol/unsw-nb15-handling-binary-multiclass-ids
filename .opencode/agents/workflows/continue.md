---
description: Resume the session from the last known state
---
# Workflow: Continue Session

You are responsible for analyzing the project state and resuming work where it left off.

2. **Determine Next Step**:
   - If a script failed, diagnose and retry (call `/debug`).
   - If a batch finished, proceed to the next batch or analysis.
   - If idle, ask the user or check the project plan in `README.md`.

3. **Action**:
   - Explicitly state what you are resuming.
   - Execute the necessary commands.

## Goal
Seamlessly pick up the context and drive the project forward without asking the user "what should I do?".