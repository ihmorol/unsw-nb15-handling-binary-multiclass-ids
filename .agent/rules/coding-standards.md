---
trigger: always_on
---

# Coding Standards

## Python
- **Style:** Follow PEP 8.
- **Type Hints:** Required for all function signatures (`def foo(x: int) -> str:`).
- **Docstrings:** Google-style docstrings for all classes and functions.
- **Imports:** Sorted and grouped (STDLIB, THIRD_PARTY, LOCAL).
- **Error Handling:** Use specific exceptions; no bare `except:`.

## Reproducibility
- **Seeds:** Always use `random_state=42` (from config).
- **Paths:** Use relative paths from project root.
- **Logging:** Use `src.utils.logging`; never `print()`.

## Commits
- **Format:** Conventional Commits (`feat: ...`, `fix: ...`, `docs: ...`).
- **Frequency:** Commit after every logical unit of work.
