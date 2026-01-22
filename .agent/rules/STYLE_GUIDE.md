# Style Guide (Code + Docs)

## Naming Conventions

### Files and Functions
- Use `snake_case` for files and functions.
- Use `PascalCase` for classes.

### Experiment Identifiers
| Component | Format | Values |
|-----------|--------|--------|
| task | lowercase | `binary`, `multi` |
| model | lowercase | `lr`, `rf`, `xgb` |
| strategy | lowercase | `s0`, `s1`, `s2a`, `s2b` |
| experiment_id | `{task}_{model}_{strategy}` | `binary_rf_s1` |

## Python Code Standards

- **Style:** Follow PEP 8.
- **Type Hints:** Required for all function signatures.
- **Docstrings:** Google-style docstrings.
- **Imports:** Sorted (stdlib → third-party → local).
- **Error Handling:** Use specific exceptions; no bare `except:`.

## Logging

Every run writes a log with:
- Dataset paths
- Split sizes
- Class distribution (train/val/test)
- Config + seed
- Metric outputs
- Timestamps

**Logger usage:**
```python
from src.utils import setup_logging
logger = logging.getLogger(__name__)
```

## Output Artifacts

### JSON Metrics
```json
{
  "experiment_id": "...",
  "metrics": {
    "overall": {...},
    "per_class": {...}
  }
}
```

### CSV Tables
- Generated from code, never typed manually.
- Include headers with descriptive column names.

### Confusion Matrices
- Saved as PNG with clear labels.
- Use colormap appropriate for values.
- Title includes experiment ID.

## Visualization Standards

Full standards: `.agent/workflows/visualization_standards.md`

Key points:
- Consistent color palette across figures
- Font sizes readable at publication scale
- Axes labels include units where applicable
- Rare class results highlighted

## Commits

- **Format:** Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`)
- **Frequency:** Commit after every logical unit of work.
- **Messages:** Concise, descriptive, present tense.
