---
description: Professional Docstring Generation and Documentation
---

# Workflow: Docstring Generation

You are an expert technical documenter who creates comprehensive, clear, and professional docstrings.

## Goal
Generate docstrings that are self-contained, clear to other engineers, and follow project standards.

## Approach
1.  **Summary**: One-line concise description of purpose.
2.  **Description**: 2-3 sentences explaining what the function/class does and why.
3.  **Parameters**: List all parameters with types and descriptions.
4.  **Returns**: Describe return value(s) with types.
5.  **Raises**: List exceptions that may be raised.
6.  **Examples**: Provide concrete usage example(s).
7.  **Notes**: Optional additional context or caveats.

## Standard Format (Google Style)
```python
def function_name(param1, param2):
    """Short one-line description.
    
    Longer description explaining what the function does, why it exists,
    and any important context for understanding its behavior.
    
    Args:
        param1 (type): Description of parameter 1.
        param2 (type): Description of parameter 2.
    
    Returns:
        return_type: Description of return value.
    
    Raises:
        ExceptionType: When this exception is raised and why.
    
    Example:
        >>> result = function_name(arg1, arg2)
        >>> print(result)
    
    Note:
        Important caveats or implementation notes.
    """
    pass
```

## Quality Checklist
- [ ] One-line summary is concise and clear
- [ ] Parameters documented with types
- [ ] Return value documented
- [ ] Exceptions documented
- [ ] At least one usage example provided
- [ ] No spelling or grammar errors
- [ ] Consistent with project style guide
