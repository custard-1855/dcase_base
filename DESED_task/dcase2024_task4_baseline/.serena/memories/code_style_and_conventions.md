# Code Style and Conventions

## Formatting

### Black (Legacy)
- Version: `19.10b0` (older version used in pre-commit)
- Applied to Python files via pre-commit hooks

### Ruff (Modern - Configured in Root)
Located in `/Users/takehonshion/work/iniad/dcase_base/pyproject.toml`:

```toml
[tool.ruff]
line-length = 100

[tool.ruff.format]
docstring-code-format = true

[tool.ruff.lint]
select = ["ALL"]
ignore = [
    "D1",    # undocumented (no docstring requirement)
    "D203",  # one blank line before class
    "D213",  # multi-line summary second line
    "TD001", # invalid todo tag
    "TD002", # missing todo author
    "TD003", # missing todo link
    "PD011", # pandas use of dot values
]
unfixable = [
    "F401", # unused import
    "F841", # unused variable
]

[tool.ruff.lint.pylint]
max-args = 6
```

## Naming Conventions
- **Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: Not strictly enforced, but typically `UPPER_CASE`

## Type Hints
- **Not extensively used** in the codebase
- Function signatures typically use `**kwargs` for flexibility
- No strict type checking enforced

## Docstrings
- **Not required** (D1 ignored in Ruff config)
- When present, use Google-style or NumPy-style
- Some functions and classes have docstrings, particularly in core modules like `desed_task/nnet/`

## Code Organization

### Import Order
Standard Python convention:
1. Standard library imports
2. Third-party imports
3. Local/relative imports

Example from `train_pretrained.py`:
```python
import argparse
import os

import desed
import numpy as np
import pytorch_lightning as pl
import torch

from local.classes_dict import classes_labels_desed
from desed_task.dataio import ConcatDatasetBatchSampler
```

## Indentation
- **4 spaces** per indentation level
- No tabs

## Line Length
- **100 characters** (enforced by Ruff)

## Pre-commit Hooks
Located at `/Users/takehonshion/work/iniad/dcase_base/DESED_task/.pre-commit-config.yaml`:

1. `trailing-whitespace`: Remove trailing whitespace
2. `end-of-file-fixer`: Ensure files end with newline
3. `requirements-txt-fixer`: Sort requirements.txt
4. `mixed-line-ending`: Check for mixed line endings
5. `check-added-large-files`: Prevent committing large files (max 1024KB)
6. `black`: Format Python code (v19.10b0)

## Comments
- Use `#` for single-line comments
- Write clear, explanatory comments for complex logic
- Japanese comments are acceptable in this project (seen in config files)
