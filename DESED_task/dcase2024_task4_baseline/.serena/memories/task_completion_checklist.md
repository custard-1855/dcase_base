# Task Completion Checklist

When you complete a coding task in this project, follow these steps:

## 1. Code Formatting

### Option A: Use Ruff (Recommended - Modern)
```bash
# Navigate to project root
cd /Users/takehonshion/work/iniad/dcase_base

# Format modified files
ruff format .

# Fix linting issues
ruff check --fix .
```

### Option B: Use Black (Legacy - for compatibility)
```bash
# Format all Python files
black .

# Or format specific file
black path/to/modified_file.py
```

## 2. Run Pre-commit Hooks

```bash
# Run all pre-commit hooks on changed files
pre-commit run

# Or run on all files to ensure consistency
pre-commit run --all-files
```

This will automatically:
- Remove trailing whitespace
- Fix end-of-file newlines
- Check for large files
- Format code with Black
- Check for mixed line endings

## 3. Verify Changes

### Check Code Quality
```bash
# Run ruff linting (no auto-fix) to see all issues
ruff check .

# Review output for any remaining issues
```

### Run Tests (if applicable)
```bash
# If you modified sebbs module
cd sebbs
pytest tests/

# Or run specific test
pytest tests/test_specific.py
```

## 4. Review Configuration Files (if modified)

If you modified YAML configuration files:
- Ensure YAML syntax is valid
- Check that paths are correct
- Verify parameter values are reasonable

## 5. Manual Review

- [ ] Code follows project conventions (snake_case, PascalCase for classes)
- [ ] Line length ≤ 100 characters
- [ ] No unused imports or variables (unless necessary)
- [ ] Comments are clear and helpful
- [ ] No debug print statements left in code
- [ ] Configuration changes are documented

## 6. Git Workflow (if committing)

```bash
# Check status
git status

# Review changes
git diff

# Stage changes
git add <files>

# Commit with meaningful message
git commit -m "Descriptive commit message"
```

## Notes

- **No explicit test suite** exists for the main baseline code, so manual testing may be required
- **Energy consumption tracking**: If you modified training code, ensure CodeCarbon tracker is still properly initialized
- **Embeddings**: If you modified embedding extraction, re-run `python extract_embeddings.py --output_dir ./embeddings`
- **Configuration sync**: Keep `confs/*.yaml` in sync with any code changes to model architecture or data loading
