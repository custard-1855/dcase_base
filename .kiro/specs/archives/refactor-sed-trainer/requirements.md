# Requirements Document

## Project Description (Input)
sed_trainer_pretrained.pyのリファクタリング:
  冗長な実装を削除し、型安全性を向上させる。具体的には、重複コードの統合、型ヒントの追加、mypy検証の導入を行う。

## Introduction
This specification addresses the refactoring of `sed_trainer_pretrained.py` (2609 lines), the core PyTorch Lightning module for sound event detection training. The refactoring aims to eliminate redundant implementations, improve type safety through comprehensive type hints, and introduce mypy validation. This enhancement aligns with the project's code quality standards established in the tech steering document and follows the Ruff-based code quality framework already in place.

## Requirements

### Requirement 1: Code Deduplication
**Objective:** As a developer, I want redundant code patterns eliminated from sed_trainer_pretrained.py, so that the codebase is maintainable and follows DRY principles.

#### Acceptance Criteria
1. When duplicate code blocks performing similar operations are identified, the sed_trainer_pretrained module shall consolidate them into reusable helper methods
2. When similar logic patterns exist across multiple methods (e.g., validation_step and test_step), the sed_trainer_pretrained module shall extract common functionality into shared private methods
3. When repeated data processing logic is detected, the sed_trainer_pretrained module shall refactor it into single-responsibility utility functions
4. The sed_trainer_pretrained module shall maintain behavioral equivalence with the original implementation after deduplication
5. The sed_trainer_pretrained module shall reduce total line count by at least 10% through consolidation

### Requirement 2: Type Hint Coverage
**Objective:** As a developer, I want comprehensive type hints added to all functions and methods in sed_trainer_pretrained.py, so that the code is self-documenting and type-safe.

#### Acceptance Criteria
1. The sed_trainer_pretrained module shall have type annotations for all method signatures including parameters and return types
2. When complex data structures (dictionaries, lists, tuples) are used, the sed_trainer_pretrained module shall use specific type hints (e.g., `Dict[str, torch.Tensor]`, `List[pd.DataFrame]`) instead of generic types
3. When optional parameters exist, the sed_trainer_pretrained module shall use `Optional[T]` type annotations
4. The sed_trainer_pretrained module shall use `typing` module constructs (Union, Tuple, Dict, List, Optional, Any) where appropriate
5. When class attributes are initialized, the sed_trainer_pretrained module shall include type annotations in the `__init__` method or as class-level annotations
6. The sed_trainer_pretrained module shall achieve 100% type annotation coverage for all public methods and 90%+ coverage for private methods

### Requirement 3: Mypy Validation Integration
**Objective:** As a developer, I want mypy static type checking integrated and passing for sed_trainer_pretrained.py, so that type errors are caught during development before runtime.

#### Acceptance Criteria
1. When mypy is executed against sed_trainer_pretrained.py, the module shall pass validation with no errors
2. If mypy strict mode is incompatible with PyTorch Lightning patterns, the module shall pass mypy validation with clearly documented exceptions in mypy.ini or pyproject.toml
3. The sed_trainer_pretrained module shall resolve all type incompatibilities with third-party libraries (pytorch_lightning, torch, pandas, numpy) through appropriate type annotations or type: ignore comments with justification
4. When ambiguous types exist (e.g., model inputs/outputs), the sed_trainer_pretrained module shall use explicit type aliases defined at module level
5. The project configuration shall include mypy in pyproject.toml with appropriate settings for the DCASE baseline codebase
6. The sed_trainer_pretrained module shall maintain compatibility with existing Ruff lint rules while satisfying mypy requirements

### Requirement 4: Code Structure Preservation
**Objective:** As a developer, I want the refactored code to maintain compatibility with existing training pipelines and configurations, so that existing experiments and workflows continue to function without modification.

#### Acceptance Criteria
1. The SEDTask4 class shall maintain the same public API (method names, signatures, return types) after refactoring
2. When existing configuration files (YAML) reference sed_trainer_pretrained.py, the module shall remain compatible without config changes
3. When existing training scripts import from sed_trainer_pretrained.py, the module shall export the same symbols with backward-compatible interfaces
4. The sed_trainer_pretrained module shall maintain identical behavior for all PyTorch Lightning lifecycle hooks (on_train_start, training_step, validation_step, etc.)
5. When experiment checkpoints from pre-refactoring code exist, the refactored module shall load them without compatibility issues

### Requirement 5: Testing and Validation
**Objective:** As a developer, I want automated verification that the refactored code maintains functional equivalence, so that regressions are detected immediately.

#### Acceptance Criteria
1. When the refactoring is complete, the development team shall execute existing test suites and verify all tests pass
2. If dedicated unit tests for sed_trainer_pretrained.py do not exist, the module shall be validated through integration tests (training runs with small datasets)
3. When validation metrics are computed using refactored code, the module shall produce identical results (within numerical precision) compared to the original implementation
4. The sed_trainer_pretrained module shall pass all Ruff lint checks after refactoring
5. When mypy validation is executed, the module shall complete successfully as defined in Requirement 3

### Requirement 6: Documentation Updates
**Objective:** As a developer, I want clear documentation of type annotations and refactored structure, so that future maintainers understand the code architecture.

#### Acceptance Criteria
1. When complex type aliases are introduced (e.g., for batch structures, score dataframes), the sed_trainer_pretrained module shall define them at module level with docstring explanations
2. When methods are refactored or extracted, the sed_trainer_pretrained module shall include docstrings following Google or NumPy style conventions
3. If significant architectural changes occur during deduplication, the module shall update class-level docstrings to reflect new structure
4. When type: ignore comments are used for mypy exceptions, the sed_trainer_pretrained module shall include inline comments explaining the rationale
5. The sed_trainer_pretrained module shall maintain consistency with existing documentation patterns in the DCASE baseline codebase

