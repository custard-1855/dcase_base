# Research & Design Decisions

---
**Purpose**: Capture discovery findings, architectural investigations, and rationale that inform the technical design for sed_trainer_pretrained.py refactoring.

**Usage**:
- Log research activities and outcomes during the discovery phase.
- Document design decision trade-offs that are too detailed for `design.md`.
- Provide references and evidence for future audits or reuse.
---

## Summary
- **Feature**: `refactor-sed-trainer`
- **Discovery Scope**: Extension (Existing System Refactoring)
- **Key Findings**:
  - Current state: 2609 lines, 39 methods, minimal type annotations (only 3/39 methods have return types)
  - Existing mypy configuration present but permissive (`disallow_untyped_defs = False`)
  - Duplicate patterns identified across `training_step`, `validation_step`, `test_step` lifecycle hooks
  - PyTorch Lightning type hinting has known compatibility challenges requiring pragmatic approach

## Research Log

### Current Codebase State Analysis
- **Context**: Understanding the refactoring scope and existing technical debt in sed_trainer_pretrained.py
- **Sources Consulted**:
  - Direct file inspection (`sed_trainer_pretrained.py`, line count: 2609)
  - Method signature analysis (39 total methods)
  - Type annotation coverage check (grep for return type annotations)
- **Findings**:
  - **Type Coverage**: Only 3 out of 39 methods (~8%) have explicit return type annotations
  - **Parameter Annotations**: No systematic parameter type hints observed
  - **Class Structure**: Single monolithic `SEDTask4(pl.LightningModule)` class with helper classes (`_NoneSafeIterator`, `SafeCollate`, `SafeDataLoader`)
  - **Duplicate Patterns**: Repeated logic in `training_step:515`, `validation_step:800`, `test_step:1213` for:
    - Batch unpacking via `_unpack_batch()`
    - Embedding extraction conditional logic (lines 814-819, 1227-1232)
    - Prediction generation for student/teacher models
    - Loss computation patterns
    - Metric accumulation logic
- **Implications**:
  - Significant refactoring opportunity for deduplication (estimated 10-15% line reduction achievable)
  - Type annotation work requires comprehensive audit across all 39 methods
  - Need to preserve PyTorch Lightning lifecycle hook signatures

### Existing Mypy Configuration Analysis
- **Context**: Verifying current static type checking infrastructure before integration
- **Sources Consulted**:
  - `mypy.ini` file analysis
  - `pyproject.toml` dev dependencies
- **Findings**:
  - **mypy.ini exists** with Python 3.11 target
  - **Permissive settings**: `disallow_untyped_defs = False` (allows untyped function definitions)
  - **Enabled checks**: `check_untyped_defs = True`, `strict_optional = True`, `warn_redundant_casts = True`
  - **Exclusions**: `exclude = (sebbs/|tests/)` - submodule and tests ignored
  - **Library ignores**: `sebbs.*` and `sed_scores_eval.*` have `ignore_missing_imports = True`
  - **Dev dependency**: `mypy>=1.19.0` already in `[dependency-groups] dev`
- **Implications**:
  - Infrastructure already in place - no new tooling installation required
  - Can incrementally tighten `disallow_untyped_defs` setting if needed
  - External library type stubs may need `type: ignore` annotations with justification

### PyTorch Lightning Type Hints Best Practices
- **Context**: Understanding framework-specific type hinting challenges and community recommendations
- **Sources Consulted**:
  - GitHub issues: Lightning-AI/pytorch-lightning#13003, #4698, #3186
  - Community forums and Stack Overflow discussions
  - PyTorch Lightning documentation (2024)
- **Findings**:
  - **Known Issues**:
    - PyTorch Lightning lacks full PEP 561 compliance (incomplete type stubs)
    - `LightningModule` attribute types sometimes conflict with assigned object types
    - PyTorch's underlying typing is "far from perfect" (affects Lightning by extension)
  - **Workarounds**:
    - Use `Protocol` for abstract base types (especially DataModules)
    - Strategic use of `type: ignore` with inline justification for framework-specific patterns
    - Focus type annotations on business logic rather than framework lifecycle hooks
  - **Best Practices**:
    - Type hint method parameters and return values where possible
    - Use `Optional[T]` for nullable types, not for default parameter values
    - Prefer `Sequence[T]` for function arguments, `List[T]` for return types
    - For Python 3.11+: Use built-in `list[int]`, `dict[str, float]` over `typing.List`, `typing.Dict`
- **Implications**:
  - Pragmatic type hinting approach required (strict typing will encounter framework limitations)
  - Document all `type: ignore` comments per Requirement 6.4
  - Focus on type safety for domain logic (loss computation, metric calculations) over Lightning internals

### Python Type Hints Evolution (3.9+ vs typing module)
- **Context**: Determining appropriate type annotation syntax for Python 3.11 codebase
- **Sources Consulted**:
  - FastAPI Python Types documentation
  - mypy official cheat sheet (1.19.0)
  - Real Python type checking guide
- **Findings**:
  - **Python 3.9+ Syntax**: Built-in types (`list[int]`, `dict[str, float]`, `tuple[int, str]`) preferred over `typing` module
  - **Tuple Specifics**:
    - Fixed-length: `tuple[float, int, str]` for 3-element tuple
    - Variable-length: `tuple[float, ...]` for homogeneous variable-length tuples
  - **Optional Semantics**: `Optional[T]` means `T | None`, NOT "parameter has default value"
  - **Type Aliases**: Recommended for complex nested types (e.g., `BatchType = tuple[torch.Tensor, torch.Tensor, list[int], list[str]]`)
  - **Dict Specifics**: `dict[KeyType, ValueType]` - both type parameters required
- **Implications**:
  - Use modern Python 3.11 native syntax (`dict`, `list`, `tuple`) throughout refactoring
  - Define type aliases at module level for complex structures (batch types, score dataframes)
  - Reserve `typing` module imports for `Optional`, `Union`, `Protocol`, `TypeAlias` only

### Code Deduplication Pattern Analysis
- **Context**: Identifying specific redundant code patterns for consolidation strategy
- **Sources Consulted**:
  - Comparative analysis of `training_step:515`, `validation_step:800`, `test_step:1213`
  - Inspection of helper method usage patterns
- **Findings**:
  - **Repeated Embedding Logic** (4 occurrences):
    ```python
    if self.hparams["pretrained"]["e2e"]:
        if self.pretrained_model.training and self.hparams["pretrained"]["freezed"]:
            self.pretrained_model.eval()
        embeddings = self.pretrained_model(embeddings)[self.hparams["net"]["embedding_type"]]
    ```
  - **Repeated Prediction Pattern** (3 occurrences per step × 2 models):
    ```python
    mels = self.mel_spec(audio)
    strong_preds_student, weak_preds_student = self.detect(mels, self.sed_student, embeddings=embeddings, ...)
    strong_preds_teacher, weak_preds_teacher = self.detect(mels, self.sed_teacher, embeddings=embeddings, ...)
    ```
  - **Shared Metric Computation**: Loss calculations follow identical patterns with different masks
  - **Batch Processing**: `_unpack_batch()` already extracted as helper (good pattern to extend)
- **Implications**:
  - Extract `_process_embeddings(embeddings)` helper method
  - Create `_compute_predictions(audio, embeddings, classes_mask)` returning both student/teacher predictions
  - Consolidate loss computation into `_compute_loss(predictions, labels, loss_type)` helper
  - Expected line reduction: 15-20% through method extraction (target: <2350 lines)

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| Monolithic Refactoring | Keep single `SEDTask4` class, extract private helper methods | Minimal API disruption, backward compatible | May still be large (target ~2300 lines) | **Selected approach** - aligns with PyTorch Lightning patterns |
| Component Decomposition | Split into `SEDTask4Base` + mixins (TrainingMixin, ValidationMixin) | Better separation of concerns | Breaking change for class structure, mixin complexity | Deferred - excessive for internal refactoring |
| Strategy Pattern | Extract step logic into separate strategy classes | High modularity, testable in isolation | Introduces indirection, non-standard for Lightning | Not aligned with framework conventions |

## Design Decisions

### Decision: Use Module-Level Type Aliases for Complex Structures

- **Context**: sed_trainer_pretrained.py handles complex nested data structures (batches, score dataframes, PSDS results) that would result in verbose inline type hints
- **Alternatives Considered**:
  1. **Inline type annotations** - Repeat full types like `dict[str, pd.DataFrame]` at each usage site
  2. **Module-level type aliases** - Define once at top of file (e.g., `ScoreDataFrameDict = dict[str, pd.DataFrame]`)
  3. **TypedDict classes** - Define structured dict types with required/optional keys
- **Selected Approach**: Module-level type aliases (Option 2)
  ```python
  # Top of sed_trainer_pretrained.py
  from typing import TypeAlias

  BatchType: TypeAlias = tuple[
      torch.Tensor,  # audio
      torch.Tensor,  # labels
      torch.Tensor,  # padded_indxs
      list[str],     # filenames
      torch.Tensor,  # embeddings
      torch.Tensor,  # valid_class_mask
  ]
  ScoreDataFrameDict: TypeAlias = dict[str, pd.DataFrame]
  PredictionTuple: TypeAlias = tuple[torch.Tensor, torch.Tensor]  # strong, weak
  ```
- **Rationale**:
  - Balances readability (clear type names) with maintainability (single source of truth)
  - Avoids boilerplate of full TypedDict definitions for simple tuple/dict structures
  - Standard practice in Python 3.11+ codebases (per FastAPI, mypy community patterns)
- **Trade-offs**:
  - **Benefits**: DRY principle, easy to update types in one location, self-documenting code
  - **Compromises**: Requires scrolling to top of file to see full type definition (mitigated by IDE "Go to Definition")
- **Follow-up**: Document all type aliases in module docstring per Requirement 6.1

### Decision: Pragmatic Type Ignore Strategy for PyTorch Lightning

- **Context**: PyTorch Lightning lacks full type stubs (PEP 561 non-compliant), causing false positives in mypy for framework lifecycle hooks
- **Alternatives Considered**:
  1. **Strict typing** - Attempt to type every Lightning interaction, fight framework limitations
  2. **Pragmatic ignores** - Use `type: ignore` selectively for framework-specific patterns with justification
  3. **No typing** - Skip type annotations on Lightning methods entirely
- **Selected Approach**: Pragmatic ignores (Option 2)
  - Type annotate method signatures fully
  - Add `# type: ignore[attr-defined]` or similar with inline comment when mypy conflicts with Lightning patterns
  - Example:
    ```python
    def training_step(
        self,
        batch: BatchType,
        batch_indx: int
    ) -> torch.Tensor:  # type: ignore[override]  # Lightning base signature uses Any
        ...
    ```
- **Rationale**:
  - Aligns with community best practices (per Lightning GitHub issues #13003, #4698)
  - Provides type safety where it matters (business logic) without framework friction
  - Satisfies Requirement 6.4 (document type: ignore with justification)
- **Trade-offs**:
  - **Benefits**: Developer productivity maintained, mypy validation passes, real type errors caught
  - **Compromises**: Not 100% type coverage (acceptable per requirements - 90%+ for private methods)
- **Follow-up**: Create inline comment template: `# type: ignore[code] - PyTorch Lightning <reason>`

### Decision: Incremental Mypy Strictness Adoption

- **Context**: Requirement 3 mandates mypy validation passing, but current codebase has `disallow_untyped_defs = False`
- **Alternatives Considered**:
  1. **Strict mode immediately** - Enable `disallow_untyped_defs = True` from start of refactoring
  2. **Incremental tightening** - Start with current settings, progressively enable stricter checks
  3. **Maintain status quo** - Keep `disallow_untyped_defs = False` indefinitely
- **Selected Approach**: Incremental tightening (Option 2)
  - **Phase 1** (current refactoring): Add all type annotations, validate with existing mypy.ini settings
  - **Phase 2** (post-refactoring): Enable `disallow_untyped_defs = True` for `local/sed_trainer_pretrained.py` via per-module override
  - **Phase 3** (future): Expand to other modules if successful
  - Configuration update in mypy.ini:
    ```ini
    [mypy-local.sed_trainer_pretrained]
    disallow_untyped_defs = True  # Phase 2 addition
    ```
- **Rationale**:
  - Reduces refactoring risk (avoid fighting mypy + refactoring simultaneously)
  - Allows validation of type annotation quality before enforcing strictness
  - Provides escape hatch if PyTorch Lightning incompatibilities emerge
- **Trade-offs**:
  - **Benefits**: Lower risk of rework, easier debugging of type issues
  - **Compromises**: Slightly longer timeline to full strict typing
- **Follow-up**: Document mypy configuration changes in Requirement 6 (documentation updates)

## Risks & Mitigations

- **Risk 1**: Type annotations break existing code due to runtime behavior differences
  - **Mitigation**: Type annotations are Python comments at runtime (no execution impact), run full test suite before/after to confirm behavioral equivalence (Requirement 5)

- **Risk 2**: Mypy validation uncovers deep type incompatibilities in third-party libraries (torch, pytorch_lightning, pandas)
  - **Mitigation**: Use `type: ignore` with justification per Requirement 3.3, leverage existing mypy.ini library exclusions

- **Risk 3**: Over-aggressive deduplication reduces code clarity or introduces performance regression
  - **Mitigation**: Limit extraction to purely duplicated logic (no speculative refactoring), validate metrics are identical per Requirement 5.3

- **Risk 4**: Line count reduction target (10%) conflicts with type annotation verbosity increase
  - **Mitigation**: Count net line change (annotations add lines, deduplication removes lines), focus on deduplication first, then annotations

- **Risk 5**: Refactored code breaks checkpoint loading from pre-refactoring experiments
  - **Mitigation**: Preserve `SEDTask4` class name, attribute names, and `__init__` signature per Requirement 4.5, test checkpoint compatibility

## References

- [PyTorch Lightning Type Hints Discussion](https://github.com/Lightning-AI/pytorch-lightning/discussions/13003) - Community patterns for typing Lightning modules
- [PyTorch Lightning Issue #4698](https://github.com/PyTorchLightning/pytorch-lightning/issues/4698) - Type annotations and mypy compatibility challenges
- [mypy Documentation - Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html) - Official mypy Python 3.11 type annotation reference
- [FastAPI Python Types Guide](https://fastapi.tiangolo.com/python-types/) - Modern Python type annotation best practices
- [Python typing Module Documentation](https://docs.python.org/3/library/typing.html) - Canonical reference for typing constructs
- Existing codebase files:
  - `mypy.ini` - Current mypy configuration baseline
  - `pyproject.toml` - Ruff lint rules, mypy dev dependency
  - `sed_trainer_pretrained.py` - Primary refactoring target
