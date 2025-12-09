# Implementation Gap Analysis: Refactor sed_trainer_pretrained.py

## Analysis Summary

- **Scope**: Refactor 2609-line `sed_trainer_pretrained.py` to eliminate redundancy, add comprehensive type hints, and integrate mypy validation
- **Key Challenge**: Balancing deduplication with PyTorch Lightning architectural constraints while preserving behavioral equivalence
- **Recommendation**: **Option C (Hybrid Approach)** - Extract reusable helpers while preserving Lightning lifecycle hooks, phased implementation with type safety first
- **Critical Findings**:
  - Significant code duplication across `training_step`, `validation_step`, `test_step` (3x embedding extraction, 3x student/teacher prediction)
  - Ground truth loading logic repeated 3 times (validation_epoch_end, test_step, on_test_epoch_end)
  - Zero type annotations currently (8 matches only from return type hints)
  - Mypy configuration exists but needs tuning for PyTorch Lightning patterns
  - No dedicated unit tests for SEDTask4 - validation relies on integration tests

---

## 1. Current State Investigation

### 1.1 Key Files and Modules

**Primary Target**:
- `DESED_task/dcase2024_task4_baseline/local/sed_trainer_pretrained.py` (2609 lines)
  - Main class: `SEDTask4(pl.LightningModule)` - PyTorch Lightning module for SED training
  - Helper classes: `SafeDataLoader`, `SafeCollate`, `_NoneSafeIterator` (batch handling)
  - Standalone functions: `select_best_auroc`, `_merge_maestro_ground_truth`, `_merge_overlapping_events`, etc.
  - 38 methods in SEDTask4 class

**Related Assets**:
- `pyproject.toml` - Ruff configuration with line-length 100, ALL rules enabled (ignoring D1, TD*, PD011)
- `mypy.ini` - Mypy configuration already exists with `check_untyped_defs = True` but `disallow_untyped_defs = False`
- `local/sebbs_wrapper/` - Recently refactored with type-safe interface (reference pattern)
- Test suite: `sebbs/tests/` (post-processing only), `local/sebbs_wrapper/tests/` (wrapper tests)

### 1.2 Architecture Patterns and Constraints

**PyTorch Lightning Lifecycle**:
- Lifecycle hooks: `on_train_start`, `training_step`, `validation_step`, `test_step`, `validation_epoch_end`, `on_test_epoch_end`, `on_train_end`, `on_test_start`
- Student-teacher architecture with EMA (exponential moving average) updates
- Multi-dataset handling (weak labels, strong synthetic labels, MAESTRO real labels)

**Dominant Patterns**:
- Teacher-student dual prediction pattern: Both models predict for each batch
- Multi-stage post-processing: Median filtering → cSEBBs → PSDS evaluation
- Buffer-based aggregation: Results accumulated in instance variables (`self.val_buffer_*`, `self.test_buffer_*`)
- Conditional execution: CMT (Confident Mean Teacher) and cSEBBs toggled via `self.hparams`

**Constraints**:
- PyTorch Lightning API compatibility: Method signatures must match Lightning protocol
- Backward compatibility: Existing YAML configs and checkpoint loading must work
- Configuration-driven behavior: Heavy reliance on `self.hparams` nested dictionaries

### 1.3 Naming and Testing Conventions

**Naming**:
- snake_case for methods and functions (consistent with project standards)
- Private methods prefixed with `_` (e.g., `_unpack_batch`, `_init_scaler`, `_maybe_wandb_log`)
- Buffer naming pattern: `{phase}_buffer_{metric}_{model_type}` (e.g., `val_buffer_sed_scores_eval_student`)

**Testing**:
- No dedicated unit tests for `sed_trainer_pretrained.py`
- Testing strategy: Integration tests via training runs with small datasets
- Existing test pattern: `sebbs_wrapper/tests/` uses pytest with type-checked stubs

### 1.4 Integration Surfaces

**Dependencies**:
- PyTorch Lightning: `pl.LightningModule` base class, `self.log()`, `self.hparams`
- External libraries without type stubs: `sed_scores_eval`, `desed_task`, `sebbs`
- Typed wrappers: `SEBBsPredictor`, `SEBBsTuner` (local/sebbs_wrapper with type hints)
- Data structures: Pandas DataFrames (scores, detections), nested dicts (ground truth, durations)

**Data Models**:
- Batch tuple: `(audio, labels, padded_indxs, filenames, embeddings, valid_class_mask)`
- Score DataFrames: Columns `["onset", "offset", *event_classes]` - sed_scores_eval format
- Ground truth: `Dict[str, List[Tuple[onset, offset, event_class]]]` structure

---

## 2. Requirements Feasibility Analysis

### 2.1 Requirement-to-Asset Map

| Requirement | Current Asset | Gap Status | Notes |
|------------|---------------|------------|-------|
| **Req 1: Code Deduplication** | 2609-line monolith with repetition | **Missing** | No extraction yet |
| R1.1: Consolidate duplicate blocks | Embedding extraction 3x (lines 564, 814, 1227) | **Missing** | `_extract_embeddings_if_needed()` helper needed |
| R1.2: Extract common validation/test logic | Student/teacher prediction 6x total | **Missing** | `_predict_student_teacher()` helper needed |
| R1.3: Refactor repeated data processing | Ground truth loading 3x (lines 964, 1257, 1772) | **Missing** | `_load_ground_truth()` helper needed |
| R1.4: Maintain behavioral equivalence | Integration tests exist (training pipeline) | **Partial** | No automated regression suite |
| R1.5: Reduce line count by 10%+ | Current: 2609 lines | **N/A** | Target: <2348 lines after deduplication |
| **Req 2: Type Hint Coverage** | ~8 type annotations (only `-> None` hints) | **Missing** | 0% coverage currently |
| R2.1: Type all method signatures | 38 methods in SEDTask4, 0 typed | **Missing** | Systematic typing needed |
| R2.2: Specific type hints for complex structures | Dicts/DataFrames untyped | **Missing** | `TypeAlias` definitions needed |
| R2.3: Optional parameter annotations | Many optional params (e.g., `embeddings=None`) | **Missing** | `Optional[torch.Tensor]` needed |
| R2.4: Use typing module constructs | No imports from `typing` | **Missing** | Add `Dict`, `List`, `Optional`, `Tuple`, `Any` |
| R2.5: Type annotations for class attributes | Buffers/metrics untyped | **Missing** | `__init__` and class-level annotations needed |
| R2.6: 100% public, 90%+ private coverage | 0% current | **Missing** | Comprehensive typing required |
| **Req 3: Mypy Validation Integration** | `mypy.ini` exists with `check_untyped_defs=True` | **Partial** | Config present but needs tuning |
| R3.1: Pass mypy validation | Not currently enforced | **Missing** | CI integration needed |
| R3.2: Document exceptions for strict mode | PyTorch Lightning incompatibilities | **Unknown** | Research needed on Lightning type issues |
| R3.3: Resolve third-party type issues | `sed_scores_eval`, `desed_task` untyped | **Constraint** | Will require `type: ignore` with justification |
| R3.4: Explicit type aliases for ambiguous types | Batch structures, score DataFrames | **Missing** | Module-level aliases needed |
| R3.5: Mypy in pyproject.toml | Currently in `mypy.ini` | **Partial** | Migration to pyproject.toml optional |
| R3.6: Ruff compatibility | Ruff configured with ALL rules | **OK** | No conflicts expected |
| **Req 4: Code Structure Preservation** | SEDTask4 class API stable | **OK** | Public API well-defined |
| R4.1: Maintain public API | Method names/signatures stable | **OK** | Refactoring is internal only |
| R4.2: YAML config compatibility | Configs reference class/module paths | **OK** | No config changes needed |
| R4.3: Import compatibility | Exports stable | **OK** | Same symbols exported |
| R4.4: Lightning lifecycle hooks identical | 7 lifecycle methods | **OK** | Signatures fixed by Lightning |
| R4.5: Checkpoint loading compatibility | Checkpoint format unchanged | **OK** | State dict keys unchanged |
| **Req 5: Testing and Validation** | Integration tests (training runs) | **Partial** | No automated regression tests |
| R5.1: Existing test suites pass | No dedicated unit tests for this file | **Gap** | Integration tests available |
| R5.2: Unit tests (if missing) | No unit tests currently | **Missing** | Integration tests sufficient per requirements |
| R5.3: Identical validation metrics | Numerical precision check | **Missing** | Pre/post refactor comparison needed |
| R5.4: Pass Ruff lint checks | Ruff configured | **OK** | Automated via pre-commit |
| R5.5: Pass mypy validation | Not enforced yet | **Missing** | As per Requirement 3 |
| **Req 6: Documentation Updates** | Minimal docstrings (Google style) | **Partial** | Some docstrings exist |
| R6.1: Type alias docstrings | No aliases defined yet | **Missing** | Will create with deduplication |
| R6.2: Method docstrings (Google/NumPy) | Partial coverage (lifecycle methods only) | **Partial** | Extend to new helpers |
| R6.3: Class-level docstring updates | SEDTask4 docstring exists but outdated | **Partial** | Update if structure changes |
| R6.4: Justify `type: ignore` with comments | No usage yet | **N/A** | Add inline rationale when needed |
| R6.5: Consistency with DCASE patterns | Existing docstrings follow Google style | **OK** | Maintain current style |

### 2.2 Key Gaps and Constraints

**Gaps**:
1. **Zero type coverage**: No type annotations across 2609 lines - complete typing needed
2. **Substantial duplication**: Embedding extraction (3x), student/teacher prediction (6x), ground truth loading (3x), PSDS computation (7x)
3. **No regression testing framework**: Integration tests exist but no automated pre/post comparison
4. **Third-party type stubs missing**: `sed_scores_eval`, `desed_task` libraries untyped - will require `type: ignore` pragmas

**Constraints**:
1. **PyTorch Lightning API**: Lifecycle hook signatures fixed (cannot change `training_step`, etc.)
2. **Configuration coupling**: Heavy use of `self.hparams` nested dicts - hard to type without structural changes
3. **Backward compatibility**: Checkpoint loading and YAML configs must remain valid
4. **Numerical equivalence requirement**: Output metrics must match exactly (within float precision)

**Research Needed**:
- PyTorch Lightning typing best practices for `LightningModule` subclasses
- Type annotations for `self.hparams` (DictConfig from OmegaConf or plain dict?)
- Mypy strict mode compatibility with PyTorch tensors and Lightning callbacks
- Type stub availability for `sed_scores_eval` library (may need custom stubs)

### 2.3 Complexity Signals

**Complexity Analysis**:
- **Algorithmic**: Moderate - mostly data processing pipelines, no complex algorithms
- **Integration**: High - 7 lifecycle hooks, multi-dataset handling, dual model predictions
- **Data Flow**: Complex - buffers accumulate across batches, multi-stage post-processing
- **External Dependencies**: High - 4 untyped libraries (`sed_scores_eval`, `desed_task`, `sebbs`, `pytorch_lightning`)

**Refactoring Risks**:
- Breaking checkpoint compatibility during extraction
- Subtle behavioral changes in aggregation logic (buffer updates)
- Type incompatibilities with PyTorch Lightning's dynamic attribute handling (`self.hparams.update()`)

---

## 3. Implementation Approach Options

### Option A: Extend Existing Components

**Approach**: Add type hints and deduplicate incrementally within the existing `SEDTask4` class structure.

**Which files/modules to extend**:
- `sed_trainer_pretrained.py`: Add type annotations to all 38 methods in-place
- Extract 4-6 private helper methods within `SEDTask4` class (e.g., `_extract_embeddings_if_needed`, `_predict_both_models`)

**Compatibility assessment**:
- ✅ Minimal API surface changes - only internal refactoring
- ✅ Lightning lifecycle methods unchanged (only internal calls modified)
- ✅ Checkpoint loading unaffected (state dict keys identical)
- ⚠️ Risk of bloating `SEDTask4` class further (currently 38 methods → ~42 methods)

**Complexity and maintainability**:
- Cognitive load: Moderate - fewer moving parts, but class remains large
- Single responsibility: ❌ Violated - `SEDTask4` handles training, validation, testing, metrics, post-processing
- File size: ⚠️ Still ~2300+ lines after 10% reduction (above ideal <1500 lines)

**Trade-offs**:
- ✅ Minimal new files - faster initial development
- ✅ Leverages existing patterns and infrastructure
- ✅ Lower risk of integration issues
- ❌ Risk of bloating existing components (class already has 38 methods)
- ❌ May complicate existing logic (more private methods to navigate)
- ❌ Doesn't address architectural concerns (God class pattern)

**Recommended if**: Time-constrained, minimal disruption prioritized, no follow-up refactoring planned.

---

### Option B: Create New Components

**Approach**: Extract substantial functionality into separate modules (e.g., `sed_types.py`, `sed_metrics.py`, `sed_postprocessing.py`) and refactor `SEDTask4` to orchestrate them.

**Rationale for new creation**:
- Clear separation of concerns: Type definitions, metrics computation, post-processing are distinct responsibilities
- Existing `SEDTask4` already complex (2609 lines, 38 methods) - extraction improves cohesion
- Opportunity to follow `sebbs_wrapper/` pattern (clean type-safe interfaces)

**Integration points**:
- `sed_types.py`: Type aliases for batch structures, score DataFrames, ground truth dicts
- `sed_metrics.py`: Functions for PSDS/F1 computation, ground truth loading, score aggregation
- `sed_postprocessing.py`: Median filtering, cSEBBs prediction wrappers
- `SEDTask4` imports and orchestrates these modules in lifecycle hooks

**Responsibility boundaries**:
- **SEDTask4**: Lightning lifecycle, training logic, optimizer/scheduler management
- **sed_metrics**: Pure functions for metrics (no state, easily testable)
- **sed_postprocessing**: Stateless post-processing pipelines
- **sed_types**: Type definitions only (no logic)

**Trade-offs**:
- ✅ Clean separation of concerns (type system, metrics, post-processing decoupled)
- ✅ Easier to test in isolation (pure functions in separate modules)
- ✅ Reduces complexity in existing components (SEDTask4 → orchestrator role)
- ✅ Aligns with `sebbs_wrapper/` refactoring pattern
- ❌ More files to navigate (+3-4 new modules)
- ❌ Requires careful interface design (function signatures, import organization)
- ❌ Higher upfront cost (module scaffolding, import refactoring)

**Recommended if**: Long-term maintainability prioritized, team willing to invest in architectural cleanup, follow-up feature development planned.

---

### Option C: Hybrid Approach (RECOMMENDED)

**Combination strategy**:
1. **Phase 1: Type Safety Foundation** (Low-risk, high-value)
   - Create `sed_types.py` for type aliases (batch structures, DataFrames, dicts)
   - Add type annotations to all methods in `sed_trainer_pretrained.py` using these aliases
   - Configure mypy and validate (add `type: ignore` where necessary for third-party libraries)

2. **Phase 2: Internal Deduplication** (Moderate-risk, medium-value)
   - Extract 5-7 private helper methods within `SEDTask4` class:
     - `_extract_embeddings_if_needed(audio, embeddings)` - dedup 3x pattern
     - `_predict_student_teacher(mels, embeddings, **kwargs)` - dedup 6x pattern
     - `_load_desed_ground_truth()` - dedup 3x pattern
     - `_compute_psds_for_dataset(scores, ground_truth, durations, **kwargs)` - dedup 7x pattern
     - `_update_scores_buffer(buffer, scores, model_type)` - reduce buffer update boilerplate
   - Keep helpers in same file (no new modules yet) to minimize disruption

3. **Phase 3: Selective Extraction** (Optional, higher-risk)
   - If Phase 2 doesn't achieve <2000 lines, extract `sed_metrics.py` for pure metric functions
   - Move `select_best_auroc`, `_merge_maestro_ground_truth`, `_merge_overlapping_events` to new module
   - Extract ground truth loading and PSDS computation to `sed_metrics.py`

**Phased implementation rationale**:
- Phase 1 provides immediate mypy value with minimal behavioral risk
- Phase 2 achieves 10%+ line reduction (estimated 2609 → ~2100-2200 lines) via targeted helpers
- Phase 3 only if needed - allows validation of Phases 1-2 before larger restructuring

**Risk mitigation**:
- Incremental rollout: Each phase independently testable via training runs
- Checkpoint compatibility: State dict unchanged (only method internals refactored)
- Numerical validation: Compare metrics pre/post each phase (automated script)
- Rollback strategy: Git branching per phase, tagged commits for safe reversion

**Trade-offs**:
- ✅ Balanced approach for complex features (type safety + deduplication + optional extraction)
- ✅ Allows iterative refinement (stop at Phase 2 if sufficient)
- ✅ Reduces risk via incremental validation
- ❌ More complex planning required (3-phase roadmap)
- ❌ Potential for inconsistency if phases not well-coordinated (mitigated by clear interfaces)

**Recommended because**:
- Addresses both type safety (Req 2-3) and deduplication (Req 1) systematically
- Minimizes risk via phased validation (Req 5.3 numerical equivalence)
- Provides exit point after Phase 2 if 10% reduction achieved without extraction
- Aligns with project constraints (backward compatibility, existing patterns)

---

## 4. Implementation Complexity & Risk

### Effort Estimation

**Size: L (1-2 weeks)**
- **Phase 1** (Type annotations): 3-4 days
  - Create `sed_types.py` with 10-15 type aliases (4 hours)
  - Annotate 38 methods in `SEDTask4` (2 days - systematic but tedious)
  - Configure mypy, resolve errors (1 day - expect 20-30 initial errors)
  - Testing: Run training pipeline, verify no runtime issues (4 hours)

- **Phase 2** (Internal deduplication): 3-4 days
  - Extract 5-7 private helper methods (2 days - careful behavioral preservation)
  - Update call sites (50+ locations across lifecycle hooks) (1 day)
  - Testing: Numerical validation pre/post (1 day - compare PSDS, F1 metrics)

- **Phase 3** (Optional extraction): 3-4 days (if needed)
  - Create `sed_metrics.py`, move 8-10 functions (1 day)
  - Update imports, refactor call sites (1 day)
  - Testing and validation (1-2 days)

**Justification**:
- Existing patterns established (Ruff, mypy config, sebbs_wrapper typing)
- Moderate complexity (data processing, no algorithmic challenges)
- Validation overhead (numerical equivalence testing after each phase)

### Risk Assessment

**Risk: Medium**

**Rationale**:
- ✅ **Mitigating factors**:
  - Mypy and Ruff infrastructure already configured
  - `sebbs_wrapper/` provides type-safe refactoring template
  - Integration tests exist for validation (training pipeline)
  - No architectural changes required (Lightning API unchanged)

- ⚠️ **Risk factors**:
  - Third-party libraries untyped (`sed_scores_eval`, `desed_task`) - requires `type: ignore` pragmas
  - PyTorch Lightning dynamic attribute handling (`self.hparams.update()`) may resist typing
  - Extensive duplication (50+ call sites to update in Phase 2) - manual error risk
  - No automated regression suite - numerical validation requires manual runs

**Specific Risks**:
1. **Type compatibility with Lightning** (Medium risk)
   - `self.hparams` is dynamically updated - may need `Any` or protocol typing
   - Lightning's `log()` and `log_dict()` have complex signatures - may need overrides
   - Mitigation: Research Lightning typing patterns, use `type: ignore` selectively with justification

2. **Behavioral regressions during deduplication** (Medium risk)
   - Subtle differences in helper extraction (e.g., missing `classes_mask` kwarg in `detect()`)
   - Buffer update order changes affecting metrics
   - Mitigation: Systematic pre/post numerical comparison, unit tests for helpers

3. **Mypy strict mode incompatibility** (Low risk)
   - Current config has `disallow_untyped_defs = False` - gradual typing safe
   - Mitigation: Keep gradual typing mode, don't enforce strict until all third-party issues resolved

---

## 5. Recommendations for Design Phase

### Preferred Approach
**Option C (Hybrid Approach)** with 3-phase implementation:
1. Type safety foundation (`sed_types.py` + annotations)
2. Internal deduplication (private helpers within `SEDTask4`)
3. Optional extraction (only if <2000 lines not achieved)

### Key Decisions for Design Phase

1. **Type Alias Definitions** (Design Phase Task)
   - Define module-level type aliases in `sed_types.py`:
     ```python
     BatchTuple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], torch.Tensor, torch.Tensor]
     ScoresDict = Dict[str, pd.DataFrame]
     GroundTruthDict = Dict[str, List[Tuple[float, float, str]]]
     AudioDurationsDict = Dict[str, float]
     ```
   - Determine if `self.hparams` should be typed as `DictConfig` (OmegaConf) or `Dict[str, Any]`

2. **Helper Method Signatures** (Design Phase Task)
   - Specify exact signatures for 5-7 private helpers
   - Determine parameter order (follow existing `detect()`, `_unpack_batch()` patterns)
   - Decide on return types (single value vs. tuple unpacking)

3. **Mypy Configuration Tuning** (Design Phase Task)
   - Evaluate if `disallow_untyped_defs = True` achievable or keep gradual typing
   - Document exceptions for third-party libraries in `mypy.ini` with rationale comments
   - Decide if migrating `mypy.ini` to `[tool.mypy]` in `pyproject.toml` worth effort

4. **Numerical Validation Strategy** (Design Phase Task)
   - Design automated comparison script:
     - Run training for N steps on small dataset (baseline)
     - Checkpoint at step N
     - Refactor code
     - Resume from checkpoint, compare metrics (PSDS, F1, loss) at step N+M
   - Define acceptable tolerance (e.g., 1e-6 for float32 precision)

5. **Documentation Standards** (Design Phase Task)
   - Confirm Google-style docstrings for all new helpers
   - Template for `type: ignore` justification comments:
     ```python
     # type: ignore[import] - sed_scores_eval lacks type stubs
     ```

### Research Items to Carry Forward

1. **PyTorch Lightning Typing Patterns** (2-3 hours)
   - Investigate official Lightning typing examples for `LightningModule` subclasses
   - Research `self.hparams` type annotation best practices (DictConfig vs. custom class)
   - Check if Lightning provides type stubs or protocols

2. **Third-party Library Type Stubs** (1-2 hours)
   - Verify if `sed-scores-eval` has type stubs on PyPI (`types-sed-scores-eval`)
   - Check `desed_task` library typing status
   - Determine if custom stub files needed (`.pyi` files in `typings/`)

3. **Mypy Strict Mode Feasibility** (1 hour)
   - Test mypy with `disallow_untyped_defs = True` on current codebase
   - Count errors and categorize (fixable vs. third-party vs. architectural)
   - Decide if strict mode is goal or gradual typing acceptable

4. **Deduplication ROI Analysis** (30 minutes)
   - Analyze exact line counts per duplicate pattern:
     - Embedding extraction (3x ~15 lines = 30 lines saved)
     - Student/teacher prediction (6x ~20 lines = 100 lines saved)
     - Ground truth loading (3x ~25 lines = 50 lines saved)
     - PSDS computation (7x ~30 lines = 180 lines saved)
   - Estimate total reduction: ~360 lines (14% reduction) achievable via helpers alone
   - Confirm Phase 3 (extraction) not needed if helpers sufficient

---

## Next Steps

### If Requirements Approved
Proceed to design phase with:
```
/kiro:spec-design refactor-sed-trainer
```

This will create a comprehensive technical design document addressing:
- Type alias definitions in `sed_types.py`
- Helper method specifications (signatures, docstrings)
- Mypy configuration updates
- Numerical validation strategy
- 3-phase implementation plan with task breakdown

### If Requirements Need Revision
Based on gap analysis findings:
- **Consider scaling back type coverage**: If 100% coverage too ambitious, target 90% (exclude complex Lightning internals)
- **Clarify numerical equivalence tolerance**: Define acceptable float precision delta (1e-6?)
- **Decide on mypy strict mode**: Confirm if gradual typing acceptable or strict mode required

---

**Warning: Requirements Not Yet Approved**

The requirements for this feature have been generated but not approved. Gap analysis can inform requirement revisions if needed. Review `.kiro/specs/refactor-sed-trainer/requirements.md` and approve before proceeding to design.

---

_Generated: 2025-12-08 via `/kiro:validate-gap refactor-sed-trainer`_
_Framework: gap-analysis.md (comprehensive investigation mode)_
_Language: English (per spec.json)_
