# Requirements Document

## Project Description (Input)
Refactor experiment directory structure to use custom hierarchical layout instead of wandb's default run-{timestamp}-{id} format. Consolidate all experiment artifacts (checkpoints, metrics, inference results, visualizations) under meaningful experiment names like experiments/{category}/{method}/{variant}/. Replace scattered wandb/exp/inference_outputs directories with unified experiment-centric structure.

## 導入

本仕様は、DCASE 2024 Task 4音響イベント検出システムにおける実験ディレクトリ構造のリファクタリングを定義する。現在のwandbデフォルト形式（`run-{timestamp}-{id}`）から、意味のある階層的レイアウト（`experiments/{category}/{method}/{variant}/`）への移行により、実験成果物の管理性と追跡性を向上させる。

## Requirements

### Requirement 1: 階層的実験ディレクトリ構造

**Objective:** 研究者として、実験を論理的なカテゴリと手法で整理したい。これにより、実験の比較と管理が容易になる。

#### Acceptance Criteria

1. The experiment directory system shall create hierarchical directory structure following the pattern `experiments/{category}/{method}/{variant}/`
2. When a new experiment is initiated, the experiment directory system shall automatically create parent directories if they do not exist
3. The experiment directory system shall support at least 3 levels of hierarchy (category, method, variant)
4. The experiment directory system shall validate directory names to prevent filesystem incompatibilities (invalid characters, path length limits)
5. When multiple experiments share the same hierarchy path, the experiment directory system shall append unique identifiers (timestamp or counter) to prevent conflicts

### Requirement 2: 実験成果物の統合管理

**Objective:** 研究者として、すべての実験成果物（チェックポイント、メトリクス、推論結果、可視化）を一箇所で管理したい。これにより、実験の再現性と結果の追跡が向上する。

#### Acceptance Criteria

1. The artifact management system shall store checkpoints under `{experiment_dir}/checkpoints/`
2. The artifact management system shall store training metrics and logs under `{experiment_dir}/metrics/`
3. The artifact management system shall store inference results under `{experiment_dir}/inference/`
4. The artifact management system shall store visualizations (UMAP, reliability diagrams) under `{experiment_dir}/visualizations/`
5. The artifact management system shall store experiment configuration snapshots under `{experiment_dir}/config/`
6. When an experiment completes, the artifact management system shall generate a manifest file listing all artifacts with metadata (timestamp, file size, content type)

**注記**: Grad-CAM可視化スクリプトは本リファクタリングの対象外とし、変更後の構造を踏まえて別途実装する。既存の可視化スクリプト（UMAP等）は新しい構造に合わせて調整される可能性がある。

### Requirement 3: wandb統合とパス解決

**Objective:** 開発者として、wandbの実験トラッキング機能を維持しながら、カスタムディレクトリ構造を使用したい。これにより、既存のワークフローが中断されない。

#### Acceptance Criteria

1. The path resolution system shall override wandb's default run directory with custom experiment path
2. When wandb logger is initialized, the path resolution system shall inject custom `dir` parameter pointing to experiment directory
3. The path resolution system shall maintain symlink or metadata mapping between wandb run ID and experiment directory path
4. The path resolution system shall provide helper functions to resolve paths (e.g., `get_checkpoint_dir()`, `get_inference_dir()`)
5. When experiment directory is queried by run ID, the path resolution system shall return corresponding experiment path within 100ms

### Requirement 4: 設定駆動の実験命名

**Objective:** 研究者として、実験名と階層構造をYAML設定ファイルで定義したい。これにより、コードを変更せずに実験構成を管理できる。

#### Acceptance Criteria

1. The configuration system shall support experiment naming parameters in YAML config (category, method, variant)
2. The configuration system shall support templating in experiment names (e.g., `{method}_{variant}_{timestamp}`)
3. When configuration is loaded, the configuration system shall validate required naming parameters are present
4. If naming parameters are missing, then the configuration system shall fall back to default values (category="default", method="baseline", variant="v1")
5. The configuration system shall allow environment variable substitution in experiment paths (e.g., `$SCRATCH_DIR/experiments`)

### Requirement 5: 実行モード別のwandBログ管理

**Objective:** 研究者として、training実行とtest-only実行、feature-extraction実行を明確に区別したい。これにより、不要なwandBログファイルの生成を防ぎ、実験ログの整理が容易になる。

**Background**: 現状では、`use_wandb=True` の場合、training、test-only、feature-extraction（推論・可視化）のすべてで新しいwandB runが作成される。これにより、実験目的と無関係なログが大量に生成され、実験の追跡が困難になっている。

#### Acceptance Criteria

1. The experiment mode management system shall recognize four distinct execution modes: `train`, `test`, `inference`, and `feature_extraction`
2. When execution mode is `train`, the system shall create a new wandb run and directory under `experiments/train/{category}/{method}/{variant}/`
3. When execution mode is `test`, the system shall either reuse the training run directory or create a minimal test-only directory under `experiments/test/{category}/{method}/{variant}/`, based on configuration
4. When execution mode is `inference` or `feature_extraction`, the system shall disable wandb run creation and store artifacts under `experiments/inference/{category}/{method}/{variant}/` without wandb synchronization
5. The manifest file shall record the execution mode (e.g., `"mode": "train"`) to distinguish experiment purposes
6. When wandb is disabled for inference/feature_extraction modes, the system shall log a clear message (e.g., "WandB disabled for feature extraction mode")
7. The configuration system shall support explicit mode specification in YAML config or via command-line argument (e.g., `--mode inference`)
8. If execution mode is not explicitly specified, then the system shall infer mode from existing indicators: presence of `test_state_dict` (test mode), `evaluation=True` parameter (inference mode), or default to train mode
9. When a test-only or inference run references a training experiment, the system shall provide a reference link in the manifest (e.g., `"parent_run_id": "..."`) for traceability