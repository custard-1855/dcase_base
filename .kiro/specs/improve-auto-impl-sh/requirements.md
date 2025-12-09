# Requirements Document

## Project Description (Input)
auto_impl.shスクリプトの包括的改善: ドキュメント修正、コミットプロセスの明確化と分離、複数スペック対応（スペック選択機能）、エラーハンドリング強化、設定の柔軟性向上、ログ機能の追加

## Introduction

`auto_impl.sh` は、Kiro仕様駆動開発において、`.kiro/specs/` 内の仕様に対して未完了タスクを自動実行する支援スクリプトである。本要求仕様では、現行スクリプトに対して以下の改善を実施する:

1. **ドキュメントの正確性向上**: スクリプトヘッダーコメントと実装の乖離を解消
2. **コミットプロセスの明確化**: テスト・Lint・型チェック確認後のコミットフローを改善
3. **複数スペック対応**: スペック選択機能の追加（対話的選択または引数指定）
4. **エラーハンドリング強化**: 失敗時のロールバック・リトライ機能の追加
5. **設定の柔軟性向上**: 環境変数・設定ファイルによるカスタマイズ可能化
6. **ログ機能の追加**: 実行履歴の記録と診断情報の出力

これにより、開発者は複数の仕様を効率的に管理し、エラー発生時の対応を容易にし、スクリプトの動作をプロジェクト要件に合わせてカスタマイズできるようになる。

## Requirements

### Requirement 1: ドキュメント正確性の向上

**Objective**: 開発者として、スクリプトヘッダーコメントと実際の実装が一致していることを確認したい。これにより、スクリプトの動作を正確に理解し、誤った前提に基づく使用を防ぐことができる。

#### Acceptance Criteria

1. When スクリプトヘッダーコメントが実装と乖離している場合、the auto_impl.sh shall ヘッダーコメントを実装の実際の動作に合わせて更新する
2. The auto_impl.sh shall 前提条件セクションに、現在の実装が要求する正確な条件（spec.jsonの必須フィールド、tasks.mdの存在、ready_for_implementationフラグ）を記載する
3. The auto_impl.sh shall 使い方セクションに、複数スペック対応後の正確な使用方法（引数オプション、対話的選択）を記載する
4. The auto_impl.sh shall 機能説明セクションに、コミットプロセスの詳細（テスト・Lint・型チェック確認フロー）を明記する

### Requirement 2: コミットプロセスの明確化と分離

**Objective**: 開発者として、タスク実装とコミットのプロセスを明確に分離し、テスト・Lint・型チェックの確認フローを透明化したい。これにより、品質保証の各ステップを理解し、必要に応じて介入できるようにする。

#### Acceptance Criteria

1. When `/impl` コマンドが完了した後、the auto_impl.sh shall テスト実行・Lint・型チェックの各検証ステップを個別に実行する
2. If テスト実行でエラーが検出された場合、then the auto_impl.sh shall エラー内容をログに記録し、コミットを中止し、修正する
3. If Lintチェックでエラーが検出された場合、then the auto_impl.sh shall エラー内容をログに記録し、コミットを中止し、修正する
4. If 型チェックでエラーが検出された場合、then the auto_impl.sh shall エラー内容をログに記録し、コミットを中止し、修正する
5. When すべての検証ステップが成功した場合、the auto_impl.sh shall コミットを実行し、コミットメッセージに検証結果の概要を含める
6. The auto_impl.sh shall 各検証ステップの実行コマンド（pytest、ruff、mypyなど）を設定ファイルまたは環境変数で指定可能にする

### Requirement 3: 複数スペック対応（スペック選択機能）

**Objective**: 開発者として、`.kiro/specs/` 内に複数の仕様が存在する場合に、実行対象のスペックを選択できるようにしたい。これにより、複数の機能を並行開発する際の効率を向上させる。

#### Acceptance Criteria

1. When `.kiro/specs/` 内に複数のスペックディレクトリ（archivesを除く）が存在する場合、the auto_impl.sh shall 対話的にスペック選択プロンプトを表示する
2. When コマンドライン引数でスペック名が指定された場合（例: `auto_impl.sh feature-name`）、the auto_impl.sh shall 対話的プロンプトをスキップし、指定されたスペックを使用する
3. If 指定されたスペック名が存在しない場合、then the auto_impl.sh shall エラーメッセージを表示し、利用可能なスペック一覧を表示して終了する
4. When `.kiro/specs/` 内にスペックが1つだけ存在する場合、the auto_impl.sh shall 自動的にそのスペックを選択し、確認プロンプトを表示しない
5. When `.kiro/specs/` 内にarchives以外のスペックが存在しない場合、then the auto_impl.sh shall エラーメッセージを表示して終了する
6. The auto_impl.sh shall 対話的選択時に、各スペックの`spec.json`から取得した基本情報（phase、ready_for_implementation、未完了タスク数）を表示する

### Requirement 4: エラーハンドリングの強化

**Objective**: 開発者として、スクリプト実行中のエラーに対して適切なハンドリングとリカバリー機能を提供したい。これにより、部分的な失敗からの復旧を容易にし、作業の中断を最小限に抑える。

#### Acceptance Criteria

1. When `/impl` コマンドが失敗した場合、the auto_impl.sh shall 失敗の詳細をログに記録し、ユーザーに（1）リトライ、（2）スキップ、（3）中止の選択肢を提示する
2. When コミットプロセスが失敗した場合、the auto_impl.sh shall 失敗の詳細をログに記録し、ユーザーに（1）手動修正後リトライ、（2）スキップ、（3）中止の選択肢を提示する
3. If ユーザーが「リトライ」を選択した場合、the auto_impl.sh shall 同じタスクを再実行する
4. If ユーザーが「スキップ」を選択した場合、the auto_impl.sh shall 現在のタスクをスキップし、次のタスクに進む（スキップしたタスクをログに記録）
5. If ユーザーが「中止」を選択した場合、the auto_impl.sh shall 実行を停止し、現在の状態をログに記録して終了する
6. The auto_impl.sh shall API制限エラー（rate limit）を検出した場合、自動的に待機時間を調整し、リトライする
7. The auto_impl.sh shall スクリプト実行中に予期しないエラーが発生した場合、エラー内容とスタックトレースをログに記録し、安全に終了する

### Requirement 5: 設定の柔軟性向上

**Objective**: 開発者として、プロジェクトごとに異なる要件に合わせてスクリプトの動作をカスタマイズできるようにしたい。これにより、多様なプロジェクト構成に対応し、再利用性を高める。

#### Acceptance Criteria

1. The auto_impl.sh shall 設定ファイル（`.kiro/auto_impl.config`）から以下の設定を読み込む: MAX_ITERATIONS、SLEEP_DURATION、CLAUDE_COMMAND、TEST_COMMAND、LINT_COMMAND、TYPECHECK_COMMAND
2. The auto_impl.sh shall 環境変数（`KIRO_AUTO_IMPL_*`形式）から設定をオーバーライド可能にする
3. The auto_impl.sh shall 設定ファイルが存在しない場合、デフォルト値を使用し、警告メッセージを表示する
4. The auto_impl.sh shall コマンドライン引数（`--max-iterations`、`--sleep`など）から設定をオーバーライド可能にする
5. The auto_impl.sh shall 設定の優先順位を「コマンドライン引数 > 環境変数 > 設定ファイル > デフォルト値」とする
6. The auto_impl.sh shall `--dry-run` オプションを提供し、実際の実行を行わずに実行計画を表示する

### Requirement 6: ログ機能の追加

**Objective**: 開発者として、スクリプトの実行履歴と診断情報を記録したい。これにより、問題発生時のトラブルシューティングを容易にし、実行結果の分析を可能にする。

#### Acceptance Criteria

1. The auto_impl.sh shall 実行ごとにタイムスタンプ付きログファイル（`.kiro/logs/auto_impl_YYYYMMDD_HHMMSS.log`）を生成する
2. When スクリプトが開始される際、the auto_impl.sh shall ログに実行開始時刻、選択されたスペック名、初期タスク数を記録する
3. When 各イテレーションが実行される際、the auto_impl.sh shall ログにイテレーション番号、残タスク数、`/impl`コマンドの出力、検証ステップの結果を記録する
4. When エラーが発生した際、the auto_impl.sh shall ログにエラー種別、エラーメッセージ、発生タイミング、スタックトレース（該当する場合）を記録する
5. When スクリプトが完了する際、the auto_impl.sh shall ログに終了時刻、総イテレーション数、完了タスク数、スキップタスク数、実行時間を記録する
6. The auto_impl.sh shall `--log-level` オプション（DEBUG、INFO、WARNING、ERROR）を提供し、ログの詳細度を制御可能にする
7. The auto_impl.sh shall ログファイルの保持期間（デフォルト30日）を設定可能にし、古いログファイルを自動削除する

