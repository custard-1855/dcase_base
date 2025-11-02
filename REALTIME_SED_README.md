# リアルタイム音響イベント検出（SED）システム

見かけ上リアルタイムに音響イベントを検出するシステムです。10秒間録音し、その音声から音響イベントを検出して可視化します。

## 機能

- 10秒間のリアルタイム音声録音
- 音響イベント検出（DCASE Task4ベース）
- フレームレベル・クリップレベルの予測
- リアルタイム可視化
  - メル・スペクトログラム
  - フレームレベル予測のヒートマップ
  - クリップレベル予測の棒グラフ

## 検出可能なイベントクラス（27クラス）

### DESED クラス（10個）
1. Alarm_bell_ringing（アラームベル）
2. Blender（ブレンダー）
3. Cat（猫）
4. Dishes（食器）
5. Dog（犬）
6. Electric_shaver_toothbrush（電動シェーバー/歯ブラシ）
7. Frying（調理音）
8. Running_water（水音）
9. Speech（会話）
10. Vacuum_cleaner（掃除機）

### MAESTRO クラス（17個）
11. cutlery and dishes（カトラリーと食器）
12. furniture dragging（家具を引きずる音）
13. people talking（会話）
14. children voices（子供の声）
15. coffee machine（コーヒーメーカー）
16. footsteps（足音）
17. large_vehicle（大型車）
18. car（車）
19. brakes_squeaking（ブレーキ音）
20. cash register beeping（レジのビープ音）
21. announcement（アナウンス）
22. shopping cart（ショッピングカート）
23. metro leaving（地下鉄発車）
24. metro approaching（地下鉄接近）
25. door opens/closes（ドアの開閉）
26. wind_blowing（風）
27. birds_singing（鳥のさえずり）

## 必要な依存関係

### 必須パッケージ
```bash
pip install sounddevice matplotlib
```

### 既存の依存関係
プロジェクトに既にインストールされているもの：
- torch
- torchaudio
- numpy
- pyyaml
- pytorch-lightning

## インストール

1. sounddeviceとmatplotlibをインストール：
```bash
pip install sounddevice matplotlib
```

2. macOSの場合、マイクアクセスの許可が必要な場合があります

## 使用方法

### 基本的な使い方（1回のみ実行）

```bash
python realtime_sed.py
```

チェックポイントを指定する場合：
```bash
python realtime_sed.py --checkpoint path/to/model.ckpt
```

### 継続モード（複数回実行）

```bash
python realtime_sed.py --continuous
```

### 可視化結果を保存

```bash
python realtime_sed.py --save-dir results/
```

### すべてのオプション

```bash
python realtime_sed.py \
    --checkpoint path/to/model.ckpt \
    --config DESED_task/dcase2024_task4_baseline/confs/pretrained.yaml \
    --device cpu \
    --save-dir results/ \
    --continuous
```

## コマンドライン引数

- `--checkpoint`: モデルチェックポイントのパス（.ckpt）
  - 指定しない場合、ランダム初期化されたモデルを使用
- `--config`: 設定ファイルのパス（デフォルト: `DESED_task/dcase2024_task4_baseline/confs/pretrained.yaml`）
- `--device`: 使用デバイス（`cpu` または `cuda`）（デフォルト: `cpu`）
- `--save-dir`: 可視化結果の保存ディレクトリ（指定しない場合は画面表示のみ）
- `--continuous`: 継続的に検出を実行（各回ごとにユーザーに確認）

## 実行の流れ

1. システムが設定とモデルをロード
2. 「録音中...」のメッセージが表示され、10秒間録音
3. 録音完了後、音響イベントを検出
4. 検出結果のサマリーがコンソールに表示
5. 可視化ウィンドウが開き、以下を表示：
   - メル・スペクトログラム
   - フレームレベル予測（時間軸上のイベント検出）
   - クリップレベル予測（全体での検出イベント）

## 出力例

```
============================================================
リアルタイムSEDシステム
============================================================
サンプルレート: 16000 Hz
録音時間: 10 秒
クラス数: 27
============================================================

10秒間録音を開始します...
録音中...
録音完了!
イベント検出中...
検出完了!

============================================================
検出結果サマリー
============================================================

検出されたイベント (3個):
------------------------------------------------------------
  Speech                         | Score: 0.856 | Time: 1.2s - 8.5s (7.3s)
  Running_water                  | Score: 0.623 | Time: 3.4s - 6.1s (2.7s)
  Dishes                         | Score: 0.512 | Time: 5.0s - 7.8s (2.8s)
============================================================
```

## トラブルシューティング

### マイクが使えない

- macOSの場合、システム環境設定 → セキュリティとプライバシー → マイク で、Terminalまたは使用中のアプリにマイクアクセスの許可が必要です
- 利用可能なデバイスを確認：
```python
import sounddevice as sd
print(sd.query_devices())
```

### CUDAエラー

GPUを使用する場合は `--device cuda` を指定しますが、PyTorchがCUDAに対応している必要があります：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### チェックポイントが見つからない

学習済みモデルがない場合、以下のいずれかを実行：
1. モデルを学習：
```bash
cd DESED_task/dcase2024_task4_baseline
python train_pretrained.py
```

2. チェックポイントなしで実行（ランダム初期化）：
```bash
python realtime_sed.py
```
注意: ランダム初期化のモデルは正確な検出ができません。デモ・テスト目的のみ。

### matplotlibウィンドウが表示されない

バックエンドの問題の可能性があります。スクリプトの先頭を以下のように変更：
```python
import matplotlib
matplotlib.use('TkAgg')  # または 'Qt5Agg'
```

## システム構成

### ファイル構造

```
dcase_base/
├── realtime_sed.py              # メインスクリプト
├── REALTIME_SED_README.md       # このファイル
└── DESED_task/
    └── dcase2024_task4_baseline/
        ├── confs/
        │   └── pretrained.yaml  # 設定ファイル
        ├── desed_task/
        │   └── nnet/
        │       └── CRNN.py      # モデル定義
        └── local/
            └── classes_dict.py  # クラスラベル定義
```

### モデルアーキテクチャ

- CRNN（Convolutional Recurrent Neural Network）
  - 7つの畳み込みブロック
  - 双方向GRU
  - フレームレベル・クリップレベルの両方の予測

### 音声処理パラメータ

- サンプリングレート: 16,000 Hz
- 録音時間: 10秒
- メル・スペクトログラム:
  - メルビン数: 128
  - FFTサイズ: 2048
  - ホップ長: 256
  - 周波数範囲: 0-8000 Hz

## カスタマイズ

### 録音時間を変更

`realtime_sed.py` の以下の部分を編集：
```python
self.duration = self.audio_max_len  # 10秒 → 任意の秒数
```

### 検出閾値を変更

`visualize_results()` メソッド内：
```python
threshold = 0.5  # 0.0-1.0の間で調整
```

### 使用するクラスを制限

特定のクラスのみを検出したい場合、`__init__()` メソッドでクラスラベルをフィルタリング：
```python
# 例: DESEDクラスのみ使用
self.class_labels = list(classes_labels_desed.keys())
```

## 技術的な詳細

### RealtimeSEDクラス

主要なメソッド：
- `record_audio()`: sounddeviceを使用して10秒間録音
- `extract_features()`: torchaudioでメル・スペクトログラムを抽出
- `detect_events()`: CRNNモデルで音響イベントを検出
- `visualize_results()`: matplotlibで検出結果を可視化
- `print_detection_summary()`: コンソールに検出結果を表示
- `run_continuous()`: 継続的に検出を実行

### 検出結果

- **強予測（strong predictions）**: フレームレベルの予測（時間軸上のどこでイベントが発生したか）
- **弱予測（weak predictions）**: クリップレベルの予測（10秒間全体でイベントが存在したか）

## ライセンスと引用

このシステムはDCASE 2024 Task 4ベースラインに基づいています。

研究で使用する場合は、DCASE 2024 Task 4を引用してください：
```
@inproceedings{dcase2024task4,
  title={DCASE 2024 Challenge Task 4: Sound Event Detection with Weak Labels and Synthetic Soundscapes},
  author={...},
  year={2024}
}
```

## 今後の改善案

- [ ] BEATs埋め込みの事前計算機能を追加
- [ ] リアルタイムストリーミング（スライディングウィンドウ）
- [ ] cSEBBs後処理の統合
- [ ] 音声ファイルからの検出モード
- [ ] Webインターフェース
- [ ] 複数閾値での評価
- [ ] イベントのオンセット・オフセット検出の精度向上
