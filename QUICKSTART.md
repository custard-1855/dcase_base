# リアルタイムSEDシステム - クイックスタート

リアルタイム音響イベント検出システムの簡単な使い方ガイドです。

## 1. 依存関係の確認

既に`sounddevice`と`matplotlib`がインストールされています。確認：

```bash
python -c "import sounddevice; import matplotlib; print('OK')"
```

## 2. 最も簡単な使い方

### 方法A: Pythonスクリプトを直接実行

```bash
python realtime_sed.py
```

これで以下が実行されます：
1. 10秒間録音開始
2. 音響イベントを検出
3. 結果を可視化（自動で画面に表示）

### 方法B: シェルスクリプトを使用

```bash
./run_realtime_sed.sh
```

## 3. よく使うパターン

### パターン1: 結果を保存

```bash
python realtime_sed.py --save-dir results/
```

`results/detection.png`に可視化結果が保存されます。

### パターン2: 継続モード（複数回実行）

```bash
python realtime_sed.py --continuous --save-dir results/
```

各回ごとに`results/detection_001.png`, `results/detection_002.png`...と保存されます。
継続するかどうか毎回確認されます。

### パターン3: 学習済みモデルを使用

```bash
# まずモデルを学習（時間がかかります）
cd DESED_task/dcase2024_task4_baseline
python train_pretrained.py

# 学習したモデルで検出
cd ../..
python realtime_sed.py --checkpoint exps/pretrained/best.ckpt
```

**注意**: チェックポイントを指定しない場合、ランダム初期化されたモデルが使われるため、
正確な検出はできません。あくまでデモ・動作確認用です。

## 4. 実行イメージ

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

検出されたイベント (2個):
------------------------------------------------------------
  Speech                         | Score: 0.723 | Time: 2.1s - 8.3s (6.2s)
  Dishes                         | Score: 0.581 | Time: 4.5s - 7.2s (2.7s)
============================================================
```

その後、3つのグラフが表示されます：
1. メル・スペクトログラム（音声の時間-周波数表現）
2. フレームレベル予測（時間軸上のイベント検出ヒートマップ）
3. クリップレベル予測（検出されたイベントの信頼度）

## 5. トラブルシューティング

### マイクへのアクセス許可エラー

macOSの場合：
1. システム環境設定 → セキュリティとプライバシー → プライバシー
2. 「マイク」を選択
3. Terminal（またはVSCodeなど）にチェックを入れる

### 利用可能なマイクデバイスを確認

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### グラフウィンドウが表示されない

バックエンドの問題かもしれません。以下を試してください：

```bash
# macOSの場合
pip install pyqt5
```

その後、`realtime_sed.py`の`matplotlib.use('TkAgg')`を`matplotlib.use('Qt5Agg')`に変更。

## 6. 検出可能なイベント

全27クラス：

**日常生活音（DESED）:**
- Alarm_bell_ringing, Blender, Cat, Dishes, Dog
- Electric_shaver_toothbrush, Frying, Running_water, Speech, Vacuum_cleaner

**都市・公共環境音（MAESTRO）:**
- 会話系: people talking, children voices, announcement
- 家庭音: cutlery and dishes, furniture dragging, coffee machine
- 交通音: car, large_vehicle, brakes_squeaking, metro approaching/leaving
- その他: footsteps, door opens/closes, shopping cart, cash register beeping
- 自然音: wind_blowing, birds_singing

## 7. よくある質問

**Q: 録音時間を変更できますか？**

A: はい。`realtime_sed.py`の`self.duration = self.audio_max_len`の部分を編集してください。
ただし、モデルは10秒での学習を想定しているため、精度が落ちる可能性があります。

**Q: なぜチェックポイントなしでも動きますか？**

A: ランダム初期化されたモデルでも推論自体は実行できますが、正確な検出はできません。
実用的には学習済みモデルが必要です。

**Q: GPUを使えますか？**

A: はい。`--device cuda`を指定してください：
```bash
python realtime_sed.py --device cuda
```

**Q: 音声ファイルから検出できますか？**

A: 現在のバージョンではマイク録音のみです。音声ファイルから検出する場合は、
`realtime_sed.py`を修正するか、既存の`test_pretrained.py`を使用してください。

## 8. 次のステップ

- 詳細な使い方: `REALTIME_SED_README.md`を参照
- モデルの学習: `DESED_task/dcase2024_task4_baseline/`のドキュメント参照
- システムのカスタマイズ: `realtime_sed.py`のコードを編集

## 9. 簡易デモ実行（チェックポイントなし）

まずは動作確認したい場合：

```bash
# 1回だけ実行（画面表示のみ）
python realtime_sed.py

# 結果を保存
python realtime_sed.py --save-dir demo_results/
```

**注意**: 学習済みモデルがないため、検出精度は期待できません。
システムの動作確認・インターフェースのテストのみに使用してください。

## 10. 実用的な使い方（学習済みモデル使用）

```bash
# 1. モデルを学習（初回のみ、数時間〜数日かかる可能性）
cd DESED_task/dcase2024_task4_baseline
python train_pretrained.py

# 2. 学習したモデルでリアルタイム検出
cd ../..
python realtime_sed.py \
    --checkpoint exps/pretrained/best.ckpt \
    --save-dir results/ \
    --continuous
```

学習済みモデルがある場合、より正確な音響イベント検出が可能になります。
