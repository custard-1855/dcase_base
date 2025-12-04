# データセット固定化レポート

## 設定

- サンプル数/クラス: 150
- ランダムシード: 42
- 生成日時: 2025-12-04 19:17:52
- 設定ファイル: confs/pretrained.yaml
- 評価対象クラス: 21 (DESED: 10, MAESTRO: 11)

## 処理ログ

### DESED Validation (synth_val)

**サンプル数のカウント結果:**

- Total: 2500
- Valid: 2500
- Skipped: 0

**クラス別サンプリング:**

| Class | Available | Target | Requested | Status |
|-------|-----------|--------|-----------|--------|
| Dog | 341 | 150 | 150 | ✓ OK |
| Speech | 2373 | 150 | 150 | ✓ OK |
| Dishes | 689 | 150 | 150 | ✓ OK |
| Frying | 377 | 150 | 150 | ✓ OK |
| Running_water | 306 | 150 | 150 | ✓ OK |
| Blender | 266 | 150 | 150 | ✓ OK |
| Cat | 284 | 150 | 150 | ✓ OK |
| Electric_shaver_toothbrush | 283 | 150 | 150 | ✓ OK |
| Alarm_bell_ringing | 400 | 150 | 150 | ✓ OK |
| Vacuum_cleaner | 251 | 150 | 150 | ✓ OK |

**選択されたサンプル数:** 1328

### DESED Unlabeled

**サンプル数のカウント結果:**

- Total: 12027
- Valid: 12027
- Skipped: 0

**選択されたサンプル数:** 1500

### MAESTRO Training

**サンプル数のカウント結果:**

- Total: 6426
- Valid: 6426
- Skipped: 0

**クラス別サンプリング:**

| Class | Available | Target | Requested | Status |
|-------|-----------|--------|-----------|--------|
| cutlery and dishes | 1428 | 150 | 150 | ✓ OK |
| people talking | 6425 | 150 | 150 | ✓ OK |
| children voices | 6385 | 150 | 150 | ✓ OK |
| footsteps | 6426 | 150 | 150 | ✓ OK |
| large_vehicle | 1395 | 150 | 150 | ✓ OK |
| car | 2612 | 150 | 150 | ✓ OK |
| brakes_squeaking | 1395 | 150 | 150 | ✓ OK |
| metro leaving | 1280 | 150 | 150 | ✓ OK |
| metro approaching | 1280 | 150 | 150 | ✓ OK |
| wind_blowing | 1217 | 150 | 150 | ✓ OK |
| birds_singing | 1217 | 150 | 150 | ✓ OK |

**選択されたサンプル数:** 1457

### MAESTRO Validation

**サンプル数のカウント結果:**

- Total: 1077
- Valid: 1077
- Skipped: 0

**クラス別サンプリング:**

| Class | Available | Target | Requested | Status |
|-------|-----------|--------|-----------|--------|
| cutlery and dishes | 202 | 150 | 150 | ✓ OK |
| people talking | 1077 | 150 | 150 | ✓ OK |
| children voices | 1063 | 150 | 150 | ✓ OK |
| footsteps | 1077 | 150 | 150 | ✓ OK |
| large_vehicle | 233 | 150 | 150 | ✓ OK |
| car | 409 | 150 | 150 | ✓ OK |
| brakes_squeaking | 233 | 150 | 150 | ✓ OK |
| metro leaving | 292 | 150 | 150 | ✓ OK |
| metro approaching | 292 | 150 | 150 | ✓ OK |
| wind_blowing | 176 | 150 | 150 | ✓ OK |
| birds_singing | 176 | 150 | 150 | ✓ OK |

**選択されたサンプル数:** 881

## データセット別サマリー

| Dataset | Total Samples | Target per Class | Classes |
|---------|--------------|------------------|--------|
| Desed Validation | 1328 | 150 | 10 |
| Desed Unlabeled | 1500 | 150 | 10 |
| Maestro Training | 1457 | 150 | 11 |
| Maestro Validation | 881 | 150 | 11 |
| **Total** | **5166** | - | **21** |

## クラス別詳細

### DESED (10 classes)

| Class | Validation | Unlabeled | Total |
|-------|-----------|-----------|-------|
| Alarm_bell_ringing | 150 | 0 | 150 |
| Blender | 150 | 0 | 150 |
| Cat | 150 | 0 | 150 |
| Dishes | 150 | 0 | 150 |
| Dog | 150 | 0 | 150 |
| Electric_shaver_toothbrush | 150 | 0 | 150 |
| Frying | 150 | 0 | 150 |
| Running_water | 150 | 0 | 150 |
| Speech | 150 | 0 | 150 |
| Vacuum_cleaner | 150 | 0 | 150 |

### MAESTRO (11 classes)

| Class | Training | Validation | Total | Fallback |
|-------|----------|-----------|-------|----------|
| people talking | 150 | 150 | 300 | No |
| children voices | 150 | 150 | 300 | No |
| metro leaving | 150 | 150 | 300 | No |
| birds_singing | 150 | 150 | 300 | No |
| metro approaching | 150 | 150 | 300 | No |
| cutlery and dishes | 150 | 150 | 300 | No |
| footsteps | 150 | 150 | 300 | No |
| brakes_squeaking | 150 | 150 | 300 | No |
| wind_blowing | 150 | 150 | 300 | No |
| large_vehicle | 150 | 150 | 300 | No |
| car | 150 | 150 | 300 | No |

## 全体のクラス分布

| Class | Total Samples |
|-------|---------------|
| Alarm_bell_ringing | 150 |
| Blender | 150 |
| Cat | 150 |
| Dishes | 150 |
| Dog | 150 |
| Electric_shaver_toothbrush | 150 |
| Frying | 150 |
| Running_water | 150 |
| Speech | 150 |
| Vacuum_cleaner | 150 |
| birds_singing | 300 |
| brakes_squeaking | 300 |
| car | 300 |
| children voices | 300 |
| cutlery and dishes | 300 |
| footsteps | 300 |
| large_vehicle | 300 |
| metro approaching | 300 |
| metro leaving | 300 |
| people talking | 300 |
| wind_blowing | 300 |

**合計サンプル数:** 5166
