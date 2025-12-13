#!/usr/bin/env python3
"""MAESTRO Real データセットの分割数を確認するスクリプト"""

import pandas as pd
import yaml
from train_pretrained import split_maestro

# 設定ファイル読み込み
with open("confs/pretrained.yaml") as f:
    config = yaml.safe_load(f)

# MAESTRO Real train TSVを読み込み
maestro_real_train_df = pd.read_csv(
    config["data"]["real_maestro_train_tsv"],
    sep="\t",
)

print(f"MAESTRO Real Train TSV 全体のレコード数: {len(maestro_real_train_df)}")
print(f"ユニークなファイル数: {maestro_real_train_df['filename'].nunique()}")
print()

# 分割実行
maestro_train_df, maestro_valid_df = split_maestro(config, maestro_real_train_df)

print("分割後:")
print(f"  Train: {len(maestro_train_df)} レコード")
print(f"  Valid: {len(maestro_valid_df)} レコード")
print(f"  合計: {len(maestro_train_df) + len(maestro_valid_df)} レコード")
print()

# ファイル数も確認
print("ユニークファイル数:")
print(f"  Train: {maestro_train_df['filename'].nunique()} ファイル")
print(f"  Valid: {maestro_valid_df['filename'].nunique()} ファイル")
print()

# 設定ファイルの split 値を確認
print(f"maestro_split 設定値: {config['training']['maestro_split']}")
print(
    f"期待される分割比率: Train {config['training']['maestro_split']:.1%}, Valid {1 - config['training']['maestro_split']:.1%}"
)
print(
    f"実際の分割比率: Train {len(maestro_train_df) / (len(maestro_train_df) + len(maestro_valid_df)):.1%}, Valid {len(maestro_valid_df) / (len(maestro_train_df) + len(maestro_valid_df)):.1%}"
)
