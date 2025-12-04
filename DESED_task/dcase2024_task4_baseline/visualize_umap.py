#!/usr/bin/env python3
"""
UMAP可視化スクリプト

推論結果から特徴量を読み込み、UMAPで2D投影して可視化する。
データセット別、ドメイン別、クラス別、モデル別の比較が可能。

使用法:
    python visualize_umap.py \
        --input_dirs inference_outputs/baseline inference_outputs/cmt_normal \
        --output_dir visualization_outputs/umap
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import umap
from tqdm import tqdm


# クラス定義
# モデル出力は27次元（DESED 10 + MAESTRO Real 17）だが、
# 固定データセットでは21クラスのみ使用

# DESED 10クラス（インデックス 0-9）
DESED_CLASSES = [
    "Alarm_bell_ringing", "Blender", "Cat", "Dishes", "Dog",
    "Electric_shaver_toothbrush", "Frying", "Running_water",
    "Speech", "Vacuum_cleaner"
]

# MAESTRO Real 17クラス（インデックス 10-26）
MAESTRO_REAL_ALL = [
    "cutlery and dishes",      # 10
    "furniture dragging",      # 11 (未使用)
    "people talking",          # 12
    "children voices",         # 13
    "coffee machine",          # 14 (未使用)
    "footsteps",               # 15
    "large_vehicle",           # 16
    "car",                     # 17
    "brakes_squeaking",        # 18
    "cash register beeping",   # 19 (未使用)
    "announcement",            # 20 (未使用)
    "shopping cart",           # 21 (未使用)
    "metro leaving",           # 22
    "metro approaching",       # 23
    "door opens/closes",       # 24 (未使用)
    "wind_blowing",            # 25
    "birds_singing",           # 26
]

# 全27クラス（モデル出力次元に対応）
ALL_CLASSES_27 = DESED_CLASSES + MAESTRO_REAL_ALL

# 固定データセットで使用している21クラス
USED_CLASS_INDICES = [
    # DESED 10クラス全て
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    # MAESTRO 11クラス
    10,  # cutlery and dishes
    12,  # people talking
    13,  # children voices
    15,  # footsteps
    16,  # large_vehicle
    17,  # car
    18,  # brakes_squeaking
    22,  # metro leaving
    23,  # metro approaching
    25,  # wind_blowing
    26,  # birds_singing
]

USED_CLASSES_21 = [ALL_CLASSES_27[i] for i in USED_CLASS_INDICES]


def load_inference_data(model_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    1つのモデルディレクトリから全データセットの推論結果を読み込む

    Returns:
        {
            'desed_validation': {'features_student': ..., 'probs_student': ..., ...},
            'maestro_training': {...},
            ...
        }
    """
    datasets = {}
    npz_files = list(model_dir.glob("*.npz"))

    for npz_file in npz_files:
        dataset_name = npz_file.stem  # 'desed_validation' など
        data = np.load(npz_file, allow_pickle=True)

        datasets[dataset_name] = {
            'features_student': data['features_student'],
            'features_teacher': data['features_teacher'],
            'probs_student': data['probs_student'],
            'probs_teacher': data['probs_teacher'],
            'filenames': data['filenames'],
        }

        # targetsはラベル付きデータのみ
        if 'targets' in data:
            datasets[dataset_name]['targets'] = data['targets']

    return datasets


def prepare_umap_data(
    models_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    feature_type: str = 'student'
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    全モデル・全データセットの特徴量を結合し、メタデータを付与

    Args:
        models_data: {model_name: {dataset_name: {key: array}}}
        feature_type: 'student' or 'teacher'

    Returns:
        features: (N, 384) 特徴量配列
        metadata: (N,) メタデータDataFrame
    """
    all_features = []
    all_metadata = []

    for model_name, datasets in models_data.items():
        for dataset_name, data in datasets.items():
            features = data[f'features_{feature_type}']
            n_samples = features.shape[0]

            # メタデータの作成
            meta = {
                'model': [model_name] * n_samples,
                'dataset': [dataset_name] * n_samples,
                'filenames': data['filenames'].tolist(),
            }

            # ドメイン情報（合成/実データ）
            if 'desed' in dataset_name:
                if 'unlabeled' in dataset_name:
                    domain = 'unlabeled'
                else:
                    domain = 'synthetic'
            else:  # maestro
                domain = 'real'
            meta['domain'] = [domain] * n_samples

            # クラスラベル（ラベル付きデータのみ）
            if 'targets' in data:
                targets = data['targets']
                # マルチラベルなので最も確率が高いクラスを代表として選択
                probs = data[f'probs_{feature_type}']
                pred_classes = np.argmax(probs, axis=1)
                meta['predicted_class'] = [ALL_CLASSES_27[i] if i < len(ALL_CLASSES_27) else f'class_{i}'
                                           for i in pred_classes]

                # 正解クラス（複数の場合は最初のもの）
                true_classes = []
                for target in targets:
                    true_indices = np.where(target > 0.5)[0]
                    if len(true_indices) > 0:
                        true_classes.append(ALL_CLASSES_27[true_indices[0]] if true_indices[0] < len(ALL_CLASSES_27)
                                          else f'class_{true_indices[0]}')
                    else:
                        true_classes.append('none')
                meta['true_class'] = true_classes
            else:
                meta['predicted_class'] = ['unlabeled'] * n_samples
                meta['true_class'] = ['unlabeled'] * n_samples

            all_features.append(features)
            all_metadata.append(pd.DataFrame(meta))

    # 結合
    features = np.concatenate(all_features, axis=0)
    metadata = pd.concat(all_metadata, ignore_index=True)

    return features, metadata


def apply_umap(features: np.ndarray, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    """UMAP次元削減を適用"""
    print(f"UMAPを適用中... (入力次元: {features.shape[1]}, 出力次元: {n_components})")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=random_state,
        verbose=True
    )

    embedding = reducer.fit_transform(features)
    print(f"UMAP完了: {embedding.shape}")

    return embedding


def plot_umap_static(
    embedding: np.ndarray,
    metadata: pd.DataFrame,
    color_by: str,
    output_path: Path,
    title: str = None
):
    """静止画版UMAP可視化（Matplotlib）"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # 色分けのための準備
    unique_values = metadata[color_by].unique()
    colors = sns.color_palette('tab20', n_colors=len(unique_values))
    color_map = {val: colors[i] for i, val in enumerate(unique_values)}

    # 各グループをプロット
    for value in unique_values:
        mask = metadata[color_by] == value
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[color_map[value]],
            label=value,
            alpha=0.6,
            s=20,
            edgecolors='none'
        )

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(title or f'UMAP Projection (colored by {color_by})', fontsize=14)

    # 凡例（多い場合は複数列）
    if len(unique_values) > 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"保存: {output_path}")


def plot_umap_interactive(
    embedding: np.ndarray,
    metadata: pd.DataFrame,
    color_by: str,
    output_path: Path,
    title: str = None
):
    """インタラクティブ版UMAP可視化（Plotly）"""
    # DataFrameに埋め込みを追加
    df = metadata.copy()
    df['UMAP_1'] = embedding[:, 0]
    df['UMAP_2'] = embedding[:, 1]

    # ホバー情報
    hover_data = ['model', 'dataset', 'domain', 'predicted_class', 'true_class']

    fig = px.scatter(
        df,
        x='UMAP_1',
        y='UMAP_2',
        color=color_by,
        hover_data=hover_data,
        title=title or f'UMAP Projection (colored by {color_by})',
        width=1200,
        height=900,
        opacity=0.7
    )

    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        font=dict(size=12),
        legend=dict(font=dict(size=10))
    )

    fig.write_html(output_path)
    print(f"保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="UMAP可視化スクリプト")
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="推論結果ディレクトリ（複数指定可）"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="出力ディレクトリ"
    )
    parser.add_argument(
        "--feature_type",
        choices=['student', 'teacher', 'both'],
        default='student',
        help="使用する特徴量"
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=2,
        help="UMAP出力次元数"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="乱数シード"
    )

    args = parser.parse_args()

    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("UMAP可視化スクリプト")
    print("="*60)

    # データ読み込み
    print("\n[1/4] データ読み込み中...")
    models_data = {}
    for input_dir in args.input_dirs:
        input_path = Path(input_dir)
        model_name = input_path.name
        print(f"  - {model_name}: {input_path}")
        models_data[model_name] = load_inference_data(input_path)

    # 特徴量タイプの決定
    feature_types = ['student', 'teacher'] if args.feature_type == 'both' else [args.feature_type]

    for feature_type in feature_types:
        print(f"\n{'='*60}")
        print(f"特徴量タイプ: {feature_type}")
        print(f"{'='*60}")

        # データ準備
        print("\n[2/4] データ準備中...")
        features, metadata = prepare_umap_data(models_data, feature_type)
        print(f"  総サンプル数: {features.shape[0]}")
        print(f"  特徴量次元: {features.shape[1]}")
        print(f"  モデル数: {metadata['model'].nunique()}")
        print(f"  データセット数: {metadata['dataset'].nunique()}")

        # UMAP適用
        print("\n[3/4] UMAP適用中...")
        embedding = apply_umap(features, args.n_components, args.random_state)

        # 可視化
        print("\n[4/4] 可視化中...")

        # サブディレクトリ
        sub_dir = output_dir / feature_type
        sub_dir.mkdir(exist_ok=True)

        # 1. データセット別
        plot_umap_static(
            embedding, metadata, 'dataset',
            sub_dir / 'dataset_comparison.png',
            f'UMAP: Dataset Comparison ({feature_type})'
        )

        # 2. ドメイン別
        plot_umap_static(
            embedding, metadata, 'domain',
            sub_dir / 'domain_comparison.png',
            f'UMAP: Domain Comparison ({feature_type})'
        )

        # 3. モデル別
        if metadata['model'].nunique() > 1:
            plot_umap_static(
                embedding, metadata, 'model',
                sub_dir / 'model_comparison.png',
                f'UMAP: Model Comparison ({feature_type})'
            )

        # 4. クラス別（予測クラス）
        # unlabeledを除外
        labeled_mask = metadata['predicted_class'] != 'unlabeled'
        if labeled_mask.sum() > 0:
            plot_umap_static(
                embedding[labeled_mask], metadata[labeled_mask], 'predicted_class',
                sub_dir / 'class_distribution_predicted.png',
                f'UMAP: Predicted Class Distribution ({feature_type})'
            )

            # 真のクラス
            plot_umap_static(
                embedding[labeled_mask], metadata[labeled_mask], 'true_class',
                sub_dir / 'class_distribution_true.png',
                f'UMAP: True Class Distribution ({feature_type})'
            )

        # 5. インタラクティブ版（全情報）
        plot_umap_interactive(
            embedding, metadata, 'dataset',
            sub_dir / 'interactive_umap.html',
            f'UMAP: Interactive Visualization ({feature_type})'
        )

        # メタデータとembeddingの保存
        np.save(sub_dir / 'umap_embedding.npy', embedding)
        metadata.to_csv(sub_dir / 'metadata.csv', index=False)
        print(f"\nメタデータ保存: {sub_dir / 'metadata.csv'}")

    print("\n" + "="*60)
    print("完了！")
    print(f"出力ディレクトリ: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
