#!/usr/bin/env python3
"""
可視化スクリプト用の共通ユーティリティモジュール

このモジュールは、DCASE 2024 Task 4の可視化スクリプトで共通して使用される
クラス定義、データローディング関数、その他のユーティリティを提供します。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from tqdm import tqdm


# --- クラス定義（DESED + MAESTRO）---
DESED_CLASSES = [
    "Alarm_bell_ringing", "Blender", "Cat", "Dishes", "Dog",
    "Electric_shaver_toothbrush", "Frying", "Running_water",
    "Speech", "Vacuum_cleaner"
]

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

ALL_CLASSES_27 = DESED_CLASSES + MAESTRO_REAL_ALL
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


# --- データローディング関数 ---
def load_inference_data(
    model_dir: Union[str, Path],
    verbose: bool = True
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    1つのモデルディレクトリから全データセットの推論結果を読み込む

    Args:
        model_dir: モデルの推論結果が格納されたディレクトリパス
        verbose: 詳細出力を表示するかどうか

    Returns:
        データセット名をキーとする辞書。各値は以下のキーを持つ辞書：
        - 'features_student': 生徒モデルの特徴量 (N, 384)
        - 'features_teacher': 教師モデルの特徴量 (N, 384)
        - 'probs_student': 生徒モデルの予測確率 (N, 27)
        - 'probs_teacher': 教師モデルの予測確率 (N, 27)
        - 'filenames': ファイル名のリスト (N,)
        - 'targets': 正解ラベル (N, 27) ※ラベル付きデータのみ

    Raises:
        FileNotFoundError: 指定されたディレクトリが存在しない場合
        ValueError: NPZファイルのフォーマットが不正な場合
    """
    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    datasets = {}
    npz_files = list(model_dir.glob("*.npz"))

    if not npz_files:
        if verbose:
            print(f"  警告: {model_dir} にNPZファイルが見つかりません")
        return datasets

    if verbose:
        npz_files = tqdm(npz_files, desc=f"Loading {model_dir.name}")

    for npz_file in npz_files:
        dataset_name = npz_file.stem  # 'desed_validation' など

        try:
            data = np.load(npz_file, allow_pickle=True)

            # 必須フィールドの確認
            required_keys = ['features_student', 'features_teacher',
                           'probs_student', 'probs_teacher', 'filenames']
            missing_keys = [k for k in required_keys if k not in data]

            if missing_keys:
                raise ValueError(f"Missing required keys in {npz_file}: {missing_keys}")

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

        except Exception as e:
            if verbose:
                print(f"  エラー: {npz_file} の読み込みに失敗: {e}")
            continue

    return datasets


# --- メタデータ作成関数 ---
def create_metadata(
    data: Dict[str, np.ndarray],
    model_name: str,
    dataset_name: str,
    feature_type: str = 'student'
) -> pd.DataFrame:
    """
    単一データセットのメタデータDataFrameを作成

    Args:
        data: load_inference_dataで読み込んだ単一データセットのデータ
        model_name: モデル名
        dataset_name: データセット名
        feature_type: 'student' または 'teacher'

    Returns:
        メタデータDataFrame with columns:
        - model: モデル名
        - dataset: データセット名
        - domain: 'synthetic', 'real', または 'unlabeled'
        - filenames: ファイル名
        - predicted_class: 予測クラス（最大確率）
        - true_class: 正解クラス（ラベル付きデータのみ）
        - max_prob: 最大予測確率
        - entropy: 予測のエントロピー
    """
    n_samples = data[f'features_{feature_type}'].shape[0]
    probs = data[f'probs_{feature_type}']

    # 基本メタデータ
    meta = {
        'model': [model_name] * n_samples,
        'dataset': [dataset_name] * n_samples,
        'filenames': data['filenames'].tolist() if data['filenames'].ndim > 0 else [data['filenames'].item()],
    }

    # ドメイン情報の判定
    if 'desed' in dataset_name.lower():
        if 'unlabeled' in dataset_name.lower():
            domain = 'unlabeled'
        else:
            domain = 'synthetic'
    elif 'maestro' in dataset_name.lower():
        domain = 'real'
    else:
        domain = 'unknown'
    meta['domain'] = [domain] * n_samples

    # 予測クラス（最大確率のクラス）
    pred_classes_idx = np.argmax(probs, axis=1)
    meta['predicted_class'] = [
        ALL_CLASSES_27[i] if i < len(ALL_CLASSES_27) else f'class_{i}'
        for i in pred_classes_idx
    ]

    # 最大予測確率
    meta['max_prob'] = np.max(probs, axis=1).tolist()

    # エントロピー（不確実性の指標）
    eps = 1e-10
    probs_clipped = np.clip(probs, eps, 1 - eps)
    entropy = -np.sum(probs_clipped * np.log(probs_clipped), axis=1)
    meta['entropy'] = entropy.tolist()

    # 正解クラス（ラベル付きデータのみ）
    if 'targets' in data:
        targets = data['targets']
        true_classes = []
        for target in targets:
            true_indices = np.where(target > 0.5)[0]
            if len(true_indices) > 0:
                # 最初の正解クラスを代表として使用
                class_name = (ALL_CLASSES_27[true_indices[0]]
                            if true_indices[0] < len(ALL_CLASSES_27)
                            else f'class_{true_indices[0]}')
                true_classes.append(class_name)
            else:
                true_classes.append('none')
        meta['true_class'] = true_classes
    else:
        meta['predicted_class_label'] = ['unlabeled'] * n_samples
        meta['true_class'] = ['unlabeled'] * n_samples

    return pd.DataFrame(meta)


# --- 複数モデル・データセットの結合 ---
def combine_multiple_models(
    models_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    feature_type: str = 'student'
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    複数モデル・複数データセットの特徴量とメタデータを結合

    Args:
        models_data: {model_name: {dataset_name: {key: array}}}
        feature_type: 'student' または 'teacher'

    Returns:
        features: 結合された特徴量配列 (N_total, 384)
        metadata: 結合されたメタデータDataFrame (N_total, columns)
    """
    all_features = []
    all_metadata = []

    for model_name, datasets in models_data.items():
        for dataset_name, data in datasets.items():
            # 特徴量
            features = data[f'features_{feature_type}']
            all_features.append(features)

            # メタデータ
            meta_df = create_metadata(data, model_name, dataset_name, feature_type)
            all_metadata.append(meta_df)

    # 結合
    features = np.concatenate(all_features, axis=0)
    metadata = pd.concat(all_metadata, ignore_index=True)

    return features, metadata

# --- データ前処理ユーティリティ---
def filter_by_domain(
    features: np.ndarray,
    metadata: pd.DataFrame,
    domains: List[str]
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    指定されたドメインのデータのみをフィルタリング

    Args:
        features: 特徴量配列
        metadata: メタデータDataFrame
        domains: フィルタリングするドメインのリスト
            例: ['synthetic', 'real']

    Returns:
        フィルタリング後の特徴量とメタデータ
    """
    mask = metadata['domain'].isin(domains)
    return features[mask], metadata[mask].reset_index(drop=True)


def filter_by_class(
    features: np.ndarray,
    metadata: pd.DataFrame,
    classes: List[str],
    use_predicted: bool = True
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    指定されたクラスのデータのみをフィルタリング

    Args:
        features: 特徴量配列
        metadata: メタデータDataFrame
        classes: フィルタリングするクラスのリスト
        use_predicted: True=予測クラスでフィルタ、False=正解クラスでフィルタ

    Returns:
        フィルタリング後の特徴量とメタデータ
    """
    column = 'predicted_class' if use_predicted else 'true_class'
    mask = metadata[column].isin(classes)
    return features[mask], metadata[mask].reset_index(drop=True)


def sample_balanced(
    features: np.ndarray,
    metadata: pd.DataFrame,
    n_samples_per_class: int,
    column: str = 'predicted_class',
    random_state: int = 42
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    各クラスから均等にサンプリング

    Args:
        features: 特徴量配列
        metadata: メタデータDataFrame
        n_samples_per_class: クラスごとのサンプル数
        column: バランシングに使用するカラム名
        random_state: 乱数シード

    Returns:
        バランスされた特徴量とメタデータ
    """
    np.random.seed(random_state)

    balanced_indices = []
    for class_name in metadata[column].unique():
        class_indices = metadata[metadata[column] == class_name].index.values
        if len(class_indices) > n_samples_per_class:
            sampled = np.random.choice(class_indices, n_samples_per_class, replace=False)
        else:
            sampled = class_indices
        balanced_indices.extend(sampled)

    balanced_indices = np.array(balanced_indices)
    return features[balanced_indices], metadata.iloc[balanced_indices].reset_index(drop=True)


# --- 統計情報の計算 ---
def compute_dataset_statistics(metadata: pd.DataFrame) -> Dict[str, any]:
    """
    データセットの統計情報を計算

    Args:
        metadata: メタデータDataFrame

    Returns:
        統計情報を含む辞書
    """
    stats = {
        'total_samples': len(metadata),
        'n_models': metadata['model'].nunique(),
        'n_datasets': metadata['dataset'].nunique(),
        'domain_distribution': metadata['domain'].value_counts().to_dict(),
        'class_distribution': metadata['predicted_class'].value_counts().to_dict(),
    }

    # ラベル付きデータがある場合
    if 'true_class' in metadata and not all(metadata['true_class'] == 'unlabeled'):
        labeled_mask = metadata['true_class'] != 'unlabeled'
        stats['n_labeled'] = labeled_mask.sum()
        stats['n_unlabeled'] = (~labeled_mask).sum()

    # エントロピー統計
    if 'entropy' in metadata:
        stats['entropy_mean'] = metadata['entropy'].mean()
        stats['entropy_std'] = metadata['entropy'].std()

    # 確信度統計
    if 'max_prob' in metadata:
        stats['confidence_mean'] = metadata['max_prob'].mean()
        stats['confidence_std'] = metadata['max_prob'].std()

    return stats


def print_statistics(stats: Dict[str, any], title: str = "Dataset Statistics"):
    """
    統計情報を見やすく表示

    Args:
        stats: 統計情報の辞書
        title: 表示タイトル
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    # 基本統計
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Number of models: {stats['n_models']}")
    print(f"Number of datasets: {stats['n_datasets']}")

    # ラベル情報
    if 'n_labeled' in stats:
        print(f"Labeled samples: {stats['n_labeled']:,}")
        print(f"Unlabeled samples: {stats['n_unlabeled']:,}")

    # ドメイン分布
    print("\nDomain distribution:")
    for domain, count in stats['domain_distribution'].items():
        percentage = (count / stats['total_samples']) * 100
        print(f"  {domain}: {count:,} ({percentage:.1f}%)")

    # エントロピー・確信度
    if 'entropy_mean' in stats:
        print(f"\nEntropy: {stats['entropy_mean']:.3f} ± {stats['entropy_std']:.3f}")
    if 'confidence_mean' in stats:
        print(f"Confidence: {stats['confidence_mean']:.3f} ± {stats['confidence_std']:.3f}")

    print(f"{'='*60}\n")