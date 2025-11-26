import random

import numpy as np
import torch


def frame_shift(mels, labels, net_pooling=4):
    bsz, n_bands, frames = mels.shape
    shifted = []
    new_labels = []
    for bindx in range(bsz):
        shift = int(random.gauss(0, 90))
        shifted.append(torch.roll(mels[bindx], shift, dims=-1))
        shift = -abs(shift) // net_pooling if shift < 0 else shift // net_pooling
        new_labels.append(torch.roll(labels[bindx], shift, dims=-1))
    return torch.stack(shifted), torch.stack(new_labels)


def mixup(data, target=None, alpha=0.2, beta=0.2, mixup_label_type="soft"):
    """Mixup data augmentation by permuting the data.

    Args:
        data: input tensor, must be a batch so data can be permuted and mixed.
        target: tensor of the target to be mixed, if None, do not return targets.
        alpha: float, the parameter to the np.random.beta distribution
        beta: float, the parameter to the np.random.beta distribution
        mixup_label_type: str, the type of mixup to be used choice between {'soft', 'hard'}.
    Returns:
        torch.Tensor of mixed data and labels if given
    """
    with torch.no_grad():
        batch_size = data.size(0)
        c = np.random.beta(alpha, beta)

        perm = torch.randperm(batch_size)

        mixed_data = c * data + (1 - c) * data[perm, :]
        if target is not None:
            if mixup_label_type == "soft":
                mixed_target = torch.clamp(
                    c * target + (1 - c) * target[perm, :], min=0, max=1
                )
            elif mixup_label_type == "hard":
                mixed_target = torch.clamp(target + target[perm, :], min=0, max=1)
            else:
                raise NotImplementedError(
                    f"mixup_label_type: {mixup_label_type} not implemented. choice in "
                    f"{'soft', 'hard'}"
                )

            return mixed_data, mixed_target
        else:
            return mixed_data


def cutmix(data, embeddings, target_c, target_f, alpha=1.0):
    """
    CutMix for Spectrogram AND BEATs Embeddings (Time-Synchronized).
    
    Args:
        data: Spectrogram tensor [batch, n_mels, time_spec].
        embeddings: BEATs features [batch, embed_dim, time_emb].
        target_c: Clip-level targets [batch, n_classes].
        target_f: Frame-level targets [batch, n_classes, time_target].
        alpha: Beta distribution parameter.
        
    Returns:
        mixed_data, mixed_embeddings, mixed_target_c, mixed_target_f
    """
    batch_size, n_mels, time_spec = data.size()
    device = data.device

    # --- 1. カットパラメータの決定 (スペクトログラム基準) ---
    if alpha > 0:
        dist = torch.distributions.beta.Beta(alpha, alpha)
        lam = dist.sample().item()
    else:
        lam = 1.0

    # カット幅 (1 - lambda)
    cut_width = int(time_spec * (1 - lam))

    # カット幅が無効な場合
    if cut_width <= 0 or cut_width >= time_spec:
        # 何もせず返す
        return data, embeddings, target_c, target_f

    # --- 2. 共通のランダム置換 (Permutation) ---
    # スペクトログラム、Embedding、ラベル全てで同じ相手と混ぜる必要がある
    perm = torch.randperm(batch_size, device=device)

    # --- 3. マスク生成ヘルパー関数 ---
    # 異なる時間解像度に対応するための内部関数
    def generate_mask(total_time, ref_start_ts, ref_total_time, ref_cut_width):
        """
        基準(Spectrogram)のカット位置を、指定された時間長(total_time)に合わせてスケーリングしマスクを生成
        """
        scale = total_time / ref_total_time
        
        # スケーリング後の開始位置と幅
        # round() で丸めることでズレを防ぐ
        scaled_start = (ref_start_ts.float() * scale).round().long()
        scaled_width = int(ref_cut_width * scale)
        
        # マスク作成
        arange = torch.arange(total_time, device=device).unsqueeze(0) # [1, T]
        start_exp = scaled_start.unsqueeze(1) # [B, 1]
        
        # [B, 1, T]
        mask = (arange >= start_exp) & (arange < start_exp + scaled_width)
        return mask.unsqueeze(1) 

    # --- 4. スペクトログラムのMix ---
    # 基準となるカット開始位置 [Batch]
    start_ts = torch.randint(0, time_spec - cut_width + 1, (batch_size,), device=device)
    
    # スペクトログラム用マスク
    mask_spec = generate_mask(time_spec, start_ts, time_spec, cut_width)
    mixed_data = torch.where(mask_spec, data[perm], data)

    # --- 5. BEATs EmbeddingsのMix ---
    # Embeddingの時間長を取得
    time_emb = embeddings.size(2)
    
    # Embedding用マスク (スペクトログラムのカット位置を投影)
    mask_emb = generate_mask(time_emb, start_ts, time_spec, cut_width)
    mixed_embeddings = torch.where(mask_emb, embeddings[perm], embeddings)

    # --- 6. ラベルのMix (Energy-based + Area Fallback) ---
    
    # --- 6.1 Strong Label (Target_F) のMix ---
    time_target = target_f.size(2)
    mask_target = generate_mask(time_target, start_ts, time_spec, cut_width)
    mixed_target_f = torch.where(mask_target, target_f[perm], target_f)

    # --- 6.2 Weak Label (Target_C) の再計算 ---
    
    # マスクの反転（背景部分）
    mask_inv = ~mask_target

    epsilon = 1e-6
    
    # 背景のエネルギー残存率
    energy_bg = (target_f * mask_inv.float()).sum(dim=2)
    total_energy_bg = target_f.sum(dim=2)
    ratio_bg_energy = energy_bg / (total_energy_bg + epsilon)
    
    # パッチのエネルギー混入率
    energy_patch = (target_f[perm] * mask_target.float()).sum(dim=2)
    total_energy_patch = target_f[perm].sum(dim=2)
    ratio_patch_energy = energy_patch / (total_energy_patch + epsilon)

    # エネルギー有無フラグ
    has_energy_bg = total_energy_bg > epsilon
    has_energy_patch = total_energy_patch > epsilon

    # [Fallback] 面積比率計算
    # target解像度でのカット率
    cut_ratio = mask_target.float().mean(dim=2).squeeze(1) # [B]
    
    # クラス次元へ拡張
    n_classes = target_c.size(1)
    cut_ratio_exp = cut_ratio.unsqueeze(1).expand(-1, n_classes)
    
    bg_area_ratio = 1.0 - cut_ratio_exp
    patch_area_ratio = cut_ratio_exp

    # 最終比率の決定 (エネルギーがない場合は面積比率を採用)
    final_ratio_bg = torch.where(has_energy_bg, ratio_bg_energy, bg_area_ratio)
    final_ratio_patch = torch.where(has_energy_patch, ratio_patch_energy, patch_area_ratio)

    # ラベル適用
    mixed_target_c_bg = target_c * final_ratio_bg
    mixed_target_c_patch = target_c[perm] * final_ratio_patch
    
    mixed_target_c = torch.clamp(mixed_target_c_bg + mixed_target_c_patch, max=1.0)

    return mixed_data, mixed_embeddings, mixed_target_c, mixed_target_f

def add_noise(mels, snrs=(6, 30), dims=(1, 2)):
    """Add white noise to mels spectrograms
    Args:
        mels: torch.tensor, mels spectrograms to apply the white noise to.
        snrs: int or tuple, the range of snrs to choose from if tuple (uniform)
        dims: tuple, the dimensions for which to compute the standard deviation (default to (1,2) because assume
            an input of a batch of mel spectrograms.
    Returns:
        torch.Tensor of mels with noise applied
    """
    if isinstance(snrs, (list, tuple)):
        snr = (snrs[0] - snrs[1]) * torch.rand(
            (mels.shape[0],), device=mels.device
        ).reshape(-1, 1, 1) + snrs[1]
    else:
        snr = snrs

    snr = 10 ** (snr / 20)  # linear domain
    sigma = torch.std(mels, dim=dims, keepdim=True) / snr
    mels = mels + torch.randn(mels.shape, device=mels.device) * sigma

    return mels


def strong_augment(data, target_c=None, target_f=None,
                   frame_shift_std=90, time_mask_max=5,
                   time_mask_prob=0.5, net_pooling=4):
    """Strong augmentation combining Frame Shift and Time Masking.

    Applies frame shift followed by time masking for strong augmentation.
    Unlike CutMix, this operates on a single sample without mixing.

    Args:
        data: torch.Tensor, input tensor of shape [batch_size, n_mels, time_frames].
        target_f: torch.Tensor or None, frame-level labels of shape
            [batch_size, n_classes, time_frames_label].
        target_c: torch.Tensor or None, clip-level labels of shape [batch_size, n_classes].
        frame_shift_std: float, standard deviation for Gaussian frame shift (default: 90).
        time_mask_max: int, maximum length of time mask (default: 5).
        time_mask_prob: float, probability of applying time masking per sample (default: 0.5).
        net_pooling: int, pooling factor for label alignment (default: 4).

    Returns:
        Augmented data and labels if provided.
    """
    import torchaudio.transforms as T

    # 1. Frame Shiftを適用（既存の関数を使用）
    if target_f is not None:
        # frame_shift関数はラベルも一緒にシフトする
        # net_poolingに応じてラベルのシフト量を調整
        shifted_data, shifted_target_f = frame_shift(data, target_f, net_pooling=net_pooling)
    else:
        # ラベルがない場合は、dataのみシフト
        batch_size, n_mels, time_frames = data.size()
        shifted = []
        for bindx in range(batch_size):
            shift = int(random.gauss(0, frame_shift_std))
            shifted.append(torch.roll(data[bindx], shift, dims=-1))
        shifted_data = torch.stack(shifted)
        shifted_target_f = None

    # 2. Time Maskingを適用（torchaudio使用）
    # iid_masks=True: バッチ内の各サンプルに異なるマスクを適用
    # p: 各サンプルにマスクを適用する確率
    time_mask = T.TimeMasking(
        time_mask_param=time_mask_max,
        iid_masks=True,
        p=time_mask_prob
    )
    masked_data = time_mask(shifted_data)

    # クリップラベルは変化しない（クリップ全体のラベルなので）
    if target_f is not None and target_c is not None:
        return masked_data, target_c, shifted_target_f
    else:
        return masked_data