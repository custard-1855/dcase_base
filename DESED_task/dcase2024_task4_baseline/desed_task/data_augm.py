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


def cutmix(data, target_c=None, target_f=None, alpha=1.0):
    """CutMix data augmentation for time axis only.

    Applies CutMix augmentation by cutting and pasting a rectangular region
    along the time axis only. Frequency axis is not modified to avoid touching
    domain information.

    Args:
        data: torch.Tensor, input tensor of shape [batch_size, n_mels, time_frames].
        target: torch.Tensor or None, target labels of shape [batch_size, n_classes, time_frames]
            or [batch_size, n_classes]. If None, do not return targets.
        alpha: float, parameter for the Beta distribution.

    Returns:
        torch.Tensor of mixed data and labels if target is given.
    """
    batch_size = data.size(0)
    n_mels = data.size(1)
    time_frames = data.size(2)

    # Sample lambda from Beta distribution
    # ベータ分布を取得
    lam = np.random.beta(alpha, alpha)

    # Random permutation for mixing
    # バッチをランダムに並べる
    perm = torch.randperm(batch_size, device=data.device)

    # Determine cut region along time axis
    # データの時間長とlambdaから,切る幅を決定
    cut_width = int(time_frames * (1 - lam))

    # Random start position
    if cut_width > 0 and cut_width < time_frames:
        start_t = np.random.randint(0, time_frames - cut_width + 1)
        end_t = start_t + cut_width
    else:
        # If cut_width is 0 or >= time_frames, no mixing
        if target_f is not None and target_c is not None:
            return data, target_c, target_f
        else:
            return data

    # Apply CutMix on time axis (frequency axis unchanged)
    mixed_data = data.clone()
    mixed_data[:, :, start_t:end_t] = data[perm, :, start_t:end_t]

    if target_f is not None and target_c is not None:
        mixed_target_f = target_f.clone()
        mixed_target_c = target_c.clone()

        # Handle both frame-level [B, K, T] and clip-level [B, K] targets
        # Frame-level target [B, K, T]
        mixed_target_f[:, :, start_t:end_t] = target_f[perm, :, start_t:end_t]
        
        # Clip-level target [B, K]
        # --- (A) オリジナルのエネルギー（確率の総和）を計算 ---
        # ゼロ除算を防ぐための微小値
        epsilon = 1e-6
        energy_orig = target_f.sum(dim=2) + epsilon
        energy_perm = target_f[perm].sum(dim=2) + epsilon

        # --- (B) CutMix後に残ったエネルギーを分離して計算 ---
        
        # Source 1 (背景画像側):
        #   CutMix範囲(start_t:end_t)以外のエネルギーが残存量となる
        #   計算: 全体 - カットされた部分
        cut_energy_bg = target_f[:, :, start_t:end_t].sum(dim=2)
        rem_energy_bg = energy_orig - epsilon - cut_energy_bg
        
        # Source 2 (パッチ画像側):
        #   CutMix範囲(start_t:end_t)に入っているエネルギーが残存量となる
        rem_energy_patch = target_f[perm, :, start_t:end_t].sum(dim=2)

        # --- (C) 残存率 (Intersection over Self) の計算 ---
        #   値域を [0, 1] に収める
        ratio_bg = torch.clamp(rem_energy_bg / energy_orig, min=0.0, max=1.0)
        ratio_patch = torch.clamp(rem_energy_patch / energy_perm, min=0.0, max=1.0)

        # --- (D) Clipラベルへの適用と統合 ---
        #   target_c (通常1.0) に残存率を掛ける
        mixed_target_c_bg = target_c * ratio_bg
        mixed_target_c_patch = target_c[perm] * ratio_patch
        
        #   両方の成分を統合 (Multi-labelなのでMax Poolingが適当)
        mixed_target_c = torch.max(mixed_target_c_bg, mixed_target_c_patch)

        return mixed_data, mixed_target_c, mixed_target_f
    else:
        return mixed_data


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
