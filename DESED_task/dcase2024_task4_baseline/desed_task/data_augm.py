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
    batch_size, n_mels, time_frames = data.size()
    device = data.device

    # ベータ分布を取得
    if alpha > 0:
        dist = torch.distributions.beta.Beta(alpha, alpha)
        lam = dist.sample().item()
    else:
        lam = 1.0

    # データの時間長とlambdaから,切る幅を決定
    #### 現在はバッチ内で同一の比率. 複雑な処理を回避するため ####
    cut_width = int(time_frames * (1 - lam))

    if cut_width <= 0 or cut_width >= time_frames:
        if target_f is not None and target_c is not None:
            return data, target_c, target_f
        return data

    # バッチをランダムに並べる
    perm = torch.randperm(batch_size, device=data.device)

    # shape: [batch_size]
    start_ts = torch.randint(0, time_frames - cut_width + 1, (batch_size,), device=device)

    # 時間インデックスを作成 [0, 1, 2, ..., T-1]
    arange_t = torch.arange(time_frames, device=device).unsqueeze(0) # [1, T]
    
    # start_ts を [B, 1] に拡張してブロードキャスト比較
    # mask: [Batch, Time] -> カットする場所がTrue
    start_ts_expanded = start_ts.unsqueeze(1)
    mask = (arange_t >= start_ts_expanded) & (arange_t < start_ts_expanded + cut_width)
    
    # 周波数軸に合わせて次元拡張: [Batch, 1, Time]
    mask_expanded = mask.unsqueeze(1)


    # maskがTrueの場所は perm データを、Falseの場所は original データを使う
    mixed_data = torch.where(mask_expanded, data[perm], data)

    if target_f is not None and target_c is not None:
        # target_fの時間次元に合わせたマスクを作成
        # BEATsのembedding未使用でフレームがずれた?
        target_time_frames = target_f.size(2)
        
        # target_fの時間解像度に合わせてstart_tsとcut_widthをスケーリング
        scale_factor = target_time_frames / time_frames
        target_cut_width = int(cut_width * scale_factor)
        target_start_ts = (start_ts.float() * scale_factor).long()
        
        # target_f用のマスクを作成
        arange_target = torch.arange(target_time_frames, device=device)
        target_start_ts_expanded = target_start_ts.unsqueeze(1)
        mask_target = (arange_target >= target_start_ts_expanded) & (arange_target < target_start_ts_expanded + target_cut_width)
        mask_target_expanded = mask_target.unsqueeze(1)
        
        # Frame-level targetも同様にマスクで合成
        mixed_target_f = torch.where(mask_target_expanded, target_f[perm], target_f)
        
        epsilon = 1e-6
        energy_orig = target_f.sum(dim=2) + epsilon
        energy_perm = target_f[perm].sum(dim=2) + epsilon

        # 背景に残る部分
        mask_inv = ~mask_target_expanded

        # target_f * mask_target_expanded.float() で、Trueの部分以外が0になる
        # 背景画像の残存エネルギーを直接算出
        rem_energy_bg = (target_f * mask_inv.float()).sum(dim=2)
        energy_total_bg = target_f.sum(dim=2)
        rem_energy_patch = (target_f[perm] * mask_target_expanded.float()).sum(dim=2)
        energy_total_patch = target_f[perm].sum(dim=2)

        # エネルギーのフラグ
        has_energy_bg = energy_total_bg > epsilon
        has_energy_patch = energy_total_patch > epsilon

        # 残存率の計算
        ratio_bg_energy = rem_energy_bg / (energy_total_bg + epsilon)
        ratio_patch_energy = rem_energy_patch / (energy_total_patch + epsilon)


        # --- フレームのエネルギーが小さい時,ラベルが信用できないので,時間比率でクリップラベルを混合 ---
        cut_ratio = mask_target_expanded.float().mean(dim=2).squeeze(1) # [B] -> 各サンプルのカット率

        n_classes = target_f.size(1)
        cut_ratio_expanded = cut_ratio.unsqueeze(1).expand(-1, n_classes)

        ratio_bg_area = 1.0 - cut_ratio_expanded
        ratio_patch_area = cut_ratio_expanded

        # エネルギーがあるクラス -> エネルギー比率を採用
        # エネルギーがないクラス -> 面積比率を採用 (弱ラベルのみのデータ救済)
        final_ratio_bg = torch.where(has_energy_bg, ratio_bg_energy, ratio_bg_area)
        final_ratio_patch = torch.where(has_energy_patch, ratio_patch_energy, ratio_patch_area)

        # --- ラベル適用 ---
        mixed_target_c_bg = target_c * final_ratio_bg
        mixed_target_c_patch = target_c[perm] * final_ratio_patch
        
        # Additive mix 線形和を取る
        mixed_target_c = mixed_target_c_bg + mixed_target_c_patch
        mixed_target_c = torch.clamp(mixed_target_c, max=1.0)
        # mixed_target_c = torch.max(mixed_target_c_bg, mixed_target_c_patch)

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


def strong_augment(data, target_f=None, target_c=None,
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