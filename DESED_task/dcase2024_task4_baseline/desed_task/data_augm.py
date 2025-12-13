import random

import numpy as np
import torch


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
                    c * target + (1 - c) * target[perm, :],
                    min=0,
                    max=1,
                )
            elif mixup_label_type == "hard":
                mixed_target = torch.clamp(target + target[perm, :], min=0, max=1)
            else:
                raise NotImplementedError(
                    f"mixup_label_type: {mixup_label_type} not implemented. choice in "
                    f"{'soft', 'hard'}",
                )

            return mixed_data, mixed_target
        return mixed_data


class MixupAugmentor:
    def __init__(self, config):
        """Args:
        config: hparams全体、もしくは mixup に関する辞書

        """
        self.mixup_type = config.get("mixup", "soft")
        self.alpha = config.get("mixup_alpha", 0.2)
        self.beta = config.get("mixup_beta", 0.2)

    def apply_mixup(self, features, embeddings, labels, start_indx, stop_indx):
        # マスクの作成
        batch_num = features.shape[0]
        current_mask = torch.zeros(batch_num).to(features.device).bool()  # device対応を追加
        current_mask[start_indx:stop_indx] = 1

        # 対象データの抽出
        masked_features = features[current_mask]
        masked_labels = labels[current_mask]
        masked_embeddings = embeddings[current_mask] if embeddings is not None else None

        # --- 修正ポイント: パラメータをここで1回だけ生成 ---
        # データ数を取得
        sub_batch_size = masked_features.size(0)

        # Mixupパラメータの生成 (alpha, betaはhparamsから取ると想定、ここではデフォルト値)
        alpha, beta = 0.2, 0.2
        lam = np.random.beta(alpha, beta)
        perm = torch.randperm(sub_batch_size).to(features.device)  # deviceを合わせる

        # --- Features の Mixup ---
        features[current_mask] = self._mixup_data(masked_features, perm, lam)

        # --- Embeddings の Mixup (存在する場合、同じ perm と lam を使用) ---
        if embeddings is not None:
            embeddings[current_mask] = self._mixup_data(masked_embeddings, perm, lam)

        # --- Labels の Mixup (最後に1回だけ、同じ perm と lam を使用) ---
        labels[current_mask] = self._mixup_target(masked_labels, perm, lam)

        return features, embeddings, labels

    def _mixup_data(self, data, perm, lam):
        """データ（features/embeddings）の混合のみを行うヘルパー関数"""
        return lam * data + (1 - lam) * data[perm, :]

    def _mixup_target(self, target, perm, lam):
        """ラベルの混合のみを行うヘルパー関数"""
        if self.mixup_type == "soft":
            mixed_target = torch.clamp(
                lam * target + (1 - lam) * target[perm, :],
                min=0,
                max=1,
            )
        elif self.mixup_type == "hard":
            # Hard mixupの実装意図が「足してクリップ」であれば元のまま
            mixed_target = torch.clamp(target + target[perm, :], min=0, max=1)
        else:
            raise NotImplementedError(
                f"mixup_label_type: {self.mixup_type} not implemented.",
            )
        return mixed_target


def cutmix(data, embeddings, target_c, target_f, alpha=1.0):
    """CutMix for Spectrogram AND BEATs Embeddings (Time-Synchronized).

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
        """基準(Spectrogram)のカット位置を、指定された時間長(total_time)に合わせてスケーリングしマスクを生成"""
        scale = total_time / ref_total_time

        # スケーリング後の開始位置と幅
        # round() で丸めることでズレを防ぐ
        scaled_start = (ref_start_ts.float() * scale).round().long()
        scaled_width = int(ref_cut_width * scale)

        # マスク作成
        arange = torch.arange(total_time, device=device).unsqueeze(0)  # [1, T]
        start_exp = scaled_start.unsqueeze(1)  # [B, 1]

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
    cut_ratio = mask_target.float().mean(dim=2).squeeze(1)  # [B]

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
            (mels.shape[0],),
            device=mels.device,
        ).reshape(-1, 1, 1) + snrs[1]
    else:
        snr = snrs

    snr = 10 ** (snr / 20)  # linear domain
    sigma = torch.std(mels, dim=dims, keepdim=True) / snr
    mels = mels + torch.randn(mels.shape, device=mels.device) * sigma

    return mels


def frame_shift(mels, labels, embeddings=None, shift_std=90, net_pooling=4):
    """Applies frame shift with zero-padding (Random Gaussian Shift).

    Args:
        mels: [B, n_mels, T]
        labels: [B, n_classes, T_label]
        embeddings: [B, dim, T_emb] or None

    """
    bsz, n_bands, frames = mels.shape

    # Embeddings情報の取得
    if embeddings is not None:
        emb_frames = embeddings.shape[-1]
    else:
        emb_frames = 0

    l_frames = labels.shape[-1] if labels is not None else 0

    # 出力用バッファ (ゼロ初期化)
    out_mels = torch.zeros_like(mels)
    out_labels = torch.zeros_like(labels) if labels is not None else None
    out_embs = torch.zeros_like(embeddings) if embeddings is not None else None

    for i in range(bsz):
        # 1. シフト量の決定 (Gaussian)
        shift = int(random.gauss(0, shift_std))

        # 2. Mel Shift
        if shift == 0:
            out_mels[i] = mels[i]
        elif shift > 0:
            if shift < frames:
                out_mels[i, :, shift:] = mels[i, :, :-shift]
        else:  # shift < 0
            shift_abs = abs(shift)
            if shift_abs < frames:
                out_mels[i, :, :-shift_abs] = mels[i, :, shift_abs:]

        # 3. Embeddings Shift (解像度に合わせてスケーリング)
        if embeddings is not None:
            scale_e = emb_frames / frames if frames > 0 else 0
            e_shift = int(shift * scale_e)

            if e_shift == 0:
                out_embs[i] = embeddings[i]
            elif e_shift > 0:
                if e_shift < emb_frames:
                    out_embs[i, :, e_shift:] = embeddings[i, :, :-e_shift]
            else:
                e_shift_abs = abs(e_shift)
                if e_shift_abs < emb_frames:
                    out_embs[i, :, :-e_shift_abs] = embeddings[i, :, e_shift_abs:]

        # 4. Label Shift (Poolingに合わせてスケーリング)
        if labels is not None:
            l_shift = int(shift / net_pooling)

            if l_shift == 0:
                out_labels[i] = labels[i]
            elif l_shift > 0:
                if l_shift < l_frames:
                    out_labels[i, :, l_shift:] = labels[i, :, :-l_shift]
            else:
                l_shift_abs = abs(l_shift)
                if l_shift_abs < l_frames:
                    out_labels[i, :, :-l_shift_abs] = labels[i, :, l_shift_abs:]

    return out_mels, out_labels, out_embs


def apply_synchronized_time_mask(mels, embeddings=None, mask_max=5):
    """Apply time masking to Mels and Embeddings synchronously.
    Unlike torchaudio.transforms.TimeMasking, this ensures both features
    are masked at the same physical time location.
    """
    B, F, T_mel = mels.shape
    masked_mels = mels.clone()
    masked_embs = embeddings.clone() if embeddings is not None else None

    # Embeddingsの時間スケール比率
    T_emb = embeddings.shape[-1] if embeddings is not None else 0
    scale = T_emb / T_mel if (embeddings is not None and T_mel > 0) else 0

    for i in range(B):
        # マスクの長さをランダム決定 (0 ~ mask_max)
        t_mask_len = random.randint(0, mask_max)
        if t_mask_len == 0 or t_mask_len >= T_mel:
            continue

        # マスク開始位置をランダム決定
        t_start = random.randint(0, T_mel - t_mask_len)

        # --- Mel Masking ---
        masked_mels[i, :, t_start : t_start + t_mask_len] = 0.0

        # --- Embeddings Masking (同期) ---
        if embeddings is not None:
            # 時間位置をEmbeddingsの解像度に変換
            e_start = int(t_start * scale)
            e_len = int(t_mask_len * scale)

            # Embeddingsの解像度が低い場合、e_lenが0になるのを防ぐか、
            # あるいは「Melでマスクされた領域に対応するEmb領域」を正確に消す
            if e_len > 0 and e_start < T_emb:
                e_end = min(e_start + e_len, T_emb)
                masked_embs[i, :, e_start:e_end] = 0.0

    return masked_mels, masked_embs


def SpecAugment(
    data,
    embeddings=None,
    target_c=None,
    target_f=None,
    frame_shift_std=90,
    time_mask_max=5,
    net_pooling=4,
):
    """Augmentation pipeline: Frame Shift -> Time Masking (Always applied).

    Args:
        data: [Batch, n_mels, time]
        embeddings: [Batch, dim, time_emb]
        target_f: Frame-level labels
        target_c: Clip-level labels (passed through)
        frame_shift_std: Std dev for Gaussian shift
        time_mask_max: Max length of time mask
        net_pooling: Pooling factor for label alignment

    """
    # 1. Frame Shift
    shifted_data, shifted_target_f, shifted_embeddings = frame_shift(
        data,
        target_f,
        embeddings=embeddings,
        shift_std=frame_shift_std,
        net_pooling=net_pooling,
    )

    # 2. Time Masking (Always applied, Synchronized)
    masked_data, masked_embeddings = apply_synchronized_time_mask(
        shifted_data,
        shifted_embeddings,
        mask_max=time_mask_max,
    )

    return masked_data, masked_embeddings, target_c, shifted_target_f
