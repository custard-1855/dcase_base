"""Confident Mean Teacher (CMT) の擬似ラベル生成, 信頼度の重み, 一貫性損失.

各関数の `self` は SEDTask4 (`phi_*`, `use_neg_sample`, `pos_neg_scale`, `median_filter`, `log` を使う).
"""

import numpy as np
import torch
import torch.nn.functional as F


def apply_cmt_postprocessing(
    self,
    y_w: torch.Tensor,
    y_s: torch.Tensor,
    index_weak: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Confident Mean Teacher postprocessing only unlabeled data.

    y_w: (batch, classes)
    y_s: (batch, classes, frames)
    """
    y_w_unlabeled = y_w[index_weak:]  # (unlabeled_batch, classes)
    y_s_unlabeled = y_s[index_weak:]  # (unlabeled_batch, classes, frames)

    if self.use_neg_sample:
        # Weak (clip-level)
        y_tilde_w = torch.where(
            y_w_unlabeled >= self.phi_pos,
            torch.ones_like(y_w_unlabeled),  # 正例 → 1.0
            torch.where(
                y_w_unlabeled <= self.phi_neg,
                torch.zeros_like(y_w_unlabeled),  # 負例 → 0.0
                y_w_unlabeled,  # 中間値 → そのまま（後で信頼度0）
            ),
        )

        # Strong (frame-level) - Weak制約なし、独立判定
        y_s_binary = torch.where(
            y_s_unlabeled >= self.phi_pos,
            torch.ones_like(y_s_unlabeled),  # 正例 → 1.0
            torch.where(
                y_s_unlabeled <= self.phi_neg,
                torch.zeros_like(y_s_unlabeled),  # 負例 → 0.0
                y_s_unlabeled,  # 中間値 → そのまま（後で信頼度0）
            ),
        )

    else:
        y_tilde_w = (y_w_unlabeled > self.phi_clip).float()  # clip
        y_w_expanded = y_tilde_w.unsqueeze(-1).expand_as(y_s_unlabeled)
        y_s_binary = y_w_expanded * (y_s_unlabeled > self.phi_frame).float()  # frame

    y_tilde_s_list = []
    original_device = y_s.device
    y_s_numpy = y_s_binary.detach().cpu().numpy()  # y_s_binary: (batch, classes, frames)

    for i in range(y_s_unlabeled.shape[0]):
        sample = y_s_numpy[i].transpose(1, 0)  # (classes, frames) -> (frames, classes)
        filtered = self.median_filter(sample)
        y_tilde_s_list.append(filtered)

    y_tilde_s_np = np.stack(y_tilde_s_list, axis=0)  # (batch, frames, classes)
    y_tilde_s_tensor = torch.from_numpy(y_tilde_s_np).to(original_device)
    y_tilde_s = y_tilde_s_tensor.transpose(1, 2)  # -> (batch, classes, frames)

    y_tilde_w = torch.cat((y_w[:index_weak], y_tilde_w), dim=0)
    y_tilde_s = torch.cat((y_s[:index_weak], y_tilde_s), dim=0)

    return y_tilde_w, y_tilde_s


def compute_cmt_confidence_weights(
    self,
    y_w: torch.Tensor,
    y_s: torch.Tensor,
    y_tilde_w: torch.Tensor,
    y_tilde_s: torch.Tensor,
    index_weak: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute confidence weights based on teacher's certainty.

    Correction: High confidence for both positive (near 1) and negative (near 0) predictions.
    """
    if self.use_neg_sample:
        # --- use neg_sample ---
        # Unlabeled部分のマスク生成
        y_w_unlabeled = y_w[index_weak:]
        y_s_unlabeled = y_s[index_weak:]

        pos_mask_w_unlabeled = (y_tilde_w[index_weak:] == 1.0).float()
        neg_mask_w_unlabeled = (y_tilde_w[index_weak:] == 0.0).float()

        pos_mask_s_unlabeled = (y_tilde_s[index_weak:] == 1.0).float()
        neg_mask_s_unlabeled = (y_tilde_s[index_weak:] == 0.0).float()

        # Unlabeled部分の信頼度計算
        c_w_pos_unlabeled = pos_mask_w_unlabeled * y_w_unlabeled
        c_w_neg_unlabeled = neg_mask_w_unlabeled * (1.0 - y_w_unlabeled)
        c_w_unlabeled = c_w_pos_unlabeled + c_w_neg_unlabeled

        c_s_pos_unlabeled = pos_mask_s_unlabeled * y_s_unlabeled
        c_s_neg_unlabeled = neg_mask_s_unlabeled * (1.0 - y_s_unlabeled)
        c_s_unlabeled = c_s_pos_unlabeled + c_s_neg_unlabeled

        # Labeled部分は常に信頼度1.0
        c_w_labeled = torch.ones_like(y_w[:index_weak])
        c_s_labeled = torch.ones_like(y_s[:index_weak])

        # 結合（pos/negは後でスケーリングに使用）
        c_w_pos = torch.cat((c_w_labeled, c_w_pos_unlabeled), dim=0)
        c_w_neg = torch.cat((torch.zeros_like(y_w[:index_weak]), c_w_neg_unlabeled), dim=0)
        c_w = torch.cat((c_w_labeled, c_w_unlabeled), dim=0)

        c_s_pos = torch.cat((c_s_labeled, c_s_pos_unlabeled), dim=0)
        c_s_neg = torch.cat((torch.zeros_like(y_s[:index_weak]), c_s_neg_unlabeled), dim=0)
        c_s = torch.cat((c_s_labeled, c_s_unlabeled), dim=0)

    else:
        # --- CMT ---
        # Unlabeled部分のみ計算（labeled部分はy_tildeが元のy_wなので除外）
        y_w_unlabeled = y_w[index_weak:]
        y_s_unlabeled = y_s[index_weak:]
        y_tilde_w_unlabeled = y_tilde_w[index_weak:]
        y_tilde_s_unlabeled = y_tilde_s[index_weak:]

        c_w_unlabeled = y_w_unlabeled * y_tilde_w_unlabeled
        y_w_expanded = y_w_unlabeled.unsqueeze(-1).expand_as(y_s_unlabeled)
        c_s_unlabeled = y_s_unlabeled * y_w_expanded * y_tilde_s_unlabeled

        # Labeled部分は常に信頼度1.0
        c_w_labeled = torch.ones_like(y_w[:index_weak])
        c_s_labeled = torch.ones_like(y_s[:index_weak])

        # 結合
        c_w = torch.cat((c_w_labeled, c_w_unlabeled), dim=0)
        c_s = torch.cat((c_s_labeled, c_s_unlabeled), dim=0)

    # 実装予定
    # 温度パラメータによる調整

    if self.use_neg_sample and self.pos_neg_scale:  # 正例と負例の量を調整
        # Unlabeled部分のみの正例と負例の総重みを計算
        pos_sum_w_unlabeled = c_w_pos_unlabeled.sum().clamp(min=1e-6)
        neg_sum_w_unlabeled = c_w_neg_unlabeled.sum().clamp(min=1e-6)

        pos_sum_s_unlabeled = c_s_pos_unlabeled.sum().clamp(min=1e-6)
        neg_sum_s_unlabeled = c_s_neg_unlabeled.sum().clamp(min=1e-6)

        # Unlabeled部分のみをスケーリング
        if pos_sum_w_unlabeled < neg_sum_w_unlabeled:
            # 負例が多い → 負例を縮小
            scale_factor = neg_sum_w_unlabeled / pos_sum_w_unlabeled
            c_w_unlabeled_scaled = c_w_pos_unlabeled + c_w_neg_unlabeled / scale_factor
        else:
            # 正例が多い → 正例を縮小
            scale_factor = pos_sum_w_unlabeled / neg_sum_w_unlabeled
            c_w_unlabeled_scaled = c_w_pos_unlabeled / scale_factor + c_w_neg_unlabeled

        if pos_sum_s_unlabeled < neg_sum_s_unlabeled:
            scale_factor = neg_sum_s_unlabeled / pos_sum_s_unlabeled
            c_s_unlabeled_scaled = c_s_pos_unlabeled + c_s_neg_unlabeled / scale_factor
        else:
            scale_factor = pos_sum_s_unlabeled / neg_sum_s_unlabeled
            c_s_unlabeled_scaled = c_s_pos_unlabeled / scale_factor + c_s_neg_unlabeled

        # Labeledとスケール済みUnlabeledを結合
        c_w = torch.cat((c_w_labeled, c_w_unlabeled_scaled), dim=0)
        c_s = torch.cat((c_s_labeled, c_s_unlabeled_scaled), dim=0)

        # ログを更新（スケール前の値を記録）
        self.log("train/cmt/pos_neg_ratio_weak", pos_sum_w_unlabeled / neg_sum_w_unlabeled)
        self.log("train/cmt/pos_neg_ratio_strong", pos_sum_s_unlabeled / neg_sum_s_unlabeled)

    return c_w, c_s


def compute_cmt_consistency_loss(
    self,
    student_w: torch.Tensor,
    student_s: torch.Tensor,
    teacher_pseudo_w: torch.Tensor,
    teacher_pseudo_s: torch.Tensor,
    confidence_w: torch.Tensor,
    confidence_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute weighted BCE loss."""
    # Weak Loss
    bce_w = F.binary_cross_entropy(student_w, teacher_pseudo_w, reduction="none")
    weighted_bce_w = confidence_w * bce_w
    loss_w_con = weighted_bce_w.mean()

    # Strong Loss
    bce_s = F.binary_cross_entropy(student_s, teacher_pseudo_s, reduction="none")
    weighted_bce_s = confidence_s * bce_s
    loss_s_con = weighted_bce_s.mean()

    self.log("train/cmt/confidence_weak_mean", confidence_w.mean())
    self.log("train/cmt/confidence_weak_std", confidence_w.std())
    self.log("train/cmt/confidence_strong_mean", confidence_s.mean())
    self.log("train/cmt/confidence_strong_std", confidence_s.std())

    return loss_w_con, loss_s_con
