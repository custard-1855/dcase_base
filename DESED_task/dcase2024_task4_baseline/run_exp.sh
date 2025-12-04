#!/bin/bash

# 実験設定
BASE_WANDB_DIR="150/MixStyle_freq"
MIXSTYLE_TYPE="resMix"
ATTN_TYPE="default"  # 基本的なattentionタイプを使用
ATTN_DEEPEN=2        # デフォルトの深さ


# ログディレクトリの作成
LOG_DIR="logs/cmt"
mkdir -p ${LOG_DIR}

# タイムスタンプ
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=========================================="
echo "Accuracy Improvement Experiments"
echo "Start Time: $(date)"
echo "=========================================="
echo ""


# only CMT


# ################################################################################
# # 実験1: Nomal
# ################################################################################
# uv run train_pretrained.py \
#     --wandb_dir ${BASE_WANDB_DIR}/Mixup_fix \
#     2>&1 | tee ${LOG_DIR}/75${TIMESTAMP}.log

# echo ""

################################################################################
# 実験3: CMT neg
################################################################################
uv run train_pretrained.py \
    --wandb_dir ${BASE_WANDB_DIR}/use_neg_sample \
    --cmt \
    --use_neg_sample \
    2>&1 | tee ${LOG_DIR}/${TIMESTAMP}.log

echo ""


# ################################################################################
# # 実験1: MixStyle
# ################################################################################
# uv run train_pretrained.py \
#     --wandb_dir ${BASE_WANDB_DIR}/\
#     --attn_type ${ATTN_TYPE} \
#     --attn_deepen ${ATTN_DEEPEN} \
#     --mixstyle_type ${MIXSTYLE_TYPE} \
#     2>&1 | tee ${LOG_DIR}/${TIMESTAMP}.log

# echo ""


# ################################################################################
# # 実験2: MixStyle + CMT
# ################################################################################
# uv run train_pretrained.py \
#     --wandb_dir ${BASE_WANDB_DIR}/+CMT_neg \
#     --attn_type ${ATTN_TYPE} \
#     --attn_deepen ${ATTN_DEEPEN} \
#     --mixstyle_type ${MIXSTYLE_TYPE} \
#     --cmt \
#     --use_neg_sample \
#     2>&1 | tee ${LOG_DIR}/${TIMESTAMP}.log

# echo ""


# ################################################################################
# # 実験2: MixStyle + SEBBs
# ################################################################################
# uv run train_pretrained.py \
#     --wandb_dir ${BASE_WANDB_DIR}/+SEBBs \
#     --attn_type ${ATTN_TYPE} \
#     --attn_deepen ${ATTN_DEEPEN} \
#     --mixstyle_type ${MIXSTYLE_TYPE} \
#     --sebbs \
#     2>&1 | tee ${LOG_DIR}/${TIMESTAMP}.log

# echo ""


# ################################################################################
# # 実験2: MixStyle + CMT + SEBBs
# ################################################################################
# uv run train_pretrained.py \
#     --wandb_dir ${BASE_WANDB_DIR}/+CMT_neg+SEBBs \
#     --attn_type ${ATTN_TYPE} \
#     --attn_deepen ${ATTN_DEEPEN} \
#     --mixstyle_type ${MIXSTYLE_TYPE} \
#     --cmt \
#     --use_neg_sample \
#     --sebbs \
#     2>&1 | tee ${LOG_DIR}/${TIMESTAMP}.log

# echo ""


# ################################################################################
# # 実験4: CMT + SEBBs
# ################################################################################
# uv run train_pretrained.py \
#     --wandb_dir ${BASE_WANDB_DIR}/CMT_frame-0.5_warm-up-50 + SEBBs \
#     --cmt \
#     --phi_frame 0.5 \
#     --warmup_epochs 50 \
#     --sebbs \
#     2>&1 | tee ${LOG_DIR}/${TIMESTAMP}.log

# echo ""


# ################################################################################
# # 実験4: SEBBs
# ################################################################################
# uv run train_pretrained.py \
#     --wandb_dir ${BASE_WANDB_DIR}/SEBBs \
#     --sebbs \
#     2>&1 | tee ${LOG_DIR}/${TIMESTAMP}.log

# echo ""



# ################################################################################
# # 実験5: sat  only time mask 50
# ################################################################################
# echo "[1/4] Running: sat"
# uv run train_pretrained.py \
#     --wandb_dir sat/150/only-time-mask-50 \
#     --sat \
#     --strong_augment_type frame_shift_time_mask \
#     --frame_shift_std 0 \
#     --time_mask_max 50 \
#     2>&1 | tee ${LOG_DIR}/only_sat${TIMESTAMP}.log

# echo ""

# ################################################################################
# # 実験6: sat frame shift + time mask
# ################################################################################
# echo "[1/4] Running: sat"
# uv run train_pretrained.py \
#     --wandb_dir sat/150/frame-shift-60_time-mask-30 \
#     --sat \
#     --strong_augment_type frame_shift_time_mask \
#     --frame_shift_std 60 \
#     --time_mask_max 30 \
#     2>&1 | tee ${LOG_DIR}/only_sat${TIMESTAMP}.log

# echo ""

