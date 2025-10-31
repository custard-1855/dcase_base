#!/bin/bash

# uv run train_pretrained.py \
#     --attn_type normal\
#     --attn_deepen false \
#     --mixstyle_type mix2attn\
#     # --cmt \
#     # --sebbs \

uv run train_pretrained.py \
    --mixstyle_type moreMix\

uv run train_pretrained.py \
    --mixstyle_type mix2attn\