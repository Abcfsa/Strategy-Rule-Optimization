#!/usr/bin/env bash
# 训练期示例（占位，算法实现后可用）：
# AIME 上自进化，产出三层记忆到 outputs/<timestamp>/
cd "$(dirname "$0")/.." && python main.py train \
  --dataset aime \
  --config configs/default.yaml \
  --seed 42
