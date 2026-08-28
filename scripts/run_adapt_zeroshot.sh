#!/usr/bin/env bash
# 测试期零适应迁移示例（占位）：
# 用 train 产出的三层记忆，在 aime_2025 上仅用策略层 + 检索组装评测
cd "$(dirname "$0")/.." && python main.py adapt \
  --dataset aime25 \
  --mode zero-shot \
  --strategy-dir "$1" \
  --config configs/default.yaml
