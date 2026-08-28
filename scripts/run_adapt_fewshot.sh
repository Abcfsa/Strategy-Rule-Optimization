#!/usr/bin/env bash
# 测试期快速适应示例（占位）：
# 给定 10 个带反馈适应样本，走 §4.5 协议后评测
cd "$(dirname "$0")/.." && python main.py adapt \
  --dataset aime25 \
  --mode few-shot \
  -n 10 \
  --strategy-dir "$1" \
  --config configs/default.yaml
