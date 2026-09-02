"""数据集加载：HotpotQA / GSM8K / MATH / AIME。

划分方法（参考 openai_api_test 脚本，统一只用 train + val，无 test）：
    - 各数据集从其 train 池 shuffle 后切分：先取 n_val 做 val，再取 n_train 做 train
      （val 在前，train 数量变动不影响 val 集稳定性）
    - HotpotQA 用原生 train/dev 文件，dev 当 val
    - AIME 用 aimo-validation-aime 作为 train+val 池

判分逻辑在 sro/grading.py 内联实现（从 openai_api_test 移植的纯函数），
SRO 自包含。答案类型：
    gsm8k / aime / math → numeric（math 走 LaTeX 数值求值）
    hotpotqa            → freeform（SQuAD 风格归一化匹配）

数据集本地路径（相对 SRO 仓库根，可在 config 覆盖）：
    ../gsm8k/main/   ../MATH/   ../HotpotQA/raw/   ../AIME/
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Optional

from .config import get_config
from .grading import extract_boxed, get_judger
from .llm import TrainSample


# ---------------------------------------------------------------------------
# 切分工具
# ---------------------------------------------------------------------------


def _shuffle_split(items: list, n_train: int, n_val: int, seed: int):
    """shuffle 后先取 n_val 做 val，再取 n_train 做 train。返回 (train, val)。"""
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    n_val = min(n_val, len(items) // 2)
    n_train = min(n_train, len(items) - n_val)
    if n_val <= 0 or n_train <= 0:
        raise ValueError(
            f"Not enough data: total={len(items)}, n_train={n_train}, n_val={n_val}"
        )
    val = [items[i] for i in idx[:n_val]]
    train = [items[i] for i in idx[n_val:n_val + n_train]]
    return train, val


def _data_dir(name: str) -> Path:
    """取数据集根目录（config 可覆盖，默认 ../<name>/）。"""
    cfg = get_config()
    p = getattr(cfg, f"{name}_path", None)
    if p:
        return Path(p)
    # 默认：SRO 仓库的同级目录
    repo_root = Path(__file__).resolve().parent.parent
    default = {
        "gsm8k": repo_root.parent / "gsm8k" / "main",
        "math": repo_root.parent / "MATH",
        "hotpotqa": repo_root.parent / "HotpotQA" / "raw",
        "aime": repo_root.parent / "AIME",
    }
    return default[name]


# ---------------------------------------------------------------------------
# 四个 loader
# ---------------------------------------------------------------------------


def _load_gsm8k(n_train: int, n_val: int, seed: int):
    """GSM8K：本地 Parquet，答案在 answer 字段 #### 之后。"""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Loading GSM8K requires the datasets library: pip install datasets") from e
    d = _data_dir("gsm8k")
    train_path = d / "train-00000-of-00001.parquet"
    ds = load_dataset("parquet", data_files={"train": str(train_path)})["train"]
    items: list[TrainSample] = []
    for it in ds:
        raw = it.get("answer", "")
        # GSM8K answer 格式：解题过程 #### 最终数字
        ans = raw.split("####")[-1].strip().replace(",", "") if "####" in raw else raw.strip()
        items.append(TrainSample(problem=it["question"], answer=ans, answer_type="numeric"))
    return _shuffle_split(items, n_train, n_val, seed)


def _load_math(n_train: int, n_val: int, seed: int):
    """MATH：本地 JSON 目录，答案在 solution 的 \\boxed{} 里。"""
    d = _data_dir("math")
    items: list[TrainSample] = []
    for jsonf in (d / "train").rglob("*.json"):
        with open(jsonf, encoding="utf-8") as f:
            obj = json.load(f)
        ans = extract_boxed(obj.get("solution", ""))
        if not ans:
            continue
        items.append(TrainSample(problem=obj["problem"], answer=ans, answer_type="numeric"))
    return _shuffle_split(items, n_train, n_val, seed)


def _load_aime(n_train: int, n_val: int, seed: int):
    """AIME：本地 Arrow，aimo-validation-aime 作为 train+val 池。"""
    try:
        from datasets import Dataset
    except ImportError as e:
        raise ImportError("Loading AIME requires the datasets library: pip install datasets") from e
    d = _data_dir("aime")
    arrow = list((d / "AI-MO___aimo-validation-aime").rglob("*.arrow"))
    if not arrow:
        raise FileNotFoundError(f"AIME arrow not found under {d}")
    ds = Dataset.from_file(str(arrow[0]))
    items: list[TrainSample] = []
    for it in ds:
        problem = it.get("problem", it.get("question", ""))
        answer = str(it.get("answer", "")).strip()
        if problem and answer:
            items.append(TrainSample(problem=problem, answer=answer, answer_type="numeric"))
    return _shuffle_split(items, n_train, n_val, seed)


def _load_hotpotqa(n_train: int, n_val: int, seed: int):
    """HotpotQA：本地 JSON，train 文件做 train 池，dev 文件做 val 池。

    与其他数据集不同：HotpotQA 用原生 train/dev 划分，不混合 shuffle。
    """
    d = _data_dir("hotpotqa")
    train_path = d / "hotpot_train_v1.1.json"
    val_path = d / "hotpot_dev_distractor_v1.json"

    def _read(path, n):
        if n <= 0:
            raise ValueError(f"HotpotQA: n must be > 0, got n={n}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rng = random.Random(seed)
        rng.shuffle(data)
        out = [TrainSample(problem=item["question"], answer=item["answer"],
                           answer_type="freeform")
               for item in data[:n]]
        return out

    train = _read(train_path, n_train)
    val = _read(val_path, n_val)
    return train, val


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

_LOADERS = {
    "gsm8k": _load_gsm8k,
    "math": _load_math,
    "aime": _load_aime,
    "hotpotqa": _load_hotpotqa,
}


def load(dataset: str, n_train: int = 50, n_val: int = 30, seed: int = 42):
    """加载指定数据集，返回 (train, val): tuple[list[TrainSample], list[TrainSample]]。

    dataset: gsm8k / math / aime / hotpotqa
    """
    if dataset not in _LOADERS:
        raise ValueError(f"unknown dataset '{dataset}'; choose from {list(_LOADERS)}")
    return _LOADERS[dataset](n_train, n_val, seed)


# 兼容 engine.set_dataset 的旧接口
def _import_eval(dataset: str):
    """返回 (judger, None)，judger 来自 sro/grading。"""
    return get_judger(dataset), None
