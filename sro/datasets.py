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
import zlib
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


def _load_aime_items() -> list[TrainSample]:
    """读 aimo-validation-aime Arrow，构造样本列表（裸答案）。"""
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
    return items


def _load_aime_gepa_split() -> tuple:
    """完整复刻 GEPA init_dataset()：Random(0) 打乱 + 对半切 + "### " 前缀。

    固定协议（不依赖 seed / 配额）。混合模式下由 load_mixed 按配额
    从两半区各取前缀子集。
    """
    items = _load_aime_items()
    random.Random(0).shuffle(items)
    mid = len(items) // 2
    train, val = items[:mid], items[mid:]
    for s in train:
        s.answer = "### " + s.answer
    for s in val:
        s.answer = "### " + s.answer
    return train, val


def _load_aime(n_train: int, n_val: int, seed: int, gepa_split: bool = False):
    """AIME：本地 Arrow，aimo-validation-aime 作为 train+val 池。

    gepa_split=True 时完全复刻 GEPA 的 init_dataset() 划分：
    - 固定 random.Random(0) 打乱（忽略 seed 参数）
    - 对半切：前半=train，后半=val（忽略 n_train/n_val）
    - 答案加 "### " 前缀（GEPA 的答案格式）
    """
    if gepa_split:
        return _load_aime_gepa_split()
    return _shuffle_split(_load_aime_items(), n_train, n_val, seed)


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


def load_mixed(name: str, n_train: int = 50, n_val: int = 30,
               seed: int = 42, aime_gepa_split: bool = False) -> tuple:
    """加载混合数据集（"a+b" 形式），返回 (train, val)。

    各数据集独立划分（先分后混，保证各集内部 val 不与自己的 train 泄漏），
    n_train/n_val 均分给各数据集，样本打 dataset 标签，最后合并 shuffle
    （同 seed）。

    aime_gepa_split=True 时仅作用于 aime 成员：aime 侧先完整复刻 GEPA
    划分（Random(0) 对半切 + ### 前缀），再按配额从两半区各取前缀子集；
    其余成员照常 seed 划分，不受该 flag 影响。
    """
    names = [n.strip() for n in name.split("+") if n.strip()]
    if len(names) < 2:
        raise ValueError(f"mixed dataset needs >=2 names: {name!r}")
    unknown = [n for n in names if n not in _LOADERS]
    if unknown:
        raise ValueError(
            f"unknown dataset(s) {unknown}; choose from {list(_LOADERS)}")
    # 均分（有余数时前面的数据集多分 1 条，保证总量贴近请求）
    k = len(names)
    tr = [n_train // k + (1 if i < n_train % k else 0) for i in range(k)]
    vl = [n_val // k + (1 if i < n_val % k else 0) for i in range(k)]
    trains, vals = [], []
    for nm, nt, nv in zip(names, tr, vl):
        if nm == "aime" and aime_gepa_split:
            # GEPA 协议侧：固定 Random(0) 对半切 + ### 前缀，按配额取各半区前缀
            g_train, g_val = _load_aime_gepa_split()
            t, v = g_train[:nt], g_val[:nv]
        else:
            # 各数据集用独立 seed 派生（避免同 seed 下各集打乱顺序耦合）。
            # crc32 跨进程稳定（内置 hash() 因 PYTHONHASHSEED 随机化，会破坏
            # 跨进程/对照实验的可复现性）
            sub_seed = seed + (zlib.crc32(nm.encode()) % 10000)
            t, v = _LOADERS[nm](nt, nv, sub_seed)
        for s in t:
            s.dataset = nm
        for s in v:
            s.dataset = nm
        trains.extend(t)
        vals.extend(v)
        print(f"  [mixed] {nm}: train {len(t)} | val {len(v)}")
    random.Random(seed).shuffle(trains)
    random.Random(seed + 1).shuffle(vals)
    return trains, vals


def load(dataset: str, n_train: int = 50, n_val: int = 30, seed: int = 42,
         gepa_split: bool = False):
    """加载指定数据集，返回 (train, val): tuple[list[TrainSample], list[TrainSample]]。

    dataset: gsm8k / math / aime / hotpotqa，或混合 "a+b"（如 gsm8k+hotpotqa；
    n_train/n_val 均分给各数据集）

    gepa_split: aime 专属 GEPA 协议（seed=0 + 对半切 + ### 前缀）。单数据集
    aime 时作用于全部数据；混合模式下仅作用于 aime 成员，其余成员不受影响。
    """
    if "+" in dataset:
        return load_mixed(dataset, n_train, n_val, seed,
                          aime_gepa_split=gepa_split)
    if dataset not in _LOADERS:
        raise ValueError(f"unknown dataset '{dataset}'; choose from {list(_LOADERS)}")
    if dataset == "aime":
        return _LOADERS[dataset](n_train, n_val, seed, gepa_split=gepa_split)
    return _LOADERS[dataset](n_train, n_val, seed)


# 兼容 engine.set_dataset 的旧接口
def _import_eval(dataset: str):
    """返回 (judger, None)，judger 来自 sro/grading。"""
    return get_judger(dataset), None
