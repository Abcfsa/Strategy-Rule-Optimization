"""数据集加载：HotpotQA / GSM8K / MATH / AIME / IFBench / LiveBench / PUPA。

划分方法（参考 openai_api_test 脚本，统一只用 train + val，无 test）：
    - 各数据集从其 train 池 shuffle 后切分：先取 n_val 做 val，再取 n_train 做 train
      （val 在前，train 数量变动不影响 val 集稳定性）
    - HotpotQA 用原生 train/dev 文件，dev 当 val
    - AIME 用 aimo-validation-aime 作为 train+val 池
    - IFBench / LiveBench 是纯 test 集（无 train 标注），按同一 _shuffle_split
      从 test 池切出 train+val 供 SRO 两阶段闭环使用

判分逻辑在 sro/grading.py 内联实现（从 openai_api_test 移植的纯函数），
SRO 自包含。答案类型：
    gsm8k / aime / math → numeric（math 走 LaTeX 数值求值）
    hotpotqa            → freeform（SQuAD 风格归一化匹配）
    ifbench / livebench → verifiable（完整回答逐约束 check_following）
    pupa                → redaction（完整输出 PII 泄漏检查）

verifiable / redaction 两类的 answer 字段不是单个答案串，而是结构化约束的
JSON 编码（见 grading.py 各 judger 的 gold 约定）；engine 对这两类不做答案
抽取，直接把模型完整 raw 交给 judger。

数据集本地路径（相对 SRO 仓库根，可在 config 覆盖）：
    ../gsm8k/main/   ../MATH/   ../HotpotQA/raw/   ../AIME/
    ../IFBench/IFBench_test/   ../livebench_instruction_following/
    ../PAPILLON/pupa/
"""

from __future__ import annotations

import csv
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
        "ifbench": repo_root.parent / "IFBench" / "IFBench_test",
        "livebench": repo_root.parent / "livebench_instruction_following",
        "pupa": repo_root.parent / "PAPILLON" / "pupa",
    }
    return default[name]


# ---------------------------------------------------------------------------
# QA / 数学类 loader（gsm8k / math / aime / hotpotqa）
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


def _load_aime(n_train: int, n_val: int, seed: int, gepa_split: bool = False):
    """AIME：本地 Arrow，aimo-validation-aime 作为 train+val 池。

    gepa_split=True 时完全复刻 GEPA 的 init_dataset() 划分：
    - 固定 random.Random(0) 打乱（忽略 seed 参数）
    - 对半切：前半=train，后半=val（忽略 n_train/n_val）
    - 答案加 "### " 前缀（GEPA 的答案格式）
    """
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

    if gepa_split:
        # 复刻 GEPA init_dataset(): random.Random(0) 打乱 + 对半切 + ### 前缀
        import random as _r
        _r.Random(0).shuffle(items)
        mid = len(items) // 2
        train = items[:mid]
        val = items[mid:]
        prefix = "### "
        for t in train:
            t.answer = prefix + t.answer
        for t in val:
            t.answer = prefix + t.answer
        return train, val

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
# 指令跟随 / 隐私脱敏 loader
# ---------------------------------------------------------------------------


def _read_parquet_rows(path: Path) -> list:
    """读 parquet 为 list[dict]。优先 datasets 库，回退 pyarrow，都没有则报错。"""
    try:
        from datasets import load_dataset
        # 显式把文件映射到 "train" 键，绕过 datasets 按文件名推断 split 的行为
        # ——livebench 的 parquet 名为 test-*.parquet，若按名推断会被判为 test
        # split，split="train" 随即报错（对齐 _load_gsm8k 的成熟写法）。
        ds = load_dataset("parquet", data_files={"train": str(path)})["train"]
        return [dict(r) for r in ds]
    except ImportError:
        pass
    try:
        import pyarrow.parquet as pq
        return pq.read_table(str(path)).to_pylist()
    except ImportError as e:
        raise ImportError(
            f"Reading {path.name} needs 'datasets' or 'pyarrow': "
            f"pip install datasets  (or: pip install pyarrow)"
        ) from e


def _encode_if_gold(prompt: str, id_list, kwargs_list) -> str:
    """把 IF 约束编码为 gold JSON（grading._evaluate_if_style 解析）。

    kwargs 转 dict 列表并【剔除 None 值】：parquet 里每条 kwargs 带全量字段
    （大多为 null），不剔除会让 gold JSON 膨胀到 1-5KB 并原样注入反思 prompt，
    烧 token 且稀释诊断信号。judger 侧的 None 过滤保留作防御。
    """
    return json.dumps({
        "prompt": prompt,
        "instruction_id_list": list(id_list),
        "kwargs": [{k: v for k, v in dict(kw).items() if v is not None}
                   for kw in kwargs_list],
    }, ensure_ascii=False)


def _load_ifbench(n_train: int, n_val: int, seed: int):
    """IFBench_test：单轮 prompt + 约束列表（OOD 约束为主）。

    answer_type=verifiable，answer=约束 gold JSON；problem 用 prompt。
    """
    d = _data_dir("ifbench")
    pq_path = d / "data" / "train-00000-of-00001.parquet"
    if not pq_path.exists():
        raise FileNotFoundError(f"IFBench parquet not found: {pq_path}")
    rows = _read_parquet_rows(pq_path)
    items: list[TrainSample] = []
    for it in rows:
        prompt = it.get("prompt", "")
        ids = it.get("instruction_id_list", [])
        kwargs = it.get("kwargs", [])
        if not prompt or not ids:
            continue
        # ids 与 kwargs 必须一一对应（错位会让 judger 抛环境错，加载期就拦下）
        if len(ids) != len(kwargs):
            raise ValueError(
                f"IFBench row key={it.get('key')!r}: {len(ids)} instruction ids "
                f"vs {len(kwargs)} kwargs — data is malformed"
            )
        items.append(TrainSample(
            problem=prompt,
            answer=_encode_if_gold(prompt, ids, kwargs),
            answer_type="verifiable",
        ))
    return _shuffle_split(items, n_train, n_val, seed)


def _load_livebench(n_train: int, n_val: int, seed: int):
    """LiveBench 指令跟随：单轮（turns[0]）+ 经典 IFEval 约束。

    400 条全为单轮；problem 用 turns[0]，answer_type=verifiable。
    """
    d = _data_dir("livebench")
    pq_path = d / "data" / "test-00000-of-00001.parquet"
    if not pq_path.exists():
        raise FileNotFoundError(
            f"LiveBench parquet not found: {pq_path} "
            f"(if it is a git-lfs pointer, run `git lfs pull` in {d})"
        )
    rows = _read_parquet_rows(pq_path)
    items: list[TrainSample] = []
    for it in rows:
        turns = it.get("turns") or []
        prompt = turns[0] if turns else it.get("task_prompt", "")
        ids = it.get("instruction_id_list", [])
        kwargs = it.get("kwargs", [])
        if not prompt or not ids:
            continue
        if len(ids) != len(kwargs):
            raise ValueError(
                f"LiveBench row question_id={it.get('question_id')!r}: "
                f"{len(ids)} instruction ids vs {len(kwargs)} kwargs — data is malformed"
            )
        items.append(TrainSample(
            problem=prompt,
            answer=_encode_if_gold(prompt, ids, kwargs),
            answer_type="verifiable",
        ))
    return _shuffle_split(items, n_train, n_val, seed)


def _load_pupa(n_train: int, n_val: int, seed: int):
    """PUPA 隐私脱敏：user_query 含 PII，要求输出脱敏后 query。

    answer_type=redaction，answer=gold JSON{pii_units, redacted_query}；
    problem 用含 PII 的 user_query。默认读 PUPA_New.csv（PUPA_PATH 可覆盖）。
    """
    d = _data_dir("pupa")
    cfg = get_config()
    # PUPA_PATH 可指到目录（取其下 PUPA_New.csv）或直接指到某个 csv
    # （如 PUPA_TNB.csv）；都未设时默认 <pupa 目录>/PUPA_New.csv
    if cfg.pupa_path:
        csv_path = Path(cfg.pupa_path)
        if csv_path.is_dir():
            csv_path = csv_path / "PUPA_New.csv"
    else:
        csv_path = d / "PUPA_New.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"PUPA csv not found: {csv_path} (set PUPA_PATH to a csv file or "
            f"the pupa directory)"
        )
    items: list[TrainSample] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            query = (row.get("user_query") or "").strip()
            pii_raw = (row.get("pii_units") or "").strip()
            redacted = (row.get("redacted_query") or "").strip()
            if not query or not pii_raw:
                continue
            pii_units = [p.strip() for p in pii_raw.split("||") if p.strip()]
            items.append(TrainSample(
                problem=query,
                answer=json.dumps({"pii_units": pii_units,
                                   "redacted_query": redacted},
                                  ensure_ascii=False),
                answer_type="redaction",
            ))
    return _shuffle_split(items, n_train, n_val, seed)


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

_LOADERS = {
    "gsm8k": _load_gsm8k,
    "math": _load_math,
    "aime": _load_aime,
    "hotpotqa": _load_hotpotqa,
    "ifbench": _load_ifbench,
    "livebench": _load_livebench,
    "pupa": _load_pupa,
}


def load(dataset: str, n_train: int = 50, n_val: int = 30, seed: int = 42,
         gepa_split: bool = False):
    """加载指定数据集，返回 (train, val): tuple[list[TrainSample], list[TrainSample]]。

    dataset: gsm8k / math / aime / hotpotqa / ifbench / livebench / pupa

    gepa_split: 仅 aime 生效，完全复刻 GEPA init_dataset() 划分（seed=0 + 对半切 + ### 前缀）。
    """
    if dataset not in _LOADERS:
        raise ValueError(f"unknown dataset '{dataset}'; choose from {list(_LOADERS)}")
    if dataset == "aime":
        return _LOADERS[dataset](n_train, n_val, seed, gepa_split=gepa_split)
    return _LOADERS[dataset](n_train, n_val, seed)


# 兼容 engine.set_dataset 的旧接口
def _import_eval(dataset: str):
    """返回 (judger, None)，judger 来自 sro/grading。"""
    return get_judger(dataset), None
