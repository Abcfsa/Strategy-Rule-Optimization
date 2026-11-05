"""数据集加载：HotpotQA / GSM8K / MATH / MATH-500 / AIME / IFBench / LiveBench / PUPA。

划分方法（train / val / test 三集，对齐 GEPA-original 协议）：
    - train：反思迭代的 rollout 池（minibatch 从中采样）
    - val：GEPA/NPO 模式下的接受判据与 Pareto 前沿评估集
    - test：优化结束后只评一次的最终报告集（Phase 0 基线也跑 test，直接可比）
    - 各数据集从其 train 池 shuffle 后切分：先取 n_val 做 val，再取 n_train 做 train，
      再取 n_test 做 test（val 在前，train 数量变动不影响 val 集稳定性；
      n_test=0 时 train/val 输出与旧版 _shuffle_split 完全一致）
    - HotpotQA 用原生 train/dev 文件：train 文件出 train，dev 文件出 val+test
    - GSM8K / MATH 的 test 用官方 test 文件（不参与 train/val 池）
    - AIME 用 aimo-validation-aime 作为 train+val 池；gepa_split 时 test 用
      MathArena___aime_2025（GEPA 协议：AIME 2025 的 30 题独立 test）
    - gepa_split=True 时对所有数据集生效：忽略 n_train/n_val/n_test/seed，
      改用 GEPA_SPLIT_SIZES 固定规模 + seed=0（详见 load() docstring）
    - MATH-500 / IFBench / LiveBench 是纯 test 集（无 train 标注），
      从池中切出 train+val+test 供 SRO 两阶段闭环使用

判分逻辑在 sro/grading.py 内联实现（从 openai_api_test 移植的纯函数），
SRO 自包含。答案类型：
    gsm8k / aime / math / math500 → numeric（math 走 LaTeX 数值求值）
    hotpotqa            → freeform（SQuAD 风格归一化匹配）
    ifbench / livebench → verifiable（完整回答逐约束 check_following）
    pupa                → redaction（完整输出 PII 泄漏检查）

verifiable / redaction 两类的 answer 字段不是单个答案串，而是结构化约束的
JSON 编码（见 grading.py 各 judger 的 gold 约定）；engine 对这两类不做答案
抽取，直接把模型完整 raw 交给 judger。

数据集本地路径（相对 SRO 仓库根，可在 config 覆盖）：
    ../gsm8k/main/   ../MATH/   ../MATH-500/   ../HotpotQA/raw/   ../AIME/
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


def _shuffle_split3(items: list, n_train: int, n_val: int, n_test: int,
                    seed: int):
    """shuffle 后依次切 val → train → test，返回 (train, val, test)。

    与旧版 _shuffle_split 的兼容性：n_test=0 时 train/val 的内容与顺序和
    旧版完全一致（同一 seed 下逐样本相同）。

    clamp 规则（池不足时按 train > val > test 优先级收缩）：
      - n_train = min(n_train, total)
      - n_val   = min(n_val,   total - n_train)
      - n_test  = min(n_test,  total - n_train - n_val)
      - 若请求了 test（n_test>0）但剩余池装不下：val/test 平分剩余（val 在前），
        保证 test 非空——test 为空会让三集协议失效，宁可 val 缩水。
    """
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    total = len(items)
    want_test = n_test > 0
    n_train = min(n_train, total)
    n_val = min(n_val, total - n_train)
    n_test = min(n_test, total - n_train - n_val)
    if want_test and n_test <= 0:
        rem = total - n_train
        if rem > 1:
            n_val = rem // 2
            n_test = rem - n_val
    if n_val <= 0 or n_train <= 0:
        raise ValueError(
            f"Not enough data: total={total}, n_train={n_train}, n_val={n_val}"
        )
    val = [items[i] for i in idx[:n_val]]
    train = [items[i] for i in idx[n_val:n_val + n_train]]
    test = [items[i] for i in idx[n_val + n_train:n_val + n_train + n_test]]
    return train, val, test


def _shuffle_split(items: list, n_train: int, n_val: int, seed: int):
    """旧接口兼容包装：返回 (train, val)。"""
    train, val, _ = _shuffle_split3(items, n_train, n_val, 0, seed)
    return train, val


def _take_first(items: list, n: int, seed: int) -> list:
    """shuffle(seed) 后取前 n 条（用于从官方 test 文件取固定规模 test 集）。"""
    if n <= 0:
        return []
    out = list(items)
    random.Random(seed).shuffle(out)
    return out[:min(n, len(out))]


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
        "math500": repo_root.parent / "MATH-500",
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


def _gsm8k_items(parquet_path: Path) -> list[TrainSample]:
    """读一个 GSM8K parquet 为 TrainSample 列表。"""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Loading GSM8K requires the datasets library: pip install datasets") from e
    ds = load_dataset("parquet", data_files={"train": str(parquet_path)})["train"]
    items: list[TrainSample] = []
    for it in ds:
        raw = it.get("answer", "")
        # GSM8K answer 格式：解题过程 #### 最终数字
        ans = raw.split("####")[-1].strip().replace(",", "") if "####" in raw else raw.strip()
        items.append(TrainSample(problem=it["question"], answer=ans, answer_type="numeric"))
    return items


def _load_gsm8k(n_train: int, n_val: int, n_test: int, seed: int):
    """GSM8K：train/val 从官方 train parquet 切；test 从官方 test parquet 取。"""
    d = _data_dir("gsm8k")
    train_items = _gsm8k_items(d / "train-00000-of-00001.parquet")
    train, val, _ = _shuffle_split3(train_items, n_train, n_val, 0, seed)
    test_path = d / "test-00000-of-00001.parquet"
    test: list[TrainSample] = []
    if n_test > 0:
        if not test_path.exists():
            raise FileNotFoundError(f"GSM8K test parquet not found: {test_path}")
        test = _take_first(_gsm8k_items(test_path), n_test, seed)
    return train, val, test


def _math_items(json_dir: Path) -> list[TrainSample]:
    """读 MATH JSON 目录为 TrainSample 列表（答案取 solution 的 \\boxed{}）。"""
    items: list[TrainSample] = []
    for jsonf in json_dir.rglob("*.json"):
        with open(jsonf, encoding="utf-8") as f:
            obj = json.load(f)
        ans = extract_boxed(obj.get("solution", ""))
        if not ans:
            continue
        items.append(TrainSample(problem=obj["problem"], answer=ans, answer_type="numeric"))
    return items


def _load_math(n_train: int, n_val: int, n_test: int, seed: int):
    """MATH：train/val 从官方 train 目录切；test 从官方 test 目录取。"""
    d = _data_dir("math")
    train_items = _math_items(d / "train")
    train, val, _ = _shuffle_split3(train_items, n_train, n_val, 0, seed)
    test: list[TrainSample] = []
    if n_test > 0:
        test_dir = d / "test"
        if not test_dir.is_dir():
            raise FileNotFoundError(f"MATH test dir not found: {test_dir}")
        test = _take_first(_math_items(test_dir), n_test, seed)
    return train, val, test


def _load_math500(n_train: int, n_val: int, n_test: int, seed: int):
    """MATH-500：OpenAI 评测子集（test.jsonl，500 题），answer 字段直接给最终答案。

    纯 test 集（无 train 标注），从池中切出 train+val+test 供 SRO 两阶段
    闭环使用。判分复用 evaluate_math（LaTeX 归一化 + 数值求值），与 MATH
    全集同一 judger。
    """
    d = _data_dir("math500")
    items: list[TrainSample] = []
    with open(d / "test.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ans = (obj.get("answer") or "").strip()
            if not obj.get("problem") or not ans:
                continue
            items.append(TrainSample(problem=obj["problem"], answer=ans,
                                     answer_type="numeric"))
    return _shuffle_split3(items, n_train, n_val, n_test, seed)


def _aime_pool_items() -> list[TrainSample]:
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


def _aime_2025_test_items() -> list[TrainSample]:
    """读 MathArena___aime_2025 Arrow（GEPA 协议的 AIME 独立 test 集，30 题）。"""
    try:
        from datasets import Dataset
    except ImportError as e:
        raise ImportError("Loading AIME-2025 requires the datasets library: pip install datasets") from e
    d = _data_dir("aime")
    arrow = list((d / "MathArena___aime_2025").rglob("*.arrow"))
    if not arrow:
        raise FileNotFoundError(
            f"AIME-2025 arrow not found under {d}/MathArena___aime_2025 "
            f"(required as GEPA-protocol test set for --gepa-split)")
    ds = Dataset.from_file(str(arrow[0]))
    items: list[TrainSample] = []
    for it in ds:
        problem = it.get("problem", it.get("question", ""))
        answer = str(it.get("answer", "")).strip()
        if problem and answer:
            items.append(TrainSample(problem=problem, answer=answer, answer_type="numeric"))
    return items


def _load_aime(n_train: int, n_val: int, n_test: int, seed: int,
               gepa_split: bool = False):
    """AIME：本地 Arrow，aimo-validation-aime 作为 train+val 池。

    gepa_split=True 时完全复刻 GEPA 的划分协议：
    - train/val：固定 random.Random(0) 打乱 aimo 池 + 对半切（45/45，忽略 seed）
    - test：MathArena___aime_2025 全部 30 题（GEPA 用 AIME 2025 做独立 test，
      论文中每题 5 次尝试取 pass@1——重复评估在 engine/main 层做，loader 只供题）
    - 三集答案统一加 "### " 前缀（GEPA 的答案格式）

    gepa_split=False 时：aimo 池 shuffle(seed) 切 train/val/test 三段。
    """
    items = _aime_pool_items()

    if gepa_split:
        # 复刻 GEPA init_dataset(): random.Random(0) 打乱 + 对半切 + ### 前缀
        random.Random(0).shuffle(items)
        mid = len(items) // 2
        train = items[:mid]
        val = items[mid:]
        test = _aime_2025_test_items()
        prefix = "### "
        for s in train + val + test:
            s.answer = prefix + s.answer
        return train, val, test

    return _shuffle_split3(items, n_train, n_val, n_test, seed)


def _load_hotpotqa(n_train: int, n_val: int, n_test: int, seed: int):
    """HotpotQA：train 文件做 train 池，dev 文件做 val+test 池。

    与其他数据集不同：HotpotQA 用原生 train/dev 划分，不混合 shuffle。
    dev（7,405 题）shuffle(seed) 后前 n_val 做 val、随后 n_test 做 test
    （GEPA 协议 150/300/300 即 train 文件取 150、dev 切 300+300）。
    """
    d = _data_dir("hotpotqa")
    train_path = d / "hotpot_train_v1.1.json"
    dev_path = d / "hotpot_dev_distractor_v1.json"

    def _read_all(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [TrainSample(problem=item["question"], answer=item["answer"],
                            answer_type="freeform")
                for item in data]

    train_items = _read_all(train_path)
    if n_train <= 0:
        raise ValueError(f"HotpotQA: n_train must be > 0, got n={n_train}")
    rng = random.Random(seed)
    rng.shuffle(train_items)
    train = train_items[:n_train]

    dev_items = _read_all(dev_path)
    rng_dev = random.Random(seed)
    rng_dev.shuffle(dev_items)
    if n_val <= 0:
        raise ValueError(f"HotpotQA: n_val must be > 0, got n={n_val}")
    val = dev_items[:n_val]
    test = dev_items[n_val:n_val + n_test] if n_test > 0 else []
    return train, val, test


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


def _load_ifbench(n_train: int, n_val: int, n_test: int, seed: int):
    """IFBench_test：单轮 prompt + 约束列表（OOD 约束为主）。

    answer_type=verifiable，answer=约束 gold JSON；problem 用 prompt。
    本地仅有 IFBench 发布集（GEPA 的 train/val 源 IF-RLVR Train 不在本地），
    故 train/val/test 都从发布集切；gepa_split 请求 150/300/294 时池不足，
    按 _shuffle_split3 的 clamp 自动降级（剩余池 val/test 平分）。
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
    return _shuffle_split3(items, n_train, n_val, n_test, seed)


def _load_livebench(n_train: int, n_val: int, n_test: int, seed: int):
    """LiveBench 指令跟随：单轮（turns[0]）+ 经典 IFEval 约束。

    400 条全为单轮；problem 用 turns[0]，answer_type=verifiable。
    gepa_split 时对齐 GEPA LiveBench-Math 协议：seed=0 三等分（~133/133/134）。
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
    return _shuffle_split3(items, n_train, n_val, n_test, seed)


def _load_pupa(n_train: int, n_val: int, n_test: int, seed: int,
               positional: bool = False):
    """PUPA 隐私脱敏：user_query 含 PII，要求输出脱敏后 query。

    answer_type=redaction，answer=gold JSON{pii_units, redacted_query}；
    problem 用含 PII 的 user_query。默认读 PUPA_New.csv（PUPA_PATH 可覆盖）。

    positional=True（GEPA 协议）：不 shuffle，按 csv 原始行序依次取
    n_train / n_val / n_test（GEPA 保留原始位置划分 111/111/221）。
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
    if positional:
        if n_train + n_val + n_test > len(items):
            raise ValueError(
                f"Not enough data: total={len(items)}, "
                f"need {n_train}+{n_val}+{n_test}")
        return (items[:n_train],
                items[n_train:n_train + n_val],
                items[n_train + n_val:n_train + n_val + n_test])
    return _shuffle_split3(items, n_train, n_val, n_test, seed)


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

_LOADERS = {
    "gsm8k": _load_gsm8k,
    "math": _load_math,
    "math500": _load_math500,
    "aime": _load_aime,
    "hotpotqa": _load_hotpotqa,
    "ifbench": _load_ifbench,
    "livebench": _load_livebench,
    "pupa": _load_pupa,
}


# ---------------------------------------------------------------------------
# GEPA 协议划分（gepa_split=True 时对所有数据集生效）
# ---------------------------------------------------------------------------

# GEPA-original (arXiv:2507.19457) 的 (train, val, test) 规模：
#   HotpotQA 150/300/300 · AIME 45/45/30(2025 独立 test) · IFBench 150/300/294
#   LiveBench-Math ~123/~123/~122（seed=0 三等分） · PUPA 111/111/221（原始位置划分）
# GSM8K / MATH / MATH-500 为 GEPA 未使用的数据集，无论文规格，
# 对齐 GEPA 量级取 150/300/300（math500 池仅 500，test 取剩余 50；
# livebench 按本地 400 条三等分 133/133/134）。
# 本地池不足时按 _shuffle_split3 的 clamp 自动收缩（ifbench 发布集仅 ~294 条，
# GEPA 的 train/val 源 IF-RLVR Train 不在本地，val/test 平分剩余池）。
GEPA_SPLIT_SIZES = {
    "gsm8k":     (150, 300, 300),
    "math":      (150, 300, 300),
    "math500":   (150, 300, 50),
    "hotpotqa":  (150, 300, 300),
    "ifbench":   (150, 300, 294),
    "livebench": (133, 133, 134),
    "pupa":      (111, 111, 221),
    # aime 走专属协议（对半切 + MathArena 2025 test），不走此表
}

GEPA_SPLIT_SEED = 0  # GEPA 固定划分 seed


def load(dataset: str, n_train: int = 50, n_val: int = 30, n_test: int = 150,
         seed: int = 42, gepa_split: bool = False):
    """加载指定数据集，返回 (train, val, test) 三元组。

    dataset: gsm8k / math / math500 / aime / hotpotqa / ifbench / livebench / pupa

    三集角色（对齐 GEPA-original 协议）：
      - train：反思迭代 rollout 池（minibatch 采样源）
      - val：GEPA/NPO 的接受判据与 Pareto 评估集
      - test：优化结束后只评一次的最终报告集

    gepa_split: True 时对**所有数据集**生效，复刻 GEPA 协议划分——忽略
    n_train / n_val / n_test / seed，改用 GEPA_SPLIT_SIZES 与 GEPA_SPLIT_SEED：
      - aime:      Random(0) 打乱 aimo 池对半切 45/45 + "### " 前缀；
                   test = MathArena___aime_2025 全部 30 题（GEPA 独立 test 协议）
      - pupa:      原始位置划分，不 shuffle，111/111/221
      - livebench: Random(0) 打乱后三等分 ~133/133/134（GEPA LiveBench 协议）
      - hotpotqa:  train 文件取 150；dev 文件 shuffle(0) 后 300 val + 300 test
      - gsm8k / math: train 池 shuffle(0) 切 150/300；官方 test 文件取 300
      - ifbench / math500: 纯 test 池 shuffle(0) 切三段（池不足自动 clamp）
    """
    if dataset not in _LOADERS:
        raise ValueError(f"unknown dataset '{dataset}'; choose from {list(_LOADERS)}")
    if gepa_split:
        if dataset == "aime":
            return _LOADERS[dataset](0, 0, 0, GEPA_SPLIT_SEED, gepa_split=True)
        nt, nv, nte = GEPA_SPLIT_SIZES[dataset]
        if dataset == "pupa":
            return _LOADERS[dataset](nt, nv, nte, GEPA_SPLIT_SEED, positional=True)
        return _LOADERS[dataset](nt, nv, nte, GEPA_SPLIT_SEED)
    return _LOADERS[dataset](n_train, n_val, n_test, seed)


# 兼容 engine.set_dataset 的旧接口
def _import_eval(dataset: str):
    """返回 (judger, None)，judger 来自 sro/grading。"""
    return get_judger(dataset), None
