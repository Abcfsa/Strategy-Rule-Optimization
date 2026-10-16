"""答案判分函数（按数据集）。

从 openai_api_test 移植的纯字符串/数值处理函数，使 SRO 自包含，
不触发 openai_api_test 顶层对 openai 库的 import。

- math    : extract_boxed + LaTeX 归一化 + AST 数值求值
- gsm8k   : 数字清理（整数/小数/分数）
- aime    : 0-999 整数清理
- hotpotqa: SQuAD 风格文本归一化 + 词边界匹配
- ifbench / livebench: 可验证指令跟随（移植自 IFBench 官方 strict 判分，
                       对完整回答逐约束 check_following，全过才算对）
- pupa    : 隐私脱敏（PII 单元泄漏检查 + 输出有效性）

IF/PUPA 的 gold 不是单个答案串，而是结构化约束（约束列表 / PII 单元），
统一编码为 JSON 字符串放进 TrainSample.answer，judger 内部解析——这样不改
TrainSample / engine 的两参数 judger 接口 (prediction, gold)。
"""

from __future__ import annotations

import ast
import json
import math as _math
import operator
import re
import string as _string


# ---------------------------------------------------------------------------
# 通用：boxed 提取
# ---------------------------------------------------------------------------


def extract_boxed(text: str) -> str:
    """从文本中提取 \\boxed{...} 的内容，正确处理嵌套花括号。

    使用最后一个 \\boxed（通常是最终答案）。
    """
    idx = text.rfind("\\boxed")
    if idx == -1:
        return ""
    start = text.find("{", idx)
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i].strip()
    return text[start + 1:].strip()


# ---------------------------------------------------------------------------
# MATH
# ---------------------------------------------------------------------------


def _math_normalize_answer(answer: str) -> str:
    """标准化 MATH 答案以便比较（保留结构，仅统一格式）。"""
    a = answer.strip().strip('$')
    a = re.sub(r'\\[,;:! ]', '', a)
    a = re.sub(r'\\(left|right)', '', a)
    a = re.sub(r'\\[dt]frac', r'\\frac', a)
    a = re.sub(r'\\text\{([^{}]*)\}', r'\1', a)
    a = re.sub(r'\\(?:mathrm|operatorname)\{([^{}]*)\}', r'\1', a)
    a = re.sub(r'\s+', ' ', a)
    return a.lower().strip()


def _simplify_latex_for_eval(s: str) -> str:
    """把常见 LaTeX 答案形式转成可计算的纯文本。"""
    s = _math_normalize_answer(s)
    s = s.replace(' ', '')
    frac_pat = r'\\frac\{([^{}]+)\}\{([^{}]+)\}'
    sqrt_pat = r'\\sqrt\{([^{}]+)\}'
    for _ in range(10):
        new_s = re.sub(sqrt_pat, r'sqrt(\1)', s)
        new_s = re.sub(frac_pat, r'(\1)/(\2)', new_s)
        if new_s == s:
            break
        s = new_s
    s = re.sub(r'\\sqrt\[([^\]]*)\]\{([^{}]+)\}', r'(\2)**(1/(\1))', s)
    s = s.replace('\\pi', 'pi')
    s = s.replace('{', '').replace('}', '')
    s = re.sub(r'\\[a-zA-Z]+', '', s)
    s = re.sub(r'(\d)([a-z(])', r'\1*\2', s)
    s = s.replace('^', '**')
    s = re.sub(r'(?<![\d.])0+(?=\d)', '', s)
    return s.strip()


def _try_numeric_eval(s: str):
    """尝试把答案求值为数字。失败返回 None。

    安全性：使用 ast 白名单求值，只允许数字常量、算术运算符、
    sqrt() 调用和 pi/e 常量，不执行任意代码。
    """
    s = _simplify_latex_for_eval(s)
    if not s:
        return None
    _BIN_OPS = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod,
    }
    _UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}

    def _eval_node(node):
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == 'sqrt' and len(node.args) == 1 and not node.keywords:
            return _math.sqrt(_eval_node(node.args[0]))
        if isinstance(node, ast.Name) and node.id in ('pi', 'e'):
            return _math.pi if node.id == 'pi' else _math.e
        raise ValueError("disallowed node")

    try:
        tree = ast.parse(s, mode='eval')
        return float(_eval_node(tree))
    except Exception:
        return None


def evaluate_math(prediction: str, ground_truth: str) -> bool:
    """MATH 答案比较（支持 LaTeX 等价形式）。"""
    if "\\boxed" in prediction:
        boxed = extract_boxed(prediction)
        if boxed:
            prediction = boxed
    if "\\boxed" in ground_truth:
        boxed = extract_boxed(ground_truth)
        if boxed:
            ground_truth = boxed

    pred_norm = _math_normalize_answer(prediction)
    gt_norm = _math_normalize_answer(ground_truth)

    if pred_norm == gt_norm:
        return True

    def _tight(s):
        return re.sub(r'\s+', '', s).replace('{', '').replace('}', '')
    if _tight(pred_norm) == _tight(gt_norm):
        return True

    pn = _try_numeric_eval(prediction)
    gn = _try_numeric_eval(ground_truth)
    if pn is not None and gn is not None:
        return abs(pn - gn) < 1e-6
    return False


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------


def _clean_gsm8k_answer(s: str) -> str:
    """清理 GSM8K 答案：去除 $、%、\\boxed{} 等包装，提取数字部分。"""
    s = s.strip().strip('$').strip()
    boxed = re.search(r'\\boxed\{([^{}]*)\}', s)
    if boxed:
        s = boxed.group(1).strip()
    s = s.strip("[]'\"").strip()
    s = s.rstrip('%').strip()
    m = re.match(r'-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*-?\d[\d,]*(?:\.\d+)?)?', s)
    if m:
        return m.group(0).replace(' ', '')
    return s


def evaluate_gsm8k(prediction: str, ground_truth: str) -> bool:
    """GSM8K 答案比较（支持整数、小数、分数）。"""
    pred = _clean_gsm8k_answer(prediction)
    gt = _clean_gsm8k_answer(ground_truth)
    if pred == gt:
        return True

    def to_num(s):
        s = s.strip()
        if '/' in s:
            parts = s.split('/')
            if len(parts) == 2:
                try:
                    return float(parts[0]) / float(parts[1])
                except (ValueError, ZeroDivisionError):
                    pass
        try:
            return float(s.replace(',', ''))
        except (ValueError, TypeError):
            return None

    pn, gn = to_num(pred), to_num(gt)
    if pn is not None and gn is not None:
        return abs(pn - gn) < 1e-6
    return False


# ---------------------------------------------------------------------------
# AIME
# ---------------------------------------------------------------------------


def _clean_aime_answer(s: str) -> str:
    """清理 AIME 答案：去除 $、\\boxed{}、逗号等包装，返回纯数字串。

    用 re.search 而非 re.match，确保 ### 前缀（GEPA 答案格式）下的数字
    也能被抽到——re.match 只从串首匹配，遇到 '### 468' 会失败。
    """
    s = s.strip().strip('$').strip()
    boxed = re.search(r'\\boxed\{([^{}]*)\}', s)
    if boxed:
        s = boxed.group(1).strip()
    s = s.strip("[]'\"").strip()
    s = s.replace(",", "").replace(" ", "")
    m = re.search(r'-?\d+', s)
    return m.group(0) if m else s


def evaluate_aime(prediction: str, ground_truth: str) -> bool:
    """判断 AIME 答案是否正确（0-999 整数）。"""
    pred_str = _clean_aime_answer(prediction)
    gt_str = _clean_aime_answer(ground_truth)
    if pred_str == gt_str:
        return True
    try:
        return int(pred_str) == int(gt_str)
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# HotpotQA
# ---------------------------------------------------------------------------


def _normalize_text(s: str) -> str:
    """SQuAD 风格文本规范化：小写、去标点、去冠词、压缩空格。"""
    s = s.lower().strip().strip("[]'\"")
    s = s.translate(str.maketrans('', '', _string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def evaluate_hotpotqa(prediction: str, ground_truth: str) -> bool:
    """HotpotQA 答案比较（规范化精确匹配 + 词边界包含匹配）。"""
    pred_clean = _normalize_text(prediction)
    gt_clean = _normalize_text(ground_truth)
    if not gt_clean:
        return not pred_clean
    if pred_clean == gt_clean:
        return True
    if not pred_clean:
        return False
    if re.search(r'\b' + re.escape(gt_clean) + r'\b', pred_clean):
        return True
    if re.search(r'\b' + re.escape(pred_clean) + r'\b', gt_clean):
        return True
    return False


# ---------------------------------------------------------------------------
# 可验证指令跟随：IFBench / LiveBench
# ---------------------------------------------------------------------------
#
# gold 是 JSON 字符串：{"prompt": str, "instruction_id_list": [str],
#                       "kwargs": [{...}]}
# prediction 是模型的【完整回答】（answer_type=verifiable 时 engine 不做答案抽取，
# 直接把 raw 交给本 judger）。判分复刻 IFBench 官方 evaluation_lib 的 strict 逻辑：
# 逐约束构造 Instruction、build_description(**kwargs)，全部 check_following 通过才算对。
#
# IFBench 官方 registry 同时覆盖 25 个经典 IFEval 约束与 58 个 OOD 约束，
# 故 ifbench 与 livebench 共用同一套 verifier 与本函数。

_IF_REGISTRY = None          # 懒加载缓存：首次调用才 import 重依赖包
_IF_IMPORT_ERROR = None      # 记录首次导入失败原因，供清晰报错


class VerifierEnvironmentError(ImportError):
    """IF verifier 环境问题（缺依赖 / 缺 nltk 语料）。

    继承 ImportError：_parse_output 对 ImportError 显式放行（不吞成 False），
    保证"环境没装好"不会被误读成"模型全错"。
    """


def _load_if_registry():
    """懒加载 IFBench verifier registry（首次调用触发 nltk/emoji 等重依赖导入）。

    缺依赖时抛出带安装指引的 VerifierEnvironmentError，而非静默判 False——
    避免"环境没装依赖"被误读成"模型全错"。导入成功后校验 nltk 语料齐备
    （上游 download_nltk_resources 在离线时静默失败，语料缺失要到 check_following
    才炸 LookupError），缓存后后续调用零开销。
    """
    global _IF_REGISTRY, _IF_IMPORT_ERROR
    if _IF_REGISTRY is not None:
        return _IF_REGISTRY
    if _IF_IMPORT_ERROR is not None:
        raise VerifierEnvironmentError(_IF_IMPORT_ERROR)
    try:
        from .ifbench_verifier import instructions_registry
        registry = instructions_registry.INSTRUCTION_DICT
        # 校验 nltk 语料：缺任何一项都会在 check_following 内部炸 LookupError，
        # 提前在这里显式报错（附离线预取指引）
        import nltk
        missing = []
        for res in ("tokenizers/punkt", "tokenizers/punkt_tab",
                    "corpora/stopwords", "taggers/averaged_perceptron_tagger_eng"):
            try:
                nltk.data.find(res)
            except LookupError:
                missing.append(res)
        if missing:
            raise LookupError(
                f"nltk resources missing: {missing}. Pre-fetch on a networked "
                f"host: python -c \"import nltk; [nltk.download(p) for p in "
                f"('punkt','punkt_tab','stopwords','averaged_perceptron_tagger_eng')]\""
            )
        _IF_REGISTRY = registry
        return _IF_REGISTRY
    except (ImportError, LookupError) as e:
        name = getattr(e, "name", None) or str(e)
        _IF_IMPORT_ERROR = (
            f"IFBench/LiveBench verifier environment problem ({name}). "
            f"Install deps with: pip install -r requirements-if.txt"
        )
        raise VerifierEnvironmentError(_IF_IMPORT_ERROR) from e


def _evaluate_if_style(prediction: str, gold: str) -> bool:
    """IFBench / LiveBench 共用的 strict 约束判分。

    对完整回答 prediction 逐条验证 gold(JSON) 里的约束；全部通过返回 True。
    gold 解析失败视为该样本判分错误（False），不抛异常中断评测。

    官方 strict 语义：空回答直接 False；check_following 收到【未 strip】的
    完整回答（部分约束如 no_whitespace / end_checker 对首尾空白敏感）。
    环境问题（VerifierEnvironmentError / nltk LookupError）显式上抛，
    不落入调用方的兜底 False。
    """
    registry = _load_if_registry()
    try:
        spec = json.loads(gold)
        instruction_id_list = spec["instruction_id_list"]
        kwargs_list = spec["kwargs"]
        prompt = spec.get("prompt", "")
    except (ValueError, KeyError, TypeError):
        return False

    if not prediction or not prediction.strip():
        return False
    if len(instruction_id_list) != len(kwargs_list):
        # gold 编码错位：数据问题，显式报错而非静默 False
        raise VerifierEnvironmentError(
            f"IF gold mismatch: {len(instruction_id_list)} ids vs "
            f"{len(kwargs_list)} kwargs"
        )

    for index, instruction_id in enumerate(instruction_id_list):
        instruction_cls = registry.get(instruction_id)
        if instruction_cls is None:
            # 未知约束 id：无法验证，保守判该条不通过（与官方 KeyError 行为一致但更稳）
            return False
        instruction = instruction_cls(instruction_id)
        # 复刻官方：剔除 None 值后再 build_description
        kwargs = {k: v for k, v in kwargs_list[index].items() if v is not None}
        try:
            instruction.build_description(**kwargs)
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=prompt)
            followed = bool(instruction.check_following(prediction))
        except VerifierEnvironmentError:
            raise
        except LookupError as e:
            # nltk 语料缺失等环境问题：显式上抛，不判 False
            raise VerifierEnvironmentError(
                f"IF verifier environment problem on '{instruction_id}': {e}"
            ) from e
        if not followed:
            return False
    return True


def evaluate_ifbench(prediction: str, gold: str) -> bool:
    """IFBench 约束判分（OOD 约束为主）。见 _evaluate_if_style。"""
    return _evaluate_if_style(prediction, gold)


def evaluate_livebench(prediction: str, gold: str) -> bool:
    """LiveBench 指令跟随判分（经典 IFEval 约束）。见 _evaluate_if_style。"""
    return _evaluate_if_style(prediction, gold)


# ---------------------------------------------------------------------------
# PUPA：隐私脱敏
# ---------------------------------------------------------------------------
#
# gold 是 JSON 字符串：{"pii_units": [str], "redacted_query": str}
# prediction 是模型输出的【完整脱敏后 query】（answer_type=redaction，不抽取答案）。
# 判分以"泄漏检查"为主（对齐 PAPILLON 的 JudgeLeakage 思路，但用确定性字符串匹配、
# 不引入 LLM judge）：任一 PII 单元仍出现在输出中即判泄漏(False)；再加有效性门槛
# （非空、长度不退化、不刷 [REDACTED] 占位符——防止"全替换"策略刷分）。
#
# 注意：PUPA_TNB 的 pii_units 含 "university" 这类泛词，匹配仍可能误报泄漏；
# 默认数据集为 PUPA_New（PII 均为具体实体），此风险较低。

_REDACTED_TOKEN = re.compile(r"\[redacted\]", re.IGNORECASE)


def _pii_leaks(prediction: str, pii_units: list) -> bool:
    """任一非平凡 PII 单元出现在 prediction 中则判泄漏。

    匹配规则：
      - 纯 ASCII 单元：按首尾字符分侧加 \\b 词边界——该侧是字母/数字才加
        （避免 "yoyo" 误中 "toyota"），是标点/空白（如 <presidio_...> 占位符、
        URL、"k.f.c."）则该侧退化为子串包含（\\b 在非词字符边缘永不成立，
        会漏检）。
      - 含非 ASCII（CJK 等）单元：直接子串包含——Python 的 \\b 在 CJK 字符
        之间永不触发（CJK 全是 \\w，无边界），加边界会整体漏检中文 PII。
    平凡单元（<2 字符或无任何词字符）跳过，避免误报。
    """
    low = prediction.lower()
    for unit in pii_units:
        u = (unit or "").strip()
        # 跳过平凡单元：Unicode 感知的词字符检查（CJK 名字也算词字符，
        # 不能只看 [0-9A-Za-z]，否则中文 PII 全部漏检）
        if len(u) < 2 or not any(ch.isalnum() for ch in u):
            continue
        ul = u.lower()
        if ul.isascii():
            pre = r"\b" if ul[0].isalnum() else ""
            post = r"\b" if ul[-1].isalnum() else ""
            if re.search(pre + re.escape(ul) + post, low):
                return True
        elif ul in low:
            return True
    return False


def evaluate_pupa(prediction: str, gold: str) -> bool:
    """PUPA 脱敏判分：无 PII 泄漏 + 输出有效（不退化、不刷占位符）。

    有效性门槛（防止"把整段 query 全换成 [REDACTED]"这类退化策略刷分，
    SRO 的优化器会主动搜索能提分的策略，指标必须抗刷）：
      1. 非空；
      2. 长度不过短（gold 的 10%，且 ≥10 字符）也不过长（gold 的 3 倍，
         且 ≥50 字符）——防退化/防注水；
      3. 去掉 [REDACTED] 占位符后须保留实质内容（非占位字符 ≥15%——
         gold 自身 p1 分位为 31%，15% 只拦全替换类输出）。
    """
    try:
        spec = json.loads(gold)
        pii_units = spec.get("pii_units", [])
        gold_redacted = spec.get("redacted_query", "") or ""
    except (ValueError, KeyError, TypeError):
        return False

    pred = (prediction or "").strip()
    if not pred:
        return False
    # 长度带：过短=退化/拒答；过长=注水
    min_len = max(10, int(0.1 * len(gold_redacted)))
    max_len = max(50, 3 * len(gold_redacted))
    if not (min_len <= len(pred) <= max_len):
        return False
    # 防刷门槛：剥掉 [REDACTED] 后仍须有实质内容
    content = _REDACTED_TOKEN.sub("", pred).strip()
    if len(content) < max(10, int(0.15 * len(pred))):
        return False
    # 主判据：任一 PII 单元泄漏即失败
    if _pii_leaks(pred, pii_units):
        return False
    return True


# ---------------------------------------------------------------------------
# 分派表
# ---------------------------------------------------------------------------

JUDGERS = {
    "math": evaluate_math,
    "gsm8k": evaluate_gsm8k,
    "aime": evaluate_aime,
    "hotpotqa": evaluate_hotpotqa,
    "ifbench": evaluate_ifbench,
    "livebench": evaluate_livebench,
    "pupa": evaluate_pupa,
}


def get_judger(dataset: str):
    return JUDGERS[dataset]
