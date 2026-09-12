"""答案判分函数（按数据集）。

从 openai_api_test 移植的纯字符串/数值处理函数，使 SRO 自包含，
不触发 openai_api_test 顶层对 openai 库的 import。

- math    : extract_boxed + LaTeX 归一化 + AST 数值求值
- gsm8k   : 数字清理（整数/小数/分数）
- aime    : 0-999 整数清理
- hotpotqa: SQuAD 风格文本归一化 + 词边界匹配
"""

from __future__ import annotations

import ast
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
    """清理 AIME 答案：去除 $、\\boxed{}、逗号等包装，返回纯数字串。"""
    s = s.strip().strip('$').strip()
    boxed = re.search(r'\\boxed\{([^{}]*)\}', s)
    if boxed:
        s = boxed.group(1).strip()
    s = s.strip("[]'\"").strip()
    s = s.replace(",", "").replace(" ", "")
    m = re.match(r'-?\d+', s)
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
# 分派表
# ---------------------------------------------------------------------------

JUDGERS = {
    "math": evaluate_math,
    "gsm8k": evaluate_gsm8k,
    "aime": evaluate_aime,
    "hotpotqa": evaluate_hotpotqa,
}


def get_judger(dataset: str):
    return JUDGERS[dataset]
