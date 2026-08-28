"""答案解析与判分。

AIME（0-999 整数）：
    parse_writer_response / evaluate_answer / _clean_aime_answer
    移植自 openai_api_test/gepa_aime.py。

MATH（LaTeX 等价）：
    normalize_answer / _try_numeric_eval / evaluate_answer
    移植自 openai_api_test/gepa_math.py。

错误分类（供蒸馏与检索用）：
    classify_error_type(prediction, ground_truth, reasoning)
    复用 gepa_aime_v2.py 的 _classify_error_type 思路：
    format_error / insufficient_reasoning / calculation_error / conceptual_error
"""
