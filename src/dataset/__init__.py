"""sro.dataset — 数据集加载。

对应技术路线报告 §5.1：
    训练/验证：AI-MO/aimo-validation-aime（90 题，本地 ../AIME/）
    测试（同域）：MathArena/aime_2025（30 题）
    跨域迁移：MATH（../MATH/）、GSM8K
    消融域：gsm8k

加载实现复用现有代码：
    - AIME（Arrow 格式）：openai_api_test/gepa_aime.py 的 load_aime
    - MATH（JSON 目录）：openai_api_test/gepa_math.py 的 load_math
"""
