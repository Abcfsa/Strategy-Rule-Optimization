"""数据集加载。

load_aime(args)：
    加载 AI-MO/aimo-validation-aime（train+val）与 MathArena/aime_2025（test），
    本地 Arrow 格式，Dataset.from_file 后按 seed 拆分 train/val。
    移植自 openai_api_test/gepa_aime.py。

load_math(args)：
    加载 MATH（train/test JSON 目录），支持难度过滤。
    移植自 openai_api_test/gepa_math.py。

两者均返回 (train_data, val_data, test_data) 三元组，
每条样本为 {problem, answer, ...}。
"""
