"""可复现的随机种子管理。

make_rng(seed)：
    返回 np.random.RandomState，用于训练/验证/测试拆分与采样。

训练/验证/测试用不同偏移的 seed（如 seed, seed+1），
避免拆分重叠（复用现有脚本的约定）。
"""
