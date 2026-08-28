"""三层记忆 JSON 读写。

save_json(obj, path) / load_json(path)：
    带 ensure_ascii=False 与 indent 的 JSON 序列化。

用于 strategy.json / rules.json / support.json 及运行产物
（archive、history、test_results 等）。
"""
