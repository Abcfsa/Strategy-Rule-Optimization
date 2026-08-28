"""YAML 配置加载。

load_config(path)：
    读取 configs/*.yaml，与 main.py 顶部默认常量合并（yaml 优先）。

配置项对应 main.py 的 CONFIGURATION PARAMETERS：
    模型、训练期进化、数据筛选、条款正则化、三层检索、快速适应等。
"""
