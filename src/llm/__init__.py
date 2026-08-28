"""sro.llm — LLM 客户端封装。

统一封装 writer / reflection 两个角色的客户端，支持可调超时：
    - writer：任务 agent（解题），默认 200s（AIME 推理时间长）
    - reflection：反射/优化器，默认 60s

复用现有实现：openai_api_test/gepa_aime.py 的
    create_client(timeout) / build_api_kwargs(...)

build_api_kwargs 处理：
    - provider 特有参数（enable_thinking / reasoning_effort）
    - qwen3 系列的显式 enable_thinking 开关
    - extra_body 注入（openai SDK v1+）
"""
