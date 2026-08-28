"""LLM 客户端工厂与调用封装。

create_client(timeout)：
    创建带超时的 OpenAI 客户端（httpx.Timeout 需显式设置全部四参数）。

build_api_kwargs(model, *, temperature, max_tokens, ...)：
    构建 chat.completions.create 的 kwargs，处理 thinking / extra_body。

call_llm(client, messages, api_kwargs, retries)：
    带重试的调用封装，返回 response 文本。

直接移植自 openai_api_test/gepa_aime.py（已验证可用）。
"""
