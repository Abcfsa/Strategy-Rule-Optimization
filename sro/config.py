"""配置加载：从 .env 读取，缺省用内置默认值。

优先级：环境变量 > .env 文件 > 本文件 DEFAULTS。
密钥只在 OPENAI_API_KEY，不入代码、不入库（见 .gitignore）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# 可选依赖：python-dotenv。无则降级为仅读环境变量。
try:
    from dotenv import load_dotenv
    _HAS_DOTENV = True
except Exception:
    _HAS_DOTENV = False


DEFAULTS = {
    "OPENAI_BASE_URL": "https://api.openai.com/v1",
    "OPENAI_API_KEY": "",
    "TASK_MODEL": "gpt-4o-mini",
    "REFLECTION_MODEL": "gpt-4o-mini",
    "TASK_TIMEOUT": "200",
    "REFLECTION_TIMEOUT": "60",
    "EMBED_MODEL": "text-embedding-3-small",
    "EMBED_DIM": "1536",
    "MATCH_THRESHOLD": "0.6",
    "TOP_K": "3",
}


@dataclass
class Config:
    openai_base_url: str
    openai_api_key: str
    task_model: str
    reflection_model: str
    task_timeout: float
    reflection_timeout: float
    embed_model: str
    embed_dim: int
    match_threshold: float
    top_k: int

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key) and not self.openai_api_key.startswith("sk-your")

    @classmethod
    def load(cls) -> "Config":
        if _HAS_DOTENV:
            load_dotenv()  # 默认从 ./.env 读
        g = lambda k: os.getenv(k, DEFAULTS[k])
        return cls(
            openai_base_url=g("OPENAI_BASE_URL"),
            openai_api_key=g("OPENAI_API_KEY"),
            task_model=g("TASK_MODEL"),
            reflection_model=g("REFLECTION_MODEL"),
            task_timeout=float(g("TASK_TIMEOUT")),
            reflection_timeout=float(g("REFLECTION_TIMEOUT")),
            embed_model=g("EMBED_MODEL"),
            embed_dim=int(g("EMBED_DIM")),
            match_threshold=float(g("MATCH_THRESHOLD")),
            top_k=int(g("TOP_K")),
        )


# 单例（懒加载：首次 import 即读 .env）
_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config.load()
    return _config
