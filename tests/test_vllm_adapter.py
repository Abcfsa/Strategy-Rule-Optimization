# -*- coding: utf-8 -*-
"""Mock tests for vLLM adaptation (no server needed; delete after use)."""
import io, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")
os.environ["OPENAI_API_KEY"] = "sk-test"
os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"

from sro.llm import _OpenAIClient, TaskLM, ReflectionLM
from sro.config import get_config

# ---- backend detection ----
assert _OpenAIClient._detect_backend("http://localhost:8000/v1") == "vllm"
assert _OpenAIClient._detect_backend("http://gpu-box:8000/v1") == "vllm"
assert _OpenAIClient._detect_backend("http://localhost:11434/v1") == "ollama"
assert _OpenAIClient._detect_backend("http://my-ollama:11434/v1") == "ollama"
assert _OpenAIClient._detect_backend("https://api.openai.com/v1") == "other"
assert _OpenAIClient._detect_backend("http://x:9999/v1", "vllm") == "vllm"  # explicit override
print("backend detection: OK")

# ---- vllm kwargs: chat_template_kwargs, no reasoning_effort hack ----
c = _OpenAIClient(base_url="http://localhost:8000/v1", api_key="sk-test")
assert c.backend == "vllm"
kw = c.build_api_kwargs("Qwen/Qwen3-8B", temperature=0.0, max_tokens=4096,
                        enable_thinking=True, max_context_len=0, extra_params="")
assert kw["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
assert "reasoning_effort" not in kw
kw_off = c.build_api_kwargs("Qwen/Qwen3-8B", temperature=0.0, max_tokens=4096,
                            enable_thinking=False, max_context_len=0, extra_params="")
assert kw_off["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
assert "reasoning_effort" not in kw_off, "vllm must not use ollama's none hack"
# vllm: no forced streaming
assert c._needs_stream(kw) is False
assert c._needs_stream(dict(kw, stream=True)) is True
print("vllm kwargs + stream logic: OK")

# ---- ollama path unchanged ----
co = _OpenAIClient(base_url="http://localhost:11434/v1", api_key="ollama")
assert co.backend == "ollama"
kwo = co.build_api_kwargs("qwen3:8b", temperature=0.0, max_tokens=4096,
                          enable_thinking=False, max_context_len=0, extra_params="")
assert kwo.get("reasoning_effort") == "none", "ollama none hack must stay"
kwo_on = co.build_api_kwargs("qwen3:8b", temperature=0.0, max_tokens=4096,
                             enable_thinking=True, max_context_len=0, extra_params="")
# openai SDK present -> extra_body path -> stream forced; absent -> v0 top-level path
if co._sdk_major() >= 1:
    assert co._needs_stream(kwo_on) is True, "ollama thinking still streams"
else:
    assert kwo_on.get("enable_thinking") is True  # v0 top-level injection (pre-existing)
print("ollama path unchanged: OK")

# ---- mock vllm server responses ----
class FakeMessage:
    def __init__(self, content, reasoning=None):
        self.content = content
        self.reasoning_content = reasoning

class FakeChoice:
    def __init__(self, content, reasoning=None, finish="stop"):
        self.message = FakeMessage(content, reasoning)
        self.finish_reason = finish

class FakeResp:
    def __init__(self, content, reasoning=None, finish="stop"):
        self.choices = [FakeChoice(content, reasoning, finish)]

class FakeCompletions:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []
    def create(self, **kw):
        self.calls.append(kw)
        return self.scripted.pop(0)

class FakeChat:
    def __init__(self, comp): self.completions = comp
class FakeClient:
    def __init__(self, comp): self.chat = FakeChat(comp)

# case 1: non-budget, thinking on, reasoning separated -> recombined trajectory
comp = FakeCompletions([FakeResp("Answer: 42", reasoning="2+2=4, times 10...")])
c._client = FakeClient(comp); c.is_real = True
out = c.chat("Qwen/Qwen3-8B", "sys", "user", enable_thinking=True, max_tokens=4096)
assert out == "<think>\n2+2=4, times 10...\n</think>\n\nAnswer: 42"
assert comp.calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
print("vllm reasoning recombination: OK")

# case 2: thinking off -> plain content, no think wrapper
comp = FakeCompletions([FakeResp("Answer: 42")])
c._client = FakeClient(comp)
out = c.chat("Qwen/Qwen3-8B", "sys", "user", enable_thinking=False, max_tokens=4096)
assert out == "Answer: 42"
print("vllm no-thinking plain path: OK")

# case 3: budget, thinking finishes naturally (stop) -> single call
comp = FakeCompletions([FakeResp("Answer: 7", reasoning="short think", finish="stop")])
c._client = FakeClient(comp)
out = c.chat("Qwen/Qwen3-8B", "sys", "user", enable_thinking=True,
             max_tokens=4096, thinking_budget=2048)
assert len(comp.calls) == 1
assert comp.calls[0]["max_tokens"] == 2048
assert out == "<think>\nshort think\n</think>\n\nAnswer: 7"
print("budget natural-finish single call: OK")

# case 4: budget exceeded (length) -> two-stage with official close injection
comp = FakeCompletions([
    FakeResp("", reasoning="long unfinished thinking", finish="length"),
    FakeResp("Answer: 99", finish="stop"),
])
c._client = FakeClient(comp)
out = c.chat("Qwen/Qwen3-8B", "sys", "user", enable_thinking=True,
             max_tokens=4096, thinking_budget=2048)
assert len(comp.calls) == 2
k1, k2 = comp.calls
assert k1["max_tokens"] == 2048
assert k2["max_tokens"] == 2048  # 4096-2048
assert k2["extra_body"]["continue_final_message"] is True
assert k2["extra_body"]["add_generation_prompt"] is False
assert k2["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
last_msg = k2["messages"][-1]
assert last_msg["role"] == "assistant"
assert "long unfinished thinking" in last_msg["content"]
assert "Considering the limited time" in last_msg["content"]
assert "</think>" in last_msg["content"]
assert out == "<think>\nlong unfinished thinking\n</think>\n\nAnswer: 99"
print("budget two-stage injection: OK")

# case 5: budget >= max_tokens or 0 -> normal path (no two-stage)
comp = FakeCompletions([FakeResp("A", reasoning="r")])
c._client = FakeClient(comp)
c.chat("Qwen/Qwen3-8B", "s", "u", enable_thinking=True, max_tokens=2048, thinking_budget=2048)
assert len(comp.calls) == 1 and comp.calls[0]["max_tokens"] == 2048
print("budget boundary (>=max) falls back: OK")

# ---- LM wiring ----
cfg = get_config()
assert cfg.task_thinking_budget == 0
t = TaskLM(thinking_budget=1024)
assert t.thinking_budget == 1024
r = ReflectionLM(thinking_budget=512)
assert r.thinking_budget == 512
t0 = TaskLM(); r0 = ReflectionLM()
assert t0.thinking_budget == 0 and r0.thinking_budget == 0
print("LM thinking_budget wiring: OK")

from sro.engine import SROEngine
e = SROEngine(task_thinking_budget=2048, reflection_thinking_budget=1024)
assert e.task_lm.thinking_budget == 2048
assert e.reflection_lm.thinking_budget == 1024
print("engine wiring: OK")

print("\nALL VLLM ADAPTATION TESTS PASSED")
