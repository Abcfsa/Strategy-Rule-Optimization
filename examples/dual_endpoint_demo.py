#!/usr/bin/env python
"""
双端点示例：TaskLM 使用本地 Ollama，ReflectionLM 使用云端 API

这个示例展示了如何配置 SRO 使用不同的端点：
- TaskLM（执行模型）: 本地 Ollama 运行的 qwen3:8b
- ReflectionLM（反思模型）: 云端 API（如 OpenAI、Anthropic）

使用方式：
1. 确保本地 Ollama 运行并拉取了 qwen3:8b
2. 配置 .env 文件（见下方模板）
3. 运行本脚本：python examples/dual_endpoint_demo.py
"""

import os
from sro.config import get_config
from sro.llm import TaskLM, ReflectionLM

# 配置示例（实际使用时放在 .env 文件中）
EXAMPLE_ENV = """
# TaskLM - 本地 Ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
TASK_MODEL=qwen3:8b
TASK_TIMEOUT=120

# ReflectionLM - 云端 API（以 OpenAI 为例）
REFLECTION_BASE_URL=https://api.openai.com/v1
REFLECTION_API_KEY=sk-your-openai-key-here
REFLECTION_MODEL=gpt-4o-mini
REFLECTION_TIMEOUT=60
"""

def print_endpoint_info():
    """打印当前配置的端点信息"""
    cfg = get_config()

    print("=" * 70)
    print("SRO 双端点配置")
    print("=" * 70)

    print("\n📍 TaskLM（执行模型）")
    print(f"   端点:     {cfg.openai_base_url}")
    print(f"   模型:     {cfg.task_model}")
    print(f"   API Key:  {cfg.openai_api_key[:20]}..." if len(cfg.openai_api_key) > 20 else f"   API Key:  {cfg.openai_api_key}")
    print(f"   超时:     {cfg.task_timeout}s")

    print("\n🧠 ReflectionLM（反思模型）")
    print(f"   端点:     {cfg.reflection_base_url}")
    print(f"   模型:     {cfg.reflection_model}")
    print(f"   API Key:  {cfg.reflection_api_key[:20]}..." if len(cfg.reflection_api_key) > 20 else f"   API Key:  {cfg.reflection_api_key}")
    print(f"   超时:     {cfg.reflection_timeout}s")

    print("\n" + "=" * 70)


def test_task_llm():
    """测试 TaskLM 调用"""
    print("\n🔬 测试 TaskLM...")

    task_lm = TaskLM()

    system_prompt = "你是一个数学解题助手。"
    user_prompt = "请计算 2 + 3 = ?"

    try:
        response = task_lm._call_llm(system_prompt, user_prompt)
        print(f"   ✓ TaskLM 响应成功")
        print(f"   回答: {response[:100]}...")
        return True
    except Exception as e:
        print(f"   ✗ TaskLM 调用失败: {e}")
        return False


def test_reflection_llm():
    """测试 ReflectionLM 调用"""
    print("\n🔬 测试 ReflectionLM...")

    reflection_lm = ReflectionLM()

    system_prompt = "你是一个元学习专家。"
    user_prompt = "请总结解决数学问题的一般策略。"

    try:
        response = reflection_lm._call_llm(system_prompt, user_prompt)
        print(f"   ✓ ReflectionLM 响应成功")
        print(f"   回答: {response[:100]}...")
        return True
    except Exception as e:
        print(f"   ✗ ReflectionLM 调用失败: {e}")
        return False


def main():
    print("\n" + "=" * 70)
    print("SRO 双端点配置演示")
    print("TaskLM: 本地 Ollama (qwen3:8b)")
    print("ReflectionLM: 云端 API")
    print("=" * 70)

    # 打印配置信息
    print_endpoint_info()

    # 测试两个模型
    task_ok = test_task_llm()
    reflection_ok = test_reflection_llm()

    # 总结
    print("\n" + "=" * 70)
    print("测试结果")
    print("=" * 70)
    print(f"  TaskLM:       {'✓ 通过' if task_ok else '✗ 失败'}")
    print(f"  ReflectionLM: {'✓ 通过' if reflection_ok else '✗ 失败'}")

    if task_ok and reflection_ok:
        print("\n🎉 双端点配置成功！可以开始运行 SRO 实验。")
    else:
        print("\n⚠️  部分测试失败，请检查配置。")

    print("\n" + "=" * 70)
    print("\n📝 .env 配置模板:")
    print(EXAMPLE_ENV)
    print("=" * 70)


if __name__ == "__main__":
    main()
