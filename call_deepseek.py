#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
'调用 LLM 接口，支持10个并发请求，失败时最多重试 LLM_MAX_RETRIES 次。'
'Args:
    messages: 用户消息列表，如 [{"role": "user", "content": "Hello"}]
    model: 模型名称，默认从环境变量 LLM_MODEL 读取
    system_content: 系统提示词
    stream: 是否流式返回

Returns:
    response 对象。若重试均失败则抛出最后一次异常。
'''
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI

ENV_PATH = Path(__file__).resolve().parent / ".env"


def _load_env_file(path: Path = ENV_PATH) -> None:
    """加载本地 .env，并保留外部环境变量的优先级。"""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


_load_env_file()

MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """懒加载 OpenAI 客户端，从环境变量读取配置。"""
    global _client
    if _client is None:
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise RuntimeError("Missing LLM_API_KEY. Please set it in .env or the shell environment.")
        base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def llm_request(
    messages,
    model=None,
    system_content="You are a helpful assistant",
    stream=False,
    max_retries: int | None = None,
):
    """
    调用 LLM 接口，失败时最多重试 LLM_MAX_RETRIES 次。

    Args:
        messages: 用户消息列表，如 [{"role": "user", "content": "Hello"}]
        model: 模型名称，默认从环境变量 LLM_MODEL 读取
        system_content: 系统提示词
        stream: 是否流式返回

    Returns:
        response 对象。若重试均失败则抛出最后一次异常。
    """
    if model is None:
        model = os.getenv("LLM_MODEL", "deepseek-chat")
    if max_retries is None:
        max_retries = MAX_RETRIES
    full_messages = [{"role": "system", "content": system_content}] + list(messages)
    last_error = None
    client = _get_client()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=full_messages,
                stream=stream,
            )
            return response
        except Exception as e:
            last_error = e
            if attempt == max_retries - 1:
                raise last_error


if __name__ == "__main__":
    response = llm_request([{"role": "user", "content": "Hello"}])
    if response:
        print(response.choices[0].message.content)
