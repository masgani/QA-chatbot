# app/core/llm_inference.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

import openai
from app.config.config import settings

logger = logging.getLogger(__name__)

_client = openai.OpenAI(
    base_url=settings.VLLM_BASE_URL,
    api_key=settings.VLLM_API_KEY,
)

def generate_completion(
    messages: List[Dict[str, Any]],
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout_s: int = 60,
    top_p: float = 1.0,
    model: Optional[str] = None,
) -> str:
    completion = _client.chat.completions.create(
        model=model or settings.VLLM_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout_s,
        top_p=top_p,
    )

    if not getattr(completion, "choices", None):
        logger.warning("LLM returned no choices: %r", completion)
        return ""

    choice0 = completion.choices[0]
    msg = choice0.message

    # Normal output
    content = (getattr(msg, "content", None) or "").strip()
    if content:
        return content

    # gpt-oss/vLLM sometimes puts the actual text into reasoning_content
    reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
    if reasoning:
        logger.warning(
            "LLM content empty; using reasoning_content instead. finish_reason=%s",
            getattr(choice0, "finish_reason", None),
        )
        return reasoning

    # Tool call args fallback (optional)
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        fn = getattr(tool_calls[0], "function", None)
        args = getattr(fn, "arguments", None)
        if isinstance(args, str) and args.strip():
            logger.warning("Using tool_call arguments as text.")
            return args.strip()

    logger.warning(
        "LLM returned empty content. finish_reason=%s msg=%r",
        getattr(choice0, "finish_reason", None),
        msg,
    )
    return ""
