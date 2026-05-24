"""
Drop-in replacement for APIBackend chat completion using Claude Agent SDK.
The agent autonomously decides whether to use tools (search, fetch, etc.)
and handles multi-round interaction internally.

LLM endpoint is configured via environment variables (same as Claude Code CLI):
  ANTHROPIC_BASE_URL  - e.g. https://api.deepseek.com/anthropic
  ANTHROPIC_AUTH_TOKEN - API key
  ANTHROPIC_MODEL     - e.g. deepseek-v4-pro
"""

from __future__ import annotations

import asyncio
import logging

import nest_asyncio

from rdagent.log import rdagent_logger as logger

ALLOWED_TOOLS = ["WebSearch", "WebFetch"]


def agent_chat_completion(user_prompt: str, system_prompt: str, json_mode: bool = False) -> str:
    """Call Claude Agent SDK. The agent handles tool calls autonomously; we only get the final result."""
    from claude_agent_sdk import query, ClaudeAgentOptions

    if json_mode:
        user_prompt += (
            "\n\nIMPORTANT: You MUST respond with a valid JSON object. "
            "Do not include any text outside the JSON object."
        )

    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    result_parts: list[str] = []

    options = ClaudeAgentOptions(allowed_tools=ALLOWED_TOOLS)

    async def _query():
        from claude_agent_sdk import ResultMessage
        async for message in query(prompt=full_prompt, options=options):
            logger.debug("SDK message type: %s", type(message).__name__)
            if isinstance(message, ResultMessage):
                logger.debug("ResultMessage is_error=%s result=%r", message.is_error, message.result[:100])
                if message.is_error:
                    raise RuntimeError(f"Claude Agent SDK error: {message.result}")
                result_parts.append(message.result)

    nest_asyncio.apply()
    asyncio.run(_query())
    result = "".join(result_parts)
    if not result:
        raise RuntimeError("Claude Agent SDK returned empty response")
    logger.info("Agent SDK response (first 500 chars): %s", result[:500])
    return result
