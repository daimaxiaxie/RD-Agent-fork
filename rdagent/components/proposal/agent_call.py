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
        async for message in query(prompt=full_prompt, options=options):
            if hasattr(message, "content"):
                for block in message.content:
                    if hasattr(block, "text"):
                        result_parts.append(block.text)

    asyncio.run(_query())
    return "".join(result_parts)
