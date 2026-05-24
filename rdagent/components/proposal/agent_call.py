"""
Drop-in replacement for APIBackend chat completion using Claude Agent SDK.
The agent autonomously decides whether to use tools (search, fetch, etc.)
and handles multi-round interaction internally.

LLM endpoint is configured via environment variables (same as Claude Code CLI):
  ANTHROPIC_BASE_URL  - e.g. https://api.deepseek.com/anthropic
  ANTHROPIC_API_KEY   - API key
  ANTHROPIC_MODEL     - e.g. deepseek-v4-pro
"""

from __future__ import annotations

import asyncio
import json

import nest_asyncio

from rdagent.log import rdagent_logger as logger

ALLOWED_TOOLS = ["WebSearch", "WebFetch"]

# JSON Schema matching the hypothesis output format expected by convert_response()
HYPOTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "reason": {"type": "string"},
        "concise_reason": {"type": "string"},
        "concise_observation": {"type": "string"},
        "concise_justification": {"type": "string"},
        "concise_knowledge": {"type": "string"},
    },
    "required": ["hypothesis", "reason"],
}


def agent_chat_completion(user_prompt: str, system_prompt: str, json_mode: bool = False) -> str:
    """Call Claude Agent SDK. The agent handles tool calls autonomously; we only get the final result."""
    from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    structured_output = None

    options_kwargs = dict(allowed_tools=ALLOWED_TOOLS)
    if json_mode:
        options_kwargs["output_format"] = {"type": "json_schema", "schema": HYPOTHESIS_SCHEMA}

    options = ClaudeAgentOptions(**options_kwargs)

    async def _query():
        nonlocal structured_output
        async for message in query(prompt=full_prompt, options=options):
            if isinstance(message, ResultMessage):
                if message.is_error:
                    raise RuntimeError(f"Claude Agent SDK error: {message.result}")
                if json_mode and message.structured_output:
                    structured_output = message.structured_output
                elif not json_mode:
                    structured_output = message.result

    nest_asyncio.apply()
    asyncio.run(_query())

    if structured_output is None:
        raise RuntimeError("Claude Agent SDK returned empty response")

    result = json.dumps(structured_output) if isinstance(structured_output, dict) else structured_output
    # logger.info(f"Agent SDK response (first 200 chars): {result[:200]}")
    return result
