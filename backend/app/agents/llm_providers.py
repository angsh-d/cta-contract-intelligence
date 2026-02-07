"""LLM Provider Protocol, implementations, and factory."""

import asyncio
import json
import logging
import time
from typing import Protocol, Optional, Any, runtime_checkable

from app.models.agent_schemas import LLMResponse

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMProvider(Protocol):
    """Structural interface every LLM backend must satisfy."""

    @property
    def provider_name(self) -> str: ...

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        model: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.0,
        response_format: Optional[str] = None,
    ) -> LLMResponse: ...

    async def complete_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[dict[str, Any]],
        *,
        model: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> LLMResponse: ...

    async def close(self) -> None: ...


# ── ClaudeProvider ──────────────────────────────────────────────

class ClaudeProvider:
    """Anthropic Claude API (Opus / Sonnet)."""

    provider_name: str = "claude"

    def __init__(self) -> None:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        self._api_key = os.environ["ANTHROPIC_API_KEY"]
        self._client = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self):
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    from anthropic import AsyncAnthropic
                    self._client = AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def complete(
        self, system_prompt, user_message, *,
        model=None, max_output_tokens=None, temperature=0.0, response_format=None,
    ) -> LLMResponse:
        client = await self._get_client()
        model = model or "claude-sonnet-4-5-20250929"
        max_output_tokens = max_output_tokens or 8192
        start = time.monotonic()
        response = await client.messages.create(
            model=model,
            max_tokens=max_output_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        content = response.content[0].text if response.content else ""
        if not content:
            logger.warning(
                "Empty Claude response: stop_reason=%s model=%s content_blocks=%d",
                response.stop_reason, model, len(response.content) if response.content else 0,
            )
        return LLMResponse(
            success=True,
            content=content,
            usage={"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
            model=model,
            latency_ms=latency_ms,
            provider=self.provider_name,
        )

    async def complete_with_tools(
        self, system_prompt, user_message, tools, *,
        model=None, max_output_tokens=None, temperature=0.0,
    ) -> LLMResponse:
        client = await self._get_client()
        model = model or "claude-sonnet-4-5-20250929"
        max_output_tokens = max_output_tokens or 8192
        start = time.monotonic()

        # Convert tool definitions to Anthropic format
        anthropic_tools = []
        for tool in tools:
            anthropic_tools.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("input_schema", {"type": "object", "properties": {}}),
            })

        # Handle user_message as string or list of messages
        if isinstance(user_message, str):
            messages = [{"role": "user", "content": user_message}]
        else:
            messages = user_message

        response = await client.messages.create(
            model=model,
            max_tokens=max_output_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages,
            tools=anthropic_tools,
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        # Extract content — may contain text and/or tool_use blocks
        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        content = "\n".join(text_parts)
        # If tool_use returned structured JSON, extract it
        if tool_calls and not content:
            content = json.dumps(tool_calls[0]["input"])

        return LLMResponse(
            success=True,
            content=content,
            usage={"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
            model=model,
            latency_ms=latency_ms,
            provider=self.provider_name,
            tool_calls=tool_calls if tool_calls else None,
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


# ── AzureOpenAIProvider ─────────────────────────────────────────

class AzureOpenAIProvider:
    """Azure OpenAI (deployment from .env)."""

    provider_name: str = "azure_openai"

    def __init__(self) -> None:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        self._api_key = os.environ["AZURE_OPENAI_API_KEY"]
        self._endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        self._deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        self._api_version = os.environ["AZURE_OPENAI_API_VERSION"]
        self._client = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self):
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    from openai import AsyncAzureOpenAI
                    self._client = AsyncAzureOpenAI(
                        api_key=self._api_key,
                        azure_endpoint=self._endpoint,
                        api_version=self._api_version,
                    )
        return self._client

    async def complete(
        self, system_prompt, user_message, *,
        model=None, max_output_tokens=None, temperature=0.0, response_format=None,
    ) -> LLMResponse:
        client = await self._get_client()
        model = model or self._deployment
        max_output_tokens = max_output_tokens or 16384
        start = time.monotonic()
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        response = await client.chat.completions.create(**kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)
        content = response.choices[0].message.content or ""
        return LLMResponse(
            success=True,
            content=content,
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            model=model,
            latency_ms=latency_ms,
            provider=self.provider_name,
        )

    async def complete_with_tools(
        self, system_prompt, user_message, tools, *,
        model=None, max_output_tokens=None, temperature=0.0,
    ) -> LLMResponse:
        client = await self._get_client()
        model = model or self._deployment
        max_output_tokens = max_output_tokens or 16384
        start = time.monotonic()

        # Convert to OpenAI function-calling format
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                },
            })

        if isinstance(user_message, str):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        else:
            messages = [{"role": "system", "content": system_prompt}] + user_message

        response = await client.chat.completions.create(
            model=model,
            max_tokens=max_output_tokens,
            temperature=temperature,
            messages=messages,
            tools=openai_tools,
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments),
                })
            if not content:
                content = json.dumps(tool_calls[0]["input"])

        return LLMResponse(
            success=True,
            content=content,
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            model=model,
            latency_ms=latency_ms,
            provider=self.provider_name,
            tool_calls=tool_calls if tool_calls else None,
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


# ── GeminiProvider ──────────────────────────────────────────────

class GeminiProvider:
    """Google Gemini API (uses google-genai SDK)."""

    provider_name: str = "gemini"

    MODEL_MAX_TOKENS: dict[str, int] = {
        "gemini-3-pro-preview": 65536,
        "gemini-2.5-flash-lite": 65536,
        "gemini-2.5-pro": 65536,
        "gemini-2.0-flash-exp": 8192,
        "gemini-1.5-flash": 8192,
        "gemini-1.5-pro": 8192,
    }

    def __init__(self) -> None:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        self._api_key = os.environ["GEMINI_API_KEY"]
        self._client = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self):
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    from google import genai
                    self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def complete(
        self, system_prompt, user_message, *,
        model=None, max_output_tokens=None, temperature=0.0, response_format=None,
    ) -> LLMResponse:
        client = await self._get_client()
        model_name = model or "gemini-3-pro-preview"
        max_output_tokens = max_output_tokens or self.MODEL_MAX_TOKENS.get(model_name, 65536)
        start = time.monotonic()
        from google.genai import types
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        if response_format == "json":
            config.response_mime_type = "application/json"
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=user_message,
            config=config,
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        return LLMResponse(
            success=True,
            content=response.text or "",
            usage={
                "input_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
                "output_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
            },
            model=model_name,
            latency_ms=latency_ms,
            provider=self.provider_name,
        )

    async def complete_with_tools(
        self, system_prompt, user_message, tools, *,
        model=None, max_output_tokens=None, temperature=0.0,
    ) -> LLMResponse:
        client = await self._get_client()
        model_name = model or "gemini-3-pro-preview"
        max_output_tokens = max_output_tokens or self.MODEL_MAX_TOKENS.get(model_name, 65536)
        start = time.monotonic()

        from google.genai import types

        # Convert tool defs to Gemini format
        gemini_tools = []
        for tool in tools:
            func_decl = types.FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=tool.get("input_schema"),
            )
            gemini_tools.append(types.Tool(function_declarations=[func_decl]))

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            tools=gemini_tools,
        )

        msg = user_message if isinstance(user_message, str) else json.dumps(user_message)
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=msg,
            config=config,
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        # Extract function calls if any — response.text raises if only function_call parts exist
        try:
            content = response.text or ""
        except ValueError:
            content = ""
        tool_calls = []
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    tool_calls.append({
                        "id": fc.name,
                        "name": fc.name,
                        "input": dict(fc.args) if fc.args else {},
                    })
            if tool_calls and not content:
                content = json.dumps(tool_calls[0]["input"])

        return LLMResponse(
            success=True,
            content=content,
            usage={
                "input_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
                "output_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
            },
            model=model_name,
            latency_ms=latency_ms,
            provider=self.provider_name,
            tool_calls=tool_calls if tool_calls else None,
        )

    async def close(self) -> None:
        self._client = None


# ── LLMProviderFactory ──────────────────────────────────────────

class LLMProviderFactory:
    """Singleton factory — one provider instance per backend."""

    _providers: dict[str, LLMProvider] = {}

    ROLE_MAP: dict[str, tuple[str, str]] = {
        "extraction":        ("claude",       "gemini"),
        "complex_reasoning": ("claude",       "azure_openai"),
        "classification":    ("claude",       "gemini"),
        "embedding":         ("azure_openai", "gemini"),
        "synthesis":         ("claude",       "azure_openai"),
    }

    @classmethod
    def get_provider(cls, name: str) -> LLMProvider:
        if name not in cls._providers:
            if name == "claude":
                cls._providers[name] = ClaudeProvider()
            elif name == "azure_openai":
                cls._providers[name] = AzureOpenAIProvider()
            elif name == "gemini":
                cls._providers[name] = GeminiProvider()
            else:
                raise ValueError(f"Unknown provider: {name}")
        return cls._providers[name]

    @classmethod
    def get_for_role(cls, role: str) -> LLMProvider:
        primary, _ = cls.ROLE_MAP[role]
        return cls.get_provider(primary)

    @classmethod
    def get_fallback_for_role(cls, role: str) -> LLMProvider:
        _, fallback = cls.ROLE_MAP[role]
        return cls.get_provider(fallback)
