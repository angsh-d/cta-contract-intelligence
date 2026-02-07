# 01 — Base Agent & LLM Abstraction Layer

> Foundation layer that every ContractIQ agent depends on.
> File locations: `backend/app/agents/base.py`, `backend/app/agents/llm_providers.py`, `backend/app/agents/prompt_loader.py`, `backend/app/agents/config.py`

---

## 1. LLMProvider Protocol

Use a **Protocol** (structural typing) rather than ABC so that any object with the right method signatures satisfies the contract — including test doubles.

```python
# backend/app/agents/llm_providers.py

from typing import Protocol, Optional, Any, runtime_checkable
from app.models.agent_schemas import LLMResponse

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
        response_format: Optional[str] = None,   # "json" | None
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

    async def close(self) -> None:
        """Clean up async HTTP clients on shutdown."""
        ...
```

### Why Protocol over ABC

| Concern | Protocol | ABC |
|---------|----------|-----|
| Mock in tests | Any object with matching methods works — zero inheritance required | Must subclass or monkeypatch |
| Third-party adapters | Wrap with a thin class; no forced inheritance tree | Forced to extend the base |
| Runtime checks | `isinstance` works with `@runtime_checkable` decorator (applied above) | Works by default |
| Explicitness | Structural — duck typing with type-checker safety | Nominal — name must appear in MRO |

---

## 2. Provider Implementations

All three providers read credentials exclusively from `.env` via `python-dotenv`. No constructor parameters for secrets.

### 2.1 ClaudeProvider

```python
import asyncio
import time

class ClaudeProvider:
    """Anthropic Claude API (Opus / Sonnet)."""

    provider_name: str = "claude"

    def __init__(self) -> None:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        self._api_key = os.environ["ANTHROPIC_API_KEY"]
        # Lazy-init Anthropic async client (lock prevents duplicate creation under concurrency)
        self._client: Optional[AsyncAnthropic] = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> AsyncAnthropic:
        if self._client is None:
            async with self._client_lock:
                if self._client is None:  # double-check after acquiring lock
                    from anthropic import AsyncAnthropic
                    self._client = AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def complete(self, system_prompt, user_message, *, model=None, max_output_tokens=None, temperature=0.0, response_format=None) -> LLMResponse:
        client = await self._get_client()
        model = model or "claude-sonnet-4-5-20250929"
        max_output_tokens = max_output_tokens or 8192
        start = time.monotonic()
        # Note: Claude does not have a native JSON mode API flag.
        # JSON compliance is enforced via prompt instruction + BaseAgent's 3-tier JSON parsing.
        response = await client.messages.create(
            model=model,
            max_tokens=max_output_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        # Guard: response.content may be empty on stop_reason="end_turn" with no output
        content = response.content[0].text if response.content else ""
        return LLMResponse(
            success=True,
            content=content,
            usage={"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
            model=model,
            latency_ms=latency_ms,
            provider=self.provider_name,
        )

    async def complete_with_tools(self, system_prompt, user_message, tools, *, model=None, max_output_tokens=None, temperature=0.0) -> LLMResponse:
        # Same pattern; pass `tools=tools` to client.messages.create
        ...

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
```

### 2.2 AzureOpenAIProvider

```python
class AzureOpenAIProvider:
    """Azure OpenAI (gpt-5-mini deployment)."""

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

    async def complete(self, system_prompt, user_message, *, model=None, max_output_tokens=None, temperature=0.0, response_format=None) -> LLMResponse:
        client = await self._get_client()
        model = model or self._deployment  # "gpt-5-mini"
        max_output_tokens = max_output_tokens or 16384
        start = time.monotonic()
        response = await client.chat.completions.create(
            model=model,
            max_tokens=max_output_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"} if response_format == "json" else None,
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        content = response.choices[0].message.content or ""
        return LLMResponse(
            success=True,
            content=content,
            usage={"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens},
            model=model,
            latency_ms=latency_ms,
            provider=self.provider_name,
        )

    async def complete_with_tools(self, system_prompt, user_message, tools, *, model=None, max_output_tokens=None, temperature=0.0) -> LLMResponse:
        # Same pattern; pass `tools=tools` to client.chat.completions.create
        ...

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
```

### 2.3 GeminiProvider

```python
class GeminiProvider:
    """Google Gemini API (uses google-genai SDK with native async support)."""

    provider_name: str = "gemini"

    # Maximum output token limits per model (from CLAUDE.md policy)
    MODEL_MAX_TOKENS: dict[str, int] = {
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

    async def complete(self, system_prompt, user_message, *, model=None, max_output_tokens=None, temperature=0.0, response_format=None) -> LLMResponse:
        client = await self._get_client()
        model_name = model or "gemini-2.5-flash-lite"
        max_output_tokens = max_output_tokens or self.MODEL_MAX_TOKENS.get(model_name, 8192)
        start = time.monotonic()
        config = {
                "system_instruction": system_prompt,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
        }
        if response_format == "json":
            config["response_mime_type"] = "application/json"
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
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
            },
            model=model_name,
            latency_ms=latency_ms,
            provider=self.provider_name,
        )

    async def complete_with_tools(self, system_prompt, user_message, tools, *, model=None, max_output_tokens=None, temperature=0.0) -> LLMResponse:
        # Same pattern; pass `tools=tools` via config
        ...

    async def close(self) -> None:
        self._client = None  # google-genai client has no explicit close
```

---

## 3. LLMProviderFactory

Reads `.env` once, creates singleton providers, and maps **agent roles** → preferred providers.

```python
# backend/app/agents/llm_providers.py (continued)

class LLMProviderFactory:
    """Singleton factory — one provider instance per backend."""

    _providers: dict[str, LLMProvider] = {}

    # Agent role → (primary_provider, fallback_provider)
    ROLE_MAP: dict[str, tuple[str, str]] = {
        "extraction":       ("claude",       "gemini"),      # Sonnet for parsing
        "complex_reasoning":("claude",       "azure_openai"),# Opus for reasoning
        "classification":   ("claude",       "gemini"),      # Sonnet for routing
        "embedding":        ("azure_openai", "gemini"),      # text-embedding-3-large
        "synthesis":        ("claude",       "azure_openai"),# Opus for truth synthesis
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
```

---

## 4. PromptLoader

All LLM prompts are stored as `.txt` files under `/prompt/` with `{variable_name}` placeholders. The loader reads them **once at startup** and caches in memory.

```python
# backend/app/agents/prompt_loader.py

import os
import re
from pathlib import Path

from app.exceptions import PromptTemplateError
# PromptTemplateError is defined centrally in backend/app/exceptions.py (see doc 07)
# as a subclass of ContractIQError.

class PromptLoader:
    """
    Loads and caches prompt templates from /prompt/ directory.

    Uses a sentinel-based substitution for {variable_name} placeholders that
    safely handles literal braces in JSON examples (escaped as {{ and }}).

    Approach: replace {{ / }} with sentinels FIRST, then substitute {var},
    then restore sentinels. This avoids lookahead regex edge cases with
    adjacent placeholders like {result_value}}}.
    """

    _PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")

    def __init__(self, prompt_dir: str | Path = "prompt") -> None:
        self._prompt_dir = Path(prompt_dir)
        self._cache: dict[str, str] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load every .txt file in prompt_dir into memory at startup."""
        if not self._prompt_dir.exists():
            raise PromptTemplateError(f"Prompt directory not found: {self._prompt_dir}")
        for txt_file in self._prompt_dir.glob("*.txt"):
            key = txt_file.stem                       # "document_parser_system"
            self._cache[key] = txt_file.read_text()

    # Sentinels for escaped braces (chosen to never appear in real prompts)
    _LBRACE_SENTINEL = "\x00LBRACE\x00"
    _RBRACE_SENTINEL = "\x00RBRACE\x00"

    def get(self, template_name: str, **variables: str) -> str:
        """
        Return a prompt with {variable_name} placeholders substituted.

        Substitution rules:
        - {variable_name} → replaced with provided value
        - {{literal_braces}} → preserved as {literal_braces} (JSON examples)

        Raises PromptTemplateError if the template is not found or a variable is missing.
        """
        raw = self._cache.get(template_name)
        if raw is None:
            raise PromptTemplateError(f"Prompt template not found: {template_name}")

        # Step 1: Protect escaped braces by replacing {{ and }} with sentinels
        temp = raw.replace("{{", self._LBRACE_SENTINEL).replace("}}", self._RBRACE_SENTINEL)

        # Step 2: Replace {variable} placeholders
        missing = []
        def _replacer(match: re.Match) -> str:
            key = match.group(1)
            if key in variables:
                return variables[key]
            missing.append(key)
            return match.group(0)

        result = self._PLACEHOLDER_RE.sub(_replacer, temp)
        if missing:
            raise PromptTemplateError(
                f"Missing variable(s) in prompt '{template_name}': {missing}"
            )

        # Step 3: Restore escaped braces
        result = result.replace(self._LBRACE_SENTINEL, "{").replace(self._RBRACE_SENTINEL, "}")
        return result

    @property
    def loaded_templates(self) -> list[str]:
        return list(self._cache.keys())
```

### Naming Convention

| Agent | Purpose | File |
|-------|---------|------|
| DocumentParserAgent | System prompt | `prompt/document_parser_system.txt` |
| DocumentParserAgent | Extraction | `prompt/document_parser_extraction.txt` |
| AmendmentTrackerAgent | System prompt | `prompt/amendment_tracker_system.txt` |
| AmendmentTrackerAgent | Analysis | `prompt/amendment_tracker_analysis.txt` |
| TemporalSequencerAgent | Date inference | `prompt/temporal_sequencer_date_inference.txt` |
| OverrideResolutionAgent | System prompt | `prompt/override_resolution_system.txt` |
| OverrideResolutionAgent | Apply overrides | `prompt/override_resolution_apply.txt` |
| ConflictDetectionAgent | System prompt | `prompt/conflict_detection_system.txt` |
| ConflictDetectionAgent | Analyze | `prompt/conflict_detection_analyze.txt` |
| DependencyMapperAgent | System prompt | `prompt/dependency_mapper_system.txt` |
| DependencyMapperAgent | Identify deps | `prompt/dependency_mapper_identify.txt` |
| RippleEffectAnalyzerAgent | System prompt | `prompt/ripple_effect_system.txt` |
| RippleEffectAnalyzerAgent | Hop analysis | `prompt/ripple_effect_hop_analysis.txt` |
| RippleEffectAnalyzerAgent | Recommendations | `prompt/ripple_effect_recommendations.txt` |
| QueryRouter | Classification | `prompt/query_router_classify.txt` |
| TruthSynthesizer | Synthesis | `prompt/truth_synthesizer_answer.txt` |
| ReusabilityAnalyzerAgent | (Phase 2) | `prompt/reusability_analyzer_system.txt` |

Total: **26 prompt files** at launch (see doc 07 §5 for the full manifest).

---

## 5. BaseAgent Abstract Class

The BaseAgent provides an **agentic execution framework** with tool use, self-verification, structured output enforcement, chain-of-thought reasoning, and confidence-gated re-processing. Every agent inherits these capabilities.

### 5.1 Supporting Infrastructure

```python
# backend/app/agents/base.py

import asyncio
import json
import re
import logging
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Union
from uuid import uuid4
from pydantic import BaseModel

from app.agents.llm_providers import LLMProvider, LLMProviderFactory
from app.agents.prompt_loader import PromptLoader
from app.agents.config import AgentConfig
from app.models.agent_schemas import LLMResponse
from app.exceptions import LLMProviderError, LLMResponseError

logger = logging.getLogger(__name__)

# ── Transient error detection ──────────────────────────────────
_TRANSIENT_STATUS_CODES = {429, 500, 502, 503}

def _is_transient(error: Exception) -> bool:
    """Check if an error is transient and worth retrying."""
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status and int(status) in _TRANSIENT_STATUS_CODES:
        return True
    if isinstance(error, (asyncio.TimeoutError, ConnectionError, OSError)):
        return True
    error_name = type(error).__name__
    if error_name in ("RateLimitError", "InternalServerError", "APITimeoutError", "ServiceUnavailableError"):
        return True
    return False

# ── Model context window limits ────────────────────────────────
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "claude-opus-4-5-20250514": 200_000,
    "claude-sonnet-4-5-20250929": 200_000,
    "gpt-5-mini": 128_000,
    "gemini-2.5-flash-lite": 1_000_000,
    "gemini-2.5-pro": 1_000_000,
}

# ── Trace context for observability ────────────────────────────
@dataclass
class LLMCallRecord:
    """Record of a single LLM API call for tracing and debugging."""
    call_id: str
    agent_name: str
    prompt_template_hash: str    # sha256 of system prompt for prompt versioning
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    model: str = ""
    provider: str = ""
    reasoning_captured: bool = False

@dataclass
class TraceContext:
    """Flows through the entire pipeline for distributed tracing and cost tracking."""
    job_id: str
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def record_call(self, record: LLMCallRecord) -> None:
        self.llm_calls.append(record)
        self.total_input_tokens += record.input_tokens
        self.total_output_tokens += record.output_tokens

@dataclass
class CallResult:
    """Rich return type from call_llm_with_metadata() — includes parsed data AND LLM metadata."""
    data: dict[str, Any] | str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    model: str = ""
    provider: str = ""

# Progress callback type: sync or async
ProgressCallback = Callable[[str, int, str], Union[None, Awaitable[None]]]
```

### 5.2 BaseAgent Class

```python
class BaseAgent(ABC):
    """
    Agentic base class for all ContractIQ agents.

    Agentic capabilities (beyond basic LLM calls):
    1. **Structured output** — call_llm() accepts a response_schema (Pydantic model)
       and uses provider-native constrained decoding (Claude tool_use, OpenAI JSON schema,
       Gemini response_schema) to guarantee schema compliance.
    2. **Tool use** — call_llm_with_tools() implements a ReAct-style loop where the LLM
       can invoke tools (DB queries, search, calculations) during reasoning.
    3. **Self-verification** — _verify_output() hook lets each agent check its own output
       before returning. Override in subclasses for domain-specific validation.
    4. **Chain-of-thought** — prompts request explicit reasoning in a 'reasoning' field.
       Reasoning is captured in output models for observability and debugging.
    5. **Confidence-gated re-processing** — when output confidence < verification_threshold,
       the agent re-runs with augmented context including a self-critique.
    6. **Token estimation** — _estimate_tokens() checks input size before LLM calls to
       prevent silent context window truncation.
    7. **Provider auto-failover** — when the primary provider fails (circuit breaker open),
       automatically attempts the fallback provider.
    8. **Trace context** — all LLM calls are recorded for cost tracking and debugging.
    9. **Multi-turn reasoning** — call_llm_conversation() enables iterative refinement
       where the LLM reviews and improves its own output across multiple turns.
    10. **Rich metadata return** — call_llm_with_metadata() returns both parsed data and
        LLM response metadata (tokens, latency, model) for agents that need to report these.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_provider: LLMProvider,
        prompt_loader: PromptLoader,
        progress_callback: Optional[ProgressCallback] = None,
        fallback_provider: Optional[LLMProvider] = None,
        trace_context: Optional[TraceContext] = None,
        llm_semaphore: Optional[asyncio.Semaphore] = None,
    ) -> None:
        self.config = config
        self.llm = llm_provider
        self._fallback_llm = fallback_provider
        self.prompts = prompt_loader
        self._progress_callback = progress_callback
        self.trace = trace_context
        self._llm_semaphore = llm_semaphore or asyncio.Semaphore(999)

    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """
        Main entry point. Subclass-specific Pydantic input → Pydantic output.
        Subclasses MUST type-narrow input_data and return their specific Output model.

        The default pattern for subclass implementations:
          1. Execute core logic (one or more LLM calls, DB queries, etc.)
          2. Return output — BaseAgent.run() wraps this with verification
        """
        ...

    async def run(self, input_data: Any) -> Any:
        """
        Agentic execution wrapper around process().
        Calls process(), then applies self-verification and confidence-gated re-processing.

        Use run() instead of process() directly when you want the full agentic loop.
        The orchestrator calls run(); process() is the subclass implementation hook.
        """
        output = await self.process(input_data)

        # Self-verification: let the agent check its own output
        output = await self._verify_output(output, input_data)

        # Confidence-gated re-processing
        confidence = getattr(output, "confidence", None) or getattr(output, "extraction_confidence", None)
        if confidence is not None and confidence < self.config.verification_threshold:
            logger.warning(
                "Low confidence %.2f (threshold %.2f) for %s — re-processing with self-critique",
                confidence, self.config.verification_threshold, self.config.agent_name,
            )
            output = await self._reprocess_with_critique(input_data, output)

        return output

    # ── Self-verification ──────────────────────────────────────

    async def _verify_output(self, output: Any, input_data: Any) -> Any:
        """
        Self-verification hook. Override in subclasses to add domain-specific checks.

        Default: no-op (returns output unchanged).
        Subclass examples:
          - AmendmentTracker: verify every modification references a real section
          - OverrideResolution: verify resolved text is consistent with amendment chain
          - ConflictDetection: verify each conflict has >= 2 evidence items
        """
        return output

    async def _reprocess_with_critique(self, input_data: Any, prior_output: Any) -> Any:
        """
        Re-run process() with augmented context including a self-critique of the prior output.
        Uses a second LLM call to critique, then feeds the critique back into process().
        """
        critique_prompt = self.prompts.get("self_verification_system")
        critique_user = (
            f"You produced this output with low confidence. Review it for errors, gaps, or ambiguities:\n\n"
            f"{json.dumps(prior_output.model_dump(mode='json') if hasattr(prior_output, 'model_dump') else str(prior_output), indent=2)}"
        )
        critique = await self.call_llm(critique_prompt, critique_user, expect_json=False)

        # Store critique context for the re-run
        if hasattr(input_data, '_critique_context'):
            input_data._critique_context = critique
        logger.info("Re-processing %s with self-critique feedback", self.config.agent_name)
        return await self.process(input_data)

    # ── LLM call with structured output ────────────────────────

    async def call_llm(
        self,
        system_prompt: str,
        user_message: str,
        *,
        expect_json: bool = True,
        model: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        response_schema: Optional[type[BaseModel]] = None,
        temperature: Optional[float] = None,
    ) -> dict[str, Any] | str:
        """
        Call the LLM with optional structured output enforcement.

        Structured output strategy (provider-native):
          - Claude: Uses tool_use with response_schema as the tool's input_schema.
            The LLM MUST produce valid JSON matching the schema.
          - Azure OpenAI: Uses response_format with JSON schema.
          - Gemini: Uses response_schema parameter for constrained decoding.
          Fallback: If structured output is not supported, uses prompt-based JSON
          with the Pydantic schema injected into the prompt + 3-tier parsing.

        Chain-of-thought: When response_schema is provided and the schema has a
        'reasoning' field, the LLM is instructed to think step-by-step before
        producing the structured output.

        Token estimation: Before sending, estimates input tokens and warns/raises
        if the prompt exceeds 90% of the model's context window.

        Auto-failover: If primary provider exhausts retries, attempts fallback.

        Retry policy: Same as before — transient errors only, exponential backoff.
        """
        model = model or self.config.model_override
        max_output_tokens = max_output_tokens or self.config.max_output_tokens
        temperature = temperature if temperature is not None else self.config.temperature

        # Token estimation — prevent silent context truncation
        self._check_token_budget(system_prompt, user_message, model)

        # Try primary provider
        try:
            return await self._call_with_retries(
                self.llm, system_prompt, user_message,
                model=model, max_output_tokens=max_output_tokens,
                expect_json=expect_json, response_schema=response_schema,
                temperature=temperature,
            )
        except LLMProviderError:
            if self._fallback_llm:
                logger.warning(
                    "Primary provider failed for %s — attempting fallback provider",
                    self.config.agent_name,
                )
                return await self._call_with_retries(
                    self._fallback_llm, system_prompt, user_message,
                    model=model, max_output_tokens=max_output_tokens,
                    expect_json=expect_json, response_schema=response_schema,
                    temperature=temperature,
                )
            raise

    async def call_llm_with_metadata(
        self,
        system_prompt: str,
        user_message: str,
        **kwargs,
    ) -> "CallResult":
        """
        Like call_llm() but returns a CallResult with both parsed data and LLM response metadata
        (tokens, latency, model, provider). Use when agents need to include metadata in their output
        (e.g., analysis_model, analysis_latency_ms).
        """
        # Delegate to _call_with_retries_rich which returns CallResult
        model = kwargs.pop("model", None) or self.config.model_override
        max_output_tokens = kwargs.pop("max_output_tokens", None) or self.config.max_output_tokens
        expect_json = kwargs.pop("expect_json", True)
        response_schema = kwargs.pop("response_schema", None)
        temperature = kwargs.pop("temperature", None)
        if temperature is None:
            temperature = self.config.temperature

        self._check_token_budget(system_prompt, user_message, model)

        try:
            return await self._call_with_retries_rich(
                self.llm, system_prompt, user_message,
                model=model, max_output_tokens=max_output_tokens,
                expect_json=expect_json, response_schema=response_schema,
                temperature=temperature,
            )
        except LLMProviderError:
            if self._fallback_llm:
                return await self._call_with_retries_rich(
                    self._fallback_llm, system_prompt, user_message,
                    model=model, max_output_tokens=max_output_tokens,
                    expect_json=expect_json, response_schema=response_schema,
                    temperature=temperature,
                )
            raise

    async def _call_with_retries_rich(self, provider, system_prompt, user_message, **kwargs) -> "CallResult":
        """Like _call_with_retries but wraps result in CallResult with metadata."""
        # Implementation delegates to _call_with_retries for the core logic,
        # but captures the LLMResponse metadata before parsing strips it.
        # (Implementation note: in production, refactor _call_with_retries to return CallResult
        # and have call_llm extract just the data. For now, this is a separate method.)
        ...  # Same retry logic as _call_with_retries, but returns CallResult

    async def _call_with_retries(
        self,
        provider: LLMProvider,
        system_prompt: str,
        user_message: str,
        *,
        model: str,
        max_output_tokens: int,
        expect_json: bool,
        response_schema: Optional[type[BaseModel]],
        temperature: float,
    ) -> dict[str, Any] | str:
        """Execute LLM call with retry logic against a specific provider."""
        prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()[:12]

        # If response_schema provided, use structured output via tool_use
        use_structured = response_schema is not None

        last_error: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                async with self._llm_semaphore:
                    logger.debug(
                        "LLM call attempt=%d agent=%s model=%s provider=%s prompt_hash=%s",
                        attempt, self.config.agent_name, model, provider.provider_name, prompt_hash,
                    )
                    start = time.monotonic()

                    if use_structured:
                        # Structured output via tool_use — schema enforcement
                        tool_def = {
                            "name": "respond",
                            "description": f"Structured response for {self.config.agent_name}",
                            "input_schema": response_schema.model_json_schema(),
                        }
                        response = await asyncio.wait_for(
                            provider.complete_with_tools(
                                system_prompt, user_message,
                                tools=[tool_def],
                                model=model,
                                max_output_tokens=max_output_tokens,
                                temperature=temperature,
                            ),
                            timeout=self.config.timeout_seconds,
                        )
                    else:
                        response = await asyncio.wait_for(
                            provider.complete(
                                system_prompt, user_message,
                                model=model,
                                max_output_tokens=max_output_tokens,
                                temperature=temperature,
                            ),
                            timeout=self.config.timeout_seconds,
                        )

                    call_latency = int((time.monotonic() - start) * 1000)

                # Record trace
                if self.trace:
                    self.trace.record_call(LLMCallRecord(
                        call_id=str(uuid4()),
                        agent_name=self.config.agent_name,
                        prompt_template_hash=prompt_hash,
                        input_tokens=response.usage.get("input_tokens", 0),
                        output_tokens=response.usage.get("output_tokens", 0),
                        latency_ms=response.latency_ms,
                        model=model,
                        provider=provider.provider_name,
                    ))

                logger.info(
                    "LLM response agent=%s tokens_in=%s tokens_out=%s latency_ms=%s provider=%s",
                    self.config.agent_name,
                    response.usage.get("input_tokens"),
                    response.usage.get("output_tokens"),
                    response.latency_ms,
                    provider.provider_name,
                )

                if not expect_json and not use_structured:
                    return response.content

                if use_structured:
                    # Tool-use response: content is already structured JSON from tool input
                    return self._parse_json_response(response.content)

                return self._parse_json_response(response.content)

            except json.JSONDecodeError:
                logger.warning("JSON parse failed for %s, re-prompting once", self.config.agent_name)
                try:
                    repair_response = await asyncio.wait_for(
                        provider.complete(
                            "Return ONLY valid JSON. No markdown, no explanation.",
                            f"Fix this into valid JSON:\n{response.content}",
                            model=model, max_output_tokens=max_output_tokens,
                        ),
                        timeout=self.config.timeout_seconds,
                    )
                    return json.loads(repair_response.content)
                except (json.JSONDecodeError, Exception) as repair_err:
                    raise LLMResponseError(
                        f"JSON repair failed for {self.config.agent_name}: {repair_err}"
                    ) from repair_err

            except Exception as e:
                if not _is_transient(e):
                    logger.error(
                        "Non-transient LLM error agent=%s error=%s — not retrying",
                        self.config.agent_name, str(e),
                    )
                    raise

                last_error = e
                if attempt < self.config.max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "Transient error attempt=%d/%d agent=%s error=%s retrying_in=%ds",
                        attempt, self.config.max_retries, self.config.agent_name, str(e), wait,
                    )
                    await asyncio.sleep(wait)

        raise LLMProviderError(
            f"All {self.config.max_retries} attempts failed for {self.config.agent_name}: {last_error}",
            provider=provider.provider_name,
        )

    # ── Tool-use agentic loop ──────────────────────────────────

    async def call_llm_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[dict[str, Any]],
        *,
        max_turns: int = 5,
        model: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        ReAct-style tool-use loop: LLM reasons, calls tools, observes results, iterates.

        The LLM can invoke tools (DB queries, semantic search, calculations) during
        its reasoning. The loop continues until the LLM produces a final answer or
        max_turns is reached.

        Tool definitions follow the Anthropic tool-use format:
        [{"name": "search_clauses", "description": "...", "input_schema": {...}}]

        Agents provide tool implementations via get_tools() and _execute_tool().
        """
        model = model or self.config.model_override
        max_output_tokens = max_output_tokens or self.config.max_output_tokens
        messages = [{"role": "user", "content": user_message}]

        for turn in range(max_turns):
            async with self._llm_semaphore:
                response = await asyncio.wait_for(
                    self.llm.complete_with_tools(
                        system_prompt, messages, tools,
                        model=model, max_output_tokens=max_output_tokens,
                        temperature=self.config.temperature,
                    ),
                    timeout=self.config.timeout_seconds,
                )

            # Check if LLM wants to call tools or is done
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_result = await self._execute_tool(
                        tool_call["name"], tool_call["input"]
                    )
                    messages.append({
                        "role": "assistant",
                        "content": response.content,
                        "tool_calls": [tool_call],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_use_id": tool_call["id"],
                        "content": json.dumps(tool_result),
                    })
            else:
                # LLM is done — parse final response
                return self._parse_json_response(response.content)

        raise LLMResponseError(
            f"Tool-use loop exceeded {max_turns} turns for {self.config.agent_name}"
        )

    async def call_llm_conversation(
        self,
        system_prompt: str,
        initial_message: str,
        *,
        max_turns: int = 3,
        model: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        refinement_prompt: str = "Review your previous response. Are there any corrections, missing details, or improvements? If yes, provide the corrected version. If your analysis is complete and accurate, return it unchanged.",
    ) -> dict[str, Any]:
        """
        Multi-turn reasoning loop: LLM produces an initial response, then iteratively
        refines it through self-review turns.

        Use this for complex analysis where a single pass may miss nuances:
        - ConflictDetection: initial scan → refinement pass for subtle conflicts
        - TemporalSequencer: initial ordering → review for non-obvious precedence
        - DependencyMapper: broad sweep → focused refinement

        The loop continues until the LLM indicates no further changes or max_turns is reached.
        """
        model = model or self.config.model_override
        max_output_tokens = max_output_tokens or self.config.max_output_tokens
        messages = [{"role": "user", "content": initial_message}]

        last_result = None
        for turn in range(max_turns):
            async with self._llm_semaphore:
                response = await asyncio.wait_for(
                    self.llm.complete(
                        system_prompt,
                        messages[-1]["content"] if turn == 0 else refinement_prompt + "\n\nYour previous response:\n" + json.dumps(last_result, indent=2),
                        model=model,
                        max_output_tokens=max_output_tokens,
                        temperature=self.config.temperature,
                    ),
                    timeout=self.config.timeout_seconds,
                )
            last_result = self._parse_json_response(response.content)

            # Check if LLM signals completion (no changes)
            if last_result.get("_refinement_complete", False) or turn == max_turns - 1:
                last_result.pop("_refinement_complete", None)
                return last_result

            # Record trace
            if self.trace:
                self.trace.record_call(LLMCallRecord(
                    call_id=str(uuid4()),
                    agent_name=self.config.agent_name,
                    prompt_template_hash="conversation_turn",
                    input_tokens=response.usage.get("input_tokens", 0),
                    output_tokens=response.usage.get("output_tokens", 0),
                    latency_ms=response.latency_ms,
                    model=model,
                    provider=self.llm.provider_name,
                ))

        return last_result

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Return tool definitions for this agent. Override in subclasses to
        enable dynamic data access during LLM reasoning.

        Standard tools available to agents with database access:
        """
        return []

    # Standard tool schemas — subclasses with DB access return these from get_tools()
    STANDARD_TOOLS: list[dict[str, Any]] = [
        {
            "name": "search_clauses",
            "description": "Semantic search for clauses matching a query, optionally filtered by category.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "category": {"type": "string", "description": "Optional clause category filter (payment, insurance, etc.)"},
                    "limit": {"type": "integer", "description": "Max results to return", "default": 5},
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_clause",
            "description": "Look up a specific clause by section number.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "section_number": {"type": "string", "description": "Section number (e.g., '7.2', 'Exhibit B-2')"},
                },
                "required": ["section_number"],
            },
        },
        {
            "name": "get_dependencies",
            "description": "Get clauses that depend on or are depended upon by a given section.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "section_number": {"type": "string"},
                    "direction": {"type": "string", "enum": ["outbound", "inbound", "both"], "default": "both"},
                },
                "required": ["section_number"],
            },
        },
        {
            "name": "get_amendment_history",
            "description": "Get the full amendment history (source chain) for a specific section.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "section_number": {"type": "string"},
                },
                "required": ["section_number"],
            },
        },
    ]

    async def _execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        """
        Execute a tool call. Override in subclasses to implement tool logic.
        Raises NotImplementedError if an unknown tool is called.
        """
        raise NotImplementedError(f"Tool '{tool_name}' not implemented in {self.config.agent_name}")

    # ── Token estimation ───────────────────────────────────────

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token for English text."""
        return len(text) // 4

    def _check_token_budget(self, system_prompt: str, user_message: str, model: str) -> None:
        """Warn or raise if input is likely to exceed context window."""
        estimated_input = self._estimate_tokens(system_prompt + user_message)
        max_output = self.config.max_output_tokens or 8192
        model_limit = MODEL_CONTEXT_LIMITS.get(model, 200_000)
        total_estimated = estimated_input + max_output

        if total_estimated > model_limit:
            raise LLMProviderError(
                f"Estimated {estimated_input} input + {max_output} output tokens "
                f"exceeds {model_limit} context limit for model {model} "
                f"in agent {self.config.agent_name}. Chunk or summarize input.",
                provider=self.llm.provider_name,
            )
        if estimated_input > model_limit * 0.85:
            logger.warning(
                "Token budget warning: agent=%s estimated_input=%d model_limit=%d (%.0f%% used)",
                self.config.agent_name, estimated_input, model_limit,
                (estimated_input / model_limit) * 100,
            )

    # ── JSON parsing ───────────────────────────────────────────

    def _parse_json_response(self, content: str) -> dict[str, Any]:
        """
        3-tier JSON extraction:
          1. Direct parse
          2. Extract from ```json ... ``` code block
          3. Raise JSONDecodeError (caller handles re-prompt)
        """
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip())

        raise json.JSONDecodeError("No valid JSON found", content, 0)

    # ── Progress reporting ─────────────────────────────────────

    async def _report_progress(self, stage: str, percent: int, message: str) -> None:
        """Fire progress callback if registered (supports both sync and async callbacks)."""
        if self._progress_callback:
            result = self._progress_callback(stage, percent, message)
            if asyncio.iscoroutine(result):
                await result
        logger.info("Progress agent=%s stage=%s pct=%d msg=%s", self.config.agent_name, stage, percent, message)
```

---

## 6. AgentConfig Dataclass

```python
# backend/app/agents/config.py

from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class AgentConfig:
    """Immutable configuration for a single agent instance."""

    agent_name: str                           # e.g. "document_parser"
    llm_role: str                             # maps to LLMProviderFactory.ROLE_MAP key
    model_override: Optional[str] = None      # e.g. "claude-opus-4-5-20250514" to force a specific model
    max_output_tokens: Optional[int] = None   # None → provider default
    max_retries: int = 3                      # total attempts (not retries — 3 means try 3 times)
    timeout_seconds: int = 120                # per-call hard timeout via asyncio.wait_for
    temperature: float = 0.0                  # task-appropriate temperature (see table below)
    verification_threshold: float = 0.75      # confidence below this triggers re-processing
```

### Default Configs per Agent

| Agent | llm_role | model_override | max_output_tokens | temperature | verification_threshold |
|-------|----------|----------------|-------------------|-------------|----------------------|
| DocumentParserAgent | `extraction` | `claude-sonnet-4-5-20250929` | 8192 | 0.0 | 0.80 |
| AmendmentTrackerAgent | `complex_reasoning` | `claude-opus-4-5-20250514` | 8192 | 0.0 | 0.75 |
| TemporalSequencerAgent | `extraction` | `claude-sonnet-4-5-20250929` | 4096 | 0.0 | 0.80 |
| OverrideResolutionAgent | `complex_reasoning` | `claude-opus-4-5-20250514` | 8192 | 0.0 | 0.75 |
| ConflictDetectionAgent | `complex_reasoning` | `claude-opus-4-5-20250514` | 8192 | 0.2 | 0.70 |
| DependencyMapperAgent | `complex_reasoning` | `claude-opus-4-5-20250514` | 8192 | 0.1 | 0.75 |
| RippleEffectAnalyzerAgent | `complex_reasoning` | `claude-opus-4-5-20250514` | 8192 | 0.2 | 0.70 |
| QueryRouter | `classification` | `claude-sonnet-4-5-20250929` | 1024 | 0.0 | 0.85 |
| TruthSynthesizer | `synthesis` | `claude-opus-4-5-20250514` | 8192 | 0.1 | 0.80 |

**Temperature rationale:**
- **0.0** for deterministic tasks (parsing, override resolution) where consistency is critical
- **0.1** for synthesis tasks where slight variation improves natural language quality
- **0.2** for exploratory tasks (conflict detection, ripple analysis) where the model should consider unlikely-but-possible interpretations

---

## 7. LLMResponse Model

```python
# backend/app/models/agent_schemas.py (excerpt — full definition in doc 02)

from pydantic import BaseModel, Field
from typing import Optional

class LLMResponse(BaseModel):
    """Standardized response from any LLM provider."""
    success: bool
    content: str = ""
    usage: dict[str, int] = Field(default_factory=dict)  # {"input_tokens": N, "output_tokens": M}
    model: str = ""
    latency_ms: int = 0
    provider: str = ""
    error: Optional[str] = None
```

---

## 8. Max Output Token Policy

Per CLAUDE.md, every LLM call MUST set `max_output_tokens` to the model's maximum. The provider implementations enforce this, but agents can override:

| Provider | Model | Max Output Tokens |
|----------|-------|-------------------|
| Claude | claude-opus-4-5-20250514 | 8192 |
| Claude | claude-sonnet-4-5-20250929 | 8192 |
| Azure OpenAI | gpt-5-mini | 16384 |
| Gemini | gemini-2.5-flash-lite | 65536 |
| Gemini | gemini-2.5-pro | 65536 |
| Gemini | gemini-2.0-flash-exp | 8192 |
| Gemini | gemini-1.5-flash | 8192 |
| Gemini | gemini-1.5-pro | 8192 |

---

## 9. File Layout Summary

```
backend/app/agents/
├── __init__.py
├── base.py              # BaseAgent ABC
├── llm_providers.py     # LLMProvider protocol + 3 implementations + factory
├── prompt_loader.py     # PromptLoader + PromptTemplateError
├── config.py            # AgentConfig dataclass
├── document_parser.py   # (doc 03)
├── amendment_tracker.py # (doc 03)
├── temporal_sequencer.py# (doc 03)
├── override_resolution.py # (doc 04)
├── conflict_detection.py  # (doc 04)
├── dependency_mapper.py   # (doc 04)
├── ripple_effect.py       # (doc 05)
├── reusability.py         # (doc 05 — Phase 2 stub)
├── query_router.py        # (doc 06)
└── truth_synthesizer.py   # (doc 06)

backend/app/models/
├── __init__.py
├── agent_schemas.py     # all Pydantic I/O models (doc 02)
├── enums.py             # shared enums (doc 02)
└── events.py            # progress event models (doc 02)

prompt/
├── document_parser_system.txt
├── document_parser_extraction.txt
├── amendment_tracker_system.txt
├── amendment_tracker_analysis.txt
├── temporal_sequencer_date_inference.txt
├── override_resolution_system.txt
├── override_resolution_apply.txt
├── conflict_detection_system.txt
├── conflict_detection_analyze.txt
├── dependency_mapper_system.txt
├── dependency_mapper_identify.txt
├── ripple_effect_system.txt
├── ripple_effect_hop_analysis.txt
├── ripple_effect_recommendations.txt
├── query_router_classify.txt
├── truth_synthesizer_answer.txt
└── reusability_analyzer_system.txt
```

---

## 10. Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| **Protocol over ABC for LLMProvider** | Structural typing means `MockLLMProvider` in tests needs zero inheritance. Any object with `complete()`, `complete_with_tools()`, and `close()` works. `@runtime_checkable` enables `isinstance()` checks. |
| **Factory with role mapping** | Agents declare intent ("I need complex reasoning"), factory resolves to concrete provider. Swapping providers requires only a config change. |
| **Prompts loaded at startup** | Eliminates filesystem I/O per LLM call. ~17 small .txt files fit comfortably in memory. |
| **Regex-based prompt substitution** | `str.format()` breaks on literal `{`/`}` in JSON examples within prompt files. Regex matches `{variable}` while preserving `{{escaped}}`. |
| **3-tier JSON parsing as fallback** | Used only when structured output (tool_use) is not available. LLMs frequently wrap JSON in markdown code blocks. |
| **Structured output via tool_use (primary)** | Claude's tool_use, OpenAI's JSON schema, and Gemini's response_schema provide schema-enforced output. Eliminates prompt-and-pray JSON issues. The Pydantic model's `model_json_schema()` is passed directly to the provider. |
| **run() wraps process()** | `run()` adds verification and confidence-gated re-processing around the subclass `process()` implementation. Orchestrator calls `run()`; agents implement `process()`. |
| **Self-verification hook** | `_verify_output()` lets each agent implement domain-specific output validation. Default is no-op — agents opt in. Catches LLM hallucinations before they propagate downstream. |
| **Confidence-gated re-processing** | When confidence < `verification_threshold`, the agent self-critiques and re-runs. This prevents low-quality outputs from flowing into downstream agents unchecked. |
| **Tool-use agentic loop** | `call_llm_with_tools()` implements a ReAct-style loop where agents can query databases, search pgvector, or look up specific clauses during reasoning — rather than pre-loading all context into the prompt. |
| **Token estimation before calls** | `_check_token_budget()` prevents silent context window truncation. Raises `LLMProviderError` when estimated input exceeds model limits. |
| **Auto-failover to fallback provider** | When primary provider exhausts retries, automatically attempts fallback. This is infrastructure resilience, not code fallback — the "never fall back" policy applies to workaround code, not provider redundancy. |
| **TraceContext for observability** | Every LLM call records prompt hash, tokens, latency, and model. Enables cost tracking, prompt versioning, and debugging of incorrect outputs by replaying the exact call. |
| **Temperature tuning per agent** | 0.0 for deterministic tasks, 0.1-0.2 for exploratory analysis. Prevents both over-creativity in extraction and over-conservatism in conflict detection. |
| **Semaphore-gated LLM calls** | All LLM calls acquire the shared semaphore, preventing parallel agents from exceeding API rate limits. |
| **Retry only transient errors** | Only 429/500/502/503/timeout/connection errors are retried. Non-transient errors (400/401/403) raise immediately — no wasted API calls. |
| **Per-call timeout enforcement** | `asyncio.wait_for(timeout_seconds)` prevents runaway LLM calls from blocking the pipeline indefinitely. |
| **Async client cleanup** | `LLMProvider.close()` method ensures HTTP connections are properly released on shutdown. Called in FastAPI lifespan shutdown. |
| **AgentConfig as frozen dataclass** | Immutable config prevents accidental mutation during processing. |
| **Lazy client initialization** | API clients created on first use, not at import time — speeds up test imports and avoids circular dependency issues. |
| **Multi-turn reasoning via call_llm_conversation()** | Complex analysis (conflict detection, dependency mapping) benefits from iterative refinement. The LLM reviews its own output and catches missed items. Limited to 3 turns by default to bound cost. |
| **Standard tool schemas in BaseAgent** | Defined once, reused by all DB-aware agents. Subclasses override `get_tools()` to return relevant subset and `_execute_tool()` to implement. Decouples tool schema from implementation. |
| **CallResult for rich metadata** | Agents like ConflictDetection and RippleEffect need to report model name and latency in their output. `call_llm_with_metadata()` avoids manual time.monotonic() tracking in each agent. |

---

## 11. New Prompt Files (Added by Agentic Architecture)

| # | File | Purpose | Used By |
|---|------|---------|---------|
| 22 | `prompt/self_verification_system.txt` | System prompt for self-critique during confidence-gated re-processing | All agents via `_reprocess_with_critique()` |

**`prompt/self_verification_system.txt`**
```
You are reviewing your own prior analysis for errors, gaps, and ambiguities.

Your task:
1. Identify any factual errors in the output (wrong section numbers, misclassified types, etc.)
2. Identify any gaps — information that should have been extracted but was missed
3. Identify any ambiguities — places where the output is uncertain and should flag this
4. Suggest specific corrections

Be thorough and self-critical. It is better to flag a false positive than to miss an actual error.
```
