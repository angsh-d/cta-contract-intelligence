"""BaseAgent abstract class — agentic execution framework for all ContractIQ agents."""

import asyncio
import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Union
from uuid import uuid4

from pydantic import BaseModel

from app.agents.circuit_breaker import CircuitBreaker
from app.agents.config import AgentConfig
from app.agents.llm_providers import LLMProvider, LLMProviderFactory
from app.agents.prompt_loader import PromptLoader
from app.exceptions import LLMProviderError, LLMResponseError
from app.models.agent_schemas import LLMResponse

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
    "claude-sonnet-4-5-20250929": 200_000,
    "gpt-5.2": 128_000,
    "gemini-3-pro-preview": 1_000_000,
    "gemini-2.5-flash-lite": 1_000_000,
    "gemini-2.5-pro": 1_000_000,
}

# ── Trace context for observability ────────────────────────────

@dataclass
class LLMCallRecord:
    """Record of a single LLM API call for tracing and debugging."""
    call_id: str
    agent_name: str
    prompt_template_hash: str
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


class BaseAgent(ABC):
    """
    Agentic base class for all ContractIQ agents.

    Capabilities:
    1. Structured output via tool_use (response_schema → constrained decoding)
    2. Tool use — ReAct-style loop (call_llm_with_tools)
    3. Self-verification — _verify_output() hook
    4. Chain-of-thought — reasoning field capture
    5. Confidence-gated re-processing
    6. Token estimation — prevent context window overflow
    7. Provider auto-failover
    8. Trace context — cost tracking and debugging
    9. Multi-turn reasoning — call_llm_conversation()
    10. Rich metadata return — call_llm_with_metadata()
    """

    # Per-provider circuit breakers (shared across all agent instances)
    _circuit_breakers: dict[str, CircuitBreaker] = {}

    # Standard tool schemas — subclasses with DB access return these from get_tools()
    STANDARD_TOOLS: list[dict[str, Any]] = [
        {
            "name": "search_clauses",
            "description": "Semantic search for clauses matching a query, optionally filtered by category.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "category": {"type": "string", "description": "Optional clause category filter"},
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
        self._critique_context: Optional[str] = None  # Set during re-processing

        # Ensure circuit breakers exist for our providers
        for provider in (llm_provider, fallback_provider):
            if provider and provider.provider_name not in BaseAgent._circuit_breakers:
                BaseAgent._circuit_breakers[provider.provider_name] = CircuitBreaker(
                    provider_name=provider.provider_name
                )

    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Main entry point. Subclass-specific Pydantic input → Pydantic output."""
        ...

    async def run(self, input_data: Any) -> Any:
        """
        Agentic execution wrapper around process().
        Applies self-verification and confidence-gated re-processing (once only).
        """
        output = await self.process(input_data)
        output = await self._verify_output(output, input_data)

        confidence = getattr(output, "confidence", None) or getattr(output, "extraction_confidence", None)
        if confidence is not None and confidence < self.config.verification_threshold:
            logger.warning(
                "Low confidence %.2f (threshold %.2f) for %s — re-processing with self-critique",
                confidence, self.config.verification_threshold, self.config.agent_name,
            )
            reprocessed = await self._reprocess_with_critique(input_data, output)
            reprocessed_confidence = getattr(reprocessed, "confidence", None) or getattr(reprocessed, "extraction_confidence", None)
            if reprocessed_confidence is not None and reprocessed_confidence > confidence:
                output = reprocessed
            else:
                logger.info("Re-processed output not better (%.2f vs %.2f) for %s — keeping original",
                           reprocessed_confidence or 0, confidence, self.config.agent_name)

        return output

    # ── Self-verification ──────────────────────────────────────

    async def _verify_output(self, output: Any, input_data: Any) -> Any:
        """Self-verification hook. Override in subclasses for domain-specific checks."""
        return output

    async def _reprocess_with_critique(self, input_data: Any, prior_output: Any) -> Any:
        """Re-run process() with augmented context including a self-critique.

        Stores critique on self._critique_context (instance attribute) so that
        process() implementations can read it for context-aware re-processing.
        """
        critique_prompt = self.prompts.get("self_verification_system")
        critique_user = (
            f"You produced this output with low confidence. Review it for errors, gaps, or ambiguities:\n\n"
            f"{json.dumps(prior_output.model_dump(mode='json') if hasattr(prior_output, 'model_dump') else str(prior_output), indent=2)}"
        )
        critique = await self.call_llm(critique_prompt, critique_user, expect_json=False)

        # Store on agent instance — subclass process() can check self._critique_context
        self._critique_context = critique
        logger.info("Re-processing %s with self-critique feedback", self.config.agent_name)
        result = await self.process(input_data)
        self._critique_context = None  # Reset after re-processing
        return result

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

        When response_schema is provided, uses tool_use for constrained decoding.
        Includes auto-failover to fallback provider on exhausted retries.
        """
        model = model or self.config.model_override
        max_output_tokens = max_output_tokens or self.config.max_output_tokens
        temperature = temperature if temperature is not None else self.config.temperature

        self._check_token_budget(system_prompt, user_message, model)

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
                    "Primary provider failed for %s — attempting fallback provider %s",
                    self.config.agent_name, self._fallback_llm.provider_name,
                )
                return await self._call_with_retries(
                    self._fallback_llm, system_prompt, user_message,
                    model=None, max_output_tokens=max_output_tokens,
                    expect_json=expect_json, response_schema=response_schema,
                    temperature=temperature,
                )
            raise

    async def call_llm_with_metadata(
        self,
        system_prompt: str,
        user_message: str,
        **kwargs,
    ) -> CallResult:
        """Like call_llm() but returns a CallResult with both parsed data and LLM metadata."""
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
                    model=None, max_output_tokens=max_output_tokens,
                    expect_json=expect_json, response_schema=response_schema,
                    temperature=temperature,
                )
            raise

    # ── Core retry engine ──────────────────────────────────────

    async def _call_with_retries(
        self,
        provider: LLMProvider,
        system_prompt: str,
        user_message: str,
        *,
        model: Optional[str],
        max_output_tokens: int,
        expect_json: bool,
        response_schema: Optional[type[BaseModel]],
        temperature: float,
    ) -> dict[str, Any] | str:
        """Execute LLM call with retry logic, circuit breaker, and optional schema validation."""
        prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()[:12]
        use_structured = response_schema is not None
        response = None

        # Circuit breaker check
        cb = BaseAgent._circuit_breakers.get(provider.provider_name)
        if cb and not await cb.can_execute():
            raise LLMProviderError(
                f"Circuit breaker OPEN for {provider.provider_name} — skipping call for {self.config.agent_name}",
                provider=provider.provider_name,
            )

        last_error: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                async with self._llm_semaphore:
                    logger.debug(
                        "LLM call attempt=%d agent=%s model=%s provider=%s prompt_hash=%s",
                        attempt, self.config.agent_name, model, provider.provider_name, prompt_hash,
                    )

                    if use_structured:
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

                # Circuit breaker — record success
                if cb:
                    await cb.record_success()

                # Record trace
                actual_model = response.model or model or "unknown"
                if self.trace:
                    self.trace.record_call(LLMCallRecord(
                        call_id=str(uuid4()),
                        agent_name=self.config.agent_name,
                        prompt_template_hash=prompt_hash,
                        input_tokens=response.usage.get("input_tokens", 0),
                        output_tokens=response.usage.get("output_tokens", 0),
                        latency_ms=response.latency_ms,
                        model=actual_model,
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

                if not response.content or not response.content.strip():
                    raise ConnectionError(f"Empty LLM response for {self.config.agent_name}")

                parsed = self._parse_json_response(response.content)

                # Schema validation when response_schema is provided
                if response_schema is not None:
                    try:
                        response_schema.model_validate(parsed)
                    except Exception as val_err:
                        logger.warning(
                            "Schema validation failed for %s: %s — returning raw parsed dict",
                            self.config.agent_name, val_err,
                        )

                return parsed

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
                # Circuit breaker — record failure for transient errors
                if cb and _is_transient(e):
                    await cb.record_failure()

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

    async def _call_with_retries_rich(
        self,
        provider: LLMProvider,
        system_prompt: str,
        user_message: str,
        *,
        model: Optional[str],
        max_output_tokens: int,
        expect_json: bool,
        response_schema: Optional[type[BaseModel]],
        temperature: float,
    ) -> CallResult:
        """Like _call_with_retries but returns CallResult with metadata."""
        prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()[:12]
        use_structured = response_schema is not None
        response = None

        cb = BaseAgent._circuit_breakers.get(provider.provider_name)
        if cb and not await cb.can_execute():
            raise LLMProviderError(
                f"Circuit breaker OPEN for {provider.provider_name}",
                provider=provider.provider_name,
            )

        last_error: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                async with self._llm_semaphore:
                    if use_structured:
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

                if cb:
                    await cb.record_success()

                actual_model = response.model or model or "unknown"
                if self.trace:
                    self.trace.record_call(LLMCallRecord(
                        call_id=str(uuid4()),
                        agent_name=self.config.agent_name,
                        prompt_template_hash=prompt_hash,
                        input_tokens=response.usage.get("input_tokens", 0),
                        output_tokens=response.usage.get("output_tokens", 0),
                        latency_ms=response.latency_ms,
                        model=actual_model,
                        provider=provider.provider_name,
                    ))

                if not expect_json and not use_structured:
                    data = response.content
                else:
                    data = self._parse_json_response(response.content)

                return CallResult(
                    data=data,
                    input_tokens=response.usage.get("input_tokens", 0),
                    output_tokens=response.usage.get("output_tokens", 0),
                    latency_ms=response.latency_ms,
                    model=response.model,
                    provider=response.provider,
                )

            except json.JSONDecodeError:
                logger.warning("JSON parse failed for %s in rich call, re-prompting", self.config.agent_name)
                try:
                    repair_response = await asyncio.wait_for(
                        provider.complete(
                            "Return ONLY valid JSON. No markdown, no explanation.",
                            f"Fix this into valid JSON:\n{response.content}",
                            model=model, max_output_tokens=max_output_tokens,
                        ),
                        timeout=self.config.timeout_seconds,
                    )
                    data = json.loads(repair_response.content)
                    return CallResult(
                        data=data,
                        input_tokens=response.usage.get("input_tokens", 0),
                        output_tokens=response.usage.get("output_tokens", 0),
                        latency_ms=response.latency_ms,
                        model=response.model,
                        provider=response.provider,
                    )
                except (json.JSONDecodeError, Exception) as repair_err:
                    raise LLMResponseError(
                        f"JSON repair failed for {self.config.agent_name}: {repair_err}"
                    ) from repair_err

            except Exception as e:
                if cb and _is_transient(e):
                    await cb.record_failure()
                if not _is_transient(e):
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
        """ReAct-style tool-use loop: LLM reasons, calls tools, observes results, iterates.

        Uses Anthropic-compatible message format:
        - Assistant message with content array (text + tool_use blocks)
        - User message with tool_result content blocks
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

            # Record trace for each turn
            if self.trace:
                self.trace.record_call(LLMCallRecord(
                    call_id=str(uuid4()),
                    agent_name=self.config.agent_name,
                    prompt_template_hash="tool_use_turn",
                    input_tokens=response.usage.get("input_tokens", 0),
                    output_tokens=response.usage.get("output_tokens", 0),
                    latency_ms=response.latency_ms,
                    model=model or "unknown",
                    provider=self.llm.provider_name,
                ))

            if response.tool_calls:
                # Build assistant message with text + tool_use blocks (Anthropic format)
                assistant_content = []
                if response.content:
                    assistant_content.append({"type": "text", "text": response.content})
                for tc in response.tool_calls:
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["input"],
                    })
                messages.append({"role": "assistant", "content": assistant_content})

                # Execute all tool calls and build tool_result blocks
                tool_results = []
                for tc in response.tool_calls:
                    result = await self._execute_tool(tc["name"], tc["input"])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": json.dumps(result) if not isinstance(result, str) else result,
                    })
                messages.append({"role": "user", "content": tool_results})
            else:
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
        """Multi-turn reasoning loop: LLM produces initial response, then iteratively refines."""
        model = model or self.config.model_override
        max_output_tokens = max_output_tokens or self.config.max_output_tokens
        messages = [{"role": "user", "content": initial_message}]

        last_result = None
        for turn in range(max_turns):
            async with self._llm_semaphore:
                prompt_content = messages[-1]["content"] if turn == 0 else (
                    refinement_prompt + "\n\nYour previous response:\n" + json.dumps(last_result, indent=2)
                )
                response = await asyncio.wait_for(
                    self.llm.complete(
                        system_prompt, prompt_content,
                        model=model,
                        max_output_tokens=max_output_tokens,
                        temperature=self.config.temperature,
                    ),
                    timeout=self.config.timeout_seconds,
                )
            last_result = self._parse_json_response(response.content)

            # Record trace BEFORE checking early return — ensures final turn is always traced
            if self.trace:
                self.trace.record_call(LLMCallRecord(
                    call_id=str(uuid4()),
                    agent_name=self.config.agent_name,
                    prompt_template_hash="conversation_turn",
                    input_tokens=response.usage.get("input_tokens", 0),
                    output_tokens=response.usage.get("output_tokens", 0),
                    latency_ms=response.latency_ms,
                    model=model or "unknown",
                    provider=self.llm.provider_name,
                ))

            if last_result.get("_refinement_complete", False) or turn == max_turns - 1:
                last_result.pop("_refinement_complete", None)
                return last_result

        return last_result

    def get_tools(self) -> list[dict[str, Any]]:
        """Return tool definitions. Override in subclasses to enable dynamic data access."""
        return []

    async def _execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        """Execute a tool call. Override in subclasses to implement tool logic."""
        raise NotImplementedError(f"Tool '{tool_name}' not implemented in {self.config.agent_name}")

    # ── Token estimation ───────────────────────────────────────

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token for English text."""
        return len(text) // 4

    def _check_token_budget(self, system_prompt: str, user_message: str, model: Optional[str]) -> None:
        """Warn or raise if input is likely to exceed context window."""
        estimated_input = self._estimate_tokens(system_prompt + user_message)
        max_output = self.config.max_output_tokens or 8192
        model_limit = MODEL_CONTEXT_LIMITS.get(model or "", 200_000)
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
