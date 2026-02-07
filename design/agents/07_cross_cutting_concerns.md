# 07 — Cross-Cutting Concerns

> Error hierarchy, retry policy, circuit breaker, logging, prompt management, model assignment, security
> File locations: `backend/app/exceptions.py`, `backend/app/agents/circuit_breaker.py`, scattered across agent modules

---

## 1. Error Hierarchy

All ContractIQ exceptions descend from a single base so callers can catch broadly or narrowly.

```python
# backend/app/exceptions.py

class ContractIQError(Exception):
    """Base exception for all ContractIQ errors."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}

# ── Document layer ───────────────────────────────────────

class DocumentExtractionError(ContractIQError):
    """PDF/DOCX could not be opened, parsed, or structured."""
    pass

# ── LLM layer ────────────────────────────────────────────

class LLMResponseError(ContractIQError):
    """LLM returned invalid/unparseable content after all retries."""
    pass

class LLMProviderError(ContractIQError):
    """LLM provider API error (auth, rate limit, timeout, server error)."""
    def __init__(self, message: str, provider: str, status_code: int | None = None, **kw):
        super().__init__(message, **kw)
        self.provider = provider
        self.status_code = status_code

# ── Agent layer ──────────────────────────────────────────

class AgentProcessingError(ContractIQError):
    """An agent's process() method failed."""
    def __init__(self, message: str, agent_name: str, **kw):
        super().__init__(message, **kw)
        self.agent_name = agent_name

# ── Database layer ───────────────────────────────────────

class DatabaseError(ContractIQError):
    """PostgreSQL (NeonDB), ChromaDB, or Redis operation failed."""
    def __init__(self, message: str, database: str, **kw):
        super().__init__(message, **kw)
        self.database = database   # "postgresql", "chromadb", "redis"

# ── Pipeline layer ───────────────────────────────────────

class PipelineError(ContractIQError):
    """Pipeline-level failure (orchestrator, task queue)."""
    def __init__(self, message: str, stage: str, **kw):
        super().__init__(message, **kw)
        self.stage = stage

# ── Prompt layer ────────────────────────────────────────

class PromptTemplateError(ContractIQError):
    """Prompt file missing or variable placeholder cannot be resolved."""
    pass
```

### Hierarchy Tree

```
ContractIQError
├── DocumentExtractionError
├── LLMResponseError
├── LLMProviderError
├── AgentProcessingError
├── DatabaseError
├── PipelineError
└── PromptTemplateError
```

---

## 2. Retry Policy

**Core principle:** Retry transient errors only. Never fall back to cached results, alternative approaches, or workarounds. Fix the actual error.

### LLM Calls

| Parameter | Value |
|-----------|-------|
| Max retries | 2 (3 total attempts) |
| Backoff | Exponential: 2s, 4s |
| Retryable errors | Rate limit (429), server error (500/502/503), timeout |
| Non-retryable | Auth failure (401/403), invalid request (400), content policy |
| After all retries fail | Raise `LLMProviderError` — never fall back |

### Database Operations

| Parameter | Value |
|-----------|-------|
| Max retries | 1 (2 total attempts) |
| Backoff | Fixed 1s |
| Retryable errors | Connection timeout, pool exhausted |
| Non-retryable | Constraint violation, syntax error |
| After all retries fail | Raise `DatabaseError` |

### Implementation in BaseAgent

```python
# Defined in BaseAgent.call_llm() — see doc 01
# The retry loop only retries TRANSIENT errors:
for attempt in range(1, self.config.max_retries + 1):
    try:
        response = await asyncio.wait_for(
            self.llm.complete(...),
            timeout=self.config.timeout_seconds,
        )
        return self._parse_json_response(response.content)
    except json.JSONDecodeError:
        # JSON repair — ONE re-prompt attempt; raise LLMResponseError if that also fails
        ...
    except Exception as e:
        if not _is_transient(e):
            raise  # non-transient errors (400/401/403) raise immediately
        wait = 2 ** attempt
        await asyncio.sleep(wait)
raise LLMProviderError(...)  # propagate — NEVER fall back
```

---

## 3. Circuit Breaker

Per-provider circuit breaker prevents cascading failures when an LLM provider is down.

```python
# backend/app/agents/circuit_breaker.py

import asyncio
import time
from enum import StrEnum

class CircuitState(StrEnum):
    CLOSED = "closed"       # normal operation
    OPEN = "open"           # blocking requests
    HALF_OPEN = "half_open" # testing with single request

class CircuitBreaker:
    """
    Per-provider circuit breaker (async-safe with asyncio.Lock).

    Thresholds:
    - failure_threshold: 5 failures within window → OPEN
    - failure_window: 300 seconds (5 minutes)
    - recovery_timeout: 30 seconds before HALF_OPEN
    """

    def __init__(
        self,
        provider_name: str,
        failure_threshold: int = 5,
        failure_window: int = 300,
        recovery_timeout: int = 30,
    ):
        self.provider_name = provider_name
        self.failure_threshold = failure_threshold
        self.failure_window = failure_window
        self.recovery_timeout = recovery_timeout

        self._state = CircuitState.CLOSED
        self._failures: list[float] = []  # timestamps of recent failures
        self._opened_at: float = 0
        self._lock = asyncio.Lock()       # protects state mutations from concurrent access

    def _check_recovery(self) -> CircuitState:
        """Check if an OPEN circuit should transition to HALF_OPEN (no mutation, read-safe)."""
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._opened_at > self.recovery_timeout:
                return CircuitState.HALF_OPEN
        return self._state

    @property
    def state(self) -> CircuitState:
        """Read current state. Transitions OPEN→HALF_OPEN are deferred to can_execute()."""
        return self._check_recovery()

    async def record_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failures.clear()  # only clear on HALF_OPEN → CLOSED transition

    async def record_failure(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._failures.append(now)
            # Prune old failures outside window
            self._failures = [t for t in self._failures if now - t < self.failure_window]

            if len(self._failures) >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = now

    async def can_execute(self) -> bool:
        async with self._lock:
            # Perform OPEN → HALF_OPEN transition under lock
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._opened_at > self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
            if self._state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
                return True
            return False  # OPEN — block
```

### Integration Point

```python
# In LLM provider wrapper (called by BaseAgent.call_llm)
if not await circuit_breaker.can_execute():
    raise LLMProviderError(
        f"Circuit breaker OPEN for {provider_name}",
        provider=provider_name,
    )
try:
    response = await provider.complete(...)
    await circuit_breaker.record_success()
    return response
except Exception as e:
    await circuit_breaker.record_failure()
    raise
```

---

## 4. Logging

All logs go to `./tmp/` directory only (per CLAUDE.md policy).

### Configuration

```python
# backend/app/logging_config.py (or use pipeline.logging_config if shared)

import logging
import os
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    """
    Configure logging for ContractIQ.

    Args:
        log_level: DEBUG, INFO, WARNING, ERROR
        log_file: Optional filename (created in ./tmp/)
    """
    tmp_dir = Path("./tmp")
    tmp_dir.mkdir(exist_ok=True)

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        file_path = tmp_dir / log_file
        handlers.append(logging.FileHandler(file_path))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s [%(funcName)s] %(message)s",
        handlers=handlers,
    )
```

### Structured Log Fields

Every agent log line includes:

| Field | Example |
|-------|---------|
| `agent_name` | `document_parser` |
| `job_id` | `abc-123-def` |
| `tokens_in` | `4521` |
| `tokens_out` | `2103` |
| `latency_ms` | `3450` |
| `model` | `claude-sonnet-4-5-20250929` |
| `provider` | `claude` |
| `stage` | `text_extraction` |

### Log Levels by Content

| Level | What Gets Logged |
|-------|-----------------|
| `DEBUG` | Full prompt content (system + user), raw LLM response |
| `INFO` | Agent start/finish, progress events, token counts, latencies |
| `WARNING` | Retries, JSON repair attempts, circuit breaker state changes |
| `ERROR` | Agent failures, pipeline errors, database errors |

### Log File Names

| Module | Log File |
|--------|----------|
| Ingestion pipeline | `./tmp/pipeline_ingestion.log` |
| Query pipeline | `./tmp/pipeline_query.log` |
| LLM providers | `./tmp/llm_providers.log` |
| Agent orchestrator | `./tmp/orchestrator.log` |
| WebSocket | `./tmp/websocket.log` |

### Distributed Tracing

Every pipeline execution creates a `TraceContext` (defined in doc 01) that flows through all agents:

```python
@dataclass
class TraceContext:
    job_id: str
    trace_id: str                    # UUID for full pipeline trace
    llm_calls: list[LLMCallRecord]  # every LLM API call recorded
    total_input_tokens: int = 0
    total_output_tokens: int = 0
```

Each `LLMCallRecord` captures:
- `prompt_template_hash` — sha256 of the system prompt for prompt versioning
- `input_tokens`, `output_tokens`, `latency_ms` — for cost and performance tracking
- `model`, `provider` — which LLM was used (including failover events)

The orchestrator creates one TraceContext per `process_contract_stack()` call and passes it to all agents. After pipeline completion, the trace is persisted to PostgreSQL for debugging and cost analysis:

```sql
CREATE TABLE pipeline_traces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id VARCHAR(255) NOT NULL,
    trace_id VARCHAR(255) NOT NULL,
    contract_stack_id UUID REFERENCES contract_stacks(id),
    total_input_tokens INT DEFAULT 0,
    total_output_tokens INT DEFAULT 0,
    total_llm_calls INT DEFAULT 0,
    total_latency_ms INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 5. Prompt Management

### Manifest — All 26 Prompt Files

| # | File | Agent | Purpose | Key Placeholders |
|---|------|-------|---------|-----------------|
| 1 | `document_parser_system.txt` | DocumentParserAgent | System prompt | (none) |
| 2 | `document_parser_extraction.txt` | DocumentParserAgent | Extraction call | `{document_type}`, `{raw_text}` |
| 3 | `amendment_tracker_system.txt` | AmendmentTrackerAgent | System prompt | (none) |
| 4 | `amendment_tracker_analysis.txt` | AmendmentTrackerAgent | Analysis call | `{amendment_number}`, `{amendment_text}`, `{original_sections}`, `{prior_modifications}` |
| 5 | `temporal_sequencer_date_inference.txt` | TemporalSequencerAgent | Date inference | (none — input via user message) |
| 6 | `override_resolution_system.txt` | OverrideResolutionAgent | System prompt | (none) |
| 7 | `override_resolution_apply.txt` | OverrideResolutionAgent | Apply overrides | `{section_number}`, `{original_text}`, `{original_source}`, `{amendment_chain}` |
| 8 | `conflict_detection_system.txt` | ConflictDetectionAgent | System prompt | (none) |
| 9 | `conflict_detection_analyze.txt` | ConflictDetectionAgent | Analyze clauses | `{grouped_clauses}`, `{dependency_graph}`, `{context}` |
| 10 | `dependency_mapper_system.txt` | DependencyMapperAgent | System prompt | (none) |
| 11 | `dependency_mapper_identify.txt` | DependencyMapperAgent | Identify deps | `{clauses}` |
| 12 | `ripple_effect_system.txt` | RippleEffectAnalyzerAgent | System prompt | (none) |
| 13 | `ripple_effect_hop_analysis.txt` | RippleEffectAnalyzerAgent | Per-hop analysis | `{section_number}`, `{current_text}`, `{proposed_text}`, `{hop_number}`, `{hop_clauses}` |
| 14 | `ripple_effect_recommendations.txt` | RippleEffectAnalyzerAgent | Recommendation synthesis | (none — input via user message) |
| 15 | `query_router_classify.txt` | QueryRouter | Query classification | (none — input via user message) |
| 16 | `truth_synthesizer_answer.txt` | TruthSynthesizer | Answer synthesis | (none — input via user message) |
| 17 | `reusability_analyzer_system.txt` | ReusabilityAnalyzerAgent | Phase 2 | (TBD) |
| 18 | `query_router_input.txt` | QueryRouter | User query input (delimited) | `{query_text}` |
| 19 | `ripple_effect_recommendations_input.txt` | RippleEffectAnalyzerAgent | Recommendation synthesis input | `{section_number}`, `{current_text}`, `{proposed_text}`, `{impacts_json}` |
| 20 | `truth_synthesizer_input.txt` | TruthSynthesizer | User query + clauses + conflicts input | `{query_text}`, `{query_type}`, `{relevant_clauses}`, `{known_conflicts}` |
| 21 | `truth_synthesizer_ripple_input.txt` | TruthSynthesizer | Ripple analysis query input | `{query_text}`, `{impact_summary}` |
| 22 | `self_verification_system.txt` | All agents | Self-critique for confidence-gated re-processing | (none) |
| 23 | `amendment_tracker_buried_scan.txt` | AmendmentTracker | Adversarial scan for buried changes | `{amendment_text}`, `{already_found}` |
| 24 | `amendment_tracker_buried_scan_input.txt` | AmendmentTracker | User prompt for buried scan | `{amendment_text}`, `{already_found}` |
| 25 | `temporal_sequencer_system.txt` | TemporalSequencerAgent | System prompt for LLM-first temporal reasoning | (none) |
| 26 | `temporal_sequencer_ordering.txt` | TemporalSequencerAgent | LLM-driven document ordering and supersession | `{documents_json}` |

### Naming Convention

```
{agent_name}_{purpose}.txt
```

Where `agent_name` matches `AgentConfig.agent_name` (snake_case) and `purpose` describes the specific call.

### Error Handling

```python
# PromptLoader raises PromptTemplateError for:
# 1. Missing template file
PromptTemplateError("Prompt template not found: nonexistent_template")

# 2. Missing variable in substitution
PromptTemplateError("Missing variable in prompt 'override_resolution_apply': 'section_number'")
```

### Demo vs. Production Prompt Separation

The conflict detection prompts contain **generic** conflict detection patterns that work for any contract stack. Demo-specific HEARTBEAT-3 pain point mapping (Pain Points #1-5) is handled as a **post-processing step** in the orchestrator, not embedded in the core prompts. This ensures:

1. **Generic prompts** — Core detection works for any clinical trial agreement, not just HEARTBEAT-3
2. **Pain point mapping** — The orchestrator maps detected conflicts to known pain points during testing/demo only
3. **Extensibility** — New conflict patterns are discovered by the LLM, not limited to a predefined list
4. **A/B testability** — Prompt changes can be evaluated against the HEARTBEAT-3 golden labels without those labels being in the prompt itself

---

## 6. Model Assignment Matrix

| Agent | Primary Provider | Primary Model | Fallback Provider | Fallback Model | Rationale |
|-------|-----------------|---------------|-------------------|----------------|-----------|
| DocumentParserAgent | Claude | claude-sonnet-4-5-20250929 | Gemini | gemini-2.5-flash-lite | Extraction task — Sonnet sufficient |
| AmendmentTrackerAgent | Claude | claude-opus-4-5-20250514 | Azure OpenAI | gpt-5-mini | Complex legal reasoning |
| TemporalSequencerAgent | Claude | claude-sonnet-4-5-20250929 | Gemini | gemini-2.5-flash-lite | Simple — mostly deterministic |
| OverrideResolutionAgent | Claude | claude-opus-4-5-20250514 | Azure OpenAI | gpt-5-mini | Complex clause resolution |
| ConflictDetectionAgent | Claude | claude-opus-4-5-20250514 | Azure OpenAI | gpt-5-mini | Complex multi-clause analysis |
| DependencyMapperAgent | Claude | claude-opus-4-5-20250514 | Azure OpenAI | gpt-5-mini | Semantic dependency identification |
| RippleEffectAnalyzerAgent | Claude | claude-opus-4-5-20250514 | Azure OpenAI | gpt-5-mini | Multi-hop reasoning |
| QueryRouter | Claude | claude-sonnet-4-5-20250929 | Gemini | gemini-2.5-flash-lite | Fast classification (<2s) |
| TruthSynthesizer | Claude | claude-opus-4-5-20250514 | Azure OpenAI | gpt-5-mini | Comprehensive synthesis |
| ReusabilityAnalyzerAgent | Claude | claude-sonnet-4-5-20250929 | Gemini | gemini-2.5-flash-lite | Phase 2 — complexity TBD |
| Embeddings | Azure OpenAI | text-embedding-3-large | Gemini | embedding-001 | Azure creds already in .env |

### Temperature Configuration

Temperature is configured per-agent in AgentConfig (see doc 01 §6) based on task characteristics:

| Task Type | Temperature | Rationale |
|-----------|-------------|-----------|
| Deterministic extraction/parsing | 0.0 | Consistency critical — same input should produce same output |
| Legal text application (overrides) | 0.0 | Faithfulness to amendment language is paramount |
| Dependency identification | 0.1 | Mostly deterministic with slight flexibility for semantic links |
| Conflict detection | 0.2 | Want the model to explore unlikely-but-possible interpretations |
| Ripple effect analysis | 0.2 | Speculative impact assessment benefits from broader exploration |
| Query classification | 0.0 | Deterministic routing |
| Answer synthesis | 0.1 | Natural language quality benefits from slight variation |

### Fallback Invocation

Fallback is automatic via provider auto-failover in BaseAgent. When the primary provider exhausts retries (or the circuit breaker opens), BaseAgent automatically attempts the fallback provider. This is infrastructure resilience, not a code workaround — the "never fall back" policy applies to masking bugs with cached/simplified results, not to LLM provider redundancy.

Auto-failover behavior:
1. Primary provider fails → BaseAgent catches `LLMProviderError`
2. Fallback provider is attempted with the same prompt and parameters
3. If fallback also fails → `LLMProviderError` propagated to orchestrator
4. All failover events are logged at WARNING level with provider names
5. When primary recovers (circuit breaker half-open), new requests route back to primary

---

## 7. Security

### Credential Management

| Rule | Implementation |
|------|----------------|
| All API keys in `.env` only | `python-dotenv` loads at startup, never hardcoded |
| No keys in logs | Logging config redacts `*_API_KEY` patterns |
| No keys in error messages | Exception constructors strip keys from details |
| No keys in WebSocket messages | Progress events contain no credentials |

### File Upload Validation

```python
# backend/app/api/upload.py

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB

async def validate_upload(file: UploadFile) -> None:
    # Check extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"File type {ext} not allowed. Use PDF or DOCX.")

    # Check size via streaming to avoid loading full content into memory for oversized files
    size = 0
    chunks = []
    while chunk := await file.read(8192):
        size += len(chunk)
        if size > MAX_UPLOAD_SIZE:
            raise HTTPException(400, f"File exceeds {MAX_UPLOAD_SIZE // (1024*1024)} MB limit.")
        chunks.append(chunk)
    content = b"".join(chunks)
    await file.seek(0)  # reset for downstream processing

    # Validate file magic bytes
    if ext == ".pdf" and not content[:4] == b"%PDF":
        raise HTTPException(400, "File does not appear to be a valid PDF.")
    if ext == ".docx" and not content[:4] == b"PK\x03\x04":
        raise HTTPException(400, "File does not appear to be a valid DOCX (ZIP/PK signature missing).")
```

### Input Sanitization

- Query text is passed directly to LLM prompts — no SQL/shell injection risk since queries only go to LLM APIs
- File paths from upload are sanitized (no directory traversal: `..` stripped)
- Contract stack IDs are UUIDs — validated by Pydantic before use
- PostgreSQL queries use parameterized queries via asyncpg (no string concatenation)

### Rate Limiting

```python
# FastAPI middleware for API rate limiting
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/contract-stacks/{stack_id}/query")
@limiter.limit("10/minute")
async def submit_query(...):
    ...
```

---

## 8. Evaluation & Feedback Framework (Phase 2)

### Confidence Calibration

Track predicted vs. actual accuracy over time:

```python
# After each user query + correction cycle:
# Store: (predicted_confidence, was_correct) tuples
# Periodically compute calibration curve
# If confidence=0.9 but accuracy=0.7 → model is overconfident → adjust prompts
```

### User Correction Feedback Loop

When users correct an answer (e.g., "the payment terms are actually Net 30, not Net 45"):
1. Store the correction as labeled training data in PostgreSQL
2. Include corrections as few-shot examples in relevant prompts
3. Track correction frequency per agent to identify systematic weaknesses

### Prompt A/B Testing

```python
# Phase 2: PromptLoader supports versioned prompts
# prompt/conflict_detection_system_v1.txt
# prompt/conflict_detection_system_v2.txt
# Orchestrator randomly assigns version per pipeline run
# Compare: conflict count, pain point detection rate, user correction rate
```

### HEARTBEAT-3 Golden Labels

The 5 pain points serve as golden evaluation labels:
- After each prompt change, re-run the HEARTBEAT-3 pipeline
- Verify all 5 pain points are still detected
- Track detection confidence over time
- Alert if any pain point drops below threshold
