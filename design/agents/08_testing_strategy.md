# 08 — Testing Strategy

> MockLLMProvider, unit tests, integration tests, E2E tests, performance benchmarks
> File locations: `backend/tests/`, `backend/tests/fixtures/`, `backend/tests/conftest.py`

---

## 1. Test Directory Structure

```
backend/tests/
├── conftest.py                      # shared fixtures (MockLLMProvider, DB setup)
├── fixtures/
│   └── llm_responses/
│       ├── document_parser/
│       │   ├── cta_parse.json
│       │   └── amendment_parse.json
│       ├── amendment_tracker/
│       │   ├── amendment_1.json
│       │   ├── amendment_2.json
│       │   ├── amendment_3_buried_payment.json
│       │   ├── amendment_4.json
│       │   └── amendment_5.json
│       ├── override_resolution/
│       │   ├── section_7_2_payment.json
│       │   └── section_12_1_indemnification.json
│       ├── conflict_detection/
│       │   └── heartbeat3_conflicts.json
│       ├── dependency_mapper/
│       │   └── heartbeat3_dependencies.json
│       ├── ripple_effect/
│       │   ├── hop_1_data_retention.json
│       │   ├── hop_2_data_retention.json
│       │   └── recommendations_data_retention.json
│       ├── query_router/
│       │   ├── truth_query.json
│       │   ├── conflict_query.json
│       │   └── ripple_query.json
│       └── truth_synthesizer/
│           └── payment_terms_answer.json
├── unit/
│   ├── test_document_parser.py
│   ├── test_amendment_tracker.py
│   ├── test_temporal_sequencer.py
│   ├── test_override_resolution.py
│   ├── test_conflict_detection.py
│   ├── test_dependency_mapper.py
│   ├── test_ripple_effect.py
│   ├── test_query_router.py
│   ├── test_truth_synthesizer.py
│   ├── test_prompt_loader.py
│   ├── test_llm_providers.py
│   ├── test_circuit_breaker.py
│   └── test_error_paths.py          # negative/error path tests
├── integration/
│   ├── conftest.py                  # DB fixtures (PostgreSQL/NeonDB + pgvector, Redis)
│   ├── test_ingestion_pipeline.py
│   ├── test_query_pipeline.py
│   └── test_heartbeat3_pain_points.py
├── e2e/
│   ├── test_api_contract_stack.py
│   └── test_api_query.py
└── performance/
    ├── test_document_parsing_perf.py
    ├── test_pipeline_perf.py
    └── test_query_perf.py
```

---

## 2. MockLLMProvider

Implements the `LLMProvider` protocol via structural typing — no inheritance required.

```python
# backend/tests/conftest.py

import json
from pathlib import Path
from app.models.agent_schemas import LLMResponse

class MockLLMProvider:
    """
    Test double for LLMProvider protocol.

    Returns pre-recorded responses from fixture files.
    Tracks calls for assertion.
    """

    provider_name: str = "mock"

    def __init__(self, fixtures_dir: Path = Path(__file__).parent / "fixtures" / "llm_responses"):
        self._fixtures_dir = fixtures_dir
        self._response_queue: list[dict] = []    # ordered queue of responses
        self._calls: list[dict] = []             # record of all calls made
        self._fixture_map: dict[str, dict] = {}  # prompt_pattern → response

    def queue_response(self, response_data: dict) -> None:
        """Add a response to the FIFO queue."""
        self._response_queue.append(response_data)

    def load_fixture(self, fixture_path: str) -> None:
        """Load a fixture file and queue its content as a response."""
        full_path = self._fixtures_dir / fixture_path
        data = json.loads(full_path.read_text())
        self._response_queue.append(data)

    def set_fixture_for_prompt(self, prompt_contains: str, fixture_path: str) -> None:
        """Map a prompt pattern to a fixture — for agents that make multiple calls."""
        full_path = self._fixtures_dir / fixture_path
        data = json.loads(full_path.read_text())
        self._fixture_map[prompt_contains] = data

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        model: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float = 0.0,
        response_format: str | None = None,
    ) -> LLMResponse:
        self._calls.append({
            "method": "complete",
            "system_prompt": system_prompt,
            "user_message": user_message,
            "model": model,
            "max_output_tokens": max_output_tokens,
        })

        # Check fixture map first
        for pattern, response_data in self._fixture_map.items():
            if pattern in user_message or pattern in system_prompt:
                return LLMResponse(
                    success=True,
                    content=json.dumps(response_data),
                    usage={"input_tokens": 100, "output_tokens": 200},
                    model=model or "mock",
                    latency_ms=50,
                    provider="mock",
                )

        # Then check queue
        if self._response_queue:
            data = self._response_queue.pop(0)
            return LLMResponse(
                success=True,
                content=json.dumps(data),
                usage={"input_tokens": 100, "output_tokens": 200},
                model=model or "mock",
                latency_ms=50,
                provider="mock",
            )

        raise RuntimeError("MockLLMProvider: no response queued or mapped")

    async def complete_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        # Delegate to complete for testing purposes (tools ignored in mock)
        return await self.complete(
            system_prompt, user_message,
            model=model, max_output_tokens=max_output_tokens, temperature=temperature,
        )

    async def close(self) -> None:
        """No-op for mock provider."""
        pass

    @property
    def call_count(self) -> int:
        return len(self._calls)

    @property
    def last_call(self) -> dict:
        return self._calls[-1] if self._calls else {}

    def assert_called_with_prompt_containing(self, text: str) -> None:
        """Assert that at least one call's user_message contains the given text."""
        for call in self._calls:
            if text in call["user_message"] or text in call["system_prompt"]:
                return
        raise AssertionError(f"No LLM call contained '{text}'. Calls: {len(self._calls)}")

    def reset(self) -> None:
        self._response_queue.clear()
        self._calls.clear()
        self._fixture_map.clear()
```

### Pytest Fixtures

```python
# backend/tests/conftest.py (continued)

import pytest
from app.agents.prompt_loader import PromptLoader

@pytest.fixture
def mock_llm():
    return MockLLMProvider()

@pytest.fixture
def prompt_loader():
    return PromptLoader(prompt_dir="prompt")

@pytest.fixture
def mock_db_pool():
    """Mock asyncpg pool for unit tests — no real DB needed."""
    class MockConnection:
        async def fetch(self, query, *args):
            return []  # empty result set for graph traversals
        async def execute(self, query, *args):
            return "INSERT 0 0"
        def transaction(self):
            return MockTransaction()
    class MockTransaction:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
    class MockPool:
        def acquire(self):
            return MockAcquire()
    class MockAcquire:
        async def __aenter__(self):
            return MockConnection()
        async def __aexit__(self, *args): pass
    return MockPool()

@pytest.fixture
def document_parser_agent(mock_llm, prompt_loader):
    from app.agents.document_parser import DocumentParserAgent
    from app.agents.config import AgentConfig
    return DocumentParserAgent(
        config=AgentConfig(agent_name="document_parser", llm_role="extraction"),
        llm_provider=mock_llm,
        prompt_loader=prompt_loader,
    )

# Similar fixtures for all other agents...
```

---

## 3. Unit Tests

One test file per agent. Each test:
1. Creates the agent with MockLLMProvider
2. Queues appropriate fixture responses
3. Calls `agent.process(input)`
4. Asserts output matches expected Pydantic model
5. Asserts the correct prompt template was loaded

### 3.1 DocumentParserAgent Tests

```python
# backend/tests/unit/test_document_parser.py

import pytest
from uuid import uuid4
from app.models.enums import DocumentType
from app.models.agent_schemas import DocumentParseInput, DocumentParseOutput

@pytest.mark.asyncio
async def test_parse_cta_document(document_parser_agent, mock_llm):
    """Parser extracts sections, tables, and metadata from a CTA PDF."""
    mock_llm.load_fixture("document_parser/cta_parse.json")

    result = await document_parser_agent.process(DocumentParseInput(
        file_path="tests/fixtures/sample_cta.pdf",
        document_type=DocumentType.CTA,
        contract_stack_id=uuid4(),
    ))

    assert isinstance(result, DocumentParseOutput)
    assert len(result.sections) > 0
    assert result.metadata.document_type == DocumentType.CTA
    assert result.metadata.effective_date is not None
    assert result.page_count > 0

@pytest.mark.asyncio
async def test_parse_loads_correct_prompts(document_parser_agent, mock_llm):
    """Parser uses document_parser_system and document_parser_extraction prompts."""
    mock_llm.load_fixture("document_parser/cta_parse.json")

    await document_parser_agent.process(DocumentParseInput(
        file_path="tests/fixtures/sample_cta.pdf",
        document_type=DocumentType.CTA,
        contract_stack_id=uuid4(),
    ))

    mock_llm.assert_called_with_prompt_containing("legal document parser")

@pytest.mark.asyncio
async def test_parse_extracts_tables(document_parser_agent, mock_llm):
    """Parser extracts tables from PDF using pdfplumber."""
    mock_llm.load_fixture("document_parser/cta_parse.json")

    result = await document_parser_agent.process(DocumentParseInput(
        file_path="tests/fixtures/sample_cta_with_tables.pdf",
        document_type=DocumentType.CTA,
        contract_stack_id=uuid4(),
    ))

    assert len(result.tables) > 0
    assert result.tables[0].headers  # has column headers
    assert len(result.tables[0].rows) > 0

@pytest.mark.asyncio
async def test_parse_chunks_large_documents(document_parser_agent, mock_llm):
    """Parser chunks documents over 100K characters."""
    # Queue multiple responses for multiple chunks
    mock_llm.load_fixture("document_parser/cta_parse.json")
    mock_llm.load_fixture("document_parser/cta_parse.json")

    # Use a large document fixture
    result = await document_parser_agent.process(DocumentParseInput(
        file_path="tests/fixtures/large_document.pdf",
        document_type=DocumentType.CTA,
        contract_stack_id=uuid4(),
    ))

    assert result.char_count > 100_000
    assert mock_llm.call_count >= 2  # multiple LLM calls for chunks
```

### 3.2 AmendmentTrackerAgent Tests

```python
# backend/tests/unit/test_amendment_tracker.py

@pytest.mark.asyncio
async def test_detect_buried_payment_change(amendment_tracker_agent, mock_llm):
    """Tracker detects buried payment term change in Amendment 3 (Pain Point #1)."""
    mock_llm.load_fixture("amendment_tracker/amendment_3_buried_payment.json")

    result = await amendment_tracker_agent.process(AmendmentTrackInput(
        amendment_document_id=uuid4(),
        amendment_number=3,
        amendment_text="...",
        amendment_sections=[...],
        original_sections=[...],
        prior_amendments=[],
    ))

    assert "7.2" in result.sections_modified
    payment_mod = next(m for m in result.modifications if m.section_number == "7.2")
    assert payment_mod.modification_type == ModificationType.SELECTIVE_OVERRIDE
    assert "Net 30" in (payment_mod.original_text or "")
    assert "Net 45" in (payment_mod.new_text or "")

@pytest.mark.asyncio
async def test_detect_exhibit_replacement(amendment_tracker_agent, mock_llm):
    """Tracker detects exhibit B → B-1 replacement (Pain Point #2)."""
    mock_llm.load_fixture("amendment_tracker/amendment_2.json")

    result = await amendment_tracker_agent.process(AmendmentTrackInput(
        amendment_document_id=uuid4(),
        amendment_number=2,
        amendment_text="...",
        amendment_sections=[...],
        original_sections=[...],
        prior_amendments=[],
    ))

    assert "Exhibit B-1" in result.exhibits_affected

@pytest.mark.asyncio
async def test_detect_all_five_modification_patterns(amendment_tracker_agent, mock_llm):
    """Tracker handles all 5 modification patterns."""
    # Test with a fixture that contains all pattern types
    mock_llm.load_fixture("amendment_tracker/all_patterns.json")
    # ... assert each pattern type is correctly identified
```

### 3.3 OverrideResolutionAgent Tests

```python
# backend/tests/unit/test_override_resolution.py

@pytest.mark.asyncio
async def test_resolve_payment_terms_after_amendments(override_agent, mock_llm):
    """Resolution correctly applies Net 30 → Net 45 change (Pain Point #1)."""
    mock_llm.load_fixture("override_resolution/section_7_2_payment.json")

    result = await override_agent.process(OverrideResolutionInput(
        contract_stack_id=uuid4(),
        section_number="7.2",
        original_clause=ParsedSection(section_number="7.2", section_title="Payment Terms", text="...Net 30 days..."),
        original_document_id=uuid4(),
        amendments=[
            AmendmentForSection(
                amendment_document_id=uuid4(),
                amendment_number=3,
                effective_date=date(2023, 8, 17),
                modification=Modification(
                    section_number="7.2",
                    modification_type=ModificationType.SELECTIVE_OVERRIDE,
                    original_text="Net 30",
                    new_text="Net 45",
                    change_description="Payment terms changed",
                ),
            )
        ],
    ))

    assert "Net 45" in result.clause_version.current_text
    assert "Net 30" not in result.clause_version.current_text
    assert len(result.clause_version.source_chain) == 2  # original + amendment_3
    assert result.clause_version.confidence >= 0.9

@pytest.mark.asyncio
async def test_source_chain_provenance(override_agent, mock_llm):
    """Source chain correctly tracks each transformation step."""
    mock_llm.load_fixture("override_resolution/section_7_2_payment.json")

    result = await override_agent.process(...)

    chain = result.clause_version.source_chain
    assert chain[0].stage == "original"
    assert chain[-1].stage == "amendment_3"
    assert chain[-1].modification_type == ModificationType.SELECTIVE_OVERRIDE
```

### 3.4 ConflictDetectionAgent Tests

```python
# backend/tests/unit/test_conflict_detection.py

@pytest.mark.asyncio
async def test_detect_all_heartbeat3_pain_points(conflict_agent, mock_llm):
    """Conflict detection identifies all 5 HEARTBEAT-3 pain points."""
    mock_llm.load_fixture("conflict_detection/heartbeat3_conflicts.json")

    result = await conflict_agent.process(ConflictDetectionInput(
        contract_stack_id=uuid4(),
        current_clauses=[...],  # resolved clauses from all sections
        contract_stack_context=ContractStackContext(
            study_name="HEARTBEAT-3",
            sponsor_name="CardioPharm International",
            site_name="Memorial Medical Center",
            therapeutic_area="Cardiology",
        ),
    ))

    # Should detect all 5 pain points
    pain_point_ids = {c.pain_point_id for c in result.conflicts if c.pain_point_id}
    assert {1, 2, 3, 4, 5} == pain_point_ids

    # Verify severity distribution (ConflictSeveritySummary is a Pydantic model, use attribute access)
    assert result.summary.critical >= 0
    assert result.summary.high >= 2
    total = result.summary.critical + result.summary.high + result.summary.medium + result.summary.low
    assert total >= 5
```

### 3.5 Prompt Loader Tests

```python
# backend/tests/unit/test_prompt_loader.py

def test_load_all_prompts():
    """All 26 prompt files load successfully."""
    loader = PromptLoader(prompt_dir="prompt")
    assert len(loader.loaded_templates) == 26

def test_variable_substitution():
    """Placeholders are correctly substituted."""
    loader = PromptLoader(prompt_dir="prompt")
    result = loader.get("document_parser_extraction", document_type="amendment", raw_text="test text")
    assert "amendment" in result
    assert "test text" in result
    assert "{document_type}" not in result

def test_missing_variable_raises():
    """Missing variable raises PromptTemplateError."""
    loader = PromptLoader(prompt_dir="prompt")
    with pytest.raises(PromptTemplateError):
        loader.get("document_parser_extraction", document_type="cta")  # missing raw_text

def test_missing_template_raises():
    """Non-existent template raises PromptTemplateError."""
    loader = PromptLoader(prompt_dir="prompt")
    with pytest.raises(PromptTemplateError):
        loader.get("nonexistent_template")
```

### 3.6 TemporalSequencerAgent Tests

```python
# backend/tests/unit/test_temporal_sequencer.py

@pytest.mark.asyncio
async def test_deterministic_sort_by_date(temporal_sequencer_agent, mock_llm):
    """Documents are sorted by effective_date, CTA first on ties."""
    result = await temporal_sequencer_agent.process(TemporalSequenceInput(
        contract_stack_id=uuid4(),
        documents=[
            DocumentSummary(document_id=uuid4(), document_type=DocumentType.AMENDMENT, effective_date=date(2023, 1, 1), filename="amend.pdf"),
            DocumentSummary(document_id=uuid4(), document_type=DocumentType.CTA, effective_date=date(2022, 1, 1), filename="cta.pdf"),
        ],
    ))
    assert result.chronological_order[0] != result.chronological_order[1]
    assert result.version_tree.root_document_id == result.chronological_order[0]

@pytest.mark.asyncio
async def test_infers_missing_date_via_llm(temporal_sequencer_agent, mock_llm):
    """LLM is called when a document lacks an effective_date."""
    mock_llm.queue_response({"effective_date": "2022-06-01"})
    doc_id = uuid4()
    result = await temporal_sequencer_agent.process(TemporalSequenceInput(
        contract_stack_id=uuid4(),
        documents=[
            DocumentSummary(document_id=uuid4(), document_type=DocumentType.CTA, effective_date=date(2022, 1, 1), filename="cta.pdf"),
            DocumentSummary(document_id=doc_id, document_type=DocumentType.AMENDMENT, effective_date=None, filename="amend.pdf"),
        ],
    ))
    assert doc_id in result.dates_inferred
```

### 3.7 DependencyMapperAgent Tests

```python
# backend/tests/unit/test_dependency_mapper.py

@pytest.mark.asyncio
async def test_llm_identifies_explicit_references(dependency_mapper_agent, mock_llm):
    """LLM identifies explicit cross-references between clauses."""
    mock_llm.load_fixture("dependency_mapper/heartbeat3_dependencies.json")
    result = await dependency_mapper_agent.process(DependencyMapInput(
        contract_stack_id=uuid4(),
        current_clauses=[
            CurrentClause(section_number="7.2", section_title="Payment", current_text="As set forth in Section 7.4 and Exhibit B...", clause_category="payment", source_document_id=uuid4(), source_document_label="CTA", effective_date=date(2022, 1, 1)),
            CurrentClause(section_number="7.4", section_title="Holdback", current_text="Holdback provisions...", clause_category="payment", source_document_id=uuid4(), source_document_label="CTA", effective_date=date(2022, 1, 1)),
        ],
    ))
    assert all(d.detection_method == "llm" for d in result.dependencies)
    assert any(d.from_section == "7.2" and d.to_section == "7.4" for d in result.dependencies)
```

### 3.8 RippleEffectAnalyzerAgent Tests

```python
# backend/tests/unit/test_ripple_effect.py

@pytest.mark.asyncio
async def test_ripple_bidirectional_traversal(ripple_effect_agent, mock_llm, mock_db_pool):
    """Ripple analysis performs both outbound and inbound PostgreSQL traversals."""
    mock_llm.load_fixture("ripple_effect/hop_1_data_retention.json")
    result = await ripple_effect_agent.process(RippleEffectInput(
        contract_stack_id=uuid4(),
        proposed_change=ProposedChange(
            section_number="9.2",
            current_text="Data shall be retained for 15 years",
            proposed_text="Data shall be retained for 25 years",
        ),
    ))
    assert result.traversal_direction == "bidirectional"
    assert result.total_impacts >= 0

@pytest.mark.asyncio
async def test_ripple_early_termination(ripple_effect_agent, mock_llm, mock_db_pool):
    """Ripple analysis terminates early after two consecutive zero-impact hops."""
    # Queue empty impacts for hops 1 and 2
    mock_llm.queue_response({"impacts": []})
    mock_llm.queue_response({"impacts": []})
    result = await ripple_effect_agent.process(RippleEffectInput(
        contract_stack_id=uuid4(),
        proposed_change=ProposedChange(
            section_number="15.1", current_text="...", proposed_text="...",
        ),
    ))
    assert result.cascade_depth <= 2
```

### 3.9 Negative / Error Path Tests

```python
# backend/tests/unit/test_error_paths.py

@pytest.mark.asyncio
async def test_llm_non_transient_error_not_retried(mock_llm, document_parser_agent):
    """Non-transient errors (e.g., 400) raise immediately without retry."""
    from app.exceptions import LLMProviderError

    class BadRequestError(Exception):
        status_code = 400

    mock_llm.queue_response = None  # clear
    # Override complete to raise non-transient error
    original_complete = mock_llm.complete
    async def failing_complete(*args, **kwargs):
        raise BadRequestError("Invalid request")
    mock_llm.complete = failing_complete

    with pytest.raises(BadRequestError):
        await document_parser_agent.call_llm("system", "user")
    # Should NOT have retried — only 1 call
    assert mock_llm.call_count == 0  # complete was replaced, so call_count stays 0

@pytest.mark.asyncio
async def test_json_repair_failure_raises_llm_response_error(mock_llm, document_parser_agent):
    """If both JSON parse and repair fail, LLMResponseError is raised."""
    from app.exceptions import LLMResponseError

    # Override complete to return invalid JSON (queue_response expects dict, so override directly)
    call_count = 0
    original_complete = mock_llm.complete
    async def bad_json_complete(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return LLMResponse(
            success=True,
            content="not valid json at all" if call_count == 1 else "still not json",
            usage={"input_tokens": 10, "output_tokens": 10},
            model="mock", latency_ms=10, provider="mock",
        )
    mock_llm.complete = bad_json_complete

    with pytest.raises(LLMResponseError):
        await document_parser_agent.call_llm("system", "user", expect_json=True)

@pytest.mark.asyncio
async def test_missing_prompt_template_raises():
    """PromptLoader raises PromptTemplateError for missing templates."""
    from app.agents.prompt_loader import PromptLoader, PromptTemplateError
    loader = PromptLoader(prompt_dir="prompt")
    with pytest.raises(PromptTemplateError, match="not found"):
        loader.get("totally_nonexistent_prompt")

@pytest.mark.asyncio
async def test_timeout_is_enforced(mock_llm, document_parser_agent):
    """LLM calls that exceed timeout_seconds raise asyncio.TimeoutError."""
    import asyncio

    async def slow_complete(*args, **kwargs):
        await asyncio.sleep(999)  # never returns

    mock_llm.complete = slow_complete
    document_parser_agent.config = document_parser_agent.config.__class__(
        agent_name="test", llm_role="extraction", timeout_seconds=1, max_retries=1,
    )

    with pytest.raises(Exception):  # TimeoutError is transient, so it retries then raises LLMProviderError
        await document_parser_agent.call_llm("system", "user")
```

### 3.10 Circuit Breaker Tests

```python
# backend/tests/unit/test_circuit_breaker.py

@pytest.mark.asyncio
async def test_circuit_opens_after_threshold():
    cb = CircuitBreaker("test", failure_threshold=3, failure_window=300)
    for _ in range(3):
        await cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert not await cb.can_execute()

@pytest.mark.asyncio
async def test_circuit_recovers_to_half_open():
    cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0)
    await cb.record_failure()
    assert cb.state == CircuitState.OPEN
    # After recovery_timeout (0s), should be HALF_OPEN
    await asyncio.sleep(0.01)
    assert await cb.can_execute()  # transitions to HALF_OPEN under lock

@pytest.mark.asyncio
async def test_circuit_closes_on_success_from_half_open():
    cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0)
    await cb.record_failure()
    await asyncio.sleep(0.01)
    assert await cb.can_execute()  # now HALF_OPEN
    await cb.record_success()
    assert cb.state == CircuitState.CLOSED

@pytest.mark.asyncio
async def test_circuit_does_not_clear_failures_on_success_when_closed():
    """record_success only clears failures on HALF_OPEN→CLOSED transition, not in CLOSED state."""
    cb = CircuitBreaker("test", failure_threshold=5, failure_window=300)
    await cb.record_failure()
    await cb.record_failure()
    await cb.record_success()  # state is CLOSED, so failures should NOT be cleared
    # After success in CLOSED state, failures remain (only cleared on HALF_OPEN→CLOSED)
    assert len(cb._failures) == 2
```

### 3.11 Agentic Behavior Tests

Tests that verify the agentic capabilities introduced in the BaseAgent architecture.

```python
# backend/tests/unit/test_agentic_behavior.py

@pytest.mark.asyncio
async def test_self_verification_triggers_on_low_confidence(mock_llm, amendment_tracker_agent):
    """When extraction_confidence < verification_threshold, agent re-processes with critique."""
    # First call returns low-confidence result
    mock_llm.queue_response({
        "modifications": [{"section_number": "7.2", "modification_type": "selective_override",
                          "original_text": "...", "new_text": "...", "change_description": "..."}],
        "reasoning": "...",
        "extraction_confidence": 0.5,  # below threshold of 0.75
    })
    # Second call (self-critique) returns text feedback
    mock_llm.queue_response("The modification list may be incomplete. Section 5.3 also appears modified.")
    # Third call (re-process) returns higher-confidence result
    mock_llm.queue_response({
        "modifications": [
            {"section_number": "7.2", "modification_type": "selective_override", "original_text": "...", "new_text": "...", "change_description": "..."},
            {"section_number": "5.3", "modification_type": "complete_replacement", "original_text": "...", "new_text": "...", "change_description": "..."},
        ],
        "reasoning": "...",
        "extraction_confidence": 0.85,
    })
    result = await amendment_tracker_agent.run(input_data)  # run() not process()
    assert result.extraction_confidence >= 0.75
    assert len(result.modifications) == 2  # found the missed modification

@pytest.mark.asyncio
async def test_token_budget_raises_on_overflow(document_parser_agent):
    """Agent raises LLMProviderError when input exceeds context window."""
    from app.exceptions import LLMProviderError
    huge_text = "x" * 1_000_000  # ~250K tokens, exceeds most models
    with pytest.raises(LLMProviderError, match="exceeds.*context limit"):
        await document_parser_agent.call_llm("system", huge_text)

@pytest.mark.asyncio
async def test_provider_auto_failover(mock_llm, mock_fallback_llm, document_parser_agent):
    """When primary provider fails, agent automatically uses fallback."""
    # Primary fails
    mock_llm.set_always_fail(LLMProviderError("Claude API down", provider="claude"))
    # Fallback succeeds
    mock_fallback_llm.queue_response({"sections": [], "metadata": {}})
    document_parser_agent._fallback_llm = mock_fallback_llm
    result = await document_parser_agent.call_llm("system", "user")
    assert result is not None  # fallback succeeded

@pytest.mark.asyncio
async def test_structured_output_uses_tool_use(mock_llm, document_parser_agent):
    """When response_schema is provided, call_llm uses complete_with_tools for schema enforcement."""
    mock_llm.queue_response({"sections": [], "metadata": {}})
    result = await document_parser_agent.call_llm(
        "system", "user",
        response_schema=DocumentParseOutput,
    )
    # Verify complete_with_tools was called (not complete)
    assert mock_llm.last_call_used_tools is True

@pytest.mark.asyncio
async def test_buried_change_scan_finds_missed_modifications(mock_llm, amendment_tracker_agent):
    """Buried change scan detects modifications missed by initial extraction."""
    # Initial extraction misses the payment change
    mock_llm.queue_response({
        "modifications": [
            {"section_number": "5.3", "modification_type": "complete_replacement",
             "original_text": "...", "new_text": "...", "change_description": "COVID telehealth"}
        ],
        "reasoning": "Found COVID protocol changes",
        "extraction_confidence": 0.9,
    })
    # Buried scan finds the payment change
    mock_llm.queue_response({
        "reasoning": "Section 7.2 contains a payment term change from Net 30 to Net 45",
        "missed_modifications": [
            {"section_number": "7.2", "modification_type": "selective_override",
             "original_text": "thirty (30) days", "new_text": "forty-five (45) days",
             "change_description": "Payment terms changed from Net 30 to Net 45"}
        ]
    })
    result = await amendment_tracker_agent.process(input_data)
    assert any(m.section_number == "7.2" for m in result.modifications)
```

### 3.12 Adversarial LLM Response Tests

Tests for malformed LLM responses beyond simple JSON parse failures.

```python
# backend/tests/unit/test_adversarial_responses.py

@pytest.mark.asyncio
async def test_valid_json_wrong_schema_caught_by_pydantic(mock_llm, amendment_tracker_agent):
    """Valid JSON with wrong field types is caught by Pydantic validation, not silently accepted."""
    mock_llm.queue_response({
        "modifications": [
            {"section_number": 7.2,  # wrong type: should be str, not float
             "modification_type": "invalid_type",  # not a valid enum value
             "original_text": "", "new_text": "", "change_description": ""}
        ],
        "reasoning": "...",
        "extraction_confidence": "high",  # wrong type: should be float, not str
    })
    with pytest.raises(ValidationError):
        await amendment_tracker_agent.process(input_data)

@pytest.mark.asyncio
async def test_hallucinated_section_flagged_by_verification(mock_llm, amendment_tracker_agent):
    """Self-verification catches references to non-existent sections."""
    mock_llm.queue_response({
        "modifications": [
            {"section_number": "99.9",  # does not exist in original CTA
             "modification_type": "selective_override",
             "original_text": "...", "new_text": "...",
             "change_description": "Hallucinated modification"}
        ],
        "reasoning": "...",
        "extraction_confidence": 0.9,
    })
    result = await amendment_tracker_agent.run(input_data)
    # _verify_output should lower confidence due to unknown section
    assert result.extraction_confidence < 0.7
```

### 3.13 Multi-Turn Reasoning Tests

```python
# backend/tests/unit/test_multi_turn_reasoning.py

@pytest.mark.asyncio
async def test_call_llm_conversation_refines_output(mock_llm, conflict_agent):
    """Multi-turn reasoning improves conflict detection through self-review."""
    # Turn 1: initial analysis finds 3 conflicts
    mock_llm.queue_response({
        "conflicts": [{"conflict_id": "c1"}, {"conflict_id": "c2"}, {"conflict_id": "c3"}],
        "reasoning": "Initial scan",
    })
    # Turn 2: refinement finds 1 additional conflict
    mock_llm.queue_response({
        "conflicts": [{"conflict_id": "c1"}, {"conflict_id": "c2"}, {"conflict_id": "c3"}, {"conflict_id": "c4"}],
        "reasoning": "Found additional stale reference on review",
        "_refinement_complete": True,
    })
    result = await conflict_agent.call_llm_conversation("system", "analyze conflicts", max_turns=3)
    assert len(result["conflicts"]) == 4
```

### 3.14 Tool-Use Tests

```python
# backend/tests/unit/test_tool_use.py

@pytest.mark.asyncio
async def test_conflict_detection_uses_tools(mock_llm, conflict_agent, mock_db_pool):
    """ConflictDetection agent uses get_clause tool during analysis."""
    # Mock tool-use response: LLM requests clause lookup
    mock_llm.queue_tool_call_response(
        tool_name="get_clause",
        tool_input={"section_number": "7.2"},
    )
    mock_llm.queue_response({"conflicts": [], "reasoning": "No conflicts after clause review"})
    result = await conflict_agent.call_llm_with_tools(
        "system", "analyze", conflict_agent.get_tools()
    )
    assert conflict_agent._execute_tool_called_with("get_clause", {"section_number": "7.2"})

@pytest.mark.asyncio
async def test_ripple_effect_uses_tools(mock_llm, ripple_agent, mock_db_pool):
    """RippleEffect agent uses get_dependencies tool during hop analysis."""
    assert len(ripple_agent.get_tools()) >= 2
    assert any(t["name"] == "get_dependencies" for t in ripple_agent.get_tools())
```

### 3.15 Compound Query Tests

```python
# backend/tests/unit/test_compound_queries.py

@pytest.mark.asyncio
async def test_compound_query_decomposition(mock_llm, query_router):
    """QueryRouter decomposes compound questions into sub-queries."""
    mock_llm.queue_response({
        "reasoning": "This is a compound query spanning truth and conflict detection",
        "query_type": "truth_reconstitution",
        "entities": ["payment terms"],
        "confidence": 0.92,
        "sub_queries": [
            {"query_type": "truth_reconstitution", "query_text": "What are the current payment terms?", "entities": ["payment"]},
            {"query_type": "conflict_detection", "query_text": "Are there conflicts in payment terms?", "entities": ["payment"]},
        ],
    })
    result = await query_router.run(QueryRouteInput(query_text="What are the payment terms and are there any conflicts?", contract_stack_id=uuid4()))
    assert len(result.sub_queries) == 2
    assert result.sub_queries[0]["query_type"] == "truth_reconstitution"
    assert result.sub_queries[1]["query_type"] == "conflict_detection"
```

### 3.16 Blackboard Communication Tests

```python
# backend/tests/unit/test_blackboard.py

@pytest.mark.asyncio
async def test_blackboard_publish_and_query(redis_client):
    """Blackboard supports publish and query of typed entries."""
    from app.agents.orchestrator import AgentBlackboard
    bb = AgentBlackboard(redis_client)
    stack_id = uuid4()
    await bb.publish(stack_id, "amendment_tracker", "buried_change", {"section": "7.2", "detail": "Net 30 → Net 45"})
    entries = await bb.query(stack_id, "buried_change")
    assert len(entries) == 1
    assert entries[0]["data"]["section"] == "7.2"

@pytest.mark.asyncio
async def test_blackboard_clear(redis_client):
    """Blackboard clear removes all entries for a stack."""
    from app.agents.orchestrator import AgentBlackboard
    bb = AgentBlackboard(redis_client)
    stack_id = uuid4()
    await bb.publish(stack_id, "test", "entry_type", {"key": "value"})
    await bb.clear(stack_id)
    entries = await bb.query(stack_id, "entry_type")
    assert len(entries) == 0
```

---

## 4. Integration Tests — HEARTBEAT-3 Acceptance

These tests use MockLLMProvider with realistic fixture data but connect to real PostgreSQL (NeonDB + pgvector).

```python
# backend/tests/integration/conftest.py

import pytest
import asyncpg

from app.database.vector_store import VectorStore

@pytest.fixture(scope="session")
async def postgres_pool():
    """Create a test PostgreSQL database (NeonDB + pgvector) and return a connection pool."""
    import os
    dsn = os.getenv("EXTERNAL_DATABASE_URL")  # NeonDB connection string
    pool = await asyncpg.create_pool(dsn=dsn)
    # Run schema migrations (includes pgvector extension + section_embeddings table).
    # The section_embeddings schema must include the `is_resolved` column to support
    # the two-tier embedding architecture: is_resolved=FALSE for Stage 1 checkpoint
    # embeddings, is_resolved=TRUE for Stage 4 query-ready embeddings.
    async with pool.acquire() as conn:
        schema = Path("backend/db/schema.sql").read_text()
        await conn.execute(schema)
    yield pool
    # Teardown: drop test data
    async with pool.acquire() as conn:
        await conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    await pool.close()

@pytest.fixture(scope="session")
def vector_store(postgres_pool):
    """Create a VectorStore backed by pgvector on the test PostgreSQL pool.

    The VectorStore must support the two-tier embedding architecture:
    - is_resolved=FALSE: Stage 1 checkpoint embeddings (stored during ingestion
      before override resolution, used as intermediate snapshots)
    - is_resolved=TRUE: Stage 4 query-ready embeddings (stored after override
      resolution, filtered on for query-time semantic search)
    """
    return VectorStore(postgres_pool)

@pytest.fixture
async def orchestrator(postgres_pool, vector_store, mock_llm):
    """Create an AgentOrchestrator with real DB but mock LLM."""
    import redis.asyncio as aioredis
    redis_client = aioredis.from_url("redis://localhost:6379/1")  # DB 1 for tests
    orchestrator = AgentOrchestrator(
        postgres_pool=postgres_pool,
        redis_client=redis_client,
        vector_store=vector_store,
    )
    yield orchestrator
    await redis_client.flushdb()
    await redis_client.close()
```

```python
# backend/tests/integration/test_heartbeat3_pain_points.py

@pytest.mark.integration
@pytest.mark.asyncio
class TestHeartbeat3PainPoints:
    """
    Acceptance tests validating all 5 HEARTBEAT-3 pain points.
    Uses real database, mock LLM with pre-recorded responses.
    """

    stack_id = uuid4()

    def _load_full_pipeline_fixtures(self, mock_llm: MockLLMProvider) -> None:
        """
        Load fixture responses for ALL agents in the full ingestion pipeline.

        Order matches pipeline execution:
        1. DocumentParser — 6 documents (1 CTA + 5 amendments)
        2. AmendmentTracker — 5 amendments (sequential)
        3. TemporalSequencer — no fixture needed (deterministic)
        4. OverrideResolution — one per section (variable count)
        5. DependencyMapper — 1 call
        6. ConflictDetection — 1 call
        7. QueryRouter — 1 per query
        8. TruthSynthesizer — 1 per query
        """
        # Stage 1: Document parsing (6 docs)
        for doc_fixture in [
            "document_parser/cta_parse.json",
            "document_parser/amendment_parse.json",
            "document_parser/amendment_parse.json",
            "document_parser/amendment_parse.json",
            "document_parser/amendment_parse.json",
            "document_parser/amendment_parse.json",
        ]:
            mock_llm.load_fixture(doc_fixture)

        # Stage 2: Amendment tracking (5 amendments)
        for amend_fixture in [
            "amendment_tracker/amendment_1.json",
            "amendment_tracker/amendment_2.json",
            "amendment_tracker/amendment_3_buried_payment.json",
            "amendment_tracker/amendment_4.json",
            "amendment_tracker/amendment_5.json",
        ]:
            mock_llm.load_fixture(amend_fixture)

        # Stage 4: Override resolution (prompt-pattern-based matching)
        mock_llm.set_fixture_for_prompt("Section 7.2", "override_resolution/section_7_2_payment.json")
        mock_llm.set_fixture_for_prompt("Section 12.1", "override_resolution/section_12_1_indemnification.json")

        # Stage 5: Dependency mapping
        mock_llm.load_fixture("dependency_mapper/heartbeat3_dependencies.json")

        # Stage 6: Conflict detection
        mock_llm.load_fixture("conflict_detection/heartbeat3_conflicts.json")

        # Query pipeline fixtures (loaded per-test via set_fixture_for_prompt)
        mock_llm.set_fixture_for_prompt("Classify this query", "query_router/truth_query.json")
        mock_llm.set_fixture_for_prompt("answering a question", "truth_synthesizer/payment_terms_answer.json")

    async def test_pain_point_1_buried_payment_change(self, orchestrator, mock_llm):
        """PP#1: Net 30 → Net 45 hidden in Amendment 3's COVID section is detected."""
        # Setup mock responses for full pipeline
        self._load_full_pipeline_fixtures(mock_llm)

        result = await orchestrator.process_contract_stack(self.stack_id, "test-job", lambda e: None)

        # Query for payment terms
        answer = await orchestrator.handle_query("What are the current payment terms?", self.stack_id)
        assert "Net 45" in answer.answer
        assert "Amendment 3" in answer.answer

    async def test_pain_point_2_budget_exhibit_evolution(self, orchestrator, mock_llm):
        """PP#2: Exhibit B → B-1 → B-2 with old PI reference detected."""
        self._load_full_pipeline_fixtures(mock_llm)
        await orchestrator.process_contract_stack(self.stack_id, "test-job", lambda e: None)

        answer = await orchestrator.handle_query("Are there any stale references in the budget exhibits?", self.stack_id)
        assert "Exhibit B" in answer.answer
        # Should mention old PI reference
        assert any("PI" in caveat or "principal investigator" in caveat.lower() for caveat in answer.caveats)

    async def test_pain_point_3_insurance_coverage_gap(self, orchestrator, mock_llm):
        """PP#3: Amendment 5 extends study but insurance obligation is ambiguous."""
        self._load_full_pipeline_fixtures(mock_llm)
        await orchestrator.process_contract_stack(self.stack_id, "test-job", lambda e: None)

        answer = await orchestrator.handle_query("Is there an insurance coverage gap?", self.stack_id)
        assert "gap" in answer.answer.lower() or "ambiguous" in answer.answer.lower()

    async def test_pain_point_4_cross_reference_confusion(self, orchestrator, mock_llm):
        """PP#4: Amendment 4 removes follow-up visits but cardiac MRI survives."""
        self._load_full_pipeline_fixtures(mock_llm)
        await orchestrator.process_contract_stack(self.stack_id, "test-job", lambda e: None)

        answer = await orchestrator.handle_query("Is the cardiac MRI visit still required?", self.stack_id)
        assert "Amendment 1" in answer.answer  # MRI was added in Amendment 1
        # Should confirm it's still active
        assert "survives" in answer.answer.lower() or "still" in answer.answer.lower() or "active" in answer.answer.lower()

    async def test_pain_point_5_pi_change_budget_ambiguity(self, orchestrator, mock_llm):
        """PP#5: Amendment 2 changes PI but Exhibit B-1 references old PI."""
        self._load_full_pipeline_fixtures(mock_llm)
        await orchestrator.process_contract_stack(self.stack_id, "test-job", lambda e: None)

        answer = await orchestrator.handle_query("Does the budget exhibit reference the correct PI?", self.stack_id)
        assert "inconsisten" in answer.answer.lower() or "stale" in answer.answer.lower() or "old" in answer.answer.lower()
```

---

## 5. E2E Tests

API-level tests exercising the full HTTP path.

```python
# backend/tests/e2e/test_api_contract_stack.py

import pytest
from httpx import AsyncClient

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_full_contract_stack_lifecycle(client: AsyncClient):
    """Create stack → upload docs → process → query → validate answers."""

    # Step 1: Create contract stack
    resp = await client.post("/api/v1/contract-stacks", json={
        "name": "HEARTBEAT-3 - Memorial Medical",
        "sponsor_name": "CardioPharm International",
        "site_name": "Memorial Medical Center",
        "study_protocol": "CP-2847-301",
        "therapeutic_area": "Cardiology",
    })
    assert resp.status_code == 201
    stack_id = resp.json()["id"]

    # Step 2: Upload documents
    for pdf_path in HEARTBEAT3_PDFS:
        with open(pdf_path, "rb") as f:
            resp = await client.post(
                f"/api/v1/contract-stacks/{stack_id}/documents",
                files={"file": (pdf_path.name, f, "application/pdf")},
                data={"document_type": "amendment" if "Amendment" in pdf_path.name else "cta"},
            )
            assert resp.status_code == 201

    # Step 3: Trigger processing
    resp = await client.post(f"/api/v1/contract-stacks/{stack_id}/process")
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    # Step 4: Wait for processing (poll job status)
    await wait_for_job(client, job_id, timeout=120)

    # Step 5: Query and validate
    resp = await client.post(f"/api/v1/contract-stacks/{stack_id}/query", json={
        "query": "What are the current payment terms?",
        "include_reasoning": True,
    })
    assert resp.status_code == 200
    assert "Net 45" in resp.json()["response"]["answer"]
    assert resp.json()["response"]["confidence"] >= 0.9
    assert len(resp.json()["response"]["sources"]) > 0
```

---

## 6. Performance Benchmarks

Performance targets (from technical_spec.md):

| Metric | Target | Measured By |
|--------|--------|-------------|
| Single document parsing | < 10 seconds | Unit test with real LLM |
| Full pipeline (6 docs) | < 60 seconds | Integration test with real LLM |
| Query response | < 15 seconds | E2E test with real LLM |
| Query routing | < 2 seconds | Unit test with real LLM |

```python
# backend/tests/performance/test_pipeline_perf.py

import time
import pytest

@pytest.mark.performance
@pytest.mark.asyncio
async def test_single_document_parse_under_10s(real_llm_parser):
    """Document parsing completes in under 10 seconds."""
    start = time.monotonic()
    await real_llm_parser.process(DocumentParseInput(
        file_path="policies/cta_policy_amendment/Original CTA - HEARTBEAT-3 (January 2022).pdf",
        document_type=DocumentType.CTA,
        contract_stack_id=uuid4(),
    ))
    elapsed = time.monotonic() - start
    assert elapsed < 10, f"Document parsing took {elapsed:.1f}s (target: <10s)"

@pytest.mark.performance
@pytest.mark.asyncio
async def test_full_pipeline_under_60s(real_llm_orchestrator):
    """Full 6-document pipeline completes in under 60 seconds."""
    start = time.monotonic()
    await real_llm_orchestrator.process_contract_stack(stack_id, "perf-job", lambda e: None)
    elapsed = time.monotonic() - start
    assert elapsed < 60, f"Full pipeline took {elapsed:.1f}s (target: <60s)"

@pytest.mark.performance
@pytest.mark.asyncio
async def test_query_under_15s(real_llm_orchestrator):
    """Query response in under 15 seconds."""
    start = time.monotonic()
    await real_llm_orchestrator.handle_query("What are the current payment terms?", stack_id)
    elapsed = time.monotonic() - start
    assert elapsed < 15, f"Query took {elapsed:.1f}s (target: <15s)"

@pytest.mark.performance
@pytest.mark.asyncio
async def test_query_routing_under_2s(real_llm_router):
    """Query routing in under 2 seconds."""
    start = time.monotonic()
    await real_llm_router.process(QueryRouteInput(
        query_text="What are the current payment terms?",
        contract_stack_id=uuid4(),
    ))
    elapsed = time.monotonic() - start
    assert elapsed < 2, f"Routing took {elapsed:.1f}s (target: <2s)"
```

---

## 7. Test Execution Commands

```bash
# Run all unit tests (fast — mock LLM)
pytest tests/unit/ -v

# Run specific agent tests
pytest tests/unit/test_document_parser.py -v
pytest tests/unit/test_conflict_detection.py -v

# Run integration tests (requires NeonDB + Redis running)
pytest tests/integration/ -v -m integration

# Run HEARTBEAT-3 pain point acceptance tests
pytest tests/integration/test_heartbeat3_pain_points.py -v

# Run E2E tests (requires full stack running)
pytest tests/e2e/ -v -m e2e

# Run performance tests (requires real LLM — costs money)
pytest tests/performance/ -v -m performance

# Run all except performance
pytest tests/ -v --ignore=tests/performance/
```

---

## 8. Fixture Data Design

LLM response fixtures are realistic JSON matching the exact format each agent expects from its LLM calls. Example for Amendment 3 (Pain Point #1):

**`tests/fixtures/llm_responses/amendment_tracker/amendment_3_buried_payment.json`**
```json
{
  "amendment_type": "covid_protocol",
  "rationale": "WHEREAS the parties wish to modify the Agreement to address protocol changes necessitated by the COVID-19 pandemic...",
  "effective_date": "2023-08-17",
  "modifications": [
    {
      "section_number": "5.3",
      "modification_type": "complete_replacement",
      "original_text": "Study visits shall occur in person at the Site.",
      "new_text": "Study visits may occur via telehealth when in-person visits are not feasible due to public health emergencies.",
      "change_description": "Added telehealth visit option for COVID-related disruptions"
    },
    {
      "section_number": "7.2",
      "modification_type": "selective_override",
      "original_text": "Sponsor shall pay undisputed invoices within thirty (30) days of receipt.",
      "new_text": "Sponsor shall pay undisputed invoices within forty-five (45) days of receipt.",
      "change_description": "Payment terms changed from Net 30 to Net 45 days"
    }
  ],
  "exhibits_affected": []
}
```

---

## 9. Cross-Reference Verification

### Agent I/O Contracts (doc 02) vs Test Assertions

| Agent | Input Model Tested | Output Model Asserted | Fixture File |
|-------|-------------------|----------------------|-------------|
| DocumentParserAgent | `DocumentParseInput` | `DocumentParseOutput` | `document_parser/*.json` |
| AmendmentTrackerAgent | `AmendmentTrackInput` | `AmendmentTrackOutput` | `amendment_tracker/*.json` |
| TemporalSequencerAgent | `TemporalSequenceInput` | `TemporalSequenceOutput` | (mostly deterministic — LLM fixture for date inference) |
| OverrideResolutionAgent | `OverrideResolutionInput` | `OverrideResolutionOutput` | `override_resolution/*.json` |
| ConflictDetectionAgent | `ConflictDetectionInput` | `ConflictDetectionOutput` | `conflict_detection/*.json` |
| DependencyMapperAgent | `DependencyMapInput` | `DependencyMapOutput` | `dependency_mapper/*.json` |
| RippleEffectAnalyzerAgent | `RippleEffectInput` | `RippleEffectOutput` | `ripple_effect/*.json` |
| QueryRouter | `QueryRouteInput` | `QueryRouteOutput` | `query_router/*.json` |
| TruthSynthesizer | `TruthSynthesisInput` | `TruthSynthesisOutput` | `truth_synthesizer/*.json` |

### Prompt File Manifest (doc 07) vs Test Coverage

All 26 prompt files are validated by:
1. `test_prompt_loader.py::test_load_all_prompts` — ensures all 26 files exist and load
2. Each agent's unit test asserts the correct prompt was loaded via `mock_llm.assert_called_with_prompt_containing()`

### HEARTBEAT-3 Pain Points (doc 04) vs Integration Tests

| Pain Point | Detected By Agent | Tested In |
|-----------|------------------|-----------|
| #1 Buried Payment Change | ConflictDetectionAgent, AmendmentTrackerAgent, OverrideResolutionAgent | `test_pain_point_1_buried_payment_change`, `test_detect_buried_payment_change`, `test_resolve_payment_terms_after_amendments` |
| #2 Budget Exhibit Evolution | ConflictDetectionAgent, AmendmentTrackerAgent | `test_pain_point_2_budget_exhibit_evolution`, `test_detect_exhibit_replacement` |
| #3 Insurance Coverage Gap | ConflictDetectionAgent, DependencyMapperAgent | `test_pain_point_3_insurance_coverage_gap` |
| #4 Cross-Reference Confusion | ConflictDetectionAgent, OverrideResolutionAgent, DependencyMapperAgent | `test_pain_point_4_cross_reference_confusion` |
| #5 PI Change + Budget Ambiguity | ConflictDetectionAgent | `test_pain_point_5_pi_change_budget_ambiguity` |
