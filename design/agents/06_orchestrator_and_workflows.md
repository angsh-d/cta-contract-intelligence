# 06 — Orchestrator & Workflows

> AgentOrchestrator, ingestion pipeline, query pipeline, WebSocket protocol, caching
> File locations: `backend/app/agents/orchestrator.py`, `backend/app/agents/query_router.py`, `backend/app/agents/truth_synthesizer.py`, `backend/app/tasks.py`

---

## 1. AgentOrchestrator

The orchestrator is the central coordination layer with LLM-driven planning capabilities. It owns all agent instances, manages the ingestion and query pipelines, supports iterative agent communication via a shared blackboard, and wires progress callbacks to WebSocket.

### Agentic Orchestration Principles

The orchestrator goes beyond a fixed pipeline:
1. **LLM-Driven Planning** — Before execution, the orchestrator assesses the document set and produces an execution plan (which stages to run, resource allocation, expected complexity).
2. **Agent Communication Protocol** — Agents can signal `needs_clarification` in their output, requesting re-analysis by a specific upstream agent. The orchestrator handles these iterative loops.
3. **Shared Blackboard** — A Redis-backed shared state where agents publish findings and other agents can subscribe to updates. Enables emergent reasoning across agents.
4. **Adaptive Execution** — The orchestrator can skip unnecessary stages (e.g., no amendment tracking for a single CTA), parallelize independent work, and re-run low-confidence stages.

### Singleton via FastAPI Lifespan

```python
# backend/app/agents/orchestrator.py

class AgentOrchestrator:
    """
    Central coordinator for all ContractIQ agents.
    Created once at application startup via FastAPI lifespan.
    Injected into route handlers via app.state.
    """

    def __init__(
        self,
        postgres_pool,          # asyncpg pool (NeonDB)
        redis_client,           # redis.asyncio.Redis
        vector_store,           # VectorStore (pgvector on NeonDB)
    ) -> None:
        self.postgres = postgres_pool
        self.redis = redis_client
        self.vector_store = vector_store
        self.blackboard = AgentBlackboard(redis_client)

        # Shared resources
        self._llm_semaphore = asyncio.Semaphore(5)  # max 5 concurrent LLM calls across all pipelines
        prompt_loader = PromptLoader(prompt_dir="prompt")

        # Agent registry — Dict[str, BaseAgent]
        self.agents: dict[str, BaseAgent] = {}
        self._init_agents(prompt_loader)

    def _init_agents(self, prompt_loader: PromptLoader) -> None:
        factory = LLMProviderFactory

        # TraceContext is created per pipeline run (see process_contract_stack),
        # but shared resources (fallback_provider, semaphore) are set at init time.
        # Each agent receives: primary provider, fallback provider, and shared semaphore.

        # Tier 1
        self.agents["document_parser"] = DocumentParserAgent(
            config=AgentConfig(agent_name="document_parser", llm_role="extraction", model_override="claude-sonnet-4-5-20250929", max_output_tokens=8192, temperature=0.0, verification_threshold=0.80),
            llm_provider=factory.get_for_role("extraction"),
            prompt_loader=prompt_loader,
            fallback_provider=factory.get_fallback_for_role("extraction"),
            llm_semaphore=self._llm_semaphore,
        )
        self.agents["amendment_tracker"] = AmendmentTrackerAgent(
            config=AgentConfig(agent_name="amendment_tracker", llm_role="complex_reasoning", model_override="claude-opus-4-5-20250514", max_output_tokens=8192, timeout_seconds=180, temperature=0.0, verification_threshold=0.75),
            llm_provider=factory.get_for_role("complex_reasoning"),
            prompt_loader=prompt_loader,
            fallback_provider=factory.get_fallback_for_role("complex_reasoning"),
            llm_semaphore=self._llm_semaphore,
        )
        self.agents["temporal_sequencer"] = TemporalSequencerAgent(
            config=AgentConfig(agent_name="temporal_sequencer", llm_role="extraction", model_override="claude-sonnet-4-5-20250929", max_output_tokens=4096, timeout_seconds=60, temperature=0.0, verification_threshold=0.80),
            llm_provider=factory.get_for_role("extraction"),
            prompt_loader=prompt_loader,
            db_pool=self.postgres,
            fallback_provider=factory.get_fallback_for_role("extraction"),
            llm_semaphore=self._llm_semaphore,
        )

        # Tier 2
        self.agents["override_resolution"] = OverrideResolutionAgent(
            config=AgentConfig(agent_name="override_resolution", llm_role="complex_reasoning", model_override="claude-opus-4-5-20250514", max_output_tokens=8192, timeout_seconds=180, temperature=0.0, verification_threshold=0.75),
            llm_provider=factory.get_for_role("complex_reasoning"),
            prompt_loader=prompt_loader,
            fallback_provider=factory.get_fallback_for_role("complex_reasoning"),
            llm_semaphore=self._llm_semaphore,
        )
        self.agents["conflict_detection"] = ConflictDetectionAgent(
            config=AgentConfig(agent_name="conflict_detection", llm_role="complex_reasoning", model_override="claude-opus-4-5-20250514", max_output_tokens=8192, timeout_seconds=300, temperature=0.2, verification_threshold=0.70),
            llm_provider=factory.get_for_role("complex_reasoning"),
            prompt_loader=prompt_loader,
            db_pool=self.postgres,
            fallback_provider=factory.get_fallback_for_role("complex_reasoning"),
            llm_semaphore=self._llm_semaphore,
        )
        self.agents["dependency_mapper"] = DependencyMapperAgent(
            config=AgentConfig(agent_name="dependency_mapper", llm_role="complex_reasoning", model_override="claude-opus-4-5-20250514", max_output_tokens=8192, temperature=0.1, verification_threshold=0.75),
            llm_provider=factory.get_for_role("complex_reasoning"),
            prompt_loader=prompt_loader,
            db_pool=self.postgres,
            fallback_provider=factory.get_fallback_for_role("complex_reasoning"),
            llm_semaphore=self._llm_semaphore,
        )

        # Tier 3
        self.agents["ripple_effect"] = RippleEffectAnalyzerAgent(
            config=AgentConfig(agent_name="ripple_effect", llm_role="complex_reasoning", model_override="claude-opus-4-5-20250514", max_output_tokens=8192, timeout_seconds=300, temperature=0.2, verification_threshold=0.70),
            llm_provider=factory.get_for_role("complex_reasoning"),
            prompt_loader=prompt_loader,
            db_pool=self.postgres,
            fallback_provider=factory.get_fallback_for_role("complex_reasoning"),
            llm_semaphore=self._llm_semaphore,
        )

        # Query pipeline
        self.agents["query_router"] = QueryRouter(
            config=AgentConfig(agent_name="query_router", llm_role="classification", model_override="claude-sonnet-4-5-20250929", max_output_tokens=1024, temperature=0.0, verification_threshold=0.85),
            llm_provider=factory.get_for_role("classification"),
            prompt_loader=prompt_loader,
            fallback_provider=factory.get_fallback_for_role("classification"),
            llm_semaphore=self._llm_semaphore,
        )
        self.agents["truth_synthesizer"] = TruthSynthesizer(
            config=AgentConfig(agent_name="truth_synthesizer", llm_role="synthesis", model_override="claude-opus-4-5-20250514", max_output_tokens=8192, temperature=0.1, verification_threshold=0.80),
            llm_provider=factory.get_for_role("synthesis"),
            prompt_loader=prompt_loader,
            fallback_provider=factory.get_fallback_for_role("synthesis"),
            llm_semaphore=self._llm_semaphore,
        )

    def get_agent(self, name: str) -> BaseAgent:
        return self.agents[name]
```

### FastAPI Lifespan Integration

```python
# backend/app/main.py (excerpt)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create shared resources + orchestrator
    postgres_pool = await create_postgres_pool()  # NeonDB via EXTERNAL_DATABASE_URL
    redis_client = await create_redis_client()
    vector_store = create_vector_store(postgres_pool)  # pgvector on NeonDB (shared pool)

    app.state.orchestrator = AgentOrchestrator(
        postgres_pool=postgres_pool,
        redis_client=redis_client,
        vector_store=vector_store,
    )

    yield

    # Shutdown: close LLM provider HTTP clients, then database connections
    for agent in app.state.orchestrator.agents.values():
        if hasattr(agent, 'llm') and hasattr(agent.llm, 'close'):
            await agent.llm.close()
    await postgres_pool.close()
    await redis_client.close()

app = FastAPI(lifespan=lifespan)
```

### Shared Blackboard (Agent Communication)

```python
class AgentBlackboard:
    """
    Redis-backed shared state for inter-agent communication.

    Agents publish findings as typed entries. Other agents can query
    the blackboard for relevant context during their analysis.
    """

    def __init__(self, redis_client):
        self.redis = redis_client

    async def publish(self, stack_id: UUID, agent_name: str, entry_type: str, data: dict) -> None:
        """Publish a finding to the blackboard."""
        key = f"blackboard:{stack_id}:{entry_type}"
        entry = {
            "agent": agent_name,
            "type": entry_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }
        await self.redis.rpush(key, json.dumps(entry))
        # Track key for SET-based cleanup (avoid SCAN in clear())
        await self.redis.sadd(f"blackboard_keys:{stack_id}", key)
        # Also publish to pub/sub for real-time subscribers
        await self.redis.publish(f"blackboard:{stack_id}", json.dumps(entry))

    async def query(self, stack_id: UUID, entry_type: str) -> list[dict]:
        """Query all entries of a given type for a contract stack."""
        key = f"blackboard:{stack_id}:{entry_type}"
        entries = await self.redis.lrange(key, 0, -1)
        return [json.loads(e) for e in entries]

    async def clear(self, stack_id: UUID) -> None:
        """Clear all blackboard entries for a contract stack.

        Uses SET-based tracking (O(M)) instead of SCAN (O(N)) — consistent
        with the cache invalidation strategy in §7.
        """
        tracking_key = f"blackboard_keys:{stack_id}"
        keys = await self.redis.smembers(tracking_key)
        if keys:
            await self.redis.delete(*keys, tracking_key)

# Entry types for blackboard communication:
# - "buried_change": AmendmentTracker publishes when it finds a buried change
# - "stale_reference": ConflictDetection publishes stale reference findings
# - "high_risk_dependency": DependencyMapper publishes critical dependency paths
# - "low_confidence_clause": OverrideResolution flags uncertain resolutions
# - "amendment_context": AmendmentTracker publishes amendment type/rationale per section
```

---

## 2. Ingestion Pipeline: `process_contract_stack()`

### Six Stages with Progress Percentages

```
 0% ──────── 30% ──── 50% ── 55% ────────── 80% ────── 90% ──── 100%
  │           │        │      │              │          │         │
  ▼           ▼        ▼      ▼              ▼          ▼         ▼
 Parse     Track    Sequence Override    Dependency  Conflict  Done
 (parallel) (seq)   (single) (parallel)  Mapping    Detection
```

| Stage | % Range | Agent(s) | Concurrency |
|-------|---------|----------|-------------|
| 1. Document Parsing | 0-30% | DocumentParserAgent | **Parallel** — all 6 docs simultaneously |
| 2. Amendment Tracking | 30-50% | AmendmentTrackerAgent | **Sequential** — each amendment needs prior results |
| 3. Temporal Sequencing | 50-55% | TemporalSequencerAgent | Single call |
| 4. Override Resolution | 55-80% | OverrideResolutionAgent | **Parallel** — each section independent |
| 5. Dependency Mapping | 80-90% | DependencyMapperAgent | Single call (all clauses) |
| 6. Conflict Detection | 90-100% | ConflictDetectionAgent | Single call (all clauses + graph) |

### Implementation

```python
class AgentOrchestrator:
    async def process_contract_stack(
        self,
        contract_stack_id: UUID,
        job_id: str,
        progress_callback: Callable[[PipelineProgressEvent], Awaitable[None]],
    ) -> dict:
        """Full ingestion pipeline for a contract stack."""

        # Invalidate any cached query results for this stack before re-processing
        await self.invalidate_cache(contract_stack_id)

        # ── Create TraceContext for this pipeline run ─────────
        trace = TraceContext(job_id=job_id)
        # Inject trace into all agents for this pipeline run
        for agent in self.agents.values():
            agent.trace = trace

        # ── Pre-flight: Assess document set complexity ─────────
        # The orchestrator examines uploaded documents before executing the pipeline.
        # This enables adaptive behavior: skipping unnecessary stages, allocating
        # more resources for complex stacks, and flagging potential issues early.
        documents = await self._get_documents(contract_stack_id)
        doc_count = len(documents)
        has_amendments = any(d.document_type == DocumentType.AMENDMENT for d in documents)

        # ── LLM-Driven Pipeline Planning ──────────────────────
        # The orchestrator uses the LLM to assess complexity and plan execution.
        # This enables adaptive behavior beyond simple flag checks.
        planner = self.get_agent("query_router")  # reuse Sonnet for fast planning
        plan_prompt = (
            f"You are planning the analysis pipeline for a contract stack.\n"
            f"Documents: {doc_count} total, "
            f"{'has amendments' if has_amendments else 'no amendments (single CTA)'}.\n"
            f"Document types: {[d.document_type.value for d in documents]}\n\n"
            f"Determine:\n"
            f"1. skip_stages: list of stages to skip (e.g., ['amendment_tracking', 'sequencing'] if no amendments)\n"
            f"2. complexity: 'low' (1-2 docs), 'medium' (3-6 docs), 'high' (7+ docs)\n"
            f"3. parallel_resolution: true if override resolution should run in parallel\n"
            f"4. warnings: any potential issues detected from document names\n\n"
            f"Return JSON: {{\"skip_stages\": [...], \"complexity\": \"...\", "
            f"\"parallel_resolution\": true, \"warnings\": [...]}}"
        )
        pipeline_plan = await planner.call_llm(
            "You are a pipeline planning assistant. Analyze the document set and plan execution.",
            plan_prompt,
        )
        skip_stages = set(pipeline_plan.get("skip_stages", []))
        logger.info("Pipeline plan: complexity=%s, skip=%s, warnings=%s",
                     pipeline_plan.get("complexity"), skip_stages, pipeline_plan.get("warnings"))

        # ── Stage 1: Parse all documents (parallel) ──────────────
        await progress_callback(PipelineProgressEvent(
            job_id=job_id, pipeline_stage="document_parsing",
            overall_percent=0, message=f"Parsing {len(documents)} documents...",
            current_agent="document_parser", timestamp=datetime.utcnow(),
        ))

        parser = self.get_agent("document_parser")
        parse_tasks = [
            parser.run(DocumentParseInput(
                file_path=doc.file_path,
                document_type=doc.document_type,
                contract_stack_id=contract_stack_id,
            ))
            for doc in documents
        ]
        # Use return_exceptions=True so one document failure doesn't cancel all others
        parse_results = await asyncio.gather(*parse_tasks, return_exceptions=True)

        # Check for failures — raise first exception if any document failed
        parsed_outputs: list[DocumentParseOutput] = []
        for i, result in enumerate(parse_results):
            if isinstance(result, Exception):
                raise PipelineError(
                    f"Document parsing failed for {documents[i].file_path}: {result}",
                    stage="document_parsing",
                )
            parsed_outputs.append(result)

        await progress_callback(PipelineProgressEvent(
            job_id=job_id, pipeline_stage="document_parsing",
            overall_percent=30, message=f"Parsed {len(parsed_outputs)} documents",
            current_agent="document_parser", timestamp=datetime.utcnow(),
        ))

        # Save parsed data to PostgreSQL
        await self._save_parsed_documents(contract_stack_id, parsed_outputs)

        # ── Stage 2: Track amendments (sequential) ───────────────
        # Validate metadata presence (required for downstream processing)
        for p in parsed_outputs:
            if p.metadata is None:
                raise PipelineError(
                    f"Document {p.document_id} has no metadata — parser failed to extract it",
                    stage="amendment_tracking",
                )

        tracker = self.get_agent("amendment_tracker")
        cta_output = next(p for p in parsed_outputs if p.metadata.document_type == DocumentType.CTA)
        amendment_outputs = [p for p in parsed_outputs if p.metadata.document_type == DocumentType.AMENDMENT]
        amendment_outputs.sort(key=lambda p: p.metadata.effective_date)

        tracking_results: list[AmendmentTrackOutput] = []
        for i, amend_output in enumerate(amendment_outputs):
            await progress_callback(PipelineProgressEvent(
                job_id=job_id, pipeline_stage="amendment_tracking",
                overall_percent=30 + int(20 * (i / len(amendment_outputs))),
                message=f"Tracking Amendment {i+1} of {len(amendment_outputs)}...",
                current_agent="amendment_tracker", timestamp=datetime.utcnow(),
            ))

            track_result = await tracker.run(AmendmentTrackInput(
                amendment_document_id=amend_output.document_id,
                amendment_number=amend_output.metadata.amendment_number or (i + 1),
                amendment_text="",  # not used directly — sections + tables are used
                amendment_sections=amend_output.sections,
                amendment_tables=amend_output.tables,     # tables critical for Pain Point #2 (budget exhibits)
                original_sections=cta_output.sections,
                original_tables=cta_output.tables,
                prior_amendments=tracking_results,  # sequential context
            ))
            tracking_results.append(track_result)

        # Save amendments to PostgreSQL
        await self._save_amendment_tracking(contract_stack_id, tracking_results)

        await progress_callback(PipelineProgressEvent(
            job_id=job_id, pipeline_stage="amendment_tracking",
            overall_percent=50, message=f"Tracked {len(tracking_results)} amendments",
            current_agent="amendment_tracker", timestamp=datetime.utcnow(),
        ))

        # ── Stage 3: Temporal sequencing ─────────────────────────
        sequencer = self.get_agent("temporal_sequencer")
        sequence_result = await sequencer.run(TemporalSequenceInput(
            contract_stack_id=contract_stack_id,
            documents=[
                DocumentSummary(
                    document_id=p.document_id,
                    document_type=p.metadata.document_type,
                    effective_date=p.metadata.effective_date,
                    document_version=f"Amendment {p.metadata.amendment_number}" if p.metadata.amendment_number else "Original CTA",
                    filename="",
                )
                for p in parsed_outputs
            ],
        ))

        await progress_callback(PipelineProgressEvent(
            job_id=job_id, pipeline_stage="temporal_sequencing",
            overall_percent=55, message="Documents sequenced",
            current_agent="temporal_sequencer", timestamp=datetime.utcnow(),
        ))

        # Save timeline and version tree to PostgreSQL
        await self._save_temporal_sequence(contract_stack_id, sequence_result)

        # Build document label lookup from sequence timeline (used downstream)
        doc_label_map: dict[UUID, str] = {
            event.document_id: event.label for event in sequence_result.timeline
        }

        # ── Stage 4: Override resolution (parallel per section) ──
        resolver = self.get_agent("override_resolution")
        sections_to_resolve = self._build_section_amendment_map(
            cta_output, tracking_results, parsed_outputs
        )

        # Limit concurrent LLM calls to prevent rate limiting (semaphore shared across pipelines)
        async def _resolve_with_limit(resolution_input):
            async with self._llm_semaphore:
                return await resolver.run(resolution_input)

        resolve_tasks = [
            _resolve_with_limit(resolution_input)
            for resolution_input in sections_to_resolve
        ]
        resolve_results = await asyncio.gather(*resolve_tasks, return_exceptions=True)

        resolved_clauses: list[OverrideResolutionOutput] = []
        for i, result in enumerate(resolve_results):
            if isinstance(result, Exception):
                raise PipelineError(
                    f"Override resolution failed for section {sections_to_resolve[i].section_number}: {result}",
                    stage="override_resolution",
                )
            resolved_clauses.append(result)

        await progress_callback(PipelineProgressEvent(
            job_id=job_id, pipeline_stage="override_resolution",
            overall_percent=80, message=f"Resolved {len(resolved_clauses)} clause versions",
            current_agent="override_resolution", timestamp=datetime.utcnow(),
        ))

        # Save resolved clauses to PostgreSQL
        await self._save_resolved_clauses(contract_stack_id, resolved_clauses)

        # Embed resolved clauses for query-time semantic search (two-tier embedding)
        # Stage 1 checkpoint embeddings (is_resolved=FALSE) were written during document parsing.
        # Now we embed the post-resolution "truth" versions with is_resolved=TRUE.
        # Only is_resolved=TRUE embeddings are searched at query time, ensuring users
        # always get the current effective clause text, not superseded versions.
        await self._embed_resolved_clauses(contract_stack_id, resolved_clauses)

        # ── Verification Gate: Check override resolution quality ────
        # Note: OverrideResolutionOutput has confidence on clause_version, not on the top-level object.
        # We also retrieve the original input for re-processing from sections_to_resolve.
        low_confidence_items: list[tuple[OverrideResolutionOutput, OverrideResolutionInput]] = []
        for i, c in enumerate(resolved_clauses):
            if c.clause_version.confidence < self.config_threshold_for("override_resolution"):
                low_confidence_items.append((c, sections_to_resolve[i]))

        if low_confidence_items:
            logger.warning(
                "%d clauses have low confidence — re-processing via run() with self-verification",
                len(low_confidence_items),
            )
            for clause_output, original_input in low_confidence_items:
                # Re-run via run() which triggers _verify_output() + confidence-gated re-processing
                re_resolved = await resolver.run(original_input)
                # Replace the low-confidence result with the re-processed one
                resolved_clauses = [
                    re_resolved if c.clause_version.section_number == clause_output.clause_version.section_number else c
                    for c in resolved_clauses
                ]

        # ── Stage 5: Dependency mapping ──────────────────────────
        mapper = self.get_agent("dependency_mapper")
        current_clauses = [
            CurrentClause(
                section_number=r.clause_version.section_number,
                section_title=r.clause_version.section_title,
                current_text=r.clause_version.current_text,
                clause_category=r.clause_version.clause_category,
                source_document_id=r.clause_version.last_modified_by,
                source_document_label=doc_label_map.get(r.clause_version.last_modified_by, "Unknown"),
                effective_date=r.clause_version.last_modified_date,
            )
            for r in resolved_clauses
        ]

        dep_result = await mapper.run(DependencyMapInput(
            contract_stack_id=contract_stack_id,
            current_clauses=current_clauses,
        ))

        await progress_callback(PipelineProgressEvent(
            job_id=job_id, pipeline_stage="dependency_mapping",
            overall_percent=90, message=f"Mapped {dep_result.total_edges} dependencies",
            current_agent="dependency_mapper", timestamp=datetime.utcnow(),
        ))

        # ── Stage 6: Conflict detection ──────────────────────────
        detector = self.get_agent("conflict_detection")
        context = await self._build_contract_context(contract_stack_id)

        conflict_result = await detector.run(ConflictDetectionInput(
            contract_stack_id=contract_stack_id,
            current_clauses=current_clauses,
            contract_stack_context=context,
            dependency_graph=dep_result.dependencies,
        ))

        # Save conflicts to PostgreSQL
        await self._save_conflicts(contract_stack_id, conflict_result.conflicts)

        # ── Agent Communication: Process needs_clarification requests ─────
        # Agents can request re-analysis by upstream agents when they detect
        # ambiguities that need resolution.
        if hasattr(conflict_result, 'needs_clarification') and conflict_result.needs_clarification:
            for clarification in conflict_result.needs_clarification:
                target_agent = clarification.target_agent
                reason = clarification.reason
                logger.info(
                    "ConflictDetection requests re-analysis from %s: %s",
                    target_agent, reason,
                )
                # Publish to blackboard for downstream visibility
                await self.blackboard.publish(
                    contract_stack_id, "conflict_detection",
                    "clarification_request",
                    {"target_agent": target_agent, "reason": reason},
                )
                # Execute re-analysis if target is an upstream agent
                if target_agent == "override_resolution" and clarification.section_number:
                    section = clarification.section_number
                    original_input = next(
                        (s for s in sections_to_resolve if s.section_number == section), None
                    )
                    if original_input:
                        re_resolved = await resolver.run(original_input)
                        resolved_clauses = [
                            re_resolved if c.clause_version.section_number == section else c
                            for c in resolved_clauses
                        ]

        await progress_callback(PipelineProgressEvent(
            job_id=job_id, pipeline_stage="complete",
            overall_percent=100,
            message=f"Pipeline complete: {len(resolved_clauses)} clauses, {len(conflict_result.conflicts)} conflicts detected",
            current_agent=None, timestamp=datetime.utcnow(),
        ))

        return {
            "clauses_processed": len(resolved_clauses),
            "conflicts_detected": len(conflict_result.conflicts),
            "dependencies_mapped": dep_result.total_edges,
            "conflict_summary": conflict_result.summary,
        }
```

### Celery Task Definition

```python
# backend/app/tasks.py

from celery import Celery
import asyncio
import json
from uuid import UUID

celery_app = Celery("contractiq", broker="redis://localhost:6379/0")

# Redis TTL for job status keys (24 hours)
JOB_KEY_TTL = 86400

@celery_app.task(bind=True)
def process_contract_stack_task(self, contract_stack_id: str):
    """
    Celery task wrapping the async ingestion pipeline.

    IMPORTANT: Celery workers run in separate processes — they do NOT share
    the FastAPI application's in-memory AgentOrchestrator. Each task must
    create its own orchestrator instance with fresh database connections.
    """
    job_id = self.request.id

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            _run_pipeline(contract_stack_id, job_id)
        )
        return result
    finally:
        loop.close()

async def _run_pipeline(contract_stack_id: str, job_id: str) -> dict:
    """Create per-task resources and run the ingestion pipeline."""
    import redis.asyncio as aioredis

    # Create fresh connections for this worker process
    postgres_pool = await create_postgres_pool()  # NeonDB
    redis_client = aioredis.from_url("redis://localhost:6379/0")
    vector_store = create_vector_store(postgres_pool)  # pgvector on NeonDB

    try:
        # Create orchestrator for this task
        orchestrator = AgentOrchestrator(
            postgres_pool=postgres_pool,
            redis_client=redis_client,
            vector_store=vector_store,
        )

        # Set initial state with TTL
        await redis_client.setex(f"job:{job_id}:status", JOB_KEY_TTL, "processing")

        async def progress_callback(event: PipelineProgressEvent):
            # Publish to Redis pub/sub for WebSocket forwarding
            await redis_client.publish(
                f"job:{job_id}:progress",
                event.model_dump_json(),
            )
            await redis_client.setex(f"job:{job_id}:progress", JOB_KEY_TTL, str(event.overall_percent))
            await redis_client.setex(f"job:{job_id}:stage", JOB_KEY_TTL, event.pipeline_stage)

        # Note: cache invalidation happens inside process_contract_stack() — no need to call again here.
        result = await orchestrator.process_contract_stack(
            UUID(contract_stack_id), job_id, progress_callback
        )
        await redis_client.setex(f"job:{job_id}:status", JOB_KEY_TTL, "completed")
        await redis_client.setex(f"job:{job_id}:result", JOB_KEY_TTL, json.dumps(result))
        return result
    except Exception as e:
        await redis_client.setex(f"job:{job_id}:status", JOB_KEY_TTL, "failed")
        await redis_client.setex(f"job:{job_id}:error", JOB_KEY_TTL, str(e))
        raise
    finally:
        await postgres_pool.close()
        await redis_client.close()
```

### State in Redis for Resumability

All job keys use `setex` with a 24-hour TTL (`JOB_KEY_TTL = 86400`) to prevent unbounded memory growth.

```
job:{job_id}:status   → "queued" | "processing" | "completed" | "failed"   (TTL: 24h)
job:{job_id}:progress → 0-100                                               (TTL: 24h)
job:{job_id}:stage    → "document_parsing" | "amendment_tracking" | ...      (TTL: 24h)
job:{job_id}:result   → JSON result (when completed)                        (TTL: 24h)
job:{job_id}:error    → error message (when failed)                         (TTL: 24h)
```

---

## 3. Query Pipeline: `handle_query()`

### Four Steps

```
User Query
     │
     ▼
┌──────────────────────────────┐
│ Step 1: Classify (QueryRouter)│  < 2 seconds target
│   - Determine query type     │
│   - Extract entities         │
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│ Step 2: Retrieve             │
│   - pgvector semantic search │
│     (is_resolved=TRUE only)  │
│   - PostgreSQL graph (CTE)   │
│   - PostgreSQL direct lookup │
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│ Step 3: Dispatch             │
│   - Route to appropriate     │
│     agent based on type      │
│   - truth → TruthSynthesizer │
│   - conflict → ConflictDet.  │
│   - ripple → RippleEffect    │
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│ Step 4: Synthesize           │
│   - TruthSynthesizer formats │
│     answer with citations    │
│   - Include caveats/conf.    │
└──────────────────────────────┘
```

### Implementation

```python
class AgentOrchestrator:
    async def handle_query(
        self, query_text: str, contract_stack_id: UUID
    ) -> TruthSynthesisOutput:
        start = time.monotonic()

        # Step 1: Check cache
        cache_key = f"query:{contract_stack_id}:{hashlib.sha256(query_text.encode()).hexdigest()}"
        cached = await self.redis.get(cache_key)
        if cached:
            return TruthSynthesisOutput.model_validate_json(cached)

        # Step 2: Classify query
        router = self.get_agent("query_router")
        route_result: QueryRouteOutput = await router.run(
            QueryRouteInput(query_text=query_text, contract_stack_id=contract_stack_id)
        )

        # Handle compound queries: execute sub-queries in parallel and merge results
        if hasattr(route_result, 'sub_queries') and len(route_result.sub_queries) > 1:
            sub_results = await asyncio.gather(*[
                self._execute_sub_query(sq, contract_stack_id)
                for sq in route_result.sub_queries
            ])
            # Synthesize all sub-query results into a unified answer
            synthesizer = self.get_agent("truth_synthesizer")
            merged_answer = await self._synthesize_compound_answer(
                query_text, sub_results, synthesizer
            )
            await self.redis.setex(cache_key, 3600, merged_answer.model_dump_json())
            await self.redis.sadd(f"cache_keys:{contract_stack_id}", cache_key)
            return merged_answer

        # Step 3: Retrieve relevant data
        relevant_clauses = await self._retrieve_clauses(
            query_text, contract_stack_id, route_result
        )
        relevant_conflicts = await self._get_relevant_conflicts(
            contract_stack_id, route_result.extracted_entities
        )

        # Step 4: Dispatch + Synthesize
        if route_result.query_type == QueryType.RIPPLE_ANALYSIS:
            # Ripple queries need special handling — extract proposed change from query
            ripple_agent = self.get_agent("ripple_effect")
            proposed_change = await self._extract_proposed_change(query_text, relevant_clauses)
            ripple_result = await ripple_agent.run(RippleEffectInput(
                contract_stack_id=contract_stack_id,
                proposed_change=proposed_change,
            ))
            # Synthesize ripple results into answer format
            answer = await self._synthesize_ripple_answer(query_text, ripple_result)
        else:
            # Truth reconstitution or conflict queries → TruthSynthesizer
            synthesizer = self.get_agent("truth_synthesizer")
            answer = await synthesizer.run(TruthSynthesisInput(
                query_text=query_text,
                query_type=route_result.query_type,
                contract_stack_id=contract_stack_id,
                relevant_clauses=relevant_clauses,
                conflicts=relevant_conflicts,
            ))

        # Cache result (1 hour TTL) and track key for SET-based invalidation
        await self.redis.setex(cache_key, 3600, answer.model_dump_json())
        await self.redis.sadd(f"cache_keys:{contract_stack_id}", cache_key)

        # Save query to PostgreSQL
        latency_ms = int((time.monotonic() - start) * 1000)
        await self._save_query(contract_stack_id, query_text, answer, latency_ms)

        return answer

    async def _retrieve_clauses(
        self, query_text: str, stack_id: UUID, route: QueryRouteOutput
    ) -> list[CurrentClause]:
        """Multi-source retrieval: pgvector + PostgreSQL graph + PostgreSQL direct lookup."""

        # pgvector semantic search (Gemini gemini-embedding-001, RETRIEVAL_QUERY)
        # Only searches embeddings WHERE is_resolved = TRUE — these are the
        # post-Stage 4 resolved clause embeddings written by _embed_resolved_clauses().
        # Raw per-document checkpoint embeddings (is_resolved=FALSE) are excluded
        # to prevent returning superseded clause versions in query results.
        similar = await self.vector_store.query_similar(
            query_text, stack_id, n_results=10, filter={"is_resolved": True}
        )
        section_numbers = set()
        for row in similar:
            sn = row.get("section_number")
            if sn:
                section_numbers.add(sn)

        # Add entity-based section matches
        for entity in route.extracted_entities:
            section_numbers.add(entity)
        section_numbers.discard(None)

        # PostgreSQL: batch lookup for all section numbers (pgvector + entities)
        clauses = []
        if section_numbers:
            rows = await self.postgres.fetch(
                "SELECT section_number, section_title, current_text, clause_category, "
                "source_document_id, effective_date FROM clauses "
                "WHERE section_number = ANY($1) AND contract_stack_id = $2 AND is_current = TRUE",
                list(section_numbers), stack_id,
            )
            for row in rows:
                clauses.append(CurrentClause(
                    section_number=row["section_number"],
                    section_title=row["section_title"] or "",
                    current_text=row["current_text"] or "",
                    clause_category=row["clause_category"] or "general",
                    source_document_id=row["source_document_id"],
                    source_document_label="",
                ))
        return clauses

    async def _extract_proposed_change(
        self, query_text: str, relevant_clauses: list[CurrentClause]
    ) -> ProposedChange:
        """
        Use LLM to extract a structured proposed change from natural language query.

        The LLM identifies which clause the user is referring to (semantically,
        not via regex), extracts the proposed modification, and structures it
        for the RippleEffectAnalyzer.
        """
        router = self.get_agent("query_router")
        clauses_context = "\n".join(
            f"Section {c.section_number} ({c.section_title}): {c.current_text[:200]}..."
            for c in relevant_clauses[:10]
        )
        system_prompt = router.prompts.get("query_router_classify")
        user_prompt = (
            f"Extract the proposed change from this query:\n\n"
            f"Query: {query_text}\n\n"
            f"Available sections:\n{clauses_context}\n\n"
            f"Return JSON: {{\"section_number\": \"...\", \"current_text\": \"...\", "
            f"\"proposed_text\": \"...\", \"change_description\": \"...\"}}"
        )
        result = await router.call_llm(system_prompt, user_prompt)

        target_clause = next(
            (c for c in relevant_clauses if c.section_number == result.get("section_number")),
            relevant_clauses[0] if relevant_clauses else None,
        )

        return ProposedChange(
            section_number=result.get("section_number", target_clause.section_number if target_clause else "unknown"),
            current_text=target_clause.current_text if target_clause else result.get("current_text", ""),
            proposed_text=result.get("proposed_text", query_text),
            change_description=result.get("change_description", query_text),
        )

    async def _synthesize_ripple_answer(
        self, query_text: str, ripple_result: RippleEffectOutput
    ) -> TruthSynthesisOutput:
        """Convert RippleEffectOutput into TruthSynthesisOutput format for API consistency."""
        synthesizer = self.get_agent("truth_synthesizer")
        # Build a structured summary of the ripple analysis
        impact_summary = json.dumps(ripple_result.model_dump(mode="json"), indent=2)

        system_prompt = synthesizer.prompts.get("truth_synthesizer_answer")
        user_prompt = synthesizer.prompts.get(
            "truth_synthesizer_ripple_input",
            query_text=query_text,
            impact_summary=impact_summary,
        )
        result = await synthesizer.call_llm(system_prompt, user_prompt)

        return TruthSynthesisOutput(
            answer=result["answer"],
            sources=[SourceCitation(**s) for s in result.get("sources", [])],
            confidence=result.get("confidence", 0.8),
            caveats=result.get("caveats", []),
            agent_reasoning=[],
        )

    async def _build_contract_context(self, contract_stack_id: UUID) -> ContractStackContext:
        """Load contract stack metadata from PostgreSQL for conflict detection context."""
        row = await self.postgres.fetchrow(
            "SELECT study_name, sponsor_name, site_name, therapeutic_area, start_date, end_date "
            "FROM contract_stacks WHERE id = $1",
            contract_stack_id,
        )
        return ContractStackContext(
            study_name=row["study_name"],
            sponsor_name=row["sponsor_name"],
            site_name=row["site_name"],
            therapeutic_area=row["therapeutic_area"],
            study_start_date=row["start_date"],
            study_end_date=row["end_date"],
        )
```

### PostgreSQL Index Requirements

These indexes are created via Alembic migration (see technical_spec.md clause_dependencies schema) for performant graph traversal:

```sql
-- Already defined in technical_spec.md clause_dependencies schema:
-- idx_clause_deps_stack ON clause_dependencies(contract_stack_id)
-- idx_clause_deps_from ON clause_dependencies(from_clause_id)
-- idx_clause_deps_to ON clause_dependencies(to_clause_id)
-- idx_clause_deps_type ON clause_dependencies(relationship_type)

-- Clause lookups by section_number + contract_stack_id (used by recursive CTEs)
CREATE INDEX IF NOT EXISTS idx_clauses_stack_section
ON clauses(contract_stack_id, section_number);

-- Document supersession lookups
CREATE INDEX IF NOT EXISTS idx_doc_supersessions_stack
ON document_supersessions(contract_stack_id);
```

---

## 4. QueryRouter

**Purpose:** Classify incoming queries and extract entities for retrieval.
**LLM:** Claude Sonnet (fast classification, <2s target)
**File:** `backend/app/agents/query_router.py`

```python
class QueryRouter(BaseAgent):
    async def process(self, input_data: QueryRouteInput) -> QueryRouteOutput:
        start = time.monotonic()
        system_prompt = self.prompts.get("query_router_classify")
        user_prompt = self.prompts.get(
            "query_router_input",
            query_text=input_data.query_text,
        )

        result = await self.call_llm(system_prompt, user_prompt)

        # Support compound queries: if the LLM identifies multiple sub-queries,
        # return them all. The orchestrator executes sub-queries in parallel.
        sub_queries = result.get("sub_queries", [])
        if not sub_queries:
            # Single query — wrap in sub_queries format for uniform handling
            sub_queries = [{
                "query_type": result["query_type"],
                "query_text": input_data.query_text,
                "entities": result.get("entities", []),
            }]

        return QueryRouteOutput(
            query_type=QueryType(result["query_type"]),  # primary type
            extracted_entities=result.get("entities", []),
            confidence=result.get("confidence", 0.9),
            routing_latency_ms=int((time.monotonic() - start) * 1000),
            llm_reasoning=result.get("reasoning", ""),
            sub_queries=sub_queries,
        )
```

### Prompt File

**`prompt/query_router_classify.txt`**
```
Classify user queries about clinical trial agreements into types:

1. truth_reconstitution — Questions about the current state of a clause or fact
   Examples: "What are the current payment terms?" "Who is the PI?" "What does Section 7.2 say?"

2. conflict_detection — Questions about conflicts, inconsistencies, or risks
   Examples: "Are there any conflicts?" "Find inconsistencies in the budget" "What risks exist?"

3. ripple_analysis — Questions about impact of changes
   Examples: "What if we change data retention to 25 years?" "Impact of extending the study?"

4. general — General questions that don't fit above categories
   Examples: "Summarize this contract" "How many amendments are there?"

Also extract entities: section numbers, person names, dates, and topics mentioned.

COMPOUND QUERIES: If the user asks a question that spans multiple types, decompose it into sub-queries.
Example: "What are the current payment terms and are there any conflicts?" → two sub-queries:
  1. truth_reconstitution: "What are the current payment terms?"
  2. conflict_detection: "Are there any conflicts related to payment terms?"

Return sub_queries only when the query genuinely spans multiple types. Simple queries should NOT be decomposed.

Before producing your final answer, reason step-by-step about your analysis.
Include your reasoning in the 'reasoning' field of the JSON output.

Return JSON:
{{
  "reasoning": "your step-by-step analysis...",
  "query_type": "truth_reconstitution|conflict_detection|ripple_analysis|general",
  "entities": ["7.2", "payment terms", "Net 45"],
  "confidence": 0.95,
  "sub_queries": [
    {{"query_type": "truth_reconstitution", "query_text": "What are the current payment terms?", "entities": ["7.2", "payment"]}},
    {{"query_type": "conflict_detection", "query_text": "Are there conflicts related to payment?", "entities": ["payment"]}}
  ]
}}
```

**`prompt/query_router_input.txt`**
```
Classify the following user query:

<user_query>
{query_text}
</user_query>

Return JSON with the classification, extracted entities, and confidence.
```

---

## 5. TruthSynthesizer

**Purpose:** Generate a comprehensive answer with source citations, caveats, and confidence.
**LLM:** Claude Opus (synthesis requires comprehensive reasoning)
**File:** `backend/app/agents/truth_synthesizer.py`

```python
class TruthSynthesizer(BaseAgent):
    async def process(self, input_data: TruthSynthesisInput) -> TruthSynthesisOutput:
        system_prompt = self.prompts.get("truth_synthesizer_answer")
        user_prompt = self.prompts.get(
            "truth_synthesizer_input",
            query_text=input_data.query_text,
            query_type=input_data.query_type.value,
            relevant_clauses=self._format_clauses(input_data.relevant_clauses),
            known_conflicts=self._format_conflicts(input_data.conflicts),
        )

        result = await self.call_llm(system_prompt, user_prompt)

        return TruthSynthesisOutput(
            answer=result["answer"],
            sources=[SourceCitation(**s) for s in result["sources"]],
            confidence=result["confidence"],
            caveats=result.get("caveats", []),
            agent_reasoning=self._build_reasoning_chain(input_data),
            llm_reasoning=result.get("reasoning", ""),
        )
```

### Prompt File

**`prompt/truth_synthesizer_answer.txt`**
```
You are answering a question about a clinical trial agreement contract stack.

Rules:
1. Base your answer ONLY on the provided clauses and conflicts. Do not invent information.
2. ALWAYS cite sources: include document name, section number, and effective date.
3. If information is ambiguous or conflicting, state that clearly with both positions.
4. If the answer requires information not in the provided clauses, say so explicitly.
5. Include confidence level based on evidence quality.
6. List any caveats: known conflicts, ambiguities, or gaps that affect the answer.

Before producing your final answer, reason step-by-step about your analysis.
Include your reasoning in the 'reasoning' field of the JSON output.

Return JSON:
{{
  "reasoning": "your step-by-step analysis...",
  "answer": "comprehensive answer with inline citations",
  "sources": [
    {{
      "document_id": "...",
      "document_name": "Amendment 3",
      "section_number": "7.2",
      "relevant_text": "exact text excerpt",
      "effective_date": "2023-08-17"
    }}
  ],
  "confidence": 0.95,
  "caveats": ["Payment terms were changed in a COVID-related amendment, which may have been unintentional"]
}}
```

---

## 6. WebSocket Protocol

### Architecture

```
Agent progress_callback
     │
     ▼
Redis Pub/Sub (channel: job:{job_id}:progress)
     │
     ▼
WebSocket Manager (subscribes to Redis)
     │
     ▼
Client WebSocket (ws://host/api/v1/ws/jobs/{job_id})
```

### Message Types

```python
# WebSocket message format
{
    "type": "progress",             # progress | stage_complete | pipeline_complete | error
    "job_id": "uuid",
    "data": {
        "pipeline_stage": "document_parsing",
        "overall_percent": 25,
        "message": "Parsing Amendment 3...",
        "current_agent": "document_parser",
        "timestamp": "2024-01-15T10:00:05Z"
    }
}
```

| Type | When Sent | Data Fields |
|------|-----------|-------------|
| `progress` | During processing | pipeline_stage, overall_percent, message, current_agent |
| `stage_complete` | When a pipeline stage finishes | stage_name, duration_ms, items_processed |
| `pipeline_complete` | When full pipeline finishes | clauses_processed, conflicts_detected, total_duration_ms |
| `error` | On fatal error | error_type, error_message, agent_name |

### WebSocket Endpoint

```python
# backend/app/api/websocket.py

@router.websocket("/api/v1/ws/jobs/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()

    pubsub = redis_client.pubsub()
    channel = f"job:{job_id}:progress"
    await pubsub.subscribe(channel)

    try:
        # Send any already-accumulated progress so client doesn't miss early events
        current_progress = await redis_client.get(f"job:{job_id}:progress")
        if current_progress:
            progress_val = current_progress.decode("utf-8") if isinstance(current_progress, bytes) else current_progress
            await websocket.send_text(json.dumps({"type": "catchup", "overall_percent": int(progress_val)}))

        # Check if job already completed/failed before we subscribed
        status_raw = await redis_client.get(f"job:{job_id}:status")
        status = status_raw.decode("utf-8") if isinstance(status_raw, bytes) else status_raw
        if status in ("completed", "failed"):
            await websocket.send_text(json.dumps({"type": "pipeline_complete", "status": status}))
            return

        async for message in pubsub.listen():
            if message["type"] == "message" or message["type"] == b"message":
                # Redis pub/sub data is bytes — decode to str for WebSocket
                data = message["data"]
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                await websocket.send_text(data)

                # Detect pipeline completion and close cleanly
                try:
                    parsed = json.loads(data)
                    if parsed.get("pipeline_stage") == "complete" or parsed.get("overall_percent") == 100:
                        await websocket.close()
                        return
                except (json.JSONDecodeError, KeyError):
                    pass
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(channel)
```

---

## 7. Caching Strategy

| Cache Key Pattern | TTL | Invalidation |
|-------------------|-----|--------------|
| `query:{stack_id}:{hash}` | 1 hour | On re-processing of contract stack |
| `clauses:{stack_id}` | Until re-processing | On re-processing |
| `conflicts:{stack_id}` | Until re-processing | On re-processing |
| `cache_keys:{stack_id}` | None (tracking set) | Deleted during invalidation |
| `blackboard:{stack_id}:{type}` | Until re-processing | Cleared via `AgentBlackboard.clear()` |

**Cache invalidation** uses Redis SET-based tracking (`cache_keys:{stack_id}`) instead of `SCAN`. Each time a cache key is written, it is also added to the tracking set via `SADD`. On invalidation, the orchestrator reads the set members via `SMEMBERS` (O(M) where M = keys for this stack) and deletes them in a single `DELETE` call, avoiding the O(N) `SCAN` over the entire Redis keyspace.

```python
async def invalidate_cache(self, contract_stack_id: UUID) -> None:
    """Invalidate all cached data for a contract stack.

    Uses Redis SET-based tracking instead of SCAN for O(M) instead of O(N) performance,
    where M = keys for this stack, N = total Redis keyspace.
    """
    tracking_key = f"cache_keys:{contract_stack_id}"
    keys = await self.redis.smembers(tracking_key)
    if keys:
        await self.redis.delete(*keys, tracking_key)
    # Also clear blackboard entries
    await self.blackboard.clear(contract_stack_id)
```

---

## 8. Implementation Notes

### Progress Callback Architecture

There are two levels of progress reporting:

| Level | Callback Signature | Scope |
|-------|-------------------|-------|
| **Pipeline** | `Callable[[PipelineProgressEvent], None]` | Orchestrator stage boundaries (0%, 30%, 50%, etc.) |
| **Agent** | `Callable[[str, int, str], None]` | Per-agent internal progress (e.g., "Parsed chunk 3/5") |

Currently, agents in `_init_agents()` are created **without** a `progress_callback`, so internal agent progress (e.g., `self._report_progress(...)` calls within `DocumentParserAgent.process()`) are no-ops. Only pipeline-level progress events reach the WebSocket.

**To enable per-agent progress in implementation:** Create an adapter that bridges the agent-level callback `(stage, percent, message)` to a `PipelineProgressEvent` and publishes it to Redis. This adapter would be passed to each agent at the start of each pipeline stage:

```python
def _make_agent_callback(self, job_id: str, agent_name: str, progress_callback):
    async def agent_cb(stage: str, percent: int, message: str):
        await progress_callback(PipelineProgressEvent(
            job_id=job_id, pipeline_stage=stage,
            overall_percent=percent,  # NOTE: needs mapping to pipeline-wide range
            message=message, current_agent=agent_name,
            timestamp=datetime.utcnow(),
        ))
    return agent_cb
```

### Prompt File Additions

This document introduces 2 additional prompt files beyond those listed in doc 07's manifest:

| # | File | Agent | Purpose | Key Placeholders |
|---|------|-------|---------|-----------------|
| 18 | `query_router_input.txt` | QueryRouter | User query input (wrapped in delimiters) | `{query_text}` |
| 19 | `ripple_effect_recommendations_input.txt` | RippleEffectAnalyzerAgent | Recommendation synthesis input | `{section_number}`, `{current_text}`, `{proposed_text}`, `{impacts_json}` |
| 20 | `truth_synthesizer_input.txt` | TruthSynthesizer | User query + clauses + conflicts input | `{query_text}`, `{query_type}`, `{relevant_clauses}`, `{known_conflicts}` |
| 21 | `truth_synthesizer_ripple_input.txt` | TruthSynthesizer | Ripple analysis query input | `{query_text}`, `{impact_summary}` |

These prompt files are included in doc 07's authoritative manifest (total: 26 prompt files).
