# 06 — Orchestrator & Workflows

> AgentOrchestrator, ingestion pipeline (with checkpoint resume), query pipeline, WebSocket protocol, in-memory caching
> File locations: `backend/app/agents/orchestrator.py`, `backend/app/agents/query_router.py`, `backend/app/agents/truth_synthesizer.py`, `backend/app/main.py`

---

## 1. AgentOrchestrator

The orchestrator is the central coordination layer with LLM-driven planning capabilities. It owns all agent instances, manages the ingestion and query pipelines, supports iterative agent communication via a shared blackboard, and wires progress callbacks to WebSocket.

### Agentic Orchestration Principles

The orchestrator goes beyond a fixed pipeline:
1. **LLM-Driven Planning** — Before execution, the orchestrator assesses the document set and produces an execution plan (which stages to run, resource allocation, expected complexity).
2. **Agent Communication Protocol** — Agents can signal `needs_clarification` in their output, requesting re-analysis by a specific upstream agent. The orchestrator handles these iterative loops.
3. **Shared Blackboard** — An in-memory shared state where agents publish findings and other agents can query for updates. Enables emergent reasoning across agents.
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
        vector_store,           # VectorStore (pgvector on NeonDB)
    ) -> None:
        self.postgres = postgres_pool
        self.vector_store = vector_store
        self.blackboard = InMemoryBlackboard()
        self.cache = InMemoryCache()

        # Shared resources
        self._llm_semaphore = asyncio.Semaphore(5)  # max 5 concurrent LLM calls across all pipelines
        # Resolve prompt dir relative to project root (one level above backend/)
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        prompt_loader = PromptLoader(prompt_dir=project_root / "prompt")

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
            config=AgentConfig(agent_name="document_parser", llm_role="extraction",
                               max_output_tokens=16000,
                               timeout_seconds=300, verification_threshold=0.80),
            llm_provider=factory.get_for_role("extraction"),
            prompt_loader=prompt_loader,
            vector_store=self.vector_store,
            fallback_provider=factory.get_fallback_for_role("extraction"),
            llm_semaphore=self._llm_semaphore,
        )
        self.agents["amendment_tracker"] = AmendmentTrackerAgent(
            config=AgentConfig(agent_name="amendment_tracker", llm_role="complex_reasoning",
                               max_output_tokens=8192,
                               timeout_seconds=180, verification_threshold=0.75),
            llm_provider=factory.get_for_role("complex_reasoning"),
            prompt_loader=prompt_loader,
            fallback_provider=factory.get_fallback_for_role("complex_reasoning"),
            llm_semaphore=self._llm_semaphore,
        )
        self.agents["temporal_sequencer"] = TemporalSequencerAgent(
            config=AgentConfig(agent_name="temporal_sequencer", llm_role="extraction",
                               max_output_tokens=4096,
                               timeout_seconds=60, verification_threshold=0.80),
            llm_provider=factory.get_for_role("extraction"),
            prompt_loader=prompt_loader,
            db_pool=self.postgres,
            fallback_provider=factory.get_fallback_for_role("extraction"),
            llm_semaphore=self._llm_semaphore,
        )

        # Tier 2
        self.agents["override_resolution"] = OverrideResolutionAgent(
            config=AgentConfig(agent_name="override_resolution", llm_role="complex_reasoning",
                               max_output_tokens=8192,
                               timeout_seconds=180, verification_threshold=0.75),
            llm_provider=factory.get_for_role("complex_reasoning"),
            prompt_loader=prompt_loader,
            fallback_provider=factory.get_fallback_for_role("complex_reasoning"),
            llm_semaphore=self._llm_semaphore,
        )
        self.agents["conflict_detection"] = ConflictDetectionAgent(
            config=AgentConfig(agent_name="conflict_detection", llm_role="complex_reasoning",
                               max_output_tokens=16000,
                               timeout_seconds=300, temperature=0.2, verification_threshold=0.70),
            llm_provider=factory.get_for_role("complex_reasoning"),
            prompt_loader=prompt_loader,
            db_pool=self.postgres,
            fallback_provider=factory.get_fallback_for_role("complex_reasoning"),
            llm_semaphore=self._llm_semaphore,
        )
        self.agents["dependency_mapper"] = DependencyMapperAgent(
            config=AgentConfig(agent_name="dependency_mapper", llm_role="complex_reasoning",
                               max_output_tokens=16000,
                               temperature=0.1, verification_threshold=0.75),
            llm_provider=factory.get_for_role("complex_reasoning"),
            prompt_loader=prompt_loader,
            db_pool=self.postgres,
            fallback_provider=factory.get_fallback_for_role("complex_reasoning"),
            llm_semaphore=self._llm_semaphore,
        )

        # Tier 3
        self.agents["ripple_effect"] = RippleEffectAnalyzerAgent(
            config=AgentConfig(agent_name="ripple_effect", llm_role="complex_reasoning",
                               max_output_tokens=16000,
                               timeout_seconds=300, temperature=0.2, verification_threshold=0.70),
            llm_provider=factory.get_for_role("complex_reasoning"),
            prompt_loader=prompt_loader,
            db_pool=self.postgres,
            fallback_provider=factory.get_fallback_for_role("complex_reasoning"),
            llm_semaphore=self._llm_semaphore,
        )

        # Query pipeline
        self.agents["query_router"] = QueryRouter(
            config=AgentConfig(agent_name="query_router", llm_role="classification",
                               max_output_tokens=1024,
                               verification_threshold=0.85),
            llm_provider=factory.get_for_role("classification"),
            prompt_loader=prompt_loader,
            fallback_provider=factory.get_fallback_for_role("classification"),
            llm_semaphore=self._llm_semaphore,
        )
        self.agents["truth_synthesizer"] = TruthSynthesizer(
            config=AgentConfig(agent_name="truth_synthesizer", llm_role="synthesis",
                               max_output_tokens=8192,
                               temperature=0.1, verification_threshold=0.80),
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
from app.database.db import create_postgres_pool, create_vector_store

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create DB pools + orchestrator. Shutdown: close connections."""
    postgres_pool = await create_postgres_pool()  # NeonDB via EXTERNAL_DATABASE_URL
    vector_store = create_vector_store(postgres_pool)  # pgvector on NeonDB (shared pool)

    # Store pool on app.state for direct use in routes
    app.state.postgres_pool = postgres_pool

    try:
        from app.agents.orchestrator import AgentOrchestrator

        app.state.orchestrator = AgentOrchestrator(
            postgres_pool=postgres_pool,
            vector_store=vector_store,
        )
        logger.info("ContractIQ ready — orchestrator initialised with %d agents",
                    len(app.state.orchestrator.agents))
    except Exception as exc:
        logger.warning("Orchestrator init failed (missing API keys?): %s — server running in limited mode", exc)
        app.state.orchestrator = None
    yield

    # Shutdown: close database connections
    await postgres_pool.close()

app = FastAPI(lifespan=lifespan)
```

### In-Memory Blackboard (Agent Communication)

```python
class InMemoryBlackboard:
    """In-memory blackboard for inter-agent communication.

    Replaces the previous Redis-backed AgentBlackboard. Uses a simple dict
    internally, keyed by stack_id + entry_type. Suitable for single-process
    deployment (no Redis dependency).
    """

    def __init__(self):
        self._entries: dict[str, list[dict]] = {}

    async def publish(self, stack_id: UUID, agent_name: str, entry_type: str, data: dict) -> None:
        key = f"blackboard:{stack_id}:{entry_type}"
        entry = {"agent": agent_name, "type": entry_type, "data": data, "timestamp": datetime.utcnow().isoformat()}
        self._entries.setdefault(key, []).append(entry)

    async def query(self, stack_id: UUID, entry_type: str) -> list[dict]:
        key = f"blackboard:{stack_id}:{entry_type}"
        return self._entries.get(key, [])

    async def clear(self, stack_id: UUID) -> None:
        prefix = f"blackboard:{stack_id}:"
        keys_to_remove = [k for k in self._entries if k.startswith(prefix)]
        for k in keys_to_remove:
            del self._entries[k]

# Entry types for blackboard communication:
# - "buried_change": AmendmentTracker publishes when it finds a buried change
# - "stale_reference": ConflictDetection publishes stale reference findings
# - "high_risk_dependency": DependencyMapper publishes critical dependency paths
# - "low_confidence_clause": OverrideResolution flags uncertain resolutions
# - "amendment_context": AmendmentTracker publishes amendment type/rationale per section
```

### In-Memory Cache (Query Result Caching)

```python
class InMemoryCache:
    """In-memory cache with TTL tracking (replaces Redis cache).

    Uses a dict with expiry timestamps for TTL support, and a separate dict
    for SET-based key tracking (used for per-stack cache invalidation).
    """

    def __init__(self):
        self._store: dict[str, tuple[str, float]] = {}  # key -> (value, expiry_time)
        self._sets: dict[str, set[str]] = {}

    async def get(self, key: str) -> Optional[str]:
        item = self._store.get(key)
        if item is None:
            return None
        value, expiry = item
        if time.time() > expiry:
            del self._store[key]
            return None
        return value

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = (value, time.time() + ttl)

    async def sadd(self, key: str, member: str) -> None:
        self._sets.setdefault(key, set()).add(member)

    async def smembers(self, key: str) -> set[str]:
        return self._sets.get(key, set())

    async def delete(self, *keys: str) -> None:
        for key in keys:
            self._store.pop(key, None)
            self._sets.pop(key, None)

    async def invalidate_for_stack(self, stack_id: UUID) -> None:
        tracking_key = f"cache_keys:{stack_id}"
        keys = await self.smembers(tracking_key)
        for k in keys:
            self._store.pop(k, None)
        self._sets.pop(tracking_key, None)
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
        """Full 6-stage ingestion pipeline with checkpoint resume.

        Each stage checks the database for existing results before running.
        If prior results exist (checkpoint), the stage is skipped and results
        are loaded from the DB. This allows the pipeline to resume from the
        last completed stage after a crash or restart.
        """

        # Invalidate any cached query results for this stack before re-processing
        await self.cache.invalidate_for_stack(contract_stack_id)

        # ── Create TraceContext for this pipeline run ─────────
        trace = TraceContext(job_id=job_id)
        for agent in self.agents.values():
            agent.trace = trace

        # Update stack status to processing
        await self.postgres.execute(
            "UPDATE contract_stacks SET processing_status = 'processing', status_updated_at = NOW() WHERE id = $1",
            contract_stack_id,
        )

        documents = await self._get_documents(contract_stack_id)

        # ── Stage 1: Parse all documents (parallel) ──────────────
        stage1_done = await self._check_stage_complete(contract_stack_id, "document_parsing")
        if stage1_done:
            logger.info("Stage 1 (document_parsing) — checkpoint found, skipping")
            await progress_callback(PipelineProgressEvent(
                job_id=job_id, pipeline_stage="document_parsing", overall_percent=30,
                message=f"Skipped parsing — {len(documents)} documents already parsed (checkpoint)",
                current_agent="document_parser", timestamp=datetime.utcnow(),
            ))
            parsed_outputs = await self._load_parsed_outputs_from_db(contract_stack_id, documents)
        else:
            await progress_callback(PipelineProgressEvent(
                job_id=job_id, pipeline_stage="document_parsing", overall_percent=0,
                message=f"Parsing {len(documents)} documents...",
                current_agent="document_parser", timestamp=datetime.utcnow(),
            ))

            parser = self.get_agent("document_parser")
            parse_tasks = [
                parser.run(DocumentParseInput(
                    document_id=doc["id"],
                    file_path=doc["file_path"],
                    document_type=DocumentType(doc["document_type"]),
                    contract_stack_id=contract_stack_id,
                ))
                for doc in documents
            ]
            parse_results = await asyncio.gather(*parse_tasks, return_exceptions=True)

            parsed_outputs: list[DocumentParseOutput] = []
            for i, result in enumerate(parse_results):
                if isinstance(result, Exception):
                    raise PipelineError(
                        f"Document parsing failed for {documents[i]['file_path']}: {result}",
                        stage="document_parsing",
                    )
                parsed_outputs.append(result)

            await progress_callback(PipelineProgressEvent(
                job_id=job_id, pipeline_stage="document_parsing", overall_percent=30,
                message=f"Parsed {len(parsed_outputs)} documents",
                current_agent="document_parser", timestamp=datetime.utcnow(),
            ))
            await self._save_parsed_documents(contract_stack_id, parsed_outputs)

        # ── Stage 2: Track amendments (sequential) ───────────────
        cta_output = next((p for p in parsed_outputs if p.metadata and p.metadata.document_type == DocumentType.CTA), None)
        if not cta_output:
            raise PipelineError("No CTA document found in parsed outputs", stage="amendment_tracking")

        stage2_done = await self._check_stage_complete(contract_stack_id, "amendment_tracking")
        if stage2_done:
            logger.info("Stage 2 (amendment_tracking) — checkpoint found, skipping")
            tracking_results = await self._load_tracking_from_db(contract_stack_id)
            await progress_callback(PipelineProgressEvent(
                job_id=job_id, pipeline_stage="amendment_tracking", overall_percent=50,
                message=f"Skipped tracking — {len(tracking_results)} amendments already tracked (checkpoint)",
                current_agent="amendment_tracker", timestamp=datetime.utcnow(),
            ))
        else:
            tracker = self.get_agent("amendment_tracker")
            amendment_outputs = [p for p in parsed_outputs if p.metadata and p.metadata.document_type == DocumentType.AMENDMENT]
            amendment_outputs.sort(key=lambda p: p.metadata.effective_date or "")

            tracking_results: list[AmendmentTrackOutput] = []
            for i, amend_output in enumerate(amendment_outputs):
                await progress_callback(PipelineProgressEvent(
                    job_id=job_id, pipeline_stage="amendment_tracking",
                    overall_percent=30 + int(20 * (i / max(len(amendment_outputs), 1))),
                    message=f"Tracking Amendment {i+1} of {len(amendment_outputs)}...",
                    current_agent="amendment_tracker", timestamp=datetime.utcnow(),
                ))
                track_result = await tracker.run(AmendmentTrackInput(
                    amendment_document_id=amend_output.document_id,
                    amendment_number=amend_output.metadata.amendment_number or (i + 1),
                    amendment_text="",
                    amendment_sections=amend_output.sections,
                    amendment_tables=amend_output.tables,
                    original_sections=cta_output.sections,
                    original_tables=cta_output.tables,
                    prior_amendments=tracking_results,
                ))
                tracking_results.append(track_result)

            await self._save_amendment_tracking(contract_stack_id, tracking_results)
            await progress_callback(PipelineProgressEvent(
                job_id=job_id, pipeline_stage="amendment_tracking", overall_percent=50,
                message=f"Tracked {len(tracking_results)} amendments",
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
                    document_version=(
                        f"Amendment {p.metadata.amendment_number}" if p.metadata.amendment_number else "Original CTA"
                    ),
                    filename="",
                )
                for p in parsed_outputs if p.metadata
            ],
        ))
        await progress_callback(PipelineProgressEvent(
            job_id=job_id, pipeline_stage="temporal_sequencing", overall_percent=55,
            message="Documents sequenced", current_agent="temporal_sequencer", timestamp=datetime.utcnow(),
        ))

        doc_label_map = {event.document_id: event.label for event in sequence_result.timeline}

        # ── Stage 4: Override resolution (parallel per section, fault-tolerant) ──
        stage4_done = await self._check_stage_complete(contract_stack_id, "override_resolution")
        if stage4_done:
            logger.info("Stage 4 (override_resolution) — checkpoint found, skipping")
            current_clauses = await self._load_resolved_clauses_from_db(contract_stack_id)

            # Ensure resolved-clause embeddings exist (may be missing if prior run
            # crashed after saving clauses but before embedding)
            if not await self.vector_store.has_resolved_embeddings(contract_stack_id):
                logger.info("Resolved embeddings missing for stack %s — re-embedding", contract_stack_id)
                await self._embed_resolved_clauses(contract_stack_id, current_clauses)

            await progress_callback(PipelineProgressEvent(
                job_id=job_id, pipeline_stage="override_resolution", overall_percent=80,
                message=f"Skipped resolution — {len(current_clauses)} clauses already resolved (checkpoint)",
                current_agent="override_resolution", timestamp=datetime.utcnow(),
            ))
        else:
            resolver = self.get_agent("override_resolution")
            sections_to_resolve = self._build_section_amendment_map(
                contract_stack_id, cta_output, tracking_results, parsed_outputs
            )

            resolve_tasks = [resolver.run(ri) for ri in sections_to_resolve]
            resolve_results = await asyncio.gather(*resolve_tasks, return_exceptions=True)

            # Fault-tolerant: failed sections are logged and skipped rather than aborting
            resolved_clauses: list[OverrideResolutionOutput] = []
            failed_sections = []
            for i, result in enumerate(resolve_results):
                if isinstance(result, Exception):
                    sect = sections_to_resolve[i].section_number
                    logger.error("Override resolution failed for section %s: %s", sect, result)
                    failed_sections.append(sect)
                    continue
                resolved_clauses.append(result)
            if failed_sections:
                logger.warning("Skipped %d failed sections in override resolution: %s", len(failed_sections), failed_sections)
            if not resolved_clauses:
                raise PipelineError("Override resolution failed for ALL sections", stage="override_resolution")

            await self._save_resolved_clauses(contract_stack_id, resolved_clauses)
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

            # Embed resolved current-truth clauses for query-time semantic search
            await self._embed_resolved_clauses(contract_stack_id, current_clauses)

            await progress_callback(PipelineProgressEvent(
                job_id=job_id, pipeline_stage="override_resolution", overall_percent=80,
                message=f"Resolved {len(resolved_clauses)} clause versions",
                current_agent="override_resolution", timestamp=datetime.utcnow(),
            ))

        # ── Stage 5: Dependency mapping ──────────────────────────
        cached_dependencies: list[ClauseDependency] = []
        stage5_done = await self._check_stage_complete(contract_stack_id, "dependency_mapping")
        if stage5_done:
            logger.info("Stage 5 (dependency_mapping) — checkpoint found, skipping")
            dep_rows = await self.postgres.fetch(
                "SELECT c1.section_number AS from_section, c2.section_number AS to_section, "
                "cd.relationship_type, cd.description, cd.confidence "
                "FROM clause_dependencies cd "
                "JOIN clauses c1 ON c1.id = cd.from_clause_id "
                "JOIN clauses c2 ON c2.id = cd.to_clause_id "
                "WHERE cd.contract_stack_id = $1", contract_stack_id,
            )
            cached_dependencies = [
                ClauseDependency(
                    from_section=r["from_section"], to_section=r["to_section"],
                    relationship_type=r["relationship_type"] or "references",
                    description=r["description"] or "", confidence=r["confidence"] or 0.8,
                )
                for r in dep_rows
            ]
            await progress_callback(PipelineProgressEvent(
                job_id=job_id, pipeline_stage="dependency_mapping", overall_percent=90,
                message=f"Skipped mapping — {len(cached_dependencies)} dependencies already mapped (checkpoint)",
                current_agent="dependency_mapper", timestamp=datetime.utcnow(),
            ))
        else:
            mapper = self.get_agent("dependency_mapper")
            dep_result = await mapper.run(DependencyMapInput(
                contract_stack_id=contract_stack_id, current_clauses=current_clauses,
            ))
            cached_dependencies = dep_result.dependencies
            await progress_callback(PipelineProgressEvent(
                job_id=job_id, pipeline_stage="dependency_mapping", overall_percent=90,
                message=f"Mapped {dep_result.total_edges} dependencies",
                current_agent="dependency_mapper", timestamp=datetime.utcnow(),
            ))

        # ── Stage 6: Conflict detection ──────────────────────────
        stage6_done = await self._check_stage_complete(contract_stack_id, "conflict_detection")
        if stage6_done:
            logger.info("Stage 6 (conflict_detection) — checkpoint found, skipping")
            conflict_count = await self.postgres.fetchval(
                "SELECT COUNT(*) FROM conflicts WHERE contract_stack_id = $1", contract_stack_id,
            )
            await progress_callback(PipelineProgressEvent(
                job_id=job_id, pipeline_stage="complete", overall_percent=100,
                message=f"Skipped detection — {conflict_count} conflicts already detected (checkpoint). Pipeline complete.",
                current_agent=None, timestamp=datetime.utcnow(),
            ))
        else:
            detector = self.get_agent("conflict_detection")
            context = await self._build_contract_context(contract_stack_id)

            conflict_result = await detector.run(ConflictDetectionInput(
                contract_stack_id=contract_stack_id,
                current_clauses=current_clauses,
                contract_stack_context=context,
                dependency_graph=cached_dependencies,
            ))
            await self._save_conflicts(contract_stack_id, conflict_result.conflicts)

            await progress_callback(PipelineProgressEvent(
                job_id=job_id, pipeline_stage="complete", overall_percent=100,
                message=f"Pipeline complete: {len(current_clauses)} clauses, {len(conflict_result.conflicts)} conflicts",
                current_agent=None, timestamp=datetime.utcnow(),
            ))

        # Save trace and update stack status
        await self._save_trace(contract_stack_id, trace)
        await self.postgres.execute(
            "UPDATE contract_stacks SET processing_status = 'completed', status_updated_at = NOW() WHERE id = $1",
            contract_stack_id,
        )

        clause_count = await self.postgres.fetchval(
            "SELECT COUNT(*) FROM clauses WHERE contract_stack_id = $1 AND is_current = TRUE", contract_stack_id,
        )
        conflict_count = await self.postgres.fetchval(
            "SELECT COUNT(*) FROM conflicts WHERE contract_stack_id = $1", contract_stack_id,
        )
        dep_count = await self.postgres.fetchval(
            "SELECT COUNT(*) FROM clause_dependencies WHERE contract_stack_id = $1", contract_stack_id,
        )

        return {
            "clauses_processed": clause_count,
            "conflicts_detected": conflict_count,
            "dependencies_mapped": dep_count,
            "trace": {"total_input_tokens": trace.total_input_tokens, "total_output_tokens": trace.total_output_tokens,
                       "total_llm_calls": len(trace.llm_calls)},
        }
```

### Resolved Clause Embedding

After Stage 4, the orchestrator embeds the resolved current-truth clauses into pgvector via `_embed_resolved_clauses()`. This ensures query-time semantic search always returns the post-resolution "truth" versions rather than superseded clause text from individual documents.

```python
    async def _embed_resolved_clauses(self, stack_id: UUID, current_clauses: list[CurrentClause]) -> None:
        """Embed resolved current-truth clauses into pgvector for query-time semantic search."""
        clause_dicts = [
            {
                "section_number": c.section_number,
                "section_title": c.section_title,
                "current_text": c.current_text,
                "clause_category": c.clause_category,
                "source_document_id": c.source_document_id,
                "effective_date": c.effective_date,
            }
            for c in current_clauses
        ]
        count = await self.vector_store.upsert_resolved_clauses(stack_id, clause_dicts)
        logger.info("Embedded %d resolved clauses for query search (stack %s)", count, stack_id)
```

### Background Job Processing (asyncio)

The actual implementation uses `asyncio.create_task()` for background processing instead of Celery workers. The FastAPI route handler creates an async background task that runs the pipeline within the same process, using the shared `AgentOrchestrator` instance from `app.state`. Job status is tracked in the PostgreSQL `contract_stacks` table (`processing_status` column) and via WebSocket progress events.

```python
# backend/app/api/routes.py (excerpt)

@router.post("/api/v1/contract-stacks/{stack_id}/process")
async def start_processing(stack_id: UUID, request: Request):
    """Kick off the ingestion pipeline as a background task."""
    orchestrator = request.app.state.orchestrator
    job_id = str(uuid4())

    async def progress_callback(event: PipelineProgressEvent):
        # Forward progress events to WebSocket subscribers
        await ws_manager.broadcast(job_id, event.model_dump_json())

    # Run pipeline in background — no Celery, no Redis
    asyncio.create_task(
        orchestrator.process_contract_stack(stack_id, job_id, progress_callback)
    )

    return {"job_id": job_id, "status": "processing"}
```

### State Tracking via PostgreSQL

Job state is persisted in the `contract_stacks` table rather than Redis keys. The pipeline updates `processing_status` at the start ("processing") and end ("completed") of each run. Checkpoint resume is handled by querying the database for existing stage results (see `_check_stage_complete()` above).

```
contract_stacks.processing_status → "pending" | "processing" | "completed" | "failed"
contract_stacks.status_updated_at → timestamp of last status change
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
        # Step 1: Check in-memory cache
        cache_key = f"query:{contract_stack_id}:{hashlib.sha256(query_text.encode()).hexdigest()}"
        cached = await self.cache.get(cache_key)
        if cached:
            return TruthSynthesisOutput.model_validate_json(cached)

        # Step 2: Classify query
        router = self.get_agent("query_router")
        route_result: QueryRouteOutput = await router.run(
            QueryRouteInput(query_text=query_text, contract_stack_id=contract_stack_id)
        )

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
        await self.cache.setex(cache_key, 3600, answer.model_dump_json())
        await self.cache.sadd(f"cache_keys:{contract_stack_id}", cache_key)

        return answer

    async def _retrieve_clauses(
        self, query_text: str, stack_id: UUID, route: QueryRouteOutput
    ) -> list[CurrentClause]:
        """Multi-source clause retrieval: pgvector + PostgreSQL (batch query)."""

        # pgvector semantic search — queries resolved clause embeddings
        similar = await self.vector_store.query_similar(query_text, stack_id, n_results=10)
        section_numbers = set()
        for row in similar:
            sn = row.get("section_number")
            if sn:
                section_numbers.add(sn)
        for entity in route.extracted_entities:
            section_numbers.add(entity)

        section_numbers.discard(None)
        clauses = []
        if section_numbers:
            # JOIN with documents to get source document label for citations
            rows = await self.postgres.fetch(
                "SELECT c.section_number, c.section_title, c.current_text, c.clause_category, "
                "c.source_document_id, c.effective_date, d.filename AS source_filename "
                "FROM clauses c LEFT JOIN documents d ON c.source_document_id = d.id "
                "WHERE c.section_number = ANY($1) AND c.contract_stack_id = $2 AND c.is_current = TRUE",
                list(section_numbers), stack_id,
            )
            for row in rows:
                clauses.append(CurrentClause(
                    section_number=row["section_number"],
                    section_title=row["section_title"] or "",
                    current_text=row["current_text"] or "",
                    clause_category=row["clause_category"] or "general",
                    source_document_id=row["source_document_id"],
                    source_document_label=row["source_filename"] or "",
                    effective_date=row["effective_date"],
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
        self, query_text: str, ripple_result
    ) -> TruthSynthesisOutput:
        """Convert RippleEffectOutput into TruthSynthesisOutput format for API consistency."""
        synthesizer = self.get_agent("truth_synthesizer")
        impact_summary = json.dumps(ripple_result.model_dump(mode="json"), indent=2)

        system_prompt = synthesizer.prompts.get("truth_synthesizer_answer")
        user_prompt = synthesizer.prompts.get(
            "truth_synthesizer_ripple_input",
            query_text=query_text,
            impact_summary=impact_summary,
        )
        result = await synthesizer.call_llm(system_prompt, user_prompt)

        # Sanitize LLM-returned source citations (validate UUID, date fields)
        raw_sources = result.get("sources", [])
        sanitized_sources = [SourceCitation(**self._sanitize_source_citation(s)) for s in raw_sources]
        return TruthSynthesisOutput(
            answer=result.get("answer", "Unable to synthesize answer from ripple analysis."),
            sources=sanitized_sources,
            confidence=result.get("confidence", 0.8),
            caveats=result.get("caveats", []),
        )

    async def _build_contract_context(self, contract_stack_id: UUID) -> ContractStackContext:
        """Load contract stack metadata from PostgreSQL for conflict detection context."""
        row = await self.postgres.fetchrow(
            "SELECT study_name, sponsor_name, site_name, therapeutic_area, start_date, end_date "
            "FROM contract_stacks WHERE id = $1",
            contract_stack_id,
        )
        if not row:
            raise PipelineError(f"Contract stack {contract_stack_id} not found", stage="context_building")
        return ContractStackContext(
            study_name=row["study_name"] or "", sponsor_name=row["sponsor_name"] or "",
            site_name=row["site_name"] or "", therapeutic_area=row["therapeutic_area"] or "",
            study_start_date=row.get("start_date"), study_end_date=row.get("end_date"),
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
**LLM:** Azure OpenAI GPT-5.2 (classification role, <2s target)
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
**LLM:** Azure OpenAI GPT-5.2 (synthesis role, comprehensive reasoning)
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
WebSocket Manager (in-process broadcast)
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

The WebSocket endpoint uses an in-process `WebSocketManager` that maintains a dict of `job_id -> set[WebSocket]` connections. The `progress_callback` passed to `process_contract_stack()` calls `ws_manager.broadcast(job_id, event_json)` which sends the event to all connected clients for that job. No Redis pub/sub is required.

```python
# backend/app/api/websocket.py

@router.websocket("/ws/jobs/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    ws_manager.connect(job_id, websocket)

    try:
        # Keep connection alive until pipeline completes or client disconnects
        while True:
            # Wait for client messages (ping/pong keepalive)
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(job_id, websocket)
```

---

## 7. Caching Strategy

All caching is handled by `InMemoryCache` (see section 1 above). No Redis is required.

| Cache Key Pattern | TTL | Invalidation |
|-------------------|-----|--------------|
| `query:{stack_id}:{hash}` | 1 hour | On re-processing of contract stack |
| `cache_keys:{stack_id}` | None (tracking set) | Deleted during invalidation |
| `blackboard:{stack_id}:{type}` | Until re-processing | Cleared via `InMemoryBlackboard.clear()` |

**Cache invalidation** uses SET-based tracking within `InMemoryCache`. Each time a cache key is written via `setex()`, it is also tracked via `sadd(f"cache_keys:{stack_id}", cache_key)`. On invalidation, `invalidate_for_stack()` reads the set members and deletes all associated cache entries.

```python
# Called at the start of process_contract_stack()
await self.cache.invalidate_for_stack(contract_stack_id)
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

**To enable per-agent progress in implementation:** Create an adapter that bridges the agent-level callback `(stage, percent, message)` to a `PipelineProgressEvent` and forwards it via the WebSocket manager. This adapter would be passed to each agent at the start of each pipeline stage:

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
