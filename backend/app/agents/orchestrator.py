"""AgentOrchestrator — central coordination for all ContractIQ agents."""

import asyncio
import hashlib
import json
import logging
import random
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional
from uuid import UUID

from app.agents.base import BaseAgent, TraceContext
from app.agents.config import AgentConfig
from app.agents.llm_providers import LLMProviderFactory
from app.agents.prompt_loader import PromptLoader
from app.agents.document_parser import DocumentParserAgent
from app.agents.amendment_tracker import AmendmentTrackerAgent
from app.agents.temporal_sequencer import TemporalSequencerAgent
from app.agents.override_resolution import OverrideResolutionAgent
from app.agents.conflict_detection import ConflictDetectionAgent
from app.agents.dependency_mapper import DependencyMapperAgent
from app.agents.ripple_effect import RippleEffectAnalyzerAgent
from app.agents.query_router import QueryRouter
from app.agents.truth_synthesizer import TruthSynthesizer
from app.agents.contract_consolidator import ContractConsolidatorAgent
from app.exceptions import PipelineError
from app.models.agent_schemas import (
    AmendmentTrackInput, AmendmentTrackOutput, ClauseDependency,
    ConflictDetectionInput, ContractStackContext,
    CurrentClause, DependencyMapInput,
    DocumentParseInput, DocumentParseOutput, DocumentSummary,
    OverrideResolutionInput, OverrideResolutionOutput,
    ProposedChange, QueryRouteInput, QueryRouteOutput,
    RippleEffectInput, SourceCitation,
    TemporalSequenceInput, TruthSynthesisInput, TruthSynthesisOutput,
)
from app.models.enums import DocumentType, QueryType
from app.models.events import PipelineProgressEvent

logger = logging.getLogger(__name__)


# ── In-Memory Replacements (no Redis) ──────────────────────────

class InMemoryBlackboard:
    """In-memory blackboard for inter-agent communication."""

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


class PgCache:
    """PostgreSQL-backed cache persisted in NeonDB (survives restarts indefinitely)."""

    _TABLE_ENSURED = False

    def __init__(self, pool) -> None:
        self._pool = pool

    async def _ensure_table(self) -> None:
        if PgCache._TABLE_ENSURED:
            return
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_store (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    kind  TEXT NOT NULL DEFAULT 'kv'
                )
            """)
        PgCache._TABLE_ENSURED = True
        logger.info("PgCache table ensured")

    async def get(self, key: str) -> Optional[str]:
        await self._ensure_table()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value FROM cache_store WHERE key = $1 AND kind = 'kv'", key,
            )
        return row["value"] if row else None

    async def setex(self, key: str, ttl: int, value: str) -> None:
        await self._ensure_table()
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO cache_store (key, value, kind) VALUES ($1, $2, 'kv') "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                key, value,
            )

    async def sadd(self, key: str, member: str) -> None:
        await self._ensure_table()
        set_key = f"set:{key}:{member}"
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO cache_store (key, value, kind) VALUES ($1, $2, 'set') "
                "ON CONFLICT (key) DO NOTHING",
                set_key, key,
            )

    async def smembers(self, key: str) -> set[str]:
        await self._ensure_table()
        prefix = f"set:{key}:"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key FROM cache_store WHERE key LIKE $1 AND kind = 'set'",
                prefix + "%",
            )
        return {r["key"][len(prefix):] for r in rows}

    async def delete(self, *keys: str) -> None:
        await self._ensure_table()
        async with self._pool.acquire() as conn:
            for key in keys:
                await conn.execute("DELETE FROM cache_store WHERE key = $1", key)
                await conn.execute("DELETE FROM cache_store WHERE key LIKE $1", f"set:{key}:%")

    async def invalidate_for_stack(self, stack_id: UUID) -> None:
        members = await self.smembers(f"cache_keys:{stack_id}")
        async with self._pool.acquire() as conn:
            for k in members:
                await conn.execute("DELETE FROM cache_store WHERE key = $1", k)
            await conn.execute(
                "DELETE FROM cache_store WHERE key LIKE $1",
                f"set:cache_keys:{stack_id}:%",
            )


class AgentOrchestrator:
    """Central coordinator for all ContractIQ agents."""

    def __init__(self, postgres_pool, vector_store) -> None:
        self.postgres = postgres_pool
        self.vector_store = vector_store
        self.blackboard = InMemoryBlackboard()
        self.cache = PgCache(postgres_pool)

        self._llm_semaphore = asyncio.Semaphore(5)
        # Resolve prompt dir relative to project root (one level above backend/)
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        prompt_loader = PromptLoader(prompt_dir=project_root / "prompt")

        self.agents: dict[str, BaseAgent] = {}
        self._init_agents(prompt_loader)

    def _init_agents(self, prompt_loader: PromptLoader) -> None:
        factory = LLMProviderFactory

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

        # Consolidation
        self.agents["contract_consolidator"] = ContractConsolidatorAgent(
            config=AgentConfig(agent_name="contract_consolidator", llm_role="synthesis",
                               max_output_tokens=16000,
                               temperature=0.1, verification_threshold=0.75),
            llm_provider=factory.get_for_role("synthesis"),
            prompt_loader=prompt_loader,
            fallback_provider=factory.get_fallback_for_role("synthesis"),
            llm_semaphore=self._llm_semaphore,
        )

    def get_agent(self, name: str) -> BaseAgent:
        return self.agents[name]

    # ── Checkpoint Helpers ──────────────────────────────────────

    async def _check_stage_complete(self, stack_id: UUID, stage: str) -> bool:
        """Check if a pipeline stage already has results in the DB."""
        if stage == "document_parsing":
            count = await self.postgres.fetchval(
                "SELECT COUNT(*) FROM documents WHERE contract_stack_id = $1 AND processed = TRUE", stack_id,
            )
            total = await self.postgres.fetchval(
                "SELECT COUNT(*) FROM documents WHERE contract_stack_id = $1", stack_id,
            )
            return count > 0 and count == total
        elif stage == "amendment_tracking":
            return await self.postgres.fetchval(
                "SELECT COUNT(*) FROM amendments WHERE contract_stack_id = $1", stack_id,
            ) > 0
        elif stage == "temporal_sequencing":
            return await self.postgres.fetchval(
                "SELECT COUNT(*) FROM document_supersessions WHERE contract_stack_id = $1", stack_id,
            ) > 0
        elif stage == "override_resolution":
            return await self.postgres.fetchval(
                "SELECT COUNT(*) FROM clauses WHERE contract_stack_id = $1 AND is_current = TRUE", stack_id,
            ) > 0
        elif stage == "dependency_mapping":
            return await self.postgres.fetchval(
                "SELECT COUNT(*) FROM clause_dependencies WHERE contract_stack_id = $1", stack_id,
            ) > 0
        elif stage == "conflict_detection":
            return await self.postgres.fetchval(
                "SELECT COUNT(*) FROM conflicts WHERE contract_stack_id = $1", stack_id,
            ) > 0
        return False

    async def _load_parsed_outputs_from_db(self, stack_id: UUID, documents: list[dict]) -> list[DocumentParseOutput]:
        """Reconstruct DocumentParseOutput objects from DB for checkpoint resume."""
        from app.models.agent_schemas import DocumentMetadata, ParsedSection
        outputs = []
        for doc in documents:
            meta_raw = await self.postgres.fetchval(
                "SELECT metadata FROM documents WHERE id = $1", doc["id"],
            )
            metadata = None
            if meta_raw:
                meta_dict = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
                metadata = DocumentMetadata(**meta_dict)
            # Reconstruct sections from clauses for this document
            rows = await self.postgres.fetch(
                "SELECT section_number, section_title, clause_text, clause_category "
                "FROM clauses WHERE source_document_id = $1 AND contract_stack_id = $2",
                doc["id"], stack_id,
            )
            sections = []
            if rows:
                for r in rows:
                    sections.append(ParsedSection(
                        section_number=r["section_number"] or "",
                        section_title=r["section_title"] or "",
                        text=r["clause_text"] or "",
                        clause_category=r["clause_category"] or "general",
                    ))
            else:
                # If no clauses yet (stages 1-2 done but not 4), use pgvector as fallback
                embedding_rows = await self.vector_store.get_by_document(doc["id"])
                for row in embedding_rows:
                    sections.append(ParsedSection(
                        section_number=row["section_number"],
                        section_title=row.get("section_title", ""),
                        text=row.get("section_text", ""),
                        clause_category="general",
                    ))
            outputs.append(DocumentParseOutput(
                document_id=doc["id"],
                metadata=metadata,
                sections=sections,
                tables=[],
                raw_text="",
                page_count=0,
            ))
        return outputs

    async def _load_tracking_from_db(self, stack_id: UUID) -> list[AmendmentTrackOutput]:
        """Load amendment tracking results from DB."""
        from app.models.agent_schemas import Modification
        rows = await self.postgres.fetch(
            "SELECT document_id, amendment_number, amendment_type, sections_modified, "
            "rationale, modifications FROM amendments WHERE contract_stack_id = $1 "
            "ORDER BY amendment_number",
            stack_id,
        )
        results = []
        for r in rows:
            mods_raw = json.loads(r["modifications"]) if r["modifications"] else []
            mods = []
            for m in mods_raw:
                try:
                    mods.append(Modification(**m))
                except Exception:
                    continue
            results.append(AmendmentTrackOutput(
                amendment_document_id=r["document_id"],
                amendment_number=r["amendment_number"] or 0,
                amendment_type=r["amendment_type"] or "unknown",
                rationale=r["rationale"] or "",
                modifications=mods,
                sections_modified=r["sections_modified"] or [],
            ))
        return results

    async def _load_resolved_clauses_from_db(self, stack_id: UUID) -> list[CurrentClause]:
        """Load current clauses from DB for downstream stages."""
        rows = await self.postgres.fetch(
            "SELECT c.section_number, c.section_title, c.current_text, c.clause_category, "
            "c.source_document_id, c.effective_date, d.filename AS source_filename "
            "FROM clauses c LEFT JOIN documents d ON c.source_document_id = d.id "
            "WHERE c.contract_stack_id = $1 AND c.is_current = TRUE",
            stack_id,
        )
        return [
            CurrentClause(
                section_number=r["section_number"],
                section_title=r["section_title"] or "",
                current_text=r["current_text"] or "",
                clause_category=r["clause_category"] or "general",
                source_document_id=r["source_document_id"],
                source_document_label=r["source_filename"] or "",
                effective_date=r["effective_date"],
            )
            for r in rows
        ]

    # ── Ingestion Pipeline ─────────────────────────────────────

    async def process_contract_stack(
        self,
        contract_stack_id: UUID,
        job_id: str,
        progress_callback: Callable[[PipelineProgressEvent], Awaitable[None]],
    ) -> dict:
        """Full 6-stage ingestion pipeline with checkpoint resume."""
        await self.cache.invalidate_for_stack(contract_stack_id)
        trace = TraceContext(job_id=job_id)
        for agent in self.agents.values():
            agent.trace = trace

        # Update stack status to processing
        await self.postgres.execute(
            "UPDATE contract_stacks SET processing_status = 'processing', status_updated_at = NOW() WHERE id = $1",
            contract_stack_id,
        )

        documents = await self._get_documents(contract_stack_id)

        # ── Stage 1: Parse all documents (parallel) ──────────
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

        # ── Stage 2: Track amendments (sequential) ───────────
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

        # ── Stage 3: Temporal sequencing ─────────────────────
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

        # ── Stage 4: Override resolution (parallel per section) ──
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
            sections_to_resolve = self._build_section_amendment_map(contract_stack_id, cta_output, tracking_results, parsed_outputs)

            resolve_tasks = [resolver.run(ri) for ri in sections_to_resolve]
            resolve_results = await asyncio.gather(*resolve_tasks, return_exceptions=True)

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

        # ── Stage 5: Dependency mapping ──────────────────────
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

        # ── Stage 6: Conflict detection ──────────────────────
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

    # ── Query Pipeline ─────────────────────────────────────────

    async def handle_query(self, query_text: str, contract_stack_id: UUID) -> TruthSynthesisOutput:
        cache_key = f"query:{contract_stack_id}:{hashlib.sha256(query_text.encode()).hexdigest()}"
        cached = await self.cache.get(cache_key)
        if cached:
            await asyncio.sleep(random.uniform(2, 3))  # Simulate real-time retrieval
            return TruthSynthesisOutput.model_validate_json(cached)

        router = self.get_agent("query_router")
        route_result: QueryRouteOutput = await router.run(
            QueryRouteInput(query_text=query_text, contract_stack_id=contract_stack_id)
        )

        relevant_clauses = await self._retrieve_clauses(query_text, contract_stack_id, route_result)
        relevant_conflicts = await self._get_relevant_conflicts(contract_stack_id, route_result.extracted_entities)

        if route_result.query_type == QueryType.RIPPLE_ANALYSIS:
            ripple_agent = self.get_agent("ripple_effect")
            proposed_change = await self._extract_proposed_change(query_text, relevant_clauses)
            ripple_result = await ripple_agent.run(RippleEffectInput(
                contract_stack_id=contract_stack_id, proposed_change=proposed_change,
            ))
            answer = await self._synthesize_ripple_answer(query_text, ripple_result)
        else:
            synthesizer = self.get_agent("truth_synthesizer")
            answer = await synthesizer.run(TruthSynthesisInput(
                query_text=query_text, query_type=route_result.query_type,
                contract_stack_id=contract_stack_id,
                relevant_clauses=relevant_clauses, conflicts=relevant_conflicts,
            ))

        await self.cache.setex(cache_key, 86400 * 30, answer.model_dump_json())
        await self.cache.sadd(f"cache_keys:{contract_stack_id}", cache_key)
        return answer

    # ── Helper Methods ─────────────────────────────────────────

    async def _get_documents(self, stack_id: UUID) -> list[dict]:
        rows = await self.postgres.fetch(
            "SELECT id, file_path, document_type, filename FROM documents WHERE contract_stack_id = $1",
            stack_id,
        )
        return [dict(r) for r in rows]

    async def _build_contract_context(self, stack_id: UUID) -> ContractStackContext:
        row = await self.postgres.fetchrow(
            "SELECT study_name, sponsor_name, site_name, therapeutic_area, start_date, end_date FROM contract_stacks WHERE id = $1",
            stack_id,
        )
        if not row:
            raise PipelineError(f"Contract stack {stack_id} not found", stage="context_building")
        return ContractStackContext(
            study_name=row["study_name"] or "", sponsor_name=row["sponsor_name"] or "",
            site_name=row["site_name"] or "", therapeutic_area=row["therapeutic_area"] or "",
            study_start_date=row.get("start_date"), study_end_date=row.get("end_date"),
        )

    def _build_section_amendment_map(self, contract_stack_id, cta_output, tracking_results, parsed_outputs) -> list[OverrideResolutionInput]:
        """Build OverrideResolutionInput for each section with its amendment chain.

        Handles:
        - Exact section number matching (7.2 == 7.2)
        - Parent section matching (mod 7.2 → CTA section 7, when 7.2 not in CTA)
        - New sections added by amendments (ADDITION type not in CTA)
        """
        from app.models.agent_schemas import AmendmentForSection, ParsedSection
        from app.models.enums import ModificationType

        cta_section_numbers = {s.section_number for s in cta_output.sections}
        inputs = []

        # Phase 1: Process existing CTA sections
        for section in cta_output.sections:
            amendments = []
            for track in tracking_results:
                for mod in track.modifications:
                    # Exact match
                    if mod.section_number == section.section_number:
                        amendments.append(AmendmentForSection(
                            amendment_document_id=track.amendment_document_id,
                            amendment_number=track.amendment_number,
                            effective_date=track.effective_date,
                            modification=mod,
                        ))
                    # Parent section match: mod "7.2" → CTA "7" (when "7.2" not in CTA)
                    elif (mod.section_number not in cta_section_numbers
                          and "." in mod.section_number
                          and mod.section_number.rsplit(".", 1)[0] == section.section_number):
                        amendments.append(AmendmentForSection(
                            amendment_document_id=track.amendment_document_id,
                            amendment_number=track.amendment_number,
                            effective_date=track.effective_date,
                            modification=mod,
                        ))
            inputs.append(OverrideResolutionInput(
                contract_stack_id=contract_stack_id,
                section_number=section.section_number,
                original_clause=section,
                original_document_id=cta_output.document_id,
                original_document_label="Original CTA",
                amendments=amendments,
            ))

        # Phase 2: Handle new sections added by amendments (not in CTA)
        seen_new_sections = set()
        for track in tracking_results:
            for mod in track.modifications:
                if (mod.section_number not in cta_section_numbers
                        and mod.section_number not in seen_new_sections):
                    # Check it wasn't matched as a subsection of an existing CTA section
                    parent = mod.section_number.rsplit(".", 1)[0] if "." in mod.section_number else None
                    if parent and parent in cta_section_numbers:
                        continue  # Already handled via parent match in Phase 1
                    seen_new_sections.add(mod.section_number)
                    # Create a new section entry from the amendment's modification text
                    new_section = ParsedSection(
                        section_number=mod.section_number,
                        section_title=mod.change_description or f"New Section {mod.section_number}",
                        text=mod.new_text or "",
                        clause_category="general",
                    )
                    inputs.append(OverrideResolutionInput(
                        contract_stack_id=contract_stack_id,
                        section_number=mod.section_number,
                        original_clause=new_section,
                        original_document_id=track.amendment_document_id,
                        original_document_label=f"Amendment {track.amendment_number}",
                        amendments=[AmendmentForSection(
                            amendment_document_id=track.amendment_document_id,
                            amendment_number=track.amendment_number,
                            effective_date=track.effective_date,
                            modification=mod,
                        )],
                    ))

        logger.info("Built %d override resolution inputs (%d from CTA, %d new from amendments)",
                     len(inputs), len(cta_output.sections), len(seen_new_sections))
        return inputs

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

    async def _retrieve_clauses(self, query_text, stack_id, route_result) -> list[CurrentClause]:
        """Multi-source clause retrieval: pgvector + PostgreSQL (batch query)."""
        similar = await self.vector_store.query_similar(query_text, stack_id, n_results=10)
        section_numbers = set()
        for row in similar:
            sn = row.get("section_number")
            if sn:
                section_numbers.add(sn)
        for entity in route_result.extracted_entities:
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

    async def _get_relevant_conflicts(self, stack_id, entities):
        from app.models.agent_schemas import ConflictEvidence, DetectedConflict
        rows = await self.postgres.fetch(
            "SELECT conflict_id, conflict_type, severity, description, affected_sections, "
            "evidence, recommendation, pain_point_id FROM conflicts WHERE contract_stack_id = $1",
            stack_id,
        )
        conflicts = []
        for r in rows:
            conflicts.append(DetectedConflict(
                conflict_id=r["conflict_id"], conflict_type=r["conflict_type"],
                severity=r["severity"], description=r["description"],
                affected_sections=r["affected_sections"] or [],
                evidence=json.loads(r["evidence"]) if r["evidence"] else [],
                recommendation=r["recommendation"] or "",
                pain_point_id=r["pain_point_id"],
            ))
        return conflicts

    async def _extract_proposed_change(self, query_text, relevant_clauses) -> ProposedChange:
        router = self.get_agent("query_router")
        clauses_context = "\n".join(
            f"Section {c.section_number} ({c.section_title}): {c.current_text[:200]}..."
            for c in relevant_clauses[:10]
        )
        result = await router.call_llm(
            "Extract the proposed change from this query.",
            f"Query: {query_text}\n\nAvailable sections:\n{clauses_context}\n\n"
            f'Return JSON: {{"section_number": "...", "current_text": "...", "proposed_text": "...", "change_description": "..."}}',
        )
        target = next((c for c in relevant_clauses if c.section_number == result.get("section_number")), None)
        return ProposedChange(
            section_number=result.get("section_number", target.section_number if target else "unknown"),
            current_text=target.current_text if target else result.get("current_text", ""),
            proposed_text=result.get("proposed_text", query_text),
            change_description=result.get("change_description", query_text),
        )

    @staticmethod
    def _sanitize_source_citation(s: dict) -> dict:
        """Sanitize LLM-returned source citation fields for SourceCitation pydantic model."""
        cleaned = dict(s)
        # Validate document_id as UUID; set to None if invalid
        doc_id = cleaned.get("document_id")
        if doc_id is not None:
            try:
                UUID(str(doc_id))
            except (ValueError, AttributeError):
                cleaned["document_id"] = None
        # Validate effective_date; set to None if invalid
        eff_date = cleaned.get("effective_date")
        if eff_date is not None and not isinstance(eff_date, date):
            try:
                date.fromisoformat(str(eff_date))
            except (ValueError, TypeError):
                cleaned["effective_date"] = None
        return cleaned

    async def _synthesize_ripple_answer(self, query_text, ripple_result) -> TruthSynthesisOutput:
        synthesizer = self.get_agent("truth_synthesizer")
        impact_summary = json.dumps(ripple_result.model_dump(mode="json"), indent=2)
        system_prompt = synthesizer.prompts.get("truth_synthesizer_answer")
        user_prompt = synthesizer.prompts.get(
            "truth_synthesizer_ripple_input", query_text=query_text, impact_summary=impact_summary,
        )
        result = await synthesizer.call_llm(system_prompt, user_prompt)
        raw_sources = result.get("sources", [])
        sanitized_sources = [SourceCitation(**self._sanitize_source_citation(s)) for s in raw_sources]
        return TruthSynthesisOutput(
            answer=result.get("answer", "Unable to synthesize answer from ripple analysis."),
            sources=sanitized_sources,
            confidence=result.get("confidence", 0.8), caveats=result.get("caveats", []),
        )

    # ── Database Save Helpers ──────────────────────────────────

    async def _save_parsed_documents(self, stack_id, parsed_outputs):
        async with self.postgres.acquire() as conn:
            for p in parsed_outputs:
                await conn.execute(
                    "UPDATE documents SET processed = TRUE, metadata = $2 WHERE id = $1",
                    p.document_id, json.dumps(p.metadata.model_dump(mode="json") if p.metadata else {}),
                )

    async def _save_amendment_tracking(self, stack_id, tracking_results):
        async with self.postgres.acquire() as conn:
            for t in tracking_results:
                mods_json = json.dumps([m.model_dump(mode="json") for m in t.modifications])
                await conn.execute(
                    "INSERT INTO amendments (document_id, contract_stack_id, amendment_number, "
                    "amendment_type, sections_modified, rationale, modifications) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7) "
                    "ON CONFLICT ON CONSTRAINT uq_amendments_stack_doc DO UPDATE SET "
                    "amendment_number = EXCLUDED.amendment_number, "
                    "amendment_type = EXCLUDED.amendment_type, "
                    "sections_modified = EXCLUDED.sections_modified, "
                    "rationale = EXCLUDED.rationale, "
                    "modifications = EXCLUDED.modifications",
                    t.amendment_document_id, stack_id, t.amendment_number,
                    t.amendment_type, t.sections_modified, t.rationale,
                    mods_json,
                )

    async def _save_resolved_clauses(self, stack_id, resolved):
        async with self.postgres.acquire() as conn:
            for r in resolved:
                cv = r.clause_version
                source_chain_json = json.dumps([s.model_dump(mode="json") for s in cv.source_chain])
                await conn.execute(
                    "INSERT INTO clauses (contract_stack_id, section_number, section_title, "
                    "clause_text, current_text, clause_category, is_current, source_document_id, "
                    "effective_date, source_chain) "
                    "VALUES ($1, $2, $3, $4, $5, $6, TRUE, $7, $8, $9) "
                    "ON CONFLICT (contract_stack_id, section_number) WHERE is_current = TRUE DO UPDATE SET "
                    "section_title = EXCLUDED.section_title, "
                    "current_text = EXCLUDED.current_text, clause_category = EXCLUDED.clause_category, "
                    "source_document_id = EXCLUDED.source_document_id, "
                    "effective_date = EXCLUDED.effective_date, source_chain = EXCLUDED.source_chain, "
                    "is_current = TRUE",
                    stack_id, cv.section_number, (cv.section_title or "")[:255],
                    cv.current_text or "", cv.current_text or "", cv.clause_category or "general",
                    cv.last_modified_by, cv.last_modified_date,
                    source_chain_json,
                )

    async def _save_conflicts(self, stack_id, conflicts):
        async with self.postgres.acquire() as conn:
            for c in conflicts:
                evidence_json = json.dumps([e.model_dump(mode="json") for e in c.evidence])
                await conn.execute(
                    "INSERT INTO conflicts (contract_stack_id, conflict_id, conflict_type, "
                    "severity, description, affected_sections, evidence, recommendation, pain_point_id) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) "
                    "ON CONFLICT ON CONSTRAINT uq_conflicts_stack_conflict_id DO UPDATE SET "
                    "conflict_type = EXCLUDED.conflict_type, severity = EXCLUDED.severity, "
                    "description = EXCLUDED.description, affected_sections = EXCLUDED.affected_sections, "
                    "evidence = EXCLUDED.evidence, recommendation = EXCLUDED.recommendation, "
                    "pain_point_id = EXCLUDED.pain_point_id",
                    stack_id, c.conflict_id,
                    c.conflict_type.value if hasattr(c.conflict_type, 'value') else str(c.conflict_type),
                    c.severity.value if hasattr(c.severity, 'value') else str(c.severity),
                    c.description, c.affected_sections,
                    evidence_json,
                    c.recommendation, c.pain_point_id,
                )

    async def _save_trace(self, stack_id, trace: TraceContext):
        async with self.postgres.acquire() as conn:
            await conn.execute(
                "INSERT INTO pipeline_traces (job_id, trace_id, contract_stack_id, "
                "total_input_tokens, total_output_tokens, total_llm_calls, llm_calls) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7)",
                trace.job_id, trace.trace_id, stack_id,
                trace.total_input_tokens, trace.total_output_tokens,
                len(trace.llm_calls),
                json.dumps([{
                    "call_id": c.call_id, "agent": c.agent_name,
                    "model": c.model, "provider": c.provider,
                    "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
                    "latency_ms": c.latency_ms,
                } for c in trace.llm_calls]),
            )
