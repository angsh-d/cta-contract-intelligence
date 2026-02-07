"""Tier 1: TemporalSequencerAgent — LLM-first temporal reasoning and version tree."""

import logging
from datetime import date
from typing import Any, Optional
from uuid import UUID

from app.agents.base import BaseAgent
from app.exceptions import LLMResponseError
from app.models.agent_schemas import (
    DocumentSummary, TemporalSequenceInput, TemporalSequenceOutput,
    TimelineEvent, VersionTree, VersionTreeNode,
)
from app.models.enums import DocumentType

logger = logging.getLogger(__name__)


class TemporalSequencerAgent(BaseAgent):
    """Determine document ordering, supersession, and version tree via LLM reasoning."""

    def __init__(self, config, llm_provider, prompt_loader, db_pool,
                 progress_callback=None, fallback_provider=None,
                 trace_context=None, llm_semaphore=None):
        super().__init__(config, llm_provider, prompt_loader, progress_callback,
                         fallback_provider, trace_context, llm_semaphore)
        self.db = db_pool

    async def process(self, input_data: TemporalSequenceInput) -> TemporalSequenceOutput:
        documents = [doc.model_copy() for doc in input_data.documents]
        dates_inferred: list[UUID] = []

        # Step 1: Infer missing dates and amendment numbers via LLM
        for i, doc in enumerate(documents):
            if doc.effective_date is None or doc.amendment_number is None:
                inferred = await self._infer_metadata(doc)
                updates = {}
                if doc.effective_date is None and inferred.get("effective_date"):
                    updates["effective_date"] = date.fromisoformat(inferred["effective_date"])
                    dates_inferred.append(doc.document_id)
                if doc.amendment_number is None and inferred.get("amendment_number"):
                    updates["amendment_number"] = inferred["amendment_number"]
                if updates:
                    documents[i] = doc.model_copy(update=updates)

        # Step 2: LLM Temporal Reasoning (primary ordering logic)
        system_prompt = self.prompts.get("temporal_sequencer_system")
        user_prompt = self.prompts.get(
            "temporal_sequencer_ordering",
            documents_json=self._format_documents_for_llm(documents),
        )
        llm_result = await self.call_llm(system_prompt, user_prompt)

        llm_order = llm_result.get("chronological_order")
        if not llm_order:
            raise LLMResponseError("TemporalSequencerAgent: LLM response missing 'chronological_order'")
        llm_supersessions = llm_result.get("supersessions", [])
        warnings = llm_result.get("warnings", [])
        reasoning = llm_result.get("reasoning", "")

        chronological_order = [UUID(doc_id) for doc_id in llm_order]

        # Step 3: Deterministic validation (cross-check)
        deterministic_order = sorted(documents, key=lambda d: (
            d.effective_date or date.min,
            0 if d.document_type == DocumentType.CTA else 1,
            d.amendment_number or 999,
        ))
        deterministic_ids = [d.document_id for d in deterministic_order]

        if chronological_order != deterministic_ids:
            divergences = []
            for i, (llm_id, det_id) in enumerate(zip(chronological_order, deterministic_ids)):
                if llm_id != det_id:
                    divergences.append(f"Position {i}: LLM={llm_id}, date-sort={det_id}")
            if divergences:
                warnings.append(
                    f"LLM ordering diverges from date-based sort at {len(divergences)} position(s): "
                    f"{'; '.join(divergences[:3])}. LLM ordering used (may indicate retroactive amendments)."
                )
                logger.warning("Temporal ordering divergence: %s", divergences)

        # Step 4: Build version tree from LLM supersession output
        doc_map = {str(d.document_id): d for d in documents}
        cta = next((d for d in documents if d.document_type == DocumentType.CTA), None)

        version_tree = VersionTree(
            root_document_id=cta.document_id if cta else documents[0].document_id,
            amendments=[
                VersionTreeNode(
                    document_id=UUID(s["successor"]),
                    amendment_number=doc_map[s["successor"]].amendment_number or 0,
                    effective_date=doc_map[s["successor"]].effective_date,
                    supersedes_document_id=UUID(s["predecessor"]),
                    label=f"Amendment {doc_map[s['successor']].amendment_number} ({doc_map[s['successor']].effective_date})",
                )
                for s in llm_supersessions
            ],
        )

        # Timeline
        timeline = [
            TimelineEvent(
                document_id=UUID(doc_id),
                event_date=doc_map[doc_id].effective_date or date.min,
                document_type=doc_map[doc_id].document_type,
                label=doc_map[doc_id].document_version or doc_map[doc_id].filename,
                amendment_number=doc_map[doc_id].amendment_number,
            )
            for doc_id in llm_order
        ]

        # Step 5: Write SUPERSEDES to PostgreSQL
        await self._write_supersedes(input_data.contract_stack_id, version_tree)

        return TemporalSequenceOutput(
            contract_stack_id=input_data.contract_stack_id,
            chronological_order=chronological_order,
            version_tree=version_tree,
            timeline=timeline,
            dates_inferred=dates_inferred,
            llm_reasoning=reasoning,
        )

    async def _infer_metadata(self, doc: DocumentSummary) -> dict:
        """Use LLM to infer effective date and amendment number from document text/filename."""
        system_prompt = self.prompts.get("temporal_sequencer_date_inference")
        user_prompt = (
            f"Filename: {doc.filename}\n"
            f"Document version: {doc.document_version or 'unknown'}\n"
            f"Document type: {doc.document_type.value}"
        )
        return await self.call_llm(system_prompt, user_prompt)

    def _format_documents_for_llm(self, documents: list[DocumentSummary]) -> str:
        parts = []
        for d in documents:
            parts.append(
                f"Document ID: {d.document_id}\n"
                f"  Type: {d.document_type.value}\n"
                f"  Filename: {d.filename}\n"
                f"  Effective Date: {d.effective_date}\n"
                f"  Amendment Number: {d.amendment_number}\n"
                f"  Version: {d.document_version or 'N/A'}"
            )
        return "\n\n".join(parts)

    async def _write_supersedes(self, contract_stack_id: UUID, version_tree: VersionTree) -> None:
        """Write SUPERSEDES relationships atomically, clearing stale rows first."""
        async with self.db.acquire() as conn:
            async with conn.transaction():
                # Clear stale supersession data for this stack before re-inserting
                await conn.execute(
                    "DELETE FROM document_supersessions WHERE contract_stack_id = $1",
                    str(contract_stack_id),
                )
                for node in version_tree.amendments:
                    await conn.execute(
                        """
                        INSERT INTO document_supersessions
                            (contract_stack_id, predecessor_document_id, successor_document_id)
                        VALUES ($1, $2, $3)
                        ON CONFLICT DO NOTHING
                        """,
                        str(contract_stack_id),
                        str(node.supersedes_document_id),
                        str(node.document_id),
                    )
