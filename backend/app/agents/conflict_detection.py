"""Tier 2: ConflictDetectionAgent — multi-pass conflict analysis with tool use."""

import json
import logging
import time
from typing import Any
from uuid import UUID

from app.agents.base import BaseAgent
from app.exceptions import LLMResponseError
from app.models.agent_schemas import (
    ClauseDependency, ConflictDetectionInput, ConflictDetectionOutput,
    ConflictSeveritySummary, CurrentClause, DetectedConflict,
)

logger = logging.getLogger(__name__)


class ConflictDetectionAgent(BaseAgent):
    """Find contradictions, ambiguities, gaps, buried changes, stale references, and temporal mismatches."""

    def __init__(self, config, llm_provider, prompt_loader, db_pool,
                 progress_callback=None, fallback_provider=None,
                 trace_context=None, llm_semaphore=None):
        super().__init__(config, llm_provider, prompt_loader, progress_callback,
                         fallback_provider, trace_context, llm_semaphore)
        self.db = db_pool
        self._current_stack_id: UUID | None = None

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            tool for tool in self.STANDARD_TOOLS
            if tool["name"] in ("get_clause", "get_dependencies", "get_amendment_history")
        ]

    async def _execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        async with self.db.acquire() as conn:
            if tool_name == "get_clause":
                row = await conn.fetchrow(
                    "SELECT section_number, section_title, current_text, clause_category "
                    "FROM clauses WHERE section_number = $1 AND contract_stack_id = $2",
                    tool_input["section_number"], self._current_stack_id,
                )
                return dict(row) if row else {"error": "Clause not found"}
            elif tool_name == "get_dependencies":
                rows = await conn.fetch(
                    """
                    SELECT cd.from_clause_id, cd.to_clause_id, cd.relationship_type, cd.description
                    FROM clause_dependencies cd
                    JOIN clauses c ON c.id = cd.from_clause_id
                    WHERE c.section_number = $1 AND cd.contract_stack_id = $2
                    UNION
                    SELECT cd.from_clause_id, cd.to_clause_id, cd.relationship_type, cd.description
                    FROM clause_dependencies cd
                    JOIN clauses c ON c.id = cd.to_clause_id
                    WHERE c.section_number = $1 AND cd.contract_stack_id = $2
                    """,
                    tool_input["section_number"], self._current_stack_id,
                )
                return [dict(r) for r in rows]
            elif tool_name == "get_amendment_history":
                rows = await conn.fetch(
                    """
                    SELECT a.amendment_number, a.modification_type, a.rationale
                    FROM amendments a
                    JOIN documents d ON d.id = a.document_id
                    JOIN clauses c ON c.source_document_id = d.id
                    WHERE a.contract_stack_id = $1 AND c.section_number = $2
                    ORDER BY a.amendment_number
                    """,
                    self._current_stack_id,
                    tool_input.get("section_number", ""),
                )
                return [dict(r) for r in rows]
        return await super()._execute_tool(tool_name, tool_input)

    async def process(self, input_data: ConflictDetectionInput) -> ConflictDetectionOutput:
        start = time.monotonic()
        self._current_stack_id = input_data.contract_stack_id

        groups = self._group_by_category(input_data.current_clauses)

        system_prompt = self.prompts.get("conflict_detection_system")
        user_prompt = self.prompts.get(
            "conflict_detection_analyze",
            grouped_clauses=self._format_groups(groups),
            context=input_data.contract_stack_context.model_dump_json(exclude_none=True),
            dependency_graph=self._format_dependency_graph(input_data.dependency_graph),
        )
        try:
            result = await self.call_llm(system_prompt, user_prompt)
        except Exception as e:
            logger.warning("ConflictDetection LLM call failed: %s — proceeding with empty conflicts", e)
            result = {}

        conflicts_raw = result.get("conflicts") or []
        conflicts = []
        for c_raw in conflicts_raw:
            try:
                conflicts.append(DetectedConflict(**c_raw))
            except Exception as e:
                logger.warning("Skipping invalid conflict: %s", e)
        latency_ms = int((time.monotonic() - start) * 1000)

        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for c in conflicts:
            sev = c.severity.value if hasattr(c.severity, 'value') else str(c.severity)
            if sev in counts:
                counts[sev] += 1

        return ConflictDetectionOutput(
            contract_stack_id=input_data.contract_stack_id,
            conflicts=conflicts,
            summary=ConflictSeveritySummary(**counts),
            analysis_model=self.config.model_override or "",
            analysis_latency_ms=latency_ms,
            llm_reasoning=result.get("reasoning", ""),
            confidence_factors=self._sanitize_confidence_factors(result.get("confidence_factors", {})),
        )

    @staticmethod
    def _sanitize_confidence_factors(raw: Any) -> dict[str, float]:
        """Coerce confidence_factors to dict[str, float], dropping non-numeric values."""
        if not isinstance(raw, dict):
            return {}
        cleaned = {}
        for k, v in raw.items():
            try:
                cleaned[str(k)] = float(v)
            except (ValueError, TypeError):
                continue
        return cleaned

    def _group_by_category(self, clauses: list[CurrentClause]) -> dict[str, list[CurrentClause]]:
        groups: dict[str, list[CurrentClause]] = {}
        for c in clauses:
            groups.setdefault(c.clause_category, []).append(c)
        return groups

    def _format_groups(self, groups: dict[str, list[CurrentClause]]) -> str:
        parts = []
        for category, clauses in groups.items():
            parts.append(f"=== {category.upper()} ===")
            for c in clauses:
                parts.append(
                    f"Section {c.section_number} ({c.section_title}) "
                    f"[Source: {c.source_document_label}, Effective: {c.effective_date}]:\n"
                    f"{c.current_text}\n"
                )
        return "\n".join(parts)

    def _format_dependency_graph(self, deps: list[ClauseDependency]) -> str:
        if not deps:
            return "(no dependencies detected)"
        lines = []
        for d in deps:
            rt = d.relationship_type.value if hasattr(d.relationship_type, 'value') else d.relationship_type
            lines.append(
                f"{d.from_section} --[{rt}]--> {d.to_section}"
                f"  ({d.description}, confidence={d.confidence:.2f})"
            )
        return "\n".join(lines)
