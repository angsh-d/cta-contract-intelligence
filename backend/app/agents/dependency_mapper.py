"""Tier 2: DependencyMapperAgent — clause dependency graph via LLM identification + PostgreSQL sync."""

import logging
from typing import Any
from uuid import UUID

from app.agents.base import BaseAgent
from app.exceptions import LLMResponseError
from app.models.agent_schemas import (
    ClauseDependency, CurrentClause, DependencyMapInput, DependencyMapOutput,
)
from app.models.enums import RelationshipType

logger = logging.getLogger(__name__)


class DependencyMapperAgent(BaseAgent):
    """Build a dependency graph of clause relationships in PostgreSQL."""

    _VALID_REL_TYPES = {rt.value for rt in RelationshipType}

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
            if tool["name"] in ("get_clause", "get_dependencies")
        ]

    async def _execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        async with self.db.acquire() as conn:
            if tool_name == "get_clause":
                row = await conn.fetchrow(
                    "SELECT section_number, section_title, current_text, clause_category "
                    "FROM clauses WHERE section_number = $1 AND contract_stack_id = $2",
                    tool_input["section_number"], str(self._current_stack_id),
                )
                return dict(row) if row else {"error": "Clause not found"}
            elif tool_name == "get_dependencies":
                rows = await conn.fetch(
                    "SELECT cd.from_clause_id, cd.to_clause_id, cd.relationship_type, cd.description "
                    "FROM clause_dependencies cd "
                    "JOIN clauses c ON c.id = cd.from_clause_id "
                    "WHERE c.section_number = $1 AND cd.contract_stack_id = $2",
                    tool_input["section_number"], str(self._current_stack_id),
                )
                return [dict(r) for r in rows]
        return await super()._execute_tool(tool_name, tool_input)

    async def process(self, input_data: DependencyMapInput) -> DependencyMapOutput:
        self._current_stack_id = input_data.contract_stack_id
        clauses = input_data.current_clauses

        # Phase 1: Single LLM call identifies ALL dependencies
        system_prompt = self.prompts.get("dependency_mapper_system")
        user_prompt = self.prompts.get(
            "dependency_mapper_identify",
            clauses=self._format_clauses(clauses),
        )
        result = await self.call_llm(system_prompt, user_prompt)
        await self._report_progress("llm_analysis", 50, "LLM dependency analysis complete")

        deps_raw = result.get("dependencies")
        if deps_raw is None:
            raise LLMResponseError("DependencyMapperAgent: LLM response missing 'dependencies'")
        all_deps = [
            ClauseDependency(
                from_section=d.get("from_section", ""),
                to_section=d.get("to_section", ""),
                relationship_type=d.get("relationship_type", "references"),
                description=d.get("description", ""),
                confidence=d.get("confidence", 0.8),
                detection_method="llm",
            )
            for d in deps_raw
        ]

        # Deduplicate
        seen = set()
        deduped_deps = []
        for d in all_deps:
            key = (d.from_section, d.to_section, d.relationship_type)
            if key not in seen:
                seen.add(key)
                deduped_deps.append(d)

        await self._report_progress("llm_analysis", 60, f"Found {len(deduped_deps)} dependencies")

        # Phase 2: PostgreSQL sync
        await self._sync_dependencies(input_data.contract_stack_id, clauses, deduped_deps)
        await self._report_progress("db_sync", 100, "Dependency graph updated")

        return DependencyMapOutput(
            contract_stack_id=input_data.contract_stack_id,
            dependencies=deduped_deps,
            total_nodes=len(clauses),
            total_edges=len(deduped_deps),
            db_synced=True,
            llm_reasoning=result.get("reasoning", ""),
        )

    def _format_clauses(self, clauses: list[CurrentClause]) -> str:
        parts = []
        for c in clauses:
            parts.append(
                f"Section {c.section_number} ({c.section_title}) [{c.clause_category}]:\n"
                f"{c.current_text}"
            )
        return "\n\n---\n\n".join(parts)

    async def _sync_dependencies(self, stack_id: UUID, clauses: list[CurrentClause], deps: list[ClauseDependency]) -> None:
        """Write clause dependencies to PostgreSQL atomically, clearing stale rows first."""
        async with self.db.acquire() as conn:
            async with conn.transaction():
                # Clear stale dependency data for this stack before re-inserting
                await conn.execute(
                    "DELETE FROM clause_dependencies WHERE contract_stack_id = $1",
                    str(stack_id),
                )
                for dep in deps:
                    await conn.execute(
                        """
                        INSERT INTO clause_dependencies
                            (contract_stack_id, from_clause_id, to_clause_id,
                             relationship_type, description, confidence, detection_method)
                        SELECT $1, f.id, t.id, $4, $5, $6, $7
                        FROM clauses f, clauses t
                        WHERE f.section_number = $2 AND f.contract_stack_id = $1
                          AND t.section_number = $3 AND t.contract_stack_id = $1
                        ON CONFLICT DO NOTHING
                        """,
                        str(stack_id),
                        dep.from_section,
                        dep.to_section,
                        dep.relationship_type.value,
                        dep.description,
                        dep.confidence,
                        dep.detection_method,
                    )
