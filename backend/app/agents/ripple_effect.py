"""Tier 3: RippleEffectAnalyzerAgent — recursive CTE traversal + per-hop LLM analysis."""

import json
import logging
import time
from typing import Any, Optional
from uuid import UUID

from app.agents.base import BaseAgent
from app.models.agent_schemas import (
    PrioritizedAction, ProposedChange, RippleEffectInput, RippleEffectOutput,
    RippleImpact, RippleRecommendations,
)

logger = logging.getLogger(__name__)


class RippleEffectAnalyzerAgent(BaseAgent):
    """Analyze ripple effects of proposed amendments via dependency graph traversal."""

    MAX_SINGLE_IMPACT_COST = 10_000_000  # $10M sanity check

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
                    tool_input["section_number"], str(self._current_stack_id),
                )
                return dict(row) if row else {"error": "Clause not found"}
            elif tool_name == "get_dependencies":
                rows = await conn.fetch(
                    "SELECT cd.from_clause_id, cd.to_clause_id, cd.relationship_type "
                    "FROM clause_dependencies cd "
                    "JOIN clauses c ON c.id = cd.from_clause_id "
                    "WHERE c.section_number = $1 AND cd.contract_stack_id = $2",
                    tool_input["section_number"], str(self._current_stack_id),
                )
                return [dict(r) for r in rows]
            elif tool_name == "get_amendment_history":
                rows = await conn.fetch(
                    "SELECT a.amendment_number, a.modification_type, a.rationale "
                    "FROM amendments a "
                    "WHERE a.contract_stack_id = $1 ORDER BY a.amendment_number",
                    str(self._current_stack_id),
                )
                return [dict(r) for r in rows]
        return await super()._execute_tool(tool_name, tool_input)

    async def process(self, input_data: RippleEffectInput) -> RippleEffectOutput:
        start = time.monotonic()
        change = input_data.proposed_change
        stack_id = input_data.contract_stack_id
        self._current_stack_id = stack_id

        # Step 1 & 2: Bidirectional PostgreSQL traversal
        outbound_paths = await self._traverse_dependencies(stack_id, change.section_number, "outbound")
        inbound_paths = await self._traverse_dependencies(stack_id, change.section_number, "inbound")

        all_paths = self._merge_paths(outbound_paths, inbound_paths)
        grouped_by_hop = self._group_by_hop(all_paths)

        await self._report_progress("graph_traversal", 20,
            f"Found {sum(len(v) for v in grouped_by_hop.values())} affected clauses across {len(grouped_by_hop)} hops")

        # Step 3: Per-hop LLM analysis with early termination
        impacts_by_hop: dict[str, list[RippleImpact]] = {}
        all_reasoning = ""

        for hop in range(1, 6):
            hop_key = f"hop_{hop}"
            hop_clauses = grouped_by_hop.get(hop, [])
            if not hop_clauses:
                break

            system_prompt = self.prompts.get("ripple_effect_system")
            user_prompt = self.prompts.get(
                "ripple_effect_hop_analysis",
                section_number=change.section_number,
                current_text=change.current_text,
                proposed_text=change.proposed_text,
                hop_number=str(hop),
                hop_clauses=self._format_hop_clauses(hop_clauses),
            )
            result = await self.call_llm(system_prompt, user_prompt)

            if result.get("reasoning"):
                all_reasoning += f"[Hop {hop}] {result['reasoning']}\n"

            raw_impacts = result.get("impacts", [])
            hop_impacts = []
            for impact in raw_impacts:
                impact_data = dict(impact)
                impact_data["hop_distance"] = hop  # Override any LLM-provided hop_distance
                hop_impacts.append(RippleImpact(**impact_data))
            impacts_by_hop[hop_key] = hop_impacts

            await self._report_progress("hop_analysis", 20 + int(50 * hop / 5),
                f"Analyzed hop {hop}: {len(hop_impacts)} impacts")

            # Early termination
            if len(hop_impacts) == 0 and hop >= 2:
                remaining = [h for h in range(hop + 1, 6) if grouped_by_hop.get(h)]
                if not remaining:
                    break

        # Step 4: Recommendation synthesis
        all_impacts = []
        for hop_list in impacts_by_hop.values():
            all_impacts.extend(hop_list)

        recommendations = await self._synthesize_recommendations(all_impacts, change)
        estimated_total_cost = self._estimate_total_cost(all_impacts)

        latency_ms = int((time.monotonic() - start) * 1000)
        cascade_depth = max(
            (int(k.split("_")[1]) for k, imps in impacts_by_hop.items() if imps),
            default=0,
        )

        return RippleEffectOutput(
            contract_stack_id=stack_id,
            proposed_change=change,
            impacts_by_hop=impacts_by_hop,
            total_impacts=len(all_impacts),
            cascade_depth=cascade_depth,
            estimated_total_cost=estimated_total_cost,
            recommendations=recommendations,
            traversal_direction="bidirectional",
            analysis_model=self.config.model_override or "",
            analysis_latency_ms=latency_ms,
            llm_reasoning=all_reasoning,
        )

    async def _traverse_dependencies(self, stack_id: UUID, section: str, direction: str) -> list[dict]:
        """Traverse clause dependencies using PostgreSQL recursive CTE."""
        async with self.db.acquire() as conn:
            if direction == "outbound":
                rows = await conn.fetch(
                    """
                    WITH RECURSIVE dependency_chain AS (
                        SELECT cd.from_clause_id, cd.to_clause_id,
                               cd.relationship_type, 1 AS hop,
                               ARRAY[cd.from_clause_id] AS path
                        FROM clause_dependencies cd
                        JOIN clauses c ON c.id = cd.from_clause_id
                        WHERE c.section_number = $1 AND cd.contract_stack_id = $2

                        UNION ALL

                        SELECT cd.from_clause_id, cd.to_clause_id,
                               cd.relationship_type, dc.hop + 1,
                               dc.path || cd.from_clause_id
                        FROM clause_dependencies cd
                        JOIN dependency_chain dc ON cd.from_clause_id = dc.to_clause_id
                        WHERE dc.hop < 5 AND cd.to_clause_id != ALL(dc.path)
                          AND cd.contract_stack_id = $2
                    )
                    SELECT dc.to_clause_id, dc.hop AS hop_distance,
                           c.section_number, c.section_title, c.clause_category, c.current_text
                    FROM dependency_chain dc
                    JOIN clauses c ON c.id = dc.to_clause_id
                    ORDER BY dc.hop
                    """,
                    section, str(stack_id),
                )
            else:
                rows = await conn.fetch(
                    """
                    WITH RECURSIVE dependency_chain AS (
                        SELECT cd.from_clause_id, cd.to_clause_id,
                               cd.relationship_type, 1 AS hop,
                               ARRAY[cd.to_clause_id] AS path
                        FROM clause_dependencies cd
                        JOIN clauses c ON c.id = cd.to_clause_id
                        WHERE c.section_number = $1 AND cd.contract_stack_id = $2

                        UNION ALL

                        SELECT cd.from_clause_id, cd.to_clause_id,
                               cd.relationship_type, dc.hop + 1,
                               dc.path || cd.to_clause_id
                        FROM clause_dependencies cd
                        JOIN dependency_chain dc ON cd.to_clause_id = dc.from_clause_id
                        WHERE dc.hop < 5 AND cd.from_clause_id != ALL(dc.path)
                          AND cd.contract_stack_id = $2
                    )
                    SELECT dc.from_clause_id AS clause_id, dc.hop AS hop_distance,
                           c.section_number, c.section_title, c.clause_category, c.current_text
                    FROM dependency_chain dc
                    JOIN clauses c ON c.id = dc.from_clause_id
                    ORDER BY dc.hop
                    """,
                    section, str(stack_id),
                )

            paths = []
            for row in rows:
                paths.append({
                    "hop_distance": row["hop_distance"],
                    "direction": direction,
                    "nodes": [{
                        "section_number": row["section_number"],
                        "section_title": row["section_title"],
                        "category": row["clause_category"],
                        "current_text": row["current_text"],
                    }],
                })
            return paths

    def _merge_paths(self, outbound: list[dict], inbound: list[dict]) -> list[dict]:
        best: dict[str, dict] = {}
        for path in outbound + inbound:
            terminal = path["nodes"][-1]["section_number"] if path["nodes"] else None
            if terminal is None:
                continue
            existing = best.get(terminal)
            if existing is None or path["hop_distance"] < existing["hop_distance"]:
                best[terminal] = path
        return list(best.values())

    def _group_by_hop(self, paths: list[dict]) -> dict[int, list[dict]]:
        grouped: dict[int, list[dict]] = {}
        for path in paths:
            hop = path["hop_distance"]
            grouped.setdefault(hop, []).append(path)
        return grouped

    def _format_hop_clauses(self, hop_clauses: list[dict]) -> str:
        parts = []
        seen = set()
        for clause in hop_clauses:
            node = clause["nodes"][-1] if clause["nodes"] else None
            if node and node["section_number"] not in seen:
                seen.add(node["section_number"])
                parts.append(
                    f"Section {node['section_number']} ({node['section_title']})\n"
                    f"Category: {node['category']}\n"
                    f"Current Text: {node.get('current_text', '[text not available]')}\n"
                )
        return "\n---\n".join(parts)

    def _estimate_total_cost(self, impacts: list[RippleImpact]) -> Optional[str]:
        total_low, total_high = 0, 0
        seen_categories: set[str] = set()
        for impact in impacts:
            dedup_key = f"{impact.impact_type}:{impact.affected_section}"
            if dedup_key in seen_categories:
                continue
            seen_categories.add(dedup_key)
            if impact.estimated_cost_low is not None:
                total_low += impact.estimated_cost_low
            if impact.estimated_cost_high is not None:
                # Cap without mutating the original impact object
                cost_high = min(impact.estimated_cost_high, self.MAX_SINGLE_IMPACT_COST)
                if cost_high < impact.estimated_cost_high:
                    logger.warning("Capping cost estimate for %s at $%s", impact.affected_section, self.MAX_SINGLE_IMPACT_COST)
                total_high += cost_high
        if total_low == 0 and total_high == 0:
            return None
        return f"${total_low:,} - ${total_high:,} (LLM-estimated, not validated)"

    async def _synthesize_recommendations(
        self, all_impacts: list[RippleImpact], change: ProposedChange
    ) -> RippleRecommendations:
        if not all_impacts:
            return RippleRecommendations(critical_actions=[], recommended_actions=[], optional_actions=[])

        system_prompt = self.prompts.get("ripple_effect_recommendations")
        user_prompt = self.prompts.get(
            "ripple_effect_recommendations_input",
            section_number=change.section_number,
            current_text=change.current_text,
            proposed_text=change.proposed_text,
            impacts_json=json.dumps([i.model_dump(mode="json") for i in all_impacts], indent=2),
        )
        result = await self.call_llm(system_prompt, user_prompt)

        return RippleRecommendations(
            critical_actions=[PrioritizedAction(**a) for a in result.get("critical_actions", [])],
            recommended_actions=[PrioritizedAction(**a) for a in result.get("recommended_actions", [])],
            optional_actions=[PrioritizedAction(**a) for a in result.get("optional_actions", [])],
        )
