# 05 — Tier 3: Analysis Agents

> RippleEffectAnalyzerAgent (full specification), ReusabilityAnalyzerAgent (Phase 2 interface only)
> File locations: `backend/app/agents/ripple_effect.py`, `backend/app/agents/reusability.py`

---

## 1. RippleEffectAnalyzerAgent

**Purpose:** Given a proposed amendment, traverse the PostgreSQL dependency graph (up to 5 hops via recursive CTE) to identify all direct and indirect impacts, synthesize recommendations with cost estimates.
**LLM:** Azure OpenAI GPT-5.2 (complex multi-hop reasoning)
**Database:** PostgreSQL/NeonDB (clause_dependencies table with recursive CTEs)
**File:** `backend/app/agents/ripple_effect.py`

### Config

```python
AgentConfig(
    agent_name="ripple_effect",
    llm_role="complex_reasoning",
    max_output_tokens=16000,
    timeout_seconds=300,
    temperature=0.2,
    verification_threshold=0.70,
)
```

### Architecture

```
Proposed Change (Section 9.2: 15yr → 25yr data retention)
     │
     ▼
┌──────────────────────────────────────────────────────┐
│ Step 1: PostgreSQL Recursive CTE (OUTBOUND)            │
│   WITH RECURSIVE from clause_dependencies             │
│   WHERE from_clause = "9.2" ... hop < 5              │
│   Group results by hop distance                      │
├──────────────────────────────────────────────────────┤
│ Step 2: PostgreSQL Recursive CTE (INBOUND)             │
│   WITH RECURSIVE from clause_dependencies             │
│   WHERE to_clause = "9.2" ... hop < 5                │
│   What depends ON the changed clause?                │
├──────────────────────────────────────────────────────┤
│ Step 3: Per-Hop LLM Analysis                          │
│   For hop = 1, 2, 3, ... (up to 5):                 │
│     - Feed affected clauses to LLM                   │
│     - Assess material impact                         │
│     - Early termination if no material impacts       │
├──────────────────────────────────────────────────────┤
│ Step 4: Recommendation Synthesis                      │
│   - Aggregate all impacts                            │
│   - Prioritize by severity + cost                    │
│   - Group related actions                            │
│   - Generate estimated total cost                    │
└──────────────────────────────────────────────────────┘
```

### Bidirectional Traversal

Two traversal directions capture the full impact picture:

| Direction | Query Pattern | Question Answered |
|-----------|---------------|-------------------|
| **OUTBOUND** | `WHERE from_clause_id = $changed ... hop < 5` | "What does this clause affect?" |
| **INBOUND** | `WHERE to_clause_id = $changed ... hop < 5` | "What depends on this clause?" |

For the data retention example:
- **Outbound:** Data retention → storage costs → vendor contracts → budget
- **Inbound:** Regulatory requirements → data retention (policy mandates this change)

### Process Flow — Detailed

```python
class RippleEffectAnalyzerAgent(BaseAgent):
    def __init__(self, config, llm_provider, prompt_loader, db_pool,
                 progress_callback=None, fallback_provider=None,
                 trace_context=None, llm_semaphore=None):
        super().__init__(config, llm_provider, prompt_loader, progress_callback,
                         fallback_provider, trace_context, llm_semaphore)
        self.db = db_pool
        self._current_stack_id = None

    def get_tools(self) -> list[dict[str, Any]]:
        """RippleEffect can dynamically query clauses and dependencies during hop analysis."""
        return [
            tool for tool in self.STANDARD_TOOLS
            if tool["name"] in ("get_clause", "get_dependencies", "get_amendment_history")
        ]

    async def _execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        """Execute tool calls for ripple effect analysis."""
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
                    "SELECT document_label, modification_type, change_description "
                    "FROM amendment_modifications "
                    "WHERE section_number = $1 AND contract_stack_id = $2 "
                    "ORDER BY effective_date",
                    tool_input["section_number"], str(self._current_stack_id),
                )
                return [dict(r) for r in rows]
        return await super()._execute_tool(tool_name, tool_input)

    async def process(self, input_data: RippleEffectInput) -> RippleEffectOutput:
        start = time.monotonic()
        change = input_data.proposed_change
        stack_id = input_data.contract_stack_id
        self._current_stack_id = stack_id

        # Step 1 & 2: Bidirectional PostgreSQL traversal (recursive CTE)
        outbound_paths = await self._traverse_dependencies(
            stack_id, change.section_number, direction="outbound"
        )
        inbound_paths = await self._traverse_dependencies(
            stack_id, change.section_number, direction="inbound"
        )

        # Merge and deduplicate paths from both directions, group by hop
        all_paths = self._merge_paths(outbound_paths, inbound_paths)
        grouped_by_hop = self._group_by_hop(all_paths)

        await self._report_progress("graph_traversal", 20,
            f"Found {sum(len(v) for v in grouped_by_hop.values())} affected clauses across {len(grouped_by_hop)} hops")

        # Step 3: Per-hop LLM analysis with early termination
        impacts_by_hop: dict[str, list[RippleImpact]] = {}
        all_reasoning: str = ""

        for hop in range(1, 6):
            hop_key = f"hop_{hop}"
            hop_clauses = grouped_by_hop.get(hop, [])

            if not hop_clauses:
                break  # no clauses at this depth

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

            hop_impacts = []
            for impact in result["impacts"]:
                try:
                    hop_impacts.append(RippleImpact(**impact, hop_distance=hop))
                except ValidationError as e:
                    logger.warning("Skipping malformed impact at hop %d: %s", hop, e)
                    continue
            impacts_by_hop[hop_key] = hop_impacts

            await self._report_progress("hop_analysis", 20 + int(50 * hop / 5),
                f"Analyzed hop {hop}: {len(hop_impacts)} impacts")

            # LLM-driven early termination: ask the LLM whether further analysis is worthwhile
            if len(hop_impacts) == 0 and hop >= 2:
                remaining_hops = [h for h in range(hop + 1, 6) if grouped_by_hop.get(h)]
                if remaining_hops:
                    continuation = await self.call_llm(
                        self.prompts.get("ripple_effect_system"),
                        f"Hop {hop} produced no material impacts. "
                        f"Remaining clauses at deeper hops: {len(remaining_hops)} hops with "
                        f"{sum(len(grouped_by_hop.get(h, [])) for h in remaining_hops)} total clauses. "
                        f"Given the nature of the proposed change (Section {change.section_number}: "
                        f"{change.change_description}), should analysis continue to deeper hops? "
                        f"Return JSON: {{\"should_continue\": true/false, \"reasoning\": \"...\"}}",
                    )
                    if not continuation.get("should_continue", True):
                        logger.info("LLM-driven termination at hop %d: %s", hop, continuation.get("reasoning", ""))
                        break
                else:
                    break  # no remaining clauses at deeper hops

        # Step 4: Recommendation synthesis
        all_impacts = []
        for hop_list in impacts_by_hop.values():
            all_impacts.extend(hop_list)

        recommendations = await self._synthesize_recommendations(all_impacts, change)

        # Estimate total cost
        estimated_total_cost = self._estimate_total_cost(all_impacts)

        latency_ms = int((time.monotonic() - start) * 1000)

        return RippleEffectOutput(
            contract_stack_id=stack_id,
            proposed_change=change,
            impacts_by_hop=impacts_by_hop,
            total_impacts=len(all_impacts),
            cascade_depth=max((h for h, imps in impacts_by_hop.items() if imps), default=0),
            estimated_total_cost=estimated_total_cost,
            recommendations=recommendations,
            traversal_direction="bidirectional",
            analysis_model=self.config.model_override,
            analysis_latency_ms=latency_ms,
            llm_reasoning=all_reasoning,
        )
```

### PostgreSQL Recursive CTE Traversal

```python
    async def _traverse_dependencies(self, stack_id: UUID, section: str, direction: str) -> list[dict]:
        """Traverse clause dependencies using PostgreSQL recursive CTE.

        All 7 relationship types are traversed: depends_on, references, modifies,
        replaces, amends, conflicts_with, supersedes.
        """
        async with self.db.acquire() as conn:
            if direction == "outbound":
                # What does this clause affect? (from → to)
                rows = await conn.fetch(
                    """
                    WITH RECURSIVE dependency_chain AS (
                        SELECT cd.from_clause_id, cd.to_clause_id,
                               cd.relationship_type, 1 AS hop,
                               ARRAY[cd.from_clause_id] AS path
                        FROM clause_dependencies cd
                        JOIN clauses c ON c.id = cd.from_clause_id
                        WHERE c.section_number = $1
                          AND cd.contract_stack_id = $2

                        UNION ALL

                        SELECT cd.from_clause_id, cd.to_clause_id,
                               cd.relationship_type, dc.hop + 1,
                               dc.path || cd.from_clause_id
                        FROM clause_dependencies cd
                        JOIN dependency_chain dc ON cd.from_clause_id = dc.to_clause_id
                        WHERE dc.hop < 5
                          AND cd.to_clause_id != ALL(dc.path)
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
            else:  # inbound
                # What depends on this clause? (to → from, reversed)
                rows = await conn.fetch(
                    """
                    WITH RECURSIVE dependency_chain AS (
                        SELECT cd.from_clause_id, cd.to_clause_id,
                               cd.relationship_type, 1 AS hop,
                               ARRAY[cd.to_clause_id] AS path
                        FROM clause_dependencies cd
                        JOIN clauses c ON c.id = cd.to_clause_id
                        WHERE c.section_number = $1
                          AND cd.contract_stack_id = $2

                        UNION ALL

                        SELECT cd.from_clause_id, cd.to_clause_id,
                               cd.relationship_type, dc.hop + 1,
                               dc.path || cd.to_clause_id
                        FROM clause_dependencies cd
                        JOIN dependency_chain dc ON cd.to_clause_id = dc.from_clause_id
                        WHERE dc.hop < 5
                          AND cd.from_clause_id != ALL(dc.path)
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
```

### Helper Methods

```python
    def _merge_paths(self, outbound: list[dict], inbound: list[dict]) -> list[dict]:
        """Merge outbound and inbound paths, deduplicating by end-node section_number.
        When the same terminal node is reachable via multiple paths, keep the SHORTEST
        hop distance to correctly classify direct vs. indirect impacts."""
        best: dict[str, dict] = {}  # terminal section_number → path with shortest hop
        for path in outbound + inbound:
            terminal = path["nodes"][-1]["section_number"] if path["nodes"] else None
            if terminal is None:
                continue
            existing = best.get(terminal)
            if existing is None or path["hop_distance"] < existing["hop_distance"]:
                best[terminal] = path
        return list(best.values())

    def _group_by_hop(self, paths: list[dict]) -> dict[int, list[dict]]:
        """Group traversal results by hop distance (integer keys)."""
        grouped: dict[int, list[dict]] = {}
        for path in paths:
            hop = path["hop_distance"]
            grouped.setdefault(hop, []).append(path)
        return grouped

    def _format_hop_clauses(self, hop_clauses: list[dict]) -> str:
        """Format clause data for LLM prompt — includes actual clause text for informed analysis."""
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

    MAX_SINGLE_IMPACT_COST = 10_000_000  # $10M sanity check per impact

    def _estimate_total_cost(self, impacts: list[RippleImpact]) -> Optional[str]:
        """
        Sum structured cost estimates from LLM-returned integer fields.
        Includes sanity checks on individual estimates and deduplication by category.
        Returns a formatted range string or None if no costs found.

        Note: These are LLM-generated estimates for directional guidance only,
        not validated financial projections.
        """
        total_low, total_high = 0, 0
        seen_categories: set[str] = set()  # deduplicate overlapping estimates
        for impact in impacts:
            # Sanity check: flag suspiciously high individual estimates
            if impact.estimated_cost_high and impact.estimated_cost_high > self.MAX_SINGLE_IMPACT_COST:
                logger.warning(
                    "Suspiciously high cost estimate $%s for %s — capping at $%s",
                    impact.estimated_cost_high, impact.affected_section, self.MAX_SINGLE_IMPACT_COST,
                )
                impact.estimated_cost_high = self.MAX_SINGLE_IMPACT_COST
            # Deduplicate: same impact_type at different hops may double-count
            dedup_key = f"{impact.impact_type}:{impact.affected_section}"
            if dedup_key in seen_categories:
                continue
            seen_categories.add(dedup_key)
            if impact.estimated_cost_low is not None:
                total_low += impact.estimated_cost_low
            if impact.estimated_cost_high is not None:
                total_high += impact.estimated_cost_high
        if total_low == 0 and total_high == 0:
            return None
        return f"${total_low:,} - ${total_high:,} (LLM-estimated, not validated)"
```

### Recommendation Synthesis

```python
    async def _synthesize_recommendations(
        self, all_impacts: list[RippleImpact], change: ProposedChange
    ) -> RippleRecommendations:
        if not all_impacts:
            return RippleRecommendations(
                critical_actions=[], recommended_actions=[], optional_actions=[]
            )

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
```

### Prompt Files

**`prompt/ripple_effect_system.txt`**
```
You are analyzing the ripple effects of a proposed contract amendment in a clinical trial agreement.

For each affected clause, determine:
1. Is it MATERIALLY impacted by the proposed change?
2. What is the nature of the impact? (cost_increase, compliance_risk, indemnification_gap, coverage_gap, schedule_impact, vendor_impact, regulatory_risk)
3. What specific action is required to address this impact?
4. What is the estimated cost or timeline impact?

Only report MATERIAL impacts. If a clause is technically connected but not practically affected, do not include it.

Be specific and concrete. Use actual dollar ranges, time periods, and action items.

Return ONLY valid JSON.
```

**`prompt/ripple_effect_hop_analysis.txt`**
```
A change is proposed to Section {section_number}:
  CURRENT: {current_text}
  PROPOSED: {proposed_text}

You are analyzing HOP {hop_number} impacts — clauses that are {hop_number} step(s) away in the dependency graph.

Affected clauses at this hop:
{hop_clauses}

For each clause, determine if the proposed change materially affects it.

Before producing your final answer, reason step-by-step about your analysis.
Include your reasoning in the 'reasoning' field of the JSON output.

Return JSON:
{{
  "reasoning": "your step-by-step analysis...",
  "impacts": [
    {{
      "affected_section": "12.1",
      "affected_section_title": "Indemnification",
      "impact_type": "indemnification_gap",
      "severity": "critical",
      "description": "Indemnification covers 7 years but data retention would be 25 years, leaving 18 years unprotected",
      "required_action": "Extend indemnification period to match data retention",
      "estimated_cost_low": 150000,
      "estimated_cost_high": 200000,
      "estimated_cost_rationale": "Additional insurance premium for 18-year extended coverage",
      "estimated_timeline": "3-4 weeks to negotiate"
    }}
  ]
}}

If NO clauses at this hop are materially affected, return: {{"impacts": []}}
```

**`prompt/ripple_effect_recommendations_input.txt`**
```
Proposed change to Section {section_number}:
FROM: {current_text}
TO: {proposed_text}

All identified impacts:
{impacts_json}

Synthesize these impacts into prioritized recommendations.
```

**`prompt/ripple_effect_recommendations.txt`**
```
Synthesize ripple effect analysis into prioritized recommendations.

Group related impacts. Prioritize by:
1. Severity (critical first)
2. Timeline urgency
3. Cost impact
4. Implementation complexity

Return JSON:
{{
  "critical_actions": [
    {{
      "priority": 1,
      "action": "Extend indemnification to 25 years",
      "reason": "18-year coverage gap creates unprotected liability",
      "estimated_cost": "$150,000 - $200,000",
      "deadline": "Before amendment execution",
      "related_sections": ["12.1", "9.2"]
    }}
  ],
  "recommended_actions": [...],
  "optional_actions": [...]
}}
```

### Demo Scenario: Data Retention 15 → 25 Years

Expected ripple analysis output for HEARTBEAT-3:

```
Proposed: Section 9.2 data retention 15 years → 25 years

Hop 1 (Direct):
  - Section 9.4 (Archive Storage): Cost increase for 10 additional years
    Estimated: $3,000-5,000/year × 10 = $30,000-50,000
  - Section 12.1 (Indemnification): Coverage gap — indemnification is 7 years
    Severity: CRITICAL

Hop 2 (Indirect):
  - Section 11.x (Insurance): Policy duration insufficient for 25-year retention
    Estimated: Additional premium $50,000-75,000
  - Exhibit B-2 (Budget): Storage line items need updating
    Estimated: Budget revision needed

Hop 3:
  - Vendor contracts: Data storage vendor agreements need extension
  - Regulatory: May require IRB notification for retention change

Estimated total cost: $1,800,000 - $2,340,000
(Includes: extended insurance, indemnification renegotiation, storage, vendor contracts, legal fees)

Recommendations:
  CRITICAL: Extend indemnification to 25 years (before amendment execution)
  RECOMMENDED: Update insurance coverage, revise budget exhibit
  OPTIONAL: Negotiate volume discount with storage vendor
```

---

## 2. ReusabilityAnalyzerAgent (Phase 2 — Interface Only)

**Purpose:** Compare contract stacks across a portfolio to identify reusable language, standard deviations, and best practices.
**Status:** Phase 2 — interface defined for API stability, no implementation until Phase 2.
**File:** `backend/app/agents/reusability.py`

### Interface Definition

```python
# backend/app/agents/reusability.py

from app.agents.base import BaseAgent
from app.agents.config import AgentConfig
from app.models.agent_schemas import LLMResponse
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID

class ReusabilityInput(BaseModel):
    """Input to ReusabilityAnalyzerAgent — Phase 2."""
    contract_stack_ids: list[UUID]         # stacks to compare
    focus_categories: list[str] = Field(default_factory=list)  # empty = all categories

class ClauseComparison(BaseModel):
    """Comparison of a clause across contract stacks."""
    section_number: str
    clause_category: str
    variations: list["ClauseVariation"]
    recommended_standard: Optional[str] = None
    deviation_risk: str                    # "low", "medium", "high"

class ClauseVariation(BaseModel):
    """One version of a clause from a specific contract stack."""
    contract_stack_id: UUID
    contract_stack_name: str
    current_text: str
    effective_date: Optional[date] = None  # aligned with doc 02 canonical definition

class ReusabilityOutput(BaseModel):
    """Output from ReusabilityAnalyzerAgent — Phase 2."""
    comparisons: list[ClauseComparison]
    reusability_score: float = Field(ge=0.0, le=1.0)
    standard_deviations: list[str]
    recommendations: list[str]

class ReusabilityAnalyzerAgent(BaseAgent):
    """Phase 2 — Not yet implemented."""

    async def process(self, input_data: ReusabilityInput) -> ReusabilityOutput:
        raise NotImplementedError(
            "ReusabilityAnalyzerAgent is scheduled for Phase 2. "
            "Interface defined for API stability."
        )
```

### Config (Phase 2)

```python
AgentConfig(
    agent_name="reusability_analyzer",
    llm_role="complex_reasoning",
    max_output_tokens=8192,
    timeout_seconds=300,
)
```

### Phase 2 Design Notes

When implemented, the reusability agent will:
1. Query PostgreSQL for clauses across multiple contract stacks
2. Group by clause_category
3. Use LLM to compare variations and identify deviations from standard language
4. Score reusability (how much boilerplate can be templated)
5. Flag non-standard clauses that deviate from best practices
