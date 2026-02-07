# 04 — Tier 2: Reasoning Agents

> OverrideResolutionAgent, ConflictDetectionAgent, DependencyMapperAgent
> File locations: `backend/app/agents/override_resolution.py`, `backend/app/agents/conflict_detection.py`, `backend/app/agents/dependency_mapper.py`
>
> **Note:** The `CrossReferenceAgent` mentioned in the technical spec is folded into `DependencyMapperAgent`, which identifies both explicit cross-references and semantic dependencies in a single LLM call. This avoids a redundant agent that would duplicate cross-reference work.

---

## Pipeline Flow

```
Tier 1 outputs
     │
     ▼
OverrideResolutionAgent (parallel per section)
     │
     ▼
Resolved current clauses
     │
     ▼
DependencyMapperAgent (single call, all clauses)
     │
     ▼
PostgreSQL clause_dependencies table
     │
     ▼
ConflictDetectionAgent (single call, all clauses + graph)
     │
     ▼
Detected conflicts
```

- **Override resolution** runs in parallel per section (each section's chain of amendments is independent).
- **Dependency mapping** must run after override resolution so it operates on current clause text.
- **Conflict detection** runs last because it uses both resolved clauses AND the dependency graph.

---

## 1. OverrideResolutionAgent

**Purpose:** For each clause section, apply all amendments in chronological order to determine the current text with full provenance.
**LLM:** Claude Opus (complex legal reasoning)
**File:** `backend/app/agents/override_resolution.py`

### Config

```python
AgentConfig(
    agent_name="override_resolution",
    llm_role="complex_reasoning",
    model_override="claude-opus-4-5-20250514",
    max_output_tokens=8192,
    max_retries=3,
    timeout_seconds=180,
)
```

### Override Logic — LLM-First Reasoning

Rather than forcing amendments into a rigid taxonomy, the agent instructs the LLM to reason from first principles about how each amendment's legal language affects the existing clause text. The LLM considers the full spectrum of amendment patterns:

- Complete replacement ("Section X is deleted and replaced with...")
- Selective phrase substitution ("the words '30 days' are replaced with '45 days'")
- Additions and insertions ("a new subsection (c) is added")
- Exhibit replacements and lineage tracking (B → B-1 → B-2)
- Conditional overrides ("Notwithstanding Section X, in the event of...")
- Exceptions ("except as modified by this Amendment")
- Composite modifications combining multiple patterns in one clause

The LLM reasons step-by-step about the legal intent, then produces the resolved clause text with full provenance.

### Process Flow

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Build Amendment Chain                            │
│   - Sort amendments for this section by effective_date  │
│   - Each amendment carries its Modification model       │
├─────────────────────────────────────────────────────────┤
│ Step 2: Apply Amendments (LLM)                           │
│   - Feed original clause + ordered amendments to LLM    │
│   - LLM applies each amendment step by step             │
│   - Returns current_text + source_chain                 │
├─────────────────────────────────────────────────────────┤
│ Step 3: Source Chain Construction                         │
│   - One SourceChainLink per stage (original → amend1    │
│     → amend3 → ...)                                    │
│   - Each link records: text at that stage, what changed │
├─────────────────────────────────────────────────────────┤
│ Step 4: Confidence Assessment                            │
│   - LLM self-assesses confidence (0-1)                  │
│   - Lower confidence when:                              │
│     - Selective overrides with ambiguous scope           │
│     - References to exhibits not in the amendment        │
│     - Multiple amendments touch same subsection          │
└─────────────────────────────────────────────────────────┘
```

### Pseudocode

```python
class OverrideResolutionAgent(BaseAgent):
    async def process(self, input_data: OverrideResolutionInput) -> OverrideResolutionOutput:
        # Sort amendments by date
        amendments = sorted(input_data.amendments, key=lambda a: a.effective_date)

        # Build the LLM prompt
        system_prompt = self.prompts.get("override_resolution_system")
        user_prompt = self.prompts.get(
            "override_resolution_apply",
            section_number=input_data.section_number,
            original_text=input_data.original_clause.text,
            original_source=input_data.original_document_label,
            amendment_chain=self._format_amendment_chain(amendments),
        )

        result = await self.call_llm(system_prompt, user_prompt)

        source_chain = [SourceChainLink(**link) for link in result["source_chain"]]

        # Determine last modification source
        if amendments:
            last_modified_by = amendments[-1].amendment_document_id
            last_modified_date = amendments[-1].effective_date
        else:
            # No amendments — original clause is the current version.
            # last_modified_date comes from the LLM result or the original document metadata.
            last_modified_by = input_data.original_document_id
            last_modified_date = date.fromisoformat(result["effective_date"]) if result.get("effective_date") else None

        return OverrideResolutionOutput(
            clause_version=ClauseVersion(
                section_number=input_data.section_number,
                section_title=input_data.original_clause.section_title,
                current_text=result["current_text"],
                source_chain=source_chain,
                last_modified_by=last_modified_by,
                last_modified_date=last_modified_date,
                confidence=result["confidence"],
                clause_category=result.get("clause_category", "general"),
            ),
            llm_reasoning=result.get("reasoning", ""),
            confidence_factors=result.get("confidence_factors", {}),
        )

    def _format_amendment_chain(self, amendments: list[AmendmentForSection]) -> str:
        parts = []
        for a in amendments:
            parts.append(
                f"Amendment {a.amendment_number} (Effective {a.effective_date}):\n"
                f"  Document ID: {a.amendment_document_id}\n"
                f"  Type: {a.modification.modification_type.value}\n"
                f"  Original text: {a.modification.original_text or '[N/A]'}\n"
                f"  New text: {a.modification.new_text or '[DELETED]'}\n"
                f"  Change: {a.modification.change_description}"
            )
        return "\n\n".join(parts)
```

### Prompt Files

**`prompt/override_resolution_system.txt`**
```
You are an expert at applying contract amendments to determine the CURRENT state of a clause.

Rules for applying amendments in chronological order:
1. COMPLETE REPLACEMENT ("deleted in its entirety and replaced"): Discard ALL prior text. Use ONLY the new text.
2. SELECTIVE OVERRIDE ("amended by deleting X and substituting Y"): Find X in the current text and replace with Y. Everything else stays.
3. ADDITION ("A new subsection is hereby added"): Append to existing text. Do not change existing text.
4. DELETION ("Section X is hereby deleted"): Mark as deleted. Current text is empty.
5. EXHIBIT REPLACEMENT ("Exhibit B replaced by Exhibit B-1"): Update all references.

Apply each amendment IN ORDER. The output of one amendment becomes the input for the next.

CRITICAL: For selective overrides, change ONLY the specified text. Do not rewrite surrounding language.

REASONING PROCESS:
Think step-by-step about each amendment:
1. What specific language does this amendment use? (e.g., "deleted and replaced", "amended by substituting")
2. What is the legal intent — complete replacement, selective modification, addition, exception, or something else?
3. How does it interact with prior amendments to this section?
4. Apply the change and verify the result makes legal sense.

Include your step-by-step reasoning in the 'reasoning' field of the output.

Before producing your final answer, reason step-by-step about your analysis.
Include your reasoning in the 'reasoning' field of the JSON output.

Return JSON with the final current text, a source chain showing each transformation step, and your confidence score.
```

**`prompt/override_resolution_apply.txt`**
```
Determine the current text of Section {section_number}.

Original text (from original CTA):
{original_text}
Source: {original_source}

Amendments to apply (in chronological order):
{amendment_chain}

Apply all amendments step by step and return:
{{
  "current_text": "the final text after all amendments",
  "clause_category": "payment|insurance|indemnification|data_retention|confidentiality|termination|general",
  "source_chain": [
    {{
      "stage": "original",
      "document_id": "...",
      "document_label": "Original CTA",
      "text": "original text",
      "change_description": null,
      "modification_type": null
    }},
    {{
      "stage": "amendment_3",
      "document_id": "...",
      "document_label": "Amendment 3 (Aug 2023)",
      "text": "text after this amendment",
      "change_description": "Payment terms changed from Net 30 to Net 45",
      "modification_type": "selective_override"
    }}
  ],
  "confidence": 0.95
}}
```

### HEARTBEAT-3 Pain Points Addressed

**Pain Point #1 — Buried Payment Change:**
When processing Section 7.2, the agent receives Amendment 3's selective override (Net 30 → Net 45) as part of the amendment chain. The source chain will explicitly show this transformation, making the buried change visible.

**Pain Point #4 — Cross-Reference Confusion:**
When processing cardiac MRI visit sections, the agent sees that Amendment 4 removed some follow-up visits but did NOT modify the cardiac MRI clause from Amendment 1. The source chain shows Amendment 1 as the last modifier, proving the cardiac MRI visit still survives.

---

## 2. ConflictDetectionAgent

**Purpose:** Find contradictions, ambiguities, gaps, buried changes, stale references, and temporal mismatches across the entire contract stack.
**LLM:** Claude Opus (complex legal reasoning across many clauses)
**File:** `backend/app/agents/conflict_detection.py`

### Config

```python
AgentConfig(
    agent_name="conflict_detection",
    llm_role="complex_reasoning",
    model_override="claude-opus-4-5-20250514",
    max_output_tokens=8192,
    max_retries=3,
    timeout_seconds=300,     # longer timeout — analyzing all clauses at once
    verification_threshold=0.70,
)
```

### Constructor

Like `DependencyMapperAgent` and `RippleEffectAnalyzerAgent`, the ConflictDetectionAgent receives a `db_pool` to query the dependency graph (via recursive CTEs) for richer conflict detection (e.g., identifying conflicts between transitively dependent clauses).

```python
def __init__(self, config, llm_provider, prompt_loader, db_pool,
             progress_callback=None, fallback_provider=None,
             trace_context=None, llm_semaphore=None):
    super().__init__(config, llm_provider, prompt_loader, progress_callback,
                     fallback_provider, trace_context, llm_semaphore)
    self.db = db_pool

    def get_tools(self) -> list[dict[str, Any]]:
        """ConflictDetection can dynamically query clauses and dependencies during analysis."""
        return [
            tool for tool in self.STANDARD_TOOLS
            if tool["name"] in ("get_clause", "get_dependencies", "get_amendment_history")
        ]

    async def _execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        """Execute tool calls against PostgreSQL and pgvector."""
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
                    "SELECT from_clause_id, to_clause_id, relationship_type, description "
                    "FROM clause_dependencies cd "
                    "JOIN clauses c ON c.id = cd.from_clause_id OR c.id = cd.to_clause_id "
                    "WHERE c.section_number = $1 AND cd.contract_stack_id = $2",
                    tool_input["section_number"], str(self._current_stack_id),
                )
                return [dict(r) for r in rows]
            elif tool_name == "get_amendment_history":
                rows = await conn.fetch(
                    "SELECT document_label, modification_type, change_description, effective_date "
                    "FROM amendment_modifications "
                    "WHERE section_number = $1 AND contract_stack_id = $2 "
                    "ORDER BY effective_date",
                    tool_input["section_number"], str(self._current_stack_id),
                )
                return [dict(r) for r in rows]
        return await super()._execute_tool(tool_name, tool_input)
```

### Process Flow — Multi-Pass Analysis

```
┌──────────────────────────────────────────────────────┐
│ Pass 1: Intra-Category Analysis                        │
│   For each clause category (payment, insurance, etc.):│
│     - Compare all clauses within the category         │
│     - Detect internal inconsistencies                 │
├──────────────────────────────────────────────────────┤
│ Pass 2: Cross-Category Analysis                        │
│   For semantically related category pairs:            │
│     - payment ↔ budget, insurance ↔ study_duration   │
│     - LLM identifies risky cross-category tensions    │
├──────────────────────────────────────────────────────┤
│ Pass 3: Tool-Augmented Deep Analysis                   │
│   LLM uses tools to dynamically query:                │
│     - get_clause() for specific clause text on demand │
│     - get_dependencies() for graph neighbors          │
│     - get_amendment_history() for source chains       │
│   Enables focused analysis without pre-loading all    │
│   context — LLM pulls what it needs during reasoning  │
├──────────────────────────────────────────────────────┤
│ Pass 4: Synthesis & Deduplication                      │
│   Merge findings from all 3 passes:                   │
│     - Deduplicate overlapping conflicts               │
│     - Assign final severity based on cross-pass signal│
│     - LLM synthesizes a coherent conflict summary     │
└──────────────────────────────────────────────────────┘
```

This replaces the single monolithic LLM call with focused, manageable passes that each operate on a subset of clauses, improving both accuracy and staying within context window limits.

### Pseudocode

```python
class ConflictDetectionAgent(BaseAgent):
    async def process(self, input_data: ConflictDetectionInput) -> ConflictDetectionOutput:
        start = time.monotonic()

        # Step 1: LLM-driven clause grouping and risk identification
        # Instead of hardcoded categories, let the LLM identify the meaningful
        # groupings and cross-category risks for THIS specific contract stack.
        groups = self._group_by_category(input_data.current_clauses)  # initial grouping by existing labels
        self._current_stack_id = input_data.contract_stack_id  # for tool-use access

        # Step 2 & 3: Combined LLM analysis (includes dependency graph for richer detection)
        system_prompt = self.prompts.get("conflict_detection_system")
        user_prompt = self.prompts.get(
            "conflict_detection_analyze",
            grouped_clauses=self._format_groups(groups),
            context=input_data.contract_stack_context.model_dump_json(exclude_none=True),
            dependency_graph=self._format_dependency_graph(input_data.dependency_graph),
        )
        result = await self.call_llm(system_prompt, user_prompt)

        conflicts = [DetectedConflict(**c) for c in result["conflicts"]]
        latency_ms = int((time.monotonic() - start) * 1000)

        # Compute typed severity summary
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for c in conflicts:
            counts[c.severity.value] += 1

        return ConflictDetectionOutput(
            contract_stack_id=input_data.contract_stack_id,
            conflicts=conflicts,
            summary=ConflictSeveritySummary(**counts),
            analysis_model=self.config.model_override,
            analysis_latency_ms=latency_ms,
            llm_reasoning=result.get("reasoning", ""),
            confidence_factors=result.get("confidence_factors", {}),
        )

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
        """Format dependency graph so LLM can detect conflicts between dependent clauses."""
        if not deps:
            return "(no dependencies detected)"
        lines = []
        for d in deps:
            lines.append(
                f"{d.from_section} --[{d.relationship_type.value}]--> {d.to_section}"
                f"  ({d.description}, confidence={d.confidence:.2f})"
            )
        return "\n".join(lines)
```

### Prompt Files

**`prompt/conflict_detection_system.txt`**
```
You are an expert at detecting conflicts, inconsistencies, and risks in clinical trial agreement contract stacks.

Analyze the provided clauses (grouped by category) and identify ALL of the following:

1. CONTRADICTIONS: Clauses that directly contradict each other
   - Example: One section says "Net 30" but the budget exhibit says "Net 45"

2. BURIED CHANGES: Important changes hidden in unrelated amendments
   - Example: Payment terms changed inside a COVID-related amendment

3. STALE REFERENCES: References to outdated persons, exhibits, or sections
   - Example: Budget exhibit still references the old PI after a PI change

4. TEMPORAL MISMATCHES: Date or duration inconsistencies
   - Example: Insurance covers through Dec 2024 but study extends to June 2025

5. GAPS: Missing information or coverage gaps
   - Example: Extended study period but no mention of extended insurance

6. AMBIGUITIES: Unclear or inconsistent language that creates risk
   - Example: "Protocol as amended" without specifying which amendments

Flag any pattern that represents a risk to the contracting parties:
- Changes buried in unrelated amendment sections
- Evolving exhibits with stale references to superseded versions
- Coverage or indemnification gaps created by study modifications
- Cross-reference confusion from partial amendments
- Personnel changes not consistently reflected across all exhibits

Discover novel conflict patterns specific to this contract stack — do not limit yourself to predefined categories.

You have access to tools for dynamically querying clause text, dependencies, and amendment history.
Use these tools when you need to examine a specific clause in detail rather than relying only on
the pre-loaded context. This is especially useful for verifying cross-references and checking
amendment provenance.

For each conflict:
- Assign severity: critical (blocks execution), high (operational impact), medium (needs clarification), low (minor)
- Provide specific evidence (exact section numbers and text excerpts)
- Give actionable recommendation

Also use the dependency graph provided to identify conflicts between transitively dependent clauses — a change in one clause may create an inconsistency in a clause that depends on it.

Before producing your final answer, reason step-by-step about your analysis.
Include your reasoning in the 'reasoning' field of the JSON output.

Return ONLY valid JSON.
```

**`prompt/conflict_detection_analyze.txt`**
```
Analyze this contract stack for conflicts.

Clauses by category:
{grouped_clauses}

Known clause dependencies (use to detect conflicts between dependent clauses):
{dependency_graph}

Contract context:
{context}

Before producing your final answer, reason step-by-step about your analysis.
Include your reasoning in the 'reasoning' field of the JSON output.

Return JSON:
{{
  "reasoning": "your step-by-step analysis...",
  "conflicts": [
    {{
      "conflict_id": "unique_id",
      "conflict_type": "contradiction|ambiguity|gap|inconsistency|buried_change|stale_reference|temporal_mismatch",
      "severity": "critical|high|medium|low",
      "description": "clear description of the conflict",
      "affected_sections": ["7.2", "Exhibit B-2"],
      "evidence": [
        {{
          "document_id": "...",
          "document_label": "Original CTA",
          "section_number": "7.2",
          "relevant_text": "original text excerpt"
        }},
        {{
          "document_id": "...",
          "document_label": "Amendment 3",
          "section_number": "7.2",
          "relevant_text": "conflicting text excerpt"
        }}
      ],
      "recommendation": "actionable recommendation",
      "pain_point_id": null
    }}
  ]
}}
```

### All 5 HEARTBEAT-3 Pain Points Addressed

| Pain Point | Conflict Type | How Detected |
|-----------|---------------|--------------|
| #1 Buried Payment Change (Net 30→45) | `buried_change` | Cross-group: payment clauses vs. original CTA shows change introduced in COVID amendment |
| #2 Budget Exhibit Evolution (B→B-1→B-2) + old PI | `stale_reference` | Intra-group: budget/personnel clauses reference old PI name after Amendment 2 changed it |
| #3 Insurance Coverage Gap | `gap` + `temporal_mismatch` | Cross-group: insurance end date vs. study extension in Amendment 5 |
| #4 Cross-Reference Confusion (cardiac MRI survives) | `ambiguity` | Cross-group: visit/procedure clauses — Amendment 4 removes follow-ups but MRI from Amendment 1 survives |
| #5 PI Change + Budget Ambiguity | `stale_reference` + `inconsistency` | Intra-group: Amendment 2 changes PI but Exhibit B-1 still references old PI |

---

## 3. DependencyMapperAgent

**Purpose:** Build a dependency graph of clause relationships (explicit references + semantic dependencies) in PostgreSQL.
**LLM:** Claude Opus (for semantic dependency identification)
**Database:** PostgreSQL/NeonDB (clause_dependencies table with recursive CTEs for traversal)
**File:** `backend/app/agents/dependency_mapper.py`

### Config

```python
AgentConfig(
    agent_name="dependency_mapper",
    llm_role="complex_reasoning",
    model_override="claude-opus-4-5-20250514",
    max_output_tokens=8192,
    max_retries=3,
    timeout_seconds=180,
)
```

### LLM-First Dependency Detection

All dependency identification — both explicit cross-references and semantic dependencies — is performed by a single LLM call. This avoids fragile regex patterns and lets the LLM understand context (e.g., "as described in Exhibit B-1" is an explicit reference that regex often misparses due to formatting variations).

```
┌─────────────────────────────────────────────────────────┐
│ Phase 1: LLM Dependency Identification (Single Call)      │
│   - Feed all clauses to LLM                              │
│   - Identify ALL dependency types in one pass:           │
│     - Explicit references ("Section 7.4", "Exhibit B")   │
│     - Functional (payment depends on budget)             │
│     - Conditional ("Subject to Section Y")               │
│     - Temporal (insurance depends on study duration)      │
│   - LLM classifies each as REFERENCES, DEPENDS_ON,      │
│     MODIFIES, etc. with confidence score                 │
│   - Self-references filtered, duplicates removed         │
├─────────────────────────────────────────────────────────┤
│ Phase 2: PostgreSQL Upsert                                │
│   - INSERT clause dependency rows (ON CONFLICT DO        │
│     NOTHING for idempotency)                             │
│   - Set properties (description, confidence,             │
│     detection_method = "llm")                            │
└─────────────────────────────────────────────────────────┘
```

### Multi-Pass Dependency Detection

The dependency mapper uses multiple focused passes for comprehensive coverage:

```python
    async def process(self, input_data: DependencyMapInput) -> DependencyMapOutput:
        clauses = input_data.current_clauses

        # Pass 1: Broad sweep — identify all dependencies in a single call
        system_prompt = self.prompts.get("dependency_mapper_system")
        user_prompt = self.prompts.get(
            "dependency_mapper_identify",
            clauses=self._format_clauses(clauses),
        )
        broad_result = await self.call_llm(system_prompt, user_prompt)
        all_deps = [ClauseDependency(**d, detection_method="llm") for d in broad_result["dependencies"]]

        # Pass 2: Per-category focused analysis — deeper analysis for high-risk categories
        for category in ["payment", "insurance", "indemnification"]:
            category_clauses = [c for c in clauses if c.clause_category == category]
            if len(category_clauses) >= 2:
                focused_deps = await self._focused_category_pass(category, category_clauses, clauses)
                all_deps.extend(focused_deps)

        # Pass 3: Gap analysis — identify clauses with zero dependencies
        connected = {d.from_section for d in all_deps} | {d.to_section for d in all_deps}
        orphans = [c for c in clauses if c.section_number not in connected]
        if orphans:
            gap_deps = await self._gap_analysis_pass(orphans, clauses)
            all_deps.extend(gap_deps)

        # Deduplicate
        deduped_deps = self._deduplicate(all_deps)
        await self._sync_dependencies(input_data.contract_stack_id, clauses, deduped_deps)

        return DependencyMapOutput(
            contract_stack_id=input_data.contract_stack_id,
            dependencies=deduped_deps,
            total_nodes=len(clauses),
            total_edges=len(deduped_deps),
            db_synced=True,
            llm_reasoning=broad_result.get("reasoning", ""),
        )
```

### Pseudocode

```python
class DependencyMapperAgent(BaseAgent):
    # Class-level constant for relationship type validation (defense-in-depth)
    _VALID_REL_TYPES = {rt.value.upper() for rt in RelationshipType}

    def __init__(self, config, llm_provider, prompt_loader, db_pool,
                 progress_callback=None, fallback_provider=None,
                 trace_context=None, llm_semaphore=None):
        super().__init__(config, llm_provider, prompt_loader, progress_callback,
                         fallback_provider, trace_context, llm_semaphore)
        self.db = db_pool

    def get_tools(self) -> list[dict[str, Any]]:
        """DependencyMapper can query existing dependencies and clause text during gap analysis."""
        return [
            tool for tool in self.STANDARD_TOOLS
            if tool["name"] in ("get_clause", "get_dependencies")
        ]

    async def _execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        """Execute tool calls for dependency mapping."""
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
        self._current_stack_id = input_data.contract_stack_id  # for tool-use access
        clauses = input_data.current_clauses

        # Phase 1: Single LLM call identifies ALL dependencies (explicit + semantic)
        system_prompt = self.prompts.get("dependency_mapper_system")
        user_prompt = self.prompts.get(
            "dependency_mapper_identify",
            clauses=self._format_clauses(clauses),
        )
        result = await self.call_llm(system_prompt, user_prompt)
        await self._report_progress("llm_analysis", 50, "LLM dependency analysis complete")

        # Extract known fields explicitly — avoids TypeError if LLM returns extra fields
        all_deps = [
            ClauseDependency(
                from_section=d["from_section"],
                to_section=d["to_section"],
                relationship_type=d["relationship_type"],
                description=d["description"],
                confidence=d.get("confidence", 0.8),
                detection_method="llm",
            )
            for d in result["dependencies"]
        ]

        # Deduplicate (LLM may identify same dependency via multiple reasoning paths)
        seen = set()
        deduped_deps = []
        for d in all_deps:
            key = (d.from_section, d.to_section, d.relationship_type)
            if key not in seen:
                seen.add(key)
                deduped_deps.append(d)

        await self._report_progress("llm_analysis", 60, f"Found {len(deduped_deps)} dependencies")

        # Phase 2: PostgreSQL
        await self._sync_dependencies(input_data.contract_stack_id, clauses, deduped_deps)
        await self._report_progress("db_sync", 100, "Dependency graph updated")

        return DependencyMapOutput(
            contract_stack_id=input_data.contract_stack_id,
            dependencies=deduped_deps,
            total_nodes=len(clauses),
            total_edges=len(deduped_deps),
            db_synced=True,
        )

    async def _sync_dependencies(self, stack_id: UUID, clauses: list[CurrentClause], deps: list[ClauseDependency]) -> None:
        """Write clause dependencies to PostgreSQL clause_dependencies table.

        Uses a single transaction for atomicity. Relies on indexes:
            idx_clause_deps_stack, idx_clause_deps_from, idx_clause_deps_to
        (created via Alembic migration, see technical_spec.md).
        """
        async with self.db.acquire() as conn:
            async with conn.transaction():
                for dep in deps:
                    rel_type = dep.relationship_type.value.upper()
                    if rel_type not in self._VALID_REL_TYPES:
                        raise ValueError(f"Unexpected relationship type: {rel_type}")

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
```

### Prompt Files

**`prompt/dependency_mapper_system.txt`**
```
You are an expert at identifying ALL dependencies between clauses in clinical trial agreements.

Analyze the provided clauses and identify every dependency — both explicit cross-references and implicit semantic dependencies.

Types of dependencies to identify:
1. EXPLICIT REFERENCES: "as defined in Section X", "pursuant to Exhibit B", "per Schedule A"
   - relationship_type: "references"
2. FUNCTIONAL: Payment terms depend on budget; budget depends on visit schedule
   - relationship_type: "depends_on"
3. CONDITIONAL: "Subject to Section Y" or "Notwithstanding Section Z"
   - relationship_type: "depends_on"
4. TEMPORAL: Insurance duration depends on study completion date
   - relationship_type: "depends_on"
5. DEFINITIONAL: Terms defined in one section used in another
   - relationship_type: "references"
6. FINANCIAL: Budget exhibits depend on procedure costs, payment terms, holdback provisions
   - relationship_type: "depends_on"

For each dependency, provide a confidence score (0.0-1.0). Explicit textual references should have confidence >= 0.95. Inferred semantic dependencies should have confidence 0.7-0.9.

Do NOT include self-references (a section referencing itself).
Deduplicate — if the same pair appears via multiple reasoning paths, include it only once.

Before producing your final answer, reason step-by-step about your analysis.
Include your reasoning in the 'reasoning' field of the JSON output.

Return ONLY valid JSON.
```

**`prompt/dependency_mapper_identify.txt`**
```
Identify ALL dependencies (explicit references AND semantic) between these clauses:

{clauses}

Before producing your final answer, reason step-by-step about your analysis.
Include your reasoning in the 'reasoning' field of the JSON output.

Return JSON with all dependencies:
{{
  "reasoning": "your step-by-step analysis...",
  "dependencies": [
    {{
      "from_section": "7.2",
      "to_section": "Exhibit B-2",
      "relationship_type": "depends_on",
      "description": "Payment terms reference budget amounts in Exhibit B-2",
      "confidence": 0.9
    }}
  ]
}}
```

### Expected HEARTBEAT-3 Dependency Graph

Key relationships the mapper should discover:

```
Payment (7.2) ──DEPENDS_ON──► Budget Exhibit (B/B-1/B-2)
Budget Exhibit ──DEPENDS_ON──► Visit Schedule (protocol)
Insurance (11.x) ──DEPENDS_ON──► Study Duration (timeline)
Indemnification (12.x) ──DEPENDS_ON──► Data Retention (9.x)
Data Retention (9.x) ──DEPENDS_ON──► Regulatory Requirements
PI Obligations ──DEPENDS_ON──► Personnel (named PI)
Cardiac MRI (from Amend 1) ──DEPENDS_ON──► Visit Schedule
Holdback (7.4) ──DEPENDS_ON──► Payment Terms (7.2)
```

---

## 4. Known Limitations and Implementation Notes

### Token Budget for ConflictDetectionAgent

The ConflictDetectionAgent sends ALL clauses to the LLM in a single call. For HEARTBEAT-3 (~20 sections, ~40K-60K input tokens), this fits within Claude Opus's 200K context window. For larger contract stacks (50+ sections), input may exceed the context window. **Implementation should add token estimation before the LLM call:**

```python
# Rough estimation: ~4 chars per token for English legal text
estimated_tokens = len(formatted_prompt) // 4
if estimated_tokens > 150_000:  # leave room for output
    # Split into per-category analysis with a cross-category synthesis pass
    ...
```

### Amendment Context for Buried Change Detection

The `CurrentClause` model currently carries `source_document_label` and `effective_date` but does NOT carry the amendment type/rationale (e.g., "covid_protocol"). This limits the ConflictDetectionAgent's ability to detect "buried changes" (Pain Point #1) — it can see that payment terms were last modified by Amendment 3 but cannot determine that Amendment 3 was primarily a COVID protocol amendment.

**Recommended enhancement for implementation:** Add `amendment_context: Optional[str]` to the `CurrentClause` model, populated by the orchestrator from `AmendmentTrackOutput.amendment_type` + `AmendmentTrackOutput.rationale`. This gives the ConflictDetection LLM the signal that a payment change was "buried" inside a COVID amendment.

### Cross-Section Override Interactions

The OverrideResolutionAgent processes each section's amendment chain independently. This is correct — it cannot detect that modifying Section A semantically invalidates Section B. That responsibility falls to the ConflictDetectionAgent (downstream), which has the full set of resolved clauses + dependency graph to identify such cross-section conflicts.
