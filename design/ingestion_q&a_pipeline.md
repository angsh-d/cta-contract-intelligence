# ContractIQ: Ingestion & Query Pipeline Documentation

> Complete pipeline reference for the 6-stage document ingestion pipeline and 4-step query pipeline.
> Source of truth: `backend/app/agents/orchestrator.py` and individual agent implementations.

---

## Architecture Overview

```
                        ┌──────────────────────────────────┐
                        │         FastAPI Application       │
                        │   /api/v1/contract-stacks         │
                        │   /api/v1/queries                 │
                        └───────────────┬──────────────────┘
                                        │
                           ┌────────────▼────────────┐
                           │    AgentOrchestrator     │
                           │  (Singleton via Lifespan)│
                           └────────────┬────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
              ▼                         ▼                         ▼
     ┌────────────────┐       ┌────────────────┐       ┌────────────────┐
     │    TIER 1       │       │    TIER 2       │       │    TIER 3       │
     │   INGESTION     │       │   REASONING     │       │   ANALYSIS      │
     │                 │       │                 │       │                 │
     │ 1. DocParser    │       │ 4. Override     │       │ 7. RippleEffect │
     │ 2. AmendTracker │       │ 5. DepMapper    │       │                 │
     │ 3. TempSequencer│       │ 6. ConflictDet  │       │                 │
     └────────────────┘       └────────────────┘       └────────────────┘

     ┌─────────────────────────────────────────────────────┐
     │                   QUERY PIPELINE                     │
     │  QueryRouter → Retrieval → Dispatch → Synthesizer    │
     └─────────────────────────────────────────────────────┘
```

---

## Ingestion Pipeline: `process_contract_stack()`

### Progress Map

```
 0% ──────── 30% ──── 50% ── 55% ────────── 80% ────── 90% ──── 100%
  │           │        │      │              │          │         │
  ▼           ▼        ▼      ▼              ▼          ▼         ▼
 Parse     Track    Sequence Override    Dependency  Conflict  Done
 (parallel) (seq)   (single) (parallel)  Mapping    Detection
```

| Stage | % Range | Agent | Concurrency | Checkpoint Resume |
|-------|---------|-------|-------------|-------------------|
| 1. Document Parsing | 0-30% | DocumentParserAgent | **Parallel** (all docs) | Yes |
| 2. Amendment Tracking | 30-50% | AmendmentTrackerAgent | **Sequential** (needs prior results) | Yes |
| 3. Temporal Sequencing | 50-55% | TemporalSequencerAgent | Single call | No (always re-runs) |
| 4. Override Resolution | 55-80% | OverrideResolutionAgent | **Parallel** (per section) | Yes |
| 5. Dependency Mapping | 80-90% | DependencyMapperAgent | Single call | Yes |
| 6. Conflict Detection | 90-100% | ConflictDetectionAgent | Single call | Yes |

### Checkpoint Resume

The orchestrator checks PostgreSQL before each stage. If prior results exist, it skips the stage and loads cached data:

```python
stage1_done = await self._check_stage_complete(stack_id, "document_parsing")
if stage1_done:
    parsed_outputs = await self._load_parsed_outputs_from_db(stack_id, documents)
```

Checks per stage:
- **Parsing**: `documents.processed = TRUE` count matches total document count
- **Tracking**: `amendments` table has rows for this stack
- **Sequencing**: `document_supersessions` table has rows (always re-runs regardless)
- **Resolution**: `clauses.is_current = TRUE` rows exist
- **Dependencies**: `clause_dependencies` rows exist
- **Conflicts**: `conflicts` rows exist

---

## Stage 1: Document Parsing (0-30%)

**Agent**: `DocumentParserAgent`
**File**: `backend/app/agents/document_parser.py`
**LLM**: Azure OpenAI GPT-5.2 (role=extraction), Gemini fallback
**Config**: `max_output_tokens=16000`, `timeout_seconds=300`, `verification_threshold=0.80`

### What It Does

Transforms raw PDFs into structured sections with metadata. All documents are parsed **in parallel** via `asyncio.gather()`.

### Process Flow

```
PDF File
   │
   ├─► Text Extraction (PyMuPDF) ──► raw_text + page_count
   │
   ├─► Table Extraction (pdfplumber) ──► list[ParsedTable]
   │
   ├─► Chunking (if text > 100K chars)
   │     └─► 50K-char chunks with 2K overlap
   │
   ├─► LLM Structuring
   │     ├─ System: document_parser_system.txt
   │     ├─ User: document_parser_extraction.txt ({document_type}, {raw_text})
   │     └─► JSON: sections[], metadata{}, extraction_confidence
   │
   ├─► Section Deduplication (when chunked)
   │
   └─► pgvector Upsert (section_embeddings table on NeonDB, is_resolved=FALSE)
         └─ Embeds section text via Gemini gemini-embedding-001 (768-dim, RETRIEVAL_DOCUMENT)
         └─ Keyed by: (contract_stack_id, document_id, section_number)
         └─ Purpose: Checkpoint fallback — per-document raw sections for pipeline resume
```

### Input / Output

**Input** (`DocumentParseInput`):
- `document_id`: UUID
- `file_path`: str (path to PDF)
- `document_type`: CTA | AMENDMENT | EXHIBIT
- `contract_stack_id`: UUID

**Output** (`DocumentParseOutput`):
- `document_id`: UUID
- `metadata`: DocumentMetadata (document_type, effective_date, amendment_number, parties, study_protocol)
- `sections`: list[ParsedSection] (section_number, section_title, text, page_numbers)
- `tables`: list[ParsedTable] (table_id, caption, headers, rows, page_number)
- `extraction_confidence`: float (0.0-1.0)

### Error Handling
- JSON parse failure: 3-tier extraction (direct parse, code block extraction, raise)
- Empty sections: raises `DocumentExtractionError`

### Database Writes
- Updates `documents.processed = TRUE` and `documents.metadata` in PostgreSQL
- Upserts checkpoint embeddings to pgvector (section_embeddings table, is_resolved=FALSE)

---

## Stage 2: Amendment Tracking (30-50%)

**Agent**: `AmendmentTrackerAgent`
**File**: `backend/app/agents/amendment_tracker.py`
**LLM**: Azure OpenAI GPT-5.2 (role=complex_reasoning), Gemini fallback
**Config**: `max_output_tokens=8192`, `timeout_seconds=180`, `verification_threshold=0.75`

### Why Sequential

Each amendment is processed one at a time because later amendments may reference earlier ones. Amendment 3 might say "as previously amended by Amendment 1" -- this requires Amendment 1's output as `prior_amendments` context.

### Process Flow

```
For each amendment (sorted by effective_date):
   │
   ├─► LLM Modification Identification
   │     ├─ System: amendment_tracker_system.txt
   │     ├─ User: amendment_tracker_analysis.txt
   │     │    ({amendment_number}, {amendment_text}, {original_sections}, {prior_modifications})
   │     └─► Identifies: section_number, modification_type, original_text, new_text
   │
   ├─► Buried Change Scan (adversarial)
   │     ├─ System: amendment_tracker_buried_scan.txt
   │     ├─ User: amendment_tracker_buried_scan_input.txt
   │     └─► Catches changes hidden in unrelated sections
   │
   └─► Self-Verification
         └─ Validates modifications reference real sections from original CTA
```

### Five Modification Patterns

| Type | Example |
|------|---------|
| `COMPLETE_REPLACEMENT` | "Section X is hereby deleted in its entirety and replaced..." |
| `SELECTIVE_OVERRIDE` | "Section X is amended by deleting 'Y' and substituting 'Z'" |
| `ADDITION` | "A new Section X.Y is hereby added..." |
| `DELETION` | "Section X is hereby deleted" |
| `EXHIBIT_REPLACEMENT` | "Exhibit B is hereby replaced by Exhibit B-1" |

### Input / Output

**Input** (`AmendmentTrackInput`):
- `amendment_document_id`: UUID
- `amendment_number`: int
- `amendment_sections`: list[ParsedSection] (from Stage 1)
- `amendment_tables`: list[ParsedTable] (critical for budget exhibits)
- `original_sections`: list[ParsedSection] (CTA sections)
- `original_tables`: list[ParsedTable]
- `prior_amendments`: list[AmendmentTrackOutput] (sequential context)

**Output** (`AmendmentTrackOutput`):
- `amendment_document_id`: UUID
- `amendment_number`: int
- `effective_date`: date | None
- `amendment_type`: str (e.g., "protocol_change", "budget_revision", "pi_change")
- `rationale`: str
- `modifications`: list[Modification]
- `sections_modified`: list[str]
- `exhibits_affected`: list[str]
- `extraction_confidence`: float

### Database Writes
- Inserts into `amendments` table (with upsert on `uq_amendments_stack_doc` constraint)
- Stores modifications as JSON blob

### HEARTBEAT-3 Pain Points Detected

| Amendment | Expected Detection |
|-----------|-------------------|
| Amendment 1 | New cardiac MRI visit section |
| Amendment 2 | PI name change, Exhibit B -> B-1 |
| **Amendment 3** | **Net 30 -> Net 45 buried in COVID section** (Pain Point #1) |
| Amendment 4 | Visit schedule changes |
| Amendment 5 | Study timeline extension, insurance ambiguity |

---

## Stage 3: Temporal Sequencing (50-55%)

**Agent**: `TemporalSequencerAgent`
**File**: `backend/app/agents/temporal_sequencer.py`
**LLM**: Azure OpenAI GPT-5.2 (role=extraction), Gemini fallback
**Config**: `max_output_tokens=4096`, `timeout_seconds=60`, `verification_threshold=0.80`

### Why LLM-First (Not Simple Date Sort)

The sequencer uses LLM reasoning because temporal ordering is not always chronological:
- Retroactive amendments (executed later, effective earlier)
- Conditional effectiveness ("upon IRB approval")
- Non-chronological precedence ("Notwithstanding the date...")
- Branching supersession (Amendment 5 may supersede Amendment 3 directly, skipping 4)

### Process Flow

```
All parsed documents
   │
   ├─► Date Inference (for documents with missing dates)
   │     ├─ Prompt: temporal_sequencer_date_inference.txt
   │     └─► Infers effective_date and amendment_number from filename/text
   │
   ├─► LLM Temporal Reasoning (PRIMARY)
   │     ├─ System: temporal_sequencer_system.txt
   │     ├─ User: temporal_sequencer_ordering.txt ({documents_json})
   │     └─► chronological_order, supersessions, warnings
   │
   ├─► Deterministic Validation (CROSS-CHECK)
   │     ├─ Sorts by (effective_date, is_cta?, amendment_number)
   │     ├─ Compares LLM order vs date-based sort
   │     └─ LLM order is authoritative if they diverge
   │
   ├─► Version Tree Construction
   │     └─ Tree with root=CTA, nodes=amendments, edges=supersedes
   │
   └─► PostgreSQL SUPERSEDES Sync
         └─ Writes to document_supersessions table atomically
```

### Output

- `chronological_order`: list[UUID] (ordered document IDs)
- `version_tree`: VersionTree (root_document_id, amendments[])
- `timeline`: list[TimelineEvent] (document_id, event_date, label)
- `doc_label_map`: derived from timeline for downstream agent use (e.g., "Amendment 3")

---

## Stage 4: Override Resolution (55-80%)

**Agent**: `OverrideResolutionAgent`
**File**: `backend/app/agents/override_resolution.py`
**LLM**: Azure OpenAI GPT-5.2 (role=complex_reasoning), Gemini fallback
**Config**: `max_output_tokens=8192`, `timeout_seconds=180`, `verification_threshold=0.75`

### What It Does

For each CTA section, applies all amendments in chronological order to determine the **current authoritative text** with full provenance (source chain).

### Process Flow

```
For each CTA section (parallel via asyncio.gather):
   │
   ├─► Build Amendment Chain
   │     └─ Matches section_number across all AmendmentTrackOutputs
   │     └─ Sorts amendments by effective_date
   │
   ├─► LLM Application
   │     ├─ System: override_resolution_system.txt
   │     ├─ User: override_resolution_apply.txt
   │     │    ({section_number}, {original_text}, {original_source}, {amendment_chain})
   │     └─► current_text, source_chain[], confidence, clause_category
   │
   └─► Source Chain Construction
         └─ Validates each link, tracks last_modified_by document_id
```

### Input / Output

**Input** (`OverrideResolutionInput`):
- `contract_stack_id`: UUID
- `section_number`: str
- `original_clause`: ParsedSection
- `original_document_id`: UUID
- `original_document_label`: str (e.g., "Original CTA")
- `amendments`: list[AmendmentForSection] (sorted by effective_date)

**Output** (`OverrideResolutionOutput`):
- `clause_version`: ClauseVersion
  - `section_number`, `section_title`, `current_text`
  - `source_chain`: list[SourceChainLink] (stage, document_label, text)
  - `last_modified_by`: UUID (document that last changed this clause)
  - `last_modified_date`: date
  - `confidence`: float
  - `clause_category`: str

### Database Writes
- Upserts into `clauses` table with `is_current = TRUE`
- Stores `source_chain` as JSON for full provenance tracking

### Downstream Data Flow

After resolution, the orchestrator builds `CurrentClause` objects used by Stages 5 and 6:

```python
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
```

### Resolved-Clause Embedding (Query Search Index)

After saving resolved clauses to PostgreSQL, the orchestrator embeds them into pgvector with `is_resolved=TRUE`. These are the **only** embeddings searched during query-time semantic search.

```python
# Orchestrator calls after _save_resolved_clauses():
await self._embed_resolved_clauses(contract_stack_id, current_clauses)
# → VectorStore.upsert_resolved_clauses() with enriched text:
#   "Section 7.2 — Payment Terms\nCategory: financial\n{current_text}"
#   Keyed by (contract_stack_id, section_number), is_resolved=TRUE
```

On checkpoint resume, the orchestrator checks `has_resolved_embeddings()` and re-embeds if missing (e.g., prior crash after saving clauses but before embedding).

---

## Stage 5: Dependency Mapping (80-90%)

**Agent**: `DependencyMapperAgent`
**File**: `backend/app/agents/dependency_mapper.py`
**LLM**: Azure OpenAI GPT-5.2 (role=complex_reasoning), Gemini fallback
**Config**: `max_output_tokens=16000`, `temperature=0.1`, `verification_threshold=0.75`

### What It Does

Analyzes all current clauses to identify inter-clause dependencies and writes the dependency graph to PostgreSQL. This graph powers ripple effect analysis.

### Relationship Types

| Type | Meaning |
|------|---------|
| `depends_on` | A requires B to be valid |
| `references` | A mentions/cross-references B |
| `modifies` | A changes/amends B |
| `replaces` | A supersedes B entirely |
| `amends` | A partially changes B |
| `conflicts_with` | A contradicts B |
| `supersedes` | A is a later version of B |

### Process Flow

```
All current clauses (from Stage 4)
   │
   ├─► LLM Dependency Identification
   │     ├─ System: dependency_mapper_system.txt
   │     ├─ User: dependency_mapper_identify.txt ({clauses})
   │     └─► dependencies[]: {from_section, to_section, relationship_type, description, confidence}
   │
   ├─► Deduplication
   │     └─ Removes duplicate edges (same from, to, relationship_type)
   │
   └─► PostgreSQL Sync
         ├─ Clears stale rows for this stack
         └─ Inserts all new dependencies atomically
```

### Database Writes
- Clears existing `clause_dependencies` rows for this contract stack
- Inserts new dependency edges with: `from_clause_id`, `to_clause_id`, `relationship_type`, `description`, `confidence`, `detection_method="llm"`

### Recursive CTE for Multi-Hop Traversal

The dependency graph enables recursive queries (used by Stage 7 - Ripple Effect):

```sql
WITH RECURSIVE dependency_chain AS (
  SELECT cd.from_clause_id, cd.to_clause_id, 1 AS hop, ARRAY[cd.from_clause_id] AS path
  FROM clause_dependencies cd
  WHERE cd.from_clause_id = $changed_clause_id AND cd.contract_stack_id = $stack_id
  UNION ALL
  SELECT cd.from_clause_id, cd.to_clause_id, dc.hop + 1, dc.path || cd.from_clause_id
  FROM clause_dependencies cd
  JOIN dependency_chain dc ON cd.from_clause_id = dc.to_clause_id
  WHERE dc.hop < 5 AND cd.to_clause_id != ALL(dc.path)
)
SELECT DISTINCT ON (to_clause_id) * FROM dependency_chain;
```

---

## Stage 6: Conflict Detection (90-100%)

**Agent**: `ConflictDetectionAgent`
**File**: `backend/app/agents/conflict_detection.py`
**LLM**: Azure OpenAI GPT-5.2 (role=complex_reasoning), Gemini fallback
**Config**: `max_output_tokens=16000`, `timeout_seconds=300`, `temperature=0.2`, `verification_threshold=0.70`

### Conflict Types

| Type | Description |
|------|-------------|
| `contradiction` | Two clauses state opposite things |
| `ambiguity` | Clause meaning is unclear |
| `gap` | Referenced but missing content |
| `inconsistency` | Temporal or logical misalignment |
| `buried_change` | Critical modification hidden in unrelated section |
| `stale_reference` | References outdated PI/institution |
| `temporal_mismatch` | Effective dates don't align with intent |

### Process Flow

```
Current clauses + dependency graph + contract context
   │
   ├─► Group Clauses by Category
   │     └─ payment, insurance, personnel, regulatory, etc.
   │
   ├─► LLM Conflict Analysis
   │     ├─ System: conflict_detection_system.txt
   │     ├─ User: conflict_detection_analyze.txt
   │     │    ({grouped_clauses}, {context}, {dependency_graph})
   │     └─► conflicts[]: {type, severity, affected_sections, description, recommendation}
   │
   └─► Severity Summarization
         └─ Counts by severity: CRITICAL, HIGH, MEDIUM, LOW
```

### Input

- `current_clauses`: list[CurrentClause] (from Stage 4)
- `contract_stack_context`: ContractStackContext (study_name, sponsor, site, therapeutic_area, dates)
- `dependency_graph`: list[ClauseDependency] (from Stage 5)

### Output

- `conflicts`: list[DetectedConflict] (conflict_type, severity, affected_sections, description, recommendation, evidence, pain_point_id)
- `conflict_summary`: counts by severity

### Database Writes
- Upserts into `conflicts` table (with constraint on `uq_conflicts_stack_conflict_id`)
- Stores evidence as JSON

### HEARTBEAT-3 Expected Conflicts

| Pain Point | Conflict Type | Severity |
|-----------|--------------|----------|
| #1: Buried Payment Change (Net 30 -> Net 45) | `buried_change` | HIGH |
| #2: Budget Exhibit Evolution (B -> B-1 -> B-2) | `inconsistency` | MEDIUM |
| #3: Insurance Coverage Gap | `gap` | HIGH |
| #4: Cross-Reference Confusion (cardiac MRI) | `inconsistency` | MEDIUM |
| #5: PI Change + Budget Ambiguity | `stale_reference` | MEDIUM |

---

## Pipeline Completion

After Stage 6, the orchestrator:

1. Saves the `TraceContext` (all LLM call metrics) to `pipeline_traces` table
2. Updates `contract_stacks.processing_status = 'completed'`
3. Returns summary:

```python
{
    "clauses_processed": <count from clauses WHERE is_current = TRUE>,
    "conflicts_detected": <count from conflicts>,
    "dependencies_mapped": <count from clause_dependencies>,
    "trace": {
        "total_input_tokens": ...,
        "total_output_tokens": ...,
        "total_llm_calls": ...
    }
}
```

---

## Query Pipeline: `handle_query()`

### Four Steps

```
User Query
     │
     ▼
┌────────────────────────────────┐
│ Step 1: Cache Check + Classify  │  < 2s target
│   - SHA256 hash for cache key  │
│   - QueryRouter: type + entities│
└────────────────────────────────┘
     │
     ▼
┌────────────────────────────────┐
│ Step 2: Multi-Source Retrieval   │
│   - pgvector semantic search   │
│     (is_resolved=TRUE only)    │
│   - PostgreSQL batch lookup    │
│   - Entity-based section match │
└────────────────────────────────┘
     │
     ▼
┌────────────────────────────────┐
│ Step 3: Dispatch by Type        │
│   - truth → TruthSynthesizer   │
│   - conflict → TruthSynthesizer│
│   - ripple → RippleEffect +    │
│              TruthSynthesizer   │
└────────────────────────────────┘
     │
     ▼
┌────────────────────────────────┐
│ Step 4: Cache + Return          │
│   - Cache result (1hr TTL)     │
│   - Return: answer + sources + │
│     confidence + caveats       │
└────────────────────────────────┘
```

---

### Step 1: QueryRouter (Classification)

**Agent**: `QueryRouter`
**File**: `backend/app/agents/query_router.py`
**LLM**: Azure OpenAI GPT-5.2 (role=classification), Gemini fallback
**Config**: `max_output_tokens=1024`, `temperature=0.0`, `verification_threshold=0.85`

**Cache Check**: Before routing, checks in-memory cache using `query:{stack_id}:{sha256(query)}` key.

**Query Types**:
| Type | Purpose | Example |
|------|---------|---------|
| `truth_reconstitution` | Current state of a clause | "What are the current payment terms?" |
| `conflict_detection` | Find conflicts/risks | "Are there any conflicts in the budget?" |
| `ripple_analysis` | Impact of proposed changes | "What if we extend the study by 6 months?" |
| `general` | General questions | "How many amendments are there?" |

**Prompts**:
- System: `query_router_classify.txt`
- User: `query_router_input.txt` (`{query_text}`)

**Output** (`QueryRouteOutput`):
- `query_type`: QueryType
- `extracted_entities`: list[str] (section numbers, names, topics)
- `confidence`: float
- `routing_latency_ms`: int

---

### Step 2: Multi-Source Retrieval

**Location**: `orchestrator._retrieve_clauses()`

Three retrieval sources merged and deduplicated:

```
1. pgvector Semantic Search (section_embeddings table, is_resolved=TRUE)
   └─ Embeds query via Gemini gemini-embedding-001 (RETRIEVAL_QUERY)
   └─ WHERE contract_stack_id = $2 AND is_resolved = TRUE
   └─ ORDER BY embedding <=> query_vector, LIMIT 10
   └─ Returns section_numbers for resolved current-truth clauses only

2. Entity-Based Matching
   └─ Adds extracted_entities as section_numbers

3. PostgreSQL Batch Lookup
   └─ Single query: WHERE section_number = ANY($1) AND is_current = TRUE
   └─ Returns CurrentClause objects
```

**Conflict Retrieval**: Separately loads all conflicts for the stack from the `conflicts` table.

---

### Step 3: Dispatch by Query Type

#### Truth/Conflict/General Queries -> TruthSynthesizer

**Agent**: `TruthSynthesizer`
**File**: `backend/app/agents/truth_synthesizer.py`
**LLM**: Azure OpenAI GPT-5.2 (role=synthesis), Gemini fallback
**Config**: `max_output_tokens=8192`, `temperature=0.1`, `verification_threshold=0.80`

**Prompts**:
- System: `truth_synthesizer_answer.txt`
- User: `truth_synthesizer_input.txt` (`{query_text}`, `{query_type}`, `{relevant_clauses}`, `{known_conflicts}`)

**Rules enforced by prompt**:
1. Base answer ONLY on provided clauses
2. ALWAYS cite sources (document name, section number, effective date)
3. State ambiguities clearly with both positions
4. Say explicitly if information is unavailable
5. Include confidence level
6. List caveats

**Output** (`TruthSynthesisOutput`):
- `answer`: str (comprehensive answer with inline citations)
- `sources`: list[SourceCitation] (document_id, document_name, section_number, relevant_text, effective_date)
- `confidence`: float
- `caveats`: list[str]

#### Ripple Analysis Queries -> RippleEffectAnalyzer + TruthSynthesizer

**Agent**: `RippleEffectAnalyzerAgent` (Tier 3)
**File**: `backend/app/agents/ripple_effect.py`
**LLM**: Azure OpenAI GPT-5.2 (role=complex_reasoning), Gemini fallback
**Config**: `max_output_tokens=16000`, `timeout_seconds=300`, `temperature=0.2`, `verification_threshold=0.70`

**Two-phase process**:

1. **Extract proposed change** (via QueryRouter LLM call):
   - LLM identifies which clause the user wants to change
   - Structures it as `ProposedChange(section_number, current_text, proposed_text, change_description)`

2. **Ripple effect analysis** (via RippleEffectAnalyzerAgent):
   ```
   Proposed Change
        │
        ├─► Bidirectional Dependency Traversal (PostgreSQL recursive CTE)
        │     ├─ Outbound: what depends on this section? (5-hop max)
        │     └─ Inbound: what does this section depend on? (5-hop max)
        │
        ├─► Per-Hop LLM Analysis (iterative)
        │     ├─ System: ripple_effect_system.txt
        │     ├─ User: ripple_effect_hop_analysis.txt
        │     │    ({section_number}, {current_text}, {proposed_text}, {hop_number}, {hop_clauses})
        │     └─ Early termination if no impacts at hop N
        │
        └─► Recommendation Synthesis
              ├─ Prompt: ripple_effect_recommendations_input.txt
              └─► PrioritizedAction[], risks[], benefits[], financial_impact_estimate
   ```

3. **Synthesize into TruthSynthesisOutput format** (via `_synthesize_ripple_answer()`):
   - Uses `truth_synthesizer_ripple_input.txt` prompt
   - Converts ripple results into standard answer format for API consistency

---

### Step 4: Cache + Return

- Result cached via `InMemoryCache.setex(cache_key, 3600, answer_json)` (1-hour TTL)
- Cache key tracked via `InMemoryCache.sadd(f"cache_keys:{stack_id}", cache_key)` for SET-based invalidation
- When the ingestion pipeline re-runs, `invalidate_for_stack()` clears all cached queries

---

## Shared Infrastructure

### Agent Base Class

All agents extend `BaseAgent` (`backend/app/agents/base.py`):

| Feature | Description |
|---------|-------------|
| `call_llm()` | Structured JSON output via LLM provider (Azure OpenAI GPT-5.2 primary) |
| `run()` | Wraps `process()` with self-verification and confidence-gated re-processing |
| `_verify_output()` | Domain-specific output validation (overridden per agent) |
| `_reprocess_with_critique()` | Re-runs with critique feedback when confidence < threshold |
| `call_llm_with_tools()` | ReAct loop: LLM reasons, calls tools, observes (up to 5 turns) |
| Retry logic | Exponential backoff (2^attempt seconds), circuit breaker |
| Fallback provider | Automatic failover to Gemini if Azure OpenAI fails |
| Token budget validation | Prevents input + output exceeding context window |
| TraceContext | Records all LLM calls for cost tracking |

### LLM Concurrency

A shared `asyncio.Semaphore(5)` limits concurrent LLM calls across all agents to prevent rate limiting.

### Inter-Agent Communication

`InMemoryBlackboard` enables agents to publish findings that other agents can query:
- `buried_change`: AmendmentTracker publishes when it finds a buried change
- `low_confidence_clause`: OverrideResolution flags uncertain resolutions
- `clarification_request`: ConflictDetection requests re-analysis from upstream agents

### Database Layer

| Store | Purpose |
|-------|---------|
| **PostgreSQL (NeonDB)** | Structured data: contract_stacks, documents, clauses, amendments, clause_dependencies, document_supersessions, conflicts, pipeline_traces |
| **pgvector (NeonDB)** | Two-tier vector embeddings in `section_embeddings` table (Gemini `gemini-embedding-001`, 768-dim): `is_resolved=FALSE` for Stage 1 checkpoint fallback, `is_resolved=TRUE` for Stage 4 resolved-clause query search |
| **InMemoryCache** | Query result caching (1-hour TTL), replaces Redis |

### Key PostgreSQL Tables

| Table | Written By Stage | Purpose |
|-------|-----------------|---------|
| `documents` | Stage 1 | Document metadata + processed flag |
| `section_embeddings` (is_resolved=FALSE) | Stage 1 | Per-document checkpoint embeddings, keyed by (stack, doc, section) |
| `amendments` | Stage 2 | Amendment tracking results + modifications JSON |
| `document_supersessions` | Stage 3 | Version tree (which doc supersedes which) |
| `clauses` | Stage 4 | Current clause text + source chain provenance |
| `section_embeddings` (is_resolved=TRUE) | Stage 4 | Resolved-clause embeddings for query-time search, keyed by (stack, section) |
| `clause_dependencies` | Stage 5 | Dependency graph edges |
| `conflicts` | Stage 6 | Detected conflicts with evidence |
| `pipeline_traces` | Completion | LLM call metrics for cost tracking |

---

## Prompt File Inventory

### Tier 1: Ingestion

| File | Agent | Placeholders |
|------|-------|-------------|
| `document_parser_system.txt` | DocumentParser | - |
| `document_parser_extraction.txt` | DocumentParser | `{document_type}`, `{raw_text}` |
| `amendment_tracker_system.txt` | AmendmentTracker | - |
| `amendment_tracker_analysis.txt` | AmendmentTracker | `{amendment_number}`, `{amendment_text}`, `{original_sections}`, `{prior_modifications}` |
| `amendment_tracker_buried_scan.txt` | AmendmentTracker | - |
| `amendment_tracker_buried_scan_input.txt` | AmendmentTracker | (buried change detection input) |
| `temporal_sequencer_system.txt` | TemporalSequencer | - |
| `temporal_sequencer_ordering.txt` | TemporalSequencer | `{documents_json}` |
| `temporal_sequencer_date_inference.txt` | TemporalSequencer | (date/amendment inference) |

### Tier 2: Reasoning

| File | Agent | Placeholders |
|------|-------|-------------|
| `override_resolution_system.txt` | OverrideResolution | - |
| `override_resolution_apply.txt` | OverrideResolution | `{section_number}`, `{original_text}`, `{original_source}`, `{amendment_chain}` |
| `dependency_mapper_system.txt` | DependencyMapper | - |
| `dependency_mapper_identify.txt` | DependencyMapper | `{clauses}` |
| `conflict_detection_system.txt` | ConflictDetection | - |
| `conflict_detection_analyze.txt` | ConflictDetection | `{grouped_clauses}`, `{context}`, `{dependency_graph}` |

### Tier 3: Analysis

| File | Agent | Placeholders |
|------|-------|-------------|
| `ripple_effect_system.txt` | RippleEffect | - |
| `ripple_effect_hop_analysis.txt` | RippleEffect | `{section_number}`, `{current_text}`, `{proposed_text}`, `{hop_number}`, `{hop_clauses}` |
| `ripple_effect_recommendations.txt` | RippleEffect | - |
| `ripple_effect_recommendations_input.txt` | RippleEffect | `{section_number}`, `{current_text}`, `{proposed_text}`, `{impacts_json}` |

### Query Pipeline

| File | Agent | Placeholders |
|------|-------|-------------|
| `query_router_classify.txt` | QueryRouter | - |
| `query_router_input.txt` | QueryRouter | `{query_text}` |
| `truth_synthesizer_answer.txt` | TruthSynthesizer | - |
| `truth_synthesizer_input.txt` | TruthSynthesizer | `{query_text}`, `{query_type}`, `{relevant_clauses}`, `{known_conflicts}` |
| `truth_synthesizer_ripple_input.txt` | TruthSynthesizer | `{query_text}`, `{impact_summary}` |

### Cross-Cutting

| File | Purpose |
|------|---------|
| `self_verification_system.txt` | Self-critique role for confidence-gated re-processing |
| `reusability_analyzer_system.txt` | Phase 2 reusability analysis (future) |

**Total: 26 prompt files**

---

## Data Flow Summary

```
PDFs ──► Stage 1 (Parse) ──► ParsedSection[] + ParsedTable[] + DocumentMetadata
                                       │
                                       ▼
         Stage 2 (Track) ──► Modification[] per amendment (sequential)
                                       │
                                       ▼
         Stage 3 (Sequence) ──► chronological_order + version_tree + doc_label_map
                                       │
                                       ▼
         Stage 4 (Resolve) ──► ClauseVersion[] with source_chain provenance
                                       │
                                       ▼
         Stage 5 (Map Deps) ──► ClauseDependency[] graph in PostgreSQL
                                       │
                                       ▼
         Stage 6 (Conflicts) ──► DetectedConflict[] with severity + evidence
                                       │
                                       ▼
                              Pipeline Complete
                              (clauses, conflicts, dependencies stored)
                                       │
                                       ▼
         User Query ──► QueryRouter ──► Retrieval (pgvector + PG) ──►
                        TruthSynthesizer / RippleEffect ──► Answer + Citations
```
