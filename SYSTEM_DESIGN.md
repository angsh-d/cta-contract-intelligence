# ContractIQ — System Design Document

**Version:** 2.0 | **Updated:** February 2026 | **Status:** Production MVP

---

## 1. Overview

**ContractIQ** is an agentic contract intelligence platform for clinical trial agreements (CTAs). It performs **truth reconstitution**, **conflict detection**, and **ripple effect analysis** across contract stacks consisting of an original CTA plus multiple amendments.

The platform ingests PDF and DOCX contract documents, runs an 11-agent pipeline to extract, track, resolve, and analyze clause-level changes, then presents a consolidated "current truth" view with full provenance tracking back to source documents.

### Key Capabilities

| Capability | Description |
|---|---|
| **Truth Reconstitution** | Programmatic text assembly applies amendments in chronological order to produce the current-state clause text without LLM rewriting |
| **Conflict Detection** | Identifies 7 conflict types (contradiction, ambiguity, gap, inconsistency, buried change, stale reference, temporal mismatch) |
| **Ripple Effect Analysis** | Multi-hop dependency traversal (max 3 hops) to surface cascading impacts of proposed changes |
| **Consolidated Contract View** | WYSIWYG document editor showing the merged contract with amendment highlighting and source document provenance |
| **Natural Language Query** | Semantic search over resolved clauses with source citations, confidence scores, and caveats |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Frontend (React 19 + TypeScript)                │
│  TipTap WYSIWYG Editor  ·  Framer Motion  ·  Tailwind CSS v4        │
│  7-Tab Interface: Overview, Health, Timeline, Query, Conflicts,      │
│                   Ripple Effects, Consolidate                        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP/REST + WebSocket
┌──────────────────────────────▼──────────────────────────────────────┐
│                      API Layer (FastAPI)                              │
│  REST: /api/v1/*  ·  WebSocket: /api/v1/ws/jobs/{job_id}            │
│  Static SPA serving from frontend/dist/                              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                    Agent Orchestrator                                 │
│  6-Stage Ingestion Pipeline  ·  Query Pipeline  ·  PgCache           │
│  In-Memory Blackboard  ·  Checkpoint Resume  ·  Progress Broadcast   │
└───────┬──────────────┬───────────────┬──────────────┬───────────────┘
        │              │               │              │
┌───────▼──────┐ ┌─────▼──────┐ ┌──────▼─────┐ ┌─────▼──────────┐
│  Tier 1      │ │  Tier 2    │ │  Tier 3    │ │  Query         │
│  Ingestion   │ │  Reasoning │ │  Analysis  │ │  Pipeline      │
│  ─ Parser    │ │  ─ Override│ │  ─ Ripple  │ │  ─ Router      │
│  ─ Amendment │ │  ─ Conflict│ │  ─ Reuse   │ │  ─ Synthesizer │
│  ─ Temporal  │ │  ─ Depend. │ │  (planned) │ │                │
│  ─ Consolid. │ │            │ │            │ │                │
└───────┬──────┘ └─────┬──────┘ └──────┬─────┘ └─────┬──────────┘
        │              │               │              │
┌───────▼──────────────▼───────────────▼──────────────▼───────────────┐
│               PostgreSQL + pgvector (NeonDB)                         │
│  10 tables  ·  HNSW cosine index  ·  Two-tier embeddings             │
│  asyncpg connection pool  ·  cache_store for PgCache                 │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                       LLM Providers                                  │
│  Primary: Azure OpenAI GPT-5.2  ·  Fallback: Google Gemini          │
│  Embeddings: gemini-embedding-001 (768-dim)                          │
│  Circuit breaker  ·  Automatic failover  ·  Role-based routing       │
└─────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.13, FastAPI, asyncio, asyncpg, Pydantic v2 |
| **Frontend** | React 19, TypeScript, Vite, Tailwind CSS v4, Framer Motion, TipTap |
| **Database** | PostgreSQL 15+ (NeonDB), pgvector extension |
| **Primary LLM** | Azure OpenAI GPT-5.2 |
| **Fallback LLM** | Google Gemini (gemini-3-pro-preview) |
| **Embeddings** | Gemini gemini-embedding-001 (768 dimensions) |
| **Document Parsing** | PyMuPDF (PDF), python-docx (DOCX), pdfplumber (tables), mammoth (DOCX→HTML) |

---

## 3. Backend

### 3.1 Application Entry Point

`backend/app/main.py` — FastAPI application with lifespan context manager.

- **Startup**: Creates asyncpg pool → VectorStore → AgentOrchestrator. Stores on `app.state`.
- **Shutdown**: Closes PostgreSQL pool.
- **Static serving**: Built React SPA from `frontend/dist/` with SPA fallback (all unmatched routes → `index.html`).
- **CORS**: Allow all origins (development mode).
- **Health check**: `GET /health` — service status + AI availability boolean.

### 3.2 Agent System

All 11 agents extend `BaseAgent` (`backend/app/agents/base.py`, 802 lines), which provides:

| Capability | Description |
|---|---|
| **Structured output** | Tool-use / response_schema for constrained JSON decoding |
| **Self-verification** | `_verify_output()` hook with confidence-gated re-processing |
| **Provider failover** | Primary LLM fails → automatic switch to fallback provider |
| **Circuit breaker** | Shared per-provider; trips after 5 failures, recovers after 60s |
| **Token estimation** | Warns at 85% context window usage to prevent truncation |
| **Trace context** | `TraceContext` records tokens, latency, model, provider per call |
| **Retry with backoff** | Exponential backoff on transient errors (429, 500, 502, 503) |
| **Multi-turn reasoning** | `call_llm_conversation()` for iterative refinement (max 3 turns) |
| **Tool-use loop** | `call_llm_with_tools()` for ReAct-style agentic execution (max 5 turns) |

#### Agent Inventory

| Agent | File | Role | Key Behavior |
|---|---|---|---|
| **DocumentParser** | `document_parser.py` | extraction | Extracts text from PDF/DOCX, chunks if >50K chars, LLM structures into sections, embeds to pgvector |
| **AmendmentTracker** | `amendment_tracker.py` | complex_reasoning | Identifies modifications per amendment; adversarial buried-change scan; extracts section_number, modification_type, original_text, new_text |
| **TemporalSequencer** | `temporal_sequencer.py` | extraction | Infers missing effective dates, builds chronological order + version tree |
| **OverrideResolution** | `override_resolution.py` | complex_reasoning | **Programmatic text assembly** (no LLM rewriting); applies modifications in order; LLM only classifies result |
| **ConflictDetection** | `conflict_detection.py` | complex_reasoning | Tool-enabled; detects 7 conflict types; groups clauses by category |
| **DependencyMapper** | `dependency_mapper.py` | complex_reasoning | Identifies 9 relationship types; builds adjacency-list dependency graph |
| **RippleEffect** | `ripple_effect.py` | complex_reasoning | Multi-hop impact analysis (max 3 hops); severity scoring; recommendations |
| **ContractConsolidator** | `contract_consolidator.py` | synthesis | Builds hierarchical document structure from resolved clauses; identifies amended sections via source_chain |
| **QueryRouter** | `query_router.py` | classification | Classifies query into 4 types; extracts section entities |
| **TruthSynthesizer** | `truth_synthesizer.py` | synthesis | Synthesizes answer from retrieved clauses + conflicts; returns sources, confidence, caveats |
| **ReusabilityAnalyzer** | `reusability.py` | — | Placeholder (not implemented) |

### 3.3 LLM Provider System

`backend/app/llm_providers.py` — Three provider implementations behind `LLMProviderFactory`:

| Provider | Model | Use |
|---|---|---|
| **AzureOpenAIProvider** | GPT-5.2 | Primary for all roles |
| **GeminiProvider** | gemini-3-pro-preview | Fallback for all roles |
| **ClaudeProvider** | claude-sonnet-4-5 | Available but not in default routing |

**Role-based routing** maps agent roles (`extraction`, `complex_reasoning`, `classification`, `synthesis`, `embedding`) to `(primary_provider, fallback_provider)` pairs. All currently route to `(azure_openai, gemini)`.

**Embedding**: Gemini `gemini-embedding-001` — 768-dimensional vectors with task-type differentiation (`RETRIEVAL_DOCUMENT` for indexing, `RETRIEVAL_QUERY` for search).

### 3.4 Orchestrator & Pipeline

`backend/app/agents/orchestrator.py` (1,050 lines) — Central coordination hub.

#### 6-Stage Ingestion Pipeline

```
Stage 1          Stage 2           Stage 3            Stage 4
Document    →    Amendment    →    Temporal      →    Override
Parsing          Tracking          Sequencing         Resolution
(parallel)       (sequential)      (single call)      (parallel)

    Stage 5              Stage 6
→   Dependency      →    Conflict
    Mapping              Detection
    (single call)        (single call)
```

| Stage | Agent | Input | Output | DB Table | Checkpoint |
|---|---|---|---|---|---|
| 1 | DocumentParser | File paths | Sections + embeddings | `section_embeddings` (is_resolved=FALSE) | `documents.processed` count |
| 2 | AmendmentTracker | CTA + amendments | Modifications per amendment | `amendments` | Amendment count |
| 3 | TemporalSequencer | All documents | Timeline + version tree | `document_supersessions` | Supersession count |
| 4 | OverrideResolution | Per-section clause + amendment chain | Resolved clause with source_chain | `clauses` (is_current=TRUE) + `section_embeddings` (is_resolved=TRUE) | Clause count |
| 5 | DependencyMapper | All current clauses | Dependency graph | `clause_dependencies` | Dependency count |
| 6 | ConflictDetection | Clauses + dependencies + context | Conflict list | `conflicts` | Conflict count |

**Checkpoint resume**: Each stage checks if DB already has results. If found, skips agent execution and loads from DB. Enables restart after crash without re-running expensive LLM stages.

**Section matching** (Stage 4): Exact match → Parent match (mod "7.2" → CTA "7") → New section (ADDITION creates new clause).

**Cache invalidation**: Consolidated contract cache is invalidated after Stage 4 completes.

**Progress broadcasting**: Real-time progress events via WebSocket to all connected clients.

#### Query Pipeline

1. `QueryRouter` classifies query type
2. Semantic search over resolved embeddings (pgvector, `is_resolved=TRUE`)
3. If ripple_analysis → `RippleEffectAnalyzer`; else → `TruthSynthesizer`
4. Results cached in PgCache (30-day TTL)

#### Helper Components

- **InMemoryBlackboard**: Key-value store for inter-agent data sharing within a pipeline run.
- **PgCache**: PostgreSQL-backed cache (`cache_store` table) with key-value + set operations. Survives restarts. Used for consolidated contracts, query results, and ripple analyses.

### 3.5 API Routes

`backend/app/api/routes.py` (819 lines) — All endpoints under `/api/v1/`.

| Endpoint | Method | Description |
|---|---|---|
| `/contract-stacks` | POST | Create new contract stack |
| `/contract-stacks` | GET | List all stacks with counts |
| `/contract-stacks/{id}` | GET | Stack detail with document/clause/conflict counts |
| `/contract-stacks/{id}/documents` | POST | Upload document (PDF/DOCX, max 50MB) |
| `/contract-stacks/{id}/documents` | GET | List documents for stack |
| `/contract-stacks/{id}/process` | POST | Trigger 6-stage pipeline (returns job_id) |
| `/jobs/{job_id}/status` | GET | Poll pipeline job status |
| `/contract-stacks/{id}/query` | POST | Natural language query |
| `/contract-stacks/{id}/analyze/conflicts` | POST | Conflict detection with severity filter |
| `/contract-stacks/{id}/analyze/ripple-effects` | POST | Ripple effect analysis (cached) |
| `/contract-stacks/{id}/consolidated` | GET | Consolidated contract (?refresh=true to invalidate cache) |
| `/contract-stacks/{id}/timeline` | GET | Temporal timeline with supersessions |
| `/contract-stacks/{id}/clauses/{section}/history` | GET | Source chain + dependencies for a clause |
| `/contract-stacks/{id}/documents/{doc_id}/clauses` | GET | All clauses for a specific document |
| `/contract-stacks/{id}/documents/{doc_id}/pdf` | GET | Serve document file (PDF inline; DOCX converted to HTML via mammoth) |

**WebSocket**: `WS /api/v1/ws/jobs/{job_id}` — Real-time pipeline progress events (in-memory queue, no Redis).

### 3.6 Prompt Management

All 28 prompts stored as `.txt` files in `/prompt` with `{variable_name}` placeholders, loaded at runtime by `PromptLoader`. No hardcoded prompts in Python code.

| Category | Prompts |
|---|---|
| **System prompts** (role definitions) | `document_parser_system`, `amendment_tracker_system`, `temporal_sequencer_system`, `override_resolution_system`, `conflict_detection_system`, `dependency_mapper_system`, `ripple_effect_system`, `query_router_classify`, `truth_synthesizer_answer`, `contract_consolidator_system`, `self_verification_system` |
| **Task prompts** (user messages) | `document_parser_extraction`, `amendment_tracker_analysis`, `amendment_tracker_buried_scan`, `temporal_sequencer_date_inference`, `temporal_sequencer_ordering`, `override_resolution_verify`, `conflict_detection_analyze`, `dependency_mapper_identify`, `ripple_effect_hop_analysis`, `ripple_effect_recommendations`, `query_router_input`, `truth_synthesizer_input`, `truth_synthesizer_ripple_input`, `contract_consolidator_assemble` |

Key prompt rules enforced:
- **CLAUSE BODY ONLY RULE**: `new_text` and `original_text` must contain only clause body content — no section headers, no amendment instruction language.
- **INLINE AMENDMENT PATTERN**: Handles amendments that describe changes inline (e.g., "Section 3.1 amended to require...") by extracting substantive content.
- **VERBATIM TEXT RULE**: Extract exact text from documents — no paraphrasing or summarization.

---

## 4. Database

### 4.1 PostgreSQL + pgvector (NeonDB)

Connected via `EXTERNAL_DATABASE_URL` (asyncpg pool). 10 core tables + 1 cache table.

### 4.2 Schema

```sql
-- Core tables
contract_stacks     -- study metadata, processing_status
documents           -- uploaded files, file_path, effective_date, amendment_number, raw_text
clauses             -- resolved clause text, source_chain (JSONB), is_current, source_document_id
amendments          -- modifications (JSONB), sections_modified, amendment_number
conflicts           -- conflict_type, severity, evidence (JSONB), recommendation
clause_dependencies -- from_clause_id, to_clause_id, relationship_type
document_supersessions -- predecessor → successor version tree
queries             -- query history
pipeline_traces     -- LLM call traces (tokens, latency, provider)

-- Vector table
section_embeddings  -- embedding vector(768), HNSW cosine index
                    -- is_resolved=FALSE: Stage 1 per-document sections
                    -- is_resolved=TRUE: Stage 4 resolved clauses (query-time search)

-- Cache table
cache_store         -- key/value + set operations (PgCache)
```

### 4.3 Two-Tier Embedding Architecture

| Tier | `is_resolved` | Written By | Purpose | Key |
|---|---|---|---|---|
| Checkpoint | `FALSE` | Stage 1 (Document Parser) | Raw per-document sections; fallback if Stage 4 hasn't run | (stack_id, document_id, section_number) |
| Query-Ready | `TRUE` | Stage 4 (Override Resolution) | Post-resolution current-truth clauses | (stack_id, section_number) |

Semantic search at query time filters on `is_resolved = TRUE` only, ensuring users always search against the resolved current state.

### 4.4 Migrations

5 Alembic migrations in `alembic/versions/`:

1. `001_initial_schema` — 9 core tables
2. `002_add_constraints_and_indexes` — unique constraints, status tracking
3. `003_add_pgvector_section_embeddings` — pgvector extension + HNSW index
4. `004_fix_section_embeddings_for_resolved_clauses` — `is_resolved` boolean + partial unique indexes
5. `005_add_raw_text_to_documents` — `raw_text` TEXT column on documents

---

## 5. Frontend

### 5.1 Stack

React 19 + TypeScript + Vite + Tailwind CSS v4 + Framer Motion. **Pure greyscale Apple-inspired design** — no colors, only blacks, greys, whites. SF Pro fonts, glass morphism, motion animations.

### 5.2 Pages

| Page | File | Description |
|---|---|---|
| **Landing** | `Landing.tsx` | Animated 6-stage pipeline visualization, hero section |
| **Dashboard** | `Dashboard.tsx` | Stats cards, recent activity, quick actions |
| **Contracts** | `StacksList.tsx` | Card grid of contract stacks, create/search |
| **Contract Detail** | `StackDetail.tsx` (3,500 lines) | 7-tab interface (see below) |

### 5.3 Contract Detail — 7-Tab Interface

#### Overview Tab
- Stack metadata (study name, sponsor, site, therapeutic area, protocol)
- Document upload (drag-drop + file picker)
- Pipeline trigger with real-time progress via WebSocket
- Stats: document count, clause count, conflict count

#### Contract Health Tab
- Clause category breakdown
- Conflict severity summary
- Confidence score distribution

#### Timeline Tab
- Horizontal scrollable timeline: MSA → Amd 1 → Amd 2 → ... → Amd N
- Documents ordered by `amendment_number` (JOIN with amendments table), with `effective_date` fallback
- Click any document → modal with extracted clauses, source chain, conflicts
- Year markers along timeline axis

#### Query Tab
- Natural language query input
- Response with answer, source citations, confidence score, caveats
- Scrollable query history

#### Conflicts Tab
- Severity filter (critical / high / medium / low)
- Conflict cards with type badge, description, affected clauses
- Evidence section with source document references
- Recommendations per conflict

#### Ripple Effects Tab
- Proposed change form (section number, current text, proposed change)
- Multi-hop impact visualization
- Impacted clause cards with severity scoring
- Actionable recommendations

#### Consolidate Tab (TipTap WYSIWYG Editor)

The flagship feature — a Word-like document editor showing the merged contract:

- **TipTap editor** (read-only by default, edit mode toggle in toolbar)
- **Hierarchical section rendering** — nested sections via recursive `buildDocumentHtml()`
- **Amendment highlighting** — Yellow `<mark>` elements on amended text with custom data attributes:
  - `data-section-number`, `data-source-document-id`, `data-amendment-source`, `data-amendment-description`
- **Custom TipTap mark** (`AmendmentHighlight`) — Preserves data-* attributes that default TipTap `Highlight` would strip
- **Provenance panel** — Click any highlighted text → right sidebar (480px) slides in showing:
  - Amendment name + section number header
  - Actual source amendment document rendered in iframe (DOCX→HTML via mammoth, PDF native)
- **Document canvas** shrinks by `marginRight: 480px` when panel is open
- **Toolbar**: Bold, Italic, Underline, Strikethrough, Headings, Lists, Alignment, Sub/Superscript, Undo/Redo, Highlight
- **Export**: PDF (jsPDF) and DOCX (docx.js) with page breaks
- **Conflict indicators**: Orange badges next to section titles with active conflicts
- **Stats bar**: Page count, section count, amended section count

### 5.4 API & State Management

- **API client** (`api/client.ts`): Fetch wrappers for all backend endpoints
- **React Query hooks** (`hooks/useApi.ts`): Declarative data fetching with caching
- **WebSocket**: Real-time pipeline progress updates
- **Types** (`types/index.ts`): 15 TypeScript interfaces covering all API responses

---

## 6. Data Flow

### 6.1 Ingestion Pipeline

```
Upload PDFs/DOCX
      │
      ▼
┌─ Stage 1: Parse ──────────────────────────────────┐
│  PyMuPDF / python-docx → raw text                  │
│  LLM structures into sections                      │
│  Embed to pgvector (is_resolved=FALSE)             │
└────────────────────────────────────────────────────┘
      │
      ▼
┌─ Stage 2: Track Amendments ────────────────────────┐
│  For each amendment document (sequential):          │
│  LLM extracts modifications (type, old/new text)   │
│  Adversarial buried-change scan                    │
│  Saves to amendments table                         │
└────────────────────────────────────────────────────┘
      │
      ▼
┌─ Stage 3: Sequence ───────────────────────────────┐
│  LLM infers missing effective dates                │
│  Builds chronological order + version tree         │
│  Saves supersession relationships                  │
└────────────────────────────────────────────────────┘
      │
      ▼
┌─ Stage 4: Resolve Overrides ──────────────────────┐
│  For each section (parallel):                      │
│  1. PROGRAMMATIC text assembly (no LLM rewriting)  │
│     Applies amendments in chronological order       │
│     Handles: replacement, selective override,       │
│     addition, deletion, exhibit replacement         │
│  2. LLM CLASSIFIES result only (category,          │
│     confidence)                                    │
│  3. Builds source_chain provenance                 │
│  Saves to clauses + section_embeddings (resolved)  │
│  Invalidates consolidated cache                    │
└────────────────────────────────────────────────────┘
      │
      ▼
┌─ Stage 5: Map Dependencies ──────────────────────┐
│  LLM identifies cross-references between clauses  │
│  9 relationship types                              │
│  Saves to clause_dependencies                      │
└────────────────────────────────────────────────────┘
      │
      ▼
┌─ Stage 6: Detect Conflicts ──────────────────────┐
│  LLM analyzes clauses + dependencies for conflicts │
│  7 conflict types with severity scoring            │
│  Tool-enabled (can query DB during analysis)       │
│  Saves to conflicts table                          │
└────────────────────────────────────────────────────┘
```

### 6.2 Override Resolution Detail

The core differentiator — **programmatic text assembly** ensures verbatim fidelity:

```python
# For each section with amendments:
current_text = original_clause_text

for amendment in sorted_by_date(amendments):
    match amendment.modification_type:
        case COMPLETE_REPLACEMENT → current_text = new_text
        case SELECTIVE_OVERRIDE  → find-and-replace (exact → normalized → fuzzy)
        case ADDITION            → current_text += "\n\n" + new_text
        case DELETION            → current_text = ""
        case EXHIBIT_REPLACEMENT → find-and-replace

# LLM only classifies the assembled text (never rewrites it)
classification = llm.classify(current_text)
```

Selective override uses a 3-tier matching strategy:
1. **Exact match**: Direct string replacement
2. **Normalized match**: Collapse whitespace differences
3. **Fuzzy match**: `difflib.SequenceMatcher` with sliding window (threshold 0.6)

Section header prefixes (e.g., "Section 7.2 (Payment Terms):") are stripped via regex before text assembly.

### 6.3 Query Pipeline

```
User Query
    │
    ▼
QueryRouter → classify type (truth/conflict/ripple/general)
    │
    ▼
pgvector semantic search (is_resolved=TRUE, top-k)
    │
    ├─ ripple_analysis → RippleEffectAnalyzer (multi-hop)
    │
    └─ other types → TruthSynthesizer
                        │
                        ▼
                   Answer + sources + confidence + caveats
```

---

## 7. Provenance & Source Chain

Every resolved clause carries a `source_chain` — an ordered list of transformations:

```json
[
  {
    "stage": "original",
    "document_id": "uuid-of-cta",
    "document_label": "Original CTA",
    "text": "Original clause body...",
    "change_description": null,
    "modification_type": null
  },
  {
    "stage": "amendment_3",
    "document_id": "uuid-of-amd-3",
    "document_label": "Amendment 3 (Effective 2011-01-01)",
    "text": "Modified clause body...",
    "change_description": "Replaced compliance placeholder with affiliates authority representation",
    "modification_type": "complete_replacement"
  }
]
```

This chain enables:
- Clicking amended text in the consolidated view → opening the source amendment document
- Auditing exactly which amendment changed which clause and when
- Displaying the full evolution of any clause through the contract stack

---

## 8. Configuration

### Environment Variables

```
AZURE_OPENAI_API_KEY          # Primary LLM
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_DEPLOYMENT
AZURE_OPENAI_API_VERSION
GEMINI_API_KEY                # Fallback LLM + embeddings
ANTHROPIC_API_KEY             # Claude (available, not in default routing)
EXTERNAL_DATABASE_URL         # NeonDB PostgreSQL connection string
```

### Development Commands

```bash
source venv/bin/activate                                    # Always activate venv first
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload  # Backend
cd frontend && npm run dev                                  # Frontend dev (port 5000)
./build.sh                                                  # Production frontend build
alembic upgrade head                                        # Run migrations
```

---

## 9. Demo Scenarios

### HEARTBEAT-3 (7 PDFs)

Cardiology trial at Memorial Medical Center. 5 known pain points serve as acceptance criteria:

| # | Pain Point | Detection |
|---|---|---|
| 1 | **Buried Payment Change**: Net 30 → Net 45 hidden in Amendment 3's COVID section | `buried_change` conflict |
| 2 | **Budget Exhibit Evolution**: Exhibit B → B-1 → B-2 across amendments | `exhibit_replacement` in source_chain |
| 3 | **Insurance Coverage Gap**: Amendment 5 extends study, insurance ambiguous | `gap` conflict (high severity) |
| 4 | **Cross-Reference Confusion**: Amendment 4 removes visits but cardiac MRI survives | `stale_reference` conflict |
| 5 | **PI Change + Budget Ambiguity**: Amendment 2 changes PI, Exhibit B-1 references old PI | `inconsistency` conflict |

### PharmaA-CRO1 (11 DOCX)

Master Services Agreement with 10 amendments spanning 2005–2020. Tests:
- High amendment volume (10 amendments)
- Inline amendment patterns ("Section 3.1 amended to require...")
- Complex selective overrides and exhibit replacements
- Volume rebate schedules and governance structures

---

## 10. File Structure

```
cta-contract-intelligence/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app, startup/shutdown
│   │   ├── logging_config.py          # Centralized logging (→ ./tmp/)
│   │   ├── exceptions.py              # Custom exceptions
│   │   ├── agents/
│   │   │   ├── base.py                # BaseAgent (802 lines) — all agent capabilities
│   │   │   ├── config.py              # AgentConfig dataclass
│   │   │   ├── circuit_breaker.py     # Circuit breaker for provider resilience
│   │   │   ├── prompt_loader.py       # Load .txt prompts with {var} substitution
│   │   │   ├── orchestrator.py        # 6-stage pipeline + query pipeline (1,050 lines)
│   │   │   ├── document_parser.py     # Stage 1: PDF/DOCX → structured sections
│   │   │   ├── amendment_tracker.py   # Stage 2: Modification extraction
│   │   │   ├── temporal_sequencer.py  # Stage 3: Chronological ordering
│   │   │   ├── override_resolution.py # Stage 4: Programmatic text assembly
│   │   │   ├── dependency_mapper.py   # Stage 5: Cross-reference graph
│   │   │   ├── conflict_detection.py  # Stage 6: Conflict identification
│   │   │   ├── contract_consolidator.py # Hierarchical document builder
│   │   │   ├── ripple_effect.py       # Multi-hop impact analysis
│   │   │   ├── query_router.py        # Query classification
│   │   │   ├── truth_synthesizer.py   # Answer synthesis
│   │   │   └── reusability.py         # Placeholder
│   │   ├── api/
│   │   │   ├── routes.py              # REST endpoints (819 lines)
│   │   │   └── websocket.py           # WebSocket progress broadcasting
│   │   ├── database/
│   │   │   ├── db.py                  # asyncpg pool creation
│   │   │   └── vector_store.py        # pgvector operations (291 lines)
│   │   └── models/
│   │       ├── agent_schemas.py       # Pydantic I/O models (500+ lines)
│   │       ├── enums.py               # All enumerations
│   │       └── events.py              # Pipeline progress/error events
│   └── uploads/                       # Uploaded document files
├── frontend/
│   ├── src/
│   │   ├── App.tsx                    # Router setup
│   │   ├── pages/
│   │   │   ├── Landing.tsx            # Animated landing page
│   │   │   ├── Dashboard.tsx          # Command center
│   │   │   ├── StacksList.tsx         # Contract list
│   │   │   └── StackDetail.tsx        # 7-tab detail view (3,500 lines)
│   │   ├── api/client.ts             # Fetch wrappers
│   │   ├── hooks/useApi.ts           # React Query hooks
│   │   └── types/index.ts            # TypeScript interfaces
│   └── dist/                          # Production build (served by backend)
├── prompt/                            # 28 prompt .txt files
├── alembic/versions/                  # 5 database migrations
├── policies/cta_policy_amendment/     # Test data (HEARTBEAT + CRO)
├── design/                            # Original design specs
├── CLAUDE.md                          # Claude Code project instructions
└── SYSTEM_DESIGN.md                   # This document
```
