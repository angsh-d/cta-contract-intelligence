# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ContractIQ** is an agentic contract intelligence platform for clinical trial agreements (CTAs). It performs truth reconstitution, conflict detection, and ripple effect analysis across contract stacks (original CTA + amendments). The canonical test case is **HEARTBEAT-3** (7 PDFs in `policies/cta_policy_amendment/`).

Detailed specs live in `design/technical_spec.md` (66KB) and agent design docs in `design/agents/`.

## Development Commands

```bash
# Virtual environment (ALWAYS activate before running Python)
source venv/bin/activate

# Run backend (from project root)
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Run frontend dev server (port 5000, proxies /api/* to backend:8000)
cd frontend && npm run dev

# Build frontend for production (served by backend as static files)
./build.sh   # or: cd frontend && npm install && npm run build

# Database migrations
alembic upgrade head

# Install Python deps
pip install --upgrade pip && pip install -r requirements.txt

# Install frontend deps
cd frontend && npm install
```

No test suite exists yet. The `design/agents/08_testing_strategy.md` documents the planned approach.

## Architecture

### Backend (`backend/app/`)

FastAPI application in `main.py` — startup creates PostgreSQL pool, vector store, and `AgentOrchestrator`. Serves the built frontend SPA from `frontend/dist/` as static files.

**Agent system** — 11 agents in `backend/app/agents/`, all extending `BaseAgent` (`base.py`, 800+ lines):
- `BaseAgent` provides: structured output via tool_use, self-verification with confidence gating, circuit breaker for provider resilience, token estimation, trace context for cost tracking, automatic provider failover
- **Tier 1 (Ingestion)**: `document_parser.py`, `amendment_tracker.py`, `temporal_sequencer.py`
- **Tier 2 (Reasoning)**: `override_resolution.py`, `conflict_detection.py`, `dependency_mapper.py`
- **Tier 3 (Analysis)**: `ripple_effect.py`, `reusability.py` (placeholder)
- **Query Pipeline**: `query_router.py`, `truth_synthesizer.py`
- **Orchestration**: `orchestrator.py` (1000+ lines) — 6-stage ingestion pipeline with checkpoint resume, in-memory blackboard for inter-agent communication, PostgreSQL-backed `PgCache` for query/ripple results

**LLM providers** (`llm_providers.py`): `LLMProviderFactory` with role-based routing. Primary: Azure OpenAI GPT-5.2. Fallback: Google Gemini. Embeddings: Gemini `gemini-embedding-001` (768-dim).

**API routes** (`api/routes.py`): REST at `/api/v1/` + WebSocket at `/api/v1/ws/` for real-time pipeline progress. Key endpoints:
- `POST /api/v1/contract-stacks/{id}/process` — triggers 6-stage pipeline (returns job_id for polling)
- `POST /api/v1/contract-stacks/{id}/query` — natural language query with semantic search
- `POST /api/v1/contract-stacks/{id}/analyze/conflicts` — conflict detection
- `POST /api/v1/contract-stacks/{id}/analyze/ripple-effects` — multi-hop ripple analysis

### Database (PostgreSQL + pgvector on NeonDB)

Connected via `EXTERNAL_DATABASE_URL` (asyncpg pool). 9 core tables + 1 vector table. 4 Alembic migrations in `alembic/versions/`.

**Two-tier embedding architecture** in `section_embeddings` table:
- `is_resolved=FALSE`: Stage 1 checkpoint embeddings (raw parsed sections per document)
- `is_resolved=TRUE`: Stage 4 query-ready embeddings (post-override-resolution)
- Query-time semantic search filters on `is_resolved=TRUE` only

Vector store implementation: `backend/app/database/vector_store.py`.

### 6-Stage Ingestion Pipeline

1. Parse PDFs → `section_embeddings` (is_resolved=FALSE)
2. Track amendments → `amendments` table
3. Sequence documents → `document_supersessions` table
4. Resolve overrides → `clauses` table (is_current=TRUE) + `section_embeddings` (is_resolved=TRUE)
5. Map dependencies → `clause_dependencies` table
6. Detect conflicts → `conflicts` table

All stages check if data already exists in DB before re-running (checkpoint resume).

### Frontend (`frontend/src/`)

React 19 + TypeScript + Vite + Tailwind CSS v4 + Framer Motion. **Pure greyscale Apple-inspired design** — no colors (blue, green, red, orange), only blacks, greys, whites.

4 pages: `Landing.tsx` (animated agent pipeline), `Dashboard.tsx` (command center), `StacksList.tsx` (contract cards), `StackDetail.tsx` (104KB, 5-tab detail view: Overview, Timeline, Query, Conflicts, Ripple Effects).

API layer: `api/client.ts` (fetch wrappers) + `hooks/useApi.ts` (React Query hooks).

## Configuration

All credentials loaded from `.env` via `python-dotenv`:
- `ANTHROPIC_API_KEY` — Claude API
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION` — Azure OpenAI (primary LLM)
- `GEMINI_API_KEY` — Google Gemini (fallback LLM + embeddings)
- `EXTERNAL_DATABASE_URL` — NeonDB PostgreSQL connection string

## Key Conventions

- **Prompts**: 27 `.txt` files in `/prompt` with `{variable_name}` placeholders, loaded at runtime via `PromptLoader` (`backend/app/agents/prompt_loader.py`). Never hardcode prompts in Python.
- **Logging**: All logs to `./tmp/` directory only. Use `from backend.app.logging_config import setup_logging`.
- **Archiving**: Old file versions go to `.archive/<timestamp>/`. No `v1`/`v2`/`_old`/`_backup` suffixes in active workspace.
- **LLM calls**: Always set `max_output_tokens` to model's maximum. For Gemini: flash-lite=65536, 2.5-pro=65536, 2.0-flash-exp=8192, 1.5-flash=8192, 1.5-pro=8192.
- **Error handling**: Fix root causes. Never write fallback code, use cached results, or bypass errors with workarounds.
- **Models**: Never use 8B models (e.g., gemini-1.5-flash-8b). Use full-size models only.
- **Inventory**: Keep `FILE_INVENTORY.md` updated after file changes. Never archive `CLAUDE.md`, `SYSTEM_DESIGN.md`, `FILE_INVENTORY.md`, or `README.md`.
- **Frontend design**: Pure greyscale only. SF Pro fonts, glass morphism, Framer Motion animations.

## Demo Scenario: HEARTBEAT-3

Test data in `policies/cta_policy_amendment/` — cardiology trial at Memorial Medical Center. 5 known pain points serve as acceptance criteria:

1. **Buried Payment Change**: Net 30 → Net 45 hidden in Amendment 3's COVID section
2. **Budget Exhibit Evolution**: Exhibit B → B-1 → B-2 across amendments
3. **Insurance Coverage Gap**: Amendment 5 extends study, insurance obligation ambiguous
4. **Cross-Reference Confusion**: Amendment 4 removes follow-up visits but cardiac MRI from Amendment 1 survives
5. **PI Change + Budget Ambiguity**: Amendment 2 changes PI but Exhibit B-1 still references old PI