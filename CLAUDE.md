# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ContractIQ** is an agentic contract intelligence platform for clinical trial agreements (CTAs). It performs truth reconstitution, conflict detection, and ripple effect analysis across contract stacks (original CTA + amendments). The canonical test case is **HEARTBEAT-3** (6 PDFs in `policies/cta_policy_amendment/`).

The full implementation specification is in `technical_spec.md` — always reference it for architecture decisions, data models, API specs, and agent designs.

## Current State

This project is in the **planning/early implementation phase**. The repository contains:
- `technical_spec.md` — comprehensive implementation spec (schemas, APIs, agents, frontend)
- `policies/cta_policy_amendment/` — HEARTBEAT-3 demo data (Original CTA + 5 Amendments + Demo Guide)
- `.env` — API credentials for Anthropic Claude, Azure OpenAI (gpt-5-mini), and Google Gemini

## Tech Stack

- **Backend**: Python 3.11+ / FastAPI / Celery / Redis
- **Databases**: PostgreSQL 15+ via NeonDB (relational + clause dependency graph via recursive CTEs + pgvector for semantic search), Redis (cache)
- **AI/ML**: Anthropic Claude API (Opus for complex reasoning, Sonnet for extraction), Azure OpenAI, Gemini
- **Document Processing**: PyMuPDF, pdfplumber, python-docx
- **Frontend**: React 18+ / TypeScript / Tailwind CSS / shadcn/ui

## Architecture

Three-tier agent system orchestrated by a central Agent Orchestrator:

- **Tier 1 — Document Ingestion**: `DocumentParserAgent`, `AmendmentTrackerAgent`, `TemporalSequencerAgent`
- **Tier 2 — Reasoning**: `OverrideResolutionAgent`, `ConflictDetectionAgent`, `DependencyMapperAgent`
- **Tier 3 — Analysis**: `RippleEffectAnalyzerAgent`, `ReusabilityAnalyzerAgent` (Phase 2)
- **Query Pipeline**: `QueryRouter`, `TruthSynthesizer`

All agents extend `BaseAgent` with `process()` and `call_llm()` methods. Data flows: PDF upload -> Tier 1 extraction -> PostgreSQL (NeonDB) storage (structured data + pgvector embeddings) -> Tier 2 reasoning -> Tier 3 analysis -> API response.

**API pattern**: REST at `/api/v1/` + WebSocket at `/api/v1/ws/` for real-time processing updates.

**Database layers**: PostgreSQL (NeonDB) stores structured data (contract_stacks, documents, clauses, amendments, conflicts, queries), the clause dependency graph (clause_dependencies table with recursive CTEs for multi-hop traversal), and vector embeddings for semantic search (section_embeddings table with pgvector HNSW cosine index, Gemini text-embedding-004, 768-dim).

## Development Commands

```bash
# Virtual environment (ALWAYS activate before running Python)
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt

# Run backend
uvicorn app.main:app --reload

# Run tests
pytest
pytest tests/test_specific.py -v          # single test file
pytest tests/test_specific.py::test_name  # single test

# Database migrations
alembic upgrade head

# Docker services (Redis only — PostgreSQL + pgvector is NeonDB cloud)
docker-compose up -d
```

## Configuration

All credentials loaded from `.env` via `python-dotenv`. Key variables:
- `ANTHROPIC_API_KEY` — Claude API
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION` — Azure OpenAI
- `GEMINI_API_KEY` — Google Gemini
- `EXTERNAL_DATABASE_URL` — NeonDB PostgreSQL connection string

## Key Conventions

- **Prompts**: All LLM prompts stored as `.txt` files in `/prompt` directory with `{variable_name}` placeholders. Never hardcode prompts in Python.
- **Logging**: All logs to `./tmp/` directory only. Use `from pipeline.logging_config import setup_logging`.
- **Archiving**: Old file versions go to `.archive/<timestamp>/`. No `v1`/`v2`/`_old`/`_backup` suffixes in the active workspace.
- **LLM calls**: Always set `max_output_tokens` to the model's maximum. For Gemini: flash-lite=65536, 2.5-pro=65536, 2.0-flash-exp=8192, 1.5-flash=8192, 1.5-pro=8192.
- **Error handling**: Fix root causes. Never write fallback code, use cached results, or bypass errors with workarounds.
- **Models**: Never use 8B models (e.g., gemini-1.5-flash-8b). Use full-size models only.
- **Inventory**: Keep `FILE_INVENTORY.md` / `PIPELINE_FILE_INVENTORY.md` updated after file changes. Never archive `CLAUDE.md`, `SYSTEM_DESIGN.md`, `FILE_INVENTORY.md`, or `README.md`.

## Demo Scenario: HEARTBEAT-3

The test data represents a cardiology trial (HEARTBEAT-3) at Memorial Medical Center with 5 known pain points:
1. **Buried Payment Change**: Net 30 -> Net 45 hidden in Amendment 3's COVID section
2. **Budget Exhibit Evolution**: Exhibit B -> B-1 -> B-2 across amendments with payment term changes
3. **Insurance Coverage Gap**: Amendment 5 extends study but insurance obligation is ambiguous
4. **Cross-Reference Confusion**: Amendment 4 removes follow-up visits but cardiac MRI from Amendment 1 survives
5. **PI Change + Budget Ambiguity**: Amendment 2 changes PI but Exhibit B-1 still references old PI

These pain points serve as acceptance criteria for the agent system.