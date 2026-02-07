# ContractIQ

## Overview
ContractIQ is an agentic contract intelligence platform for clinical trial agreements (CTAs). It performs truth reconstitution, conflict detection, and ripple effect analysis across contract stacks (original CTA + amendments).

**Current State**: Backend API running on FastAPI. The orchestrator with AI agents requires API keys (ANTHROPIC_API_KEY, etc.) to be configured for full functionality.

## Tech Stack
- **Language**: Python 3.12
- **Framework**: FastAPI + Uvicorn
- **Database**: PostgreSQL (Replit built-in), ChromaDB (local vector store)
- **AI/ML**: Anthropic Claude, Azure OpenAI, Google Gemini
- **Document Processing**: PyMuPDF, pdfplumber

## Project Architecture
```
backend/
  app/
    main.py          - FastAPI entry point with lifespan management
    api/
      routes.py      - REST API routes (/api/v1/)
      websocket.py   - WebSocket for real-time updates
    agents/          - AI agent implementations (Tier 1-3)
      orchestrator.py - Central agent orchestrator
      llm_providers.py - LLM provider abstractions
    database/
      db.py          - PostgreSQL + ChromaDB connection factories
    models/          - Pydantic schemas and enums
alembic/             - Database migrations
prompt/              - LLM prompt templates (.txt files)
policies/            - Demo CTA documents (HEARTBEAT-3)
```

## Development Commands
```bash
# Run server (configured as workflow)
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload

# Database migrations
alembic upgrade head
```

## Environment Variables
- `DATABASE_URL` - PostgreSQL connection (auto-configured by Replit)
- `ANTHROPIC_API_KEY` - Required for AI agent functionality
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION` - Azure OpenAI (fallback)
- `GEMINI_API_KEY` - Google Gemini (fallback)

## Key Endpoints
- `GET /health` - Health check
- `POST /api/v1/contract-stacks` - Create contract stack
- `POST /api/v1/contract-stacks/{id}/documents` - Upload document
- `POST /api/v1/contract-stacks/{id}/process` - Process with AI agents
- `POST /api/v1/contract-stacks/{id}/query` - Query contract stack

## Recent Changes
- 2026-02-07: Initial Replit setup - configured PostgreSQL, ran migrations, set up workflow on port 5000
- DB connection updated to use Replit's DATABASE_URL (falls back from EXTERNAL_DATABASE_URL)
- Added processing_status column to contract_stacks schema
- Made orchestrator initialization graceful (server starts without API keys in limited mode)
