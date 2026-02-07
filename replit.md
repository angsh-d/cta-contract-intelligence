# ContractIQ

## Overview
ContractIQ is an agentic contract intelligence platform for clinical trial agreements (CTAs). It performs truth reconstitution, conflict detection, and ripple effect analysis across contract stacks (original CTA + amendments).

**Current State**: Full-stack application with FastAPI backend and React frontend. Backend runs on port 8000, frontend (Vite) on port 5000 with proxy to backend. AI agents require API keys for full functionality.

## Tech Stack
- **Backend**: Python 3.12, FastAPI + Uvicorn
- **Frontend**: React 18 + TypeScript, Vite, Tailwind CSS v4, Framer Motion
- **Database**: PostgreSQL + pgvector (Replit built-in / NeonDB for vector search). Uses a two-tier pgvector embedding architecture: `is_resolved=FALSE` rows hold Stage 1 checkpoint embeddings (raw parsed sections), while `is_resolved=TRUE` rows hold Stage 4 query-ready embeddings (post-override-resolution). Query-time semantic search filters on `is_resolved=TRUE` only.
- **AI/ML**: Anthropic Claude, Azure OpenAI, Google Gemini
- **Document Processing**: PyMuPDF, pdfplumber
- **UI Libraries**: Lucide React (icons), TanStack React Query, React Router

## Project Architecture
```
backend/
  app/
    main.py            - FastAPI entry point with lifespan management
    api/
      routes.py        - REST API routes (/api/v1/)
      websocket.py     - WebSocket for real-time updates
    agents/            - AI agent implementations (Tier 1-3)
      orchestrator.py  - Central agent orchestrator
      llm_providers.py - LLM provider abstractions
    database/
      db.py            - PostgreSQL + pgvector connection factories
    models/            - Pydantic schemas and enums
frontend/
  src/
    main.tsx           - React entry point
    App.tsx            - Routes configuration
    index.css          - Tailwind + Apple design system
    api/client.ts      - API client functions
    hooks/useApi.ts    - React Query hooks
    types/index.ts     - TypeScript type definitions
    components/
      AppShell.tsx     - Sidebar layout with navigation
    pages/
      Dashboard.tsx    - Overview with stats and actions
      StacksList.tsx   - Contract list with create modal
      StackDetail.tsx  - Detail view with 5 tabs (Overview, Timeline, Query, Conflicts, Ripple)
alembic/               - Database migrations
prompt/                - LLM prompt templates (.txt files)
policies/              - Demo CTA documents (HEARTBEAT-3)
```

## Development Commands
```bash
# Combined workflow (configured): Backend on 8000 + Frontend on 5000
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
cd frontend && npm run dev

# Database migrations
alembic upgrade head
```

## Environment Variables
- `DATABASE_URL` - PostgreSQL connection (auto-configured by Replit, preferred)
- `EXTERNAL_DATABASE_URL` - Fallback PostgreSQL connection (NeonDB)
- `ANTHROPIC_API_KEY` - Required for AI agent functionality
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION` - Azure OpenAI (fallback)
- `GEMINI_API_KEY` - Google Gemini (fallback)

## Key Endpoints
- `GET /health` - Health check (includes ai_available status)
- `GET /api/v1/contract-stacks` - List all contract stacks
- `POST /api/v1/contract-stacks` - Create contract stack
- `GET /api/v1/contract-stacks/{id}` - Get stack details
- `POST /api/v1/contract-stacks/{id}/documents` - Upload document
- `POST /api/v1/contract-stacks/{id}/process` - Process with AI agents
- `POST /api/v1/contract-stacks/{id}/query` - Query contract stack
- `GET /api/v1/contract-stacks/{id}/timeline` - Get timeline
- `POST /api/v1/contract-stacks/{id}/analyze/conflicts` - Detect conflicts

## Frontend Design
- Pure greyscale Apple-inspired UI: SF Pro fonts, #1d1d1f text, #f5f5f7 backgrounds, glass morphism
- NO colors allowed — blacks, greys, whites only throughout all pages
- Responsive sidebar navigation with mobile drawer
- Framer Motion page transitions, stagger animations, spring-based tab indicators
- 5-tab segmented control on contract detail (Overview, Timeline, Query, Conflicts, Ripple Effects)
- Apple Messages-style AI chat interface with suggested queries, typing indicator, source citations
- Dashboard with 9-agent architecture preview (3 tiers: Ingestion, Reasoning, Analysis)
- Landing page with animated agent pipeline visualization and stat counters
- Premium contract cards with therapeutic area badges and status pills
- Vertical timeline with dark CTA root node and amendment chain
- Conflict detection with severity-based cards and graduated border indicators
- Ripple effect analysis with section/current/proposed text inputs

## User Preferences
- Apple-inspired, best-in-class UX design — PURE GREYSCALE ONLY
- Clean, minimal interface virtually indistinguishable from Apple products
- No colors (blue, green, orange, red) — only blacks, greys, whites

## Recent Changes
- 2026-02-07: Initial Replit setup - configured PostgreSQL, ran migrations
- 2026-02-07: Built complete Apple-inspired React frontend with 5 pages
- 2026-02-07: Fixed DB connection to prefer DATABASE_URL over EXTERNAL_DATABASE_URL
- 2026-02-07: Combined backend+frontend into single workflow
- 2026-02-07: Deployment configured for autoscale with frontend build step
- 2026-02-07: Major UX overhaul — StackDetail with 5 premium tabs (cinematic metrics, timeline, Messages-style chat, conflict cards, ripple effects)
- 2026-02-07: Dashboard redesigned as clinical operations command center with 9-agent architecture preview
- 2026-02-07: StacksList enhanced with premium cards, therapeutic area badges, stagger animations
- 2026-02-07: Landing page enhanced with animated agent pipeline and stat counters
- 2026-02-07: Fixed StacksList animation bug (data loading after mount)
