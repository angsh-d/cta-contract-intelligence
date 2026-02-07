"""ContractIQ — FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.api.websocket import router as ws_router
from app.database.db import create_postgres_pool, create_vector_store
from app.logging_config import setup_logging

load_dotenv()
setup_logging(log_level="INFO", log_file="contractiq.log")

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create DB pools + orchestrator. Shutdown: close connections."""
    logger.info("Starting ContractIQ...")

    postgres_pool = await create_postgres_pool()
    vector_store = create_vector_store(postgres_pool)

    # Store pool on app.state for direct use in routes
    app.state.postgres_pool = postgres_pool

    try:
        from app.agents.orchestrator import AgentOrchestrator

        app.state.orchestrator = AgentOrchestrator(
            postgres_pool=postgres_pool,
            vector_store=vector_store,
        )
        logger.info("ContractIQ ready — orchestrator initialised with %d agents",
                    len(app.state.orchestrator.agents))
    except Exception as exc:
        logger.warning("Orchestrator init failed (missing API keys?): %s — server running in limited mode", exc)
        app.state.orchestrator = None
    yield

    # Shutdown
    logger.info("Shutting down ContractIQ...")
    await postgres_pool.close()
    logger.info("ContractIQ stopped.")


app = FastAPI(
    title="ContractIQ",
    description="Agentic contract intelligence for clinical trial agreements",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(ws_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "contractiq",
        "ai_available": app.state.orchestrator is not None,
    }
