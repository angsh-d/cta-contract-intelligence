"""Database connection factories for PostgreSQL (NeonDB) and pgvector."""

import os
import asyncpg
from dotenv import load_dotenv

from app.database.vector_store import VectorStore

load_dotenv()


async def create_postgres_pool() -> asyncpg.Pool:
    """Create asyncpg connection pool to NeonDB PostgreSQL."""
    database_url = os.environ.get("DATABASE_URL") or os.environ.get("EXTERNAL_DATABASE_URL", "")
    return await asyncpg.create_pool(
        database_url,
        min_size=2,
        max_size=10,
        command_timeout=60,
    )


def create_vector_store(postgres_pool: asyncpg.Pool) -> VectorStore:
    """Create VectorStore backed by pgvector on the shared PostgreSQL pool."""
    return VectorStore(postgres_pool)
