"""Database connection factories for PostgreSQL (NeonDB) and ChromaDB."""

import os
from pathlib import Path
import asyncpg
import chromadb
from dotenv import load_dotenv

load_dotenv()


async def create_postgres_pool() -> asyncpg.Pool:
    """Create asyncpg connection pool to NeonDB PostgreSQL."""
    database_url = os.environ.get("EXTERNAL_DATABASE_URL") or os.environ["DATABASE_URL"]
    return await asyncpg.create_pool(
        database_url,
        min_size=2,
        max_size=10,
        command_timeout=60,
    )


def create_chroma_collection(
    collection_name: str = "contractiq_clauses",
    persist_dir: str | None = None,
) -> chromadb.Collection:
    """Create or get ChromaDB collection for clause embeddings."""
    persist_dir = persist_dir or os.environ.get("CHROMA_PERSIST_DIR", "./chroma_data")
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
