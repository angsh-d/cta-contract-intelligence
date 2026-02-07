"""VectorStore — pgvector-backed semantic search using Gemini embeddings."""

import logging
import os
from typing import Optional
from uuid import UUID

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

_UPSERT_SQL = """
    INSERT INTO section_embeddings
        (contract_stack_id, document_id, section_number, section_title,
         section_text, effective_date, embedding, embedding_model)
    VALUES ($1, $2, $3, $4, $5, $6::date, $7::vector, $8)
    ON CONFLICT (contract_stack_id, section_number)
    DO UPDATE SET
        document_id = EXCLUDED.document_id,
        section_title = EXCLUDED.section_title,
        section_text = EXCLUDED.section_text,
        effective_date = EXCLUDED.effective_date,
        embedding = EXCLUDED.embedding,
        embedding_model = EXCLUDED.embedding_model
"""


class VectorStore:
    """Wraps pgvector operations on the section_embeddings table."""

    EMBEDDING_MODEL = "gemini-embedding-001"
    EMBEDDING_DIM = 768

    def __init__(self, postgres_pool) -> None:
        self._pool = postgres_pool
        self._genai_client: Optional[genai.Client] = None

    def _get_genai_client(self) -> genai.Client:
        if self._genai_client is None:
            api_key = os.environ.get("GEMINI_API_KEY", "")
            self._genai_client = genai.Client(api_key=api_key)
        return self._genai_client

    async def _generate_embeddings(
        self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float]]:
        """Generate embeddings via Gemini gemini-embedding-001 (async).

        Args:
            texts: Texts to embed.
            task_type: RETRIEVAL_DOCUMENT for indexing, RETRIEVAL_QUERY for search.
        """
        client = self._get_genai_client()
        result = await client.aio.models.embed_content(
            model=self.EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.EMBEDDING_DIM,
            ),
        )
        return [e.values for e in result.embeddings]

    async def upsert_embeddings(
        self,
        contract_stack_id: UUID,
        sections: list[dict],
        document_id: Optional[UUID] = None,
        effective_date: Optional[str] = None,
    ) -> int:
        """Generate embeddings and upsert into section_embeddings.

        Args:
            contract_stack_id: The contract stack UUID.
            sections: List of dicts with keys: section_number, section_title, text.
            document_id: Optional source document UUID.
            effective_date: Optional effective date string (YYYY-MM-DD).

        Returns:
            Number of sections upserted.
        """
        if not sections:
            return 0

        texts = [s["text"][:8000] for s in sections]
        embeddings = await self._generate_embeddings(texts, task_type="RETRIEVAL_DOCUMENT")

        args = []
        for section, embedding in zip(sections, embeddings):
            embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
            args.append((
                contract_stack_id,
                document_id,
                section["section_number"],
                section.get("section_title", ""),
                section["text"][:8000],
                effective_date,
                embedding_str,
                self.EMBEDDING_MODEL,
            ))

        async with self._pool.acquire() as conn:
            await conn.executemany(_UPSERT_SQL, args)

        logger.info("Upserted %d embeddings for stack %s", len(sections), contract_stack_id)
        return len(sections)

    async def query_similar(
        self,
        query_text: str,
        contract_stack_id: UUID,
        n_results: int = 10,
    ) -> list[dict]:
        """Semantic search: embed query and find nearest sections via cosine distance.

        Returns list of dicts with keys: section_number, section_title, section_text,
        document_id, effective_date, distance.
        """
        embeddings = await self._generate_embeddings([query_text], task_type="RETRIEVAL_QUERY")
        query_embedding_str = "[" + ",".join(str(v) for v in embeddings[0]) + "]"

        rows = await self._pool.fetch(
            """
            SELECT section_number, section_title, section_text,
                   document_id, effective_date,
                   embedding <=> $1::vector AS distance
            FROM section_embeddings
            WHERE contract_stack_id = $2
            ORDER BY embedding <=> $1::vector
            LIMIT $3
            """,
            query_embedding_str,
            contract_stack_id,
            n_results,
        )

        return [
            {
                "section_number": r["section_number"],
                "section_title": r["section_title"] or "",
                "section_text": r["section_text"],
                "document_id": r["document_id"],
                "effective_date": r["effective_date"],
                "distance": r["distance"],
            }
            for r in rows
        ]

    async def get_by_document(self, document_id: UUID) -> list[dict]:
        """Retrieve all embeddings for a given document (checkpoint fallback).

        Returns list of dicts with keys: section_number, section_title, section_text,
        effective_date.
        """
        rows = await self._pool.fetch(
            """
            SELECT section_number, section_title, section_text, effective_date
            FROM section_embeddings
            WHERE document_id = $1
            ORDER BY section_number
            """,
            document_id,
        )

        return [
            {
                "section_number": r["section_number"],
                "section_title": r["section_title"] or "",
                "section_text": r["section_text"],
                "effective_date": r["effective_date"],
            }
            for r in rows
        ]
