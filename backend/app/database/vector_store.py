"""VectorStore — pgvector-backed semantic search using Gemini embeddings.

Two classes of embeddings in section_embeddings table:
  - is_resolved=FALSE  (Stage 1): Raw per-document sections for checkpoint fallback.
    Unique key: (contract_stack_id, document_id, section_number)
  - is_resolved=TRUE   (Stage 4): Resolved current-truth clauses for query search.
    Unique key: (contract_stack_id, section_number)
"""

import logging
import os
from datetime import date, datetime
from typing import Optional
from uuid import UUID

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Stage 1: per-document section embeddings (checkpoint fallback)
_UPSERT_DOC_SQL = """
    INSERT INTO section_embeddings
        (contract_stack_id, document_id, section_number, section_title,
         section_text, effective_date, embedding, embedding_model, is_resolved)
    VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8, FALSE)
    ON CONFLICT (contract_stack_id, document_id, section_number)
        WHERE is_resolved = FALSE
    DO UPDATE SET
        section_title = EXCLUDED.section_title,
        section_text = EXCLUDED.section_text,
        effective_date = EXCLUDED.effective_date,
        embedding = EXCLUDED.embedding,
        embedding_model = EXCLUDED.embedding_model
"""

# Stage 4: resolved-clause embeddings (query search)
_UPSERT_RESOLVED_SQL = """
    INSERT INTO section_embeddings
        (contract_stack_id, document_id, section_number, section_title,
         section_text, effective_date, embedding, embedding_model, is_resolved)
    VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8, TRUE)
    ON CONFLICT (contract_stack_id, section_number)
        WHERE is_resolved = TRUE
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
        all_embeddings: list[list[float]] = []
        batch_size = 100  # Gemini API limit
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            result = await client.aio.models.embed_content(
                model=self.EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.EMBEDDING_DIM,
                ),
            )
            all_embeddings.extend(e.values for e in result.embeddings)
        return all_embeddings

    @staticmethod
    def _parse_date(value) -> Optional[date]:
        """Convert string/datetime/date to datetime.date for asyncpg."""
        if value is None:
            return None
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError:
                return None
        return None

    # ── Stage 1: Document-level embeddings (checkpoint fallback) ──

    async def upsert_embeddings(
        self,
        contract_stack_id: UUID,
        sections: list[dict],
        document_id: UUID,
        effective_date: Optional[str] = None,
    ) -> int:
        """Embed raw parsed sections from a single document (Stage 1).

        Keyed by (contract_stack_id, document_id, section_number).
        Each document's sections are stored independently — no cross-document
        overwrites. Used for checkpoint resume when Stage 4 hasn't run yet.
        """
        if not sections:
            return 0

        texts = [s["text"][:8000] for s in sections]
        embeddings = await self._generate_embeddings(texts, task_type="RETRIEVAL_DOCUMENT")

        eff_date = self._parse_date(effective_date)

        args = []
        for section, embedding in zip(sections, embeddings):
            embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
            args.append((
                contract_stack_id,
                document_id,
                section["section_number"],
                section.get("section_title", ""),
                section["text"][:8000],
                eff_date,
                embedding_str,
                self.EMBEDDING_MODEL,
            ))

        async with self._pool.acquire() as conn:
            await conn.executemany(_UPSERT_DOC_SQL, args)

        logger.info("Upserted %d document embeddings for stack %s doc %s",
                     len(sections), contract_stack_id, document_id)
        return len(sections)

    # ── Stage 4: Resolved-clause embeddings (query search) ────────

    async def upsert_resolved_clauses(
        self,
        contract_stack_id: UUID,
        clauses: list[dict],
    ) -> int:
        """Embed resolved current-truth clauses after override resolution (Stage 4).

        Keyed by (contract_stack_id, section_number). These are the ONLY
        embeddings searched during query-time semantic search.

        Each clause dict must have: section_number, section_title, current_text,
        clause_category, source_document_id, effective_date.
        """
        if not clauses:
            return 0

        # Build enriched text for embedding: title + category context + full text.
        # This gives the embedding model structural context so "payment terms"
        # matches both the title and content of Section 7.2.
        texts = []
        for c in clauses:
            parts = [f"Section {c['section_number']}"]
            if c.get("section_title"):
                parts[0] += f" — {c['section_title']}"
            if c.get("clause_category") and c["clause_category"] != "general":
                parts.append(f"Category: {c['clause_category']}")
            parts.append(c["current_text"][:8000])
            texts.append("\n".join(parts))

        embeddings = await self._generate_embeddings(texts, task_type="RETRIEVAL_DOCUMENT")

        args = []
        for clause, embedding in zip(clauses, embeddings):
            embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
            args.append((
                contract_stack_id,
                clause.get("source_document_id"),         # traceability
                clause["section_number"],
                (clause.get("section_title") or "")[:255],
                clause["current_text"][:8000],
                self._parse_date(clause.get("effective_date")),
                embedding_str,
                self.EMBEDDING_MODEL,
            ))

        async with self._pool.acquire() as conn:
            await conn.executemany(_UPSERT_RESOLVED_SQL, args)

        logger.info("Upserted %d resolved-clause embeddings for stack %s",
                     len(clauses), contract_stack_id)
        return len(clauses)

    # ── Query-time semantic search (resolved clauses only) ────────

    async def query_similar(
        self,
        query_text: str,
        contract_stack_id: UUID,
        n_results: int = 10,
    ) -> list[dict]:
        """Semantic search against resolved current-truth clauses.

        Only searches is_resolved=TRUE rows — the canonical section numbers
        that match the clauses table. Returns section_numbers suitable for
        direct batch lookup in the clauses table.
        """
        embeddings = await self._generate_embeddings([query_text], task_type="RETRIEVAL_QUERY")
        query_embedding_str = "[" + ",".join(str(v) for v in embeddings[0]) + "]"

        rows = await self._pool.fetch(
            """
            SELECT section_number, section_title, section_text,
                   document_id, effective_date,
                   embedding <=> $1::vector AS distance
            FROM section_embeddings
            WHERE contract_stack_id = $2 AND is_resolved = TRUE
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

    # ── Checkpoint fallback (document-level sections) ─────────────

    async def get_by_document(self, document_id: UUID) -> list[dict]:
        """Retrieve raw parsed sections for a specific document (checkpoint).

        Only returns is_resolved=FALSE rows for the given document_id.
        Used when Stage 1 completed but Stage 4 hasn't populated the
        clauses table yet.
        """
        rows = await self._pool.fetch(
            """
            SELECT section_number, section_title, section_text, effective_date
            FROM section_embeddings
            WHERE document_id = $1 AND is_resolved = FALSE
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

    async def has_resolved_embeddings(self, contract_stack_id: UUID) -> bool:
        """Check if resolved-clause embeddings exist for a contract stack."""
        count = await self._pool.fetchval(
            "SELECT COUNT(*) FROM section_embeddings "
            "WHERE contract_stack_id = $1 AND is_resolved = TRUE",
            contract_stack_id,
        )
        return count > 0
