"""Enable pgvector and create section_embeddings table.

Revision ID: 003
Revises: 002
Create Date: 2026-02-07
"""
from alembic import op

revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension (NeonDB supports this natively)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Section embeddings table — written in Stage 1 (document parsing),
    # joined to clauses via (contract_stack_id, section_number)
    op.execute("""
        CREATE TABLE section_embeddings (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            contract_stack_id UUID NOT NULL REFERENCES contract_stacks(id) ON DELETE CASCADE,
            document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
            section_number VARCHAR(50) NOT NULL,
            section_title VARCHAR(255),
            section_text TEXT NOT NULL,
            effective_date DATE,
            embedding vector(768) NOT NULL,
            embedding_model VARCHAR(100) NOT NULL DEFAULT 'gemini-embedding-001',
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)

    # Upsert target — one embedding per (stack, section)
    op.execute("""
        CREATE UNIQUE INDEX uq_section_embeddings_stack_section
            ON section_embeddings(contract_stack_id, section_number)
    """)

    # HNSW cosine index for semantic search
    op.execute("""
        CREATE INDEX idx_section_embeddings_hnsw
            ON section_embeddings USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
    """)

    # Lookup index (contract_stack_id is already the leading column in the
    # unique index, so a single-column index on it would be redundant)
    op.execute("CREATE INDEX idx_section_embeddings_document ON section_embeddings(document_id)")

    # Reuse trigger function from migration 002
    op.execute("""
        CREATE TRIGGER trg_section_embeddings_updated_at
            BEFORE UPDATE ON section_embeddings
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_section_embeddings_updated_at ON section_embeddings")
    op.execute("DROP TABLE IF EXISTS section_embeddings")
    op.execute("DROP EXTENSION IF EXISTS vector")
