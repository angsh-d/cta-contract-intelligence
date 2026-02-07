"""Add is_resolved column and fix unique constraints on section_embeddings.

Stage 1 (document parsing) creates per-document embeddings (is_resolved=FALSE).
After Stage 4 (override resolution), resolved clause embeddings are created
(is_resolved=TRUE) for query-time semantic search.

Revision ID: 004
Revises: 003
Create Date: 2026-02-07
"""
from alembic import op

revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add is_resolved flag to distinguish document-level vs resolved-clause embeddings
    op.execute("""
        ALTER TABLE section_embeddings
        ADD COLUMN is_resolved BOOLEAN NOT NULL DEFAULT FALSE
    """)

    # Drop the old unique index (was: one embedding per stack+section — caused
    # overwrites when multiple documents share section numbers)
    op.execute("DROP INDEX IF EXISTS uq_section_embeddings_stack_section")

    # Partial unique index for document-level embeddings (Stage 1 checkpoint):
    # One embedding per (stack, document, section) when is_resolved=FALSE
    op.execute("""
        CREATE UNIQUE INDEX uq_section_embeddings_doc_section
            ON section_embeddings(contract_stack_id, document_id, section_number)
            WHERE is_resolved = FALSE
    """)

    # Partial unique index for resolved-clause embeddings (query search):
    # One embedding per (stack, section) when is_resolved=TRUE
    op.execute("""
        CREATE UNIQUE INDEX uq_section_embeddings_resolved
            ON section_embeddings(contract_stack_id, section_number)
            WHERE is_resolved = TRUE
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_section_embeddings_resolved")
    op.execute("DROP INDEX IF EXISTS uq_section_embeddings_doc_section")
    op.execute("ALTER TABLE section_embeddings DROP COLUMN IF EXISTS is_resolved")
    # Restore original unique index
    op.execute("""
        CREATE UNIQUE INDEX uq_section_embeddings_stack_section
            ON section_embeddings(contract_stack_id, section_number)
    """)
