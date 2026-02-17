"""Add raw_text column to documents table for verbatim text preservation.

Stores the full extracted text from each PDF/DOCX so that downstream
stages can reference the exact source wording without re-extraction.

Revision ID: 005
Revises: 004
Create Date: 2026-02-16
"""
from alembic import op

revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS raw_text TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS raw_text")
