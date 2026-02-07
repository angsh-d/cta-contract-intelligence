"""Add UNIQUE constraints, missing indexes, updated_at trigger, fix columns.

Revision ID: 002
Revises: 001
Create Date: 2026-02-07
"""
from alembic import op

revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Fix contract_stacks: ensure name + study_name both work ──
    # The routes INSERT uses study_name; set name to default to study_name
    op.execute("ALTER TABLE contract_stacks ALTER COLUMN name DROP NOT NULL")
    op.execute("""
        ALTER TABLE contract_stacks ADD COLUMN IF NOT EXISTS
        status_updated_at TIMESTAMP DEFAULT NOW()
    """)

    # ── Add UNIQUE constraints required by ON CONFLICT clauses ──

    # clauses: orchestrator._save_resolved_clauses uses ON CONFLICT (contract_stack_id, section_number)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_clauses_stack_section
        ON clauses(contract_stack_id, section_number)
        WHERE is_current = TRUE
    """)

    # amendments: prevent duplicates on re-processing
    op.execute("""
        ALTER TABLE amendments
        ADD CONSTRAINT uq_amendments_stack_doc UNIQUE(contract_stack_id, document_id)
    """)

    # conflicts: prevent duplicates on re-processing
    op.execute("""
        ALTER TABLE conflicts
        ADD CONSTRAINT uq_conflicts_stack_conflict_id UNIQUE(contract_stack_id, conflict_id)
    """)

    # ── Additional indexes for query performance ──
    op.execute("CREATE INDEX IF NOT EXISTS idx_clauses_source_doc ON clauses(source_document_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_traces_job ON pipeline_traces(job_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_traces_stack ON pipeline_traces(contract_stack_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_conflicts_type ON conflicts(conflict_type)")

    # ── updated_at trigger function ──
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """)
    for table in ('contract_stacks', 'documents', 'clauses'):
        op.execute(f"""
            DROP TRIGGER IF EXISTS trg_{table}_updated_at ON {table};
            CREATE TRIGGER trg_{table}_updated_at
                BEFORE UPDATE ON {table}
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()
        """)


def downgrade() -> None:
    for table in ('contract_stacks', 'documents', 'clauses'):
        op.execute(f"DROP TRIGGER IF EXISTS trg_{table}_updated_at ON {table}")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column()")
    op.execute("DROP INDEX IF EXISTS idx_conflicts_type")
    op.execute("DROP INDEX IF EXISTS idx_pipeline_traces_stack")
    op.execute("DROP INDEX IF EXISTS idx_pipeline_traces_job")
    op.execute("DROP INDEX IF EXISTS idx_clauses_source_doc")
    op.execute("ALTER TABLE conflicts DROP CONSTRAINT IF EXISTS uq_conflicts_stack_conflict_id")
    op.execute("ALTER TABLE amendments DROP CONSTRAINT IF EXISTS uq_amendments_stack_doc")
    op.execute("DROP INDEX IF EXISTS uq_clauses_stack_section")
    op.execute("ALTER TABLE contract_stacks DROP COLUMN IF EXISTS status_updated_at")
    op.execute("ALTER TABLE contract_stacks ALTER COLUMN name SET NOT NULL")
