"""Initial schema — 9 tables for ContractIQ.

Revision ID: 001
Revises:
Create Date: 2026-02-07
"""
from alembic import op
import sqlalchemy as sa

revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # contract_stacks
    op.execute("""
        CREATE TABLE IF NOT EXISTS contract_stacks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            study_name VARCHAR(255),
            sponsor_name VARCHAR(255),
            site_name VARCHAR(255),
            study_protocol VARCHAR(100),
            therapeutic_area VARCHAR(100),
            start_date DATE,
            end_date DATE,
            status VARCHAR(50) DEFAULT 'active',
            processing_status VARCHAR(50) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_contract_stacks_name ON contract_stacks(name)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_contract_stacks_status ON contract_stacks(status)")

    # documents
    op.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            contract_stack_id UUID NOT NULL REFERENCES contract_stacks(id),
            document_type VARCHAR(50) NOT NULL,
            filename VARCHAR(255) NOT NULL,
            file_path TEXT NOT NULL,
            file_size_bytes INTEGER,
            upload_date TIMESTAMP DEFAULT NOW(),
            effective_date DATE,
            execution_date DATE,
            document_version VARCHAR(50),
            amendment_number INTEGER,
            processed BOOLEAN DEFAULT FALSE,
            processing_error TEXT,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_contract_stack ON documents(contract_stack_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_effective_date ON documents(effective_date)")

    # clauses
    op.execute("""
        CREATE TABLE IF NOT EXISTS clauses (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            contract_stack_id UUID NOT NULL REFERENCES contract_stacks(id),
            source_document_id UUID REFERENCES documents(id),
            section_number VARCHAR(50),
            section_title VARCHAR(255),
            clause_text TEXT NOT NULL,
            current_text TEXT,
            clause_category VARCHAR(100),
            is_current BOOLEAN DEFAULT TRUE,
            overridden_by_document_id UUID REFERENCES documents(id),
            effective_date DATE,
            source_chain JSONB,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_clauses_contract_stack ON clauses(contract_stack_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_clauses_section ON clauses(section_number)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_clauses_category ON clauses(clause_category)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_clauses_current ON clauses(is_current)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_clauses_stack_section ON clauses(contract_stack_id, section_number)")

    # amendments
    op.execute("""
        CREATE TABLE IF NOT EXISTS amendments (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL REFERENCES documents(id),
            contract_stack_id UUID NOT NULL REFERENCES contract_stacks(id),
            amendment_number INTEGER,
            amendment_type VARCHAR(100),
            sections_modified TEXT[],
            modification_type VARCHAR(50),
            rationale TEXT,
            modifications JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_amendments_contract_stack ON amendments(contract_stack_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_amendments_number ON amendments(amendment_number)")

    # conflicts
    op.execute("""
        CREATE TABLE IF NOT EXISTS conflicts (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            contract_stack_id UUID NOT NULL REFERENCES contract_stacks(id),
            conflict_id VARCHAR(255),
            conflict_type VARCHAR(100),
            severity VARCHAR(50),
            description TEXT NOT NULL,
            affected_sections TEXT[],
            evidence JSONB,
            recommendation TEXT,
            pain_point_id INTEGER,
            detected_at TIMESTAMP DEFAULT NOW(),
            resolved BOOLEAN DEFAULT FALSE,
            resolution_notes TEXT
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_conflicts_contract_stack ON conflicts(contract_stack_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_conflicts_severity ON conflicts(severity)")

    # queries
    op.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            contract_stack_id UUID REFERENCES contract_stacks(id),
            query_text TEXT NOT NULL,
            query_type VARCHAR(100),
            response JSONB,
            agent_reasoning JSONB,
            execution_time_ms INTEGER,
            success BOOLEAN,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_queries_contract_stack ON queries(contract_stack_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_queries_created_at ON queries(created_at)")

    # clause_dependencies
    op.execute("""
        CREATE TABLE IF NOT EXISTS clause_dependencies (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            contract_stack_id UUID NOT NULL REFERENCES contract_stacks(id),
            from_clause_id UUID NOT NULL REFERENCES clauses(id),
            to_clause_id UUID NOT NULL REFERENCES clauses(id),
            relationship_type VARCHAR(50) NOT NULL,
            description TEXT,
            confidence FLOAT DEFAULT 0.8,
            detection_method VARCHAR(50) DEFAULT 'llm',
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(contract_stack_id, from_clause_id, to_clause_id, relationship_type)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_clause_deps_stack ON clause_dependencies(contract_stack_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_clause_deps_from ON clause_dependencies(from_clause_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_clause_deps_to ON clause_dependencies(to_clause_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_clause_deps_type ON clause_dependencies(relationship_type)")

    # document_supersessions
    op.execute("""
        CREATE TABLE IF NOT EXISTS document_supersessions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            contract_stack_id UUID NOT NULL REFERENCES contract_stacks(id),
            predecessor_document_id UUID NOT NULL REFERENCES documents(id),
            successor_document_id UUID NOT NULL REFERENCES documents(id),
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(contract_stack_id, predecessor_document_id, successor_document_id)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_doc_supersessions_stack ON document_supersessions(contract_stack_id)")

    # pipeline_traces
    op.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_traces (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            job_id VARCHAR(255) NOT NULL,
            trace_id VARCHAR(255) NOT NULL,
            contract_stack_id UUID REFERENCES contract_stacks(id),
            total_input_tokens INT DEFAULT 0,
            total_output_tokens INT DEFAULT 0,
            total_llm_calls INT DEFAULT 0,
            total_latency_ms INT DEFAULT 0,
            llm_calls JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS pipeline_traces CASCADE")
    op.execute("DROP TABLE IF EXISTS document_supersessions CASCADE")
    op.execute("DROP TABLE IF EXISTS clause_dependencies CASCADE")
    op.execute("DROP TABLE IF EXISTS queries CASCADE")
    op.execute("DROP TABLE IF EXISTS conflicts CASCADE")
    op.execute("DROP TABLE IF EXISTS amendments CASCADE")
    op.execute("DROP TABLE IF EXISTS clauses CASCADE")
    op.execute("DROP TABLE IF EXISTS documents CASCADE")
    op.execute("DROP TABLE IF EXISTS contract_stacks CASCADE")
