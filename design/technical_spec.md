# ContractIQ - Technical Implementation Specification
## For Claude Code Implementation

---

# PROJECT OVERVIEW

**Project Name:** ContractIQ MVP  
**Objective:** Build agentic contract intelligence platform for clinical trial agreements  
**Target:** Demonstrate truth reconstitution, conflict detection, and ripple effect analysis  
**Timeline:** 8-week MVP, iterative development  
**Demo Scenario:** HEARTBEAT-3 contract stack (6 documents)

---

# SYSTEM ARCHITECTURE

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                         │
│  - Document Upload UI                                        │
│  - Query Interface                                           │
│  - Results Visualization                                     │
│  - Agent Reasoning Display                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓ HTTP/REST
┌─────────────────────────────────────────────────────────────┐
│                   API Layer (FastAPI)                        │
│  - Document ingestion endpoints                              │
│  - Query processing endpoints                                │
│  - Agent orchestration                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Agent Orchestrator (Python)                     │
│  - Manages agent workflows                                   │
│  - Coordinates multi-agent reasoning                         │
│  - Handles state and context                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────┬──────────────────┬─────────────────────┐
│  Document        │   Reasoning      │   Analysis          │
│  Agents          │   Agents         │   Agents            │
│  - Parser        │   - Override     │   - Truth           │
│  - Amendment     │   - Conflict     │   - Ripple          │
│  - Temporal      │   - Cross-Ref    │   - Risk            │
└──────────────────┴──────────────────┴─────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PostgreSQL + pgvector (NeonDB)                             │
│  - Structured data + clause dependency graph (recursive CTEs)│
│  - Two-tier vector embeddings (section_embeddings table):   │
│    * is_resolved=FALSE: Stage 1 per-document checkpoint     │
│      fallback (keyed by stack, doc, section)                │
│    * is_resolved=TRUE: Stage 4 resolved-clause query search │
│      (keyed by stack, section)                              │
│  - Semantic search via cosine distance (<=> operator)       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Azure OpenAI GPT-5.2 (Primary LLM)              │
│  - All agents: extraction, reasoning, classification         │
│  - Gemini (fallback LLM for all roles)                       │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Backend
- **Language:** Python 3.11+
- **Web Framework:** FastAPI
- **Async Runtime:** asyncio, aiohttp
- **Task Processing:** asyncio background tasks (in-memory)

### Databases
- **Relational:** PostgreSQL 15+ via NeonDB (primary data store + clause dependency graph via recursive CTEs)
- **Vector Store:** pgvector on NeonDB (section_embeddings table, HNSW cosine index). Two-tier architecture: `is_resolved=FALSE` rows serve as Stage 1 per-document checkpoint fallback (keyed by stack, doc, section); `is_resolved=TRUE` rows serve as Stage 4 resolved-clause query search (keyed by stack, section). Query-time searches filter on `is_resolved = TRUE`; checkpoint lookups filter on `is_resolved = FALSE`
- **Cache:** InMemoryCache (query result caching with TTL, replaces Redis)

### AI/ML
- **Primary LLM:** Azure OpenAI GPT-5.2 (all agents: extraction, complex reasoning, classification, synthesis)
- **Fallback LLM:** Google Gemini (gemini-3-pro-preview, automatic failover)
- **Embeddings:** Gemini gemini-embedding-001 (768-dim, via GEMINI_API_KEY in .env). Task types: `RETRIEVAL_DOCUMENT` for indexing, `RETRIEVAL_QUERY` for search
- **Document Processing:** PyMuPDF, pdfplumber, python-docx
- **NLP:** spaCy (optional for entity extraction)

### Frontend
- **Framework:** React 18+ with TypeScript
- **UI Library:** Tailwind CSS + shadcn/ui
- **State Management:** Zustand or React Query
- **Visualization:** Recharts, D3.js (for graphs)
- **Markdown:** react-markdown (for formatted responses)

### Infrastructure
- **Deployment:** Docker + Docker Compose (development)
- **Production:** AWS ECS or Google Cloud Run (future)
- **File Storage:** S3-compatible object storage
- **Monitoring:** Sentry, Prometheus (future)

---

# DATA MODELS

## PostgreSQL Schema

### documents
```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contract_stack_id UUID NOT NULL REFERENCES contract_stacks(id),
    document_type VARCHAR(50) NOT NULL, -- 'cta', 'amendment', 'exhibit'
    filename VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER,
    upload_date TIMESTAMP DEFAULT NOW(),
    effective_date DATE,
    execution_date DATE,
    document_version VARCHAR(50), -- e.g., 'Amendment 3', 'Protocol 2.0'
    processed BOOLEAN DEFAULT FALSE,
    processing_error TEXT,
    metadata JSONB, -- flexible storage for additional info
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_documents_contract_stack ON documents(contract_stack_id);
CREATE INDEX idx_documents_type ON documents(document_type);
CREATE INDEX idx_documents_effective_date ON documents(effective_date);
```

### contract_stacks
```sql
CREATE TABLE contract_stacks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL, -- e.g., 'HEARTBEAT-3 - Memorial Medical'
    sponsor_name VARCHAR(255),
    site_name VARCHAR(255),
    study_protocol VARCHAR(100), -- e.g., 'CP-2847-301'
    therapeutic_area VARCHAR(100),
    status VARCHAR(50) DEFAULT 'active', -- 'active', 'completed', 'archived'
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_contract_stacks_name ON contract_stacks(name);
CREATE INDEX idx_contract_stacks_status ON contract_stacks(status);
```

### clauses
```sql
CREATE TABLE clauses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contract_stack_id UUID NOT NULL REFERENCES contract_stacks(id),
    source_document_id UUID REFERENCES documents(id),
    section_number VARCHAR(50), -- e.g., '7.2', '12.1(a)'
    section_title VARCHAR(255), -- e.g., 'Payment Terms', 'Indemnification'
    clause_text TEXT NOT NULL,
    clause_category VARCHAR(100), -- 'payment', 'indemnification', 'data_retention', etc.
    is_current BOOLEAN DEFAULT TRUE, -- false if overridden by amendment
    overridden_by_document_id UUID REFERENCES documents(id),
    effective_date DATE,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_clauses_contract_stack ON clauses(contract_stack_id);
CREATE INDEX idx_clauses_section ON clauses(section_number);
CREATE INDEX idx_clauses_category ON clauses(clause_category);
CREATE INDEX idx_clauses_current ON clauses(is_current);
```

### amendments
```sql
CREATE TABLE amendments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id),
    contract_stack_id UUID NOT NULL REFERENCES contract_stacks(id),
    amendment_number INTEGER, -- 1, 2, 3, etc.
    amendment_type VARCHAR(100), -- 'protocol_change', 'budget_revision', 'pi_change', etc.
    sections_modified TEXT[], -- array of section numbers modified
    modification_type VARCHAR(50), -- 'complete_replacement', 'selective_override', 'addition'
    rationale TEXT, -- extracted from recitals
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_amendments_contract_stack ON amendments(contract_stack_id);
CREATE INDEX idx_amendments_number ON amendments(amendment_number);
```

### conflicts
```sql
CREATE TABLE conflicts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contract_stack_id UUID NOT NULL REFERENCES contract_stacks(id),
    conflict_type VARCHAR(100), -- 'contradiction', 'ambiguity', 'gap', 'inconsistency'
    severity VARCHAR(50), -- 'critical', 'high', 'medium', 'low'
    clause_ids UUID[], -- array of clause IDs involved in conflict
    description TEXT NOT NULL,
    recommendation TEXT,
    detected_at TIMESTAMP DEFAULT NOW(),
    resolved BOOLEAN DEFAULT FALSE,
    resolution_notes TEXT
);

CREATE INDEX idx_conflicts_contract_stack ON conflicts(contract_stack_id);
CREATE INDEX idx_conflicts_severity ON conflicts(severity);
CREATE INDEX idx_conflicts_resolved ON conflicts(resolved);
```

### queries
```sql
CREATE TABLE queries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contract_stack_id UUID REFERENCES contract_stacks(id),
    query_text TEXT NOT NULL,
    query_type VARCHAR(100), -- 'truth_reconstitution', 'conflict_detection', 'ripple_analysis'
    response TEXT,
    agent_reasoning JSONB, -- stores the multi-agent conversation
    execution_time_ms INTEGER,
    success BOOLEAN,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_queries_contract_stack ON queries(contract_stack_id);
CREATE INDEX idx_queries_created_at ON queries(created_at);
```

## Clause Dependency Graph (PostgreSQL)

Clause relationships are stored in PostgreSQL using a `clause_dependencies` table. Multi-hop traversal (up to 5 hops for ripple effect analysis) uses **recursive CTEs**, which are efficient for the expected data volume (~50-100 clauses, ~200-500 edges per contract stack).

### clause_dependencies
```sql
CREATE TABLE clause_dependencies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contract_stack_id UUID NOT NULL REFERENCES contract_stacks(id),
    from_clause_id UUID NOT NULL REFERENCES clauses(id),
    to_clause_id UUID NOT NULL REFERENCES clauses(id),
    relationship_type VARCHAR(50) NOT NULL, -- 'depends_on', 'references', 'modifies', 'replaces', 'amends', 'conflicts_with', 'supersedes'
    description TEXT,
    confidence FLOAT DEFAULT 0.8,
    detection_method VARCHAR(50) DEFAULT 'llm', -- 'llm' (all dependencies identified by LLM)
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_clause_deps_stack ON clause_dependencies(contract_stack_id);
CREATE INDEX idx_clause_deps_from ON clause_dependencies(from_clause_id);
CREATE INDEX idx_clause_deps_to ON clause_dependencies(to_clause_id);
CREATE INDEX idx_clause_deps_type ON clause_dependencies(relationship_type);
```

### document_supersessions
```sql
CREATE TABLE document_supersessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contract_stack_id UUID NOT NULL REFERENCES contract_stacks(id),
    predecessor_document_id UUID NOT NULL REFERENCES documents(id),
    successor_document_id UUID NOT NULL REFERENCES documents(id),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_doc_supersessions_stack ON document_supersessions(contract_stack_id);
```

### section_embeddings
```sql
CREATE TABLE section_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contract_stack_id UUID NOT NULL REFERENCES contract_stacks(id),
    document_id UUID REFERENCES documents(id),          -- populated for checkpoint rows; NULL for resolved rows
    section_number VARCHAR(50) NOT NULL,
    section_title VARCHAR(255),
    content_text TEXT NOT NULL,                          -- the text that was embedded
    embedding vector(768) NOT NULL,                     -- gemini-embedding-001 output
    is_resolved BOOLEAN NOT NULL DEFAULT FALSE,         -- FALSE = Stage 1 checkpoint, TRUE = Stage 4 resolved
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Two-tier partial unique indexes:
-- Stage 1 checkpoint fallback: one embedding per (stack, document, section)
CREATE UNIQUE INDEX uq_section_embeddings_doc_section
    ON section_embeddings (contract_stack_id, document_id, section_number)
    WHERE is_resolved = FALSE;

-- Stage 4 resolved-clause query search: one embedding per (stack, section)
CREATE UNIQUE INDEX uq_section_embeddings_resolved
    ON section_embeddings (contract_stack_id, section_number)
    WHERE is_resolved = TRUE;

-- HNSW index for fast cosine similarity search on resolved embeddings
CREATE INDEX idx_section_embeddings_hnsw
    ON section_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX idx_section_embeddings_stack ON section_embeddings(contract_stack_id);
CREATE INDEX idx_section_embeddings_resolved ON section_embeddings(is_resolved);
```

### Multi-Hop Traversal (Recursive CTE)
```sql
-- Find all clauses affected by a change to a given clause (up to 5 hops)
WITH RECURSIVE dependency_chain AS (
    -- Base case: direct dependencies from the changed clause
    SELECT cd.from_clause_id, cd.to_clause_id, cd.relationship_type,
           cd.description, cd.confidence, 1 AS hop,
           ARRAY[cd.from_clause_id] AS path
    FROM clause_dependencies cd
    WHERE cd.from_clause_id = $changed_clause_id
      AND cd.contract_stack_id = $stack_id

    UNION ALL

    -- Recursive case: follow dependencies up to 5 hops
    SELECT cd.from_clause_id, cd.to_clause_id, cd.relationship_type,
           cd.description, cd.confidence, dc.hop + 1,
           dc.path || cd.from_clause_id
    FROM clause_dependencies cd
    JOIN dependency_chain dc ON cd.from_clause_id = dc.to_clause_id
    WHERE dc.hop < 5
      AND cd.to_clause_id != ALL(dc.path)  -- prevent cycles
      AND cd.contract_stack_id = $stack_id
)
SELECT DISTINCT ON (to_clause_id)
    to_clause_id, relationship_type, description, confidence, hop, path
FROM dependency_chain
ORDER BY to_clause_id, hop;  -- prefer shortest path
```

---

# API SPECIFICATIONS

## REST API Endpoints

### Document Management

**POST /api/v1/contract-stacks**
Create a new contract stack
```json
Request:
{
    "name": "HEARTBEAT-3 - Memorial Medical",
    "sponsor_name": "CardioPharm International",
    "site_name": "Memorial Medical Center",
    "study_protocol": "CP-2847-301",
    "therapeutic_area": "Cardiology"
}

Response:
{
    "id": "uuid",
    "name": "HEARTBEAT-3 - Memorial Medical",
    "status": "active",
    "created_at": "2024-01-15T10:00:00Z"
}
```

**POST /api/v1/contract-stacks/{stack_id}/documents**
Upload document to contract stack
```json
Request: multipart/form-data
- file: (binary)
- document_type: "cta" | "amendment" | "exhibit"
- effective_date: "2022-01-15" (optional)
- document_version: "Amendment 3" (optional)

Response:
{
    "id": "uuid",
    "filename": "Amendment_3.pdf",
    "status": "processing",
    "processing_job_id": "uuid"
}
```

**GET /api/v1/contract-stacks/{stack_id}/documents**
List all documents in a contract stack
```json
Response:
{
    "contract_stack_id": "uuid",
    "documents": [
        {
            "id": "uuid",
            "document_type": "cta",
            "filename": "Original_CTA.pdf",
            "effective_date": "2022-01-15",
            "processed": true
        },
        ...
    ]
}
```

### Query & Analysis

**POST /api/v1/contract-stacks/{stack_id}/query**
Submit a query for analysis
```json
Request:
{
    "query": "What are the current payment terms?",
    "query_type": "truth_reconstitution", // optional
    "include_reasoning": true // show agent conversation
}

Response:
{
    "query_id": "uuid",
    "query": "What are the current payment terms?",
    "response": {
        "answer": "Net 45 days per Amendment 3, Section 4, dated August 17, 2023",
        "sources": [
            {
                "document_id": "uuid",
                "document_name": "Amendment_3.pdf",
                "section": "7.2",
                "text": "...Sponsor shall pay undisputed invoices within forty-five (45) days...",
                "effective_date": "2023-08-17"
            }
        ],
        "confidence": 0.98
    },
    "agent_reasoning": [
        {
            "agent": "query_router",
            "action": "routing query to truth_reconstitution_agent",
            "timestamp": "2024-01-15T10:00:01Z"
        },
        ...
    ],
    "execution_time_ms": 8234
}
```

**POST /api/v1/contract-stacks/{stack_id}/analyze/conflicts**
Detect conflicts in contract stack
```json
Request:
{
    "severity_threshold": "medium" // only return medium+ severity
}

Response:
{
    "conflicts": [
        {
            "id": "uuid",
            "conflict_type": "buried_change",
            "severity": "high",
            "description": "Payment terms changed from Net 30 to Net 45 in Amendment 3, buried in COVID-related amendment",
            "affected_clauses": [
                {
                    "section": "7.2",
                    "original_text": "...Net 30...",
                    "current_text": "...Net 45..."
                }
            ],
            "recommendation": "Update budget documents to reflect Net 45 payment terms"
        },
        ...
    ],
    "summary": {
        "critical": 0,
        "high": 2,
        "medium": 3,
        "low": 1
    }
}
```

**POST /api/v1/contract-stacks/{stack_id}/analyze/ripple-effects**
Analyze ripple effects of proposed amendment
```json
Request:
{
    "proposed_change": {
        "section": "9.2",
        "current_text": "15 years data retention",
        "proposed_text": "25 years data retention"
    }
}

Response:
{
    "direct_impacts": [
        {
            "section": "9.4",
            "impact_type": "cost_increase",
            "description": "Archive storage costs increase by 8 years",
            "estimated_cost": "$3,000-5,000"
        }
    ],
    "indirect_impacts": [
        {
            "section": "12.1",
            "impact_type": "indemnification_gap",
            "description": "Indemnification covers 7 years but data retention is 25 years",
            "severity": "critical",
            "recommendation": "Extend indemnification to 25 years"
        },
        ...
    ],
    "cascade_depth": 3,
    "total_impacts": 18,
    "estimated_total_cost": "$1,800,000 - $2,340,000",
    "recommendations": [
        "Extend indemnification coverage to 25 years",
        "Update insurance policy duration",
        "Renegotiate vendor contracts for extended retention"
    ]
}
```

### Real-Time Processing

**GET /api/v1/jobs/{job_id}/status**
Check status of background processing job
```json
Response:
{
    "job_id": "uuid",
    "status": "processing", // "queued", "processing", "completed", "failed"
    "progress": 45, // percentage
    "message": "Extracting clauses from Amendment 3...",
    "created_at": "2024-01-15T10:00:00Z",
    "updated_at": "2024-01-15T10:00:30Z"
}
```

**WebSocket /api/v1/ws/jobs/{job_id}**
Real-time updates for processing jobs
```json
Message format:
{
    "type": "progress",
    "job_id": "uuid",
    "progress": 60,
    "message": "Building dependency graph..."
}
```

---

# AGENT IMPLEMENTATIONS

## Agent Architecture

Each agent follows this pattern:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from app.agents.llm_providers import LLMProvider

class BaseAgent(ABC):
    def __init__(self, config: AgentConfig, llm_provider: LLMProvider,
                 prompt_loader: PromptLoader, fallback_provider: LLMProvider = None,
                 llm_semaphore: asyncio.Semaphore = None):
        self.config = config
        self.llm_provider = llm_provider          # Azure OpenAI GPT-5.2 (primary)
        self.fallback_provider = fallback_provider  # Gemini (fallback)
        self.prompt_loader = prompt_loader
        self._llm_semaphore = llm_semaphore

    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing method - must be implemented by each agent"""
        pass

    async def call_llm(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_output_tokens: int = None,
        temperature: float = 0.0,
        response_format: str = None,
    ) -> LLMResponse:
        """Wrapper for LLM API calls with retry, fallback, and error handling"""
        try:
            response = await self.llm_provider.complete(
                system_prompt=system_prompt,
                user_message=user_message,
                max_output_tokens=max_output_tokens or self.config.max_output_tokens,
                temperature=temperature,
                response_format=response_format,
            )
            return response
        except Exception as e:
            if self.fallback_provider:
                return await self.fallback_provider.complete(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    max_output_tokens=max_output_tokens or self.config.max_output_tokens,
                    temperature=temperature,
                    response_format=response_format,
                )
            raise
```

## Tier 1: Document Ingestion Agents

### DocumentParserAgent

**Purpose:** Extract text, structure, and metadata from PDFs/Word docs

**Implementation:**
```python
class DocumentParserAgent(BaseAgent):
    """LLM: Azure OpenAI GPT-5.2 (role=extraction), Gemini fallback.
    Config: max_output_tokens=16000, verification_threshold=0.80"""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input:
            - file_path: path to document
            - document_type: 'cta', 'amendment', 'exhibit'

        Output:
            - extracted_text: full text
            - sections: list of {section_number, title, text}
            - tables: list of extracted tables
            - metadata: {effective_date, parties, etc}
        """
        file_path = input_data['file_path']
        document_type = input_data['document_type']

        # Step 1: Extract raw text
        raw_text = self._extract_text(file_path)

        # Step 2: Use Azure OpenAI GPT-5.2 to structure the text
        system_prompt = """You are a legal document parser specializing in clinical trial agreements.
        
        Extract the following from the document:
        1. Document type (CTA, Amendment, Exhibit)
        2. Effective date
        3. Parties (Sponsor name, Institution name)
        4. Section structure (section numbers, titles, text)
        5. Any tables (convert to structured format)
        
        Return as JSON."""
        
        user_message = f"""Parse this {document_type}:

{raw_text}

Return structured JSON with sections, metadata, and tables."""
        
        result = await self.call_llm(system_prompt, user_message)
        
        if result['success']:
            parsed_data = json.loads(result['content'][0].text)
            return {
                "success": True,
                "parsed_data": parsed_data,
                "raw_text": raw_text
            }
        else:
            return result
            
    def _extract_text(self, file_path: str) -> str:
        """Extract raw text using PyMuPDF or pdfplumber"""
        import fitz  # PyMuPDF
        
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
```

### AmendmentTrackerAgent

**Purpose:** Identify what each amendment modifies

**Implementation:**
```python
class AmendmentTrackerAgent(BaseAgent):
    """LLM: Azure OpenAI GPT-5.2 (role=complex_reasoning), Gemini fallback.
    Config: max_output_tokens=8192, timeout_seconds=180, verification_threshold=0.75"""
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input:
            - amendment_text: full text of amendment
            - amendment_number: 1, 2, 3, etc.
            
        Output:
            - sections_modified: list of section numbers
            - modification_type: 'complete_replacement', 'selective_override', 'addition'
            - modification_details: for each section, what changed
            - rationale: extracted from recitals
        """
        
        system_prompt = """You are an expert at analyzing contract amendments.

Your task is to identify:
1. Which sections are modified (look for "Section X is hereby amended", "Section Y is deleted and replaced", etc.)
2. Type of modification:
   - complete_replacement: entire section replaced
   - selective_override: only part of section changed (e.g., "Section 7.2 is amended by deleting 'Net 30' and substituting 'Net 45'")
   - addition: new section added
3. For each modification, extract the OLD text and NEW text
4. Extract the rationale from the WHEREAS clauses

Return as structured JSON."""

        user_message = f"""Analyze this amendment:

{input_data['amendment_text']}

Return JSON with:
{{
    "sections_modified": ["7.2", "12.1"],
    "modifications": [
        {{
            "section": "7.2",
            "type": "selective_override",
            "old_text": "...Net 30...",
            "new_text": "...Net 45...",
            "change_description": "Payment terms changed from Net 30 to Net 45"
        }}
    ],
    "rationale": "Extracted from recitals"
}}"""

        result = await self.call_llm(system_prompt, user_message)
        
        if result['success']:
            return {
                "success": True,
                "tracking_data": json.loads(result['content'][0].text)
            }
        else:
            return result
```

### TemporalSequencerAgent

**Purpose:** Order documents chronologically and track versioning

**Implementation:**
```python
class TemporalSequencerAgent(BaseAgent):
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input:
            - documents: list of {id, effective_date, document_type, version}
            
        Output:
            - chronological_order: ordered list of document IDs
            - version_tree: mapping of which documents supersede which
            - timeline: visual timeline data
        """
        documents = input_data['documents']
        
        # Sort by effective_date
        sorted_docs = sorted(documents, key=lambda x: x['effective_date'])
        
        # Build version tree
        version_tree = {}
        original_cta = next((d for d in sorted_docs if d['document_type'] == 'cta'), None)
        
        if original_cta:
            version_tree['root'] = original_cta['id']
            version_tree['amendments'] = []
            
            for doc in sorted_docs:
                if doc['document_type'] == 'amendment':
                    version_tree['amendments'].append({
                        'id': doc['id'],
                        'version': doc['version'],
                        'effective_date': doc['effective_date'],
                        'supersedes': version_tree['amendments'][-1]['id'] if version_tree['amendments'] else original_cta['id']
                    })
        
        return {
            "success": True,
            "chronological_order": [d['id'] for d in sorted_docs],
            "version_tree": version_tree,
            "timeline": self._create_timeline(sorted_docs)
        }
        
    def _create_timeline(self, documents: List[Dict]) -> List[Dict]:
        """Create timeline visualization data"""
        return [
            {
                "date": doc['effective_date'],
                "event": f"{doc['document_type']}: {doc['version']}",
                "document_id": doc['id']
            }
            for doc in documents
        ]
```

## Tier 2: Reasoning Agents

### OverrideResolutionAgent

**Purpose:** Determine current state of each clause after amendments

**Implementation:**
```python
class OverrideResolutionAgent(BaseAgent):
    """LLM: Azure OpenAI GPT-5.2 (role=complex_reasoning), Gemini fallback.
    Config: max_output_tokens=8192, timeout_seconds=180, verification_threshold=0.75"""
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input:
            - section_number: "7.2"
            - original_clause: {text, source_doc}
            - amendments: list of amendments that touched this section
            
        Output:
            - current_text: the text after all amendments applied
            - source_chain: provenance showing how we got here
            - confidence: 0-1 score
        """
        
        section = input_data['section_number']
        original = input_data['original_clause']
        amendments = input_data['amendments']
        
        system_prompt = """You are an expert at applying contract amendments to determine current clause text.

Given an original clause and a series of amendments, determine the CURRENT state of the clause.

Rules:
1. "Section X is hereby deleted in its entirety and replaced with..." → complete replacement
2. "Section X is amended by deleting 'Y' and substituting 'Z'" → selective override (only change Y to Z)
3. "A new Section X.Y is hereby added" → addition (doesn't affect existing sections)
4. Apply amendments in chronological order
5. Later amendments override earlier amendments

Return the final current text with full provenance."""

        # Build amendment chain
        amendment_chain = ""
        for i, amend in enumerate(amendments, 1):
            amendment_chain += f"\n\nAmendment {i} (Effective {amend['effective_date']}):\n{amend['modification_text']}"
        
        user_message = f"""Original Clause (Section {section}):
{original['text']}
Source: {original['source_doc']}

Amendments:{amendment_chain}

Determine the CURRENT text of Section {section} after applying all amendments in order.

Return JSON:
{{
    "current_text": "...",
    "source_chain": [
        {{"stage": "original", "text": "...", "source": "..."}},
        {{"stage": "amendment_1", "text": "...", "source": "...", "change": "..."}}
    ],
    "confidence": 0.95
}}"""

        result = await self.call_llm(system_prompt, user_message)
        
        if result['success']:
            return {
                "success": True,
                "resolution": json.loads(result['content'][0].text)
            }
        else:
            return result
```

### ConflictDetectionAgent

**Purpose:** Find contradictions and inconsistencies

**Implementation:**
```python
class ConflictDetectionAgent(BaseAgent):
    """LLM: Azure OpenAI GPT-5.2 (role=complex_reasoning), Gemini fallback.
    Config: max_output_tokens=16000, timeout_seconds=300, temperature=0.2, verification_threshold=0.70"""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input:
            - clauses: list of all current clauses
            - contract_stack_context: metadata about the contract
            
        Output:
            - conflicts: list of detected conflicts
            - each conflict includes: type, severity, description, recommendation
        """
        
        clauses = input_data['clauses']
        context = input_data.get('contract_stack_context', {})
        
        system_prompt = """You are an expert at detecting conflicts in clinical trial agreements.

Analyze the provided clauses and identify:

1. **Contradictions**: Clauses that directly contradict each other
   Example: Section 7.2 says "Net 30" but budget shows "Net 45"

2. **Gaps**: Missing information or coverage gaps
   Example: Insurance covers study through Dec 2024 but study extends to June 2025

3. **Ambiguities**: Unclear or inconsistent language
   Example: Amendment 4 says "Protocol as amended" without specifying which version

4. **Buried Changes**: Important changes hidden in unrelated amendments
   Example: Payment terms changed in COVID amendment

For each conflict, assess severity:
- critical: Blocks contract execution, legal risk
- high: Significant operational impact
- medium: Requires clarification
- low: Minor inconsistency

Return structured JSON."""

        # Group clauses by category for analysis
        payment_clauses = [c for c in clauses if c['category'] == 'payment']
        insurance_clauses = [c for c in clauses if c['category'] == 'insurance']
        # ... etc
        
        user_message = f"""Analyze this contract for conflicts:

Payment Clauses:
{json.dumps(payment_clauses, indent=2)}

Insurance Clauses:
{json.dumps(insurance_clauses, indent=2)}

Context:
- Study ends: {context.get('study_end_date', 'unknown')}
- Therapeutic area: {context.get('therapeutic_area', 'unknown')}

Identify all conflicts. Return JSON:
{{
    "conflicts": [
        {{
            "type": "buried_change",
            "severity": "high",
            "affected_clauses": ["7.2"],
            "description": "...",
            "recommendation": "..."
        }}
    ]
}}"""

        result = await self.call_llm(system_prompt, user_message)
        
        if result['success']:
            conflicts = json.loads(result['content'][0].text)['conflicts']
            return {
                "success": True,
                "conflicts": conflicts
            }
        else:
            return result
```

### DependencyMapperAgent

**Purpose:** Build knowledge graph of clause relationships

**Implementation:**
```python
class DependencyMapperAgent(BaseAgent):
    """LLM: Azure OpenAI GPT-5.2 (role=complex_reasoning), Gemini fallback.
    Config: max_output_tokens=16000, temperature=0.1, verification_threshold=0.75"""

    def __init__(self, config, llm_provider, prompt_loader, db_pool, **kwargs):
        super().__init__(config, llm_provider, prompt_loader, **kwargs)
        self.db = db_pool
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input:
            - clauses: all clauses in contract
            
        Output:
            - dependency_graph: nodes and edges
            - saves to PostgreSQL clause_dependencies table
        """
        
        clauses = input_data['clauses']
        contract_stack_id = input_data['contract_stack_id']
        
        # Step 1: Use Azure OpenAI GPT-5.2 to identify ALL dependencies (explicit + semantic)
        system_prompt = """You are an expert at identifying dependencies between contract clauses.

Analyze clauses and identify ALL relationships in a single pass:

1. **Explicit references**: "as defined in Section X", "per Exhibit B"
2. **Functional dependencies**: Payment terms depend on budget
3. **Conditional dependencies**: "Subject to Section Y"
4. **Temporal dependencies**: Insurance duration depends on study completion date

Return as graph structure."""

        user_message = f"""Identify dependencies between these clauses:

{json.dumps(clauses, indent=2)}

Return JSON:
{{
    "dependencies": [
        {{
            "from_clause": "7.2",
            "to_clause": "7.4",
            "relationship_type": "depends_on",
            "description": "Payment terms reference holdback clause"
        }}
    ]
}}"""

        result = await self.call_llm(system_prompt, user_message)
        
        if result['success']:
            dependencies = json.loads(result['content'][0].text)['dependencies']
            
            # Step 3: Save to PostgreSQL
            await self._save_dependencies(contract_stack_id, clauses, dependencies)

            return {
                "success": True,
                "dependencies": dependencies
            }
        else:
            return result

    async def _save_dependencies(self, stack_id: str, clauses: List[Dict], dependencies: List[Dict]):
        """Save clause dependencies to PostgreSQL"""
        async with self.db.acquire() as conn:
            for dep in dependencies:
                await conn.execute(
                    """
                    INSERT INTO clause_dependencies
                        (contract_stack_id, from_clause_id, to_clause_id,
                         relationship_type, description, detection_method)
                    SELECT $1, f.id, t.id, $4, $5, 'llm'
                    FROM clauses f, clauses t
                    WHERE f.section_number = $2 AND f.contract_stack_id = $1
                      AND t.section_number = $3 AND t.contract_stack_id = $1
                    ON CONFLICT DO NOTHING
                    """,
                    stack_id, dep['from_clause'], dep['to_clause'],
                    dep['relationship_type'], dep['description']
                )
```

## Tier 3: Analysis Agents

### RippleEffectAnalyzerAgent

**Purpose:** Multi-hop reasoning for cascade impacts

**Implementation:**
```python
class RippleEffectAnalyzerAgent(BaseAgent):
    """LLM: Azure OpenAI GPT-5.2 (role=complex_reasoning), Gemini fallback.
    Config: max_output_tokens=16000, timeout_seconds=300, temperature=0.2, verification_threshold=0.70"""

    def __init__(self, config, llm_provider, prompt_loader, db_pool, **kwargs):
        super().__init__(config, llm_provider, prompt_loader, **kwargs)
        self.db = db_pool
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input:
            - proposed_change: {section, old_text, new_text}
            - contract_stack_id
            
        Output:
            - impacts: hierarchical list of impacts by hop distance
            - recommendations: prioritized actions
        """
        
        change = input_data['proposed_change']
        stack_id = input_data['contract_stack_id']
        
        # Step 1: Query PostgreSQL for dependency graph
        dependency_paths = await self._get_dependency_paths(stack_id, change['section'])
        
        # Step 2: Analyze each hop using Azure OpenAI GPT-5.2
        impacts_by_hop = {}
        
        for hop in range(1, 6):  # up to 5 hops
            hop_clauses = [p for p in dependency_paths if len(p) == hop + 1]
            
            if not hop_clauses:
                break
                
            system_prompt = f"""You are analyzing ripple effects of a contract amendment.

A change is proposed to Section {change['section']}:
OLD: {change['old_text']}
NEW: {change['new_text']}

You are analyzing HOP {hop} impacts - clauses that are {hop} step(s) away in the dependency graph.

For each clause, determine:
1. Is it materially impacted by this change?
2. What is the nature of the impact?
3. What action is required?
4. What is the estimated cost/timeline impact?

Be specific and concrete."""

            user_message = f"""Analyze these Hop {hop} clauses:

{json.dumps(hop_clauses, indent=2)}

Return JSON with material impacts only:
{{
    "impacts": [
        {{
            "affected_section": "12.1",
            "impact_type": "indemnification_gap",
            "severity": "critical",
            "description": "...",
            "required_action": "...",
            "estimated_cost": "...",
            "estimated_timeline": "..."
        }}
    ]
}}"""

            result = await self.call_llm(system_prompt, user_message)
            
            if result['success']:
                hop_impacts = json.loads(result['content'][0].text)['impacts']
                impacts_by_hop[f'hop_{hop}'] = hop_impacts
        
        # Step 3: Synthesize recommendations
        all_impacts = []
        for hop_impacts in impacts_by_hop.values():
            all_impacts.extend(hop_impacts)
            
        recommendations = await self._generate_recommendations(all_impacts)
        
        return {
            "success": True,
            "impacts_by_hop": impacts_by_hop,
            "total_impacts": len(all_impacts),
            "cascade_depth": len(impacts_by_hop),
            "recommendations": recommendations
        }
        
    async def _get_dependency_paths(self, stack_id: str, from_section: str) -> List[List[Dict]]:
        """Query PostgreSQL for dependency paths using recursive CTE"""
        async with self.db.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH RECURSIVE dependency_chain AS (
                    SELECT cd.from_clause_id, cd.to_clause_id,
                           cd.relationship_type, cd.description, 1 AS hop,
                           ARRAY[cd.from_clause_id] AS path
                    FROM clause_dependencies cd
                    JOIN clauses c ON c.id = cd.from_clause_id
                    WHERE c.section_number = $1
                      AND cd.contract_stack_id = $2

                    UNION ALL

                    SELECT cd.from_clause_id, cd.to_clause_id,
                           cd.relationship_type, cd.description, dc.hop + 1,
                           dc.path || cd.from_clause_id
                    FROM clause_dependencies cd
                    JOIN dependency_chain dc ON cd.from_clause_id = dc.to_clause_id
                    WHERE dc.hop < 5
                      AND cd.to_clause_id != ALL(dc.path)
                      AND cd.contract_stack_id = $2
                )
                SELECT dc.to_clause_id, dc.hop, dc.relationship_type,
                       c.section_number, c.section_title, c.clause_category
                FROM dependency_chain dc
                JOIN clauses c ON c.id = dc.to_clause_id
                ORDER BY dc.hop
                """,
                from_section, stack_id
            )

            # Group by hop distance
            paths_by_hop = {}
            for row in rows:
                hop = row['hop']
                if hop not in paths_by_hop:
                    paths_by_hop[hop] = []
                paths_by_hop[hop].append({
                    'section': row['section_number'],
                    'title': row['section_title'],
                    'category': row['clause_category']
                })

            return paths_by_hop
            
    async def _generate_recommendations(self, impacts: List[Dict]) -> List[Dict]:
        """Use Azure OpenAI GPT-5.2 to prioritize and synthesize recommendations"""
        system_prompt = """You are synthesizing ripple effect analysis into actionable recommendations.

Prioritize by:
1. Severity (critical → high → medium → low)
2. Timeline urgency
3. Cost impact
4. Complexity

Group related impacts and provide clear action items."""

        user_message = f"""Synthesize these impacts into prioritized recommendations:

{json.dumps(impacts, indent=2)}

Return JSON:
{{
    "critical_actions": [
        {{
            "priority": 1,
            "action": "...",
            "reason": "...",
            "estimated_cost": "...",
            "deadline": "..."
        }}
    ],
    "recommended_actions": [...],
    "optional_actions": [...]
}}"""

        result = await self.call_llm(system_prompt, user_message)
        
        if result['success']:
            return json.loads(result['content'][0].text)
        else:
            return {"error": "Failed to generate recommendations"}
```

## Agent Orchestrator

**Purpose:** Coordinate multi-agent workflows

**Implementation:**
```python
class AgentOrchestrator:
    def __init__(self, postgres_pool, vector_store):
        self.postgres = postgres_pool
        self.vector_store = vector_store
        self.blackboard = InMemoryBlackboard()
        self.cache = InMemoryCache()

        # LLMProviderFactory resolves Azure OpenAI GPT-5.2 (primary) + Gemini (fallback) per role
        # Initialize all agents with config, provider, prompt_loader, fallback, and semaphore
        self._init_agents(prompt_loader)
        
    async def process_contract_stack(self, contract_stack_id: str) -> Dict[str, Any]:
        """
        Full pipeline to process a contract stack:
        1. Parse all documents
        2. Sequence them temporally
        3. Track amendments
        4. Resolve overrides
        5. Build dependency graph
        6. Detect conflicts
        """
        
        # Step 1: Get all documents
        documents = await self._get_documents(contract_stack_id)
        
        # Step 2: Parse each document
        parsed_docs = []
        for doc in documents:
            parse_result = await self.parser.process({
                'file_path': doc['file_path'],
                'document_type': doc['document_type']
            })
            parsed_docs.append(parse_result['parsed_data'])
            
        # Step 3: Sequence documents
        sequence_result = await self.temporal_sequencer.process({
            'documents': [
                {
                    'id': doc['id'],
                    'effective_date': doc['effective_date'],
                    'document_type': doc['document_type'],
                    'version': doc['document_version']
                }
                for doc in documents
            ]
        })
        
        # Step 4: Track amendments
        amendments = [d for d in documents if d['document_type'] == 'amendment']
        tracking_results = []
        
        for amendment in amendments:
            track_result = await self.amendment_tracker.process({
                'amendment_text': amendment['text'],
                'amendment_number': amendment['amendment_number']
            })
            tracking_results.append(track_result)
            
        # Step 5: Resolve current state of all clauses
        # Group clauses by section, apply amendments in order
        clauses_by_section = self._group_clauses_by_section(parsed_docs)
        current_clauses = []
        
        for section, clause_history in clauses_by_section.items():
            resolution = await self.override_resolver.process({
                'section_number': section,
                'original_clause': clause_history[0],
                'amendments': [t for t in tracking_results if section in t['sections_modified']]
            })
            current_clauses.append(resolution['resolution'])
            
        # Step 6: Build dependency graph
        await self.dependency_mapper.process({
            'clauses': current_clauses,
            'contract_stack_id': contract_stack_id
        })
        
        # Step 7: Detect conflicts
        conflicts = await self.conflict_detector.process({
            'clauses': current_clauses,
            'contract_stack_context': {
                'study_end_date': '2025-06-30',  # from DB
                'therapeutic_area': 'Cardiology'
            }
        })
        
        # Save to database
        await self._save_processing_results(
            contract_stack_id,
            current_clauses,
            conflicts['conflicts']
        )
        
        return {
            "success": True,
            "clauses_processed": len(current_clauses),
            "conflicts_detected": len(conflicts['conflicts']),
            "dependency_graph_built": True
        }
        
    async def handle_query(self, query: str, contract_stack_id: str) -> Dict[str, Any]:
        """
        Route and handle user query
        """
        
        # Step 1: Classify query type
        query_type = await self._classify_query(query)
        
        # Step 2: Route to appropriate workflow
        if query_type == 'truth_reconstitution':
            return await self._handle_truth_query(query, contract_stack_id)
            
        elif query_type == 'conflict_detection':
            return await self._handle_conflict_query(query, contract_stack_id)
            
        elif query_type == 'ripple_analysis':
            return await self._handle_ripple_query(query, contract_stack_id)
            
        else:
            return {"error": "Unknown query type"}
            
    async def _classify_query(self, query: str) -> str:
        """Use QueryRouter agent (Azure OpenAI GPT-5.2) to classify query type"""
        # Delegates to QueryRouter agent which uses Azure OpenAI GPT-5.2
        # with max_output_tokens=1024, verification_threshold=0.85
        route_output = await self.agents["query_router"].run(
            QueryRouteInput(query_text=query)
        )
        return route_output.query_type
        
    async def _handle_truth_query(self, query: str, contract_stack_id: str) -> Dict[str, Any]:
        """Handle truth reconstitution queries"""

        # Get relevant clauses from pgvector semantic search (is_resolved=TRUE)
        relevant_clauses = await self._retrieve_clauses(query, contract_stack_id)

        # Use TruthSynthesizer agent (Azure OpenAI GPT-5.2) to synthesize answer
        # Config: max_output_tokens=8192, temperature=0.1, verification_threshold=0.80
        synthesis_output = await self.agents["truth_synthesizer"].run(
            TruthSynthesisInput(
                query_text=query,
                query_type="truth_reconstitution",
                relevant_clauses=relevant_clauses,
                known_conflicts=[],
            )
        )

        return {
            "success": True,
            "answer": synthesis_output.answer,
            "sources": synthesis_output.sources,
            "confidence": synthesis_output.confidence,
            "caveats": synthesis_output.caveats,
        }
```

---

# FRONTEND IMPLEMENTATION

## React Application Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── DocumentUpload.tsx
│   │   ├── ContractStackList.tsx
│   │   ├── QueryInterface.tsx
│   │   ├── ResultsDisplay.tsx
│   │   ├── AgentReasoningViewer.tsx
│   │   ├── ConflictsList.tsx
│   │   ├── RippleEffectVisualizer.tsx
│   │   └── DependencyGraph.tsx
│   ├── pages/
│   │   ├── Home.tsx
│   │   ├── ContractStack.tsx
│   │   ├── Analysis.tsx
│   │   └── Settings.tsx
│   ├── hooks/
│   │   ├── useContractStack.ts
│   │   ├── useQuery.ts
│   │   └── useWebSocket.ts
│   ├── services/
│   │   ├── api.ts
│   │   └── websocket.ts
│   ├── types/
│   │   └── index.ts
│   ├── App.tsx
│   └── main.tsx
├── package.json
└── tsconfig.json
```

## Key Components

### QueryInterface.tsx
```typescript
import React, { useState } from 'react';
import { useQuery } from '../hooks/useQuery';

interface QueryInterfaceProps {
    contractStackId: string;
}

export const QueryInterface: React.FC<QueryInterfaceProps> = ({ contractStackId }) => {
    const [query, setQuery] = useState('');
    const { submitQuery, result, loading, error } = useQuery(contractStackId);
    
    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        await submitQuery(query);
    };
    
    return (
        <div className="query-interface">
            <form onSubmit={handleSubmit}>
                <textarea
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Ask about this contract... e.g., 'What are the current payment terms?'"
                    className="w-full p-4 border rounded-lg"
                    rows={4}
                />
                <button
                    type="submit"
                    disabled={loading}
                    className="mt-2 px-6 py-2 bg-blue-600 text-white rounded-lg"
                >
                    {loading ? 'Analyzing...' : 'Submit Query'}
                </button>
            </form>
            
            {result && (
                <div className="mt-6">
                    <ResultsDisplay result={result} />
                </div>
            )}
            
            {error && (
                <div className="mt-4 p-4 bg-red-50 text-red-700 rounded-lg">
                    {error}
                </div>
            )}
        </div>
    );
};
```

### ResultsDisplay.tsx
```typescript
import React from 'react';
import { QueryResult } from '../types';

interface ResultsDisplayProps {
    result: QueryResult;
}

export const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ result }) => {
    return (
        <div className="results-display bg-white rounded-lg shadow-lg p-6">
            <div className="answer mb-6">
                <h3 className="text-lg font-semibold mb-2">Answer</h3>
                <p className="text-gray-800">{result.response.answer}</p>
                <div className="mt-2 text-sm text-gray-500">
                    Confidence: {(result.response.confidence * 100).toFixed(0)}%
                </div>
            </div>
            
            <div className="sources mb-6">
                <h3 className="text-lg font-semibold mb-2">Sources</h3>
                {result.response.sources.map((source, idx) => (
                    <div key={idx} className="mb-3 p-3 bg-gray-50 rounded border-l-4 border-blue-500">
                        <div className="font-medium text-sm text-gray-700">
                            {source.document_name} - Section {source.section}
                        </div>
                        <div className="text-xs text-gray-500 mt-1">
                            Effective: {source.effective_date}
                        </div>
                        <div className="text-sm text-gray-800 mt-2 italic">
                            "{source.text.substring(0, 200)}..."
                        </div>
                        <a
                            href={`#doc-${source.document_id}`}
                            className="text-blue-600 text-sm mt-2 inline-block"
                        >
                            View full document →
                        </a>
                    </div>
                ))}
            </div>
            
            {result.agent_reasoning && (
                <AgentReasoningViewer reasoning={result.agent_reasoning} />
            )}
        </div>
    );
};
```

### AgentReasoningViewer.tsx
```typescript
import React, { useState } from 'react';
import { AgentReasoning } from '../types';

interface AgentReasoningViewerProps {
    reasoning: AgentReasoning[];
}

export const AgentReasoningViewer: React.FC<AgentReasoningViewerProps> = ({ reasoning }) => {
    const [expanded, setExpanded] = useState(false);
    
    return (
        <div className="agent-reasoning border-t pt-4">
            <button
                onClick={() => setExpanded(!expanded)}
                className="flex items-center text-sm text-gray-600 hover:text-gray-800"
            >
                <span className="mr-2">{expanded ? '▼' : '▶'}</span>
                Show Agent Reasoning ({reasoning.length} steps)
            </button>
            
            {expanded && (
                <div className="mt-4 space-y-2">
                    {reasoning.map((step, idx) => (
                        <div key={idx} className="flex items-start p-3 bg-gray-50 rounded">
                            <div className="flex-shrink-0 w-8 h-8 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">
                                {idx + 1}
                            </div>
                            <div className="ml-3 flex-grow">
                                <div className="text-sm font-medium text-gray-700">
                                    {step.agent}
                                </div>
                                <div className="text-sm text-gray-600 mt-1">
                                    {step.action}
                                </div>
                                <div className="text-xs text-gray-400 mt-1">
                                    {new Date(step.timestamp).toLocaleTimeString()}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};
```

---

# DEPLOYMENT & OPERATIONS

## Docker Compose Setup

**docker-compose.yml**
```yaml
version: '3.8'

services:
  api:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - EXTERNAL_DATABASE_URL=${EXTERNAL_DATABASE_URL}
      - AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - AZURE_OPENAI_DEPLOYMENT=${AZURE_OPENAI_DEPLOYMENT}
      - AZURE_OPENAI_API_VERSION=${AZURE_OPENAI_API_VERSION}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./uploads:/app/uploads
      # Vector embeddings stored in NeonDB via pgvector (two-tier: is_resolved=FALSE for Stage 1 checkpoint fallback, is_resolved=TRUE for Stage 4 query search) — no local volume needed
      # Background processing uses asyncio tasks (no Celery/Redis required)
      # Query caching uses InMemoryCache with TTL (no Redis required)

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - VITE_API_URL=http://localhost:8000
    depends_on:
      - api

  # Note: Redis and Celery are not currently used.
  # Background tasks use asyncio, caching uses InMemoryCache + InMemoryBlackboard.
  # Redis may be added as an optional future enhancement for distributed caching.
```

## Environment Variables

**.env**
```bash
# Database (NeonDB)
EXTERNAL_DATABASE_URL=postgresql://neondb_owner:...@ep-....neon.tech/neondb?sslmode=require

# Azure OpenAI (Primary LLM — GPT-5.2 for all agents)
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://...openai.azure.com
AZURE_OPENAI_DEPLOYMENT=gpt-5.2
AZURE_OPENAI_API_VERSION=2024-10-01-preview

# Gemini (Fallback LLM + Embeddings)
GEMINI_API_KEY=...

# Anthropic (optional, ClaudeProvider available but not used in current role map)
ANTHROPIC_API_KEY=sk-ant-...

# Vector embeddings use pgvector on NeonDB (two-tier: is_resolved=FALSE for Stage 1 per-document checkpoint fallback, is_resolved=TRUE for Stage 4 resolved-clause query search)
# Embedding model: Gemini gemini-embedding-001 (task_type=RETRIEVAL_DOCUMENT for indexing, RETRIEVAL_QUERY for search; uses GEMINI_API_KEY above)

# No Redis required — caching uses InMemoryCache, inter-agent communication uses InMemoryBlackboard
# No Celery required — background processing uses asyncio tasks

# Application
APP_ENV=development
LOG_LEVEL=INFO
UPLOAD_DIR=/app/uploads
MAX_UPLOAD_SIZE_MB=50
```

---

# TESTING STRATEGY

## Unit Tests

**backend/tests/test_agents.py**
```python
import pytest
from app.agents import DocumentParserAgent, AmendmentTrackerAgent

@pytest.mark.asyncio
async def test_document_parser():
    """Test document parser can extract sections (uses Azure OpenAI GPT-5.2)"""
    agent = DocumentParserAgent(mock_llm_config())
    
    result = await agent.process({
        'file_path': 'tests/fixtures/sample_cta.pdf',
        'document_type': 'cta'
    })
    
    assert result['success'] == True
    assert len(result['parsed_data']['sections']) > 0
    assert 'Payment Terms' in [s['title'] for s in result['parsed_data']['sections']]

@pytest.mark.asyncio
async def test_amendment_tracker():
    """Test amendment tracker identifies modifications"""
    agent = AmendmentTrackerAgent(mock_llm_config())
    
    amendment_text = """
    Section 7.2 of the Agreement is hereby amended to read as follows:
    "Payment terms shall be Net 45 days from invoice receipt."
    """
    
    result = await agent.process({
        'amendment_text': amendment_text,
        'amendment_number': 3
    })
    
    assert result['success'] == True
    assert '7.2' in result['tracking_data']['sections_modified']
    assert result['tracking_data']['modifications'][0]['type'] == 'complete_replacement'
```

## Integration Tests

**backend/tests/test_integration.py**
```python
@pytest.mark.asyncio
async def test_full_pipeline():
    """Test full contract processing pipeline"""
    orchestrator = AgentOrchestrator(
        postgres_pool=postgres_pool,
        vector_store=vector_store,
    )
    
    # Create test contract stack
    stack_id = await create_test_contract_stack()
    
    # Upload test documents
    await upload_test_documents(stack_id)
    
    # Process
    result = await orchestrator.process_contract_stack(stack_id)
    
    assert result['success'] == True
    assert result['clauses_processed'] > 0
    assert result['dependency_graph_built'] is True
    
    # Query
    query_result = await orchestrator.handle_query(
        "What are the current payment terms?",
        stack_id
    )
    
    assert query_result['success'] == True
    assert 'Net 45' in query_result['answer']
```

## Demo Test Data

**Use the HEARTBEAT-3 contract stack as canonical test data:**
- 6 documents (CTA + 5 amendments)
- Known conflicts (buried payment change, insurance gap)
- Known dependencies (payment → budget → holdback)

---

# IMPLEMENTATION PHASES

## Phase 1: MVP (Weeks 1-8)

### Week 1-2: Infrastructure
- [ ] Set up project structure
- [ ] Configure Docker Compose
- [ ] Create database schemas
- [ ] Set up NeonDB (PostgreSQL + pgvector)
- [ ] Implement basic API structure (FastAPI)
- [ ] Create React frontend skeleton

### Week 3-4: Document Ingestion
- [ ] Implement DocumentParserAgent
- [ ] Implement AmendmentTrackerAgent
- [ ] Implement TemporalSequencerAgent
- [ ] Create document upload UI
- [ ] Implement file storage (S3 or local)
- [ ] Add background processing (asyncio tasks)

### Week 5-6: Core Intelligence
- [ ] Implement OverrideResolutionAgent
- [ ] Implement ConflictDetectionAgent
- [ ] Implement DependencyMapperAgent
- [ ] Build clause dependency graph in PostgreSQL
- [ ] Create query interface UI
- [ ] Implement results display

### Week 7-8: Demo Polish
- [ ] Load HEARTBEAT-3 test data
- [ ] Implement agent reasoning visualization
- [ ] Add conflict detection UI
- [ ] Create demo script
- [ ] Performance optimization
- [ ] Bug fixes and testing

**Deliverable:** Working demo with truth reconstitution and conflict detection

## Phase 2: Advanced Analysis (Weeks 9-16)

### Week 9-10: Ripple Effects
- [ ] Implement RippleEffectAnalyzerAgent
- [ ] Add multi-hop reasoning
- [ ] Create visualization for cascade impacts
- [ ] Implement recommendation engine

### Week 11-12: Reusability Analysis
- [ ] Implement ReusabilityAnalyzerAgent
- [ ] Add cross-contract comparison
- [ ] Create gap analysis UI
- [ ] Build recommendation system

### Week 13-14: Portfolio Features
- [ ] Multi-contract processing
- [ ] Portfolio analytics dashboard
- [ ] Cross-portfolio search
- [ ] Trend detection

### Week 15-16: Integration & Polish
- [ ] API refinement
- [ ] Performance optimization
- [ ] Comprehensive testing
- [ ] Documentation

**Deliverable:** Full-featured pilot-ready platform

---

# SUCCESS CRITERIA

## Technical Metrics
- [ ] Document processing: <30 seconds per contract stack
- [ ] Query response: <15 seconds for complex multi-hop reasoning
- [ ] Accuracy: >90% precision on conflict detection (validated against manual review)
- [ ] Uptime: 99%+ during demo/pilot

## Demo Metrics
- [ ] "Wow moment" achieved in first 60 seconds
- [ ] All 6 demo scenarios execute successfully
- [ ] Agent reasoning is visible and understandable
- [ ] Zero errors during live demo

## User Experience
- [ ] Upload and process 6-document stack in <5 minutes
- [ ] Query interface responds instantly
- [ ] Results are clear and actionable
- [ ] Provenance is always visible

---

# NEXT STEPS FOR CLAUDE CODE

To implement this solution:

1. **Start with Phase 1, Weeks 1-2** - Infrastructure setup
2. **Use this specification as context** for all implementation
3. **Test incrementally** - Don't wait until the end
4. **Focus on HEARTBEAT-3 demo** - Use that as north star
5. **Prioritize working code over perfect code** - Iterate quickly

**Key files to generate first:**
1. `backend/app/main.py` - FastAPI application
2. `backend/app/models.py` - SQLAlchemy models
3. `backend/app/agents/base.py` - BaseAgent class
4. `backend/app/agents/document_parser.py` - First agent
5. `backend/app/database.py` - Database connections
6. `frontend/src/App.tsx` - React application
7. `docker-compose.yml` - Development environment

**Testing approach:**
- Use HEARTBEAT-3 PDFs as test data
- Validate each agent independently
- Build integration tests for full pipeline
- Manual testing of UI/UX

**This specification is complete and ready for implementation by Claude Code.**