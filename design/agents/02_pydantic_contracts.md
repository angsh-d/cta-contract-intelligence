# 02 — Pydantic I/O Contracts

> Every agent input and output defined as strict Pydantic v2 models.
> File locations: `backend/app/models/enums.py`, `backend/app/models/agent_schemas.py`, `backend/app/models/events.py`

---

## 1. Shared Enums

```python
# backend/app/models/enums.py

from enum import StrEnum

class DocumentType(StrEnum):
    CTA = "cta"
    AMENDMENT = "amendment"
    EXHIBIT = "exhibit"

class ModificationType(StrEnum):
    COMPLETE_REPLACEMENT = "complete_replacement"     # "Section X is hereby deleted in its entirety and replaced"
    SELECTIVE_OVERRIDE = "selective_override"          # "Section X is amended by deleting Y and substituting Z"
    ADDITION = "addition"                             # "A new Section X.Y is hereby added"
    DELETION = "deletion"                             # "Section X is hereby deleted"
    EXHIBIT_REPLACEMENT = "exhibit_replacement"       # "Exhibit B is replaced by Exhibit B-1"

class ConflictType(StrEnum):
    CONTRADICTION = "contradiction"       # Clauses directly contradict
    AMBIGUITY = "ambiguity"               # Unclear or inconsistent language
    GAP = "gap"                           # Missing coverage
    INCONSISTENCY = "inconsistency"       # Related clauses don't align
    BURIED_CHANGE = "buried_change"       # Important change hidden in unrelated amendment
    STALE_REFERENCE = "stale_reference"   # References outdated person/exhibit/section
    TEMPORAL_MISMATCH = "temporal_mismatch" # Date/duration inconsistencies

class ConflictSeverity(StrEnum):
    CRITICAL = "critical"   # Blocks execution, legal risk
    HIGH = "high"           # Significant operational impact
    MEDIUM = "medium"       # Requires clarification
    LOW = "low"             # Minor inconsistency

class QueryType(StrEnum):
    TRUTH_RECONSTITUTION = "truth_reconstitution"
    CONFLICT_DETECTION = "conflict_detection"
    RIPPLE_ANALYSIS = "ripple_analysis"
    GENERAL = "general"

class RelationshipType(StrEnum):
    DEPENDS_ON = "depends_on"
    REFERENCES = "references"
    CONFLICTS_WITH = "conflicts_with"
    SUPERSEDES = "supersedes"
    MODIFIES = "modifies"
    REPLACES = "replaces"
    AMENDS = "amends"
    CONTAINS = "contains"
    HAS_CLAUSE = "has_clause"
```

---

## 2. LLM Response Model (shared)

```python
# backend/app/models/agent_schemas.py

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Optional
from datetime import date, datetime
from uuid import UUID
from app.models.enums import *

class LLMResponse(BaseModel):
    """Standardized response from any LLM provider."""
    success: bool
    content: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    model: str = ""
    latency_ms: int = 0
    provider: str = ""
    error: Optional[str] = None
```

---

## 3. Tier 1 — Document Ingestion Models

### 3.1 DocumentParserAgent

```python
class DocumentMetadata(BaseModel):
    """Extracted document metadata."""
    document_type: DocumentType
    effective_date: Optional[date] = None
    execution_date: Optional[date] = None
    title: str = ""
    amendment_number: Optional[int] = None
    parties: list["Party"] = Field(default_factory=list)
    study_protocol: Optional[str] = None

class Party(BaseModel):
    """Contracting party."""
    name: str
    role: str                          # "sponsor", "institution", "investigator"
    address: Optional[str] = None

class ParsedSection(BaseModel):
    """A single section extracted from a document."""
    section_number: str                # "7.2", "12.1(a)"
    section_title: str                 # "Payment Terms"
    text: str
    page_numbers: list[int] = Field(default_factory=list)
    subsections: list["ParsedSection"] = Field(default_factory=list)

class ParsedTable(BaseModel):
    """A table extracted from a document."""
    table_id: str
    caption: Optional[str] = None
    headers: list[str]
    rows: list[list[str]]
    page_number: int
    source_section: Optional[str] = None   # section_number this table belongs to

class DocumentParseInput(BaseModel):
    """Input to DocumentParserAgent.process()."""
    model_config = ConfigDict(frozen=True)

    file_path: str
    document_type: DocumentType
    contract_stack_id: UUID

class DocumentParseOutput(BaseModel):
    """Output from DocumentParserAgent.process().

    Note: metadata may be None if the LLM failed to extract metadata from all
    chunks of a large document. The orchestrator should validate metadata presence
    before passing to downstream agents.
    """
    document_id: UUID
    metadata: Optional[DocumentMetadata] = None
    sections: list[ParsedSection]
    tables: list[ParsedTable]
    raw_text: str
    char_count: int
    page_count: int
    extraction_model: str              # which LLM was used
    extraction_latency_ms: int
    llm_reasoning: str = ""
    extraction_confidence: float = Field(ge=0.0, le=1.0, default=0.9)
```

### 3.2 AmendmentTrackerAgent

```python
class Modification(BaseModel):
    """A single modification made by an amendment."""
    section_number: str
    modification_type: ModificationType
    original_text: Optional[str] = None    # text before this amendment
    new_text: Optional[str] = None         # text after this amendment (None for deletions)
    change_description: str                # human-readable summary
    exhibit_reference: Optional[str] = None  # "Exhibit B-1" if applicable

    @model_validator(mode='after')
    def validate_modification_consistency(self) -> 'Modification':
        if self.modification_type == ModificationType.DELETION:
            if self.new_text is not None:
                raise ValueError("Deletions must have new_text=None")
            if not self.original_text:
                raise ValueError("Deletions must specify original_text")
        elif self.modification_type != ModificationType.ADDITION:
            if not self.original_text:
                raise ValueError(f"{self.modification_type} requires original_text")
        return self
```

### Semantic Validators

The `@model_validator` decorator enforces semantic constraints beyond basic type checking. Key examples:

```python
class Modification(BaseModel):
    # ... existing fields ...

    @model_validator(mode='after')
    def validate_modification_consistency(self) -> 'Modification':
        if self.modification_type == ModificationType.DELETION:
            if self.new_text is not None:
                raise ValueError("Deletions must have new_text=None")
            if not self.original_text:
                raise ValueError("Deletions must specify original_text")
        elif self.modification_type != ModificationType.ADDITION:
            if not self.original_text:
                raise ValueError(f"{self.modification_type} requires original_text")
        return self

class DetectedConflict(BaseModel):
    evidence: list[ConflictEvidence] = Field(min_length=2)  # conflicts require at least 2 evidence items
```

```python
class AmendmentTrackInput(BaseModel):
    """Input to AmendmentTrackerAgent.process()."""
    model_config = ConfigDict(frozen=True)

    amendment_document_id: UUID
    amendment_number: int
    amendment_text: str
    amendment_sections: list[ParsedSection]
    amendment_tables: list["ParsedTable"] = Field(default_factory=list)  # tables from amendment (Pain Point #2)
    original_sections: list[ParsedSection]   # from original CTA for comparison
    original_tables: list["ParsedTable"] = Field(default_factory=list)   # tables from original CTA
    prior_amendments: list["AmendmentTrackOutput"] = Field(default_factory=list)

class AmendmentTrackOutput(BaseModel):
    """Output from AmendmentTrackerAgent.process()."""
    amendment_document_id: UUID
    amendment_number: int
    effective_date: Optional[date] = None
    amendment_type: str                    # "protocol_change", "budget_revision", "pi_change", etc.
    rationale: str                         # extracted from WHEREAS clauses
    modifications: list[Modification]
    sections_modified: list[str]           # section numbers only, for quick lookup
    exhibits_affected: list[str]           # ["Exhibit B", "Exhibit B-1"]
    llm_reasoning: str = ""
    extraction_confidence: float = Field(ge=0.0, le=1.0, default=0.9)
```

### 3.3 TemporalSequencerAgent

```python
class TimelineEvent(BaseModel):
    """A single event on the contract timeline."""
    document_id: UUID
    event_date: date
    document_type: DocumentType
    label: str                             # "Original CTA", "Amendment 3 - COVID Protocol Changes"
    amendment_number: Optional[int] = None

class VersionTree(BaseModel):
    """Tree structure showing document supersession."""
    root_document_id: UUID
    amendments: list["VersionTreeNode"]

class VersionTreeNode(BaseModel):
    """Single node in version tree."""
    document_id: UUID
    amendment_number: int
    effective_date: Optional[date] = None  # may be inferred later by LLM
    supersedes_document_id: UUID           # which document this amends
    label: str

class TemporalSequenceInput(BaseModel):
    """Input to TemporalSequencerAgent.process()."""
    model_config = ConfigDict(frozen=True)

    contract_stack_id: UUID
    documents: list["DocumentSummary"]

class DocumentSummary(BaseModel):
    """Lightweight document info for sequencing."""
    document_id: UUID
    document_type: DocumentType
    effective_date: Optional[date] = None
    document_version: Optional[str] = None
    amendment_number: Optional[int] = None  # from DocumentMetadata, for correct non-sequential numbering
    filename: str

class TemporalSequenceOutput(BaseModel):
    """Output from TemporalSequencerAgent.process()."""
    contract_stack_id: UUID
    chronological_order: list[UUID]        # document IDs in date order
    version_tree: VersionTree
    timeline: list[TimelineEvent]
    dates_inferred: list[UUID]             # doc IDs where LLM inferred the date
    confidence: float = Field(ge=0.0, le=1.0, default=0.9)  # for verification gate
    llm_reasoning: str = ""
```

---

## 4. Tier 2 — Reasoning Models

### 4.1 OverrideResolutionAgent

```python
class SourceChainLink(BaseModel):
    """One step in the provenance chain showing how a clause evolved."""
    stage: str                             # "original", "amendment_1", "amendment_3"
    document_id: UUID
    document_label: str                    # "Original CTA (Jan 2022)"
    text: str                              # clause text at this stage
    change_description: Optional[str] = None
    modification_type: Optional[ModificationType] = None

class ClauseVersion(BaseModel):
    """The resolved current state of a single clause."""
    section_number: str
    section_title: str
    current_text: str                      # the text after all amendments applied
    source_chain: list[SourceChainLink]
    last_modified_by: UUID                 # document_id of most recent amendment
    last_modified_date: Optional[date] = None
    confidence: float = Field(ge=0.0, le=1.0)
    clause_category: str                   # "payment", "insurance", "indemnification", etc.

class OverrideResolutionInput(BaseModel):
    """Input to OverrideResolutionAgent.process()."""
    model_config = ConfigDict(frozen=True)

    contract_stack_id: UUID
    section_number: str
    original_clause: ParsedSection
    original_document_id: UUID
    original_document_label: str           # human-readable, e.g. "Original CTA (Jan 2022)"
    amendments: list["AmendmentForSection"]

class AmendmentForSection(BaseModel):
    """Amendment data relevant to a specific section."""
    amendment_document_id: UUID
    amendment_number: int
    effective_date: date
    modification: Modification             # the specific mod for this section

class OverrideResolutionOutput(BaseModel):
    """Output from OverrideResolutionAgent.process()."""
    clause_version: ClauseVersion
    llm_reasoning: str = ""
    confidence_factors: dict[str, float] = Field(default_factory=dict)  # e.g., {"amendment_clarity": 0.9, "section_match": 0.8}
```

### 4.2 ConflictDetectionAgent

```python
class ConflictEvidence(BaseModel):
    """Evidence supporting a detected conflict."""
    document_id: UUID
    document_label: str
    section_number: str
    relevant_text: str                     # excerpt demonstrating the conflict

class DetectedConflict(BaseModel):
    """A single conflict found in the contract stack."""
    conflict_id: str                       # deterministic ID for dedup
    conflict_type: ConflictType
    severity: ConflictSeverity
    description: str
    affected_sections: list[str]
    evidence: list[ConflictEvidence]       # at least 2 pieces of evidence
    recommendation: str
    pain_point_id: Optional[int] = None    # maps to HEARTBEAT-3 pain points 1-5

class CurrentClause(BaseModel):
    """A clause in its current resolved state, ready for conflict analysis."""
    section_number: str
    section_title: str
    current_text: str
    clause_category: str
    source_document_id: UUID
    source_document_label: str
    effective_date: Optional[date] = None  # None when original clause has no metadata date

class ConflictDetectionInput(BaseModel):
    """Input to ConflictDetectionAgent.process()."""
    model_config = ConfigDict(frozen=True)

    contract_stack_id: UUID
    current_clauses: list[CurrentClause]
    contract_stack_context: "ContractStackContext"
    dependency_graph: list["ClauseDependency"] = Field(default_factory=list)  # from DependencyMapperAgent

class ContractStackContext(BaseModel):
    """Contextual info about the contract stack for richer conflict detection."""
    study_name: str
    sponsor_name: str
    site_name: str
    therapeutic_area: str
    study_start_date: Optional[date] = None
    study_end_date: Optional[date] = None
    current_pi: Optional[str] = None

class ConflictSeveritySummary(BaseModel):
    """Typed severity counts for conflict detection output."""
    critical: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0

class ConflictDetectionOutput(BaseModel):
    """Output from ConflictDetectionAgent.process()."""
    contract_stack_id: UUID
    conflicts: list[DetectedConflict]
    summary: ConflictSeveritySummary
    analysis_model: str
    analysis_latency_ms: int
    llm_reasoning: str = ""
    confidence_factors: dict[str, float] = Field(default_factory=dict)  # e.g., {"amendment_clarity": 0.9, "section_match": 0.8}
    needs_clarification: list["ClarificationRequest"] = Field(default_factory=list)  # inter-agent communication
```

### 4.3 DependencyMapperAgent

```python
class ClauseDependency(BaseModel):
    """A single dependency between two clauses."""
    from_section: str
    to_section: str
    relationship_type: RelationshipType
    description: str
    detection_method: str = "llm"          # always "llm" (LLM-first design)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)

class DependencyMapInput(BaseModel):
    """Input to DependencyMapperAgent.process()."""
    model_config = ConfigDict(frozen=True)

    contract_stack_id: UUID
    current_clauses: list[CurrentClause]

class DependencyMapOutput(BaseModel):
    """Output from DependencyMapperAgent.process()."""
    contract_stack_id: UUID
    dependencies: list[ClauseDependency]
    total_nodes: int                       # clause count
    total_edges: int                       # dependency count
    db_synced: bool                        # True if written to clause_dependencies table
    llm_reasoning: str = ""
```

---

## 5. Tier 3 — Analysis Models

### 5.1 RippleEffectAnalyzerAgent

```python
class ProposedChange(BaseModel):
    """A proposed amendment to analyze for ripple effects."""
    section_number: str
    current_text: str
    proposed_text: str
    change_description: Optional[str] = None

class RippleImpact(BaseModel):
    """A single impact discovered during ripple analysis."""
    affected_section: str
    affected_section_title: str
    hop_distance: int                      # 1 = direct, 2 = indirect, etc.
    impact_type: str                       # "cost_increase", "indemnification_gap", "compliance_risk", etc.
    severity: ConflictSeverity
    description: str
    required_action: str
    estimated_cost_low: Optional[int] = None   # lower bound in USD (e.g., 3000)
    estimated_cost_high: Optional[int] = None  # upper bound in USD (e.g., 5000)
    estimated_cost_rationale: Optional[str] = None  # LLM explanation of cost estimate
    estimated_timeline: Optional[str] = None

class PrioritizedAction(BaseModel):
    """An action item from ripple analysis, prioritized."""
    priority: int                          # 1 = highest
    action: str
    reason: str
    estimated_cost: Optional[str] = None
    deadline: Optional[str] = None
    related_sections: list[str]

class RippleRecommendations(BaseModel):
    """Synthesized recommendations from all ripple impacts."""
    critical_actions: list[PrioritizedAction]
    recommended_actions: list[PrioritizedAction]
    optional_actions: list[PrioritizedAction]

class RippleEffectInput(BaseModel):
    """Input to RippleEffectAnalyzerAgent.process()."""
    model_config = ConfigDict(frozen=True)

    contract_stack_id: UUID
    proposed_change: ProposedChange

class RippleEffectOutput(BaseModel):
    """Output from RippleEffectAnalyzerAgent.process()."""
    contract_stack_id: UUID
    proposed_change: ProposedChange
    impacts_by_hop: dict[str, list[RippleImpact]]   # {"hop_1": [...], "hop_2": [...]}
    total_impacts: int
    cascade_depth: int                     # max hop reached
    estimated_total_cost: Optional[str] = None
    recommendations: RippleRecommendations
    traversal_direction: str               # "outbound", "inbound", "bidirectional"
    analysis_model: str
    analysis_latency_ms: int
    llm_reasoning: str = ""
```

---

## 6. Orchestrator & Query Models

### 6.1 QueryRouter

```python
class QueryRouteInput(BaseModel):
    """Input to QueryRouter."""
    model_config = ConfigDict(frozen=True)

    query_text: str
    contract_stack_id: UUID

class QueryRouteOutput(BaseModel):
    """Output from QueryRouter."""
    query_type: QueryType
    extracted_entities: list[str]          # section numbers, names, dates mentioned
    confidence: float = Field(ge=0.0, le=1.0)
    routing_latency_ms: int
    llm_reasoning: str = ""
    sub_queries: list["SubQuery"] = Field(default_factory=list)  # for compound queries
```

### 6.2 TruthSynthesizer

```python
class SourceCitation(BaseModel):
    """A citation to a source document/section."""
    document_id: UUID
    document_name: str
    section_number: str
    relevant_text: str                     # excerpt
    effective_date: Optional[date] = None  # LLM may omit or return non-ISO format

class TruthSynthesisInput(BaseModel):
    """Input to TruthSynthesizer."""
    model_config = ConfigDict(frozen=True)

    query_text: str
    query_type: QueryType
    contract_stack_id: UUID
    relevant_clauses: list[CurrentClause]
    conflicts: list[DetectedConflict]      # any known conflicts relevant to this query

class TruthSynthesisOutput(BaseModel):
    """Output from TruthSynthesizer."""
    answer: str
    sources: list[SourceCitation]
    confidence: float = Field(ge=0.0, le=1.0)
    caveats: list[str]                     # any ambiguities or uncertainties
    agent_reasoning: list["AgentReasoningStep"]
    llm_reasoning: str = ""

class AgentReasoningStep(BaseModel):
    """One step in the multi-agent reasoning chain."""
    agent: str
    action: str
    timestamp: datetime
    duration_ms: Optional[int] = None

class SubQuery(BaseModel):
    """A decomposed sub-query from a compound question."""
    query_type: QueryType
    query_text: str
    entities: list[str] = Field(default_factory=list)

class BlackboardEntry(BaseModel):
    """A single entry on the agent communication blackboard."""
    agent: str
    entry_type: str                    # "buried_change", "stale_reference", "clarification_request", etc.
    data: dict[str, Any]
    timestamp: datetime

class ClarificationRequest(BaseModel):
    """An agent's request for re-analysis by another agent."""
    target_agent: str                  # agent to re-run
    reason: str                        # why clarification is needed
    section_number: Optional[str] = None  # specific section to re-examine
    context: Optional[str] = None      # additional context for the re-run
```

---

## 7. Event Models (WebSocket / Progress)

```python
# backend/app/models/events.py

from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class AgentProgressEvent(BaseModel):
    """Progress update from a single agent."""
    job_id: str
    agent_name: str
    stage: str                             # "parsing", "extracting_tables", etc.
    percent_complete: int                  # 0-100 within this agent's scope
    message: str
    timestamp: datetime

class PipelineProgressEvent(BaseModel):
    """Overall pipeline progress event."""
    job_id: str
    pipeline_stage: str                    # "document_parsing", "amendment_tracking", etc.
    overall_percent: int                   # 0-100 for entire pipeline
    message: str
    current_agent: Optional[str] = None
    timestamp: datetime

class PipelineCompleteEvent(BaseModel):
    """Emitted when full pipeline finishes."""
    job_id: str
    success: bool
    total_documents: int
    total_clauses: int
    total_conflicts: int
    total_dependencies: int
    total_duration_ms: int
    timestamp: datetime

class PipelineErrorEvent(BaseModel):
    """Emitted when pipeline encounters a fatal error."""
    job_id: str
    error_type: str
    error_message: str
    agent_name: Optional[str] = None
    stage: Optional[str] = None
    timestamp: datetime
```

---

## 8. Model Validation Rules

All models use Pydantic v2 with these conventions:

| Rule | Implementation |
|------|----------------|
| **UUIDs** | `UUID` type, generated server-side via `uuid4()` |
| **Dates** | `date` for calendar dates, `datetime` for timestamps (always UTC) |
| **Enums** | `StrEnum` so JSON serialization is human-readable strings |
| **Optional fields** | `Optional[T] = None` — never use bare `Optional[T]` without default |
| **Lists** | `list[T] = Field(default_factory=list)` — never mutable defaults |
| **Confidence** | `float = Field(ge=0.0, le=1.0)` — bounded |
| **Nested models** | Forward references via `"ModelName"` strings, resolved by `model_rebuild()` |
| **Immutability** | Input models have `model_config = ConfigDict(frozen=True)` — shown explicitly on each `*Input` class above |
| **Serialization** | All models support `.model_dump(mode="json")` for API responses |

---

## 9. Reusability Models (Phase 2)

Defined here for API stability. Implementation deferred to Phase 2 (see doc 05).

```python
class ClauseVariation(BaseModel):
    """One version of a clause from a specific contract stack."""
    contract_stack_id: UUID
    contract_stack_name: str
    current_text: str
    effective_date: Optional[date] = None

class ClauseComparison(BaseModel):
    """Comparison of a clause across contract stacks."""
    section_number: str
    clause_category: str
    variations: list[ClauseVariation]
    recommended_standard: Optional[str] = None
    deviation_risk: str                    # "low", "medium", "high"

class ReusabilityInput(BaseModel):
    """Input to ReusabilityAnalyzerAgent — Phase 2."""
    model_config = ConfigDict(frozen=True)

    contract_stack_ids: list[UUID]
    focus_categories: list[str] = Field(default_factory=list)

class ReusabilityOutput(BaseModel):
    """Output from ReusabilityAnalyzerAgent — Phase 2."""
    comparisons: list[ClauseComparison]
    reusability_score: float = Field(ge=0.0, le=1.0)
    standard_deviations: list[str]
    recommendations: list[str]
    llm_reasoning: str = ""
```

---

## 10. Cross-Reference: Agent → I/O Models

| Agent | Input Model | Output Model |
|-------|-------------|--------------|
| DocumentParserAgent | `DocumentParseInput` | `DocumentParseOutput` |
| AmendmentTrackerAgent | `AmendmentTrackInput` | `AmendmentTrackOutput` |
| TemporalSequencerAgent | `TemporalSequenceInput` | `TemporalSequenceOutput` |
| OverrideResolutionAgent | `OverrideResolutionInput` | `OverrideResolutionOutput` |
| ConflictDetectionAgent | `ConflictDetectionInput` | `ConflictDetectionOutput` |
| DependencyMapperAgent | `DependencyMapInput` | `DependencyMapOutput` |
| RippleEffectAnalyzerAgent | `RippleEffectInput` | `RippleEffectOutput` |
| ReusabilityAnalyzerAgent | `ReusabilityInput` | `ReusabilityOutput` |
| QueryRouter | `QueryRouteInput` | `QueryRouteOutput` |
| TruthSynthesizer | `TruthSynthesisInput` | `TruthSynthesisOutput` |
