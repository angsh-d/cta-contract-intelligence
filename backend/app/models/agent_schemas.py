"""All Pydantic I/O models for ContractIQ agents."""

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Optional, Any
from datetime import date, datetime
from uuid import UUID

from app.models.enums import (
    DocumentType, ModificationType, ConflictType, ConflictSeverity,
    QueryType, RelationshipType,
)


# ── LLM Response (shared) ──────────────────────────────────────

class LLMResponse(BaseModel):
    """Standardized response from any LLM provider."""
    success: bool
    content: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    model: str = ""
    latency_ms: int = 0
    provider: str = ""
    error: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None


# ── Tier 1: Document Ingestion ──────────────────────────────────

class Party(BaseModel):
    """Contracting party."""
    name: str
    role: str
    address: Optional[str] = None


class DocumentMetadata(BaseModel):
    """Extracted document metadata."""
    document_type: DocumentType
    effective_date: Optional[date] = None
    execution_date: Optional[date] = None
    title: str = ""
    amendment_number: Optional[int] = None
    parties: list[Party] = Field(default_factory=list)
    study_protocol: Optional[str] = None


class ParsedSection(BaseModel):
    """A single section extracted from a document."""
    section_number: str
    section_title: str
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
    source_section: Optional[str] = None


class DocumentParseInput(BaseModel):
    """Input to DocumentParserAgent.process()."""
    model_config = ConfigDict(frozen=True)

    document_id: UUID
    file_path: str
    document_type: DocumentType
    contract_stack_id: UUID


class DocumentParseOutput(BaseModel):
    """Output from DocumentParserAgent.process()."""
    document_id: UUID
    metadata: Optional[DocumentMetadata] = None
    sections: list[ParsedSection]
    tables: list[ParsedTable]
    raw_text: str = ""
    char_count: int = 0
    page_count: int = 0
    extraction_model: str = ""
    extraction_latency_ms: int = 0
    llm_reasoning: str = ""
    extraction_confidence: float = Field(ge=0.0, le=1.0, default=0.9)


# ── Amendment Tracker ───────────────────────────────────────────

class Modification(BaseModel):
    """A single modification made by an amendment."""
    section_number: str
    modification_type: ModificationType
    original_text: Optional[str] = None
    new_text: Optional[str] = None
    change_description: str
    exhibit_reference: Optional[str] = None

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


class AmendmentTrackOutput(BaseModel):
    """Output from AmendmentTrackerAgent.process()."""
    amendment_document_id: UUID
    amendment_number: int
    effective_date: Optional[date] = None
    amendment_type: str = "unknown"
    rationale: str = ""
    modifications: list[Modification] = Field(default_factory=list)
    sections_modified: list[str] = Field(default_factory=list)
    exhibits_affected: list[str] = Field(default_factory=list)
    llm_reasoning: str = ""
    extraction_confidence: float = Field(ge=0.0, le=1.0, default=0.9)


class AmendmentTrackInput(BaseModel):
    """Input to AmendmentTrackerAgent.process()."""
    model_config = ConfigDict(frozen=True)

    amendment_document_id: UUID
    amendment_number: int
    amendment_text: str
    amendment_sections: list[ParsedSection]
    amendment_tables: list[ParsedTable] = Field(default_factory=list)
    original_sections: list[ParsedSection]
    original_tables: list[ParsedTable] = Field(default_factory=list)
    prior_amendments: list[AmendmentTrackOutput] = Field(default_factory=list)


# ── Temporal Sequencer ──────────────────────────────────────────

class DocumentSummary(BaseModel):
    """Lightweight document info for sequencing."""
    document_id: UUID
    document_type: DocumentType
    effective_date: Optional[date] = None
    document_version: Optional[str] = None
    amendment_number: Optional[int] = None
    filename: str


class TimelineEvent(BaseModel):
    """A single event on the contract timeline."""
    document_id: UUID
    event_date: date
    document_type: DocumentType
    label: str
    amendment_number: Optional[int] = None


class VersionTreeNode(BaseModel):
    """Single node in version tree."""
    document_id: UUID
    amendment_number: int
    effective_date: Optional[date] = None
    supersedes_document_id: UUID
    label: str


class VersionTree(BaseModel):
    """Tree structure showing document supersession."""
    root_document_id: UUID
    amendments: list[VersionTreeNode]


class TemporalSequenceInput(BaseModel):
    """Input to TemporalSequencerAgent.process()."""
    model_config = ConfigDict(frozen=True)

    contract_stack_id: UUID
    documents: list[DocumentSummary]


class TemporalSequenceOutput(BaseModel):
    """Output from TemporalSequencerAgent.process()."""
    contract_stack_id: UUID
    chronological_order: list[UUID]
    version_tree: VersionTree
    timeline: list[TimelineEvent]
    dates_inferred: list[UUID] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.9)
    llm_reasoning: str = ""


# ── Tier 2: Override Resolution ─────────────────────────────────

class SourceChainLink(BaseModel):
    """One step in the provenance chain showing how a clause evolved."""
    stage: str = ""
    document_id: str = ""  # LLM returns labels like "CTA_2021", not UUIDs
    document_label: str = ""
    text: str = ""
    change_description: Optional[str] = None
    modification_type: Optional[str] = None  # LLM may return non-enum values


class ClauseVersion(BaseModel):
    """The resolved current state of a single clause."""
    section_number: str
    section_title: str = ""
    current_text: str = ""
    source_chain: list[SourceChainLink] = Field(default_factory=list)
    last_modified_by: Optional[UUID] = None
    last_modified_date: Optional[date] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    clause_category: str = "general"


class AmendmentForSection(BaseModel):
    """Amendment data relevant to a specific section."""
    amendment_document_id: UUID
    amendment_number: int
    effective_date: Optional[date] = None
    modification: Modification


class OverrideResolutionInput(BaseModel):
    """Input to OverrideResolutionAgent.process()."""
    model_config = ConfigDict(frozen=True)

    contract_stack_id: UUID
    section_number: str
    original_clause: ParsedSection
    original_document_id: UUID
    original_document_label: str
    amendments: list[AmendmentForSection]


class OverrideResolutionOutput(BaseModel):
    """Output from OverrideResolutionAgent.process()."""
    clause_version: ClauseVersion
    llm_reasoning: str = ""
    confidence_factors: dict[str, float] = Field(default_factory=dict)


# ── Tier 2: Conflict Detection ──────────────────────────────────

class CurrentClause(BaseModel):
    """A clause in its current resolved state, ready for conflict analysis."""
    section_number: str
    section_title: str = ""
    current_text: str = ""
    clause_category: str = "general"
    source_document_id: Optional[UUID] = None
    source_document_label: str = ""
    effective_date: Optional[date] = None


class ContractStackContext(BaseModel):
    """Contextual info about the contract stack for richer conflict detection."""
    study_name: str
    sponsor_name: str
    site_name: str
    therapeutic_area: str
    study_start_date: Optional[date] = None
    study_end_date: Optional[date] = None
    current_pi: Optional[str] = None


class ConflictEvidence(BaseModel):
    """Evidence supporting a detected conflict."""
    document_id: str = ""  # LLM returns labels, not UUIDs
    document_label: str = ""
    section_number: str = ""
    relevant_text: str = ""


class DetectedConflict(BaseModel):
    """A single conflict found in the contract stack."""
    conflict_id: str = ""
    conflict_type: ConflictType = ConflictType.CONTRADICTION
    severity: ConflictSeverity = ConflictSeverity.MEDIUM
    description: str = ""
    affected_sections: list[str] = Field(default_factory=list)
    evidence: list[ConflictEvidence] = Field(default_factory=list)
    recommendation: str = ""
    pain_point_id: Optional[int] = None


class ClarificationRequest(BaseModel):
    """An agent's request for re-analysis by another agent."""
    target_agent: str
    reason: str
    section_number: Optional[str] = None
    context: Optional[str] = None


class ConflictSeveritySummary(BaseModel):
    """Typed severity counts for conflict detection output."""
    critical: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0


class ConflictDetectionInput(BaseModel):
    """Input to ConflictDetectionAgent.process()."""
    model_config = ConfigDict(frozen=True)

    contract_stack_id: UUID
    current_clauses: list[CurrentClause]
    contract_stack_context: ContractStackContext
    dependency_graph: list["ClauseDependency"] = Field(default_factory=list)


class ConflictDetectionOutput(BaseModel):
    """Output from ConflictDetectionAgent.process()."""
    contract_stack_id: UUID
    conflicts: list[DetectedConflict]
    summary: ConflictSeveritySummary
    analysis_model: str
    analysis_latency_ms: int
    llm_reasoning: str = ""
    confidence_factors: dict[str, float] = Field(default_factory=dict)
    needs_clarification: list[ClarificationRequest] = Field(default_factory=list)


# ── Tier 2: Dependency Mapper ───────────────────────────────────

class ClauseDependency(BaseModel):
    """A single dependency between two clauses."""
    from_section: str
    to_section: str
    relationship_type: str = "references"  # LLM may return non-enum values
    description: str = ""
    detection_method: str = "llm"
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
    total_nodes: int
    total_edges: int
    db_synced: bool
    llm_reasoning: str = ""


# ── Tier 3: Ripple Effect ──────────────────────────────────────

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
    hop_distance: int
    impact_type: str
    severity: ConflictSeverity
    description: str
    required_action: str
    cascade_path: Optional[str] = None
    estimated_cost_low: Optional[int] = None
    estimated_cost_high: Optional[int] = None
    estimated_cost_rationale: Optional[str] = None
    estimated_timeline: Optional[str] = None


class PrioritizedAction(BaseModel):
    """An action item from ripple analysis, prioritized."""
    priority: int
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
    impacts_by_hop: dict[str, list[RippleImpact]]
    total_impacts: int
    cascade_depth: int
    estimated_total_cost: Optional[str] = None
    recommendations: RippleRecommendations
    traversal_direction: str
    analysis_model: str
    analysis_latency_ms: int
    llm_reasoning: str = ""


# ── Query Pipeline ──────────────────────────────────────────────

class SubQuery(BaseModel):
    """A decomposed sub-query from a compound question."""
    query_type: QueryType
    query_text: str
    entities: list[str] = Field(default_factory=list)


class QueryRouteInput(BaseModel):
    """Input to QueryRouter."""
    model_config = ConfigDict(frozen=True)

    query_text: str
    contract_stack_id: UUID


class QueryRouteOutput(BaseModel):
    """Output from QueryRouter."""
    query_type: QueryType
    extracted_entities: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    routing_latency_ms: int
    llm_reasoning: str = ""
    sub_queries: list[SubQuery] = Field(default_factory=list)


class SourceCitation(BaseModel):
    """A citation to a source document/section."""
    document_id: Optional[UUID] = None
    document_name: str = ""
    section_number: str = ""
    relevant_text: str = ""
    effective_date: Optional[date] = None


class AgentReasoningStep(BaseModel):
    """One step in the multi-agent reasoning chain."""
    agent: str
    action: str
    timestamp: datetime
    duration_ms: Optional[int] = None


class TruthSynthesisInput(BaseModel):
    """Input to TruthSynthesizer."""
    model_config = ConfigDict(frozen=True)

    query_text: str
    query_type: QueryType
    contract_stack_id: UUID
    relevant_clauses: list[CurrentClause]
    conflicts: list[DetectedConflict] = Field(default_factory=list)


class TruthSynthesisOutput(BaseModel):
    """Output from TruthSynthesizer."""
    answer: str
    sources: list[SourceCitation]
    confidence: float = Field(ge=0.0, le=1.0)
    caveats: list[str] = Field(default_factory=list)
    agent_reasoning: list[AgentReasoningStep] = Field(default_factory=list)
    llm_reasoning: str = ""


# ── Blackboard ──────────────────────────────────────────────────

class BlackboardEntry(BaseModel):
    """A single entry on the agent communication blackboard."""
    agent: str
    entry_type: str
    data: dict[str, Any]
    timestamp: datetime


# ── Reusability (Phase 2) ──────────────────────────────────────

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
    deviation_risk: str


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


# Resolve forward references
ParsedSection.model_rebuild()
ConflictDetectionInput.model_rebuild()
