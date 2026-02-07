"""Shared enums for ContractIQ agent system."""

from enum import StrEnum


class DocumentType(StrEnum):
    CTA = "cta"
    AMENDMENT = "amendment"
    EXHIBIT = "exhibit"


class ModificationType(StrEnum):
    COMPLETE_REPLACEMENT = "complete_replacement"
    SELECTIVE_OVERRIDE = "selective_override"
    ADDITION = "addition"
    DELETION = "deletion"
    EXHIBIT_REPLACEMENT = "exhibit_replacement"


class ConflictType(StrEnum):
    CONTRADICTION = "contradiction"
    AMBIGUITY = "ambiguity"
    GAP = "gap"
    INCONSISTENCY = "inconsistency"
    BURIED_CHANGE = "buried_change"
    STALE_REFERENCE = "stale_reference"
    TEMPORAL_MISMATCH = "temporal_mismatch"


class ConflictSeverity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


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
