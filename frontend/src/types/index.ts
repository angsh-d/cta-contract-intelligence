export interface ContractStack {
  id: string;
  name: string;
  sponsor_name: string;
  site_name: string;
  therapeutic_area: string | null;
  study_protocol: string | null;
  processing_status: string | null;
  created_at: string | null;
  counts?: {
    documents: number;
    clauses: number;
    conflicts: number;
  };
}

export interface Document {
  id: string;
  document_type: string;
  filename: string;
  effective_date: string | null;
  processed: boolean;
  document_version?: string | null;
}

export interface TimelineEntry {
  document_id: string;
  document_type: string;
  filename: string;
  effective_date: string | null;
  document_version: string | null;
}

export interface Supersession {
  predecessor: string;
  successor: string;
}

export interface TimelineData {
  contract_stack_id: string;
  timeline: TimelineEntry[];
  supersessions: Supersession[];
}

export interface QueryResponse {
  query_id: string;
  query: string;
  response: {
    answer: string;
    sources: Array<{
      document_id?: string;
      section_number?: string;
      text?: string;
      confidence?: number;
    }>;
    confidence: number;
    caveats: string[];
  };
  execution_time_ms: number;
}

export interface ConflictEvidence {
  document_id: string;
  document_label: string;
  section_number: string;
  relevant_text: string;
}

export interface Conflict {
  id: string;
  conflict_type: string;
  severity: string;
  description: string;
  affected_clauses: string[];
  evidence: ConflictEvidence[];
  recommendation: string;
  pain_point_id: number | null;
}

export interface ConflictAnalysis {
  conflicts: Conflict[];
  summary: Record<string, number>;
}

export interface ClauseHistory {
  section_number: string;
  section_title: string;
  current_text: string;
  clause_category: string;
  effective_date: string | null;
  source_chain: Array<{
    document_id: string;
    document_type: string;
    text: string;
    effective_date: string;
  }>;
  dependencies: Array<{
    related_section: string;
    relationship_type: string;
    description: string;
    confidence: number | null;
  }>;
}

export interface SourceChainLink {
  stage: string;
  document_id: string;
  document_label: string;
  text: string;
  change_description: string | null;
  modification_type: string | null;
}

export interface ClauseConflict {
  conflict_id: string;
  conflict_type: string;
  severity: string;
  description: string;
  recommendation: string;
  pain_point_id: number | null;
}

export interface DocumentClause {
  section_number: string;
  section_title: string;
  current_text: string;
  clause_category: string;
  is_current: boolean;
  effective_date: string | null;
  source_chain: SourceChainLink[];
  conflicts: ClauseConflict[];
}

export interface DocumentClausesResponse {
  document_id: string;
  filename: string;
  document_type: string;
  clauses: DocumentClause[];
}

export interface JobStatus {
  job_id: string;
  status: string;
  progress: number;
  message: string;
  created_at?: string;
  updated_at?: string;
}
