import {
  ContractStack,
  Document,
  DocumentClausesResponse,
  TimelineData,
  QueryResponse,
  ConflictAnalysis,
  ClauseHistory,
  JobStatus,
} from '../types/index';

const BASE = '';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export const api = {
  health: () => request<{ status: string; service: string; ai_available: boolean }>('/health'),

  listStacks: () => request<{ stacks: ContractStack[] }>('/api/v1/contract-stacks'),

  getStack: (id: string) => request<ContractStack>(`/api/v1/contract-stacks/${id}`),

  createStack: (data: { name: string; sponsor_name: string; site_name: string; study_protocol?: string; therapeutic_area?: string }) =>
    request<{ id: string; name: string; status: string; created_at: string }>('/api/v1/contract-stacks', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  listDocuments: (stackId: string) => request<{ contract_stack_id: string; documents: Document[] }>(`/api/v1/contract-stacks/${stackId}/documents`),

  uploadDocument: async (stackId: string, file: File, documentType: string, effectiveDate?: string) => {
    const form = new FormData();
    form.append('file', file);
    form.append('document_type', documentType);
    if (effectiveDate) form.append('effective_date', effectiveDate);
    const res = await fetch(`/api/v1/contract-stacks/${stackId}/documents`, { method: 'POST', body: form });
    if (!res.ok) { const err = await res.json().catch(() => ({ detail: res.statusText })); throw new Error(err.detail); }
    return res.json();
  },

  processStack: (stackId: string) => request<{ job_id: string; status: string }>(`/api/v1/contract-stacks/${stackId}/process`, { method: 'POST' }),

  getJobStatus: (jobId: string) => request<JobStatus>(`/api/v1/jobs/${jobId}/status`),

  getTimeline: (stackId: string) => request<TimelineData>(`/api/v1/contract-stacks/${stackId}/timeline`),

  queryStack: (stackId: string, query: string) =>
    request<QueryResponse>(`/api/v1/contract-stacks/${stackId}/query`, {
      method: 'POST',
      body: JSON.stringify({ query, include_reasoning: true }),
    }),

  analyzeConflicts: (stackId: string, severityThreshold = 'medium') =>
    request<ConflictAnalysis>(`/api/v1/contract-stacks/${stackId}/analyze/conflicts`, {
      method: 'POST',
      body: JSON.stringify({ severity_threshold: severityThreshold }),
    }),

  analyzeRippleEffects: (stackId: string, proposedChange: Record<string, unknown>) =>
    request<Record<string, unknown>>(`/api/v1/contract-stacks/${stackId}/analyze/ripple-effects`, {
      method: 'POST',
      body: JSON.stringify({ proposed_change: proposedChange }),
    }),

  getClauseHistory: (stackId: string, sectionNumber: string) =>
    request<ClauseHistory>(`/api/v1/contract-stacks/${stackId}/clauses/${sectionNumber}/history`),

  getDocumentClauses: (stackId: string, documentId: string) =>
    request<DocumentClausesResponse>(`/api/v1/contract-stacks/${stackId}/documents/${documentId}/clauses`),

  getDocumentPdfUrl: (stackId: string, documentId: string) =>
    `/api/v1/contract-stacks/${stackId}/documents/${documentId}/pdf`,
};
