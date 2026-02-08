import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';

export function useHealth() {
  return useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: 30000 });
}

export function useStacks() {
  return useQuery({ queryKey: ['stacks'], queryFn: api.listStacks });
}

export function useStack(id: string) {
  return useQuery({ queryKey: ['stack', id], queryFn: () => api.getStack(id), enabled: !!id });
}

export function useDocuments(stackId: string) {
  return useQuery({ queryKey: ['documents', stackId], queryFn: () => api.listDocuments(stackId), enabled: !!stackId });
}

export function useTimeline(stackId: string) {
  return useQuery({ queryKey: ['timeline', stackId], queryFn: () => api.getTimeline(stackId), enabled: !!stackId });
}

export function useCreateStack() {
  const qc = useQueryClient();
  return useMutation({ mutationFn: api.createStack, onSuccess: () => qc.invalidateQueries({ queryKey: ['stacks'] }) });
}

export function useUploadDocument(stackId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ file, documentType, effectiveDate }: { file: File; documentType: string; effectiveDate?: string }) =>
      api.uploadDocument(stackId, file, documentType, effectiveDate),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['documents', stackId] });
      qc.invalidateQueries({ queryKey: ['stack', stackId] });
    },
  });
}

export function useProcessStack(stackId: string) {
  return useMutation({ mutationFn: () => api.processStack(stackId) });
}

export function useQueryStack(stackId: string) {
  return useMutation({ mutationFn: (query: string) => api.queryStack(stackId, query) });
}

export function useConflicts(stackId: string) {
  return useMutation({ mutationFn: (threshold?: string) => api.analyzeConflicts(stackId, threshold) });
}

export function useDocumentClauses(stackId: string, documentId: string | null) {
  return useQuery({
    queryKey: ['documentClauses', stackId, documentId],
    queryFn: () => api.getDocumentClauses(stackId, documentId!),
    enabled: !!stackId && !!documentId,
  });
}

export function useRippleEffects(stackId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (change: Record<string, unknown>) => api.analyzeRippleEffects(stackId, change),
    onSuccess: (data, variables) => {
      // Cache result in query cache so it persists across tab switches
      const cacheKey = JSON.stringify(variables);
      qc.setQueryData(['rippleResult', stackId, cacheKey], data);
    },
  });
}

export function useCachedRippleResult(stackId: string, change: Record<string, unknown> | null) {
  const cacheKey = change ? JSON.stringify(change) : '';
  return useQuery({
    queryKey: ['rippleResult', stackId, cacheKey],
    queryFn: () => null,
    enabled: false,  // Never auto-fetch — populated by mutation onSuccess
    staleTime: Infinity,
  });
}
