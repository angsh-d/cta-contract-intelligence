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

export function useRippleEffects(stackId: string) {
  return useMutation({ mutationFn: (change: Record<string, unknown>) => api.analyzeRippleEffects(stackId, change) });
}
