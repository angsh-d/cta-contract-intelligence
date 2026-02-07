import { useState, useRef, useCallback, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  ArrowLeft,
  Upload,
  Play,
  FileText,
  Clock,
  MessageSquare,
  AlertTriangle,
  Waves,
  CheckCircle2,
  X,
  File,
  Send,
  Loader2,
  ChevronRight,
  Info,
  Shield,
  Circle,
} from 'lucide-react'
import { useStack, useDocuments, useTimeline, useUploadDocument, useProcessStack, useQueryStack, useConflicts } from '../hooks/useApi'
import type { Conflict, QueryResponse, TimelineEntry } from '../types'

const TABS = [
  { id: 'overview', label: 'Overview', icon: FileText },
  { id: 'timeline', label: 'Timeline', icon: Clock },
  { id: 'query', label: 'Query', icon: MessageSquare },
  { id: 'conflicts', label: 'Conflicts', icon: AlertTriangle },
  { id: 'ripple', label: 'Ripple Effects', icon: Waves },
]

export default function StackDetail() {
  const { id = '', tab } = useParams()
  const navigate = useNavigate()
  const activeTab = tab || 'overview'
  const { data: stack, isLoading: stackLoading } = useStack(id)
  const { data: docsData } = useDocuments(id)
  const documents = docsData?.documents || []

  if (stackLoading) {
    return (
      <div className="max-w-6xl mx-auto px-6 lg:px-10 py-10">
        <div className="h-8 w-48 skeleton mb-4" />
        <div className="h-4 w-32 skeleton mb-8" />
        <div className="h-64 skeleton rounded-2xl" />
      </div>
    )
  }

  if (!stack) {
    return (
      <div className="max-w-6xl mx-auto px-6 lg:px-10 py-20 text-center">
        <Shield className="w-12 h-12 text-apple-light mx-auto mb-4" />
        <h2 className="text-[20px] font-semibold text-apple-black mb-1">Contract not found</h2>
        <p className="text-[14px] text-apple-gray">This contract stack may have been removed.</p>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto px-6 lg:px-10 py-8">
      <button onClick={() => navigate('/stacks')} className="flex items-center gap-1.5 text-[13px] font-medium text-apple-gray hover:text-apple-black transition-colors mb-6">
        <ArrowLeft className="w-4 h-4" />
        Back to Contracts
      </button>

      <div className="mb-8">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-[28px] font-semibold tracking-tight text-apple-black">{stack.name}</h1>
            <p className="text-[14px] text-apple-gray mt-1">
              {stack.sponsor_name} · {stack.site_name}
              {stack.therapeutic_area && ` · ${stack.therapeutic_area}`}
            </p>
          </div>
          <StatusBadge status={stack.processing_status} />
        </div>
      </div>

      <div className="flex gap-1 p-1 bg-apple-silver/40 rounded-xl mb-8 overflow-x-auto">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => navigate(`/stacks/${id}/${t.id}`)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-[13px] font-medium whitespace-nowrap transition-all duration-200 ${
              activeTab === t.id
                ? 'bg-white text-apple-black shadow-sm'
                : 'text-apple-gray hover:text-apple-dark'
            }`}
          >
            <t.icon className="w-4 h-4" />
            {t.label}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.2 }}
        >
          {activeTab === 'overview' && <OverviewTab stackId={id} stack={stack} documents={documents} />}
          {activeTab === 'timeline' && <TimelineTab stackId={id} />}
          {activeTab === 'query' && <QueryTab stackId={id} />}
          {activeTab === 'conflicts' && <ConflictsTab stackId={id} />}
          {activeTab === 'ripple' && <RippleTab stackId={id} />}
        </motion.div>
      </AnimatePresence>
    </div>
  )
}

function StatusBadge({ status }: { status: string | null }) {
  const s = status || 'pending'
  const map: Record<string, { bg: string; text: string; dot: string; label: string }> = {
    completed: { bg: 'bg-apple-black/[0.06]', text: 'text-apple-dark', dot: 'bg-apple-dark', label: 'Processed' },
    processing: { bg: 'bg-apple-silver', text: 'text-apple-dark', dot: 'bg-apple-dark2', label: 'Processing' },
    pending: { bg: 'bg-apple-bg', text: 'text-apple-gray', dot: 'bg-apple-light', label: 'Pending' },
    created: { bg: 'bg-apple-bg', text: 'text-apple-gray', dot: 'bg-apple-light', label: 'Created' },
  }
  const c = map[s] || map.pending
  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[12px] font-medium ${c.bg} ${c.text}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${c.dot}`} />
      {c.label}
    </span>
  )
}

function OverviewTab({ stackId, stack, documents }: { stackId: string; stack: any; documents: any[] }) {
  const [showUpload, setShowUpload] = useState(false)
  const uploadMutation = useUploadDocument(stackId)
  const processMutation = useProcessStack(stackId)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleUpload = async (files: FileList | null) => {
    if (!files) return
    for (const file of Array.from(files)) {
      await uploadMutation.mutateAsync({ file, documentType: file.name.toLowerCase().includes('amendment') ? 'amendment' : 'cta' })
    }
    setShowUpload(false)
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <MetricCard label="Documents" value={stack.counts?.documents ?? documents.length} icon={FileText} />
        <MetricCard label="Clauses Extracted" value={stack.counts?.clauses ?? 0} icon={Shield} />
        <MetricCard label="Conflicts Found" value={stack.counts?.conflicts ?? 0} icon={AlertTriangle} />
      </div>

      <div className="flex gap-3">
        <button
          onClick={() => setShowUpload(true)}
          className="flex items-center gap-2 px-4 py-2.5 bg-white border border-black/[0.04] text-apple-black text-[13px] font-medium rounded-full hover:bg-apple-bg transition-colors"
        >
          <Upload className="w-4 h-4" />
          Upload Documents
        </button>
        {documents.length > 0 && (
          <button
            onClick={() => processMutation.mutate()}
            disabled={processMutation.isPending}
            className="flex items-center gap-2 px-4 py-2.5 bg-apple-black text-white text-[13px] font-medium rounded-full hover:bg-apple-dark transition-colors disabled:opacity-40"
          >
            {processMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            {processMutation.isPending ? 'Processing...' : 'Process Stack'}
          </button>
        )}
      </div>

      {processMutation.isSuccess && (
        <div className="flex items-center gap-2 px-4 py-3 bg-apple-silver/50 rounded-xl text-[13px] text-apple-dark font-medium">
          <CheckCircle2 className="w-4 h-4" />
          Processing started. Check back shortly for results.
        </div>
      )}

      <div>
        <h3 className="text-[15px] font-semibold text-apple-black mb-3">Documents</h3>
        {documents.length === 0 ? (
          <div className="text-center py-12 bg-white rounded-2xl border border-black/[0.04]">
            <File className="w-10 h-10 text-apple-light mx-auto mb-3" />
            <p className="text-[14px] text-apple-gray">No documents uploaded yet</p>
          </div>
        ) : (
          <div className="bg-white rounded-2xl border border-black/[0.04] overflow-hidden">
            {documents.map((doc: any, i: number) => (
              <div key={doc.id} className={`flex items-center justify-between px-5 py-3.5 ${i < documents.length - 1 ? 'border-b border-black/[0.04]' : ''}`}>
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-9 h-9 rounded-lg bg-apple-bg flex items-center justify-center flex-shrink-0">
                    <FileText className="w-4 h-4 text-apple-dark" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-[13px] font-medium text-apple-black truncate">{doc.filename}</p>
                    <p className="text-[11px] text-apple-gray capitalize">{doc.document_type.replace('_', ' ')}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {doc.effective_date && <span className="text-[11px] text-apple-gray">{doc.effective_date}</span>}
                  {doc.processed ? (
                    <CheckCircle2 className="w-4 h-4 text-apple-dark" />
                  ) : (
                    <Circle className="w-4 h-4 text-apple-light" />
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <AnimatePresence>
        {showUpload && (
          <>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 bg-black/30 backdrop-blur-sm z-50" onClick={() => setShowUpload(false)} />
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="fixed inset-x-4 top-[15%] sm:inset-auto sm:left-1/2 sm:top-1/2 sm:-translate-x-1/2 sm:-translate-y-1/2 w-auto sm:w-[440px] bg-white/95 backdrop-blur-xl rounded-2xl shadow-2xl z-50 border border-black/[0.04]"
            >
              <div className="flex items-center justify-between px-6 py-4 border-b border-black/[0.04]">
                <h2 className="text-[17px] font-semibold text-apple-black">Upload Documents</h2>
                <button onClick={() => setShowUpload(false)} className="p-1.5 rounded-lg hover:bg-apple-silver/50 transition-colors">
                  <X className="w-5 h-5 text-apple-gray" />
                </button>
              </div>
              <div className="p-6">
                <div
                  className={`border-2 border-dashed rounded-2xl p-10 text-center transition-colors cursor-pointer ${
                    dragOver ? 'border-apple-dark bg-apple-bg' : 'border-apple-silver hover:border-apple-light'
                  }`}
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={(e) => { e.preventDefault(); setDragOver(false); handleUpload(e.dataTransfer.files) }}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload className="w-8 h-8 text-apple-gray mx-auto mb-3" />
                  <p className="text-[14px] font-medium text-apple-black mb-1">Drop files here or click to browse</p>
                  <p className="text-[12px] text-apple-gray">Supports PDF, DOCX files up to 50MB</p>
                  <input ref={fileInputRef} type="file" multiple accept=".pdf,.docx,.doc" className="hidden" onChange={(e) => handleUpload(e.target.files)} />
                </div>
                {uploadMutation.isPending && (
                  <div className="flex items-center justify-center gap-2 mt-4 text-[13px] text-apple-dark">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Uploading...
                  </div>
                )}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  )
}

function MetricCard({ label, value, icon: Icon }: { label: string; value: number; icon: any }) {
  return (
    <div className="bg-white rounded-2xl border border-black/[0.04] p-5">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[12px] font-medium text-apple-gray uppercase tracking-wider">{label}</span>
        <Icon className="w-4 h-4 text-apple-gray" />
      </div>
      <p className="text-[24px] font-semibold tracking-tight text-apple-black">{value}</p>
    </div>
  )
}

function TimelineTab({ stackId }: { stackId: string }) {
  const { data, isLoading } = useTimeline(stackId)
  const timeline = data?.timeline || []

  if (isLoading) return <div className="h-64 skeleton rounded-2xl" />

  if (timeline.length === 0) {
    return (
      <div className="text-center py-16">
        <Clock className="w-10 h-10 text-apple-light mx-auto mb-3" />
        <p className="text-[15px] font-medium text-apple-black mb-1">No timeline data</p>
        <p className="text-[13px] text-apple-gray">Process your documents to generate the timeline</p>
      </div>
    )
  }

  return (
    <div className="relative">
      <div className="absolute left-6 top-0 bottom-0 w-px bg-apple-silver" />
      <div className="space-y-0">
        {timeline.map((entry: TimelineEntry, i: number) => (
          <motion.div
            key={entry.document_id}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.08 }}
            className="relative flex gap-5 pl-0 py-4"
          >
            <div className="relative z-10 flex-shrink-0">
              <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                entry.document_type === 'cta' ? 'bg-apple-black' : 'bg-white border-2 border-apple-silver'
              }`}>
                <FileText className={`w-5 h-5 ${entry.document_type === 'cta' ? 'text-white' : 'text-apple-dark'}`} />
              </div>
            </div>
            <div className="flex-1 bg-white rounded-2xl border border-black/[0.04] p-4">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-[14px] font-medium text-apple-black">{entry.filename}</p>
                  <p className="text-[12px] text-apple-gray mt-0.5 capitalize">{entry.document_type.replace('_', ' ')}</p>
                </div>
                {entry.effective_date && (
                  <span className="text-[12px] text-apple-gray bg-apple-bg px-2.5 py-1 rounded-lg font-medium">
                    {entry.effective_date}
                  </span>
                )}
              </div>
              {entry.document_version && (
                <span className="inline-block mt-2 text-[11px] text-apple-dark bg-apple-silver/60 px-2 py-0.5 rounded-md font-medium">
                  {entry.document_version}
                </span>
              )}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  )
}

function QueryTab({ stackId }: { stackId: string }) {
  const [query, setQuery] = useState('')
  const queryMutation = useQueryStack(stackId)
  const [history, setHistory] = useState<Array<{ query: string; response: QueryResponse }>>([])
  const chatEndRef = useRef<HTMLDivElement>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return
    const q = query.trim()
    setQuery('')
    try {
      const result = await queryMutation.mutateAsync(q)
      setHistory((prev) => [...prev, { query: q, response: result }])
    } catch {}
  }

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [history])

  return (
    <div className="flex flex-col h-[calc(100vh-320px)] min-h-[400px]">
      <div className="flex-1 overflow-y-auto space-y-6 pb-4">
        {history.length === 0 && !queryMutation.isPending && (
          <div className="text-center py-16">
            <div className="w-16 h-16 rounded-2xl bg-apple-bg flex items-center justify-center mx-auto mb-4">
              <MessageSquare className="w-8 h-8 text-apple-gray" />
            </div>
            <h3 className="text-[17px] font-semibold text-apple-black mb-1">Ask anything about your contracts</h3>
            <p className="text-[14px] text-apple-gray max-w-md mx-auto">
              Query payment terms, clause history, obligations, and more. AI will trace through all amendments to find the current truth.
            </p>
            <div className="flex flex-wrap justify-center gap-2 mt-6">
              {[
                'What are the current payment terms?',
                'Show the history of Section 7.2',
                'What insurance obligations exist?',
              ].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => setQuery(suggestion)}
                  className="px-3.5 py-2 bg-white border border-black/[0.04] rounded-xl text-[12px] text-apple-dark hover:bg-apple-bg transition-colors"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {history.map((item, i) => (
          <div key={i} className="space-y-4">
            <div className="flex justify-end">
              <div className="max-w-[80%] bg-apple-dark text-white px-4 py-3 rounded-2xl rounded-br-md">
                <p className="text-[14px]">{item.query}</p>
              </div>
            </div>
            <div className="flex justify-start">
              <div className="max-w-[85%] bg-apple-bg border border-black/[0.04] px-5 py-4 rounded-2xl rounded-bl-md">
                <p className="text-[14px] text-apple-black leading-relaxed whitespace-pre-wrap">{item.response.response.answer}</p>
                {item.response.response.sources.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-black/[0.04]">
                    <p className="text-[11px] font-medium text-apple-gray uppercase tracking-wider mb-2">Sources</p>
                    <div className="space-y-1">
                      {item.response.response.sources.map((src, j) => (
                        <div key={j} className="flex items-center gap-2 text-[12px] text-apple-dark">
                          <FileText className="w-3 h-3" />
                          <span>{src.section_number || src.document_id || `Source ${j + 1}`}</span>
                          {src.confidence != null && <span className="text-apple-gray">({Math.round(src.confidence * 100)}%)</span>}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                <div className="flex items-center gap-3 mt-3 text-[11px] text-apple-gray">
                  <span>Confidence: {Math.round(item.response.response.confidence * 100)}%</span>
                  <span>{item.response.execution_time_ms}ms</span>
                </div>
              </div>
            </div>
          </div>
        ))}

        {queryMutation.isPending && (
          <div className="flex justify-start">
            <div className="bg-apple-bg border border-black/[0.04] px-5 py-4 rounded-2xl">
              <div className="flex items-center gap-2 text-[13px] text-apple-gray">
                <Loader2 className="w-4 h-4 animate-spin text-apple-dark" />
                Analyzing contracts...
              </div>
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="flex gap-3 pt-4 border-t border-black/[0.04]">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask about your contract terms..."
          className="flex-1 px-4 py-3 bg-white rounded-xl border border-black/[0.04] text-[14px] text-apple-black placeholder:text-apple-light focus:outline-none focus:ring-2 focus:ring-apple-dark/20 focus:border-apple-dark/30 transition-all"
        />
        <button
          type="submit"
          disabled={!query.trim() || queryMutation.isPending}
          className="px-4 py-3 bg-apple-black text-white rounded-xl hover:bg-apple-dark transition-colors disabled:opacity-30"
        >
          <Send className="w-4 h-4" />
        </button>
      </form>
    </div>
  )
}

function ConflictsTab({ stackId }: { stackId: string }) {
  const conflictsMutation = useConflicts(stackId)
  const [analyzed, setAnalyzed] = useState(false)

  const handleAnalyze = async () => {
    await conflictsMutation.mutateAsync('medium')
    setAnalyzed(true)
  }

  const data = conflictsMutation.data
  const conflicts = data?.conflicts || []

  const severityConfig: Record<string, { border: string; bg: string; weight: string }> = {
    critical: { border: 'border-apple-black/30', bg: 'bg-apple-black/[0.03]', weight: 'border-l-4 border-l-apple-black' },
    high: { border: 'border-apple-dark/20', bg: 'bg-apple-dark/[0.02]', weight: 'border-l-4 border-l-apple-dark2' },
    medium: { border: 'border-apple-gray/20', bg: 'bg-apple-gray/[0.02]', weight: 'border-l-4 border-l-apple-gray' },
    low: { border: 'border-apple-light/40', bg: 'bg-apple-light/[0.02]', weight: 'border-l-4 border-l-apple-light' },
  }

  if (!analyzed && !conflictsMutation.isPending) {
    return (
      <div className="text-center py-16">
        <div className="w-16 h-16 rounded-2xl bg-apple-bg flex items-center justify-center mx-auto mb-4">
          <AlertTriangle className="w-8 h-8 text-apple-gray" />
        </div>
        <h3 className="text-[17px] font-semibold text-apple-black mb-1">Conflict Analysis</h3>
        <p className="text-[14px] text-apple-gray mb-6 max-w-md mx-auto">
          Run AI-powered analysis to detect hidden conflicts, contradictions, and risks across all amendments.
        </p>
        <button
          onClick={handleAnalyze}
          className="inline-flex items-center gap-2 px-5 py-2.5 bg-apple-black text-white text-[13px] font-medium rounded-full hover:bg-apple-dark transition-colors"
        >
          <AlertTriangle className="w-4 h-4" />
          Analyze Conflicts
        </button>
      </div>
    )
  }

  if (conflictsMutation.isPending) {
    return (
      <div className="text-center py-16">
        <Loader2 className="w-8 h-8 text-apple-dark animate-spin mx-auto mb-4" />
        <p className="text-[15px] font-medium text-apple-black">Analyzing conflicts...</p>
        <p className="text-[13px] text-apple-gray mt-1">This may take a moment</p>
      </div>
    )
  }

  return (
    <div>
      {data?.summary && Object.keys(data.summary).length > 0 && (
        <div className="flex gap-3 mb-6 flex-wrap">
          {Object.entries(data.summary).map(([severity, count]) => {
            const dotShade: Record<string, string> = {
              critical: 'bg-apple-black',
              high: 'bg-apple-dark2',
              medium: 'bg-apple-gray',
              low: 'bg-apple-light',
            }
            return (
              <div key={severity} className="flex items-center gap-2 px-3 py-1.5 rounded-full text-[12px] font-medium bg-apple-bg border border-black/[0.04]">
                <span className={`w-1.5 h-1.5 rounded-full ${dotShade[severity] || 'bg-apple-light'}`} />
                <span className="capitalize">{severity}: {count}</span>
              </div>
            )
          })}
        </div>
      )}

      {conflicts.length === 0 && (
        <div className="text-center py-12">
          <CheckCircle2 className="w-10 h-10 text-apple-dark mx-auto mb-3" />
          <p className="text-[15px] font-medium text-apple-black">No conflicts detected</p>
          <p className="text-[13px] text-apple-gray mt-1">Your contract stack appears consistent</p>
        </div>
      )}

      <div className="space-y-3">
        {conflicts.map((conflict: Conflict) => {
          const sc = severityConfig[conflict.severity] || severityConfig.low
          return (
            <div key={conflict.id} className={`bg-white rounded-2xl border border-black/[0.04] p-5 ${sc.weight}`}>
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-apple-dark" />
                  <span className="text-[13px] font-semibold text-apple-black capitalize">{conflict.conflict_type.replace('_', ' ')}</span>
                </div>
                <span className="px-2.5 py-0.5 rounded-full text-[11px] font-medium uppercase tracking-wider bg-apple-bg text-apple-gray">
                  {conflict.severity}
                </span>
              </div>
              <p className="text-[14px] text-apple-dark leading-relaxed mb-3">{conflict.description}</p>
              {conflict.recommendation && (
                <div className="flex gap-2 bg-apple-bg/60 rounded-xl p-3">
                  <Info className="w-4 h-4 text-apple-gray flex-shrink-0 mt-0.5" />
                  <p className="text-[13px] text-apple-dark">{conflict.recommendation}</p>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function RippleTab({ stackId }: { stackId: string }) {
  const [change, setChange] = useState({ clause_section: '', change_type: 'modify', description: '' })
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`/api/v1/contract-stacks/${stackId}/analyze/ripple-effects`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ proposed_change: change }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Analysis failed' }))
        throw new Error(err.detail)
      }
      setResult(await res.json())
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="bg-white rounded-2xl border border-black/[0.04] p-6 mb-6">
        <h3 className="text-[15px] font-semibold text-apple-black mb-4">Propose a Change</h3>
        <form onSubmit={handleAnalyze} className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-[12px] font-medium text-apple-gray mb-1.5">Section</label>
              <input
                type="text"
                value={change.clause_section}
                onChange={(e) => setChange({ ...change, clause_section: e.target.value })}
                placeholder="e.g. 7.2"
                className="w-full px-3.5 py-2.5 bg-apple-bg rounded-xl border border-black/[0.04] text-[14px] text-apple-black placeholder:text-apple-light focus:outline-none focus:ring-2 focus:ring-apple-dark/20 transition-all"
              />
            </div>
            <div>
              <label className="block text-[12px] font-medium text-apple-gray mb-1.5">Change Type</label>
              <select
                value={change.change_type}
                onChange={(e) => setChange({ ...change, change_type: e.target.value })}
                className="w-full px-3.5 py-2.5 bg-apple-bg rounded-xl border border-black/[0.04] text-[14px] text-apple-black focus:outline-none focus:ring-2 focus:ring-apple-dark/20 transition-all appearance-none"
              >
                <option value="modify">Modify</option>
                <option value="add">Add</option>
                <option value="remove">Remove</option>
                <option value="extend">Extend</option>
              </select>
            </div>
          </div>
          <div>
            <label className="block text-[12px] font-medium text-apple-gray mb-1.5">Description</label>
            <textarea
              value={change.description}
              onChange={(e) => setChange({ ...change, description: e.target.value })}
              placeholder="Describe the proposed change..."
              rows={3}
              className="w-full px-3.5 py-2.5 bg-apple-bg rounded-xl border border-black/[0.04] text-[14px] text-apple-black placeholder:text-apple-light focus:outline-none focus:ring-2 focus:ring-apple-dark/20 transition-all resize-none"
            />
          </div>
          <button
            type="submit"
            disabled={loading || !change.description}
            className="flex items-center gap-2 px-5 py-2.5 bg-apple-black text-white text-[13px] font-medium rounded-full hover:bg-apple-dark transition-colors disabled:opacity-40"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Waves className="w-4 h-4" />}
            {loading ? 'Analyzing...' : 'Analyze Ripple Effects'}
          </button>
        </form>
      </div>

      {error && (
        <div className="flex items-center gap-2 px-4 py-3 bg-apple-silver/50 rounded-xl text-[13px] text-apple-dark font-medium mb-6">
          <AlertTriangle className="w-4 h-4" />
          {error}
        </div>
      )}

      {result && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="bg-white rounded-2xl border border-black/[0.04] p-6">
          <h3 className="text-[15px] font-semibold text-apple-black mb-4">Analysis Results</h3>
          <pre className="text-[13px] text-apple-dark bg-apple-bg rounded-xl p-4 overflow-x-auto whitespace-pre-wrap font-mono leading-relaxed">
            {JSON.stringify(result, null, 2)}
          </pre>
        </motion.div>
      )}

      {!result && !loading && !error && (
        <div className="text-center py-12">
          <Waves className="w-10 h-10 text-apple-light mx-auto mb-3" />
          <p className="text-[15px] font-medium text-apple-black mb-1">Ripple Effect Analysis</p>
          <p className="text-[13px] text-apple-gray max-w-md mx-auto">
            Propose a contract change above to see how it ripples through all related clauses, obligations, and dependencies.
          </p>
        </div>
      )}
    </div>
  )
}
