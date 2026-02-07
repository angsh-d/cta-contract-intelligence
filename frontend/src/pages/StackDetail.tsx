import { useState, useRef, useEffect } from 'react'
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
  Info,
  Shield,
  Circle,
  Zap,
  Search,
  BookOpen,
  Layers,
  GitBranch,
  Hash,
} from 'lucide-react'
import {
  useStack,
  useDocuments,
  useTimeline,
  useUploadDocument,
  useProcessStack,
  useQueryStack,
  useConflicts,
  useRippleEffects,
} from '../hooks/useApi'
import type { Conflict, QueryResponse, TimelineEntry } from '../types'

const TABS = [
  { id: 'overview', label: 'Overview', icon: Layers },
  { id: 'timeline', label: 'Timeline', icon: GitBranch },
  { id: 'query', label: 'Query', icon: MessageSquare },
  { id: 'conflicts', label: 'Conflicts', icon: Shield },
  { id: 'ripple', label: 'Ripple Effects', icon: Waves },
]

const fadeUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -10 },
}

const stagger = {
  animate: { transition: { staggerChildren: 0.06 } },
}

export default function StackDetail() {
  const { id = '', tab } = useParams()
  const navigate = useNavigate()
  const activeTab = tab || 'overview'
  const { data: stack, isLoading: stackLoading } = useStack(id)
  const { data: docsData } = useDocuments(id)
  const documents = docsData?.documents || []

  if (stackLoading) {
    return (
      <div className="max-w-7xl mx-auto px-6 lg:px-16 py-16">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
          <div className="h-6 w-32 skeleton" />
          <div className="h-12 w-96 skeleton" />
          <div className="h-5 w-64 skeleton" />
          <div className="h-px bg-apple-silver/60 my-8" />
          <div className="grid grid-cols-3 gap-6">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-40 skeleton rounded-3xl" />
            ))}
          </div>
        </motion.div>
      </div>
    )
  }

  if (!stack) {
    return (
      <motion.div {...fadeUp} className="max-w-7xl mx-auto px-6 lg:px-16 py-32 text-center">
        <div className="w-20 h-20 rounded-full bg-apple-bg flex items-center justify-center mx-auto mb-6">
          <Shield className="w-10 h-10 text-apple-light" />
        </div>
        <h2 className="text-[28px] font-semibold text-apple-black tracking-tight mb-2">Contract not found</h2>
        <p className="text-[17px] text-apple-gray leading-relaxed">This contract stack may have been removed or is no longer available.</p>
        <button
          onClick={() => navigate('/stacks')}
          className="mt-8 px-6 py-3 bg-apple-black text-white text-[15px] font-medium rounded-full hover:bg-apple-dark transition-all duration-300"
        >
          Back to Contracts
        </button>
      </motion.div>
    )
  }

  return (
    <div className="min-h-screen bg-apple-white">
      <div className="max-w-7xl mx-auto px-6 lg:px-16 py-10">
        <motion.button
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3 }}
          onClick={() => navigate('/stacks')}
          className="group flex items-center gap-2 text-[15px] font-medium text-apple-gray hover:text-apple-black transition-colors duration-300 mb-10"
        >
          <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform duration-300" />
          Contracts
        </motion.button>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] }}
          className="mb-12"
        >
          <div className="flex items-start justify-between gap-6">
            <div className="space-y-3">
              <h1 className="text-[40px] font-bold tracking-tight text-apple-pure leading-[1.1]">
                {stack.name}
              </h1>
              <p className="text-[17px] text-apple-gray2 leading-relaxed">
                {stack.sponsor_name}
                <span className="mx-2 text-apple-light">·</span>
                {stack.site_name}
                {stack.therapeutic_area && (
                  <>
                    <span className="mx-2 text-apple-light">·</span>
                    {stack.therapeutic_area}
                  </>
                )}
              </p>
              {stack.study_protocol && (
                <p className="text-[14px] text-apple-gray font-mono tracking-wide">{stack.study_protocol}</p>
              )}
            </div>
            <StatusBadge status={stack.processing_status} />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.15 }}
          className="relative flex gap-1 p-1.5 bg-apple-bg/80 backdrop-blur-xl rounded-2xl mb-12 overflow-x-auto border border-black/[0.03]"
        >
          {TABS.map((t) => {
            const isActive = activeTab === t.id
            return (
              <button
                key={t.id}
                onClick={() => navigate(`/stacks/${id}/${t.id}`)}
                className={`relative flex items-center gap-2.5 px-5 py-3 rounded-xl text-[14px] font-medium whitespace-nowrap transition-all duration-300 ${
                  isActive
                    ? 'text-apple-black'
                    : 'text-apple-gray hover:text-apple-dark2'
                }`}
              >
                {isActive && (
                  <motion.div
                    layoutId="activeTab"
                    className="absolute inset-0 bg-white rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.08),0_1px_2px_rgba(0,0,0,0.04)]"
                    transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                  />
                )}
                <span className="relative z-10 flex items-center gap-2.5">
                  <t.icon className="w-4 h-4" />
                  {t.label}
                </span>
              </button>
            )
          })}
        </motion.div>

        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.35, ease: [0.25, 0.46, 0.45, 0.94] }}
          >
            {activeTab === 'overview' && <OverviewTab stackId={id} stack={stack} documents={documents} />}
            {activeTab === 'timeline' && <TimelineTab stackId={id} />}
            {activeTab === 'query' && <QueryTab stackId={id} />}
            {activeTab === 'conflicts' && <ConflictsTab stackId={id} />}
            {activeTab === 'ripple' && <RippleTab stackId={id} />}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  )
}

function StatusBadge({ status }: { status: string | null }) {
  const s = status || 'pending'
  const map: Record<string, { bg: string; text: string; dot: string; label: string }> = {
    completed: { bg: 'bg-apple-black/[0.06]', text: 'text-apple-dark', dot: 'bg-apple-dark', label: 'Processed' },
    processing: { bg: 'bg-apple-silver/80', text: 'text-apple-dark', dot: 'bg-apple-dark2', label: 'Processing' },
    pending: { bg: 'bg-apple-bg', text: 'text-apple-gray', dot: 'bg-apple-light', label: 'Pending' },
    created: { bg: 'bg-apple-bg', text: 'text-apple-gray', dot: 'bg-apple-light', label: 'Created' },
  }
  const c = map[s] || map.pending
  return (
    <motion.span
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-[13px] font-medium ${c.bg} ${c.text}`}
    >
      <span className={`w-2 h-2 rounded-full ${c.dot} ${s === 'processing' ? 'animate-pulse' : ''}`} />
      {c.label}
    </motion.span>
  )
}

function MetricCard({ label, value, icon: Icon, delay = 0 }: { label: string; value: number; icon: any; delay?: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay, ease: [0.25, 0.46, 0.45, 0.94] }}
      className="group relative bg-white/80 backdrop-blur-xl rounded-3xl border border-black/[0.04] p-8 hover:shadow-[0_8px_30px_rgba(0,0,0,0.06)] transition-all duration-500 overflow-hidden"
    >
      <div className="absolute top-0 right-0 w-32 h-32 bg-apple-bg/40 rounded-full -translate-y-1/2 translate-x-1/2 group-hover:scale-150 transition-transform duration-700" />
      <div className="relative">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-2xl bg-apple-bg flex items-center justify-center">
            <Icon className="w-5 h-5 text-apple-dark2" />
          </div>
          <span className="text-[13px] font-medium text-apple-gray uppercase tracking-[0.08em]">{label}</span>
        </div>
        <p className="text-[48px] font-bold tracking-tight text-apple-pure leading-none">{value}</p>
      </div>
    </motion.div>
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
    <div className="space-y-10">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
        <MetricCard label="Documents" value={stack.counts?.documents ?? documents.length} icon={FileText} delay={0} />
        <MetricCard label="Clauses Extracted" value={stack.counts?.clauses ?? 0} icon={BookOpen} delay={0.08} />
        <MetricCard label="Conflicts Found" value={stack.counts?.conflicts ?? 0} icon={AlertTriangle} delay={0.16} />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.2 }}
        className="flex gap-4"
      >
        <button
          onClick={() => setShowUpload(true)}
          className="group flex items-center gap-2.5 px-6 py-3 bg-white border border-black/[0.06] text-apple-black text-[14px] font-medium rounded-full hover:bg-apple-bg hover:border-black/[0.1] transition-all duration-300 shadow-[0_1px_2px_rgba(0,0,0,0.04)]"
        >
          <Upload className="w-4 h-4 group-hover:-translate-y-0.5 transition-transform duration-300" />
          Upload Documents
        </button>
        {documents.length > 0 && (
          <button
            onClick={() => processMutation.mutate()}
            disabled={processMutation.isPending}
            className="group flex items-center gap-2.5 px-6 py-3 bg-apple-pure text-white text-[14px] font-medium rounded-full hover:bg-apple-dark transition-all duration-300 disabled:opacity-40 shadow-[0_2px_8px_rgba(0,0,0,0.15)]"
          >
            {processMutation.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4 group-hover:scale-110 transition-transform duration-300" />
            )}
            {processMutation.isPending ? 'Processing...' : 'Process Stack'}
          </button>
        )}
      </motion.div>

      <AnimatePresence>
        {processMutation.isSuccess && (
          <motion.div
            initial={{ opacity: 0, y: -8, height: 0 }}
            animate={{ opacity: 1, y: 0, height: 'auto' }}
            exit={{ opacity: 0, y: -8, height: 0 }}
            className="flex items-center gap-3 px-6 py-4 bg-apple-bg/80 backdrop-blur-sm rounded-2xl border border-black/[0.04]"
          >
            <CheckCircle2 className="w-5 h-5 text-apple-dark" />
            <p className="text-[14px] text-apple-dark font-medium">Processing initiated successfully. Results will appear shortly.</p>
          </motion.div>
        )}
      </AnimatePresence>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.25 }}
      >
        <h3 className="text-[20px] font-semibold text-apple-black tracking-tight mb-5">Documents</h3>
        {documents.length === 0 ? (
          <div className="text-center py-20 bg-apple-offwhite rounded-3xl border border-black/[0.03]">
            <div className="w-16 h-16 rounded-full bg-apple-bg flex items-center justify-center mx-auto mb-4">
              <File className="w-8 h-8 text-apple-light" />
            </div>
            <p className="text-[17px] font-medium text-apple-black mb-1">No documents yet</p>
            <p className="text-[14px] text-apple-gray">Upload your clinical trial agreements to get started</p>
          </div>
        ) : (
          <motion.div variants={stagger} initial="initial" animate="animate" className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {documents.map((doc: any, i: number) => (
              <motion.div
                key={doc.id}
                variants={fadeUp}
                transition={{ duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
                className="group relative bg-white rounded-2xl border border-black/[0.04] p-5 hover:shadow-[0_8px_24px_rgba(0,0,0,0.06)] hover:border-black/[0.08] transition-all duration-400 cursor-default"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="w-11 h-11 rounded-xl bg-apple-bg flex items-center justify-center">
                    <FileText className="w-5 h-5 text-apple-dark2" />
                  </div>
                  {doc.processed ? (
                    <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-apple-black/[0.05]">
                      <CheckCircle2 className="w-3.5 h-3.5 text-apple-dark" />
                      <span className="text-[11px] font-medium text-apple-dark">Processed</span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-apple-bg">
                      <Circle className="w-3.5 h-3.5 text-apple-light" />
                      <span className="text-[11px] font-medium text-apple-gray">Pending</span>
                    </div>
                  )}
                </div>
                <p className="text-[14px] font-semibold text-apple-black truncate mb-1.5">{doc.filename}</p>
                <div className="flex items-center gap-2">
                  <span className="px-2.5 py-0.5 rounded-lg bg-apple-bg text-[11px] font-medium text-apple-dark2 capitalize">
                    {doc.document_type.replace('_', ' ')}
                  </span>
                  {doc.effective_date && (
                    <span className="text-[11px] text-apple-gray font-medium">{doc.effective_date}</span>
                  )}
                </div>
                {doc.document_version && (
                  <span className="inline-block mt-2.5 text-[11px] text-apple-gray font-mono bg-apple-silver/40 px-2 py-0.5 rounded-md">
                    v{doc.document_version}
                  </span>
                )}
              </motion.div>
            ))}
          </motion.div>
        )}
      </motion.div>

      <AnimatePresence>
        {showUpload && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.25 }}
              className="fixed inset-0 bg-black/40 backdrop-blur-md z-50"
              onClick={() => setShowUpload(false)}
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.92, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 10 }}
              transition={{ duration: 0.35, ease: [0.25, 0.46, 0.45, 0.94] }}
              className="fixed inset-x-4 top-[12%] sm:inset-auto sm:left-1/2 sm:top-1/2 sm:-translate-x-1/2 sm:-translate-y-1/2 w-auto sm:w-[480px] bg-white/95 backdrop-blur-2xl rounded-3xl shadow-[0_24px_80px_rgba(0,0,0,0.15)] z-50 border border-black/[0.06]"
            >
              <div className="flex items-center justify-between px-7 py-5 border-b border-black/[0.04]">
                <h2 className="text-[19px] font-semibold text-apple-black tracking-tight">Upload Documents</h2>
                <button onClick={() => setShowUpload(false)} className="p-2 rounded-xl hover:bg-apple-bg transition-colors duration-200">
                  <X className="w-5 h-5 text-apple-gray" />
                </button>
              </div>
              <div className="p-7">
                <div
                  className={`border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 cursor-pointer ${
                    dragOver ? 'border-apple-dark bg-apple-bg/80 scale-[1.01]' : 'border-apple-silver hover:border-apple-light hover:bg-apple-offwhite'
                  }`}
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={(e) => { e.preventDefault(); setDragOver(false); handleUpload(e.dataTransfer.files) }}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <div className="w-14 h-14 rounded-2xl bg-apple-bg flex items-center justify-center mx-auto mb-4">
                    <Upload className="w-7 h-7 text-apple-gray" />
                  </div>
                  <p className="text-[15px] font-semibold text-apple-black mb-1">Drop files here or click to browse</p>
                  <p className="text-[13px] text-apple-gray">Supports PDF, DOCX files up to 50MB</p>
                  <input ref={fileInputRef} type="file" multiple accept=".pdf,.docx,.doc" className="hidden" onChange={(e) => handleUpload(e.target.files)} />
                </div>
                {uploadMutation.isPending && (
                  <div className="flex items-center justify-center gap-2.5 mt-5 text-[14px] text-apple-dark font-medium">
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

function TimelineTab({ stackId }: { stackId: string }) {
  const { data, isLoading } = useTimeline(stackId)
  const timeline = data?.timeline || []

  if (isLoading) {
    return (
      <div className="space-y-6">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-24 skeleton rounded-2xl" />
        ))}
      </div>
    )
  }

  if (timeline.length === 0) {
    return (
      <motion.div {...fadeUp} className="text-center py-24">
        <div className="w-20 h-20 rounded-full bg-apple-bg flex items-center justify-center mx-auto mb-6">
          <Clock className="w-10 h-10 text-apple-light" />
        </div>
        <h3 className="text-[22px] font-semibold text-apple-black tracking-tight mb-2">No timeline data</h3>
        <p className="text-[15px] text-apple-gray max-w-sm mx-auto leading-relaxed">
          Process your documents to visualize the complete evolution of your contract stack.
        </p>
      </motion.div>
    )
  }

  return (
    <div className="relative max-w-3xl mx-auto py-4">
      <div className="absolute left-8 top-0 bottom-0 w-[2px] bg-gradient-to-b from-apple-dark via-apple-silver to-apple-bg" />

      <div className="space-y-2">
        {timeline.map((entry: TimelineEntry, i: number) => {
          const isRoot = entry.document_type === 'cta'
          return (
            <motion.div
              key={entry.document_id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1, duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] }}
              className="relative flex gap-6 pl-0 py-3"
            >
              <div className="relative z-10 flex-shrink-0">
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: i * 0.1 + 0.15, type: 'spring', stiffness: 300, damping: 20 }}
                  className={`w-16 h-16 rounded-2xl flex items-center justify-center shadow-[0_2px_8px_rgba(0,0,0,0.08)] ${
                    isRoot
                      ? 'bg-apple-pure shadow-[0_4px_16px_rgba(0,0,0,0.2)]'
                      : 'bg-white border-2 border-apple-silver'
                  }`}
                >
                  <FileText className={`w-6 h-6 ${isRoot ? 'text-white' : 'text-apple-dark2'}`} />
                </motion.div>
              </div>

              <div className={`flex-1 rounded-2xl p-5 transition-all duration-300 ${
                isRoot
                  ? 'bg-apple-pure text-white shadow-[0_4px_20px_rgba(0,0,0,0.15)]'
                  : 'bg-white border border-black/[0.04] hover:shadow-[0_4px_16px_rgba(0,0,0,0.06)]'
              }`}>
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <p className={`text-[16px] font-semibold truncate ${isRoot ? 'text-white' : 'text-apple-black'}`}>
                      {entry.filename}
                    </p>
                    <p className={`text-[13px] mt-1 capitalize ${isRoot ? 'text-white/70' : 'text-apple-gray'}`}>
                      {entry.document_type.replace('_', ' ')}
                    </p>
                  </div>
                  {entry.effective_date && (
                    <span className={`shrink-0 text-[13px] font-medium px-3 py-1.5 rounded-xl ${
                      isRoot ? 'bg-white/15 text-white' : 'bg-apple-bg text-apple-dark2'
                    }`}>
                      {entry.effective_date}
                    </span>
                  )}
                </div>
                {entry.document_version && (
                  <span className={`inline-block mt-3 text-[12px] font-mono px-2.5 py-1 rounded-lg ${
                    isRoot ? 'bg-white/10 text-white/80' : 'bg-apple-silver/50 text-apple-dark2'
                  }`}>
                    v{entry.document_version}
                  </span>
                )}
              </div>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}

function QueryTab({ stackId }: { stackId: string }) {
  const [query, setQuery] = useState('')
  const queryMutation = useQueryStack(stackId)
  const [history, setHistory] = useState<Array<{ query: string; response: QueryResponse }>>([])
  const chatEndRef = useRef<HTMLDivElement>(null)

  const suggestions = [
    'What are the current payment terms?',
    'Show the history of Section 7.2',
    'What insurance obligations exist?',
    'Who is the current PI?',
    'What are the holdback provisions?',
  ]

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

  const handleSuggestion = (s: string) => {
    setQuery(s)
  }

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [history, queryMutation.isPending])

  return (
    <div className="flex flex-col h-[calc(100vh-320px)] min-h-[500px]">
      <div className="flex-1 overflow-y-auto pb-6 space-y-5">
        {history.length === 0 && !queryMutation.isPending && (
          <motion.div {...fadeUp} transition={{ duration: 0.5 }} className="text-center py-16">
            <div className="w-20 h-20 rounded-[28px] bg-apple-bg flex items-center justify-center mx-auto mb-6">
              <MessageSquare className="w-10 h-10 text-apple-gray" />
            </div>
            <h3 className="text-[24px] font-bold text-apple-pure tracking-tight mb-2">
              Contract Intelligence
            </h3>
            <p className="text-[16px] text-apple-gray max-w-lg mx-auto leading-relaxed mb-8">
              Ask anything about your contracts. AI agents trace through every amendment to surface the current truth.
            </p>
            <motion.div
              variants={stagger}
              initial="initial"
              animate="animate"
              className="flex flex-wrap justify-center gap-2.5 max-w-2xl mx-auto"
            >
              {suggestions.map((s) => (
                <motion.button
                  key={s}
                  variants={fadeUp}
                  onClick={() => handleSuggestion(s)}
                  className="px-4 py-2.5 bg-white border border-black/[0.05] rounded-full text-[13px] text-apple-dark2 font-medium hover:bg-apple-bg hover:border-black/[0.08] transition-all duration-300 shadow-[0_1px_2px_rgba(0,0,0,0.03)]"
                >
                  {s}
                </motion.button>
              ))}
            </motion.div>
          </motion.div>
        )}

        {history.map((item, i) => (
          <div key={i} className="space-y-4">
            <motion.div
              initial={{ opacity: 0, y: 10, x: 20 }}
              animate={{ opacity: 1, y: 0, x: 0 }}
              transition={{ duration: 0.35 }}
              className="flex justify-end"
            >
              <div className="max-w-[75%] bg-apple-dark text-white px-5 py-3.5 rounded-[20px] rounded-br-lg shadow-[0_2px_8px_rgba(0,0,0,0.12)]">
                <p className="text-[15px] leading-relaxed">{item.query}</p>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 10, x: -20 }}
              animate={{ opacity: 1, y: 0, x: 0 }}
              transition={{ duration: 0.4, delay: 0.1 }}
              className="flex justify-start"
            >
              <div className="max-w-[85%] bg-apple-bg border border-black/[0.03] px-6 py-5 rounded-[20px] rounded-bl-lg">
                <p className="text-[15px] text-apple-black leading-[1.7] whitespace-pre-wrap">{item.response.response.answer}</p>

                {item.response.response.sources.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-black/[0.05]">
                    <p className="text-[11px] font-semibold text-apple-gray uppercase tracking-[0.1em] mb-2.5">Sources</p>
                    <div className="flex flex-wrap gap-2">
                      {item.response.response.sources.map((src, j) => (
                        <div key={j} className="flex items-center gap-1.5 px-2.5 py-1 bg-white rounded-lg border border-black/[0.04] text-[12px] text-apple-dark2">
                          <FileText className="w-3 h-3 text-apple-gray" />
                          <span className="font-medium">{src.section_number || src.document_id || `Source ${j + 1}`}</span>
                          {src.confidence != null && (
                            <span className="text-apple-gray ml-1">{Math.round(src.confidence * 100)}%</span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="flex items-center gap-4 mt-4 text-[12px] text-apple-gray">
                  <span className="flex items-center gap-1.5">
                    <div className="w-1.5 h-1.5 rounded-full bg-apple-dark" />
                    {Math.round(item.response.response.confidence * 100)}% confidence
                  </span>
                  <span>{item.response.execution_time_ms}ms</span>
                </div>

                {item.response.response.caveats && item.response.response.caveats.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-black/[0.04]">
                    {item.response.response.caveats.map((caveat, k) => (
                      <p key={k} className="text-[12px] text-apple-gray italic">{caveat}</p>
                    ))}
                  </div>
                )}
              </div>
            </motion.div>
          </div>
        ))}

        {queryMutation.isPending && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex justify-start"
          >
            <div className="bg-apple-bg border border-black/[0.03] px-6 py-5 rounded-[20px] rounded-bl-lg">
              <div className="flex items-center gap-1.5">
                {[0, 1, 2].map((dot) => (
                  <motion.div
                    key={dot}
                    className="w-2.5 h-2.5 rounded-full bg-apple-gray"
                    animate={{ opacity: [0.3, 1, 0.3], scale: [0.85, 1.1, 0.85] }}
                    transition={{ duration: 1.2, repeat: Infinity, delay: dot * 0.2 }}
                  />
                ))}
              </div>
            </div>
          </motion.div>
        )}
        <div ref={chatEndRef} />
      </div>

      <div className="pt-4 border-t border-black/[0.04]">
        <form onSubmit={handleSubmit} className="flex gap-3">
          <div className="flex-1 relative">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask about your contracts..."
              className="w-full px-5 py-3.5 bg-apple-offwhite rounded-2xl border border-black/[0.04] text-[15px] text-apple-black placeholder:text-apple-light focus:outline-none focus:border-apple-dark/20 focus:bg-white focus:shadow-[0_0_0_3px_rgba(0,0,0,0.04)] transition-all duration-300"
            />
          </div>
          <motion.button
            type="submit"
            disabled={!query.trim() || queryMutation.isPending}
            whileTap={{ scale: 0.95 }}
            className="px-5 py-3.5 bg-apple-pure text-white rounded-2xl hover:bg-apple-dark transition-all duration-300 disabled:opacity-20 shadow-[0_2px_8px_rgba(0,0,0,0.12)]"
          >
            <Send className="w-5 h-5" />
          </motion.button>
        </form>
      </div>
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

  const severityConfig: Record<string, { weight: string; dot: string }> = {
    critical: { weight: 'border-l-[5px] border-l-apple-pure', dot: 'bg-apple-pure' },
    high: { weight: 'border-l-4 border-l-apple-dark', dot: 'bg-apple-dark' },
    medium: { weight: 'border-l-[3px] border-l-apple-gray', dot: 'bg-apple-gray' },
    low: { weight: 'border-l-2 border-l-apple-light', dot: 'bg-apple-light' },
  }

  if (!analyzed && !conflictsMutation.isPending) {
    return (
      <motion.div {...fadeUp} transition={{ duration: 0.5 }} className="text-center py-24">
        <div className="w-20 h-20 rounded-full bg-apple-bg flex items-center justify-center mx-auto mb-6">
          <Shield className="w-10 h-10 text-apple-gray" />
        </div>
        <h3 className="text-[24px] font-bold text-apple-pure tracking-tight mb-2">Conflict Detection</h3>
        <p className="text-[16px] text-apple-gray max-w-md mx-auto leading-relaxed mb-8">
          AI agents will analyze every clause across all amendments to detect hidden conflicts, contradictions, and compliance risks.
        </p>
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={handleAnalyze}
          className="inline-flex items-center gap-2.5 px-7 py-3.5 bg-apple-pure text-white text-[15px] font-semibold rounded-full hover:bg-apple-dark transition-all duration-300 shadow-[0_4px_16px_rgba(0,0,0,0.2)]"
        >
          <Zap className="w-5 h-5" />
          Detect Hidden Conflicts
        </motion.button>
      </motion.div>
    )
  }

  if (conflictsMutation.isPending) {
    return (
      <motion.div {...fadeUp} className="text-center py-24">
        <motion.div
          animate={{ scale: [1, 1.15, 1], opacity: [0.6, 1, 0.6] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="w-20 h-20 rounded-full bg-apple-bg flex items-center justify-center mx-auto mb-6"
        >
          <Shield className="w-10 h-10 text-apple-dark" />
        </motion.div>
        <p className="text-[20px] font-semibold text-apple-black tracking-tight mb-2">AI agents analyzing contracts...</p>
        <p className="text-[15px] text-apple-gray">Scanning clauses for contradictions and risks</p>
      </motion.div>
    )
  }

  return (
    <motion.div {...fadeUp} transition={{ duration: 0.4 }}>
      {data?.summary && Object.keys(data.summary).length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex gap-3 mb-8 flex-wrap"
        >
          {Object.entries(data.summary).map(([severity, count]) => {
            const sc = severityConfig[severity] || severityConfig.low
            return (
              <div key={severity} className="flex items-center gap-2.5 px-4 py-2 rounded-full text-[13px] font-semibold bg-white border border-black/[0.06] shadow-[0_1px_3px_rgba(0,0,0,0.04)]">
                <span className={`w-2.5 h-2.5 rounded-full ${sc.dot}`} />
                <span className="capitalize text-apple-dark">{severity}</span>
                <span className="text-apple-gray">{count}</span>
              </div>
            )
          })}
        </motion.div>
      )}

      {conflicts.length === 0 && (
        <div className="text-center py-16">
          <div className="w-16 h-16 rounded-full bg-apple-bg flex items-center justify-center mx-auto mb-4">
            <CheckCircle2 className="w-8 h-8 text-apple-dark" />
          </div>
          <p className="text-[20px] font-semibold text-apple-black tracking-tight mb-1">No conflicts detected</p>
          <p className="text-[15px] text-apple-gray">Your contract stack appears consistent</p>
        </div>
      )}

      <motion.div variants={stagger} initial="initial" animate="animate" className="space-y-4">
        {conflicts.map((conflict: Conflict) => {
          const sc = severityConfig[conflict.severity] || severityConfig.low
          return (
            <motion.div
              key={conflict.id}
              variants={fadeUp}
              transition={{ duration: 0.4 }}
              className={`bg-white rounded-2xl border border-black/[0.04] p-6 ${sc.weight} hover:shadow-[0_4px_16px_rgba(0,0,0,0.06)] transition-shadow duration-300`}
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <AlertTriangle className="w-5 h-5 text-apple-dark2" />
                  <span className="text-[15px] font-semibold text-apple-black capitalize">
                    {conflict.conflict_type.replace('_', ' ')}
                  </span>
                </div>
                <span className="px-3 py-1 rounded-full text-[11px] font-bold uppercase tracking-[0.08em] bg-apple-bg text-apple-gray2">
                  {conflict.severity}
                </span>
              </div>

              <p className="text-[15px] text-apple-dark leading-[1.7] mb-4">{conflict.description}</p>

              {conflict.affected_clauses && conflict.affected_clauses.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mb-4">
                  {conflict.affected_clauses.map((clause, j) => (
                    <span key={j} className="px-2.5 py-1 bg-apple-silver/50 rounded-lg text-[12px] font-mono text-apple-dark2">
                      {clause}
                    </span>
                  ))}
                </div>
              )}

              {conflict.recommendation && (
                <div className="flex gap-3 bg-apple-offwhite rounded-xl p-4 border border-black/[0.02]">
                  <Info className="w-4 h-4 text-apple-gray flex-shrink-0 mt-0.5" />
                  <p className="text-[14px] text-apple-dark2 leading-relaxed">{conflict.recommendation}</p>
                </div>
              )}

              {conflict.pain_point_id != null && (
                <div className="flex items-center gap-2 mt-3 text-[12px] text-apple-gray">
                  <Hash className="w-3.5 h-3.5" />
                  Pain Point {conflict.pain_point_id}
                </div>
              )}
            </motion.div>
          )
        })}
      </motion.div>
    </motion.div>
  )
}

function RippleTab({ stackId }: { stackId: string }) {
  const [sectionNumber, setSectionNumber] = useState('')
  const [currentText, setCurrentText] = useState('')
  const [proposedText, setProposedText] = useState('')
  const rippleMutation = useRippleEffects(stackId)

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!sectionNumber.trim()) return
    await rippleMutation.mutateAsync({
      clause_section: sectionNumber.trim(),
      current_text: currentText.trim(),
      proposed_text: proposedText.trim(),
      change_type: 'modify',
      description: `Modify section ${sectionNumber.trim()}`,
    })
  }

  const result = rippleMutation.data as any

  const hopLabels: Record<number, { label: string; description: string }> = {
    1: { label: 'Direct Impact', description: 'Immediately affected clauses' },
    2: { label: 'Indirect Impact', description: 'Secondary dependencies' },
    3: { label: 'Cascade Effect', description: 'Tertiary ripple effects' },
  }

  const groupedImpacts: Record<number, any[]> = {}
  if (result?.impacts) {
    result.impacts.forEach((impact: any) => {
      const hop = impact.hop_level || impact.hop || 1
      if (!groupedImpacts[hop]) groupedImpacts[hop] = []
      groupedImpacts[hop].push(impact)
    })
  }

  return (
    <div className="space-y-8">
      <motion.div
        {...fadeUp}
        transition={{ duration: 0.4 }}
        className="bg-white rounded-3xl border border-black/[0.04] p-8"
      >
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-2xl bg-apple-bg flex items-center justify-center">
            <Waves className="w-5 h-5 text-apple-dark2" />
          </div>
          <div>
            <h3 className="text-[18px] font-semibold text-apple-black tracking-tight">Propose a Change</h3>
            <p className="text-[13px] text-apple-gray">Analyze the ripple effect of contract modifications</p>
          </div>
        </div>

        <form onSubmit={handleAnalyze} className="space-y-5">
          <div>
            <label className="block text-[13px] font-medium text-apple-dark2 mb-2">Section Number</label>
            <input
              type="text"
              value={sectionNumber}
              onChange={(e) => setSectionNumber(e.target.value)}
              placeholder="e.g. 7.2"
              className="w-full px-4 py-3 bg-apple-offwhite rounded-xl border border-black/[0.04] text-[15px] text-apple-black placeholder:text-apple-light focus:outline-none focus:border-apple-dark/20 focus:bg-white focus:shadow-[0_0_0_3px_rgba(0,0,0,0.04)] transition-all duration-300"
            />
          </div>
          <div>
            <label className="block text-[13px] font-medium text-apple-dark2 mb-2">Current Text</label>
            <textarea
              value={currentText}
              onChange={(e) => setCurrentText(e.target.value)}
              placeholder="Paste the current clause text..."
              rows={3}
              className="w-full px-4 py-3 bg-apple-offwhite rounded-xl border border-black/[0.04] text-[15px] text-apple-black placeholder:text-apple-light focus:outline-none focus:border-apple-dark/20 focus:bg-white focus:shadow-[0_0_0_3px_rgba(0,0,0,0.04)] transition-all duration-300 resize-none"
            />
          </div>
          <div>
            <label className="block text-[13px] font-medium text-apple-dark2 mb-2">Proposed Text</label>
            <textarea
              value={proposedText}
              onChange={(e) => setProposedText(e.target.value)}
              placeholder="Enter the proposed replacement text..."
              rows={3}
              className="w-full px-4 py-3 bg-apple-offwhite rounded-xl border border-black/[0.04] text-[15px] text-apple-black placeholder:text-apple-light focus:outline-none focus:border-apple-dark/20 focus:bg-white focus:shadow-[0_0_0_3px_rgba(0,0,0,0.04)] transition-all duration-300 resize-none"
            />
          </div>
          <motion.button
            type="submit"
            disabled={!sectionNumber.trim() || rippleMutation.isPending}
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.98 }}
            className="flex items-center gap-2.5 px-7 py-3.5 bg-apple-pure text-white text-[15px] font-semibold rounded-full hover:bg-apple-dark transition-all duration-300 disabled:opacity-30 shadow-[0_2px_12px_rgba(0,0,0,0.15)]"
          >
            {rippleMutation.isPending ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Zap className="w-5 h-5" />
            )}
            {rippleMutation.isPending ? 'Analyzing...' : 'Analyze Impact'}
          </motion.button>
        </form>
      </motion.div>

      {rippleMutation.isPending && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center py-16"
        >
          <div className="relative w-24 h-24 mx-auto mb-6">
            {[0, 1, 2].map((ring) => (
              <motion.div
                key={ring}
                className="absolute inset-0 rounded-full border-2 border-apple-dark/20"
                animate={{
                  scale: [1, 1.8 + ring * 0.4],
                  opacity: [0.6, 0],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  delay: ring * 0.4,
                  ease: 'easeOut',
                }}
              />
            ))}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-12 h-12 rounded-full bg-apple-bg flex items-center justify-center">
                <Waves className="w-6 h-6 text-apple-dark" />
              </div>
            </div>
          </div>
          <p className="text-[18px] font-semibold text-apple-black tracking-tight">Tracing ripple effects...</p>
          <p className="text-[14px] text-apple-gray mt-1">Mapping dependencies across all documents</p>
        </motion.div>
      )}

      {result && !rippleMutation.isPending && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="space-y-8"
        >
          {Object.entries(groupedImpacts).sort(([a], [b]) => Number(a) - Number(b)).map(([hop, impacts]) => {
            const hopNum = Number(hop)
            const meta = hopLabels[hopNum] || { label: `Hop ${hop}`, description: 'Extended effects' }
            return (
              <div key={hop}>
                <div className="flex items-center gap-3 mb-4">
                  <div className={`w-8 h-8 rounded-xl flex items-center justify-center ${
                    hopNum === 1 ? 'bg-apple-pure' : hopNum === 2 ? 'bg-apple-dark2' : 'bg-apple-gray'
                  }`}>
                    <span className="text-[13px] font-bold text-white">{hop}</span>
                  </div>
                  <div>
                    <h4 className="text-[16px] font-semibold text-apple-black">{meta.label}</h4>
                    <p className="text-[12px] text-apple-gray">{meta.description}</p>
                  </div>
                </div>

                <div className="space-y-3 ml-11">
                  {impacts.map((impact: any, j: number) => (
                    <motion.div
                      key={j}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: j * 0.06 }}
                      className="bg-white rounded-2xl border border-black/[0.04] p-5 hover:shadow-[0_4px_12px_rgba(0,0,0,0.04)] transition-shadow duration-300"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          {impact.section && (
                            <span className="px-2.5 py-1 bg-apple-bg rounded-lg text-[13px] font-mono font-medium text-apple-dark2">
                              {impact.section || impact.affected_section}
                            </span>
                          )}
                          {(impact.impact_type || impact.type) && (
                            <span className="px-2.5 py-1 bg-apple-silver/50 rounded-lg text-[12px] font-medium text-apple-gray2 capitalize">
                              {(impact.impact_type || impact.type).replace('_', ' ')}
                            </span>
                          )}
                        </div>
                        {impact.severity && (
                          <span className="text-[11px] font-bold uppercase tracking-wider text-apple-gray">
                            {impact.severity}
                          </span>
                        )}
                      </div>
                      <p className="text-[14px] text-apple-dark leading-relaxed">
                        {impact.description || impact.explanation}
                      </p>
                    </motion.div>
                  ))}
                </div>
              </div>
            )
          })}

          {(result.summary || result.recommendations || result.estimated_cost || result.estimated_timeline) && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-apple-offwhite rounded-3xl border border-black/[0.03] p-7"
            >
              <h4 className="text-[17px] font-semibold text-apple-black tracking-tight mb-4">Impact Summary</h4>
              {result.summary && (
                <p className="text-[15px] text-apple-dark2 leading-relaxed mb-4">
                  {typeof result.summary === 'string' ? result.summary : JSON.stringify(result.summary)}
                </p>
              )}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {result.total_impacts != null && (
                  <div className="bg-white rounded-2xl p-5 border border-black/[0.03]">
                    <p className="text-[12px] font-medium text-apple-gray uppercase tracking-wider mb-1">Total Impacts</p>
                    <p className="text-[28px] font-bold text-apple-pure tracking-tight">{result.total_impacts}</p>
                  </div>
                )}
                {result.estimated_cost && (
                  <div className="bg-white rounded-2xl p-5 border border-black/[0.03]">
                    <p className="text-[12px] font-medium text-apple-gray uppercase tracking-wider mb-1">Estimated Cost</p>
                    <p className="text-[28px] font-bold text-apple-pure tracking-tight">{result.estimated_cost}</p>
                  </div>
                )}
                {result.estimated_timeline && (
                  <div className="bg-white rounded-2xl p-5 border border-black/[0.03]">
                    <p className="text-[12px] font-medium text-apple-gray uppercase tracking-wider mb-1">Timeline</p>
                    <p className="text-[28px] font-bold text-apple-pure tracking-tight">{result.estimated_timeline}</p>
                  </div>
                )}
                {result.risk_level && (
                  <div className="bg-white rounded-2xl p-5 border border-black/[0.03]">
                    <p className="text-[12px] font-medium text-apple-gray uppercase tracking-wider mb-1">Risk Level</p>
                    <p className="text-[28px] font-bold text-apple-pure tracking-tight capitalize">{result.risk_level}</p>
                  </div>
                )}
              </div>
              {result.recommendations && Array.isArray(result.recommendations) && result.recommendations.length > 0 && (
                <div className="mt-5 pt-5 border-t border-black/[0.04]">
                  <p className="text-[13px] font-semibold text-apple-dark mb-3">Recommendations</p>
                  <div className="space-y-2">
                    {result.recommendations.map((rec: any, k: number) => (
                      <div key={k} className="flex gap-3 items-start">
                        <div className="w-5 h-5 rounded-full bg-apple-bg flex items-center justify-center flex-shrink-0 mt-0.5">
                          <span className="text-[10px] font-bold text-apple-dark2">{k + 1}</span>
                        </div>
                        <p className="text-[14px] text-apple-dark2 leading-relaxed">
                          {typeof rec === 'string' ? rec : rec.text || rec.recommendation || JSON.stringify(rec)}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </motion.div>
      )}

      {rippleMutation.isError && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-3 px-6 py-4 bg-apple-bg rounded-2xl border border-black/[0.04]"
        >
          <AlertTriangle className="w-5 h-5 text-apple-dark2 flex-shrink-0" />
          <p className="text-[14px] text-apple-dark">
            {(rippleMutation.error as Error)?.message || 'Analysis failed. Please try again.'}
          </p>
        </motion.div>
      )}
    </div>
  )
}
