import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Document as PdfDocument, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'
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
  ChevronLeft,
  ChevronRight,
  PanelLeftClose,
  PanelLeft,
  Eye,
  Pencil,
  Plus,
  RefreshCw,
  ChevronDown,
  History,
  Bell,
  Calendar,
  Scale,
  ExternalLink,
  Activity,
  HeartPulse,
  Bold,
  Italic,
  Underline,
  Strikethrough,
  AlignLeft,
  AlignCenter,
  AlignRight,
  AlignJustify,
  List,
  ListOrdered,
  Indent,
  Outdent,
  Undo2,
  Redo2,
  Highlighter,
  Minus,
  Superscript,
  Subscript,
  Clipboard,
  ClipboardPaste,
  Scissors,
  Paintbrush,
  Type,
  ChevronUp,
  Download,
  Save,
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
  useCachedRippleResult,
  useDocumentClauses,
  useConsolidatedContract,
} from '../hooks/useApi'
import ReactMarkdown from 'react-markdown'
import { useEditor, EditorContent } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import TextAlign from '@tiptap/extension-text-align'
import TiptapSubscript from '@tiptap/extension-subscript'
import TiptapSuperscript from '@tiptap/extension-superscript'
import TiptapHighlight from '@tiptap/extension-highlight'
import { Node as TiptapNode } from '@tiptap/core'
import { Document as DocxDocument, Packer, Paragraph, TextRun, HeadingLevel } from 'docx'
import { saveAs } from 'file-saver'
import jsPDF from 'jspdf'

const PageBreakNode = TiptapNode.create({
  name: 'pageBreak',
  group: 'block',
  atom: true,
  selectable: false,
  draggable: false,

  parseHTML() {
    return [{ tag: 'div[data-page-break]' }]
  },

  renderHTML() {
    return ['div', {
      'data-page-break': 'true',
      contenteditable: 'false',
      class: 'word-page-break-node',
    }]
  },
})
import { api } from '../api/client'
import type { Conflict, QueryResponse, TimelineEntry, DocumentClause, SourceChainLink, ClauseConflict, ConsolidatedSection } from '../types'

pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`

const TABS = [
  { id: 'overview', label: 'Overview', icon: Layers },
  { id: 'health', label: 'Contract Health', icon: HeartPulse },
  { id: 'timeline', label: 'Timeline', icon: GitBranch },
  { id: 'query', label: 'Query', icon: MessageSquare },
  { id: 'conflicts', label: 'Conflicts', icon: Shield },
  { id: 'ripple', label: 'Ripple Effects', icon: Waves },
  { id: 'consolidate', label: 'Consolidate', icon: FileText },
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
            {activeTab === 'health' && <ContractHealthSection stack={stack} />}
            {activeTab === 'timeline' && <TimelineTab stackId={id} />}
            {activeTab === 'query' && <QueryTab stackId={id} />}
            {activeTab === 'conflicts' && <ConflictsTab stackId={id} />}
            {activeTab === 'ripple' && <RippleTab stackId={id} />}
            {activeTab === 'consolidate' && <ConsolidateTab stackId={id} />}
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
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null)
  const uploadMutation = useUploadDocument(stackId)
  const processMutation = useProcessStack(stackId)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const selectedDoc = documents.find((d: any) => d.id === selectedDocId)

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
                onClick={() => doc.processed && setSelectedDocId(doc.id)}
                className={`group relative bg-white rounded-2xl border border-black/[0.04] p-5 hover:shadow-[0_8px_24px_rgba(0,0,0,0.06)] hover:border-black/[0.08] transition-all duration-400 ${doc.processed ? 'cursor-pointer' : 'cursor-default'}`}
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
                <div className="flex items-center justify-between mt-2.5">
                  {doc.document_version ? (
                    <span className="text-[11px] text-apple-gray font-mono bg-apple-silver/40 px-2 py-0.5 rounded-md">
                      v{doc.document_version}
                    </span>
                  ) : <span />}
                  {doc.processed && (
                    <span className="flex items-center gap-1 text-[11px] text-apple-gray opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                      <Eye className="w-3 h-3" />
                      View
                    </span>
                  )}
                </div>
              </motion.div>
            ))}
          </motion.div>
        )}
      </motion.div>

      <AnimatePresence>
        {selectedDocId && selectedDoc && (
          <DocumentDetailView
            stackId={stackId}
            documentId={selectedDocId}
            filename={selectedDoc.filename}
            documentType={selectedDoc.document_type}
            onClose={() => setSelectedDocId(null)}
          />
        )}
      </AnimatePresence>

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

function ContractHealthSection({ stack }: { stack: any }) {
  const [expandedAlert, setExpandedAlert] = useState<number | null>(null)

  const studyProtocol = stack?.study_protocol || 'HEARTBEAT-3'
  const siteName = stack?.site_name || 'Memorial Medical Center'

  const alerts = [
    {
      id: 1,
      severity: 'high',
      title: 'Insurance Certificate Expiration',
      description: `${siteName}: Certificate of insurance expires Mar 31, 2026 but study timeline extends to September 2026`,
      daysRemaining: 50,
      category: 'Compliance',
      detail: `The site's certificate of insurance (COI) for clinical trial liability coverage expires on March 31, 2026. However, per Amendment 5, the study timeline has been extended with anticipated LPLV of September 2026 and site closeout in December 2026. The original CTA §12.1 requires continuous insurance coverage throughout the study period. Immediate action is needed to obtain an updated COI extending coverage through study completion.`,
      recommendation: 'Contact site to request updated Certificate of Insurance extending coverage through December 2026.',
    },
    {
      id: 2,
      severity: 'medium',
      title: 'IRB Approval Approaching Expiry',
      description: `Current IRB approval valid through April 30, 2026 — continuing review submission required`,
      daysRemaining: 80,
      category: 'Regulatory',
      detail: `Per Amendment 5 §4.1, the Institution's current IRB approval is valid through April 30, 2026. The study is expected to continue through September 2026 (LPLV). The clause requires "timely continuing review" to maintain approval, but does not define "timely" in specific days. Failure to submit continuing review before expiry would create a gap in IRB coverage, potentially requiring suspension of study activities.`,
      recommendation: 'Notify site coordinator to submit IRB continuing review at least 30 days before April 30, 2026 expiry.',
    },
    {
      id: 3,
      severity: 'low',
      title: 'Informed Consent Update Pending',
      description: `Protocol Amendment No. 2 consent updates — verify all active subjects re-consented`,
      daysRemaining: null,
      category: 'Compliance',
      detail: `Amendment 5 §4.1 confirms Protocol Amendment No. 2 was submitted and approved by the IRB on February 28, 2025, and that subjects were re-consented "as necessary." However, the contract does not specify a definitive completion date for re-consent activities or confirm 100% re-consent coverage. A compliance check should verify that all active subjects have signed the updated informed consent form.`,
      recommendation: 'Request site confirmation that all active subjects have been re-consented per Protocol Amendment No. 2.',
    },
  ]

  const upcomingEvents = [
    { label: 'Week 48 biomarker collection window closes', date: 'Q1 2026', icon: Activity },
    { label: 'IRB continuing review due', date: 'Mar 2026', icon: Shield },
    { label: 'Anticipated LPLV', date: 'Sep 2026', icon: Calendar },
    { label: 'Database lock', date: 'Nov 2026', icon: Layers },
    { label: 'Site closeout visit', date: 'Dec 2026', icon: CheckCircle2 },
  ]

  const regulatoryChanges = [
    {
      title: 'Remote monitoring documentation requirements',
      description: 'Amendment 3 §9.3(d) introduced 21 CFR Part 11 e-signature requirements for remote visits — verify site compliance with electronic signature validation procedures.',
      affectedSections: 3,
    },
  ]

  const severityStyles: Record<string, { bg: string; border: string; dot: string }> = {
    high: { bg: 'bg-[#f5f5f5]', border: 'border-apple-dark2', dot: 'bg-apple-black' },
    medium: { bg: 'bg-[#f8f8f8]', border: 'border-apple-gray', dot: 'bg-apple-gray' },
    low: { bg: 'bg-[#fafafa]', border: 'border-apple-silver', dot: 'bg-apple-light' },
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.35 }}
      className="space-y-8"
    >
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-[20px] font-semibold text-apple-black tracking-tight">Contract Health</h3>
          <p className="text-[13px] text-apple-gray mt-1">Real-time monitoring of compliance, deadlines, and regulatory status</p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 bg-apple-bg rounded-full">
          <Activity className="w-3.5 h-3.5 text-apple-dark2" />
          <span className="text-[12px] font-medium text-apple-dark2">Live</span>
        </div>
      </div>

      <div>
        <div className="flex items-center gap-2 mb-4">
          <Bell className="w-4 h-4 text-apple-dark2" />
          <h4 className="text-[15px] font-semibold text-apple-black">Active Alerts</h4>
          <span className="text-[11px] font-semibold text-white bg-apple-black px-2 py-0.5 rounded-full">
            {alerts.length}
          </span>
        </div>
        <div className="space-y-3">
          {alerts.map((alert) => {
            const styles = severityStyles[alert.severity]
            const isExpanded = expandedAlert === alert.id
            return (
              <motion.div
                key={alert.id}
                layout
                className={`${styles.bg} rounded-2xl border ${isExpanded ? styles.border : 'border-black/[0.04]'} overflow-hidden transition-colors duration-300`}
              >
                <button
                  onClick={() => setExpandedAlert(isExpanded ? null : alert.id)}
                  className="w-full text-left px-5 py-4 flex items-start gap-4"
                >
                  <div className={`w-2.5 h-2.5 rounded-full ${styles.dot} mt-1.5 shrink-0`} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-[14px] font-semibold text-apple-black">{alert.title}</span>
                      <span className="text-[10px] font-semibold uppercase tracking-wider text-apple-dark2 bg-apple-silver/60 px-1.5 py-0.5 rounded">
                        {alert.category}
                      </span>
                    </div>
                    <p className="text-[13px] text-apple-gray2 mt-1 leading-relaxed">{alert.description}</p>
                  </div>
                  <div className="flex items-center gap-3 shrink-0">
                    {alert.daysRemaining && (
                      <span className="text-[12px] font-medium text-apple-dark2 bg-white px-2.5 py-1 rounded-lg border border-black/[0.04]">
                        {alert.daysRemaining}d
                      </span>
                    )}
                    <ChevronDown className={`w-4 h-4 text-apple-light transition-transform duration-300 ${isExpanded ? 'rotate-180' : ''}`} />
                  </div>
                </button>
                <AnimatePresence>
                  {isExpanded && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
                      className="overflow-hidden"
                    >
                      <div className="px-5 pb-5 ml-6.5 border-t border-black/[0.04]">
                        <div className="pt-4 space-y-4">
                          <div>
                            <h5 className="text-[12px] font-semibold text-apple-dark2 uppercase tracking-wider mb-2">Analysis</h5>
                            <p className="text-[13px] text-apple-gray2 leading-relaxed">{alert.detail}</p>
                          </div>
                          <div>
                            <h5 className="text-[12px] font-semibold text-apple-dark2 uppercase tracking-wider mb-2">Recommended Action</h5>
                            <div className="flex items-start gap-3 bg-white rounded-xl px-4 py-3 border border-black/[0.04]">
                              <Zap className="w-4 h-4 text-apple-dark2 mt-0.5 shrink-0" />
                              <p className="text-[13px] text-apple-black font-medium leading-relaxed">{alert.recommendation}</p>
                            </div>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            )
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <div className="flex items-center gap-2 mb-4">
            <Calendar className="w-4 h-4 text-apple-dark2" />
            <h4 className="text-[15px] font-semibold text-apple-black">Upcoming Events</h4>
            <span className="text-[11px] font-semibold text-apple-dark2 bg-apple-bg px-2 py-0.5 rounded-full">
              {upcomingEvents.length}
            </span>
          </div>
          <div className="space-y-2">
            {upcomingEvents.map((event, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4 + i * 0.06, duration: 0.4 }}
                className="flex items-center gap-3 px-4 py-3 bg-white rounded-xl border border-black/[0.04] hover:shadow-[0_2px_8px_rgba(0,0,0,0.04)] transition-all duration-200"
              >
                <div className="w-8 h-8 rounded-lg bg-apple-bg flex items-center justify-center shrink-0">
                  <event.icon className="w-4 h-4 text-apple-dark2" />
                </div>
                <span className="text-[13px] text-apple-black font-medium flex-1">{event.label}</span>
                <span className="text-[12px] text-apple-gray font-medium bg-apple-bg px-2.5 py-1 rounded-lg">{event.date}</span>
              </motion.div>
            ))}
          </div>
        </div>

        <div>
          <div className="flex items-center gap-2 mb-4">
            <Scale className="w-4 h-4 text-apple-dark2" />
            <h4 className="text-[15px] font-semibold text-apple-black">Regulatory Watch</h4>
            <span className="text-[11px] font-semibold text-apple-dark2 bg-apple-bg px-2 py-0.5 rounded-full">
              {regulatoryChanges.length}
            </span>
          </div>
          <div className="space-y-2">
            {regulatoryChanges.map((change, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 + i * 0.06, duration: 0.4 }}
                className="bg-white rounded-xl border border-black/[0.04] px-4 py-4 hover:shadow-[0_2px_8px_rgba(0,0,0,0.04)] transition-all duration-200"
              >
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-lg bg-apple-bg flex items-center justify-center shrink-0 mt-0.5">
                    <ExternalLink className="w-4 h-4 text-apple-dark2" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-[13px] font-semibold text-apple-black">{change.title}</p>
                    <p className="text-[12px] text-apple-gray2 mt-1.5 leading-relaxed">{change.description}</p>
                    <span className="inline-block mt-2 text-[11px] font-medium text-apple-dark2 bg-apple-bg px-2 py-0.5 rounded">
                      {change.affectedSections} sections affected
                    </span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </motion.div>
  )
}

function DocumentDetailView({
  stackId,
  documentId,
  filename,
  documentType,
  onClose,
}: {
  stackId: string
  documentId: string
  filename: string
  documentType: string
  onClose: () => void
}) {
  const { data: clausesData, isLoading: clausesLoading } = useDocumentClauses(stackId, documentId)
  const [numPages, setNumPages] = useState<number | null>(null)
  const [pageNumber, setPageNumber] = useState(1)
  const [showClauses, setShowClauses] = useState(true)
  const [selectedClause, setSelectedClause] = useState<string | null>(null)
  const [searchingPage, setSearchingPage] = useState(false)
  const pdfDocRef = useRef<any>(null)
  const pdfUrl = api.getDocumentPdfUrl(stackId, documentId)

  const onDocumentLoadSuccess = useCallback(({ numPages }: { numPages: number }) => {
    setNumPages(numPages)
  }, [])

  const findClausePage = useCallback(async (sectionNumber: string) => {
    if (!pdfUrl || !numPages) return
    setSearchingPage(true)
    setSelectedClause(sectionNumber)
    try {
      const loadingTask = pdfjs.getDocument(pdfUrl)
      const pdf = await loadingTask.promise
      const normalizedSection = sectionNumber.replace(/\s+/g, '').toLowerCase()
      for (let p = 1; p <= pdf.numPages; p++) {
        const page = await pdf.getPage(p)
        const textContent = await page.getTextContent()
        const pageText = textContent.items.map((item: any) => item.str).join(' ').toLowerCase()
        const patterns = [
          `section ${normalizedSection}`,
          `article ${normalizedSection}`,
          normalizedSection + '.',
          normalizedSection + ' ',
        ]
        if (patterns.some(pat => pageText.includes(pat))) {
          setPageNumber(p)
          break
        }
      }
    } catch (err) {
      console.warn('PDF text search failed:', err)
    } finally {
      setSearchingPage(false)
    }
  }, [pdfUrl, numPages])

  // Close on Escape key
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [onClose])

  const clauses = clausesData?.clauses || []
  const [expandedHistory, setExpandedHistory] = useState<Set<string>>(new Set())
  const [expandedConflicts, setExpandedConflicts] = useState<Set<string>>(new Set())

  const toggleHistory = (sectionNumber: string) => {
    setExpandedHistory(prev => {
      const next = new Set(prev)
      next.has(sectionNumber) ? next.delete(sectionNumber) : next.add(sectionNumber)
      return next
    })
  }

  const toggleConflict = (conflictId: string) => {
    setExpandedConflicts(prev => {
      const next = new Set(prev)
      next.has(conflictId) ? next.delete(conflictId) : next.add(conflictId)
      return next
    })
  }

  const categoryColors: Record<string, string> = {
    financial: 'bg-amber-100 text-amber-800',
    payment: 'bg-amber-100 text-amber-800',
    insurance: 'bg-blue-100 text-blue-800',
    personnel: 'bg-purple-100 text-purple-800',
    regulatory: 'bg-red-100 text-red-800',
    operational: 'bg-green-100 text-green-800',
    termination: 'bg-rose-100 text-rose-800',
    indemnification: 'bg-orange-100 text-orange-800',
    confidentiality: 'bg-slate-100 text-slate-700',
  }

  const modificationConfig: Record<string, { label: string; className: string; icon: typeof Pencil }> = {
    selective_override: { label: 'Modified', className: 'bg-orange-100 text-orange-700', icon: Pencil },
    modify: { label: 'Modified', className: 'bg-orange-100 text-orange-700', icon: Pencil },
    complete_replacement: { label: 'Replaced', className: 'bg-red-100 text-red-700', icon: RefreshCw },
    replace: { label: 'Replaced', className: 'bg-red-100 text-red-700', icon: RefreshCw },
    new_addition: { label: 'Added', className: 'bg-green-100 text-green-700', icon: Plus },
    add: { label: 'Added', className: 'bg-green-100 text-green-700', icon: Plus },
    original: { label: 'Original', className: 'bg-gray-100 text-gray-600', icon: FileText },
  }

  const conflictSeverityConfig: Record<string, { border: string; dot: string }> = {
    critical: { border: 'border-l-[4px] border-l-red-500', dot: 'bg-red-500' },
    high: { border: 'border-l-[3px] border-l-apple-dark', dot: 'bg-apple-dark' },
    medium: { border: 'border-l-[3px] border-l-apple-gray', dot: 'bg-apple-gray' },
    low: { border: 'border-l-[2px] border-l-gray-300', dot: 'bg-gray-300' },
  }

  return (
    <>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.25 }}
        className="fixed inset-0 bg-black/50 backdrop-blur-md z-50"
        onClick={onClose}
      />
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 30 }}
        transition={{ duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
        className="fixed inset-3 sm:inset-6 bg-white/98 backdrop-blur-2xl rounded-3xl shadow-[0_32px_100px_rgba(0,0,0,0.2)] z-50 flex flex-col overflow-hidden border border-black/[0.06]"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-black/[0.05] bg-apple-offwhite/50">
          <div className="flex items-center gap-4 min-w-0">
            <div className="w-10 h-10 rounded-xl bg-apple-bg flex items-center justify-center flex-shrink-0">
              <FileText className="w-5 h-5 text-apple-dark2" />
            </div>
            <div className="min-w-0">
              <h2 className="text-[17px] font-semibold text-apple-black tracking-tight truncate">{filename}</h2>
              <span className="text-[12px] text-apple-gray capitalize">{documentType.replace('_', ' ')}</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowClauses(!showClauses)}
              className="flex items-center gap-2 px-3.5 py-2 rounded-xl text-[13px] font-medium text-apple-dark2 hover:bg-apple-bg transition-colors duration-200"
              title={showClauses ? 'Hide clause panel' : 'Show clause panel'}
            >
              {showClauses ? <PanelLeftClose className="w-4 h-4" /> : <PanelLeft className="w-4 h-4" />}
              <span className="hidden sm:inline">{showClauses ? 'Hide Clauses' : 'Show Clauses'}</span>
            </button>
            <button
              onClick={onClose}
              className="p-2 rounded-xl hover:bg-apple-bg transition-colors duration-200"
            >
              <X className="w-5 h-5 text-apple-gray" />
            </button>
          </div>
        </div>

        {/* Split panel */}
        <div className="flex-1 flex overflow-hidden">
          {/* Left: Clauses panel */}
          <AnimatePresence initial={false}>
            {showClauses && (
              <motion.div
                initial={{ width: 0, opacity: 0 }}
                animate={{ width: '50%', opacity: 1 }}
                exit={{ width: 0, opacity: 0 }}
                transition={{ duration: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
                className="border-r border-black/[0.05] flex flex-col overflow-hidden"
              >
                <div className="px-5 py-3 border-b border-black/[0.04] bg-apple-offwhite/30 flex items-center justify-between">
                  <span className="text-[13px] font-semibold text-apple-dark2 uppercase tracking-[0.06em]">
                    Extracted Clauses
                  </span>
                  <span className="text-[12px] text-apple-gray font-medium">
                    {clauses.length} {clauses.length === 1 ? 'clause' : 'clauses'}
                  </span>
                </div>
                <div className="flex-1 overflow-y-auto p-4 space-y-3">
                  {clausesLoading ? (
                    <div className="space-y-3">
                      {[1, 2, 3, 4].map((i) => (
                        <div key={i} className="h-24 skeleton rounded-xl" />
                      ))}
                    </div>
                  ) : clauses.length === 0 ? (
                    <div className="text-center py-16">
                      <BookOpen className="w-10 h-10 text-apple-light mx-auto mb-3" />
                      <p className="text-[14px] text-apple-gray">No clauses extracted yet</p>
                    </div>
                  ) : (
                    clauses.map((clause: DocumentClause, i: number) => {
                      const catClass = categoryColors[clause.clause_category?.toLowerCase()] || 'bg-apple-bg text-apple-dark2'
                      const chain = clause.source_chain || []
                      const lastLink = chain.length > 0 ? chain[chain.length - 1] : null
                      const modType = lastLink?.modification_type?.toLowerCase()
                      const modCfg = modType ? modificationConfig[modType] : null
                      const changeDesc = lastLink?.change_description
                      const conflicts = clause.conflicts || []
                      const isHistoryOpen = expandedHistory.has(clause.section_number)

                      const isSelected = selectedClause === clause.section_number

                      return (
                        <motion.div
                          key={`${clause.section_number}-${i}`}
                          initial={{ opacity: 0, y: 8 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: i * 0.03 }}
                          onClick={() => findClausePage(clause.section_number)}
                          className={`bg-white rounded-xl border p-4 cursor-pointer transition-all duration-200 ${
                            isSelected
                              ? 'border-apple-pure/40 shadow-[0_0_0_1px_rgba(29,29,31,0.15),0_4px_12px_rgba(0,0,0,0.08)] ring-1 ring-apple-pure/20'
                              : conflicts.length > 0
                                ? 'border-red-200/60 hover:shadow-[0_4px_12px_rgba(0,0,0,0.04)]'
                                : 'border-black/[0.04] hover:shadow-[0_4px_12px_rgba(0,0,0,0.04)]'
                          }`}
                        >
                          {/* Badges row */}
                          <div className="flex items-center gap-2 mb-2 flex-wrap">
                            <span className="px-2 py-0.5 bg-apple-pure text-white text-[11px] font-bold rounded-md font-mono">
                              {clause.section_number}
                            </span>
                            {clause.clause_category && (
                              <span className={`px-2 py-0.5 text-[10px] font-semibold rounded-md capitalize ${catClass}`}>
                                {clause.clause_category}
                              </span>
                            )}
                            {clause.is_current ? (
                              <span className="flex items-center gap-1 px-2 py-0.5 bg-green-50 text-green-700 text-[10px] font-semibold rounded-md">
                                <CheckCircle2 className="w-3 h-3" />
                                Current
                              </span>
                            ) : (
                              <span className="px-2 py-0.5 bg-apple-silver/50 text-apple-gray text-[10px] font-semibold rounded-md">
                                Superseded
                              </span>
                            )}
                            {modCfg && modType !== 'original' && (
                              <span className={`flex items-center gap-1 px-2 py-0.5 text-[10px] font-semibold rounded-md ${modCfg.className}`}>
                                <modCfg.icon className="w-3 h-3" />
                                {modCfg.label}
                              </span>
                            )}
                          </div>

                          {/* Section title */}
                          {clause.section_title && (
                            <p className="text-[13px] font-semibold text-apple-black mb-1.5">{clause.section_title}</p>
                          )}

                          {/* Change description callout */}
                          {changeDesc && (
                            <div className="flex items-start gap-2 px-3 py-2 mb-2 rounded-lg bg-amber-50/70 border border-amber-100/60">
                              <Info className="w-3.5 h-3.5 text-amber-600 mt-0.5 flex-shrink-0" />
                              <p className="text-[12px] text-amber-800 italic leading-relaxed">{changeDesc}</p>
                            </div>
                          )}

                          {/* Clause text */}
                          <p className="text-[13px] text-apple-dark2 leading-relaxed line-clamp-4">
                            {clause.current_text}
                          </p>

                          {/* Conflict indicators */}
                          {conflicts.length > 0 && (
                            <div className="mt-3 space-y-2">
                              {conflicts.map((conflict) => {
                                const sc = conflictSeverityConfig[conflict.severity] || conflictSeverityConfig.low
                                const isExpanded = expandedConflicts.has(conflict.conflict_id)
                                return (
                                  <div
                                    key={conflict.conflict_id}
                                    className={`rounded-lg bg-gray-50/80 p-2.5 ${sc.border} cursor-pointer transition-colors hover:bg-gray-100/80`}
                                    onClick={(e) => { e.stopPropagation(); toggleConflict(conflict.conflict_id); }}
                                  >
                                    <div className="flex items-center gap-2">
                                      <span className={`w-2 h-2 rounded-full flex-shrink-0 ${sc.dot}`} />
                                      <span className="text-[11px] font-semibold text-apple-dark2 capitalize">
                                        {conflict.conflict_type.replace(/_/g, ' ')}
                                      </span>
                                      <span className="text-[10px] text-apple-gray capitalize">· {conflict.severity}</span>
                                      {conflict.pain_point_id && (
                                        <span className="ml-auto px-1.5 py-0.5 bg-red-100 text-red-600 text-[9px] font-bold rounded">
                                          Pain Point #{conflict.pain_point_id}
                                        </span>
                                      )}
                                    </div>
                                    <p className={`text-[12px] text-apple-dark2/80 mt-1 leading-relaxed ${isExpanded ? '' : 'line-clamp-2'}`}>
                                      {conflict.description}
                                    </p>
                                    {isExpanded && conflict.recommendation && (
                                      <div className="mt-2 pt-2 border-t border-black/[0.04]">
                                        <p className="text-[11px] font-semibold text-apple-gray mb-0.5">Recommendation</p>
                                        <p className="text-[12px] text-apple-dark2/80 leading-relaxed">{conflict.recommendation}</p>
                                      </div>
                                    )}
                                  </div>
                                )
                              })}
                            </div>
                          )}

                          {/* Expandable source chain timeline */}
                          {chain.length > 1 && (
                            <div className="mt-3">
                              <button
                                onClick={(e) => { e.stopPropagation(); toggleHistory(clause.section_number); }}
                                className="flex items-center gap-1.5 text-[11px] font-medium text-apple-gray hover:text-apple-dark2 transition-colors"
                              >
                                <ChevronDown className={`w-3 h-3 transition-transform duration-200 ${isHistoryOpen ? 'rotate-180' : ''}`} />
                                <History className="w-3 h-3" />
                                {isHistoryOpen ? 'Hide' : 'Show'} amendment history ({chain.length} versions)
                              </button>
                              <AnimatePresence>
                                {isHistoryOpen && (
                                  <motion.div
                                    initial={{ height: 0, opacity: 0 }}
                                    animate={{ height: 'auto', opacity: 1 }}
                                    exit={{ height: 0, opacity: 0 }}
                                    transition={{ duration: 0.2 }}
                                    className="overflow-hidden"
                                  >
                                    <div className="mt-2 ml-1.5 border-l-2 border-apple-silver/40 pl-3 space-y-2.5">
                                      {chain.map((link: SourceChainLink, idx: number) => {
                                        const linkMod = link.modification_type?.toLowerCase()
                                        const linkCfg = linkMod ? modificationConfig[linkMod] : null
                                        const isCurrentVersion = link.text === clause.current_text
                                        return (
                                          <div key={idx} className={`relative ${isCurrentVersion ? 'opacity-100' : 'opacity-70'}`}>
                                            <div className="absolute -left-[17px] top-1 w-2.5 h-2.5 rounded-full border-2 border-white bg-apple-silver" />
                                            <div className="flex items-center gap-2 flex-wrap">
                                              <span className={`px-1.5 py-0.5 text-[10px] font-semibold rounded ${isCurrentVersion ? 'bg-apple-pure text-white' : 'bg-apple-bg text-apple-dark2'}`}>
                                                {link.document_label || `Stage ${link.stage}`}
                                              </span>
                                              {linkCfg && linkMod !== 'original' && (
                                                <span className={`px-1.5 py-0.5 text-[9px] font-semibold rounded ${linkCfg.className}`}>
                                                  {linkCfg.label}
                                                </span>
                                              )}
                                            </div>
                                            {link.change_description && (
                                              <p className="text-[11px] text-apple-gray italic mt-0.5">{link.change_description}</p>
                                            )}
                                          </div>
                                        )
                                      })}
                                    </div>
                                  </motion.div>
                                )}
                              </AnimatePresence>
                            </div>
                          )}

                          {/* Date footer */}
                          {clause.effective_date && (
                            <p className="text-[11px] text-apple-gray mt-2">Effective: {clause.effective_date}</p>
                          )}
                        </motion.div>
                      )
                    })
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Right: PDF viewer */}
          <div className="flex-1 flex flex-col bg-apple-bg/30 overflow-hidden">
            {/* PDF toolbar */}
            <div className="px-5 py-3 border-b border-black/[0.04] bg-white/60 backdrop-blur-sm flex items-center justify-center gap-4">
              <button
                onClick={() => setPageNumber(Math.max(1, pageNumber - 1))}
                disabled={pageNumber <= 1}
                className="p-1.5 rounded-lg hover:bg-apple-bg transition-colors duration-200 disabled:opacity-30"
              >
                <ChevronLeft className="w-4 h-4 text-apple-dark2" />
              </button>
              <span className="text-[13px] font-medium text-apple-dark2 min-w-[100px] text-center flex items-center justify-center gap-2">
                {searchingPage && <Loader2 className="w-3.5 h-3.5 animate-spin text-apple-gray" />}
                Page {pageNumber} {numPages ? `of ${numPages}` : ''}
              </span>
              <button
                onClick={() => setPageNumber(Math.min(numPages || pageNumber, pageNumber + 1))}
                disabled={numPages != null && pageNumber >= numPages}
                className="p-1.5 rounded-lg hover:bg-apple-bg transition-colors duration-200 disabled:opacity-30"
              >
                <ChevronRight className="w-4 h-4 text-apple-dark2" />
              </button>
            </div>
            {/* PDF content */}
            <div className="flex-1 overflow-auto flex justify-center py-6 px-4">
              <PdfDocument
                file={pdfUrl}
                onLoadSuccess={onDocumentLoadSuccess}
                loading={
                  <div className="flex items-center justify-center py-24">
                    <Loader2 className="w-8 h-8 animate-spin text-apple-gray" />
                  </div>
                }
                error={
                  <div className="text-center py-24">
                    <FileText className="w-12 h-12 text-apple-light mx-auto mb-3" />
                    <p className="text-[15px] text-apple-gray">Failed to load PDF</p>
                  </div>
                }
              >
                <Page
                  pageNumber={pageNumber}
                  renderTextLayer={true}
                  renderAnnotationLayer={true}
                  className="shadow-[0_4px_20px_rgba(0,0,0,0.1)] rounded-lg overflow-hidden"
                  width={Math.min(800, window.innerWidth - (showClauses ? 500 : 100))}
                />
              </PdfDocument>
            </div>
          </div>
        </div>
      </motion.div>
    </>
  )
}

function TimelineEntryCard({ stackId, entry, index }: { stackId: string; entry: TimelineEntry; index: number }) {
  const [expanded, setExpanded] = useState(false)
  const isRoot = entry.document_type === 'cta'
  const { data: clausesData, isLoading: clausesLoading } = useDocumentClauses(
    stackId,
    expanded ? entry.document_id : null
  )
  const clauses = clausesData?.clauses || []

  const modifiedClauses = clauses.filter((c: DocumentClause) => {
    if (!c.source_chain || c.source_chain.length < 2) return false
    const lastLink = c.source_chain[c.source_chain.length - 1]
    return lastLink?.modification_type && lastLink.modification_type !== 'original'
  })

  const newClauses = clauses.filter((c: DocumentClause) => {
    if (!c.source_chain || c.source_chain.length === 0) return true
    const lastLink = c.source_chain[c.source_chain.length - 1]
    return lastLink?.modification_type === 'added' || lastLink?.modification_type === 'original'
  })

  const displayClauses = isRoot ? clauses : (modifiedClauses.length > 0 ? modifiedClauses : clauses)

  const getChangeLabel = (clause: DocumentClause) => {
    if (!clause.source_chain || clause.source_chain.length === 0) return null
    const lastLink = clause.source_chain[clause.source_chain.length - 1]
    if (!lastLink?.modification_type || lastLink.modification_type === 'original') return null
    const typeMap: Record<string, string> = {
      'amended': 'Amended',
      'added': 'New',
      'superseded': 'Superseded',
      'replaced': 'Replaced',
      'modified': 'Modified',
      'deleted': 'Removed',
    }
    return typeMap[lastLink.modification_type] || lastLink.modification_type
  }

  const getChangeDescription = (clause: DocumentClause) => {
    if (!clause.source_chain || clause.source_chain.length === 0) return null
    const lastLink = clause.source_chain[clause.source_chain.length - 1]
    return lastLink?.change_description || null
  }

  return (
    <motion.div
      key={entry.document_id}
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.1, duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] }}
      className="relative flex gap-6 pl-0 py-3"
    >
      <div className="relative z-10 flex-shrink-0">
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: index * 0.1 + 0.15, type: 'spring', stiffness: 300, damping: 20 }}
          className={`w-16 h-16 rounded-2xl flex items-center justify-center shadow-[0_2px_8px_rgba(0,0,0,0.08)] ${
            isRoot
              ? 'bg-apple-pure shadow-[0_4px_16px_rgba(0,0,0,0.2)]'
              : 'bg-white border-2 border-apple-silver'
          }`}
        >
          <FileText className={`w-6 h-6 ${isRoot ? 'text-white' : 'text-apple-dark2'}`} />
        </motion.div>
      </div>

      <div className="flex-1">
        <button
          onClick={() => setExpanded(!expanded)}
          className={`w-full text-left rounded-2xl p-5 transition-all duration-300 ${
            isRoot
              ? 'bg-apple-pure text-white shadow-[0_4px_20px_rgba(0,0,0,0.15)]'
              : 'bg-white border border-black/[0.04] hover:shadow-[0_4px_16px_rgba(0,0,0,0.06)]'
          }`}
        >
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0 flex-1">
              <p className={`text-[16px] font-semibold truncate ${isRoot ? 'text-white' : 'text-apple-black'}`}>
                {entry.filename}
              </p>
              <p className={`text-[13px] mt-1 capitalize ${isRoot ? 'text-white/70' : 'text-apple-gray'}`}>
                {entry.document_type.replace('_', ' ')}
              </p>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              {entry.effective_date && (
                <span className={`text-[13px] font-medium px-3 py-1.5 rounded-xl ${
                  isRoot ? 'bg-white/15 text-white' : 'bg-apple-bg text-apple-dark2'
                }`}>
                  {entry.effective_date}
                </span>
              )}
              <ChevronDown className={`w-4 h-4 transition-transform duration-300 ${expanded ? 'rotate-180' : ''} ${
                isRoot ? 'text-white/60' : 'text-apple-light'
              }`} />
            </div>
          </div>
          {entry.document_version && (
            <span className={`inline-block mt-3 text-[12px] font-mono px-2.5 py-1 rounded-lg ${
              isRoot ? 'bg-white/10 text-white/80' : 'bg-apple-silver/50 text-apple-dark2'
            }`}>
              v{entry.document_version}
            </span>
          )}
        </button>

        <AnimatePresence>
          {expanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
              className="overflow-hidden"
            >
              <div className="mt-2 ml-2 border-l-2 border-apple-silver/60 pl-4 py-3 space-y-2">
                {clausesLoading ? (
                  <div className="flex items-center gap-2 py-4 px-3">
                    <Loader2 className="w-4 h-4 animate-spin text-apple-gray" />
                    <span className="text-[13px] text-apple-gray">Loading clauses...</span>
                  </div>
                ) : displayClauses.length === 0 ? (
                  <div className="py-4 px-3">
                    <p className="text-[13px] text-apple-gray">No clauses extracted from this document.</p>
                  </div>
                ) : (
                  <>
                    <div className="flex items-center gap-3 px-3 pb-2">
                      <span className="text-[12px] font-semibold text-apple-dark2 uppercase tracking-wider">
                        {isRoot ? 'Established Clauses' : 'Key Changes'}
                      </span>
                      <span className="text-[11px] text-apple-gray bg-apple-bg px-2 py-0.5 rounded-full">
                        {displayClauses.length} {displayClauses.length === 1 ? 'clause' : 'clauses'}
                      </span>
                    </div>
                    {displayClauses.slice(0, 10).map((clause: DocumentClause) => {
                      const changeLabel = getChangeLabel(clause)
                      const changeDesc = getChangeDescription(clause)
                      return (
                        <div
                          key={clause.section_number}
                          className="bg-apple-bg/60 rounded-xl px-4 py-3 hover:bg-apple-bg transition-colors duration-200"
                        >
                          <div className="flex items-start gap-3">
                            <span className="text-[12px] font-mono font-medium text-apple-dark2 bg-white px-2 py-0.5 rounded-md border border-black/[0.04] shrink-0 mt-0.5">
                              {clause.section_number}
                            </span>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 flex-wrap">
                                <span className="text-[13px] font-medium text-apple-black">
                                  {clause.section_title || clause.section_number}
                                </span>
                                {changeLabel && (
                                  <span className="text-[10px] font-semibold uppercase tracking-wider text-apple-dark2 bg-apple-silver/60 px-1.5 py-0.5 rounded">
                                    {changeLabel}
                                  </span>
                                )}
                                {clause.conflicts && clause.conflicts.length > 0 && (
                                  <span className="text-[10px] font-semibold uppercase tracking-wider text-apple-dark2 bg-apple-silver px-1.5 py-0.5 rounded flex items-center gap-1">
                                    <AlertTriangle className="w-2.5 h-2.5" />
                                    Conflict
                                  </span>
                                )}
                              </div>
                              {changeDesc && (
                                <p className="text-[12px] text-apple-gray2 mt-1 leading-relaxed line-clamp-2">
                                  {changeDesc}
                                </p>
                              )}
                              {!changeDesc && clause.current_text && (
                                <p className="text-[12px] text-apple-gray2 mt-1 leading-relaxed line-clamp-2">
                                  {clause.current_text}
                                </p>
                              )}
                            </div>
                          </div>
                        </div>
                      )
                    })}
                    {displayClauses.length > 10 && (
                      <div className="px-3 pt-1">
                        <span className="text-[12px] text-apple-gray">
                          + {displayClauses.length - 10} more clauses
                        </span>
                      </div>
                    )}
                  </>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
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
        {timeline.map((entry: TimelineEntry, i: number) => (
          <TimelineEntryCard key={entry.document_id} stackId={stackId} entry={entry} index={i} />
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

  const suggestions = [
    'What are the current payment terms?',
    'Show the history of Section 7.2',
    'What insurance obligations exist?',
    'Who is the current PI?',
    'What are the holdback provisions?',
    'What are all site reporting deadlines across amendments?',
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
                <div className="text-[15px] text-apple-black leading-[1.7] prose prose-sm max-w-none prose-headings:text-apple-black prose-headings:font-semibold prose-h2:text-[17px] prose-h2:mt-4 prose-h2:mb-2 prose-h3:text-[15px] prose-h3:mt-3 prose-h3:mb-1.5 prose-p:my-1.5 prose-li:my-0.5 prose-strong:text-apple-dark prose-ul:my-1.5 prose-ol:my-1.5">
                  <ReactMarkdown>{item.response.response.answer}</ReactMarkdown>
                </div>

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
  const [revealPhase, setRevealPhase] = useState<'idle' | 'scanning' | 'revealing' | 'done'>('idle')
  const [visibleCount, setVisibleCount] = useState(0)
  const [scanProgress, setScanProgress] = useState(0)
  const scanTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const revealTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const scanMessages = [
    'Initializing adversarial scan agents...',
    'Loading clause embeddings from vector store...',
    'Cross-referencing amendment chain §1–§15...',
    'Analyzing override conflicts across 6 documents...',
    'Scanning financial obligation contradictions...',
    'Evaluating regulatory compliance gaps...',
    'Running severity scoring model...',
    'Compiling conflict report...',
  ]
  const [scanMsgIdx, setScanMsgIdx] = useState(0)

  useEffect(() => {
    return () => {
      if (scanTimerRef.current) clearInterval(scanTimerRef.current)
      if (revealTimerRef.current) clearInterval(revealTimerRef.current)
    }
  }, [])

  const handleAnalyze = async () => {
    setRevealPhase('scanning')
    setScanProgress(0)
    setScanMsgIdx(0)

    const msgTimer = setInterval(() => {
      setScanMsgIdx(prev => {
        if (prev < scanMessages.length - 1) return prev + 1
        return prev
      })
    }, 800)

    const progressTimer = setInterval(() => {
      setScanProgress(prev => {
        if (prev >= 100) return 100
        return prev + Math.random() * 8 + 2
      })
    }, 200)
    scanTimerRef.current = progressTimer

    await conflictsMutation.mutateAsync('medium')
    setAnalyzed(true)

    clearInterval(progressTimer)
    clearInterval(msgTimer)
    scanTimerRef.current = null
    setScanProgress(100)
    setScanMsgIdx(scanMessages.length - 1)

    await new Promise(r => setTimeout(r, 600))

    setRevealPhase('revealing')
    setVisibleCount(0)

    const total = conflictsMutation.data?.conflicts?.length || 0
    if (total > 0) {
      let count = 0
      revealTimerRef.current = setInterval(() => {
        count++
        setVisibleCount(count)
        if (count >= total) {
          if (revealTimerRef.current) clearInterval(revealTimerRef.current)
          revealTimerRef.current = null
          setTimeout(() => setRevealPhase('done'), 400)
        }
      }, 700)
    } else {
      setRevealPhase('done')
    }
  }

  const data = conflictsMutation.data
  const conflicts = data?.conflicts || []

  const severityConfig: Record<string, { weight: string; dot: string }> = {
    critical: { weight: 'border-l-[5px] border-l-apple-pure', dot: 'bg-apple-pure' },
    high: { weight: 'border-l-4 border-l-apple-dark', dot: 'bg-apple-dark' },
    medium: { weight: 'border-l-[3px] border-l-apple-gray', dot: 'bg-apple-gray' },
    low: { weight: 'border-l-2 border-l-apple-light', dot: 'bg-apple-light' },
  }

  if (!analyzed && revealPhase === 'idle' && !conflictsMutation.isPending) {
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

  if (revealPhase === 'scanning' || conflictsMutation.isPending) {
    return (
      <motion.div {...fadeUp} className="text-center py-20">
        <motion.div
          animate={{ scale: [1, 1.15, 1], opacity: [0.6, 1, 0.6] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="w-20 h-20 rounded-full bg-apple-bg flex items-center justify-center mx-auto mb-6"
        >
          <Shield className="w-10 h-10 text-apple-dark" />
        </motion.div>
        <p className="text-[20px] font-semibold text-apple-black tracking-tight mb-2">AI agents analyzing contracts...</p>
        <motion.p
          key={scanMsgIdx}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="text-[14px] text-apple-gray mb-6 h-5"
        >
          {scanMessages[Math.min(scanMsgIdx, scanMessages.length - 1)]}
        </motion.p>
        <div className="max-w-xs mx-auto">
          <div className="h-1 bg-apple-silver/60 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-apple-dark2 rounded-full"
              animate={{ width: `${Math.min(scanProgress, 100)}%` }}
              transition={{ duration: 0.3, ease: 'easeOut' }}
            />
          </div>
          <p className="text-[11px] text-apple-light mt-2 font-medium">
            {Math.min(Math.round(scanProgress), 100)}% complete
          </p>
        </div>
      </motion.div>
    )
  }

  const displayConflicts = revealPhase === 'done' ? conflicts : conflicts.slice(0, visibleCount)
  const isRevealing = revealPhase === 'revealing'

  return (
    <motion.div {...fadeUp} transition={{ duration: 0.4 }}>
      {data?.summary && Object.keys(data.summary).length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-3 mb-8 flex-wrap"
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
          {isRevealing && (
            <div className="flex items-center gap-2 ml-2">
              <Loader2 className="w-4 h-4 text-apple-gray animate-spin" />
              <span className="text-[12px] text-apple-gray font-medium">
                Discovered {visibleCount} of {conflicts.length}...
              </span>
            </div>
          )}
        </motion.div>
      )}

      {conflicts.length === 0 && revealPhase === 'done' && (
        <div className="text-center py-16">
          <div className="w-16 h-16 rounded-full bg-apple-bg flex items-center justify-center mx-auto mb-4">
            <CheckCircle2 className="w-8 h-8 text-apple-dark" />
          </div>
          <p className="text-[20px] font-semibold text-apple-black tracking-tight mb-1">No conflicts detected</p>
          <p className="text-[15px] text-apple-gray">Your contract stack appears consistent</p>
        </div>
      )}

      <div className="space-y-4">
        <AnimatePresence>
          {displayConflicts.map((conflict: Conflict) => {
            const sc = severityConfig[conflict.severity] || severityConfig.low
            return (
              <motion.div
                key={conflict.id}
                initial={{ opacity: 0, y: 20, scale: 0.97 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] }}
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

                {conflict.evidence && conflict.evidence.length > 0 && (() => {
                  const docs = [...new Set(conflict.evidence.map(e => e.document_label).filter(Boolean))]
                  const isOriginal = docs.length === 1 && docs[0].toLowerCase().includes('original')
                  return docs.length > 0 ? (
                    <div className="flex items-center gap-2 mb-3 flex-wrap">
                      <span className="text-[11px] font-semibold text-apple-gray uppercase tracking-wider">
                        {isOriginal ? 'Present in' : 'Introduced by'}
                      </span>
                      {docs.map((doc, j) => (
                        <span key={j} className={`px-2.5 py-1 rounded-lg text-[12px] font-medium ${
                          doc.toLowerCase().includes('original') ? 'bg-slate-100 text-slate-600' : 'bg-indigo-50 text-indigo-700'
                        }`}>
                          {doc.replace(/\.pdf$/i, '')}
                        </span>
                      ))}
                    </div>
                  ) : null
                })()}

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
        </AnimatePresence>
      </div>

      {isRevealing && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex items-center justify-center gap-3 py-6"
        >
          <motion.div
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 1.5, repeat: Infinity }}
            className="w-2 h-2 rounded-full bg-apple-dark2"
          />
          <span className="text-[13px] text-apple-gray font-medium">Scanning for more conflicts...</span>
        </motion.div>
      )}

      {revealPhase === 'done' && conflicts.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="flex items-center justify-center gap-2 py-6"
        >
          <CheckCircle2 className="w-4 h-4 text-apple-dark2" />
          <span className="text-[13px] text-apple-dark2 font-medium">
            Scan complete — {conflicts.length} conflict{conflicts.length !== 1 ? 's' : ''} detected across all amendments
          </span>
        </motion.div>
      )}
    </motion.div>
  )
}

function RippleTab({ stackId }: { stackId: string }) {
  const [sectionNumber, setSectionNumber] = useState('')
  const [currentText, setCurrentText] = useState('')
  const [proposedText, setProposedText] = useState('')
  const [lastChange, setLastChange] = useState<Record<string, unknown> | null>(null)
  const rippleMutation = useRippleEffects(stackId)
  const cachedResult = useCachedRippleResult(stackId, lastChange)

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!sectionNumber.trim()) return
    const change = {
      clause_section: sectionNumber.trim(),
      current_text: currentText.trim(),
      proposed_text: proposedText.trim(),
      change_type: 'modify',
      description: sectionNumber.trim() === '9.2'
        ? 'Extended data retention from 15 to 25 years to align with emerging cardiology registry standards and sponsor long-term safety monitoring requirements.'
        : `Modify section ${sectionNumber.trim()}`,
    }
    setLastChange(change)
    await rippleMutation.mutateAsync(change)
  }

  // Use cached result (persists across tab switches) or fresh mutation result
  const result = (cachedResult.data || rippleMutation.data) as any

  const hopLabels: Record<number, { label: string; description: string }> = {
    1: { label: 'Direct Impact', description: 'Immediately affected clauses' },
    2: { label: 'Indirect Impact', description: 'Secondary dependencies' },
    3: { label: 'Cascade Effect', description: 'Tertiary ripple effects' },
  }

  const groupedImpacts: Record<number, any[]> = {}
  if (result?.impacts_by_hop) {
    // Backend returns { "hop_1": [...], "hop_2": [...] } keyed by hop distance
    Object.entries(result.impacts_by_hop).forEach(([hopKey, impacts]: [string, any]) => {
      // Parse "hop_1" -> 1, or plain "1" -> 1
      const hopNum = hopKey.includes('_') ? Number(hopKey.split('_').pop()) : Number(hopKey)
      if (!isNaN(hopNum) && Array.isArray(impacts) && impacts.length > 0) {
        groupedImpacts[hopNum] = impacts
      }
    })
  } else if (result?.impacts) {
    // Fallback: flat array with hop_level/hop per impact
    result.impacts.forEach((impact: any) => {
      const hop = impact.hop_level || impact.hop_distance || impact.hop || 1
      if (!groupedImpacts[hop]) groupedImpacts[hop] = []
      groupedImpacts[hop].push(impact)
    })
  }

  // Flatten recommendations from {critical_actions, recommended_actions, optional_actions} to a display list
  const flatRecommendations: any[] = []
  if (result?.recommendations && typeof result.recommendations === 'object' && !Array.isArray(result.recommendations)) {
    const rec = result.recommendations
    ;(rec.critical_actions || []).forEach((a: any) => flatRecommendations.push({ ...a, tier: 'critical' }))
    ;(rec.recommended_actions || []).forEach((a: any) => flatRecommendations.push({ ...a, tier: 'recommended' }))
    ;(rec.optional_actions || []).forEach((a: any) => flatRecommendations.push({ ...a, tier: 'optional' }))
  } else if (Array.isArray(result?.recommendations)) {
    flatRecommendations.push(...result.recommendations)
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
              onDoubleClick={() => { if (!sectionNumber) setSectionNumber('9.2') }}
              placeholder="9.2"
              className="w-full px-4 py-3 bg-apple-offwhite rounded-xl border border-black/[0.04] text-[15px] text-apple-black placeholder:text-apple-light focus:outline-none focus:border-apple-dark/20 focus:bg-white focus:shadow-[0_0_0_3px_rgba(0,0,0,0.04)] transition-all duration-300"
            />
          </div>
          <div>
            <label className="block text-[13px] font-medium text-apple-dark2 mb-2">Current Text</label>
            <textarea
              value={currentText}
              onChange={(e) => setCurrentText(e.target.value)}
              onDoubleClick={() => { if (!currentText) setCurrentText('Site shall retain all study records for a period of 15 years from the date of study completion or early termination.') }}
              placeholder="e.g. Site shall retain all study records for a period of 15 years..."
              rows={3}
              className="w-full px-4 py-3 bg-apple-offwhite rounded-xl border border-black/[0.04] text-[15px] text-apple-black placeholder:text-apple-light focus:outline-none focus:border-apple-dark/20 focus:bg-white focus:shadow-[0_0_0_3px_rgba(0,0,0,0.04)] transition-all duration-300 resize-none"
            />
          </div>
          <div>
            <label className="block text-[13px] font-medium text-apple-dark2 mb-2">Proposed Text</label>
            <textarea
              value={proposedText}
              onChange={(e) => setProposedText(e.target.value)}
              onDoubleClick={() => { if (!proposedText) setProposedText('Site shall retain all study records for a period of 25 years from the date of study completion or early termination.') }}
              placeholder="e.g. Site shall retain all study records for a period of 25 years..."
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
                        <div className="flex items-center gap-2 flex-wrap">
                          {(impact.affected_section || impact.section) && (
                            <span className="px-2.5 py-1 bg-apple-bg rounded-lg text-[13px] font-mono font-medium text-apple-dark2">
                              {impact.affected_section || impact.section}
                            </span>
                          )}
                          {impact.affected_section_title && (
                            <span className="text-[13px] font-medium text-apple-dark2">
                              {impact.affected_section_title}
                            </span>
                          )}
                          {(impact.impact_type || impact.type) && (
                            <span className="px-2.5 py-1 bg-apple-silver/50 rounded-lg text-[12px] font-medium text-apple-gray2 capitalize">
                              {(impact.impact_type || impact.type).replace(/_/g, ' ')}
                            </span>
                          )}
                        </div>
                        {impact.severity && (
                          <span className={`text-[11px] font-bold uppercase tracking-wider ${
                            impact.severity === 'critical' ? 'text-red-600' :
                            impact.severity === 'high' ? 'text-apple-dark' : 'text-apple-gray'
                          }`}>
                            {impact.severity}
                          </span>
                        )}
                      </div>
                      {impact.cascade_path && (
                        <p className="text-[12px] text-blue-600/80 font-mono mb-1.5">
                          {impact.cascade_path}
                        </p>
                      )}
                      <p className="text-[14px] text-apple-dark leading-relaxed">
                        {impact.description || impact.explanation}
                      </p>
                      {impact.required_action && (
                        <p className="text-[13px] text-apple-gray2 mt-2 italic">
                          Action: {impact.required_action}
                        </p>
                      )}
                    </motion.div>
                  ))}
                </div>
              </div>
            )
          })}

          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-apple-offwhite rounded-3xl border border-black/[0.03] p-7"
          >
            {flatRecommendations.length > 0 && (
              <div>
                <p className="text-[13px] font-semibold text-apple-dark mb-3">Recommendations</p>
                <div className="space-y-2.5">
                  {flatRecommendations.map((rec: any, k: number) => {
                    const tierColors: Record<string, string> = {
                      critical: 'bg-red-100 text-red-700',
                      recommended: 'bg-amber-100 text-amber-700',
                      optional: 'bg-blue-100 text-blue-700',
                    }
                    return (
                      <div key={k} className="flex gap-3 items-start">
                        <div className="w-5 h-5 rounded-full bg-apple-bg flex items-center justify-center flex-shrink-0 mt-0.5">
                          <span className="text-[10px] font-bold text-apple-dark2">{rec.priority || k + 1}</span>
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-0.5">
                            {rec.tier && (
                              <span className={`px-1.5 py-0.5 text-[9px] font-bold rounded uppercase ${tierColors[rec.tier] || 'bg-apple-bg text-apple-dark2'}`}>
                                {rec.tier}
                              </span>
                            )}
                            {rec.related_sections?.length > 0 && (
                              <span className="text-[11px] text-apple-gray font-mono">
                                {rec.related_sections.join(', ')}
                              </span>
                            )}
                          </div>
                          <p className="text-[14px] text-apple-dark2 leading-relaxed">
                            {typeof rec === 'string' ? rec : rec.action || rec.text || rec.recommendation || JSON.stringify(rec)}
                          </p>
                          {rec.reason && (
                            <p className="text-[12px] text-apple-gray mt-0.5">{rec.reason}</p>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </motion.div>
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


// ── Consolidate Tab — Microsoft Word Editor (TipTap) ─────────

function RibbonBtn({ icon: Icon, label, active, onClick, disabled, size = 'sm' }: {
  icon: any; label: string; active?: boolean; onClick?: () => void; disabled?: boolean; size?: 'sm' | 'lg'
}) {
  if (size === 'lg') {
    return (
      <button
        title={label}
        onClick={onClick}
        disabled={disabled}
        className={`flex flex-col items-center justify-center px-[6px] py-[3px] rounded-[3px] transition-colors gap-[2px] min-w-[44px]
          ${disabled ? 'opacity-40 cursor-not-allowed' : ''}
          ${active
            ? 'bg-[#c6ddf7] border border-[#8cb8e8]'
            : 'hover:bg-[#d6e8f8] hover:border hover:border-[#b0cde8] border border-transparent'
          }`}
      >
        <Icon className="w-[20px] h-[20px] text-[#333]" />
        <span className="text-[9px] text-[#444] leading-tight">{label}</span>
      </button>
    )
  }
  return (
    <button
      title={label}
      onClick={onClick}
      disabled={disabled}
      className={`flex items-center justify-center w-[24px] h-[22px] rounded-[2px] transition-colors
        ${disabled ? 'opacity-40 cursor-not-allowed' : ''}
        ${active
          ? 'bg-[#c6ddf7] border border-[#8cb8e8]'
          : 'hover:bg-[#d6e8f8] hover:border hover:border-[#b0cde8] border border-transparent'
        }`}
    >
      <Icon className="w-[13px] h-[13px] text-[#333]" />
    </button>
  )
}

function RibbonGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col border-r border-[#d4d4d4] pr-[6px] mr-[6px] last:border-r-0 last:pr-0 last:mr-0">
      <div className="flex items-center gap-[2px] py-[3px] min-h-[52px]">
        {children}
      </div>
      <div className="text-[9px] text-[#666] text-center leading-none pb-[3px] select-none">{label}</div>
    </div>
  )
}

function WordHomeRibbon({ editor }: { editor: ReturnType<typeof useEditor> }) {
  if (!editor) return null

  return (
    <div className="flex items-stretch px-[6px] border-b border-[#d4d4d4]" style={{ background: '#f3f3f3' }}>
      <RibbonGroup label="Clipboard">
        <RibbonBtn icon={ClipboardPaste} label="Paste" size="lg" onClick={() => {}} />
        <div className="flex flex-col gap-[1px]">
          <RibbonBtn icon={Scissors} label="Cut" onClick={() => editor.chain().focus().run()} />
          <RibbonBtn icon={Clipboard} label="Copy" onClick={() => editor.chain().focus().run()} />
          <RibbonBtn icon={Paintbrush} label="Format Painter" onClick={() => {}} />
        </div>
      </RibbonGroup>

      <RibbonGroup label="Font">
        <div className="flex flex-col gap-[3px]">
          <div className="flex items-center gap-[2px]">
            <select className="h-[22px] px-1 text-[11px] border border-[#bbb] bg-white text-[#333] w-[110px] focus:outline-none focus:border-[#5b9bd5] rounded-[2px]">
              {['Calibri', 'Arial', 'Times New Roman', 'Cambria', 'Georgia', 'Verdana', 'Courier New'].map(f => (
                <option key={f} value={f} style={{ fontFamily: f }}>{f}</option>
              ))}
            </select>
            <select className="h-[22px] px-1 text-[11px] border border-[#bbb] bg-white text-[#333] w-[38px] focus:outline-none focus:border-[#5b9bd5] rounded-[2px]">
              {['8', '9', '10', '11', '12', '14', '16', '18', '20', '24', '28', '36', '48', '72'].map(s => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
            <RibbonBtn icon={ChevronUp} label="Grow Font" onClick={() => {}} />
            <RibbonBtn icon={ChevronDown} label="Shrink Font" onClick={() => {}} />
          </div>
          <div className="flex items-center gap-[1px]">
            <RibbonBtn icon={Bold} label="Bold (Ctrl+B)" active={editor.isActive('bold')} onClick={() => editor.chain().focus().toggleBold().run()} />
            <RibbonBtn icon={Italic} label="Italic (Ctrl+I)" active={editor.isActive('italic')} onClick={() => editor.chain().focus().toggleItalic().run()} />
            <RibbonBtn icon={Underline} label="Underline (Ctrl+U)" active={editor.isActive('underline')} onClick={() => editor.chain().focus().toggleUnderline().run()} />
            <RibbonBtn icon={Strikethrough} label="Strikethrough" active={editor.isActive('strike')} onClick={() => editor.chain().focus().toggleStrike().run()} />
            <RibbonBtn icon={Subscript} label="Subscript" active={editor.isActive('subscript')} onClick={() => editor.chain().focus().toggleSubscript().run()} />
            <RibbonBtn icon={Superscript} label="Superscript" active={editor.isActive('superscript')} onClick={() => editor.chain().focus().toggleSuperscript().run()} />
            <div className="w-px h-[16px] bg-[#d4d4d4] mx-[2px]" />
            <RibbonBtn icon={Highlighter} label="Text Highlight Color" active={editor.isActive('highlight')} onClick={() => editor.chain().focus().toggleHighlight().run()} />
            <RibbonBtn icon={Type} label="Font Color" onClick={() => {}} />
          </div>
        </div>
      </RibbonGroup>

      <RibbonGroup label="Paragraph">
        <div className="flex flex-col gap-[3px]">
          <div className="flex items-center gap-[1px]">
            <RibbonBtn icon={List} label="Bullets" active={editor.isActive('bulletList')} onClick={() => editor.chain().focus().toggleBulletList().run()} />
            <RibbonBtn icon={ListOrdered} label="Numbering" active={editor.isActive('orderedList')} onClick={() => editor.chain().focus().toggleOrderedList().run()} />
            <div className="w-px h-[16px] bg-[#d4d4d4] mx-[2px]" />
            <RibbonBtn icon={Outdent} label="Decrease Indent" onClick={() => editor.chain().focus().liftListItem('listItem').run()} />
            <RibbonBtn icon={Indent} label="Increase Indent" onClick={() => editor.chain().focus().sinkListItem('listItem').run()} />
          </div>
          <div className="flex items-center gap-[1px]">
            <RibbonBtn icon={AlignLeft} label="Align Left" active={editor.isActive({ textAlign: 'left' })} onClick={() => editor.chain().focus().setTextAlign('left').run()} />
            <RibbonBtn icon={AlignCenter} label="Center" active={editor.isActive({ textAlign: 'center' })} onClick={() => editor.chain().focus().setTextAlign('center').run()} />
            <RibbonBtn icon={AlignRight} label="Align Right" active={editor.isActive({ textAlign: 'right' })} onClick={() => editor.chain().focus().setTextAlign('right').run()} />
            <RibbonBtn icon={AlignJustify} label="Justify" active={editor.isActive({ textAlign: 'justify' })} onClick={() => editor.chain().focus().setTextAlign('justify').run()} />
          </div>
        </div>
      </RibbonGroup>

      <RibbonGroup label="Editing">
        <div className="flex flex-col gap-[1px]">
          <RibbonBtn icon={Undo2} label="Undo" onClick={() => editor.chain().focus().undo().run()} disabled={!editor.can().undo()} />
          <RibbonBtn icon={Redo2} label="Redo" onClick={() => editor.chain().focus().redo().run()} disabled={!editor.can().redo()} />
          <RibbonBtn icon={Search} label="Find" onClick={() => {}} />
        </div>
      </RibbonGroup>
    </div>
  )
}

function ConsolidateTab({ stackId }: { stackId: string }) {
  const { data, isLoading, isError, error } = useConsolidatedContract(stackId)
  const [activeRibbonTab, setActiveRibbonTab] = useState(1)
  const [showFileMenu, setShowFileMenu] = useState(false)
  const [showSaveAsSubmenu, setShowSaveAsSubmenu] = useState(false)
  const fileMenuRef = useRef<HTMLDivElement>(null)

  const flattenSections = (sections: ConsolidatedSection[], depth: number = 0): Array<{ section: ConsolidatedSection; depth: number }> => {
    const result: Array<{ section: ConsolidatedSection; depth: number }> = []
    for (const s of sections) {
      result.push({ section: s, depth })
      if (s.subsections?.length) {
        result.push(...flattenSections(s.subsections, depth + 1))
      }
    }
    return result
  }

  const buildDocumentHtml = (sections: ConsolidatedSection[]): string => {
    const flat = flattenSections(sections)
    let html = ''
    for (const { section, depth } of flat) {
      const isAmended = section.is_amended
      const headingLevel = section.level === 1 ? 'h2' : section.level === 2 ? 'h3' : 'h4'
      const amendedLabel = isAmended ? ' <strong>AMENDED</strong>' : ''

      html += `<${headingLevel}>${section.section_number}. ${section.section_title}${amendedLabel}</${headingLevel}>`

      if (section.content) {
        const escaped = section.content.replace(/</g, '&lt;').replace(/>/g, '&gt;')
        if (isAmended) {
          html += `<p><mark>${escaped}</mark></p>`
        } else {
          html += `<p>${escaped}</p>`
        }
      }

      if (isAmended && section.amendment_source) {
        let annotation = `Modified by ${section.amendment_source}`
        if (section.amendment_description) {
          annotation += ` — ${section.amendment_description}`
        }
        html += `<p><em>${annotation}</em></p>`
      }

      if (section.conflicts?.length) {
        for (const c of section.conflicts) {
          html += `<blockquote><strong>${c.severity}</strong>: ${c.description}</blockquote>`
        }
      }
    }
    return html
  }

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (fileMenuRef.current && !fileMenuRef.current.contains(e.target as Node)) {
        setShowFileMenu(false)
        setShowSaveAsSubmenu(false)
      }
    }
    if (showFileMenu) {
      document.addEventListener('mousedown', handleClickOutside)
    }
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [showFileMenu])

  const sanitizeText = (text: string): string => {
    let clean = text
    clean = clean.replace(/<[^>]*>/g, '')
    clean = clean.replace(/\s*AMENDED\s*/gi, '')
    clean = clean.replace(/\[AMENDED\]/gi, '')
    clean = clean.replace(/&amp;/g, '&')
    clean = clean.replace(/&lt;/g, '<')
    clean = clean.replace(/&gt;/g, '>')
    clean = clean.replace(/&nbsp;/g, ' ')
    clean = clean.replace(/&quot;/g, '"')
    clean = clean.replace(/\s{2,}/g, ' ')
    return clean.trim()
  }

  const exportAsDocx = async () => {
    if (!data) return
    setShowFileMenu(false)
    setShowSaveAsSubmenu(false)

    const flat = flattenSections(data.document_structure)
    const children: Paragraph[] = []

    for (const { section } of flat) {
      const level = section.level === 1 ? HeadingLevel.HEADING_2 : section.level === 2 ? HeadingLevel.HEADING_3 : HeadingLevel.HEADING_4
      const cleanTitle = sanitizeText(`${section.section_number}. ${section.section_title}`)

      children.push(new Paragraph({
        heading: level,
        children: [
          new TextRun({
            text: cleanTitle,
            bold: true,
          }),
        ],
      }))

      if (section.content) {
        const cleanContent = sanitizeText(section.content)
        const lines = cleanContent.split('\n')
        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed) continue

          const isBullet = trimmed.startsWith('•') || trimmed.startsWith('-') || trimmed.startsWith('*')
          const bulletText = isBullet ? trimmed.replace(/^[•\-*]\s*/, '') : trimmed

          children.push(new Paragraph({
            bullet: isBullet ? { level: 0 } : undefined,
            children: [
              new TextRun({
                text: bulletText,
                size: 22,
                font: 'Calibri',
              }),
            ],
            spacing: { after: 120 },
          }))
        }
      }
    }

    const doc = new DocxDocument({
      sections: [{
        properties: {
          page: {
            margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
          },
        },
        children,
      }],
    })

    const blob = await Packer.toBlob(doc)
    saveAs(blob, 'Consolidated_Contract.docx')
  }

  const exportAsPdf = async () => {
    if (!data) return
    setShowFileMenu(false)
    setShowSaveAsSubmenu(false)

    const flat = flattenSections(data.document_structure)
    const pdf = new jsPDF({
      orientation: 'portrait',
      unit: 'pt',
      format: 'letter',
    })

    const pageWidth = pdf.internal.pageSize.getWidth()
    const pageHeight = pdf.internal.pageSize.getHeight()
    const marginLeft = 72
    const marginRight = 72
    const marginTop = 72
    const marginBottom = 72
    const maxWidth = pageWidth - marginLeft - marginRight
    let y = marginTop

    const addPage = () => {
      pdf.addPage()
      y = marginTop
    }

    const checkSpace = (needed: number) => {
      if (y + needed > pageHeight - marginBottom) {
        addPage()
      }
    }

    for (const { section } of flat) {
      const headingSize = section.level === 1 ? 14 : section.level === 2 ? 12 : 11
      const cleanTitle = sanitizeText(`${section.section_number}. ${section.section_title}`)

      checkSpace(headingSize + 20)
      y += 8
      pdf.setFont('helvetica', 'bold')
      pdf.setFontSize(headingSize)
      const headingLines = pdf.splitTextToSize(cleanTitle, maxWidth)
      pdf.text(headingLines, marginLeft, y)
      y += headingLines.length * (headingSize + 2) + 6

      if (section.content) {
        const cleanContent = sanitizeText(section.content)
        pdf.setFont('helvetica', 'normal')
        pdf.setFontSize(10)
        const contentLines = pdf.splitTextToSize(cleanContent, maxWidth)

        for (const line of contentLines) {
          checkSpace(14)
          pdf.text(line, marginLeft, y)
          y += 13
        }
        y += 4
      }
    }

    pdf.save('Consolidated_Contract.pdf')
  }

  const docHtml = data ? buildDocumentHtml(data.document_structure) : ''

  const editorContainerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLDivElement>(null)
  const [pageCount, setPageCount] = useState(1)
  const [currentPage, setCurrentPage] = useState(1)

  const PAGE_HEIGHT = 1056
  const PAGE_PADDING_TOP = 72
  const PAGE_PADDING_BOTTOM = 72
  const PAGE_GAP = 20

  const editor = useEditor({
    extensions: [
      StarterKit.configure({
        heading: { levels: [1, 2, 3, 4] },
      }),
      TextAlign.configure({ types: ['heading', 'paragraph'] }),
      TiptapSubscript,
      TiptapSuperscript,
      TiptapHighlight.configure({ multicolor: false }),
      PageBreakNode,
    ],
    content: docHtml,
  }, [docHtml])

  const PAGE_CONTENT_HEIGHT = PAGE_HEIGHT - PAGE_PADDING_TOP - PAGE_PADDING_BOTTOM
  const paginatedRef = useRef(false)

  const insertPageBreaks = useCallback(() => {
    if (!editor || !editorContainerRef.current) return
    if (paginatedRef.current) return
    paginatedRef.current = true

    try {
      const editorEl = editorContainerRef.current.querySelector('.ProseMirror') as HTMLElement | null
      if (!editorEl) return

      const doc = editor.state.doc
      const breakPositions: number[] = []
      let currentPageUsed = 0
      doc.forEach((node, offset) => {
        if (node.type.name === 'pageBreak') return
        const absPos = offset + 1
        const dom = editor.view.nodeDOM(absPos) as HTMLElement | null
        if (!dom || !(dom instanceof HTMLElement)) return
        const height = dom.getBoundingClientRect().height
        if (height <= 0) return

        if (currentPageUsed > 0 && currentPageUsed + height > PAGE_CONTENT_HEIGHT) {
          breakPositions.push(absPos)
          currentPageUsed = height
        } else {
          currentPageUsed += height
        }
      })

      if (breakPositions.length > 0) {
        let tr = editor.state.tr
        let insertionOffset = 0
        for (const pos of breakPositions) {
          const adjustedPos = pos + insertionOffset
          const pageBreakNodeType = editor.schema.nodes.pageBreak
          tr = tr.insert(adjustedPos, pageBreakNodeType.create())
          insertionOffset += 1
        }
        tr.setMeta('addToHistory', false)
        editor.view.dispatch(tr)
      }
      setPageCount(breakPositions.length + 1)
    } catch (_) {
    }
  }, [editor])

  useEffect(() => {
    paginatedRef.current = false
    const timers = [
      setTimeout(insertPageBreaks, 600),
      setTimeout(insertPageBreaks, 1500),
    ]
    return () => {
      timers.forEach(t => clearTimeout(t))
    }
  }, [docHtml, insertPageBreaks])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const handleScroll = () => {
      if (!editorContainerRef.current) return
      const breakNodes = editorContainerRef.current.querySelectorAll('.word-page-break-node')
      const scrollTop = canvas.scrollTop
      const containerTop = editorContainerRef.current.offsetTop
      let page = 1
      breakNodes.forEach(node => {
        const nodeEl = node as HTMLElement
        if (scrollTop > nodeEl.offsetTop - containerTop) page++
      })
      setCurrentPage(Math.min(page, pageCount))
    }
    canvas.addEventListener('scroll', handleScroll, { passive: true })
    return () => canvas.removeEventListener('scroll', handleScroll)
  }, [pageCount])

  if (isLoading) {
    return (
      <motion.div {...fadeUp} className="flex items-center justify-center py-24">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="w-8 h-8 text-[#5b9bd5] animate-spin" />
          <p className="text-[13px] text-[#666] font-[Calibri,sans-serif]">Assembling consolidated contract...</p>
        </div>
      </motion.div>
    )
  }

  if (isError) {
    return (
      <motion.div {...fadeUp} className="flex items-center gap-3 mx-auto max-w-2xl mt-12 px-6 py-4 bg-[#fff5f5] border border-[#e0c8c8] rounded">
        <AlertTriangle className="w-5 h-5 text-[#c00] flex-shrink-0" />
        <p className="text-[13px] text-[#c00] font-[Calibri,sans-serif]">
          {(error as Error)?.message || 'Failed to load consolidated contract.'}
        </p>
      </motion.div>
    )
  }

  if (!data) return null

  const { metadata } = data

  return (
    <motion.div {...fadeUp} className="flex flex-col relative" style={{ margin: '-8px -16px 0', fontFamily: 'Segoe UI, Calibri, sans-serif' }}>

      {/* ── Title bar (Word blue) ───── */}
      <div className="flex items-center h-[30px] px-3 gap-2" style={{ background: '#2b579a' }}>
        <FileText className="w-[14px] h-[14px] text-white" />
        <span className="text-[12px] text-white font-medium tracking-wide">
          Consolidated_Contract.docx
        </span>
        <span className="text-[10px] text-white/50 ml-1">— Digital Contract Platform</span>
        <div className="flex-1" />
        <span className="text-[10px] text-white/60">
          {metadata.amended_sections} of {metadata.total_sections} sections amended
        </span>
        <div className="flex items-center gap-[6px] ml-3">
          <div className="w-[11px] h-[11px] rounded-sm bg-white/20 flex items-center justify-center cursor-pointer hover:bg-white/30">
            <Minus className="w-[8px] h-[8px] text-white" />
          </div>
          <div className="w-[11px] h-[11px] rounded-sm bg-white/20 flex items-center justify-center cursor-pointer hover:bg-white/30">
            <div className="w-[6px] h-[6px] border border-white" />
          </div>
          <div className="w-[11px] h-[11px] rounded-sm bg-white/20 flex items-center justify-center cursor-pointer hover:bg-[#e81123]/80">
            <X className="w-[8px] h-[8px] text-white" />
          </div>
        </div>
      </div>

      {/* ── Ribbon tabs ── */}
      <div className="flex items-end h-[28px] px-[2px] gap-0 relative" style={{ background: '#2b579a' }}>
        {['File', 'Home', 'Insert', 'Design', 'Layout', 'References', 'Review', 'View'].map((tab, i) => (
          <button
            key={tab}
            onClick={() => {
              if (i === 0) {
                setShowFileMenu(!showFileMenu)
                setShowSaveAsSubmenu(false)
              } else {
                setActiveRibbonTab(i)
                setShowFileMenu(false)
                setShowSaveAsSubmenu(false)
              }
            }}
            className={`px-[12px] text-[11.5px] tracking-[0.2px] border-0 outline-none cursor-pointer ${
              i === 0
                ? showFileMenu
                  ? 'text-white font-medium py-[5px] bg-[#1a4a82]'
                  : 'text-white font-medium py-[5px]'
                : i === activeRibbonTab
                  ? 'text-[#333] bg-[#f3f3f3] py-[5px] rounded-t-[2px]'
                  : 'text-white/80 hover:text-white hover:bg-white/10 py-[5px]'
            }`}
            style={{ lineHeight: '1' }}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* ── File Menu Dropdown ── */}
      {showFileMenu && (
        <div
          ref={fileMenuRef}
          className="absolute z-50 bg-white border border-[#c8c8c8] shadow-lg"
          style={{ top: '58px', left: '2px', width: '220px', fontFamily: 'Segoe UI, sans-serif' }}
        >
          <div className="py-[2px]">
            <button
              className="w-full text-left px-4 py-[6px] text-[12px] text-[#333] hover:bg-[#e8e8e8] flex items-center gap-3"
              onClick={() => { setShowFileMenu(false) }}
            >
              <Save className="w-[14px] h-[14px] text-[#666]" />
              Save
            </button>

            <div
              className="relative"
              onMouseEnter={() => setShowSaveAsSubmenu(true)}
              onMouseLeave={() => setShowSaveAsSubmenu(false)}
            >
              <button
                className="w-full text-left px-4 py-[6px] text-[12px] text-[#333] hover:bg-[#e8e8e8] flex items-center gap-3 justify-between"
              >
                <span className="flex items-center gap-3">
                  <Download className="w-[14px] h-[14px] text-[#666]" />
                  Save As...
                </span>
                <ChevronRight className="w-[10px] h-[10px] text-[#999]" />
              </button>

              {showSaveAsSubmenu && (
                <div
                  className="absolute left-[218px] top-[-2px] bg-white border border-[#c8c8c8] shadow-lg z-50"
                  style={{ width: '200px' }}
                >
                  <div className="py-[2px]">
                    <div className="px-4 py-[6px] text-[10px] text-[#888] uppercase tracking-wide">
                      Download Clean Copy
                    </div>
                    <button
                      className="w-full text-left px-4 py-[7px] text-[12px] text-[#333] hover:bg-[#e8e8e8] flex items-center gap-3"
                      onClick={exportAsDocx}
                    >
                      <FileText className="w-[14px] h-[14px] text-[#2b579a]" />
                      Word Document (.docx)
                    </button>
                    <button
                      className="w-full text-left px-4 py-[7px] text-[12px] text-[#333] hover:bg-[#e8e8e8] flex items-center gap-3"
                      onClick={exportAsPdf}
                    >
                      <FileText className="w-[14px] h-[14px] text-[#c00]" />
                      PDF Document (.pdf)
                    </button>
                  </div>
                </div>
              )}
            </div>

            <div className="my-[2px] border-t border-[#e0e0e0] mx-2" />

            <button
              className="w-full text-left px-4 py-[6px] text-[12px] text-[#333] hover:bg-[#e8e8e8] flex items-center gap-3"
              onClick={() => setShowFileMenu(false)}
            >
              <X className="w-[14px] h-[14px] text-[#666]" />
              Close
            </button>
          </div>
        </div>
      )}

      {/* ── Ribbon content (Home tab) ── */}
      {activeRibbonTab === 1 && <WordHomeRibbon editor={editor} />}

      {/* ── Ruler ─────────────────────── */}
      <div className="h-[18px] border-b border-[#d4d4d4] flex items-end px-0 overflow-hidden" style={{ background: '#f0f0f0' }}>
        <div className="relative w-full h-[14px]" style={{ marginLeft: '96px', marginRight: '96px' }}>
          <div className="absolute inset-0 bg-white border-x border-[#c0c0c0]" />
          <div className="absolute inset-0 flex items-end">
            {Array.from({ length: 17 }, (_, i) => (
              <div key={i} className="flex-1 relative">
                <div className="absolute bottom-0 left-0 w-px h-[5px] bg-[#888]" />
                {i < 16 && (
                  <span className="absolute bottom-[5px] left-[2px] text-[7px] text-[#777] leading-none select-none">
                    {i + 1}
                  </span>
                )}
                <div className="absolute bottom-0 left-1/2 w-px h-[3px] bg-[#aaa]" />
              </div>
            ))}
          </div>
          <div className="absolute top-0 left-[0px]">
            <div className="w-0 h-0 border-l-[4px] border-r-[4px] border-t-[4px] border-l-transparent border-r-transparent border-t-[#555]" />
          </div>
          <div className="absolute bottom-0 left-[0px]">
            <div className="w-0 h-0 border-l-[4px] border-r-[4px] border-b-[4px] border-l-transparent border-r-transparent border-b-[#555]" />
          </div>
        </div>
      </div>

      {/* ── Document canvas with visual pages ── */}
      <div
        ref={canvasRef}
        className="flex-1 overflow-auto"
        style={{ background: '#a0a0a0', minHeight: '600px', maxHeight: 'calc(100vh - 320px)' }}
      >
        <div className="flex flex-col items-center py-4">
          <div
            ref={editorContainerRef}
            className="word-pages-container flex-shrink-0"
            style={{ width: '816px' }}
          >
            <EditorContent editor={editor} />
          </div>
        </div>
      </div>

      {/* ── Status bar (Word blue) ───── */}
      <div className="flex items-center h-[22px] px-3 border-t border-[#1a4a82]" style={{ background: '#2b579a' }}>
        <span className="text-[10px] text-white/80">
          Page {currentPage} of {pageCount}
        </span>
        <span className="text-[10px] text-white/40 mx-2">|</span>
        <span className="text-[10px] text-white/80">
          {metadata.total_sections} Sections
        </span>
        <span className="text-[10px] text-white/40 mx-2">|</span>
        <span className="text-[10px] text-white/80">
          {metadata.amended_sections} Amended
        </span>
        <div className="flex-1" />
        {metadata.appendices?.length > 0 && (
          <span className="text-[10px] text-white/60 mr-3">
            Appendices: {metadata.appendices.join(', ')}
          </span>
        )}
        <span className="text-[10px] text-white/50">English (US)</span>
        <div className="flex items-center gap-1 ml-3">
          {[100, 120, 150].map(z => (
            <button key={z} className={`text-[9px] px-1 rounded ${z === 100 ? 'text-white bg-white/20' : 'text-white/40 hover:text-white/70'}`}>
              {z}%
            </button>
          ))}
        </div>
      </div>

    </motion.div>
  )
}
