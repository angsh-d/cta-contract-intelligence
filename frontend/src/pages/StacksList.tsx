import { useState } from 'react'
import { Link } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Plus,
  Shield,
  Search,
  X,
  ChevronRight,
  FileStack,
  Building2,
  MapPin,
} from 'lucide-react'
import { useStacks, useCreateStack } from '../hooks/useApi'

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.04 } },
}
const item = {
  hidden: { opacity: 0, y: 8 },
  show: { opacity: 1, y: 0, transition: { duration: 0.3 } },
}

export default function StacksList() {
  const { data, isLoading } = useStacks()
  const createMutation = useCreateStack()
  const [showCreate, setShowCreate] = useState(false)
  const [search, setSearch] = useState('')
  const [form, setForm] = useState({ name: '', sponsor_name: '', site_name: '', study_protocol: '', therapeutic_area: '' })

  const stacks = (data?.stacks || []).filter(
    (s) => !search || s.name.toLowerCase().includes(search.toLowerCase()) || s.sponsor_name.toLowerCase().includes(search.toLowerCase())
  )

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault()
    await createMutation.mutateAsync(form)
    setForm({ name: '', sponsor_name: '', site_name: '', study_protocol: '', therapeutic_area: '' })
    setShowCreate(false)
  }

  return (
    <div className="max-w-6xl mx-auto px-6 lg:px-10 py-10">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-[28px] font-semibold tracking-tight text-apple-black">Contracts</h1>
          <p className="text-[14px] text-apple-gray mt-0.5">Manage your clinical trial agreement stacks</p>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="flex items-center gap-2 px-4 py-2.5 bg-apple-black text-white text-[13px] font-medium rounded-full hover:bg-apple-dark transition-colors"
        >
          <Plus className="w-4 h-4" />
          New Stack
        </button>
      </div>

      <div className="relative mb-6">
        <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-apple-gray" />
        <input
          type="text"
          placeholder="Search contracts..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full pl-11 pr-4 py-3 bg-white rounded-xl border border-black/[0.04] text-[14px] text-apple-black placeholder:text-apple-light focus:outline-none focus:ring-2 focus:ring-apple-dark/20 focus:border-apple-dark/30 transition-all"
        />
      </div>

      {isLoading && (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-20 skeleton rounded-2xl" />
          ))}
        </div>
      )}

      {!isLoading && stacks.length === 0 && (
        <div className="text-center py-20">
          <div className="w-16 h-16 rounded-2xl bg-apple-silver/40 flex items-center justify-center mx-auto mb-4">
            <FileStack className="w-8 h-8 text-apple-gray" />
          </div>
          <h3 className="text-[17px] font-semibold text-apple-black mb-1">
            {search ? 'No matches found' : 'No contracts yet'}
          </h3>
          <p className="text-[14px] text-apple-gray">
            {search ? 'Try a different search term' : 'Create your first contract stack to begin'}
          </p>
        </div>
      )}

      <motion.div variants={container} initial="hidden" animate="show" className="space-y-2">
        {stacks.map((stack) => (
          <motion.div key={stack.id} variants={item}>
            <Link
              to={`/stacks/${stack.id}`}
              className="flex items-center justify-between px-5 py-4 bg-white rounded-2xl border border-black/[0.04] hover:border-apple-light hover:shadow-sm transition-all duration-200 group"
            >
              <div className="flex items-center gap-4 min-w-0">
                <div className="w-11 h-11 rounded-xl bg-apple-bg flex items-center justify-center flex-shrink-0">
                  <Shield className="w-5 h-5 text-apple-dark" />
                </div>
                <div className="min-w-0">
                  <p className="text-[14px] font-medium text-apple-black truncate">{stack.name}</p>
                  <div className="flex items-center gap-3 mt-0.5">
                    <span className="flex items-center gap-1 text-[12px] text-apple-gray">
                      <Building2 className="w-3 h-3" /> {stack.sponsor_name}
                    </span>
                    <span className="flex items-center gap-1 text-[12px] text-apple-gray">
                      <MapPin className="w-3 h-3" /> {stack.site_name}
                    </span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <StatusPill status={stack.processing_status} />
                <ChevronRight className="w-4 h-4 text-apple-light group-hover:text-apple-gray transition-colors" />
              </div>
            </Link>
          </motion.div>
        ))}
      </motion.div>

      <AnimatePresence>
        {showCreate && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/30 backdrop-blur-sm z-50"
              onClick={() => setShowCreate(false)}
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              transition={{ type: 'spring', damping: 25, stiffness: 300 }}
              className="fixed inset-x-4 top-[10%] sm:inset-auto sm:left-1/2 sm:top-1/2 sm:-translate-x-1/2 sm:-translate-y-1/2 w-auto sm:w-[480px] bg-white/95 backdrop-blur-xl rounded-2xl shadow-2xl z-50 border border-black/[0.04]"
            >
              <div className="flex items-center justify-between px-6 py-4 border-b border-black/[0.04]">
                <h2 className="text-[17px] font-semibold text-apple-black">New Contract Stack</h2>
                <button onClick={() => setShowCreate(false)} className="p-1.5 rounded-lg hover:bg-apple-silver/50 transition-colors">
                  <X className="w-5 h-5 text-apple-gray" />
                </button>
              </div>
              <form onSubmit={handleCreate} className="p-6 space-y-4">
                <FormField label="Study Name (required)" value={form.name} onChange={(v) => setForm({ ...form, name: v })} placeholder="e.g. HEARTBEAT-3" required />
                <FormField label="Sponsor (required)" value={form.sponsor_name} onChange={(v) => setForm({ ...form, sponsor_name: v })} placeholder="e.g. CardioVita Therapeutics" required />
                <FormField label="Site (required)" value={form.site_name} onChange={(v) => setForm({ ...form, site_name: v })} placeholder="e.g. Memorial Medical Center" required />
                <FormField label="Protocol" value={form.study_protocol} onChange={(v) => setForm({ ...form, study_protocol: v })} placeholder="e.g. CP-2847-301" />
                <FormField label="Therapeutic Area" value={form.therapeutic_area} onChange={(v) => setForm({ ...form, therapeutic_area: v })} placeholder="e.g. Cardiology" />
                <div className="flex justify-end gap-3 pt-2">
                  <button type="button" onClick={() => setShowCreate(false)} className="px-4 py-2 text-[13px] font-medium text-apple-gray hover:text-apple-black transition-colors">
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={!form.name || !form.sponsor_name || !form.site_name || createMutation.isPending}
                    className="px-5 py-2 bg-apple-black text-white text-[13px] font-medium rounded-full hover:bg-apple-dark transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    {createMutation.isPending ? 'Creating...' : 'Create Stack'}
                  </button>
                </div>
              </form>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  )
}

function FormField({ label, value, onChange, placeholder, required }: {
  label: string; value: string; onChange: (v: string) => void; placeholder: string; required?: boolean;
}) {
  return (
    <div>
      <label className="block text-[12px] font-medium text-apple-gray mb-1.5">
        {label}
      </label>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        required={required}
        className="w-full px-3.5 py-2.5 bg-apple-bg rounded-xl border border-black/[0.04] text-[14px] text-apple-black placeholder:text-apple-light focus:outline-none focus:ring-2 focus:ring-apple-dark/20 focus:border-apple-dark/30 transition-all"
      />
    </div>
  )
}

function StatusPill({ status }: { status: string | null }) {
  const config: Record<string, { bg: string; text: string; label: string }> = {
    completed: { bg: 'bg-apple-black/[0.06]', text: 'text-apple-dark', label: 'Processed' },
    processing: { bg: 'bg-apple-silver', text: 'text-apple-dark', label: 'Processing' },
    pending: { bg: 'bg-apple-bg', text: 'text-apple-gray', label: 'Pending' },
    created: { bg: 'bg-apple-bg', text: 'text-apple-gray', label: 'Created' },
  }
  const c = config[status || 'pending'] || config.pending
  return (
    <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-medium ${c.bg} ${c.text}`}>
      {c.label}
    </span>
  )
}
