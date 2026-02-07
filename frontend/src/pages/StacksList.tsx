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
  show: { opacity: 1, transition: { staggerChildren: 0.06 } },
}
const item = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.25, 0.1, 0.25, 1] as const } },
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
    <div className="min-h-screen bg-apple-offwhite">
      <div className="max-w-[960px] mx-auto px-6 lg:px-10 py-14">
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.25, 0.1, 0.25, 1] }}
          className="flex items-center justify-between mb-10"
        >
          <div>
            <h1 className="text-[32px] font-semibold tracking-[-0.02em] text-apple-black">Contracts</h1>
            <p className="text-[14px] text-apple-gray mt-1">Manage your clinical trial agreement stacks</p>
          </div>
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setShowCreate(true)}
            className="flex items-center gap-2 px-5 py-2.5 bg-apple-black text-white text-[13px] font-medium rounded-full hover:bg-apple-dark transition-all duration-300 shadow-sm"
          >
            <Plus className="w-4 h-4" strokeWidth={2} />
            New Stack
          </motion.button>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1, ease: [0.25, 0.1, 0.25, 1] }}
          className="relative mb-8"
        >
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-[16px] h-[16px] text-apple-gray" />
          <input
            type="text"
            placeholder="Search contracts..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-11 pr-4 py-3.5 bg-white/80 backdrop-blur-lg rounded-2xl border border-apple-silver/60 text-[14px] text-apple-black placeholder:text-apple-light focus:outline-none focus:ring-2 focus:ring-apple-dark/10 focus:border-apple-light focus:bg-white transition-all duration-300 shadow-[0_1px_3px_rgba(0,0,0,0.02)]"
          />
          {search && (
            <button
              onClick={() => setSearch('')}
              className="absolute right-4 top-1/2 -translate-y-1/2 p-0.5 rounded-full hover:bg-apple-silver/50 transition-colors"
            >
              <X className="w-3.5 h-3.5 text-apple-gray" />
            </button>
          )}
        </motion.div>

        {isLoading && (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: i * 0.1 }}
                className="h-[84px] skeleton rounded-2xl"
              />
            ))}
          </div>
        )}

        {!isLoading && stacks.length === 0 && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            className="text-center py-24"
          >
            <div className="w-20 h-20 rounded-[20px] bg-white border border-apple-silver/60 flex items-center justify-center mx-auto mb-5 shadow-[0_2px_8px_rgba(0,0,0,0.04)]">
              <FileStack className="w-9 h-9 text-apple-gray" />
            </div>
            <h3 className="text-[20px] font-semibold text-apple-black mb-2 tracking-[-0.01em]">
              {search ? 'No matches found' : 'No contracts yet'}
            </h3>
            <p className="text-[15px] text-apple-gray mb-8 max-w-[320px] mx-auto leading-relaxed">
              {search ? 'Try a different search term' : 'Create your first contract stack to get started with AI-powered analysis'}
            </p>
            {!search && (
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setShowCreate(true)}
                className="inline-flex items-center gap-2 px-6 py-3 bg-apple-black text-white text-[14px] font-medium rounded-full hover:bg-apple-dark transition-all duration-300"
              >
                <Plus className="w-4 h-4" />
                Create Stack
              </motion.button>
            )}
          </motion.div>
        )}

        {!isLoading && stacks.length > 0 && (
        <motion.div variants={container} initial="hidden" animate="show" className="space-y-2.5" key={stacks.length}>
          {stacks.map((stack) => (
            <motion.div key={stack.id} variants={item}>
              <Link
                to={`/stacks/${stack.id}`}
                className="flex items-center justify-between px-5 py-4.5 bg-white/90 backdrop-blur-sm rounded-2xl border border-apple-silver/50 hover:border-apple-light hover:shadow-[0_4px_16px_rgba(0,0,0,0.06)] hover:bg-white transition-all duration-300 group"
              >
                <div className="flex items-center gap-4 min-w-0">
                  <div className="w-12 h-12 rounded-[14px] bg-apple-bg border border-apple-silver/40 flex items-center justify-center flex-shrink-0">
                    <Shield className="w-5.5 h-5.5 text-apple-dark2" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-[14px] font-medium text-apple-black truncate tracking-[-0.01em]">{stack.name}</p>
                    <div className="flex items-center gap-3 mt-1">
                      <span className="flex items-center gap-1 text-[12px] text-apple-gray">
                        <Building2 className="w-3 h-3" /> {stack.sponsor_name}
                      </span>
                      <span className="flex items-center gap-1 text-[12px] text-apple-gray">
                        <MapPin className="w-3 h-3" /> {stack.site_name}
                      </span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-3 flex-shrink-0">
                  {stack.therapeutic_area && (
                    <span className="hidden sm:inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-apple-bg border border-apple-silver/40 text-apple-gray2">
                      {stack.therapeutic_area}
                    </span>
                  )}
                  <StatusPill status={stack.processing_status} />
                  <ChevronRight className="w-4 h-4 text-apple-light group-hover:text-apple-gray group-hover:translate-x-0.5 transition-all duration-300" />
                </div>
              </Link>
            </motion.div>
          ))}
        </motion.div>
        )}

        <AnimatePresence>
          {showCreate && (
            <>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.25 }}
                className="fixed inset-0 bg-apple-black/25 backdrop-blur-md z-50"
                onClick={() => setShowCreate(false)}
              />
              <motion.div
                initial={{ opacity: 0, scale: 0.96, y: 24 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.96, y: 24 }}
                transition={{ type: 'spring', damping: 28, stiffness: 320 }}
                className="fixed inset-x-4 top-[8%] sm:inset-auto sm:left-1/2 sm:top-1/2 sm:-translate-x-1/2 sm:-translate-y-1/2 w-auto sm:w-[500px] bg-white/95 backdrop-blur-2xl rounded-[20px] shadow-[0_24px_80px_rgba(0,0,0,0.15)] z-50 border border-white/50"
              >
                <div className="flex items-center justify-between px-7 py-5 border-b border-apple-silver/40">
                  <h2 className="text-[18px] font-semibold text-apple-black tracking-[-0.01em]">New Contract Stack</h2>
                  <button
                    onClick={() => setShowCreate(false)}
                    className="p-1.5 rounded-full hover:bg-apple-silver/50 transition-colors duration-200"
                  >
                    <X className="w-5 h-5 text-apple-gray" />
                  </button>
                </div>
                <form onSubmit={handleCreate} className="p-7 space-y-5">
                  <FormField label="Study Name" value={form.name} onChange={(v) => setForm({ ...form, name: v })} placeholder="e.g. HEARTBEAT-3" required />
                  <FormField label="Sponsor" value={form.sponsor_name} onChange={(v) => setForm({ ...form, sponsor_name: v })} placeholder="e.g. CardioVita Therapeutics" required />
                  <FormField label="Site" value={form.site_name} onChange={(v) => setForm({ ...form, site_name: v })} placeholder="e.g. Memorial Medical Center" required />
                  <div className="grid grid-cols-2 gap-4">
                    <FormField label="Protocol" value={form.study_protocol} onChange={(v) => setForm({ ...form, study_protocol: v })} placeholder="e.g. CP-2847-301" />
                    <FormField label="Therapeutic Area" value={form.therapeutic_area} onChange={(v) => setForm({ ...form, therapeutic_area: v })} placeholder="e.g. Cardiology" />
                  </div>
                  <div className="flex justify-end gap-3 pt-3">
                    <button
                      type="button"
                      onClick={() => setShowCreate(false)}
                      className="px-5 py-2.5 text-[13px] font-medium text-apple-gray2 hover:text-apple-black rounded-full hover:bg-apple-bg transition-all duration-200"
                    >
                      Cancel
                    </button>
                    <motion.button
                      whileHover={{ scale: 1.01 }}
                      whileTap={{ scale: 0.98 }}
                      type="submit"
                      disabled={!form.name || !form.sponsor_name || !form.site_name || createMutation.isPending}
                      className="px-6 py-2.5 bg-apple-black text-white text-[13px] font-medium rounded-full hover:bg-apple-dark transition-all duration-300 disabled:opacity-30 disabled:cursor-not-allowed shadow-sm"
                    >
                      {createMutation.isPending ? 'Creating…' : 'Create Stack'}
                    </motion.button>
                  </div>
                </form>
              </motion.div>
            </>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

function FormField({ label, value, onChange, placeholder, required }: {
  label: string; value: string; onChange: (v: string) => void; placeholder: string; required?: boolean;
}) {
  return (
    <div>
      <label className="block text-[12px] font-medium text-apple-gray2 mb-2 tracking-wide uppercase">
        {label}{required && ' *'}
      </label>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        required={required}
        className="w-full px-4 py-3 bg-apple-bg/60 rounded-xl border border-apple-silver/50 text-[14px] text-apple-black placeholder:text-apple-light/80 focus:outline-none focus:ring-2 focus:ring-apple-dark/10 focus:border-apple-light focus:bg-white transition-all duration-300"
      />
    </div>
  )
}

function StatusPill({ status }: { status: string | null }) {
  const config: Record<string, { bg: string; text: string; label: string; dot: string }> = {
    completed: { bg: 'bg-apple-dark/[0.06]', text: 'text-apple-dark', label: 'Processed', dot: 'bg-apple-dark' },
    processing: { bg: 'bg-apple-silver/80', text: 'text-apple-dark2', label: 'Processing', dot: 'bg-apple-gray' },
    pending: { bg: 'bg-apple-bg', text: 'text-apple-gray', label: 'Pending', dot: 'bg-apple-light' },
    created: { bg: 'bg-apple-bg', text: 'text-apple-gray', label: 'Created', dot: 'bg-apple-light' },
  }
  const c = config[status || 'pending'] || config.pending
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium ${c.bg} ${c.text}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${c.dot}`} />
      {c.label}
    </span>
  )
}
