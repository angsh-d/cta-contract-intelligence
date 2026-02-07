import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  FileStack,
  AlertTriangle,
  Search,
  ArrowRight,
  Plus,
  Shield,
  Clock,
  CheckCircle2,
  Activity,
} from 'lucide-react'
import { useStacks, useHealth } from '../hooks/useApi'

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.06 } },
}
const item = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.25, 0.1, 0.25, 1] as const } },
}

export default function Dashboard() {
  const { data: stacksData } = useStacks()
  const { data: healthData } = useHealth()
  const stacks = stacksData?.stacks || []
  const aiAvailable = healthData?.ai_available ?? false

  const totalStacks = stacks.length
  const processed = stacks.filter((s) => s.processing_status === 'completed').length
  const pending = stacks.filter((s) => s.processing_status === 'pending' || s.processing_status === null).length

  return (
    <div className="max-w-6xl mx-auto px-6 lg:px-10 py-10">
      <motion.div variants={container} initial="hidden" animate="show">
        <motion.div variants={item} className="mb-10">
          <h1 className="text-[32px] font-semibold tracking-tight text-apple-black">
            Welcome back
          </h1>
          <p className="text-[15px] text-apple-gray mt-1">
            Your contract intelligence overview
          </p>
        </motion.div>

        <motion.div variants={item} className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
          <StatCard icon={FileStack} label="Contract Stacks" value={totalStacks} />
          <StatCard icon={CheckCircle2} label="Processed" value={processed} />
          <StatCard icon={Clock} label="Pending" value={pending} />
          <StatCard
            icon={Activity}
            label="AI Engine"
            value={aiAvailable ? 'Active' : 'Offline'}
          />
        </motion.div>

        <motion.div variants={item} className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-10">
          <ActionCard
            to="/stacks"
            icon={Plus}
            title="New Contract Stack"
            description="Upload and analyze a new set of clinical trial agreements"
            dark
          />
          <ActionCard
            to="/stacks"
            icon={Search}
            title="Query Contracts"
            description="Ask questions about your contract terms and obligations"
          />
          <ActionCard
            to="/stacks"
            icon={AlertTriangle}
            title="Detect Conflicts"
            description="Run AI-powered analysis to find hidden risks"
          />
        </motion.div>

        {stacks.length > 0 && (
          <motion.div variants={item}>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-[17px] font-semibold text-apple-black">Recent Contracts</h2>
              <Link to="/stacks" className="flex items-center gap-1 text-[13px] font-medium text-apple-gray hover:text-apple-black transition-colors">
                View all <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>
            <div className="bg-white rounded-2xl border border-black/[0.04] overflow-hidden">
              {stacks.slice(0, 5).map((stack, i) => (
                <Link
                  key={stack.id}
                  to={`/stacks/${stack.id}`}
                  className={`flex items-center justify-between px-5 py-4 hover:bg-apple-bg/50 transition-colors ${
                    i < Math.min(stacks.length, 5) - 1 ? 'border-b border-black/[0.04]' : ''
                  }`}
                >
                  <div className="flex items-center gap-4 min-w-0">
                    <div className="w-10 h-10 rounded-xl bg-apple-bg flex items-center justify-center flex-shrink-0">
                      <Shield className="w-5 h-5 text-apple-dark" />
                    </div>
                    <div className="min-w-0">
                      <p className="text-[14px] font-medium text-apple-black truncate">{stack.name}</p>
                      <p className="text-[12px] text-apple-gray truncate">{stack.sponsor_name} · {stack.site_name}</p>
                    </div>
                  </div>
                  <StatusPill status={stack.processing_status} />
                </Link>
              ))}
            </div>
          </motion.div>
        )}

        {stacks.length === 0 && (
          <motion.div variants={item} className="text-center py-20">
            <div className="w-16 h-16 rounded-2xl bg-apple-silver/40 flex items-center justify-center mx-auto mb-4">
              <FileStack className="w-8 h-8 text-apple-gray" />
            </div>
            <h3 className="text-[17px] font-semibold text-apple-black mb-1">No contracts yet</h3>
            <p className="text-[14px] text-apple-gray mb-6">Create your first contract stack to get started</p>
            <Link
              to="/stacks"
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-apple-black text-white text-[13px] font-medium rounded-full hover:bg-apple-dark transition-colors"
            >
              <Plus className="w-4 h-4" />
              New Contract Stack
            </Link>
          </motion.div>
        )}
      </motion.div>
    </div>
  )
}

function StatCard({ icon: Icon, label, value }: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string | number;
}) {
  return (
    <div className="bg-white rounded-2xl border border-black/[0.04] p-5">
      <div className="flex items-center justify-between mb-3">
        <span className="text-[12px] font-medium text-apple-gray uppercase tracking-wider">{label}</span>
        <Icon className="w-4 h-4 text-apple-gray" />
      </div>
      <p className="text-[28px] font-semibold tracking-tight text-apple-black">
        {value}
      </p>
    </div>
  )
}

function ActionCard({ to, icon: Icon, title, description, dark }: {
  to: string;
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
  dark?: boolean;
}) {
  return (
    <Link
      to={to}
      className={`group relative rounded-2xl p-6 overflow-hidden transition-all duration-300 border ${
        dark
          ? 'bg-apple-black text-white border-apple-black hover:bg-apple-dark'
          : 'bg-white text-apple-black border-black/[0.04] hover:border-apple-light hover:shadow-sm'
      }`}
    >
      <Icon className={`w-8 h-8 mb-4 ${dark ? 'opacity-80' : 'text-apple-dark'}`} />
      <h3 className="text-[15px] font-semibold mb-1">{title}</h3>
      <p className={`text-[13px] leading-relaxed ${dark ? 'opacity-70' : 'text-apple-gray'}`}>{description}</p>
      <ArrowRight className={`w-4 h-4 mt-4 group-hover:translate-x-1 transition-all ${dark ? 'opacity-50 group-hover:opacity-80' : 'text-apple-light group-hover:text-apple-gray'}`} />
    </Link>
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
