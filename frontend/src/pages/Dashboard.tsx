import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  FileStack,
  Search,
  ArrowRight,
  Plus,
  Shield,
  Clock,
  CheckCircle2,
  Activity,
  Layers,
  GitBranch,
  Zap,
  ChevronRight,
  Cpu,
  CircleDot,
} from 'lucide-react'
import { useStacks, useHealth } from '../hooks/useApi'

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.07 } },
}
const item = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.25, 0.1, 0.25, 1] as const } },
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
    <div className="max-w-6xl mx-auto px-6 lg:px-10 py-12">
      <motion.div variants={container} initial="hidden" animate="show">

        <motion.div variants={item} className="mb-12">
          <h1 className="text-[32px] font-semibold tracking-tight text-apple-black leading-tight">
            Welcome back
          </h1>
          <p className="text-[15px] text-apple-gray mt-1.5 tracking-normal">
            Your contract intelligence overview
          </p>
        </motion.div>

        <motion.div variants={item} className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-14">
          <StatCard icon={FileStack} label="Contract Stacks" value={totalStacks} />
          <StatCard icon={CheckCircle2} label="Processed" value={processed} />
          <StatCard icon={Clock} label="Pending" value={pending} />
          <StatCard
            icon={Activity}
            label="AI Engine"
            value={aiAvailable ? 'Active' : 'Offline'}
          />
        </motion.div>

        <motion.div variants={item} className="mb-3">
          <h2 className="text-[22px] font-semibold text-apple-black tracking-tight">
            Agentic AI Capabilities
          </h2>
          <p className="text-[14px] text-apple-gray mt-1">
            Intelligent contract analysis powered by specialized agents
          </p>
        </motion.div>

        <motion.div variants={item} className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-14">
          <ActionCard
            to="/stacks"
            icon={Layers}
            title="Truth Reconstitution"
            description="AI traces every amendment, reconstructing the authoritative truth"
            dark
          />
          <ActionCard
            to="/stacks"
            icon={Search}
            title="Conflict Detection"
            description="Detect buried changes, contradictions, and coverage gaps"
          />
          <ActionCard
            to="/stacks"
            icon={GitBranch}
            title="Ripple Analysis"
            description="Predict downstream impact before executing changes"
          />
        </motion.div>

        {stacks.length > 0 && (
          <motion.div variants={item} className="mb-14">
            <div className="flex items-center justify-between mb-5">
              <h2 className="text-[20px] font-semibold text-apple-black tracking-tight">Recent Contracts</h2>
              <Link to="/stacks" className="flex items-center gap-1 text-[13px] font-medium text-apple-gray hover:text-apple-black transition-colors duration-200">
                View all <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>
            <div className="bg-white rounded-2xl border border-black/[0.04] overflow-hidden shadow-[0_0_0_0.5px_rgba(0,0,0,0.02)]">
              {stacks.slice(0, 5).map((stack, i) => (
                <Link
                  key={stack.id}
                  to={`/stacks/${stack.id}`}
                  className={`group flex items-center justify-between px-5 py-4 hover:bg-apple-bg/60 transition-colors duration-200 ${
                    i < Math.min(stacks.length, 5) - 1 ? 'border-b border-black/[0.04]' : ''
                  }`}
                >
                  <div className="flex items-center gap-4 min-w-0">
                    <div className="w-10 h-10 rounded-xl bg-apple-bg flex items-center justify-center flex-shrink-0">
                      <Shield className="w-5 h-5 text-apple-dark2" />
                    </div>
                    <div className="min-w-0">
                      <p className="text-[14px] font-medium text-apple-black truncate">{stack.name}</p>
                      <p className="text-[12px] text-apple-gray truncate">{stack.sponsor_name} · {stack.site_name}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <StatusPill status={stack.processing_status} />
                    <ChevronRight className="w-4 h-4 text-apple-light opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
                  </div>
                </Link>
              ))}
            </div>
          </motion.div>
        )}

        {stacks.length === 0 && (
          <motion.div variants={item} className="text-center py-24 mb-14">
            <div className="w-16 h-16 rounded-2xl bg-apple-silver/40 flex items-center justify-center mx-auto mb-5">
              <FileStack className="w-8 h-8 text-apple-gray" />
            </div>
            <h3 className="text-[17px] font-semibold text-apple-black mb-1.5">No contracts yet</h3>
            <p className="text-[14px] text-apple-gray mb-7 max-w-xs mx-auto leading-relaxed">
              Create your first contract stack to get started with AI-powered analysis
            </p>
            <Link
              to="/stacks"
              className="inline-flex items-center gap-2 px-6 py-2.5 bg-apple-black text-white text-[13px] font-medium rounded-full hover:bg-apple-dark transition-colors duration-200"
            >
              <Plus className="w-4 h-4" />
              New Contract Stack
            </Link>
          </motion.div>
        )}

        <motion.div variants={item}>
          <div className="rounded-2xl border border-black/[0.04] bg-white/80 backdrop-blur-xl p-8 shadow-[0_0_0_0.5px_rgba(0,0,0,0.02)]">
            <div className="text-center mb-8">
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-apple-bg border border-black/[0.04] mb-4">
                <Cpu className="w-3.5 h-3.5 text-apple-gray2" />
                <span className="text-[11px] font-medium text-apple-gray2 uppercase tracking-wider">Architecture</span>
              </div>
              <h2 className="text-[20px] font-semibold text-apple-black tracking-tight">
                Powered by 9 Specialized AI Agents
              </h2>
              <p className="text-[13px] text-apple-gray mt-1.5">
                A multi-tier pipeline for comprehensive contract intelligence
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 relative">
              <div className="hidden md:block absolute top-1/2 left-1/3 w-[1px] h-12 -translate-y-1/2 bg-gradient-to-b from-transparent via-apple-light to-transparent" />
              <div className="hidden md:block absolute top-1/2 left-2/3 w-[1px] h-12 -translate-y-1/2 bg-gradient-to-b from-transparent via-apple-light to-transparent" />

              <AgentTier
                tier="Tier 1"
                label="Ingestion"
                agents={['Parser', 'Amendment Tracker', 'Temporal Sequencer']}
              />
              <AgentTier
                tier="Tier 2"
                label="Reasoning"
                agents={['Override Resolution', 'Dependency Mapper', 'Conflict Detection']}
              />
              <AgentTier
                tier="Tier 3"
                label="Analysis"
                agents={['Ripple Effect', 'Truth Synthesis', 'Query Router']}
              />
            </div>

            <div className="flex items-center justify-center gap-1.5 mt-8">
              {[...Array(9)].map((_, i) => (
                <CircleDot key={i} className="w-2.5 h-2.5 text-apple-light" />
              ))}
            </div>
          </div>
        </motion.div>

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
    <div className="bg-white rounded-2xl border border-black/[0.04] p-5 shadow-[0_0_0_0.5px_rgba(0,0,0,0.02)] hover:shadow-[0_2px_8px_rgba(0,0,0,0.04)] transition-shadow duration-300">
      <div className="flex items-center justify-between mb-4">
        <span className="text-[11px] font-medium text-apple-gray uppercase tracking-wider">{label}</span>
        <div className="w-7 h-7 rounded-lg bg-apple-bg flex items-center justify-center">
          <Icon className="w-3.5 h-3.5 text-apple-gray2" />
        </div>
      </div>
      <p className="text-[28px] font-semibold tracking-tight text-apple-black leading-none">
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
      className={`group relative rounded-2xl p-7 overflow-hidden transition-all duration-300 border ${
        dark
          ? 'bg-apple-black text-white border-apple-black hover:bg-apple-dark shadow-[0_4px_20px_rgba(0,0,0,0.15)]'
          : 'bg-white text-apple-black border-black/[0.04] hover:border-apple-light hover:shadow-[0_2px_12px_rgba(0,0,0,0.06)]'
      }`}
    >
      <div className={`w-10 h-10 rounded-xl flex items-center justify-center mb-5 ${
        dark ? 'bg-white/10' : 'bg-apple-bg'
      }`}>
        <Icon className={`w-5 h-5 ${dark ? 'text-white/90' : 'text-apple-dark2'}`} />
      </div>
      <h3 className="text-[16px] font-semibold mb-2 tracking-tight">{title}</h3>
      <p className={`text-[13px] leading-relaxed ${dark ? 'text-white/60' : 'text-apple-gray'}`}>{description}</p>
      <div className={`flex items-center gap-1 mt-5 text-[12px] font-medium ${
        dark ? 'text-white/40 group-hover:text-white/70' : 'text-apple-light group-hover:text-apple-gray'
      } transition-colors duration-200`}>
        <span>Explore</span>
        <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform duration-200" />
      </div>
    </Link>
  )
}

function StatusPill({ status }: { status: string | null }) {
  const config: Record<string, { bg: string; text: string; label: string }> = {
    completed: { bg: 'bg-apple-black/[0.06]', text: 'text-apple-dark', label: 'Processed' },
    processing: { bg: 'bg-apple-silver', text: 'text-apple-dark2', label: 'Processing' },
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

function AgentTier({ tier, label, agents }: {
  tier: string;
  label: string;
  agents: string[];
}) {
  return (
    <div className="text-center">
      <div className="mb-3">
        <span className="text-[10px] font-semibold text-apple-gray uppercase tracking-widest">{tier}</span>
        <p className="text-[14px] font-semibold text-apple-black mt-0.5">{label}</p>
      </div>
      <div className="flex flex-col gap-2 items-center">
        {agents.map((agent) => (
          <div
            key={agent}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-apple-bg border border-black/[0.04] text-[11px] font-medium text-apple-dark2"
          >
            <Zap className="w-2.5 h-2.5 text-apple-gray" />
            {agent}
          </div>
        ))}
      </div>
    </div>
  )
}
