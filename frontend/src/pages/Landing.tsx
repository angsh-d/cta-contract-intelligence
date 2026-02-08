import { useRef, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion, useInView, useScroll, useTransform } from 'framer-motion'
import {
  Shield,
  Layers,
  AlertTriangle,
  Waves,
  Upload,
  Cpu,
  Search,
  CheckCircle2,
  ArrowRight,
  Network,
  Zap,
  Brain,
} from 'lucide-react'

function FadeUp({ children, className = '', delay = 0 }: { children: React.ReactNode; className?: string; delay?: number }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, margin: '-80px' })
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 40 }}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 40 }}
      transition={{ duration: 0.8, delay, ease: [0.25, 0.1, 0.25, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  )
}

function AnimatedCounter({ value, suffix = '', duration = 2 }: { value: number; suffix?: string; duration?: number }) {
  const [count, setCount] = useState(0)
  const ref = useRef(null)
  const inView = useInView(ref, { once: true })

  useEffect(() => {
    if (!inView) return
    let start = 0
    const step = Math.ceil(value / (duration * 60))
    const timer = setInterval(() => {
      start += step
      if (start >= value) {
        setCount(value)
        clearInterval(timer)
      } else {
        setCount(start)
      }
    }, 1000 / 60)
    return () => clearInterval(timer)
  }, [inView, value, duration])

  return <span ref={ref}>{count}{suffix}</span>
}

const pipelineStages = [
  { label: 'Parse', icon: Upload },
  { label: 'Sequence', icon: Layers },
  { label: 'Extract', icon: Search },
  { label: 'Resolve', icon: Zap },
  { label: 'Map', icon: Network },
  { label: 'Synthesize', icon: Brain },
]

const features = [
  {
    icon: Layers,
    title: 'Truth Reconstitution',
    subtitle: 'The whole truth. Nothing but.',
    description: 'A multi-agent chain of reasoning traces every amendment back to its origin, reconstructing the current truth across your entire contract stack. Each agent verifies, cross-references, and validates — so no version ambiguity survives.',
  },
  {
    icon: AlertTriangle,
    title: 'Conflict Detection',
    subtitle: 'Hidden risks. Surfaced instantly.',
    description: 'Adversarial scanning agents detect buried changes and contradictions between amendments in seconds, surfacing risks hidden deep in complex document chains that manual review would miss. Every clause is stress-tested against every other.',
  },
  {
    icon: Waves,
    title: 'Ripple Effect Analysis',
    subtitle: 'See the impact before it happens.',
    description: 'Multi-hop cascade analysis traces the downstream impact of any proposed change up to 5 levels deep. Understand how one clause modification ripples through every related obligation, dependency, and cross-reference.',
  },
  {
    icon: Network,
    title: 'Agentic Architecture',
    subtitle: 'Intelligence that collaborates.',
    description: '9 specialized agents collaborate through a tiered pipeline. Each agent brings domain expertise, from document parsing to dependency mapping to predictive analysis. Together, they deliver insights no single model could achieve.',
  },
]

const steps = [
  { num: '01', icon: Upload, title: 'Upload', description: 'Upload your CTA and amendments. PDFs are parsed by specialized document agents that understand clinical trial agreement structure.' },
  { num: '02', icon: Cpu, title: 'Process', description: '9 AI agents extract clauses, track amendments, resolve overrides, map dependencies. Every change is sequenced and verified.' },
  { num: '03', icon: Search, title: 'Analyze', description: 'Query in natural language. AI reconstructs truth from the full amendment chain, citing sources and confidence levels.' },
  { num: '04', icon: CheckCircle2, title: 'Decide', description: 'Detect conflicts, analyze ripple effects, and make data-driven decisions with complete, verified contract intelligence.' },
]

export default function Landing() {
  const heroRef = useRef(null)
  const { scrollYProgress } = useScroll({ target: heroRef, offset: ['start start', 'end start'] })
  const heroOpacity = useTransform(scrollYProgress, [0, 0.5], [1, 0])
  const heroScale = useTransform(scrollYProgress, [0, 0.5], [1, 0.97])

  return (
    <div className="min-h-screen bg-white">
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-xl border-b border-black/[0.04]">
        <div className="max-w-[1120px] mx-auto px-6 flex items-center justify-between h-[48px]">
          <Link to="/" className="flex items-center gap-2.5">
            <Shield className="w-[18px] h-[18px] text-apple-black" />
            <span className="text-[14px] font-semibold tracking-[-0.01em] text-apple-black">ContractIQ</span>
          </Link>
          <div className="flex items-center gap-8">
            <a href="#features" className="text-[12px] font-medium text-apple-gray2 hover:text-apple-black transition-colors hidden sm:block">Features</a>
            <a href="#how-it-works" className="text-[12px] font-medium text-apple-gray2 hover:text-apple-black transition-colors hidden sm:block">How It Works</a>
            <Link
              to="/dashboard"
              className="px-4 py-1.5 bg-apple-black text-white text-[12px] font-medium rounded-full hover:bg-apple-dark transition-colors"
            >
              Get Started
            </Link>
          </div>
        </div>
      </nav>

      <section ref={heroRef} className="relative min-h-screen flex items-center justify-center px-6 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-white via-apple-offwhite to-apple-bg" />
        <motion.div style={{ opacity: heroOpacity, scale: heroScale }} className="relative max-w-[980px] mx-auto text-center pt-[48px]">
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-[14px] font-medium text-apple-gray tracking-[0.08em] uppercase mb-8"
          >
            Agentic Contract Intelligence
          </motion.p>
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="text-[56px] sm:text-[80px] lg:text-[96px] font-bold tracking-[-0.03em] text-apple-black leading-[0.95]"
          >
            Clinical Trial Contracts.
          </motion.h1>
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.45 }}
            className="text-[56px] sm:text-[80px] lg:text-[96px] font-bold tracking-[-0.03em] text-apple-gray leading-[0.95] mt-1"
          >
            Decoded.
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.65 }}
            className="text-[19px] sm:text-[21px] text-apple-gray2 max-w-[640px] mx-auto mt-10 leading-[1.5]"
          >
            AI-powered truth reconstitution across your entire amendment chain. See what changed, what conflicts, and what it means.
          </motion.p>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.85 }}
            className="flex items-center justify-center gap-5 mt-12"
          >
            <Link
              to="/dashboard"
              className="inline-flex items-center gap-2.5 px-8 py-4 bg-apple-black text-white text-[17px] font-medium rounded-full hover:bg-apple-dark transition-all duration-300"
            >
              Get Started
              <ArrowRight className="w-[18px] h-[18px]" />
            </Link>
            <a
              href="#features"
              className="inline-flex items-center gap-2.5 px-8 py-4 text-apple-black text-[17px] font-medium rounded-full border border-apple-light hover:border-apple-gray transition-all duration-300"
            >
              Learn More
            </a>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 1.1 }}
            className="mt-20"
          >
            <div className="flex items-center justify-center gap-3 sm:gap-4 mb-8">
              {pipelineStages.map((stage, i) => (
                <motion.div
                  key={stage.label}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.4, delay: 1.3 + i * 0.1 }}
                  className="flex items-center gap-3 sm:gap-4"
                >
                  <div className="flex flex-col items-center gap-2">
                    <div className="w-10 h-10 sm:w-11 sm:h-11 rounded-xl bg-white/80 backdrop-blur-sm border border-apple-silver/60 flex items-center justify-center shadow-[0_2px_8px_rgba(0,0,0,0.04)]">
                      <stage.icon className="w-4 h-4 sm:w-[18px] sm:h-[18px] text-apple-dark2" />
                    </div>
                    <span className="text-[10px] font-medium text-apple-gray tracking-wide">{stage.label}</span>
                  </div>
                  {i < pipelineStages.length - 1 && (
                    <motion.div
                      initial={{ scaleX: 0 }}
                      animate={{ scaleX: 1 }}
                      transition={{ duration: 0.3, delay: 1.5 + i * 0.1 }}
                      className="w-4 sm:w-6 h-px bg-apple-light mb-5 origin-left"
                    />
                  )}
                </motion.div>
              ))}
            </div>

            <div className="flex items-center justify-center gap-6 sm:gap-10">
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 2 }}
                className="text-center"
              >
                <p className="text-[28px] sm:text-[32px] font-bold text-apple-black tracking-[-0.02em]">
                  <AnimatedCounter value={9} duration={1.5} />
                </p>
                <p className="text-[11px] sm:text-[12px] font-medium text-apple-gray tracking-wide uppercase mt-1">AI Agents</p>
              </motion.div>
              <div className="w-px h-8 bg-apple-silver" />
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 2.15 }}
                className="text-center"
              >
                <p className="text-[28px] sm:text-[32px] font-bold text-apple-black tracking-[-0.02em]">
                  <AnimatedCounter value={6} duration={1.5} />
                </p>
                <p className="text-[11px] sm:text-[12px] font-medium text-apple-gray tracking-wide uppercase mt-1">Stage Pipeline</p>
              </motion.div>
              <div className="w-px h-8 bg-apple-silver" />
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 2.3 }}
                className="text-center"
              >
                <p className="text-[11px] sm:text-[12px] font-medium text-apple-gray tracking-wide uppercase">Real-Time</p>
                <p className="text-[28px] sm:text-[32px] font-bold text-apple-black tracking-[-0.02em] mt-[-2px]">Intelligence</p>
              </motion.div>
            </div>
          </motion.div>
        </motion.div>
      </section>

      <section id="features" className="py-32 sm:py-40 px-6 bg-apple-bg">
        <div className="max-w-[1120px] mx-auto">
          <FadeUp className="text-center mb-24">
            <p className="text-[14px] font-medium text-apple-gray tracking-[0.08em] uppercase mb-5">Capabilities</p>
            <h2 className="text-[44px] sm:text-[56px] lg:text-[64px] font-bold tracking-[-0.03em] text-apple-black leading-[1.05]">
              Intelligence at every layer.
            </h2>
          </FadeUp>

          <div className="space-y-8">
            {features.map((feature, i) => (
              <FadeUp key={feature.title} delay={i * 0.08}>
                <div className="bg-white rounded-[24px] p-10 sm:p-14 border border-black/[0.04] flex flex-col lg:flex-row lg:items-center gap-8 lg:gap-16 hover:shadow-[0_8px_30px_rgba(0,0,0,0.04)] transition-shadow duration-500">
                  <div className="flex-shrink-0">
                    <div className="w-16 h-16 rounded-2xl bg-apple-bg flex items-center justify-center">
                      <feature.icon className="w-8 h-8 text-apple-dark" />
                    </div>
                  </div>
                  <div className="flex-1">
                    <p className="text-[12px] font-medium text-apple-gray tracking-[0.08em] uppercase mb-2">{feature.title}</p>
                    <h3 className="text-[28px] sm:text-[36px] font-bold text-apple-black tracking-[-0.02em] leading-[1.1] mb-4">{feature.subtitle}</h3>
                    <p className="text-[17px] text-apple-gray2 leading-[1.6] max-w-[560px]">{feature.description}</p>
                  </div>
                </div>
              </FadeUp>
            ))}
          </div>
        </div>
      </section>

      <section id="how-it-works" className="py-32 sm:py-40 px-6 bg-white">
        <div className="max-w-[1120px] mx-auto">
          <FadeUp className="text-center mb-24">
            <p className="text-[14px] font-medium text-apple-gray tracking-[0.08em] uppercase mb-5">How It Works</p>
            <h2 className="text-[44px] sm:text-[56px] lg:text-[64px] font-bold tracking-[-0.03em] text-apple-black leading-[1.05]">
              From upload to insight.
            </h2>
          </FadeUp>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-10 lg:gap-8">
            {steps.map((step, i) => (
              <FadeUp key={step.title} delay={i * 0.12}>
                <div className="text-center sm:text-left">
                  <span className="text-[64px] font-bold text-apple-silver tracking-[-0.04em] leading-none">{step.num}</span>
                  <div className="w-14 h-14 rounded-2xl bg-apple-bg flex items-center justify-center mt-6 mb-6 mx-auto sm:mx-0">
                    <step.icon className="w-7 h-7 text-apple-dark" />
                  </div>
                  <h3 className="text-[22px] font-bold text-apple-black tracking-[-0.01em] mb-3">{step.title}</h3>
                  <p className="text-[15px] text-apple-gray leading-[1.6]">{step.description}</p>
                </div>
              </FadeUp>
            ))}
          </div>
        </div>
      </section>

      <section className="py-32 sm:py-40 px-6 bg-apple-black">
        <FadeUp className="text-center">
          <h2 className="text-[44px] sm:text-[56px] lg:text-[64px] font-bold tracking-[-0.03em] text-white leading-[1.05] mb-6">
            Ready to see the truth?
          </h2>
          <p className="text-[19px] text-apple-gray max-w-[520px] mx-auto mb-12 leading-[1.5]">
            Start analyzing your clinical trial agreements with AI-powered intelligence.
          </p>
          <Link
            to="/dashboard"
            className="inline-flex items-center gap-2.5 px-8 py-4 bg-white text-apple-black text-[17px] font-medium rounded-full hover:bg-apple-silver transition-all duration-300"
          >
            Get Started
            <ArrowRight className="w-[18px] h-[18px]" />
          </Link>
        </FadeUp>
      </section>

      <footer className="py-8 px-6 bg-apple-bg border-t border-black/[0.04]">
        <div className="max-w-[1120px] mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4 text-apple-gray" />
            <span className="text-[12px] text-apple-gray font-medium">ContractIQ</span>
          </div>
          <p className="text-[12px] text-apple-gray">
            AI-Powered Contract Intelligence for Clinical Trials
          </p>
        </div>
      </footer>
    </div>
  )
}
