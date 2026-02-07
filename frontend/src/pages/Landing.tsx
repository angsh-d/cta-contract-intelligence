import { useRef } from 'react'
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
} from 'lucide-react'

function FadeUp({ children, className = '', delay = 0 }: { children: React.ReactNode; className?: string; delay?: number }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, margin: '-100px' })
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 50 }}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 }}
      transition={{ duration: 0.9, delay, ease: [0.25, 0.1, 0.25, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  )
}

const features = [
  {
    icon: Layers,
    title: 'Truth Reconstitution',
    subtitle: 'The whole truth. Nothing but.',
    description: 'AI traces every amendment back to its origin, reconstructing the current truth across your entire contract stack. No more guessing which version is authoritative.',
  },
  {
    icon: AlertTriangle,
    title: 'Conflict Detection',
    subtitle: 'Hidden risks. Surfaced instantly.',
    description: 'Contradictions between amendments are revealed in seconds, surfacing risks buried deep in complex document chains that manual review would miss.',
  },
  {
    icon: Waves,
    title: 'Ripple Effect Analysis',
    subtitle: 'See the impact before it happens.',
    description: 'Predict the downstream impact of any proposed change. Understand how one clause modification ripples through every related obligation and dependency.',
  },
]

const steps = [
  { num: '01', icon: Upload, title: 'Upload', description: 'Upload your CTA and all amendments in any order. We handle the rest.' },
  { num: '02', icon: Cpu, title: 'Process', description: 'AI parses, sequences, and extracts every clause with precision.' },
  { num: '03', icon: Search, title: 'Analyze', description: 'Query terms, detect conflicts, trace clause history instantly.' },
  { num: '04', icon: CheckCircle2, title: 'Decide', description: 'Make informed decisions with complete, verified contract truth.' },
]

export default function Landing() {
  const heroRef = useRef(null)
  const { scrollYProgress } = useScroll({ target: heroRef, offset: ['start start', 'end start'] })
  const heroOpacity = useTransform(scrollYProgress, [0, 0.5], [1, 0])
  const heroScale = useTransform(scrollYProgress, [0, 0.5], [1, 0.97])

  return (
    <div className="min-h-screen bg-white">
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-xl border-b border-black/[0.04]">
        <div className="max-w-[1120px] mx-auto px-6 flex items-center justify-between h-[44px]">
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
        <motion.div style={{ opacity: heroOpacity, scale: heroScale }} className="relative max-w-[980px] mx-auto text-center pt-[44px]">
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-[14px] font-medium text-apple-gray tracking-[0.08em] uppercase mb-8"
          >
            Contract Intelligence Platform
          </motion.p>
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="text-[56px] sm:text-[80px] lg:text-[96px] font-bold tracking-[-0.03em] text-apple-black leading-[0.95]"
          >
            Contract Intelligence.
          </motion.h1>
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.45 }}
            className="text-[56px] sm:text-[80px] lg:text-[96px] font-bold tracking-[-0.03em] text-apple-gray leading-[0.95] mt-1"
          >
            Reimagined.
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.65 }}
            className="text-[19px] sm:text-[21px] text-apple-gray2 max-w-[640px] mx-auto mt-10 leading-[1.5]"
          >
            AI-powered analysis for clinical trial agreements. Trace amendments, detect conflicts, and understand the ripple effects of every change.
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
              <FadeUp key={feature.title} delay={i * 0.1}>
                <div className="bg-white rounded-[24px] p-10 sm:p-14 border border-black/[0.04] flex flex-col lg:flex-row lg:items-center gap-8 lg:gap-16">
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
