import { ReactNode, useState } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LayoutDashboard,
  FileStack,
  Menu,
  X,
  Shield,
  Zap,
} from 'lucide-react'

const NAV_ITEMS = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/stacks', icon: FileStack, label: 'Contracts' },
]

export default function AppShell({ children }: { children: ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const location = useLocation()

  return (
    <div className="flex h-screen overflow-hidden bg-apple-bg">
      <aside className="hidden lg:flex flex-col w-64 bg-white/80 backdrop-blur-xl border-r border-apple-silver/60">
        <div className="flex items-center gap-3 px-6 h-16 border-b border-apple-silver/40">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-apple-black to-apple-dark flex items-center justify-center">
            <Shield className="w-4 h-4 text-white" />
          </div>
          <span className="text-[15px] font-semibold tracking-tight text-apple-black">ContractIQ</span>
        </div>
        <nav className="flex-1 px-3 py-4 space-y-1">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-xl text-[13px] font-medium transition-all duration-200 ${
                  isActive
                    ? 'bg-apple-black text-white shadow-sm'
                    : 'text-apple-gray hover:text-apple-black hover:bg-apple-silver/50'
                }`
              }
            >
              <item.icon className="w-[18px] h-[18px]" />
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="px-4 py-4 border-t border-apple-silver/40">
          <div className="flex items-center gap-2 px-2 py-2 rounded-lg bg-apple-bg/60">
            <Zap className="w-3.5 h-3.5 text-apple-green" />
            <span className="text-[11px] text-apple-gray font-medium">AI Engine Active</span>
          </div>
        </div>
      </aside>

      <AnimatePresence>
        {sidebarOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40 lg:hidden"
              onClick={() => setSidebarOpen(false)}
            />
            <motion.aside
              initial={{ x: -280 }}
              animate={{ x: 0 }}
              exit={{ x: -280 }}
              transition={{ type: 'spring', damping: 25, stiffness: 300 }}
              className="fixed inset-y-0 left-0 w-64 bg-white/95 backdrop-blur-xl border-r border-apple-silver/60 z-50 lg:hidden"
            >
              <div className="flex items-center justify-between px-6 h-16 border-b border-apple-silver/40">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-apple-black to-apple-dark flex items-center justify-center">
                    <Shield className="w-4 h-4 text-white" />
                  </div>
                  <span className="text-[15px] font-semibold tracking-tight">ContractIQ</span>
                </div>
                <button onClick={() => setSidebarOpen(false)} className="p-1.5 rounded-lg hover:bg-apple-silver/50 transition-colors">
                  <X className="w-5 h-5 text-apple-gray" />
                </button>
              </div>
              <nav className="px-3 py-4 space-y-1">
                {NAV_ITEMS.map((item) => (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    end={item.to === '/'}
                    onClick={() => setSidebarOpen(false)}
                    className={({ isActive }) =>
                      `flex items-center gap-3 px-3 py-2.5 rounded-xl text-[13px] font-medium transition-all duration-200 ${
                        isActive
                          ? 'bg-apple-black text-white shadow-sm'
                          : 'text-apple-gray hover:text-apple-black hover:bg-apple-silver/50'
                      }`
                    }
                  >
                    <item.icon className="w-[18px] h-[18px]" />
                    {item.label}
                  </NavLink>
                ))}
              </nav>
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      <div className="flex-1 flex flex-col min-w-0">
        <header className="flex items-center h-14 px-4 lg:px-8 border-b border-apple-silver/40 bg-white/60 backdrop-blur-xl lg:hidden">
          <button onClick={() => setSidebarOpen(true)} className="p-2 -ml-2 rounded-lg hover:bg-apple-silver/50 transition-colors">
            <Menu className="w-5 h-5 text-apple-dark" />
          </button>
          <div className="flex items-center gap-2 ml-3">
            <Shield className="w-5 h-5 text-apple-black" />
            <span className="text-[14px] font-semibold tracking-tight">ContractIQ</span>
          </div>
        </header>
        <main className="flex-1 overflow-y-auto">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, ease: [0.25, 0.1, 0.25, 1] }}
          >
            {children}
          </motion.div>
        </main>
      </div>
    </div>
  )
}
