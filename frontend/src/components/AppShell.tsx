import { useState } from 'react'
import { NavLink, useLocation, Outlet } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LayoutDashboard,
  FileStack,
  Menu,
  X,
  Zap,
  User,
} from 'lucide-react'

const NAV_ITEMS = [
  { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/stacks', icon: FileStack, label: 'Contracts' },
]

export default function AppShell() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const location = useLocation()

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-apple-bg">
      <header className="flex items-center h-12 px-5 border-b border-black/[0.06] bg-white shrink-0 z-50">
        <div className="flex items-center gap-3">
          <img
            src="https://www.saama.com/wp-content/uploads/saama_logo.svg"
            alt="Saama"
            className="h-5"
          />
          <div className="w-[1px] h-5 bg-black/15" />
          <span className="text-[13px] font-medium text-apple-dark2 tracking-tight">
            Digital Contract Platform
          </span>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <div className="w-7 h-7 rounded-full bg-apple-dark2 flex items-center justify-center">
            <User className="w-3.5 h-3.5 text-white" />
          </div>
          <span className="text-[13px] font-medium text-apple-dark2">Angshuman Deb</span>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        <aside className="hidden lg:flex flex-col w-56 bg-white/80 backdrop-blur-xl border-r border-black/[0.04]">
          <nav className="flex-1 px-3 py-4 space-y-1">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/dashboard'}
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
          <div className="px-4 py-4 border-t border-black/[0.04]">
            <div className="flex items-center gap-2 px-2 py-2 rounded-lg bg-apple-bg/60">
              <Zap className="w-3.5 h-3.5 text-apple-gray" />
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
                className="fixed top-12 bottom-0 left-0 w-56 bg-white/95 backdrop-blur-xl border-r border-black/[0.04] z-50 lg:hidden"
              >
                <nav className="px-3 py-4 space-y-1">
                  {NAV_ITEMS.map((item) => (
                    <NavLink
                      key={item.to}
                      to={item.to}
                      end={item.to === '/dashboard'}
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
          <div className="flex items-center h-10 px-4 lg:hidden border-b border-black/[0.04] bg-white/60 backdrop-blur-xl">
            <button onClick={() => setSidebarOpen(true)} className="p-1.5 -ml-1.5 rounded-lg hover:bg-apple-silver/50 transition-colors">
              <Menu className="w-5 h-5 text-apple-dark" />
            </button>
          </div>
          <main className="flex-1 overflow-y-auto">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, ease: [0.25, 0.1, 0.25, 1] }}
            >
              <Outlet />
            </motion.div>
          </main>
        </div>
      </div>
    </div>
  )
}
