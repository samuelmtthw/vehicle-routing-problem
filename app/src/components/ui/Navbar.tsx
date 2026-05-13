import { NavLink } from 'react-router-dom'
import { LayoutDashboard, List, Map } from 'lucide-react'
import { clsx } from 'clsx'

const NAV = [
  { to: '/',         icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/scenarios', icon: List,            label: 'Scenarios'  },
]

export function Navbar() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 h-14 flex items-center px-6 gap-8"
      style={{ background: 'rgba(8,9,12,.92)', borderBottom: '1px solid var(--border)', backdropFilter: 'blur(12px)' }}>
      <div className="flex items-center gap-2 mr-4">
        <div className="w-7 h-7 rounded-lg flex items-center justify-center"
          style={{ background: 'linear-gradient(135deg,#4f8ef7,#a855f7)' }}>
          <Map size={14} className="text-white" />
        </div>
        <span className="font-bold text-sm tracking-tight">HCVRP<span className="text-[var(--text3)]">.research</span></span>
      </div>

      <div className="flex items-center gap-1">
        {NAV.map(({ to, icon: Icon, label }) => (
          <NavLink key={to} to={to} end className={({ isActive }) => clsx(
            'flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-150',
            isActive
              ? 'bg-[var(--bg3)] text-[var(--text)] border border-[var(--border2)]'
              : 'text-[var(--text3)] hover:text-[var(--text2)] hover:bg-[var(--bg3)]'
          )}>
            <Icon size={15} />
            {label}
          </NavLink>
        ))}
      </div>

      <div className="ml-auto flex items-center gap-3">
        <span className="text-[11px] font-mono text-[var(--text3)] px-2 py-1 rounded-md bg-[var(--bg3)] border border-[var(--border)]">
          41 scenarios · 7 depots
        </span>
      </div>
    </nav>
  )
}
