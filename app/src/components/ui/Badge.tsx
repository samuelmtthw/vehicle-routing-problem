import { clsx } from 'clsx'

interface Props {
  children: React.ReactNode
  variant?: 'default' | 'green' | 'blue' | 'orange' | 'purple' | 'red' | 'yellow'
  className?: string
}

const variants = {
  default: 'bg-[var(--bg4)] text-[var(--text2)]',
  green:   'bg-green-950/60 text-emerald-400',
  blue:    'bg-blue-950/60 text-blue-400',
  orange:  'bg-orange-950/60 text-orange-400',
  purple:  'bg-purple-950/60 text-purple-400',
  red:     'bg-red-950/60 text-red-400',
  yellow:  'bg-yellow-950/60 text-yellow-400',
}

export function Badge({ children, variant = 'default', className }: Props) {
  return (
    <span className={clsx(
      'inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-semibold font-mono border border-white/5',
      variants[variant], className
    )}>
      {children}
    </span>
  )
}
