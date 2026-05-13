import { clsx } from 'clsx'

interface Props {
  children: React.ReactNode
  className?: string
  onClick?: () => void
}

export function Card({ children, className, onClick }: Props) {
  return (
    <div
      onClick={onClick}
      className={clsx(
        'rounded-xl border bg-[var(--bg2)] border-[var(--border)]',
        onClick && 'cursor-pointer hover:border-[var(--border2)] hover:bg-[var(--bg3)] transition-all duration-150',
        className
      )}
    >
      {children}
    </div>
  )
}
