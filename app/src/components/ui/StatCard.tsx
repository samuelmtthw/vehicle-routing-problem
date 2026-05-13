import { clsx } from 'clsx'

interface Props {
  label: string
  value: string | number
  sub?: string
  accent?: string
  delta?: number
  deltaLabel?: string
}

export function StatCard({ label, value, sub, accent, delta, deltaLabel }: Props) {
  const isPositive = delta !== undefined && delta > 0
  const isNegative = delta !== undefined && delta < 0

  return (
    <div
      className="rounded-xl p-4 border border-[var(--border)] bg-[var(--bg2)] relative overflow-hidden"
      style={{ borderTopColor: accent ?? 'var(--border)', borderTopWidth: 2 }}
    >
      <div className="text-[10px] font-mono tracking-widest text-[var(--text3)] uppercase mb-2">{label}</div>
      <div className="text-2xl font-bold font-mono text-[var(--text)] leading-none">{value}</div>
      {sub && <div className="text-xs text-[var(--text3)] mt-1.5">{sub}</div>}
      {delta !== undefined && (
        <div className={clsx(
          'text-[11px] font-mono mt-1.5 flex items-center gap-1',
          isPositive ? 'text-emerald-400' : isNegative ? 'text-red-400' : 'text-[var(--text3)]'
        )}>
          {isPositive ? '↑' : isNegative ? '↓' : '—'}
          {Math.abs(delta).toFixed(1)}% {deltaLabel ?? 'vs heuristic'}
        </div>
      )}
    </div>
  )
}
