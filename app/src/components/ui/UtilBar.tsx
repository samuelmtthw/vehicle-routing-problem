interface Props {
  label: string
  value: number
  detail?: string
  color?: string
}

export function UtilBar({ label, value, detail, color = 'var(--blue)' }: Props) {
  return (
    <div className="mb-2">
      <div className="flex justify-between text-[10px] font-mono text-[var(--text3)] mb-1">
        <span>{label}</span>
        <span>{detail ?? value.toFixed(1) + '%'}</span>
      </div>
      <div className="h-1.5 rounded-full bg-[var(--bg5)] overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${Math.min(value, 100)}%`, background: color }}
        />
      </div>
    </div>
  )
}
