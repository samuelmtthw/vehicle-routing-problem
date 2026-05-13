import { useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, ArrowUpDown, MapPin, Package, Truck, Route, Clock } from 'lucide-react'
import type { ScenarioMeta } from '../types'
import { Card } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { fmtKm, fmtRp, fmtS, isMethodResult, DEPOT_COLORS, SOURCE_COLORS } from '../lib/utils'
import scenariosMeta from '../data/scenarios_meta.json'

const scenarios = scenariosMeta as unknown as ScenarioMeta[]
const depots    = ['All', ...Array.from(new Set(scenarios.map(s => s.depot)))]
const sources   = ['All', 'COMPANY-1', 'COMPANY-2']

export default function Scenarios() {
  const navigate   = useNavigate()
  const [search,   setSearch]   = useState('')
  const [depot,    setDepot]    = useState('All')
  const [source,   setSource]   = useState('All')
  const [sortKey,  setSortKey]  = useState<string>('date')
  const [sortDir,  setSortDir]  = useState<'asc'|'desc'>('asc')

  const filtered = useMemo(() => {
    let list = [...scenarios]
    if (depot  !== 'All') list = list.filter(s => s.depot  === depot)
    if (source !== 'All') list = list.filter(s => s.source === source)
    if (search) list = list.filter(s =>
      s.depot.toLowerCase().includes(search.toLowerCase()) ||
      s.date.includes(search) ||
      s.source.toLowerCase().includes(search.toLowerCase())
    )
    list.sort((a, b) => {
      let va: number | string = 0, vb: number | string = 0
      if (sortKey === 'date')      { va = a.date; vb = b.date }
      if (sortKey === 'awb_count') { va = a.awb_count; vb = b.awb_count }
      if (sortKey === 'h_veh')     { va = a.heuristic.vehicles_used; vb = b.heuristic.vehicles_used }
      if (sortKey === 'h_dist')    { va = a.heuristic.total_distance_km; vb = b.heuristic.total_distance_km }
      if (sortKey === 'h_cost')    { va = a.heuristic.total_cost_rp; vb = b.heuristic.total_cost_rp }
      if (typeof va === 'string') return sortDir === 'asc' ? va.localeCompare(vb as string) : (vb as string).localeCompare(va)
      return sortDir === 'asc' ? va - (vb as number) : (vb as number) - va
    })
    return list
  }, [search, depot, source, sortKey, sortDir])

  const sort = (key: string) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDir('asc') }
  }

  const SortBtn = ({ k, label }: { k: string; label: string }) => (
    <button onClick={() => sort(k)}
      className="flex items-center gap-1 text-[9px] font-mono tracking-widest text-[var(--text3)] hover:text-[var(--text2)] uppercase transition-colors whitespace-nowrap">
      {label}
      <ArrowUpDown size={10} className={sortKey === k ? 'text-[var(--blue)]' : ''} />
    </button>
  )

  return (
    <div className="p-6 fade-in">
      <div className="mb-6">
        <h1 className="text-2xl font-bold mb-1">Scenarios</h1>
        <p className="text-sm text-[var(--text3)]">{scenarios.length} scenarios across 7 depots</p>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 mb-5">
        <div className="relative flex-1 min-w-48">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text3)]" />
          <input
            value={search} onChange={e => setSearch(e.target.value)}
            placeholder="Search depot, date, company..."
            className="w-full pl-8 pr-3 py-2 text-sm bg-[var(--bg2)] border border-[var(--border)] rounded-lg text-[var(--text)] placeholder:text-[var(--text3)] focus:outline-none focus:border-[var(--border2)]"
          />
        </div>
        <select value={depot} onChange={e => setDepot(e.target.value)}
          className="px-3 py-2 text-sm bg-[var(--bg2)] border border-[var(--border)] rounded-lg text-[var(--text)] focus:outline-none focus:border-[var(--border2)]">
          {depots.map(d => <option key={d} value={d}>{d === 'All' ? 'All Depots' : d.replace('Depot ','')}</option>)}
        </select>
        <select value={source} onChange={e => setSource(e.target.value)}
          className="px-3 py-2 text-sm bg-[var(--bg2)] border border-[var(--border)] rounded-lg text-[var(--text)] focus:outline-none focus:border-[var(--border2)]">
          {sources.map(s => <option key={s} value={s}>{s === 'All' ? 'All Companies' : s}</option>)}
        </select>
        <span className="px-3 py-2 text-sm font-mono text-[var(--text3)] bg-[var(--bg2)] border border-[var(--border)] rounded-lg">
          {filtered.length} results
        </span>
      </div>

      {/* Table */}
      <Card className="overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="px-4 py-3 text-left"><SortBtn k="date"      label="Date" /></th>
                <th className="px-4 py-3 text-left"><span className="text-[9px] font-mono tracking-widest text-[var(--text3)] uppercase">Company</span></th>
                <th className="px-4 py-3 text-left"><span className="text-[9px] font-mono tracking-widest text-[var(--text3)] uppercase">Depot</span></th>
                <th className="px-4 py-3 text-left"><span className="text-[9px] font-mono tracking-widest text-[var(--text3)] uppercase">Location</span></th>
                <th className="px-4 py-3 text-right"><SortBtn k="awb_count" label="AWBs" /></th>
                <th className="px-4 py-3 text-right"><SortBtn k="h_veh"     label="H Vehicles" /></th>
                <th className="px-4 py-3 text-right"><span className="text-[9px] font-mono tracking-widest text-[var(--text3)] uppercase">I Vehicles</span></th>
                <th className="px-4 py-3 text-right"><SortBtn k="h_dist"    label="H Distance" /></th>
                <th className="px-4 py-3 text-right"><span className="text-[9px] font-mono tracking-widest text-[var(--text3)] uppercase">I Distance</span></th>
                <th className="px-4 py-3 text-right"><SortBtn k="h_cost"    label="H Cost" /></th>
                <th className="px-4 py-3 text-right"><span className="text-[9px] font-mono tracking-widest text-[var(--text3)] uppercase">ILP Status</span></th>
                <th className="px-4 py-3 text-right"></th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(s => {
                const ilp = isMethodResult(s.ilp) ? s.ilp : null
                const depotColor = DEPOT_COLORS[s.depot] ?? '#4f8ef7'
                const srcColor   = SOURCE_COLORS[s.source] ?? '#4f8ef7'
                return (
                  <tr key={s.id}
                    className="border-b border-[var(--border)] hover:bg-[var(--bg3)] cursor-pointer transition-colors group"
                    onClick={() => navigate(`/scenarios/${encodeURIComponent(s.id)}`)}>
                    <td className="px-4 py-3 font-mono text-[var(--text2)] text-xs">{s.date}</td>
                    <td className="px-4 py-3">
                      <span className="font-mono text-[10px] px-1.5 py-0.5 rounded font-bold"
                        style={{ background: s.source === 'COMPANY-1' ? '#0a2010' : '#0a0a20', color: srcColor }}>
                        {s.source}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full shrink-0" style={{ background: depotColor }} />
                        <span className="font-medium text-sm">{s.depot.replace('Depot ','')}</span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-xs text-[var(--text3)]">
                      {s.depot_city}, {s.depot_province}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <span className="font-mono text-[var(--text)] font-semibold">{s.awb_count}</span>
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-xs text-orange-400">{s.heuristic.vehicles_used}</td>
                    <td className="px-4 py-3 text-right font-mono text-xs text-blue-400">{ilp?.vehicles_used ?? '—'}</td>
                    <td className="px-4 py-3 text-right font-mono text-xs text-orange-400">{fmtKm(s.heuristic.total_distance_km)}</td>
                    <td className="px-4 py-3 text-right font-mono text-xs text-blue-400">{ilp ? fmtKm(ilp.total_distance_km) : '—'}</td>
                    <td className="px-4 py-3 text-right font-mono text-xs text-[var(--text2)]">{fmtRp(s.heuristic.total_cost_rp)}</td>
                    <td className="px-4 py-3 text-right">
                      {!s.ilp ? <Badge variant="default">pending</Badge>
                        : s.ilp.status === 'no_solution' ? <Badge variant="red">no solution</Badge>
                        : <Badge variant="green">solved</Badge>}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <span className="text-xs text-[var(--text3)] group-hover:text-[var(--blue)] transition-colors font-mono">View →</span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}
