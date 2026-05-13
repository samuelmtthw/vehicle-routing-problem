import { useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  LineChart, Line, ScatterChart, Scatter, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from 'recharts'
import type { ScenarioMeta } from '../types'
import { StatCard } from '../components/ui/StatCard'
import { Card } from '../components/ui/Card'
import { fmtKm, fmtRp, fmtS, pctDiff, isMethodResult } from '../lib/utils'
import scenariosMeta from '../data/scenarios_meta.json'

const scenarios = scenariosMeta as unknown as ScenarioMeta[]
const solved = scenarios.filter(s => isMethodResult(s.ilp))

const TICK = { fill: '#4d5670', fontSize: 10, fontFamily: 'Space Mono' }
const GRID = { stroke: '#1c2030', strokeDasharray: '3 3' }
const TT = {
  contentStyle: { background: '#161922', border: '1px solid #252b3b', borderRadius: 10, fontSize: 11, fontFamily: 'Space Mono' },
  labelStyle: { color: '#eceef5', fontWeight: 600 },
  itemStyle: { color: '#8892aa' },
}
const LEG = { wrapperStyle: { fontSize: 10, fontFamily: 'Space Mono' } }

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card className="p-5">
      <div className="text-[10px] font-mono tracking-widest text-[var(--text3)] uppercase mb-4">{title}</div>
      {children}
    </Card>
  )
}

export default function Dashboard() {
  const avgH = useMemo(() => ({
    vehicles: scenarios.reduce((a, s) => a + s.heuristic.vehicles_used, 0) / scenarios.length,
    dist:     scenarios.reduce((a, s) => a + s.heuristic.total_distance_km, 0) / scenarios.length,
    cost:     scenarios.reduce((a, s) => a + s.heuristic.total_cost_rp, 0) / scenarios.length,
    runtime:  scenarios.reduce((a, s) => a + s.heuristic.runtime_sec, 0) / scenarios.length,
    w_util:   scenarios.reduce((a, s) => a + s.heuristic.avg_weight_util_pct, 0) / scenarios.length,
    v_util:   scenarios.reduce((a, s) => a + s.heuristic.avg_volume_util_pct, 0) / scenarios.length,
  }), [])

  const avgI = useMemo(() => {
    if (!solved.length) return null
    return {
      vehicles: solved.reduce((a, s) => a + (isMethodResult(s.ilp) ? s.ilp.vehicles_used : 0), 0) / solved.length,
      dist:     solved.reduce((a, s) => a + (isMethodResult(s.ilp) ? s.ilp.total_distance_km : 0), 0) / solved.length,
      cost:     solved.reduce((a, s) => a + (isMethodResult(s.ilp) ? s.ilp.total_cost_rp : 0), 0) / solved.length,
      runtime:  solved.reduce((a, s) => a + (isMethodResult(s.ilp) ? s.ilp.runtime_sec : 0), 0) / solved.length,
      w_util:   solved.reduce((a, s) => a + (isMethodResult(s.ilp) ? s.ilp.avg_weight_util_pct : 0), 0) / solved.length,
      v_util:   solved.reduce((a, s) => a + (isMethodResult(s.ilp) ? s.ilp.avg_volume_util_pct : 0), 0) / solved.length,
    }
  }, [])

  const byDepot = useMemo(() => {
    const g: Record<string, ScenarioMeta[]> = {}
    scenarios.forEach(s => { if (!g[s.depot]) g[s.depot] = []; g[s.depot].push(s) })
    return Object.entries(g).map(([depot, rows]) => {
      const ilpRows = rows.filter(r => isMethodResult(r.ilp))
      const avg = (fn: (s: ScenarioMeta) => number, arr: ScenarioMeta[]) =>
        arr.length ? Math.round(arr.reduce((a, s) => a + fn(s), 0) / arr.length * 10) / 10 : null
      return {
        depot:  depot.replace('Depot ', ''),
        h_veh:  avg(s => s.heuristic.vehicles_used, rows),
        i_veh:  avg(s => isMethodResult(s.ilp) ? s.ilp.vehicles_used : 0, ilpRows),
        h_dist: avg(s => s.heuristic.total_distance_km, rows),
        i_dist: avg(s => isMethodResult(s.ilp) ? s.ilp.total_distance_km : 0, ilpRows),
        h_cost: avg(s => s.heuristic.total_cost_rp, rows),
        i_cost: avg(s => isMethodResult(s.ilp) ? s.ilp.total_cost_rp : 0, ilpRows),
        h_w:    avg(s => s.heuristic.avg_weight_util_pct, rows),
        i_w:    avg(s => isMethodResult(s.ilp) ? s.ilp.avg_weight_util_pct : 0, ilpRows),
        h_v:    avg(s => s.heuristic.avg_volume_util_pct, rows),
        i_v:    avg(s => isMethodResult(s.ilp) ? s.ilp.avg_volume_util_pct : 0, ilpRows),
      }
    })
  }, [])

  const scatter = useMemo(() =>
    solved.map(s => ({
      name:     s.depot.replace('Depot ', '') + ' ' + s.date,
      awbs:     s.awb_count,
      h_dist:   s.heuristic.total_distance_km,
      i_dist:   isMethodResult(s.ilp) ? s.ilp.total_distance_km : 0,
      diff_pct: isMethodResult(s.ilp) ? pctDiff(s.heuristic.total_distance_km, s.ilp.total_distance_km) : 0,
    }))
  , [])

  const timeSeries = useMemo(() =>
    scenarios
      .filter(s => s.depot === 'Depot Jatiasih')
      .sort((a, b) => a.date.localeCompare(b.date))
      .map(s => ({
        date:   s.date.slice(5),
        h_veh:  s.heuristic.vehicles_used,
        i_veh:  isMethodResult(s.ilp) ? s.ilp.vehicles_used : null,
        h_dist: s.heuristic.total_distance_km,
        i_dist: isMethodResult(s.ilp) ? s.ilp.total_distance_km : null,
      }))
  , [])

  const radarData = useMemo(() => {
    if (!avgI) return []
    const norm = (a: number, b: number) => ({ H: Math.round((a / Math.max(a, b)) * 100), I: Math.round((b / Math.max(a, b)) * 100) })
    return [
      { metric: 'Vehicles', ...norm(avgH.vehicles, avgI.vehicles) },
      { metric: 'Distance', ...norm(avgH.dist,     avgI.dist) },
      { metric: 'Cost',     ...norm(avgH.cost,     avgI.cost) },
      { metric: 'Runtime',  ...norm(avgH.runtime,  avgI.runtime) },
      { metric: 'W Util%',  ...norm(avgH.w_util,   avgI.w_util) },
      { metric: 'V Util%',  ...norm(avgH.v_util,   avgI.v_util) },
    ]
  }, [avgH, avgI])

  return (
    <div className="p-6 space-y-6 fade-in">
      <div>
        <h1 className="text-2xl font-bold mb-1">Dashboard</h1>
        <p className="text-sm text-[var(--text3)]">
          K-Means + Bin Packing vs ILP + OR-Tools — {scenarios.length} scenarios, 7 depots, 2 clients
        </p>
      </div>

      {/* Heuristic stats */}
      <div>
        <div className="text-[10px] font-mono tracking-widest text-[var(--text3)] uppercase mb-3 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-orange-400 inline-block" />
          Method 1 — K-Means + Bin Packing (all 41 scenarios)
        </div>
        <div className="grid grid-cols-3 lg:grid-cols-6 gap-3">
          <StatCard label="Avg Vehicles"  value={avgH.vehicles.toFixed(1)}      accent="#f97316" />
          <StatCard label="Avg Distance"  value={fmtKm(Math.round(avgH.dist))}  accent="#4f8ef7" />
          <StatCard label="Avg Fuel Cost" value={fmtRp(Math.round(avgH.cost))}  accent="#a855f7" />
          <StatCard label="Avg Runtime"   value={fmtS(avgH.runtime)}            accent="#eab308" />
          <StatCard label="Avg W Util"    value={avgH.w_util.toFixed(1) + '%'}  accent="#22d3a0" />
          <StatCard label="Avg V Util"    value={avgH.v_util.toFixed(1) + '%'}  accent="#06b6d4" />
        </div>
      </div>

      {/* ILP stats */}
      {avgI && (
        <div>
          <div className="text-[10px] font-mono tracking-widest text-[var(--text3)] uppercase mb-3 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-blue-400 inline-block" />
            Method 2 — ILP + OR-Tools ({solved.length} solved, 1 no_solution)
          </div>
          <div className="grid grid-cols-3 lg:grid-cols-6 gap-3">
            <StatCard label="Avg Vehicles"  value={avgI.vehicles.toFixed(1)}      accent="#f97316" delta={pctDiff(avgH.vehicles, avgI.vehicles)} />
            <StatCard label="Avg Distance"  value={fmtKm(Math.round(avgI.dist))}  accent="#4f8ef7" delta={pctDiff(avgH.dist, avgI.dist)} />
            <StatCard label="Avg Fuel Cost" value={fmtRp(Math.round(avgI.cost))}  accent="#a855f7" delta={pctDiff(avgH.cost, avgI.cost)} />
            <StatCard label="Avg Runtime"   value={fmtS(avgI.runtime)}            accent="#eab308" delta={pctDiff(avgH.runtime, avgI.runtime)} />
            <StatCard label="Avg W Util"    value={avgI.w_util.toFixed(1) + '%'}  accent="#22d3a0" delta={pctDiff(avgH.w_util, avgI.w_util)} />
            <StatCard label="Avg V Util"    value={avgI.v_util.toFixed(1) + '%'}  accent="#06b6d4" delta={pctDiff(avgH.v_util, avgI.v_util)} />
          </div>
        </div>
      )}

      {/* Charts grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ChartCard title="Avg Vehicles Used — by Depot">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={byDepot} margin={{ top:0, right:0, bottom:0, left:-10 }}>
              <CartesianGrid {...GRID} />
              <XAxis dataKey="depot" tick={TICK} />
              <YAxis tick={TICK} />
              <Tooltip {...TT} />
              <Legend {...LEG} />
              <Bar dataKey="h_veh" name="Heuristic" fill="#f97316" radius={[3,3,0,0]} />
              <Bar dataKey="i_veh" name="ILP"       fill="#4f8ef7" radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Avg Total Distance km — by Depot">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={byDepot} margin={{ top:0, right:0, bottom:0, left:-10 }}>
              <CartesianGrid {...GRID} />
              <XAxis dataKey="depot" tick={TICK} />
              <YAxis tick={TICK} />
              <Tooltip {...TT} />
              <Legend {...LEG} />
              <Bar dataKey="h_dist" name="Heuristic" fill="#f97316" radius={[3,3,0,0]} />
              <Bar dataKey="i_dist" name="ILP"       fill="#4f8ef7" radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Avg Fuel Cost (Rp) — by Depot">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={byDepot} margin={{ top:0, right:0, bottom:0, left:10 }}>
              <CartesianGrid {...GRID} />
              <XAxis dataKey="depot" tick={TICK} />
              <YAxis tick={TICK} tickFormatter={v => 'Rp' + Math.round((v as number)/1000) + 'k'} />
              <Tooltip {...TT} formatter={(v: unknown) => ['Rp ' + (v as number).toLocaleString('id-ID'), '']} />
              <Legend {...LEG} />
              <Bar dataKey="h_cost" name="Heuristic" fill="#a855f7" radius={[3,3,0,0]} />
              <Bar dataKey="i_cost" name="ILP"       fill="#22d3a0" radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Avg Capacity Utilisation % — by Depot">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={byDepot} margin={{ top:0, right:0, bottom:0, left:-10 }}>
              <CartesianGrid {...GRID} />
              <XAxis dataKey="depot" tick={TICK} />
              <YAxis tick={TICK} domain={[0, 100]} />
              <Tooltip {...TT} />
              <Legend {...LEG} />
              <Bar dataKey="h_w" name="H Weight%" fill="#f97316"   radius={[3,3,0,0]} />
              <Bar dataKey="h_v" name="H Volume%" fill="#f9731688" radius={[3,3,0,0]} />
              <Bar dataKey="i_w" name="I Weight%" fill="#4f8ef7"   radius={[3,3,0,0]} />
              <Bar dataKey="i_v" name="I Volume%" fill="#4f8ef788" radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Time series */}
      {timeSeries.length > 0 && (
        <ChartCard title="Depot Jatiasih — Vehicles & Distance Over Time (COMPANY-2, 27 days)">
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={timeSeries} margin={{ top:0, right:16, bottom:0, left:-10 }}>
              <CartesianGrid {...GRID} />
              <XAxis dataKey="date" tick={TICK} />
              <YAxis yAxisId="l" tick={TICK} />
              <YAxis yAxisId="r" orientation="right" tick={TICK} />
              <Tooltip {...TT} />
              <Legend {...LEG} />
              <Line yAxisId="l" type="monotone" dataKey="h_dist" stroke="#f97316" dot={false} strokeWidth={2}   name="H Distance km" />
              <Line yAxisId="l" type="monotone" dataKey="i_dist" stroke="#4f8ef7" dot={false} strokeWidth={2}   name="I Distance km" />
              <Line yAxisId="r" type="monotone" dataKey="h_veh"  stroke="#f97316" dot={false} strokeWidth={1.5} strokeDasharray="4 2" name="H Vehicles" />
              <Line yAxisId="r" type="monotone" dataKey="i_veh"  stroke="#4f8ef7" dot={false} strokeWidth={1.5} strokeDasharray="4 2" name="I Vehicles" />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ChartCard title="Distance: Heuristic vs ILP per Scenario (scatter)">
          <ResponsiveContainer width="100%" height={260}>
            <ScatterChart margin={{ top:0, right:0, bottom:16, left:-10 }}>
              <CartesianGrid {...GRID} />
              <XAxis dataKey="h_dist" name="Heuristic" tick={TICK} label={{ value:'Heuristic km', position:'insideBottom', offset:-4, fill:'#4d5670', fontSize:10 }} />
              <YAxis dataKey="i_dist" name="ILP" tick={TICK} label={{ value:'ILP km', angle:-90, position:'insideLeft', fill:'#4d5670', fontSize:10 }} />
              <Tooltip cursor={{ strokeDasharray:'3 3', stroke:'var(--border2)' }}
                content={({ payload }) => {
                  if (!payload?.length) return null
                  const d = payload[0]?.payload as { name: string; h_dist: number; i_dist: number; diff_pct: number }
                  return (
                    <div style={{ background:'var(--bg3)', border:'1px solid var(--border2)', borderRadius:8, padding:'8px 12px', fontSize:11, fontFamily:'Space Mono' }}>
                      <div style={{ color:'var(--text)', marginBottom:4, fontSize:10 }}>{d?.name}</div>
                      <div style={{ color:'#f97316' }}>Heuristic: {d?.h_dist?.toFixed(1)} km</div>
                      <div style={{ color:'#4f8ef7' }}>ILP: {d?.i_dist?.toFixed(1)} km</div>
                      <div style={{ color: (d?.diff_pct ?? 0) > 0 ? '#ef4444' : '#22d3a0' }}>
                        {(d?.diff_pct ?? 0) > 0 ? '+' : ''}{d?.diff_pct?.toFixed(1)}%
                      </div>
                    </div>
                  )
                }}
              />
              <Scatter data={scatter} fill="#4f8ef7" fillOpacity={0.8} />
            </ScatterChart>
          </ResponsiveContainer>
        </ChartCard>

        {radarData.length > 0 && (
          <ChartCard title="Profile Comparison — normalised (lower = better for Vehicles/Distance/Cost/Runtime)">
            <ResponsiveContainer width="100%" height={260}>
              <RadarChart data={radarData} margin={{ top:10, right:20, bottom:10, left:20 }}>
                <PolarGrid stroke="#1c2030" />
                <PolarAngleAxis dataKey="metric" tick={{ fill:'#4d5670', fontSize:10, fontFamily:'Space Mono' }} />
                <PolarRadiusAxis tick={{ fill:'#4d5670', fontSize:9 }} domain={[0,100]} />
                <Radar name="Heuristic" dataKey="H" stroke="#f97316" fill="#f97316" fillOpacity={0.15} />
                <Radar name="ILP"       dataKey="I" stroke="#4f8ef7" fill="#4f8ef7" fillOpacity={0.15} />
                <Legend {...LEG} />
              </RadarChart>
            </ResponsiveContainer>
          </ChartCard>
        )}
      </div>

      {/* Full comparison table */}
      <Card className="overflow-hidden">
        <div className="px-5 py-4 border-b border-[var(--border)]">
          <div className="text-[10px] font-mono tracking-widest text-[var(--text3)] uppercase">Per-Scenario Comparison Table</div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[11px]">
            <thead>
              <tr className="border-b border-[var(--border)]">
                {['Source','Depot','Date','AWBs','H Veh','I Veh','Δ Veh%','H Dist','I Dist','Δ Dist%','H Cost','I Cost','Δ Cost%','H Runtime','I Runtime'].map(h => (
                  <th key={h} className="px-3 py-2.5 text-left font-mono text-[9px] tracking-widest text-[var(--text3)] uppercase whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {scenarios.map((s, i) => {
                const ilp    = isMethodResult(s.ilp) ? s.ilp : null
                const dVeh   = ilp ? pctDiff(s.heuristic.vehicles_used, ilp.vehicles_used) : null
                const dDist  = ilp ? pctDiff(s.heuristic.total_distance_km, ilp.total_distance_km) : null
                const dCost  = ilp ? pctDiff(s.heuristic.total_cost_rp, ilp.total_cost_rp) : null
                const delta  = (v: number | null) => {
                  if (v === null) return <span className="text-[var(--text3)]">—</span>
                  return <span className={v < 0 ? 'text-emerald-400 font-semibold' : v === 0 ? 'text-[var(--text3)]' : 'text-red-400'}>
                    {v > 0 ? '+' : ''}{v.toFixed(1)}%
                  </span>
                }
                return (
                  <tr key={i} className="border-b border-[var(--border)] hover:bg-[var(--bg3)] transition-colors">
                    <td className="px-3 py-2">
                      <span className="font-mono text-[9px] px-1.5 py-0.5 rounded font-bold"
                        style={{ background: s.source === 'COMPANY-1' ? '#0a2010' : '#0a0a20', color: s.source === 'COMPANY-1' ? '#22d3a0' : '#a855f7' }}>
                        {s.source}
                      </span>
                    </td>
                    <td className="px-3 py-2 font-mono text-[var(--text2)]">{s.depot.replace('Depot ','')}</td>
                    <td className="px-3 py-2 font-mono text-[var(--text2)]">{s.date}</td>
                    <td className="px-3 py-2 font-mono text-[var(--text)]">{s.awb_count}</td>
                    <td className="px-3 py-2 font-mono text-orange-400">{s.heuristic.vehicles_used}</td>
                    <td className="px-3 py-2 font-mono text-blue-400">{ilp?.vehicles_used ?? '—'}</td>
                    <td className="px-3 py-2 font-mono">{delta(dVeh)}</td>
                    <td className="px-3 py-2 font-mono text-orange-400">{s.heuristic.total_distance_km.toFixed(1)}</td>
                    <td className="px-3 py-2 font-mono text-blue-400">{ilp?.total_distance_km.toFixed(1) ?? '—'}</td>
                    <td className="px-3 py-2 font-mono">{delta(dDist)}</td>
                    <td className="px-3 py-2 font-mono text-orange-400">{fmtRp(s.heuristic.total_cost_rp)}</td>
                    <td className="px-3 py-2 font-mono text-blue-400">{ilp ? fmtRp(ilp.total_cost_rp) : '—'}</td>
                    <td className="px-3 py-2 font-mono">{delta(dCost)}</td>
                    <td className="px-3 py-2 font-mono text-[var(--text2)]">{s.heuristic.runtime_sec.toFixed(2)}s</td>
                    <td className="px-3 py-2 font-mono text-[var(--text2)]">{ilp ? ilp.runtime_sec.toFixed(2) + 's' : '—'}</td>
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
