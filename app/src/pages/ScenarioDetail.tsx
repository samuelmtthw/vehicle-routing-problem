import { useState, useEffect, useRef, useMemo } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import L from 'leaflet'
import { ArrowLeft, MapPin, Package, Truck, Route, Clock, ChevronRight, AlertCircle } from 'lucide-react'
import type { Scenario } from '../types'
import { UtilBar } from '../components/ui/UtilBar'
import { Badge } from '../components/ui/Badge'
import { fmtKm, fmtRp, fmtS, fmtPct, isMethodResult, VEHICLE_COLORS, DEPOT_COLORS } from '../lib/utils'
import scenariosData from '../data/scenarios.json'

const allScenarios = scenariosData as unknown as Scenario[]

type Method = 'heuristic' | 'ilp'

export default function ScenarioDetail() {
  const { id }     = useParams<{ id: string }>()
  const navigate   = useNavigate()
  const [method, setMethod] = useState<Method>('heuristic')

  const scenario = useMemo(() =>
    allScenarios.find(s => s.id === decodeURIComponent(id ?? ''))
  , [id])

  const mapRef    = useRef<L.Map | null>(null)
  const layersRef = useRef<L.Layer[]>([])
  const markersRef = useRef<L.Layer[]>([])

  // Initialise map
  useEffect(() => {
    if (!scenario) return
    if (mapRef.current) { mapRef.current.remove(); mapRef.current = null }

    const map = L.map('scenario-map', { zoomControl: true, attributionControl: true })
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', { maxZoom: 18 }).addTo(map)
    mapRef.current = map

    return () => { map.remove(); mapRef.current = null }
  }, [scenario?.id])

  // Render markers when scenario or method changes
  useEffect(() => {
    if (!scenario || !mapRef.current) return
    const map = mapRef.current

    // Clear previous layers
    markersRef.current.forEach(l => map.removeLayer(l))
    markersRef.current = []

    const bounds: L.LatLngExpression[] = []
    const dp: L.LatLngExpression = [scenario.depot_lat, scenario.depot_lng]
    bounds.push(dp)

    // Depot marker
    const depotIcon = L.divIcon({
      html: `<div style="width:18px;height:18px;background:#eab308;border:2.5px solid #fff;border-radius:50%;box-shadow:0 0 14px #eab30888;"></div>`,
      className: '', iconAnchor: [9, 9],
    })
    const dm = L.marker(dp, { icon: depotIcon, zIndexOffset: 1000 }).bindPopup(
      `<div style="padding:10px 12px;">
        <div style="font-weight:700;font-size:13px;margin-bottom:4px;">📦 DEPOT</div>
        <div style="color:#8892aa;font-size:11px;">${scenario.depot}</div>
        <div style="color:#8892aa;font-size:11px;">${scenario.depot_city}, ${scenario.depot_province}</div>
      </div>`
    )
    dm.addTo(map)
    markersRef.current.push(dm)

    // AWB markers with tooltips
    scenario.awbs.forEach((awb, idx) => {
      const color = '#4f8ef7'
      bounds.push([awb.lat, awb.lng])

      const icon = L.divIcon({
        html: `<div style="
          width:28px;height:28px;
          background:${color}22;
          border:1.5px solid ${color};
          border-radius:50%;
          display:flex;align-items:center;justify-content:center;
          font-family:'Space Mono',monospace;font-size:9px;font-weight:700;
          color:${color};cursor:pointer;
          transition:all .15s;
          box-shadow:0 2px 8px rgba(0,0,0,.4);
        ">${idx + 1}</div>`,
        className: '', iconAnchor: [14, 14],
      })

      const itemsHtml = awb.items.map(it =>
        `<div style="display:flex;justify-content:space-between;gap:16px;padding:2px 0;border-bottom:1px solid #252b3b;">
          <span style="color:#8892aa;">${it.quantity} ${it.unit}</span>
          <span style="color:#eceef5;">${it.weight_kg.toFixed(2)} kg · ${it.volume_cbm.toFixed(3)} CBM</span>
        </div>`
      ).join('')

      const popup = L.popup({ maxWidth: 300, className: '' }).setContent(
        `<div style="padding:12px 14px;min-width:240px;">
          <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
            <span style="background:#4f8ef722;border:1px solid #4f8ef7;border-radius:4px;padding:2px 6px;font-family:'Space Mono',monospace;font-size:10px;color:#4f8ef7;font-weight:700;">${awb.awb_id}</span>
          </div>
          <div style="font-weight:600;font-size:13px;color:#eceef5;margin-bottom:2px;">${awb.location_name}</div>
          <div style="font-size:11px;color:#8892aa;margin-bottom:6px;">${awb.city}, ${awb.province}</div>
          <div style="font-size:10px;color:#4d5670;margin-bottom:8px;">${awb.address}</div>
          <div style="display:flex;gap:12px;margin-bottom:8px;">
            <span style="font-size:11px;color:#22d3a0;font-family:'Space Mono',monospace;">${awb.total_weight.toFixed(2)} kg</span>
            <span style="font-size:11px;color:#a855f7;font-family:'Space Mono',monospace;">${awb.total_volume.toFixed(3)} CBM</span>
            <span style="font-size:11px;color:#4d5670;font-family:'Space Mono',monospace;">${awb.customer}</span>
          </div>
          <div style="font-family:'Space Mono',monospace;font-size:10px;color:#4d5670;margin-bottom:4px;letter-spacing:.06em;">ITEMS (${awb.items.length})</div>
          <div style="font-family:'Space Mono',monospace;font-size:10px;">${itemsHtml}</div>
        </div>`
      )

      const marker = L.marker([awb.lat, awb.lng], { icon })
      marker.bindPopup(popup)
      marker.on('mouseover', () => marker.openPopup())
      marker.addTo(map)
      markersRef.current.push(marker)
    })

    if (bounds.length > 1) {
      map.fitBounds(bounds as L.LatLngBoundsExpression, { padding: [40, 40] })
    }
  }, [scenario, method])

  if (!scenario) {
    return (
      <div className="flex items-center justify-center h-full text-[var(--text3)] font-mono text-sm">
        Scenario not found
      </div>
    )
  }

  const activeResult = method === 'heuristic' ? scenario.heuristic : (isMethodResult(scenario.ilp) ? scenario.ilp : null)
  const depotColor   = DEPOT_COLORS[scenario.depot] ?? '#4f8ef7'

  return (
    <div className="flex h-full overflow-hidden">
      {/* Sidebar */}
      <div className="w-80 shrink-0 flex flex-col overflow-y-auto border-r border-[var(--border)]"
        style={{ background: 'var(--bg2)' }}>

        {/* Back button + header */}
        <div className="p-4 border-b border-[var(--border)]">
          <button onClick={() => navigate('/scenarios')}
            className="flex items-center gap-1.5 text-xs text-[var(--text3)] hover:text-[var(--text)] transition-colors mb-4 font-mono">
            <ArrowLeft size={13} /> Back to Scenarios
          </button>
          <div className="flex items-center gap-2 mb-1">
            <span className="w-3 h-3 rounded-full shrink-0" style={{ background: depotColor }} />
            <span className="font-bold text-base">{scenario.depot.replace('Depot ','')}</span>
          </div>
          <div className="text-xs text-[var(--text3)] font-mono mb-3">{scenario.date} · {scenario.depot_city}</div>
          <div className="flex gap-2">
            <span className="font-mono text-[10px] px-1.5 py-0.5 rounded font-bold"
              style={{ background: scenario.source === 'COMPANY-1' ? '#0a2010' : '#0a0a20', color: scenario.source === 'COMPANY-1' ? '#22d3a0' : '#a855f7' }}>
              {scenario.source}
            </span>
            <Badge variant="blue">{scenario.awb_count} AWBs</Badge>
          </div>
        </div>

        {/* Method tabs */}
        <div className="p-4 border-b border-[var(--border)]">
          <div className="text-[9px] font-mono tracking-widest text-[var(--text3)] uppercase mb-2">Select Method</div>
          <div className="flex flex-col gap-2">
            <button
              onClick={() => setMethod('heuristic')}
              className={`px-3 py-2.5 rounded-lg text-sm font-medium text-left transition-all border ${
                method === 'heuristic'
                  ? 'bg-orange-950/40 border-orange-500/50 text-orange-300'
                  : 'bg-[var(--bg3)] border-[var(--border)] text-[var(--text2)] hover:border-[var(--border2)]'
              }`}>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-orange-400 shrink-0" />
                <div>
                  <div className="font-semibold text-xs">Method 1</div>
                  <div className="text-[10px] opacity-70">K-Means + Bin Packing</div>
                </div>
              </div>
            </button>
            <button
              onClick={() => setMethod('ilp')}
              className={`px-3 py-2.5 rounded-lg text-sm font-medium text-left transition-all border ${
                method === 'ilp'
                  ? 'bg-blue-950/40 border-blue-500/50 text-blue-300'
                  : 'bg-[var(--bg3)] border-[var(--border)] text-[var(--text2)] hover:border-[var(--border2)]'
              }`}>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-blue-400 shrink-0" />
                <div>
                  <div className="font-semibold text-xs">Method 2</div>
                  <div className="text-[10px] opacity-70">ILP + OR-Tools</div>
                </div>
              </div>
            </button>
          </div>
        </div>

        {/* Method results */}
        <div className="p-4 border-b border-[var(--border)]">
          {activeResult ? (
            <>
              <div className="text-[9px] font-mono tracking-widest text-[var(--text3)] uppercase mb-3">
                {method === 'heuristic' ? 'Heuristic Results' : 'ILP Results'}
              </div>
              <div className="grid grid-cols-2 gap-2 mb-4">
                {[
                  { label: 'Vehicles', value: activeResult.vehicles_used, icon: Truck, color: '#f97316' },
                  { label: 'Distance', value: fmtKm(activeResult.total_distance_km), icon: Route, color: '#4f8ef7' },
                  { label: 'Fuel Cost', value: fmtRp(activeResult.total_cost_rp), icon: Package, color: '#a855f7' },
                  { label: 'Runtime', value: fmtS(activeResult.runtime_sec), icon: Clock, color: '#eab308' },
                ].map(({ label, value, icon: Icon, color }) => (
                  <div key={label} className="rounded-lg p-3 border border-[var(--border)] bg-[var(--bg3)]"
                    style={{ borderTopColor: color, borderTopWidth: 2 }}>
                    <div className="flex items-center gap-1 mb-1">
                      <Icon size={10} style={{ color }} />
                      <span className="text-[9px] font-mono tracking-wide text-[var(--text3)] uppercase">{label}</span>
                    </div>
                    <div className="font-bold font-mono text-sm" style={{ color }}>{value}</div>
                  </div>
                ))}
              </div>

              {/* Utilisation bars */}
              <UtilBar label="Weight Utilisation" value={activeResult.avg_weight_util_pct}
                detail={`${activeResult.avg_weight_util_pct.toFixed(1)}%`} color="#22d3a0" />
              <UtilBar label="Volume Utilisation" value={activeResult.avg_volume_util_pct}
                detail={`${activeResult.avg_volume_util_pct.toFixed(1)}%`} color="#a855f7" />

              {/* Vehicle breakdown */}
              <div className="mt-4">
                <div className="text-[9px] font-mono tracking-widest text-[var(--text3)] uppercase mb-2">Vehicle Breakdown</div>
                <div className="space-y-1.5">
                  {Object.entries(activeResult.vehicle_breakdown).map(([vtype, count]) => (
                    <div key={vtype} className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full" style={{ background: VEHICLE_COLORS[vtype] ?? '#888' }} />
                        <span className="text-xs text-[var(--text2)]">{vtype}</span>
                      </div>
                      <span className="font-mono text-xs font-bold" style={{ color: VEHICLE_COLORS[vtype] ?? '#888' }}>
                        ×{count}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {method === 'heuristic' && (activeResult as typeof scenario.heuristic).k_clusters !== undefined && (
                <div className="mt-3 pt-3 border-t border-[var(--border)]">
                  <span className="text-[10px] font-mono text-[var(--text3)]">
                    K-Means clusters: <span className="text-[var(--text)]">{(activeResult as typeof scenario.heuristic).k_clusters}</span>
                  </span>
                </div>
              )}

              {method === 'ilp' && (activeResult as NonNullable<typeof scenario.ilp> & { ortools_objective?: number }).ortools_objective && (
                <div className="mt-3 pt-3 border-t border-[var(--border)]">
                  <span className="text-[10px] font-mono text-[var(--text3)]">
                    OR-Tools objective: <span className="text-[var(--text)]">
                      {((activeResult as NonNullable<typeof scenario.ilp> & { ortools_objective?: number }).ortools_objective ?? 0).toLocaleString()}
                    </span>
                  </span>
                </div>
              )}
            </>
          ) : (
            <div className="flex items-center gap-2 p-3 rounded-lg bg-red-950/30 border border-red-500/20">
              <AlertCircle size={14} className="text-red-400 shrink-0" />
              <div>
                <div className="text-xs font-semibold text-red-300">No Solution</div>
                <div className="text-[10px] text-red-400/70 mt-0.5">ILP solver timed out or found no feasible solution for this scenario.</div>
              </div>
            </div>
          )}
        </div>

        {/* AWB list */}
        <div className="p-4 flex-1">
          <div className="text-[9px] font-mono tracking-widest text-[var(--text3)] uppercase mb-3">
            AWB List ({scenario.awbs.length})
          </div>
          <div className="space-y-1.5">
            {scenario.awbs.map((awb, i) => (
              <div key={awb.awb_id}
                className="flex items-center gap-2 p-2.5 rounded-lg bg-[var(--bg3)] border border-[var(--border)] hover:border-[var(--border2)] transition-colors cursor-default">
                <span className="w-5 h-5 rounded-full bg-[var(--bg4)] flex items-center justify-center font-mono text-[9px] text-[var(--text3)] shrink-0">
                  {i + 1}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="text-xs font-medium text-[var(--text)] truncate">{awb.location_name}</div>
                  <div className="text-[10px] font-mono text-[var(--text3)] truncate">{awb.awb_id}</div>
                </div>
                <div className="text-right shrink-0">
                  <div className="text-[10px] font-mono text-emerald-400">{awb.total_weight.toFixed(0)}kg</div>
                  <div className="text-[10px] font-mono text-purple-400">{awb.total_volume.toFixed(3)}cbm</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main content: map */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Map header */}
        <div className="px-5 py-3 border-b border-[var(--border)] flex items-center gap-3 shrink-0"
          style={{ background: 'var(--bg2)' }}>
          <MapPin size={14} style={{ color: depotColor }} />
          <span className="font-semibold text-sm">{scenario.depot} · {scenario.date}</span>
          <span className="text-xs text-[var(--text3)] font-mono">{scenario.awbs.length} delivery locations</span>
          <div className="ml-auto flex items-center gap-2">
            <span className="text-[10px] font-mono text-[var(--text3)]">
              {method === 'heuristic'
                ? `K=${scenario.heuristic.k_clusters ?? '—'} clusters · OSRM distances`
                : `OR-Tools · 60s limit`}
            </span>
            <Badge variant={method === 'heuristic' ? 'orange' : 'blue'}>
              {method === 'heuristic' ? 'Heuristic' : 'ILP'}
            </Badge>
          </div>
        </div>

        {/* Leaflet map */}
        <div id="scenario-map" className="flex-1" />

        {/* Legend */}
        <div className="px-5 py-2.5 border-t border-[var(--border)] flex items-center gap-4 text-[10px] font-mono text-[var(--text3)]"
          style={{ background: 'var(--bg2)' }}>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-yellow-400 inline-block" />
            Depot
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-blue-400 inline-block" />
            Delivery Location
          </div>
          <span className="ml-auto text-[var(--text3)]">Hover / click marker to see AWB details</span>
        </div>
      </div>
    </div>
  )
}
