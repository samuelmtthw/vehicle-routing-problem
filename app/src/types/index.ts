export interface VehicleBreakdown { [key: string]: number }

export interface MethodResult {
  vehicles_used: number
  vehicle_breakdown: VehicleBreakdown
  total_distance_km: number
  total_cost_rp: number
  avg_weight_util_pct: number
  avg_volume_util_pct: number
  runtime_sec: number
  distance_method?: string
  status?: string
  k_clusters?: number
  ortools_objective?: number
}

export interface AWBItem {
  quantity: number
  unit: string
  weight_kg: number
  volume_cbm: number
}

export interface AWB {
  awb_id: string
  lat: number
  lng: number
  location_name: string
  address: string
  city: string
  province: string
  customer: string
  total_weight: number
  total_volume: number
  items: AWBItem[]
}

export interface ScenarioMeta {
  id: string
  source: string
  depot: string
  date: string
  depot_lat: number
  depot_lng: number
  depot_city: string
  depot_province: string
  awb_count: number
  heuristic: MethodResult
  ilp: MethodResult | { status: 'no_solution' } | null
}

export interface Scenario extends ScenarioMeta {
  awbs: AWB[]
}
