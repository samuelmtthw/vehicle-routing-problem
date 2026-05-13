import type { MethodResult } from '../types'

export const fmtKm  = (v: number) => v.toLocaleString('id-ID', { maximumFractionDigits: 1 }) + ' km'
export const fmtRp  = (v: number) => 'Rp ' + Math.round(v).toLocaleString('id-ID')
export const fmtPct = (v: number) => v.toFixed(1) + '%'
export const fmtS   = (v: number) => v.toFixed(2) + 's'
export const fmtNum = (v: number) => v.toLocaleString('id-ID')

export const VEHICLE_COLORS: Record<string, string> = {
  'Blind Van':    '#4f8ef7',
  'Pickup Bak':   '#22d3a0',
  'Engkel (CDE)': '#f97316',
  'CDD Box':      '#a855f7',
}

export const DEPOT_COLORS: Record<string, string> = {
  'Depot Balikpapan': '#4f8ef7',
  'Depot Batam':      '#22d3a0',
  'Depot Jatiasih':   '#a855f7',
  'Depot Makassar':   '#f97316',
  'Depot Manado':     '#ef4444',
  'Depot Medan':      '#eab308',
  'Depot Pekanbaru':  '#06b6d4',
}

export const SOURCE_COLORS: Record<string, string> = {
  'COMPANY-1': '#22d3a0',
  'COMPANY-2': '#a855f7',
}

export function pctDiff(a: number, b: number) {
  if (a === 0) return 0
  return ((b - a) / a) * 100
}

export function isMethodResult(x: MethodResult | { status: 'no_solution' } | null): x is MethodResult {
  if (!x) return false
  return 'vehicles_used' in x
}
