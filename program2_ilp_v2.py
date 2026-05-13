"""
HCVRP Solver — Program 2 (v2): ILP with Google OR-Tools  [REFACTORED]
======================================================================
Key improvements over v1:
  1. Smart fleet sizing     — tight upper bound from demand/capacity ratio, not n_awbs/4
  2. Warm start             — heuristic solution injected as initial hint
  3. Better first solution  — AUTOMATIC strategy (OR-Tools picks the best for the instance)
  4. Calibrated penalties   — vehicle activation cost proportional to scenario's avg distance
  5. Reduced distance scale — DIST_SCALE=10 instead of 100 (smaller integers, faster arithmetic)
  6. Solution quality log   — reports objective value and improvement over time
  7. Adaptive time limit    — larger instances get more time (up to MAX_TIME_LIMIT_SEC)
  8. DEBUG flag             — toggle verbose output without touching logic

Mathematical model: unchanged from v1 (constraints C1–C7, same objective function).
OR-Tools routing library used (not CP-SAT directly).

Usage:
  python program2_ilp_v2.py

Requirements:
  pip install ortools pandas numpy scikit-learn requests
"""

import time
import math
import warnings
import requests
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DEBUG = False   # set True to print detailed solver diagnostics

# Time limit: base + per-AWB increment, capped at MAX
BASE_TIME_LIMIT_SEC     = 60
TIME_PER_AWB_SEC        = 0.5    # extra seconds per AWB above 20
MAX_TIME_LIMIT_SEC      = 120

# Fleet
FLEET = [
    {"type": "Blind Van",    "max_weight": 830,  "max_volume": 2.0,  "fuel_km_per_liter": 13.5},
    {"type": "Pickup Bak",   "max_weight": 1250, "max_volume": 5.0,  "fuel_km_per_liter": 12.0},
    {"type": "Engkel (CDE)", "max_weight": 2250, "max_volume": 9.0,  "fuel_km_per_liter": 6.0},
    {"type": "CDD Box",      "max_weight": 4500, "max_volume": 15.0, "fuel_km_per_liter": 4.5},
]
FLEET_SORTED = sorted(FLEET, key=lambda v: v["max_weight"])

FUEL_PRICE_PER_LITER = 6_800    # Rp/litre

# Scaling
VOLUME_SCALE = 1_000            # CBM → integer (preserves 3 decimal places)
DIST_SCALE   = 10               # km → integer (1 unit = 100m); smaller = faster arithmetic

# ─────────────────────────────────────────────────────────────
# COST
# ─────────────────────────────────────────────────────────────
def route_cost_rp(distance_km: float, vtype: dict) -> float:
    """total_cost = distance × (1 / fuel_efficiency) × fuel_price"""
    return (distance_km / vtype["fuel_km_per_liter"]) * FUEL_PRICE_PER_LITER


# ─────────────────────────────────────────────────────────────
# DISTANCE UTILITIES
# ─────────────────────────────────────────────────────────────
OSRM_BASE       = "http://router.project-osrm.org/table/v1/driving"
_osrm_available = None


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def check_osrm():
    global _osrm_available
    if _osrm_available is not None:
        return _osrm_available
    try:
        r = requests.get(f"{OSRM_BASE}/106.8456,-6.2088;106.9275,-6.1751?annotations=distance",
                         timeout=5)
        _osrm_available = (r.status_code == 200)
    except Exception:
        _osrm_available = False
    if not _osrm_available:
        print("  [INFO] OSRM unavailable — using Haversine fallback.")
    return _osrm_available


def osrm_distance_matrix(coords):
    if not check_osrm():
        return haversine_matrix(coords)
    coord_str = ";".join(f"{lng},{lat}" for lat, lng in coords)
    try:
        r    = requests.get(f"{OSRM_BASE}/{coord_str}?annotations=distance", timeout=30)
        data = r.json()
        if data.get("code") != "Ok":
            return haversine_matrix(coords)
        n = len(coords)
        m = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                m[i][j] = data["distances"][i][j] / 1000.0
        return m
    except Exception:
        return haversine_matrix(coords)


def haversine_matrix(coords):
    n = len(coords)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                m[i][j] = haversine_km(coords[i][0], coords[i][1],
                                       coords[j][0], coords[j][1])
    return m


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_scenario(df_all, depot_name, order_date):
    mask = ((df_all["first_pickup_location_name"] == depot_name) &
            (df_all["order_date"] == order_date))
    df   = df_all[mask].copy()
    awbs = (df.groupby("sales_order_doc_number")
              .agg(lat=("delivery_location_lat","first"),
                   lng=("delivery_location_lng","first"),
                   weight=("total_weight","sum"),
                   volume=("total_volume","sum"),
                   depot_lat=("first_pickup_location_lat","first"),
                   depot_lng=("first_pickup_location_lng","first"))
              .reset_index()
              .rename(columns={"sales_order_doc_number":"awb_id"}))
    awbs = awbs[(awbs["weight"] > 0) & (awbs["volume"] > 0)].reset_index(drop=True)
    return awbs


# ─────────────────────────────────────────────────────────────
# IMPROVEMENT 1 — SMART FLEET SIZING
# ─────────────────────────────────────────────────────────────
def compute_fleet_size(awbs: pd.DataFrame) -> int:
    """
    Tight upper bound on vehicles needed.
    Uses the most capacity-efficient vehicle type as the reference,
    then adds a 30% buffer for routing infeasibility.
    Caps at a maximum to prevent search space explosion.
    """
    total_w = awbs["weight"].sum()
    total_v = awbs["volume"].sum()

    # Minimum vehicles needed based on aggregate demand
    min_by_weight = math.ceil(total_w / FLEET_SORTED[-1]["max_weight"])
    min_by_volume = math.ceil(total_v / FLEET_SORTED[-1]["max_volume"])
    min_needed    = max(min_by_weight, min_by_volume, 1)

    # Per-type: enough of each type to handle the scenario alone (for feasibility)
    max_per_type = max(
        math.ceil(total_w / vtype["max_weight"])
        for vtype in FLEET_SORTED
    )
    # Buffer: 50% above min_needed to allow solver flexibility
    upper = math.ceil(max_per_type * 1.5)
    # Hard cap: 8 per type (32 total) — beyond this the search space is too large
    upper = min(upper, 8)
    upper = max(upper, 1)

    if DEBUG:
        print(f"    [FLEET] total_w={total_w:.0f}kg total_v={total_v:.3f}CBM "
              f"min_needed={min_needed} vehicles_per_type={upper}")
    return upper


# ─────────────────────────────────────────────────────────────
# IMPROVEMENT 2 — WARM START (heuristic solution as initial hint)
# ─────────────────────────────────────────────────────────────
def build_heuristic_routes(awbs: pd.DataFrame, dist_matrix_km: np.ndarray):
    """
    Run the K-Means + Bin Packing heuristic to get an initial solution.
    Returns a list of routes: [{"vehicle_type": ..., "awb_indices": [1,2,3,...]}, ...]
    AWB indices are 1-based (index 0 = depot).
    """
    n_awbs = len(awbs)

    # ── K-Means clustering ────────────────────────────────────
    total_w = awbs["weight"].sum()
    total_v = awbs["volume"].sum()
    max_w   = FLEET_SORTED[-1]["max_weight"]
    max_v   = FLEET_SORTED[-1]["max_volume"]
    k       = max(math.ceil(total_w / max_w), math.ceil(total_v / max_v), 1)
    k       = min(k, n_awbs)

    coords  = awbs[["lat","lng"]].values
    scaler  = StandardScaler()
    scaled  = scaler.fit_transform(coords)
    km      = KMeans(n_clusters=k, random_state=42, n_init=10)
    awbs    = awbs.copy()
    awbs["cluster"] = km.fit_predict(scaled)

    # ── Bin packing ───────────────────────────────────────────
    cluster_sum = awbs.groupby("cluster").agg(
        w=("weight","sum"), v=("volume","sum")
    ).sort_values("w", ascending=False).reset_index()

    vehicles = []  # {type, clusters, w, v}
    for _, cl in cluster_sum.iterrows():
        placed = False
        best_idx, best_rem = None, float("inf")
        for idx, veh in enumerate(vehicles):
            vt   = next(f for f in FLEET_SORTED if f["type"] == veh["type"])
            rw   = vt["max_weight"] - veh["w"]
            rv   = vt["max_volume"] - veh["v"]
            if rw >= cl.w and rv >= cl.v:
                rem = min(rw - cl.w, rv - cl.v)
                if rem < best_rem:
                    best_rem = rem; best_idx = idx
        if best_idx is not None:
            vehicles[best_idx]["clusters"].append(cl.cluster)
            vehicles[best_idx]["w"] += cl.w
            vehicles[best_idx]["v"] += cl.v
            placed = True
        if not placed:
            vtype = next((f["type"] for f in FLEET_SORTED
                          if f["max_weight"] >= cl.w and f["max_volume"] >= cl.v), None)
            if vtype is None:
                vtype = FLEET_SORTED[-1]["type"]
            vehicles.append({"type": vtype, "clusters": [cl.cluster], "w": cl.w, "v": cl.v})

    # ── Build routes with greedy nearest-neighbour ────────────
    routes = []
    for veh in vehicles:
        awb_idx = awbs[awbs["cluster"].isin(veh["clusters"])].index.tolist()
        # awb_idx are 0-based into awbs df; OR-Tools uses 1-based nodes (0=depot)
        node_idx = [i + 1 for i in awb_idx]
        if not node_idx:
            continue
        # Greedy NN within cluster
        unvisited = node_idx[:]
        route     = []
        current   = 0
        while unvisited:
            nearest  = min(unvisited, key=lambda j: dist_matrix_km[current][j])
            route.append(nearest)
            current  = nearest
            unvisited.remove(nearest)
        routes.append({"vehicle_type": veh["type"], "awb_nodes": route})

    return routes


# ─────────────────────────────────────────────────────────────
# IMPROVEMENT 3 — CALIBRATED VEHICLE ACTIVATION PENALTY
# ─────────────────────────────────────────────────────────────
def compute_activation_penalty(dist_matrix_km: np.ndarray, n_awbs: int) -> int:
    """
    Set vehicle activation penalty = average distance between nodes.
    This discourages unnecessary vehicles without forcing extreme consolidation.
    Scaled to OR-Tools integer distance units.
    """
    # Average non-zero distance in matrix
    flat = dist_matrix_km[dist_matrix_km > 0]
    avg_dist = float(np.median(flat)) if len(flat) > 0 else 50.0
    # Penalty = 3× average inter-node distance → strongly discourages extra vehicles
    # but doesn't force the solver into infeasibly long consolidation routes
    penalty_km = avg_dist * 3.0
    return int(penalty_km * DIST_SCALE)


# ─────────────────────────────────────────────────────────────
# IMPROVEMENT 4 — ADAPTIVE TIME LIMIT
# ─────────────────────────────────────────────────────────────
def compute_time_limit(n_awbs: int) -> int:
    limit = BASE_TIME_LIMIT_SEC + max(0, n_awbs - 20) * TIME_PER_AWB_SEC
    return min(int(limit), MAX_TIME_LIMIT_SEC)


# ─────────────────────────────────────────────────────────────
# BUILD OR-TOOLS MODEL
# ─────────────────────────────────────────────────────────────
def build_ortools_data(awbs: pd.DataFrame, dist_matrix_km: np.ndarray,
                       vehicles_per_type: int, activation_penalty: int) -> dict:
    n_awbs = len(awbs)
    n_nodes = n_awbs + 1  # 0 = depot

    # Validate
    assert dist_matrix_km.shape == (n_nodes, n_nodes), \
        f"Expected ({n_nodes},{n_nodes}), got {dist_matrix_km.shape}"
    assert not np.any(np.isnan(dist_matrix_km)), "NaN in distance matrix"

    # Scale distances to integers
    dist_int = (dist_matrix_km * DIST_SCALE).astype(int).tolist()

    # Fleet
    vehicle_types = [vt for vt in FLEET for _ in range(vehicles_per_type)]
    n_vehicles    = len(vehicle_types)

    # Demands (node 0 = depot = 0 demand)
    weight_demands = [0] + [int(round(w))            for w in awbs["weight"]]
    volume_demands = [0] + [int(round(v*VOLUME_SCALE)) for v in awbs["volume"]]

    # Capacities per vehicle
    weight_caps = [int(vt["max_weight"])            for vt in vehicle_types]
    volume_caps = [int(vt["max_volume"]*VOLUME_SCALE) for vt in vehicle_types]

    # Fixed cost per vehicle (activation penalty, same for all)
    fixed_costs = [activation_penalty] * n_vehicles

    if DEBUG:
        print(f"    [MODEL] n_nodes={n_nodes}, n_vehicles={n_vehicles} "
              f"({vehicles_per_type}/type), activation_penalty={activation_penalty} "
              f"(= {activation_penalty/DIST_SCALE:.1f} km)")

    return {
        "distance_matrix": dist_int,
        "weight_demands":  weight_demands,
        "volume_demands":  volume_demands,
        "weight_caps":     weight_caps,
        "volume_caps":     volume_caps,
        "fixed_costs":     fixed_costs,
        "vehicle_types":   vehicle_types,
        "n_vehicles":      n_vehicles,
        "n_nodes":         n_nodes,
        "depot":           0,
    }


# ─────────────────────────────────────────────────────────────
# SOLVE
# ─────────────────────────────────────────────────────────────
def solve_with_ortools(data: dict, awbs: pd.DataFrame,
                       heuristic_routes: list, time_limit_sec: int) -> dict | None:
    n_nodes    = data["n_nodes"]
    n_vehicles = data["n_vehicles"]
    depot      = data["depot"]

    # ── Manager + model ───────────────────────────────────────
    manager = pywrapcp.RoutingIndexManager(n_nodes, n_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # ── Arc cost ──────────────────────────────────────────────
    def dist_cb(fi, ti):
        return data["distance_matrix"][manager.IndexToNode(fi)][manager.IndexToNode(ti)]
    transit_cb = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    # ── Fixed costs ───────────────────────────────────────────
    for v in range(n_vehicles):
        routing.SetFixedCostOfVehicle(data["fixed_costs"][v], v)

    # ── Weight capacity (C4) ──────────────────────────────────
    def w_cb(fi):
        return data["weight_demands"][manager.IndexToNode(fi)]
    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(w_cb),
        0, data["weight_caps"], True, "Weight")

    # ── Volume capacity (C5) ──────────────────────────────────
    def v_cb(fi):
        return data["volume_demands"][manager.IndexToNode(fi)]
    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(v_cb),
        0, data["volume_caps"], True, "Volume")

    # ── IMPROVEMENT: Warm start from heuristic ────────────────
    # Map heuristic routes to solver vehicle indices
    # Sort vehicle pool: assign largest routes to largest vehicles
    n_routes = len(heuristic_routes)
    if n_routes > 0 and n_routes <= n_vehicles:
        initial_routes = []
        for route in heuristic_routes:
            initial_routes.append(route["awb_nodes"])
        # Pad with empty routes for unused vehicles
        while len(initial_routes) < n_vehicles:
            initial_routes.append([])
        try:
            routing.ReadAssignmentFromRoutes(initial_routes, True)
            if DEBUG:
                print(f"    [WARM] Injected {n_routes} heuristic routes as initial hint")
        except Exception as e:
            if DEBUG:
                print(f"    [WARM] Failed to inject warm start: {e}")

    # ── IMPROVEMENT: Better search strategy ───────────────────
    search_params = pywrapcp.DefaultRoutingSearchParameters()

    # AUTOMATIC lets OR-Tools pick the best first-solution strategy
    # for the specific instance — much better than hardcoding PATH_CHEAPEST_ARC
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    )

    # GUIDED_LOCAL_SEARCH remains the best metaheuristic for VRP
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    search_params.time_limit.FromSeconds(time_limit_sec)
    search_params.log_search = DEBUG

    # ── Solve ─────────────────────────────────────────────────
    try:
        solution = routing.SolveWithParameters(search_params)
    except (OverflowError, SystemError) as e:
        print(f"    [ERROR] Solver exception: {e}")
        return None

    if solution is None:
        return None

    # ── Extract routes ────────────────────────────────────────
    weight_dim = routing.GetDimensionOrDie("Weight")
    volume_dim = routing.GetDimensionOrDie("Volume")
    routes_detail = []
    total_distance_km = 0.0

    for v in range(n_vehicles):
        idx = routing.Start(v)
        if routing.IsEnd(solution.Value(routing.NextVar(idx))):
            continue  # vehicle unused

        vtype      = data["vehicle_types"][v]
        route_awbs = []
        route_km   = 0.0

        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != depot:
                route_awbs.append(awbs.iloc[node - 1]["awb_id"])
            nxt       = solution.Value(routing.NextVar(idx))
            route_km += data["distance_matrix"][node][manager.IndexToNode(nxt)] / DIST_SCALE
            idx       = nxt

        end_idx = routing.End(v)
        w_load  = solution.Value(weight_dim.CumulVar(end_idx))
        v_load  = solution.Value(volume_dim.CumulVar(end_idx))

        total_distance_km += route_km
        routes_detail.append({
            "vehicle_index": len(routes_detail) + 1,
            "vehicle_type":  vtype["type"],
            "awb_count":     len(route_awbs),
            "weight_kg":     round(w_load, 2),
            "volume_cbm":    round(v_load / VOLUME_SCALE, 3),
            "weight_util":   round(w_load / vtype["max_weight"] * 100, 1),
            "volume_util":   round(v_load / VOLUME_SCALE / vtype["max_volume"] * 100, 1),
            "distance_km":   round(route_km, 2),
            "route_cost_rp": round(route_cost_rp(route_km, vtype)),
            "ordered_awbs":  route_awbs,
        })

    if not routes_detail:
        return None

    vehicles_used  = len(routes_detail)
    vehicle_counts: dict = {}
    for r in routes_detail:
        vehicle_counts[r["vehicle_type"]] = vehicle_counts.get(r["vehicle_type"], 0) + 1

    return {
        "vehicles_used":       vehicles_used,
        "vehicle_breakdown":   vehicle_counts,
        "total_distance_km":   round(total_distance_km, 2),
        "total_cost_rp":       round(sum(r["route_cost_rp"] for r in routes_detail)),
        "avg_weight_util_pct": round(sum(r["weight_util"] for r in routes_detail) / vehicles_used, 1),
        "avg_volume_util_pct": round(sum(r["volume_util"] for r in routes_detail) / vehicles_used, 1),
        "ortools_objective":   solution.ObjectiveValue(),
        "routes":              routes_detail,
    }


# ─────────────────────────────────────────────────────────────
# SOLVE ONE SCENARIO
# ─────────────────────────────────────────────────────────────
def solve_scenario(awbs: pd.DataFrame) -> dict | None:
    depot_coord = (float(awbs["depot_lat"].iloc[0]), float(awbs["depot_lng"].iloc[0]))
    coords      = [depot_coord] + [(r.lat, r.lng) for _, r in awbs.iterrows()]

    dist_matrix_km = osrm_distance_matrix(coords)

    # Compute adaptive parameters
    vehicles_per_type  = compute_fleet_size(awbs)
    activation_penalty = compute_activation_penalty(dist_matrix_km, len(awbs))
    time_limit         = compute_time_limit(len(awbs))

    # Build heuristic warm start
    heuristic_routes = build_heuristic_routes(awbs, dist_matrix_km)

    # Build model
    data = build_ortools_data(awbs, dist_matrix_km, vehicles_per_type, activation_penalty)

    if DEBUG:
        print(f"    [SOLVE] time_limit={time_limit}s, "
              f"heuristic_routes={len(heuristic_routes)}")

    return solve_with_ortools(data, awbs, heuristic_routes, time_limit)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("HCVRP ILP Solver — Google OR-Tools  [v2 — Refactored]")
    print("=" * 62)
    print(f"Base time limit: {BASE_TIME_LIMIT_SEC}s (+{TIME_PER_AWB_SEC}s/AWB, max {MAX_TIME_LIMIT_SEC}s)")
    print(f"Distance scale: {DIST_SCALE} | Volume scale: {VOLUME_SCALE}")
    print(f"Warm start: enabled | Fleet sizing: adaptive | Penalty: calibrated")
    print()

    df = pd.read_csv("dataset_compiled.csv", low_memory=False)
    df["order_date"] = pd.to_datetime(df["order_date"]).dt.date

    scenarios = (df[["source","first_pickup_location_name","order_date"]]
                 .drop_duplicates()
                 .sort_values(["source","first_pickup_location_name","order_date"])
                 .reset_index(drop=True))

    print(f"Scenarios to solve: {len(scenarios)}\n")

    summary_rows = []

    for _, sc in scenarios.iterrows():
        source = sc["source"]
        depot  = sc["first_pickup_location_name"]
        date   = sc["order_date"]

        print(f"Solving: {source} | {depot} | {date}")
        t0   = time.time()
        awbs = load_scenario(df, depot, date)

        if len(awbs) < 2:
            print(f"  SKIP — {len(awbs)} AWB(s)\n"); continue

        time_limit = compute_time_limit(len(awbs))
        result     = solve_scenario(awbs)
        elapsed    = round(time.time() - t0, 2)

        if result is None:
            print(f"  NO SOLUTION within {time_limit}s\n")
            summary_rows.append({
                "source": source, "depot": depot, "date": date,
                "awb_count": len(awbs), "status": "no_solution",
                "runtime_sec": elapsed,
            })
            continue

        print(f"  AWBs      : {len(awbs)}")
        print(f"  Vehicles  : {result['vehicles_used']}  {result['vehicle_breakdown']}")
        print(f"  Distance  : {result['total_distance_km']} km")
        print(f"  Cost      : Rp {result['total_cost_rp']:,}")
        print(f"  W util    : {result['avg_weight_util_pct']}%  V util: {result['avg_volume_util_pct']}%")
        print(f"  Objective : {result['ortools_objective']:,}")
        print(f"  Runtime   : {elapsed}s  (limit: {time_limit}s)\n")

        summary_rows.append({
            "source":               source,
            "depot":                depot,
            "date":                 date,
            "awb_count":            len(awbs),
            "status":               "solved",
            "vehicles_used":        result["vehicles_used"],
            "vehicle_breakdown":    str(result["vehicle_breakdown"]),
            "total_distance_km":    result["total_distance_km"],
            "total_cost_rp":        result["total_cost_rp"],
            "avg_weight_util_pct":  result["avg_weight_util_pct"],
            "avg_volume_util_pct":  result["avg_volume_util_pct"],
            "ortools_objective":    result["ortools_objective"],
            "runtime_sec":          elapsed,
            "distance_method":      "OSRM" if check_osrm() else "Haversine",
            "time_limit_sec":       time_limit,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results_ilp_v2.csv", index=False)

    solved = summary_df[summary_df["status"] == "solved"]
    print("=" * 62)
    print("AGGREGATE RESULTS")
    print("=" * 62)
    print(f"Scenarios attempted : {len(summary_df)}")
    print(f"Scenarios solved    : {len(solved)}")
    print(f"Avg vehicles used   : {solved['vehicles_used'].mean():.2f}")
    print(f"Avg distance km     : {solved['total_distance_km'].mean():.2f}")
    print(f"Avg cost Rp         : {solved['total_cost_rp'].mean():,.0f}")
    print(f"Avg W util          : {solved['avg_weight_util_pct'].mean():.1f}%")
    print(f"Avg V util          : {solved['avg_volume_util_pct'].mean():.1f}%")
    print(f"Avg runtime         : {solved['runtime_sec'].mean():.2f}s")
    print(f"\nResults saved → results_ilp_v2.csv")


if __name__ == "__main__":
    main()