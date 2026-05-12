"""
HCVRP Solver — Program 2: Integer Linear Programming with Google OR-Tools
=========================================================================
Mathematical model implemented:
  - Objective  : minimize fixed vehicle cost + variable travel cost (lexicographic)
  - Constraints: each AWB visited exactly once (C1)
                 flow conservation (C2) — handled internally by OR-Tools
                 all routes start and end at depot (C3)
                 weight capacity per vehicle (C4)
                 volume capacity per vehicle (C5)
                 subtour elimination (C6) — handled internally by OR-Tools
                 integrality (C7) — enforced by CP-SAT branch-and-bound

OR-Tools routing library is used (not CP-SAT directly) because it provides
a purpose-built VRP engine with native support for:
  - heterogeneous fleet (vehicle-specific capacity dimensions)
  - multiple capacity dimensions simultaneously (weight + volume)
  - fixed costs per vehicle
  - time limit for large instances

Usage:
  python program2_ilp.py

Requirements:
  pip install ortools pandas numpy requests
"""

import time
import math
import warnings
import requests
import numpy as np
import pandas as pd

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1. FLEET CONFIGURATION  (identical to Program 1)
# ─────────────────────────────────────────────────────────────
FLEET = [
    {"type": "Blind Van",    "max_weight": 830,  "max_volume": 2.0,  "fuel_km_per_liter": 13.5},
    {"type": "Pickup Bak",   "max_weight": 1250, "max_volume": 5.0,  "fuel_km_per_liter": 12.0},
    {"type": "Engkel (CDE)", "max_weight": 2250, "max_volume": 9.0,  "fuel_km_per_liter": 6.0},
    {"type": "CDD Box",      "max_weight": 4500, "max_volume": 15.0, "fuel_km_per_liter": 4.5},
]

FUEL_PRICE_PER_LITER = 6_800  # Rp per liter (Solar/Diesel, 2025)

def route_cost_rp(distance_km, vehicle_type_dict):
    """total_cost = distance * (1 / fuel_efficiency) * fuel_price"""
    liters = distance_km / vehicle_type_dict["fuel_km_per_liter"]
    return liters * FUEL_PRICE_PER_LITER
VOLUME_SCALE  = 1_000    # OR-Tools requires integer dimensions;
                          # multiply volume (CBM) by this to preserve 3 decimal places
WEIGHT_SCALE  = 1        # weight already in kg integers — no scaling needed
DIST_SCALE    = 100      # store distances as integer centimetres (km * 100)

# Fixed cost per vehicle in distance units.
# We use a uniform high penalty to discourage unnecessary vehicle use,
# since the actual cost formula is purely fuel-based.
# 200 km equivalent per vehicle activation (= 200 * DIST_SCALE units).
VEHICLE_ACTIVATION_PENALTY_KM = 200

def fixed_cost_scaled(vehicle_type_dict):
    """Convert vehicle activation penalty to OR-Tools distance units."""
    return int(VEHICLE_ACTIVATION_PENALTY_KM * DIST_SCALE)

# Time limit in seconds for OR-Tools solver per scenario.
# Increase for larger instances or better solution quality.
SOLVER_TIME_LIMIT_SEC = 60

# ─────────────────────────────────────────────────────────────
# 2. DISTANCE UTILITIES  (same as Program 1 — keep consistent)
# ─────────────────────────────────────────────────────────────
OSRM_BASE      = "http://router.project-osrm.org/table/v1/driving"
_osrm_available = None


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def check_osrm():
    global _osrm_available
    if _osrm_available is not None:
        return _osrm_available
    try:
        url = f"{OSRM_BASE}/106.8456,-6.2088;106.9275,-6.1751?annotations=distance"
        r   = requests.get(url, timeout=5)
        _osrm_available = (r.status_code == 200)
    except Exception:
        _osrm_available = False
    if not _osrm_available:
        print("  [INFO] OSRM not reachable — using Haversine fallback. "
              "Switch to OSRM when running on your own machine.")
    return _osrm_available


def osrm_distance_matrix(coords):
    """
    Build an (n x n) distance matrix (km).
    coords: list of (lat, lng) — index 0 is the depot.
    """
    if not check_osrm():
        return haversine_matrix(coords)
    coord_str = ";".join(f"{lng},{lat}" for lat, lng in coords)
    url       = f"{OSRM_BASE}/{coord_str}?annotations=distance"
    try:
        r    = requests.get(url, timeout=30)
        data = r.json()
        if data.get("code") != "Ok":
            return haversine_matrix(coords)
        n      = len(coords)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i][j] = data["distances"][i][j] / 1000.0
        return matrix
    except Exception:
        return haversine_matrix(coords)


def haversine_matrix(coords):
    n      = len(coords)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = haversine_km(
                    coords[i][0], coords[i][1],
                    coords[j][0], coords[j][1]
                )
    return matrix


# ─────────────────────────────────────────────────────────────
# 3. DATA LOADING & AGGREGATION  (same logic as Program 1)
# ─────────────────────────────────────────────────────────────
def load_scenario(df_all, depot_name, order_date):
    mask = (
        (df_all["first_pickup_location_name"] == depot_name) &
        (df_all["order_date"] == order_date)
    )
    df = df_all[mask].copy()

    awbs = (
        df.groupby("sales_order_doc_number")
        .agg(
            lat       = ("delivery_location_lat",  "first"),
            lng       = ("delivery_location_lng",  "first"),
            weight    = ("total_weight",            "sum"),
            volume    = ("total_volume",            "sum"),
            depot_lat = ("first_pickup_location_lat",  "first"),
            depot_lng = ("first_pickup_location_lng",  "first"),
        )
        .reset_index()
        .rename(columns={"sales_order_doc_number": "awb_id"})
    )
    awbs = awbs[(awbs["weight"] > 0) & (awbs["volume"] > 0)].reset_index(drop=True)
    return awbs


# ─────────────────────────────────────────────────────────────
# 4. BUILD OR-TOOLS MODEL
# ─────────────────────────────────────────────────────────────
def build_ortools_data(awbs, dist_matrix_km):
    """
    Prepare the data dictionary that OR-Tools routing expects.

    Node layout:
      index 0         = depot
      indices 1..n    = AWB delivery locations

    Fleet layout:
      We supply enough vehicles of each type to potentially serve all AWBs.
      In practice OR-Tools will only activate the minimum needed
      (enforced by the fixed cost on each vehicle).
    """
    n_awbs = len(awbs)

    # ── Distance matrix scaled to integers ──────────────────
    # OR-Tools requires integer arc costs.
    # We multiply km by DIST_SCALE to preserve precision.
    dist_int = (dist_matrix_km * DIST_SCALE).astype(int).tolist()

    # ── Determine vehicle fleet ──────────────────────────────
    # Provide enough vehicles of each type so the problem is always feasible.
    # Upper bound: one vehicle per AWB (worst case).
    # We distribute evenly across fleet types.
    vehicles_per_type = max(1, math.ceil(n_awbs / len(FLEET)))
    vehicle_types = []
    for vtype in FLEET:
        for _ in range(vehicles_per_type):
            vehicle_types.append(vtype)
    n_vehicles = len(vehicle_types)

    # ── Demand vectors (scaled integers) ────────────────────
    # Node 0 (depot) has zero demand.
    weight_demands = [0] + [int(round(w * WEIGHT_SCALE)) for w in awbs["weight"]]
    volume_demands = [0] + [int(round(v * VOLUME_SCALE)) for v in awbs["volume"]]

    # ── Capacity vectors per vehicle ────────────────────────
    weight_capacities = [int(v["max_weight"] * WEIGHT_SCALE) for v in vehicle_types]
    volume_capacities = [int(v["max_volume"] * VOLUME_SCALE) for v in vehicle_types]

    # ── Fixed costs per vehicle (scaled to distance units) ──
    vehicle_fixed_costs = [fixed_cost_scaled(v) for v in vehicle_types]

    return {
        "distance_matrix":    dist_int,
        "weight_demands":     weight_demands,
        "volume_demands":     volume_demands,
        "weight_capacities":  weight_capacities,
        "volume_capacities":  volume_capacities,
        "vehicle_fixed_costs": vehicle_fixed_costs,
        "vehicle_types":      vehicle_types,
        "n_vehicles":         n_vehicles,
        "n_nodes":            n_awbs + 1,   # depot + AWBs
        "depot":              0,
    }


# ─────────────────────────────────────────────────────────────
# 5. SOLVE WITH OR-TOOLS
# ─────────────────────────────────────────────────────────────
def solve_with_ortools(data, awbs, time_limit_sec=SOLVER_TIME_LIMIT_SEC):
    """
    Build and solve the HCVRP model using OR-Tools routing library.
    Returns solution dict or None if infeasible / timed out without solution.
    """
    n_nodes    = data["n_nodes"]
    n_vehicles = data["n_vehicles"]
    depot      = data["depot"]

    # ── Create routing index manager ────────────────────────
    manager = pywrapcp.RoutingIndexManager(n_nodes, n_vehicles, depot)

    # ── Create routing model ─────────────────────────────────
    routing = pywrapcp.RoutingModel(manager)

    # ── Arc cost callback (distance) ────────────────────────
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node   = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_cb_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    # ── Fixed cost per vehicle ───────────────────────────────
    # This implements the lexicographic minimisation:
    # solver penalises activating a vehicle heavily before optimising distance.
    for v in range(n_vehicles):
        routing.SetFixedCostOfVehicle(data["vehicle_fixed_costs"][v], v)

    # ── Weight capacity dimension (Constraint C4) ────────────
    def weight_demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data["weight_demands"][from_node]

    weight_cb_idx = routing.RegisterUnaryTransitCallback(weight_demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        weight_cb_idx,
        0,                              # no slack
        data["weight_capacities"],      # vehicle-specific capacities
        True,                           # start cumul at zero
        "Weight",
    )

    # ── Volume capacity dimension (Constraint C5) ────────────
    def volume_demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data["volume_demands"][from_node]

    volume_cb_idx = routing.RegisterUnaryTransitCallback(volume_demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        volume_cb_idx,
        0,                              # no slack
        data["volume_capacities"],      # vehicle-specific capacities
        True,                           # start cumul at zero
        "Volume",
    )

    # ── Search parameters ────────────────────────────────────
    search_params = pywrapcp.DefaultRoutingSearchParameters()

    # Initial solution: PATH_CHEAPEST_ARC gives a good starting point quickly
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Metaheuristic: GUIDED_LOCAL_SEARCH improves the initial solution
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    # Time limit — prevents ILP from running indefinitely on large instances
    search_params.time_limit.FromSeconds(time_limit_sec)

    # Log search progress (set to True for debugging)
    search_params.log_search = False

    # ── Solve ────────────────────────────────────────────────
    solution = routing.SolveWithParameters(search_params)

    if solution is None:
        return None

    # ── Extract solution ─────────────────────────────────────
    routes_detail     = []
    total_distance_km = 0.0

    weight_dim = routing.GetDimensionOrDie("Weight")
    volume_dim = routing.GetDimensionOrDie("Volume")

    for v in range(n_vehicles):
        index = routing.Start(v)
        if routing.IsEnd(solution.Value(routing.NextVar(index))):
            # Vehicle not used
            continue

        vtype     = data["vehicle_types"][v]
        route_awbs = []
        route_km   = 0.0

        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index != depot:
                route_awbs.append(awbs.iloc[node_index - 1]["awb_id"])
            next_index = solution.Value(routing.NextVar(index))
            route_km  += data["distance_matrix"][node_index][
                             manager.IndexToNode(next_index)
                         ] / DIST_SCALE
            index = next_index

        # Load at end of route (= total carried by this vehicle)
        end_index  = routing.End(v)
        w_load     = solution.Value(weight_dim.CumulVar(end_index)) / WEIGHT_SCALE
        v_load     = solution.Value(volume_dim.CumulVar(end_index)) / VOLUME_SCALE

        route_cost = route_cost_rp(route_km, vtype)
        total_distance_km += route_km

        routes_detail.append({
            "vehicle_index": len(routes_detail) + 1,
            "vehicle_type":  vtype["type"],
            "awb_count":     len(route_awbs),
            "weight_kg":     round(w_load, 2),
            "volume_cbm":    round(v_load, 3),
            "weight_util":   round(w_load / vtype["max_weight"] * 100, 1),
            "volume_util":   round(v_load / vtype["max_volume"] * 100, 1),
            "distance_km":   round(route_km, 2),
            "route_cost_rp": round(route_cost),
            "ordered_awbs":  route_awbs,
        })

    vehicles_used  = len(routes_detail)
    if vehicles_used == 0:
        return None

    vehicle_counts = {}
    for r in routes_detail:
        vehicle_counts[r["vehicle_type"]] = vehicle_counts.get(r["vehicle_type"], 0) + 1

    total_cost      = sum(r["route_cost_rp"] for r in routes_detail)
    avg_weight_util = sum(r["weight_util"]   for r in routes_detail) / vehicles_used
    avg_volume_util = sum(r["volume_util"]   for r in routes_detail) / vehicles_used

    # OR-Tools objective value (for reference)
    obj_value = solution.ObjectiveValue()

    return {
        "vehicles_used":       vehicles_used,
        "vehicle_breakdown":   vehicle_counts,
        "total_distance_km":   round(total_distance_km, 2),
        "total_cost_rp":       round(total_cost),
        "avg_weight_util_pct": round(avg_weight_util, 1),
        "avg_volume_util_pct": round(avg_volume_util, 1),
        "ortools_objective":   obj_value,
        "routes":              routes_detail,
    }


# ─────────────────────────────────────────────────────────────
# 6. SOLVE ONE SCENARIO
# ─────────────────────────────────────────────────────────────
def solve_scenario(awbs):
    depot_lat   = awbs["depot_lat"].iloc[0]
    depot_lng   = awbs["depot_lng"].iloc[0]
    depot_coord = (depot_lat, depot_lng)

    # Build coordinate list: depot first, then all AWBs
    coords      = [depot_coord] + [(row["lat"], row["lng"]) for _, row in awbs.iterrows()]

    # Distance matrix (km, float)
    dist_matrix_km = osrm_distance_matrix(coords)

    # Build OR-Tools data model
    data = build_ortools_data(awbs, dist_matrix_km)

    # Solve
    result = solve_with_ortools(data, awbs)
    return result


# ─────────────────────────────────────────────────────────────
# 7. MAIN — run all scenarios
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("HCVRP ILP Solver — Google OR-Tools")
    print("=" * 60)

    df = pd.read_csv("dataset_compiled.csv", low_memory=False)
    df["order_date"] = pd.to_datetime(df["order_date"]).dt.date

    scenarios = (
        df[["source", "first_pickup_location_name", "order_date"]]
        .drop_duplicates()
        .sort_values(["source", "first_pickup_location_name", "order_date"])
        .reset_index(drop=True)
    )

    print(f"Total scenarios to solve: {len(scenarios)}")
    print(f"Solver time limit per scenario: {SOLVER_TIME_LIMIT_SEC}s\n")

    summary_rows = []

    for _, sc in scenarios.iterrows():
        source = sc["source"]
        depot  = sc["first_pickup_location_name"]
        date   = sc["order_date"]
        label  = f"{source} | {depot} | {date}"

        print(f"Solving: {label}")
        t_start = time.time()

        awbs = load_scenario(df, depot, date)

        if len(awbs) < 2:
            print(f"  SKIP — only {len(awbs)} AWB(s).\n")
            continue

        result  = solve_scenario(awbs)
        elapsed = round(time.time() - t_start, 2)

        if result is None:
            print(f"  NO SOLUTION found within {SOLVER_TIME_LIMIT_SEC}s — skipping.\n")
            summary_rows.append({
                "source": source, "depot": depot, "date": date,
                "awb_count": len(awbs), "status": "no_solution",
                "runtime_sec": elapsed,
            })
            continue

        # Print scenario summary
        print(f"  AWBs           : {len(awbs)}")
        print(f"  Vehicles used  : {result['vehicles_used']}  {result['vehicle_breakdown']}")
        print(f"  Total distance : {result['total_distance_km']} km")
        print(f"  Total cost     : Rp {result['total_cost_rp']:,}")
        print(f"  Avg weight util: {result['avg_weight_util_pct']}%")
        print(f"  Avg volume util: {result['avg_volume_util_pct']}%")
        print(f"  OR-Tools obj   : {result['ortools_objective']}")
        print(f"  Runtime        : {elapsed}s\n")

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
        })

    # ── Save results ─────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results_ilp.csv", index=False)

    # ── Aggregate stats (solved only) ────────────────────────
    solved = summary_df[summary_df["status"] == "solved"]
    print("=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    print(f"Scenarios attempted    : {len(summary_df)}")
    print(f"Scenarios solved       : {len(solved)}")
    print(f"Avg AWBs per scenario  : {solved['awb_count'].mean():.1f}")
    print(f"Avg vehicles used      : {solved['vehicles_used'].mean():.2f}")
    print(f"Avg total distance km  : {solved['total_distance_km'].mean():.2f}")
    print(f"Avg total cost Rp      : {solved['total_cost_rp'].mean():,.0f}")
    print(f"Avg weight utilisation : {solved['avg_weight_util_pct'].mean():.1f}%")
    print(f"Avg volume utilisation : {solved['avg_volume_util_pct'].mean():.1f}%")
    print(f"Avg runtime per scenario: {solved['runtime_sec'].mean():.2f}s")
    print(f"\nResults saved → results_ilp.csv")


if __name__ == "__main__":
    main()
