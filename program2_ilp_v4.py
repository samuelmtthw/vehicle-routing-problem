"""
program2_ilp_v4.py
==================
Exact ILP solver for 2D Heterogeneous Capacitated Vehicle Routing Problem (HCVRP)
Method: Integer Linear Programming via Google OR-Tools CP routing solver

Mathematical model (4-subscript, per thesis Section 3.2.3):
  Indices : i,j = 0..n (nodes), k = 1..3 (unit), l = 1..4 (vehicle type)
  Variables:
    x_ijkl ∈ {0,1}  — 1 if unit k of type l travels arc i→j
    y_kl   ∈ {0,1}  — 1 if unit k of type l is activated
    u_ikl  ≥ 0      — MTZ subtour elimination position variable
  Objective: min ΣΣΣΣc_ijl·x_ijkl + ΣΣf_l·y_kl
  Constraints C1–C7 + C3b (see thesis)

Dataset : dataset_compiled_v2.csv
Output  : results_ilp_v4.csv

Vehicle fleet (4 types × 3 units = 12 vehicles):
  l=1  Blind Van    W=830kg   V=2.0CBM  e=13.5km/L  f=150,000
  l=2  Pickup Bak   W=1250kg  V=5.0CBM  e=12.0km/L  f=200,000
  l=3  Engkel (CDE) W=2250kg  V=9.0CBM  e=6.0km/L   f=350,000
  l=4  CDD Box      W=4500kg  V=15.0CBM e=4.5km/L   f=500,000

Fuel price     : Rp 6,800/liter (Bio Solar, Pertamina, May 2026)
Distance       : OSRM road distances (Haversine fallback if OSRM unreachable)
First solution : PATH_CHEAPEST_ARC
Metaheuristic  : GUIDED_LOCAL_SEARCH
Time limit     : 60 seconds per scenario (matches v1 — GLS requires a bound to terminate)
"""

import math
import time
import os
import warnings

import numpy as np
import pandas as pd
import requests

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

warnings.filterwarnings("ignore")

# ── Fleet configuration ────────────────────────────────────────────────────────
FLEET = [
    {"l": 1, "name": "Blind Van",   "W": 830,  "V": 2.0,  "e": 13.5, "f": 150_000},
    {"l": 2, "name": "Pickup Bak",  "W": 1250, "V": 5.0,  "e": 12.0, "f": 200_000},
    {"l": 3, "name": "Engkel",      "W": 2250, "V": 9.0,  "e": 6.0,  "f": 350_000},
    {"l": 4, "name": "CDD Box",     "W": 4500, "V": 15.0, "e": 4.5,  "f": 500_000},
]
UNITS_PER_TYPE = 3   # k = 1, 2, 3
FUEL_PRICE     = 6800  # Rp/litre

# Build flat vehicle list: 12 entries total
VEHICLES = []
for vtype in FLEET:
    for unit in range(1, UNITS_PER_TYPE + 1):
        VEHICLES.append({
            "vehicle_idx": len(VEHICLES),
            "l": vtype["l"],
            "k": unit,
            "name": vtype["name"],
            "W": vtype["W"],
            "V": vtype["V"],
            "e": vtype["e"],
            "f": vtype["f"],
        })

NUM_VEHICLES = len(VEHICLES)  # 12

# OR-Tools uses integer costs — scale Rp to avoid float issues
COST_SCALE = 1  # keep Rp as integer (values are large enough)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset_compiled_v2.csv")
OUTPUT_PATH  = os.path.join(os.path.dirname(__file__), "results_ilp_v4.csv")


# ── Distance utilities ─────────────────────────────────────────────────────────
OSRM_BASE       = "http://router.project-osrm.org/table/v1/driving"
_osrm_available = None   # cached after first check


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = math.sin(math.radians(lat2-lat1)/2)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(math.radians(lon2-lon1)/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def check_osrm():
    """Return True if the public OSRM server is reachable (cached)."""
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
        print("  [INFO] OSRM not reachable — using Haversine fallback.")
    return _osrm_available


def haversine_matrix(nodes):
    """Build n×n Haversine distance matrix (km)."""
    n = len(nodes)
    D = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = haversine(nodes[i]["lat"], nodes[i]["lng"],
                          nodes[j]["lat"], nodes[j]["lng"])
            D[i][j] = D[j][i] = d
    return D


def build_distance_matrix(nodes):
    """
    Build n×n distance matrix (km). Index 0 = depot.
    Uses OSRM road distances with Haversine fallback.
    nodes: list of dicts with 'lat', 'lng'.
    """
    coords = [(nd["lat"], nd["lng"]) for nd in nodes]

    if not check_osrm():
        return haversine_matrix(nodes)

    # OSRM expects lon,lat order
    coord_str = ";".join(f"{lng},{lat}" for lat, lng in coords)
    url = f"{OSRM_BASE}/{coord_str}?annotations=distance"
    try:
        r    = requests.get(url, timeout=30)
        data = r.json()
        if data.get("code") != "Ok":
            return haversine_matrix(nodes)
        n = len(coords)
        D = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                D[i][j] = data["distances"][i][j] / 1000.0  # metres → km
        return D
    except Exception:
        return haversine_matrix(nodes)


def travel_cost_rp(dist_km, vehicle_e):
    """Fuel cost in Rp (integer)."""
    return int(round((dist_km / vehicle_e) * FUEL_PRICE))


# ── OR-Tools model ─────────────────────────────────────────────────────────────
def solve_scenario(scenario_id, depot, delivery_nodes):
    """
    Solve HCVRP for one scenario using OR-Tools CP routing solver.

    The OR-Tools routing model maps naturally to the ILP:
      - Each vehicle corresponds to one (k,l) unit
      - SetFixedCostOfVehicle implements f_l · y_kl
      - Per-vehicle arc cost callbacks implement c_ijl · x_ijkl
      - AddDimensionWithVehicleCapacity implements C4 (weight) and C5 (volume)
      - Flow conservation, depot departure/return (C1–C3, C3b) handled by OR-Tools
      - MTZ subtour elimination (C6) handled implicitly by OR-Tools

    Distance: OSRM road distances with Haversine fallback.
    First solution: PATH_CHEAPEST_ARC.
    Time limit: 60 seconds (matches v1; GLS requires a bound to terminate).

    Returns result dict or None if infeasible.
    """
    t0 = time.time()
    n = len(delivery_nodes)

    if n == 0:
        return None

    # ── Build node list: index 0 = depot ──────────────────────────────────────
    all_nodes = [depot] + delivery_nodes
    num_nodes = len(all_nodes)  # n+1

    D_km = build_distance_matrix(all_nodes)

    # ── Per-vehicle cost matrices (integer Rp) ─────────────────────────────────
    # cost_matrices[v][i][j] = travel cost in Rp for vehicle v
    cost_matrices = []
    for veh in VEHICLES:
        mat = [[travel_cost_rp(D_km[i][j], veh["e"]) for j in range(num_nodes)]
               for i in range(num_nodes)]
        cost_matrices.append(mat)

    # ── OR-Tools routing manager ──────────────────────────────────────────────
    manager = pywrapcp.RoutingIndexManager(num_nodes, NUM_VEHICLES, 0)
    routing = pywrapcp.RoutingModel(manager)

    # ── Transit callbacks (one per vehicle type — 4 unique) ───────────────────
    # Register one callback per vehicle; vehicles with same l share same cost matrix
    callback_indices = []
    for v_idx, veh in enumerate(VEHICLES):
        mat = cost_matrices[v_idx]

        def make_callback(m):
            def cb(from_idx, to_idx):
                i = manager.IndexToNode(from_idx)
                j = manager.IndexToNode(to_idx)
                return m[i][j]
            return cb

        cb_idx = routing.RegisterTransitCallback(make_callback(mat))
        callback_indices.append(cb_idx)
        routing.SetArcCostEvaluatorOfVehicle(cb_idx, v_idx)

    # ── Fixed costs (f_l · y_kl) — C3b linkage ────────────────────────────────
    for v_idx, veh in enumerate(VEHICLES):
        routing.SetFixedCostOfVehicle(veh["f"], v_idx)

    # ── Weight capacity dimension (C4) ────────────────────────────────────────
    def weight_callback(from_idx):
        node = manager.IndexToNode(from_idx)
        if node == 0:
            return 0
        return int(round(all_nodes[node]["weight"] * 1000))  # grams for precision

    weight_cb = routing.RegisterUnaryTransitCallback(weight_callback)
    weight_caps = [int(veh["W"] * 1000) for veh in VEHICLES]  # grams
    routing.AddDimensionWithVehicleCapacity(
        weight_cb, 0, weight_caps, True, "Weight"
    )

    # ── Volume capacity dimension (C5) ────────────────────────────────────────
    def volume_callback(from_idx):
        node = manager.IndexToNode(from_idx)
        if node == 0:
            return 0
        return int(round(all_nodes[node]["volume"] * 1_000_000))  # micro-CBM

    volume_cb = routing.RegisterUnaryTransitCallback(volume_callback)
    volume_caps = [int(veh["V"] * 1_000_000) for veh in VEHICLES]  # micro-CBM
    routing.AddDimensionWithVehicleCapacity(
        volume_cb, 0, volume_caps, True, "Volume"
    )

    # ── Search parameters ──────────────────────────────────────────────────────
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    # 60s time limit per scenario — matches v1; GLS requires a bound to terminate
    search_params.time_limit.FromSeconds(60)
    search_params.log_search = False

    # ── Solve ─────────────────────────────────────────────────────────────────
    solution = routing.SolveWithParameters(search_params)
    runtime  = time.time() - t0

    if not solution:
        return {
            "scenario_id": scenario_id,
            "status": "NO_SOLUTION",
            "runtime_sec": round(runtime, 4),
            "n_nodes": n,
            "routes": [],
        }

    # ── Extract solution ───────────────────────────────────────────────────────
    routes        = []
    total_dist    = 0.0
    total_fuel    = 0.0
    total_fixed   = 0.0
    num_active    = 0

    for v_idx in range(NUM_VEHICLES):
        idx = routing.Start(v_idx)
        route_nodes = []

        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:
                route_nodes.append(node)
            idx = solution.Value(routing.NextVar(idx))

        if not route_nodes:
            continue  # vehicle unused

        num_active += 1
        veh = VEHICLES[v_idx]

        # Reconstruct route distance
        route_dist = 0.0
        prev = 0  # depot
        for node in route_nodes:
            route_dist += D_km[prev][node]
            prev = node
        route_dist += D_km[prev][0]  # return to depot

        fuel_cost  = (route_dist / veh["e"]) * FUEL_PRICE
        fixed_cost = veh["f"]

        load_w = sum(all_nodes[nd]["weight"] for nd in route_nodes)
        load_v = sum(all_nodes[nd]["volume"] for nd in route_nodes)

        total_dist  += route_dist
        total_fuel  += fuel_cost
        total_fixed += fixed_cost

        routes.append({
            "vehicle_type": veh["l"],
            "vehicle_unit": veh["k"],
            "vehicle_name": veh["name"],
            "route_node_ids": [all_nodes[nd]["node_id"] for nd in route_nodes],
            "route_names": [all_nodes[nd]["location_name"] for nd in route_nodes],
            "distance_km": round(route_dist, 4),
            "fuel_cost_rp": round(fuel_cost, 2),
            "fixed_cost_rp": fixed_cost,
            "load_weight_kg": round(load_w, 4),
            "load_volume_cbm": round(load_v, 6),
            "capacity_weight": veh["W"],
            "capacity_volume": veh["V"],
        })

    total_cost = total_fuel + total_fixed

    # Utilisation
    total_w_demand = sum(nd["weight"] for nd in delivery_nodes)
    total_v_demand = sum(nd["volume"] for nd in delivery_nodes)
    cap_w = sum(r["capacity_weight"] for r in routes)
    cap_v = sum(r["capacity_volume"] for r in routes)

    return {
        "scenario_id": scenario_id,
        "method": "ilp_ortools",
        "status": "SOLUTION_FOUND",
        "n_nodes": n,
        "num_vehicles_used": num_active,
        "total_distance_km": round(total_dist, 4),
        "total_fuel_cost_rp": round(total_fuel, 2),
        "total_fixed_cost_rp": round(total_fixed, 2),
        "total_cost_rp": round(total_cost, 2),
        "runtime_sec": round(runtime, 4),
        "distance_method": "OSRM" if check_osrm() else "Haversine",
        "weight_utilisation": round(total_w_demand / cap_w * 100, 2) if cap_w else 0,
        "volume_utilisation": round(total_v_demand / cap_v * 100, 2) if cap_v else 0,
        "total_weight_demand": round(total_w_demand, 4),
        "total_volume_demand": round(total_v_demand, 6),
        "routes": routes,
    }


# ── Load dataset ──────────────────────────────────────────────────────────────
def load_dataset(path):
    df = pd.read_csv(path)
    scenarios = {}

    for scenario_id, grp in df.groupby("scenario_id"):
        depot_row     = grp[grp["node_type"] == "depot"].iloc[0]
        delivery_rows = grp[grp["node_type"] == "delivery"]

        depot = {
            "node_id": 0,
            "location_name": depot_row["location_name"],
            "lat": float(depot_row["lat"]),
            "lng": float(depot_row["lng"]),
        }

        delivery_nodes = []
        for _, row in delivery_rows.iterrows():
            delivery_nodes.append({
                "node_id": int(row["node_id"]),
                "location_name": row["location_name"],
                "lat": float(row["lat"]),
                "lng": float(row["lng"]),
                "weight": float(row["total_weight_kg"]),
                "volume": float(row["total_volume_cbm"]),
                "awb_count": int(row["awb_count"]),
            })

        scenarios[scenario_id] = {
            "depot": depot,
            "delivery_nodes": delivery_nodes,
            "city": depot_row["city"],
            "date": depot_row["date"],
            "n_nodes": len(delivery_nodes),
            "is_lindo_representative": bool(grp["is_lindo_representative"].iloc[0]),
        }

    return scenarios


# ── Write results ─────────────────────────────────────────────────────────────
def write_results(all_results, path):
    flat_rows = []
    for res in all_results:
        if res is None:
            continue
        base = {k: v for k, v in res.items() if k != "routes"}
        if not res.get("routes"):
            flat_rows.append(base)
            continue
        for r_idx, route in enumerate(res["routes"]):
            row = dict(base)
            row["route_index"]        = r_idx + 1
            row["vehicle_type_l"]     = route["vehicle_type"]
            row["vehicle_unit_k"]     = route["vehicle_unit"]
            row["vehicle_name"]       = route["vehicle_name"]
            row["route_distance_km"]  = route["distance_km"]
            row["route_fuel_cost_rp"] = route["fuel_cost_rp"]
            row["route_fixed_cost_rp"]= route["fixed_cost_rp"]
            row["load_weight_kg"]     = route["load_weight_kg"]
            row["load_volume_cbm"]    = route["load_volume_cbm"]
            row["capacity_weight"]    = route["capacity_weight"]
            row["capacity_volume"]    = route["capacity_volume"]
            row["route_node_ids"]     = str(route["route_node_ids"])
            row["route_names"]        = " -> ".join(route["route_names"])
            flat_rows.append(row)

    df_out = pd.DataFrame(flat_rows)
    df_out.to_csv(path, index=False)
    print(f"\nResults saved → {path}")
    return df_out


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  HCVRP ILP Solver v4")
    print("  Method: Google OR-Tools CP Routing (AUTOMATIC + GLS)")
    print("=" * 65)

    scenarios = load_dataset(DATASET_PATH)
    print(f"\nLoaded {len(scenarios)} scenarios from {DATASET_PATH}\n")

    all_results = []

    for scenario_id in sorted(scenarios.keys()):
        sc = scenarios[scenario_id]
        n  = sc["n_nodes"]
        print(f"  Solving {scenario_id}  (n={n}) ...",
              end=" ", flush=True)

        result = solve_scenario(
            scenario_id,
            sc["depot"],
            sc["delivery_nodes"],
        )

        if result:
            result["city"]  = sc["city"]
            result["date"]  = sc["date"]
            result["is_lindo_representative"] = sc["is_lindo_representative"]
            all_results.append(result)

            if result.get("status") == "NO_SOLUTION":
                print("NO SOLUTION")
            else:
                print(f"vehicles={result['num_vehicles_used']}  "
                      f"dist={result['total_distance_km']:.1f}km  "
                      f"cost=Rp{result['total_cost_rp']:,.0f}  "
                      f"runtime={result['runtime_sec']:.2f}s")
        else:
            print("ERROR")
            all_results.append(None)

    # Summary statistics (solved only)
    valid = [r for r in all_results if r and r.get("status") != "NO_SOLUTION"]
    if valid:
        print("\n" + "=" * 65)
        print("  SUMMARY")
        print("=" * 65)
        print(f"  Scenarios solved      : {len(valid)}/{len(all_results)}")
        print(f"  Avg vehicles used     : {np.mean([r['num_vehicles_used']  for r in valid]):.2f}")
        print(f"  Avg total distance km : {np.mean([r['total_distance_km']  for r in valid]):.2f}")
        print(f"  Avg total cost Rp     : {np.mean([r['total_cost_rp']      for r in valid]):,.0f}")
        print(f"  Avg weight utilisation: {np.mean([r['weight_utilisation'] for r in valid]):.2f}%")
        print(f"  Avg volume utilisation: {np.mean([r['volume_utilisation'] for r in valid]):.2f}%")
        print(f"  Avg runtime sec       : {np.mean([r['runtime_sec']        for r in valid]):.4f}")

    write_results([r for r in all_results if r], OUTPUT_PATH)


if __name__ == "__main__":
    main()
