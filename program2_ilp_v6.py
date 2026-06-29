"""
program2_ilp_v6.py
==================
Exact ILP solver for 2D Heterogeneous Capacitated Vehicle Routing Problem (HCVRP)
Method: Pure Integer Linear Programming via Google OR-Tools CP-SAT (branch & bound,
no local-search metaheuristic). Unlike v5 (which used the OR-Tools constraint-programming
*routing* solver with a Guided Local Search metaheuristic — i.e. a heuristic, not an exact
method), this version formulates and solves the explicit MILP directly with CP-SAT and
reports a provable optimum (or a certified bound if the time limit is hit).

Mathematical model (per thesis Section 3.2.3), flattened over the 12 (k,l) vehicle units
as a single vehicle index v = 0..11:
  Indices : i,j = 0..n (nodes, 0 = depot), v = 0..11 (vehicle unit, encodes (k,l))
  Variables:
    x_ijv  ∈ {0,1}  — 1 if vehicle v travels arc i→j                       (C1/C2/C6)
    y_v    ∈ {0,1}  — 1 if vehicle v is activated                          (C3b)
    z_iv   ∈ {0,1}  — 1 if vehicle v serves customer i                     (C3)
    u_i    ∈ [1,n]  — MTZ subtour-elimination position of customer i       (C6)
  Objective: min ΣΣΣ c_ijv·x_ijv + Σ f_v·y_v
  Constraints:
    C1  depot out-degree   : Σ_j x_0jv = y_v                  ∀v
    C2  depot in-degree    : Σ_j xj0v  = y_v                  ∀v
    C3  flow conservation  : Σ_j xijv  = Σ_j xjiv = z_iv       ∀i∈customers, ∀v
    C3b vehicle activation : z_iv ≤ y_v                        ∀i,v
    C4  weight capacity    : Σ_i z_iv·w_i ≤ W_v                ∀v
    C5  volume capacity    : Σ_i z_iv·vol_i ≤ V_v               ∀v
    C6  subtour elimination: u_i - u_j + n·Σ_v xijv ≤ n-1      ∀i≠j ∈ customers
    C7  unique assignment  : Σ_v z_iv = 1                       ∀i∈customers
  Symmetry breaking: identical units of the same vehicle type are used in a fixed
  order (y for unit k ≥ y for unit k+1) to cut the search space without affecting
  the optimum.

Dataset : dataset_compiled_v2.csv
Output  : results_ilp_v6.csv

Vehicle fleet (4 types × 3 units = 12 vehicles):
  l=1  Blind Van    W=830kg   V=2.0CBM  e=13.5km/L  fuel=Gasoline  P_BBM=Rp10,000/L  f=150,000
  l=2  Pickup Bak   W=1250kg  V=5.0CBM  e=12.0km/L  fuel=Gasoline  P_BBM=Rp10,000/L  f=200,000
  l=3  Engkel (CDE) W=2250kg  V=9.0CBM  e=6.0km/L   fuel=Diesel    P_BBM=Rp6,800/L   f=350,000
  l=4  CDD Box      W=4500kg  V=15.0CBM e=4.5km/L   fuel=Diesel    P_BBM=Rp6,800/L   f=500,000

Fuel prices (Pertamina, May 2026):
  Gasoline (Pertalite/Pertamax) : Rp 10,000/liter  → Blind Van, Pickup Bak
  Diesel   (Bio Solar)          : Rp  6,800/liter  → Engkel, CDD Box

Distance     : OSRM road distances (Haversine fallback if OSRM unreachable)
Solver       : OR-Tools CP-SAT (exact MILP branch & bound)
Time limit   : 120 seconds per scenario (safety net; small instances solve well within this)
"""

import math
import time
import os
import warnings

import numpy as np
import pandas as pd
import requests

from ortools.sat.python import cp_model

warnings.filterwarnings("ignore")

# ── Fleet configuration ────────────────────────────────────────────────────────
FUEL_GASOLINE = 10_000   # Rp/litre — Blind Van, Pickup Bak
FUEL_DIESEL   =  6_800   # Rp/litre — Engkel, CDD Box

FLEET = [
    {"l": 1, "name": "Blind Van",  "W": 830,  "V": 2.0,  "e": 13.5, "f": 150_000, "pbbm": FUEL_GASOLINE},
    {"l": 2, "name": "Pickup Bak", "W": 1250, "V": 5.0,  "e": 12.0, "f": 200_000, "pbbm": FUEL_GASOLINE},
    {"l": 3, "name": "Engkel",     "W": 2250, "V": 9.0,  "e": 6.0,  "f": 350_000, "pbbm": FUEL_DIESEL},
    {"l": 4, "name": "CDD Box",    "W": 4500, "V": 15.0, "e": 4.5,  "f": 500_000, "pbbm": FUEL_DIESEL},
]
UNITS_PER_TYPE = 3   # k = 1, 2, 3

# Build flat vehicle list: 12 entries total
VEHICLES = []
for vtype in FLEET:
    for unit in range(1, UNITS_PER_TYPE + 1):
        VEHICLES.append({
            "vehicle_idx": len(VEHICLES),
            "l":    vtype["l"],
            "k":    unit,
            "name": vtype["name"],
            "W":    vtype["W"],
            "V":    vtype["V"],
            "e":    vtype["e"],
            "f":    vtype["f"],
            "pbbm": vtype["pbbm"],
        })

NUM_VEHICLES = len(VEHICLES)  # 12

# Indices of vehicles grouped by type, in unit order — used for symmetry breaking
VEHICLE_TYPE_GROUPS = []
for vtype in FLEET:
    group = [v["vehicle_idx"] for v in VEHICLES if v["l"] == vtype["l"]]
    VEHICLE_TYPE_GROUPS.append(group)

DATASET_PATH    = os.path.join(os.path.dirname(__file__), "dataset_compiled_v2.csv")
OUTPUT_PATH     = os.path.join(os.path.dirname(__file__), "results_ilp_v6.csv")
TIME_LIMIT_SEC  = 120


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


def travel_cost_rp(dist_km, vehicle_e, vehicle_pbbm):
    """Fuel cost in Rp (integer): uses per-vehicle fuel price (gasoline vs diesel)."""
    return int(round((dist_km / vehicle_e) * vehicle_pbbm))


# ── CP-SAT model ────────────────────────────────────────────────────────────────
def solve_scenario(scenario_id, depot, delivery_nodes, time_limit_sec=TIME_LIMIT_SEC):
    """
    Solve HCVRP for one scenario as a pure ILP using OR-Tools CP-SAT.

    This is an exact branch-and-bound MILP solve (no local-search metaheuristic):
      - x[v,i,j]  : arc i->j used by vehicle v                      (C1,C2,C6)
      - y[v]      : vehicle v activated -> SetFixedCostOfVehicle equivalent f_v·y_v (C3b)
      - z[v,i]    : vehicle v serves customer i                     (C3,C7)
      - u[i]      : MTZ position variable, shared across vehicles since each
                    customer is served by exactly one vehicle                (C6)

    Distance: OSRM road distances with Haversine fallback.
    Time limit: 120 seconds (safety net only — instances here are small enough
    that CP-SAT is expected to prove optimality well before the limit).

    Returns result dict or None if infeasible.
    """
    t0 = time.time()
    n = len(delivery_nodes)

    if n == 0:
        return None

    # ── Build node list: index 0 = depot ──────────────────────────────────────
    all_nodes = [depot] + delivery_nodes
    num_nodes = len(all_nodes)  # n+1
    customers = list(range(1, num_nodes))

    D_km = build_distance_matrix(all_nodes)

    # ── Per-vehicle cost matrices (integer Rp) ─────────────────────────────────
    cost_matrices = []
    for veh in VEHICLES:
        mat = [[travel_cost_rp(D_km[i][j], veh["e"], veh["pbbm"]) for j in range(num_nodes)]
               for i in range(num_nodes)]
        cost_matrices.append(mat)

    weight_g  = {i: int(round(all_nodes[i]["weight"] * 1000)) for i in customers}        # grams
    volume_uc = {i: int(round(all_nodes[i]["volume"] * 1_000_000)) for i in customers}    # micro-CBM

    # ── Build CP-SAT model ─────────────────────────────────────────────────────
    model = cp_model.CpModel()
    V = NUM_VEHICLES

    x = {}
    for v in range(V):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    x[v, i, j] = model.NewBoolVar(f"x_{v}_{i}_{j}")

    y = [model.NewBoolVar(f"y_{v}") for v in range(V)]
    z = {(v, i): model.NewBoolVar(f"z_{v}_{i}") for v in range(V) for i in customers}

    # C3 — flow conservation, linking in/out degree to assignment z[v,i]
    for v in range(V):
        for i in customers:
            model.Add(sum(x[v, i, j] for j in range(num_nodes) if j != i) == z[v, i])
            model.Add(sum(x[v, j, i] for j in range(num_nodes) if j != i) == z[v, i])

    # C7 — each customer served exactly once
    for i in customers:
        model.Add(sum(z[v, i] for v in range(V)) == 1)

    # C1, C2 — depot degree linked to vehicle activation y_v
    for v in range(V):
        model.Add(sum(x[v, 0, j] for j in customers) == y[v])
        model.Add(sum(x[v, j, 0] for j in customers) == y[v])

    # C3b — a vehicle can only serve a customer if it is activated
    for v in range(V):
        for i in customers:
            model.Add(z[v, i] <= y[v])

    # C4, C5 — weight / volume capacity per vehicle
    for v in range(V):
        veh = VEHICLES[v]
        model.Add(sum(z[v, i] * weight_g[i] for i in customers) <= int(veh["W"] * 1000))
        model.Add(sum(z[v, i] * volume_uc[i] for i in customers) <= int(veh["V"] * 1_000_000))

    # C6 — MTZ subtour elimination (global position var: each customer has exactly
    # one serving vehicle, so a single u_i per customer suffices)
    u = {i: model.NewIntVar(1, n, f"u_{i}") for i in customers}
    for i in customers:
        for j in customers:
            if i != j:
                arc_total = sum(x[v, i, j] for v in range(V))
                model.Add(u[i] - u[j] + n * arc_total <= n - 1)

    # Symmetry breaking — identical units of a type are used in a fixed order
    for group in VEHICLE_TYPE_GROUPS:
        for a, b in zip(group, group[1:]):
            model.Add(y[a] >= y[b])

    # ── Objective: travel cost + fixed activation cost ────────────────────────
    travel_cost = sum(
        cost_matrices[v][i][j] * x[v, i, j]
        for v in range(V)
        for i in range(num_nodes)
        for j in range(num_nodes)
        if i != j
    )
    fixed_cost = sum(VEHICLES[v]["f"] * y[v] for v in range(V))
    model.Minimize(travel_cost + fixed_cost)

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers  = os.cpu_count() or 8
    status  = solver.Solve(model)
    runtime = time.time() - t0

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "scenario_id": scenario_id,
            "status": "NO_SOLUTION",
            "runtime_sec": round(runtime, 4),
            "n_nodes": n,
            "routes": [],
        }

    is_optimal = (status == cp_model.OPTIMAL)

    # ── Extract solution ───────────────────────────────────────────────────────
    routes      = []
    total_dist  = 0.0
    total_fuel  = 0.0
    total_fixed = 0.0
    num_active  = 0

    for v in range(V):
        if solver.Value(y[v]) == 0:
            continue

        # Trace the route starting from depot
        route_nodes = []
        current = 0
        while True:
            next_node = None
            for j in range(num_nodes):
                if j != current and solver.Value(x[v, current, j]) == 1:
                    next_node = j
                    break
            if next_node is None or next_node == 0:
                break
            route_nodes.append(next_node)
            current = next_node

        if not route_nodes:
            continue  # vehicle activated but empty (shouldn't happen)

        num_active += 1
        veh = VEHICLES[v]

        route_dist = 0.0
        prev = 0
        for node in route_nodes:
            route_dist += D_km[prev][node]
            prev = node
        route_dist += D_km[prev][0]

        fuel_cost  = (route_dist / veh["e"]) * veh["pbbm"]
        fixed_cost_v = veh["f"]

        load_w = sum(all_nodes[nd]["weight"] for nd in route_nodes)
        load_v = sum(all_nodes[nd]["volume"] for nd in route_nodes)

        total_dist  += route_dist
        total_fuel  += fuel_cost
        total_fixed += fixed_cost_v

        depot_name = all_nodes[0]["location_name"]
        depot_id   = all_nodes[0]["node_id"]

        routes.append({
            "vehicle_type": veh["l"],
            "vehicle_unit": veh["k"],
            "vehicle_name": veh["name"],
            "route_node_ids":      [all_nodes[nd]["node_id"] for nd in route_nodes],
            "route_names":         [all_nodes[nd]["location_name"] for nd in route_nodes],
            "route_full_node_ids": [depot_id] + [all_nodes[nd]["node_id"] for nd in route_nodes] + [depot_id],
            "route_full":          depot_name + " -> " + " -> ".join(all_nodes[nd]["location_name"] for nd in route_nodes) + " -> " + depot_name,
            "distance_km": round(route_dist, 4),
            "fuel_cost_rp": round(fuel_cost, 2),
            "fixed_cost_rp": fixed_cost_v,
            "load_weight_kg": round(load_w, 4),
            "load_volume_cbm": round(load_v, 6),
            "capacity_weight": veh["W"],
            "capacity_volume": veh["V"],
        })

    total_cost = total_fuel + total_fixed

    total_w_demand = sum(nd["weight"] for nd in delivery_nodes)
    total_v_demand = sum(nd["volume"] for nd in delivery_nodes)
    cap_w = sum(r["capacity_weight"] for r in routes)
    cap_v = sum(r["capacity_volume"] for r in routes)

    return {
        "scenario_id": scenario_id,
        "method": "ilp_cpsat",
        "status": "OPTIMAL" if is_optimal else "FEASIBLE_TIME_LIMIT",
        "is_optimal": is_optimal,
        "objective_lower_bound_rp": round(solver.BestObjectiveBound(), 2),
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
            row["route_full_node_ids"]= str(route["route_full_node_ids"])
            row["route_full"]         = route["route_full"]
            flat_rows.append(row)

    df_out = pd.DataFrame(flat_rows)
    df_out.to_csv(path, index=False)
    print(f"\nResults saved → {path}")
    return df_out


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  HCVRP Pure ILP Solver v6")
    print("  Method: Google OR-Tools CP-SAT (exact MILP, branch & bound)")
    print("  Fuel: Gasoline Rp10,000/L (Blind Van, Pickup Bak) | Diesel Rp6,800/L (Engkel, CDD Box)")
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
                print(f"[{result['status']}]  "
                      f"vehicles={result['num_vehicles_used']}  "
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
        n_optimal = sum(1 for r in valid if r.get("is_optimal"))
        print(f"  Scenarios solved      : {len(valid)}/{len(all_results)}")
        print(f"  Proved optimal        : {n_optimal}/{len(valid)}")
        print(f"  Avg vehicles used     : {np.mean([r['num_vehicles_used']  for r in valid]):.2f}")
        print(f"  Avg total distance km : {np.mean([r['total_distance_km']  for r in valid]):.2f}")
        print(f"  Avg total cost Rp     : {np.mean([r['total_cost_rp']      for r in valid]):,.0f}")
        print(f"  Avg weight utilisation: {np.mean([r['weight_utilisation'] for r in valid]):.2f}%")
        print(f"  Avg volume utilisation: {np.mean([r['volume_utilisation'] for r in valid]):.2f}%")
        print(f"  Avg runtime sec       : {np.mean([r['runtime_sec']        for r in valid]):.4f}")

    write_results([r for r in all_results if r], OUTPUT_PATH)


if __name__ == "__main__":
    main()
