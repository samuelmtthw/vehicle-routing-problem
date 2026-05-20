"""
program1_heuristic_v4.py
========================
Heuristic solver for 2D Heterogeneous Capacitated Vehicle Routing Problem (HCVRP)
Method: K-Means clustering + Best-Fit Decreasing (BFD) bin packing + Greedy Nearest Neighbor routing

Dataset : dataset_compiled_v2.csv
Output  : results_heuristic_v3.csv

Vehicle fleet (4 types × 3 units = 12 vehicles):
  l=1  Blind Van    W=830kg   V=2.0CBM  e=13.5km/L
  l=2  Pickup Bak   W=1250kg  V=5.0CBM  e=12.0km/L
  l=3  Engkel (CDE) W=2250kg  V=9.0CBM  e=6.0km/L
  l=4  CDD Box      W=4500kg  V=15.0CBM e=4.5km/L

Fuel price: Rp 6,800/liter (Bio Solar, Pertamina, May 2026)
Fixed cost per vehicle activation: f_l (set high to penalise fleet size)
Distance: Haversine (road OSRM not available offline)
"""

import math
import time
import copy
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# ── Fleet configuration ────────────────────────────────────────────────────────
FLEET = [
    {"l": 1, "name": "Blind Van",   "W": 830,  "V": 2.0,  "e": 13.5},
    {"l": 2, "name": "Pickup Bak",  "W": 1250, "V": 5.0,  "e": 12.0},
    {"l": 3, "name": "Engkel",      "W": 2250, "V": 9.0,  "e": 6.0},
    {"l": 4, "name": "CDD Box",     "W": 4500, "V": 15.0, "e": 4.5},
]
UNITS_PER_TYPE = 3          # k = 1,2,3 per vehicle type
FUEL_PRICE     = 6800       # Rp/litre
FIXED_COST     = {1: 150_000, 2: 200_000, 3: 350_000, 4: 500_000}  # Rp per activation

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset_compiled_v2.csv")
OUTPUT_PATH  = os.path.join(os.path.dirname(__file__), "results_heuristic_v4.csv")


# ── Distance utilities ─────────────────────────────────────────────────────────
OSRM_BASE      = "http://router.project-osrm.org/table/v1/driving"
_osrm_available = None   # cached after first check


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


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
    """Build n×n distance matrix (km) using Haversine."""
    n = len(nodes)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
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
        n      = len(coords)
        D      = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                D[i][j] = data["distances"][i][j] / 1000.0  # metres → km
        return D
    except Exception:
        return haversine_matrix(nodes)


def travel_cost(dist_km, vehicle_type_l):
    """Fuel cost in Rp for a given distance and vehicle type."""
    e = FLEET[vehicle_type_l - 1]["e"]
    return (dist_km / e) * FUEL_PRICE


# ── Best-Fit Decreasing (BFD) bin packing ─────────────────────────────────────
def bfd_assign_nodes_to_vehicles(nodes, vehicle_slots):
    """
    Assign delivery nodes to vehicle slots using Best-Fit Decreasing.
    Nodes sorted descending by weight.
    Each slot: {'l': vehicle_type, 'rem_w': remaining_weight, 'rem_v': remaining_volume, 'nodes': []}

    Returns list of slots with assigned nodes.
    """
    # Sort nodes descending by weight (primary), volume (secondary)
    sorted_nodes = sorted(nodes, key=lambda x: (-x["weight"], -x["volume"]))

    for node in sorted_nodes:
        best_slot = None
        best_slack = float("inf")

        for slot in vehicle_slots:
            if slot["rem_w"] >= node["weight"] and slot["rem_v"] >= node["volume"]:
                # tightest fit: minimise remaining capacity after insertion
                slack = (slot["rem_w"] - node["weight"]) + (slot["rem_v"] - node["volume"]) * 100
                if slack < best_slack:
                    best_slack = slack
                    best_slot = slot

        if best_slot is None:
            # No slot fits — return None to signal infeasibility
            return None

        best_slot["nodes"].append(node)
        best_slot["rem_w"] -= node["weight"]
        best_slot["rem_v"] -= node["volume"]

    return vehicle_slots


# ── Greedy Nearest Neighbour routing ──────────────────────────────────────────
def greedy_nearest_neighbour(depot_idx, node_indices, D):
    """
    Build a route starting and ending at depot using nearest neighbour heuristic.
    Returns ordered list of indices (excluding depot start/end) and total distance.
    """
    if not node_indices:
        return [], 0.0

    unvisited = list(node_indices)
    route = []
    current = depot_idx
    total_dist = 0.0

    while unvisited:
        nearest = min(unvisited, key=lambda j: D[current][j])
        total_dist += D[current][nearest]
        route.append(nearest)
        current = nearest
        unvisited.remove(nearest)

    # return to depot
    total_dist += D[current][depot_idx]
    return route, total_dist


# ── K-Means clustering ────────────────────────────────────────────────────────
def kmeans_cluster(delivery_nodes, k):
    """
    Cluster delivery_nodes into k groups by lat/lng.
    Returns list of lists (one list of node dicts per cluster).
    Handles edge cases: k > n, all same location.
    """
    n = len(delivery_nodes)
    k = min(k, n)

    coords = np.array([[nd["lat"], nd["lng"]] for nd in delivery_nodes])

    if k == 1 or n == 1:
        return [delivery_nodes]

    # Use k-means++ init for better stability
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(coords)

    clusters = defaultdict(list)
    for i, nd in enumerate(delivery_nodes):
        clusters[labels[i]].append(nd)

    return [v for v in clusters.values() if v]


# ── Main solver per scenario ───────────────────────────────────────────────────
def solve_scenario(scenario_id, depot, delivery_nodes):
    """
    Solve one HCVRP scenario with the heuristic approach.

    Strategy:
    1. Try k clusters (k = 1 .. min(n, total_vehicles))
    2. For each k, assign clusters to vehicle slots via BFD
    3. Route each slot with greedy nearest neighbour
    4. Pick k that gives fewest vehicles (primary), then least total cost (secondary)

    Returns result dict.
    """
    t0 = time.time()
    n = len(delivery_nodes)

    if n == 0:
        return None

    # Build all nodes list: index 0 = depot
    all_nodes = [depot] + delivery_nodes
    D = build_distance_matrix(all_nodes)  # 0 = depot

    total_vehicles = len(FLEET) * UNITS_PER_TYPE  # 12

    # ── Pre-check: flag nodes that exceed the largest vehicle ─────────────────
    max_W = max(v["W"] for v in FLEET)
    max_V = max(v["V"] for v in FLEET)
    oversized = [nd for nd in delivery_nodes
                 if nd["weight"] > max_W or nd["volume"] > max_V]
    if oversized:
        names = [nd["location_name"] for nd in oversized]
        return {
            "scenario_id": scenario_id,
            "method": "heuristic_kmeans_bfd_gnn",
            "status": "INFEASIBLE_OVERSIZED_NODE",
            "infeasible_nodes": names,
            "n_nodes": n,
            "num_vehicles_used": 0,
            "total_distance_km": 0.0,
            "total_fuel_cost_rp": 0.0,
            "total_fixed_cost_rp": 0.0,
            "total_cost_rp": 0.0,
            "best_k_clusters": 0,
            "weight_utilisation": 0.0,
            "volume_utilisation": 0.0,
            "total_weight_demand": round(sum(nd["weight"] for nd in delivery_nodes), 4),
            "total_volume_demand": round(sum(nd["volume"] for nd in delivery_nodes), 6),
            "runtime_sec": round(time.time() - t0, 4),
            "routes": [],
        }

    best_result = None
    best_num_vehicles = float("inf")
    best_total_cost   = float("inf")

    # Build base vehicle slots template (deep-copied per iteration)
    base_slots = []
    for vtype in FLEET:
        for unit in range(1, UNITS_PER_TYPE + 1):
            base_slots.append({
                "l": vtype["l"],
                "k": unit,
                "name": vtype["name"],
                "W": vtype["W"],
                "V": vtype["V"],
                "rem_w": vtype["W"],
                "rem_v": vtype["V"],
                "nodes": [],
            })

    # Try different k values for K-Means
    max_k = min(n, total_vehicles)

    for k in range(1, max_k + 1):
        clusters = kmeans_cluster(delivery_nodes, k)

        # Deep-copy slots so each k iteration starts fresh
        vehicle_slots = copy.deepcopy(base_slots)

        assigned = bfd_assign_nodes_to_vehicles(delivery_nodes, vehicle_slots)

        if assigned is None:
            continue  # infeasible packing for this configuration

        # Count active vehicles and compute costs
        active_slots = [s for s in assigned if s["nodes"]]
        num_vehicles = len(active_slots)

        total_fuel_cost  = 0.0
        total_fixed_cost = 0.0
        total_dist       = 0.0
        routes           = []

        for slot in active_slots:
            node_indices = [all_nodes.index(nd) for nd in slot["nodes"]]
            route_order, dist = greedy_nearest_neighbour(0, node_indices, D)
            fuel = travel_cost(dist, slot["l"])
            fixed = FIXED_COST[slot["l"]]

            total_dist       += dist
            total_fuel_cost  += fuel
            total_fixed_cost += fixed

            routes.append({
                "vehicle_type": slot["l"],
                "vehicle_unit": slot["k"],
                "vehicle_name": slot["name"],
                "route_node_ids": [all_nodes[i]["node_id"] for i in route_order],
                "route_names": [all_nodes[i]["location_name"] for i in route_order],
                "distance_km": round(dist, 4),
                "fuel_cost_rp": round(fuel, 2),
                "fixed_cost_rp": fixed,
                "load_weight_kg": round(slot["W"] - slot["rem_w"], 4),
                "load_volume_cbm": round(slot["V"] - slot["rem_v"], 6),
                "capacity_weight": slot["W"],
                "capacity_volume": slot["V"],
            })

        total_cost = total_fuel_cost + total_fixed_cost

        # Lexicographic: minimise vehicles first, then total cost
        if (num_vehicles < best_num_vehicles or
                (num_vehicles == best_num_vehicles and total_cost < best_total_cost)):
            best_num_vehicles = num_vehicles
            best_total_cost   = total_cost
            best_result = {
                "scenario_id": scenario_id,
                "method": "heuristic_kmeans_bfd_gnn",
                "n_nodes": n,
                "num_vehicles_used": num_vehicles,
                "total_distance_km": round(total_dist, 4),
                "total_fuel_cost_rp": round(total_fuel_cost, 2),
                "total_fixed_cost_rp": round(total_fixed_cost, 2),
                "total_cost_rp": round(total_cost, 2),
                "best_k_clusters": k,
                "routes": routes,
            }

    runtime = time.time() - t0

    if best_result:
        # Utilisation
        total_weight_demand = sum(nd["weight"] for nd in delivery_nodes)
        total_volume_demand = sum(nd["volume"] for nd in delivery_nodes)
        total_cap_w = sum(r["capacity_weight"] for r in best_result["routes"])
        total_cap_v = sum(r["capacity_volume"] for r in best_result["routes"])

        best_result["runtime_sec"]          = round(runtime, 4)
        best_result["weight_utilisation"]   = round(total_weight_demand / total_cap_w * 100, 2) if total_cap_w else 0
        best_result["volume_utilisation"]   = round(total_volume_demand / total_cap_v * 100, 2) if total_cap_v else 0
        best_result["total_weight_demand"]  = round(total_weight_demand, 4)
        best_result["total_volume_demand"]  = round(total_volume_demand, 6)

    return best_result


# ── Load dataset ──────────────────────────────────────────────────────────────
def load_dataset(path):
    df = pd.read_csv(path)
    scenarios = {}

    for scenario_id, grp in df.groupby("scenario_id"):
        depot_row = grp[grp["node_type"] == "depot"].iloc[0]
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
        if not res["routes"]:
            flat_rows.append(base)
            continue
        for r_idx, route in enumerate(res["routes"]):
            row = dict(base)
            row["route_index"]       = r_idx + 1
            row["vehicle_type_l"]    = route["vehicle_type"]
            row["vehicle_unit_k"]    = route["vehicle_unit"]
            row["vehicle_name"]      = route["vehicle_name"]
            row["route_distance_km"] = route["distance_km"]
            row["route_fuel_cost_rp"]= route["fuel_cost_rp"]
            row["route_fixed_cost_rp"]= route["fixed_cost_rp"]
            row["load_weight_kg"]    = route["load_weight_kg"]
            row["load_volume_cbm"]   = route["load_volume_cbm"]
            row["capacity_weight"]   = route["capacity_weight"]
            row["capacity_volume"]   = route["capacity_volume"]
            row["route_node_ids"]    = str(route["route_node_ids"])
            row["route_names"]       = " -> ".join(route["route_names"])
            flat_rows.append(row)

    df_out = pd.DataFrame(flat_rows)
    df_out.to_csv(path, index=False)
    print(f"\nResults saved → {path}")
    return df_out


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  HCVRP Heuristic Solver v4")
    print("  Method: K-Means + BFD + Greedy Nearest Neighbour")
    print("=" * 65)

    scenarios = load_dataset(DATASET_PATH)
    print(f"\nLoaded {len(scenarios)} scenarios from {DATASET_PATH}\n")

    all_results = []

    for scenario_id in sorted(scenarios.keys()):
        sc = scenarios[scenario_id]
        print(f"  Solving {scenario_id}  (n={sc['n_nodes']}) ...", end=" ", flush=True)

        result = solve_scenario(scenario_id, sc["depot"], sc["delivery_nodes"])

        if result:
            result["city"]  = sc["city"]
            result["date"]  = sc["date"]
            result["is_lindo_representative"] = sc["is_lindo_representative"]
            result["distance_method"] = "OSRM" if check_osrm() else "Haversine"
            all_results.append(result)
            print(f"vehicles={result['num_vehicles_used']}  "
                  f"dist={result['total_distance_km']:.1f}km  "
                  f"cost=Rp{result['total_cost_rp']:,.0f}  "
                  f"runtime={result['runtime_sec']:.2f}s")
        else:
            print("INFEASIBLE")
            all_results.append(None)

    # Summary statistics
    valid = [r for r in all_results if r]
    if valid:
        print("\n" + "=" * 65)
        print("  SUMMARY")
        print("=" * 65)
        avg_veh  = np.mean([r["num_vehicles_used"]  for r in valid])
        avg_dist = np.mean([r["total_distance_km"]  for r in valid])
        avg_cost = np.mean([r["total_cost_rp"]      for r in valid])
        avg_wutil= np.mean([r["weight_utilisation"] for r in valid])
        avg_vutil= np.mean([r["volume_utilisation"] for r in valid])
        avg_rt   = np.mean([r["runtime_sec"]        for r in valid])
        print(f"  Scenarios solved      : {len(valid)}/{len(all_results)}")
        print(f"  Avg vehicles used     : {avg_veh:.2f}")
        print(f"  Avg total distance km : {avg_dist:.2f}")
        print(f"  Avg total cost Rp     : {avg_cost:,.0f}")
        print(f"  Avg weight utilisation: {avg_wutil:.2f}%")
        print(f"  Avg volume utilisation: {avg_vutil:.2f}%")
        print(f"  Avg runtime sec       : {avg_rt:.4f}")

    write_results(valid, OUTPUT_PATH)


if __name__ == "__main__":
    main()
