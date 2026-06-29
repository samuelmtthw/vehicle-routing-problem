"""
program1_heuristic_v6.py
========================
Heuristic solver for 2D Heterogeneous Capacitated Vehicle Routing Problem (HCVRP)

Method: K-Means clustering + Best-Fit Decreasing (BFD) bin packing + Greedy Nearest Neighbor routing
        — cluster-first, route-second —

Key fix vs v5: BFD is now applied PER CLUSTER (not globally).
  Each cluster's nodes are packed into vehicle slots independently,
  which makes K-Means actually meaningful: geographic grouping now
  constrains which nodes can be consolidated.

Pipeline per k:
  1. K-Means splits delivery nodes into k geographic clusters
  2. For each cluster, BFD allocates its nodes into the best-fitting
     vehicle slots drawn from the global fleet pool
  3. GNN builds the intra-cluster route for each active vehicle slot
  4. Lexicographic selection picks the k with fewest vehicles, then
     lowest total cost

Dataset : dataset_compiled_v2.csv
Output  : results_heuristic_v6.csv

Vehicle fleet (4 types × 3 units = 12 vehicles):
  l=1  Blind Van    W=830kg   V=2.0CBM  e=13.5km/L  fuel=Gasoline  P_BBM=Rp10,000/L  f=Rp150,000
  l=2  Pickup Bak   W=1250kg  V=5.0CBM  e=12.0km/L  fuel=Gasoline  P_BBM=Rp10,000/L  f=Rp200,000
  l=3  Engkel (CDE) W=2250kg  V=9.0CBM  e=6.0km/L   fuel=Diesel    P_BBM=Rp6,800/L   f=Rp350,000
  l=4  CDD Box      W=4500kg  V=15.0CBM e=4.5km/L   fuel=Diesel    P_BBM=Rp6,800/L   f=Rp500,000

Fuel prices (Pertamina, May 2026):
  Gasoline (Pertalite) : Rp 10,000/liter → Blind Van, Pickup Bak
  Diesel   (Bio Solar) : Rp  6,800/liter → Engkel, CDD Box

Distance: OSRM road distances (Haversine fallback if OSRM unreachable)
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

FUEL_GASOLINE = 10_000   # Rp/litre — Blind Van, Pickup Bak
FUEL_DIESEL   =  6_800   # Rp/litre — Engkel, CDD Box

FLEET = [
    {"l": 1, "name": "Blind Van",  "W":  830, "V":  2.0, "e": 13.5, "pbbm": FUEL_GASOLINE},
    {"l": 2, "name": "Pickup Bak", "W": 1250, "V":  5.0, "e": 12.0, "pbbm": FUEL_GASOLINE},
    {"l": 3, "name": "Engkel",     "W": 2250, "V":  9.0, "e":  6.0, "pbbm": FUEL_DIESEL},
    {"l": 4, "name": "CDD Box",    "W": 4500, "V": 15.0, "e":  4.5, "pbbm": FUEL_DIESEL},
]

UNITS_PER_TYPE = 3                                    # k = 1,2,3 per vehicle type
FIXED_COST     = {1: 150_000, 2: 200_000, 3: 350_000, 4: 500_000}  # Rp per activation

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset_compiled_v2.csv")
OUTPUT_PATH  = os.path.join(os.path.dirname(__file__), "results_heuristic_v6.csv")

# ── Distance utilities ─────────────────────────────────────────────────────────

OSRM_BASE       = "http://router.project-osrm.org/table/v1/driving"
_osrm_available = None  # cached after first check


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi        = math.radians(lat2 - lat1)
    dlambda     = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def check_osrm():
    """Return True if the public OSRM server is reachable (result cached)."""
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
    if not check_osrm():
        return haversine_matrix(nodes)

    # OSRM expects lon,lat order
    coord_str = ";".join(f"{nd['lng']},{nd['lat']}" for nd in nodes)
    url = f"{OSRM_BASE}/{coord_str}?annotations=distance"
    try:
        r    = requests.get(url, timeout=30)
        data = r.json()
        if data.get("code") != "Ok":
            return haversine_matrix(nodes)
        n = len(nodes)
        D = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                D[i][j] = data["distances"][i][j] / 1000.0  # metres → km
        return D
    except Exception:
        return haversine_matrix(nodes)


def travel_cost(dist_km, vehicle_type_l):
    """Fuel cost in Rp using per-vehicle fuel price (gasoline vs diesel)."""
    vtype = FLEET[vehicle_type_l - 1]
    return (dist_km / vtype["e"]) * vtype["pbbm"]


# ── K-Means clustering ─────────────────────────────────────────────────────────

def kmeans_cluster(delivery_nodes, k):
    """
    Cluster delivery_nodes into k geographic groups by lat/lng.
    Returns list of lists (one list of node dicts per cluster).
    """
    n = len(delivery_nodes)
    k = min(k, n)
    coords = np.array([[nd["lat"], nd["lng"]] for nd in delivery_nodes])

    if k == 1 or n == 1:
        return [delivery_nodes]

    km     = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(coords)

    clusters = defaultdict(list)
    for i, nd in enumerate(delivery_nodes):
        clusters[labels[i]].append(nd)

    return [v for v in clusters.values() if v]


# ── Best-Fit Decreasing (BFD) bin packing  ────────────────────────────────────

def make_fresh_slots():
    """
    Build the full list of 12 vehicle slots in ascending capacity order.
    Each call returns a brand-new deep-copyable list.
    """
    slots = []
    for vtype in FLEET:
        for unit in range(1, UNITS_PER_TYPE + 1):
            slots.append({
                "l":     vtype["l"],
                "k":     unit,
                "name":  vtype["name"],
                "W":     vtype["W"],
                "V":     vtype["V"],
                "rem_w": vtype["W"],
                "rem_v": vtype["V"],
                "nodes": [],
            })
    return slots


def bfd_pack_cluster(cluster_nodes, available_slots):
    """
    Pack all nodes in cluster_nodes into available_slots using BFD (tightest-fit).

    Nodes are sorted descending by weight.
    Slot selection: minimise slack = (rem_w - w) + λ*(rem_v - v), λ=100.

    Returns:
        (used_slots, remaining_slots) on success
        None                          if any node cannot be packed
    """
    sorted_nodes = sorted(cluster_nodes, key=lambda x: (-x["weight"], -x["volume"]))

    used_indices = set()

    for node in sorted_nodes:
        best_idx   = None
        best_slack = float("inf")

        for idx, slot in enumerate(available_slots):
            if slot["rem_w"] >= node["weight"] and slot["rem_v"] >= node["volume"]:
                slack = (slot["rem_w"] - node["weight"]) + (slot["rem_v"] - node["volume"]) * 100
                if slack < best_slack:
                    best_slack = slack
                    best_idx   = idx

        if best_idx is None:
            return None  # infeasible: no slot can fit this node

        available_slots[best_idx]["nodes"].append(node)
        available_slots[best_idx]["rem_w"] -= node["weight"]
        available_slots[best_idx]["rem_v"] -= node["volume"]
        used_indices.add(best_idx)

    used_slots      = [available_slots[i] for i in sorted(used_indices)]
    remaining_slots = [available_slots[i] for i in range(len(available_slots))
                       if i not in used_indices]
    return used_slots, remaining_slots


# ── Greedy Nearest Neighbour routing ──────────────────────────────────────────

def greedy_nearest_neighbour(depot_idx, node_indices, D):
    """
    Build a route starting and ending at depot using nearest-neighbour heuristic.
    Returns ordered list of node indices (depot excluded) and total distance (km).
    """
    if not node_indices:
        return [], 0.0

    unvisited  = list(node_indices)
    route      = []
    current    = depot_idx
    total_dist = 0.0

    while unvisited:
        nearest = min(unvisited, key=lambda j: D[current][j])
        total_dist += D[current][nearest]
        route.append(nearest)
        current = nearest
        unvisited.remove(nearest)

    total_dist += D[current][depot_idx]   # return to depot
    return route, total_dist


# ── Per-cluster packing + routing ─────────────────────────────────────────────

def pack_and_route_clusters(clusters, all_nodes, D):
    """
    Given a list of clusters (each a list of node dicts), assign vehicles
    to each cluster via per-cluster BFD, then route via GNN.

    Vehicle slots are drawn from a shared pool and consumed cluster by cluster
    in descending order of cluster total weight (largest cluster gets first pick).

    Returns:
        list of route dicts on success
        None if any cluster is infeasible
    """
    # Sort clusters: largest total weight first so they get priority on big vehicles
    clusters_sorted = sorted(clusters, key=lambda c: -sum(n["weight"] for n in c))

    available_slots = make_fresh_slots()
    all_active_slots = []

    for cluster_nodes in clusters_sorted:
        result = bfd_pack_cluster(cluster_nodes, available_slots)
        if result is None:
            return None  # this k is infeasible
        used_slots, available_slots = result
        all_active_slots.extend(used_slots)

    # Build routes via GNN for each active slot
    routes = []
    for slot in all_active_slots:
        if not slot["nodes"]:
            continue  # safety guard

        # Map node dicts → indices in all_nodes (0 = depot)
        node_indices = [all_nodes.index(nd) for nd in slot["nodes"]]
        route_order, dist = greedy_nearest_neighbour(0, node_indices, D)

        fuel  = travel_cost(dist, slot["l"])
        fixed = FIXED_COST[slot["l"]]

        depot_id   = all_nodes[0]["node_id"]
        depot_name = all_nodes[0]["location_name"]

        routes.append({
            "vehicle_type":        slot["l"],
            "vehicle_unit":        slot["k"],
            "vehicle_name":        slot["name"],
            "route_node_ids":      [all_nodes[i]["node_id"] for i in route_order],
            "route_names":         [all_nodes[i]["location_name"] for i in route_order],
            "route_full_node_ids": [depot_id] + [all_nodes[i]["node_id"] for i in route_order] + [depot_id],
            "route_full":          depot_name + " -> "
                                   + " -> ".join(all_nodes[i]["location_name"] for i in route_order)
                                   + " -> " + depot_name,
            "distance_km":         round(dist, 4),
            "fuel_cost_rp":        round(fuel, 2),
            "fixed_cost_rp":       fixed,
            "load_weight_kg":      round(slot["W"] - slot["rem_w"], 4),
            "load_volume_cbm":     round(slot["V"] - slot["rem_v"], 6),
            "capacity_weight":     slot["W"],
            "capacity_volume":     slot["V"],
        })

    return routes


# ── Main solver per scenario ───────────────────────────────────────────────────

def solve_scenario(scenario_id, depot, delivery_nodes):
    """
    Solve one HCVRP scenario with the corrected cluster-first, route-second heuristic.

    For each k (1 .. min(n, 12)):
        1. K-Means clusters nodes geographically into k groups
        2. BFD packs each cluster's nodes into vehicle slots (per-cluster)
        3. GNN routes each active vehicle slot
        4. Lexicographic selection: fewest vehicles → lowest total cost

    Returns a result dict.
    """
    t0 = time.time()
    n  = len(delivery_nodes)

    if n == 0:
        return None

    # Index 0 = depot in the global node list
    all_nodes = [depot] + delivery_nodes
    D         = build_distance_matrix(all_nodes)

    total_vehicles = len(FLEET) * UNITS_PER_TYPE  # 12

    # ── Pre-check: any single node exceeding largest vehicle? ─────────────────
    max_W = max(v["W"] for v in FLEET)
    max_V = max(v["V"] for v in FLEET)
    oversized = [nd for nd in delivery_nodes
                 if nd["weight"] > max_W or nd["volume"] > max_V]
    if oversized:
        names = [nd["location_name"] for nd in oversized]
        return {
            "scenario_id":        scenario_id,
            "method":             "heuristic_kmeans_per_cluster_bfd_gnn",
            "status":             "INFEASIBLE_OVERSIZED_NODE",
            "infeasible_nodes":   names,
            "n_nodes":            n,
            "num_vehicles_used":  0,
            "total_distance_km":  0.0,
            "total_fuel_cost_rp": 0.0,
            "total_fixed_cost_rp":0.0,
            "total_cost_rp":      0.0,
            "best_k_clusters":    0,
            "weight_utilisation": 0.0,
            "volume_utilisation": 0.0,
            "total_weight_demand":round(sum(nd["weight"] for nd in delivery_nodes), 4),
            "total_volume_demand":round(sum(nd["volume"] for nd in delivery_nodes), 6),
            "runtime_sec":        round(time.time() - t0, 4),
            "routes":             [],
        }

    best_result      = None
    best_num_vehicles = float("inf")
    best_total_cost  = float("inf")

    max_k = min(n, total_vehicles)

    for k in range(1, max_k + 1):
        # Step 1: geographic clustering
        clusters = kmeans_cluster(delivery_nodes, k)

        # Step 2 + 3: per-cluster BFD packing + GNN routing
        routes = pack_and_route_clusters(clusters, all_nodes, D)

        if routes is None:
            # At least one cluster was infeasible for this k — skip
            continue

        num_vehicles    = len(routes)
        total_dist      = sum(r["distance_km"]   for r in routes)
        total_fuel_cost = sum(r["fuel_cost_rp"]  for r in routes)
        total_fixed_cost= sum(r["fixed_cost_rp"] for r in routes)
        total_cost      = total_fuel_cost + total_fixed_cost

        # Lexicographic: fewer vehicles first, then lower cost
        if (num_vehicles < best_num_vehicles or
                (num_vehicles == best_num_vehicles and total_cost < best_total_cost)):
            best_num_vehicles = num_vehicles
            best_total_cost   = total_cost
            best_result = {
                "scenario_id":         scenario_id,
                "method":              "heuristic_kmeans_per_cluster_bfd_gnn",
                "status":              "FEASIBLE",
                "n_nodes":             n,
                "num_vehicles_used":   num_vehicles,
                "total_distance_km":   round(total_dist, 4),
                "total_fuel_cost_rp":  round(total_fuel_cost, 2),
                "total_fixed_cost_rp": round(total_fixed_cost, 2),
                "total_cost_rp":       round(total_cost, 2),
                "best_k_clusters":     k,
                "routes":              routes,
            }

    runtime = time.time() - t0

    if best_result is None:
        # Every k was infeasible — extremely unlikely given pre-check
        return {
            "scenario_id":         scenario_id,
            "method":              "heuristic_kmeans_per_cluster_bfd_gnn",
            "status":              "INFEASIBLE",
            "n_nodes":             n,
            "num_vehicles_used":   0,
            "total_distance_km":   0.0,
            "total_fuel_cost_rp":  0.0,
            "total_fixed_cost_rp": 0.0,
            "total_cost_rp":       0.0,
            "best_k_clusters":     0,
            "weight_utilisation":  0.0,
            "volume_utilisation":  0.0,
            "total_weight_demand": round(sum(nd["weight"] for nd in delivery_nodes), 4),
            "total_volume_demand": round(sum(nd["volume"] for nd in delivery_nodes), 6),
            "runtime_sec":         round(runtime, 4),
            "routes":              [],
        }

    # Compute utilisation across active vehicles
    total_weight_demand  = sum(nd["weight"] for nd in delivery_nodes)
    total_volume_demand  = sum(nd["volume"] for nd in delivery_nodes)
    total_cap_w = sum(r["capacity_weight"] for r in best_result["routes"])
    total_cap_v = sum(r["capacity_volume"] for r in best_result["routes"])

    best_result["runtime_sec"]        = round(runtime, 4)
    best_result["weight_utilisation"] = round(total_weight_demand / total_cap_w * 100, 2) if total_cap_w else 0.0
    best_result["volume_utilisation"] = round(total_volume_demand / total_cap_v * 100, 2) if total_cap_v else 0.0
    best_result["total_weight_demand"]= round(total_weight_demand, 4)
    best_result["total_volume_demand"]= round(total_volume_demand, 6)

    return best_result


# ── Load dataset ───────────────────────────────────────────────────────────────

def load_dataset(path):
    df        = pd.read_csv(path)
    scenarios = {}

    for scenario_id, grp in df.groupby("scenario_id"):
        depot_row     = grp[grp["node_type"] == "depot"].iloc[0]
        delivery_rows = grp[grp["node_type"] == "delivery"]

        depot = {
            "node_id":       0,
            "location_name": depot_row["location_name"],
            "lat":           float(depot_row["lat"]),
            "lng":           float(depot_row["lng"]),
        }

        delivery_nodes = []
        for _, row in delivery_rows.iterrows():
            delivery_nodes.append({
                "node_id":       int(row["node_id"]),
                "location_name": row["location_name"],
                "lat":           float(row["lat"]),
                "lng":           float(row["lng"]),
                "weight":        float(row["total_weight_kg"]),
                "volume":        float(row["total_volume_cbm"]),
                "awb_count":     int(row["awb_count"]),
            })

        scenarios[scenario_id] = {
            "depot":                  depot,
            "delivery_nodes":         delivery_nodes,
            "city":                   depot_row["city"],
            "date":                   depot_row["date"],
            "n_nodes":                len(delivery_nodes),
            "is_lindo_representative":bool(grp["is_lindo_representative"].iloc[0]),
        }

    return scenarios


# ── Write results ──────────────────────────────────────────────────────────────

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


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  HCVRP Heuristic Solver v6")
    print("  Method: K-Means (per-cluster) + BFD + Greedy Nearest Neighbour")
    print("  Fuel: Gasoline Rp10,000/L (Blind Van, Pickup Bak)")
    print("        Diesel   Rp 6,800/L (Engkel, CDD Box)")
    print("  Key change vs v5: BFD applied PER CLUSTER, not globally")
    print("=" * 70)

    scenarios = load_dataset(DATASET_PATH)
    print(f"\nLoaded {len(scenarios)} scenarios from {DATASET_PATH}\n")

    all_results = []

    for scenario_id in sorted(scenarios.keys()):
        sc = scenarios[scenario_id]
        print(f"  Solving {scenario_id} (n={sc['n_nodes']}) ...", end=" ", flush=True)

        result = solve_scenario(scenario_id, sc["depot"], sc["delivery_nodes"])

        if result:
            result["city"]                   = sc["city"]
            result["date"]                   = sc["date"]
            result["is_lindo_representative"]= sc["is_lindo_representative"]
            result["distance_method"]        = "OSRM" if check_osrm() else "Haversine"
            all_results.append(result)

            status = result.get("status", "?")
            if status == "FEASIBLE":
                print(f"k={result['best_k_clusters']}  "
                      f"vehicles={result['num_vehicles_used']}  "
                      f"dist={result['total_distance_km']:.1f}km  "
                      f"cost=Rp{result['total_cost_rp']:,.0f}  "
                      f"runtime={result['runtime_sec']:.2f}s")
            else:
                print(f"{status}")
        else:
            print("SKIPPED (0 nodes)")
            all_results.append(None)

    # ── Summary ───────────────────────────────────────────────────────────────
    valid = [r for r in all_results if r and r.get("status") == "FEASIBLE"]

    if valid:
        print("\n" + "=" * 70)
        print("  SUMMARY")
        print("=" * 70)
        avg_veh   = np.mean([r["num_vehicles_used"]   for r in valid])
        avg_dist  = np.mean([r["total_distance_km"]   for r in valid])
        avg_cost  = np.mean([r["total_cost_rp"]        for r in valid])
        avg_wutil = np.mean([r["weight_utilisation"]   for r in valid])
        avg_vutil = np.mean([r["volume_utilisation"]   for r in valid])
        avg_rt    = np.mean([r["runtime_sec"]           for r in valid])

        print(f"  Scenarios solved        : {len(valid)}/{len(all_results)}")
        print(f"  Avg vehicles used       : {avg_veh:.2f}")
        print(f"  Avg total distance (km) : {avg_dist:.2f}")
        print(f"  Avg total cost (Rp)     : {avg_cost:,.0f}")
        print(f"  Avg weight utilisation  : {avg_wutil:.2f}%")
        print(f"  Avg volume utilisation  : {avg_vutil:.2f}%")
        print(f"  Avg runtime (sec)       : {avg_rt:.4f}")

    write_results(valid, OUTPUT_PATH)


if __name__ == "__main__":
    main()