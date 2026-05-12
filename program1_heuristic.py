"""
HCVRP Solver — Program 1: K-Means + Bin Packing Heuristic
==========================================================
Pipeline:
  1. Load & aggregate AWBs (sum weight + volume per AWB for a given depot+date)
  2. K-Means clustering  — group AWBs by geographic proximity
  3. Bin Packing         — assign clusters to vehicles (weight + volume capacity)
  4. Greedy Nearest Neighbor — order stops within each vehicle's route
  5. OSRM API            — real road distances (falls back to Haversine if unreachable)
  6. Output results per scenario + aggregate summary CSV

Usage:
  python program1_heuristic.py

Requirements:
  pip install pandas numpy scikit-learn requests
"""

import time
import math
import warnings
import itertools
import requests
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1. FLEET CONFIGURATION
# ─────────────────────────────────────────────────────────────
FLEET = [
    {"type": "Blind Van",    "max_weight": 830,  "max_volume": 2.0,  "fuel_km_per_liter": 13.5},
    {"type": "Pickup Bak",   "max_weight": 1250, "max_volume": 5.0,  "fuel_km_per_liter": 12.0},
    {"type": "Engkel (CDE)", "max_weight": 2250, "max_volume": 9.0,  "fuel_km_per_liter": 6.0},
    {"type": "CDD Box",      "max_weight": 4500, "max_volume": 15.0, "fuel_km_per_liter": 4.5},
]
# Sort ascending by capacity so Bin Packing tries smallest fitting vehicle first
FLEET_SORTED = sorted(FLEET, key=lambda v: v["max_weight"])

FUEL_PRICE_PER_LITER = 6_800  # Rp per liter (Solar/Diesel, 2025)

def route_cost_rp(distance_km, vehicle_type_dict):
    """total_cost = distance * (1 / fuel_efficiency) * fuel_price"""
    liters = distance_km / vehicle_type_dict["fuel_km_per_liter"]
    return liters * FUEL_PRICE_PER_LITER

# ─────────────────────────────────────────────────────────────
# 2. DISTANCE UTILITIES
# ─────────────────────────────────────────────────────────────
OSRM_BASE = "http://router.project-osrm.org/table/v1/driving"
_osrm_available = None   # cached after first check


def haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line distance in km using the Haversine formula."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def check_osrm():
    """Return True if the public OSRM server is reachable."""
    global _osrm_available
    if _osrm_available is not None:
        return _osrm_available
    try:
        url = f"{OSRM_BASE}/106.8456,-6.2088;106.9275,-6.1751?annotations=duration,distance"
        r = requests.get(url, timeout=5)
        _osrm_available = (r.status_code == 200)
    except Exception:
        _osrm_available = False
    if not _osrm_available:
        print("  [INFO] OSRM not reachable — using Haversine fallback. "
              "Switch to OSRM when running on your own machine.")
    return _osrm_available


def osrm_distance_matrix(coords):
    """
    Build an (n x n) distance matrix (km) via OSRM table API.
    coords: list of (lat, lng) tuples — index 0 is the depot.
    Falls back to Haversine if OSRM is unavailable.
    """
    if not check_osrm():
        return haversine_matrix(coords)

    # OSRM expects lon,lat order
    coord_str = ";".join(f"{lng},{lat}" for lat, lng in coords)
    url = f"{OSRM_BASE}/{coord_str}?annotations=distance"
    try:
        r = requests.get(url, timeout=30)
        data = r.json()
        if data.get("code") != "Ok":
            return haversine_matrix(coords)
        distances_m = data["distances"]           # metres
        n = len(coords)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i][j] = distances_m[i][j] / 1000.0   # convert to km
        return matrix
    except Exception:
        return haversine_matrix(coords)


def haversine_matrix(coords):
    """Build an (n x n) distance matrix using Haversine."""
    n = len(coords)
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
# 3. DATA LOADING & AGGREGATION
# ─────────────────────────────────────────────────────────────
def load_scenario(df_all, depot_name, order_date):
    """
    Extract and aggregate one scenario from the compiled dataset.
    Returns a DataFrame with one row per AWB:
      [awb_id, lat, lng, weight, volume, depot_lat, depot_lng]
    """
    mask = (
        (df_all["first_pickup_location_name"] == depot_name) &
        (df_all["order_date"] == order_date)
    )
    df = df_all[mask].copy()

    # Aggregate rows that share the same AWB (multiple items on one order)
    awbs = (
        df.groupby("sales_order_doc_number")
        .agg(
            lat=("delivery_location_lat", "first"),
            lng=("delivery_location_lng", "first"),
            weight=("total_weight", "sum"),
            volume=("total_volume", "sum"),
            depot_lat=("first_pickup_location_lat", "first"),
            depot_lng=("first_pickup_location_lng", "first"),
        )
        .reset_index()
        .rename(columns={"sales_order_doc_number": "awb_id"})
    )

    # Drop any AWBs with zero or missing weight/volume
    awbs = awbs[(awbs["weight"] > 0) & (awbs["volume"] > 0)].reset_index(drop=True)
    return awbs


# ─────────────────────────────────────────────────────────────
# 4. K-MEANS CLUSTERING
# ─────────────────────────────────────────────────────────────
def determine_k(awbs):
    """
    Estimate the minimum number of clusters needed so that each cluster
    is likely to fit within at least the largest vehicle's capacity.
    This is a conservative lower bound; Bin Packing will further split
    clusters that exceed vehicle capacity.
    """
    total_weight = awbs["weight"].sum()
    total_volume = awbs["volume"].sum()
    max_w = FLEET_SORTED[-1]["max_weight"]
    max_v = FLEET_SORTED[-1]["max_volume"]
    k_weight = math.ceil(total_weight / max_w)
    k_volume = math.ceil(total_volume / max_v)
    k = max(k_weight, k_volume, 1)
    # Cap at number of AWBs
    return min(k, len(awbs))


def kmeans_cluster(awbs):
    """
    Cluster AWBs geographically using K-Means.
    Returns awbs DataFrame with a new 'cluster' column.
    """
    k = determine_k(awbs)
    coords = awbs[["lat", "lng"]].values

    # Normalise coordinates so lat/lng differences are weighted equally
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    awbs = awbs.copy()
    awbs["cluster"] = km.fit_predict(coords_scaled)
    return awbs, k


# ─────────────────────────────────────────────────────────────
# 5. BIN PACKING — assign clusters to vehicles
# ─────────────────────────────────────────────────────────────
def bin_pack_clusters(awbs):
    """
    Assign clusters to vehicles using a First-Fit Decreasing (FFD) strategy.
    Each vehicle gets a subset of clusters whose combined weight and volume
    do not exceed that vehicle's capacity.

    Returns a list of vehicle assignments:
      [{"vehicle_type": str, "awb_ids": [...], "weight": float, "volume": float}, ...]
    """
    # Summarise each cluster
    cluster_summary = (
        awbs.groupby("cluster")
        .agg(weight=("weight", "sum"), volume=("volume", "sum"))
        .reset_index()
    )

    # Sort clusters by weight descending (FFD heuristic)
    cluster_summary = cluster_summary.sort_values("weight", ascending=False).reset_index(drop=True)

    vehicles = []   # list of dicts: {vehicle_type, clusters, weight, volume}

    for _, cl in cluster_summary.iterrows():
        cl_weight = cl["weight"]
        cl_volume = cl["volume"]
        cl_id     = cl["cluster"]

        # Try to fit into an existing vehicle (best fit — least remaining capacity)
        placed = False
        best_fit_idx = None
        best_fit_remaining = float("inf")

        for idx, v in enumerate(vehicles):
            v_type   = next(f for f in FLEET_SORTED if f["type"] == v["vehicle_type"])
            rem_w    = v_type["max_weight"] - v["weight"]
            rem_v    = v_type["max_volume"] - v["volume"]
            if rem_w >= cl_weight and rem_v >= cl_volume:
                # Fits — check if this is a tighter fit than current best
                remaining = min(rem_w - cl_weight, rem_v - cl_volume)
                if remaining < best_fit_remaining:
                    best_fit_remaining = remaining
                    best_fit_idx       = idx

        if best_fit_idx is not None:
            vehicles[best_fit_idx]["clusters"].append(cl_id)
            vehicles[best_fit_idx]["weight"] += cl_weight
            vehicles[best_fit_idx]["volume"] += cl_volume
            placed = True

        if not placed:
            # Open a new vehicle — pick the smallest type that fits this cluster
            assigned_type = None
            for vtype in FLEET_SORTED:
                if vtype["max_weight"] >= cl_weight and vtype["max_volume"] >= cl_volume:
                    assigned_type = vtype["type"]
                    break
            if assigned_type is None:
                # Cluster exceeds even the largest vehicle — must split it
                # Split cluster AWBs into sub-clusters that fit
                cl_awbs = awbs[awbs["cluster"] == cl_id].copy()
                sub_vehicles = _split_oversized_cluster(cl_awbs)
                vehicles.extend(sub_vehicles)
                continue

            vehicles.append({
                "vehicle_type": assigned_type,
                "clusters":     [cl_id],
                "weight":       cl_weight,
                "volume":       cl_volume,
            })

    # Expand clusters → AWB IDs
    result = []
    for v in vehicles:
        awb_ids = awbs[awbs["cluster"].isin(v["clusters"])]["awb_id"].tolist()
        result.append({
            "vehicle_type": v["vehicle_type"],
            "awb_ids":      awb_ids,
            "weight":       v["weight"],
            "volume":       v["volume"],
        })
    return result


def _split_oversized_cluster(cl_awbs):
    """
    If a single cluster exceeds the largest vehicle's capacity,
    greedily split it into sub-groups that each fit a vehicle.
    """
    sub_vehicles = []
    remaining    = cl_awbs.copy().reset_index(drop=True)

    while len(remaining) > 0:
        current_w, current_v = 0, 0
        assigned_type = None
        group_indices = []

        for i, row in remaining.iterrows():
            for vtype in FLEET_SORTED:
                if (current_w + row["weight"] <= vtype["max_weight"] and
                        current_v + row["volume"] <= vtype["max_volume"]):
                    if assigned_type is None or (
                        vtype["max_weight"] <= next(
                            f["max_weight"] for f in FLEET_SORTED if f["type"] == assigned_type
                        )
                    ):
                        assigned_type = vtype["type"]
                if (assigned_type and
                        current_w + row["weight"] <= next(
                            f["max_weight"] for f in FLEET_SORTED if f["type"] == assigned_type
                        ) and
                        current_v + row["volume"] <= next(
                            f["max_volume"] for f in FLEET_SORTED if f["type"] == assigned_type
                        )):
                    group_indices.append(i)
                    current_w += row["weight"]
                    current_v += row["volume"]
                    break

        if not group_indices:
            # Single AWB that exceeds all vehicles — assign to largest vehicle anyway
            idx = remaining.index[0]
            sub_vehicles.append({
                "vehicle_type": FLEET_SORTED[-1]["type"],
                "clusters":     [],
                "weight":       remaining.loc[idx, "weight"],
                "volume":       remaining.loc[idx, "volume"],
                "_awb_ids":     [remaining.loc[idx, "awb_id"]],
            })
            remaining = remaining.drop(index=idx).reset_index(drop=True)
        else:
            group = remaining.loc[group_indices]
            sub_vehicles.append({
                "vehicle_type": assigned_type,
                "clusters":     [],
                "weight":       current_w,
                "volume":       current_v,
                "_awb_ids":     group["awb_id"].tolist(),
            })
            remaining = remaining.drop(index=group_indices).reset_index(drop=True)

    # Normalise format
    for v in sub_vehicles:
        if "_awb_ids" in v:
            v["awb_ids"] = v.pop("_awb_ids")
    return sub_vehicles


# ─────────────────────────────────────────────────────────────
# 6. GREEDY NEAREST NEIGHBOR ROUTING
# ─────────────────────────────────────────────────────────────
def greedy_route(depot_coord, stop_coords, dist_matrix):
    """
    Build a route using Nearest Neighbor heuristic.
    depot_coord : (lat, lng) — index 0 in dist_matrix
    stop_coords : list of (lat, lng) — indices 1..n in dist_matrix
    dist_matrix : (n+1) x (n+1) matrix where row/col 0 = depot

    Returns ordered list of stop indices (1-based into stop_coords)
    and total round-trip distance in km.
    """
    n         = len(stop_coords)
    unvisited = list(range(1, n + 1))   # 1-indexed (0 = depot)
    route     = []
    current   = 0   # start at depot
    total_km  = 0.0

    while unvisited:
        nearest    = min(unvisited, key=lambda j: dist_matrix[current][j])
        total_km  += dist_matrix[current][nearest]
        route.append(nearest)
        current    = nearest
        unvisited.remove(nearest)

    # Return to depot
    total_km += dist_matrix[current][0]
    return route, total_km


# ─────────────────────────────────────────────────────────────
# 7. SOLVE ONE SCENARIO
# ─────────────────────────────────────────────────────────────
def solve_scenario(awbs):
    """
    Run the full heuristic pipeline for one scenario.
    Returns a dict of results.
    """
    depot_lat = awbs["depot_lat"].iloc[0]
    depot_lng = awbs["depot_lng"].iloc[0]
    depot_coord = (depot_lat, depot_lng)

    # Step 1 — K-Means clustering
    awbs_clustered, k_used = kmeans_cluster(awbs)

    # Step 2 — Bin Packing
    vehicle_assignments = bin_pack_clusters(awbs_clustered)

    # Step 3 — Route each vehicle
    total_distance_km = 0.0
    routes_detail     = []

    for v_idx, vehicle in enumerate(vehicle_assignments):
        awb_ids    = vehicle["awb_ids"]
        v_awbs     = awbs_clustered[awbs_clustered["awb_id"].isin(awb_ids)]
        stop_coords = [(row["lat"], row["lng"]) for _, row in v_awbs.iterrows()]

        # Build distance matrix (depot + stops)
        all_coords   = [depot_coord] + stop_coords
        dist_matrix  = osrm_distance_matrix(all_coords)

        # Greedy nearest neighbour
        route_order, route_km = greedy_route(depot_coord, stop_coords, dist_matrix)

        # Map route indices back to AWB IDs
        ordered_awbs = [awb_ids[i - 1] for i in route_order]

        # Vehicle cost
        vtype      = next(f for f in FLEET_SORTED if f["type"] == vehicle["vehicle_type"])
        route_cost = route_cost_rp(route_km, vtype)

        total_distance_km += route_km
        routes_detail.append({
            "vehicle_index": v_idx + 1,
            "vehicle_type":  vehicle["vehicle_type"],
            "awb_count":     len(awb_ids),
            "weight_kg":     round(vehicle["weight"], 2),
            "volume_cbm":    round(vehicle["volume"], 3),
            "weight_util":   round(vehicle["weight"] / vtype["max_weight"] * 100, 1),
            "volume_util":   round(vehicle["volume"] / vtype["max_volume"] * 100, 1),
            "distance_km":   round(route_km, 2),
            "route_cost_rp": round(route_cost),
            "ordered_awbs":  ordered_awbs,
        })

    # Summary
    vehicles_used   = len(vehicle_assignments)
    vehicle_counts  = {}
    for v in routes_detail:
        vehicle_counts[v["vehicle_type"]] = vehicle_counts.get(v["vehicle_type"], 0) + 1

    total_cost = sum(r["route_cost_rp"] for r in routes_detail)
    avg_weight_util = sum(r["weight_util"] for r in routes_detail) / vehicles_used
    avg_volume_util = sum(r["volume_util"] for r in routes_detail) / vehicles_used

    return {
        "vehicles_used":      vehicles_used,
        "vehicle_breakdown":  vehicle_counts,
        "total_distance_km":  round(total_distance_km, 2),
        "total_cost_rp":      round(total_cost),  # fuel cost only: distance * (1/efficiency) * fuel_price
        "avg_weight_util_pct": round(avg_weight_util, 1),
        "avg_volume_util_pct": round(avg_volume_util, 1),
        "k_clusters_used":    k_used,
        "routes":             routes_detail,
    }


# ─────────────────────────────────────────────────────────────
# 8. MAIN — run all scenarios
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("HCVRP Heuristic Solver — K-Means + Bin Packing")
    print("=" * 60)

    # Load compiled dataset
    df = pd.read_csv("dataset_compiled.csv", low_memory=False)
    df["order_date"] = pd.to_datetime(df["order_date"]).dt.date

    # Get all unique scenarios
    scenarios = (
        df[["source", "first_pickup_location_name", "order_date"]]
        .drop_duplicates()
        .sort_values(["source", "first_pickup_location_name", "order_date"])
        .reset_index(drop=True)
    )

    print(f"Total scenarios to solve: {len(scenarios)}\n")

    summary_rows = []

    for _, sc in scenarios.iterrows():
        source     = sc["source"]
        depot      = sc["first_pickup_location_name"]
        date       = sc["order_date"]
        label      = f"{source} | {depot} | {date}"

        print(f"Solving: {label}")
        t_start = time.time()

        # Load & aggregate AWBs for this scenario
        awbs = load_scenario(df, depot, date)

        if len(awbs) < 2:
            print(f"  SKIP — only {len(awbs)} AWB(s), not enough for routing.\n")
            continue

        # Solve
        result = solve_scenario(awbs)
        elapsed = round(time.time() - t_start, 2)

        # Print scenario summary
        print(f"  AWBs           : {len(awbs)}")
        print(f"  Clusters (k)   : {result['k_clusters_used']}")
        print(f"  Vehicles used  : {result['vehicles_used']}  {result['vehicle_breakdown']}")
        print(f"  Total distance : {result['total_distance_km']} km")
        print(f"  Total cost     : Rp {result['total_cost_rp']:,}")
        print(f"  Avg weight util: {result['avg_weight_util_pct']}%")
        print(f"  Avg volume util: {result['avg_volume_util_pct']}%")
        print(f"  Runtime        : {elapsed}s\n")

        summary_rows.append({
            "source":             source,
            "depot":              depot,
            "date":               date,
            "awb_count":          len(awbs),
            "k_clusters":         result["k_clusters_used"],
            "vehicles_used":      result["vehicles_used"],
            "vehicle_breakdown":  str(result["vehicle_breakdown"]),
            "total_distance_km":  result["total_distance_km"],
            "total_cost_rp":      result["total_cost_rp"],
            "avg_weight_util_pct": result["avg_weight_util_pct"],
            "avg_volume_util_pct": result["avg_volume_util_pct"],
            "runtime_sec":        elapsed,
            "distance_method":    "OSRM" if check_osrm() else "Haversine",
        })

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results_heuristic.csv", index=False)

    # Print aggregate stats
    print("=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    print(f"Scenarios solved       : {len(summary_df)}")
    print(f"Avg AWBs per scenario  : {summary_df['awb_count'].mean():.1f}")
    print(f"Avg vehicles used      : {summary_df['vehicles_used'].mean():.2f}")
    print(f"Avg total distance km  : {summary_df['total_distance_km'].mean():.2f}")
    print(f"Avg total cost Rp      : {summary_df['total_cost_rp'].mean():,.0f}")
    print(f"Avg weight utilisation : {summary_df['avg_weight_util_pct'].mean():.1f}%")
    print(f"Avg volume utilisation : {summary_df['avg_volume_util_pct'].mean():.1f}%")
    print(f"Avg runtime per scenario: {summary_df['runtime_sec'].mean():.2f}s")
    print(f"\nResults saved → results_heuristic.csv")


if __name__ == "__main__":
    main()
