"""
=============================================================================
METHOD 1: K-MEANS CLUSTERING + FIRST-FIT DECREASING BIN PACKING
Heterogeneous Capacitated Vehicle Routing Problem (HCVRP)
Weight + Volume Dual-Capacity Constraints

Journal: "Integer Linear Programming Based Optimization Model for
          Heterogeneous Capacitated Vehicle Routing Problem with
          Google OR-Tools Implementation"

Description:
    Heuristic baseline approach using two-stage pipeline:
    1. K-Means clustering to group SOs geographically per origin
    2. First-Fit Decreasing (FFD) bin packing to assign SOs to vehicles
       respecting weight and volume constraints
    3. Nearest Neighbor heuristic to sequence stops within each route

Constraints:
    - SOs cannot be split across vehicles
    - Single pickup point per route (multi-drop only)
    - SOs grouped only within same origin warehouse
    - Each route assigned optimal vehicle type based on load

Dependencies:
    pip install pandas numpy scikit-learn scipy
=============================================================================
"""

import pandas as pd
import numpy as np
import math
import time
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

warnings.filterwarnings("ignore")

# =============================================================================
# FLEET CONFIGURATION — Indonesian Logistics Standards
# =============================================================================
FLEET_TYPES = [
    {"type": "Pickup",   "max_weight_kg": 1500,  "max_volume_cbm": 2.0,  "cost_per_km": 5000},
    {"type": "CDE",      "max_weight_kg": 2000,  "max_volume_cbm": 6.0,  "cost_per_km": 7000},
    {"type": "CDD",      "max_weight_kg": 4000,  "max_volume_cbm": 12.0, "cost_per_km": 9000},
    {"type": "CDD Long", "max_weight_kg": 6000,  "max_volume_cbm": 20.0, "cost_per_km": 11000},
    {"type": "Fuso",     "max_weight_kg": 8000,  "max_volume_cbm": 25.0, "cost_per_km": 14000},
    {"type": "Tronton",  "max_weight_kg": 15000, "max_volume_cbm": 30.0, "cost_per_km": 18000},
]
# Sort ascending by capacity so we always pick the smallest fitting vehicle
FLEET_TYPES = sorted(FLEET_TYPES, key=lambda x: x["max_weight_kg"])


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two coordinates in kilometers."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.asin(math.sqrt(a))


def select_vehicle_type(total_weight, total_volume):
    """
    Select the smallest vehicle type that can accommodate the given
    weight and volume. Returns None if no vehicle type is sufficient.
    """
    for fleet in FLEET_TYPES:
        if total_weight <= fleet["max_weight_kg"] and total_volume <= fleet["max_volume_cbm"]:
            return fleet
    return None  # Oversized — should not happen if data is clean


def nearest_neighbor_route(stops, origin_lat, origin_lng):
    """
    Nearest Neighbor heuristic to sequence delivery stops.
    Starts from origin, always visits closest unvisited stop next.
    Returns ordered list of stop indices and total route distance (km).
    """
    if not stops:
        return [], 0.0

    unvisited = list(range(len(stops)))
    route = []
    current_lat, current_lng = origin_lat, origin_lng
    total_dist = 0.0

    while unvisited:
        nearest_idx = min(
            unvisited,
            key=lambda i: haversine_km(current_lat, current_lng,
                                       stops[i]["lat"], stops[i]["lng"])
        )
        stop = stops[nearest_idx]
        dist = haversine_km(current_lat, current_lng, stop["lat"], stop["lng"])
        total_dist += dist
        current_lat, current_lng = stop["lat"], stop["lng"]
        route.append(nearest_idx)
        unvisited.remove(nearest_idx)

    # Return to origin (depot)
    total_dist += haversine_km(current_lat, current_lng, origin_lat, origin_lng)
    return route, total_dist


def compute_route_distance(route_stops, origin_lat, origin_lng):
    """Compute total distance of a given ordered route."""
    if not route_stops:
        return 0.0
    total = haversine_km(origin_lat, origin_lng, route_stops[0]["lat"], route_stops[0]["lng"])
    for i in range(len(route_stops) - 1):
        total += haversine_km(
            route_stops[i]["lat"], route_stops[i]["lng"],
            route_stops[i+1]["lat"], route_stops[i+1]["lng"]
        )
    total += haversine_km(route_stops[-1]["lat"], route_stops[-1]["lng"], origin_lat, origin_lng)
    return total


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================
def load_and_preprocess(filepath):
    """
    Load raw CSV, aggregate to SO-level, return clean DataFrame.
    Each row = one Sales Order with total weight, volume, and coordinates.
    """
    print("Loading data...")
    df = pd.read_csv(filepath, low_memory=False)

    # Aggregate line items to SO level
    so_df = df.groupby("sales_order_doc_number").agg(
        total_weight_kg=("total_weight", "sum"),
        total_volume_cbm=("total_volume", "sum"),
        line_item_count=("item_name", "count"),
        delivery_lat=("delivery_location_lat", "first"),
        delivery_lng=("delivery_location_lng", "first"),
        delivery_name=("delivery_location_name", "first"),
        pickup_name=("first_pickup_location_name", "first"),
        pickup_lat=("first_pickup_location_lat", "first"),
        pickup_lng=("first_pickup_location_lng", "first"),
        existing_group=("group", "first"),
    ).reset_index()

    # Drop SOs with missing coordinates or zero load
    before = len(so_df)
    so_df = so_df.dropna(subset=["delivery_lat", "delivery_lng", "pickup_lat", "pickup_lng"])
    so_df = so_df[so_df["total_weight_kg"] > 0]
    so_df = so_df[so_df["total_volume_cbm"] > 0]
    after = len(so_df)

    print(f"  Loaded {before} SOs → {after} valid SOs after cleaning")
    print(f"  Origins: {so_df['pickup_name'].value_counts().to_dict()}")
    return so_df.reset_index(drop=True)


# =============================================================================
# STAGE 1: K-MEANS CLUSTERING
# =============================================================================
def estimate_k(so_group_df, fleet_types):
    """
    Estimate number of clusters (vehicles) needed for a group of SOs.
    Uses the largest vehicle type as reference to get a lower bound,
    then multiplies by a buffer factor to account for geographic spread.
    """
    largest = fleet_types[-1]
    total_w = so_group_df["total_weight_kg"].sum()
    total_v = so_group_df["total_volume_cbm"].sum()
    k_by_weight = math.ceil(total_w / largest["max_weight_kg"])
    k_by_volume = math.ceil(total_v / largest["max_volume_cbm"])
    k_base = max(k_by_weight, k_by_volume, 1)
    # Buffer factor 1.3 to account for geographic constraints
    return max(math.ceil(k_base * 1.3), 2)


def kmeans_cluster_sos(so_group_df, origin_name):
    """
    Apply K-Means clustering to SOs within one origin warehouse.
    Features: delivery lat/lng (geographic proximity).
    Returns so_group_df with a 'cluster' column added.
    """
    coords = so_group_df[["delivery_lat", "delivery_lng"]].values

    if len(so_group_df) == 1:
        so_group_df = so_group_df.copy()
        so_group_df["cluster"] = 0
        return so_group_df

    k = estimate_k(so_group_df, FLEET_TYPES)
    k = min(k, len(so_group_df))  # Can't have more clusters than SOs

    print(f"  [{origin_name}] {len(so_group_df)} SOs → K-Means with K={k}")

    # Scale coordinates for clustering
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    # Run K-Means with multiple restarts for stability
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=15,
        max_iter=500,
        random_state=42
    )
    labels = kmeans.fit_predict(coords_scaled)

    so_group_df = so_group_df.copy()
    so_group_df["cluster"] = labels
    return so_group_df


# =============================================================================
# STAGE 2: FIRST-FIT DECREASING (FFD) BIN PACKING
# =============================================================================
def ffd_bin_packing(cluster_sos, origin_lat, origin_lng):
    """
    First-Fit Decreasing bin packing within a K-Means cluster.

    Algorithm:
    1. Sort SOs by total_weight descending (largest first)
    2. Try to fit each SO into an existing bin (vehicle)
    3. If no existing bin fits, open a new bin
    4. Each bin gets the smallest vehicle type that fits its total load

    Each bin = one delivery route departing from origin_lat/lng.

    Returns list of route dicts with assigned SOs, vehicle type, and metrics.
    """
    # Sort SOs: heaviest first (FFD strategy)
    sorted_sos = cluster_sos.sort_values("total_weight_kg", ascending=False).to_dict("records")

    bins = []  # Each bin: {"sos": [...], "weight": float, "volume": float}

    for so in sorted_sos:
        placed = False
        for b in bins:
            # Check if SO fits in this bin using the largest vehicle type
            # We'll assign actual vehicle type after bin is closed
            new_w = b["weight"] + so["total_weight_kg"]
            new_v = b["volume"] + so["total_volume_cbm"]
            # Can any vehicle fit this combined load?
            if select_vehicle_type(new_w, new_v) is not None:
                b["sos"].append(so)
                b["weight"] = new_w
                b["volume"] = new_v
                placed = True
                break

        if not placed:
            # Check if this single SO fits in any vehicle at all
            vehicle = select_vehicle_type(so["total_weight_kg"], so["total_volume_cbm"])
            if vehicle is None:
                print(f"    WARNING: SO {so['sales_order_doc_number']} "
                      f"({so['total_weight_kg']:.1f}kg, {so['total_volume_cbm']:.3f}cbm) "
                      f"exceeds all vehicle capacities — assigning to largest vehicle.")
                vehicle = FLEET_TYPES[-1]
            bins.append({
                "sos": [so],
                "weight": so["total_weight_kg"],
                "volume": so["total_volume_cbm"]
            })

    # Build route objects
    routes = []
    for i, b in enumerate(bins):
        vehicle = select_vehicle_type(b["weight"], b["volume"])
        if vehicle is None:
            vehicle = FLEET_TYPES[-1]

        stops = [{"so": s["sales_order_doc_number"],
                  "name": s["delivery_name"],
                  "lat": s["delivery_lat"],
                  "lng": s["delivery_lng"],
                  "weight": s["total_weight_kg"],
                  "volume": s["total_volume_cbm"]} for s in b["sos"]]

        # Sequence stops via Nearest Neighbor
        ordered_indices, dist_km = nearest_neighbor_route(stops, origin_lat, origin_lng)
        ordered_stops = [stops[i] for i in ordered_indices]

        routes.append({
            "route_id": i,
            "vehicle_type": vehicle["type"],
            "max_weight_kg": vehicle["max_weight_kg"],
            "max_volume_cbm": vehicle["max_volume_cbm"],
            "total_weight_kg": b["weight"],
            "total_volume_cbm": b["volume"],
            "weight_utilization_pct": round(b["weight"] / vehicle["max_weight_kg"] * 100, 2),
            "volume_utilization_pct": round(b["volume"] / vehicle["max_volume_cbm"] * 100, 2),
            "num_stops": len(stops),
            "distance_km": round(dist_km, 3),
            "stops": ordered_stops,
        })

    return routes


# =============================================================================
# MAIN OPTIMIZATION PIPELINE
# =============================================================================
def optimize_kmeans_binpacking(filepath):
    """
    Full pipeline:
      Load → group by origin → K-Means cluster → FFD bin pack → NN route
    Returns summary DataFrame and detailed route list.
    """
    start_time = time.time()

    # Load data
    so_df = load_and_preprocess(filepath)

    all_routes = []
    global_route_id = 0

    print("\n--- Stage 1 & 2: Clustering + Bin Packing per Origin ---")
    for origin_name, group in so_df.groupby("pickup_name"):
        origin_lat = group["pickup_lat"].iloc[0]
        origin_lng = group["pickup_lng"].iloc[0]

        # Stage 1: K-Means clustering
        clustered = kmeans_cluster_sos(group, origin_name)

        origin_routes = []
        for cluster_id, cluster_group in clustered.groupby("cluster"):
            # Stage 2: FFD bin packing within each cluster
            routes = ffd_bin_packing(cluster_group, origin_lat, origin_lng)
            for r in routes:
                r["origin"] = origin_name
                r["cluster_id"] = cluster_id
                r["global_route_id"] = global_route_id
                global_route_id += 1
                origin_routes.append(r)

        all_routes.extend(origin_routes)
        print(f"  [{origin_name}] → {len(origin_routes)} routes")

    elapsed = time.time() - start_time

    # ==========================================================================
    # BUILD RESULTS
    # ==========================================================================
    summary_rows = []
    for r in all_routes:
        summary_rows.append({
            "route_id": r["global_route_id"],
            "origin": r["origin"],
            "vehicle_type": r["vehicle_type"],
            "num_stops": r["num_stops"],
            "total_weight_kg": round(r["total_weight_kg"], 3),
            "max_weight_kg": r["max_weight_kg"],
            "total_volume_cbm": round(r["total_volume_cbm"], 4),
            "max_volume_cbm": r["max_volume_cbm"],
            "weight_utilization_pct": r["weight_utilization_pct"],
            "volume_utilization_pct": r["volume_utilization_pct"],
            "distance_km": r["distance_km"],
            "stop_sequence": " → ".join([s["so"] for s in r["stops"]]),
        })

    summary_df = pd.DataFrame(summary_rows)

    # ==========================================================================
    # PRINT PERFORMANCE REPORT
    # ==========================================================================
    print("\n" + "=" * 70)
    print("  METHOD 1: K-MEANS + BIN PACKING — RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Computation time       : {elapsed:.2f} seconds")
    print(f"  Total routes (vehicles): {len(all_routes)}")
    print(f"  Total distance (km)    : {summary_df['distance_km'].sum():.2f}")
    print(f"  Avg weight utilization : {summary_df['weight_utilization_pct'].mean():.1f}%")
    print(f"  Avg volume utilization : {summary_df['volume_utilization_pct'].mean():.1f}%")
    print(f"  Total SOs served       : {summary_df['num_stops'].sum()}")

    print("\n  Vehicles used by type:")
    vtype_counts = summary_df.groupby(["origin", "vehicle_type"]).size().reset_index(name="count")
    print(vtype_counts.to_string(index=False))

    print("\n  Per-origin summary:")
    origin_summary = summary_df.groupby("origin").agg(
        routes=("route_id", "count"),
        total_dist_km=("distance_km", "sum"),
        avg_weight_util=("weight_utilization_pct", "mean"),
        avg_vol_util=("volume_utilization_pct", "mean"),
    ).round(2)
    print(origin_summary.to_string())
    print("=" * 70)

    return summary_df, all_routes, elapsed


# =============================================================================
# EXPORT RESULTS
# =============================================================================
def export_results(summary_df, all_routes, output_prefix="method1_kmeans"):
    """Export route summary and detailed stop sequences to CSV."""

    # Summary CSV
    summary_path = f"{output_prefix}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Exported route summary → {summary_path}")

    # Detailed stops CSV
    detail_rows = []
    for r in all_routes:
        for seq, stop in enumerate(r["stops"], start=1):
            detail_rows.append({
                "global_route_id": r["global_route_id"],
                "origin": r["origin"],
                "vehicle_type": r["vehicle_type"],
                "stop_sequence": seq,
                "sales_order_no": stop["so"],
                "delivery_name": stop["name"],
                "delivery_lat": stop["lat"],
                "delivery_lng": stop["lng"],
                "weight_kg": round(stop["weight"], 3),
                "volume_cbm": round(stop["volume"], 4),
            })

    detail_df = pd.DataFrame(detail_rows)
    detail_path = f"{output_prefix}_stop_detail.csv"
    detail_df.to_csv(detail_path, index=False)
    print(f"  Exported stop detail   → {detail_path}")

    return summary_path, detail_path


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    DATA_PATH = "_SELECT_toso_tosoli_FROM_t_opt_sales_orders_toso_LEFT_JOIN_t_opt_202602191544.csv"

    print("=" * 70)
    print("  HCVRP — METHOD 1: K-MEANS CLUSTERING + FFD BIN PACKING")
    print("=" * 70 + "\n")

    summary_df, all_routes, elapsed = optimize_kmeans_binpacking(DATA_PATH)
    export_results(summary_df, all_routes)

    print(f"\nDone in {elapsed:.2f}s")
