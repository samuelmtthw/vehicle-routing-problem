"""
build_distance_matrix_s05.py
============================
Builds a road-distance matrix for all S05 locations (11 nodes: depot + 10
delivery nodes) using the public OSRM routing server, then writes the
results to CSV.

Distance pairs produced
-----------------------
  depot → n1 … n10          (10 pairs)
  n1    → n2 … n10          ( 9 pairs)
  n2    → n3 … n10          ( 8 pairs)
  …
  n9    → n10               ( 1 pair)
  Total : 55 unique directed pairs (upper triangle)

Outputs
-------
  distance_matrix_s05_pairwise.csv   – one row per pair (from/to/distance)
  distance_matrix_s05_full.csv       – 11×11 full matrix table

Usage
-----
  python build_distance_matrix_s05.py
"""

import math
import sys

import pandas as pd
import requests

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET_PATH   = "dataset_compiled_v2.csv"
SCENARIO_ID    = "S05_Batam_2025-12-18"
OSRM_BASE      = "http://router.project-osrm.org/table/v1/driving"
OUTPUT_PAIRS   = "distance_matrix_s05_pairwise.csv"
OUTPUT_MATRIX  = "distance_matrix_s05_full.csv"


# ── Haversine fallback ─────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(phi1) * math.cos(phi2)
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── OSRM availability check ────────────────────────────────────────────────────
def check_osrm():
    try:
        test_url = (f"{OSRM_BASE}/104.065,1.110;104.056,1.129"
                    f"?annotations=distance")
        r = requests.get(test_url, timeout=5)
        ok = r.status_code == 200
    except Exception:
        ok = False
    if not ok:
        print("  [INFO] OSRM not reachable — falling back to Haversine distances.")
    return ok


# ── Distance matrix builders ───────────────────────────────────────────────────
def haversine_matrix(nodes):
    n = len(nodes)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(nodes[i]["lat"], nodes[i]["lng"],
                          nodes[j]["lat"], nodes[j]["lng"])
            D[i][j] = D[j][i] = d
    return D


def osrm_matrix(nodes):
    """
    Query OSRM /table endpoint for all n×n road distances (km).
    Returns the full symmetric matrix.
    """
    # OSRM expects lon,lat order
    coord_str = ";".join(f"{nd['lng']},{nd['lat']}" for nd in nodes)
    url = f"{OSRM_BASE}/{coord_str}?annotations=distance"
    print(f"  Querying OSRM: {url[:80]}…")
    try:
        r    = requests.get(url, timeout=30)
        data = r.json()
        if data.get("code") != "Ok":
            print(f"  [WARN] OSRM returned code={data.get('code')}. "
                  f"Falling back to Haversine.")
            return None, "Haversine (OSRM error)"
        n = len(nodes)
        D = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                D[i][j] = data["distances"][i][j] / 1000.0  # m → km
        return D, "OSRM"
    except Exception as exc:
        print(f"  [WARN] OSRM request failed ({exc}). Falling back to Haversine.")
        return None, "Haversine (exception)"


def build_matrix(nodes):
    if check_osrm():
        D, method = osrm_matrix(nodes)
        if D is not None:
            return D, method
    return haversine_matrix(nodes), "Haversine"


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # 1. Load dataset and filter S05
    print(f"Loading dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    s05 = df[df["scenario_id"] == SCENARIO_ID].sort_values("node_id").reset_index(drop=True)

    if s05.empty:
        sys.exit(f"[ERROR] Scenario '{SCENARIO_ID}' not found in {DATASET_PATH}.")

    print(f"Scenario : {SCENARIO_ID}")
    print(f"Nodes    : {len(s05)} (depot + {len(s05) - 1} delivery)")
    print()

    # 2. Build node list
    nodes = []
    for _, row in s05.iterrows():
        nodes.append({
            "node_id":  int(row["node_id"]),
            "label":    "depot" if row["node_type"] == "depot" else f"n{int(row['node_id'])}",
            "name":     row["location_name"],
            "lat":      float(row["lat"]),
            "lng":      float(row["lng"]),
        })

    # Print node summary
    print(f"{'ID':<6} {'Label':<8} {'Name':<20} {'Lat':>10} {'Lng':>10}")
    print("-" * 58)
    for nd in nodes:
        print(f"{nd['node_id']:<6} {nd['label']:<8} {nd['name']:<20} "
              f"{nd['lat']:>10.4f} {nd['lng']:>10.4f}")
    print()

    # 3. Build distance matrix
    print("Building distance matrix…")
    D, method = build_matrix(nodes)
    print(f"  Distance method : {method}")
    print()

    n = len(nodes)

    # 4. Build pairwise CSV (upper triangle only, depot→n1 … n9→n10)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                "from_id":    nodes[i]["node_id"],
                "from_label": nodes[i]["label"],
                "from_name":  nodes[i]["name"],
                "to_id":      nodes[j]["node_id"],
                "to_label":   nodes[j]["label"],
                "to_name":    nodes[j]["name"],
                "distance_km": round(D[i][j], 4),
            })

    df_pairs = pd.DataFrame(pairs)
    df_pairs.to_csv(OUTPUT_PAIRS, index=False)
    print(f"Pairwise results saved  → {OUTPUT_PAIRS}  ({len(df_pairs)} rows)")

    # 5. Build full 11×11 matrix CSV
    labels = [nd["label"] for nd in nodes]
    matrix_data = {}
    for i, lbl_i in enumerate(labels):
        matrix_data[lbl_i] = {lbl_j: round(D[i][j], 4) for j, lbl_j in enumerate(labels)}

    df_matrix = pd.DataFrame(matrix_data, index=labels).T
    df_matrix.index.name = "from \\ to"
    df_matrix.to_csv(OUTPUT_MATRIX)
    print(f"Full matrix saved       → {OUTPUT_MATRIX}  ({n}×{n})")
    print()

    # 6. Print a readable summary to console
    print("=" * 72)
    print(f"DISTANCE MATRIX SUMMARY — {SCENARIO_ID}  [{method}]")
    print("=" * 72)
    # Header row
    header = f"{'':12s}" + "".join(f"{lbl:>10s}" for lbl in labels)
    print(header)
    print("-" * len(header))
    for i, lbl_i in enumerate(labels):
        row_str = f"{lbl_i:<12s}"
        for j in range(n):
            if i == j:
                row_str += f"{'—':>10s}"
            else:
                row_str += f"{D[i][j]:>10.2f}"
        print(row_str)
    print()
    print("Values are road distances in kilometres.")
    print(f"Diagonal (—) = same location (distance = 0).")


if __name__ == "__main__":
    main()
