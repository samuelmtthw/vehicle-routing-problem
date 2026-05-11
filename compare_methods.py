"""
=============================================================================
COMPARISON RUNNER
Runs both methods and produces a side-by-side performance comparison.

Usage:
    # Run both methods and compare:
    python3 compare_methods.py

    # Run only Method 1 (no OR-Tools needed):
    python3 compare_methods.py --method1-only

Dependencies:
    Method 1 only : pip install pandas numpy scikit-learn scipy
    Both methods  : pip install pandas numpy scikit-learn scipy ortools
=============================================================================
"""

import sys
import time
import pandas as pd

DATA_PATH = "_SELECT_toso_tosoli_FROM_t_opt_sales_orders_toso_LEFT_JOIN_t_opt_202602191544.csv"

# ─────────────────────────────────────────────────────────────────────────────
# RUN METHOD 1: K-Means + Bin Packing
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "█" * 70)
print("  RUNNING METHOD 1: K-MEANS + FIRST-FIT DECREASING BIN PACKING")
print("█" * 70)

from method1_kmeans_binpacking import optimize_kmeans_binpacking, export_results as export_m1
summary_m1, routes_m1, elapsed_m1 = optimize_kmeans_binpacking(DATA_PATH)
export_m1(summary_m1, routes_m1, output_prefix="method1_kmeans")

# ─────────────────────────────────────────────────────────────────────────────
# RUN METHOD 2: ILP + OR-Tools (only if available and not --method1-only)
# ─────────────────────────────────────────────────────────────────────────────
run_m2 = "--method1-only" not in sys.argv

summary_m2 = None
elapsed_m2 = None

if run_m2:
    try:
        from ortools.constraint_solver import pywraprouting
        print("\n" + "█" * 70)
        print("  RUNNING METHOD 2: ILP + GOOGLE OR-TOOLS")
        print("█" * 70)
        from method2_ilp_ortools import optimize_ilp_ortools, export_results as export_m2
        summary_m2, routes_m2, elapsed_m2 = optimize_ilp_ortools(DATA_PATH)
        export_m2(summary_m2, routes_m2, output_prefix="method2_ortools")
    except ImportError:
        print("\n  [SKIP] OR-Tools not installed. Run: pip install ortools")
        print("         Only Method 1 results will be shown.\n")
        run_m2 = False

# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
if run_m2 and summary_m2 is not None:
    from method2_ilp_ortools import compare_methods
    compare_methods(summary_m1, elapsed_m1, summary_m2, elapsed_m2)

    # Export combined comparison to CSV
    comp_rows = []
    for metric, (v1, v2) in {
        "Total routes (vehicles)": (len(summary_m1), len(summary_m2)),
        "Total distance (km)": (round(summary_m1["distance_km"].sum(), 2),
                                round(summary_m2["distance_km"].sum(), 2)),
        "Avg weight utilization (%)": (round(summary_m1["weight_utilization_pct"].mean(), 2),
                                       round(summary_m2["weight_utilization_pct"].mean(), 2)),
        "Avg volume utilization (%)": (round(summary_m1["volume_utilization_pct"].mean(), 2),
                                       round(summary_m2["volume_utilization_pct"].mean(), 2)),
        "Computation time (s)": (round(elapsed_m1, 2), round(elapsed_m2, 2)),
        "Total SOs served": (int(summary_m1["num_stops"].sum()),
                             int(summary_m2["num_stops"].sum())),
    }.items():
        comp_rows.append({
            "Metric": metric,
            "Method1_KMeans_BinPacking": v1,
            "Method2_ILP_ORTools": v2,
        })
    pd.DataFrame(comp_rows).to_csv("comparison_results.csv", index=False)
    print("\n  Exported comparison table → comparison_results.csv")

else:
    print("\n" + "=" * 70)
    print("  METHOD 1 RESULTS ONLY (OR-Tools not available)")
    print("=" * 70)
    print(f"  Routes generated : {len(summary_m1)}")
    print(f"  Total distance   : {summary_m1['distance_km'].sum():.2f} km")
    print(f"  Avg weight util  : {summary_m1['weight_utilization_pct'].mean():.1f}%")
    print(f"  Avg volume util  : {summary_m1['volume_utilization_pct'].mean():.1f}%")
    print(f"  Computation time : {elapsed_m1:.2f}s")
    print("=" * 70)

print("\nAll done!")
