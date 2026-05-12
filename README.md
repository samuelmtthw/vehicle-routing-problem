# HCVRP Solver — Thesis Research Code
**Heterogeneous Capacitated Vehicle Routing Problem**
Comparison: K-Means + Bin Packing (Heuristic) vs. ILP with Google OR-Tools (Exact)

---

## Contents

```
hcvrp_solver/
├── README.md                  ← this file
├── dataset_compiled.csv       ← anonymized dataset (41 scenarios, 2 clients, 7 depots)
├── program1_heuristic.py      ← Program 1: K-Means + Bin Packing heuristic
└── program2_ilp.py            ← Program 2: ILP solver using Google OR-Tools
```

---

## Requirements

Python 3.9 or higher is recommended.

Install all dependencies with:

```bash
pip install pandas numpy scikit-learn requests ortools
```

---

## How to Run

Both programs read from `dataset_compiled.csv` and must be run from the same folder as the dataset.

**Program 1 — Heuristic (K-Means + Bin Packing):**
```bash
python program1_heuristic.py
```
Output: `results_heuristic.csv`
Expected runtime: ~2–5 seconds total for all 41 scenarios.

**Program 2 — Exact (ILP with OR-Tools):**
```bash
python program2_ilp.py
```
Output: `results_ilp.csv`
Expected runtime: ~30–60 minutes total (up to 60s per scenario by default).

To adjust the solver time limit, edit this line in `program2_ilp.py`:
```python
SOLVER_TIME_LIMIT_SEC = 60   # increase for better solution quality
```

---

## Distance Calculation

Both programs use the **OSRM public API** (`router.project-osrm.org`) to retrieve real-world road distances. If OSRM is unreachable, both programs automatically fall back to **Haversine (straight-line)** distance and print an info message.

To use Google Maps Distance Matrix API instead, replace the `osrm_distance_matrix()` function in both programs with a call to:
```
https://maps.googleapis.com/maps/api/distancematrix/json
```
and supply your API key.

---

## Dataset

`dataset_compiled.csv` contains **41 routing scenarios** across **7 depots** and **2 clients**, spanning July 2025 – January 2026. All personally identifiable information has been anonymized:

| Field | Anonymization |
|-------|--------------|
| Customer name | `CLIENT_01`, `CLIENT_02` |
| AWB number | `AWB_00001` ... `AWB_01385` |
| Delivery location name | `Ruko XXXX`, `Mall XXXX`, `FnB XXXX`, etc. |
| Delivery address | `[District: X, City: Y]` |
| Item name | `Goods` |
| Depot name | `Depot Jatiasih`, `Depot Balikpapan`, etc. |
| GPS coordinates | Kept as-is (required for routing) |

Scenarios used:

| Source | Depot | Dates | AWBs/day |
|--------|-------|-------|----------|
| CLIENT_01 (KK) | Depot Balikpapan | Dec 26 | 38 |
| CLIENT_01 (KK) | Depot Batam | Dec 18, Dec 24 | 28–29 |
| CLIENT_01 (KK) | Depot Makassar | Dec 19, Dec 23 | 26–63 |
| CLIENT_01 (KK) | Depot Manado | Dec 18, Dec 24 | 17–32 |
| CLIENT_01 (KK) | Depot Medan | Dec 23, Dec 24 | 25–83 |
| CLIENT_01 (KK) | Depot Pekanbaru | Dec 8, 11, 18, 23, 24 | 21–44 |
| CLIENT_02 (HANGRY) | Depot Jatiasih | 27 days, Jul 2025–Jan 2026 | 28–46 |

---

## Fleet Configuration

Both programs use the same heterogeneous fleet:

| Vehicle Type | Max Weight (kg) | Max Volume (CBM) | Fixed Cost (Rp) |
|-------------|----------------|-----------------|----------------|
| Blind Van | 830 | 2.0 | 150,000 |
| Pickup Bak | 1,250 | 5.0 | 200,000 |
| Engkel (CDE) | 2,250 | 9.0 | 300,000 |
| CDD Box | 4,500 | 15.0 | 450,000 |

Capacity values are midpoints of industry-standard ranges for Indonesian logistics vehicles,
cross-referenced against Waresix, Deliveree, and ALD Logistik published specifications.
Variable travel cost: Rp 5,000/km (uniform across vehicle types).

---

## Output Columns

Both `results_heuristic.csv` and `results_ilp.csv` share the same structure:

| Column | Description |
|--------|-------------|
| `source` | Dataset origin (KK or HANGRY) |
| `depot` | Depot name |
| `date` | Order date |
| `awb_count` | Number of AWBs in this scenario |
| `vehicles_used` | Total vehicles dispatched |
| `vehicle_breakdown` | Count per vehicle type |
| `total_distance_km` | Total road distance across all routes |
| `total_cost_rp` | Total cost (fixed + variable) in Rupiah |
| `avg_weight_util_pct` | Average weight utilisation across vehicles (%) |
| `avg_volume_util_pct` | Average volume utilisation across vehicles (%) |
| `runtime_sec` | Wall-clock time to solve this scenario |
| `distance_method` | OSRM or Haversine |

`results_ilp.csv` additionally contains:
- `ortools_objective` — raw OR-Tools objective value
- `status` — `solved` or `no_solution`

---

## Comparing Results

To merge both result files for comparison:

```python
import pandas as pd

h = pd.read_csv("results_heuristic.csv")
i = pd.read_csv("results_ilp.csv")

h = h.rename(columns={c: f"h_{c}" for c in h.columns if c not in ["source","depot","date"]})
i = i.rename(columns={c: f"i_{c}" for c in i.columns if c not in ["source","depot","date"]})

comparison = h.merge(i, on=["source","depot","date"])
comparison["dist_improvement_pct"] = round(
    (comparison["h_total_distance_km"] - comparison["i_total_distance_km"])
    / comparison["h_total_distance_km"] * 100, 2
)
comparison.to_csv("results_comparison.csv", index=False)
print(comparison[["depot","date","h_vehicles_used","i_vehicles_used",
                   "h_total_distance_km","i_total_distance_km","dist_improvement_pct"]].to_string())
```

---

## Mathematical Model Reference

The ILP formulation implemented in `program2_ilp.py` corresponds to the model
described in thesis Section 3.2.3. The complete constraint set is:

- **C1** Each AWB visited exactly once
- **C2** Flow conservation
- **C3** All routes depart from and return to depot
- **C4** Weight capacity per vehicle
- **C5** Volume capacity per vehicle
- **C6** Subtour elimination (handled internally by OR-Tools)
- **C7** Binary integrality (enforced by CP-SAT branch-and-bound)

---

## Notes for Thesis

- DHL dataset was excluded from experiments: average 651 AWBs/day per depot, which exceeds ILP tractability limits. This is discussed in Chapter 5 (Discussion) as a finding on scalability limitations of exact methods.
- Jakarta depot (CLIENT_01) was also excluded for the same reason (221–634 AWBs/day). The heuristic can be run on Jakarta data separately to demonstrate scalability.
- All results with `distance_method = Haversine` should be re-run with OSRM enabled before final submission to ensure road distances are used throughout.
