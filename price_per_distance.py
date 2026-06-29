"""Calculate per-leg transport costs for the Engkel and CDD routes.

The script reads a pairwise distance table, looks up each route leg,
and calculates:

	cost = distance_km / efficiency_km_per_liter * fuel_price_rp_per_liter

It writes one CSV row per leg, with route totals repeated on each row.

Default routes:
	Engkel   -> 0, 1, 5, 8, 0
	CDD Box  -> 0, 4, 9, 6, 2, 10, 7, 3, 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ENGKEL_ROUTE = [0, 1, 5, 8, 0]
CDD_ROUTE = [0, 4, 9, 6, 2, 10, 7, 3, 0]

VEHICLES = {
	"Engkel (CDE)": {
		"efficiency_km_per_l": 6.0,
		"pbbm_rp_per_l": 6_800,
		"route": ENGKEL_ROUTE,
	},
	"CDD Box": {
		"efficiency_km_per_l": 4.5,
		"pbbm_rp_per_l": 6_800,
		"route": CDD_ROUTE,
	},
}


def node_meta(node_id: int) -> dict[str, str]:
	"""Return default labels and names for a node id."""
	if node_id == 0:
		return {"label": "depot", "name": "Depot Batam"}
	return {"label": f"n{node_id}", "name": f"Node_{node_id:02d}"}


def build_distance_lookup(df: pd.DataFrame) -> dict[tuple[int, int], dict[str, object]]:
	"""Build a directed lookup for distance matrices.

	Supports both:
	- the full matrix CSV from build_distance_matrix_s05.py
	- the pairwise upper-triangle CSV as a fallback
	"""
	lookup: dict[tuple[int, int], dict[str, object]] = {}

	if "from \\ to" in df.columns:
		labels = [str(col) for col in df.columns if col != "from \\ to"]
		label_to_id = {"depot": 0}
		for label in labels:
			if label.startswith("n"):
				label_to_id[label] = int(label[1:])

		for _, row in df.iterrows():
			from_label = str(row["from \\ to"])
			from_id = label_to_id.get(from_label)
			if from_id is None:
				continue
			for to_label in labels:
				to_id = label_to_id.get(to_label)
				if to_id is None:
					continue
				lookup[(from_id, to_id)] = {
					"from_label": from_label,
					"from_name": node_meta(from_id)["name"],
					"to_label": to_label,
					"to_name": node_meta(to_id)["name"],
					"distance_km": float(row[to_label]),
				}
		return lookup

	for row in df.itertuples(index=False):
		from_id = int(row.from_id)
		to_id = int(row.to_id)
		distance_km = float(row.distance_km)

		lookup[(from_id, to_id)] = {
			"from_label": getattr(row, "from_label", node_meta(from_id)["label"]),
			"from_name": getattr(row, "from_name", node_meta(from_id)["name"]),
			"to_label": getattr(row, "to_label", node_meta(to_id)["label"]),
			"to_name": getattr(row, "to_name", node_meta(to_id)["name"]),
			"distance_km": distance_km,
		}

		lookup[(to_id, from_id)] = {
			"from_label": getattr(row, "to_label", node_meta(to_id)["label"]),
			"from_name": getattr(row, "to_name", node_meta(to_id)["name"]),
			"to_label": getattr(row, "from_label", node_meta(from_id)["label"]),
			"to_name": getattr(row, "from_name", node_meta(from_id)["name"]),
			"distance_km": distance_km,
		}

	return lookup


def cost_rp(distance_km: float, efficiency_km_per_l: float, pbbm_rp_per_l: float) -> int:
	"""Calculate rounded fuel cost in rupiah."""
	return int(round((distance_km / efficiency_km_per_l) * pbbm_rp_per_l))


def route_rows(route_name: str, vehicle_name: str, route: list[int], lookup: dict[tuple[int, int], dict[str, object]], efficiency_km_per_l: float, pbbm_rp_per_l: float) -> list[dict[str, object]]:
	"""Expand a route into one row per leg."""
	rows: list[dict[str, object]] = []
	total_distance_km = 0.0
	total_cost_rp = 0

	for leg_order, (from_id, to_id) in enumerate(zip(route, route[1:]), start=1):
		if from_id == to_id:
			distance_km = 0.0
			meta = {
				"from_label": node_meta(from_id)["label"],
				"from_name": node_meta(from_id)["name"],
				"to_label": node_meta(to_id)["label"],
				"to_name": node_meta(to_id)["name"],
			}
		else:
			meta = lookup.get((from_id, to_id))
			if meta is None:
				raise KeyError(f"No directed distance found for leg {from_id} -> {to_id}")
			distance_km = float(meta["distance_km"])

		leg_cost = cost_rp(distance_km, efficiency_km_per_l, pbbm_rp_per_l)
		total_distance_km += distance_km
		total_cost_rp += leg_cost

		rows.append({
			"route_name": route_name,
			"vehicle_name": vehicle_name,
			"leg_order": leg_order,
			"from_id": from_id,
			"from_label": meta["from_label"],
			"from_name": meta["from_name"],
			"to_id": to_id,
			"to_label": meta["to_label"],
			"to_name": meta["to_name"],
			"distance_km": round(distance_km, 4),
			"efficiency_km_per_l": efficiency_km_per_l,
			"pbbm_rp_per_l": pbbm_rp_per_l,
			"leg_cost_rp": leg_cost,
			"route_total_distance_km": round(total_distance_km, 4),
			"route_total_cost_rp": total_cost_rp,
		})

	return rows


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Calculate per-leg prices for Engkel and CDD routes.")
	parser.add_argument(
		"--input",
		default="distance_matrix_s05_full.csv",
		help="Directed full matrix CSV file (preferred). Pairwise CSV is accepted as fallback.",
	)
	parser.add_argument(
		"--output",
		default="price_per_distance_results.csv",
		help="Output CSV file.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	base_dir = Path(__file__).resolve().parent
	input_path = (base_dir / args.input).resolve()
	output_path = (base_dir / args.output).resolve()

	df = pd.read_csv(input_path)
	lookup = build_distance_lookup(df)

	all_rows: list[dict[str, object]] = []
	for vehicle_name, config in VEHICLES.items():
		all_rows.extend(
			route_rows(
				route_name=vehicle_name,
				vehicle_name=vehicle_name,
				route=config["route"],
				lookup=lookup,
				efficiency_km_per_l=config["efficiency_km_per_l"],
				pbbm_rp_per_l=config["pbbm_rp_per_l"],
			)
		)

	result_df = pd.DataFrame(all_rows)
	result_df.to_csv(output_path, index=False)

	print(f"Wrote {len(result_df)} leg rows to {output_path}")
	print(
		result_df[["route_name", "route_total_distance_km", "route_total_cost_rp"]]
		.drop_duplicates()
		.to_string(index=False)
	)


if __name__ == "__main__":
	main()

