"""Build and freeze the SOFAE graph.

Input: a CSV or Parquet table with one row per patient and the onset time of
each event in the columns

    kidney, respiratory, cardiovascular, brain, haemostasis, liver, sepsis, septic_shock

(empty when the event did not occur). Onset times are either datetimes
(``--datetime``) or hours on a common time axis within each patient. MODS
onset is derived as the second earliest organ failure.

Output: the graph file (.npz) used by ``train_sepsis_most.py`` and its edge
list (.csv).

    python scripts/build/build_sofae_graph.py --onsets onsets.parquet --out sofae_graph.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sepsis_most.constants import ORGANS
from sepsis_most.graph import build_sofae_graph

INPUT_COLUMNS = list(ORGANS) + ["sepsis", "septic_shock"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onsets", required=True, help="CSV or Parquet table of event onset times")
    ap.add_argument("--out", required=True, help="output graph file (.npz)")
    ap.add_argument("--datetime", action="store_true", help="parse onset columns as datetimes")
    ap.add_argument("--min-support", type=int, default=50, help="minimum supporting patients per edge")
    ap.add_argument("--dim", type=int, default=8, help="embedding dimension")
    args = ap.parse_args()

    path = Path(args.onsets)
    table = pd.read_parquet(path) if path.suffix in (".parquet", ".pq") else pd.read_csv(path)
    missing = [c for c in INPUT_COLUMNS if c not in table.columns]
    if missing:
        raise SystemExit(f"missing columns: {missing}")
    table = table[INPUT_COLUMNS].copy()
    if args.datetime:
        for c in INPUT_COLUMNS:
            table[c] = pd.to_datetime(table[c], errors="coerce")

    graph = build_sofae_graph(table, min_support=args.min_support, dim=args.dim)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    graph.save(out)
    edges = graph.edges()
    edges.to_csv(out.with_suffix(".edges.csv"), index=False)
    print(f"patients: {len(table):,}")
    print("node counts:", dict(zip(graph.nodes, graph.relations.counts.tolist())))
    print(f"edges with >= {args.min_support} supporting patients: {len(edges)}")
    print(f"saved {out} and {out.with_suffix('.edges.csv')}")


if __name__ == "__main__":
    main()
