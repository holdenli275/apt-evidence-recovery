#!/usr/bin/env python3
"""Recompute the eleven shared reporting rules in the manuscript from cached candidates."""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bash_src"))
import subgraph_rescoring as scoring

HOSTS = (
    "cadets", "theia", "trace", "SysClient0051", "SysClient0201",
    "SysClient0501", "SimulatedUbuntu", "SimulatedW10", "SimulatedWS12",
)
VIEWS = ("fullz", "TGN_fullz")
TOP_K = (2, 4, 8, 15, 30)
QUANTILES = (0.5, 0.7, 0.8, 0.9)
CUTOFFS = (1, 10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="release root containing analysis/cache_rescore")
    parser.add_argument("--output", type=Path,
                        help="output CSV (default: <root>/analysis/knob_sensitivity.csv)")
    args = parser.parse_args()
    root = args.root.resolve()
    rows = []
    for host in HOSTS:
        for view in VIEWS:
            for run in range(3):
                path = root / "analysis/cache_rescore" / f"prep_{host}_{view}_run{run}.pkl"
                with path.open("rb") as handle:
                    data = pickle.load(handle)
                scores = np.asarray([
                    scoring.score_of(data, index, None)
                    for index in range(len(data["subgraphs"]))
                ])
                order = np.argsort(-scores, kind="stable")
                row = {"host": host, "tag": view, "run": run}
                for k in TOP_K:
                    row[f"f1_topk_{k}"] = round(
                        scoring.eval_selection(data, order[:k])["f1"], 6
                    )
                for q in QUANTILES:
                    threshold = float(np.quantile(scores, q)) if len(scores) else 0.0
                    kept = np.flatnonzero(scores > threshold)
                    row[f"f1_q_{q}"] = round(
                        scoring.eval_selection(data, kept)["f1"], 6
                    ) if len(kept) else 0.0
                for cutoff in CUTOFFS:
                    kept = np.flatnonzero(scores > cutoff)
                    row[f"f1_tau_{cutoff}"] = round(
                        scoring.eval_selection(data, kept)["f1"], 6
                    ) if len(kept) else 0.0
                rows.append(row)
    if len(rows) != 54:
        raise AssertionError(f"expected 54 manuscript rows, got {len(rows)}")
    output = args.output or root / "analysis/knob_sensitivity.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} paired rule-sensitivity rows to {output}")


if __name__ == "__main__":
    main()
