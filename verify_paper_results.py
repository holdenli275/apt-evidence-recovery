#!/usr/bin/env python3
"""Check the manuscript's archived Aggregate/Joint tables from release CSVs."""

from __future__ import annotations

import csv
import argparse
import statistics
from collections import defaultdict
from pathlib import Path


HOSTS = (
    ("darpa_tc3", "cadets"), ("darpa_tc3", "theia"), ("darpa_tc3", "trace"),
    ("darpa_optc", "SysClient0051"), ("darpa_optc", "SysClient0201"),
    ("darpa_optc", "SysClient0501"), ("nodlink", "SimulatedUbuntu"),
    ("nodlink", "SimulatedW10"), ("nodlink", "SimulatedWS12"),
)
VIEWS = ("fullz", "TGN_fullz")
RULES = (
    ("top-2", "f1_topk_2", .614, .687),
    ("top-4", "f1_topk_4", .708, .861),
    ("top-8", "f1_topk_8", .847, .876),
    ("top-15", "f1_topk_15", .863, .890),
    ("top-30", "f1_topk_30", .852, .888),
    ("q50", "f1_q_0.5", .868, .880),
    ("q70", "f1_q_0.7", .801, .860),
    ("q80", "f1_q_0.8", .715, .861),
    ("q90", "f1_q_0.9", .649, .738),
    ("cutoff 1", "f1_tau_1", .852, .888),
    ("cutoff 10", "f1_tau_10", .752, .901),
)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def close(label: str, actual: float, expected: float, tolerance: float = .001) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: {actual:.6f}, expected {expected:.3f}")
    print(f"{label}: {actual:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    root = parser.parse_args().root
    node = defaultdict(list)
    for family, host in HOSTS:
        for view in VIEWS:
            path = (root / "dataset" / family / host / "experiments" / "results"
                    / f"Full_Script_Test_{view}" / f"T3_OCRGCN_{host}_{view}"
                    / "anomaly_results_summary.csv")
            selected = [float(row["f_measure"]) for row in rows(path)
                        if row["run"] in ("0", "1", "2")]
            if len(selected) != 3:
                raise AssertionError(f"expected three node runs: {path}")
            node[view].extend(selected)
    close("Table 2 node Aggregate", statistics.mean(node["fullz"]), .680)
    close("Table 2 node Joint", statistics.mean(node["TGN_fullz"]), .627)

    report = defaultdict(list)
    for row in rows(root / "analysis/subgraph_rescoring.csv"):
        if row["variant"] in VIEWS and row["rule"] == "default_tau10" \
                and row["run"] in ("0", "1", "2"):
            report[row["variant"]].append(float(row["real_F1"]))
    if any(len(report[view]) != 27 for view in VIEWS):
        raise AssertionError("expected 27 default-report rows per view")
    close("Table 1 report Aggregate", statistics.mean(report["fullz"]), .752)
    close("Table 1 report Joint", statistics.mean(report["TGN_fullz"]), .901)
    close("Table 2 minimum Joint", min(report["TGN_fullz"]), .266)

    knob = [row for row in rows(root / "analysis/knob_sensitivity.csv")
            if row["tag"] in VIEWS and row["run"] in ("0", "1", "2")]
    for name, column, aggregate, joint in RULES:
        for view, expected in (("fullz", aggregate), ("TGN_fullz", joint)):
            values = [float(row[column]) for row in knob if row["tag"] == view]
            if len(values) != 27:
                raise AssertionError(f"expected 27 {name} rows for {view}")
            close(f"Table 3 {name} {view}", statistics.mean(values), expected)
    print("Manuscript archived Table 1-3 checks passed")


if __name__ == "__main__":
    main()
