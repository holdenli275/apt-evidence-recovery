# Verifying the reported numbers

Three commands re-derive the paper's reported quantities from the CSVs in this
release. None of them needs a GPU, a dataset download, or network access.

## 1. The check to run first

```bash
PYTHONPATH=src python verify_paper_results.py
```

Stdlib only. It reads the 18 `anomaly_results_summary.csv` files, the archived
`analysis/subgraph_rescoring.csv` rule table, and `analysis/knob_sensitivity.csv`,
and asserts every value against the manuscript with a tolerance of 0.001. It
exits non-zero on any mismatch. Expected output (abridged):

```
Table 2 node Aggregate: 0.680
Table 2 node Joint: 0.627
Table 1 report Aggregate: 0.752
Table 1 report Joint: 0.901
Table 2 minimum Joint: 0.266
Table 3 top-2 fullz: 0.614
Table 3 top-4 TGN_fullz: 0.861
...
Table 3 cutoff 10 fullz: 0.752
Table 3 cutoff 10 TGN_fullz: 0.901
Manuscript archived Table 1-3 checks passed
```

Twenty-eight lines are printed in total: the four Table 1/2 headline values, the
Joint minimum, the twenty-two entries of the eleven-rule table in both views
(both `f1_tau_10` entries repeat the Table 1 values), and the closing
confirmation line.

The invariants it enforces are worth stating explicitly, because they are what
makes the check a check rather than a print:

- exactly three runs per host per view for the node summaries (54 node rows);
- exactly 27 default-rule rows per view in `analysis/subgraph_rescoring.csv`;
- exactly 27 rows per `(view, rule)` cell in `analysis/knob_sensitivity.csv`.

## 2. Operating point and calibration

```bash
PYTHONPATH=src python analysis/mechanism/operating_point.py
```

Needs NumPy and pandas. Reads the same `analysis/knob_sensitivity.csv` and prints
the per-host calibrated-rule table plus the arithmetic means over the 18-run
subset:

```
         val_mean  default_mean  oracle_mean
variant
FUSED      0.8676        0.8908       0.9042
STATIC     0.8422        0.7804       0.9054
```

`default_mean` is the fixed-rule mean, `val_mean` the mean after selecting the
rule on run 0, `oracle_mean` the per-run best rule. The calibrated gap
0.8676 - 0.8422 = 0.0254 is the 0.025 quoted in the manuscript; `oracle_mean`
moves slightly the other way, which is the operating-point caveat.

The command also writes `analysis/mechanism/OPERATING_POINT.md` and rewrites
`operating_point_fixed.csv` / `operating_point_calibrated.csv`. In a clean
checkout both CSVs come back **byte-identical** to the shipped files, and the
Markdown report is a new generated file that is not part of the package.

## 3. Paired and leave-one-out statistics

```bash
PYTHONPATH=src python analysis/paired_significance.py
```

Needs NumPy and SciPy. Reads `analysis/subgraph_rescoring.csv` and the node
summaries; prints Wilcoxon, sign-test, paired-t, and bootstrap results. Expected
headline figures:

```
- mean STATIC = 0.7516, mean FUSED = 0.9014, mean Δ = +0.1499
- bootstrap 95% CI (run-level) = [+0.0516, +0.2681]
- wins/ties/losses = 15/8/4
- host-level Wilcoxon (n=9) two-sided p = 0.098
```

Two caveats on reproducibility, both visible in the output:

- the script rewrites `analysis/paired_significance.csv` with last-digit
  differences in a few bootstrap and Wilcoxon statistics, because those depend
  on the SciPy/NumPy build; the differences are ≤ 0.002 and never change a
  conclusion;
- it additionally writes `analysis/paired_significance.md`, a generated summary
  that is not part of the package.

## 4. What cannot be re-run from this tree

- `analysis/knob_sensitivity.py` and all four `bash_src/*.py` scripts import the
  detector stack (`torch_geometric`, and for `bash_src` also the corpus loader)
  and read `analysis/cache_rescore/`, which is **not** shipped. They document the
  original computation and will fail on import or on a missing cache path:

  ```
  ModuleNotFoundError: No module named 'torch_geometric'
  ```

- Detector training and the graph-construction stages need the processed PyG
  graphs, node feature tensors, TGN embeddings, and the GraphDB ingestion
  environment. All are excluded; see `MANIFEST.md` "What is deliberately absent".

So the release supports the paper's *verification* path end to end (every quoted
number is recomputed from shipped CSVs) and its *audit* path (the archived rule
tables and scripts are readable), but not end-to-end retraining.
