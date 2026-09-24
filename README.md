# Complementary Behavior Representations for APT Evidence Recovery

Code, archived experiment results, and verification scripts for the paper
*Complementary Behavior Representations for APT Evidence Recovery*. Author names
are withheld for double-blind review and will be added on acceptance.

The paper asks whether a node representation that looks better at the node level
is also the one that produces better *reported evidence*. A joint representation
concatenates a relational interaction profile with a compact recurrent
event-history state, `x_v = [a_v || h_v]`, and the three views are compared
through the whole recovery pipeline: one-class R-GCN detection, one-hop
correlation, and reporting. Across nine hosts from three benchmarks and 27 paired
detector runs, the joint view raises mean two-hop-tolerant reported-set F1 from
0.752 to 0.901 while mean node-level F1 *falls* from 0.680 to 0.627 - the
ranking depends on which interface is evaluated.

## Main archived results

All values are recomputed from the CSVs in `dataset/` and `analysis/` by
`verify_paper_results.py`; nothing below is quoted from the manuscript by hand.

| Quantity | Aggregate | Joint | Temporal |
|---|---:|---:|---:|
| Node-level F1 (mean, 27 runs) | 0.680 | 0.627 | - |
| Reported-set F1, default rule (mean, 27 runs) | 0.752 | **0.901** | 0.549 |
| Reported-set F1, minimum run | 0.000 | **0.266** | - |
| Reported-set F1, median run | 0.988 | 0.997 | - |
| Calibrated rule (mean, runs 1-2, 18 runs) | 0.842 | 0.868 | - |

- The default-rule advantage is 0.110 over the 18-run calibration subset and
  0.025 after same-host rule calibration, so part of the gain is an
  operating-point effect rather than a fixed effect size.
- Across the eleven shared reporting rules, mean gains range from 0.012 to
  0.153. Tail behaviour is not uniformly better: `top-2` leaves a zero minimum
  for both views, and `top-15`/`q50` have higher minima for Aggregate.
- Paired over runs, Joint records 15 wins, 8 exact ties, and 4 losses. The mean
  gain mostly reflects fewer near-zero runs, not improvements on already strong
  runs.
- The gains are positive in all 27 paired runs; dropping one host leaves
  0.085-0.169 and dropping one benchmark leaves 0.113-0.178. Benchmark-wise
  means are 0.223 (TC3), 0.134 (OpTC), 0.093 (NODLINK), and flipping the three
  benchmark blocks gives p = 0.25 over eight sign assignments.
- The exploratory type-aware retention diagnostic reaches 0.957 mean F1 for
  Joint and 0.870 for Aggregate, but it adopts a typed rule in only 9 of 54
  folds, and one tie-break was introduced after inspecting outcomes.

## Claim boundaries

The boundaries below are the ones the released artifacts support; do not read
them more broadly.

- Evidence recovery here is **retrospective and transductive**: every run reads
  the full observed host trace, so the results describe recovery from a
  collected trace rather than live detection.
- The primary metric is the OCR-APT **two-hop-tolerant reported-set F1**, which
  credits nearby malicious nodes and exempts nearby benign context. It is not
  strict reported-node accuracy, and node counts omit attack-path structure.
- Replication over seeds 360/361/362 measures **detector-initialization
  variation only**; the temporal encoder is trained once per host and reused, so
  the paired runs are not independent draws over the full pipeline.
- The joint advantage is **within-catalog and within-benchmark**. Nothing in
  this release tests transfer to unseen attacks, unseen hosts, or independent
  collectors, and the sign-flip test is not significant.
- Added dimensionality, capacity, and normalization remain alternative
  explanations for the gain; dimension-matched and shuffled-history controls
  would separate them.
- The type-aware retention diagnostic is **exploratory and label-assisted**: it
  selects rules on one detector initialization using host labels from the other
  two, and its reported gains include a post-hoc tie-break. The manuscript does
  not claim a new decoder or cross-host generalization from it.

## Data layout

- `dataset/<benchmark>/<host>/experiments/results/Full_Script_Test_{fullz,TGN_fullz}/T3_OCRGCN_*/`
  holds the per-run node summaries, candidate-statistics CSVs, and the original
  two-hop-tolerant report summaries for the stated OCRGCN runs. The `fullz` view
  is Aggregate; `TGN_fullz` is Joint.
- `dataset/.../results/Full_Script_Test_TGNonly/` holds the archived
  Temporal-only report summaries. The feature-construction script for that arm
  is included, its input tensors are not.
- `analysis/` holds the archived rule-sensitivity, paired-comparison, and
  calibrated-rule CSVs behind the manuscript's Tables 1-3, plus the scripts that
  regenerate them.
- `bash_src/` holds the candidate replay, eleven-rule evaluation, same-host
  calibration, and type-mask gate scripts as they were run.
- `src/` holds graph encoding, OCRGCN detection, the one-hop correlation path,
  and the compact TGN encoder. Its in-tree `src/pygod` copy is limited to the
  OCRGCN implementation and the utilities it imports.
- `MANIFEST.json` gives the source group, size, and SHA-256 for every file in the
  original package; `MANIFEST.md` is the release inventory, including what was
  deliberately left out.

## Verifying the reported numbers

The verification path needs no GPU, no training, and no data beyond this
release. From the release root:

```bash
PYTHONPATH=src python verify_paper_results.py      # Tables 1-3, exits non-zero on mismatch
PYTHONPATH=src python analysis/mechanism/operating_point.py   # operating-point and calibration summary
PYTHONPATH=src python analysis/knob_sensitivity.py --help     # eleven-rule sweep over the archived table
PYTHONPATH=src python analysis/paired_significance.py         # paired and leave-one-out statistics
```

`verify_paper_results.py` re-derives every value it checks from
`dataset/.../anomaly_results_summary.csv`, `analysis/subgraph_rescoring.csv`,
and `analysis/knob_sensitivity.csv`, and raises if a value is off by more than
0.001. `docs/VERIFICATION.md` lists the expected output of each command.

The sweep, replay, and feature-construction scripts in `bash_src/` document the
original computations, but their large inputs (candidate replay caches,
candidate graph tensors, per-node alarm CSVs, processed PyG graphs, node feature
tensors, TGN embeddings) are **not** part of this compact release, so those
scripts cannot be re-run end to end from this tree alone. There is no
`analysis/cache_rescore/` in this package.

## Code and environment

The experiments used Python 3.11, PyTorch 2.6.0+cu124, and PyTorch Geometric
2.6.1. Install a platform-appropriate PyTorch build, then `requirements.txt`.
Set `PYTHONPATH=src` to import the included detector code. The GraphDB reporter
reads repository endpoints from `config.json`; `config.example.json` contains
only local placeholders. No API credentials are bundled.

## Scope and provenance

The baseline OCR-APT code derives from <https://github.com/CoDS-GCS/OCR-APT> and
retains the upstream Apache-2.0 `LICENSE`, which covers this release. Original
benchmark sources: DARPA TC3
(<https://github.com/darpa-i2o/Transparent-Computing>), OpTC
(<https://github.com/FiveDirections/OpTC-data>), and the OCR-APT/NODLINK dataset
record (<https://doi.org/10.5281/zenodo.17254415>). Those sources keep their own
terms; this release redistributes no raw logs and does not relicense them.
`docs/DATA_PROVENANCE_AND_LICENSES.md` records the per-directory situation.

## Citation

See `CITATION.cff` (GitHub renders it through *Cite this repository*) and
`paper/artifact_citation.tex` for the data-availability paragraph and the
artifact entry.

```bibtex
@inproceedings{anon2026complementary,
  title     = {Complementary Behavior Representations for {APT} Evidence Recovery},
  author    = {Anonymous Authors},
  booktitle = {Manuscript under review},
  year      = {2026}
}
```
