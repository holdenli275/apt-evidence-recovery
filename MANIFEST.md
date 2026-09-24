# Release inventory

This repository is the compact code-and-results release for
*Complementary Behavior Representations for APT Evidence Recovery*. It ships the
archived evidence-recovery results, the scripts that regenerate the manuscript's
reported tables from them, the detector and encoder code used for the reported
runs, and the per-file checksum manifest of the original package.

Where this file and `MANIFEST.json` disagree, `MANIFEST.json` is authoritative
for file identity: it records the group, size, and SHA-256 of every file in the
package as it was cut. `MANIFEST.md` describes what the release contains and,
more importantly, what it deliberately leaves out.

## Contents

| Path | Groups in `MANIFEST.json` | Files | What it is |
|---|---|---:|---|
| `dataset/` | `candidate_statistics` (54), `run_result` (36), `temporal_only_summary` (9) | 99 | Per-run node summaries, candidate-statistics CSVs, and original two-hop-tolerant report summaries for the nine hosts and three views; plus the archived Temporal-only summaries. |
| `src/` | `source` (9), `source_dependency` (12) | 21 | Graph encoding, OCRGCN training/detection, one-hop correlation, the compact TGN encoder, and the vendored PyGOD subset the OCRGCN path imports. |
| `bash_src/` | `source` (4) | 4 | Candidate replay, eleven-rule sweep, same-host calibration, and type-mask gate as run. |
| `analysis/` | `derived_table` (6), `source` (3) | 9 | The rule-sensitivity, paired-comparison, and calibrated-rule tables behind Tables 1-3, plus the scripts that regenerate them. |
| `verify_paper_results.py` | `documentation` (1) | 1 | Stdlib-only re-derivation of every quantity the manuscript quotes from the shipped CSVs. |
| `LICENSE` (`source`); `README.md`, `config.example.json`, `requirements.txt` (`documentation`) | | 4 | Release metadata. `LICENSE` is the upstream Apache-2.0 text retained from OCR-APT. |

The counts above are the 138 files recorded in `MANIFEST.json`, regrouped by
top-level path; `MANIFEST.json` is not self-listed. The release adds `README.md`
(rewritten),
`MANIFEST.md`, `VERSION.md`, `CITATION.cff`, `.zenodo.json`, `.gitignore`,
`docs/`, and `paper/`.

## Naming

The manuscript's **Aggregate** view is the `fullz` tag in every archived file and
directory (`Full_Script_Test_fullz`, `tag=fullz`); **Joint** is `TGN_fullz`; and
**Temporal-only** is `TGNonly`. The joint representation is the concatenation
`x_v = [a_v || h_v]`, with the 32-dimensional event-history block last, so the
Temporal-only view is the last 32 columns of the Joint feature tensors. The
internal names survive because the archived CSVs and the recorded commands use
them verbatim.

Three kinds of edit separate this tree from the original package.

1. Anonymization. The verification script was renamed in this release - its
   original name carried the submission's file name - and now ships as
   `verify_paper_results.py`; the docstring in `analysis/knob_sensitivity.py` and
   the success message in `verify_paper_results.py` no longer name the
   manuscript file.
2. Ground-truth removal. The `covered_ioc` column was deleted from all 54
   candidate-statistics CSVs; every other column in those files is unchanged.
3. Release scaffolding: `README.md` rewritten, plus `MANIFEST.md`, `VERSION.md`,
   `CITATION.cff`, `.zenodo.json`, `.gitignore`, `docs/`, and `paper/`.

`MANIFEST.json` was regenerated after those edits, so its hashes match this tree
and differ from the original package for the files touched by (1) and (2).

## What is deliberately absent

The paper's experiments run on raw audit logs through a GraphDB-backed ingestion
pipeline. None of that is redistributable, and the compact release excludes it.
Also excluded, because it is regenerable or because it is large:

- raw audit logs, GraphDB repositories, and the ingestion environment;
- processed PyG graphs and node feature tensors;
- TGN embeddings (`tgn_embeddings/<host>_tgn.pt`) and the Temporal-only input
  tensors;
- candidate replay caches (`analysis/cache_rescore/`), candidate graph tensors
  (`*_constructed_subgraphs_nx.pt`), and per-node raised-alarm CSVs;
- model weights and detector checkpoints;
- other detector implementations and the unrelated `AgentLinux` host
  configuration; log-erasure, CAC, VCTF, Unified, StreamSpot, and LLM experiment
  lines;
- benchmark ground-truth IOC material, including the `covered_ioc` column that
  the detector writes into the candidate-statistics CSVs (see "Ground-truth IOC
  material" below).

Consequences a reader should know before trying to re-run anything:

- `bash_src/*.py` and `analysis/knob_sensitivity.py` import the detector stack
  (`torch_geometric`) and read `analysis/cache_rescore/`, which is not in this
  release. They document the original computation; they cannot be executed from
  this tree alone.
- `verify_paper_results.py`, `analysis/mechanism/operating_point.py`, and
  `analysis/paired_significance.py` need only the shipped CSVs (NumPy, pandas,
  SciPy) and do run from this tree.
- The Temporal-only feature-construction script is included; its inputs are not.

## Ground-truth IOC material

The upstream OCR-APT codebase pairs detection with an investigation stage that
matches a benchmark ground-truth IOC list (`groundTruth_IOCs.json`) against the
constructed subgraphs and writes human-readable attack descriptions. That stage
is still present in the released code - `investigate_subgraphs`,
`label_query_graph_ioc`, and `get_attack_description` in
`src/detect_anomalous_subgraphs.py`, and `get_investigation_queries` in
`src/sparql_queries.py` - because it ships inside the entry point that produced
the reported runs. It is **not** part of the paper's pipeline, its outputs
(`investigation/..._attack_description_subgraph_*.csv`) are not in this release,
and `get_investigation_queries` has no call site.

What the original package also carried, and this release removes, is the
`covered_ioc` column that the detector writes into
`*_correlated_subgraphs_statistics.csv`: 54 files, 148 of roughly 1000 cells
non-empty, holding benchmark ground-truth IOC strings - attack tool names, file
names, and network addresses taken from the benchmark's own ground-truth lists.
Those values are benchmark labels rather than model output, and the column is
reproduced nowhere else in this release. No shipped script reads it, and
`verify_paper_results.py` never touches it, so deleting it changes no reported
number. The paper's own numbers are computed from the node summaries,
`analysis/subgraph_rescoring.csv`, and `analysis/knob_sensitivity.csv`.

Neither the IOC list nor the investigation outputs are redistributed here.

## Claim boundaries

The release supports a within-catalog, retrospective, transductive comparison of
two node representations evaluated at the final reporting interface. It does not
support claims of cross-host or cross-attack generalization, of live detection,
or of strict reported-node accuracy; the primary metric is two-hop tolerant. The
type-aware retention diagnostic is exploratory and label-assisted. See the
"Claim boundaries" section of `README.md`.
