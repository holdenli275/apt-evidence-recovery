# Data provenance and licensing

## What each part of the release is

| Path | Origin | Notes |
|---|---|---|
| `src/`, `bash_src/` | The authors' experiment code, built on the OCR-APT baseline | Also contains a vendored subset of PyGOD (see below). |
| `src/pygod/` | Vendored from PyGOD | Limited to the OCRGCN implementation and the utilities it imports. Upstream headers are retained. |
| `dataset/` | Outputs of the authors' detector and reporting runs | Per-run node summaries, candidate statistics, and reported-node summaries. No raw audit logs. |
| `analysis/` | Derived tables computed from the above | Rule sensitivity, paired comparisons, calibration. |
| `verify_paper_results.py` | The authors' verification script | Re-derives every quoted number from the shipped CSVs. |

## Upstream sources

All three benchmark families are public. This release redistributes **no raw
audit logs**; it carries per-run summaries and derived statistics that the
authors computed from them.

- **OCR-APT** (baseline code): <https://github.com/CoDS-GCS/OCR-APT>. The
  one-hop correlation path, the OCRGCN detector, and the reporting interface
  derive from this codebase, and the Apache-2.0 `LICENSE` in this repository is
  the upstream text retained unchanged. The upstream terms therefore govern the
  released code.
- **DARPA TC3** (cadets, theia, trace): 
  <https://github.com/darpa-i2o/Transparent-Computing>. The engagement data
  release keeps its own terms.
- **OpTC** (SysClient0051, SysClient0201, SysClient0501):
  <https://github.com/FiveDirections/OpTC-data>. Released by Five Directions for
  research use; its terms are unchanged by this release.
- **NODLINK / OCR-APT simulated collection** (SimulatedUbuntu, SimulatedW10,
  SimulatedWS12): the dataset record at
  <https://doi.org/10.5281/zenodo.17254415>.

PyGOD, PyTorch, PyTorch Geometric, OGB, and SciPy are dependencies, not
redistributed content; each keeps its own license and is installed from its own
distribution.

## Licensing scope of this repository

- **Code** (`src/`, `bash_src/`, `analysis/*.py`, `verify_paper_results.py`):
  Apache-2.0, as in `LICENSE`. New code in this release is offered under the same
  terms as the upstream baseline it extends.
- **Released result artifacts** (`dataset/`, `analysis/*.csv`, `MANIFEST.json`):
  released for research use together with the code. They are the authors'
  derived outputs and are not the raw benchmark logs.
- **Upstream content**: unchanged and not relicensed. If you redistribute the
  DARPA TC3, OpTC, or NODLINK raw data, go to those sources for the terms that
  apply.

Nothing in this repository grants rights to the raw audit logs of any benchmark,
and no raw log content is present.

## Personal data

The hosts are emulated benchmark machines drawn from public attack-emulation
collections. The released CSVs contain process names, entity-type identifiers,
node counts, alarm counts, and F1 statistics. Paths recorded in the candidate
statistics are the emulated hosts' own filesystem paths (for example
`C:\Users\Administrator\...` and `/Users/Public/...`), which are properties of
the published benchmark images rather than of any real person.

The `covered_ioc` column that the detector writes into the candidate-statistics
CSVs under `--get_node_attrs` carried benchmark ground-truth IOC strings (tool
names, file names, addresses). It is **not** part of this release: the column was
deleted from all 54 files, and neither the ground-truth IOC list nor the
investigation outputs are redistributed. See `MANIFEST.md` "Ground-truth IOC
material".

No contributor identity, credential, endpoint, API key, or private host name is
part of this release. `config.example.json` contains localhost placeholders
only; `config.json` is git-ignored and is never shipped.
