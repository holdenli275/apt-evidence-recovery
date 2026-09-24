# Environment and commands

## Environment

The reported runs used Python 3.11, PyTorch 2.6.0+cu124, and PyTorch Geometric
2.6.1. For the verification path only, NumPy, pandas, and SciPy are enough:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the full experiment stack (including
`torch-geometric==2.6.1`). Install a platform-appropriate PyTorch build first if
you want the detector code importable; the three verification commands in
`docs/VERIFICATION.md` do not need it.

`PYTHONPATH=src` is required whenever the released code is imported, because the
package layout predates installation and `src/` is the import root.

## Running the verification path

```bash
PYTHONPATH=src python verify_paper_results.py
PYTHONPATH=src python analysis/mechanism/operating_point.py
PYTHONPATH=src python analysis/paired_significance.py
```

All three run from the release root and read only shipped files. Run them in a
copy if you want the tree to stay byte-identical to `MANIFEST.json`: two of them
rewrite their own generated CSVs.

## Running the detector path

```bash
cp config.example.json config.json      # then point the repository URLs at your GraphDB
PYTHONPATH=src python src/train_gnn_models.py --help
PYTHONPATH=src python src/detect_anomalous_subgraphs.py --help
```

Both entry points import the detector stack at module level (`torch`,
`torch_geometric`, and the vendored `pygod`), so even `--help` needs a working
PyTorch/PyG install; without one they fail with
`ModuleNotFoundError: No module named 'torch_geometric'`.

The two entry points expect the corpus layout that `src/encode_to_PyG.py`
produces and `src/dataset_pyg_custom.py` consumes - an OGB-style per-host
directory:

```
<host>/features/<node_type>/node-features.pt     node feature tensors, one file per type
<host>/mapping/<node_type>_entidx2name.csv       feature-row index -> node UUID
<host>/raw/node-label/<node_type>/node-label.csv node labels
<host>/split/<split_type>/{train,valid,test}.csv.gz   or split_dict.pt
<host>/processed/geometric_data_processed.pt     PyG cache built from the above
```

None of those tensors or mappings are part of this release.

The reporter stage reads its GraphDB endpoints from `config.json`.
`config.example.json` contains localhost placeholders and no credentials; no API
key, token, or password is bundled anywhere in this tree.

## Reconstructing a reported run

The recorded pipeline for a host is:

1. build typed provenance graphs and per-type entity mappings from the raw
   audit logs (GraphDB-backed; not shipped);
2. build the node feature tensors for the two views - `fullz` (interaction
   profile) and `TGN_fullz` (profile concatenated with the 32-dimensional
   event-history state). `bash_src/build_tgnonly_features.py` documents how the
   Temporal-only view is derived as the last 32 columns of the Joint tensors;
3. train the one-class R-GCN detector for a given seed
   (`src/train_gnn_models.py`), then run detection and correlation
   (`src/detect_anomalous_subgraphs.py`), which writes the per-run node summaries
   and candidate subgraphs;
4. select and report candidates under a retention rule; the archived rule
   outcomes for all eleven shared rules, and the same-host calibration, are in
   `analysis/knob_sensitivity.csv` and `analysis/mechanism/`.

Steps 1 and 3 need the excluded inputs. Steps 2 and 4 are documented by shipped
scripts whose data inputs are also excluded; `verify_paper_results.py` checks the
recorded outcomes of step 4 without re-running it.

## Detector seeds

The three reported seeds are 360, 361, and 362. The temporal encoder is trained
once per host and reused across them, so the paired comparisons measure
detector-initialization variation only.
