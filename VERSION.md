# Version history

## Version 1.0 - 2026-09-24

First public release: the compact code-and-results package for
*Complementary Behavior Representations for APT Evidence Recovery*.

Contents:

- nine-host, three-view archived results for 27 paired detector runs (seeds
  360/361/362) over DARPA TC3, OpTC, and the NODLINK simulated collection;
- the eleven shared reporting rules, the same-host calibration tables, and the
  paired / leave-one-out statistics behind the manuscript's Tables 1-3;
- the exploratory type-aware retention diagnostic and its rule-selection
  records;
- the detector, correlation, reporting, and TGN-encoder code paths used for the
  reported runs, with the vendored PyGOD subset they import;
- `verify_paper_results.py`, a stdlib-only re-derivation of every quantity the
  manuscript quotes from the shipped CSVs.

The original internal package name carried the submission filename; this release
renames the verification script to `verify_paper_results.py`, removes the
manuscript filename from the two scripts that mentioned it, and deletes the
`covered_ioc` column - benchmark ground-truth IOC strings - from all 54
candidate-statistics CSVs. `MANIFEST.json` was regenerated accordingly, and
`MANIFEST.md` records the IOC provenance of the removed column.

## Previously

The internal package cut, dated 2026-09-24, held 138 files and is summarized by
the `MANIFEST.json` shipped here. No earlier release exists.
