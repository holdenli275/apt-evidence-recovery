# Citation material

`artifact_citation.tex` holds the two versions of the artifact paragraph that the
manuscript's availability statement needs, plus the BibTeX entry for the release
itself.

- **Option A** is the review-time wording: it points at the anonymized mirror and
  names no repository owner. Use it while the submission is under double-blind
  review, together with the anonymized `CITATION.cff` and `.zenodo.json` in the
  review copy of this repository.
- **Option B** is the post-acceptance wording: named repository, version DOI, and
  the concept DOI.

Replace the `OWNER`, `XXXXXXX`, `YYYYYYY`, and mirror-URL placeholders once the
release exists. `MANIFEST.md` records what the release does and does not cover,
and `docs/VERIFICATION.md` lists the commands a reader can run to reproduce the
reported numbers without retraining.
