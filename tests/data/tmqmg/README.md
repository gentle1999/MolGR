# Offline tmQMg-Corrected fixtures

The sole tmQMg reference is [tmQMg-Corrected](https://github.com/gentle1999/tmQMg-Corrected),
release **2026-11-23**, commit `aac372de0e859a6daf451084abe84a9ae58ad8e6`.
The full release has 74,547 inputs and 72,040 targets; it is not vendored here.

## Frozen subset

`published/cases.json` contains **941 eligible inputs and 941 published
native reference graphs**; excluded release inputs are not retained. It records the
release manifest SHA-256, original XYZ bytes, charge/multiplicity, original input
ordinal, eligibility/exclusion reason, native RDKit payload and per-case hashes.
`published/release-manifest.json` preserves the exact pinned upstream manifest.

The subset covers the grouped CSV/XYZ fixtures here, the eight former reviewed
fixtures in `../reviewed/tmqmg`, tmQMg entries in the mixed reconstruction
fixtures, the 706-case regression selection in `published/regression_case_ids.json`, standalone XYZ
regressions, the subprocess concurrency regression, the LIBZOM SMILES regression, direct candidate-search controls
(`AROJEW`, `CEHHAB`), and historical candidate-search input ordinals. Duplicate XYZ paths
are retained where existing local tests or stress manifests use them; their bytes
must match the same released input.

Use `scripts.tmqmg_fixture_data.fixture()` for inputs and `reference_graph()` for
targets. `regression_case_ids()` reads the stable 706-case selection; its manifest
contains no duplicate input or graph payload. Tests need no SQLite database or
local review service. Corrections and their evidence are maintained in the upstream
tmQMg-Corrected project; this repository consumes its published targets.
**SMILES are display-only.** Targets are decoded from the release's
`NATIVE_GRAPH_ZLIB_BASE64`, not regenerated from SMILES or reapproved local graphs.
Reviewed SDFs are exact release records and retain that native payload.

Whole-target comparisons follow the release policy: no chirality, ignore metal
coordination bonds, require proven resonance equivalence, `max_resonance=100000`.
The benchmark additionally enforces a 60-second subprocess comparison deadline;
direct unit-test equivalence calls do not impose that subprocess deadline.

## Eligible-only fixtures and local regressions

The 34 previously retained release-excluded inputs have been removed from the
JSON subset, XYZ copies, CSV/geometry/review manifests and associated
test parameters or dedicated regressions. They are not moved into boundary tests.
The generator filters release eligibility before writing any fixture store, and
the offline loader rejects a subset containing excluded inputs. The complete
upstream manifest is retained unchanged as provenance, not as fixture selection.

`fixture_sources.json` selects local regressions. Its `selection_classification` /
`selection_reason` fields record historical selection context, not current
reference validity. Local bond, geometry, search-stage and backend-parity
assertions are not alternative accepted whole-molecule targets. All retained
cases have `benchmark=true` and no exclusion reason.

The 706-case regression selection also contains 36 eligible coupled-spin graphs whose **published
native properties** explicitly say `STANDARD_RECONSTRUCTION_SCOPE =
spin_coupling_out_of_scope`. A data integrity test checks their input multiplicity
and ligand-local radical assignments. The benchmark scores all 36 published targets.
Pytest collects reconstruction and backend-parity checks for the other 670 cases.

## Regression test layout

`tests/test_tmqmg_published_regression.py` replaces the misleading historical
`test_reviewed_nitrosyl_graphs.py` name: the 706 selected targets are not all
nitrosyls. Published-target/input-conservation checks run independently for C++
and Python. A separate test compares the backends **with coordination bonds**.
Reconstruction is cached per case; each assertion uses its own molecule copy.
Sulfur target/local-control checks and strict canonical-output parity are also
separate tests. A target failure must not hide another backend or a parity check.

Known algorithm failures remain ordinary failures: no new xfail, alternative
answers, weakened parity checks, or extra dataset exclusions. The existing
coupled-spin repair scope is unchanged; the benchmark still scores those targets.
`DIRECT_CASE_IDS` in the generator selects direct test inputs that have no XYZ
manifest; integrity tests require all of them to have released native targets.
Candidate-search tests use stable case IDs (for example `AFUGEQ`, formerly input
ordinal 627), with a separate integrity check preserving the historical ordinal
selection. Ordinals are never interpreted as target indices.

## Regeneration and verification

From the repository root, with RDKit 2026.03.1 (the release writer):

```bash
.venv-benchmark/bin/python -m scripts.freeze_tmqmg_fixtures \
  --dataset .local/datasets/tmqmg-corrected
```

The checkout must contain the pinned `manifest.json`, `data/`, `evidence/inputs/`
and `sources/` license notices. The generator checks the manifest and selected
XYZ/SDF/metadata part hashes before writing, refreshes eligible inputs, extracts exact
native graphs, refreshes the ID-only regression selection and removes excluded inputs and
retired approval SDFs.
It does not consult a historical CSV, manual-review database or current MolGR
predictions. The original CSV/XYZ generator interface is no longer supported.

Ordinary tests use only this offline subset, with no network or external dataset.
Regenerate fixtures only from the pinned release, not from historical local
review exports.

## Attribution

The new annotations are © tmQMg-Corrected contributors, CC BY 4.0. This is an
unmodified subset of the released native annotations, repackaged as JSON and SDF. Coordinates and identifiers originate in tmQMg by Kneiding and colleagues.
See `published/sources/DATA-LICENSE.md` and the accompanying license texts for
upstream attribution and terms. MolGR-specific local regression assertions are
not additions to the dataset's accepted reference set.
