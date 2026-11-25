# Benchmarks on released molecular geometries

[中文](README.zh-CN.md)

This harness imports source coordinates, reconstructs each released conformer,
and aggregates matches with equal weight per parent molecule. It reuses the
[existing adapters](../smiles_xyz_benchmark/methods) and equivalence settings.
No reference bonds are passed into reconstruction. Raw data, prepared SQLite
stores, worker logs and per-conformer results stay in Git-ignored directories.

## Dataset status

| Dataset | Source | Import and evaluation policy | Current validation |
| --- | --- | --- | --- |
| GEOM-QM9 | [GEOM release](https://doi.org/10.7910/DVN/JNGTDF) | All released conformers, grouped by source molecule | Importer unit-tested; full download pending |
| GEOM-Drugs | [GEOM release](https://doi.org/10.7910/DVN/JNGTDF) | All released conformers, grouped by source molecule | First 3 complete molecules / 744 conformers used for pipeline validation; not a representative sample |
| BorylXAT-DB | [Zenodo v2](https://doi.org/10.5281/zenodo.21330198) | Stable structures; transition states and other categories retained separately from stable-graph accuracy | Existing 30 corrected fixtures available; full database download currently returns HTTP 403 |
| BDE | Release still to be identified | SDF import requires source attribution and an explicit multiplicity property | No dataset-specific measurement yet |
| tmQMg-Corrected | [2026-11-23 release](https://github.com/gentle1999/tmQMg-Corrected) | Pinned native Lewis graphs; release eligibility and comparison policy | Full import counts verified; no full performance result |

The historical tmQMg comparison used
72,578 draft targets, not the current release. Pilot measurements are not
full-dataset results; the published standard requires a new evaluation.

## Reference policy

GEOM's [official tutorial](https://github.com/learningmatter-mit/geom/blob/master/tutorials/02_loading_rdkit_mols.ipynb)
notes that some geometries dissociate and their graphs differ from the parent
SMILES. Each conformer's released `rd_mol` supplies its reference graph; the
parent SMILES only identifies the aggregation group. The released graphs were
assigned with xyz2mol, so reference agreement is not an independent chemical
accuracy measurement for that method family. Conformer energy, degeneracy,
source ID and statistical weight are retained as evidence, but do not select
or weight the main comparison. Every stored conformer is used once; `totalconfs`
and degeneracy do not create additional coordinate records.

Input charge comes from the source. GEOM singlet/doublet targets are inferred
from a source graph with zero/one formal radical electron, respectively. A
multiradical graph alone does not establish spin coupling: its multiplicity
remains unresolved. BorylXAT open-shell categories additionally require a
spin-density sum consistent with one unpaired electron. Spin densities are not
atom-localized integer radical counts.

BorylXAT reuses the [source-derived correction and geometry checks](../../tests/data/borylxat/README.md)
for B and reactant/product complexes. A failed correction leaves the reference
unresolved; it does not replace it with a MolGR output. Other stable categories
retain source graphs and undergo composition, charge and radical-count checks.
Transition states are not assigned an integer-bond reference accuracy. The full
importer requires the published file size and MD5; it has not yet been validated
against all source rows.

Matching allows resonance and defaults to ignoring stereochemistry and metal
coordination bonds. GEOM, BorylXAT and BDE default to `max_resonance=5000` and a
5-second comparison limit; tmQMg-Corrected uses the publisher's 100,000 and
60 seconds. These are source configurations executed by the same runner.
Generic importers check composition, source charge, electron parity and formal
radical count; these checks do not prove chemical correctness. tmQMg eligibility
and input multiplicity come directly from the release: formal radical counts
must not be used to reinterpret coupled-spin states.
All sources retain native RDKit reference graphs and SHA-256 hashes; SMILES is
display-only, never the reference serialization. Schema 1 stores must be prepared
again. Unknown spin states and exclusions remain visible in coverage counts;
only `valid` references enter the reconstruction/scoring queue.

## Prepare local data

Use the [benchmark environment](../README.md). The public GEOM release is about
50 GB; the prepared store requires additional space. Its `.tar.gz` file is an
uncompressed tar in this release. The importer reads members sequentially without
extracting the archive. It only permits the RDKit molecule constructor when
unpickling data.

```bash
mkdir -p .local/datasets/geom .local/datasets/borylxat/21330198
curl --fail --location --continue-at - \
  'https://dataverse.harvard.edu/api/access/datafile/4327252?format=original' \
  --output .local/datasets/geom/rdkit_folder.tar.gz
curl --fail --location --continue-at - \
  'https://zenodo.org/records/21330198/files/BorylXAT-DB.db?download=1' \
  --output .local/datasets/borylxat/21330198/BorylXAT-DB.db

python -m benchmarks.dataset_xyz_benchmark.prepare \
  --dataset geom-qm9 --source .local/datasets/geom/rdkit_folder.tar.gz \
  --out .local/datasets/benchmark-cases/geom-qm9.sqlite
python -m benchmarks.dataset_xyz_benchmark.prepare \
  --dataset geom-drugs --source .local/datasets/geom/rdkit_folder.tar.gz \
  --out .local/datasets/benchmark-cases/geom-drugs.sqlite
python -m benchmarks.dataset_xyz_benchmark.prepare \
  --dataset borylxat --source .local/datasets/borylxat/21330198/BorylXAT-DB.db \
  --out .local/datasets/benchmark-cases/borylxat.sqlite
```

The pinned GEOM archive is 50,137,579,520 bytes, MD5
`e8f2168b7050652db22c976be25c450e`. BorylXAT v2 is 291,348,480 bytes, MD5
`d486fbcc36f856bbebf303b5cda4950f`. Full imports reject other files.
`--limit-molecules N` permits a small pilot, marks it explicitly, and imports
**all conformers** of each selected molecule. Its archive-order selection is
for pipeline validation only. An interrupted import remains marked incomplete
and cannot be benchmarked. Importing again requires a new output file.

Run the existing BorylXAT fixtures without a network download:

```bash
python -m benchmarks.dataset_xyz_benchmark.prepare \
  --dataset borylxat-fixtures --source tests/data/borylxat \
  --out .local/datasets/benchmark-cases/borylxat-fixtures.sqlite
```

## tmQMg-Corrected

The sole standard is [tmQMg-Corrected 2026-11-23](https://github.com/gentle1999/tmQMg-Corrected),
tag `v2026-11-23`, revision `aac372de0e859a6daf451084abe84a9ae58ad8e6`.
The dataset is maintained in its own repository, not bundled with MolGR. The
download is large; run from the MolGR repository root:

```bash
git clone --depth 1 --branch v2026-11-23 \
  https://github.com/gentle1999/tmQMg-Corrected.git \
  .local/datasets/tmqmg-corrected
git -C .local/datasets/tmqmg-corrected rev-parse HEAD
# aac372de0e859a6daf451084abe84a9ae58ad8e6

python -m benchmarks.dataset_xyz_benchmark.prepare \
  --dataset tmqmg-corrected --source .local/datasets/tmqmg-corrected \
  --out .local/datasets/benchmark-cases/tmqmg-corrected.sqlite
python -m benchmarks.dataset_xyz_benchmark.native_batch \
  --cases .local/datasets/benchmark-cases/tmqmg-corrected.sqlite \
  --workers 8 --checkers 4 \
  --out .local/reports/dataset-performance/tmqmg-corrected-batch
python -m benchmarks.dataset_xyz_benchmark.report \
  --cases .local/datasets/benchmark-cases/tmqmg-corrected.sqlite \
  --results .local/reports/dataset-performance/tmqmg-corrected-batch/molgr_cpp.sqlite \
  --out .local/reports/dataset-performance/tmqmg-corrected-batch/molgr_cpp-summary.json
```

Development full-set checks use eight native reconstruction workers and four independent RDKit postprocessing and scoring processes by default. Each case is rebuilt once. The checker retains double-precision coordinates and MolGR electron annotations and applies the release comparison policy to the resulting graph. The log reports end-to-end throughput. Use the general `run` module for individual latency measurements and isolated process watchdogs. A repeat with matching provenance resumes at the last committed batch.

- Manifest SHA-256: `ea04563079dfe4344601f6c896e36c24a847a69e4f320ec4f906a69203c3faf3`.
- Layout v4: `data/*.sdf`, `evidence/inputs/*.xyz`, `evidence/inputs/*.jsonl`,
  indexed by repository-relative paths. An SDF-only archive is insufficient.
- Of 74,547 inputs, 72,040 have native Lewis targets. The 2,507 exclusions comprise
  1,189 boron clusters, 354 boron hydrogen bridges, 770 missing-H inputs and
  194 electronic-state exclusions. Published singlet targets with coupled ligand radicals
  count toward accuracy, including when MolGR’s radical budget cannot reproduce them.

Use `--source` for another complete checkout. Import verifies the pinned manifest,
consumed XYZ/SDF part hashes, original XYZ identity and native graph hashes.
There is no implicit download or fallback to old drafts or original CSV labels.
Full imports additionally validate part and release counts. Excluded inputs stay
in coverage statistics, not the evaluation queue. Reports retain release totals,
exclusion categories, the actual imported scope and comparison policy.
For a small check, add `--limit-molecules 10` during preparation and use fresh
store/result paths. This selects source-order parent molecules (including excluded
inputs), not a random sample or an index into eligible targets.

Reconstruction receives only XYZ, source charge and input multiplicity, never
reference bonds, atom-local electronic assignments or historical local answers.
Comparison uses the published native graph directly, without local alternative
answers or a SMILES round trip. The [offline fixtures](../../tests/data/tmqmg/README.md)
are a release-derived regression subset, not another standard. The draft's
**99.41% / 72,578 targets** is not a result for this release.

New Lewis annotations are **CC BY 4.0**: credit tmQMg-Corrected, identify the pinned
release and any changes, and cite tmQMg. Consult the upstream
[data licenses and attribution](https://github.com/gentle1999/tmQMg-Corrected/blob/aac372de0e859a6daf451084abe84a9ae58ad8e6/sources/DATA-LICENSE.md).
MolGR's software license does not replace dataset licenses.

## Measure and resume

Run each method separately; multiprocessing distributes molecules/conformers,
with one logical CPU and one native thread per worker. The Linux harness uses
persistent spawned workers, a default 30-second reconstruction timeout and the
source-configured comparison timeout. `--comparison-timeout` may override only
unlocked policies, never a publisher-pinned comparison limit. A worker crash or
timeout is retained in the result table.
Returned graphs are compared even when the adapter reports an API error.
Input reading, process startup, warmup and reference comparison are outside the
timed adapter call. xyz2mol_tm's transition-metal entry point is not an organic
baseline for GEOM, BorylXAT or BDE; it is available for tmQMg.

```bash
python -m benchmarks.dataset_xyz_benchmark.run \
  --cases .local/datasets/benchmark-cases/geom-drugs.sqlite \
  --method molgr_cpp --workers 14 \
  --out .local/reports/dataset-performance/geom-drugs
python -m benchmarks.dataset_xyz_benchmark.report \
  --cases .local/datasets/benchmark-cases/geom-drugs.sqlite \
  --results .local/reports/dataset-performance/geom-drugs/molgr_cpp.sqlite \
  --out .local/reports/dataset-performance/geom-drugs/molgr_cpp-summary.json
```

Repeat with `rdkit_determine_bonds`, `openbabel_read_xyz`, `cell2mol_v2`,
`xyzgraph_cheminf_full` and `molgr_fallback`. Reusing an output directory resumes
completed records only when the input hash, code, native binary, package versions,
CPU allocation and timeout settings agree. Do not mix runs from different code
versions into one timing comparison.


cell2mol organic outputs can contain an exact-element RDKit query atom. The
adapter copies its ordinary atom fields into a concrete atom, preserving bonds,
charges, isotope, stereo, hydrogen/radical counts, atom order and coordinates.
Ambiguous atom/bond queries remain errors. This conversion enables graph
comparison without changing the upstream chemical assignment.

Each method writes `<method>.sqlite`, `<method>-provenance.json` and worker logs.
Report JSON includes source/run provenance, comparison policy, coverage and rates;
there is no dataset-specific CSV output path.

## Aggregation

For molecule *i* with *nᵢ* eligible conformers, let *mᵢ* be its matched conformers.
The main rate is the mean of *mᵢ / nᵢ* over molecules, giving every molecule equal
weight. Also report the fraction of molecules for which **all eligible conformers
match**, and the unweighted per-conformer rate. Timeouts, missing output graphs,
and inconclusive comparisons do not count as matches. Reference exclusions are
method-independent and their coverage must accompany all rates.

The reporter refuses an unfinished run by default. `--allow-partial` is available
for progress inspection; it retains the full eligible denominator and labels
the result incomplete. It never promotes a pilot into a full-dataset result.
No cross-dataset headline figure is published until the full inputs and runs
are available.
