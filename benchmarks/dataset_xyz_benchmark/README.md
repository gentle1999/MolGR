# Benchmarks on released molecular geometries

[中文](README.zh-CN.md)

This harness imports source coordinates, reconstructs each selected conformer,
and aggregates matches with equal weight per parent molecule. It reuses the
[existing adapters](../smiles_xyz_benchmark/methods) and equivalence settings.
No reference bonds are passed into reconstruction. Raw data, prepared SQLite
stores, worker logs and per-conformer results stay in Git-ignored directories.

## Dataset status

| Dataset | Molecular class | Inputs in the cross-dataset comparison |
| --- | --- | --- |
| [GEOM-QM9](https://doi.org/10.7910/DVN/JNGTDF) | Small organic molecules | One released conformer per parent |
| [GEOM-Drugs](https://doi.org/10.7910/DVN/JNGTDF) | Drug-like molecules | One conformer per parent in a fixed sample of 100,000 parents |
| [BorylXAT-DB](https://doi.org/10.5281/zenodo.21330198) | Boryl radicals and their halogen-transfer reaction partners | Unique optimized reactant and product components with mapped reaction SMILES as references |
| [tmQMg-Corrected](https://github.com/gentle1999/tmQMg-Corrected) | Singlet transition-metal complexes | Structures with a native Lewis reference in the pinned release |
| [BDE](https://doi.org/10.6084/m9.figshare.12158646.v1) | Small organic radical fragments | One optimized SDF geometry per carbon-containing doublet fragment |

In the cross-dataset comparison, each reconstruction has a 10 s deadline and the subsequent reference comparison has a separate 5 s deadline. Each dataset has its own denominator, including failed and timed-out reconstructions.

## Reference policy

GEOM's [official tutorial](https://github.com/learningmatter-mit/geom/blob/master/tutorials/02_loading_rdkit_mols.ipynb)
notes that some geometries dissociate and their graphs differ from the parent
SMILES. Each conformer's released `rd_mol` supplies its reference graph; the
parent SMILES only identifies the aggregation group. The released graphs were
assigned with xyz2mol, so reference agreement is not an independent chemical
accuracy measurement for that method family. Each selected structure retains its
released coordinates, energy, degeneracy, source ID and statistical weight.

Both GEOM commands below use seed `20261125`. QM9 selects one conformer per
released parent in archive order and records the reference status for every
selection. Drugs ranks distinct pickle paths from `summary_drugs.json` by
seeded SHA-256, then takes the first 100,000 distinct parents with an eligible
reference for the selected conformer. A missing member, empty ensemble or
ineligible reference advances the selection to the next ranked parent. Both
datasets select the conformer by hashing the dataset name, seed and archive
member path, using rejection sampling to select a uniform zero-based index.
For scoring, an RDKit atom query that specifies a single exact element can be
stored as a concrete atom. Both the source and scoring graph hashes are retained
for eligible structures.

The [QM9 input indices](manifests/geom-qm9.csv.gz) and [Drugs input indices](manifests/geom-drugs.csv.gz) give the exact selections in case-store order. Both files are gzipped CSVs. To load a geometry, open the pickle at `archive_member` in the pinned release and read `entry["conformers"][conformer_index]["rd_mol"]`.

| Column | Meaning |
| --- | --- |
| `archive_parent_index` | Zero-based position among regular `.pickle` members in the same subset, in physical tar order. |
| `conformer_index` | Zero-based position in that parent’s released `conformers` list. |
| `case_id`, `source_geom_id` | One-based SQLite case ID and the publisher’s `geom_id` for the selected conformer, respectively. |
| `reference_status` | Use `valid` for the scored cohort. QM9 includes all 133,232 selections, of which 133,123 were scored. All 100,000 Drugs selections were scored. |
| `sample_rank` (Drugs only) | One-based rank before eligibility filtering and parent deduplication. Skipped candidates leave gaps. |

Input charge comes from the source. GEOM singlet/doublet targets are inferred
from a source graph with zero/one formal radical electron, respectively. A
multiradical graph alone does not establish spin coupling: its multiplicity
remains unresolved.

BorylXAT uses the Zenodo v2 `BorylXAT-DB.db`. Each reaction links four independently optimized species: the boryl complex radical (`complex_r`) and chlorinated substrate (`Cl`) on the reactant side, and the chlorinated boron complex (`complex_p`) and carbon radical (`c_radical`) on the product side. Each geometry comes from the final step of its own `GS_OPT` calculation. Reconstruct each component separately and count each structure key once.

The reference is the corresponding fragment of the mapped reaction SMILES on the TS record. Track the mapped chlorine transferred from carbon to boron to identify the fragments and verify conservation of every explicit hydrogen and other atom across the reaction. Preserve the optimized atom order; graph comparison allows atom permutations. Report components with missing or conflicting reaction labels as unresolved.

BDE uses the optimized structures published by St. John et al. A fragment must contain carbon and have one unpaired electron in the SDF bond table. Its atomic spin densities (`AtomSpins`) must sum to 1 within 0.0001. Closed-shell parents and carbon-free fragments fall outside this cohort. Reconstruction receives the explicit-H coordinates, SDF formal charge and doublet multiplicity through the benchmark adapter. Individual method APIs may use only a subset of these inputs. The bond table from the same record supplies the scoring reference.

Across all five datasets, the RDKit and xyz2mol_tm (local) adapters select radical assignment for neutral open-shell inputs with `allowChargedFragments=False` and `allow_charged_fragments=False`, respectively. Closed-shell and charged inputs use the upstream formal-charge mode. The xyz2mol_tm adapter scopes the CLI valence table to each call: its radical mode changes C, N and O valences, and the next molecule must start with its own settings. When converting xyzgraph output to RDKit, disable implicit hydrogens on every atom because the XYZ already includes them. The cell2mol v2 molecular XYZ entry point accepts total charge and has no multiplicity parameter.

Matching allows resonance and defaults to ignoring stereochemistry and metal
coordination bonds. GEOM, BorylXAT and BDE default to `max_resonance=5000` and a
5-second comparison limit. The standalone tmQMg-Corrected release specifies 100,000 and
60 seconds. The cross-dataset timing comparison keeps its 100,000-state budget and uses the shared 5 s comparison deadline.
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
mkdir -p .local/datasets/geom
curl --fail --location --continue-at - \
  'https://dataverse.harvard.edu/api/access/datafile/4327252?format=original' \
  --output .local/datasets/geom/rdkit_folder.tar.gz
mkdir -p .local/datasets/borylxat/21330198
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
  --out .local/datasets/benchmark-cases/borylxat-optimized.sqlite

mkdir -p .local/datasets/bde
curl --fail --location --continue-at - \
  https://ndownloader.figshare.com/files/22357962 \
  --output .local/datasets/bde/20200415_radical_database.sdf.gz
python -m benchmarks.dataset_xyz_benchmark.prepare \
  --dataset bde --source .local/datasets/bde/20200415_radical_database.sdf.gz \
  --out .local/datasets/benchmark-cases/bde.sqlite
```

The pinned GEOM archive is 50,137,579,520 bytes, MD5
`e8f2168b7050652db22c976be25c450e`. BorylXAT imports verify the published database size and MD5. Rebuild older stores to use optimized component geometries with reaction-derived labels.
For a GEOM pilot, `--limit-molecules N` imports **all conformers** of the first N
parents in archive order and marks the store as a pilot. Omit this option to
reproduce the fixed benchmark cohorts above. An interrupted import remains marked incomplete
and cannot be benchmarked. Importing again requires a new output file.

Run the separate Zenodo regression fixtures offline:

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
source-configured comparison timeout. For the five-dataset comparison, use
`--timeout 10 --comparison-timeout 5` for every dataset, including tmQMg-Corrected.
Omit `--comparison-timeout` to use the source default (60 s for tmQMg-Corrected).
An explicit deadline can shorten a pinned limit. The run provenance records both
the source policy and the effective policy. Use separate output directories for
runs with different limits. A worker crash or
timeout is retained in the result table.
Returned graphs are compared even when the adapter reports an API error.
Input reading, process startup, warmup and reference comparison are outside the
timed adapter call. Use `--method xyz2mol_tm` for every dataset. The worker reads the element symbols
from each XYZ before timing. An element on the upstream TMC list (including La
and Lu) selects the Hückel entry. All other inputs use `xyz2mol_local.xyz2mol`.
The result detail records the entry as `upstream_entry`. The explicit
`--method xyz2mol_tm_local` option still selects the general entry directly.

```bash
python -m benchmarks.dataset_xyz_benchmark.run \
  --cases .local/datasets/benchmark-cases/geom-drugs.sqlite \
  --method molgr_cpp --workers 64 --timeout 10 --comparison-timeout 5 \
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
