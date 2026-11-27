# Benchmarks

[English](README.md) | [中文](README.zh-CN.md)

This directory contains benchmark entrypoints, benchmark-specific dependencies, and
benchmark documentation. Benchmark dependencies are intentionally isolated from the
root project dependency graph so they do not affect package releases.

## Available Benchmarks

- `smiles_xyz_benchmark`: builds XYZ cases from SMILES inputs and compares reconstruction methods.
- `molfile_xyz_benchmark`: loads `.mol`, `.molfile`, and `.sdf` inputs, converts them to XYZ cases,
  and runs the same method registry.
- `dataset_xyz_benchmark`: imports released geometries for GEOM, BorylXAT, BDE and tmQMg-Corrected.
  See the unified [dataset preparation, execution and reporting guide](dataset_xyz_benchmark/README.md).

All benchmark entrypoints use the same shared method registry:

- `rdkit_determine_bonds`
- `openbabel_read_xyz`
- `cell2mol_v2`
- `molgr_fallback`
- `molgr_cpp`
- `xyzgraph_cheminf_full`
- `xyz2mol_tm`

`cell2mol_v2`, `xyzgraph_cheminf_full`, and `xyz2mol_tm` are comparison baselines. `molgr_fallback`
uses the Python reference backend, and `molgr_cpp` uses the default C++ backend.

## Environment

- Project runtime: Python `>=3.8`
- Benchmark runtime: Python `>=3.10,<3.12`
- Recommended package manager: `uv`
- Benchmark dependency file: [`pyproject.toml`](pyproject.toml)
- Benchmark lock file: [`uv.lock`](uv.lock)
- Default benchmark virtualenv: `.venv-benchmark`
- Default benchmark Python executable: `python3.10`

The benchmark runtime is narrower than the package runtime because optional comparison
stacks such as `xyzgraph`, `cell2mol`, and `cosymlib` have older dependency constraints.
The benchmark dependency set also keeps `numpy<2` for compatibility with those stacks.

## Quickstart

Create or refresh the dedicated benchmark environment from the repository root:

```bash
bash scripts/benchmark_env.sh create
```

Run a small SMILES/XYZ benchmark:

```bash
bash scripts/benchmark_env.sh run python benchmarks/smiles_xyz_benchmark/run.py \
  --input tests/test_cases.csv \
  --limit 10 \
  --out benchmarks/_runs/smiles-demo
```

Run a small Molfile/SDF benchmark:

```bash
bash scripts/benchmark_env.sh run python benchmarks/molfile_xyz_benchmark/run.py \
  --input tests/data/sdf/MoNNMo.sdf \
  --limit 1 \
  --out benchmarks/_runs/molfile-demo
```

Released datasets use `python -m benchmarks.dataset_xyz_benchmark.prepare`,
`.run` and `.report`; [commands and source policies](dataset_xyz_benchmark/README.md)
are maintained in that directory. For backend parity, run `--method molgr_cpp`
and `--method molgr_fallback` separately on the same prepared store.

For repeated commands, switch the current shell to the benchmark project:

```bash
eval "$(bash scripts/benchmark_env.sh env)"
```

Then use normal `uv run` commands:

```bash
uv run python benchmarks/smiles_xyz_benchmark/run.py \
  --input tests/test_cases.csv \
  --limit 10 \
  --out benchmarks/_runs/smiles-demo
```

Switch back to the default project environment:

```bash
unset UV_PROJECT UV_PROJECT_ENVIRONMENT UV_PYTHON
```

## Inputs

SMILES/XYZ benchmark:

- `--input` points to a text or CSV-style file containing SMILES.
- Header rows named `smiles`, `canonicalsmiles`, or `canonicalsmi` are accepted.
- Each SMILES is embedded with RDKit, optimized with UFF, converted to XYZ, and compared
  against the original RDKit molecule.

Molfile/SDF benchmark:

- `--input` accepts one `.mol`, `.molfile`, or `.sdf` file, or a directory scanned recursively.
- Recommended fixture root: [`../tests/data/sdf/`](../tests/data/sdf/).
- Nested fixture categories under [`../tests/data/sdf/`](../tests/data/sdf/) are supported.
- Each input structure is loaded with RDKit, converted to an XYZ case, and compared against
  the source molecule.

For SMILES/molfile runners, `--limit` caps cases and `--out` selects the output
directory. Dataset runs use preparation-time `--limit-molecules` and run-time
`--method` / `--workers`.

## Outputs

SMILES/molfile runs write two main files under `--out`; dataset runs use the
unified SQLite/JSON workflow described above:

- `results.csv`: one row per `(case, method)` attempt with status, prediction,
  equivalence result, and timing fields.
- `summary.csv`: aggregated metrics by method, including counts and latency statistics.

The important `results.csv` columns are:

- `case_idx`
- `method_id`
- `input_smiles`
- `ground_truth_smiles`
- `status`
- `error`
- `predicted_smiles`
- `equivalent`
- `equivalence_method`
- `evaluator_decision`
- `reference_status`
- `reference_reason`
- `timing_ms_total`
- `method_ms`
- `equivalence_ms`
- `timing_ms_breakdown_json`

Timing columns are flattened when present. `timing_ms_breakdown_json` preserves the full
method-specific timing dictionary for profiling.

## Interpreting Results

`reference_status` determines the accuracy denominator: `valid` references enter;
`excluded` records have confirmed reference-error evidence; `unresolved` references
remain unassessed. Eligibility is independent of method outcomes.

Use `evaluator_decision` for graph matching: `equivalent`, `not_equivalent`, and
`inconclusive` mean matched, mismatched, and unresolved comparison. API `status`
is independent. Retained graphs are compared after errors, and confirmed
equivalence counts as a match. Attempts without a comparison decision remain in
the eligible-reference denominator without adding a match.

In `summary.csv`, `success_count`, `fail_count`, `inconclusive_count`, and
`runtime_failure_count` partition eligible references; the last field counts
attempts with no comparison decision. `api_error_count` separately records errors,
and `matched_with_errors` counts matching graphs from error-status attempts.
Use `results.csv` to inspect each error, decision, and timing.

## Optional `cell2mol_v2` Note

`cell2mol_v2` is optional as a benchmark baseline. The benchmark dependency file points
to the upstream `cell2mol` `v2` Git revision. If you enable or redistribute this baseline,
ensure your usage complies with its license terms, including GPL obligations where applicable.

## Optional xyz2mol_tm baseline

After creating the benchmark environment, install the pinned upstream checkout:

```bash
git clone https://github.com/jensengroup/xyz2mol_tm.git .local/third-party/xyz2mol_tm
git -C .local/third-party/xyz2mol_tm checkout db2734b084045ef895528c7a3d5040d38da23983
uv pip install --python .venv-benchmark/bin/python --no-deps -e .local/third-party/xyz2mol_tm
```

Editable installation is needed because this revision's wheel package list omits
the Hückel subpackage. The existing benchmark environment supplies RDKit (including
`rdEHTTools`), NumPy, and NetworkX. Reinstall this optional checkout after environment
sync if it was removed. The clone and raw results remain under Git-ignored `.local/`.

Select `--methods xyz2mol_tm` in the tmQMg runners. The adapter calls
`get_tmc_mol(xyz_file, overall_charge, with_stereo=False)` and evaluates the returned
RDKit graph. This entrypoint does not accept multiplicity. Its SMILES round trip
discards coordinates, so the adapter retains upstream dative bonds without adding
geometry-based MolGR postprocessing. File I/O and serialization are timed. Failures
remain errors; an available graph is retained if only serialization fails.

The shared registry loads RDKit's Hückel and MolStandardize modules before native
MolGR/Open Babel adapters to avoid an import-order crash observed with the benchmark
environment's Linux RDKit wheel. Import the registry before MolGR in an existing
Python process; if the unsafe order has already occurred, the adapter reports
that a fresh process is needed. Upstream reconstruction code is unmodified.
The performance runner records the installed version, source hashes, and checkout revision.

The upstream tmQMg SMILES were generated with the xyz2mol_tm Hückel method. The current benchmark uses only the 72,040 released Lewis graphs from tmQMg-Corrected 2026-11-23, with its published evidence tiers and exclusions. Original labels are provenance, not ground truth. [tmQMg](https://github.com/uiocompcat/tmQMg#data) · [Corrected targets](dataset_xyz_benchmark/README.md#tmqmg-corrected)

## Benchmark-Specific Docs

- [`smiles_xyz_benchmark/README.md`](smiles_xyz_benchmark/README.md)
- [`smiles_xyz_benchmark/README.zh-CN.md`](smiles_xyz_benchmark/README.zh-CN.md)
- [`molfile_xyz_benchmark/README.md`](molfile_xyz_benchmark/README.md)
- [`molfile_xyz_benchmark/README.zh-CN.md`](molfile_xyz_benchmark/README.zh-CN.md)
- [`dataset_xyz_benchmark/README.md#tmqmg-corrected`](dataset_xyz_benchmark/README.md#tmqmg-corrected)
- [`dataset_xyz_benchmark/README.zh-CN.md#tmqmg-corrected`](dataset_xyz_benchmark/README.zh-CN.md#tmqmg-corrected)

## Additional source-geometry datasets

[GEOM-QM9, GEOM-Drugs, BorylXAT-DB, tmQMg-Corrected and BDE](dataset_xyz_benchmark/README.md): fixed GEOM sampling, molecule-level aggregation, resumable multiprocessing, and current data availability. Pilot validation is reported separately from full-dataset results.
