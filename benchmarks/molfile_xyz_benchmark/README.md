# Molfile/SDF XYZ Benchmark

[English](README.md) | [中文](README.zh-CN.md)

This benchmark compares molecule reconstruction methods on XYZ cases generated from
`.mol`, `.molfile`, or `.sdf` fixtures. It is useful for regression checks against
curated molecules with explicit coordinates, including metal-containing systems.

## Case Preparation

For each molfile/SDF fixture, [`../../scripts/molgr_cases_molfile.py`](../../scripts/molgr_cases_molfile.py):

1. Loads the source file with RDKit using `sanitize=False`, `removeHs=False`, and
   `strictParsing=False`.
2. Verifies that conformer coordinates are present.
3. Computes total formal charge and radical electron count.
4. Converts the structure to an XYZ block.
5. Uses the source RDKit molecule as the ground-truth structure for equivalence checking.

If case preparation fails, the benchmark records a skipped row for each method.

## Methods

This benchmark reuses the shared method registry from `smiles_xyz_benchmark`:

- `rdkit_determine_bonds`
- `openbabel_read_xyz`
- `cell2mol_v2`
- `molgr_fallback`
- `molgr_cpp`
- `xyzgraph_cheminf_full`
- `xyz2mol_tm` (optional, for transition-metal complexes)

## Environment

Use the dedicated benchmark environment documented in [`../README.md`](../README.md).
The environment uses Python `>=3.10,<3.12` and keeps benchmark-only dependencies in
[`../pyproject.toml`](../pyproject.toml).

Create or refresh it from the repository root:

```bash
bash scripts/benchmark_env.sh create
```

## Run

Recommended direct run:

```bash
bash scripts/benchmark_env.sh run python benchmarks/molfile_xyz_benchmark/run.py \
  --input tests/data/sdf/MoNNMo.sdf \
  --limit 1 \
  --out benchmarks/_runs/molfile-demo
```

Optional shell switch for repeated benchmark commands:

```bash
eval "$(bash scripts/benchmark_env.sh env)"
```

Then run:

```bash
uv run python benchmarks/molfile_xyz_benchmark/run.py \
  --input tests/data/sdf/MoNNMo.sdf \
  --limit 1 \
  --out benchmarks/_runs/molfile-demo
```

Restore the normal project environment:

```bash
unset UV_PROJECT UV_PROJECT_ENVIRONMENT UV_PYTHON
```

## Inputs

`--input` accepts either:

- a single `.mol`, `.molfile`, or `.sdf` file
- a directory recursively scanned for those suffixes

Recommended fixture root: [`../../tests/data/sdf/`](../../tests/data/sdf/).

Nested directories under [`../../tests/data/sdf/`](../../tests/data/sdf/) are supported,
so you can run either a specific category path or the whole fixture tree.

`--limit` caps the number of discovered files after recursive path sorting.

## Outputs

The output directory contains:

- `results.csv`: one row per `(case, method)` run.
- `summary.csv`: aggregate counts and latency statistics by method.

The CSV schema is shared with `smiles_xyz_benchmark`. Timing fields are flattened where
available, and `timing_ms_breakdown_json` keeps the full timing dictionary.
