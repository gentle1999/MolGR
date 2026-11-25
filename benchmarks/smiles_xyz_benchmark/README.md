# SMILES/XYZ Benchmark

[English](README.md) | [中文](README.zh-CN.md)

This benchmark compares molecule reconstruction methods on XYZ cases generated from
SMILES inputs. It is useful for quick regression checks because the input file is compact
and RDKit can provide the reference molecule.

## Case Preparation

For each SMILES entry, [`../../scripts/molgr_cases_smiles_csv.py`](../../scripts/molgr_cases_smiles_csv.py):

1. Parses the SMILES with RDKit.
2. Generates the canonical reference SMILES.
3. Adds hydrogens.
4. Embeds a 3D conformer with RDKit.
5. Optimizes the conformer with UFF.
6. Computes total formal charge and radical electron count.
7. Converts the molecule to an XYZ block.

If case preparation fails, the benchmark records a skipped row for each method.

## Methods

The benchmark uses the shared method registry from
[`methods/__init__.py`](methods/__init__.py):

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

The default is serial row execution; `molgr_cpp` still keeps its internal target-bucket
parallelism. Add `--jobs N` when measuring batch throughput so the native batch pool owns
the C++ worker budget.

Create or refresh it from the repository root:

```bash
bash scripts/benchmark_env.sh create
```

## Run

Recommended direct run:

```bash
bash scripts/benchmark_env.sh run python benchmarks/smiles_xyz_benchmark/run.py \
  --input tests/test_cases.csv \
  --limit 10 \
  --out benchmarks/_runs/smiles-demo
```

Optional shell switch for repeated benchmark commands:

```bash
eval "$(bash scripts/benchmark_env.sh env)"
```

Then run:

```bash
uv run python benchmarks/smiles_xyz_benchmark/run.py \
  --input tests/test_cases.csv \
  --limit 10 \
  --out benchmarks/_runs/smiles-demo
```

Restore the normal project environment:

```bash
unset UV_PROJECT UV_PROJECT_ENVIRONMENT UV_PYTHON
```

## Inputs

`--input` points to a text or CSV-style file containing SMILES. The loader accepts:

- a plain file with one SMILES per non-empty line
- a header named `smiles`
- a header named `canonicalsmiles`
- a header named `canonicalsmi`
- a two-line `general` plus SMILES header layout

`--limit` applies after header parsing and caps the number of SMILES entries used for the run.

## Outputs

The output directory contains:

- `results.csv`: one row per `(case, method)` run.
- `summary.csv`: aggregate counts and latency statistics by method.

The CSV schema is shared with `molfile_xyz_benchmark`. Timing fields are flattened where
available, and `timing_ms_breakdown_json` keeps the full timing dictionary.

## Reproducibility

For comparable runs:

- keep the input file and `--limit` fixed
- keep the method registry fixed
- use the same [`../uv.lock`](../uv.lock)
- record the Git commit and output directory name with each run
