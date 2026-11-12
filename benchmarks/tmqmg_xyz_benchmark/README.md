# tmQMg XYZ Benchmark

This benchmark compares molecule reconstruction methods on tmQMg CSV/XYZ pairs.
It reuses the shared method registry from the other benchmark entrypoints.

## Inputs

`--csv` points to a tmQMg metadata CSV with at least `id`, `smiles`, and `charge` columns.
`--xyz-dir` points to a directory containing `<id>.xyz` files.
The official source, pinned revision, checksums, and download commands are in the
[molecule review development guide](../../docs/development/MOLECULE_REVIEW_TOOL.zh-CN.md#获取-tmqmg-数据).

Reconstruction input is restricted to the XYZ coordinates, the dataset global
charge, and this benchmark's fixed closed-shell multiplicity (one; zero radical
electrons). The tmQMg `smiles` field is loaded only after reconstruction as a
reference graph for comparison and formula-consistency diagnostics; it never
contributes bonds, charges, radicals, or candidate-selection information.

Subset selection flags:

- `--start-row` and `--end-row` filter by 1-based CSV row index.
- `--ids` filters by tmQMg id. Repeat the flag or pass comma-separated values.
- `--limit` caps the number of selected rows after filtering.

Execution flags:

- `--methods` filters the shared method registry. Use `molgr_cpp,molgr_fallback`
  for C++/Python backend parity checks.
- `--cpp-accelerations default|all` selects the C++ backend acceleration preset.
  Both presets keep target-bucket parallelism enabled; `all` additionally enables
  optional C++ accelerations such as the vendor UFF atom-typing cache.
- `--enable-uff-atom-typing-cache` enables the vendor UFF atom-typing cache when
  using the default preset.
- `--process-workers 1` (the default) runs rows serially; `molgr_cpp` still keeps
  its internal target-bucket parallelism. For `molgr_cpp` with `N > 1`, one
  benchmark subprocess owns a native batch pool of `N` workers instead of stacking
  external processes. Other methods may still split across subprocesses when `N > 1`.
- `--case-timeout-seconds` sets the per-method per-case wall-time limit; use `0` to
  disable it.

Backend parity check:

```bash
bash scripts/benchmark_env.sh run python benchmarks/tmqmg_xyz_benchmark/run.py \
  --csv /path/to/tmqmg.csv \
  --xyz-dir /path/to/xyz \
  --limit 1000 \
  --out benchmarks/_runs/tmqmg-parity \
  --progress-every 50 \
  --case-timeout-seconds 1.0 \
  --cpp-accelerations all \
  --methods molgr_cpp,molgr_fallback
```

## Equivalence search

Reference and SMILES round-trip comparisons use the evaluator's default resonance
search budget of 5,000 (up to 10,000 visited states across both search directions,
regions, and mappings). Search exhaustion remains inconclusive. Atom-mapping
limits and the maximum resonance-path length are unchanged.

## Outputs

The benchmark writes `results.csv` and `summary.csv` to `--out`.

## Accuracy exclusions

The versioned rules in `comparison_annotations.json` exclude 1,176 structures
containing at least four boron atoms from reconstruction-accuracy comparisons.
Conventional Lewis graphs used for both the tmQMg and MolGR answers cannot
represent multicenter 3-center-2-electron (3c-2e) bonding in these boron
clusters, so neither answer is treated as assessable. Reconstruction may still
run to retain candidate and timing diagnostics, but these cases do not enter the
accuracy denominator. `YULBOY` is the explicitly recorded exception to this
tmQMg revision's annotation set.
