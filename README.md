# MolGR

[English](README.md) | [中文](README.zh-CN.md)

[![CI](https://github.com/gentle1999/MolGR/actions/workflows/ci.yaml/badge.svg)](https://github.com/gentle1999/MolGR/actions/workflows/ci.yaml)
[![PyPI](https://img.shields.io/pypi/v/molgr.svg)](https://pypi.org/project/molgr/)
[![PyPI downloads](https://img.shields.io/pypi/dm/molgr.svg)](https://pypi.org/project/molgr/)
[![Wheel](https://img.shields.io/pypi/wheel/molgr.svg)](https://pypi.org/project/molgr/)
![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-4C8BF5)
![Python](https://img.shields.io/badge/python-3.8--3.14-3776AB?logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)
![Development status](https://img.shields.io/badge/status-active%20development-F59E0B)
![Typing](https://img.shields.io/badge/typing-PEP%20561%20typed-2F74C0)
![Languages](https://img.shields.io/badge/languages-Python%20%7C%20C%2B%2B-00599C)
![Ruff](https://img.shields.io/badge/lint%20%26%20format-Ruff-D7FF64?logo=ruff&logoColor=261230)
[![License](https://img.shields.io/badge/license-GPL--2.0-blue.svg)](LICENSE)

MolGR stands for Moleculer Graph Reconstructor.

MolGR is a Python package for reconstructing molecular graphs from XYZ coordinates.
It accepts an XYZ block, total charge, and spin multiplicity, and returns an
`rdkit.Chem.Mol` with bond orders, a 3D conformer, optional dative bonds, and optional
stereochemistry.

MolGR currently provides two switchable reconstruction backends:

- `cpp`: the default backend, exposed through `molgr._core`, for regular use and
  performance-sensitive workflows.
- `python`: the Python reference implementation in [`src/molgr/fallback/`](src/molgr/fallback/),
  used for semantic alignment, debugging, and regression tests.

## Quick Usage

The main entrypoint is `molgr.interface.xyz_to_rdmol`:

```python
from molgr.interface import xyz_to_rdmol

xyz = """3
water
O 0.000000 0.000000 0.000000
H 0.758602 0.000000 0.504284
H -0.758602 0.000000 0.504284
"""

mol = xyz_to_rdmol(
    xyz,
    total_charge=0,
    spin_multiplicity=1,
    backend="cpp",  # "python" is also available
)
```

Before reconstruction, MolGR calculates the total electron count as the sum of
the atomic numbers minus `total_charge` and rejects impossible spin
multiplicities. In particular, even-electron systems require odd multiplicities,
while odd-electron systems require even multiplicities. The accepted
`spin_multiplicity` is then normalized as
`total_radical_electrons = spin_multiplicity - 1`. By default, MolGR also
completes possible dative bonds and stereochemistry. Pass
`make_dative_bonds=False` or `make_stereochemistry=False` to disable those
post-processing steps.

## Configuration

Runtime configuration is dataclass-based. The package-level
[`molgr.config.CONFIG`](src/molgr/config.py) object is used by default, and callers may
either mutate that global object or pass an explicit [`MolGRConfig`](src/molgr/config.py)
to `xyz_to_rdmol(..., config=...)`.

```python
from molgr.config import CONFIG

CONFIG.cpp_backend.enable_target_bucket_parallelism = True
CONFIG.cpp_backend.target_bucket_parallel_threshold = 1
CONFIG.cpp_backend.target_bucket_parallel_max_threads = None
```

On Windows, `CONFIG.cpp_backend.max_threads` defaults to `1`. Enabling C++
backend thread parallelism on Windows is not recommended: concurrent Open Babel
reconstruction can cause native access violations that Python cannot catch. Keep
`max_threads=1` unless the complete workload has been validated locally. Linux
and macOS keep the automatic `None` default.

The C++ backend is the default accelerated implementation of the Python fallback
semantics. C++-only switches may change scheduling, caching, or thread-safe vendor
implementations, but they must not change the selected molecule for the same
`MolGRConfig`.

### Dangerous-use warnings

Low-level APIs that still expose raw `OBMol*` pointers emit one
`[UNSAFE_OPENBABEL]` warning per API name on first use. These pointers have manual
ownership, must not be shared across threads, and must be freed exactly once after
all access has finished. Prefer `xyz_to_rdmol()`, `MoleculeData`, and the native
batch API. Raising the C++ log threshold hides the message but does not change
Open Babel's thread-safety or lifetime constraints. Use process isolation for
untrusted inputs or code that must call other raw Open Babel/Pybel APIs.

MolGR rejects calls from a POSIX child created by `fork` after MolGR/Open Babel
was imported. Such a child inherits inconsistent native mutex, worker-pool, and
Open Babel global state. Use `multiprocessing.get_context("spawn")` on Linux and
macOS; Windows already uses `spawn` and is unaffected. Forking before the child
first imports MolGR, or immediately executing a fresh program after `fork`, does
not inherit initialized MolGR state.

Calling `molgr.interface.xyz_to_rdmol()` directly inside a `spawn` worker is
also supported. When the process pool owns cross-molecule parallelism, use
`max_threads=1` and disable single-molecule native parallelism in each worker
so process count and native worker count do not multiply unexpectedly.

## Advanced Usage

### Parallel execution

For low-latency or benchmark runs, call `xyz_to_rdmol()` serially. Each C++ call
may still parallelize the metal target buckets internally, so an outer Python
worker pool is not needed.

For high-throughput workloads, use the native batch API. It accepts any finite
iterable, runs reconstruction in a bounded C++ worker pool, and streams
`(input, result, status)` triples. The input is included even when
`ordered=False`, so completion-order results remain associated with the correct
XYZ request.

```python
from molgr import ReconstructionBatchRequest, iter_xyz_to_rdmol_batch

requests = (
    ReconstructionBatchRequest(xyz, total_charge=0, spin_multiplicity=1)
    for xyz in xyz_blocks
)

for request, molecule, status in iter_xyz_to_rdmol_batch(
    requests,
    backend="cpp",
    max_workers=None,  # select the native worker count automatically
):
    consume(request, molecule, status)
```

When the batch uses more than one worker, MolGR disables per-molecule target-bucket
and candidate-scoring parallelism to avoid nested oversubscription. A one-worker
batch retains the normal per-molecule parallel strategy. Do not wrap the native
batch API in `joblib`, a thread pool, or a process pool; set `max_workers` on the
batch call instead. The `python` backend remains sequential as the semantic
reference implementation.

## Installation

MolGR requires Python `>=3.8`. Runtime and build dependencies are declared in
[`pyproject.toml`](pyproject.toml). The default development workflow uses `uv`.

### C++ Build Requirements

Building from source compiles the `molgr._core` extension. The local build environment
must provide:

- CMake `>=3.15`.
- A C++17-capable compiler:
  - Linux: GCC 9+ or Clang 10+.
  - macOS: Apple Clang from a current Xcode Command Line Tools installation.
  - Windows: MSVC Build Tools 2019+ with the C++ workload.
- Python development headers for the active Python interpreter. In virtual environments
  created by `uv`, this is usually provided by the selected Python installation.
- Python build packages from [`pyproject.toml`](pyproject.toml): `scikit-build-core`,
  `pybind11`, `setuptools-scm`, and `openbabel-wheel`.
- A working OpenBabel wheel for the target platform. The CMake build locates OpenBabel
  headers and libraries from the installed `openbabel-wheel` package.

Install from source and build the C++ extension:

```bash
cd MolGR
uv sync --dev
uv pip install -e . -v --no-build-isolation
```

Editable installation builds the C++ extension through `scikit-build-core`, `pybind11`,
and the OpenBabel package available in the current Python environment. After changing C++
code or pybind11 bindings, rebuild before running affected tests:

```bash
make cpp-build
```

To mimic CI-style dependency resolution against PyPI instead of the local mirror configured in
[`pyproject.toml`](pyproject.toml), pass explicit indexes:

```bash
uv sync --dev --index https://pypi.org/simple --default-index https://pypi.org/simple
uv pip install -e . -v --no-build-isolation \
  --index https://pypi.org/simple \
  --default-index https://pypi.org/simple
```

## Project Layout

- [`src/molgr/interface.py`](src/molgr/interface.py): public API, config resolution,
  backend routing, and RDKit post-processing.
- [`src/molgr/fallback/`](src/molgr/fallback/): Python reference implementation.
- [`src/cpp/`](src/cpp/): C++ reconstruction implementation.
- [`src/bindings/`](src/bindings/): pybind11 bindings for `molgr._core`.
- [`src/molgr/utils/`](src/molgr/utils/): conversion, equivalence, and RDKit post-processing utilities.
- [`tests/`](tests/): public behavior tests, fallback tests, converter tests, and C++ parity tests.
- [`benchmarks/`](benchmarks/): benchmark entrypoints, dedicated dependencies, and benchmark docs.
- [`docs/`](docs/): architecture and release documentation.

## Algorithm Overview

MolGR treats the Python fallback as the semantic reference and the C++ backend as the accelerated
path. Both backends follow the same high-level reconstruction flow:

1. Parse the XYZ input and determine reconstruction targets from total charge,
   spin multiplicity, and runtime config.
2. Route to `backend="cpp"` or `backend="python"`.
3. For metal-free structures, search neighboring-radical seeds in increasing
   charge-separation discrepancy layers. The layers share resonance deduplication state and stop
   at the first valid layer; deformed-pi and bond-break recovery run only if every primary layer is empty.
4. For metal-containing structures, temporarily strip metals from the organic core, enumerate
   possible metal states, and group those states by the induced organic charge/radical target.
5. Reconstruct each organic target bucket once, then reuse it across corresponding metal candidates.
6. Score organic candidates or combined metal-organic candidates and select the best state.
7. Convert backend output to an RDKit molecule and optionally complete dative bonds and stereochemistry.

See [`docs/architecture/ALGORITHM_ARCHITECTURE.md`](docs/architecture/ALGORITHM_ARCHITECTURE.md)
for the full call graph, metal search, resonance recovery, and C++ optimization notes.

## Development

Create or refresh the development environment:

```bash
uv sync --dev
```

Build the extension in editable mode:

```bash
uv pip install -e . -v --no-build-isolation
```

Common checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pyright
uv run pytest
```

Build source and wheel distributions:

```bash
uv build
```

The root [`makefile`](makefile) provides common development shortcuts:

- `make format`: run the Ruff formatter; modifies files.
- `make lint`: run `ruff check . --fix`; modifies files.
- `make test`: run pytest.
- `make type-check`: run MyPy and Pyright on `src`.
- `make cpp-build`: rebuild the editable C++ extension and refresh C++ IDE config.
- `make stubs`: regenerate type stubs for the compiled extension.

`make stubs` runs `pybind11-stubgen` against `molgr._core`, keeps the public
`pipeline` stubs and the development/parity helper stubs under
[`src/molgr/_core/dev/`](src/molgr/_core/dev/), then formats and lints the generated
`.pyi` files.

Optional VSCode/clangd C++ configuration:

```bash
uv run python scripts/gen_vscode_config_with_ob.py --build-dir build
```

## Testing

Run the full test suite:

```bash
uv run pytest
```

Useful focused tests during development:

```bash
uv run pytest tests/test_main.py
uv run pytest tests/test_backend_reconstruction_regression.py
uv run pytest tests/test_fallback_resonance.py
uv run pytest tests/test_fallback_get_possible_metal_radicals.py
```

When changing C++, pybind11 bindings, or the `_core` public surface:

1. Rebuild the extension with `make cpp-build`.
2. Run affected C++/Python parity tests, for example
   [`tests/test_backend_reconstruction_regression.py`](tests/test_backend_reconstruction_regression.py)
   [`tests/test_cpp_python_metal_candidate_parity.py`](tests/test_cpp_python_metal_candidate_parity.py),
   [`tests/test_cpp_uff_atom_typing_cache.py`](tests/test_cpp_uff_atom_typing_cache.py),
   and [`tests/test_force_field_scoring_policy.py`](tests/test_force_field_scoring_policy.py).
3. If binding interfaces changed, run `make stubs` and inspect the generated `.pyi` files.

The C++ backend is an acceleration of the Python fallback semantics. Keep the
backend parity guardrails in
[`docs/architecture/ALGORITHM_ARCHITECTURE.md`](docs/architecture/ALGORITHM_ARCHITECTURE.md#cpppython-parity-guardrails)
in sync when changing SMARTS matching, UFF setup state, resonance selection,
or C++-only acceleration switches.

Some tests depend on OpenBabel. They may be skipped when the dependency is unavailable.

## Benchmark

Benchmark dependencies live in [`benchmarks/pyproject.toml`](benchmarks/pyproject.toml), separate
from the release dependency graph and the root [`uv.lock`](uv.lock). Benchmarks use a dedicated environment:
default path `.venv-benchmark`, default Python executable `python3.10`.

Create or refresh the benchmark environment:

```bash
bash scripts/benchmark_env.sh create
```

Run the SMILES/XYZ benchmark:

```bash
bash scripts/benchmark_env.sh run python benchmarks/smiles_xyz_benchmark/run.py \
  --input tests/test_cases.csv \
  --limit 10 \
  --out benchmarks/_runs/smiles-demo
```

Run the Molfile/SDF benchmark:

```bash
bash scripts/benchmark_env.sh run python benchmarks/molfile_xyz_benchmark/run.py \
  --input tests/data/sdf/MoNNMo.sdf \
  --limit 1 \
  --out benchmarks/_runs/molfile-demo
```

Compare both backends on a small tmQMg-Corrected sample on Linux. First [install the pinned release](benchmarks/dataset_xyz_benchmark/README.md#tmqmg-corrected).

```bash
bash scripts/benchmark_env.sh run python -m benchmarks.dataset_xyz_benchmark.prepare \
  --dataset tmqmg-corrected --source .local/datasets/tmqmg-corrected \
  --limit-molecules 10 --out benchmarks/_runs/tmqmg-demo/cases.sqlite

for method in molgr_cpp molgr_fallback; do
  bash scripts/benchmark_env.sh run python -m benchmarks.dataset_xyz_benchmark.run \
    --cases benchmarks/_runs/tmqmg-demo/cases.sqlite --method "$method" \
    --workers 1 --out benchmarks/_runs/tmqmg-demo/results
done
```

For repeated benchmark commands, switch the current shell environment:

```bash
eval "$(bash scripts/benchmark_env.sh env)"
```

Restore the default project environment:

```bash
unset UV_PROJECT UV_PROJECT_ENVIRONMENT UV_PYTHON
```

See [`benchmarks/README.md`](benchmarks/README.md) or
[`benchmarks/README.zh-CN.md`](benchmarks/README.zh-CN.md) for details.

## Documentation

- [`docs/README.md`](docs/README.md): English documentation index.
- [`docs/README.zh-CN.md`](docs/README.zh-CN.md): Chinese documentation index.
- [`docs/architecture/ALGORITHM_ARCHITECTURE.md`](docs/architecture/ALGORITHM_ARCHITECTURE.md):
  English algorithm architecture.
- [`docs/architecture/ALGORITHM_ARCHITECTURE.zh-CN.md`](docs/architecture/ALGORITHM_ARCHITECTURE.zh-CN.md):
  Chinese algorithm architecture.
- [`docs/release/DEVELOPMENT_RELEASE_GUIDE.md`](docs/release/DEVELOPMENT_RELEASE_GUIDE.md):
  English development and release guide.
- [`docs/release/DEVELOPMENT_RELEASE_GUIDE.zh-CN.md`](docs/release/DEVELOPMENT_RELEASE_GUIDE.zh-CN.md):
  Chinese development and release guide.
- [`benchmarks/README.md`](benchmarks/README.md): English benchmark environment, commands, and outputs.
- [`benchmarks/README.zh-CN.md`](benchmarks/README.zh-CN.md): Chinese benchmark environment, commands, and outputs.

## License

MolGR is released under the GNU General Public License v2. See [`LICENSE`](LICENSE).
