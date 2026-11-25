"""Batch the selected XYZ regression inputs and return an independent graph per assertion."""

from __future__ import annotations

import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from typing import TYPE_CHECKING, Literal

import pytest

from benchmarks.tmqmg_concurrency_stress import available_cpu_count


if TYPE_CHECKING:
    from molgr import ReconstructionBatchRequest, ReconstructionDiagnostics


# Keep both backends and output modes for these inputs in the short CI suite.
_PR_FIXTURES = {
    # Metal-state ranking, charge and radical localization.
    "ABAGAM",
    "ABELOK",
    "ABIRIO",
    # Side-on coordination, chelates and ring resonance.
    "ACEKAU",
    "VIVNIZ",
    "KAYWOA",
    # Bounded equivalence proofs.
    "BASDAC",
    "BEBQIK",
    "YURTOW",
    # Published and reviewed sulfur assignments.
    "ADURUP",
    "ALILOX",
    "KIWCEY",
    "KIWCIC",
}


def fixture_marks(case_id: str) -> tuple[pytest.MarkDecorator, ...]:
    """Reserve the remaining bulk XYZ inputs for full-suite runs."""
    return () if case_id in _PR_FIXTURES else (pytest.mark.exhaustive,)


def _batch(
    frames: list[ReconstructionBatchRequest],
    backend: Literal["cpp", "python"],
    kekulize: bool,
    workers: int,
) -> dict[ReconstructionBatchRequest, tuple[bytes | None, ReconstructionDiagnostics | None]]:
    """Run one public batch and preserve full-precision geometry across processes."""
    from rdkit import Chem

    from molgr import MolGRConfig, iter_xyz_to_rdmol_batch

    config = MolGRConfig()
    config.cpp_backend.max_threads = 1
    config.cpp_backend.target_bucket_parallel_max_threads = 1
    results = {}
    for request, molecule, diagnostics in iter_xyz_to_rdmol_batch(
        frames, backend=backend, kekulize=kekulize, max_workers=workers, config=config
    ):
        # RDKit's default process transport loses atom properties and rounds XYZ.
        graph = (
            molecule.ToBinary(
                Chem.PropertyPickleOptions.AllProps | Chem.PropertyPickleOptions.CoordsAsDouble
            )
            if molecule is not None and diagnostics is None
            else None
        )
        results[request] = graph, diagnostics
    return results


@pytest.fixture(scope="session")
def rebuild_fixture(request: pytest.FixtureRequest):
    """Rebuild only collected frames, keeping failures and graph copies case-local."""
    from rdkit import Chem

    groups = defaultdict(dict)
    for item in request.session.items:
        if "rebuild_fixture" in getattr(item, "fixturenames", ()):
            params = item.callspec.params
            backends = (params["backend"],) if "backend" in params else ("cpp", "python")
            for backend in backends:
                groups[backend, params.get("kekulize", True)][params["frame"]] = None

    # Limit prefetch so the first failing case is reported promptly. Multiple
    # assertions and fixture families can reuse exactly the same request.
    batches = {}
    for mode, frames in groups.items():
        ordered = list(frames)
        for i in range(0, len(ordered), 32):
            batch = ordered[i : i + 32]
            for frame in batch:
                batches[mode, frame] = batch

    results = defaultdict(dict)
    workers = min(4, available_cpu_count())
    pool = None
    with ExitStack() as resources:

        def rebuild(
            frame: ReconstructionBatchRequest,
            backend: Literal["cpp", "python"],
            kekulize: bool = True,
        ) -> Chem.Mol:
            nonlocal pool
            mode = backend, kekulize
            cache = results[mode]
            if frame not in cache:
                batch = batches[mode, frame]
                if backend == "python" and workers > 1 and len(batch) >= 8:
                    if pool is None:
                        pool = resources.enter_context(
                            ProcessPoolExecutor(
                                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
                            )
                        )
                    # Small groups let free workers take another ligand when
                    # a neighboring input has an expensive electronic search.
                    pending = [
                        pool.submit(_batch, batch[i : i + 2], backend, kekulize, 1)
                        for i in range(0, len(batch), 2)
                    ]
                    for future in pending:
                        cache.update(future.result())
                else:
                    cache.update(
                        _batch(batch, backend, kekulize, workers if backend == "cpp" else 1)
                    )
            graph, diagnostics = cache[frame]
            assert diagnostics is None and graph is not None, diagnostics
            return Chem.Mol(graph)

        yield rebuild
