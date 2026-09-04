# MolGR Documentation

[English](README.md) | [中文](README.zh-CN.md)

This directory contains project documentation beyond the root README.

## Start Here

- [`../README.md`](../README.md): project overview, installation, development commands, tests, and benchmark quickstart.
- [`../README.zh-CN.md`](../README.zh-CN.md): Chinese project README.
- [`../benchmarks/README.md`](../benchmarks/README.md): benchmark environment, shared method list, input formats, and output schema.
- [`../benchmarks/README.zh-CN.md`](../benchmarks/README.zh-CN.md): Chinese benchmark documentation.

## Architecture

- [`architecture/ALGORITHM_ARCHITECTURE.md`](architecture/ALGORITHM_ARCHITECTURE.md): English algorithm architecture reference.
- [`architecture/ALGORITHM_ARCHITECTURE.zh-CN.md`](architecture/ALGORITHM_ARCHITECTURE.zh-CN.md): Chinese algorithm architecture reference.
- [`architecture/ALGORITHM_ARCHITECTURE.svg`](architecture/ALGORITHM_ARCHITECTURE.svg): compact publication-oriented architecture figure.
- [`architecture/ALGORITHM_ARCHITECTURE.drawio`](architecture/ALGORITHM_ARCHITECTURE.drawio): editable source for the architecture figure.

These files cover the public `xyz_to_rdmol(...)` API, Python fallback reference backend,
C++ `_core` backend, metal-aware reconstruction, resonance recovery, scoring, C++ optimizations,
and maintenance boundaries.

## Release

- [`release/DEVELOPMENT_RELEASE_GUIDE.md`](release/DEVELOPMENT_RELEASE_GUIDE.md): English development and release guide.
- [`release/DEVELOPMENT_RELEASE_GUIDE.zh-CN.md`](release/DEVELOPMENT_RELEASE_GUIDE.zh-CN.md): Chinese development and release guide.

These files describe the Gitea internal package channels, GitHub/PyPI release path,
wheel build matrix, tag policy, and required CI variables.

## Development Tools

- [`development/MOLECULE_REVIEW_TOOL.md`](development/MOLECULE_REVIEW_TOOL.md): English molecule graph review/trace tool guide.
- [`development/MOLECULE_REVIEW_TOOL.zh-CN.md`](development/MOLECULE_REVIEW_TOOL.zh-CN.md): Chinese runtime, queue, review-state, and fixture guide.
