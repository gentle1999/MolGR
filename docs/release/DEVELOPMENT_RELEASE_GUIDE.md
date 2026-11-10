# Development and Release Guide

[English](DEVELOPMENT_RELEASE_GUIDE.md) | [中文](DEVELOPMENT_RELEASE_GUIDE.zh-CN.md)

This project uses a three-channel release model:

1. Pushes to `develop` publish internal test packages in Gitea.
2. Prerelease tags on `main` (`rc` / `.dev`) publish internal prerelease packages in Gitea.
3. Stable GitHub Releases from stable `main` tags publish official packages to PyPI.

## Release Flow Diagram

```mermaid
flowchart TD
    A[Local development] --> B[Push to Gitea]
    B --> C{Trigger type}

    C -->|push develop| D[Gitea CI: build/test + internal test publish]
    C -->|rc/.dev tag reachable from main| E[Gitea CI: build/test + internal prerelease publish]
    C -->|push/PR main| F[Gitea CI: build/test only]

    D --> G[Mirror to GitHub]
    E --> G
    F --> G

    G --> H{GitHub event}
    H -->|manual build_release_wheels=true| L[Build and test release wheels only]
    H -->|published + prerelease=false + target=main + stable tag vX.Y.Z| I[GitHub CI: cibuildwheel + manylinux + sdist]
    H -->|prerelease or non-stable tag| J[No PyPI publish]

    I --> K[Publish to PyPI]
```

## CI Roles

### Gitea workflow

File: [`.gitea/workflows/ci.yaml`](../../.gitea/workflows/ci.yaml)

- Linux matrix compile/test for Python `3.8` to `3.14`.
- Dependency index uses intranet mirror
  [`https://mirrors.zju.edu.cn/pypi/web/simple`](https://mirrors.zju.edu.cn/pypi/web/simple).
- Internal wheel publishing uses fine-grained `cibuildwheel` jobs so Gitea can
  schedule wheels across multiple runners:
  - Linux x86_64 wheels run on `ubuntu-latest` runners.
  - Linux aarch64 wheels run on the native `ubuntu-24.04-arm64` runner.
  - `cp38`-`cp312` use `manylinux2014` / glibc `>=2.17`.
  - `cp313`-`cp314` use `manylinux_2_28`.
  - macOS arm64 wheels are split by CPython ABI and run on `macos-arm64`
    runners.
- macOS arm64 runners must have official Python.org installs for the targeted
  Python versions; `cibuildwheel` will not build macOS wheels with `brew` or
  `uv` Python installations.
- Registry base URL is read from repository variable `SERVER_URL`.
- Publish to the internal registry when either condition is met:
  - `push` to `develop` (test packages), or
  - `push` tag `v*` with `rc` or `.dev`, validated as reachable from `main`
    (prerelease packages).

### GitHub workflow

File: [`.github/workflows/ci.yaml`](../../.github/workflows/ci.yaml)

- Cross-platform verification job (`build-and-test`) for `3.8` to `3.14`.
- GitHub-hosted test jobs use the `setup-python` tool cache and the `setup-uv`
  package cache, split by operating system and Python version.
- Lint runs a pinned Ruff through `uvx` without installing MolGR's RDKit/Open Babel
  runtime stack. Test jobs install runtime requirements directly from `pyproject.toml`
  instead of resolving through the locally mirrored `uv.lock`.
- Benchmark dependencies live in [`benchmarks/pyproject.toml`](../../benchmarks/pyproject.toml) and are intentionally excluded from root package metadata and release dependency resolution.
- Formal release wheel job uses `cibuildwheel` with platform-specific wheel groups:
  - Linux `cp38`-`cp312`: `manylinux2014` / glibc `>=2.17` for `x86_64` and `aarch64`.
  - Linux `cp313`-`cp314`: `manylinux_2_28` / glibc `>=2.28` for `x86_64` and `aarch64`.
  - Windows: AMD64; `delvewheel` repairs native dependencies while leaving
    `openbabel-3.dll` to the declared Open Babel package. CPython 3.8 uses
    `delvewheel 1.10.0`; newer Python versions use `delvewheel 1.13.0` because
    `delvewheel 1.11+` requires Python 3.9 or newer. CMake links the matching
    `openbabel/bin/openbabel-3.lib` import library supplied by both package families.
  - macOS: native `x86_64` on `macos-15-intel` and explicit `arm64` on `macos-latest`. Both retain `MACOSX_DEPLOYMENT_TARGET=14.0`.
- Runtime dependency metadata keeps `rdkit>=2023.9.6` unpinned so modern systems can install newer RDKit wheels, while older glibc systems can fall back to the newest compatible RDKit wheel available for their Python and platform. The root [`uv.lock`](../../uv.lock) is only the local development resolution and may choose older RDKit for some Python splits; it is not the published wheel's version cap.
- Automatic release deployment publishes to PyPI only when all are true:
  - `release.published`
  - `prerelease == false`
  - `target_commitish == main`
  - tag starts with `v`
  - tag does not contain `rc` / `.dev`
- A manual `workflow_dispatch` run with `build_release_wheels=true` builds and
  tests the complete release wheel matrix and uploads workflow artifacts for
  inspection, without publishing to PyPI or attaching files to a GitHub Release.

## Tag Synchronization Policy

Mirror is managed by Gitea project settings.

## Required Gitea Variables and Runner Settings

- Repository variable: `SERVER_URL`
  - Example: [`https://gitea.example.com:13000`](https://gitea.example.com:13000)
  - Used to compose publish/check URLs for internal registry upload.
- Package install steps explicitly clear proxy variables before running `pip`
  or `uv`.
- There is no hard-coded proxy fallback in the workflow. If action download
  needs a proxy, configure it on the runner/service environment because action
  download happens before job steps.
- Repository secrets:
  - `OWNER`
  - `PASSWORD`
Runner/action resolution:

- The current Gitea workflow uses `checkout@v6` from the configured internal action URL.
- Configure action source at Gitea server level, for example
  `DEFAULT_ACTIONS_URL=self` with mirrored actions, or `github` with outbound network.
- Workflow steps do not configure package-install proxies.

With one-way mirror `Gitea -> GitHub`:

- tags created in local/Gitea sync to GitHub;
- tags created only in GitHub do not sync back automatically.

Recommended operation:

1. Create tags in local/Gitea first.
2. Let mirror sync tags to GitHub.
3. Create GitHub Release from the synced stable tag.

## Operating Procedure

1. Develop on feature branches.
2. Merge to `develop` for continuous internal test-package publishing.
3. Merge to `main` after validation.
4. For internal prerelease package validation, tag `main` with `vX.Y.ZrcN` or `vX.Y.Z.devN`.
5. Before the first public release, optionally run the GitHub workflow manually with
   `build_release_wheels=true` and verify every wheel artifact.
6. For formal release, tag `main` with stable `vX.Y.Z` and publish a non-prerelease GitHub Release.
7. Verify PyPI artifacts and GitHub Release assets after deploy.

### Recovering an interrupted stable release

After fixing CI on `main`, dispatch the workflow on `main` with `release_tag`
set to an existing stable tag (for example `v0.1.10`). This explicitly enables
publishing. The workflow verifies that the tag is reachable from `main` and has
a published, non-prerelease GitHub Release targeting `main`, then pins all
build, test and deployment checkouts to that tag's commit. The release tag is
not moved. Leaving `release_tag` empty preserves the existing build-only
behavior of `build_release_wheels=true`.

```sh
gh workflow run ci.yaml --ref main -f release_tag=v0.1.10
```

## Guardrails

- `main` branch pushes must not publish internal test packages directly.
- Use `develop` for continuous internal test package channel.
- Use `main` prerelease tags for internal prerelease channel.
- Use stable `vX.Y.Z` tags for formal release only.
