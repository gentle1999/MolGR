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

MolGR 名称来源于 Moleculer Graph Reconstructor。

MolGR 是一个从 XYZ 坐标重建分子图的 Python 包。它接收 XYZ 文本、总电荷和自旋多重度，
输出带有键级、三维构象、可选配位键和可选立体化学信息的 `rdkit.Chem.Mol`。

当前项目有两套可切换后端：

- `cpp`：默认后端，通过 `molgr._core` 暴露 C++ 加速实现，适合常规使用和性能敏感场景。
- `python`：位于 [`src/molgr/fallback/`](src/molgr/fallback/) 的 Python 参考实现，用于语义对齐、调试和回归测试。

## 快速使用

主要入口是 `molgr.interface.xyz_to_rdmol`：

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
    backend="cpp",  # 也可以使用 "python"
)
```

重建前，MolGR 会用原子序数之和减去 `total_charge` 得到总电子数，并拒绝不可能的
自旋多重度：偶电子体系只能使用奇数多重度，奇电子体系只能使用偶数多重度。通过校验后，
`spin_multiplicity` 会在入口处转换为
`total_radical_electrons = spin_multiplicity - 1`。默认会补充可能的配位键和立体化学信息；
需要关闭时可传入 `make_dative_bonds=False` 或 `make_stereochemistry=False`。

## 配置

运行时配置基于 dataclass。默认使用包级全局对象
[`molgr.config.CONFIG`](src/molgr/config.py)；调用方可以直接修改这个全局配置对象，也可以向
`xyz_to_rdmol(..., config=...)` 传入独立的 [`MolGRConfig`](src/molgr/config.py)。

```python
from molgr.config import CONFIG

CONFIG.cpp_backend.enable_target_bucket_parallelism = True
CONFIG.cpp_backend.target_bucket_parallel_threshold = 1
CONFIG.cpp_backend.target_bucket_parallel_max_threads = None
```

Windows 平台的 `CONFIG.cpp_backend.max_threads` 默认为 `1`。不建议在 Windows 上开启
C++ 后端线程并行：Open Babel 并发重建可能触发 Python 无法捕获的原生 access violation。
除非已经在本地完整验证实际工作负载，否则应保持 `max_threads=1`。Linux 和 macOS 仍使用
自动并行的默认值 `None`。

C++ 后端是 Python fallback 语义的默认加速实现。C++ 专属开关可以改变调度、缓存或线程安全
vendor 实现，但同一个 `MolGRConfig` 下不能改变最终入选分子。

### 危险行为告警

仍然暴露裸 `OBMol*` 的低层接口会在第一次调用时输出一次
`[UNSAFE_OPENBABEL]` 警告。此类指针由调用方手动管理，不能跨线程共享，必须在所有访问结束后且
只释放一次；优先使用 `xyz_to_rdmol()`、`MoleculeData` 和 native batch API。提高 C++ 日志级别可以
隐藏告警，但不会改变 Open Babel 的线程安全或生命周期约束。对于不受信任输入或必须调用其他裸
Open Babel/Pybel API 的场景，仍应使用进程级隔离。

MolGR 会拒绝在已经 import MolGR/Open Babel 的父进程中通过 POSIX `fork` 创建的子进程继续
调用 MolGR，因为子进程继承的 native mutex、线程池和 Open Babel 全局状态可能不一致。Linux
和 macOS 多进程任务应使用 `multiprocessing.get_context("spawn")`；Windows 默认使用 `spawn`，
不受此限制。先 `fork` 再在子进程中首次 import MolGR，或 `fork` 后立即 `exec` 新程序，也不会
继承已初始化的 MolGR 状态。

在 `spawn` 子进程中直接调用 `molgr.interface.xyz_to_rdmol()` 也是支持的；此时建议让进程池
承担跨分子并行，并在每个子进程使用 `max_threads=1`、关闭单分子内部并行，避免进程数与
native worker 数相乘造成过量调度。

## 高级使用

### 并行执行策略

对于低延迟场景或 benchmark，应串行调用 `xyz_to_rdmol()`。每次 C++ 调用仍可在内部自动并行
处理金属价态目标桶，因此不需要再套一层 Python worker 池。

对于高吞吐批量任务，应使用 native batch API。它接受任意有限迭代器，由有界 C++ worker 池
完成重建，并流式返回 `(input, result, status)` 三元组。即使使用默认的 `ordered=False`，每个
结果也会携带原始输入，不会丢失 XYZ 与重建结果的对应关系。

```python
from molgr import ReconstructionBatchRequest, iter_xyz_to_rdmol_batch

requests = (
    ReconstructionBatchRequest(xyz, total_charge=0, spin_multiplicity=1)
    for xyz in xyz_blocks
)

for request, molecule, status in iter_xyz_to_rdmol_batch(
    requests,
    backend="cpp",
    max_workers=None,  # 自动选择 native worker 数量
):
    consume(request, molecule, status)
```

当 batch 使用多个 worker 时，MolGR 会自动关闭单分子内部的目标桶和候选评分并行，避免嵌套
并行造成线程过量；batch 只有一个 worker 时仍保留单分子内部并行。不要在 native batch 外层
再套 `joblib`、线程池或进程池，应直接通过 batch 的 `max_workers` 控制并行数。`python`
后端继续保持串行，作为语义参考实现。

## 安装

MolGR 需要 Python `>=3.8`。运行时依赖和构建依赖以 [`pyproject.toml`](pyproject.toml) 为准；
默认开发工作流使用 `uv`。

### C++ 编译环境要求

从源码构建会编译 `molgr._core` 扩展。本地编译环境需要提供：

- CMake `>=3.15`。
- 支持 C++17 的编译器：
  - Linux：GCC 9+ 或 Clang 10+。
  - macOS：当前 Xcode Command Line Tools 提供的 Apple Clang。
  - Windows：MSVC Build Tools 2019+，并安装 C++ workload。
- 当前 Python 解释器对应的 Python development headers。使用 `uv` 创建虚拟环境时，
  通常由所选 Python 安装提供。
- [`pyproject.toml`](pyproject.toml) 中声明的 Python 构建包：`scikit-build-core`、
  `pybind11`、`setuptools-scm` 和 `openbabel-wheel`。
- 目标平台可用的 OpenBabel wheel。CMake 会从已安装的 `openbabel-wheel` 包中定位
  OpenBabel 头文件和库文件。

从源码安装并构建 C++ 扩展：

```bash
cd MolGR
uv sync --dev
uv pip install -e . -v --no-build-isolation
```

可编辑安装会通过 `scikit-build-core`、`pybind11` 和当前 Python 环境中的 OpenBabel 构建
C++ 扩展。如果修改了 C++ 代码或 pybind11 绑定，运行相关测试前需要重新构建：

```bash
make cpp-build
```

如果需要模拟 CI 中直接使用 PyPI 的依赖解析方式，而不是使用 [`pyproject.toml`](pyproject.toml) 中配置的本地镜像，
可以显式指定 index：

```bash
uv sync --dev --index https://pypi.org/simple --default-index https://pypi.org/simple
uv pip install -e . -v --no-build-isolation \
  --index https://pypi.org/simple \
  --default-index https://pypi.org/simple
```

## 项目结构

- [`src/molgr/interface.py`](src/molgr/interface.py)：公开 API，负责配置解析、后端选择和 RDKit 后处理。
- [`src/molgr/fallback/`](src/molgr/fallback/)：Python 参考实现。
- [`src/cpp/`](src/cpp/)：C++ 重建实现。
- [`src/bindings/`](src/bindings/)：`molgr._core` 的 pybind11 绑定。
- [`src/molgr/utils/`](src/molgr/utils/)：转换、等价性检查和 RDKit 后处理工具。
- [`tests/`](tests/)：公开行为测试、fallback 测试、转换器测试和 C++ 一致性测试。
- [`benchmarks/`](benchmarks/)：benchmark 入口、专用依赖和运行说明。
- [`docs/`](docs/)：架构说明和发布流程文档。

## 算法概览

MolGR 把 Python fallback 作为语义参考，把 C++ 后端作为加速路径。两套后端遵循同一套
高层重建逻辑：

1. 解析 XYZ 输入，并根据总电荷、自旋多重度和运行时配置确定重建目标。
2. 根据 `backend="cpp"` 或 `backend="python"` 分发到对应后端。
3. 对不含金属的结构，按电荷分离动作数递增搜索邻位自由基种子层；各层共享共振去重状态，并在首个有效层停止。
   只有全部主层为空时，才进入畸变 pi 键和断键恢复层。
4. 对含金属的结构，先把金属从有机骨架中临时分离，枚举可能的金属状态，并把这些状态按照
   诱导出的有机部分电荷和自由基目标分桶。
5. 每个有机目标桶只重建一次，然后复用于对应的金属候选组合。
6. 对有机候选或“金属 + 有机骨架”组合候选评分，选择最优状态。
7. 将后端结果转换为 RDKit 分子，并按需补充配位键和立体化学信息。

更完整的调用图、金属搜索、共振恢复和 C++ 优化说明见
[`docs/architecture/ALGORITHM_ARCHITECTURE.zh-CN.md`](docs/architecture/ALGORITHM_ARCHITECTURE.zh-CN.md)。

## 开发

创建或刷新开发环境：

```bash
uv sync --dev
```

以可编辑模式构建扩展：

```bash
uv pip install -e . -v --no-build-isolation
```

常用检查命令：

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pyright
uv run pytest
```

构建源码包和 wheel：

```bash
uv build
```

根目录 [`makefile`](makefile) 提供常用开发快捷命令：

- `make format`：运行 Ruff formatter，会修改文件。
- `make lint`：运行 `ruff check . --fix`，会修改文件。
- `make test`：运行 pytest。
- `make type-check`：对 `src` 运行 MyPy 和 Pyright。
- `make cpp-build`：重新构建可编辑 C++ 扩展，并刷新 C++ IDE 配置。
- `make stubs`：重新生成编译扩展的类型存根。

`make stubs` 会对 `molgr._core` 运行 `pybind11-stubgen`，保留公开 `pipeline`
stubs 以及 [`src/molgr/_core/dev/`](src/molgr/_core/dev/) 下的开发/一致性测试 helper
stubs，然后格式化并 lint 生成的 `.pyi` 文件。

可选：根据当前环境生成 VSCode/clangd C++ 配置：

```bash
uv run python scripts/gen_vscode_config_with_ob.py --build-dir build
```

## 测试

运行完整测试：

```bash
uv run pytest
```

开发时可以运行更小范围的测试：

```bash
uv run pytest tests/test_main.py
uv run pytest tests/test_backend_reconstruction_regression.py
uv run pytest tests/test_fallback_resonance.py
uv run pytest tests/test_fallback_get_possible_metal_radicals.py
```

修改 C++、pybind11 绑定或 `_core` 公开接口时，推荐流程是：

1. 使用 `make cpp-build` 重新构建扩展。
2. 运行相关的 C++/Python 后端一致性测试，例如
   [`tests/test_backend_reconstruction_regression.py`](tests/test_backend_reconstruction_regression.py)
   [`tests/test_cpp_python_metal_candidate_parity.py`](tests/test_cpp_python_metal_candidate_parity.py)、
   [`tests/test_cpp_uff_atom_typing_cache.py`](tests/test_cpp_uff_atom_typing_cache.py)
   和 [`tests/test_force_field_scoring_policy.py`](tests/test_force_field_scoring_policy.py)。
3. 如果绑定接口变化，运行 `make stubs`，并检查生成的 `.pyi` 文件。

C++ 后端是 Python fallback 语义的加速实现。修改 SMARTS 匹配、UFF setup 状态、
共振候选选择或 C++ 专属加速开关时，需要同步检查
[`docs/architecture/ALGORITHM_ARCHITECTURE.zh-CN.md`](docs/architecture/ALGORITHM_ARCHITECTURE.zh-CN.md#cpppython-后端一致性护栏)
中的后端一致性护栏。

部分测试依赖 OpenBabel；如果当前环境缺少对应依赖，这些测试可能会被跳过。

## Benchmark

Benchmark 依赖与主项目依赖分离在 [`benchmarks/pyproject.toml`](benchmarks/pyproject.toml) 中；它们不参与发布包的依赖解析，
也不参与主 [`uv.lock`](uv.lock)。benchmark 使用独立环境，默认路径是 `.venv-benchmark`，默认 Python 是
`python3.10`。

创建或刷新 benchmark 环境：

```bash
bash scripts/benchmark_env.sh create
```

运行 SMILES/XYZ benchmark：

```bash
bash scripts/benchmark_env.sh run python benchmarks/smiles_xyz_benchmark/run.py \
  --input tests/test_cases.csv \
  --limit 10 \
  --out benchmarks/_runs/smiles-demo
```

运行 Molfile/SDF benchmark：

```bash
bash scripts/benchmark_env.sh run python benchmarks/molfile_xyz_benchmark/run.py \
  --input tests/data/sdf/MoNNMo.sdf \
  --limit 1 \
  --out benchmarks/_runs/molfile-demo
```

在 Linux 上比较两个后端对同一组 tmQMg-Corrected 样本的重建结果。先按[数据准备说明](benchmarks/dataset_xyz_benchmark/README.zh-CN.md#tmqmg-corrected)获取指定版本。

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

连续运行 benchmark 命令时，也可以切换当前 shell 环境：

```bash
eval "$(bash scripts/benchmark_env.sh env)"
```

恢复默认项目环境：

```bash
unset UV_PROJECT UV_PROJECT_ENVIRONMENT UV_PYTHON
```

更多说明见 [`benchmarks/README.md`](benchmarks/README.md) 或
[`benchmarks/README.zh-CN.md`](benchmarks/README.zh-CN.md)。

## 文档导航

- [`docs/README.md`](docs/README.md)：英文文档索引。
- [`docs/README.zh-CN.md`](docs/README.zh-CN.md)：中文文档索引。
- [`docs/architecture/ALGORITHM_ARCHITECTURE.md`](docs/architecture/ALGORITHM_ARCHITECTURE.md)：英文算法架构说明。
- [`docs/architecture/ALGORITHM_ARCHITECTURE.zh-CN.md`](docs/architecture/ALGORITHM_ARCHITECTURE.zh-CN.md)：中文算法架构说明。
- [`docs/release/DEVELOPMENT_RELEASE_GUIDE.md`](docs/release/DEVELOPMENT_RELEASE_GUIDE.md)：英文开发与发布指南。
- [`docs/release/DEVELOPMENT_RELEASE_GUIDE.zh-CN.md`](docs/release/DEVELOPMENT_RELEASE_GUIDE.zh-CN.md)：中文开发与发布指南。
- [`benchmarks/README.md`](benchmarks/README.md)：英文 benchmark 环境、运行命令和输出说明。
- [`benchmarks/README.zh-CN.md`](benchmarks/README.zh-CN.md)：中文 benchmark 环境、运行命令和输出说明。

## 协议

MolGR 使用 GNU General Public License v2 发布。完整协议内容见 [`LICENSE`](LICENSE)。
