# Benchmarks

[English](README.md) | [中文](README.zh-CN.md)

本目录包含 benchmark 入口、benchmark 专用依赖和 benchmark 说明。Benchmark 依赖与根项目依赖图隔离，
不会影响正式发布包的依赖解析。

## 可用 Benchmark

- `smiles_xyz_benchmark`：从 SMILES 输入生成 XYZ case，并比较不同重建方法。
- `molfile_xyz_benchmark`：读取 `.mol`、`.molfile` 和 `.sdf` 输入，转换为 XYZ case，
  并运行同一套方法注册表。
- `dataset_xyz_benchmark`：统一读取已发布的分子坐标，支持 GEOM、BorylXAT、BDE 和 tmQMg-Corrected。
  数据来源、准备、运行与汇总统一见[数据集评测说明](dataset_xyz_benchmark/README.zh-CN.md)。

所有 benchmark 入口都使用同一套共享方法注册表：

- `rdkit_determine_bonds`
- `openbabel_read_xyz`
- `cell2mol_v2`
- `molgr_fallback`
- `molgr_cpp`
- `xyzgraph_cheminf_full`
- `xyz2mol_tm`

`cell2mol_v2`、`xyzgraph_cheminf_full` 和 `xyz2mol_tm` 是对比基线。`molgr_fallback` 使用 Python 参考后端，
`molgr_cpp` 使用默认 C++ 后端。

## 环境

- 项目运行时：Python `>=3.8`
- Benchmark 运行时：Python `>=3.10,<3.12`
- 推荐包管理器：`uv`
- Benchmark 依赖文件：[`pyproject.toml`](pyproject.toml)
- Benchmark lock 文件：[`uv.lock`](uv.lock)
- 默认 benchmark 虚拟环境：`.venv-benchmark`
- 默认 benchmark Python 可执行文件：`python3.10`

Benchmark 运行时范围比项目运行时更窄，因为 `xyzgraph`、`cell2mol` 和 `cosymlib`
等可选对比栈有较旧的依赖约束。Benchmark 依赖集也保留 `numpy<2`，以兼容这些栈。

## 快速开始

在仓库根目录创建或刷新专用 benchmark 环境：

```bash
bash scripts/benchmark_env.sh create
```

运行一个小规模 SMILES/XYZ benchmark：

```bash
bash scripts/benchmark_env.sh run python benchmarks/smiles_xyz_benchmark/run.py \
  --input tests/test_cases.csv \
  --limit 10 \
  --out benchmarks/_runs/smiles-demo
```

运行一个小规模 Molfile/SDF benchmark：

```bash
bash scripts/benchmark_env.sh run python benchmarks/molfile_xyz_benchmark/run.py \
  --input tests/data/sdf/MoNNMo.sdf \
  --limit 1 \
  --out benchmarks/_runs/molfile-demo
```

已发布数据集统一使用 `python -m benchmarks.dataset_xyz_benchmark.prepare`、
`.run` 和 `.report`；[完整命令和来源说明](dataset_xyz_benchmark/README.zh-CN.md)只在该目录维护。
后端对齐时，对同一个输入库分别运行 `--method molgr_cpp` 和 `--method molgr_fallback`。

如果需要连续运行 benchmark 命令，可以把当前 shell 切到 benchmark project：

```bash
eval "$(bash scripts/benchmark_env.sh env)"
```

然后使用普通 `uv run` 命令：

```bash
uv run python benchmarks/smiles_xyz_benchmark/run.py \
  --input tests/test_cases.csv \
  --limit 10 \
  --out benchmarks/_runs/smiles-demo
```

恢复默认项目环境：

```bash
unset UV_PROJECT UV_PROJECT_ENVIRONMENT UV_PYTHON
```

## 输入

SMILES/XYZ benchmark：

- `--input` 指向包含 SMILES 的文本或 CSV 风格文件。
- 接受名为 `smiles`、`canonicalsmiles` 或 `canonicalsmi` 的表头。
- 每个 SMILES 会由 RDKit 嵌入、UFF 优化、转换为 XYZ，并与原始 RDKit 分子比较。

Molfile/SDF benchmark：

- `--input` 接受单个 `.mol`、`.molfile` 或 `.sdf` 文件，也接受递归扫描的目录。
- 推荐 fixture 根目录：[`../tests/data/sdf/`](../tests/data/sdf/)。
- 支持 [`../tests/data/sdf/`](../tests/data/sdf/) 下的嵌套 fixture 分类。
- 每个输入结构会由 RDKit 读取、转换为 XYZ case，并与源分子比较。

SMILES/Molfile 入口的 `--limit` 限制 case 数，`--out` 指定输出目录。
数据集入口使用准备阶段的 `--limit-molecules` 和运行阶段的 `--method`、`--workers`。

## 输出

SMILES/Molfile 入口在 `--out` 目录下写入两个主文件；数据集入口使用上述统一 SQLite/JSON 流程：

- `results.csv`：每个 `(case, method)` 尝试一行，包含状态、预测、等价性结果和计时字段。
- `summary.csv`：按方法聚合的指标，包括数量和延迟统计。

`results.csv` 的主要列包括：

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

出现的计时列会被展开。`timing_ms_breakdown_json` 保留完整的方法专用计时字典，便于 profiling。

## 结果解读

`reference_status` 决定准确率分母：`valid` 进入，`excluded` 表示已有证据确认参考错误，`unresolved` 表示参考尚不可判定。参考筛选独立于算法结果。

以 `evaluator_decision` 统计图匹配：`equivalent`、`not_equivalent`、`inconclusive` 分别计为匹配、不匹配、无法判定。API `status` 与其独立；报错后保留的图仍参与比较，确认等价则计入匹配数。没有比较结果的尝试保留在分母中，但不计匹配。

`summary.csv` 中的 `success_count`、`fail_count`、`inconclusive_count` 和 `runtime_failure_count` 划分可用参考；最后一项统计未获得比较判定的尝试。`api_error_count` 单独记录报错数，`matched_with_errors` 记录报错图中的匹配数。`results.csv` 保留逐项错误、判定和计时，便于复核。

## 可选 `cell2mol_v2` 说明

`cell2mol_v2` 是可选 benchmark 基线。Benchmark 依赖文件指向上游 `cell2mol` 的 `v2`
Git revision。如果启用或重新分发这个基线，请确认用法符合其许可证条款，包括可能的 GPL 义务。

## 可选 xyz2mol_tm 基线

创建基准环境后，安装固定版本的上游代码：

```bash
git clone https://github.com/jensengroup/xyz2mol_tm.git .local/third-party/xyz2mol_tm
git -C .local/third-party/xyz2mol_tm checkout db2734b084045ef895528c7a3d5040d38da23983
uv pip install --python .venv-benchmark/bin/python --no-deps -e .local/third-party/xyz2mol_tm
```

该版本的 wheel 打包列表遗漏 Hückel 子包，因此使用 editable 安装。现有基准环境提供 RDKit（含 `rdEHTTools`）、NumPy 和 NetworkX。环境同步若移除了该可选包，需重新安装。克隆代码和原始结果均保存在 Git 忽略的 `.local/` 中。

tmQMg 运行器通过 `--methods xyz2mol_tm` 选择此方法。适配器调用 `get_tmc_mol(xyz_file, overall_charge, with_stereo=False)`，评估其返回的 RDKit 图。该入口不接收多重度；其 SMILES 往返会丢弃坐标，因此保留上游配位键，不添加依赖几何的 MolGR 后处理。临时文件读写和序列化计入耗时。失败保留为错误；仅序列化失败时保留已返回的图。

共享注册表先加载 RDKit 的 Hückel 和 MolStandardize 模块，再加载 MolGR/Open Babel 原生适配器，以避免基准环境 Linux RDKit wheel 中已复现的加载顺序崩溃。已有 Python 进程应先导入注册表，再导入 MolGR；若加载顺序已不安全，适配器会提示改用新进程。上游重建算法未修改。性能运行器记录安装版本、源码哈希及克隆版本。

上游 tmQMg 的原始 SMILES 由 xyz2mol_tm 的 Hückel 方法生成。当前评测仅使用 tmQMg-Corrected 2026-11-23 发布的 72,040 张 Lewis 图及其证据等级和排除规则；原始标签仅作来源记录，不是标准答案。 [tmQMg](https://github.com/uiocompcat/tmQMg#data) · [校正集](dataset_xyz_benchmark/README.zh-CN.md#tmqmg-corrected)

## 子 Benchmark 文档

- [`smiles_xyz_benchmark/README.zh-CN.md`](smiles_xyz_benchmark/README.zh-CN.md)
- [`smiles_xyz_benchmark/README.md`](smiles_xyz_benchmark/README.md)
- [`molfile_xyz_benchmark/README.zh-CN.md`](molfile_xyz_benchmark/README.zh-CN.md)
- [`molfile_xyz_benchmark/README.md`](molfile_xyz_benchmark/README.md)
- [`dataset_xyz_benchmark/README.zh-CN.md#tmqmg-corrected`](dataset_xyz_benchmark/README.zh-CN.md#tmqmg-corrected)
- [`dataset_xyz_benchmark/README.md#tmqmg-corrected`](dataset_xyz_benchmark/README.md#tmqmg-corrected)

## 更多原始几何数据集

[GEOM-QM9、GEOM-Drugs、BorylXAT-DB 与待确认的 BDE 来源](dataset_xyz_benchmark/README.zh-CN.md)：全部构象导入、按分子汇总、可续跑多进程及当前数据可用性。小规模流程验证与全量评测分开报告。
