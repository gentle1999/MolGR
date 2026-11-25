# Molfile/SDF XYZ Benchmark

[English](README.md) | [中文](README.zh-CN.md)

这个 benchmark 在由 `.mol`、`.molfile` 或 `.sdf` fixture 生成的 XYZ case 上比较分子重建方法。
它适合针对带显式坐标的 curated molecules 做回归检查，包括含金属体系。

## Case 准备

对每个 molfile/SDF fixture，[`../../scripts/molgr_cases_molfile.py`](../../scripts/molgr_cases_molfile.py) 会：

1. 使用 RDKit 读取源文件，并设置 `sanitize=False`、`removeHs=False` 和 `strictParsing=False`。
2. 检查是否存在构象坐标。
3. 计算总形式电荷和自由基电子数。
4. 将结构转换为 XYZ block。
5. 使用源 RDKit 分子作为等价性检查的 ground truth。

如果 case 准备失败，benchmark 会为每个方法记录一行 skipped。

## 方法

本 benchmark 复用 `smiles_xyz_benchmark` 的共享方法注册表：

- `rdkit_determine_bonds`
- `openbabel_read_xyz`
- `cell2mol_v2`
- `molgr_fallback`
- `molgr_cpp`
- `xyzgraph_cheminf_full`
- `xyz2mol_tm`（可选，面向过渡金属配合物）

## 环境

使用 [`../README.zh-CN.md`](../README.zh-CN.md) 中说明的专用 benchmark 环境。
该环境使用 Python `>=3.10,<3.12`，benchmark-only 依赖位于 [`../pyproject.toml`](../pyproject.toml)。

在仓库根目录创建或刷新环境：

```bash
bash scripts/benchmark_env.sh create
```

## 运行

推荐直接运行：

```bash
bash scripts/benchmark_env.sh run python benchmarks/molfile_xyz_benchmark/run.py \
  --input tests/data/sdf/MoNNMo.sdf \
  --limit 1 \
  --out benchmarks/_runs/molfile-demo
```

如果需要连续运行 benchmark 命令，可以切换当前 shell：

```bash
eval "$(bash scripts/benchmark_env.sh env)"
```

然后运行：

```bash
uv run python benchmarks/molfile_xyz_benchmark/run.py \
  --input tests/data/sdf/MoNNMo.sdf \
  --limit 1 \
  --out benchmarks/_runs/molfile-demo
```

恢复默认项目环境：

```bash
unset UV_PROJECT UV_PROJECT_ENVIRONMENT UV_PYTHON
```

## 输入

`--input` 接受：

- 单个 `.mol`、`.molfile` 或 `.sdf` 文件
- 一个递归扫描这些后缀的目录

推荐 fixture 根目录：[`../../tests/data/sdf/`](../../tests/data/sdf/)。

支持 [`../../tests/data/sdf/`](../../tests/data/sdf/) 下的嵌套目录，因此可以运行某个类别路径，
也可以运行整个 fixture tree。

`--limit` 在递归路径排序后限制发现的文件数量。

## 输出

输出目录包含：

- `results.csv`：每个 `(case, method)` 运行一行。
- `summary.csv`：按方法聚合的数量和延迟统计。

CSV schema 与 `smiles_xyz_benchmark` 共享。可用的计时字段会被展开，
`timing_ms_breakdown_json` 保留完整计时字典。
