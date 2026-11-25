# SMILES/XYZ Benchmark

[English](README.md) | [中文](README.zh-CN.md)

这个 benchmark 在由 SMILES 生成的 XYZ case 上比较分子重建方法。它适合快速回归检查，
因为输入文件紧凑，并且 RDKit 可以提供参考分子。

## Case 准备

对每个 SMILES 条目，[`../../scripts/molgr_cases_smiles_csv.py`](../../scripts/molgr_cases_smiles_csv.py) 会：

1. 使用 RDKit 解析 SMILES。
2. 生成 canonical reference SMILES。
3. 添加氢原子。
4. 使用 RDKit 嵌入 3D 构象。
5. 使用 UFF 优化构象。
6. 计算总形式电荷和自由基电子数。
7. 将分子转换为 XYZ block。

如果 case 准备失败，benchmark 会为每个方法记录一行 skipped。

## 方法

Benchmark 使用 [`methods/__init__.py`](methods/__init__.py) 中的共享方法注册表：

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

默认按行串行执行；`molgr_cpp` 的单分子调用仍保留后端 target-bucket 并行。需要批量
吞吐时可给入口增加 `--jobs N`，由 native batch 池统一管理 C++ worker。

在仓库根目录创建或刷新环境：

```bash
bash scripts/benchmark_env.sh create
```

## 运行

推荐直接运行：

```bash
bash scripts/benchmark_env.sh run python benchmarks/smiles_xyz_benchmark/run.py \
  --input tests/test_cases.csv \
  --limit 10 \
  --out benchmarks/_runs/smiles-demo
```

如果需要连续运行 benchmark 命令，可以切换当前 shell：

```bash
eval "$(bash scripts/benchmark_env.sh env)"
```

然后运行：

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

`--input` 指向包含 SMILES 的文本或 CSV 风格文件。loader 接受：

- 每个非空行一个 SMILES 的普通文件
- 名为 `smiles` 的表头
- 名为 `canonicalsmiles` 的表头
- 名为 `canonicalsmi` 的表头
- 两行 `general` + SMILES 表头布局

`--limit` 在表头解析后生效，用于限制本次运行使用的 SMILES 条目数。

## 输出

输出目录包含：

- `results.csv`：每个 `(case, method)` 运行一行。
- `summary.csv`：按方法聚合的数量和延迟统计。

CSV schema 与 `molfile_xyz_benchmark` 共享。可用的计时字段会被展开，
`timing_ms_breakdown_json` 保留完整计时字典。

## 可复现性

为了比较不同运行：

- 固定输入文件和 `--limit`
- 固定方法注册表
- 使用同一个 [`../uv.lock`](../uv.lock)
- 记录 Git commit 和输出目录名
