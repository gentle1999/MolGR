# tmQMg XYZ Benchmark

这个 benchmark 用 tmQMg 的 CSV/XYZ 对比较不同重建方法，复用同一套方法注册表。

## 输入

`--csv` 指向 tmQMg 元数据 CSV，至少需要 `id`、`smiles`、`charge` 列。
`--xyz-dir` 指向包含 `<id>.xyz` 文件的目录。

官方数据来源、固定版本、SHA-256 和下载命令见
[`MOLECULE_REVIEW_TOOL.zh-CN.md`](../../docs/development/MOLECULE_REVIEW_TOOL.zh-CN.md#获取-tmqmg-数据)。

重建输入严格限于 XYZ 坐标、数据集给出的全局电荷，以及本 benchmark 固定的闭壳层
多重度 1（自由基电子数为 0）。tmQMg 的 `smiles` 字段只在重建完成后作为参考图用于
比较和分子式一致性诊断；它绝不参与键、局部电荷、自由基或候选选择。

子集筛选参数：

- `--start-row` 和 `--end-row` 按 1-based CSV 行号筛选。
- `--ids` 按 tmQMg id 筛选，可以重复传入或使用逗号分隔。
- `--limit` 在筛选后限制 case 数量。

执行参数：

- `--methods` 限制共享方法注册表。验证 C++/Python 后端对齐时使用
  `molgr_cpp,molgr_fallback`。
- `--cpp-accelerations default|all` 选择 C++ 后端加速 preset。两个 preset 都保持
  target-bucket 并行开启；`all` 会额外开启 vendor UFF atom-typing cache 等可选 C++ 加速项。
- `--enable-uff-atom-typing-cache` 在 default preset 下开启 vendor UFF atom-typing cache。
- `--process-workers 1`（默认）按行串行运行；`molgr_cpp` 仍开启单分子内部的
  target-bucket 并行。对 `molgr_cpp` 使用 `N > 1` 时只启动一个 benchmark 子进程，
  将 `N` 交给 native batch worker 池，不再叠加外部进程。其他方法才会在 `N > 1`
  时拆分外部 worker。
- `--case-timeout-seconds` 设置每个方法、每个 case 的 wall-time 限制；传 `0` 表示关闭。

后端对齐检查示例：

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

## 等价搜索

参考图比较和 SMILES 往返检查使用判定器默认的 5,000 共振搜索预算，
双向搜索、所有区域及映射合计最多访问 10,000 个状态。搜索耗尽仍归为
“无法判定”，原子映射上限和共振路径长度上限保持不变。

## 输出

benchmark 会在 `--out` 下写出 `results.csv` 和 `summary.csv`。

## 准确率排除项

版本化规则 `comparison_annotations.json` 将含有至少 4 个硼原子的 1176 个结构排除在
重建准确率比较之外。tmQMg 答案和 MolGR 答案使用的普通 Lewis 图都不能表达这些硼簇中
的多中心三中心二电子（3C2E）键，因此双方答案均标记为“不可判定”，而不是判定任一方
正确或错误。benchmark 仍可运行重建以保留候选结构和耗时诊断，但这些条目不进入准确率
分母。`YULBOY` 是当前 tmQMg 版本标记集合中明确记录的例外。
