# 基于已发布分子几何的跨数据集评测

[English](README.md)

这个入口读取原始坐标，对每个已发布构象重建，再按母分子等权汇总。
复用[已有算法适配器](../smiles_xyz_benchmark/methods)和匹配规则，重建输入不含参考键。
原始下载、压缩 SQLite 输入、工作进程日志和逐构象结果均保存在 Git 忽略目录。

## 数据集状态

| 数据集 | 来源 | 导入和评测范围 | 当前验证状态 |
| --- | --- | --- | --- |
| GEOM-QM9 | [GEOM 发布版](https://doi.org/10.7910/DVN/JNGTDF) | 全部已发布构象，按母分子分组 | 导入器单元测试通过；完整下载待完成 |
| GEOM-Drugs | [GEOM 发布版](https://doi.org/10.7910/DVN/JNGTDF) | 全部已发布构象，按母分子分组 | 已用最先下载的 3 个完整分子、744 个构象验证入口；不具有抽样代表性 |
| BorylXAT-DB | [Zenodo v2](https://doi.org/10.5281/zenodo.21330198) | 稳定结构；过渡态和其他类别单独保留，不纳入稳定分子图正确率 | 已有 30 个校正测试结构；完整数据库下载暂时返回 HTTP 403 |
| BDE | 具体发布版尚待确认 | SDF 导入要求来源链接和明确的多重度属性 | 尚无该数据集的测量结果 |
| tmQMg-Corrected | [2026-11-23 发布版](https://github.com/gentle1999/tmQMg-Corrected) | 固定发布版原生 Lewis 图；排除规则和比较预算来自 manifest | 全量导入计数验证通过；尚无全量性能成绩 |

早期 tmQMg 性能对比 使用旧草案的 72,578 个目标，
不是当前发布版成绩。小规模流程验证不是全量结果；新版标准需要重新评测。

## 参考答案与构象口径

[GEOM 官方示例](https://github.com/learningmatter-mit/geom/blob/master/tutorials/02_loading_rdkit_mols.ipynb)
指出，部分几何发生了解离，其图结构会与母分子 SMILES 不同。因此每个构象使用它随附的
`rd_mol` 图作为参考，母分子 SMILES 仅用于汇总分组。发布图由 xyz2mol 赋值，
对这类算法而言，参考匹配率仍不构成独立的化学正确率。
保留构象能量、简并度、来源 ID 和统计权重，但不据此筛选或加权。
每个已存储构象使用一次；`totalconfs` 和简并度不会生成额外坐标记录。

电荷来自数据源。GEOM 的零／一个形式自由基电子分别按源图推定为单重态／双重态；
多自由基图不能单独确定自旋耦合，多重度保持未确定。
BorylXAT 开壳层类别还要求自旋密度之和与一个未配对电子一致。
连续的自旋密度不被当作整数原子自由基数。

BorylXAT 的 B 和反应物／产物配合物复用[源图校正与几何校验](../../tests/data/borylxat/README.md)。
校正失败的参考保持无法判定，不用 MolGR 输出替代。
其他稳定类别保留源图，检查组成、电荷和自由基数。
过渡态不赋予整数键图的参考正确率。全量导入必须通过官方文件大小和 MD5 检查；
目前尚未在全部源数据库记录上验证导入器。

匹配允许共振，默认忽略立体标记和金属配位键。GEOM、BorylXAT 和 BDE 默认使用
`max_resonance=5000`、比较限时 5 秒；tmQMg-Corrected 使用发布版规定的
100,000 和 60 秒。这些设置保存为数据源配置，由同一运行器执行，不另设评分入口。
通用导入器检查元素组成、源电荷、电子奇偶性和形式自由基数，但这些检查不证明化学正确性。
tmQMg 的资格和输入多重度直接采用发布版，不能用形式自由基数重新推断耦合自旋。
所有数据源均保存原生 RDKit 参考图和 SHA-256；SMILES 仅供展示，不参与参考图还原。
旧 schema 1 输入库需重新准备。自旋未确定和参考排除数量保留在覆盖统计中，
只有 `valid` 参考进入重建和评分队列。

## 准备本地数据

使用[基准环境](../README.zh-CN.md)。GEOM 官方文件约 50 GB，导入后的数据库还需要额外空间。
此发布版的 `.tar.gz` 实际为未压缩 tar。导入器顺序读取成员，不解包全部文件；
反序列化时仅允许 RDKit 分子构造器。

```bash
mkdir -p .local/datasets/geom .local/datasets/borylxat/21330198
curl --fail --location --continue-at - \
  'https://dataverse.harvard.edu/api/access/datafile/4327252?format=original' \
  --output .local/datasets/geom/rdkit_folder.tar.gz
curl --fail --location --continue-at - \
  'https://zenodo.org/records/21330198/files/BorylXAT-DB.db?download=1' \
  --output .local/datasets/borylxat/21330198/BorylXAT-DB.db

python -m benchmarks.dataset_xyz_benchmark.prepare \
  --dataset geom-qm9 --source .local/datasets/geom/rdkit_folder.tar.gz \
  --out .local/datasets/benchmark-cases/geom-qm9.sqlite
python -m benchmarks.dataset_xyz_benchmark.prepare \
  --dataset geom-drugs --source .local/datasets/geom/rdkit_folder.tar.gz \
  --out .local/datasets/benchmark-cases/geom-drugs.sqlite
python -m benchmarks.dataset_xyz_benchmark.prepare \
  --dataset borylxat --source .local/datasets/borylxat/21330198/BorylXAT-DB.db \
  --out .local/datasets/benchmark-cases/borylxat.sqlite
```

固定的 GEOM 文件为 50,137,579,520 字节，MD5 为
`e8f2168b7050652db22c976be25c450e`；BorylXAT v2 为 291,348,480 字节，MD5 为
`d486fbcc36f856bbebf303b5cda4950f`。全量导入拒绝不匹配文件。
`--limit-molecules N` 可用于小规模流程验证，仍导入所选分子的**全部构象**，
并明确标记为 pilot。归档顺序选择不能作为随机抽样。
导入中断后保留未完成标记，不能开始评测；重新导入需使用新的输出文件。

无需下载即可验证已有 BorylXAT 测试结构：

```bash
python -m benchmarks.dataset_xyz_benchmark.prepare \
  --dataset borylxat-fixtures --source tests/data/borylxat \
  --out .local/datasets/benchmark-cases/borylxat-fixtures.sqlite
```

## tmQMg-Corrected

唯一标准是 [tmQMg-Corrected 2026-11-23](https://github.com/gentle1999/tmQMg-Corrected)，
固定 tag `v2026-11-23`、revision `aac372de0e859a6daf451084abe84a9ae58ad8e6`。
数据在独立仓库维护，不随 MolGR 打包。下载较大；从 MolGR 根目录运行：

```bash
git clone --depth 1 --branch v2026-11-23 \
  https://github.com/gentle1999/tmQMg-Corrected.git \
  .local/datasets/tmqmg-corrected
git -C .local/datasets/tmqmg-corrected rev-parse HEAD
# aac372de0e859a6daf451084abe84a9ae58ad8e6

python -m benchmarks.dataset_xyz_benchmark.prepare \
  --dataset tmqmg-corrected --source .local/datasets/tmqmg-corrected \
  --out .local/datasets/benchmark-cases/tmqmg-corrected.sqlite
python -m benchmarks.dataset_xyz_benchmark.native_batch \
  --cases .local/datasets/benchmark-cases/tmqmg-corrected.sqlite \
  --workers 8 --checkers 4 \
  --out .local/reports/dataset-performance/tmqmg-corrected-batch
python -m benchmarks.dataset_xyz_benchmark.report \
  --cases .local/datasets/benchmark-cases/tmqmg-corrected.sqlite \
  --results .local/reports/dataset-performance/tmqmg-corrected-batch/molgr_cpp.sqlite \
  --out .local/reports/dataset-performance/tmqmg-corrected-batch/molgr_cpp-summary.json
```

开发期全量判定默认使用 8 个原生重建线程和 4 个独立的 RDKit 后处理与评分进程。每例只重建一次。评分端保留双精度坐标和 MolGR 电子属性，按发布版规则比较实际分子图。日志给出含检查的端到端吞吐。需要逐分子延迟或进程隔离的硬超时控制时，使用通用的 `run` 入口。相同来源和配置可从已写入的批次继续。

- Manifest SHA-256：`ea04563079dfe4344601f6c896e36c24a847a69e4f320ec4f906a69203c3faf3`。
- Layout v4：`data/*.sdf`、`evidence/inputs/*.xyz`、`evidence/inputs/*.jsonl`，
  manifest 使用相对仓库根目录的路径。只有 SDF 的归档不足以重建。
- 74,547 个输入中，72,040 个有发布的原生 Lewis 图；2,507 个排除项包括
  1,189 个硼簇、354 个硼氢桥、770 个缺氢输入和 194 个因电子态证据排除的输入。发布版保留的单重态配体双自由基进入准确率分母；
  MolGR 按总自由基预算重建，无法匹配这些参考图时按实际结果计分。

可用 `--source` 指定其他完整 checkout。导入器验证固定 manifest、所读 XYZ/SDF 分片哈希、
原始 XYZ 身份和原生图哈希，不隐式下载、不退回旧草案或原始 CSV 标签。
完整导入还检查分片及全量计数。排除项保留在覆盖统计，不进入评测队列；
报告同时携带发布版总量、排除分类、实际导入范围和比较策略。
小规模检查在准备时添加 `--limit-molecules 10`，并使用新的输入库和结果目录。
该选项按源顺序选择母分子（包括排除项），不是随机样本，也不是按合格目标编号选择。

重建只接收 XYZ、源电荷和源多重度；参考键、原子电子赋值及本地历史答案不进入重建。
比较直接使用发布的原生图，不接受本地备选答案，不按 SMILES 往返重建参考图。
[离线 fixtures](../../tests/data/tmqmg/README.md) 只是发布版的回归子集，不是第二套标准。
旧草案的 **99.41% / 72,578 个目标** 不能作为本发布版成绩。

新增 Lewis 标注采用 **CC BY 4.0**：注明 tmQMg-Corrected、固定发布版及所作变更，并引用 tmQMg。
参见上游[数据许可与署名](https://github.com/gentle1999/tmQMg-Corrected/blob/aac372de0e859a6daf451084abe84a9ae58ad8e6/sources/DATA-LICENSE.md)。
MolGR 的软件许可不替代数据许可。

## 测量与续跑

每种方法独立运行，多进程分配分子／构象，每个工作进程绑定一个逻辑 CPU，内部一个原生线程。
Linux 入口使用持久的 spawn 工作进程，默认重建限时 30 秒，匹配限时由数据源配置决定。
`--comparison-timeout` 只允许覆盖未锁定的配置，不能改写发布版固定的比较限时。
进程崩溃和超时仍写入结果；算法报错后保留的图照常匹配。
计时不包含输入读取、进程启动、预热和参考匹配。
xyz2mol_tm 的过渡金属入口可用于 tmQMg，不作为 GEOM、BorylXAT 或 BDE 的有机基线。

```bash
python -m benchmarks.dataset_xyz_benchmark.run \
  --cases .local/datasets/benchmark-cases/geom-drugs.sqlite \
  --method molgr_cpp --workers 14 \
  --out .local/reports/dataset-performance/geom-drugs
python -m benchmarks.dataset_xyz_benchmark.report \
  --cases .local/datasets/benchmark-cases/geom-drugs.sqlite \
  --results .local/reports/dataset-performance/geom-drugs/molgr_cpp.sqlite \
  --out .local/reports/dataset-performance/geom-drugs/molgr_cpp-summary.json
```

依次替换为 `rdkit_determine_bonds`、`openbabel_read_xyz`、`cell2mol_v2`、
`xyzgraph_cheminf_full` 和 `molgr_fallback`。同一输出目录仅在输入哈希、代码、原生二进制、
依赖版本、CPU 和限时设置一致时续跑，不把不同代码版本的耗时混成一组结果。


cell2mol 的有机输出可能残留仅限定元素的 RDKit 查询原子。适配器将其普通原子字段复制为
确定的原子，保留键、电荷、同位素、立体标记、氢数、自由基数、原子顺序和坐标。
有歧义的原子或键查询仍报错；此转换使图可以比较，不改变上游的化学赋值。

每种方法写入 `<method>.sqlite`、`<method>-provenance.json` 和工作进程日志；
报告 JSON 包含来源／运行溯源、比较配置、覆盖范围和汇总比例，不另写 tmQMg 专用 CSV。

## 汇总

设分子 *i* 有 *nᵢ* 个可用参考构象，其中 *mᵢ* 个匹配，主匹配率是各分子
*mᵢ / nᵢ* 的平均值，每个分子权重相同。同时报告**全部可用构象均匹配**的分子比例，
以及不加权的逐构象匹配率。超时、无输出图和无法判定均不计为匹配。
参考排除与算法无关，必须与各项匹配率一起报告覆盖范围。

汇总器默认拒绝未完成的运行。`--allow-partial` 可查看进度，但保留完整可用参考分母，
并标记为未完成。pilot 不会被自动提升为全量结果。
完整数据和测试完成前，不发布跨数据集的主性能图。
