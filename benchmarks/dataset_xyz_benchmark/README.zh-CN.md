# 基于已发布分子几何的跨数据集评测

[English](README.md)

这个入口读取原始坐标，对每个入选构象重建，再按母分子等权汇总。
复用[已有算法适配器](../smiles_xyz_benchmark/methods)和匹配规则，重建输入不含参考键。
原始下载、压缩 SQLite 输入、工作进程日志和逐构象结果均保存在 Git 忽略目录。

## 数据集状态

| 数据集 | 评估的分子类型 | 跨数据集对比采用的输入 |
| --- | --- | --- |
| [GEOM-QM9](https://doi.org/10.7910/DVN/JNGTDF) | 有机小分子 | 每个母分子固定取一个发布构象 |
| [GEOM-Drugs](https://doi.org/10.7910/DVN/JNGTDF) | 药物样分子 | 固定抽取 100,000 个母分子，每个取一个构象 |
| [BorylXAT-DB](https://doi.org/10.5281/zenodo.21330198) | 硼自由基及其卤原子转移反应组分 | 按结构键去重的优化反应物与产物，用映射反应 SMILES 作参考 |
| [tmQMg-Corrected](https://github.com/gentle1999/tmQMg-Corrected) | 单重态过渡金属配合物 | 发布版中有原生 Lewis 参考图的结构 |
| [BDE](https://doi.org/10.6084/m9.figshare.12158646.v1) | 有机小分子自由基碎片 | 每个含碳双重态碎片取发布 SDF 中的优化结构 |

跨数据集对比中，每次重建限时 10 秒，随后独立进行的参考图比较限时 5 秒。各数据集分别统计匹配率，分母包括重建失败和超时的输入。

## 参考答案与构象口径

[GEOM 官方示例](https://github.com/learningmatter-mit/geom/blob/master/tutorials/02_loading_rdkit_mols.ipynb)
指出，部分几何发生了解离，其图结构会与母分子 SMILES 不同。因此每个构象使用它随附的
`rd_mol` 图作为参考，母分子 SMILES 仅用于汇总分组。发布图由 xyz2mol 赋值，
对这类算法而言，参考匹配率仍不构成独立的化学正确率。
入选构象保留发布时的坐标、能量、简并度、来源 ID 和统计权重。

下列两条 GEOM 命令均使用种子 `20261125`。QM9 按归档顺序为每个母分子选取一个构象，并记录每个入选构象的参考图状态。Drugs 对 `summary_drugs.json` 中去重的 pickle 路径按带种子的 SHA-256 排序，依次选出 100,000 个母分子。若归档成员缺失、构象集为空或抽中构象的参考图不合格，则顺延至下一个母分子。两套数据均由数据集名、种子和归档成员路径确定构象，通过拒绝采样均匀选取从零开始的构象编号。对合格参考图，若查询仅指定一个确切元素，可将其转换为普通 RDKit 原子用于评分，同时记录源图和评分图的哈希。

[QM9 输入索引](manifests/geom-qm9.csv.gz)和 [Drugs 输入索引](manifests/geom-drugs.csv.gz)按输入库顺序列出实际选中的母分子和构象，格式为 gzip 压缩的 CSV。要读取某条记录的几何，从固定发布归档中打开 `archive_member` 指定的 pickle，取 `entry["conformers"][conformer_index]["rd_mol"]` 即可。

| 字段 | 含义 |
| --- | --- |
| `archive_parent_index` | 母分子在对应子集中从 0 开始的索引。按 tar 中的物理顺序，仅计入该子集的普通 `.pickle` 文件。 |
| `conformer_index` | 在该母分子发布的 `conformers` 列表中从 0 开始的索引。 |
| `case_id`、`source_geom_id` | 前者是从 1 开始的 SQLite 输入编号，后者是发布者给入选构象分配的 `geom_id`。 |
| `reference_status` | 筛选 `valid` 即得实际评分清单。QM9 收录全部 133,232 条抽样记录，其中 133,123 条参与评分；Drugs 的 100,000 条均参与评分。 |
| `sample_rank`（仅 Drugs） | 资格筛选和母分子去重前的抽样名次，从 1 开始。跳过的候选会留下空号。 |

电荷来自数据源。GEOM 的零／一个形式自由基电子分别按源图推定为单重态／双重态；
多自由基图不能单独确定自旋耦合，多重度保持未确定。

BorylXAT 使用 Zenodo v2 的 `BorylXAT-DB.db`。每条反应关联四个独立优化的结构：反应物侧的硼自由基配合物（`complex_r`）和含氯底物（`Cl`），产物侧的硼氯配合物（`complex_p`）和碳自由基（`c_radical`）。坐标取各自 `GS_OPT` 计算的最后一帧。每个组分分别重建，重复出现的结构键只计一次。

参考图来自反应记录中 mapped reaction SMILES 对应侧的片段。用映射号追踪从碳转移到硼的氯，确认片段角色；逐个验证显式氢和其他原子在两侧守恒。组分坐标保留其原始原子顺序，图比较允许原子重排。没有反应标注或标注冲突的结构单列为无法判定。

BDE 使用 St. John 等人发布的优化结构 SDF。保留含碳且键表标有一个未配对电子的记录，并核对逐原子自旋密度（`AtomSpins`）之和在 1 的 ±0.0001 内。闭壳层母分子和无碳碎片不进入分母。电荷取自 SDF 的形式电荷，适配器接收显式氢坐标、总电荷和双重态多重度，各算法按自身接口取用。评分使用同一记录的键表。

五套数据共用同一规则：对于中性开壳层输入，RDKit 和 xyz2mol_tm 通用入口分别使用 `allowChargedFragments=False` 和 `allow_charged_fragments=False` 启用自由基赋值。闭壳层和带净电荷的输入采用上游的形式电荷赋值。xyz2mol_tm 在每次调用中独立设置价态表，自由基模式对 C、N、O 价态的调整只对当前分子生效。xyzgraph 的输出转为 RDKit 图时，须禁用隐式补氢，因为 XYZ 已含全部氢原子。cell2mol v2 的分子 XYZ 入口接收总电荷，没有多重度参数。

匹配允许共振，默认忽略立体标记和金属配位键。GEOM、BorylXAT 和 BDE 默认使用
`max_resonance=5000`、比较限时 5 秒。tmQMg-Corrected 独立发布版的设置为 100,000 和 60 秒。跨数据集耗时对比沿用其 100,000 个状态的搜索预算，比较限时统一为 5 秒。
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
mkdir -p .local/datasets/geom
curl --fail --location --continue-at - \
  'https://dataverse.harvard.edu/api/access/datafile/4327252?format=original' \
  --output .local/datasets/geom/rdkit_folder.tar.gz
mkdir -p .local/datasets/borylxat/21330198
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
  --out .local/datasets/benchmark-cases/borylxat-optimized.sqlite

mkdir -p .local/datasets/bde
curl --fail --location --continue-at - \
  https://ndownloader.figshare.com/files/22357962 \
  --output .local/datasets/bde/20200415_radical_database.sdf.gz
python -m benchmarks.dataset_xyz_benchmark.prepare \
  --dataset bde --source .local/datasets/bde/20200415_radical_database.sdf.gz \
  --out .local/datasets/benchmark-cases/bde.sqlite
```

固定的 GEOM 文件为 50,137,579,520 字节，MD5 为
`e8f2168b7050652db22c976be25c450e`。BorylXAT 导入校验发布数据库的大小和 MD5；已有输入库需重新准备，以使用独立优化的坐标与反应标注。
GEOM 的 `--limit-molecules N` 按归档顺序取前 N 个母分子的**全部构象**，并将输入库标记为 pilot。复现上表的正式 benchmark 时省略该选项。
导入中断后保留未完成标记，不能开始评测；重新导入需使用新的输出文件。

另有独立的 Zenodo 回归夹具，可离线运行：

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
复现五套数据的对比时，全部使用 `--timeout 10 --comparison-timeout 5`，包括 tmQMg-Corrected。
省略 `--comparison-timeout` 时沿用数据源默认值（tmQMg-Corrected 为 60 秒）。显式限时可以缩短发布版的时限。运行记录分别保存来源策略和实际策略。不同限时的结果写入不同目录。
进程崩溃和超时仍写入结果；算法报错后保留的图照常匹配。
计时不包含输入读取、进程启动、预热和参考匹配。
五套数据均可使用 `--method xyz2mol_tm`。工作进程在计时前读取每个 XYZ 的元素符号。含上游 TMC 列表中任一元素（包括 La 和 Lu）的结构走 Hückel 入口，其余结构走 `xyz2mol_local.xyz2mol`。逐条结果中的 `upstream_entry` 记录实际入口。显式传入 `--method xyz2mol_tm_local` 仍可直接使用通用入口。

```bash
python -m benchmarks.dataset_xyz_benchmark.run \
  --cases .local/datasets/benchmark-cases/geom-drugs.sqlite \
  --method molgr_cpp --workers 64 --timeout 10 --comparison-timeout 5 \
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
