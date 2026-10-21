# PlotSeeker

基于 Bangumi Archive 的多实体 RAG 检索系统。目标是根据模糊剧情、角色特征、
世界观设定或碎片线索定位最可能对应的作品。

当前进度：**P0-P6、P8 已完成；P7 仅完成扩容预算分析；P9 Agent 实验性实现（确定性 + LLM CLI），默认不启用**。
所有结果以实际运行日志和报告为准，未确认信息见 [`docs/unconfirmed.md`](docs/unconfirmed.md)。

## 当前审计结果（2026-10-21 实测）

- 9 份文件合计 6,764,913 条有效记录，解析/解码/结构异常 0 行。
- 作品 695,909；角色 223,927；人物 103,653；章节 1,709,524。
- 作品简介非空 572,333；角色简介非空 170,975；章节描述非空 110,840。
- 全量审计耗时 143.2 秒，进程峰值内存 758.3 MiB，9 份文件 SHA256 已写入报告。
- 关联表存在孤儿记录（最大 7,744 条）与 1 条重复自然键；`person-relations` 的
  person/character ID 空间大量重叠，P2 需按 `person_type` 分命名空间处理。
- 详细分布、样例与首期索引建议见 [`docs/data-audit.md`](docs/data-audit.md)。

## 当前数据库结果（2026-10-21 实测）

- 全量导入耗时 135.3 秒，进程峰值内存 421.1 MiB，数据库 1.645 GiB。
- 数据库：`storage/plotseeker.db`；导入报告：`storage/import-report.json`。
- 行数：作品 695,909、角色 223,927、人物 103,653、章节 1,709,524。
- 关联表：作品-角色 451,258、作品-人物 2,220,090、作品关系 939,430、
  人物-角色 291,476、人物关系 129,645。
- `subject_relations` 原始 939,431 行中有 1 条完全重复自然键，按主键忽略。
- `PRAGMA integrity_check = ok`，主键重复 0，`PRAGMA foreign_keys = 1`，
  外键违规 0。原始归档存在悬空引用，因此未声明硬外键，改用导入后完整性查询。
- 同一数据集完整重建两次，表行数与完整性结果一致；覆盖已有数据库必须显式
  `--overwrite`。

## 当前文档构建结果（2026-10-21 实测）

- 全量构建耗时 39.2 秒，进程峰值内存 29.9 MiB。
- 输出：`data/processed/documents/`，清单：`data/processed/documents/manifest.json`。
- 共 **846,606** 篇文档：作品 572,333、角色 170,405、章节 103,868。
- 作品过滤 123,576 条简介为空；角色过滤 52,952 条空简介、570 条无可用作品关联；
  章节过滤 1,598,684 条空描述、683 条无有效作品、4,016 条仅制作信息、
  2,273 条过短或无法判定。
- 输出约 942 MiB；固定种子抽样 **50/50** 条可追溯到原始实体。
- 重复全量构建得到相同 JSONL SHA256。
- 文档文件 SHA256、行数、截断统计和过滤原因均写入 manifest。

## 当前向量索引结果（2026-10-21 实测）

- 模型：`BAAI/bge-m3`，revision `5617a9f61b028005a4858fdac845db406aefb181`，
  1024 维，CLS pooling + L2 归一化，`max_length=1024`，GPU fp16。
- 抽样：10,000 篇真实文档，作品 4,000 / 角色 3,000 / 章节 3,000。
- 索引：3 个 Chroma collection（`p4_subject_docs`、`p4_character_docs`、
  `p4_episode_docs`），cosine distance，HNSW `ef_search=200`。
- 完整重建耗时 319.7 秒，其中编码 187.2 秒，峰值内存 3,682.3 MiB，
  GPU 峰值分配 1,459 MiB，索引体积 113.2 MiB。
- 重新打开索引后三集合条数与 10,000 个抽样 ID 全部核对通过；
  30 条已知目标 sanity query 为 30/30 hit@1、30/30 hit@5，
  P50 34.97 ms、P95 72.94 ms。
- 断点恢复已实际测试：在 20 个批次后停止，重新运行后从 checkpoint 继续完成。
- 显式重建已实际测试；旧 collection 删除后遗留的 HNSW 目录会在重建时清理。
- 产物：`storage/vector_store/p4_checkpoint.json`、
  `storage/vector_store/p4_report.json`；模型缓存：
  `storage/model_cache/bge-m3/`。

## 当前关键词与混合召回结果（2026-10-21 实测）

- 关键词索引：`storage/fts/p5_keyword_10k.db`，与 P4 相同的 10,000 篇文档。
- 同时建立 `fts_jieba` 与 `fts_bigram`；默认分词 **jieba**。
- 分词比较（479 条名称/片段查询）：
  - jieba Recall@5 `0.9395`，MRR@10 `0.9184`；
  - bigram Recall@5 `0.8935`，MRR@10 `0.8769`。
- 关键词索引 60.3 MiB，构建 38.4 秒。
- 查询预处理：NFKC、OpenCC 繁转简、标点归一、停用词、32 token 上限。
- 混合召回：RRF `k=60`；角色主作品权重 1.0，额外作品最多 4 个、权重 0.35。
- 证据包含 doc_id、来源、两路排名、文本片段、关系路径和贡献值。
- 低置信结果带 `warnings`，不会把向量近邻伪装成确定答案。
- 命令行 Demo：`python scripts/search.py "失去记忆的魔王在现代学校" --mode hybrid`。

## 当前评测结果（2026-10-21 实测）

- 评测集：`benchmarks/queries.jsonl`，128 条（118 可回答、10 合成负样例）。
- 覆盖：名称、别名、作品简介、角色简介、章节剧情、跨类型、繁简、碎片、负样例。
- 报告：`docs/evaluation.md`；明细：`storage/evaluation/p6_report.json`；
  失败记录：`benchmarks/failures.md`。

| 模式 | Recall@5 | MRR@10 | Top-1 | P50 | P95 | Confident FP（10 负样例） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| keyword | 1.0 | 0.9890 | 0.9831 | 58.76 ms | 85.42 ms | 0 |
| vector | 0.9915 | 0.9703 | 0.9576 | 61.89 ms | 82.02 ms | 0 |
| hybrid | 1.0 | 0.9958 | 0.9915 | 120.53 ms | 153.07 ms | 0 |

混合检索 Recall@5 与最佳单路持平，MRR@10 和 Top-1 有提升。评测查询由已索引文本
生成，不是独立人工查询集，指标可能偏乐观；详见 `docs/evaluation.md`。

## 当前 API 结果（2026-10-21 实测）

```powershell
python scripts/serve.py --host 127.0.0.1 --port 8000 --device cuda
```

- `GET /health`：索引、模型缓存、SQLite 状态。
- `GET /search?q=...&top_k=10&mode=hybrid&candidate_k=50`：返回候选作品与证据。
- 已实际启动服务并完成 `/health`、keyword、hybrid 请求。
- 首次 hybrid/vector 请求加载 BGE-M3 约 13-16 秒，之后进程内复用。
- 空/超长查询、非法参数返回 422；索引缺失返回 503；超时返回 504。

## P9 Agent 实验结果（实验性，默认不启用）

- 实现位置：`src/agent/`、`scripts/agent.py`、`scripts/evaluate_agent.py`。
- 工具：作品/角色/章节检索、相关作品、澄清追问。
- 无 LLM；确定性路由 + 最多 4 步工具预算。
- 对比评测：`docs/agent-evaluation.md`。

| 变体 | R@5 | MRR@10 | Top-1 | 可回答澄清率 | 负样例澄清率 | 平均工具调用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.0 | 0.9958 | 0.9915 | 0.0 | 0.0 | 0.0 |
| agent | 1.0 | 0.9958 | 0.9915 | 0.0 | 0.0 | 1.16 |
| agent_clarify_low | 1.0 | 0.9958 | 0.9915 | 0.0 | 1.0 | 1.24 |

默认 Agent 未提升 Recall@5，却增加工具调用，因此不推荐默认启用。
`--clarify-low` 模式在合成负样例上把 10/10 转为澄清，可作为低置信/不可回答查询的
可选策略，但仍需真实用户查询验证。详见 `docs/agent.md`。

## 目录结构

```text
PlotSeeker/
├── data/
│   ├── raw/                  # 9 份原始 JSONL，只读，不修改
│   └── processed/            # 清洗后的中间数据
│       └── documents/        # P3 文档 JSONL + manifest.json
├── storage/                  # SQLite / 模型缓存 / 向量索引等可重建产物
│   ├── plotseeker.db
│   ├── model_cache/bge-m3/
│   ├── vector_store/         # Chroma + p4_checkpoint.json + p4_report.json
│   ├── fts/                  # P5 FTS5 数据库与构建报告
│   └── evaluation/           # P6 JSON 评测明细
├── scripts/
│   ├── audit_data.py         # P1 流式审计入口
│   ├── import_sqlite.py      # P2 SQLite 导入入口
│   ├── build_documents.py    # P3 检索文档构建入口
│   ├── download_model.py     # P4 固定 revision 的模型下载
│   ├── build_index.py        # P4 向量索引构建入口
│   ├── build_keyword_index.py # P5 FTS5 关键词索引
│   ├── search.py             # P5 命令行检索 Demo
│   ├── build_benchmark.py    # P6 开发评测集生成
│   ├── evaluate.py           # P6 三基线评测
│   ├── serve.py              # P8 FastAPI 启动入口
│   ├── agent.py              # P9 确定性 Agent CLI
│   ├── agent_llm.py          # P9 OpenAI-compatible LLM Agent CLI
│   └── evaluate_agent.py     # P9 Agent 对比评测
├── src/
│   ├── config.py             # 统一路径配置
│   ├── ingestion/
│   │   ├── jsonl.py          # 流式 JSONL 读取
│   │   ├── infobox.py        # infobox 与别名解析
│   │   ├── text_quality.py   # 章节描述质量判断与清洗
│   │   ├── audit.py          # 审计与字段分析
│   │   ├── importer.py       # SQLite Schema、导入与完整性检查
│   │   ├── document_builder.py # P3 文档模板、截断与溯源
│   │   └── resources.py      # 峰值内存统计
│   ├── embeddings/
│   │   ├── base.py           # Embedder 协议
│   │   ├── bge_m3.py         # BGE-M3 加载、CLS pooling、归一化
│   │   ├── fake.py           # 测试用确定性假向量
│   │   ├── chroma_store.py   # Chroma 持久化与元数据映射
│   │   └── indexer.py        # 抽样、断点、重建、校验与报告
│   ├── retrieval/
│   │   ├── preprocess.py     # 中文归一化、jieba/bigram、停用词
│   │   ├── keyword_index.py  # FTS5 构建与关键词检索
│   │   ├── vector_search.py  # P4 索引查询适配
│   │   ├── relations.py      # 角色/章节 -> 作品
│   │   ├── fusion.py         # RRF 融合与证据
│   │   └── search.py         # 统一检索服务与置信度
│   ├── evaluation/
│   │   ├── benchmark_builder.py
│   │   ├── metrics.py
│   │   └── runner.py
│   ├── api/
│   │   └── app.py            # FastAPI /health /search
│   └── agent/
│       ├── tools.py          # 作品/角色/章节/关系/澄清工具
│       ├── agent.py          # 确定性单 Agent 策略
│       ├── llm_config.py     # API Key / Base URL / Model 配置
│       ├── llm_agent.py      # OpenAI-compatible tool-calling Agent
│       └── evaluation.py     # 启用/禁用 Agent 对比
├── tests/
│   ├── test_audit.py
│   ├── test_import_sqlite.py
│   ├── test_document_builder.py
│   ├── test_vector_index.py
│   ├── test_keyword_retrieval.py
│   ├── test_hybrid_retrieval.py
│   ├── test_api.py
│   ├── test_agent.py
│   └── test_llm_agent.py
├── benchmarks/               # P6 queries.jsonl、benchmark_manifest.json、failures.md
├── docs/
│   ├── plan.md               # 项目总计划
│   ├── remaining-work.md     # 阶段状态与剩余工作
│   ├── data-audit.md         # 审计脚本生成的全量报告
│   ├── evaluation.md         # P6 评测报告
│   ├── architecture.md       # P8 架构与接口说明
│   ├── agent.md              # P9 Agent 工具、策略与结论
│   ├── agent-evaluation.md   # P9 启用/禁用对比评测
│   ├── agent-llm.md          # LLM Agent CLI 配置与限制
│   ├── scaling.md            # P7 扩容预算分析
│   └── unconfirmed.md        # 需要人工确认的信息
├── config/
│   ├── agent.example.json    # 可提交的 LLM 配置模板
│   ├── agent.json            # 本地 API Key，已被 .gitignore 排除
│   └── README.md
├── scratch/                  # 临时实验脚本与运行日志
├── requirements.txt
├── requirements.lock.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

`data/raw/`、`storage/`、`scratch/` 已被 `.gitignore` 排除，避免原始数据、
索引和临时文件进入版本库。

## 环境

要求 Python 3.11+。当前依赖统一安装在系统 Python
（`E:\python\Lib\site-packages`）；项目 `.venv` 只作为解释器入口，
通过 `--system-site-packages` 继承全局包，不再在项目内安装业务依赖。

复现顺序如下，**必须在未激活 `.venv` 的系统 Python 中安装**：

```powershell
# 1. 全局安装项目依赖（不要先激活 .venv）
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps chromadb==1.5.9

# 2. 创建只继承全局包的解释器入口
python -m venv --system-site-packages .venv
```

不要在新 `.venv` 内执行 `pip install -r requirements.txt`，否则依赖会再次写入
项目的 `.venv\Lib\site-packages`。

当前已验证的环境：

- 系统 Python：`chromadb 1.5.9`、`fastapi 0.139.0`、`uvicorn 0.46.0`、
  `httpx 0.27.0`、`pytest 9.0.3`、`torch 2.6.0+cu124`、
  `transformers 5.12.1`、`huggingface-hub 1.22.0`、`sentencepiece 0.2.2`、
  `numpy 2.2.6`。
- 项目 `.venv`：仅含 `pip`，`include-system-site-packages = true`。
- `onnxruntime-gpu 1.26.0` 提供 `onnxruntime` 模块；不要安装 CPU 版
  `onnxruntime` 覆盖它。因此 chromadb 使用 `--no-deps` 安装。
- GPU：NVIDIA RTX 3060 Laptop，6 GB 显存，CUDA 12.4 可用。

如果新环境没有 CUDA PyTorch，需要先从 PyTorch 官方 CUDA 索引安装与当前
`transformers` 兼容的版本，再运行 P4。

`data/raw/` 为只读输入；模型权重、Chroma 索引和 checkpoint 都在 `storage/`
下，可随时重建，不会进入 Git。

## 运行审计

全量审计（读取 9 份文件，生成 `docs/data-audit.md`）：

```powershell
python scripts/audit_data.py
```

冒烟审计（每个文件只读前 5000 行，用于验证脚本；不做关联完整性结论）：

```powershell
python scripts/audit_data.py --limit 5000 --output scratch/data-audit-smoke.md
```

可选参数：

- `--raw-dir`：原始数据目录，默认 `data/raw`。
- `--output`：Markdown 报告路径，默认 `docs/data-audit.md`。
- `--sample-size`：章节描述质量分析的抽样条数，默认 3000。
- `--seed`：抽样随机种子，默认 42。

审计脚本以二进制只读方式打开原始文件，同一遍读取中计算 SHA256、行数、
字段完整性、关联完整性和 infobox 解析统计。报告采用临时文件加原子替换写入。

## 构建 SQLite 数据库

首次全量导入：

```powershell
python scripts/import_sqlite.py
```

覆盖重建（先写入临时库，成功后原子替换；缺少该参数时会拒绝覆盖）：

```powershell
python scripts/import_sqlite.py --overwrite
```

小规模冒烟导入（默认使用独立的 `storage/plotseeker-smoke.db` 和
`storage/import-report-smoke.json`）：

```powershell
python scripts/import_sqlite.py --limit 5000
```

参数：

- `--db` / `--report`：自定义数据库与报告路径。
- `--batch-size`：每批写入行数，默认 5000。
- `--limit`：每个文件只读前 N 行；关联行只有在两端实体都已导入时才写入，
  被过滤的行单独记录，不当作孤儿数据。
- `--overwrite`：允许替换已有数据库。

数据库表：

- 主表：`subject`、`character`、`person`、`episode`
- 关联表：`subject_characters`、`subject_persons`、`subject_relations`、
  `person_characters`、`person_relations`
- 元数据：`import_metadata`（Schema 版本、源文件 SHA256、行数、完整性结果等）

原始 `infobox`、`tags`、`meta_tags`、`score_details`、`favorite`、`career`
等字段分别以原文或 JSON 文本保留，不丢弃业务字段。`source_file` 记录主表
实体来源，原始 ID 全部保留。`type`、`platform`、`role`、`relation_type` 等
码值按原始整数保存，未在 P2 阶段解释业务含义，分布见审计报告。

导入报告还包括行数对账、主键重复检查、孤儿关联、`person_relations` 的
person/character 命名空间统计，以及角色→作品、章节→作品等抽样 SQL。

## 构建检索文档

全量构建（默认读取 `storage/plotseeker.db`，写入 `data/processed/documents/`）：

```powershell
python scripts/build_documents.py
```

覆盖重建：

```powershell
python scripts/build_documents.py --overwrite
```

冒烟构建（每类最多 50 篇）：

```powershell
python scripts/build_documents.py --limit 50
```

输出文件：

- `subjects.jsonl`：名称、中文名、infobox 别名、标签、简介
- `characters.jsonl`：角色名、别名、所属作品、角色简介
- `episodes.jsonl`：所属作品、章节名、播出日期、经质量过滤的剧情
- `manifest.json`：行数、SHA256、过滤原因、截断统计、溯源抽样结果

每个文档包含稳定 `doc_id`、`entity_type`、`entity_id`、`subject_id`、
`source`、`text` 和 `metadata`。原始 ID 与来源文件全部保留；`type`/`platform`
等未解释码值只进入 metadata，不进入检索文本。

截断规则：作品简介 4,000 字、角色简介 3,000 字、章节剧情 2,000 字，
全文分别不超过 6,000 / 5,000 / 4,000 字；别名、标签、作品列表也有上限。
所有截断记录在文档 `metadata.truncated_fields`，并在 manifest 汇总。

章节只保留 `narrative` / `mixed` 描述；`mixed` 会删除制作信息行。空描述、
仅制作信息、无法判定、无有效作品的章节均过滤并记录原因。

> 当前沙箱中 `.venv` 进程可能带 Low Mandatory Level，写入 `data/processed`
> 时可能出现 `WinError 5`。遇到时可用系统 Python 运行构建脚本，或调整目标目录
> 的完整性标签；普通本地环境不需要此操作。P4 已把 Chroma/SQLite 的临时目录
> 指向 `storage/vector_store/_tmp`，不依赖系统 TEMP 的 ACL。

## 下载 Embedding 模型

默认通过 `hf-mirror.com` 下载并自动回退到 `huggingface.co`：

```powershell
python scripts/download_model.py
```

可选参数：

- `--repo`：模型仓库，默认 `BAAI/bge-m3`。
- `--revision`：固定 revision，默认
  `5617a9f61b028005a4858fdac845db406aefb181`。
- `--endpoint` / `--fallback-endpoint`：主端点和回退端点。
- `--output`：本地目录，默认 `storage/model_cache/bge-m3`。

脚本只下载 `transformers` 所需文件（约 2.3 GB），写入
`plotseeker-model.json` 记录仓库、revision、实际端点和下载时间。下载显式使用
匿名请求，不依赖本机已过期的 HuggingFace token。

## 构建向量索引

全量 10,000 篇文档（作品 4,000 / 角色 3,000 / 章节 3,000）：

```powershell
python scripts/build_index.py
```

显式删除并重建目标 collection：

```powershell
python scripts/build_index.py --rebuild
```

中断恢复：重新运行同一条构建命令即可。checkpoint 默认位于
`storage/vector_store/p4_checkpoint.json`，每批成功写入后原子更新，并在恢复时
逐条核对已记录 ID 是否仍在索引中。

离线冒烟（不加载 BGE-M3）：

```powershell
python scripts/build_index.py --fake-embedder --sample-sizes 50,50,50 `
    --store-dir storage/vector_store_smoke
```

常用参数：

- `--manifest`：P3 文档清单，默认 `data/processed/documents/manifest.json`。
- `--sample-sizes`：三类文档抽样数，默认 `4000,3000,3000`。
- `--batch-size`：编码与写入批大小，默认 16。
- `--max-length`：BGE-M3 tokenizer 最大长度，默认 1024。
- `--device`：`auto` / `cuda` / `cpu`。
- `--sanity-queries`：已知目标 sanity query 数，默认 30；`0` 可关闭。
- `--stop-after-batches`：调试用中断钩子，实际用于验证断点恢复。
- `--no-verify-source`：跳过 P3 JSONL 的 SHA256 校验，不建议用于正式构建。

索引结构：

- `p4_subject_docs`、`p4_character_docs`、`p4_episode_docs` 三个 collection。
- cosine distance；HNSW `ef_construction=200`、`max_neighbors=64`、
  `ef_search=200`。
- Chroma metadata 只放标量；别名、标签、作品 ID 列表等转为 JSON 字符串。
- 文档原文同时写入 Chroma，便于后续证据返回。

构建前会校验 manifest、JSONL SHA256、行数和确定性抽样 ID。已有完整索引时
默认拒绝覆盖；模型、文档或抽样参数变化也会拒绝混用。`--rebuild` 会显式删除
三个目标 collection，并清理 Chroma 1.5.x 删除 collection 后遗留的 HNSW 目录。

产物：

- `storage/vector_store/`：Chroma 持久化索引。
- `storage/vector_store/p4_checkpoint.json`：身份、批次、已处理 ID。
- `storage/vector_store/p4_report.json`：条数校验、编码速度、内存/显存、
  索引体积、查询延迟、sanity 结果和重建信息。

P4 只构建 10,000 篇抽样文档，不在本阶段进行百万级全量编码。

## 构建关键词索引

从 P4 checkpoint 读取同一批 10,000 个文档 ID，构建独立 FTS5 数据库：

```powershell
python scripts/build_keyword_index.py
```

覆盖重建：

```powershell
python scripts/build_keyword_index.py --rebuild
```

默认同时建立 `fts_jieba` 与 `fts_bigram`，用名称/片段查询比较后选择默认分词。
输出：

- `storage/fts/p5_keyword_10k.db`：文档表 + 两张 FTS5 表。
- `storage/fts/p5_keyword_report.json`：文档数、体积、分词比较、构建耗时。

关键词索引已有文件时默认拒绝覆盖；构建使用临时文件 + 原子替换。

## 命令行检索

```powershell
# 混合检索
python scripts/search.py "失去记忆的魔王在现代学校" --mode hybrid --top-k 5

# 仅关键词
python scripts/search.py "时间循环" --mode keyword

# 机器可读
python scripts/search.py "妹妹用脚抵住门" --mode hybrid --json
```

结果包含候选作品、置信度、warning、两路排名、文本片段、关系路径和贡献值。
不存在的实体不会凭空生成；低置信度的纯向量近邻会带 `weak_keyword_coverage`
或 `vector_only_no_keyword_match`。

## 生成评测集与运行评测

```powershell
python scripts/build_benchmark.py
python scripts/evaluate.py
```

输出：

- `benchmarks/queries.jsonl`：128 条开发评测查询。
- `benchmarks/benchmark_manifest.json`：查询分布和数据来源。
- `benchmarks/failures.md`：逐条失败记录。
- `storage/evaluation/p6_report.json`：指标与每条查询明细。
- `docs/evaluation.md`：自动生成的评测报告。

评测集由已索引文本生成，不是独立人工查询集；详见 `docs/evaluation.md` 的限制章节。

## 启动 API 服务

```powershell
python scripts/serve.py --host 127.0.0.1 --port 8000 --device cuda
```

接口：

- `GET /health`
- `GET /search?q=...&top_k=10&mode=hybrid&candidate_k=50`

示例：

```powershell
curl "http://127.0.0.1:8000/search?q=时间循环&mode=hybrid&top_k=5"
```

首次 hybrid/vector 请求会加载 BGE-M3；之后复用进程内模型与索引。
超时、索引缺失、非法参数分别返回 504 / 503 / 422。架构与错误处理见
[`docs/architecture.md`](docs/architecture.md)。

## 运行 Agent（实验性）

```powershell
python scripts/agent.py "时间循环学园有什么相关作品"
python scripts/agent.py "妹妹用脚抵住门" --clarify-low
python scripts/agent.py --tools
python scripts/evaluate_agent.py
```

默认模式在低置信时返回候选并带 warning；`--clarify-low` 会直接追问补充线索。
Agent 不调用 LLM，工具结果全部可追溯到 P2/P5 数据。

## LLM Agent CLI（本地实验）

确定性 Agent 之外，另有基于 OpenAI-compatible tool calling 的 LLM CLI。
配置文件 `config/agent.json` 不进入 Git；首次使用先填入 Key：

```powershell
notepad config\agent.json
# 如果文件被删除，可重新复制模板：
Copy-Item config\agent.example.json config\agent.json -Force
```

```json
{
  "api_key": "sk-你的Key",
  "base_url": "https://api.deepseek.com/v1",
  "model": "deepseek-chat"
}
```

也可以用 `PLOTSEEKER_LLM_API_KEY`、`PLOTSEEKER_LLM_BASE_URL`、
`PLOTSEEKER_LLM_MODEL` 覆盖。

```powershell
# 只检查配置，不调用 API、不加载模型
python scripts/agent_llm.py --check-config

# 本地查询
python scripts/agent_llm.py "失去记忆的魔王在现代学校"
python scripts/agent_llm.py "时间循环学园有什么相关作品" --json
```

LLM 只能通过本地工具检索，证据仍来自 P2/P5；配置、兼容性和限制见
[`docs/agent-llm.md`](docs/agent-llm.md)。

## P7 / P9 状态

- P7：未执行全量编码。10k 指标不能外推百万级；扩容预算、去重和失败回滚检查项见
  [`docs/scaling.md`](docs/scaling.md)。
- P9：Agent 已实验性实现，但默认 Agent 未提升 Recall@5，只增加工具调用，
  因此不推荐默认启用。`--clarify-low` 可作为低置信/不可回答查询的可选模式，
  仍需真实用户查询验证。详见 `docs/agent.md`。

## 测试

```powershell
python -m pytest
```



当前共 74 个测试，覆盖：JSONL 流式读取、infobox/别名解析、语言与章节描述
启发式、审计端到端流程、SQLite 全量导入与覆盖保护、P3 文档构建、P4 假向量
构建/Chroma 持久化/断点恢复/重建、P5 预处理/FTS5/混合召回/关系扩展、
P6 指标与报告结构、P8 API 参数校验/超时/错误映射，以及 P9 工具规格、路由、
关系扩展、澄清和证据溯源，以及 LLM Agent 的配置/工具调用/重试/澄清/参数错误。
测试不依赖真实 BGE-M3、真实 10k 索引或外部 LLM API。
