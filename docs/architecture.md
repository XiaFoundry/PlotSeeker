# PlotSeeker 架构说明

> 更新时间：2026-10-21
> 覆盖：P0-P6、P8，以及实验性 P9 Agent；P7 仅做规模分析。

## 1. 目标与边界

PlotSeeker 根据模糊剧情、角色特征、世界观或碎片线索定位最可能的作品。
系统返回的是**可追溯的检索证据**，不是 LLM 生成的剧情解释。

当前已验证的检索范围：

- 原始数据：Bangumi Archive 9 份 JSONL，6,764,913 条记录。
- 结构化库：`storage/plotseeker.db`，9 个业务表 + 导入元数据。
- 检索文档：P3 全量 846,606 篇；P4/P5/P6 使用固定 10,000 篇抽样。
- 向量索引：3 个 Chroma collection，BGE-M3 1024 维。
- 关键词索引：SQLite FTS5，同一批 10,000 篇文档，jieba 默认分词。
- API：FastAPI `/health` 与 `/search`，进程级复用模型和索引。

未做全量向量编码，也未做独立人工查询集；相关限制见
`docs/unconfirmed.md` 与 `docs/evaluation.md`。

## 2. 数据流

```text
data/raw/*.jsonlines
        |
        v
P1 audit_data.py  -> docs/data-audit.md
        |
        v
P2 import_sqlite.py -> storage/plotseeker.db
        |
        v
P3 build_documents.py -> data/processed/documents/*.jsonl + manifest.json
        |
        +------------------------------+
        |                              |
        v                              v
P4 build_index.py                 P5 build_keyword_index.py
Chroma + BGE-M3                   SQLite FTS5
storage/vector_store/             storage/fts/
        |                              |
        +--------------+---------------+
                       v
             P5 RetrievalService
       keyword / vector / hybrid (RRF)
                       |
                       v
         角色/章节 -> 作品关系扩展
                       |
                       v
         证据组装 + 置信度 + CLI / P8 API
                       |
                       +------> P9 Agent tools -> 澄清/关系扩展
                       |
                       v
             P6 evaluate.py -> docs/evaluation.md
```

## 3. 目录与职责

| 路径 | 职责 |
| --- | --- |
| `src/ingestion/` | P1/P2/P3：审计、SQLite 导入、文档构建 |
| `src/embeddings/` | P4：BGE-M3、Chroma、抽样、checkpoint、重建 |
| `src/retrieval/preprocess.py` | 中文归一化、jieba/bigram 分词、查询停用词 |
| `src/retrieval/keyword_index.py` | FTS5 构建、分词对比、关键词检索 |
| `src/retrieval/vector_search.py` | P4 Chroma 查询适配与模型一致性检查 |
| `src/retrieval/relations.py` | 角色/章节命中到作品的映射 |
| `src/retrieval/fusion.py` | RRF 融合与证据组装 |
| `src/retrieval/search.py` | 统一检索服务、置信度与模式选择 |
| `src/evaluation/` | P6 评测集生成、指标、基线与报告 |
| `src/api/` | P8 FastAPI 服务 |
| `src/agent/` | P9 工具层、确定性单 Agent、对比评测 |
| `scripts/` | 各阶段命令行入口 |
| `tests/` | 单元、集成、API 合约测试 |

## 4. P4 向量索引

- 模型：`BAAI/bge-m3`，revision `5617a9f...`，1024 维，CLS pooling，L2 归一化。
- 抽样：4,000 作品 / 3,000 角色 / 3,000 章节，共 10,000 篇。
- Chroma collection：`p4_subject_docs`、`p4_character_docs`、`p4_episode_docs`。
- cosine distance；HNSW `ef_construction=200`、`max_neighbors=64`、`ef_search=200`。
- checkpoint 记录 source SHA256、模型身份、抽样指纹、批次和已处理 ID。
- `--rebuild` 删除目标 collection 并清理遗留 HNSW 目录。
- 产物：`storage/vector_store/p4_checkpoint.json`、`p4_report.json`。

## 5. P5 关键词与混合召回

### 5.1 关键词索引

- 从 P4 checkpoint 读取 10,000 个 `doc_id`，保证两路检索覆盖同一批文档。
- 独立数据库 `storage/fts/p5_keyword_10k.db`，不修改 P2 主库。
- 同时建立 `fts_jieba` 与 `fts_bigram` 两张 FTS5 表。
- 构建时用名称/片段查询比较两种分词，报告写入
  `storage/fts/p5_keyword_report.json`；本次 jieba 默认。
- 文档表保存原始文本、metadata、jieba/bigram token，用于覆盖率判断。
- 构建使用临时文件 + 原子替换；已有索引默认拒绝覆盖。

### 5.2 查询预处理

- NFKC、大小写归一、可选 OpenCC 繁转简。
- 标点/空白归一。
- jieba `lcut_for_search`；查询侧过滤常见停用词和模板词。
- 查询 token 去重并限制为 32 个，避免长文本 OR 查询过慢。
- 入库与查询共用 `preprocess.py`。

### 5.3 向量检索

- `VectorSearcher` 读取 P4 checkpoint，验证模型 ID、维度、完成状态。
- 查询向量与入库向量使用同一 BGE-M3、CLS pooling、L2 归一化。
- 三个 collection 各取候选，按相似度合并并赋全局 rank。

### 5.4 RRF 与关系扩展

- 关键词与向量各自产生文档级 rank。
- 默认 RRF `k=60`，每个 subject 在每个通道只取最佳贡献，避免文档数量堆叠。
- subject 直接贡献权重 1.0；角色主作品 1.0；角色额外作品最多 4 个，权重 0.35。
- episode 只映射其 `subject_id`。
- 不存在于 `subject` 表的 ID 不会生成候选。
- 每个候选最多保留 6 条证据，每通道最多 3 条。

### 5.5 置信度

- `high`：关键词覆盖率达标且向量也有命中。
- `medium`：仅关键词覆盖率达标。
- `low`：只有弱关键词或纯向量近邻。
- `none`：无候选。
- 关键词覆盖率 = 查询 token 在文档 token 中的命中比例；
  阈值 0.75 在开发集上校准，尚未独立验证。
- 低置信度结果仍可能返回真实近邻，但 API 会带 `warnings`，不伪装成确定答案。

## 6. P6 评测

- 评测集：`benchmarks/queries.jsonl`，128 条（118 可回答、10 合成负样例）。
- 线索类型：名称、别名、作品简介、角色简介、章节剧情、跨类型、繁简、碎片、负样例。
- 基线：keyword / vector / hybrid。
- 指标：Recall@5、MRR@10、Top-1、P50/P95、索引体积、峰值内存、负样例误报。
- 每条指标可在 `storage/evaluation/p6_report.json` 追溯到具体 query_id。
- 报告：`docs/evaluation.md`；失败记录：`benchmarks/failures.md`。
- 评测集由已索引文本生成，不是独立人工查询集；指标可能偏乐观。

## 7. P8 API

### 7.1 启动

```powershell
python scripts/serve.py --host 127.0.0.1 --port 8000 --device cuda
```

模型与索引在首次语义查询时加载，之后进程内复用；健康检查不加载模型。

### 7.2 接口

- `GET /health`
  - 返回 `status`、`api_version`、关键词索引/向量索引/模型缓存/SQLite 状态。
  - 向量索引要求 checkpoint 状态为 `complete`；全部满足时为 `ok`，否则 `degraded`。
- `GET /search?q=...&top_k=10&mode=hybrid&candidate_k=50`
  - `mode`：`keyword` / `vector` / `hybrid`。
  - `top_k`：1-50；`candidate_k`：1-200；`q` 最长 2,000 字符。
  - 返回 `confidence`、`warnings`、两路计数、耗时、候选与证据。

### 7.3 错误处理

| 状态码 | 场景 |
| --- | --- |
| 422 | 空查询、超长查询、非法 mode/top_k/candidate_k |
| 503 | 关键词索引、向量索引、模型或 SQLite 缺失 |
| 504 | 请求超过 `PLOTSEEKER_SEARCH_TIMEOUT_SECONDS`（默认 30 秒） |
| 500 | 未预期内部错误 |

超时只终止请求等待，后台线程中的模型编码可能继续执行；当前没有任务取消机制。

### 7.4 限制

- 无认证、无限流、无 CORS、无持久化请求日志。
- 首次 hybrid/vector 请求包含约 13-16 秒模型加载时间。
- 单进程单模型；多 worker 会重复加载模型与索引。
- GPU 编码由检索服务全局锁串行化，避免并发模型调用冲突。

## 8. 复现命令

```powershell
# P1-P3
python scripts/audit_data.py
python scripts/import_sqlite.py --overwrite
python scripts/build_documents.py --overwrite

# P4
python scripts/download_model.py
python scripts/build_index.py --rebuild

# P5
python scripts/build_keyword_index.py --rebuild
python scripts/search.py "失去记忆的魔王在现代学校" --mode hybrid

# P6
python scripts/build_benchmark.py
python scripts/evaluate.py

# P8
python scripts/serve.py --port 8000
```

## 9. P7 与 P9 状态

- P7 未启动全量扩容。10k 指标不能外推百万级；扩容前需要文本级去重、
  token 总量、磁盘预算、断点恢复和回滚策略评估。见 `docs/scaling.md`。
- P9 已实现实验性单 Agent：`src/agent/`、`scripts/agent.py`、
  `scripts/evaluate_agent.py`。
  - 工具：作品/角色/章节检索、相关作品、澄清追问。
  - 无 LLM；路由和澄清由确定性策略控制。
  - 默认 Agent 在开发集上没有提升 Recall@5，且增加工具调用，不建议默认启用。
  - `clarify_on_low=True` 保守澄清模式在合成负样例上把 10/10 转为澄清，
    但需要真实用户查询验证。
  - 详见 `docs/agent.md` 与 `docs/agent-evaluation.md`。
