# PlotSeeker 剩余工作清单

> 更新时间：2026-10-21
> 依据：`docs/plan.md` 与当前实际产物
> 用途：记录已完成、进行中、未开始的工作，以及各阶段的实测结果和验收条件。

## 1. 总体进度

| 阶段 | 状态 | 主要产物 |
| --- | --- | --- |
| P0 环境与骨架 | 已完成 | `.venv`、`requirements.txt`、`requirements.lock.txt`、目录、README |
| P1 全量数据审计 | 已完成 | `scripts/audit_data.py`、`docs/data-audit.md` |
| P2 SQLite 结构化库 | 已完成 | `storage/plotseeker.db`、`storage/import-report.json` |
| P3 检索文档构建 | 已完成 | `data/processed/documents/*.jsonl`、`manifest.json` |
| P4 向量索引 | 已完成 | `scripts/build_index.py`、`src/embeddings/`、`storage/vector_store/` |
| P5 关键词与混合召回 | 已完成 | `src/retrieval/`、`scripts/build_keyword_index.py`、`scripts/search.py`、`storage/fts/` |
| P6 评测体系 | 已完成 | `benchmarks/`、`src/evaluation/`、`scripts/evaluate.py`、`docs/evaluation.md` |
| P7 扩容与优化 | 部分完成（仅预算分析） | `docs/scaling.md`，未执行全量编码 |
| P8 FastAPI 与完整文档 | 已完成 | `src/api/`、`scripts/serve.py`、`docs/architecture.md`、`tests/test_api.py` |
| P9 Agent 扩展（可选） | 实验性完成（默认不启用） | `src/agent/`、`scripts/agent.py`、`scripts/agent_llm.py`、`config/agent.example.json`、`docs/agent.md`、`docs/agent-llm.md` |

## 2. 当前已可用的产物

| 产物 | 实测结果 |
| --- | --- |
| 原始 JSONL | 9 份，6,764,913 条有效记录，解析异常 0 |
| SQLite | 1.645 GiB，9 个业务表 + `import_metadata`，完整性检查通过 |
| 作品文档 | 572,333 篇，653.3 MiB |
| 角色文档 | 170,405 篇，168.7 MiB |
| 章节文档 | 103,868 篇，120.1 MiB |
| 文档合计 | 846,606 篇，约 942 MiB |
| 文档溯源 | 固定种子 50/50 条通过校验 |
| 文档确定性 | 重复全量构建得到相同 JSONL SHA256 |
| BGE-M3 | revision `5617a9f...`，1024 维，本地 2.27 GB 权重 |
| 向量索引 | 10,000 篇，3 个 Chroma collection，113.2 MiB |
| 索引校验 | 10,000 个 doc_id 全部回溯通过；重开索引后条数一致 |
| 索引 sanity | 30/30 hit@1，30/30 hit@5；P50 34.97 ms / P95 72.94 ms |
| 关键词索引 | 10,000 篇，60.3 MiB，默认 jieba |
| P6 开发评测集 | 128 条（118 可回答 / 10 负样例） |
| P6 hybrid | Recall@5 1.0，MRR@10 0.9958，Top-1 0.9915 |
| P8 API | `/health`、`/search` 已实际启动并返回 200 |
| P9 Agent | 默认与 hybrid 持平；`clarify-low` 负样例 10/10 转为澄清 |

关键过滤量：

- 作品：过滤 123,576 条空简介。
- 角色：过滤 52,952 条空简介、570 条无可用作品关联。
- 章节：过滤 1,598,684 条空描述、683 条无有效作品、4,016 条仅制作信息、
  2,273 条过短或无法判定。

## 3. P4：向量索引（已完成）

### 3.1 最终环境

- GPU：NVIDIA RTX 3060 Laptop，6 GB 显存，CUDA 12.4 可用。
- 系统 Python（`E:\python\Lib\site-packages`）：`chromadb 1.5.9`、
  `pytest 9.0.3`、`torch 2.6.0+cu124`、`transformers 5.12.1`、
  `huggingface-hub 1.22.0`、`sentencepiece 0.2.2`、`numpy 2.2.6`。
- 项目 `.venv`：`--system-site-packages`，仅保留 `pip`，业务包全部从全局继承。
- 模型：`BAAI/bge-m3`，revision `5617a9f61b028005a4858fdac845db406aefb181`，
  1024 维，CLS pooling，L2 归一化，`max_length=1024`，GPU fp16。
- 模型缓存：`storage/model_cache/bge-m3/`，权重 2,271,145,830 字节。

### 3.2 实施结果

- [x] 重建 `.venv` 并启用 `--system-site-packages`，业务依赖迁移到全局 Python。
- [x] 在全局 Python 安装并固定 `chromadb==1.5.9`；`.venv` 不再包含业务包。
- [x] 通过镜像/直连下载 BGE-M3，并写入 `plotseeker-model.json`。
- [x] 实现 `src/embeddings/`：
  - `base.py`：Embedder 协议；
  - `bge_m3.py`：加载、CLS pooling、L2 归一化、设备/fp16/批大小；
  - `fake.py`：测试用确定性假向量；
  - `chroma_store.py`：持久化客户端、元数据映射、孤儿 HNSW 清理；
  - `indexer.py`：抽样、批次写入、checkpoint、重建、重开校验、报告。
- [x] 实现 `scripts/download_model.py` 与 `scripts/build_index.py`。
- [x] 确定性抽样 10,000 篇：作品 4,000 / 角色 3,000 / 章节 3,000。
- [x] 三个 collection：`p4_subject_docs`、`p4_character_docs`、
  `p4_episode_docs`；cosine distance；HNSW `ef_construction=200`、
  `max_neighbors=64`、`ef_search=200`。
- [x] Chroma metadata 只放标量，列表字段转 JSON 字符串。
- [x] checkpoint 记录源文件 SHA256、manifest SHA256、模型身份、维度、
  抽样指纹、批大小、每批已处理 ID；每批原子更新。
- [x] 断点恢复时逐条核对 checkpoint 中 ID 是否仍在索引中；源文档/模型/
  抽样变化会拒绝复用旧索引。
- [x] `--rebuild` 显式删除三个目标 collection；已有完整索引时默认拒绝覆盖。
- [x] 重建时清理 Chroma 删除 collection 后遗留的 HNSW 目录。
- [x] 产出 `storage/vector_store/p4_report.json` 与 `p4_checkpoint.json`。
- [x] 编写 `tests/test_vector_index.py`；P4 相关测试通过，全套 74 个测试通过。
- [x] 真实 10,000 条文档编码、写入、重开并检索。
- [x] 实际测试一次 10,000 条中断恢复与一次 10,000 条显式重建。

### 3.3 实测指标

| 指标 | 结果 |
| --- | --- |
| 文档数 | 10,000（4,000 / 3,000 / 3,000） |
| 完整重建耗时 | 319.7 秒 |
| 编码耗时 / 速度 | 187.2 秒 / 53.41 docs/s |
| 写入 Chroma 耗时 | 104.3 秒 |
| 总吞吐 | 31.28 docs/s |
| 进程峰值内存 | 3,682.3 MiB |
| GPU 峰值分配 / 保留 | 1,459.4 / 1,746.0 MiB |
| 索引体积 | 113.2 MiB |
| 重开校验 | 10,000/10,000 IDs 通过 |
| Sanity | 30/30 hit@1，30/30 hit@5 |
| 查询延迟 | P50 34.97 ms，P95 72.94 ms |
| 断点恢复 | 20 批后停止，恢复后处理剩余 606 批 |
| 重建 | 旧索引 config 与计数写入报告；孤儿 HNSW 目录 3 个被清理 |

### 3.4 P4 验收标准

- [x] 10,000 条真实文档完成编码、写入 Chroma，并能在重新打开索引后检索。
- [x] 集合条数与抽样文档 ID 可以核对。
- [x] 断点恢复至少实际测试一次；重建至少实际测试一次。
- [x] 记录编码速度、峰值内存/显存、索引体积、查询 P50/P95。
- [x] 查询向量与入库向量使用同一模型、同一池化方式和同一归一化规则。
- [x] 小规模检索结果可定位到 P3 文档与原始实体。
- [x] 未进行百万级全量编码；P4 只做 10,000 篇抽样。

### 3.5 P4 遗留说明

- HuggingFace 镜像会重定向到 `huggingface.co`；当前下载脚本先试镜像，
  元数据失败后自动回退直连。复现时优先使用脚本，不要手工拼接 URL。
- Chroma 1.5.9 的 Rust SQLite 迁移依赖可写临时目录；系统 TEMP 在沙箱中
  可能 ACL 受限，`chroma_store.py` 已将 `TEMP`/`TMP`/`SQLITE_TMPDIR`
  指向 `storage/vector_store/_tmp`。
- HNSW 是近似检索；`ef_search=200` 在 10k 规模下 sanity 已全部命中，
  后续扩容时需重新评估查询延迟与召回。
- 模型权重、Chroma 索引和 checkpoint 均为可重建产物，不进入 Git。
- P7 扩容前需要结合 P5/P6 的召回与延迟结果决定是否扩大抽样或索引全量。

## 4. P5：关键词检索与混合召回（已完成）

### 4.1 实施结果

- [x] 从 P4 checkpoint 读取同一批 10,000 个 `doc_id`，保证关键词/向量同范围。
- [x] 独立 FTS5 数据库 `storage/fts/p5_keyword_10k.db`，不修改 P2 主库。
- [x] 同时建立 `fts_jieba` 与 `fts_bigram`，比较后默认 jieba。
- [x] 统一预处理：NFKC、OpenCC 繁转简、标点归一、停用词、查询 token 上限 32。
- [x] 关键词、向量、hybrid 三模式与 RRF（k=60）。
- [x] 角色/章节命中映射到作品；主作品权重 1.0，额外角色作品最多 4 个、权重 0.35。
- [x] 证据包含 doc_id、来源、排名、片段、关系路径与贡献值。
- [x] CLI Demo：`python scripts/search.py "查询" --mode hybrid`。
- [x] 关键词/混合检索测试：`tests/test_keyword_retrieval.py`、`tests/test_hybrid_retrieval.py`。
- [x] 负样例置信度：关键词覆盖率阈值 0.75；低置信结果带 warning，不伪装成确定答案。

### 4.2 实测

- 关键词索引：10,000 文档，60.3 MiB，构建 38.4 秒。
- 分词比较（479 条名称/片段查询）：
  - jieba Recall@5 0.9395，MRR@10 0.9184；
  - bigram Recall@5 0.8935，MRR@10 0.8769。
- 关键词 P50/P95：58.76 / 85.42 ms。
- 向量 P50/P95：61.89 / 82.02 ms。
- hybrid P50/P95：120.53 / 153.07 ms。

## 5. P6：评测体系（已完成）

- [x] 生成 `benchmarks/queries.jsonl`：128 条，118 可回答、10 合成负样例。
- [x] 覆盖名称、别名、作品简介、角色简介、章节剧情、跨类型、繁简、碎片、负样例。
- [x] 三种基线：keyword / vector / hybrid。
- [x] 指标：Recall@5、MRR@10、Top-1、P50/P95、索引体积、峰值内存。
- [x] 每条指标可追溯到 `storage/evaluation/p6_report.json` 的 query_id。
- [x] 失败记录：`benchmarks/failures.md`，含失败原因与归因。
- [x] 生成 `docs/evaluation.md` 与复现命令。
- [x] 如实记录 hybrid Recall 未超过最佳单路，但 MRR/Top-1 有提升。
- [x] 负样例区分 Any FP 与 Confident FP；置信阈值在开发集校准。

实测（118 可回答）：

| 模式 | Recall@5 | MRR@10 | Top-1 | P50 | P95 | Confident FP (10 负样例) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| keyword | 1.0 | 0.9890 | 0.9831 | 58.76 ms | 85.42 ms | 0 |
| vector | 0.9915 | 0.9703 | 0.9576 | 61.89 ms | 82.02 ms | 0 |
| hybrid | 1.0 | 0.9958 | 0.9915 | 120.53 ms | 153.07 ms | 0 |

限制：查询由已索引文本生成，不是独立人工查询集；指标可能偏乐观。

## 6. P7：资源扩容与检索优化（仅完成预算分析）

- [x] 记录 10k 实测吞吐、显存、索引体积和查询延迟。
- [x] 编写 `docs/scaling.md`：全量线性估算、磁盘量级、扩容前检查项。
- [ ] 未执行全量编码。
- [ ] 未做文本级去重和真实 token 总量统计。
- [ ] 未做 50k/100k 扩容实验。
- [ ] 未引入 reranker。
- [ ] 未做文档更新/删除/索引版本管理。

**原因：** 10k 指标不能直接外推百万级；需要人工确认资源预算和独立评测集。

## 7. P8：FastAPI 服务与完整文档（已完成）

- [x] `GET /health`
- [x] `GET /search?q=...&top_k=10&mode=hybrid&candidate_k=50`
- [x] 空查询、超长查询、非法 mode/top_k/candidate_k 校验。
- [x] 503 索引/模型缺失、504 超时、500 内部错误。
- [x] 模型与索引进程级复用；首次语义查询承担模型加载。
- [x] API 合约测试：`tests/test_api.py`。
- [x] 真实服务启动并通过 `/health`、keyword、hybrid 查询。
- [x] 编写 `docs/architecture.md`。
- [x] README 补充启动命令、API 示例与评测命令。

限制：无认证、无限流、无 CORS；超时不会取消后台模型线程；多 worker 会重复加载模型。

## 8. P9：Agent 扩展（实验性完成）

- [x] `search_subjects`、`search_characters`、`search_episodes`
- [x] `get_related_subjects`
- [x] `ask_user_for_clarification`
- [x] CLI：`python scripts/agent.py`
- [x] 工具规格、确定性路由、最多 4 步预算
- [x] 对比启用/禁用 Agent 的检索质量、工具调用次数与延迟
- [x] 报告：`docs/agent.md`、`docs/agent-evaluation.md`、
  `storage/evaluation/p9_agent_report.json`
- [x] LLM CLI：`scripts/agent_llm.py`，OpenAI-compatible tool calling
- [x] 本地配置：`config/agent.json`（已 gitignore）、`config/agent.example.json`
- [x] LLM 配置/限制文档：`docs/agent-llm.md`
- [ ] 使用真实 API Key 做端到端人工验收
- [ ] 验证目标 LLM 服务的 function calling 兼容性

### 实测结论

| 变体 | R@5 | MRR@10 | Top-1 | 可回答澄清率 | 负样例澄清率 | 平均工具调用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.0 | 0.9958 | 0.9915 | 0.0 | 0.0 | 0.0 |
| agent | 1.0 | 0.9958 | 0.9915 | 0.0 | 0.0 | 1.16 |
| agent_clarify_low | 1.0 | 0.9958 | 0.9915 | 0.0 | 1.0 | 1.24 |

- 默认 Agent 没有提升 Recall@5，却增加工具调用，**不建议默认启用**。
- 保守澄清模式在开发集上保持回答质量，并把 10/10 合成负样例转为澄清；
  可作为低置信/不可回答查询的可选策略。
- 开发集仍是自动生成，不能证明真实用户场景的净收益；需要独立人工查询集。
- 未接入 P8 API；当前只提供 CLI 和评测脚本。

## 9. 当前环境注意事项

- 业务依赖已迁移到系统 Python `E:\python\Lib\site-packages`；
  `.venv` 使用 `--system-site-packages`，仅含 `pip`。
  `requirements.lock.txt` 记录全局包版本和系统 ML 包版本。
- 必须在未激活 `.venv` 时执行全局 `pip install`；在 `.venv` 内安装会把包
  重新写回项目目录。
- `onnxruntime-gpu 1.26.0` 提供 `onnxruntime` 模块；chromadb 使用
  `--no-deps` 安装，不能安装 CPU 版 onnxruntime 覆盖 GPU 版。
- 当前沙箱中的 `.venv` 进程可能带 Low Mandatory Level，写入
  `data/processed` 可能报 `WinError 5`；P3 全量文档使用系统 Python 构建。
  P4 已将 Chroma/SQLite 临时目录固定到 `storage/vector_store/_tmp`。
- HuggingFace 镜像会重定向到 `huggingface.co`；`download_model.py` 会自动
  回退直连，模型缓存放在 `storage/model_cache/`。
- Chroma 删除 collection 后会遗留 HNSW 目录，`--rebuild` 已实现清理。
- 原始数据、SQLite、文档 JSONL、模型权重和索引均不应提交到 Git。
- 项目根目录的 `pip-*` 临时目录已清理；`.venv` 只保留 pip，业务包在全局 Python。
- jieba 缓存已移到 `storage/cache/jieba/jieba.cache`，不再生成根目录 `jieba.cache`。
- pytest 临时目录改为 `scratch/pytest-tmp`，不再在根目录生成 `pytest-of-*`。

## 10. 建议的下一步顺序

1. 人工验收 P5 关键词/混合检索：查询结果、置信度、负样例表现。
2. 人工验收 P6 评测集与 `docs/evaluation.md`；重点确认开发集是否过于简单。
3. 人工验收 P8 API：启动、错误码、超时、证据字段与资源占用。
4. 补充独立人工查询集后再决定 P7 是否扩容。
5. 用独立人工查询验证 P9 的 `clarify_on_low` 是否真的改善歧义处理。
6. 不在未确认前执行全量 84.6 万篇向量编码。

未确认信息统一记录在 `docs/unconfirmed.md`。
