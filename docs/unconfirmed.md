# PlotSeeker 未确认信息与待人工检查项

> 生成时间：2026-10-21
> 用途：记录自动施工过程中做出的假设、临时参数、未完成阶段和需要人工确认的结论。
> 原则：以下内容不应被当作最终验收结论；确认前不要据此启动全量编码或对外承诺指标。

## 1. 当前完成范围

| 阶段 | 状态 | 说明 |
| --- | --- | --- |
| P0-P4 | 已完成 | 环境、审计、SQLite、P3 文档、P4 10k 向量索引 |
| P5 | 已完成 | FTS5 关键词索引、混合召回、关系扩展、证据、CLI |
| P6 | 已完成（开发集） | 128 条自动生成查询、三种基线、评测报告 |
| P7 | 仅预算分析 | 未执行全量/扩容编码，见 `docs/scaling.md` |
| P8 | 已完成 | FastAPI `/health`、`/search`、API 测试、架构文档 |
| P9 | 实验性完成 | 确定性 Agent 默认不启用；另提供 OpenAI-compatible LLM CLI |

## 2. 需要优先确认的结论

### 2.1 P6 评测集不是独立人工查询集

- `benchmarks/queries.jsonl` 的查询由已索引文档文本生成。
- 名称、别名、简介、章节、片段查询可能与索引文本共享措辞。
- 因此 keyword/hybrid 的 Recall@5=1.0 可能偏乐观。
- 当前结果只能证明链路可用和可复现，不能证明真实模糊查询达到该水平。
- 需要人工补充独立查询集后重新评测。

### 2.2 P6 可接受答案只取主作品

- 角色可能关联多个作品；当前标签只使用 P3 文档的 `subject_id`。
- 如果用户目标是对应角色的其他作品，当前评测会判为错误。
- 需要确认是否扩展 `acceptable_subject_ids`，以及如何定义多作品可接受集合。

### 2.3 负样例与置信度阈值

- 10 条负样例是人工合成文本，可能偶然命中真实词汇。
- 关键词置信度阈值 `0.75` 基于关键词 token 覆盖率，在开发集上校准。
- 当前结果：
  - Any false positive = 10/10（向量总会返回近邻）；
  - Confident false positive = 0/10。
- 低置信结果仍会返回候选，但带 `warnings`；需要确认产品是否应直接隐藏低置信候选。
- 阈值没有独立验证，可能需要按真实查询调整。
- 实际例子：`失去记忆的魔王在现代学校` 的向量最高相似度约 0.539，
  但关键词覆盖率只有 0.4，因此系统标为 `low` 并带 warning。
  这表示置信度策略偏保守，不表示检索一定错误。

### 2.4 RRF 与关系扩展参数

当前固定值：

- RRF `k=60`。
- 每个 subject 在每个通道只取最佳文档贡献，避免文档数量堆叠。
- 角色主作品权重 `1.0`。
- 角色额外作品最多 `4` 个，权重 `0.35`。
- 每个候选最多 `6` 条证据，每个通道最多 `3` 条。
- `candidate_k=50`。

这些参数未做系统消融实验。需要确认是否接受当前默认值，或进入 P7 做加权/消融。

### 2.5 关键词预处理

- 默认 jieba；bigram 仅作为对比表保留。
- 查询侧过滤停用词和模板词，去重并限制 32 个 token。
- 分词比较结果来自 479 条名称/片段查询，不是人工查询集。
- 需要确认停用词表和 32 token 上限是否影响真实短查询/长剧情查询。

## 3. P4 冻结与既有风险

- P4 `p4_report.json`、`p4_checkpoint.json` 和正常构建逻辑保持冻结。
- 为修复测试中的中断恢复客户端清理，`src/embeddings/indexer.py` 的异常停止分支
  增加了 Chroma 缓存清理；未改变正常构建、抽样或索引格式，也未重建 P4 索引。
- 小于等于 100 篇的测试构建会使用 HNSW `sync_threshold=1`，避免 Chroma 未刷盘
  导致“重新打开后找不到段”；现有 10k 索引配置未变。
- 当前 P4 sanity 30/30 使用建库文档前 200 字，只能证明索引/元数据/查询链路，
  不是独立质量评测。
- P4 10k 构建指标不能外推百万级。
- P4 checkpoint 的 `processed_ids` 用于 P5 抽样，若 P4 重建，P5 关键词索引需重建。
- Chroma 1.5.9 删除 collection 会遗留 HNSW 目录，`--rebuild` 已清理。

## 4. 未执行或未验证的内容

### 4.1 P7 全量扩容

未执行：

- 文本级精确/近似去重。
- 真实 token 总量和 1024 token 截断比例。
- 50k/100k 扩容实验。
- 全量关键词索引与全量向量索引。
- reranker、热度/年份先验、文档删除/更新机制。
- 全量规模断点恢复和失败回滚演练。

`docs/scaling.md` 中的全量体积/时间是线性估算，未验证，不得作为承诺。

### 4.2 P9 Agent

已实现实验性单 Agent，不调用 LLM：

- `search_subjects`
- `search_characters`
- `search_episodes`
- `get_related_subjects`
- `ask_user_for_clarification`

开发集对比结论：

- 默认 Agent Recall@5 与直接 hybrid 持平，但平均工具调用 1.16 次；不建议默认启用。
- `clarify_on_low=True` 在开发集上保持回答质量，把 10/10 合成负样例转为澄清；
  可作为低置信/不可回答查询的可选策略。
- 延迟测量受缓存和调用顺序影响，不把 Agent 的 P50 较低解释为速度提升。
- 未接入 P8 API；`subject_relations.relation_type` 仍为原始整数码值。
- 真实短查询可能被保守澄清模式过度追问；需要独立人工查询验证。

### 4.3 LLM Agent CLI

已实现：

- `scripts/agent_llm.py`：OpenAI-compatible Chat Completions + tool calling。
- `config/agent.json`：本地 API Key、Base URL、模型名；已被 `.gitignore` 排除。
- `config/agent.example.json`：可提交模板。
- `docs/agent-llm.md`：配置、命令和限制。

未验证/需要人工确认：

- 没有真实 API Key，因此未调用外部 LLM 做端到端验收。
- 默认示例指向 DeepSeek `deepseek-chat`；其他 OpenAI-compatible 服务的
  `tools` / `tool_choice` / `max_tokens` 兼容性未验证。
- 模型必须支持 function calling；不支持时只会得到 `llm_did_not_call_tools` warning。
- 没有流式输出、自动重试退避、成本预算、多轮会话和并发控制。
- LLM 只能通过本地工具取证据；但最终自然语言回答仍由模型生成，需要人工抽查
  是否忠实于工具结果。
- 未接入 P8 API；API Key 不通过 `/search` 传递。

## 5. P8 API 未确认项

当前实现：

- `GET /health`
- `GET /search?q=...&top_k=10&mode=hybrid&candidate_k=50`
- 422 / 503 / 504 / 500 错误映射。
- 进程级模型与索引复用。
- 默认超时 30 秒，可用 `PLOTSEEKER_SEARCH_TIMEOUT_SECONDS` 调整。

未确认/未实现：

- 无认证、无限流、无 CORS、无持久化请求日志。
- 超时只停止请求等待，后台模型线程可能继续执行。
- 多 worker 会重复加载模型和索引；当前建议单 worker。
- 健康检查会检查文件存在和 P4 checkpoint 是否为 complete，但不会实际执行检索。
- 首次 hybrid/vector 请求有约 13-16 秒冷启动。
- 低置信候选是否应返回给前端，需要产品确认。

## 6. 环境与依赖未确认项

- 依赖已迁移到系统 Python `E:\python\Lib\site-packages`；`.venv` 只含 `pip`。
- `onnxruntime-gpu 1.26.0` 提供 `onnxruntime` 模块，但 `pip check` 仍提示
  `chromadb requires onnxruntime`。这是元数据不一致，不影响当前显式传入向量的用法。
- `fastmcp`/`mcp` 与 `httpx 0.27.0` 的冲突在本次改动前已存在，未处理。
- 如果在新机器上重装，需要先安装 CUDA PyTorch，再使用全局锁文件；
  不要在 `.venv` 内安装业务包，否则会重新写入项目目录。

## 7. 产物与报告

| 产物 | 路径 |
| --- | --- |
| P4 索引报告 | `storage/vector_store/p4_report.json` |
| P4 checkpoint | `storage/vector_store/p4_checkpoint.json` |
| P5 关键词库 | `storage/fts/p5_keyword_10k.db` |
| P5 构建报告 | `storage/fts/p5_keyword_report.json` |
| P6 评测集 | `benchmarks/queries.jsonl` |
| P6 评测明细 | `storage/evaluation/p6_report.json` |
| P6 评测报告 | `docs/evaluation.md` |
| P6 失败记录 | `benchmarks/failures.md` |
| P8 架构文档 | `docs/architecture.md` |
| P9 Agent 报告 | `storage/evaluation/p9_agent_report.json` |
| P9 Agent 对比 | `docs/agent-evaluation.md` |
| P9 Agent 说明 | `docs/agent.md` |
| LLM Agent CLI 说明 | `docs/agent-llm.md` |
| LLM 配置模板 | `config/agent.example.json` |
| P7 扩容分析 | `docs/scaling.md` |

## 8. 本轮验证记录

- `python -m pytest`：74 个测试通过。
- P5 关键词索引：10,000 文档，60.3 MiB，默认 jieba。
- P6 最终评测（118 可回答 / 10 负样例）：
  - keyword Recall@5=1.0，MRR@10=0.9890，Top-1=0.9831；
  - vector Recall@5=0.9915，MRR@10=0.9703，Top-1=0.9576；
  - hybrid Recall@5=1.0，MRR@10=0.9958，Top-1=0.9915；
  - Confident false positive = 0/10。
- P8 真实服务启动成功；`/health`、keyword、hybrid 请求均返回 200。
- P8 首次 hybrid 请求约 15.9 秒（含 BGE-M3 加载），之后复用进程内模型。
- P9 Agent 已实际运行并生成 `docs/agent-evaluation.md`；默认模式未显示质量提升。
- P9 默认 Agent 与直接 hybrid 质量持平，平均 1.16 次工具调用；不建议默认启用。
- P9 `agent_clarify_low` 把 10/10 合成负样例转为澄清，可回答澄清率 0%。
- P4 索引与 checkpoint 未被本次 P5-P9 施工重建。

## 9. 建议人工检查顺序

1. 阅读 `docs/evaluation.md`，确认开发集是否过于简单。
2. 运行 `python scripts/search.py` 抽查 10-20 条真实模糊查询。
3. 检查低置信候选和负样例是否符合产品预期。
4. 检查 P8 API 的错误码、证据字段和冷启动体验。
5. 决定是否补充独立人工查询集。
6. 只有确认评测集和资源预算后，才启动 P7 扩容实验。
7. 验证 P9 Agent 的 `clarify_on_low` 是否改善真实歧义查询。

## 10. 本轮未修改的既有产物

- `data/raw/` 未修改。
- `storage/plotseeker.db` 未重建。
- `storage/vector_store/` 的 P4 索引和 checkpoint 未重建。
- P4 测试和 P4 代码保持可用；P5 测试仅新增文件。
