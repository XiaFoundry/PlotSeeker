# PlotSeeker P9 Agent 说明

> 更新时间：2026-10-21
> 状态：实验性实现，默认不启用；保守澄清模式可选。
> 版本：`p9-v1`，确定性工具 Agent；另提供 OpenAI-compatible LLM CLI（`docs/agent-llm.md`）。

## 1. 目标

Agent 不替代底层召回，而是判断：

- 应该查作品、角色还是章节；
- 低置信或无结果时是否换检索入口；
- 用户是否在问“相关作品”；
- 证据不足时是否追问补充线索。

## 2. 工具

| 工具 | 输入 | 输出 | 数据来源 |
| --- | --- | --- | --- |
| `search_subjects` | `query`, `top_k` | 作品候选 + 证据 | P5 hybrid |
| `search_characters` | `query`, `top_k` | 角色文档 + 所属作品 | P5 FTS/向量 |
| `search_episodes` | `query`, `top_k` | 章节文档 + 所属作品 | P5 FTS/向量 |
| `get_related_subjects` | `subject_id`, `limit` | 相关作品 + 原始 `relation_type` | P2 `subject_relations` |
| `ask_user_for_clarification` | `question`, `options`, `reason` | 结构化追问 | Agent 策略 |

所有工具返回真实 `doc_id`、`subject_id` 和文本片段；缺失的 subject 行会被跳过，
不会生成不存在的实体。`relation_type` 保持 P2 原始整数码值，未擅自解释业务含义。

## 3. 策略

1. 根据关键词路由：
   - 角色/人物/主角/能力 → `search_characters`
   - 第 X 话/章节/某一话 → `search_episodes`
   - 其他 → `search_subjects`
2. 置信度不足时，最多再尝试另外两个入口（受 `max_steps` 限制）。
3. 查询包含“相关/类似/续作/同系列”等词且已有作品命中时，调用
   `get_related_subjects`。
4. 没有任何候选时调用澄清工具。
5. 两种低置信策略：
   - 默认：返回低置信候选 + `low_confidence_candidates` warning；
   - `clarify_on_low=True`：低置信时直接追问，不返回低置信候选。

## 4. 命令

确定性 Agent：

```powershell
# 默认模式
python scripts/agent.py "失去记忆的魔王在现代学校"

# JSON 输出
python scripts/agent.py "时间循环学园有什么相关作品" --json

# 低置信时追问
python scripts/agent.py "妹妹用脚抵住门" --clarify-low

# 查看工具规格
python scripts/agent.py --tools
```

LLM 版本（需要 API Key）：

```powershell
python scripts/agent_llm.py "失去记忆的魔王在现代学校"
```

配置与限制见 [`docs/agent-llm.md`](agent-llm.md)。

## 5. 对比评测

```powershell
python scripts/evaluate_agent.py
```

输出：

- `storage/evaluation/p9_agent_report.json`
- `docs/agent-evaluation.md`

评测集复用 P6 的 128 条开发查询。基线为直接 hybrid 检索。

### 实测结果

| 变体 | R@5 | MRR@10 | Top-1 | 可回答澄清率 | 负样例澄清率 | 平均工具调用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.0 | 0.9958 | 0.9915 | 0.0 | 0.0 | 0.0 |
| agent | 1.0 | 0.9958 | 0.9915 | 0.0 | 0.0 | 1.16 |
| agent_clarify_low | 1.0 | 0.9958 | 0.9915 | 0.0 | 1.0 | 1.24 |

- 默认 Agent 没有提升 Recall@5，却增加了工具调用，**不建议默认启用**。
- `agent_clarify_low` 在开发集上保持回答质量，并把 10/10 合成负样例转为澄清，
  可作为“低置信/不可回答查询”的可选策略。
- 延迟受缓存和调用顺序影响，本报告不把 Agent 的 P50 较低解释为速度提升。
- 开发集由索引文本生成，不能替代真实用户查询；合成负样例也不能证明真实歧义处理能力。

## 6. 局限与待确认

- 没有 LLM 规划，路由依赖正则和置信度启发式。
- 角色/章节工具只按主作品映射，角色多作品关系未作为最终候选展开。
- `subject_relations` 的 `relation_type` 未解释，当前只原样返回。
- 澄清策略可能在真实短查询上过度追问；默认关闭低置信追问。
- 未接入 P8 API；当前仅 CLI 和评测脚本。
- 没有独立人工查询集证明 Agent 的净收益。
