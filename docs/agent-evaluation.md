# PlotSeeker P9 Agent 对比评测

> 本报告由 `scripts/evaluate_agent.py` 自动生成，请勿手工编辑。

- 生成时间：2026-10-21T04:36:40+00:00
- 执行命令：`python scripts/evaluate_agent.py`
- 查询集：`E:\PlotSeeker\benchmarks\queries.jsonl`，128 条
- 基线：直接 hybrid 检索（无 Agent 编排）。
- Agent 版本：确定性工具编排，不调用 LLM。

## 1. 总体指标

| 变体 | R@5 | MRR@10 | Top-1 | 可回答澄清率 | P50 | P95 | 平均工具调用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.0 | 0.9958 | 0.9915 | 0.0 | 185.75 ms | 273.22 ms | 0.0 |
| agent | 1.0 | 0.9958 | 0.9915 | 0.0 | 147.73 ms | 220.39 ms | 1.16 |
| agent_clarify_low | 1.0 | 0.9958 | 0.9915 | 0.0 | 142.23 ms | 199.57 ms | 1.24 |

## 2. 不可回答查询

| 变体 | 负样例澄清率 | Any FP | Confident FP |
| --- | ---: | ---: | ---: |
| baseline | 0.0 | 10 | 0 |
| agent | 0.0 | 10 | 0 |
| agent_clarify_low | 1.0 | 0 | 0 |

## 3. 结论

默认 Agent Recall@5 与直接 hybrid 持平，且平均工具调用 1.16 次，未证明默认启用有质量收益。 保守澄清模式在开发集上保持 Recall@5，可回答澄清率 0.00%，负样例澄清率 100.00%。该模式可作为低置信/不可回答查询的可选策略，但合成负样例不能替代真实用户验证。

## 4. 失败样例

### baseline（10）

| query_id | 查询 | 类型 | 原因 | target_rank | 工具 |
| --- | --- | --- | --- | ---: | --- |
| q119 | 找一部主角叫 Zzyzx Qwerty 的动画 | unanswerable | weak_false_positive | - | hybrid |
| q120 | 有一部作品，剧情是紫色大象在月球开便利店 | unanswerable | weak_false_positive | - | hybrid |
| q121 | 角色名字是赵不存在，能力是控制时间 | unanswerable | weak_false_positive | - | hybrid |
| q122 | 想找一部关于会说话的蓝色仙人掌拯救银河系的漫画 | unanswerable | weak_false_positive | - | hybrid |
| q123 | 有一部作品，主角是一只穿西装的仓鼠侦探 | unanswerable | weak_false_positive | - | hybrid |
| q124 | 记得一部动画，反派用微波炉统治世界 | unanswerable | weak_false_positive | - | hybrid |
| q125 | 找一部作品，里面有叫 Qwertyuiop 的魔法学院 | unanswerable | weak_false_positive | - | hybrid |
| q126 | 有一部关于企鹅参加奥运会的青春故事 | unanswerable | weak_false_positive | - | hybrid |

### agent（10）

| query_id | 查询 | 类型 | 原因 | target_rank | 工具 |
| --- | --- | --- | --- | ---: | --- |
| q119 | 找一部主角叫 Zzyzx Qwerty 的动画 | unanswerable | weak_false_positive | - | characters -> episodes -> subjects |
| q120 | 有一部作品，剧情是紫色大象在月球开便利店 | unanswerable | weak_false_positive | - | subjects -> characters -> episodes |
| q121 | 角色名字是赵不存在，能力是控制时间 | unanswerable | weak_false_positive | - | characters -> episodes -> subjects |
| q122 | 想找一部关于会说话的蓝色仙人掌拯救银河系的漫画 | unanswerable | weak_false_positive | - | subjects -> characters -> episodes |
| q123 | 有一部作品，主角是一只穿西装的仓鼠侦探 | unanswerable | weak_false_positive | - | characters -> episodes -> subjects |
| q124 | 记得一部动画，反派用微波炉统治世界 | unanswerable | weak_false_positive | - | subjects -> characters -> episodes |
| q125 | 找一部作品，里面有叫 Qwertyuiop 的魔法学院 | unanswerable | weak_false_positive | - | subjects -> characters -> episodes |
| q126 | 有一部关于企鹅参加奥运会的青春故事 | unanswerable | weak_false_positive | - | subjects -> characters -> episodes |

### agent_clarify_low（10）

| query_id | 查询 | 类型 | 原因 | target_rank | 工具 |
| --- | --- | --- | --- | ---: | --- |
| q119 | 找一部主角叫 Zzyzx Qwerty 的动画 | unanswerable | correctly_clarified | - | characters -> episodes -> subjects |
| q120 | 有一部作品，剧情是紫色大象在月球开便利店 | unanswerable | correctly_clarified | - | subjects -> characters -> episodes |
| q121 | 角色名字是赵不存在，能力是控制时间 | unanswerable | correctly_clarified | - | characters -> episodes -> subjects |
| q122 | 想找一部关于会说话的蓝色仙人掌拯救银河系的漫画 | unanswerable | correctly_clarified | - | subjects -> characters -> episodes |
| q123 | 有一部作品，主角是一只穿西装的仓鼠侦探 | unanswerable | correctly_clarified | - | characters -> episodes -> subjects |
| q124 | 记得一部动画，反派用微波炉统治世界 | unanswerable | correctly_clarified | - | subjects -> characters -> episodes |
| q125 | 找一部作品，里面有叫 Qwertyuiop 的魔法学院 | unanswerable | correctly_clarified | - | subjects -> characters -> episodes |
| q126 | 有一部关于企鹅参加奥运会的青春故事 | unanswerable | correctly_clarified | - | subjects -> characters -> episodes |

## 5. 限制

- 评测集由已索引文本生成，不是独立人工查询集。
- Agent 不调用 LLM，工具调用成本按工具次数和额外延迟衡量，不含 token 成本。
- 低置信候选默认仍返回；`agent_clarify_low` 才在低置信时直接追问。
- 单进程测量，首次请求包含模型和索引加载时间。

