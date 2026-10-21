# PlotSeeker P6 检索评测报告

> 本报告由 `scripts/evaluate.py` 自动生成，请勿手工编辑。

- 生成时间：2026-10-21T03:31:28+00:00
- 执行命令：`python scripts/evaluate.py`
- 查询集：`E:\PlotSeeker\benchmarks\queries.jsonl`，128 条（可回答 118，不可回答 10）
- 索引范围：P4 10,000 篇抽样文档 + 同批 FTS5 关键词索引
- P4 config_id：`618c2efa859f16b9675c864b5ffe0912538939df5aeb26920b747429f4e1d0c1`

## 1. 总体指标

| 模式 | 查询数 | Recall@5 | MRR@10 | Top-1 | P50 (ms) | P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| keyword | 118 | 1.0 | 0.989 | 0.9831 | 58.76 | 85.42 |
| vector | 118 | 0.9915 | 0.9703 | 0.9576 | 61.89 | 82.02 |
| hybrid | 118 | 1.0 | 0.9958 | 0.9915 | 120.53 | 153.07 |

**结论：** 混合检索 Recall@5=1.0 与最佳单路持平。 MRR 或 Top-1 有提升，但 Recall@5 未提升。

## 2. 按线索类型

| 线索类型 | 查询数 | keyword R@5 | keyword MRR | keyword Top1 | vector R@5 | vector MRR | vector Top1 | hybrid R@5 | hybrid MRR | hybrid Top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| alias | 10 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| character_summary | 20 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9417 | 0.9 | 1.0 | 1.0 | 1.0 |
| cross_type_character | 10 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| cross_type_episode | 10 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| episode_plot | 20 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9667 | 0.95 | 1.0 | 1.0 | 1.0 |
| fragment | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 0.8667 | 0.8 | 1.0 | 1.0 | 1.0 |
| name | 18 | 1.0 | 0.9278 | 0.8889 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| subject_summary | 20 | 1.0 | 1.0 | 1.0 | 0.95 | 0.95 | 0.95 | 1.0 | 0.975 | 0.95 |
| traditional | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

## 3. 不可回答查询

| 模式 | Any false positive | Confident false positive | Any FP rate | Confident FP rate |
| --- | ---: | ---: | ---: | ---: |
| keyword | 10 | 0 | 1.0 | 0.0 |
| vector | 10 | 0 | 1.0 | 0.0 |
| hybrid | 10 | 0 | 1.0 | 0.0 |

## 4. 资源与延迟

| 项目 | 数值 |
| --- | ---: |
| 关键词索引体积 | 60.3 MiB |
| P4 向量索引体积 | 113.2 MiB |
| 进程峰值内存 | 3682.0 MiB |
| 模型 | BAAI/bge-m3 / 5617a9f61b028005a4858fdac845db406aefb181 |
| 设备 | cuda |

## 5. 失败案例

完整失败记录：`E:\PlotSeeker\benchmarks\failures.md`

### keyword 失败样例（10 条）
| query_id | 查询 | 线索类型 | 失败原因 | 归因 | target_rank | top_ids |
| --- | --- | --- | --- | --- | ---: | --- |
| q119 | 找一部主角叫 Zzyzx Qwerty 的动画 | unanswerable | weak_false_positive | 问题歧义或负样例 | - | 10456, 475779, 320423 |
| q120 | 有一部作品，剧情是紫色大象在月球开便利店 | unanswerable | weak_false_positive | 问题歧义或负样例 | - | 229514, 323753, 112984 |
| q121 | 角色名字是赵不存在，能力是控制时间 | unanswerable | weak_false_positive | 问题歧义或负样例 | - | 76889, 159725, 283197 |
| q122 | 想找一部关于会说话的蓝色仙人掌拯救银河系的漫画 | unanswerable | weak_false_positive | 问题歧义或负样例 | - | 73044, 306742, 510266 |
| q123 | 有一部作品，主角是一只穿西装的仓鼠侦探 | unanswerable | weak_false_positive | 问题歧义或负样例 | - | 522276, 475779, 899 |
| q124 | 记得一部动画，反派用微波炉统治世界 | unanswerable | weak_false_positive | 问题歧义或负样例 | - | 435131, 61339, 265831 |
| q125 | 找一部作品，里面有叫 Qwertyuiop 的魔法学院 | unanswerable | weak_false_positive | 问题歧义或负样例 | - | 345823, 410928, 661805 |
| q126 | 有一部关于企鹅参加奥运会的青春故事 | unanswerable | weak_false_positive | 问题歧义或负样例 | - | 331946, 490245, 361296 |

### vector 失败样例（11 条）
| query_id | 查询 | 线索类型 | 失败原因 | 归因 | target_rank | top_ids |
| --- | --- | --- | --- | --- | ---: | --- |
| q011 | 找一部作品，剧情大概是：ある夏の日、偶然引き寄せられた小説家と謎の美女。そしてそこから二人が紡ぎ出す、狂おしい愛の記録。 | subject_summary | vector_not_in_top10 | 向量召回 | - | 463530, 618127, 509739 |
| q119 | 找一部主角叫 Zzyzx Qwerty 的动画 | unanswerable | weak_false_positive | 问题歧义或负样例 | - | 32248, 975, 383338 |
| q120 | 有一部作品，剧情是紫色大象在月球开便利店 | unanswerable | weak_false_positive | 问题歧义或负样例 | - | 33310, 220056, 13131 |
| q121 | 角色名字是赵不存在，能力是控制时间 | unanswerable | weak_false_positive | 问题歧义或负样例 | - | 159725, 811, 131605 |
| q122 | 想找一部关于会说话的蓝色仙人掌拯救银河系的漫画 | unanswerable | weak_false_positive | 问题歧义或负样例 | - | 389389, 12782, 1860 |
| q123 | 有一部作品，主角是一只穿西装的仓鼠侦探 | unanswerable | weak_false_positive | 问题歧义或负样例 | - | 66239, 389389, 5871 |
| q124 | 记得一部动画，反派用微波炉统治世界 | unanswerable | weak_false_positive | 问题歧义或负样例 | - | 38668, 607653, 241595 |
| q125 | 找一部作品，里面有叫 Qwertyuiop 的魔法学院 | unanswerable | weak_false_positive | 问题歧义或负样例 | - | 195212, 453904, 345823 |

### hybrid 失败样例（10 条）
| query_id | 查询 | 线索类型 | 失败原因 | 归因 | target_rank | top_ids |
| --- | --- | --- | --- | --- | ---: | --- |
| q119 | 找一部主角叫 Zzyzx Qwerty 的动画 | unanswerable | weak_false_positive | 问题歧义或负样例 | - | 20026, 2055, 390318 |
| q120 | 有一部作品，剧情是紫色大象在月球开便利店 | unanswerable | weak_false_positive | 问题歧义或负样例 | - | 5114, 112984, 939 |
| q121 | 角色名字是赵不存在，能力是控制时间 | unanswerable | weak_false_positive | 问题歧义或负样例 | - | 159725, 109378, 2193 |
| q122 | 想找一部关于会说话的蓝色仙人掌拯救银河系的漫画 | unanswerable | weak_false_positive | 问题歧义或负样例 | - | 210139, 207960, 1860 |
| q123 | 有一部作品，主角是一只穿西装的仓鼠侦探 | unanswerable | weak_false_positive | 问题歧义或负样例 | - | 66239, 43215, 411493 |
| q124 | 记得一部动画，反派用微波炉统治世界 | unanswerable | weak_false_positive | 问题歧义或负样例 | - | 26391, 32248, 94985 |
| q125 | 找一部作品，里面有叫 Qwertyuiop 的魔法学院 | unanswerable | weak_false_positive | 问题歧义或负样例 | - | 345823, 661805, 453904 |
| q126 | 有一部关于企鹅参加奥运会的青春故事 | unanswerable | weak_false_positive | 问题歧义或负样例 | - | 331946, 202371, 361296 |

## 6. 评测限制

- 查询由已索引文档文本生成，不是独立人工查询集；部分查询与索引文本共享措辞。
- 仅覆盖 P4 的 10,000 篇抽样文档，不代表 84.6 万篇全量数据。
- 可接受答案只使用 P3 文档记录的主作品，角色跨作品关系未全部列入可接受集合。
- 不可回答查询是合成负样例，可能偶然命中真实词汇。
- 置信度使用关键词覆盖率阈值；阈值在开发集上校准，不是独立验证。
- Any false positive 包含低置信度近邻；Confident false positive 才反映高置信误报。
- 延迟为预热后单进程测量，首次请求还包含模型和索引加载时间。
- 本报告是 P6 开发基线，不是最终检索质量结论。

## 7. 数据文件

- JSON 报告：`E:\PlotSeeker\storage\evaluation\p6_report.json`
- 失败记录：`E:\PlotSeeker\benchmarks\failures.md`

