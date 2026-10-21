# LLM Agent CLI

> 更新时间：2026-10-21
> 状态：本地实验功能，未接入 P8 API。
> 接口：OpenAI-compatible Chat Completions + function/tool calling。

## 1. 配置 API Key

编辑 **不进入 Git** 的 `config/agent.json`：

```json
{
  "provider": "openai-compatible",
  "api_key": "sk-你的Key",
  "base_url": "https://api.deepseek.com/v1",
  "model": "deepseek-chat",
  "temperature": 0.2,
  "max_tokens": 1200,
  "timeout_seconds": 90,
  "max_tool_rounds": 4,
  "top_k": 5
}
```

`config/agent.json` 已在 `.gitignore` 中。也可以复制示例：

```powershell
Copy-Item config\agent.example.json config\agent.json -Force
```

或使用环境变量覆盖：

```powershell
$env:PLOTSEEKER_LLM_API_KEY = "sk-..."
$env:PLOTSEEKER_LLM_BASE_URL = "https://api.deepseek.com/v1"
$env:PLOTSEEKER_LLM_MODEL = "deepseek-chat"
```

`base_url` 需要是 OpenAI 兼容根地址，程序会请求：

```text
{base_url}/chat/completions
```

## 2. 启动

```powershell
# 检查配置（不会调用 API，也不会加载 BGE）
python scripts/agent_llm.py --check-config

# 本地查询
python scripts/agent_llm.py "失去记忆的魔王在现代学校"

# JSON 输出
python scripts/agent_llm.py "时间循环学园有什么相关作品" --json

# 查看工具定义
python scripts/agent_llm.py --tools
```

可选覆盖：

```powershell
python scripts/agent_llm.py "查询" `
    --base-url https://api.openai.com/v1 `
    --model gpt-4o-mini `
    --max-tool-rounds 4 `
    --top-k 5
```

Key 建议只放配置文件或环境变量，不建议长期写在命令行历史里。

## 3. 执行流程

```text
用户查询
  -> LLM 选择工具
  -> P9 AgentTools 执行本地检索
  -> 工具结果（doc_id / subject_id / 证据片段）回填给 LLM
  -> LLM 生成带证据的最终回答，或调用澄清工具
```

工具：

- `search_subjects`
- `search_characters`
- `search_episodes`
- `get_related_subjects`
- `ask_user_for_clarification`

所有证据来自 P2/P5，不来自 LLM 记忆。若模型没有调用任何工具，程序会追加一次
“必须先调用检索工具”的提示；仍然不调用时结果带 `llm_did_not_call_tools` warning。

## 4. 输出

CLI 默认打印：

- 回答
- 模型、工具轮次、耗时
- prompt/completion/total tokens
- 工具调用列表与摘要
- 警告或澄清问题

`--json` 输出完整结构，包括：

```json
{
  "query": "...",
  "answer": "...",
  "clarification": null,
  "tool_calls": [],
  "rounds": 1,
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
  "warnings": [],
  "elapsed_ms": 0,
  "model": "...",
  "base_url": "...",
  "used_llm": true
}
```

## 5. 安全与限制

- API Key 只从 `config/agent.json` 或环境变量读取，不写入日志、报告或错误消息。
- `config/agent.json` 已被 `.gitignore` 排除；`agent.example.json` 不含真实 Key。
- 当前未调用真实外部 API 做验收；本地测试需要自备 Key。
- 不同 OpenAI-compatible 服务对 `tools`、`tool_choice`、`max_tokens` 的兼容性
  可能不同；推荐使用支持 function calling 的模型。
- 如果模型不支持工具调用，CLI 会得到 `llm_did_not_call_tools` warning，而不是
  自动切换成纯文本生成。
- 没有流式输出、重试退避、成本预算、并发控制和多轮对话会话；每次 CLI 调用是
  独立请求。
- 未接入 P8 API；API Key 不通过 `/search` 传递。
