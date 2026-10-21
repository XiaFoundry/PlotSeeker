# Agent 本地配置

`agent.json` 保存本地 LLM API Key，已被 `.gitignore` 排除，不要提交。
首次使用复制示例并填入 Key：

```powershell
Copy-Item config\agent.example.json config\agent.json -Force
notepad config\agent.json
```

字段：

- `provider`：当前只支持 `openai-compatible`。
- `api_key`：你的 API Key。
- `base_url`：OpenAI 兼容接口根地址，例如：
  - DeepSeek：`https://api.deepseek.com/v1`
  - OpenAI：`https://api.openai.com/v1`
  - 本地兼容服务：`http://127.0.0.1:11434/v1`
- `model`：支持 tool calling 的模型名。
- `temperature`、`max_tokens`、`timeout_seconds`：请求参数。
- `max_tool_rounds`：最多工具调用轮数。
- `top_k`：单次检索返回条数。

也可以用环境变量覆盖 Key 和端点：

```powershell
$env:PLOTSEEKER_LLM_API_KEY = "sk-..."
$env:PLOTSEEKER_LLM_BASE_URL = "https://api.deepseek.com/v1"
$env:PLOTSEEKER_LLM_MODEL = "deepseek-chat"
```

Key 不会写入日志或报告。
