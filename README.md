# LiteBot

LiteBot 是一个面向真实任务执行场景的轻量级个人 AI Agent。它在 nanobot
`v0.1.4.post6` 的基础上，重点补齐了长上下文管理、结构化情景记忆、工具运行时安全、
统一生命周期钩子，以及可重复的离线评测与回归审计能力。

当前开发分支：`litebot-dev`。

> LiteBot 目前仍兼容上游的 Python 包名、命令行入口和默认数据目录。因此下文中的
> `nanobot` 命令、`nanobot.*` 导入路径和 `~/.nanobot` 目录是现阶段真实可用的接口，
> 并不代表项目仍以 nanobot 作为产品名称。详见[命名与兼容性](#命名与兼容性)。

## LiteBot 增强了什么

### 长上下文管理

- 根据模型上下文窗口和输出预留空间计算有效 token 预算。
- 在软阈值处滚动压缩较早对话，同时保留系统提示、当前请求和最近轮次。
- 在硬阈值前执行多轮收敛；受保护内容本身超限时明确返回错误，不静默截断。
- 记录压缩来源、估算 token、压缩轮次和保留消息等遥测信息。

### 大型工具结果卸载与按需检索

- 默认将超过 8 KiB 的工具结果保存为会话级 artifact，避免完整结果反复进入模型上下文。
- 使用 `search_tool_result` 在 UTF-8 文本/JSON artifact 中定位内容。
- 使用 `get_tool_result` 按字符偏移读取有限页面。
- artifact 具有会话归属、校验和、读取/搜索额度、TTL 和垃圾回收保护。

### 结构化情景记忆

- 将值得长期保留的事件写入 `memory/HISTORY.jsonl`，同时维护便于人工查看的 Markdown 镜像。
- 在新请求到来时进行轻量、可解释的 Top-K 相关性检索。
- 对检索到的记忆进行去重，并以受保护上下文注入当前 Agent 运行。
- 保留原有 `MEMORY.md` 长期记忆机制，并支持历史数据迁移。

### 工具运行时安全

- 所有工具调用统一经过 `ALLOW` / `CONFIRM` / `DENY` 策略决策。
- 支持按工具名、参数正则、工作区边界、敏感路径和重复调用进行治理。
- `CONFIRM` 在宿主未提供异步确认处理器时采用 fail-closed 行为。
- 生成工作区内的脱敏 JSONL 审计记录，并对工具错误和部分执行结果进行结构化分类。

### 统一生命周期钩子

LiteBot 可在会话、Agent、LLM、工具、上下文和记忆阶段注册有序、隔离失败的钩子，
覆盖 `SessionStart`、`AgentRunStart`、`PreLLMCall`、`PreToolUse`、
`ContextCompact`、`MemoryWrite`、`ToolAudit` 等事件。钩子可允许、拒绝或修改运行行为，
同时保留旧版 `AgentHook` 兼容层。

### 评测与回归审计

- v2 评测框架包含 72 个固定离线核心用例。
- 覆盖上下文管理、记忆检索、工具安全、工具可靠性、多步推理和回归六类能力。
- 核心用例不依赖真实模型、网络、子进程或 LLM 裁判。
- 支持具名 baseline、JSON/Markdown 对比报告、结果 schema 校验和哈希链审计轨迹。
- 另提供长上下文、大工具结果、情景记忆、生命周期钩子和工具安全的专项基准。

LiteBot 同时继承上游已有能力，包括多模型 Provider、MCP、Skills、子 Agent、Cron、
Heartbeat，以及 Telegram、Discord、WhatsApp、飞书、钉钉、Slack、QQ、Matrix、
邮件、企业微信和微信等渠道集成。

## 环境要求

- Python 3.11 或 3.12
- 推荐使用 [uv](https://docs.astral.sh/uv/)
- 使用真实模型时，需要配置相应 Provider 的 API Key 或 OAuth 登录

## 本地安装

使用 uv：

```bash
uv sync --extra dev
```

或使用 pip：

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
```

初始化配置和工作区：

```bash
nanobot onboard --wizard
```

默认配置文件位于 `~/.nanobot/config.json`，默认工作区位于
`~/.nanobot/workspace`。

## 快速开始

发送单条消息：

```bash
nanobot agent -m "你好，请总结当前工作区的项目结构"
```

进入交互模式：

```bash
nanobot agent
```

启动消息渠道、Heartbeat 和 Cron 网关：

```bash
nanobot gateway
```

检查配置和运行状态：

```bash
nanobot status
nanobot channels status
```

使用独立配置和工作区：

```bash
nanobot onboard --config ~/.litebot/config.json --workspace ~/.litebot/workspace
nanobot agent --config ~/.litebot/config.json --workspace ~/.litebot/workspace
nanobot gateway --config ~/.litebot/config.json
```

`--config` 用于选择实例配置；`--workspace` 可覆盖该配置中的默认工作区。

## 配置示例

配置模型、上下文窗口和本地时区：

```json
{
  "agents": {
    "defaults": {
      "provider": "openrouter",
      "model": "anthropic/claude-sonnet-4",
      "contextWindowTokens": 65536,
      "maxTokens": 8192,
      "timezone": "Asia/Shanghai"
    }
  },
  "providers": {
    "openrouter": {
      "apiKey": "YOUR_API_KEY"
    }
  }
}
```

配置上下文管理：

```json
{
  "agents": {
    "defaults": {
      "contextManagement": {
        "recentTurns": 8,
        "softThreshold": 0.8,
        "hardThreshold": 0.92,
        "compactionTarget": 0.68,
        "toolOffloadThresholdBytes": 8192,
        "artifactPageSize": 4096,
        "artifactMaxReadsPerSession": 3,
        "artifactMaxSearchesPerSession": 5,
        "artifactMaxReturnedCharsPerSession": 12288,
        "artifactTtlDays": 30
      }
    }
  }
}
```

阈值必须满足：

```text
compactionTarget < softThreshold < hardThreshold
```

配置工具策略：

```json
{
  "tools": {
    "restrictToWorkspace": true,
    "policy": {
      "enabled": true,
      "defaultAction": "confirm",
      "duplicateWindowSeconds": 5,
      "auditPath": "logs/tool-audit.jsonl",
      "rules": [
        {
          "id": "deny-destructive-shell",
          "tools": ["exec"],
          "action": "deny",
          "reason": "禁止高风险删除命令",
          "priority": 100,
          "arguments": {
            "command": {
              "regex": ["(?i)\\brm\\s+-rf\\b", "(?i)\\bformat\\b"]
            }
          }
        }
      ]
    }
  }
}
```

配置键同时接受 camelCase 和 snake_case。`auditPath` 必须位于 Agent 工作区内。

## 运行测试

```bash
python -m pytest
```

只运行本分支新增的核心测试：

```bash
python -m pytest \
  tests/agent/test_context_manager.py \
  tests/agent/test_context_tool_offload.py \
  tests/agent/test_episodic_memory.py \
  tests/agent/test_hook_manager.py \
  tests/security/test_tool_policy.py \
  tests/tools/test_get_tool_result.py
```

## 运行评测

验证和查看 v2 用例：

```bash
python -m benchmarks validate
python -m benchmarks list --category tool_safety
```

运行完整离线核心评测或单个用例：

```bash
python -m benchmarks run --profile core
python -m benchmarks run --case tool_safety.explicit_deny
```

维护 baseline 并比较回归：

```bash
python -m benchmarks baseline promote \
  --run benchmarks/results/<run-id> \
  --profile core \
  --name phase4

python -m benchmarks baseline set-default --profile core --name phase4
python -m benchmarks compare \
  --baseline core/default \
  --current benchmarks/results/<run-id>
```

核心比较会对正确性、覆盖率、审计和 manifest 回归返回失败；延迟、token 和工具轮次变化
默认作为警告，使用 `--strict-performance` 可将其升级为失败。

更多专项基准和输出格式见 [benchmarks/README.md](benchmarks/README.md)。

## 项目结构

```text
nanobot/
├── agent/
│   ├── loop.py              # Agent 主循环与上下文/记忆编排
│   ├── runner.py            # LLM 与工具运行生命周期
│   ├── context_manager.py   # token 预算、压缩与大型结果卸载
│   ├── episodic_memory.py   # 结构化情景记忆与检索
│   ├── hook.py              # 统一生命周期钩子
│   └── tools/
│       └── tool_result.py   # artifact 搜索与分页读取
├── security/
│   └── tool_policy.py       # 工具策略、脱敏、审计和错误分类
├── session/
│   ├── manager.py           # 会话持久化
│   └── artifacts.py         # 会话级工具结果存储
├── channels/                # 消息渠道
├── providers/               # 模型 Provider
├── config/                  # 配置 schema 与加载逻辑
├── skills/                  # 内置 Skills
├── cron/                    # 定时任务
├── heartbeat/               # 主动唤醒
└── cli/                     # 命令行入口

benchmarks/                  # v2 评测框架、用例、schema 与专项基准
tests/                       # 单元、集成和回归测试
bridge/                      # WhatsApp Bridge
```

## 命名与兼容性

当前代码处于“产品名称已变更、内部接口尚未迁移”的过渡阶段：

| 层级 | 当前值 | 说明 |
|---|---|---|
| 项目/产品名称 | LiteBot | README、评测 schema 和新增能力使用该名称 |
| Python 分发名 | `nanobot-ai` | `pyproject.toml` 中的安装包名称 |
| Python 包名 | `nanobot` | 全部内部 import 和第三方扩展依赖的接口 |
| CLI 命令 | `nanobot` | 当前唯一已注册的命令行入口 |
| 默认数据目录 | `~/.nanobot` | 配置、工作区和运行数据的默认位置 |
| 容器与服务名 | `nanobot` | Docker Compose、镜像示例和 systemd 约定 |

README 已将项目本身描述为 LiteBot，但不会把仍然有效的兼容接口伪装成已经完成迁移。
如果要正式发布为独立的 LiteBot 产品，还需要一次有计划的全仓库重命名；不应对源码做简单的
全局文本替换。建议至少覆盖：

1. `pyproject.toml` 的分发名、CLI entry point、构建包路径和项目元数据。
2. `nanobot/` Python 包目录、全部 import、日志 namespace、测试和插件/扩展接口。
3. `~/.nanobot` 默认目录，并为既有配置、workspace、session、memory 和 cron 数据提供迁移或兼容回退。
4. Dockerfile、Compose service、镜像名、systemd 示例、环境变量及部署脚本。
5. Logo、图片、文档、贡献指南、安全策略、仓库链接和发布元数据。
6. 向后兼容策略，例如暂时保留 `nanobot` CLI alias 和 `nanobot` import shim，并标注弃用周期。

如果当前目标只是让 `litebot-dev` 分支准确展示其新增能力，更新 README 足够；如果目标是打包、
安装或对外发布一个名为 LiteBot 的独立项目，只改 README 不够。

## 已知限制

- artifact 搜索和分页位置是解码后的字符偏移，不是原始字节偏移；当前不提供二进制 artifact API。
- artifact 检索按会话和 artifact 限流。默认最多搜索 5 次、分页读取 3 次、累计返回
  12,288 个字符；命中搜索会重置局部读取计数。
- 如果受保护的系统提示和当前用户消息本身已超过有效预算，LiteBot 会返回上下文溢出错误。
- `CONFIRM` 策略需要宿主提供异步确认处理器；普通 CLI/网关流程没有确认能力时会拒绝执行。
- `live` 和 `judge` 评测需要显式启用，并可能依赖真实 Provider、网络、配额或外部服务。

## 项目来源与许可证

LiteBot 基于 HKUDS/nanobot `v0.1.4.post6` 继续开发，并保留原项目的 MIT License 和版权声明。
本分支新增实现聚焦于 Agent 在长任务中的上下文效率、记忆可用性、工具安全和可验证回归。

详见 [LICENSE](LICENSE)。
