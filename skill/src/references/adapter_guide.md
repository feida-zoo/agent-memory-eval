# Adapter 接入指南

membench 与被测 Agent 之间只有一个协议：`BaseMemoryAdapter`（见 `membench/adapters/base.py`）。

## 三个挂点在被测 Agent 源码中的典型位置

| membench 方法 | 在 Agent 源码中找什么 | 注意事项 |
|---|---|---|
| `reset()` | 记忆存储的清空入口 | 必须真清空（向量库 collection、事实表、情景缓冲、图谱）。测试环境用独立实例/库，绝不碰生产数据 |
| `ingest(messages)` | 记忆的**写入路径**：对话结束时的抽取/总结/入库逻辑 | 传入 `messages: [{"role","content","timestamp"}]`；返回写入耗时与 LLM 调用数 |
| `query(question)` | 记忆的**读取路径** + 回答生成 | `retrieved` 字段必须填**实际注入 prompt 的记忆条目原文**（用于 evidence_recall）；填不了检索细节时至少返回注入的记忆文本 |

## Python 直接接入（推荐）

```python
from membench import BaseMemoryAdapter, IngestStats, QueryResult

class MyAgentAdapter(BaseMemoryAdapter):
    name = "my-agent-v1"          # 版本写进 name，便于 compare 区分

    def __init__(self, platform: str = "default"):
        # platform 用于多平台对比：在此切换接入配置（如不同 IM 平台的记忆命名空间）
        ...

    def reset(self) -> None: ...
    def ingest(self, messages) -> IngestStats: ...
    def query(self, question) -> QueryResult: ...
```

注册方式二选一：
1. 在 `membench/adapters/__init__.py` 的 `ADAPTERS` 注册表中加一行（改 membench 源码）；
2. 不动 membench：写 `my_adapter.py` 后用 Runner API 直接传实例（见 `examples/quickstart.py`）。

## HTTP 接入（Agent 是独立服务时）

在 Agent 侧实现三个端点，然后用内置适配器：

```bash
python -m membench.cli run --adapter http --judge heuristic   # 端点地址见 adapters/http.py 的配置方式
```

- `POST /reset` → 200
- `POST /ingest`  body: `{"messages": [...]}` → `{"input_tokens":int,"llm_calls":int,"latency_ms":float}`
- `POST /query`   body: `{"question": str}` → `{"answer":str,"retrieved":[str],...}`

## 判分器接入（三选一）

| 方式 | 命令 | 适用场景 |
|---|---|---|
| `--judge heuristic` | 零配置 | 首次接入跑通；stale/abstention 红线指标始终规则判定 |
| `--judge llm` | 配 `--llm-base-url/--llm-api-key/--llm-model` | **正式评估首选**：第三方模型，避免自评偏差 |
| `--judge client` | 配 `--client-llm "my_agent.llm:LLMClient"` | 复用 agent 自己的 LLM 快速验证；报告需注明同源偏差 |

`--client-llm` 接受 `module:ClassName` 类路径（无参实例化），或返回字符串的函数。类需带 `.chat(prompt)->str` 或 `.complete(prompt)->str` 方法；若类名/方法名不同，继承 `membench.judges.client.ClientLLMJudge` 并重写 `_call()` 即可。

## 常见接错（自查清单）

- [ ] `query` 里把问题直接丢给底座模型而没走记忆检索 → 分数虚高，证据召回字段暴露
- [ ] `reset` 只清了会话没清记忆库 → 跨 dialogue 串扰，single_hop 异常高、abstention 异常低
- [ ] `ingest` 里同步调了生成模型做总结但没计入 `llm_calls` → 成本指标失真
- [ ] timestamp 没传给记忆系统 → temporal 题系统性失败（办公集里大量"昨天/下周"依赖时间锚）
