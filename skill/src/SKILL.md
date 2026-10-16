---
name: agent-memory-eval
description: >
  对有源码的 Agent（尤其是通用办公助手类 Agent）进行记忆系统评测与优化迭代的工程化技能。
  内置 membench 评测工程（adapter 协议 + 六维指标 + 回归门禁 + 报告生成）与办公域黄金测试集。
  触发场景：评测/测试 Agent 的记忆系统、记忆能力打分、记忆系统优化前后对比（A/B）、
  两个 commit 的记忆回归对比、多平台（如企微/飞书/钉钉接入）记忆表现对比、平台×commit 矩阵对比、
  记忆系统优化闭环（诊断→建议→plan→执行→AB→报告）、
  基于用户真实对话日志构造/删除用户专属测试集。
  关键词：记忆评测、memory benchmark、记忆回归、A/B 对比、stale_rate、黄金集、优化闭环、可比性校验。
---

# agent-memory-eval：Agent 记忆系统评测技能

本技能封装 membench 评测工程（`membench/`，自带 README 与测试），覆盖**接入 → 评测 → 对比 → 优化闭环**的完整工作流。
**所有评测报告必须输出到本 skill 的 `output/` 目录，文件名带 `_YYMMDD_hhmmss` 时间戳后缀，报告开头注明所用测试集（membench 自动生成，勿删）。**

## ⛔ 两个强制检查点（不得跳过）

1. **任何优化/改动完成后，先询问用户是否执行 A/B 对比，不得径自开跑。** 你是"提方案并按用户确认执行"的角色，不是"自动改完就自信跑分"的角色。
2. **启动 A/B 前，先与用户确认数据集范围（单集 / 多集 / 全量）、A/B 分组（基线 vs 候选）与平台/标签。** 数据集口径不一致、平台/标签错配，都会让对比结论不可归因。

> 纪律背景：`compare` 只**测量差异**，不**判定**两臂是否只差一个变量。可比性必须由你（优化者）自己保证——固定模型与 harness、只切优化开关，并用 `--require-comparable` 做前置校验。

## 被测对象定位：单用户个人 Agent

**目标被测系统是单用户个人 Agent，记忆属于用户个人。** 该定位影响评测口径：

- **多用户权限题已屏蔽**：GateMem 内置集中 140 道 privacy-refuse 题（测"同一记忆对不同身份的选择性披露"）与单用户场景无关，已在数据集内标记 `ignore=true`，评测引擎自动跳过——不删除原始数据，口径可追溯，未来若产品演变为多用户形态可随时恢复。
- **safety（删除后遗忘）依然重要**：用户说"忘掉我上周的体检结果"，agent 之后不应再记起——这是数据主权问题，与单/多用户无关，GateMem 的 222 道 safety 题全部保留参评。
- **残余风险缺口**：单用户场景下的隐私风险以"对外输出泄漏"（agent 帮你写邮件时带出私密记忆）和"提示注入套取"形式存在，现有评测集未覆盖；如需要可在用户测试集中补充这类题。

## 目录约定

| 路径 | 用途 |
|---|---|
| `membench/` | 评测工程本体（CLI: `python -m membench.cli ...`） |
| `membench/membench/data/office_dataset.jsonl` | **内置**办公域黄金集（20 对话 / 86 题 / 7 类题型），业务验收门禁用 |
| `membench/membench/data/sample_dataset.jsonl` | **内置**冒烟迷你集（仅验证管线连通，不作评估结论） |
| `~/.membench/datasets/` | **按需下载**数据集的落盘目录（`locomo` / `longmemeval_oracle` / `longmemeval_s` / `gatemem_office` / `beam_1m` / `beam_10m`） |
| `user_dataset/` | 用户专属测试集存放目录（与内置集物理隔离） |
| `output/` | 所有评估/对比报告的输出目录（实验子目录：`eval_*` / `ab_*` / `matrix_*`） |
| `docs/` | 《Agent 记忆系统深度研究报告》及配图，供随时查阅背景知识 |
| `references/adapter_guide.md` | 接入指南（三挂点、Python/HTTP 接入、判分器） |
| `references/compare_workflows.md` | commit / 平台 / 矩阵对比详解 + 可比性前置校验 |
| `references/optimize_workflow.md` | 记忆系统优化闭环操作手册（六阶段） |
| `references/http_bridge_guide.md` | 非 Python 系统 HTTP bridge 接入 checklist |
| `references/user_dataset_guide.md` | 用户测试集构造指南 |
| `scripts/eval_two_commits.sh` | 两个 commit 的 A/B 评测模板脚本 |
| `membench/examples/quickstart.py` | 真实 adapter 编写参考（把假想记忆系统包成 BaseMemoryAdapter） |
| `membench/examples/optimize_loop.md` | 优化闭环教程（基线 → 单变量改动 → compare 门禁 → 迭代） |
| `membench/examples/http_bridge_node/` | 非 Python（Node/Electron）HTTP bridge 可复制模板 |

## 工作流 0：接入被测 Agent（首次使用必做）

1. 阅读 `references/adapter_guide.md`，在被测 Agent 源码中定位三个挂点：记忆清空、记忆写入、记忆读取+回答。
2. 实现 `BaseMemoryAdapter` 三方法（`reset` / `ingest` / `query`），或在 Agent 侧暴露 `POST /reset|/ingest|/query` 三个端点后用内置 HttpAdapter。
   - **若被测 Agent 的记忆能力本身就是"LLM + 工具调用"**（ingest 与 query 是同构的 tool-call 循环），不要各写一遍骨架——直接用 `membench.agent_loop.AgentLoopAdapter` 复用循环（见「本波新能力」）。
3. **生成的适配器必须显式持久化**（否则跨 session 丢失）：
   ```bash
   cd <skill目录>/membench
   python -m membench.cli adapter install /path/to/your_adapter.py --name <适配器名>
   # 落盘到 ~/.membench/adapters/，之后任何 session 直接 --adapter <适配器名> 可用
   # 查看：python -m membench.cli adapter list；删除：adapter remove <名>
   ```
   临时/CI 场景也可用环境变量 `MEMBENCH_ADAPTER_MODULES=module.path`（同名时环境变量优先于持久化）。
4. 冒烟验证（**membench 命令一律在 `<skill目录>/membench/` 项目根内执行**）：
   ```bash
   cd <skill目录>/membench
   python -m membench.cli run --adapter <你的adapter> --dataset membench/data/office_dataset.jsonl --judge heuristic
   ```
5. **验收标准**：pipeline 跑通且 `stale_rate` 与 `abstention_accuracy` 非零缺陷暴露合理；若全部满分，警惕 adapter 接错（如 query 没走记忆路径而是直接问底座模型）。

## 工作流 1：用户测试集管理（严格显式触发）

**铁律：构造和删除都必须由用户显式要求，绝不自动执行。评测时若用户测试集存在则默认使用，但不得主动建议之外的任何写操作。**

- **构造**（用户说"用我的真实数据构造测试集"等）：按 `references/user_dataset_guide.md` 将用户对话日志合成为 membench JSONL（schema 校验 + evidence 原文校验），然后：
  ```bash
  cd <skill目录>/membench
  MEMBENCH_USER_DATA_DIR=<skill目录>/user_dataset python -m membench.cli user-data install <合成文件>
  ```
- **删除**（用户说"删除我的测试集"等）：先向用户确认，再执行 `user-data delete`。
- **查看**：`user-data status`。

## 工作流 1.5：数据集介绍与按需下载

用户问"支持哪些测试集 / 这个分数算好还是差"时，用内置介绍命令（不要凭记忆回答，以命令输出为准）：

```bash
cd <skill目录>/membench
python -m membench.cli dataset info                 # 列举全部数据集：来源、特点、基准线要点
python -m membench.cli dataset info locomo          # 单个数据集详细介绍（含实际统计与用法示例）
python -m membench.cli dataset fetch locomo         # 下载并转换落盘到 ~/.membench/datasets/
python -m membench.cli dataset fetch longmemeval_oracle
python -m membench.cli dataset fetch beam_1m
```

**数据集口径（务必记牢，别再按记忆误答）：**

| 数据集 | 状态 | 说明 |
|---|---|---|
| `office` | **内置** | 办公域黄金集（`membench/data/office_dataset.jsonl`），业务验收门禁 |
| `sample` | **内置** | 冒烟迷你集（`membench/data/sample_dataset.jsonl`） |
| `locomo` | **按需下载** | `dataset fetch locomo`（10 对话 / 1986 题，英文日常对话，横向对标） |
| `longmemeval_oracle` | **按需下载** | `dataset fetch longmemeval_oracle`（500 题，只含证据会话） |
| `longmemeval_s` | **按需下载** | `dataset fetch longmemeval_s`（500 题，~50 会话/题，~268MB，检索压力测试） |
| `gatemem_office` | **按需下载** | `dataset fetch gatemem_office`（17 episodes / 547 检查点，utility/safety 两维有效） |
| `beam_1m` / `beam_10m` | **按需下载** | 百万/千万 token 级压力测试（体积大，不内置） |
| `statemembench` | 暂不可用 | 数据未公开，需向作者索取 |

- **只有 `office` / `sample` 是内置的**，其余全在 `~/.membench/datasets/`（可用 `MEMBENCH_DATASETS_DIR` 覆盖）。
- 每个条目的「基准线」小节说明看到评分如何判断好坏（如 LoCoMo 人类上限 F1≈87.9%、近年记忆系统 ACC 57%~77% 参考区间）。

## 工作流 2：运行评测，输出评估报告

```bash
cd <本skill目录>/membench
export MEMBENCH_USER_DATA_DIR=<skill目录>/user_dataset
python -m membench.cli run --adapter <adapter> --judge heuristic --report-out ../output/
# 或指定测试集 / 平台 / commit / 驱动 LLM / 适配器版本：
python -m membench.cli run --adapter <adapter> --dataset <路径> --platform <平台名> --commit <sha> \
  --driver-llm <驱动模型标识> --adapter-version <适配器版本> --report-out ../output/
# 用按需下载的基准（先 fetch；文件在 ~/.membench/datasets/）：
python -m membench.cli run --adapter <adapter> --dataset ~/.membench/datasets/locomo.jsonl --report-out ../output/
# 完整清单与介绍：python -m membench.cli dataset info
```

**产物目录约定（本波已落地的布局，报告与 run 就近归属）：**

| 命令 | 产物目录 | 内含 |
|---|---|---|
| `run --report-out DIR` | `DIR/eval_<adapter>_<platform>_<ts>/` | `report.md`、`metrics.json`、`fig_*.png`、`runs/<run_id>/`（自包含） |
| `evaluate --report-out DIR` | `DIR/eval_<adapter>_<platform>_<ts>/` | `evaluate_report.md`、`evaluate_summary.json`、`fig_dataset_overview.png`、`runs/<各数据集 run_id>/` |
| `compare --report-out DIR` | `DIR/ab_<tag>_<ts>/` | `A/{report.md,metrics.json,fig_*.png,runs/}`、`B/{...}`、`compare_report.md`（+`fig_metric_diff.png`、`fig_category_diff.png`）、可选 `optimization_report.md` |
| `matrix --report-out DIR` | `DIR/matrix_<ts>/` | 矩阵报告、`matrix_result.json`、配图、`runs/<label>/` |

不传 `--report-out` 时，runs 默认落到 `<skill目录>/output/runs/`（skill 环境）或工程根 `runs/`（开发环境）。命名 helper（`membench.layout.subdir_for` / `ab_subdir` / `matrix_subdir`）是公开可复用 API，不必自己重写约定。

### 工作流 2.5：全量评估（多数据集融合总分）

用户问"整体打个分 / 全部数据集跑一遍"时用 `evaluate`：对办公域槽位（用户集>内置office）+ locomo + longmemeval_oracle + gatemem_office + BEAM-1M 依次评测，跨数据集**宏平均**融合出综合评级，输出一份全量报告：

```bash
python -m membench.cli evaluate --adapter <adapter> --judge llm --report-out ../output/
python -m membench.cli evaluate --adapter <adapter> --quick      # 跳过 BEAM，分钟级日常回归
python -m membench.cli evaluate --adapter <adapter> --beam-10m   # 1M 换成 10M 全集终验
python -m membench.cli evaluate --adapter <adapter> --include-lme-s  # 追加 LongMemEval-S（检索/精确率压力测试）
```

用户问"两版系统 A/B 对比一下、跨数据集一次出结论"时用 `evaluate-compare`：两套 adapter × 同一组数据集，分别跑完后产出**逐数据集 diff + 逐数据集题型 diff + 跨数据集融合 diff + 融合/per-dataset 两层门禁 + 对比图**，落 `<报告父目录>/ab_<tag>_<ts>/` 自包含结构（旧版 run 的 capabilities 未落账时自动推断并在报告头标注来源）：

```bash
python -m membench.cli evaluate-compare \
    --baseline-adapter <adapter_A> --candidate-adapter <adapter_B> \
    --judge llm --report-out ../output/ --tag "优化前vs优化后"
# 两侧可分别指定 driver_llm / adapter-version 供 compare 可比性前置校验
```

> **全量评估前必须先 fetch 用到的按需下载数据集**（locomo / longmemeval_oracle / gatemem_office / beam_1m 等），否则会报错提示先 `dataset fetch <key>`。命令：`python -m membench.cli dataset fetch <key>`（落 `~/.membench/datasets/`）。

解读要点（向用户说明）：
- **宏平均**：每个指标先在各数据集内算分再跨集等权平均；指标只在具备对应标注能力的集上参与（无 gold_stale 的集不参与 stale_rate，无 evidence 标注的集不参与 evidence_recall），不会用"记 0"稀释总分。
- **门禁独立于总分**：办公域槽位 stale_rate>0.03 一票否决，总评"不达标"、退出码 1，无论综合评级多高。
- BEAM 未 fetch 时会报错提示先 `dataset fetch beam_1m`；heuristic 判分跑 BEAM 会警告失真，正式全量评估用 `--judge llm`。

**判分器三选一**（按可信度和可用性排序）：
- `--judge llm`：第三方模型（OpenAI 兼容协议），**正式评估首选**，与公共基准口径一致。配 `--llm-base-url/--llm-api-key/--llm-model`。
- `--judge client`：复用 **agent 自己的 LLM** 做判分（自评，可能系统性偏高，但可用于快速验证）。配 `--client-llm "my_agent.llm:LLMClient"`（类路径，会被无参实例化）。适合记忆系统针对自家 LLM 做过优化、想先看自评效果的场景。
- `--judge heuristic`：零成本默认，规则判定。stale/abstention 红线指标始终由规则判定，不受 LLM 影响。

测试集选择优先级（membench 自动执行，向用户说明实际使用了哪个）：
**用户显式指定 > 用户测试集（如已构造）> 内置办公黄金集**。

## 工作流 3：两个 commit 的 A/B 对比

详见 `references/compare_workflows.md`。要点：用 `git worktree add` 分别检出两个 commit，各自构建并以 `--commit <sha>` 标签跑同一测试集，然后：

```bash
python -m membench.cli compare --baseline runs/<旧commit的run> --candidate runs/<新commit的run> \
  --require-comparable --tag <标签> --report-out ../output/
```

- **`--require-comparable`**：对比前做**前置校验**——驱动/判分模型、适配器实现版本、数据集指纹等关键维度不一致即判"不可比"，禁止当 A/B 用（避免"变量变了还照给 PASS"）。
- **`--tag`**：给本次对比命名，决定 `ab_<tag>_<ts>/` 目录名。
- 其余 commit 对比模板见 `scripts/eval_two_commits.sh`。

门禁：accuracy 降 >0.02 或 stale_rate 升 >0.02 → FAIL（退出码 1，可接 CI）。

## 工作流 4：记忆系统优化闭环

完整操作手册见 `references/optimize_workflow.md`，入门教程见 `membench/examples/optimize_loop.md`。六阶段：

```
诊断（读 report.md：六维指标 / 分题型 / 失败样例 Top10 / 门禁 → 结论"问题出在哪"）
  → 优化建议（可执行方向清单，按收益排序）
  → 优化 plan（拆成单变量改动、排优先级、界定验收指标）
  → 执行（落地单个改动；范围明确且独立时才交 subagent，默认当前会话执行）
  → AB 对比（消融 + 可比性前置校验）
  → 优化报告（方案 + 可归因效果，落 ab_<tag>_<ts>/optimization_report.md）
```

| 阶段 | 动作与命令 | 产物 | 检查点 |
|---|---|---|---|
| 1 诊断 | 读基线 `report.md`（六维指标、by_category、失败样例 Top10、门禁），对照 `optimize_loop.md` 的"现象→短板维度"表定位 | 诊断结论（问题维度 + 证据） | — |
| 2 优化建议 | 给出可执行方向清单，按预期收益排序 | 建议清单 | — |
| 3 优化 plan | 每条拆成**单变量**改动、排优先级、界定验收指标（含门禁阈值） | plan（一人一步） | — |
| 4 执行 | 落地单个改动（只改一处；默认当前会话执行，范围明确且独立时才用 subagent） | 代码改动 | **⛔ 检查点①：改完后先问用户是否跑 A/B** |
| 5 AB 对比 | 固定模型/harness、只切优化开关；先 `--require-comparable` 校验再 `compare` | `ab_<tag>_<ts>/` | **⛔ 检查点②：跑前先确认数据集范围、A/B 分组与平台/标签** |
| 6 优化报告 | 记录方案 + 可归因效果（含门禁结论） | `ab_<tag>_<ts>/optimization_report.md` | — |

- 每次只改一个变量；数据集与判分器在两次 run 之间保持不变（见工作流 2 / 3）。
- 失败/回退样例里若出现大量"答案其实对但判 False"，先切 `--judge llm` 再定胜负（见 `references/optimize_workflow.md`）。
- **`compare` 只测差异、不判定消融**；可比性由你自己保证（如上）。

## 工作流 5：两个平台对比

同一 commit、同一测试集，分别以 `--platform <平台A>` / `--platform <平台B>` 跑两次（adapter 内按平台切换接入配置），然后 `compare`。报告中明确指出平台差异最大的题型（by_category diff）。

## 工作流 6：平台 × commit 矩阵对比

四个组合各跑一次（标签 `平台@commit`），然后：

```bash
python -m membench.cli matrix \
  --runs "wx@<shaA>=runs/<run1>" "fs@<shaA>=runs/<run2>" \
         "wx@<shaB>=runs/<run3>" "fs@<shaB>=runs/<run4>" \
  --report-out ../output/
```

产出矩阵透视表（每指标一张 平台×commit 表）+ 全部两两门禁 + 分组条形图。

## 本波新能力速查

### agent-loop 驱动复用（ingest 与 query 同一套循环）

被测 Agent 的 ingest（喂会话让 LLM 调写入工具落库）与 query（让 LLM 调检索工具作答）往往是**同一套 tool-call 骨架**。用 `AgentLoopAdapter` 把循环、统计与证据回填交给 membench，接入方只给 `{llm_client, tools, system_prompt}`：

```python
from membench.agent_loop import AgentLoopAdapter

adapter = AgentLoopAdapter(
    llm_client="my_agent.llm:LLMClient",   # module:ClassName 类路径（无参实例化），与 --judge client 同约定
    ingest_tools=[{"name": "write_memory", "schema": {...}, "handler": write_memory}],
    query_tools=[{"name": "retrieve", "schema": {...}, "handler": retrieve}],  # 缺省复用 ingest_tools
    system_prompt="你是一个带长期记忆的办公助手。",
)
```

- 确切签名：`AgentLoopAdapter(llm_client, ingest_tools, query_tools=None, system_prompt=None, name="agent-loop", max_rounds=8, retrieved_from=None, ingest_prompt=None, query_prompt=None)`。
- `llm_client` 可为对象 / 可调用 / `"module:ClassName"` 类路径；`tools` 元素为 `{"name","schema","handler"}`。
- membench 负责循环、`IngestStats`/`QueryResult` 统计与 `retrieved` 回填（检索类工具名含 `retrieve`/`search`/`recall` 即被识别；不满足时传自定义 `retrieved_from`）。
- 把实例交给 Runner API 即可评测（见 `membench/examples/quickstart.py` 的用法）。
- **适用边界**：仅当被测系统的写入/读取函数与 LLM 客户端能在 **Python 侧**调用时适用；非 Python 系统（Node/Electron）的 tool-call 循环必须留在被测进程内，membench 无法复用——走 HttpAdapter 三端点（见下一节）。

### HTTP bridge 与 doctor（非 Python 系统）

被测 Agent 不是 Python（Node/Electron）且记忆是库内能力时，写一个极薄 HTTP bridge 暴露三端点，再用内置 HttpAdapter。细节见 `references/http_bridge_guide.md`，可复制模板见 `membench/examples/http_bridge_node/`。

跑评测前先做连通性 + 契约校验：

```bash
python -m membench.cli doctor --target <url>     # 连通性 + 三端点契约字段校验（reset/ingest/query）
```

- 校验 `/reset` 2xx、`/ingest` 2xx（含 `input_tokens`/`llm_calls` 数值校验）、`/query` 2xx + JSON + 字符串 `answer`（`retrieved` 若出现须为字符串列表）。
- 底层实现为 `membench.doctor.check_target(base_url, timeout=10.0, api_key=None)`，只依赖标准库。

### run 元数据与可比性

- `run` 新增 `--driver-llm` / `--adapter-version`；`metrics.json` 记 `driver_llm` / `judge_model` / `adapter_version` / `dataset_fingerprint` / `schema_version`，用于回溯"这次分数是用什么跑出来的"。
- `compare` 新增 `--require-comparable` / `--tag`：前者做可比性前置校验（关键维度不一致 → 不可比，禁止当 A/B），后者决定 `ab_<tag>_<ts>/` 目录名。
- 数据集指纹 `membench.data.schema.dataset_fingerprint(path) -> {"sha256", "bytes", "name"}`；可用 `membench.data.info.resolve_available_path(key)` 解析某数据集当前可用路径（内置优先，其次 `datasets_dir()`）。

### `--shards` 的分片维度

- `--shards N --shard-index I` 配 `fuse` 融合同一评测的多个分片 run；`--shard-by` 决定分片维度。
- `--shard-by question`（默认）按**全局题号**分片（`i % N == I`），不含入选题的对话整段跳过——适合"少量超长对话承载大量题"的集，ingest 真省。
- `--shard-by dialogue` 按**对话整段**分片（`d % N == I`），分片之间 ingest 互不重复——"几乎每个对话都有题"的 ingest 密集型数据集（如 office、BEAM-1M）**必须用这个**；用题目级分片会让每个分片仍对全部对话 ingest，成本 ×N 反而更慢。
- 两种维度都能用 `fuse` 融合，但**不许混维融合**（fuse 会明确报错）。

## 结果解读准则

- 报告中的比率型指标以 **"评价（百分比）"** 形式呈现（如 `优秀（92.3%）`），四级：优秀 / 良好 / 中等 / 较差；报告开头有 **综合评级**（六项质量指标连续点数均值，1.0~4.0，档内随数值线性变化，满分 4.0）。效率成本指标（延迟/token/调用次数）依赖部署环境，不参与评级；**诊断指标**（检索召回率@5 / 过度拒答率）仅展示与归因、不参与评级。各数据集特有的分数参照区间用 `dataset info <key>` 查询。
- 办公 Agent 的**一票否决指标是 `stale_rate`**（陈旧召回，目标 ≤0.03）；红线指标是 `abstention_accuracy`（PII/凭证拒答，目标 =1.0）。
- 先看 report.md 的"失败样例 Top10"定位短板题型，再决定优化方向；每次优化只改一个变量。
- 背景知识与指标方法论见 `docs/Agent记忆系统深度研究报告.md` 第 4 章。
