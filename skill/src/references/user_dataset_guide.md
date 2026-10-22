# 用户测试集构造指南

> 触发纪律：**仅在用户显式要求时执行**（"用我的真实数据构造测试集"）；删除同理，且删除前必须向用户确认。

## 输入：用户真实对话日志

接受常见格式：JSONL/JSON 对话导出、IM 聊天记录导出、客服工单流转记录。最低要求字段：消息文本、发言人、时间戳、会话边界（session/工单 id）。

## 构造流程

1. **清洗**：去掉系统消息、按钮事件、空消息；按 session 边界分组；按时间排序。
2. **筛选候选片段**：优先选择包含以下信号的对话——
   - 事实陈述（"截止日是 X""负责人是 Y""预算是 Z 万"）
   - 事实更新（"改成""推迟到""换了"）→ knowledge_update 题
   - 偏好表达（"以后都…""我习惯…"）→ preference 题
   - 时间锚（"昨天""下周三""上次会议"）→ temporal 题
   - 敏感信息（密码、证件号、薪酬）→ abstention 题（**原文绝不进 evidence**）
3. **出题**：参照内置集 `membench/membench/data/office_dataset.jsonl` 的题型配比（single_hop 约 40%，multi_hop 20%，knowledge_update ≥10%，temporal ≥10%，preference ≥10%，abstention ≥5%）。
4. **字段规则**（schema 强制）：
   - `timestamp` 写 **ISO 8601、精度按原始数据**（`2026-04-06T09:00` 或只到日 `2026-04-06`；不要为了"统一"补时分秒，也不要写中文/散文时间）——公开数据集已统一成这个口径，自建集跟上，适配器只需一套解析；
   - `evidence` 必须是对话原文的**逐字子串**；
   - knowledge_update 题必须填 `gold_stale`（旧值），且旧值必须出现在对话原文中；
   - abstention 题 `gold_answer` 固定为 `<ABSTAIN>`，`evidence` 为空；
   - `qid` 全局唯一（建议 `{dialogue_id}_q{n}`）。
5. **自检**（构造后必须运行）：
   ```python
   from membench.data.schema import load_dataset
   ds = load_dataset("user_dataset_draft.jsonl")   # schema 校验
   # 逐题检查 evidence/gold_stale 为原文子串（脚本见 membench README FAQ）
   ```
6. **安装**：
   ```bash
   MEMBENCH_USER_DATA_DIR=<skill目录>/user_dataset \
   python -m membench.cli user-data install user_dataset_draft.jsonl
   ```

## 隐私红线

- 真实 PII（手机号、身份证、薪酬数字）**替换为占位符后再出题**（如 `138****1234` → `13800001234`），保结构不保真值；
- 敏感信息只允许以 abstention 题形式存在（考"不该记住"），绝不允许作为 gold_answer 被复述；
- 用户测试集留在本地 `user_dataset/` 目录，不打包、不上传。

## 删除

用户显式要求后，先确认再执行：

```bash
MEMBENCH_USER_DATA_DIR=<skill目录>/user_dataset python -m membench.cli user-data delete
```
