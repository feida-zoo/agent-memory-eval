"""数据集注册表与介绍：membench 支持的全部评测数据集的元信息。

提供两类查询：

- 概览：``list_datasets()`` / ``describe_all()`` —— 列举所有数据集并逐一简介；
- 详情：``get_dataset_info(key)`` / ``describe(key)`` —— 单个数据集的详细介绍，
  含测试集来源、特点、基准线（看到评分如何判断好坏）与接入状态。

对 ``status="builtin"`` 且文件确实存在的条目，``describe()`` 会附带用
``schema.load_dataset`` 统计出的实际 dialogue/question 数（异常时静默跳过）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: 内置数据集所在目录（membench/data/）。
_DATA_DIR = Path(__file__).resolve().parent

#: status 取值 -> 中文标签。
_STATUS_LABELS: dict[str, str] = {
    "builtin": "内置",
    "fetch": "需按需下载",
    "unavailable": "暂不可用",
}


@dataclass
class DatasetInfo:
    """单个数据集的元信息。

    Attributes:
        key: 数据集唯一键（CLI 与注册表中使用）。
        name: 显示名。
        builtin_file: 内置 jsonl 文件名（位于 membench/data/ 下），无内置文件时为 None。
        status: 接入状态，"builtin" 内置 / "fetch" 需按需下载 / "unavailable" 暂不可用。
        source: 测试集来源（含论文/仓库链接）。
        scale: 规模描述。
        features: 测试集特点（题型覆盖与局限）。
        baseline: 基准线说明（看到评分如何判断好坏）。
        usage: 使用示例命令。
    """

    key: str
    name: str
    builtin_file: str | None
    status: str
    source: str
    scale: str
    features: str
    baseline: str
    usage: str


#: 全部支持的数据集注册表（key -> DatasetInfo）。
DATASET_REGISTRY: dict[str, DatasetInfo] = {
    "office": DatasetInfo(
        key="office",
        name="内置办公域黄金集",
        builtin_file="office_dataset.jsonl",
        status="builtin",
        source="membench 自建合成（面向通用办公助手类 agent）。",
        scale="20 对话 / 86 题 / 7 类题型全覆盖。",
        features=(
            "含 gold_stale 标注，可测 stale_rate（陈旧召回）；含 abstention 题；"
            "规模小、跑一轮快，适合日常回归与优化迭代。"
        ),
        baseline=(
            "项目验收线 accuracy≥0.85、stale_rate≤0.03（办公场景红线）、"
            "abstention_accuracy≥0.80、evidence_recall≥0.90。"
        ),
        usage=(
            "python -m membench.cli run --adapter <你的适配器>"
            "（run 不传 --dataset 时默认回退到该内置数据集）"
        ),
    ),
    "sample": DatasetInfo(
        key="sample",
        name="冒烟测试迷你集",
        builtin_file="sample_dataset.jsonl",
        status="builtin",
        source="membench 自建合成。",
        scale="极小规模（数段对话 / 数道题）。",
        features="冒烟测试用迷你集，仅用于验证管线连通性，不用于评估结论。",
        baseline="不设基准线：能完整跑通 ingest/检索/判分管线即视为通过，分数无参考意义。",
        usage="python -m membench.cli run --adapter mock-strong --dataset membench/data/sample_dataset.jsonl",
    ),
    "locomo": DatasetInfo(
        key="locomo",
        name="LoCoMo 超长多会话对话",
        builtin_file=None,
        status="fetch",
        source=(
            "LoCoMo（Maharana et al., ACL 2024，arXiv:2402.17753，Snap Research）。"
        ),
        scale="10 段超长多会话对话 / 1986 题。",
        features=(
            "平均每段约 300 turns / 9k token / 35 sessions；"
            "题型含 single-hop/multi-hop/temporal/open-domain/adversarial"
            "（已映射到 membench 7 类）。"
        ),
        baseline=(
            "人类上限 F1≈87.9%；原论文 GPT-4 仅 32.1 F1（token-overlap 口径，偏严）；"
            "近年记忆系统整体 ACC 参考区间 57%~77%（Naive RAG≈63.6%、Mem0≈68.4%、"
            "TSM≈76.7%，arXiv:2601.07468）；厂商自报最高 92.5% 但口径不一"
            "（judge/检索深度不同）需谨慎对比；另有独立审计指出其答案键约 6.4% 错误率，"
            "绝对分仅供参考，重点看相对提升。"
        ),
        usage="python -m membench.cli run --adapter <你的适配器> --dataset ~/.membench/datasets/locomo.jsonl（先 dataset fetch locomo）",
    ),
    "longmemeval_oracle": DatasetInfo(
        key="longmemeval_oracle",
        name="LongMemEval-Oracle 证据会话问答",
        builtin_file=None,
        status="fetch",
        source=(
            "LongMemEval（Wu et al., ICLR 2025，arXiv:2410.10813，UCLA），"
            "oracle 变体（每题只给证据会话，隔离检索噪声）。"
        ),
        scale="500 题。",
        features=(
            "覆盖信息抽取/多会话推理/时间推理/知识更新/偏好 5 类；"
            "evidence 取自证据会话原文（完整保留、不截断，按 session 级口径做 containment 匹配）；"
            "会话按 haystack_dates 升序注入；question_date 保留并追加在问题末尾；"
            "knowledge-update 无过期答案标注，stale_rate 恒 0。"
        ),
        baseline=(
            "oracle 条件下 GPT-4o 约 87~92%、Llama-3.1-8B 约 71%；作为参照，完整 S 变体"
            "（约 115k token 历史）上 GPT-4o 全文读约 60.6~64%、商用在线助手 "
            "ChatGPT(GPT-4o) 57.7%、Coze 32.9%——oracle 分数显著高于这些是预期，"
            "低于 70% 说明回答/判分管线有问题。"
        ),
        usage="python -m membench.cli run --adapter <你的适配器> --dataset ~/.membench/datasets/longmemeval_oracle.jsonl（先 dataset fetch longmemeval_oracle）",
    ),
    "longmemeval_s": DatasetInfo(
        key="longmemeval_s",
        name="LongMemEval-S 长历史检索压力测试",
        builtin_file=None,
        status="fetch",
        source=(
            "LongMemEval（Wu et al., ICLR 2025，arXiv:2410.10813，UCLA），"
            "S 变体（每题约 50 个历史会话、约 115k token，含真实干扰会话）。"
        ),
        scale="500 题，每题 ~50 sessions / ~115k token。",
        features=(
            "与 oracle 同题但提供完整干扰历史，测的是检索/精确率而非纯回答能力；"
            "evidence 已从 answer_session_ids 解析为 session 级原文（非空）；"
            "knowledge-update 无过期答案标注，stale_rate 恒 0；"
            "体积大（~268MB），ingest 成本高，适合阶段性评估而非每次回归。"
        ),
        baseline=(
            "S 变体上 GPT-4o 全文读约 60.6~64%、商用在线助手 ChatGPT(GPT-4o) 57.7%、"
            "Coze 32.9%；与 oracle 变体的分差（oracle - S）≈ 检索/精确率不足的损失，"
            "分差越小说明记忆检索越精准。"
        ),
        usage="python -m membench.cli run --adapter <你的适配器> --dataset ~/.membench/datasets/longmemeval_s.jsonl（先 dataset fetch longmemeval_s）",
    ),
    "beam_1m": DatasetInfo(
        key="beam_1m",
        name="BEAM 1M 单对话压力测试",
        builtin_file=None,
        status="fetch",
        source=(
            "BEAM（Tavakoli et al. 2025，arXiv:2510.27246，"
            "https://github.com/mohammadtavakoli78/BEAM）。"
        ),
        scale="35 个 chat × 约 1M token/个，每 chat 20 题（10 类题型 × 2）共约 700 题。",
        features=(
            "百万 token 级单对话压力测试；题型含 abstention/contradiction_resolution/"
            "event_ordering/information_extraction/instruction_following/knowledge_update/"
            "multi_session_reasoning/preference_following/summarization/temporal_reasoning"
            "（已映射 membench 7 类）；turn 级 time_anchor 时间锚点；"
            "ingest 成本高（35×1M token 写入），注意时间与 API 费用。"
        ),
        baseline=(
            "无官方榜单；Mem0 自报 1M 集 64.1%；原论文显示长上下文模型随对话变长显著退化、"
            "其方法 LIGHT 较最优基线提升 3.5~12.69 个百分点——"
            "建议以自身优化前后对比为主，跑通即有价值。"
        ),
        usage="python -m membench.cli dataset fetch beam_1m",
    ),
    "beam_10m": DatasetInfo(
        key="beam_10m",
        name="BEAM 10M 全集终验",
        builtin_file=None,
        status="fetch",
        source=(
            "BEAM（Tavakoli et al. 2025，arXiv:2510.27246，"
            "https://github.com/mohammadtavakoli78/BEAM）。"
        ),
        scale="10 个 chat × 约 10M token/个，约 200 题，全集。",
        features=(
            "千万 token 级，目前公开记忆基准中最大量级；"
            "ingest 成本极高（串行写入以小时计），"
            "建议先用 beam_1m 迭代、定型后跑 10m 终验。"
        ),
        baseline=(
            "无官方榜单；Mem0 自报 10M 集 48.6%——在该量级 50% 上下已是可报道水平。"
        ),
        usage="python -m membench.cli dataset fetch beam_10m",
    ),
    "gatemem_office": DatasetInfo(
        key="gatemem_office",
        name="GateMem 办公三维记忆基准",
        builtin_file=None,
        status="fetch",
        source="GateMem 办公场景记忆基准（arXiv:2606.18829，github.com/rzhub/GateMem；episodes + checkpoints 结构，多 principal 办公环境）。",
        scale="17 episodes / 547 检查点（utility 154 + privacy 171 + safety 222）。",
        features=(
            "唯一同时覆盖 utility（该答）/privacy（该拒，按 asker 身份）/safety"
            "（删除后不该记起）三维的办公记忆基准；privacy/safety 题已映射为 abstention。"
            "单用户适配：内置集中 140 道 privacy-refuse 题（多用户权限语义）已标记 "
            "ignore=true，评测引擎自动跳过不计分；保留 safety-no_memory 222 题"
            "（删除后遗忘，单用户同样适用）与 utility 154 题。"
            "**时点语义**：每个 checkpoint 的 as_of_turn_id 转成 as_of（已 ingest 的"
            "消息条数），episode 标 ingest_strategy=interleaved——评测边 ingest 边提问，"
            "问的是「当时能看到什么」；因此需要 --shard-by dialogue，且不支持 --replay。"
            "加 --trace-state 可逐 as_of 导出状态，产出失败归因（没写入 / 没检索到 / "
            "检索到但答错）。"
            "判分：checkpoint 的 judge_spec（include 正则 / not_include / leak_targets）"
            "按确定性 rubric 执行，泄露 leak_targets 记为红线泄露；"
            "局限：asker 身份感知未建模；evidence 为空（无法逐题判断证据是否写入，"
            "归因相关题记 unknown）。"
        ),
        baseline=(
            "无公开基线；理想系统 utility 类 accuracy→100% 且 safety 类 "
            "abstention_accuracy→100%；安全类任何泄露（删除了还答得出）都比 "
            "utility 低分更严重，优先看 abstention_accuracy。"
        ),
        usage="python -m membench.cli run --adapter <你的适配器> --dataset ~/.membench/datasets/gatemem_office.jsonl（先 dataset fetch gatemem_office）",
    ),
    "statemembench": DatasetInfo(
        key="statemembench",
        name="StateMemBench 状态记忆基准",
        builtin_file=None,
        status="unavailable",
        source="StateMemBench（arXiv:2608.19652）。",
        scale="234 场景 / 322 探针。",
        features=(
            "专注状态记忆与陈旧信息检测；截至 2026-09 未发现公开数据与代码，需向作者索取；"
            "获取后可按 membench.data.convert_* 模式编写转换器接入。"
        ),
        baseline="暂无公开基准线（数据未公开，尚无法跑分）。",
        usage="暂不可用；获取数据后编写转换器接入。",
    ),
}


def list_datasets() -> list[DatasetInfo]:
    """按注册顺序返回全部数据集元信息。"""
    return list(DATASET_REGISTRY.values())


def get_dataset_info(key: str) -> DatasetInfo:
    """按 key 取数据集元信息。

    Args:
        key: 数据集键。

    Returns:
        DatasetInfo: 对应数据集的元信息。

    Raises:
        KeyError: key 未知，错误信息中列出全部可用 key。
    """
    try:
        return DATASET_REGISTRY[key]
    except KeyError:
        available = ", ".join(DATASET_REGISTRY)
        raise KeyError(f"未知数据集: {key}。可用 key: {available}") from None


def builtin_path(key: str) -> Path | None:
    """返回内置数据集文件的绝对路径；非内置条目返回 None。"""
    info = get_dataset_info(key)
    if info.builtin_file is None:
        return None
    return _DATA_DIR / info.builtin_file


def datasets_dir() -> Path:
    """按需下载数据集的落盘目录（``MEMBENCH_DATASETS_DIR`` > ``~/.membench/datasets``）。"""
    env = os.environ.get("MEMBENCH_DATASETS_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".membench" / "datasets"


def resolve_available_path(key: str) -> Path | None:
    """解析某数据集当前可用的文件路径（内置文件优先，其次按需下载目录）。

    查找顺序：

    1. ``membench/data/<builtin_file>``（若注册表标了内置文件名且文件存在）；
    2. ``datasets_dir()/<key>.jsonl``（``dataset fetch`` 的落盘位置）。

    Args:
        key: 数据集键。

    Returns:
        可用文件路径；两处都不存在时返回 None（调用方据此提示先 ``dataset fetch``）。

    Raises:
        KeyError: key 未知。
    """
    info = get_dataset_info(key)
    if info.builtin_file is not None:
        path = _DATA_DIR / info.builtin_file
        if path.is_file():
            return path
    fetched = datasets_dir() / f"{key}.jsonl"
    if fetched.is_file():
        return fetched
    return None


def infer_dataset_source(path: str | Path) -> "str | None":
    """按**文件路径**反推数据集来源标签；认不出来时返回 ``None``。

    为什么需要它：``dataset fetch`` 的公开基准集路径一旦被显式传给 CLI
    （``--dataset ~/.membench/datasets/locomo.jsonl``，或编排器 ``--common``
    先把清单解析成路径再逐个下发），解析层看到的是"一个存在的文件"，默认会
    记成 ``user-specified``——报告里于是把公开基准集标成"用户自定义"。

    判定顺序：

    1. 路径等于某条目的内置文件（``membench/data/<builtin_file>``）→ ``office``
       记 ``builtin-office``，其余记 ``builtin``；
    2. 路径位于 :func:`datasets_dir` 且文件名 stem 命中注册表里的 ``fetch``
       条目 → ``fetch``；
    3. 其余 → ``None``（调用方按 ``user-specified`` 兜底）。

    Args:
        path: 数据集 JSONL 路径（相对路径按当前工作目录解析）。

    Returns:
        来源标签或 ``None``。
    """
    try:
        resolved = Path(path).expanduser().resolve()
    except OSError:
        return None
    for info in DATASET_REGISTRY.values():
        if info.builtin_file is None:
            continue
        if resolved == (_DATA_DIR / info.builtin_file).resolve():
            return "builtin-office" if info.key == "office" else "builtin"
    fetch_dir = datasets_dir().resolve()
    if resolved.parent == fetch_dir:
        info = DATASET_REGISTRY.get(resolved.stem)
        if info is not None and info.status == "fetch":
            return "fetch"
    return None


def dataset_key_hint() -> str:
    """一行可用数据集 key 清单，按「能否直接使用」分组。

    分组口径：

    - 可直接用：:func:`resolve_available_path` 能解析到文件的 key（内置文件
      或已 ``dataset fetch``）；
    - 需先 dataset fetch：注册表标 ``status="fetch"`` 但本地还没有文件的 key；
    - 暂不可用：注册表标 ``status="unavailable"`` 的 key。

    Returns:
        形如 ``可直接用: office, sample；需先 dataset fetch: locomo, …`` 的字符串。
    """
    ready: list[str] = []
    fetchable: list[str] = []
    unavailable: list[str] = []
    for info in DATASET_REGISTRY.values():
        if resolve_available_path(info.key) is not None:
            ready.append(info.key)
        elif info.status == "fetch":
            fetchable.append(info.key)
        else:
            unavailable.append(info.key)
    parts = [f"可直接用: {', '.join(ready) or '（无）'}"]
    if fetchable:
        parts.append(f"需先 dataset fetch: {', '.join(fetchable)}")
    if unavailable:
        parts.append(f"暂不可用: {', '.join(unavailable)}")
    return "；".join(parts)


def _first_sentences(text: str, n: int) -> str:
    """取中文文本的前 n 句（按"。"切分，保留句号）。"""
    parts = [p for p in text.split("。") if p.strip()]
    head = parts[:n]
    return "。".join(head) + ("。" if head else "")


def _actual_stats(info: DatasetInfo) -> str | None:
    """对内置且文件存在的条目统计实际 dialogue/question 数；异常时静默返回 None。"""
    if info.status != "builtin" or info.builtin_file is None:
        return None
    path = _DATA_DIR / info.builtin_file
    if not path.is_file():
        return None
    try:
        from membench.data.schema import load_dataset

        dialogues = load_dataset(path)
        n_q = sum(len(d.questions) for d in dialogues)
        return f"{len(dialogues)} 对话 / {n_q} 题"
    except Exception:
        return None


def describe(key: str) -> str:
    """生成单个数据集的详细多行中文介绍。

    Args:
        key: 数据集键（未知 key 抛 KeyError，见 get_dataset_info）。

    Returns:
        str: 含名称/来源/规模/特点/基准线/接入状态/用法的多行介绍；
            内置且文件存在时附带实际统计。
    """
    info = get_dataset_info(key)
    status_label = _STATUS_LABELS.get(info.status, info.status)
    lines = [
        f"【{info.key}】{info.name}",
        f"接入状态: {status_label}（{info.status}）",
    ]
    if info.builtin_file is not None:
        lines.append(f"内置文件: membench/data/{info.builtin_file}")
        stats = _actual_stats(info)
        if stats is not None:
            lines.append(f"实际统计: {stats}")
    lines += [
        f"来源: {info.source}",
        f"规模: {info.scale}",
        f"特点: {info.features}",
        f"基准线（如何判断分数好坏）: {info.baseline}",
        f"用法: {info.usage}",
    ]
    return "\n".join(lines)


def describe_all() -> str:
    """生成全部数据集的概览：每个一段（名称 + status 标签 + 简介 + 基准线要点）。"""
    blocks: list[str] = []
    for info in DATASET_REGISTRY.values():
        status_label = _STATUS_LABELS.get(info.status, info.status)
        intro = _first_sentences(info.scale + info.features, 3)
        baseline_point = _first_sentences(info.baseline, 1)
        blocks.append(
            f"== {info.key} — {info.name} [{status_label}] ==\n"
            f"{intro}\n"
            f"基准线要点: {baseline_point}"
        )
    header = f"membench 支持的数据集（共 {len(DATASET_REGISTRY)} 个，详情用 dataset info <key> 查看）:"
    return header + "\n\n" + "\n\n".join(blocks)
