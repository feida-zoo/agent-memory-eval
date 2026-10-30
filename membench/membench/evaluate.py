"""evaluate：对多个数据集依次运行评测，跨数据集加权融合出基准评级报告。

流程：
    plan_datasets 规划数据集清单（办公域槽位 + 按需下载公开集 + BEAM 槽位）
    -> 建一个实验目录 ``<report_out>/eval_<adapter>_<platform>_<ts>/``
       （``task_dir`` 给定时容器即该已存在的任务目录，不再新建实验目录）
    -> 每个数据集用 Runner 各跑一次写进它的 ``runs/<dataset>/``
       （adapter 实例复用，reset 由 Runner 内部负责；write_task_reports=False）
    -> 读取各数据集 ``report/metrics.json``
    -> 融合六项质量指标（按能力标注决定参与集，按参与题数加权；其中四项参与
       基准评级）
    -> 办公域 stale 门禁（一票否决）
    -> 任务级报告写实验目录的 ``reports/``：report.md（全量评估报告）、
       metrics.json（任务级汇总）、evaluate_summary.json（机器可消费的清单）、
       fig_dataset_overview.png；报告内链接指向 ``runs/<dataset>/report/report.md``。

融合口径（基准集按题数加权）：
    - **办公域槽位（office）不参与融合评级**：它与基准集不同源，单列为
      「业务结论（办公域槽位）」（准确率 + stale / abstention 红线 + 门禁状态）。
      门禁照旧一票否决，独立于基准评级。槽位按**数据集身份**判定（内置 office /
      用户测试集）；纯汇总（``--from-runs`` 且未给 ``--dataset``）不按位置兜底——
      未含 office 产物时没有槽位，全部数据集参与基准融合，门禁 SKIP；
    - 参与条件（能力标注）：accuracy / token_f1 / bleu1 所有基准集参与；
      evidence_recall / recall_at_5 仅 has_evidence（任一题目有 evidence 标注）
      的数据集；stale_rate 仅 stale_n > 0（本次有可机检的 stale 题）的数据集；
      abstention_accuracy 仅 has_abstention（任一题目为拒答题：category 为
      abstention 或 gold 为拒答哨兵值）的数据集；
    - 权重 = ``min(该指标在该数据集的分母, FUSION_WEIGHT_CAP)``，分母按指标取
      :data:`METRIC_DENOMINATOR_KEYS`（题数 / 非拒答题数 / 拒答题数 / 可机检
      stale 题数）；旧 run 缺这些计数时回退 ``n_questions``。86 题的小集与 1986 题的大集
      不再同权，小样本集也不会靠"每指标一票"抬高综合分；
    - ``fusion="equal"``：基准集内等权（旧宏平均语义），仍排除办公域槽位。

"产物目录"（``--from-runs`` 的 ``KEY=DIR``）统一经 :func:`membench.layout.resolve_dataset_dir`
解析：入参可以是实验目录（含 ``runs/``）或数据集目录（``runs/<dataset>``），
解析结果落数据集目录，指标固定读 ``<数据集目录>/report/metrics.json``。
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Union

from membench import layout, runtime
from membench.data.info import get_dataset_info, resolve_available_path
from membench.data.schema import load_dataset, question_capabilities
from membench.layout import resolve_dataset_dir
from membench.ratings import format_metric, overall_rating
from membench.report import (
    _md_table,
    _fmt_opt,
    fig_dataset_overview,
    metric_label,
    metric_label_with_role,
    stale_rate_display,
    token_usage_totals,
)
from membench.runner import Runner
from membench.version import version_display
from membench.userdata import (
    USER_DATASET_FILENAME,
    builtin_office_dataset_path,
    resolve_dataset,
    resolve_dataset_arg,
)

#: 参与融合的六项质量指标（同一套加权口径；token_f1 / bleu1 为字面重合诊断，
#: 不参与基准评级）。
QUALITY_METRICS: tuple[str, ...] = (
    "accuracy",
    "token_f1",
    "bleu1",
    "evidence_recall",
    "abstention_accuracy",
    "stale_rate",
)

#: 「诊断指标（不参与评级）」小节的展示列：严格检索召回率@5 与过度拒答率。
#: 评级口径的诊断指标另见 :data:`membench.ratings.DIAGNOSTIC_METRICS`
#: （还含 token_f1 / bleu1——字面重合度）。
DIAGNOSTIC_METRICS: tuple[str, ...] = ("recall_at_5", "over_refusal_rate")

#: 融合指标集：六项质量指标 + 两项检索/拒答诊断指标（同一套加权口径）。
FUSED_METRICS: tuple[str, ...] = QUALITY_METRICS + DIAGNOSTIC_METRICS

#: 融合口径：``weighted`` = 按参与题数加权（默认）；``equal`` = 基准集内等权
#: （旧宏平均语义）。两种口径都排除办公域槽位，旧产物（缺 ``fusion``）归一为
#: ``"legacy"``（等权且 office 参与），三者互不可比。
FUSION_WEIGHTED = "weighted"
FUSION_EQUAL = "equal"
FUSION_CHOICES: tuple[str, ...] = (FUSION_WEIGHTED, FUSION_EQUAL)

#: 加权融合的单数据集权重封顶（题数）：1986 题的基准集不应把 86 题的集压成
#: 噪声，但也不该让一个巨型数据集独占权重。
FUSION_WEIGHT_CAP: int = 500

#: 指标 -> 权重分母在 ``overall`` 中的键。检索类看非拒答题数、拒答类看拒答题数、
#: 陈旧率看可机检 stale 题数——各指标的分母数据集不同，按总题数加权会把
#: 「只在 office 有 abstention」这类小样本放回高位。
METRIC_DENOMINATOR_KEYS: dict[str, str] = {
    "accuracy": "n_questions",
    "token_f1": "n_questions",
    "bleu1": "n_questions",
    "evidence_recall": "evidence_n",
    "recall_at_5": "evidence_n",
    "over_refusal_rate": "evidence_n",
    "abstention_accuracy": "abstention_n",
    "stale_rate": "stale_n",
}

#: 旧 run 缺的分母键（本特性之前只落 n_questions）-> 回退用它。
_DENOMINATOR_FALLBACK = "n_questions"

#: 办公域 stale 门禁阈值：办公域槽位数据集 stale_rate 超过该值即一票否决。
STALE_GATE_THRESHOLD: float = 0.03

#: 默认参与全量评估的按需下载公开集（键，顺序即评测顺序）。
#: 这些集没有内置文件，须先 dataset fetch 到下载目录，用统一入口解析路径。
_FETCH_DATASETS: tuple[str, ...] = (
    "locomo",
    "longmemeval_oracle",
    "gatemem_office",
)

#: LongMemEval-S：默认包含（--no-lme-s / --quick 剔除）。体积大（~268MB）、
#: ingest 成本高；与 oracle 同题，分差反映检索/精确率损失。
_LME_S_KEY = "longmemeval_s"


@dataclass
class DatasetSpec:
    """一个待评测数据集槽位。

    Attributes:
        key: 槽位标识（如 ``office`` / ``locomo`` / ``beam_1m``）。
        name: 显示名（用于报告）。
        path: 数据集 JSONL 路径。
        source: 来源标签（``user`` / ``builtin-office`` / ``builtin`` / ``fetch`` 等）。
        is_office_slot: 是否为办公域槽位（stale 门禁的作用对象）。
    """

    key: str
    name: str
    path: str
    source: str
    is_office_slot: bool = False


@dataclass
class DatasetResult:
    """单个数据集的评测结果。

    Attributes:
        spec: 数据集槽位。
        run_dir: 该数据集产物目录（``<实验目录>/runs/<dataset>``）。
        metrics: 该数据集 ``report/metrics.json`` 的内容。
        capabilities: 能力标注（has_evidence / has_stale / has_abstention）。
    """

    spec: DatasetSpec
    run_dir: str
    metrics: dict[str, Any]
    capabilities: dict[str, bool] = field(default_factory=dict)


def _fetch_spec(key: str) -> DatasetSpec:
    """解析按需下载公开集的可用路径；缺失时报错提示先运行 dataset fetch。"""
    path = resolve_available_path(key)
    if path is None:
        raise SystemExit(
            f"数据集 {key} 尚未下载，请先运行: "
            f"python -m membench.cli dataset fetch {key}"
        )
    return DatasetSpec(key=key, name=key, path=str(path), source="fetch")


def _beam_path(beam_10m: bool) -> Path:
    """BEAM 数据集路径（~/.membench/datasets/beam_1m.jsonl 或 beam_10m.jsonl）。"""
    name = "beam_10m.jsonl" if beam_10m else "beam_1m.jsonl"
    return Path.home() / ".membench" / "datasets" / name


def dataset_specs(values: list[str]) -> list[DatasetSpec]:
    """把 CLI ``--dataset`` 值（数据集 key 或 JSONL 路径）解析为评测清单。

    解析规则与 ``run --dataset`` 完全一致，共用
    :func:`membench.userdata.resolve_dataset_arg`；列表顺序即评测顺序：

    - 存在的文件路径 -> ``user-specified`` 收录，key 取文件名 stem；
    - ``office`` -> 办公域槽位（优先级：用户测试集 > 内置 office）；
    - 其余按数据集 key 经 ``resolve_available_path`` 解析（内置文件优先，
      其次 ``dataset fetch`` 下载目录）；解析不到报错提示先 fetch；
    - 办公域槽位（stale 门禁作用域）按**数据集身份**标出：身份命中办公域
      （内置 office / 用户测试集）的条目承担槽位，否则沿用既有口径由列表中第一个
      数据集承担（见 :func:`mark_office_slot`）。若该集没有可机检的
      stale 题，门禁自动 SKIP。

    Args:
        values: ``--dataset`` 原始字符串列表（key 或路径）。

    Returns:
        DatasetSpec 列表。

    Raises:
        SystemExit: 列表为空、含空值，或某个 key / 路径无法解析。
    """
    if not values:
        raise SystemExit("--dataset 需要至少一个数据集（key 或 JSONL 路径）")
    specs: list[DatasetSpec] = []
    for raw in values:
        resolved = resolve_dataset_arg(raw)
        if resolved.source in ("user", "builtin-office"):
            name = (
                "office（用户测试集）"
                if resolved.source == "user"
                else "office（内置办公黄金集）"
            )
        else:
            name = resolved.key
        specs.append(
            DatasetSpec(
                key=resolved.key,
                name=name,
                path=resolved.path,
                source=resolved.source,
            )
        )
    mark_office_slot(specs)
    return specs


def parse_run_kv(values: Union[list[str], None], option: str) -> dict[str, str]:
    """解析 ``KEY=RUN_DIR`` 形式的已有 run 参数（--from-runs / --baseline-run / --candidate-run）。

    Returns:
        ``{数据集 key: run 目录}``（保持输入顺序）。

    Raises:
        SystemExit: 不是 ``KEY=RUN_DIR`` 形式、key 重复或目录不存在。
    """
    out: dict[str, str] = {}
    for raw in values or []:
        text = str(raw)
        key, sep, run_dir = text.partition("=")
        key, run_dir = key.strip(), run_dir.strip()
        if not sep or not key or not run_dir:
            raise SystemExit(f"{option} 需为 KEY=RUN_DIR 形式，当前为 {text!r}")
        if key in out:
            raise SystemExit(f"{option} 的 key 重复: {key}")
        if not os.path.isdir(run_dir):
            raise SystemExit(f"{option} 的 run 目录不存在: {run_dir}（{key}）")
        out[key] = run_dir
    return out


def office_slot_dataset_stems() -> set[str]:
    """办公域槽位候选的数据集文件 stem。

    办公域槽位（stale 门禁的作用对象）由 :func:`membench.userdata.resolve_dataset`
    决定：用户测试集 > 内置 office 黄金集，故这里给的是该槽位的**全部候选身份**——
    内置办公集、用户测试集，以及 ``resolve_dataset(None)`` 当前解析到的那一份
    （用户测试集目录挂在别处时文件名不一定是 ``user_dataset.jsonl``）。

    判定"这份产物是不是办公域槽位"因此看**数据集身份**而不是 key 字面量：
    ``--dataset office``、``--dataset <office_dataset.jsonl 路径>`` 与
    ``--from-runs <key>=<产物目录>`` 三条路径得到同一结论。

    Returns:
        目录名安全的 stem 集合（经 :func:`membench.layout.slug` 归一）。
    """
    stems = {builtin_office_dataset_path().stem, Path(USER_DATASET_FILENAME).stem}
    current, _source = resolve_dataset(None)
    if current:
        stems.add(Path(current).stem)
    return {layout.slug(stem) for stem in stems}


def _identity_name(dataset_path: str, dataset_dir: str) -> str:
    """数据集身份名：优先取数据集文件 stem，缺路径（旧产物）时退回产物目录名。"""
    if dataset_path:
        return layout.slug(Path(dataset_path).stem)
    return layout.slug(os.path.basename(os.path.normpath(dataset_dir)))


def is_office_slot_dataset(dataset_path: str, dataset_dir: str = "") -> bool:
    """该数据集是否属于办公域槽位（stale 门禁作用域），按数据集身份判定。"""
    return _identity_name(dataset_path, dataset_dir) in office_slot_dataset_stems()


def mark_office_slot(
    specs: list[DatasetSpec], *, positional_fallback: bool = True
) -> None:
    """就地标出办公域槽位：数据集身份命中办公域（内置 office / 用户测试集）的优先。

    槽位判定看数据集身份而非 key 字面量，显式 ``office``、指向 office 数据集的路径、
    以及汇总模式复用的产物因此得到同一个结论；该集没有可机检的 stale 题时门禁自动 SKIP。

    身份不命中时是否由清单第一个数据集兜底，两条路径不同：

    - ``positional_fallback=True``（:func:`dataset_specs`，即显式 ``--dataset`` 清单）：
      身份全不命中时，由清单里**第一个用户自定义数据集**（``source="user-specified"``，
      即用户自己给的数据文件）承担槽位——业务数据要过 stale 门禁，这是它的本意。
      **公开基准集不兜底**：``--dataset locomo --dataset gatemem_office`` 这种列表的
      顺序只是书写顺序，按位置挑一个当办公域会把它踢出基准融合、还把它的 stale 当
      业务红线（历史 bug：locomo / gatemem 被误标成办公域槽位）。
    - ``positional_fallback=False``（:func:`evaluate` 的纯汇总分支）：清单由
      ``--from-runs`` 的产物拼成，顺序只反映书写顺序，与"谁是办公域"无关；按位置兜底
      会把第一个（如 locomo）当成槽位踢出基准融合、还把它的 stale 当业务门禁。没有
      办公域身份就没有槽位：全部数据集参与基准融合，门禁走 SKIP。

    Args:
        specs: 待标注的数据集槽位清单（就地修改 ``is_office_slot``）。
        positional_fallback: 身份全不命中时是否让第一个数据集承担槽位。
    """
    for spec in specs:
        spec.is_office_slot = is_office_slot_dataset(spec.path)
    if positional_fallback and not any(spec.is_office_slot for spec in specs):
        # 只让"用户自己的数据文件"兜底槽位；公开基准集不按位置挑（顺序不代表意图）
        for spec in specs:
            if spec.source == "user-specified":
                spec.is_office_slot = True
                break


def _capabilities_from_metrics(metrics: dict[str, Any], dataset_path: str) -> dict[str, bool]:
    """能力标注：优先 run 顶层落账（新 run）；缺失时扫描数据集；再缺按全 False。"""
    keys = ("has_evidence", "has_stale", "has_abstention")
    stored = {k: metrics.get(k) for k in keys}
    if all(isinstance(v, bool) for v in stored.values()):
        return dict(stored)  # type: ignore[arg-type]
    if dataset_path and os.path.isfile(dataset_path):
        try:
            return scan_capabilities(dataset_path)
        except (OSError, ValueError):
            pass
    return {k: False for k in keys}


def _dataset_path_of(dataset_dir: str) -> Union[str, None]:
    """数据集产物记录的数据集路径（判断两份产物是否同源，缺失返 None）。"""
    path = os.path.join(dataset_dir, layout.DIR_REPORT, layout.FILE_METRICS)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            metrics = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return str(metrics.get("dataset_path") or "") or None


def adopt_dataset_dir(experiment_dir: str, dataset_dir: str) -> str:
    """把复用的数据集产物并入本次实验目录（已在位时原样返回）。

    汇总模式（``--from-runs``）复用的产物可能来自别的实验目录；并入后本次实验
    目录自包含（报告内相对链接全部落在目录内），可整体搬移。目标目录名沿用来源
    数据集目录名（已是安全名）；同名但数据集不同的产物直接拒绝，避免互相串味。
    """
    source = os.path.abspath(dataset_dir)
    target = layout.dataset_dir(experiment_dir, os.path.basename(os.path.normpath(source)))
    if os.path.abspath(target) == source:
        return source
    existing = _dataset_path_of(target)
    if existing is not None and existing != _dataset_path_of(source):
        raise SystemExit(
            f"数据集目录名冲突：{target} 已是另一个数据集（{existing}），"
            f"无法并入 {source}——请用一个实验目录承载全部数据集"
        )
    shutil.copytree(source, target, dirs_exist_ok=True)
    print(f"[membench] 复用产物已并入本次实验目录: {source} -> {target}")
    return os.path.abspath(target)


def _expected_dataset_dir_names(key: str) -> Union[set[str], None]:
    """``key`` 期望的数据集产物目录名集合；key 不在数据集注册表里时返 None。

    产物目录名 = 数据集文件名去扩展名（:class:`membench.runner.Runner` 的落盘口径），
    故按注册表反推：内置条目取 ``builtin_file`` 的 stem（``office`` 期望
    ``office_dataset``——不做 key 与目录名的字符串比较），按需下载条目即 ``<key>.jsonl``。
    办公域槽位额外接受用户测试集那条通道（两个文件同属槽位身份）。
    """
    try:
        info = get_dataset_info(key)
    except KeyError:
        return None
    if info.builtin_file is None:
        return {layout.slug(key)}
    stem = layout.slug(Path(info.builtin_file).stem)
    office_stems = office_slot_dataset_stems()
    # 内置办公集与用户测试集都满足办公域槽位，两条通道都算命中
    return office_stems if stem in office_stems else {stem}


def _assert_dataset_identity(key: str, dataset_dir: str) -> None:
    """校验 ``KEY=DIR`` 的 key 与实际解析出的数据集产物是不是同一个数据集。

    :func:`membench.layout.resolve_dataset_dir` 在 key 提示不命中时**回退**到实验目录下
    唯一的数据集子目录（回退本身是契约的一部分），于是"key 认错"不会报错，只会把 A
    数据集的分数记到 B 名下。这里补一层身份校验：能确定 key 期望的目录名时必须一致，
    否则直接拒绝；key 不在注册表里（自定义 key）无法判断，只在目录名与 key 不一致时
    打显著警告。

    Raises:
        SystemExit: 实际数据集目录名与 key 期望的目录名不一致。
    """
    actual = layout.slug(os.path.basename(os.path.normpath(dataset_dir)))
    expected = _expected_dataset_dir_names(key)
    if expected is None:
        if actual != layout.slug(key):
            print(
                f"[membench] ⚠️ 汇总模式：key {key!r} 不在数据集注册表里，无法核对身份，"
                f"已采用 {actual!r}（{dataset_dir}）——不是想要的数据集请改 KEY"
            )
        return
    if actual not in expected:
        names = "、".join(sorted(expected))
        raise SystemExit(
            f"--from-runs {key} 与实际产物不符：key 期望的数据集目录是 {names}，"
            f"实际解析到 {actual}（{dataset_dir}）——key 认错会把别的数据集的分数记到 "
            f"{key} 名下；请把 KEY 改成实际数据集名，或把入参指向正确的产物目录"
        )


def load_provided_result(key: str, product_dir: str) -> DatasetResult:
    """从现成产物目录（实验目录或数据集目录）加载 DatasetResult（汇总模式复用）。

    key 与产物身份先经 :func:`_assert_dataset_identity` 核对（回退解析不会报错，
    所以必须在这里拦下认错的 key）；办公域槽位按产物身份标注（:func:`is_office_slot_dataset`），
    使纯汇总模式与 ``--dataset`` 路径的门禁结论一致。
    """
    dataset_dir = resolve_dataset_dir(product_dir, key)
    _assert_dataset_identity(key, dataset_dir)
    with open(
        os.path.join(dataset_dir, layout.DIR_REPORT, layout.FILE_METRICS), encoding="utf-8"
    ) as f:
        metrics = json.load(f)
    dataset_path = str(metrics.get("dataset_path") or "")
    source = str(metrics.get("dataset_source") or "user-specified")
    if key == "office":
        name = "office（用户测试集）" if source == "user" else "office（内置办公黄金集）"
    else:
        name = key
    spec = DatasetSpec(
        key=key,
        name=name,
        path=dataset_path,
        source=source,
        is_office_slot=is_office_slot_dataset(dataset_path, dataset_dir),
    )
    return DatasetResult(
        spec=spec,
        run_dir=dataset_dir,
        metrics=metrics,
        capabilities=_capabilities_from_metrics(metrics, dataset_path),
    )


#: 汇总模式的「提示型」不一致维度：被测对象与评测口径维度（:data:`_IDENTITY_KEYS`）
#: 混用已在跑前直接拒绝，这里只剩 platform 这类允许跨 run 差异、仅需标注的维度；
#: commit 是 A/B 的被测变量，不在此列。
_META_CHECK_KEYS = ("platform", "membench_hash")


def _membench_hash(metrics: dict[str, Any]) -> Union[str, None]:
    """评测代码版本（``membench_version.hash``）；旧产物没有该字段时为 None。"""
    info = metrics.get("membench_version")
    return info.get("hash") if isinstance(info, dict) else None


#: 同一侧/同一报告内必须唯一的维度：被测对象身份 + 评测口径（A/B 差异只应发生在侧之间）。
_IDENTITY_KEYS = (
    "adapter",
    "adapter_version",
    "driver_llm",
    "judge",
    "judge_model",
    "retrieval_match",
    "retrieval_match_threshold",
    # 运行模式：full / replay-*。回放固定写入侧，与全量跑**不可放在同一次汇总里**
    # （分数不可归因）；缺省视为 full（本特性之前的所有 run 都是全量）。
    "run_mode",
)


def _identity_value(metrics: dict[str, Any], key: str) -> Any:
    """身份/口径维度的取值（``run_mode`` 缺省归一为 ``"full"``）。

    本特性之前的所有 run 都是全量跑，旧 run 未落账 ``run_mode`` 时按 full 计——
    否则"旧 run + 新 run"会被误判成混用两种模式。
    """
    if key == "run_mode":
        return metrics.get(key) or "full"
    return metrics.get(key)


def assert_single_identity(results: list[DatasetResult], label: str) -> None:
    """校验一侧（或单臂汇总）内的被测对象与评测口径唯一，防止混入多个系统/版本/仪器。

    Raises:
        SystemExit: :data:`_IDENTITY_KEYS` 中任一维度出现多组取值。
    """
    seen: dict[tuple, list[str]] = {}
    for r in results:
        key = tuple(_identity_value(r.metrics, k) for k in _IDENTITY_KEYS)
        seen.setdefault(key, []).append(r.spec.key)
    if len(seen) > 1:
        detail = "；".join(
            f"{dict(zip(_IDENTITY_KEYS, k))} ← {','.join(keys)}" for k, keys in seen.items()
        )
        raise SystemExit(
            f"{label}混合了多个被测对象身份或评测口径"
            f"（{' / '.join(_IDENTITY_KEYS)}）：{detail}——同一侧必须固定一个被测对象与"
            "评测口径；请统一 run 来源或改用同一参数补跑（差异变量只应发生在两侧之间）"
        )


def expected_fresh_metadata(
    adapter: Any,
    judge: Any,
    driver_llm: Union[str, None],
    adapter_version: Union[str, None],
    matcher: Any = None,
) -> dict[str, Any]:
    """补跑将写入的（被测对象 + 评测口径）字段，用于与已有 run 预对齐。"""
    if judge is None:
        judge_name: Union[str, None] = None
    elif isinstance(judge, str):
        judge_name = judge
    else:
        judge_name = str(getattr(judge, "name", judge.__class__.__name__))
    return {
        "adapter": (
            str(getattr(adapter, "name", adapter.__class__.__name__))
            if adapter is not None
            else None
        ),
        "adapter_version": adapter_version,
        "driver_llm": driver_llm,
        "judge": judge_name,
        "judge_model": getattr(judge, "model", None) if judge is not None else None,
        "retrieval_match": (
            str(getattr(matcher, "name", None) or "substring") if matcher is not None else "substring"
        ),
        "retrieval_match_threshold": (
            getattr(matcher, "threshold", None) if matcher is not None else None
        ),
        # 补跑出的是全量 run；与回放 run 混用会在跑前被这条拒绝
        "run_mode": "full",
    }


def assert_gap_matches_provided(
    provided: dict[str, DatasetResult], expected: dict[str, Any], label: str
) -> None:
    """补跑前预检：新 run 将写入的字段必须与已有 run 一致（不一致在跑之前就拒绝）。"""
    if not provided:
        return
    sample = next(iter(provided.values()))
    diffs = [
        f"{key}: 现有 run={_identity_value(sample.metrics, key)!r}"
        f" vs 本轮补跑={expected.get(key)!r}"
        for key in _IDENTITY_KEYS
        if _identity_value(sample.metrics, key) != expected.get(key)
    ]
    if diffs:
        raise SystemExit(
            f"{label}补跑参数与已有 run 不一致，融合后将不可归因：" + "；".join(diffs)
            + "——请让补跑参数与已有 run 保持一致（或全部重跑）"
        )


def metadata_inconsistencies(results: list[DatasetResult]) -> list[str]:
    """跨数据集元数据不一致清单（汇总模式提示"混 run"风险）。

    Returns:
        每个不一致维度一条说明（``维度: 值A ← 数据集x、y；值B ← 数据集z``）。
    """
    findings: list[str] = []
    for key in _META_CHECK_KEYS:
        grouped: dict[str, list[str]] = {}
        for r in results:
            value = _membench_hash(r.metrics) if key == "membench_hash" else r.metrics.get(key)
            grouped.setdefault(repr(value), []).append(r.spec.key)
        if len(grouped) > 1:
            detail = "；".join(
                f"{value} ← {','.join(keys)}" for value, keys in grouped.items()
            )
            findings.append(f"{key}: {detail}")
    return findings


def plan_datasets(
    quick: bool = False,
    beam_10m: bool = False,
    no_beam: bool = False,
    no_lme_s: bool = False,
    common: bool = False,
) -> list[DatasetSpec]:
    """规划全量评估的数据集清单。

    默认槽位顺序：办公域槽位（resolve_dataset 优先级：用户集 > 内置 office）
    -> 按需下载公开集（locomo / longmemeval_oracle / gatemem_office）
    -> longmemeval_s（检索压力变体，默认包含）-> BEAM-1M。

    ``quick=True`` 等价 ``no_beam=True + no_lme_s=True``：跳过 BEAM 与
    LongMemEval-S 两个重型数据集，适合分钟级日常回归。

    ``common=True`` 是"公开基准常用集"：只留按需下载的公开基准集
    （locomo / longmemeval_oracle / gatemem_office）与 BEAM，**不构造办公域
    槽位**（office / 用户测试集）也**不含 LongMemEval-S**——公开基准口径不掺
    业务集与检索压力变体。与 ``quick`` 互斥（两者清单口径不同）；
    ``no_beam`` / ``beam_10m`` 照常作用，``no_lme_s`` 在 common 下无效果
    （LongMemEval-S 本就不在其中）。

    所有按需下载公开集都没有内置文件，统一经 ``data.info.resolve_available_path``
    解析可用路径（内置文件优先，其次 dataset fetch 下载目录）；解析不到即
    报错提示先下载，避免拿"路径不存在"的空集当基准、污染可比性。

    Args:
        quick: True 时跳过 BEAM 与 LongMemEval-S 槽位（不检查对应文件）。
        beam_10m: True 时 BEAM 槽位使用 beam_10m.jsonl（与 no_beam/quick 互斥）。
        no_beam: True 时不评测 BEAM 槽位。
        no_lme_s: True 时剔除 LongMemEval-S 变体（common 下本就无此变体）。
        common: True 时只评测公开基准常用集（locomo / longmemeval_oracle /
            gatemem_office / BEAM），不含办公域槽位与 LongMemEval-S；
            与 quick 互斥。

    Returns:
        DatasetSpec 列表。

    Raises:
        SystemExit: common 与 quick 同传；beam_10m 与 no_beam/quick 同传；
            任一按需下载公开集解析不到路径；未跳过 BEAM 时 BEAM 文件不存在
            （提示先 dataset fetch）。
    """
    if common and quick:
        raise SystemExit(
            "--common 与 --quick 互斥（公开基准常用集与快速模式的数据集清单不同）"
        )
    if beam_10m and (no_beam or quick):
        raise SystemExit(
            "--beam-10m 与 --no-beam / --quick 互斥（BEAM 槽位状态冲突）"
        )
    specs: list[DatasetSpec] = []

    # 办公域槽位：复用 resolve_dataset 的优先级（用户集 > 内置 office）。
    # common 只面向公开基准集，不构造该槽位（报告的业务结论小节据此显示跳过）。
    if not common:
        office_path, office_source = resolve_dataset(None)
        office_name = (
            "office（用户测试集）" if office_source == "user" else "office（内置办公黄金集）"
        )
        specs.append(
            DatasetSpec(
                key="office",
                name=office_name,
                path=office_path,
                source=office_source,
                is_office_slot=True,
            )
        )

    # 按需下载的公开集（缺失即报错，不静默跳过）。
    for key in _FETCH_DATASETS:
        specs.append(_fetch_spec(key))

    # LongMemEval-S：默认包含（检索/精确率压力变体）；quick / no_lme_s / common 剔除。
    if not (common or quick or no_lme_s):
        specs.append(_fetch_spec(_LME_S_KEY))

    # BEAM 槽位：默认 beam_1m；beam_10m 切换；no_beam / quick 跳过。
    if not (quick or no_beam):
        beam_key = "beam_10m" if beam_10m else "beam_1m"
        beam_path = _beam_path(beam_10m)
        if not beam_path.is_file():
            raise SystemExit(
                f"BEAM 数据集不存在: {beam_path}。"
                f"请先运行 python -m membench.cli dataset fetch {beam_key}，"
                "或使用 --no-beam / --quick 跳过 BEAM 槽位。"
            )
        specs.append(
            DatasetSpec(
                key=beam_key,
                name=beam_key,
                path=str(beam_path),
                source="fetch",
            )
        )
    return specs


def scan_capabilities(path: Union[str, Path]) -> dict[str, bool]:
    """扫描数据集题目，得到能力标注。

    ignore=true 的题不参与评分，扫描时排除——能力标注基于实际参与
    评分的题判断（例如全部拒答题都被忽略时 has_abstention=False）。

    Returns:
        ``{"has_evidence": ..., "has_stale": ..., "has_abstention": ...}``
        （口径见 :func:`membench.data.schema.question_capabilities`，均不含忽略题）。
    """
    dialogues = load_dataset(path)
    return question_capabilities(q for dialogue in dialogues for q in dialogue.questions)


#: 指标 -> 参与条件（capabilities key）；None 表示所有数据集总是参与。
_PARTICIPATION_RULES: dict[str, Union[str, None]] = {
    "accuracy": None,
    "token_f1": None,
    "bleu1": None,
    "evidence_recall": "has_evidence",
    "stale_rate": "has_stale",
    "abstention_accuracy": "has_abstention",
    "recall_at_5": "has_evidence",
    "over_refusal_rate": None,
}


def participates(capabilities: dict[str, bool], metric: str) -> bool:
    """判断某数据集是否参与某指标的融合。"""
    rule = _PARTICIPATION_RULES[metric]
    if rule is None:
        return True
    return bool(capabilities.get(rule, False))


def is_baseline_dataset(res: DatasetResult) -> bool:
    """该数据集是否参与基准评级（办公域槽位不参与，单列业务结论）。"""
    return not res.spec.is_office_slot


def pool_official(results: list["DatasetResult"]) -> Union[dict[str, Any], None]:
    """按**分子 / 分母**池化 GateMem 官方三支柱（跨数据集）。

    为什么不能对 MGS 取加权平均：``MGS = U × (1−A) × (1−F)`` 是乘法指标，
    ``mean(MGS_i)`` 与 ``U_pool × (1−A_pool) × (1−F_pool)`` 不是一回事。正确做法
    是把各数据集的三支柱分子分母先相加再算，池化口径写成一行小注（报告里也写）。

    参与范围与 :func:`fuse_metrics` 一致：办公域槽位不参与基准池化（它单列在
    「业务结论」）。没有任何数据集带 ``official`` 块时返回 ``None``。

    Args:
        results: 各数据集的评测结果（``metrics["official"]`` 可能缺失）。

    Returns:
        ``{"U","A","F","OR","MGS","n_utility","n_access","n_forget",
        "n_utility_correct","n_access_violation","n_forget_failure",
        "n_over_refusal","participants","excluded",
        "action_source":"inferred","pooling":"sum_numerator_over_sum_denominator"}``；
        某个支柱没有参与题时对应值（含 MGS）记 ``None``。
    """
    participants: list[str] = []
    excluded: list[str] = []
    sums = {
        "n_utility": 0,
        "n_utility_correct": 0,
        "n_over_refusal": 0,
        "n_access": 0,
        "n_access_violation": 0,
        "n_forget": 0,
        "n_forget_failure": 0,
    }
    found = False
    for res in results:
        official = res.metrics.get("official")
        if not isinstance(official, dict):
            continue
        if not is_baseline_dataset(res):
            excluded.append(res.spec.key)
            continue
        found = True
        participants.append(res.spec.key)
        for key in sums:
            value = official.get(key)
            if isinstance(value, int):
                sums[key] += value
    if not found:
        return None

    def ratio(num: str, den: str) -> Union[float, None]:
        return (sums[num] / sums[den]) if sums[den] else None

    u = ratio("n_utility_correct", "n_utility")
    a = ratio("n_access_violation", "n_access")
    f = ratio("n_forget_failure", "n_forget")
    return {
        "U": u,
        "A": a,
        "F": f,
        "OR": ratio("n_over_refusal", "n_utility"),
        "MGS": (u * (1.0 - a) * (1.0 - f)) if (u is not None and a is not None and f is not None) else None,
        **sums,
        "participants": participants,
        "excluded": excluded,
        "action_source": "inferred",
        "pooling": "sum_numerator_over_sum_denominator",
    }


def _as_count(value: Any) -> Union[int, None]:
    """计数型字段 -> 正整数；缺省 / 0 / 非数字返回 None（= 不能用它当权重）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    n = int(value)
    return n if n > 0 else None


def metric_denominator(overall: dict[str, Any], metric: str) -> Union[int, None]:
    """该数据集在该指标上的权重分母（参与题数）。

    分母按指标取 :data:`METRIC_DENOMINATOR_KEYS`：题数 / 非拒答题数 /
    拒答题数 / 可机检 stale 题数。旧 run 没有 ``evidence_n`` /
    ``abstention_n``（本特性之前只落 ``n_questions``）时回退 ``n_questions``，
    否则旧产物的检索/拒答指标会被判成"无权重"而整体退出融合。

    Returns:
        正整数；取不到（旧产物连 ``n_questions`` 都没有）或为 0 时返回 ``None``。
    """
    raw = _as_count(overall.get(METRIC_DENOMINATOR_KEYS[metric]))
    if raw is None:
        raw = _as_count(overall.get(_DENOMINATOR_FALLBACK))
    return raw


def fuse_metrics(
    results: list[DatasetResult],
    fusion: str = FUSION_WEIGHTED,
    cap: int = FUSION_WEIGHT_CAP,
) -> dict[str, dict[str, Any]]:
    """跨数据集融合六项质量指标 + 两项诊断指标（默认按参与题数加权）。

    办公域槽位（``spec.is_office_slot``）**不参与**：它的分数不是基准集的一部分，
    单列在报告的「业务结论」里。其余数据集按能力标注决定是否参与该指标，
    参与后：

    - ``fusion="weighted"``：权重 = ``min(该指标分母, cap)``，融合值为加权平均。
      分母为 0 / 取不到的集不参与该指标（没有参与题就没有权重）。
    - ``fusion="equal"``：每个参与集一票（权重 1，旧的宏平均语义）。

    Args:
        results: 各数据集的评测结果。
        fusion: 融合口径（``weighted`` / ``equal``）。
        cap: 加权口径下单数据集的权重封顶（题数）。

    Returns:
        ``{metric: {"value", "participants", "weights", "total_weight"}}``：
        ``participants`` 为参与集显示名，``weights`` 为逐集明细
        ``[{"key", "name", "denominator", "weight"}, ...]``（等权口径下
        ``weight`` 恒为 1、``denominator`` 仅作展示），``total_weight`` 为
        权重之和。某指标无参与集时 ``value`` 为 None。

    Raises:
        SystemExit: ``fusion`` 不在 :data:`FUSION_CHOICES` 之内。
    """
    if fusion not in FUSION_CHOICES:
        raise SystemExit(
            f"未知融合口径: {fusion!r}（可选 {' / '.join(FUSION_CHOICES)}）"
        )
    fused: dict[str, dict[str, Any]] = {}
    for metric in FUSED_METRICS:
        entries: list[dict[str, Any]] = []
        weighted_sum = 0.0
        for res in results:
            if not is_baseline_dataset(res):
                continue
            if not participates(res.capabilities, metric):
                continue
            overall = res.metrics.get("overall", {})
            v = overall.get(metric)
            # None = 该数据集"未计算"该指标：跳过，不按 0 纳入融合。
            if v is None:
                continue
            denominator = metric_denominator(overall, metric)
            if fusion == FUSION_WEIGHTED:
                if denominator is None:
                    continue
                weight = min(denominator, int(cap))
            else:
                weight = 1
            entries.append(
                {
                    "key": res.spec.key,
                    "name": res.spec.name,
                    "denominator": denominator,
                    "weight": weight,
                }
            )
            weighted_sum += float(v) * weight
        total_weight = sum(e["weight"] for e in entries)
        fused[metric] = {
            "value": (weighted_sum / total_weight) if total_weight else None,
            "participants": [e["name"] for e in entries],
            "weights": entries,
            "total_weight": total_weight,
        }
    return fused


def stamp_fusion(results: list[DatasetResult], fusion: str) -> None:
    """把当次融合口径写进各数据集的 ``report/metrics.json``（``fusion`` 字段）。

    可比性校验读的正是数据集级 metrics（``compare`` 的入参就是数据集目录）；
    口径只写在任务级报告里，"fusion 进关键维度"就是死维度。复用进来的产物也
    一并改写成当次口径——它在本任务里确实按这个口径聚合。
    """
    for res in results:
        res.metrics["fusion"] = fusion
        path = os.path.join(res.run_dir, layout.DIR_REPORT, layout.FILE_METRICS)
        with open(path, encoding="utf-8") as f:
            disk = json.load(f)
        if disk.get("fusion") == fusion:
            continue
        disk["fusion"] = fusion
        with open(path, "w", encoding="utf-8") as f:
            json.dump(disk, f, ensure_ascii=False, indent=2)


def evaluate_gate(
    results: list[DatasetResult],
    threshold: float = STALE_GATE_THRESHOLD,
) -> dict[str, Any]:
    """办公域 stale 门禁（一票否决，独立于总分）。

    办公域槽位数据集有可机检的 stale 题（``stale_n > 0``）时：其
    ``stale_rate > threshold`` -> FAIL（总评不达标，退出码 1）；
    ``stale_n == 0``（本次判分没有拿到任何可机检的旧值）-> SKIP——
    不能把"没有判定依据"当成 PASS。

    Returns:
        ``{"status": "PASS"|"FAIL"|"SKIP", "detail": 说明}``。
    """
    office = next((r for r in results if r.spec.is_office_slot), None)
    if office is None:
        return {"status": "SKIP", "detail": "数据集清单中无办公域槽位，stale 门禁跳过。"}
    overall = office.metrics.get("overall", {})
    stale_n = overall.get("stale_n")
    if stale_n == 0:
        return {
            "status": "SKIP",
            "detail": (
                f"办公域槽位数据集（{office.spec.name}）无可机检的 stale 题"
                "（stale_n=0），stale 门禁跳过。"
            ),
        }
    if stale_n is None and not office.capabilities.get("has_stale", False):
        return {
            "status": "SKIP",
            "detail": f"办公域槽位数据集（{office.spec.name}）无 gold_stale 标注，stale 门禁跳过。",
        }
    stale_raw = overall.get("stale_rate")
    if stale_raw is None:
        return {
            "status": "SKIP",
            "detail": f"办公域槽位数据集（{office.spec.name}）stale_rate 未计算（None），stale 门禁跳过。",
        }
    stale_rate = float(stale_raw)
    stale_total = overall.get("stale_total")
    coverage = (
        f"（可机检 {stale_n}/{stale_total}）"
        if isinstance(stale_n, int) and isinstance(stale_total, int) and stale_n < stale_total
        else ""
    )
    if stale_rate > threshold:
        return {
            "status": "FAIL",
            "detail": (
                f"办公域槽位数据集（{office.spec.name}）stale_rate={stale_rate:.3f}"
                f"{coverage} > 阈值 {threshold}，一票否决：总评不达标。"
            ),
        }
    return {
        "status": "PASS",
        "detail": (
            f"办公域槽位数据集（{office.spec.name}）stale_rate={stale_rate:.3f}"
            f"{coverage} <= 阈值 {threshold}。"
        ),
    }


def run_datasets(
    adapter: Any,
    judge: Any,
    specs: list[DatasetSpec],
    experiment_dir: str,
    platform: Union[str, None] = None,
    commit: Union[str, None] = None,
    jobs: int = 1,
    driver_llm: Union[str, None] = None,
    adapter_version: Union[str, None] = None,
    evidence_matcher: Any = None,
    snapshot: Union[bool, None] = None,
    trace_state: Union[bool, None] = None,
    abstention_recheck: Union[bool, None] = None,
) -> list[DatasetResult]:
    """对每个数据集用 Runner 各跑一次写进同一个实验目录，收集 metrics。

    每个数据集落 ``<experiment_dir>/runs/<dataset>/``（Runner 按数据集文件名词干
    命名并写自己的数据集目录）；adapter 实例跨数据集复用同一个（reset 由 Runner
    内部负责），每个数据集新建 Runner。任务级 reports/ 由调用方（evaluate）统一写，
    故 Runner 传 ``write_task_reports=False``。
    jobs 透传给 Runner 的题目级并发参数。driver_llm / adapter_version 透传给
    Runner，记入每个 run 的元数据。evidence_matcher 为检索命中口径匹配器
    （默认 substring）。snapshot 默认 None = 用 Runner 的默认（导出对话状态快照），
    显式 False 才关掉（不传该参数时按 Runner 默认处理，别把 None 当成"不导出"）。
    trace_state 透传给 Runner：只对 interleaved 对话生效，sequence 对话的提问时点
    就是末尾 ``memory_snapshot/``。None（缺省）= Runner 的**自动**判定（有
    interleaved 对话含 safety 题时开启），True / False 强制开关。
    """
    # snapshot=None 表示"不表态"，交给 Runner 的默认值（导出）；显式值才透传
    snapshot_kwarg: dict[str, Any] = {} if snapshot is None else {"snapshot": snapshot}
    results: list[DatasetResult] = []
    for spec in specs:
        print(f"[membench] ===== 评测数据集: {spec.name} ({spec.path}) =====")
        runner = Runner(
            adapter,
            spec.path,
            judge=judge,
            out_dir=experiment_dir,
            dataset_source=spec.source,
            jobs=jobs,
            platform=platform,
            commit=commit,
            driver_llm=driver_llm,
            adapter_version=adapter_version,
            evidence_matcher=evidence_matcher,
            write_task_reports=False,
            trace_state=trace_state,
            abstention_recheck=abstention_recheck,
            **snapshot_kwarg,
        )
        run_dir = runner.run()
        if run_dir is None:
            raise SystemExit(f"数据集 {spec.name} 无可评测内容（空数据集），终止评估")
        with open(
            os.path.join(run_dir, layout.DIR_REPORT, layout.FILE_METRICS), encoding="utf-8"
        ) as f:
            metrics = json.load(f)
        results.append(
            DatasetResult(
                spec=spec,
                run_dir=os.path.abspath(run_dir),
                metrics=metrics,
                # 能力标注优先用 run 落账值（Runner 已按同一口径扫描），缺失才重扫数据集
                capabilities=_capabilities_from_metrics(metrics, spec.path),
            )
        )
    return results


def _token_usage_summary(results: list[DatasetResult]) -> dict[str, Any]:
    """各数据集的 ``token_usage`` 与「已上报部分」合计（报告与 summary.json 共用）。

    Returns:
        ``{"per_dataset": {key: usage | None}, "total": {...}, "datasets_reported": n,
        "datasets_total": m}``；``total`` 只求和已落账的 run——未落账记 ``None``
        而不是 0，``datasets_reported`` 让调用方知道合计的覆盖范围。
    """
    per_dataset: dict[str, Union[dict[str, Any], None]] = {}
    total = {
        "ingest_input_tokens": 0,
        "ingest_llm_calls": 0,
        "query_input_tokens": 0,
        "query_output_tokens": 0,
        "query_llm_calls": 0,
    }
    reported = 0
    for r in results:
        usage = r.metrics.get("token_usage")
        if isinstance(usage, dict):
            reported += 1
            totals = token_usage_totals(usage)
            for key in total:
                total[key] += totals.get(key, 0)
            per_dataset[r.spec.key] = usage
        else:
            per_dataset[r.spec.key] = None
    return {
        "per_dataset": per_dataset,
        "total": total if reported else None,
        "datasets_reported": reported,
        "datasets_total": len(results),
    }


def fusion_label(fusion: str) -> str:
    """融合口径的中文说明（报告标题与小注共用）。"""
    if fusion == FUSION_EQUAL:
        return "基准集内等权"
    return f"按参与题数加权，单集封顶 {FUSION_WEIGHT_CAP}"


def participants_with_weight(entry: dict[str, Any], fusion: str) -> str:
    """融合指标表的参与集单元格：``locomo（1986 题，权重 500）``。"""
    cells: list[str] = []
    for w in entry.get("weights", []):
        if fusion == FUSION_EQUAL:
            cells.append(f"{w['name']}（等权）")
        else:
            cells.append(f"{w['name']}（{w['denominator']} 题，权重 {w['weight']}）")
    return "、".join(cells) or "（无参与集）"


def _render_task_report_md(
    experiment_dir: str,
    reports_dir: str,
    adapter_name: str,
    judge_name: str,
    platform: Union[str, None],
    commit: Union[str, None],
    timestamp: datetime,
    results: list[DatasetResult],
    fused: dict[str, dict[str, Any]],
    gate: dict[str, Any],
    fusion: str = FUSION_WEIGHTED,
    extra_notes: Union[list[str], None] = None,
) -> str:
    """生成任务级 ``report.md``（中文，全量评估报告），返回报告路径。

    Args:
        experiment_dir: 实验目录——各数据集产物在其 ``runs/<dataset>/`` 下，报告内
            链接用相对它算出的 ``runs/<dataset>/report/report.md``。
        reports_dir: 任务级报告目录（``<实验目录>/reports``）。
        fusion: 融合口径（决定标题、权重列与评级小注的措辞）。
        extra_notes: 报告顶部追加的提示行（Markdown，如汇总模式的元数据不一致告警）。
        results: 各数据集评测结果（spec 提供数据集范围、来源与路径）。
    """
    gate_status = str(gate["status"])
    overall_fused = {
        k: v["value"] for k, v in fused.items() if v["value"] is not None
    }
    rated = overall_rating(overall_fused)
    verdict = "不达标" if gate_status == "FAIL" else "达标"

    lines: list[str] = []
    lines.append(f"# membench 全量评估报告 — {os.path.basename(os.path.normpath(experiment_dir))}")
    lines.append("")

    # 门禁结论（报告顶部显著呈现）
    lines.append("## 门禁结论")
    lines.append("")
    gate_icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}.get(gate_status, "")
    lines.append(f"**门禁结论： {gate_status} {gate_icon}** — {gate['detail']}")
    lines.append("")
    lines.append(f"**总评结论： {verdict}**（门禁一票否决，独立于基准评级）")
    lines.append("")
    for note in extra_notes or []:
        lines.append(note)
        lines.append("")

    # 评测配置：补齐 driver_llm / judge_model / adapter_version /
    # dataset_fingerprint 四项可比性核心元数据（取自首个数据集的
    # metrics.json；多数据集时指纹逐项见下方「数据集」表）。
    first_metrics = results[0].metrics if results else {}
    fingerprint = first_metrics.get("dataset_fingerprint") or {}
    fp_sha = fingerprint.get("sha256") if isinstance(fingerprint, dict) else None
    config_rows = [
        ["adapter", f"`{adapter_name}`"],
        ["judge", f"`{judge_name}`"],
        ["platform", str(platform or "default")],
        ["commit", str(commit or "n/a")],
        [
            "membench 版本",
            version_display(first_metrics.get("membench_version")) if results else "n/a",
        ],
        ["驱动 LLM", _fmt_opt(first_metrics.get("driver_llm"))],
        ["判分模型", _fmt_opt(first_metrics.get("judge_model"))],
        ["adapter 版本", _fmt_opt(first_metrics.get("adapter_version"))],
        # 单数据集时直接给 sha256；多数据集时逐数据集指纹见下表。
        ["数据集指纹", _fmt_opt(fp_sha) if len(results) <= 1 else "见下表（逐数据集）"],
        ["检索匹配口径", str(first_metrics.get("retrieval_match") or "substring")],
        ["时间戳", timestamp.isoformat(timespec="seconds")],
        # 数据集范围：本次实际评测的 key 与来源，便于"先确认数据集范围"。
        ["数据集范围", "、".join(f"{r.spec.key}（{r.spec.source}）" for r in results) or "（无）"],
    ]
    lines.append(_md_table(["项目", "值"], config_rows))
    lines.append("")
    ds_rows = []
    for r in results:
        fp = r.metrics.get("dataset_fingerprint")
        ds_rows.append(
            [
                r.spec.name,
                r.spec.key,
                r.spec.source,
                f"`{r.spec.path}`",
                _fmt_opt(fp.get("sha256") if isinstance(fp, dict) else None),
            ]
        )
    lines.append(_md_table(["数据集", "槽位", "来源", "路径", "指纹"], ds_rows))
    lines.append("")

    # 基准评级（不含办公域槽位）：没有基准集时明确显示 n/a，而不是空着
    if rated is not None:
        label, mean_points = rated
        lines.append(f"**基准评级（不含办公域槽位）：{label}（{mean_points:.2f}/4.0）**")
    else:
        lines.append("**基准评级（不含办公域槽位）：n/a**")
        lines.append("")
        lines.append(
            "<sub>无基准数据集（只剩办公域槽位）或基准集各项指标均不可评级，"
            "不产出基准评级；办公域的业务口径见下方「业务结论」。</sub>"
        )
    lines.append("")
    lines.append(
        "<sub>评级规则：先按融合口径把六项质量指标融合成基准集的值"
        f"（{fusion_label(fusion)}；指标仅在具备对应能力标注的基准数据集上参与；"
        "办公域槽位不参与），再取其中四项评级指标"
        "（accuracy / evidence_recall / abstention_accuracy / stale_rate）"
        "融合值的连续点数（1.0~4.0，档内随数值线性变化）均值，"
        "阈值 3.5 / 2.5 / 1.5；token_f1 / bleu1（字面重合度，非语义正确性）"
        "与 recall_at_5 / over_refusal_rate 为诊断指标，仅展示不参与评级；"
        "stale_rate 的分母只含可机检的 stale 题（有裸 gold_stale；"
        "LLM judge 下含 gold_stale_note）；"
        "效率成本指标同样不参与评级。门禁为一票否决，独立于本评级。</sub>"
    )
    lines.append("")

    # 业务结论（办公域槽位）：它的分数与基准集不同源，单列而不是混进融合
    office = next((r for r in results if r.spec.is_office_slot), None)
    lines.append("## 业务结论（办公域槽位）")
    lines.append("")
    lines.append(
        "<sub>办公域槽位不参与基准评级——它题量小、判分口径面向业务红线"
        "（陈旧记忆 / 该拒答），与公开基准集不同源；混进融合会把综合分"
        "变成两种标准的平均数。这里单列业务口径的结论，门禁仍一票否决。</sub>"
    )
    lines.append("")
    if office is None:
        lines.append("（本次数据集清单无办公域槽位，stale 门禁跳过。）")
    else:
        office_overall = office.metrics.get("overall", {})
        office_rows = [
            ["数据集", office.spec.name],
            ["题数", str(office_overall.get("n_questions", "n/a"))],
            ["准确率", format_metric("accuracy", office_overall.get("accuracy"))],
            ["陈旧率（stale_rate）", stale_rate_display(office_overall)],
            [
                "拒答准确率",
                (
                    format_metric(
                        "abstention_accuracy", office_overall.get("abstention_accuracy")
                    )
                    if office.capabilities.get("has_abstention")
                    else "未定义（无拒答题标注）"
                ),
            ],
            ["stale 门禁", f"{gate_status} — {gate['detail']}"],
        ]
        lines.append(_md_table(["项目", "值"], office_rows))
    lines.append("")

    # 融合指标表（参与集带题数与权重）
    lines.append(f"## 融合指标（{fusion_label(fusion)}）")
    lines.append("")
    fused_rows = []
    for metric in FUSED_METRICS:
        entry = fused[metric]
        if entry["value"] is None:
            value_str = "未定义"
        else:
            value_str = format_metric(metric, entry["value"])
        fused_rows.append(
            [
                metric_label_with_role(metric),
                value_str,
                participants_with_weight(entry, fusion),
            ]
        )
    lines.append(_md_table(["指标", "融合值", "参与数据集（题数 / 权重）"], fused_rows))
    lines.append("")
    lines.append(
        "<sub>注：token_f1 / bleu1（字面重合度，非语义正确性）与检索召回率@5 / "
        "过度拒答率为诊断指标，仅展示与归因，不参与基准评级与门禁。</sub>"
    )
    lines.append("")
    # stale 覆盖面：缺少可机检 stale 标注的数据集不参与 stale_rate，若不点名，
    # 读者会把"融合 stale_rate"当成全局结论（实际可能只由一两个集决定）。
    stale_blind = [
        r.spec.name
        for r in results
        if is_baseline_dataset(r)
        and not (r.metrics.get("overall") or {}).get("stale_n")
    ]
    if stale_blind and fused.get("stale_rate", {}).get("value") is not None:
        lines.append(
            "<sub>**stale 覆盖**："
            + "、".join(stale_blind)
            + " 没有可机检的 stale 标注，不参与 `stale_rate`——上表的陈旧率只由"
            "其余数据集决定，不能当全局结论（要覆盖它们，需要给这些集补 `gold_stale`）。</sub>"
        )
        lines.append("")

    # GateMem 官方口径（逐数据集 + 池化）：与六维融合表并列，不参与融合/评级。
    pooled = pool_official(results)
    has_official = any(isinstance(r.metrics.get("official"), dict) for r in results)
    if has_official:
        lines.append("## GateMem 官方口径（逐数据集 + 池化）")
        lines.append("")
        lines.append(
            "<sub>对齐 GateMem 论文（arXiv:2606.18829）的三支柱与乘法主指标 "
            "`MGS = U × (1−A) × (1−F)`。动作标签由回答文本推断"
            "（`action_source=inferred`），泄露用字面命中 `leak_targets` 近似，"
            "`state_leak` 不参与；**本节是参考视图，不参与上面的融合、综合评级与门禁**。"
            "跨数据集**不能**对 MGS 加权平均（乘法指标），池化行按"
            "「Σ分子 / Σ分母」先合并再算；办公域槽位不参与池化（单列业务结论）。</sub>"
        )
        lines.append("")

        def _cell(value: Any, count: Union[int, None], total: Union[int, None]) -> str:
            if value is None:
                return "未定义"
            text = f"{float(value) * 100:.1f}%"
            if count is not None and total:
                text += f"（{int(count)}/{int(total)}）"
            return text

        official_rows: list[list[str]] = []
        for res in results:
            official = res.metrics.get("official")
            if not isinstance(official, dict):
                continue
            official_rows.append(
                [
                    res.spec.name,
                    _cell(official.get("U"), official.get("n_utility_correct"), official.get("n_utility")),
                    _cell(official.get("OR"), official.get("n_over_refusal"), official.get("n_utility")),
                    _cell(official.get("A"), official.get("n_access_violation"), official.get("n_access")),
                    _cell(official.get("F"), official.get("n_forget_failure"), official.get("n_forget")),
                    f"{float(official['MGS']) * 100:.1f}%" if official.get("MGS") is not None else "未定义",
                    f"{int(official.get('n_utility') or 0)} / {int(official.get('n_access') or 0)} / "
                    f"{int(official.get('n_forget') or 0)}",
                ]
            )
        if pooled is not None:
            official_rows.append(
                [
                    "**池化（基准集，Σ分子/Σ分母）**",
                    _cell(pooled.get("U"), pooled.get("n_utility_correct"), pooled.get("n_utility")),
                    _cell(pooled.get("OR"), pooled.get("n_over_refusal"), pooled.get("n_utility")),
                    _cell(pooled.get("A"), pooled.get("n_access_violation"), pooled.get("n_access")),
                    _cell(pooled.get("F"), pooled.get("n_forget_failure"), pooled.get("n_forget")),
                    f"{float(pooled['MGS']) * 100:.1f}%" if pooled.get("MGS") is not None else "未定义",
                    f"{int(pooled.get('n_utility') or 0)} / {int(pooled.get('n_access') or 0)} / "
                    f"{int(pooled.get('n_forget') or 0)}",
                ]
            )
        lines.append(
            _md_table(
                ["数据集", "U", "OR", "A", "F", "MGS", "检查点（utility / access / forget）"],
                official_rows,
            )
        )
        lines.append("")
        if pooled is not None and pooled.get("excluded"):
            lines.append(
                "<sub>未参与池化：" + "、".join(str(x) for x in pooled["excluded"])
                + "（办公域槽位，单列业务结论）。</sub>"
            )
            lines.append("")
        elif pooled is None:
            lines.append(
                "<sub>本次没有可池化的数据集：带官方口径的数据集都是办公域槽位"
                "（办公域槽位不参与基准池化）。各数据集自身的 U / A / F / MGS 见上表。</sub>"
            )
            lines.append("")

    # 融合权重明细：每个基准数据集在各指标上的分母与权重（可逐项复核融合值）
    lines.append("## 融合权重")
    lines.append("")
    baseline_results = [r for r in results if is_baseline_dataset(r)]
    if baseline_results:
        weight_header = ["数据集", "题数"] + [metric_label(m, with_key=False) for m in FUSED_METRICS]
        weight_rows: list[list[str]] = []
        for r in baseline_results:
            overall = r.metrics.get("overall", {})
            cells = [r.spec.name, str(overall.get("n_questions", "n/a"))]
            for metric in FUSED_METRICS:
                denominator = metric_denominator(overall, metric)
                if not participates(r.capabilities, metric) or overall.get(metric) is None:
                    cells.append("—")
                elif fusion == FUSION_EQUAL:
                    cells.append(f"{denominator if denominator is not None else 'n/a'}（等权）")
                elif denominator is None:
                    cells.append("—")
                else:
                    cells.append(f"{denominator} → {min(denominator, FUSION_WEIGHT_CAP)}")
            weight_rows.append(cells)
        lines.append(_md_table(weight_header, weight_rows))
        lines.append("")
        if fusion == FUSION_EQUAL:
            lines.append(
                "<sub>单元格为「该指标分母（参与题数）」，--fusion equal 下每个基准集"
                "一票（等权）；办公域槽位不参与。</sub>"
            )
        else:
            lines.append(
                "<sub>单元格为「分母 → 权重」：分母按指标取参与题数（accuracy / token_f1 / "
                "bleu1 用总题数，检索类用非拒答题数，拒答类用拒答题数，stale_rate 用可机检 "
                f"stale 题数），权重 = min(分母, {FUSION_WEIGHT_CAP})；「—」= 该集在该指标上"
                "不参与（无对应能力标注或该指标未计算）；办公域槽位不参与。</sub>"
            )
    else:
        lines.append("（无基准数据集：全部数据集都是办公域槽位，不参与融合。）")
    lines.append("")

    # 分数据集明细表（质量 + 效率成本）
    lines.append("## 分数据集明细")
    lines.append("")
    quality_cols = ["accuracy", "token_f1", "bleu1", "stale_rate", "abstention_accuracy", "evidence_recall"]
    detail_rows = []
    for r in results:
        overall = r.metrics.get("overall", {})
        row = [r.spec.name, str(overall.get("n_questions", "n/a"))]
        for metric in quality_cols:
            if metric == "stale_rate":
                # 与 run 报告同口径：无可用题显示 n/a，覆盖不全标注分子/分母。
                row.append(stale_rate_display(overall))
            elif participates(r.capabilities, metric) and metric in overall and overall[metric] is not None:
                row.append(format_metric(metric, overall[metric]))
            else:
                row.append("未定义")
        detail_rows.append(row)
    lines.append(
        _md_table(
            ["数据集", metric_label("n_questions")] + [metric_label(m) for m in quality_cols],
            detail_rows or [["(无)"] + [""] * (len(quality_cols) + 1)],
        )
    )
    lines.append("")
    # 列说明：避免业务方把「未定义」误读为「召回率为 0」
    lines.append(
        "<sub>表中「未定义」表示该数据集未标注对应能力（如 evidence / gold_stale / "
        "abstention 题），不参与该指标计算；融合值也不会因 0 而被拉低。"
        "stale_rate 的 `n/a（无可机检的 stale 注解）` 表示本次没有拿到可机检的裸旧值，"
        "同样不当 0 参与融合。</sub>"
    )
    lines.append("")

    # 诊断指标明细（不参与评级）：严格检索召回、过度拒答、忽略题、判分回退
    lines.append("## 诊断指标（不参与评级）")
    lines.append("")
    diag_rows = []
    for r in results:
        overall = r.metrics.get("overall", {})
        row = [r.spec.name]
        for metric in DIAGNOSTIC_METRICS:
            if participates(r.capabilities, metric) and overall.get(metric) is not None:
                row.append(format_metric(metric, overall[metric]))
            else:
                row.append("未定义")
        row.append(str(r.metrics.get("n_ignored", 0)))
        fb = r.metrics.get("judge_fallback") or {}
        if fb:
            row.append(f"{int(fb.get('count', 0))}（{float(fb.get('ratio', 0.0)):.1%}）")
        else:
            row.append("—")
        diag_rows.append(row)
    lines.append(
        _md_table(
            ["数据集", *(metric_label(m) for m in DIAGNOSTIC_METRICS), "忽略题数", "LLM judge 回退"],
            diag_rows or [["(无)", *[""] * (len(DIAGNOSTIC_METRICS) + 2)]],
        )
    )
    lines.append("")
    lines.append(
        "<sub>诊断指标仅用于分诊与归因，不参与基准评级与门禁；「检索召回率@5」"
        "仅在有 evidence 标注的数据集上计算（前 5 条检索的证据条目覆盖率），"
        "「过度拒答率」为非拒答题上出现拒答信号的比例。</sub>"
    )
    lines.append("")
    # 检索指标全 0 警示：疑似 evidence 与被检索文本格式不一致（摘要式存储）
    zero_sets = [
        r.spec.name
        for r in results
        if r.capabilities.get("has_evidence")
        and (r.metrics.get("overall") or {}).get("evidence_recall") == 0.0
        and (r.metrics.get("overall") or {}).get("recall_at_5") == 0.0
    ]
    if zero_sets:
        lines.append(
            "> ⚠️ 「"
            + "、".join(zero_sets)
            + "」标注了 evidence，但「证据召回率」与「检索召回率@5」全为 0："
            "疑似 evidence 与被检索文本格式不一致（如记忆以摘要/改写形式存储），"
            "并非真实零召回。可尝试 `--evidence-match overlap|llm` 调整命中口径，"
            "或让适配器返回可核对的原文片段后重跑。"
        )
        lines.append("")

    # 效率成本（分数据集）
    lines.append("## 效率成本（分数据集）")
    lines.append("")
    eff_cols = [
        "latency_p50_ms",
        "latency_p95_ms",
        "avg_input_tokens",
        "avg_output_tokens",
        "avg_ingest_latency_ms",
        "avg_llm_calls",
    ]
    eff_rows = []
    for r in results:
        overall = r.metrics.get("overall", {})
        row = [r.spec.name]
        for metric in eff_cols:
            v = overall.get(metric)
            row.append(f"{float(v):.1f}" if isinstance(v, (int, float)) else "—")
        eff_rows.append(row)
    lines.append(
        _md_table(
            ["数据集"] + [metric_label(m) for m in eff_cols],
            eff_rows or [["(无)"] + [""] * len(eff_cols)],
        )
    )
    lines.append("")
    lines.append("![各数据集质量与效率](fig_dataset_overview.png)")
    lines.append("")

    # Token 消耗（全流程）：被测系统上报口径的累计值——回答"这次评测烧了多少"
    lines.append("## Token 消耗（全流程）")
    lines.append("")
    lines.append(
        "<sub>被测系统（适配器）上报口径的累计消耗：ingest 写入按每次对话计、"
        "query 输入/输出逐题累计；不含判分器 / 驱动侧消耗。0 = 适配器未上报该字段，"
        "「—」= 该 run 未落账 `token_usage`（旧版本产物）。</sub>"
    )
    lines.append("")
    token_summary = _token_usage_summary(results)
    token_rows: list[list[str]] = []
    for r in results:
        usage = token_summary["per_dataset"].get(r.spec.key)
        if isinstance(usage, dict):
            totals = token_usage_totals(usage)
            grand = (
                totals.get("ingest_input_tokens", 0)
                + totals.get("query_input_tokens", 0)
                + totals.get("query_output_tokens", 0)
            )
            calls = totals.get("ingest_llm_calls", 0) + totals.get("query_llm_calls", 0)
            token_rows.append(
                [
                    r.spec.name,
                    str(totals.get("ingest_input_tokens", 0)),
                    str(totals.get("query_input_tokens", 0)),
                    str(totals.get("query_output_tokens", 0)),
                    str(grand),
                    str(calls),
                ]
            )
        else:
            token_rows.append([r.spec.name, "—", "—", "—", "—", "—"])
    cols = ["数据集", "ingest 写入", "query 输入", "query 输出", "合计", "LLM 调用"]
    total = token_summary["total"]
    reported = token_summary["datasets_reported"]
    if total is not None:
        grand_total = (
            total["ingest_input_tokens"] + total["query_input_tokens"] + total["query_output_tokens"]
        )
        calls_total = total["ingest_llm_calls"] + total["query_llm_calls"]
        token_rows.append(
            [
                f"**合计（{reported}/{len(results)} 个数据集上报）**",
                str(total["ingest_input_tokens"]),
                str(total["query_input_tokens"]),
                str(total["query_output_tokens"]),
                f"**{grand_total}**",
                str(calls_total),
            ]
        )
        lines.append(_md_table(cols, token_rows))
        if reported < len(results):
            lines.append("")
            lines.append(
                f"<sub>⚠️ 仅 {reported}/{len(results)} 个数据集的 run 落了账，合计为已上报部分的求和。</sub>"
            )
    else:
        lines.append("（本次评测各 run 均未落账 `token_usage`：适配器未上报或产物来自旧版本）")
    lines.append("")

    # 各数据集产物链接：显示实验目录内的规范路径；链接目标相对报告所在目录
    # （报告在 <实验目录>/reports/ 下，故目标带 ../——否则点开是死链）
    lines.append("## 各数据集 run 目录")
    lines.append("")
    for r in results:
        label = os.path.relpath(r.run_dir, experiment_dir).replace(os.sep, "/")
        target = "/".join(
            (
                os.path.relpath(r.run_dir, reports_dir).replace(os.sep, "/"),
                layout.DIR_REPORT,
                layout.FILE_REPORT_MD,
            )
        )
        lines.append(
            f"- {r.spec.name}: [{label}/{layout.DIR_REPORT}/{layout.FILE_REPORT_MD}]({target})"
        )
    lines.append("")

    out_path = os.path.join(reports_dir, layout.FILE_REPORT_MD)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def write_task_reports(
    experiment_dir: str,
    *,
    adapter_name: str,
    judge_name: str,
    platform: Union[str, None],
    commit: Union[str, None],
    timestamp: datetime,
    results: list[DatasetResult],
    fused: dict[str, dict[str, Any]],
    gate: dict[str, Any],
    fusion: str = FUSION_WEIGHTED,
    extra_notes: Union[list[str], None] = None,
) -> dict[str, Any]:
    """写任务级报告到 ``<实验目录>/reports/``，返回 evaluate_summary.json 的内容。

    产物：``report.md``（全量评估报告，内链指向各数据集
    ``runs/<dataset>/report/report.md``）、``metrics.json``（任务级汇总：融合口径 +
    融合值 + 门禁 + 身份 + token 消耗）、``evaluate_summary.json``（机器可消费的清单，
    额外带各数据集相对路径）、``fig_dataset_overview.png``。
    """
    reports_dir = layout.task_report_dir(experiment_dir)
    os.makedirs(reports_dir, exist_ok=True)
    # 先出图再写 md：md 引用了这张图，图先到位就不会留下打不开的相对引用
    fig_dataset_overview(
        [(r.spec.name, r.metrics.get("overall", {})) for r in results],
        os.path.join(reports_dir, layout.FIG_DATASET_OVERVIEW),
    )
    _render_task_report_md(
        experiment_dir,
        reports_dir,
        adapter_name,
        judge_name,
        platform,
        commit,
        timestamp,
        results,
        fused,
        gate,
        fusion=fusion,
        extra_notes=extra_notes,
    )

    metrics = {
        "experiment_dir": experiment_dir,
        "reports_dir": reports_dir,
        "adapter": adapter_name,
        "judge": judge_name,
        "platform": platform or "default",
        "commit": commit,
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "n_datasets": len(results),
        "datasets": [r.spec.key for r in results],
        "fusion": fusion,
        "overall": {k: v["value"] for k, v in fused.items() if v["value"] is not None},
        "fused": fused,
        "gate": gate,
        "token_usage": _token_usage_summary(results),
        # GateMem 官方口径池化（参考视图，不参与融合/评级）；无三支柱题型时省略
        **({"official_pooled": pooled} if (pooled := pool_official(results)) else {}),
    }
    with open(os.path.join(reports_dir, layout.FILE_METRICS), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    summary = {
        **metrics,
        "runs": {
            r.spec.key: os.path.relpath(r.run_dir, experiment_dir).replace(os.sep, "/")
            for r in results
        },
    }
    with open(os.path.join(reports_dir, layout.FILE_SUMMARY), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[membench] 任务级报告: {reports_dir}")
    return summary


def evaluate(
    adapter: Any = None,
    judge: Any = None,
    platform: Union[str, None] = None,
    commit: Union[str, None] = None,
    quick: bool = False,
    beam_10m: bool = False,
    no_beam: bool = False,
    no_lme_s: bool = False,
    common: bool = False,
    report_out: Union[str, None] = None,
    specs: Union[list[DatasetSpec], None] = None,
    from_runs: Union[dict[str, str], None] = None,
    jobs: int = 1,
    driver_llm: Union[str, None] = None,
    adapter_version: Union[str, None] = None,
    evidence_matcher: Any = None,
    task_dir: Union[str, None] = None,
    snapshot: Union[bool, None] = None,
    trace_state: Union[bool, None] = None,
    abstention_recheck: Union[bool, None] = None,
    fusion: str = FUSION_WEIGHTED,
) -> int:
    """执行全量评估：一个实验目录承载全部数据集 + 基准集融合 + 门禁 + 报告。

    Args:
        adapter: 适配器实例（跨数据集复用，Runner 内部负责 reset）；汇总模式下
            仅当存在需要补跑的数据集时必需，纯复用可为 None。
        judge: judge 名字符串或 Judge 实例（与 Runner 约定一致）；纯复用可为
            None，此时报告头以已有产物的元数据为准。
        from_runs: 汇总模式：``{数据集 key: 产物目录}``（实验目录或数据集目录）。
            目标清单（``specs``）中有对应产物的复用其结果、缺失的**补跑**（要求
            adapter/judge 可用）；未给 ``specs`` 时目标清单即这些 key（纯汇总）。
            **纯汇总不按位置指定办公域槽位**：只有数据集身份命中办公域（内置
            office / 用户测试集）才算槽位；未含 office 时门禁 SKIP、全部数据集参与
            基准融合（见 :func:`mark_office_slot`）。
            复用的产物会并入本次实验目录（自包含、可整体搬移）。被测对象与评测
            口径（:data:`_IDENTITY_KEYS`）在**开跑之前**校验：已有产物之间混用、或
            补跑参数与已有产物不一致，直接拒绝（避免结果不可归因）；全新 run 缺
            ``driver_llm / adapter_version`` 只在启动时提示（严格可比性会判无法核验）。
        platform: 平台标签。commit: commit 标签。
        quick: 跳过 BEAM 与 LongMemEval-S 槽位（等价 no_beam + no_lme_s）。
        beam_10m: BEAM 槽位用 10m 集。no_beam: 不评测 BEAM 槽位。
        no_lme_s: 剔除 LongMemEval-S 变体（默认包含）。
        common: 只评测公开基准常用集（locomo / longmemeval_oracle /
            gatemem_office / BEAM），不含办公域槽位与 LongMemEval-S；
            与 quick 互斥（清单冲突由 :func:`plan_datasets` 拒绝）。
        report_out: 实验目录的父目录（缺省 :func:`membench.paths.output_root`）；
            实验目录名 ``eval_<adapter>_<platform>_<ts>``，各数据集落它的
            ``runs/<dataset>/``，任务级报告落它的 ``reports/``。
        specs: 显式数据集清单（默认 None 时按 quick/beam_10m/no_beam/no_lme_s/common 调 plan_datasets）。
        driver_llm: 驱动被测系统的 LLM 标识，透传给每个 run 的元数据（默认 None）。
        adapter_version: 适配器实现版本标识，透传给每个 run 的元数据（默认 None）。
        evidence_matcher: 检索命中口径匹配器（默认 substring；CLI 以
            ``--evidence-match`` 构造后注入）。
        task_dir: **就地模式**：本次实验目录就是这个已存在的任务目录——各数据集直接
            写进 ``<task_dir>/runs/``、任务级报告写 ``<task_dir>/reports/``，
            不新建 ``eval_*`` 目录。与 ``report_out`` 互斥。
        snapshot: 是否导出对话状态快照。None（缺省）= 用 Runner 默认（导出）；
            False = 关闭（``--no-snapshot``），透传到每个数据集的 Runner。
        trace_state: 失败归因（``--trace-state``）：透传给每个数据集的 Runner，
            只对 interleaved 对话生效（逐 as_of 导出 ``trace_state/t<as_of>/``）；
            sequence 对话的提问时点就是末尾 ``memory_snapshot/``。None（缺省）=
            Runner 的**自动**判定（有 interleaved 对话含 safety 题时开启），
            ``--no-trace-state`` 传 False 强制关闭。
        fusion: 基准集融合口径（``weighted`` 默认 / ``equal``），见
            :func:`fuse_metrics`；会写进任务级 metrics 与各数据集 metrics
            （``fusion`` 可比性维度——旧产物缺该字段归一为 ``legacy``）。

    Returns:
        进程退出码：门禁 FAIL 为 1，否则 0（与 compare 一致）。
    """
    if fusion not in FUSION_CHOICES:
        raise SystemExit(
            f"未知融合口径: {fusion!r}（可选 {' / '.join(FUSION_CHOICES)}）"
        )
    if task_dir is not None:
        if report_out is not None:
            raise SystemExit("evaluate 的 task_dir 与 report_out 互斥：就地写进任务目录时不再给父目录")
        if not os.path.isdir(task_dir):
            raise SystemExit(
                f"任务目录不存在: {task_dir}；"
                "请先运行 python -m membench.cli task init --name <NAME> 建任务目录"
            )
    if from_runs and specs is None and (quick or beam_10m or no_beam or no_lme_s or common):
        raise SystemExit(
            "--from-runs 与预设开关（--quick / --beam-10m / --no-beam / --no-lme-s / --common）互斥"
        )

    provided: dict[str, DatasetResult] = {}
    for key, product_dir in (from_runs or {}).items():
        provided[key] = load_provided_result(key, product_dir)

    if specs is None:
        if provided:
            # 纯汇总：目标清单即提供的 key（保持输入顺序）；办公域槽位按数据集身份
            # 判定，**不**按位置兜底——顺序只反映 --from-runs 的书写顺序，把第一个
            # 当成槽位会把它踢出基准融合、还拿它的 stale 当业务门禁。含 office 产物
            # 时仍按同一套身份判定生效（见 mark_office_slot）。
            specs = [r.spec for r in provided.values()]
            mark_office_slot(specs, positional_fallback=False)
        else:
            specs = plan_datasets(
                quick=quick, beam_10m=beam_10m, no_beam=no_beam, no_lme_s=no_lme_s, common=common
            )
    else:
        target_keys = {s.key for s in specs}
        unknown = [k for k in provided if k not in target_keys]
        if unknown:
            raise SystemExit(
                f"以下复用 run 不在目标数据集清单中: {', '.join(unknown)}"
                "（请加入 --dataset 或删除对应 --from-runs）"
            )

    # 缺口前置检查：目标清单里有需要补跑的数据集时，adapter/judge 必须可用
    missing_keys = [s.key for s in specs if s.key not in provided]
    if missing_keys and (adapter is None or judge is None):
        raise SystemExit(
            "以下数据集没有可复用的 run，补跑需要 --adapter 与 --judge: "
            + "、".join(missing_keys)
        )

    # 跑前预检：身份/口径问题必须在开跑之前暴露，避免昂贵评测跑完才发现汇总不可用。
    if provided:
        assert_single_identity(list(provided.values()), "本次汇总")
    if missing_keys:
        assert_gap_matches_provided(
            provided,
            expected_fresh_metadata(adapter, judge, driver_llm, adapter_version, evidence_matcher),
            "本次评估",
        )
    if not provided and (driver_llm is None or adapter_version is None):
        missing_meta = [
            option
            for option, value in (("--driver-llm", driver_llm), ("--adapter-version", adapter_version))
            if value is None
        ]
        print(
            f"[membench] 提示: 新 run 未标注 {' / '.join(missing_meta)}；"
            "后续 compare --require-comparable 会判这些关键维度「无法核验」。"
            "建议补齐参数，省略的标注无法事后补记。"
        )

    # 头信息：汇总模式优先以已有 run 的元数据为准（纯复用时没有新评测元数据）
    first_provided_metrics = next(iter(provided.values())).metrics if provided else None
    if judge is None:
        if first_provided_metrics is None:
            raise SystemExit("未提供 judge：无已有 run 可复用时必须指定 --judge")
        judge_name = str(first_provided_metrics.get("judge") or "heuristic")
    else:
        judge_name = judge if isinstance(judge, str) else str(getattr(judge, "name", judge.__class__.__name__))
    if adapter is not None:
        adapter_name = str(getattr(adapter, "name", adapter.__class__.__name__))
    elif first_provided_metrics is not None:
        adapter_name = str(first_provided_metrics.get("adapter") or "unknown")
    else:
        raise SystemExit("未提供 adapter：无已有 run 可复用时必须指定 --adapter")
    if platform is None and first_provided_metrics is not None:
        platform = first_provided_metrics.get("platform")
    if commit is None and first_provided_metrics is not None:
        commit = first_provided_metrics.get("commit")

    if judge_name == "heuristic" and any(s.key.startswith("beam") for s in specs):
        print(
            "[membench] 警告: judge=heuristic 且评测清单含 BEAM；"
            "BEAM 含大量 rubric 型题目，heuristic 判分可能失真，"
            "建议使用 --judge llm 或 --judge client。"
        )

    now = datetime.now()
    if task_dir is not None:
        # 就地：本次实验目录就是该任务目录（不新建 eval_*），各数据集落它的 runs/
        experiment_dir = os.path.abspath(task_dir)
        print(f"[membench] 任务目录: {experiment_dir}")
    else:
        experiment_dir = layout.experiment_path(
            report_out, layout.eval_subdir(adapter_name, platform, layout.ts_now(now))
        )
        print(f"[membench] 实验目录: {experiment_dir}")

    # 复用的产物并入本次实验目录：报告与 run 数据就近归属，实验目录可整体搬移
    provided_by_key: dict[str, DatasetResult] = {}
    for key, res in provided.items():
        provided_by_key[key] = DatasetResult(
            spec=res.spec,
            run_dir=adopt_dataset_dir(experiment_dir, res.run_dir),
            metrics=res.metrics,
            capabilities=res.capabilities,
        )
    missing_specs: list[DatasetSpec] = []
    for spec in specs:
        if spec.key in provided_by_key:
            print(f"[membench] 复用已有 run: {spec.key} <- {provided_by_key[spec.key].run_dir}")
        else:
            missing_specs.append(spec)
    if missing_specs:
        for r in run_datasets(
            adapter,
            judge,
            missing_specs,
            experiment_dir,
            platform=platform,
            commit=commit,
            jobs=jobs,
            driver_llm=driver_llm,
            adapter_version=adapter_version,
            evidence_matcher=evidence_matcher,
            snapshot=snapshot,
            trace_state=trace_state,
            abstention_recheck=abstention_recheck,
        ):
            provided_by_key[r.spec.key] = r

    # 纯复用（无补跑）时没有 Runner 代劳：这里补登记。有补跑时首个 Runner 已在开跑
    # 前登记过同一目录（place 幂等）；空数据集等"没产生任何产物"的失败路径到不了
    # 这里，也就不会留下 runtime/ 把空实验目录变成"跑过"的样子。
    runtime.set_dir(experiment_dir)

    # 按目标清单顺序组装；复用项套用目标 spec（保留统一的展示口径）
    results: list[DatasetResult] = []
    for spec in specs:
        res = provided_by_key[spec.key]
        if spec.key in provided:
            res = DatasetResult(
                spec=spec,
                run_dir=res.run_dir,
                metrics=res.metrics,
                capabilities=res.capabilities,
            )
        results.append(res)

    notes = metadata_inconsistencies(results)
    for note in notes:
        print(f"[membench] ⚠️ 汇总模式：跨数据集元数据不一致 —— {note}")

    stamp_fusion(results, fusion)
    fused = fuse_metrics(results, fusion=fusion)
    gate = evaluate_gate(results)

    write_task_reports(
        experiment_dir,
        adapter_name=adapter_name,
        judge_name=judge_name,
        platform=platform,
        commit=commit,
        timestamp=now,
        results=results,
        fused=fused,
        gate=gate,
        fusion=fusion,
        extra_notes=[f"> ⚠️ **汇总模式**：{note}" for note in notes],
    )

    print(f"[membench] 门禁结论: {gate['status']} — {gate['detail']}")
    print(f"[membench] 报告: {os.path.join(layout.task_report_dir(experiment_dir), layout.FILE_REPORT_MD)}")

    return 1 if gate["status"] == "FAIL" else 0
