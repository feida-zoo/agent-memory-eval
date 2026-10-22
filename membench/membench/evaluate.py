"""evaluate：对多个数据集依次运行评测，跨数据集宏平均融合出综合评分报告。

流程：
    plan_datasets 规划数据集清单（办公域槽位 + 按需下载公开集 + BEAM 槽位）
    -> 每个数据集用 Runner 各跑一次（adapter 实例复用，reset 由 Runner 内部负责）
    -> 读取各 run_dir 的 metrics.json
    -> 宏平均融合六项质量指标（按能力标注决定参与集）
    -> 办公域 stale 门禁（一票否决）
    -> 报告目录（report_out 缺省时 <out_dir>/evaluate_<ts>/，
       指定时 <report_out>/eval_<adapter>_<platform>_<ts>/）内写
       evaluate_report.md + evaluate_summary.json，并把各数据集 run 目录
       复制进该目录的 runs/ 下，使报告与 run 数据就近归属、自包含。

融合口径（宏平均）：
    - accuracy / token_f1 / bleu1：所有数据集参与；
    - evidence_recall：仅 has_evidence（任一题目有 evidence 标注）的数据集参与；
    - stale_rate：仅 has_stale（任一题目有 gold_stale 标注）的数据集参与；
    - abstention_accuracy：仅 has_abstention（任一题目为拒答题：category 为
      abstention 或 gold 为拒答哨兵值）的数据集参与；
    融合值 = 参与集该指标 overall 值的算术平均。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Union

from membench import layout, paths
from membench.data.info import resolve_available_path
from membench.data.schema import load_dataset, question_capabilities
from membench.ratings import format_metric, overall_rating
from membench.report import (
    TOKEN_USAGE_FIELD_ZH,
    _md_table,
    _fmt_opt,
    fig_dataset_overview,
    metric_label,
    token_usage_totals,
)
from membench.runner import Runner
from membench.userdata import resolve_dataset

#: 参与融合的六项质量指标（宏平均口径）。
QUALITY_METRICS: tuple[str, ...] = (
    "accuracy",
    "token_f1",
    "bleu1",
    "evidence_recall",
    "abstention_accuracy",
    "stale_rate",
)

#: 诊断指标（展示但不参与综合评级与门禁）：严格检索召回率@5 与过度拒答率。
DIAGNOSTIC_METRICS: tuple[str, ...] = ("recall_at_5", "over_refusal_rate")

#: 融合指标集：六项质量指标 + 两项诊断指标（宏平均口径一致）。
FUSED_METRICS: tuple[str, ...] = QUALITY_METRICS + DIAGNOSTIC_METRICS

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
        run_dir: Runner 产出的 run 目录。
        metrics: 该 run 的 metrics.json 内容。
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

    解析规则（列表顺序即评测顺序）：

    - 存在的文件路径 -> ``user-specified`` 收录，key 取文件名 stem；
    - ``office`` -> 办公域槽位（优先级：用户测试集 > 内置 office）；
    - 其余按数据集 key 经 ``resolve_available_path`` 解析（内置文件优先，
      其次 ``dataset fetch`` 下载目录）；解析不到报错提示先 fetch；
    - 列表中第一个数据集承担办公域槽位（stale 门禁作用域；显式给出
      ``office`` 时以它为准）。若该集无 gold_stale 标注，门禁自动 SKIP。

    Args:
        values: ``--dataset`` 原始字符串列表（key 或路径）。

    Returns:
        DatasetSpec 列表。

    Raises:
        SystemExit: 列表为空、含空值，或某个 key 无法解析。
    """
    if not values:
        raise SystemExit("--dataset 需要至少一个数据集（key 或 JSONL 路径）")
    specs: list[DatasetSpec] = []
    for raw in values:
        value = raw.strip()
        if not value:
            raise SystemExit("--dataset 收到空值（key 或 JSONL 路径）")
        if os.path.exists(value):
            path = Path(value)
            specs.append(
                DatasetSpec(
                    key=path.stem,
                    name=path.stem,
                    path=str(path.resolve()),
                    source="user-specified",
                )
            )
        elif value == "office":
            office_path, office_source = resolve_dataset(None)
            office_name = (
                "office（用户测试集）"
                if office_source == "user"
                else "office（内置办公黄金集）"
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
        else:
            resolved = resolve_available_path(value)
            if resolved is None:
                raise SystemExit(
                    f"数据集 {value!r} 无法解析：既不是存在的文件路径，也不是可用"
                    f"的数据集 key（可用 key 见 list-datasets）；若需按需下载，"
                    f"先运行 python -m membench.cli dataset fetch {value}"
                )
            specs.append(
                DatasetSpec(key=value, name=value, path=str(resolved), source="fetch")
            )
    # 标出办公域槽位：显式 office 优先，否则列表第一个数据集承担
    if not any(spec.is_office_slot for spec in specs):
        specs[0].is_office_slot = True
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


def load_provided_result(key: str, run_dir: str) -> DatasetResult:
    """从现成 run 目录加载 DatasetResult（汇总模式复用已有评测结果）。"""
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.isfile(metrics_path):
        raise SystemExit(f"run 目录缺少 metrics.json: {run_dir}（{key}）")
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    dataset_path = str(metrics.get("dataset_path") or "")
    source = str(metrics.get("dataset_source") or "user-specified")
    if key == "office":
        name = "office（用户测试集）" if source == "user" else "office（内置办公黄金集）"
    else:
        name = key
    spec = DatasetSpec(key=key, name=name, path=dataset_path, source=source)
    return DatasetResult(
        spec=spec,
        run_dir=run_dir,
        metrics=metrics,
        capabilities=_capabilities_from_metrics(metrics, dataset_path),
    )


#: 汇总模式的「提示型」不一致维度：被测对象与评测口径维度（:data:`_IDENTITY_KEYS`）
#: 混用已在跑前直接拒绝，这里只剩 platform 这类允许跨 run 差异、仅需标注的维度；
#: commit 是 A/B 的被测变量，不在此列。
_META_CHECK_KEYS = ("platform",)


#: 同一侧/同一报告内必须唯一的维度：被测对象身份 + 评测口径（A/B 差异只应发生在侧之间）。
_IDENTITY_KEYS = (
    "adapter",
    "adapter_version",
    "driver_llm",
    "judge",
    "judge_model",
    "retrieval_match",
    "retrieval_match_threshold",
)


def assert_single_identity(results: list[DatasetResult], label: str) -> None:
    """校验一侧（或单臂汇总）内的被测对象与评测口径唯一，防止混入多个系统/版本/仪器。

    Raises:
        SystemExit: :data:`_IDENTITY_KEYS` 中任一维度出现多组取值。
    """
    seen: dict[tuple, list[str]] = {}
    for r in results:
        key = tuple(r.metrics.get(k) for k in _IDENTITY_KEYS)
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
    }


def assert_gap_matches_provided(
    provided: dict[str, DatasetResult], expected: dict[str, Any], label: str
) -> None:
    """补跑前预检：新 run 将写入的字段必须与已有 run 一致（不一致在跑之前就拒绝）。"""
    if not provided:
        return
    sample = next(iter(provided.values()))
    diffs = [
        f"{key}: 现有 run={sample.metrics.get(key)!r} vs 本轮补跑={expected.get(key)!r}"
        for key in _IDENTITY_KEYS
        if sample.metrics.get(key) != expected.get(key)
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
            grouped.setdefault(repr(r.metrics.get(key)), []).append(r.spec.key)
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
) -> list[DatasetSpec]:
    """规划全量评估的数据集清单。

    默认槽位顺序：办公域槽位（resolve_dataset 优先级：用户集 > 内置 office）
    -> 按需下载公开集（locomo / longmemeval_oracle / gatemem_office）
    -> longmemeval_s（检索压力变体，默认包含）-> BEAM-1M。

    ``quick=True`` 等价 ``no_beam=True + no_lme_s=True``：跳过 BEAM 与
    LongMemEval-S 两个重型数据集，适合分钟级日常回归。

    所有按需下载公开集都没有内置文件，统一经 ``data.info.resolve_available_path``
    解析可用路径（内置文件优先，其次 dataset fetch 下载目录）；解析不到即
    报错提示先下载，避免拿"路径不存在"的空集当基准、污染可比性。

    Args:
        quick: True 时跳过 BEAM 与 LongMemEval-S 槽位（不检查对应文件）。
        beam_10m: True 时 BEAM 槽位使用 beam_10m.jsonl（与 no_beam/quick 互斥）。
        no_beam: True 时不评测 BEAM 槽位。
        no_lme_s: True 时剔除 LongMemEval-S 变体。

    Returns:
        DatasetSpec 列表。

    Raises:
        SystemExit: beam_10m 与 no_beam/quick 同传；任一按需下载公开集解析
            不到路径；未跳过 BEAM 时 BEAM 文件不存在（提示先 dataset fetch）。
    """
    if beam_10m and (no_beam or quick):
        raise SystemExit(
            "--beam-10m 与 --no-beam / --quick 互斥（BEAM 槽位状态冲突）"
        )
    specs: list[DatasetSpec] = []

    # 办公域槽位：复用 resolve_dataset 的优先级（用户集 > 内置 office）。
    office_path, office_source = resolve_dataset(None)
    office_name = "office（用户测试集）" if office_source == "user" else "office（内置办公黄金集）"
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

    # LongMemEval-S：默认包含（检索/精确率压力变体）；quick / no_lme_s 剔除。
    if not (quick or no_lme_s):
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
    """判断某数据集是否参与某指标的宏平均融合。"""
    rule = _PARTICIPATION_RULES[metric]
    if rule is None:
        return True
    return bool(capabilities.get(rule, False))


def fuse_metrics(results: list[DatasetResult]) -> dict[str, dict[str, Any]]:
    """跨数据集宏平均融合六项质量指标 + 两项诊断指标。

    Args:
        results: 各数据集的评测结果。

    Returns:
        ``{metric: {"value": 融合值, "participants": [参与数据集显示名, ...]}}``；
        某指标无任何参与集时 value 为 None。
    """
    fused: dict[str, dict[str, Any]] = {}
    for metric in FUSED_METRICS:
        values: list[float] = []
        names: list[str] = []
        for res in results:
            if not participates(res.capabilities, metric):
                continue
            overall = res.metrics.get("overall", {})
            v = overall.get(metric)
            # None = 该数据集"未计算"该指标：跳过，不按 0 纳入融合。
            if v is None:
                continue
            values.append(float(v))
            names.append(res.spec.name)
        fused[metric] = {
            "value": (sum(values) / len(values)) if values else None,
            "participants": names,
        }
    return fused


def evaluate_gate(
    results: list[DatasetResult],
    threshold: float = STALE_GATE_THRESHOLD,
) -> dict[str, Any]:
    """办公域 stale 门禁（一票否决，独立于总分）。

    办公域槽位数据集若有 has_stale 标注：其 stale_rate > threshold
    -> FAIL（总评不达标，退出码 1）；无 stale 标注 -> SKIP。

    Returns:
        ``{"status": "PASS"|"FAIL"|"SKIP", "detail": 说明}``。
    """
    office = next((r for r in results if r.spec.is_office_slot), None)
    if office is None:
        return {"status": "SKIP", "detail": "数据集清单中无办公域槽位，stale 门禁跳过。"}
    if not office.capabilities.get("has_stale", False):
        return {
            "status": "SKIP",
            "detail": f"办公域槽位数据集（{office.spec.name}）无 gold_stale 标注，stale 门禁跳过。",
        }
    stale_raw = office.metrics.get("overall", {}).get("stale_rate")
    if stale_raw is None:
        return {
            "status": "SKIP",
            "detail": f"办公域槽位数据集（{office.spec.name}）stale_rate 未计算（None），stale 门禁跳过。",
        }
    stale_rate = float(stale_raw)
    if stale_rate > threshold:
        return {
            "status": "FAIL",
            "detail": (
                f"办公域槽位数据集（{office.spec.name}）stale_rate={stale_rate:.3f} "
                f"> 阈值 {threshold}，一票否决：总评不达标。"
            ),
        }
    return {
        "status": "PASS",
        "detail": (
            f"办公域槽位数据集（{office.spec.name}）stale_rate={stale_rate:.3f} "
            f"<= 阈值 {threshold}。"
        ),
    }


def run_datasets(
    adapter: Any,
    judge: Any,
    specs: list[DatasetSpec],
    out_dir: str,
    platform: Union[str, None] = None,
    commit: Union[str, None] = None,
    jobs: int = 1,
    driver_llm: Union[str, None] = None,
    adapter_version: Union[str, None] = None,
    evidence_matcher: Any = None,
) -> list[DatasetResult]:
    """对每个数据集用 Runner 各跑一次，收集 metrics.json。

    adapter 实例跨数据集复用同一个（reset 由 Runner 内部负责）；
    每个数据集新建 Runner。jobs 透传给 Runner 的题目级并发参数。
    driver_llm / adapter_version 透传给 Runner，记入每个 run 的元数据。
    evidence_matcher 为检索命中口径匹配器（默认 substring）。
    """
    results: list[DatasetResult] = []
    for spec in specs:
        print(f"[membench] ===== 评测数据集: {spec.name} ({spec.path}) =====")
        runner = Runner(
            adapter,
            spec.path,
            judge=judge,
            out_dir=out_dir,
            platform=platform,
            commit=commit,
            dataset_source=spec.source,
            jobs=jobs,
            driver_llm=driver_llm,
            adapter_version=adapter_version,
            evidence_matcher=evidence_matcher,
        )
        run_dir = runner.run()
        if run_dir is None:
            raise SystemExit(f"数据集 {spec.name} 无可评测内容（空数据集），终止评估")
        with open(os.path.join(run_dir, "metrics.json"), encoding="utf-8") as f:
            metrics = json.load(f)
        results.append(
            DatasetResult(
                spec=spec,
                run_dir=run_dir,
                metrics=metrics,
                capabilities=scan_capabilities(spec.path),
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


def write_evaluate_report(
    eval_dir: str,
    adapter_name: str,
    judge_name: str,
    platform: Union[str, None],
    commit: Union[str, None],
    timestamp: datetime,
    results: list[DatasetResult],
    fused: dict[str, dict[str, Any]],
    gate: dict[str, Any],
    extra_notes: Union[list[str], None] = None,
) -> str:
    """生成 evaluate_report.md（中文），写入报告目录并返回报告路径。

    Args:
        extra_notes: 报告顶部追加的提示行（Markdown，如汇总模式的元数据不一致告警）。

    Args:
        eval_dir: 报告目录。各数据集 run 副本已复制到其 ``runs/<run_id>/`` 下，
            故报告中的 run 链接使用相对该目录的 ``runs/<run_id>/report.md``。
        results: 各数据集评测结果（spec 提供数据集范围、来源与路径）。
    """
    gate_status = str(gate["status"])
    overall_fused = {
        k: v["value"] for k, v in fused.items() if v["value"] is not None
    }
    rated = overall_rating(overall_fused)
    verdict = "不达标" if gate_status == "FAIL" else "达标"

    lines: list[str] = []
    lines.append(f"# membench 全量评估报告 — evaluate_{timestamp.strftime('%Y%m%d_%H%M%S')}")
    lines.append("")

    # 门禁结论（报告顶部显著呈现）
    lines.append("## 门禁结论")
    lines.append("")
    gate_icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}.get(gate_status, "")
    lines.append(f"**门禁结论： {gate_status} {gate_icon}** — {gate['detail']}")
    lines.append("")
    lines.append(f"**总评结论： {verdict}**（门禁一票否决，独立于综合评级）")
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

    # 综合评级
    if rated is not None:
        label, mean_points = rated
        lines.append(f"**综合评级：{label}（{mean_points:.2f}/4.0）**")
        lines.append("")
        lines.append(
            "<sub>评级规则：先对六项质量指标（accuracy / token_f1 / bleu1 / "
            "evidence_recall / abstention_accuracy / stale_rate）做跨数据集宏平均融合"
            "（指标仅在具备对应能力标注的数据集上参与），再取融合值的连续点数"
            "（1.0~4.0，档内随数值线性变化）均值，阈值 3.5 / 2.5 / 1.5；"
            "效率成本指标与两项诊断指标（检索召回率@5 / 过度拒答率）不参与评级。"
            "门禁为一票否决，独立于本评级。</sub>"
        )
        lines.append("")

    # 融合指标表
    lines.append("## 融合指标（跨数据集宏平均）")
    lines.append("")
    fused_rows = []
    for metric in FUSED_METRICS:
        entry = fused[metric]
        if entry["value"] is None:
            value_str = "未定义"
            participants_str = "（无参与集）"
        else:
            value_str = format_metric(metric, entry["value"])
            participants_str = "、".join(entry["participants"]) or "（无参与集）"
        fused_rows.append(
            [metric_label(metric), value_str, participants_str]
        )
    lines.append(_md_table(["指标", "融合值", "参与数据集"], fused_rows))
    lines.append("")
    lines.append(
        "<sub>注：检索召回率@5 与过度拒答率为诊断指标（展示与归因用），"
        "不参与综合评级与门禁。</sub>"
    )
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
            if participates(r.capabilities, metric) and metric in overall and overall[metric] is not None:
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
        "abstention 题），不参与该指标计算；融合值也不会因 0 而被拉低。</sub>"
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
        "<sub>诊断指标仅用于分诊与归因，不参与综合评级与门禁；「检索召回率@5」"
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

    # 各数据集 run 目录链接（指向报告目录下复制后的 runs/<run_id>/）
    lines.append("## 各数据集 run 目录")
    lines.append("")
    for r in results:
        rel = f"runs/{os.path.basename(os.path.normpath(r.run_dir))}"
        lines.append(f"- {r.spec.name}: [{rel}]({rel}/report.md)")
    lines.append("")

    out_path = os.path.join(eval_dir, "evaluate_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 报告配图：各数据集质量 + 效率分组条形图
    fig_dataset_overview(
        [(r.spec.name, r.metrics.get("overall", {})) for r in results],
        os.path.join(eval_dir, "fig_dataset_overview.png"),
    )
    return out_path


def evaluate(
    adapter: Any = None,
    judge: Any = None,
    out_dir: Union[str, None] = None,
    platform: Union[str, None] = None,
    commit: Union[str, None] = None,
    quick: bool = False,
    beam_10m: bool = False,
    no_beam: bool = False,
    no_lme_s: bool = False,
    report_out: Union[str, None] = None,
    specs: Union[list[DatasetSpec], None] = None,
    from_runs: Union[dict[str, str], None] = None,
    jobs: int = 1,
    driver_llm: Union[str, None] = None,
    adapter_version: Union[str, None] = None,
    evidence_matcher: Any = None,
) -> int:
    """执行全量评估：多数据集依次评测 + 宏平均融合 + 门禁 + 报告。

    Args:
        adapter: 适配器实例（跨数据集复用，Runner 内部负责 reset）；汇总模式下
            仅当存在需要补跑的数据集时必需，纯复用可为 None。
        judge: judge 名字符串或 Judge 实例（与 Runner 约定一致）；纯复用可为
            None，此时报告头以已有 run 的元数据为准。
        from_runs: 汇总模式：``{数据集 key: 已有 run 目录}``。目标清单（``specs``）
            中有对应 run 的复用其结果、缺失的**补跑**（要求 adapter/judge 可用）；
            未给 ``specs`` 时目标清单即这些 key（纯汇总）。被测对象与评测口径
            （:data:`_IDENTITY_KEYS`）在**开跑之前**校验：已有 run 之间混用、或
            补跑参数与已有 run 不一致，直接拒绝（避免结果不可归因）；全新 run 缺
            ``driver_llm / adapter_version`` 只在启动时提示（严格可比性会判无法核验）。
        out_dir: run 数据与缺省报告目录的父目录；None 时默认 paths.runs_dir()。
        platform: 平台标签。commit: commit 标签。
        quick: 跳过 BEAM 与 LongMemEval-S 槽位（等价 no_beam + no_lme_s）。
        beam_10m: BEAM 槽位用 10m 集。no_beam: 不评测 BEAM 槽位。
        no_lme_s: 剔除 LongMemEval-S 变体（默认包含）。
        report_out: 报告根目录。给定后报告落
            ``<report_out>/eval_<adapter>_<platform>_<ts>/``；None 时落
            ``<out_dir>/evaluate_<ts>/``。两种情况都把各数据集 run 目录复制进
            报告目录的 ``runs/`` 下（自包含，便于就近归属与搬移）。
        specs: 显式数据集清单（默认 None 时按 quick/beam_10m/no_beam/no_lme_s 调 plan_datasets）。
        driver_llm: 驱动被测系统的 LLM 标识，透传给每个 run 的元数据（默认 None）。
        adapter_version: 适配器实现版本标识，透传给每个 run 的元数据（默认 None）。
        evidence_matcher: 检索命中口径匹配器（默认 substring；CLI 以
            ``--evidence-match`` 构造后注入）。

    Returns:
        进程退出码：门禁 FAIL 为 1，否则 0（与 compare 一致）。
    """
    if out_dir is None:
        out_dir = paths.runs_dir()

    if from_runs and specs is None and (quick or beam_10m or no_beam or no_lme_s):
        raise SystemExit("--from-runs 与预设开关（--quick / --beam-10m / --no-beam / --no-lme-s）互斥")

    provided: dict[str, DatasetResult] = {}
    for key, run_dir in (from_runs or {}).items():
        provided[key] = load_provided_result(key, run_dir)

    if specs is None:
        if provided:
            # 纯汇总：目标清单即提供的 key（保持输入顺序）
            specs = [r.spec for r in provided.values()]
        else:
            specs = plan_datasets(quick=quick, beam_10m=beam_10m, no_beam=no_beam, no_lme_s=no_lme_s)
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
    # 报告目录：显式 report_out 时用自包含的 eval_<adapter>_<platform>_<ts>/；
    # 否则保持历史位置 <out_dir>/evaluate_<ts>/（不引入新的测试副作用）。
    if report_out:
        report_dir = layout.experiment_path(
            report_out, layout.subdir_for(adapter_name, platform, layout.ts_now(now))
        )
    else:
        report_dir = os.path.join(out_dir, f"evaluate_{now.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(report_dir, exist_ok=True)

    provided_by_key = dict(provided)
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
            out_dir,
            platform=platform,
            commit=commit,
            jobs=jobs,
            driver_llm=driver_llm,
            adapter_version=adapter_version,
            evidence_matcher=evidence_matcher,
        ):
            provided_by_key[r.spec.key] = r

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

    fused = fuse_metrics(results)
    gate = evaluate_gate(results)

    # 把各数据集 run 目录复制进报告目录的 runs/ 下，使报告与 run 数据就近归属、可整体搬移；
    # 报告里的链接据此指向 runs/<run_id>/report.md。
    copied_runs = layout.copy_runs_into(
        os.path.join(report_dir, "runs"), [r.run_dir for r in results]
    )

    report_path = write_evaluate_report(
        report_dir,
        adapter_name,
        judge_name,
        platform,
        commit,
        now,
        results,
        fused,
        gate,
        extra_notes=[f"> ⚠️ **汇总模式**：跨数据集元数据不一致 —— {note}" for note in notes],
    )

    # 汇总 JSON，便于程序化消费。
    summary = {
        "eval_dir": report_dir,
        "experiment_dir": report_dir,
        "adapter": adapter_name,
        "judge": judge_name,
        "platform": platform or "default",
        "commit": commit,
        "timestamp": now.isoformat(timespec="seconds"),
        "datasets": [r.spec.key for r in results],
        "runs": {
            r.spec.key: os.path.relpath(d, report_dir)
            for r, d in zip(results, copied_runs)
        },
        "fused": fused,
        "gate": gate,
        "token_usage": _token_usage_summary(results),
    }
    with open(os.path.join(report_dir, "evaluate_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[membench] 门禁结论: {gate['status']} — {gate['detail']}")
    print(f"[membench] 报告: {report_path}")

    return 1 if gate["status"] == "FAIL" else 0
