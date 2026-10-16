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
from membench.report import _md_table, _fmt_opt, fig_dataset_overview, metric_label
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

#: 办公域 stale 门禁阈值：办公域槽位数据集 stale_rate 超过该值即一票否决。
STALE_GATE_THRESHOLD: float = 0.03

#: 默认参与全量评估的按需下载公开集（键，顺序即评测顺序）。
#: 这些集没有内置文件，须先 dataset fetch 到下载目录，用统一入口解析路径。
_FETCH_DATASETS: tuple[str, ...] = (
    "locomo",
    "longmemeval_oracle",
    "gatemem_office",
)

#: 可选公开集：--include-lme-s 时追加。体积大（~268MB）、ingest 成本高，默认不入全量；
#: 与 oracle 同题，分差反映检索/精确率损失。
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


def plan_datasets(
    quick: bool = False, beam_10m: bool = False, include_lme_s: bool = False
) -> list[DatasetSpec]:
    """规划全量评估的数据集清单。

    槽位顺序：办公域槽位（resolve_dataset 优先级：用户集 > 内置 office）
    -> 按需下载公开集（locomo / longmemeval_oracle / gatemem_office）
    -> 可选：longmemeval_s（--include-lme-s 时追加）
    -> BEAM 槽位（--quick 时跳过；--beam-10m 时换成 10m 集）。

    四个公开集都没有内置文件，统一经 ``data.info.resolve_available_path``
    解析可用路径（内置文件优先，其次 dataset fetch 下载目录）；解析不到即
    报错提示先下载，避免拿"路径不存在"的空集当基准、污染可比性。

    Args:
        quick: True 时跳过 BEAM 槽位（不检查文件是否存在）。
        beam_10m: True 时 BEAM 槽位使用 beam_10m.jsonl。
        include_lme_s: True 时追加 LongMemEval-S 变体（大体积、高成本）。

    Returns:
        DatasetSpec 列表。

    Raises:
        SystemExit: 任一按需下载公开集解析不到路径，或非 quick 模式且 BEAM
            文件不存在时，提示先运行 dataset fetch。
    """
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

    # 可选：LongMemEval-S（大体积、高成本，默认不入全量）。
    if include_lme_s:
        specs.append(_fetch_spec(_LME_S_KEY))

    # BEAM 槽位。
    if not quick:
        beam_key = "beam_10m" if beam_10m else "beam_1m"
        beam_path = _beam_path(beam_10m)
        if not beam_path.is_file():
            raise SystemExit(
                f"BEAM 数据集不存在: {beam_path}。"
                f"请先运行 python -m membench.cli dataset fetch {beam_key}，"
                "或使用 --quick 跳过 BEAM 槽位。"
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
}


def participates(capabilities: dict[str, bool], metric: str) -> bool:
    """判断某数据集是否参与某指标的宏平均融合。"""
    rule = _PARTICIPATION_RULES[metric]
    if rule is None:
        return True
    return bool(capabilities.get(rule, False))


def fuse_metrics(results: list[DatasetResult]) -> dict[str, dict[str, Any]]:
    """跨数据集宏平均融合六项质量指标。

    Args:
        results: 各数据集的评测结果。

    Returns:
        ``{metric: {"value": 融合值, "participants": [参与数据集显示名, ...]}}``；
        某指标无任何参与集时 value 为 None。
    """
    fused: dict[str, dict[str, Any]] = {}
    for metric in QUALITY_METRICS:
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
) -> list[DatasetResult]:
    """对每个数据集用 Runner 各跑一次，收集 metrics.json。

    adapter 实例跨数据集复用同一个（reset 由 Runner 内部负责）；
    每个数据集新建 Runner。jobs 透传给 Runner 的题目级并发参数。
    driver_llm / adapter_version 透传给 Runner，记入每个 run 的元数据。
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
            driver_llm=driver_llm,
            adapter_version=adapter_version,
        )
        run_dir = runner.run()
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
) -> str:
    """生成 evaluate_report.md（中文），写入报告目录并返回报告路径。

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
            "效率成本指标不参与评级。门禁为一票否决，独立于本评级。</sub>"
        )
        lines.append("")

    # 融合指标表
    lines.append("## 融合指标（跨数据集宏平均）")
    lines.append("")
    fused_rows = []
    for metric in QUALITY_METRICS:
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
    adapter: Any,
    judge: Any,
    out_dir: Union[str, None] = None,
    platform: Union[str, None] = None,
    commit: Union[str, None] = None,
    quick: bool = False,
    beam_10m: bool = False,
    include_lme_s: bool = False,
    report_out: Union[str, None] = None,
    specs: Union[list[DatasetSpec], None] = None,
    jobs: int = 1,
    driver_llm: Union[str, None] = None,
    adapter_version: Union[str, None] = None,
) -> int:
    """执行全量评估：多数据集依次评测 + 宏平均融合 + 门禁 + 报告。

    Args:
        adapter: 适配器实例（跨数据集复用，Runner 内部负责 reset）。
        judge: judge 名字符串或 Judge 实例（与 Runner 约定一致）。
        out_dir: run 数据与缺省报告目录的父目录；None 时默认 paths.runs_dir()。
        platform: 平台标签。commit: commit 标签。
        quick: 跳过 BEAM 槽位。beam_10m: BEAM 槽位用 10m 集。
        include_lme_s: 追加 LongMemEval-S 变体（大体积、高成本）。
        report_out: 报告根目录。给定后报告落
            ``<report_out>/eval_<adapter>_<platform>_<ts>/``；None 时落
            ``<out_dir>/evaluate_<ts>/``。两种情况都把各数据集 run 目录复制进
            报告目录的 ``runs/`` 下（自包含，便于就近归属与搬移）。
        specs: 显式数据集清单（默认 None 时按 quick/beam_10m/include_lme_s 调 plan_datasets）。
        driver_llm: 驱动被测系统的 LLM 标识，透传给每个 run 的元数据（默认 None）。
        adapter_version: 适配器实现版本标识，透传给每个 run 的元数据（默认 None）。

    Returns:
        进程退出码：门禁 FAIL 为 1，否则 0（与 compare 一致）。
    """
    if out_dir is None:
        out_dir = paths.runs_dir()

    if specs is None:
        specs = plan_datasets(quick=quick, beam_10m=beam_10m, include_lme_s=include_lme_s)

    judge_name = judge if isinstance(judge, str) else str(getattr(judge, "name", judge.__class__.__name__))
    adapter_name = str(getattr(adapter, "name", adapter.__class__.__name__))

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

    results = run_datasets(
        adapter,
        judge,
        specs,
        out_dir,
        platform=platform,
        commit=commit,
        jobs=jobs,
        driver_llm=driver_llm,
        adapter_version=adapter_version,
    )
    fused = fuse_metrics(results)
    gate = evaluate_gate(results)

    # 把各数据集 run 目录复制进报告目录的 runs/ 下，使报告与 run 数据就近归属、可整体搬移；
    # 报告里的链接据此指向 runs/<run_id>/report.md。
    copied_runs = layout.copy_runs_into(
        os.path.join(report_dir, "runs"), [r.run_dir for r in results]
    )

    report_path = write_evaluate_report(
        report_dir, adapter_name, judge_name, platform, commit, now, results, fused, gate
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
    }
    with open(os.path.join(report_dir, "evaluate_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[membench] 门禁结论: {gate['status']} — {gate['detail']}")
    print(f"[membench] 报告: {report_path}")

    return 1 if gate["status"] == "FAIL" else 0
