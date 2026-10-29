"""compare：对比两份评测产物（基线 vs 候选），产出逐指标 diff 与回归门禁判定。

门禁规则（SPEC 3.6，阈值可由 CLI 覆盖）：
- accuracy 下降 > acc_threshold（默认 0.02） -> FAIL
- stale_rate 上升 > stale_threshold（默认 0.02） -> FAIL
- latency_p95 上升 > latency_warn（默认 30%） -> WARN
- 其余指标变动仅报告，不门禁。

入参（``--baseline`` / ``--candidate``）是「产物目录」：既可以是**实验目录**（含
``runs/``，其下恰好一个数据集目录），也可以是**数据集目录**
（``<实验目录>/runs/<dataset>``）；指标一律读 ``<数据集目录>/report/metrics.json``。
旧版产物（没有 ``report/`` 层级，或旧的 ``ab_`` 实验目录）不再兼容，直接报错要求重跑
（见 :func:`membench.layout.resolve_dataset_dir`）。

产物布局（见 :mod:`membench.layout`）::

    <out_dir>/compare_<name>_<platform>_<ts>/
    ├── A/  B/                   # 各为一个完整 eval 实验形态（runs/ + reports/）
    └── reports/{comparison.md, fig_metric_diff.png, fig_category_diff.png}

``out_dir`` 是实验目录的**父目录**（缺省 :func:`membench.paths.output_root`）。A/ B/
在入参解析、产物齐备校验与可比性核对完成之后才落地——入参有问题或产物不完整时不留
半个实验目录。

**就地模式**（``task_dir``）：一次 A/B 本来就是**一个任务目录**，两臂已在
``<task_dir>/A`` 与 ``<task_dir>/B``（``task init --arms 2`` 建的骨架，各臂各自跑过
``run`` / ``fuse``）。此时不新建 ``compare_*`` 目录、也不复制两臂——只把
``comparison.md`` 与两张对比图写进 ``<task_dir>/reports/``，两臂产物原样留在原位；
两臂缺省就是该任务目录的 A/ B/，也可显式给 ``dir_a`` / ``dir_b``。

可比性前置校验：比较前先核对驱动/判分模型、适配器实现与数据集
指纹等维度；关键维度缺失或不一致时给出 ``comparability`` 门禁，避免结果不可归因
却照给 PASS。
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Sequence

from membench import layout, runtime
from membench.layout import resolve_dataset_dir
from membench.paths import output_root
from membench.report import _md_table, copy_report, render_compare_report

# 参与逐指标 diff 的 overall 指标（排除计数项）
_DIFF_KEYS: list[str] = [
    # 质量指标（参与评级）
    "accuracy",
    "token_f1",
    "bleu1",
    "evidence_recall",
    "abstention_accuracy",
    "stale_rate",
    # 诊断指标（不参与评级）
    "recall_at_5",
    "over_refusal_rate",
    # 效率成本
    "latency_p50_ms",
    "latency_p95_ms",
    "avg_input_tokens",
    "avg_output_tokens",
    "avg_ingest_latency_ms",
    "avg_llm_calls",
]

#: 可比性维度：任一 critical 维度不一致即视为不可比；
#: 缺值时不记 mismatch，而记入 ``unverifiable``（strict 下同样判不可比）。
CRITICAL_COMPARABILITY_DIMS: tuple[str, ...] = (
    "adapter",
    "adapter_version",
    "dataset_sha256",
    "judge",
    "judge_model",
    "driver_llm",
    "platform",
    # 运行模式：full / replay-restore / replay-rebuild。回放固定写入侧，
    # 与全量分**不可比**（缺省视为 full——本特性之前的所有 run 都是全量跑）。
    "run_mode",
    # 题目数：分数算在哪个题集上。两侧题目数不同说明分数来自不同题目总体
    # （典型如 --limit 冒烟 vs 全量、分片融合只覆盖一部分），diff 不可归因；
    # 与数据集指纹同级——数据集身份没变但题目集变了，同样不能当 A/B。
    "n_questions",
)

#: 次要维度：不一致仅降级为 WARN 门禁，不判 FAIL。
WARN_COMPARABILITY_DIMS: tuple[str, ...] = (
    "limit",
    "shards",
    "shard_index",
    "adapter_meta",
    # 检索命中口径（substring / overlap / llm）不同则分数不可直接对比，
    # 降级 WARN 提示；旧版 run 未落账该字段时两侧同缺，不判不一致。
    "retrieval_match",
    # 融合口径：weighted（按题数加权、office 不参与）/ equal（基准集内等权）/
    # legacy（旧产物：等权且 office 参与，缺省归一）。它只影响**多数据集融合**
    # 的数值，单数据集的指标与它无关——因此不能当关键维度，否则"单集 run 产物
    # vs evaluate 产物"这种常规对比会被误判不可比。降级 WARN，报告里照样列出。
    "fusion",
)

#: 提示维度：仅记录（``same=False``），不影响结论与门禁。
#: ``commit`` 不同是 A/B 的被测变量，**不得**判为不可比。
INFO_COMPARABILITY_DIMS: tuple[str, ...] = ("commit", "dataset_path", "dataset")

#: 维度 -> 严重级别。
_SEVERITY_BY_DIM: dict[str, str] = {
    **{dim: "critical" for dim in CRITICAL_COMPARABILITY_DIMS},
    **{dim: "warn" for dim in WARN_COMPARABILITY_DIMS},
    **{dim: "info" for dim in INFO_COMPARABILITY_DIMS},
}

#: 全量比较维度（critical -> warn -> info，决定 checks 顺序）。
_COMPARABILITY_DIMS: tuple[str, ...] = (
    CRITICAL_COMPARABILITY_DIMS + WARN_COMPARABILITY_DIMS + INFO_COMPARABILITY_DIMS
)


def _load_metrics(target: str) -> dict[str, Any]:
    """读取「产物目录」（实验目录或数据集目录）的数据集级指标。

    入参是对话级产物目录、旧版产物结构或半成品（缺 ``report/metrics.json``）时，
    :func:`membench.layout.resolve_dataset_dir` 直接 SystemExit 并给出可操作提示；
    本函数只负责把已确认存在的指标读出来。
    """
    dataset_dir = resolve_dataset_dir(target)
    path = os.path.join(dataset_dir, layout.DIR_REPORT, layout.FILE_METRICS)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _diff_overall(overall_a: dict[str, Any], overall_b: dict[str, Any]) -> dict[str, dict[str, float]]:
    """逐指标 diff（b - a）。

    任一侧指标为 None（"未计算"）时跳过，不产出 diff——与
    ``summary._mean`` 空集返 None 的语义一致。
    """
    diffs: dict[str, dict[str, float]] = {}
    for key in _DIFF_KEYS:
        if key in overall_a and key in overall_b:
            va = overall_a[key]
            vb = overall_b[key]
            if va is None or vb is None:
                continue
            va_f, vb_f = float(va), float(vb)
            diffs[key] = {"a": va_f, "b": vb_f, "diff": vb_f - va_f}
    return diffs


def _eval_gates(
    overall_a: dict[str, Any],
    overall_b: dict[str, Any],
    acc_threshold: float,
    stale_threshold: float,
    latency_warn: float,
) -> list[dict[str, Any]]:
    """按 SPEC 3.6 逐条评估门禁。"""
    gates: list[dict[str, Any]] = []

    acc_a_raw = overall_a.get("accuracy")
    acc_b_raw = overall_b.get("accuracy")
    # 任一侧"未计算"（None）→ 出 SKIP 行显式说明，不强行转 0 污染判定。
    if acc_a_raw is None or acc_b_raw is None:
        gates.append(
            {
                "name": "accuracy_regression",
                "status": "SKIP",
                "detail": "指标未计算（None），跳过判定",
            }
        )
    else:
        acc_a = float(acc_a_raw)
        acc_b = float(acc_b_raw)
        acc_diff = acc_b - acc_a
        gates.append(
            {
                "name": "accuracy_regression",
                "status": "FAIL" if acc_diff < -acc_threshold else "PASS",
                "detail": f"accuracy diff={acc_diff:+.3f}，阈值 -{acc_threshold}",
            }
        )

    stale_a_raw = overall_a.get("stale_rate")
    stale_b_raw = overall_b.get("stale_rate")
    if stale_a_raw is None or stale_b_raw is None:
        gates.append(
            {
                "name": "stale_rate_regression",
                "status": "SKIP",
                "detail": "指标未计算（None），跳过判定",
            }
        )
    else:
        stale_a = float(stale_a_raw)
        stale_b = float(stale_b_raw)
        stale_diff = stale_b - stale_a
        gates.append(
            {
                "name": "stale_rate_regression",
                "status": "FAIL" if stale_diff > stale_threshold else "PASS",
                "detail": f"stale_rate diff={stale_diff:+.3f}，阈值 +{stale_threshold}",
            }
        )

    p95_a_raw = overall_a.get("latency_p95_ms")
    p95_b_raw = overall_b.get("latency_p95_ms")
    p95_a = float(p95_a_raw) if p95_a_raw is not None else 0.0
    p95_b = float(p95_b_raw) if p95_b_raw is not None else 0.0
    if p95_a > 0:
        p95_rel = (p95_b - p95_a) / p95_a
        gates.append(
            {
                "name": "latency_p95_regression",
                "status": "WARN" if p95_rel > latency_warn else "PASS",
                "detail": f"latency_p95 相对变化={p95_rel:+.1%}，告警阈值 +{latency_warn:.0%}",
            }
        )
    else:
        gates.append(
            {
                "name": "latency_p95_regression",
                "status": "PASS",
                "detail": "基线 latency_p95 为 0 或未计算，跳过相对变化判定",
            }
        )
    return gates


def _comparability_value(metrics: dict[str, Any], dim: str) -> Any:
    """取某维度在 metrics 中的值（``dataset_sha256`` 走指纹子字段）。

    ``run_mode`` 缺省归一为 ``"full"``：本特性之前的所有 run 都是全量跑，
    旧 run 不该因为"没落账这个字段"被记成无法核验。

    ``fusion`` 缺省归一为 ``"legacy"``：旧产物的融合口径是"基准集内等权且
    办公域槽位参与"，与 ``weighted`` / ``equal`` 都不可比——口径变了，分数
    本来就不能直接比。

    ``n_questions`` 顶层没有——它由 :func:`membench.metrics.summary.aggregate`
    写在 ``overall`` 下，故取 ``overall`` 兜底；只读顶层会让该维度恒为
    ``None == None``（死维度，冒烟产物与全量产物因此被当成可比）。
    """
    if dim == "dataset_sha256":
        fingerprint = metrics.get("dataset_fingerprint") or {}
        return fingerprint.get("sha256") if isinstance(fingerprint, dict) else None
    if dim == "run_mode":
        return metrics.get("run_mode") or "full"
    if dim == "fusion":
        return metrics.get("fusion") or "legacy"
    if dim == "n_questions":
        value = metrics.get("n_questions")
        if value is not None:
            return value
        overall = metrics.get("overall")
        return overall.get("n_questions") if isinstance(overall, dict) else None
    return metrics.get(dim)


def _is_missing(value: Any) -> bool:
    """critical 维度的「缺失」判定（None / 空串）；warn/info 维度的 None 是合法取值。"""
    return value is None or value == ""


#: 规则判分器：不存在"判分模型"这一维度。两侧同为规则判分时 judge_model 记
#: 「不适用」（same=True），而非记成无法核验——否则用 heuristic 判分的 A/B
#: 在 strict 模式下会因"判分模型缺失"无条件 FAIL。
_RULE_BASED_JUDGES: tuple[str, ...] = ("heuristic",)


def _is_not_applicable(dim: str, metrics_a: dict[str, Any], metrics_b: dict[str, Any]) -> bool:
    """判断某维度是否「不适用」（区别于元数据缺失，如规则判分无模型标识）。"""
    if dim != "judge_model":
        return False
    judge_a = metrics_a.get("judge")
    return judge_a == metrics_b.get("judge") and judge_a in _RULE_BASED_JUDGES


def _check_reason(check: dict[str, Any]) -> str:
    """把一条 check 转成可读的「不可比原因」文本。"""
    label = {"critical": "关键", "warn": "次要", "info": "提示"}.get(
        check["severity"], check["severity"]
    )
    return f"[{label}] {check['dimension']}: {check['detail']}"


def check_comparability(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    *,
    downgrade: Sequence[str] = (),
) -> dict[str, Any]:
    """核对两个 run 的元数据是否可比。

    ``downgrade`` 里的维度按 **info** 级比较：仍记录 ``same`` 与差异，但既不影响
    ``comparable``、也不进门禁——多线对比用它把 ``adapter_version`` 降级（多线
    场景里版本号正是被测变量，不能拿它判"不可比"）。

    取值规则：``dataset_sha256`` 取 ``metrics["dataset_fingerprint"]["sha256"]``，
    其余维度取 metrics 顶层字段。

    - critical 维度任一值缺失 -> 记入 ``unverifiable``（不记 mismatch）；
      两值都在且不同 -> 记入 ``mismatches``；相同 -> ``same=True``。
      例外：两侧判分器同为规则判分（``heuristic``）时 ``judge_model`` 记
      「不适用」（``same=True``）——规则判分本就没有模型标识。
    - warn / info 维度同样比较：warn 不一致只降级 WARN 门禁，info 不一致仅记录
      （``same=False``）但不影响结论与门禁。
    - ``comparable = 无 critical mismatch``（缺值不算 mismatch）。

    Returns:
        ``{"comparable": bool, "checks": [...], "mismatches": [...],
        "unverifiable": [...], "reasons": [str]}``。
    """
    checks: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    unverifiable: list[dict[str, Any]] = []

    for dim in _COMPARABILITY_DIMS:
        severity = "info" if dim in downgrade else _SEVERITY_BY_DIM[dim]
        a = _comparability_value(metrics_a, dim)
        b = _comparability_value(metrics_b, dim)
        if severity == "critical" and _is_not_applicable(dim, metrics_a, metrics_b):
            check = {
                "dimension": dim,
                "a": a,
                "b": b,
                "same": True,
                "severity": severity,
                "detail": "两侧均未记录（规则判分无模型标识，不适用）",
            }
        elif severity == "critical" and (_is_missing(a) or _is_missing(b)):
            check = {
                "dimension": dim,
                "a": a,
                "b": b,
                "same": False,
                "severity": severity,
                "detail": f"关键维度缺失，无法核验（A={a!r}, B={b!r}）",
            }
            unverifiable.append(check)
        elif a != b:
            check = {
                "dimension": dim,
                "a": a,
                "b": b,
                "same": False,
                "severity": severity,
                "detail": f"A={a} != B={b}",
            }
            mismatches.append(check)
        else:
            check = {
                "dimension": dim,
                "a": a,
                "b": b,
                "same": True,
                "severity": severity,
                "detail": "一致",
            }
        checks.append(check)

    comparable = not any(c["severity"] == "critical" for c in mismatches)
    # 「不可比原因」只列 mismatch（非 info 级）与无法核验项；info 级差异
    # （commit / dataset_path / dataset）只在逐维度表里呈现。
    reasons = [_check_reason(c) for c in mismatches if c["severity"] != "info"] + [
        _check_reason(c) for c in unverifiable
    ]
    return {
        "comparable": comparable,
        "checks": checks,
        "mismatches": mismatches,
        "unverifiable": unverifiable,
        "reasons": reasons,
    }


def _comparability_gate(
    comparability: dict[str, Any], require_comparable: bool
) -> dict[str, Any]:
    """把可比性结论转成首位门禁条目。

    strict（``require_comparable``）下 critical mismatch 或 critical unverifiable
    -> FAIL；非 strict 下 critical mismatch -> WARN；任一 warn 级 mismatch -> WARN
    （strict 下同样 WARN，不升 FAIL）；否则 PASS。
    """
    critical_mismatch = [c for c in comparability["mismatches"] if c["severity"] == "critical"]
    warn_mismatch = [c for c in comparability["mismatches"] if c["severity"] == "warn"]
    unverifiable = comparability["unverifiable"]

    if require_comparable and (critical_mismatch or unverifiable):
        status = "FAIL"
    elif critical_mismatch or warn_mismatch:
        status = "WARN"
    else:
        status = "PASS"

    strict_note = "（strict 模式）" if require_comparable else ""
    detail = (
        f"关键维度不一致 {len(critical_mismatch)} 项，"
        f"次要维度不一致 {len(warn_mismatch)} 项，"
        f"无法核验 {len(unverifiable)} 项{strict_note}"
    )
    return {"name": "comparability", "status": status, "detail": detail}


def _run_id(metrics: dict[str, Any], target: str) -> str:
    """run 身份：优先 metrics 的 run_id，缺失时退回入参目录名。"""
    return str(metrics.get("run_id") or os.path.basename(os.path.normpath(target)))


def _series_name(metrics: dict[str, Any]) -> str:
    """实验目录名里的被测系统片段（adapter，缺失时退回 run_id）。"""
    return str(metrics.get("adapter") or metrics.get("run_id") or "unknown")


def _platform_tag(metrics_a: dict[str, Any], metrics_b: dict[str, Any]) -> str:
    """实验目录名里的平台片段：两臂同平台取该平台，不同平台拼 ``p1-p2``。"""
    platform_a = str(metrics_a.get("platform") or "default")
    platform_b = str(metrics_b.get("platform") or "default")
    return platform_a if platform_a == platform_b else f"{platform_a}-{platform_b}"


def _dataset_name(metrics: dict[str, Any], target: str) -> str:
    """数据集名：优先 metrics 记的名字，退回入参目录名（都按 layout 规则 slug 化）。"""
    return layout.slug(str(metrics.get("dataset") or os.path.basename(os.path.normpath(target))))


#: 数据集产物里支撑 A/B 副本的必需文件（缺任一即视为产物不完整）。
_REQUIRED_PRODUCT_FILES: tuple[str, ...] = (layout.FILE_METRICS, layout.FILE_REPORT_MD)


def _require_dataset_product(dataset_dir: str) -> None:
    """校验数据集产物齐备（在复制 A/B 副本 / 就地写报告之前调用）。

    缺文件要在这里就拦下：复制之后再炸会留下半个实验目录（A/ 已落地、
    ``reports/`` 未生成）。措辞与 :func:`membench.layout._require_metrics` 一致。

    Raises:
        SystemExit: ``report/`` 下缺 :data:`_REQUIRED_PRODUCT_FILES` 中的任一文件。
    """
    for name in _REQUIRED_PRODUCT_FILES:
        path = os.path.join(dataset_dir, layout.DIR_REPORT, name)
        if not os.path.isfile(path):
            raise SystemExit(f"数据集目录缺少 {name}（产物不完整）: {dataset_dir}")


def _ensure_task_reports(arm_dir: str, dataset_dir: str) -> None:
    """补齐 A/B 副本的任务级 ``reports/``（形态与 Runner 的 write_task_reports 一致）。"""
    task_dir = layout.task_report_dir(arm_dir)
    if os.path.isfile(os.path.join(task_dir, layout.FILE_REPORT_MD)):
        return
    report_dir = os.path.join(dataset_dir, layout.DIR_REPORT)
    copy_report(
        os.path.join(report_dir, layout.FILE_REPORT_MD),
        task_dir,
        layout.FILE_REPORT_MD,
        extra_files=[os.path.join(report_dir, layout.FILE_METRICS)],
    )


def _copy_arm(compare_dir: str, arm: str, target: str, metrics: dict[str, Any]) -> str:
    """把一侧产物整理成 ``<compare_dir>/<arm>/``——一个完整的 eval 实验形态。

    入参是实验目录时整树复制（连同 ``runs/`` 与 ``reports/``）；入参是数据集目录时
    构造成 ``runs/<dataset>/`` 并从数据集报告生成任务级 ``reports/``，与
    :class:`membench.runner.Runner` 的产物形态一致。
    """
    src = os.path.abspath(target)
    arm_dir = os.path.join(compare_dir, arm)
    if os.path.isdir(layout.runs_root(src)):
        shutil.copytree(src, arm_dir)
        # 副本里的数据集目录：源实验目录内的相对位置原样保留
        in_arm = os.path.join(arm_dir, os.path.relpath(resolve_dataset_dir(src), src))
    else:
        in_arm = layout.dataset_dir(arm_dir, _dataset_name(metrics, src))
        shutil.copytree(src, in_arm)
    _ensure_task_reports(arm_dir, in_arm)
    return arm_dir


def _append_sources_section(md_path: str, sources: dict[str, dict[str, str]]) -> None:
    """在 comparison.md 末尾追加「来源与产物」小节（两侧源目录路径与 run_id）。"""
    rows = [
        [
            role,
            f"`{src['target']}`",
            f"`{src['dataset_dir']}`",
            f"`{src['run_id']}`",
            f"`{os.path.basename(src['arm_dir'])}/`",
        ]
        for role, src in sources.items()
    ]
    block = "\n".join(
        [
            "",
            "## 来源与产物",
            "",
            _md_table(["侧", "源产物目录", "数据集目录", "run_id", "本实验副本"], rows),
            "",
        ]
    )
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(block)


def compare_runs(
    dir_a: str | None = None,
    dir_b: str | None = None,
    acc_threshold: float = 0.02,
    stale_threshold: float = 0.02,
    latency_warn: float = 0.30,
    require_comparable: bool = False,
    out_dir: str | None = None,
    name: str | None = None,
    platform: str | None = None,
    task_dir: str | None = None,
) -> dict[str, Any]:
    """对比两侧的产物目录（实验目录或数据集目录）。

    Args:
        dir_a: 基线产物目录；``task_dir`` 给定时缺省为该任务目录的 ``A/``。
        dir_b: 候选产物目录；``task_dir`` 给定时缺省为该任务目录的 ``B/``。
        acc_threshold: accuracy 允许的最大下降幅度。
        stale_threshold: stale_rate 允许的最大上升幅度。
        latency_warn: latency_p95 相对上升的告警比例（0.30 = 30%）。
        require_comparable: strict 模式；关键维度不一致或缺失 -> 门禁 FAIL。
        out_dir: compare 实验目录的**父目录**；缺省 :func:`membench.paths.output_root`。
        name: 实验目录名里的标签；缺省 ``<A 的 adapter>_vs_<B 的 adapter>``。
        platform: 实验目录名里的平台；缺省按两臂的 platform 取（不同则 ``p1-p2``）。
        task_dir: **就地模式**：输出容器即这个已存在的任务目录——两臂已在
            ``<task_dir>/A`` ``<task_dir>/B``（缺省从这里读，也可用 ``dir_a`` /
            ``dir_b`` 显式指定），报告与两张对比图写 ``<task_dir>/reports/``，
            不新建 ``compare_*`` 目录、不复制两臂。与 ``out_dir`` 互斥。

    Returns:
        ``{"verdict", "diffs", "gates", "run_a", "run_b", "compare_dir",
        "report_path", "arm_a", "arm_b", "sources", "comparability"}``；``gates``
        首位为 ``comparability`` 门禁，``compare_dir`` 指向本次输出容器
        （新建的 ``compare_<name>_<platform>_<ts>`` 实验目录，或就地模式下的
        ``task_dir``）。
    """
    if task_dir is not None:
        if out_dir is not None:
            raise SystemExit("compare 的 task_dir 与 out_dir 互斥：就地对比只给任务目录")
        if not os.path.isdir(task_dir):
            raise SystemExit(
                f"任务目录不存在: {task_dir}；"
                "请先运行 python -m membench.cli task init --name <NAME> --arms 2"
            )
        # 就地：两臂就在任务目录里，缺省读 <task_dir>/A 与 <task_dir>/B
        container = os.path.abspath(task_dir)
        dir_a = os.path.abspath(dir_a) if dir_a else layout.arm_dir(container, layout.DIR_ARM_A)
        dir_b = os.path.abspath(dir_b) if dir_b else layout.arm_dir(container, layout.DIR_ARM_B)
    else:
        if not dir_a or not dir_b:
            raise SystemExit(
                "compare 需要两侧产物目录（dir_a / dir_b），或就地模式的任务目录 task_dir"
            )
        container = ""
    dataset_a = resolve_dataset_dir(dir_a)
    dataset_b = resolve_dataset_dir(dir_b)
    # 产物齐备校验与指标读取都在写报告 / 复制之前完成：两侧都必须是完整产物。
    _require_dataset_product(dataset_a)
    _require_dataset_product(dataset_b)
    metrics_a = _load_metrics(dir_a)
    metrics_b = _load_metrics(dir_b)
    overall_a = metrics_a.get("overall", {})
    overall_b = metrics_b.get("overall", {})

    diffs = _diff_overall(overall_a, overall_b)
    comparability = check_comparability(metrics_a, metrics_b)
    gates = [
        _comparability_gate(comparability, require_comparable),
        *_eval_gates(overall_a, overall_b, acc_threshold, stale_threshold, latency_warn),
    ]

    if any(g["status"] == "FAIL" for g in gates):
        verdict = "FAIL"
    elif any(g["status"] == "WARN" for g in gates):
        verdict = "WARN"
    else:
        verdict = "PASS"

    id_a = _run_id(metrics_a, dir_a)
    id_b = _run_id(metrics_b, dir_b)
    # 入参解析、产物齐备校验与可比性核对都已完成，此后才落地报告与 A/B 副本。
    if task_dir is not None:
        # 就地：容器就是任务目录，两臂已在位——只写 reports/，不新建目录也不复制
        compare_dir = container
        arm_a = layout.arm_dir(container, layout.DIR_ARM_A)
        arm_b = layout.arm_dir(container, layout.DIR_ARM_B)
    else:
        compare_dir = layout.experiment_path(
            out_dir or output_root(),
            layout.compare_subdir(
                name or f"{_series_name(metrics_a)}_vs_{_series_name(metrics_b)}",
                platform or _platform_tag(metrics_a, metrics_b),
            ),
        )
        arm_a = _copy_arm(compare_dir, layout.DIR_ARM_A, dir_a, metrics_a)
        arm_b = _copy_arm(compare_dir, layout.DIR_ARM_B, dir_b, metrics_b)
    # 对比实验目录已确定（新建的 compare_* 或就地任务目录）：运行时数据归它
    runtime.set_dir(compare_dir)
    reports_dir = layout.task_report_dir(compare_dir)
    render_compare_report(
        reports_dir, metrics_a, metrics_b, diffs, gates, verdict, comparability=comparability
    )
    report_path = os.path.join(reports_dir, layout.FILE_COMPARISON_MD)
    sources = {
        "A（基线）": {
            "target": dir_a,
            "dataset_dir": dataset_a,
            "run_id": id_a,
            "arm_dir": arm_a,
        },
        "B（候选）": {
            "target": dir_b,
            "dataset_dir": dataset_b,
            "run_id": id_b,
            "arm_dir": arm_b,
        },
    }
    _append_sources_section(report_path, sources)

    # 逐条打印不可比原因（含"关键维度无法核验"），默认模式也不静默
    for reason in comparability["reasons"]:
        print(f"[membench] 不可比原因: {reason}")

    print(f"[membench] compare {id_a} -> {id_b}: {verdict}")
    for gate in gates:
        print(f"[membench]   gate {gate['name']}: {gate['status']} ({gate['detail']})")
    print(f"[membench] report: {report_path}")
    arm_role = "臂（就地）" if task_dir is not None else "副本"
    print(f"[membench] A {arm_role}: {arm_a}")
    print(f"[membench] B {arm_role}: {arm_b}")

    return {
        "verdict": verdict,
        "diffs": diffs,
        "gates": gates,
        "run_a": id_a,
        "run_b": id_b,
        "compare_dir": compare_dir,
        "report_path": report_path,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "sources": sources,
        "comparability": comparability,
    }
