"""membench 命令行入口：run / evaluate / evaluate-compare / fuse / compare /
matrix / doctor / adapter / user-data / dataset / list-datasets。

用法（SPEC 第 6 节）：
    python -m membench.cli run --adapter mock-strong \
        [--dataset membench/data/sample_dataset.jsonl] --judge heuristic [--out runs/] \
        [--platform wx --commit a1b2c3] [--driver-llm gpt-4o] \
        [--adapter-version a1b2c3] [--report-out DIR] \
        [--evidence-match substring|overlap|llm]
    python -m membench.cli evaluate --adapter mock-strong \
        [--judge heuristic] [--platform wx --commit a1b2c3] [--out runs/] \
        [--driver-llm gpt-4o] [--adapter-version a1b2c3] \
        [--report-out DIR] [--quick | --beam-10m | --no-beam] [--no-lme-s] \
        [--evidence-match substring|overlap|llm] \
        [--dataset KEY|PATH]...   # 显式数据集清单（与预设开关互斥）
    python -m membench.cli evaluate-compare --baseline-adapter A --candidate-adapter B \
        [--judge heuristic] [--platform wx --commit a1b2c3] [--out runs/] \
        [--driver-llm-a gpt-4o] [--adapter-version-a a1b2c3] \
        [--driver-llm-b gpt-4o-mini] [--adapter-version-b d4e5f6] \
        [--acc-threshold 0.02] [--stale-threshold 0.02] [--latency-warn 0.30] \
        [--report-out DIR] [--tag <label>] [--quick | --beam-10m | --no-beam] \
        [--no-lme-s] [--evidence-match substring|overlap|llm] \
        [--dataset KEY|PATH]...   # 两侧使用同一清单（与预设开关互斥）
    python -m membench.cli fuse --runs runs/<shard0>,runs/<shard1> --out runs/<fused>
    python -m membench.cli compare --baseline runs/<run_a> --candidate runs/<run_b> \
        [--acc-threshold 0.02] [--stale-threshold 0.02] [--latency-warn 0.30] \
        [--require-comparable] [--tag <label>] [--report-out DIR]
    python -m membench.cli matrix --runs "wx@a1b2c3=runs/dir1" "fs@a1b2c3=runs/dir2" ... \
        [--out DIR] [--report-out DIR]
    python -m membench.cli doctor --target http://127.0.0.1:8787 \
        [--timeout 10.0] [--api-key K]
    python -m membench.cli adapter install <adapter.py> | list | remove <name>
    python -m membench.cli user-data install <jsonl路径> | status | delete
    python -m membench.cli dataset info [<key>] | dataset fetch <key> [--out DIR]
    python -m membench.cli list-datasets

--report-out 统一语义：实验目录（``eval_*`` / ``ab_*`` / ``matrix_*``）的
父目录；缺省时用 paths.output_root()（MEMBENCH_OUTPUT_DIR > skill 的 output/ >
工程根）。一次实验一个自包含子目录（报告 + 对应 run 副本就近归属）。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

from membench import layout, paths
from membench.adapters import ADAPTERS
from membench.compare import compare_runs
from membench.matrix import compare_matrix
from membench.runner import Runner
from membench.userdata import (
    delete_user_dataset,
    install_user_dataset,
    resolve_dataset,
    user_dataset_status,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="membench",
        description="membench — Agent 记忆系统评测框架",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = sub.add_parser("run", help="对一个记忆系统运行评测数据集")
    p_run.add_argument("--adapter", required=True, help="适配器名（见 list-datasets 提示 / ADAPTERS 注册表）")
    p_run.add_argument(
        "--dataset",
        default=None,
        help="数据集 JSONL 路径；不传时依次回退到已安装的用户测试集、内置 office 数据集"
        "（可用 dataset info 查看全部支持数据集）",
    )
    p_run.add_argument("--judge", default="heuristic", choices=["heuristic", "llm", "client"], help="判分器（默认 heuristic；llm=OpenAI兼容接口；client=agent内置LLM）")
    p_run.add_argument("--out", default=None, help="run 输出父目录（默认 skill/output/runs/）")
    p_run.add_argument("--platform", default=None, help="平台标签（如 wx / fs），记入 run_id 与元数据")
    p_run.add_argument("--commit", default=None, help="commit 标签（如 a1b2c3），记入元数据")
    p_run.add_argument("--driver-llm", default=None, help="驱动被测系统的 LLM 标识（如 gpt-4o），记入元数据供 compare 可比性校验")
    p_run.add_argument("--adapter-version", default=None, help="适配器实现版本/commit，记入元数据供 compare 可比性校验")
    p_run.add_argument("--report-out", default=None, help="实验目录父目录：产出 <DIR>/eval_<adapter>_<platform>_<ts>/（报告+metrics+run 副本）")
    p_run.add_argument("--llm-base-url", default=None, help="judge=llm 时的 OpenAI 兼容 base_url")
    p_run.add_argument("--llm-api-key", default=None, help="judge=llm 时的 API key")
    p_run.add_argument("--llm-model", default=None, help="judge=llm 时的模型名")
    p_run.add_argument("--llm-timeout", type=float, default=None, help="judge=llm 时的请求超时秒数（默认用 judge 内建值 90s）")
    p_run.add_argument(
        "--evidence-match",
        choices=("substring", "overlap", "llm"),
        default="substring",
        help="检索证据命中口径：substring=逐字子串（默认，历史口径）；overlap=token 包含度容错近原文改写；llm=语义判定（复用 --llm-* 配置，失败回退 substring）",
    )
    p_run.add_argument("--evidence-match-threshold", type=float, default=0.6, help="evidence-match=overlap 时的包含度阈值（默认 0.6）")
    p_run.add_argument("--jobs", type=int, default=1, help="题目级并发数（默认 1；ingest 始终串行；query / judge / 证据匹配在此并发下执行，要求 adapter.query 与 judge 线程安全）")
    p_run.add_argument("--client-llm", default=None, help="judge=client 时的 agent 内置 LLM 类路径，如 'my_agent.llm:LLMClient'")
    p_run.add_argument("--limit", type=int, default=None, help="冒烟模式：只评测（分片后的）前 N 道题，run_id 与 metadata 记录 limit")
    p_run.add_argument("--adapter-meta", default=None, help='适配器配置自由文本（"key=value;..."），原样记入 metadata["adapter_meta"]')
    p_run.add_argument("--shards", type=int, default=1, help="数据集分片总数（默认 1 不分片）；上限=对话数（--shard-by dialogue）或题目数（默认 question 维度），超出直接报错")
    p_run.add_argument("--shard-index", type=int, default=0, help="本分片序号（从 0 开始，需 < --shards；与 --limit 组合时先分片再截断）")
    p_run.add_argument("--shard-by", choices=["question", "dialogue"], default="question", help="分片维度：question 按题目（默认）；dialogue 整段选取对话、分片间不重复 ingest，适合几乎每个对话都有题的数据集（如 BEAM）")
    p_run.set_defaults(func=_cmd_run)

    # --- compare ---
    p_cmp = sub.add_parser("compare", help="对比两个 run（基线 vs 候选）并输出门禁结论")
    p_cmp.add_argument("--baseline", required=True, help="基线 run 目录")
    p_cmp.add_argument("--candidate", required=True, help="候选 run 目录")
    p_cmp.add_argument("--acc-threshold", type=float, default=0.02, help="accuracy 允许最大下降（默认 0.02）")
    p_cmp.add_argument("--stale-threshold", type=float, default=0.02, help="stale_rate 允许最大上升（默认 0.02）")
    p_cmp.add_argument("--latency-warn", type=float, default=0.30, help="latency_p95 上升告警比例（默认 0.30）")
    p_cmp.add_argument("--require-comparable", action="store_true", help="strict 可比性：关键维度不一致或无法核验时门禁 FAIL")
    p_cmp.add_argument("--tag", default=None, help="A/B 实验目录标签（缺省由两个 run_id 拼成 <a>_vs_<b>）")
    p_cmp.add_argument("--report-out", default=None, help="实验目录父目录：产出 <DIR>/ab_<tag>_<ts>/（A/B 各自带 runs/ 副本 + compare_report.md）")
    p_cmp.set_defaults(func=_cmd_compare)

    # --- fuse ---
    p_fuse = sub.add_parser("fuse", help="融合同一评测的多个分片 run 为一个完整 run 目录")
    p_fuse.add_argument("--runs", required=True, help="逗号分隔的分片 run 目录列表（需覆盖全部 shard_index）")
    p_fuse.add_argument("--out", required=True, help="融合 run 输出目录")
    p_fuse.set_defaults(func=_cmd_fuse)

    # --- matrix ---
    p_mtx = sub.add_parser("matrix", help="多 run 矩阵对比（platform × commit 维度）")
    p_mtx.add_argument(
        "--runs",
        required=True,
        nargs="+",
        help='带标签的 run 目录，形如 "platform@commit=runs/dir"（label 也接受任意字符串）',
    )
    p_mtx.add_argument("--out", default=None, help="matrix 产物输出目录（缺省 matrix_<时间戳>/）")
    p_mtx.add_argument("--report-out", default=None, help="实验目录父目录：产出 <DIR>/matrix_<ts>/（--out 优先于本参数）")
    p_mtx.set_defaults(func=_cmd_matrix)

    # --- user-data ---
    p_ud = sub.add_parser("user-data", help="用户测试集管理（install / status / delete）")
    ud_sub = p_ud.add_subparsers(dest="user_data_command", required=True)
    p_ud_install = ud_sub.add_parser("install", help="校验并安装用户测试集")
    p_ud_install.add_argument("src", help="用户测试集 JSONL 路径")
    p_ud_install.set_defaults(func=_cmd_user_data_install)
    p_ud_status = ud_sub.add_parser("status", help="查看用户测试集状态")
    p_ud_status.set_defaults(func=_cmd_user_data_status)
    p_ud_delete = ud_sub.add_parser("delete", help="删除用户测试集")
    p_ud_delete.set_defaults(func=_cmd_user_data_delete)

    # --- evaluate ---
    p_ev = sub.add_parser("evaluate", help="全量评估：多数据集依次评测 + 宏平均融合综合评分 + 办公域 stale 门禁")
    p_ev.add_argument("--adapter", required=True, help="适配器名（见 list-datasets 提示 / ADAPTERS 注册表）")
    p_ev.add_argument("--judge", default="heuristic", choices=["heuristic", "llm", "client"], help="判分器（默认 heuristic；llm=OpenAI兼容接口；client=agent内置LLM）")
    p_ev.add_argument("--platform", default=None, help="平台标签（如 wx / fs），记入 run_id 与元数据")
    p_ev.add_argument("--commit", default=None, help="commit 标签（如 a1b2c3），记入元数据")
    p_ev.add_argument("--out", default=None, help="run 与 evaluate 报告输出父目录（默认 skill/output/runs/）")
    p_ev.add_argument("--driver-llm", default=None, help="驱动被测系统的 LLM 标识（如 gpt-4o），记入每个 run 的元数据")
    p_ev.add_argument("--adapter-version", default=None, help="适配器实现版本/commit，记入每个 run 的元数据")
    p_ev.add_argument("--report-out", default=None, help="实验目录父目录：产出 <DIR>/eval_<adapter>_<platform>_<ts>/")
    ev_group = p_ev.add_mutually_exclusive_group()
    ev_group.add_argument("--quick", action="store_true", help="快速模式：跳过 BEAM 与 LongMemEval-S（等价 --no-beam --no-lme-s，分钟级回归）")
    ev_group.add_argument("--beam-10m", action="store_true", help="BEAM 槽位使用 beam_10m.jsonl（默认 beam_1m.jsonl）")
    p_ev.add_argument("--no-beam", action="store_true", help="不评测 BEAM 槽位（与 --beam-10m 互斥）")
    p_ev.add_argument("--no-lme-s", action="store_true", help="剔除 LongMemEval-S 变体（默认包含；~268MB、高成本）")
    p_ev.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="KEY|PATH",
        help="显式指定评测数据集（数据集 key 或 JSONL 路径，可重复，列表中第一个承担办公域槽位）；与预设开关（--quick / --beam-10m / --no-beam / --no-lme-s）互斥",
    )
    p_ev.add_argument("--llm-base-url", default=None, help="judge=llm 时的 OpenAI 兼容 base_url")
    p_ev.add_argument("--llm-api-key", default=None, help="judge=llm 时的 API key")
    p_ev.add_argument("--llm-model", default=None, help="judge=llm 时的模型名")
    p_ev.add_argument("--llm-timeout", type=float, default=None, help="judge=llm 时的请求超时秒数（默认用 judge 内建值 90s）")
    p_ev.add_argument(
        "--evidence-match",
        choices=("substring", "overlap", "llm"),
        default="substring",
        help="检索证据命中口径：substring=逐字子串（默认，历史口径）；overlap=token 包含度容错近原文改写；llm=语义判定（复用 --llm-* 配置，失败回退 substring）",
    )
    p_ev.add_argument("--evidence-match-threshold", type=float, default=0.6, help="evidence-match=overlap 时的包含度阈值（默认 0.6）")
    p_ev.add_argument("--jobs", type=int, default=1, help="题目级并发数（默认 1；ingest 始终串行；query / judge / 证据匹配在此并发下执行，要求 adapter.query 与 judge 线程安全）")
    p_ev.add_argument("--client-llm", default=None, help="judge=client 时的 agent 内置 LLM 类路径，如 'my_agent.llm:LLMClient'")
    p_ev.set_defaults(func=_cmd_evaluate)

    # --- evaluate-compare ---
    p_ev_cmp = sub.add_parser(
        "evaluate-compare",
        help="两组（baseline vs candidate）× 多数据集融合 A/B 对比；产出 ab_<tag>_<ts>/",
    )
    p_ev_cmp.add_argument("--baseline-adapter", required=True, help="基线适配器名")
    p_ev_cmp.add_argument("--candidate-adapter", required=True, help="候选适配器名")
    p_ev_cmp.add_argument("--judge", default="heuristic", choices=["heuristic", "llm", "client"], help="判分器（默认 heuristic）")
    p_ev_cmp.add_argument("--platform", default=None, help="平台标签（如 wx / fs），记入两侧 run 元数据")
    p_ev_cmp.add_argument("--commit", default=None, help="commit 标签，记入两侧 run 元数据")
    p_ev_cmp.add_argument("--out", default=None, help="run 输出父目录（默认 paths.runs_dir()）")
    p_ev_cmp.add_argument("--driver-llm-a", default=None, help="基线侧的 driver LLM 标识")
    p_ev_cmp.add_argument("--driver-llm-b", default=None, help="候选侧的 driver LLM 标识")
    p_ev_cmp.add_argument("--adapter-version-a", default=None, help="基线侧 adapter 版本/commit")
    p_ev_cmp.add_argument("--adapter-version-b", default=None, help="候选侧 adapter 版本/commit")
    p_ev_cmp.add_argument("--acc-threshold", type=float, default=0.02, help="accuracy 允许最大下降（融合维度，默认 0.02）")
    p_ev_cmp.add_argument("--stale-threshold", type=float, default=0.02, help="stale_rate 允许最大上升（融合维度，默认 0.02）")
    p_ev_cmp.add_argument("--latency-warn", type=float, default=0.30, help="latency_p95 相对上升告警比例（融合维度，默认 0.30）")
    p_ev_cmp.add_argument("--tag", default=None, help="A/B 实验目录标签（缺省由两个 adapter 名拼成）")
    p_ev_cmp.add_argument("--report-out", default=None, help="实验目录父目录：产出 <DIR>/ab_<tag>_<ts>/")
    ev_cmp_group = p_ev_cmp.add_mutually_exclusive_group()
    ev_cmp_group.add_argument("--quick", action="store_true", help="快速模式：跳过 BEAM 与 LongMemEval-S（等价 --no-beam --no-lme-s）")
    ev_cmp_group.add_argument("--beam-10m", action="store_true", help="BEAM 槽位使用 beam_10m.jsonl")
    p_ev_cmp.add_argument("--no-beam", action="store_true", help="不评测 BEAM 槽位（与 --beam-10m 互斥）")
    p_ev_cmp.add_argument("--no-lme-s", action="store_true", help="剔除 LongMemEval-S 变体（默认包含）")
    p_ev_cmp.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="KEY|PATH",
        help="显式指定评测数据集（数据集 key 或 JSONL 路径，可重复，两侧使用同一清单，第一个承担办公域槽位）；与预设开关（--quick / --beam-10m / --no-beam / --no-lme-s）互斥",
    )
    p_ev_cmp.add_argument("--llm-base-url", default=None, help="judge=llm 时的 OpenAI 兼容 base_url")
    p_ev_cmp.add_argument("--llm-api-key", default=None, help="judge=llm 时的 API key")
    p_ev_cmp.add_argument("--llm-model", default=None, help="judge=llm 时的模型名")
    p_ev_cmp.add_argument("--llm-timeout", type=float, default=None, help="judge=llm 时的请求超时秒数（默认 90s）")
    p_ev_cmp.add_argument(
        "--evidence-match",
        choices=("substring", "overlap", "llm"),
        default="substring",
        help="检索证据命中口径（两侧一致）：substring=逐字子串（默认，历史口径）；overlap=token 包含度容错近原文改写；llm=语义判定（复用 --llm-* 配置，失败回退 substring）",
    )
    p_ev_cmp.add_argument("--evidence-match-threshold", type=float, default=0.6, help="evidence-match=overlap 时的包含度阈值（默认 0.6）")
    p_ev_cmp.add_argument("--jobs", type=int, default=1, help="题目级并发数（默认 1；ingest 始终串行；query / judge / 证据匹配在此并发下执行，要求 adapter.query 与 judge 线程安全）")
    p_ev_cmp.add_argument("--client-llm", default=None, help="judge=client 时的 agent 内置 LLM 类路径")
    p_ev_cmp.set_defaults(func=_cmd_evaluate_compare)

    # --- adapter ---
    p_ad = sub.add_parser("adapter", help="业务适配器持久化管理（install / list / remove）")
    ad_sub = p_ad.add_subparsers(dest="adapter_command", required=True)
    p_ad_install = ad_sub.add_parser("install", help="把适配器 .py 落盘到持久化目录，跨 session 可用")
    p_ad_install.add_argument("src", help="适配器 .py 源文件路径")
    p_ad_install.add_argument("--name", default=None, help="注册名（缺省用文件名 stem；源码可推断时以源码为准）")
    p_ad_install.set_defaults(func=_cmd_adapter_install)
    p_ad_list = ad_sub.add_parser("list", help="列出内置与持久化适配器")
    p_ad_list.set_defaults(func=_cmd_adapter_list)
    p_ad_remove = ad_sub.add_parser("remove", help="删除持久化适配器")
    p_ad_remove.add_argument("name", help="安装时的文件名（<name>.py / <name>.json）")
    p_ad_remove.set_defaults(func=_cmd_adapter_remove)

    # --- list-datasets ---
    p_ls = sub.add_parser("list-datasets", help="列出内置数据集")
    p_ls.set_defaults(func=_cmd_list_datasets)

    # --- dataset ---
    p_ds = sub.add_parser("dataset", help="数据集介绍与按需下载（info / fetch）")
    ds_sub = p_ds.add_subparsers(dest="dataset_command", required=True)
    p_ds_info = ds_sub.add_parser("info", help="介绍数据集；不传 name 时概览全部支持的数据集")
    p_ds_info.add_argument("name", nargs="?", default=None, help="数据集 key（可选；未知 key 列出全部可用 key）")
    p_ds_info.set_defaults(func=_cmd_dataset_info)
    p_ds_fetch = ds_sub.add_parser("fetch", help="下载并转换需按需获取的数据集（locomo / longmemeval_* / gatemem_office / beam_*）")
    p_ds_fetch.add_argument("name", help="数据集 key（仅支持 status=fetch 的条目）")
    p_ds_fetch.add_argument("--out", default=None, help="输出目录（默认 MEMBENCH_DATASETS_DIR 或 ~/.membench/datasets/）")
    p_ds_fetch.set_defaults(func=_cmd_dataset_fetch)

    # --- doctor ---
    p_doc = sub.add_parser("doctor", help="探测被测服务是否符合 HttpAdapter 契约（reset/ingest/query 三端点）")
    p_doc.add_argument("--target", required=True, help="被测服务根地址，如 http://127.0.0.1:8787")
    p_doc.add_argument("--timeout", type=float, default=10.0, help="单次请求超时秒数（默认 10.0）")
    p_doc.add_argument("--api-key", default=None, help="非空时给每个请求加 Authorization: Bearer <key>")
    p_doc.set_defaults(func=_cmd_doctor)

    return parser


def _make_adapter(name: str):
    """按注册表名实例化适配器。"""
    if name not in ADAPTERS:
        available = ", ".join(sorted(ADAPTERS)) or "(空)"
        raise SystemExit(f"未知适配器: {name}。可用: {available}")
    return ADAPTERS[name]()


def _make_judge(args: argparse.Namespace):
    """构造判分器；judge=llm 时把 --llm-* 参数传给 LLMJudge，
    judge=client 时用 --client-llm 指定的类路径构造 ClientLLMJudge。"""
    if args.judge == "llm":
        from membench.judges.llm import LLMJudge

        kwargs = {}
        # --llm-timeout 为 None 时使用 LLMJudge 内建默认值（90s）
        if getattr(args, "llm_timeout", None) is not None:
            kwargs["timeout"] = args.llm_timeout
        return LLMJudge(
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            model=args.llm_model,
            **kwargs,
        )
    if args.judge == "client":
        from membench.judges.client import ClientLLMJudge

        if not args.client_llm:
            raise SystemExit(
                "--judge client 需要 --client-llm 'module:ClassName'（agent 内置 LLM 客户端类路径）"
            )
        return ClientLLMJudge(args.client_llm)
    return args.judge  # "heuristic"，由 Runner 经 get_judge 解析


def _make_evidence_matcher(args: argparse.Namespace):
    """构造检索命中口径匹配器（--evidence-match；llm 复用 --llm-* 配置）。"""
    from membench.metrics.evidence import build_evidence_matcher

    try:
        return build_evidence_matcher(
            args.evidence_match,
            threshold=args.evidence_match_threshold,
            llm_base_url=getattr(args, "llm_base_url", None),
            llm_api_key=getattr(args, "llm_api_key", None),
            llm_model=getattr(args, "llm_model", None),
            llm_timeout=getattr(args, "llm_timeout", None) or 90.0,
        )
    except ValueError as exc:
        raise SystemExit(f"[membench] 参数错误: {exc}")


def _cmd_run(args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit < 1:
        raise SystemExit(f"--limit 必须 >= 1，当前为 {args.limit}")
    if args.shards < 1:
        raise SystemExit(f"--shards 必须 >= 1，当前为 {args.shards}")
    if not 0 <= args.shard_index < args.shards:
        raise SystemExit(
            f"--shard-index 必须满足 0 <= I < --shards，当前为 {args.shard_index}/{args.shards}"
        )
    adapter = _make_adapter(args.adapter)
    judge = _make_judge(args)
    evidence_matcher = _make_evidence_matcher(args)
    dataset_path, dataset_source = resolve_dataset(args.dataset)
    print(f"[membench] 测试集: {dataset_path} (来源: {dataset_source})")
    runner = Runner(
        adapter,
        dataset_path,
        judge=judge,
        out_dir=args.out,
        platform=args.platform,
        commit=args.commit,
        dataset_source=dataset_source,
        jobs=args.jobs,
        limit=args.limit,
        shards=args.shards,
        shard_index=args.shard_index,
        shard_by=args.shard_by,
        adapter_meta=args.adapter_meta,
        driver_llm=args.driver_llm,
        adapter_version=args.adapter_version,
        report_out=args.report_out,
        evidence_matcher=evidence_matcher,
    )
    try:
        run_dir = runner.run()
    except ValueError as exc:
        # 数据集相关参数错误（分片数超限、schema 校验失败等）→ 友好退出
        raise SystemExit(f"[membench] 参数错误: {exc}")
    # report_out 时 Runner 已自建 <DIR>/eval_<adapter>_<platform>_<ts>/ 并复制 run 副本
    print(f"[membench] 完成: {run_dir}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    result = compare_runs(
        args.baseline,
        args.candidate,
        acc_threshold=args.acc_threshold,
        stale_threshold=args.stale_threshold,
        latency_warn=args.latency_warn,
        require_comparable=args.require_comparable,
        out_dir=args.report_out,
        tag=args.tag,
    )
    # compare_runs 已打印 verdict / 逐门禁 / 不可比原因 / 报告路径；此处补实验目录便于脚本消费
    print(f"[membench] verdict: {result['verdict']}")
    print(f"[membench] compare_dir: {result['compare_dir']}")
    # FAIL 时以非零码退出，便于 CI 门禁集成
    return 1 if result["verdict"] == "FAIL" else 0


def _cmd_fuse(args: argparse.Namespace) -> int:
    """fuse：融合多个分片 run；stale 门禁 FAIL 时以非零码退出。"""
    from membench.fuse import fuse_runs

    run_dirs = [d.strip() for d in args.runs.split(",") if d.strip()]
    try:
        result = fuse_runs(run_dirs, args.out)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"[membench] {exc}")
    gate = result["gate"]
    print(f"[membench] 融合完成: {result['n_questions']} 题 -> {result['out_dir']}")
    print(f"[membench] stale 门禁: {gate['status']} — {gate['detail']}")
    return 1 if gate["status"] == "FAIL" else 0


def _cmd_matrix(args: argparse.Namespace) -> int:
    labeled_runs: list[tuple[str, str]] = []
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"--runs 项缺少 '=' 分隔符: {spec}（形如 label=runs/dir）")
        label, _, run_dir = spec.partition("=")
        if not label or not run_dir:
            raise SystemExit(f"--runs 项格式非法: {spec}（形如 label=runs/dir）")
        labeled_runs.append((label, run_dir))

    if args.out:
        out_dir = args.out
    elif args.report_out:
        # report_out 语义统一为实验目录父目录
        out_dir = layout.experiment_path(args.report_out, layout.matrix_subdir())
    else:
        out_dir = layout.experiment_path(paths.output_root(), layout.matrix_subdir())
    result = compare_matrix(labeled_runs, out_dir)
    worst = "PASS"
    for pair in result["pairwise"]:
        if pair["verdict"] == "FAIL":
            worst = "FAIL"
        elif pair["verdict"] == "WARN" and worst != "FAIL":
            worst = "WARN"
    print(f"[membench] matrix verdict: {worst}")
    print(f"[membench] report: {result['report_path']}")
    return 0


def _cmd_user_data_install(args: argparse.Namespace) -> int:
    try:
        stats = install_user_dataset(args.src)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"安装失败，测试集校验未通过: {exc}")
    print(f"[membench] 用户测试集已安装: {stats['path']}")
    print(
        f"[membench] dialogues={stats['n_dialogues']} questions={stats['n_questions']} "
        f"source={stats['source']}"
    )
    return 0


def _cmd_user_data_status(args: argparse.Namespace) -> int:
    status = user_dataset_status()
    if not status["exists"]:
        print("[membench] 未安装用户测试集（run 不传 --dataset 时将回退内置 office 数据集）")
        return 0
    print(f"[membench] 用户测试集: {status['path']}")
    meta = status["metadata"]
    if meta:
        print(f"[membench] installed_at={meta.get('installed_at', 'n/a')} "
              f"source={meta.get('source', 'n/a')} "
              f"dialogues={meta.get('n_dialogues', 'n/a')} "
              f"questions={meta.get('n_questions', 'n/a')}")
    else:
        print("[membench] (metadata.json 缺失或损坏)")
    return 0


def _cmd_user_data_delete(args: argparse.Namespace) -> int:
    existed = delete_user_dataset()
    if existed:
        print("[membench] 用户测试集已删除")
    else:
        print("[membench] 本就没有用户测试集，无需删除")
    return 0


def _cmd_adapter_install(args: argparse.Namespace) -> int:
    """adapter install：落盘 + 校验；不要求当前进程已加载该适配器。"""
    from membench.adapters import install_adapter

    try:
        result = install_adapter(args.src, name=args.name)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"安装失败: {exc}")
    print(f"[membench] 适配器已安装: {result['path']}")
    print(f"[membench] 注册名: {result['name']}  installed_at: {result['installed_at']}")
    print("[membench] 下次运行任意 membench 命令时将自动加载，无需再设环境变量")
    return 0


def _cmd_adapter_list(args: argparse.Namespace) -> int:
    """adapter list：表格列出内置适配器与持久化适配器（标注来源）。"""
    from membench.adapters import ADAPTERS, list_installed_adapters

    installed = list_installed_adapters()
    installed_names = {m.get("name", "") for m in installed}
    builtin_names = sorted(n for n in ADAPTERS if n not in installed_names)

    rows: list[tuple[str, str, str, str, str]] = []
    for n in builtin_names:
        cls = ADAPTERS[n]
        rows.append((n, cls.__name__, "-", "-", "内置"))
    for m in installed:
        rows.append(
            (
                str(m.get("name", "")),
                str(m.get("class_name", "") or "?"),
                str(m.get("installed_at", "")),
                str(m.get("source_path", "")),
                "持久化",
            )
        )
    if not rows:
        print("[membench] 暂无可用适配器")
        return 0
    headers = ("注册名", "类名", "安装时间", "来源路径", "来源")
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(5)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))
    return 0


def _cmd_adapter_remove(args: argparse.Namespace) -> int:
    """adapter remove：删除持久化适配器文件。"""
    from membench.adapters import remove_adapter

    if remove_adapter(args.name):
        print(f"[membench] 持久化适配器 '{args.name}' 已删除")
    else:
        print(f"[membench] 持久化目录中没有名为 '{args.name}' 的适配器")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from membench.evaluate import dataset_specs, evaluate

    if args.jobs < 1:
        raise SystemExit(f"--jobs 必须 >= 1，当前为 {args.jobs}")
    if args.beam_10m and args.no_beam:
        raise SystemExit("--beam-10m 与 --no-beam 互斥（BEAM 槽位状态冲突）")
    if args.dataset and (
        args.quick or args.beam_10m or args.no_beam or args.no_lme_s
    ):
        raise SystemExit(
            "--dataset 与预设开关（--quick / --beam-10m / --no-beam / --no-lme-s）"
            "互斥：显式清单时请直接枚举全部数据集"
        )
    specs = dataset_specs(args.dataset) if args.dataset else None
    adapter = _make_adapter(args.adapter)
    judge = _make_judge(args)
    return evaluate(
        adapter,
        judge,
        out_dir=args.out,
        platform=args.platform,
        commit=args.commit,
        quick=args.quick,
        beam_10m=args.beam_10m,
        no_beam=args.no_beam,
        no_lme_s=args.no_lme_s,
        report_out=args.report_out,
        jobs=args.jobs,
        driver_llm=args.driver_llm,
        adapter_version=args.adapter_version,
        evidence_matcher=_make_evidence_matcher(args),
        specs=specs,
    )


def _cmd_evaluate_compare(args: argparse.Namespace) -> int:
    """两组（baseline vs candidate）× 多数据集融合 A/B 对比。"""
    from membench.evaluate import dataset_specs
    from membench.evaluate_compare import evaluate_compare

    if args.jobs < 1:
        raise SystemExit(f"--jobs 必须 >= 1，当前为 {args.jobs}")
    if args.beam_10m and args.no_beam:
        raise SystemExit("--beam-10m 与 --no-beam 互斥（BEAM 槽位状态冲突）")
    if args.dataset and (
        args.quick or args.beam_10m or args.no_beam or args.no_lme_s
    ):
        raise SystemExit(
            "--dataset 与预设开关（--quick / --beam-10m / --no-beam / --no-lme-s）"
            "互斥：显式清单时请直接枚举全部数据集"
        )
    specs = dataset_specs(args.dataset) if args.dataset else None
    baseline_adapter = _make_adapter(args.baseline_adapter)
    candidate_adapter = _make_adapter(args.candidate_adapter)
    judge = _make_judge(args)
    result = evaluate_compare(
        baseline_adapter,
        candidate_adapter,
        judge,
        out_dir=args.out,
        platform=args.platform,
        commit=args.commit,
        quick=args.quick,
        beam_10m=args.beam_10m,
        no_beam=args.no_beam,
        no_lme_s=args.no_lme_s,
        report_out=args.report_out,
        tag=args.tag,
        jobs=args.jobs,
        driver_llm_baseline=args.driver_llm_a,
        driver_llm_candidate=args.driver_llm_b,
        adapter_version_baseline=args.adapter_version_a,
        adapter_version_candidate=args.adapter_version_b,
        acc_threshold=args.acc_threshold,
        stale_threshold=args.stale_threshold,
        latency_warn=args.latency_warn,
        evidence_matcher=_make_evidence_matcher(args),
        specs=specs,
    )
    return 1 if result["verdict"] == "FAIL" else 0


def _cmd_list_datasets(args: argparse.Namespace) -> int:
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    print("内置数据集（membench/data/*.jsonl）:")
    found = False
    if os.path.isdir(data_dir):
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".jsonl"):
                continue
            found = True
            path = os.path.join(data_dir, fname)
            info = ""
            try:
                from membench.data.schema import load_dataset

                dialogues = load_dataset(path)
                n_q = sum(len(d.questions) for d in dialogues)
                info = f"  ({len(dialogues)} dialogues, {n_q} questions)"
            except Exception:
                pass
            print(f"  {path}{info}")
    if not found:
        print("  (未找到内置数据集)")
    print("\n可用适配器: " + (", ".join(sorted(ADAPTERS)) or "(无)"))
    return 0


def _cmd_dataset_info(args: argparse.Namespace) -> int:
    """dataset info：无 name 概览全部，有 name 详细介绍单个数据集。"""
    from membench.data import info as dataset_info

    if args.name is None:
        print(dataset_info.describe_all())
        return 0
    try:
        print(dataset_info.describe(args.name))
    except KeyError as exc:
        print(f"[membench] {exc.args[0]}", file=sys.stderr)
        return 2
    return 0


def _cmd_dataset_fetch(args: argparse.Namespace) -> int:
    """dataset fetch：按需下载并转换 status=fetch 的数据集。

    下载+转换实现集中在 ``membench.data.fetch``（已有完整产物则跳过，幂等），CLI 只做
    参数校验与退出码映射：内置数据集无需下载（提示路径，退出码 0）；
    未知 key / 暂不可用 -> 2（并给出可手动获取的来源）；下载失败 -> 1。
    """
    from membench.data import fetch as dataset_fetch
    from membench.data import info as dataset_info

    try:
        entry = dataset_info.get_dataset_info(args.name)
    except KeyError as exc:
        print(f"[membench] {exc.args[0]}", file=sys.stderr)
        return 2

    if entry.status == "builtin":
        path = dataset_info.builtin_path(entry.key)
        print(f"[membench] {entry.key} 为内置数据集，无需下载: {path}")
        print(
            f"[membench] 运行示例: python -m membench.cli run "
            f"--adapter <你的适配器> --dataset {path}"
        )
        return 0

    if not dataset_fetch.is_fetchable(entry.key):
        label = {"unavailable": "暂不可用"}.get(entry.status, entry.status)
        print(
            f"[membench] {entry.key} {label}，不支持按需下载；"
            f"可手动获取来源: {entry.source}",
            file=sys.stderr,
        )
        return 2

    try:
        out_path = dataset_fetch.fetch_dataset(entry.key, args.out)
    except KeyError as exc:  # 注册表标了 fetch 但下载后端未登记
        print(f"[membench] {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:  # 下载源未确证或下载失败
        print(f"[membench] {exc}", file=sys.stderr)
        return 1
    print(f"[membench] 数据集已就绪: {out_path}")
    print(f"[membench] 运行示例: python -m membench.cli run --adapter <你的适配器> --dataset {out_path}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """doctor：探测被测服务 reset/ingest/query 三端点是否符合 HttpAdapter 契约。"""
    from membench.doctor import check_target

    result = check_target(args.target, timeout=args.timeout, api_key=args.api_key)
    for check in result["checks"]:
        print(f"[membench] doctor {check['name']}: {check['status']} — {check['detail']}")
    return 0 if result["ok"] else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口，返回进程退出码。"""
    # 在任何子命令解析前加载外部适配器插件（MEMBENCH_ADAPTER_MODULES）。
    from membench.adapters import load_external_adapters

    load_external_adapters()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except RuntimeError as exc:
        # 如缺少可选依赖（matplotlib）等运行时错误：输出友好提示而非堆栈
        print(f"[membench] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
