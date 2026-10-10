"""membench 命令行入口：run / compare / matrix / user-data / list-datasets。

用法（SPEC 第 6 节）：
    python -m membench.cli run --adapter mock-strong \
        [--dataset membench/data/sample_dataset.jsonl] --judge heuristic --out runs/ \
        [--platform wx --commit a1b2c3] [--report-out DIR]
    python -m membench.cli evaluate --adapter mock-strong \
        [--judge heuristic] [--platform wx --commit a1b2c3] [--out runs/] \
        [--report-out DIR] [--quick | --beam-10m]
    python -m membench.cli compare --baseline runs/<run_a> --candidate runs/<run_b> \
        [--acc-threshold 0.02] [--report-out DIR]
    python -m membench.cli matrix --runs "wx@a1b2c3=runs/dir1" "fs@a1b2c3=runs/dir2" ... \
        [--report-out DIR]
    python -m membench.cli user-data install <jsonl路径> | status | delete
    python -m membench.cli list-datasets
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from membench.adapters import ADAPTERS
from membench.compare import compare_runs
from membench.matrix import compare_matrix
from membench.report import copy_report
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
    p_run.add_argument("--report-out", default=None, help="额外把报告复制到该目录（带时间戳文件名）")
    p_run.add_argument("--llm-base-url", default=None, help="judge=llm 时的 OpenAI 兼容 base_url")
    p_run.add_argument("--llm-api-key", default=None, help="judge=llm 时的 API key")
    p_run.add_argument("--llm-model", default=None, help="judge=llm 时的模型名")
    p_run.add_argument("--llm-timeout", type=float, default=None, help="judge=llm 时的请求超时秒数（默认用 judge 内建值 90s）")
    p_run.add_argument("--jobs", type=int, default=1, help="题目级并发数（默认 1；ingest 始终串行，要求 adapter.query 与 judge 线程安全）")
    p_run.add_argument("--client-llm", default=None, help="judge=client 时的 agent 内置 LLM 类路径，如 'my_agent.llm:LLMClient'")
    p_run.set_defaults(func=_cmd_run)

    # --- compare ---
    p_cmp = sub.add_parser("compare", help="对比两个 run（基线 vs 候选）并输出门禁结论")
    p_cmp.add_argument("--baseline", required=True, help="基线 run 目录")
    p_cmp.add_argument("--candidate", required=True, help="候选 run 目录")
    p_cmp.add_argument("--acc-threshold", type=float, default=0.02, help="accuracy 允许最大下降（默认 0.02）")
    p_cmp.add_argument("--stale-threshold", type=float, default=0.02, help="stale_rate 允许最大上升（默认 0.02）")
    p_cmp.add_argument("--latency-warn", type=float, default=0.30, help="latency_p95 上升告警比例（默认 0.30）")
    p_cmp.add_argument("--report-out", default=None, help="额外把报告复制到该目录（带时间戳文件名）")
    p_cmp.set_defaults(func=_cmd_compare)

    # --- matrix ---
    p_mtx = sub.add_parser("matrix", help="多 run 矩阵对比（platform × commit 维度）")
    p_mtx.add_argument(
        "--runs",
        required=True,
        nargs="+",
        help='带标签的 run 目录，形如 "platform@commit=runs/dir"（label 也接受任意字符串）',
    )
    p_mtx.add_argument("--out", default=None, help="matrix 产物输出目录（默认 skill/output/matrix_<时间戳>）")
    p_mtx.add_argument("--report-out", default=None, help="额外把报告复制到该目录（带时间戳文件名）")
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
    p_ev.add_argument("--report-out", default=None, help="额外把报告复制到该目录（带时间戳文件名）")
    ev_group = p_ev.add_mutually_exclusive_group()
    ev_group.add_argument("--quick", action="store_true", help="快速模式：跳过 BEAM 槽位（不检查 BEAM 文件是否存在）")
    ev_group.add_argument("--beam-10m", action="store_true", help="BEAM 槽位使用 beam_10m.jsonl（默认 beam_1m.jsonl）")
    p_ev.add_argument("--include-lme-s", action="store_true", help="追加 LongMemEval-S 变体（~268MB、高成本，与 oracle 分差反映检索/精确率损失）")
    p_ev.add_argument("--llm-base-url", default=None, help="judge=llm 时的 OpenAI 兼容 base_url")
    p_ev.add_argument("--llm-api-key", default=None, help="judge=llm 时的 API key")
    p_ev.add_argument("--llm-model", default=None, help="judge=llm 时的模型名")
    p_ev.add_argument("--llm-timeout", type=float, default=None, help="judge=llm 时的请求超时秒数（默认用 judge 内建值 90s）")
    p_ev.add_argument("--jobs", type=int, default=1, help="题目级并发数（默认 1；ingest 始终串行，要求 adapter.query 与 judge 线程安全）")
    p_ev.add_argument("--client-llm", default=None, help="judge=client 时的 agent 内置 LLM 类路径，如 'my_agent.llm:LLMClient'")
    p_ev.set_defaults(func=_cmd_evaluate)

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
    p_ds_fetch = ds_sub.add_parser("fetch", help="下载并转换需按需获取的数据集（如 beam_1m / beam_10m）")
    p_ds_fetch.add_argument("name", help="数据集 key（仅支持 status=fetch 的条目）")
    p_ds_fetch.add_argument("--out", default=None, help="输出目录（默认 ~/.membench/datasets/）")
    p_ds_fetch.set_defaults(func=_cmd_dataset_fetch)

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


def _cmd_run(args: argparse.Namespace) -> int:
    adapter = _make_adapter(args.adapter)
    judge = _make_judge(args)
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
    )
    run_dir = runner.run()
    if args.report_out:
        ts = datetime.now().strftime("%y%m%d_%H%M%S")
        platform_tag = args.platform or "default"
        # 每次评测一个独立子目录，避免多次评测产物混在同一目录
        sub_dir = os.path.join(
            args.report_out, f"eval_{adapter.name}_{platform_tag}_{ts}"
        )
        copy_report(
            os.path.join(run_dir, "report.md"),
            sub_dir,
            "report.md",
            "",
            extra_files=[
                os.path.join(run_dir, "results.jsonl"),
                os.path.join(run_dir, "metrics.json"),
            ],
        )
        print(f"[membench] 报告已复制: {sub_dir}/")
    print(f"[membench] 完成: {run_dir}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    result = compare_runs(
        args.baseline,
        args.candidate,
        acc_threshold=args.acc_threshold,
        stale_threshold=args.stale_threshold,
        latency_warn=args.latency_warn,
    )
    print(f"[membench] verdict: {result['verdict']}")
    if args.report_out:
        ts = datetime.now().strftime("%y%m%d_%H%M%S")
        # 每次对比一个独立子目录
        sub_dir = os.path.join(
            args.report_out, f"compare_{result['run_a']}_{result['run_b']}_{ts}"
        )
        copy_report(
            os.path.join(result["compare_dir"], "compare_report.md"),
            sub_dir,
            "compare_report.md",
            "",
        )
        print(f"[membench] 报告已复制: {sub_dir}/")
    # FAIL 时以非零码退出，便于 CI 门禁集成
    return 1 if result["verdict"] == "FAIL" else 0


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
    else:
        # 默认输出到 skill/output/matrix_<时间戳>/；开发环境退到工程根 runs/
        pkg_dir = os.path.dirname(os.path.abspath(__file__))  # membench/membench/
        proj_root = os.path.dirname(pkg_dir)  # membench/（工程根）
        skill_root = os.path.dirname(proj_root)  # skill 根目录
        if os.path.isfile(os.path.join(skill_root, "SKILL.md")):
            out_dir = os.path.join(
                skill_root, "output", f"matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        else:
            out_dir = os.path.join(
                proj_root, "runs", f"matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    result = compare_matrix(labeled_runs, out_dir)
    worst = "PASS"
    for pair in result["pairwise"]:
        if pair["verdict"] == "FAIL":
            worst = "FAIL"
        elif pair["verdict"] == "WARN" and worst != "FAIL":
            worst = "WARN"
    print(f"[membench] matrix verdict: {worst}")
    print(f"[membench] report: {result['report_path']}")
    if args.report_out:
        ts = datetime.now().strftime("%y%m%d_%H%M%S")
        dest = copy_report(
            result["report_path"],
            args.report_out,
            f"matrix_report_{ts}.md",
            ts,
        )
        print(f"[membench] 报告已复制: {dest}")
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
    from membench.evaluate import evaluate

    if args.jobs < 1:
        raise SystemExit(f"--jobs 必须 >= 1，当前为 {args.jobs}")
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
        include_lme_s=args.include_lme_s,
        report_out=args.report_out,
        jobs=args.jobs,
    )


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


#: BEAM 原始数据在 GitHub raw 上的基础 URL。
_BEAM_RAW_BASE = "https://raw.githubusercontent.com/mohammadtavakoli78/BEAM/main/"

#: status=fetch 条目的下载配置：key -> (远端子目录, chat 数, convert_beam 的 scale 标识)。
_FETCH_CONFIG: dict[str, tuple[str, int, str]] = {
    "beam_1m": ("1M", 35, "1m"),
    "beam_10m": ("10M", 10, "10m"),
}

#: 简单数据集的下载配置：key -> [(源URL, 本地保存名), ...]（按转换器所需文件列出）。
#: LoCoMo / LongMemEval / GateMem 都是"下几个文件 + 跑转换器"即可，不需要 BEAM 的多 chat 循环。
_FETCH_SIMPLE: dict[str, list[tuple[str, str]]] = {
    "locomo": [
        ("https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json", "locomo10.json"),
    ],
    "longmemeval_oracle": [
        ("https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json", "longmemeval_oracle.json"),
    ],
    "longmemeval_s": [
        ("https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json", "longmemeval_s_cleaned.json"),
    ],
    "gatemem_office": [
        ("https://raw.githubusercontent.com/rzhub/GateMem/main/bench/data/office/episodes.jsonl", "episodes.jsonl"),
        ("https://raw.githubusercontent.com/rzhub/GateMem/main/bench/data/office/checkpoints.jsonl", "checkpoints.jsonl"),
    ],
}

#: 每个 chat 目录需要下载的文件（相对路径）。
#: 注意：BEAM 仓库实际结构是 chats/{scale}/{id}/probing_questions/probing_questions.json，
#: 不是 chats/{scale}/{id}/pq.json（pq.json 是早期版本或本地缓存的命名）。
_BEAM_CHAT_FILES = ("chat.json", "probing_questions/probing_questions.json")


def _download_file(url: str, dest: Path, retries: int = 3) -> None:
    """下载单个文件（重试 retries 次）；已存在且可解析为 JSON 时跳过（断点续传）。"""
    import json
    import urllib.request

    if dest.is_file():
        try:
            json.loads(dest.read_text(encoding="utf-8"))
            return  # 已下载且完整，跳过
        except (OSError, ValueError):
            pass  # 损坏的半成品，重新下载
    last_exc: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                dest.write_bytes(resp.read())
            return
        except Exception as exc:  # noqa: BLE001 - 网络错误统一重试
            last_exc = exc
            print(f"[membench] 下载失败（第 {attempt}/{retries} 次）: {url}: {exc}", file=sys.stderr)
    raise RuntimeError(f"下载失败（已重试 {retries} 次）: {url}: {last_exc}")


def _fetch_beam(name: str, out_dir: Path) -> Path:
    """下载 BEAM 某规模原始数据并转换为 membench JSONL，返回输出文件路径。"""
    subdir, n_chats, scale = _FETCH_CONFIG[name]
    raw_dir = out_dir / f".{name}_raw"
    for i in range(1, n_chats + 1):
        chat_dir = raw_dir / str(i)
        chat_dir.mkdir(parents=True, exist_ok=True)
        # chat.json 直接下载
        _download_file(
            f"{_BEAM_RAW_BASE}chats/{subdir}/{i}/chat.json",
            chat_dir / "chat.json",
        )
        # probing_questions 在子目录里，下载后保存到 chat_dir 根（转换器期望的位置）
        _download_file(
            f"{_BEAM_RAW_BASE}chats/{subdir}/{i}/probing_questions/probing_questions.json",
            chat_dir / "probing_questions.json",
        )
        print(f"[membench] 已下载 chat {i}/{n_chats}", file=sys.stderr)

    try:
        from membench.data.convert_beam import convert_beam_split
    except ImportError as exc:
        raise SystemExit(
            f"[membench] 转换模块 membench.data.convert_beam 不可用（可能尚未就绪）: {exc}。"
            f"原始数据已下载到 {raw_dir}，稍后可重试。"
        )

    out_path = out_dir / f"{name}.jsonl"
    stats = convert_beam_split(raw_dir, scale, out_path)
    print(
        f"[membench] 转换完成: {stats['dialogues']} dialogues, {stats['questions']} questions"
        f"（跳过 {stats['skipped']} 个损坏 chat）"
    )
    return out_path


def _fetch_simple(name: str, out_dir: Path) -> Path:
    """下载单文件数据集（LoCoMo / LongMemEval / GateMem）并转换为 membench JSONL。"""
    raw_dir = out_dir / f".{name}_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded: dict[str, Path] = {}
    for url, filename in _FETCH_SIMPLE[name]:
        dest = raw_dir / filename
        _download_file(url, dest)
        downloaded[filename] = dest
        print(f"[membench] 已下载 {filename}", file=sys.stderr)

    out_path = out_dir / f"{name}.jsonl"
    if name == "locomo":
        from membench.data.convert_locomo import convert_locomo
        stats = convert_locomo(str(downloaded["locomo10.json"]), str(out_path))
        print(f"[membench] 转换完成: {stats.get('dialogues', '?')} dialogues, {stats.get('questions', '?')} questions")
    elif name.startswith("longmemeval"):
        from membench.data.convert_longmemeval import convert_longmemeval
        src = downloaded.get("longmemeval_s_cleaned.json") or downloaded.get("longmemeval_oracle.json")
        stats = convert_longmemeval(str(src), str(out_path))
        print(f"[membench] 转换完成: {stats.get('questions', '?')} questions")
    elif name == "gatemem_office":
        from membench.data.convert_gatemem import convert_gatemem
        stats = convert_gatemem(
            downloaded["episodes.jsonl"], downloaded["checkpoints.jsonl"], out_path,
            ignore_privacy_refuse=True,
        )
        print(f"[membench] 转换完成: {stats.get('questions', '?')} questions（含 ignore 标记）")
    return out_path


def _cmd_dataset_fetch(args: argparse.Namespace) -> int:
    """dataset fetch：按需下载并转换 status=fetch 的数据集。"""
    from membench.data import info as dataset_info

    try:
        entry = dataset_info.get_dataset_info(args.name)
    except KeyError as exc:
        print(f"[membench] {exc.args[0]}", file=sys.stderr)
        return 2

    if entry.status == "builtin":
        path = dataset_info.builtin_path(entry.key)
        print(f"[membench] {entry.key} 为内置数据集，无需下载: {path}")
        print(f"[membench] 直接运行: python -m membench.cli run --adapter <你的适配器> --dataset {path}")
        return 0
    if entry.status == "unavailable":
        print(f"[membench] {entry.key} 暂不可用，无法下载。原因: {entry.features}")
        return 2
    if entry.key not in _FETCH_CONFIG and entry.key not in _FETCH_SIMPLE:
        print(f"[membench] {entry.key} 暂无下载配置，请按以下来源手动获取: {entry.source}", file=sys.stderr)
        return 2

    out_dir = Path(args.out).expanduser() if args.out else Path.home() / ".membench" / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if entry.key in _FETCH_CONFIG:
            out_path = _fetch_beam(entry.key, out_dir)
        elif entry.key in _FETCH_SIMPLE:
            out_path = _fetch_simple(entry.key, out_dir)
        else:
            raise RuntimeError(f"{entry.key} 暂无下载配置")
    except RuntimeError as exc:
        print(f"[membench] {exc}", file=sys.stderr)
        return 1
    print(f"[membench] 数据集已就绪: {out_path}")
    print(f"[membench] 运行示例: python -m membench.cli run --adapter <你的适配器> --dataset {out_path}")
    return 0


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
