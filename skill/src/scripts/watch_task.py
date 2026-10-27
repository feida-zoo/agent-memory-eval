#!/usr/bin/env python3
"""长跑监控：轮询一个任务目录的进度，给 agent 的"后台执行 + 定时查看"用。

用法::

    python watch_task.py TASK [--interval 20] [--until-done] [--max-wait 3600] [--json]

每次轮询打印一行紧凑进度：已运行时长、各臂 / 各数据集的分片进度（读分片账本）、
数据集级产物是否齐备、快照池桶数（回放任务）、``orchestrator.json`` 的状态与退出码、
最后一条命令的退出码。``--until-done`` 持续轮询直到完成（成功或失败）或超时。

退出码：

- ``0``：已完成且成功（``orchestrator.json`` 的 ``exit_code == 0``），或单次查看；
- ``1``：已完成但失败（退出码非零，或编排进程已不在却仍停在 ``running``）；
- ``2``：超时，或没有 ``orchestrator.json``（非 ``run_task.py --background`` 任务，
  无法判断"完成"）。

``--json`` 每次轮询输出一行机器可读 JSON（JSONL），stdout 只有这些行；人读的提示
一律走 stderr。分片账本 / 快照池的读取全部复用 ``membench.layout`` 的公开 helper，
本脚本不重新解析账本格式。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_task import (  # noqa: E402
    DIR_RUNTIME,
    FILE_ORCHESTRATOR,
    find_membench_root,
)

#: 退出码约定（见模块 docstring）。
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_TIMEOUT = 2


def _modules(root: Path) -> tuple[Any, Any]:
    """按工程根导入 ``membench.layout`` / ``membench.runtime``。"""
    path = str(root)
    if path not in sys.path:
        sys.path.insert(0, path)
    import importlib

    return (
        importlib.import_module("membench.layout"),
        importlib.import_module("membench.runtime"),
    )


# ---------------------------------------------------------------------------
# 采集
# ---------------------------------------------------------------------------


def _parse_time(value: Any) -> Optional[float]:
    """ISO 时间 → epoch 秒；解析不了返回 None。"""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _pid_alive(pid: Any) -> Optional[bool]:
    """进程是否还在（``None`` = 没有 pid，无从判断）。"""
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_orchestrator(task: str) -> Optional[dict[str, Any]]:
    """读 ``<任务>/runtime/orchestrator.json``（没有 / 损坏返回 None）。"""
    path = os.path.join(task, DIR_RUNTIME, FILE_ORCHESTRATOR)
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _orchestrator_state(task: str) -> Optional[dict[str, Any]]:
    """编排状态 + 进程存活判定；``running`` 但进程已不在时报 ``stalled``。"""
    data = _read_orchestrator(task)
    if data is None:
        return None
    status = str(data.get("status") or "unknown")
    pid = data.get("pid")
    alive = _pid_alive(pid) if status == "running" else None
    effective = "stalled" if status == "running" and alive is False else status
    return {
        "status": status,
        "effective_status": effective,
        "exit_code": data.get("exit_code"),
        "pid": pid,
        "pid_alive": alive,
        "started_at": data.get("started_at"),
        "finished_at": data.get("finished_at"),
        "log": data.get("log"),
    }


def _arm_roots(task: str, layout: Any) -> list[tuple[str, str]]:
    """任务下的臂（与 ``task status`` 同口径；单臂任务把任务根当唯一实验根）。"""
    arms = [
        (label, os.path.join(task, label))
        for label in (layout.DIR_ARM_A, layout.DIR_ARM_B)
        if os.path.isdir(os.path.join(task, label))
    ]
    return arms or [("", task)]


def _dataset_progress(arm_root: str, layout: Any) -> list[dict[str, Any]]:
    """一个臂下各数据集的分片进度与数据集级产物状态（账本读取全走 layout）。"""
    runs = os.path.join(arm_root, layout.DIR_RUNS)
    if not os.path.isdir(runs):
        return []
    progress: list[dict[str, Any]] = []
    for name in sorted(os.listdir(runs)):
        dataset_dir = os.path.join(runs, name)
        if name in (layout.DIR_REPORT, "__pycache__") or name.startswith("."):
            continue
        if not os.path.isdir(dataset_dir):
            continue
        ready = os.path.isfile(
            os.path.join(dataset_dir, layout.DIR_REPORT, layout.FILE_METRICS)
        )
        error: Optional[str] = None
        try:
            ledger = layout.read_shard_ledger(dataset_dir)
        except SystemExit as exc:  # 坏账本只报告，不中断进度视图
            ledger, error = None, str(exc)
        if ledger is None:
            shards = 1
            done = 1 if ready else 0
            running = 0
            missing = 0 if ready else 1
        else:
            shards = int(ledger.get("shards") or 1)
            entries = {
                key: entry
                for key, entry in (ledger.get("entries") or {}).items()
                if isinstance(entry, dict)
            }
            done = sum(1 for entry in entries.values() if entry.get("status") == "done")
            running = sum(1 for entry in entries.values() if entry.get("status") == "running")
            missing = max(0, shards - len(entries))
        progress.append(
            {
                "name": name,
                "ready": ready,
                "shards": shards,
                "done": done,
                "running": running,
                "missing": missing,
                "error": error,
            }
        )
    return progress


def _collect_commands(task: str, arms: list[tuple[str, str]], runtime: Any) -> list[dict]:
    """任务根与各臂 runtime 里的全部命令记录（按开始时间排序）。"""
    records: list[dict] = []
    roots = [task] + [root for label, root in arms if label]
    for root in roots:
        records.extend(runtime.read_commands(root))
    records.sort(key=lambda item: (str(item.get("started_at") or ""), int(item.get("pid") or 0)))
    return records


def collect(task: str) -> dict[str, Any]:
    """采集一次进度快照（JSON 结构，见模块 docstring）。"""
    root = find_membench_root()
    layout, runtime = _modules(root)
    task = os.path.abspath(task)
    arms = _arm_roots(task, layout)
    arm_snapshots: list[dict[str, Any]] = []
    for label, arm_root in arms:
        datasets = _dataset_progress(arm_root, layout)
        arm_snapshots.append(
            {
                "label": label,
                "root": arm_root,
                "datasets": datasets,
                "datasets_ready": sum(1 for item in datasets if item["ready"]),
                "datasets_total": len(datasets),
            }
        )

    commands = _collect_commands(task, arms, runtime)
    last = commands[-1] if commands else None
    orchestrator = _orchestrator_state(task)

    started = (
        _parse_time(orchestrator.get("started_at")) if orchestrator is not None else None
    )
    if started is None and commands:
        started = _parse_time(commands[0].get("started_at"))
    elapsed = None
    if started is not None:
        # 已结束的任务显示总时长（finished_at），不要让它随轮询一直涨
        finished = orchestrator.get("finished_at") if orchestrator is not None else None
        if finished is None and commands:
            finished = commands[-1].get("finished_at")
        ended = _parse_time(finished)
        elapsed = max(0.0, (ended if ended is not None else time.time()) - started)

    snapshot_pool = None
    if layout.has_snapshot_pool(task):
        pool_root = layout.snapshot_pool_root(task)
        snapshot_pool = {
            "root": pool_root,
            "buckets": len(layout.list_snapshot_buckets(pool_root)),
        }

    return {
        "task": task,
        "elapsed_seconds": elapsed,
        "orchestrator": orchestrator,
        "arms": arm_snapshots,
        "snapshot_pool": snapshot_pool,
        "last_command": (
            {
                "command": last.get("command"),
                "exit_code": last.get("exit_code"),
                "started_at": last.get("started_at"),
                "duration_ms": last.get("duration_ms"),
                "log": last.get("log"),
            }
            if last is not None
            else None
        ),
        "failed_commands": sum(
            1
            for item in commands
            if isinstance(item.get("exit_code"), int) and item["exit_code"] != 0
        ),
    }


# ---------------------------------------------------------------------------
# 完成判定与展示
# ---------------------------------------------------------------------------


def completion(snapshot: dict[str, Any]) -> tuple[bool, Optional[bool], str]:
    """``(是否完成, 是否成功, 说明)``。

    只有 ``run_task.py --background`` 写的 ``orchestrator.json`` 能判定完成：
    ``done`` / ``failed`` 为终态；``running`` 但 pid 已不在视为 ``stalled``（失败）。
    没有 orchestrator 时返回 ``(True, None, 说明)``——由调用方以超时语义退出。
    """
    orchestrator = snapshot.get("orchestrator")
    if orchestrator is None:
        return (
            True,
            None,
            "没有 orchestrator.json：无法判断完成（该任务不是 run_task.py --background 启动的）",
        )
    status = orchestrator.get("effective_status")
    if status in ("done", "failed", "stalled"):
        code = orchestrator.get("exit_code")
        ok = status == "done" and code == 0
        return True, ok, f"orchestrator {status}（exit_code={code}）"
    return False, None, ""


def _format_elapsed(seconds: Optional[float]) -> str:
    """``HH:MM:SS``（未知返回 ``n/a``）。"""
    if seconds is None:
        return "n/a"
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def _arm_text(arm: dict[str, Any]) -> str:
    """一个臂的分片进度片段：``A: office 2/4(跑1)✗``。"""
    label = arm["label"] or "单臂"
    datasets = arm["datasets"]
    if not datasets:
        return f"{label}: 无数据集"
    parts = []
    for item in datasets:
        state = f"{item['done']}/{item['shards']}片"
        if item["running"]:
            state += f"(跑{item['running']})"
        state += "·产物✓" if item["ready"] else "·产物✗"
        if item["error"]:
            state += "!"
        parts.append(f"{item['name']} {state}")
    return f"{label}: " + " ".join(parts)


def format_line(snapshot: dict[str, Any]) -> str:
    """单行紧凑进度（人读）。"""
    pieces = [f"[watch] {_format_elapsed(snapshot['elapsed_seconds'])}"]
    pieces.extend(_arm_text(arm) for arm in snapshot["arms"])
    if snapshot["snapshot_pool"] is not None:
        pieces.append(f"快照池 {snapshot['snapshot_pool']['buckets']} 桶")
    orchestrator = snapshot["orchestrator"]
    if orchestrator is None:
        pieces.append("orchestrator 无（非后台任务）")
    else:
        status = orchestrator["effective_status"]
        pid = orchestrator.get("pid")
        suffix = f"(pid={pid})" if pid else ""
        if status in ("done", "failed"):
            pieces.append(f"orchestrator {status} exit={orchestrator.get('exit_code')}")
        elif status == "stalled":
            pieces.append(f"orchestrator stalled{suffix}（进程已不在）")
        else:
            pieces.append(f"orchestrator {status}{suffix}")
    last = snapshot["last_command"]
    if last is None:
        pieces.append("暂无命令记录")
    else:
        pieces.append(f"最后命令 {last.get('command')} exit={last.get('exit_code')}")
    if snapshot["failed_commands"]:
        pieces.append(f"失败命令 {snapshot['failed_commands']} 条")
    return " | ".join(pieces)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="watch_task.py",
        description="轮询 membench 任务目录的进度（分片账本 / 产物 / 快照池 / 后台编排状态）。",
        epilog=(
            "退出码: 0=完成且成功（或单次查看）；1=完成但失败；2=超时或无法判断完成。"
        ),
    )
    parser.add_argument("task", metavar="TASK", help="任务目录（run_task.py 的 stdout 那一行）")
    parser.add_argument("--interval", type=float, default=20.0, help="轮询间隔秒数（默认 20）")
    parser.add_argument(
        "--until-done",
        action="store_true",
        help="持续轮询直到完成（成功 / 失败）或超时；默认只看一次",
    )
    parser.add_argument(
        "--max-wait",
        type=float,
        default=3600.0,
        help="--until-done 的最长等待秒数（默认 3600）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="每次轮询输出一行机器可读 JSON（JSONL）；stdout 只有 JSON",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI 入口：返回 0 / 1 / 2（见模块 docstring）。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.interval <= 0:
        print("[watch] 参数错误: --interval 必须 > 0", file=sys.stderr)
        return EXIT_TIMEOUT
    if args.max_wait <= 0:
        print("[watch] 参数错误: --max-wait 必须 > 0", file=sys.stderr)
        return EXIT_TIMEOUT
    if not os.path.isdir(args.task):
        print(f"[watch] 任务目录不存在: {args.task}", file=sys.stderr)
        return EXIT_TIMEOUT

    deadline = time.monotonic() + args.max_wait
    while True:
        try:
            snapshot = collect(args.task)
        except SystemExit as exc:  # 找不到 membench 工程根
            print(f"[watch] {exc}", file=sys.stderr)
            return EXIT_TIMEOUT
        done, ok, reason = completion(snapshot)
        payload = {**snapshot, "done": done, "ok": ok, "reason": reason}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False), flush=True)
        else:
            print(format_line(snapshot), flush=True)
        if not args.until_done:
            return EXIT_OK
        if done:
            if ok is True:
                print(f"[watch] 完成: {reason}", file=sys.stderr)
                return EXIT_OK
            if ok is False:
                print(f"[watch] 失败: {reason}", file=sys.stderr)
                return EXIT_FAILED
            print(f"[watch] 无法判断完成: {reason}", file=sys.stderr)
            return EXIT_TIMEOUT
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(
                f"[watch] 超时：已等待 {args.max_wait:.0f}s，任务仍未完成"
                f"（最后状态：{(snapshot['orchestrator'] or {}).get('effective_status') or 'n/a'}）",
                file=sys.stderr,
            )
            return EXIT_TIMEOUT
        time.sleep(min(args.interval, remaining))


if __name__ == "__main__":
    sys.exit(main())
