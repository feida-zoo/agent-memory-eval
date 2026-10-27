#!/usr/bin/env python3
"""membench 标准流程编排器：把「建任务目录 → 跑分片 → 融合 → A/B 门禁 → 自检」
做成一条不会漏步骤的命令。

用法::

    run_task.py eval --adapter NAME (--dataset KEY|PATH)... \
        --judge {llm,client,heuristic} --recall-mode {substring,overlap,llm} \
        [--platform P] [--commit SHA] [--driver-llm M] [--adapter-version V] \
        [--shards N|max] [--jobs N] [--limit N] [--report-out DIR] [--no-snapshot] \
        [--name TAG] [--evidence-match-threshold X] [--llm-* ...] \
        [--adapter-arg KEY=VALUE]... [--task-dir DIR] [--background] [--dry-run]

    run_task.py ab --baseline-adapter NAME --candidate-adapter NAME \
        (--dataset KEY|PATH)... --judge {...} --recall-mode {...} \
        [--baseline-version V] [--candidate-version V] [--driver-llm-a/-b M] \
        [--platform P] [--commit SHA] [--shards N|max] [--jobs N] [--limit N] \
        [--name TAG] [--report-out DIR] [--no-snapshot] \
        [--adapter-arg-a/-b KEY=VALUE]... [--task-dir DIR] [--background] [--dry-run]

固定流程（不许省步骤）
----------------------

``eval``：

1. ``task init --arms 1`` 建任务目录；
2. 单数据集：直接 ``run --task-dir``；多数据集且**无分片、无 limit**：直接走官方
   ``evaluate --task-dir --dataset ...``；多数据集但**有分片或 limit**：逐数据集
   ``run --task-dir``（数据集之间串行，分片之间并行），最后用
   ``evaluate --task-dir --from-runs`` 汇总任务级报告。**后一条分支是因为
   ``evaluate`` 不支持 ``--shards`` / ``--limit``；逐数据集 run 让多数据集也能
   分片与冒烟。**
3. 分片（``--shards > 1`` 或 ``max``）时，每片 ``--shard-by dialogue`` 并行启动，
   全部 wait；跑完对该数据集目录执行 ``fuse --task-dir`` 就地补齐。
4. 交付前自检（``verify_artifacts.verify_task``）。

``ab``：

1. ``task init --arms 2`` 建 ``A/``、``B/`` 两臂；
2. 两臂各自按上面的方式跑同一批数据集（各自的 adapter / 版本 / driver 参数）；
3. 每个数据集分片跑完后，对该臂的数据集目录 ``fuse --task-dir``；
4. 单数据集：``compare --task-dir --require-comparable``；
   多数据集：先逐数据集 ``compare --task-dir --require-comparable`` 做 strict
   门禁，再用 ``evaluate-compare --task-dir``（复用两臂已有产物）写一份跨数据集
   融合对比报告；
5. 交付前自检。

硬约束
------

- ``--judge heuristic`` 只允许与 ``--limit`` 同时出现（小样本冒烟）；正式评测请用
  ``llm`` / ``client``。
- ``--recall-mode`` 映射到 membench 的 ``--evidence-match``。摘要 / 改写式记忆请用
  ``llm``，否则 ``evidence_recall`` 会系统性偏低。
- ``compare --require-comparable`` 是 strict 门禁：关键维度（``adapter`` /
  ``adapter_version`` / ``driver_llm`` / ``platform`` / ``judge`` 等）缺失或不同
  都会判 FAIL。A/B 只应让被测变量（通常是 ``commit``）不同；正式对比请显式给
  ``--driver-llm-a/-b`` 与 ``--baseline-version/--candidate-version``。
- stdout 只打印任务目录绝对路径（最后一行），进度 / 子进程输出 / 自检摘要全部走
  stderr，因此 ``TASK=$(run_task.py ...)`` 可以直接接走路径；退出码见下。
- ``--background``：父进程先建（或复用 ``--task-dir``）任务目录，把本次编排派生到
  后台（独立会话，输出落 ``<任务>/runtime/logs/orchestrator-<ts>.log``），写
  ``<任务>/runtime/orchestrator.json``（pid / argv / started_at / log /
  status=running）后**stdout 只打印任务目录一行并立即退出 0**；子进程结束时把
  status / exit_code / finished_at 原子更新掉。长跑请配 ``watch_task.py`` 定时看
  进度，不要在同一轮里前台阻塞等待。``--background`` 与 ``--dry-run`` 互斥。
- ``--task-dir DIR``：复用**已存在**的任务目录（跳过 ``task init``），前后台子进程
  共用同一个目录；缺 ``runs/``（eval）或 ``A/`` ``B/``（ab）时拒绝。

退出码
------

- ``0``：流程成功且自检通过（A/B 门禁 PASS / WARN）；
- ``1``：流程成功、自检通过，但 A/B 门禁 FAIL（``evaluate`` 的办公域 stale 门禁
  FAIL 同样用 1；如实向上传递）；
- ``2``：流程失败（参数错误、task init / run / fuse / evaluate / compare 崩溃）；
- ``3``：流程跑完但交付前自检失败（产物不完整 / 不一致 / 有死链）。

自检失败优先于门禁 FAIL：产物本身不完整时，退出码是 3 而不是 1。
``--dry-run`` 只打印将要执行的命令序列（``$TASK`` 是占位符），不建目录、不跑评测；
数据集目录名是预测值（``office`` 槽位与自定义 key 的实际目录名可能不同）。
``--background`` 的父进程只表示"派生成功"（退出码 0）；本次编排的真实退出码写在
``<任务>/runtime/orchestrator.json`` 的 ``exit_code``，用 ``watch_task.py`` 读取。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

#: 本脚本所在目录（同时放 verify_artifacts.py）；直接 import 复用自检逻辑。
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from verify_artifacts import verify_task  # noqa: E402

#: 退出码约定（见模块 docstring）。
EXIT_OK = 0
EXIT_GATE_FAIL = 1
EXIT_FLOW_FAIL = 2
EXIT_VERIFY_FAIL = 3

#: 后台编排的落点（与 membench 的实验级 runtime 同一棵树）。
DIR_RUNTIME = "runtime"
DIR_LOGS = "logs"
FILE_ORCHESTRATOR = "orchestrator.json"

#: 子进程环境变量：标识"我是 --background 派生的编排子进程"（结束时要回写状态）。
ORCHESTRATOR_ENV = "MEMBENCH_ORCHESTRATOR_CHILD"

#: membench 工程根候选（相对脚本目录）：仓库内是 ../../../membench，打包后是 ../membench。
_MEMBENCH_ROOT_CANDIDATES: tuple[str, ...] = (
    "../membench",
    "../../membench",
    "../../../membench",
)

#: 门禁结论的几种输出形态（compare / fuse / evaluate / evaluate-compare）。
_VERDICT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\[membench\] verdict: (PASS|WARN|FAIL)"),
    re.compile(r"\[membench\] 门禁结论: (PASS|WARN|FAIL)"),
    re.compile(r"\[membench\] stale 门禁: (PASS|WARN|FAIL)"),
    re.compile(r"\[membench\] evaluate-compare verdict: (PASS|WARN|FAIL)"),
)

#: 与 membench.layout.slug 相同的目录名安全化规则（dry-run 预测数据集目录用）。
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


# ---------------------------------------------------------------------------
# 异常与数据结构
# ---------------------------------------------------------------------------


class UsageError(Exception):
    """参数错误（退出码 2）。"""


class FlowFailed(Exception):
    """流程失败（退出码 2）；``task_dir`` 已知时供调用方打印现场。"""

    def __init__(self, message: str, task_dir: Optional[str] = None) -> None:
        super().__init__(message)
        self.task_dir = task_dir


@dataclass
class DatasetArg:
    """一个 ``--dataset`` 入参解析后的形态。"""

    raw: str
    cli_value: str
    key: str
    path_based: bool

    @property
    def display(self) -> str:
        """人看的名字。"""
        return self.raw


@dataclass
class ArmArgs:
    """一条臂的 adapter 与元数据。"""

    adapter: str
    driver_llm: Optional[str] = None
    adapter_version: Optional[str] = None
    adapter_args: list[str] = field(default_factory=list)


@dataclass
class CommonArgs:
    """两种模式共用的评测参数。"""

    judge: str
    recall_mode: str
    platform: Optional[str] = None
    commit: Optional[str] = None
    jobs: Optional[int] = None
    limit: Optional[int] = None
    no_snapshot: bool = False
    evidence_match_threshold: Optional[float] = None
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    llm_timeout: Optional[float] = None
    client_llm: Optional[str] = None


@dataclass
class ResolvedDataset:
    """实际跑完的数据集：入参 + 落盘的数据集目录。"""

    dataset: DatasetArg
    directory: str


@dataclass
class FlowState:
    """一次流程的可变结果。"""

    task_dir: str
    gate_fail: bool = False
    datasets: list[ResolvedDataset] = field(default_factory=list)
    arm_datasets: dict[str, dict[str, ResolvedDataset]] = field(default_factory=dict)


@dataclass
class EvalOptions:
    """``eval`` 模式的完整参数。"""

    common: CommonArgs
    arm: ArmArgs
    datasets: list[DatasetArg]
    name: str
    report_out: Optional[str]
    shards: int | str
    task_dir: Optional[str] = None

    @property
    def use_evaluate(self) -> bool:
        """多数据集且无分片 / limit：直接用官方 ``evaluate --dataset`` 路径。

        ``evaluate`` 不支持 ``--shards`` / ``--limit``；这两种情况下退回
        「逐数据集 run + evaluate --from-runs 汇总」（语义等价，且能分片/冒烟）。
        """
        return (
            len(self.datasets) > 1
            and self.shards == 1
            and self.common.limit is None
        )


@dataclass
class AbOptions:
    """``ab`` 模式的完整参数。"""

    common: CommonArgs
    baseline: ArmArgs
    candidate: ArmArgs
    datasets: list[DatasetArg]
    name: str
    report_out: Optional[str]
    shards: int | str
    task_dir: Optional[str] = None


@dataclass
class CommandResult:
    """一次子进程调用的结果。"""

    argv: list[str]
    returncode: int
    output: str = ""


# ---------------------------------------------------------------------------
# 工程根与命令拼装
# ---------------------------------------------------------------------------


def find_membench_root() -> Path:
    """按约定顺序定位 membench 工程根；找不到时报错并列出尝试过的路径。"""
    tried: list[str] = []
    for rel in _MEMBENCH_ROOT_CANDIDATES:
        candidate = (SCRIPT_DIR / rel).resolve()
        tried.append(str(candidate))
        if (candidate / "membench" / "cli.py").is_file():
            return candidate
    raise SystemExit(
        "[run_task] 找不到 membench 工程根（需要包含 membench/cli.py）。已尝试:\n  "
        + "\n  ".join(tried)
        + "\n请确认脚本与 membench 的相对位置（打包后为 <skill>/scripts/ 与 "
        "<skill>/membench/），或重新构建 skill 包。"
    )


def _cli_argv(root: Path, argv: Sequence[str]) -> list[str]:
    """``python -m membench.cli ...`` 的完整 argv（用当前解释器）。"""
    return [sys.executable, "-m", "membench.cli", *argv]


def _slug(text: str) -> str:
    """与 membench.layout.slug 一致的安全名（空串回退 default）。"""
    cleaned = _UNSAFE_RE.sub("-", str(text)).strip("-")
    return cleaned or "default"


def _parse_shards(value: str) -> int | str:
    """``--shards``：>= 1 的整数或 ``max``。"""
    text = str(value).strip().lower()
    if text == "max":
        return "max"
    try:
        number = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--shards 需为 >= 1 的整数或 'max'") from exc
    if number < 1:
        raise argparse.ArgumentTypeError(f"--shards 必须 >= 1 或 'max'，当前为 {number}")
    return number


def _resolve_caller_path(value: str) -> str:
    """把调用方 cwd 下的路径转成绝对路径（不要求存在）。"""
    path = Path(os.path.expanduser(value))
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve())


def _resolve_dataset(raw: str) -> DatasetArg:
    """解析 ``--dataset``：存在的路径转绝对路径，否则按数据集 key 原样传递。

    membench 子进程的 cwd 是工程根，而用户在任意目录调用本脚本；相对路径必须在
    调用方 cwd 下解析后传绝对路径，否则会被工程根重新解释。
    """
    candidate = Path(os.path.expanduser(raw))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.exists():
        if not candidate.is_file():
            raise UsageError(f"--dataset 路径不是文件: {raw}")
        resolved = candidate.resolve()
        return DatasetArg(
            raw=raw,
            cli_value=str(resolved),
            key=resolved.stem,
            path_based=True,
        )
    return DatasetArg(raw=raw, cli_value=raw, key=raw, path_based=False)


def _collect_datasets(values: Sequence[str]) -> list[DatasetArg]:
    """解析并去重 ``--dataset`` 列表。"""
    datasets: list[DatasetArg] = []
    seen: dict[str, str] = {}
    for raw in values:
        text = str(raw).strip()
        if not text:
            raise UsageError("--dataset 收到空值")
        dataset = _resolve_dataset(text)
        # 产物目录名经 layout.slug 归一；用归一后的名字查重，才能拦住 "a b" / "a-b"
        # 这类最终写进同一目录的不同 key。
        dir_key = _slug(dataset.key)
        if dir_key in seen:
            raise UsageError(
                f"数据集产物目录名重复: {dir_key!r}（{seen[dir_key]} 与 {text}）；"
                "同一数据集请只给一次，两个同名文件请先改名"
            )
        seen[dir_key] = text
        datasets.append(dataset)
    return datasets


def _heuristic_smoke_note(common: CommonArgs) -> str:
    """heuristic 冒烟提示。"""
    return (
        f"judge=heuristic 且 limit={common.limit}：这是小样本冒烟，"
        "不能当正式结论（正式评测请用 --judge llm/client）"
    )


# ---------------------------------------------------------------------------
# 任务目录 / 后台编排
# ---------------------------------------------------------------------------


def _membench_module(root: Path, name: str) -> Any:
    """按工程根导入 ``membench.<name>``（目录名 / 脱敏 / 账本逻辑不重复实现）。"""
    path = str(root)
    if path not in sys.path:
        sys.path.insert(0, path)
    import importlib

    return importlib.import_module(f"membench.{name}")


def _validate_task_dir(task_dir: str, arms: int, root: Path) -> None:
    """``--task-dir`` 必须是已有任务骨架（``runs/`` 或 ``A/`` ``B/``，且带 ``reports/``）。"""
    layout = _membench_module(root, "layout")
    if not os.path.isdir(task_dir):
        raise UsageError(
            f"--task-dir 不存在: {task_dir}；先运行 task init 建任务目录，"
            "或去掉 --task-dir 让本脚本自己建"
        )
    needed = (
        (layout.DIR_ARM_A, layout.DIR_ARM_B, layout.DIR_REPORTS)
        if arms == 2
        else (layout.DIR_RUNS, layout.DIR_REPORTS)
    )
    missing = [name for name in needed if not os.path.isdir(os.path.join(task_dir, name))]
    if missing:
        shape = "A/ B/ reports/" if arms == 2 else "runs/ reports/"
        raise UsageError(
            f"--task-dir 不是 arms={arms} 的任务目录（缺 {'、'.join(missing)}）: {task_dir}；"
            f"该形态需要 {shape}"
        )


def _strip_flag(argv: Sequence[str], flag: str) -> list[str]:
    """去掉一个无值开关（``--background``）；它后面紧跟的是别的选项 / 值，不能吞掉。"""
    return [token for token in argv if token != flag]


def _strip_option(argv: Sequence[str], option: str) -> list[str]:
    """去掉 ``--opt VALUE`` / ``--opt=VALUE`` 两种形态（子进程参数重建用）。"""
    result: list[str] = []
    skip_value = False
    for token in argv:
        if skip_value:
            skip_value = False
            continue
        if token == option:
            skip_value = True
            continue
        if token.startswith(f"{option}="):
            continue
        result.append(token)
    return result


def _iso_now() -> str:
    """本地时间（带时区偏移，精确到秒）。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _orchestrator_path(task_dir: str) -> str:
    """``<任务>/runtime/orchestrator.json``。"""
    return os.path.join(task_dir, DIR_RUNTIME, FILE_ORCHESTRATOR)


def _write_json_atomic(path: str, data: dict[str, Any]) -> None:
    """同目录唯一临时文件 + ``os.replace``（父 / 子进程可能同时更新）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _finish_orchestrator(task_dir: str, exit_code: int) -> None:
    """子进程结束时回写 ``orchestrator.json`` 的最终状态（原子写）。

    只更新仍在 ``running`` 的记录：同一个任务目录被人工重跑（前台 ``--task-dir``）
    时，不应把上一次后台编排的历史结论改掉。
    """
    path = _orchestrator_path(task_dir)
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return
    if not isinstance(data, dict) or data.get("status") != "running":
        return
    data["status"] = "done" if exit_code == 0 else "failed"
    data["exit_code"] = int(exit_code)
    data["finished_at"] = _iso_now()
    try:
        _write_json_atomic(path, data)
    except OSError:
        pass


def _start_background(
    options: EvalOptions | AbOptions,
    root: Path,
    raw_argv: Sequence[str],
) -> int:
    """把本次编排派生到后台，stdout 只打印任务目录一行后立即返回 0。

    父进程负责建任务目录（或校验 ``--task-dir``）、开日志、写 ``orchestrator.json``；
    子进程以独立会话运行同一份参数（去掉 ``--background``，补上 ``--task-dir``），
    结束时由 :func:`_finish_orchestrator` 回写状态与退出码。
    """
    arms = 1 if isinstance(options, EvalOptions) else 2
    if options.task_dir is not None:
        task_dir = options.task_dir
    else:
        executor = Executor(root)
        task_dir = executor.init_task(
            _task_init_argv(
                name=options.name,
                arms=arms,
                platform=options.common.platform,
                report_out=options.report_out,
            )
        )

    runtime = _membench_module(root, "runtime")
    log_dir = os.path.join(task_dir, DIR_RUNTIME, DIR_LOGS)
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"orchestrator-{stamp}.log")
    suffix = 1
    while os.path.exists(log_path):
        log_path = os.path.join(log_dir, f"orchestrator-{stamp}-{suffix}.log")
        suffix += 1

    child_argv = _strip_option(_strip_flag(raw_argv, "--background"), "--task-dir")
    child_argv += ["--task-dir", task_dir]
    command = [sys.executable, str(Path(__file__).resolve()), *child_argv]
    env = os.environ.copy()
    env[ORCHESTRATOR_ENV] = "1"
    with open(log_path, "ab") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=os.getcwd(),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _write_json_atomic(
        _orchestrator_path(task_dir),
        {
            "schema": 1,
            "pid": process.pid,
            "argv": runtime.redact_argv(raw_argv),
            "command": "eval" if isinstance(options, EvalOptions) else "ab",
            "started_at": _iso_now(),
            "log": os.path.relpath(log_path, task_dir).replace(os.sep, "/"),
            "status": "running",
            "exit_code": None,
            "finished_at": None,
        },
    )
    print(f"[run_task] 后台编排已启动: pid={process.pid}，日志 {log_path}", file=sys.stderr)
    print(task_dir)
    return EXIT_OK


# ---------------------------------------------------------------------------
# 子进程执行
# ---------------------------------------------------------------------------


def _redacted_argv(root: Path, argv: Sequence[str]) -> list[str]:
    """命令展示用的 argv：密钥参数掩成 ``***``（日志会长期留存，不能把 key 打进去）。"""
    try:
        return _membench_module(root, "runtime").redact_argv(argv)
    except Exception:  # noqa: BLE001 - 展示逻辑不能因为导入失败中断
        return list(argv)


class Executor:
    """在 membench 工程根下执行 CLI 子进程；stdout 保持干净。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.env = os.environ.copy()
        # 相对路径的产物根按调用方 cwd 解释，避免子进程 cwd（工程根）改变语义
        out_env = self.env.get("MEMBENCH_OUTPUT_DIR")
        if out_env and not os.path.isabs(out_env):
            self.env["MEMBENCH_OUTPUT_DIR"] = _resolve_caller_path(out_env)

    @staticmethod
    def _stderr_target() -> Any:
        """子进程输出落点：终端直接继承；被捕获（pytest 等）时改用 PIPE。"""
        try:
            sys.stderr.fileno()
            return sys.stderr
        except (AttributeError, OSError, ValueError):
            return subprocess.PIPE

    def _cmd(self, argv: Sequence[str]) -> list[str]:
        """完整命令。"""
        return _cli_argv(self.root, argv)

    def display(self, argv: Sequence[str]) -> str:
        """可读命令（dry-run 用；密钥参数已脱敏）。"""
        return shlex.join(self._cmd(_redacted_argv(self.root, argv)))

    def init_task(self, argv: Sequence[str]) -> str:
        """``task init``：stdout 只取路径，stderr 继承。"""
        target = self._stderr_target()
        proc = subprocess.run(
            self._cmd(argv),
            cwd=str(self.root),
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=target,
            text=True,
        )
        if target is subprocess.PIPE and proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        if proc.returncode != 0:
            raise FlowFailed(f"task init 失败（退出码 {proc.returncode}）")
        lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
        if not lines:
            raise FlowFailed("task init 没有在 stdout 打印任务目录路径")
        task_dir = lines[-1]
        if not os.path.isabs(task_dir):
            task_dir = str((self.root / task_dir).resolve())
        if not os.path.isdir(task_dir):
            raise FlowFailed(f"task init 返回的目录不存在: {task_dir}")
        print(f"[run_task] 任务目录: {task_dir}", file=sys.stderr)
        return task_dir

    def run_stream(self, argv: Sequence[str]) -> CommandResult:
        """跑命令并实时输出（run 用）；返回退出码。"""
        print(f"[run_task] 执行: {self.display(argv)}", file=sys.stderr)
        target = self._stderr_target()
        proc = subprocess.run(
            self._cmd(argv),
            cwd=str(self.root),
            env=self.env,
            stdout=target,
            stderr=subprocess.STDOUT if target is subprocess.PIPE else target,
            text=True,
        )
        if target is subprocess.PIPE and proc.stdout:
            print(proc.stdout, end="", file=sys.stderr)
        return CommandResult(list(argv), proc.returncode, proc.stdout or "")

    def run_capture(self, argv: Sequence[str]) -> CommandResult:
        """跑命令并捕获输出（fuse / compare / evaluate 用，便于解析门禁结论）。"""
        print(f"[run_task] 执行: {self.display(argv)}", file=sys.stderr)
        proc = subprocess.run(
            self._cmd(argv),
            cwd=str(self.root),
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output = proc.stdout or ""
        if output:
            print(output, end="", file=sys.stderr)
        return CommandResult(list(argv), proc.returncode, output)

    def run_parallel(self, argvs: Sequence[Sequence[str]]) -> list[CommandResult]:
        """Popen 同时起 N 个进程，全部 wait；返回每个的退出码与输出。"""
        target = self._stderr_target()
        processes: list[tuple[list[str], subprocess.Popen[str]]] = []
        for index, argv in enumerate(argvs, 1):
            print(
                f"[run_task] 并行启动 {index}/{len(argvs)}: {self.display(argv)}",
                file=sys.stderr,
            )
            processes.append(
                (
                    list(argv),
                    subprocess.Popen(
                        self._cmd(argv),
                        cwd=str(self.root),
                        env=self.env,
                        stdout=target,
                        stderr=subprocess.STDOUT if target is subprocess.PIPE else target,
                        text=True,
                    ),
                )
            )
        results: list[CommandResult] = []
        for argv, proc in processes:
            if target is subprocess.PIPE:
                output, _ = proc.communicate()
                if output:
                    print(output, end="", file=sys.stderr)
                results.append(CommandResult(list(argv), proc.returncode, output or ""))
            else:
                proc.wait()
                results.append(CommandResult(list(argv), proc.returncode, ""))
        return results


# ---------------------------------------------------------------------------
# 命令拼装
# ---------------------------------------------------------------------------


def _task_init_argv(
    *,
    name: str,
    arms: int,
    platform: Optional[str],
    report_out: Optional[str],
) -> list[str]:
    """``task init`` 参数。"""
    argv = ["task", "init", "--arms", str(arms), "--name", name]
    if platform:
        argv += ["--platform", platform]
    if report_out:
        argv += ["--report-out", report_out]
    return argv


def _run_argv(
    *,
    arm: ArmArgs,
    dataset: DatasetArg,
    target_dir: str,
    common: CommonArgs,
    shards: Optional[int | str] = None,
    shard_index: Optional[int] = None,
) -> list[str]:
    """``run --task-dir`` 参数；``shards`` 非空时按对话级分片。"""
    argv = [
        "run",
        "--adapter",
        arm.adapter,
        "--dataset",
        dataset.cli_value,
        "--judge",
        common.judge,
        "--evidence-match",
        common.recall_mode,
        "--task-dir",
        target_dir,
    ]
    if common.platform:
        argv += ["--platform", common.platform]
    if common.commit:
        argv += ["--commit", common.commit]
    if arm.driver_llm:
        argv += ["--driver-llm", arm.driver_llm]
    if arm.adapter_version:
        argv += ["--adapter-version", arm.adapter_version]
    if common.jobs is not None:
        argv += ["--jobs", str(common.jobs)]
    if common.limit is not None:
        argv += ["--limit", str(common.limit)]
    if common.no_snapshot:
        argv += ["--no-snapshot"]
    if common.evidence_match_threshold is not None:
        argv += ["--evidence-match-threshold", str(common.evidence_match_threshold)]
    if common.llm_base_url:
        argv += ["--llm-base-url", common.llm_base_url]
    if common.llm_api_key:
        argv += ["--llm-api-key", common.llm_api_key]
    if common.llm_model:
        argv += ["--llm-model", common.llm_model]
    if common.llm_timeout is not None:
        argv += ["--llm-timeout", str(common.llm_timeout)]
    if common.client_llm:
        argv += ["--client-llm", common.client_llm]
    for item in arm.adapter_args:
        argv += ["--adapter-arg", item]
    if shards is not None:
        argv += [
            "--shards",
            str(shards),
            "--shard-index",
            str(shard_index or 0),
            "--shard-by",
            "dialogue",
        ]
    return argv


def _evaluate_argv(
    *,
    options: EvalOptions,
    task_dir: str,
) -> list[str]:
    """官方多数据集路径：``evaluate --task-dir --dataset ...``（无分片 / limit）。

    只在 :attr:`EvalOptions.use_evaluate` 为真时使用；``evaluate`` 不支持分片与
    ``--limit``，其余情况由 ``_execute_eval`` 逐数据集 run 后用 ``--from-runs`` 汇总。
    """
    common = options.common
    argv = ["evaluate", "--adapter", options.arm.adapter]
    for dataset in options.datasets:
        argv += ["--dataset", dataset.cli_value]
    argv += [
        "--judge",
        common.judge,
        "--evidence-match",
        common.recall_mode,
        "--task-dir",
        task_dir,
    ]
    if common.platform:
        argv += ["--platform", common.platform]
    if common.commit:
        argv += ["--commit", common.commit]
    if options.arm.driver_llm:
        argv += ["--driver-llm", options.arm.driver_llm]
    if options.arm.adapter_version:
        argv += ["--adapter-version", options.arm.adapter_version]
    if common.jobs is not None:
        argv += ["--jobs", str(common.jobs)]
    if common.no_snapshot:
        argv += ["--no-snapshot"]
    if common.evidence_match_threshold is not None:
        argv += ["--evidence-match-threshold", str(common.evidence_match_threshold)]
    if common.llm_base_url:
        argv += ["--llm-base-url", common.llm_base_url]
    if common.llm_api_key:
        argv += ["--llm-api-key", common.llm_api_key]
    if common.llm_model:
        argv += ["--llm-model", common.llm_model]
    if common.llm_timeout is not None:
        argv += ["--llm-timeout", str(common.llm_timeout)]
    if common.client_llm:
        argv += ["--client-llm", common.client_llm]
    for item in options.arm.adapter_args:
        argv += ["--adapter-arg", item]
    return argv


def _fuse_argv(dataset_dir: str) -> list[str]:
    """``fuse --task-dir`` 就地补齐一个数据集（也接受 ``runs/<dataset>``）。"""
    return ["fuse", "--task-dir", dataset_dir]


def _evaluate_from_runs_argv(
    task_dir: str, datasets: Sequence[ResolvedDataset]
) -> list[str]:
    """``evaluate --task-dir --from-runs`` 汇总多数据集任务级报告。"""
    argv = ["evaluate", "--task-dir", task_dir]
    for item in datasets:
        argv += ["--from-runs", f"{item.dataset.key}={item.directory}"]
    return argv


def _compare_argv(
    task_dir: str,
    *,
    baseline: Optional[str] = None,
    candidate: Optional[str] = None,
) -> list[str]:
    """``compare --task-dir --require-comparable``；多数据集时显式给两臂目录。"""
    argv = ["compare", "--task-dir", task_dir, "--require-comparable"]
    if baseline and candidate:
        argv += ["--baseline", baseline, "--candidate", candidate]
    return argv


def _evaluate_compare_argv(
    task_dir: str, arm_datasets: dict[str, dict[str, ResolvedDataset]], datasets: Sequence[DatasetArg]
) -> list[str]:
    """``evaluate-compare --task-dir`` 复用两臂已有产物，写跨数据集对比报告。"""
    argv = ["evaluate-compare", "--task-dir", task_dir]
    for dataset in datasets:
        baseline = arm_datasets["A"][dataset.key]
        candidate = arm_datasets["B"][dataset.key]
        argv += ["--baseline-run", f"{dataset.key}={baseline.directory}"]
        argv += ["--candidate-run", f"{dataset.key}={candidate.directory}"]
    return argv


# ---------------------------------------------------------------------------
# 真实流程
# ---------------------------------------------------------------------------


def _child_dirs(path: str) -> set[str]:
    """目录下的子目录名集合（目录不存在时为空）。"""
    try:
        return {name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))}
    except OSError:
        return set()


def _complete_report_exists(dataset_dir: str) -> bool:
    """数据集级 ``report/metrics.json`` 存在且非空。"""
    path = os.path.join(dataset_dir, "report", "metrics.json")
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _has_ledger(dataset_dir: str) -> bool:
    """数据集目录里是否有分片账本。"""
    if os.path.isfile(os.path.join(dataset_dir, "shards.json")):
        return True
    try:
        return any(
            name.startswith("shards.") and name.endswith(".json")
            for name in os.listdir(dataset_dir)
        )
    except OSError:
        return False


def _discover_dataset_dir(
    exp_root: str, before: set[str], *, task_dir: str
) -> str:
    """本次 run 新产生的数据集目录（只允许一个）。"""
    runs = os.path.join(exp_root, "runs")
    after = _child_dirs(runs)
    new = sorted(after - before)
    if len(new) == 1:
        return os.path.join(runs, new[0])
    if not new:
        raise FlowFailed(
            "本次 run 没有产生新的数据集目录：可能是空数据集、分片索引越界，"
            "或数据集名与已有产物重名（membench 拒绝覆盖）",
            task_dir,
        )
    raise FlowFailed(
        f"本次 run 产生了多个新的数据集目录（{', '.join(new)}），无法确定融合哪一个",
        task_dir,
    )


def _read_ledger_shards(dataset_dir: str, *, task_dir: str) -> int:
    """从分片账本读有效分片数（``--shards max`` 探测用）。"""
    paths = [os.path.join(dataset_dir, "shards.json")]
    try:
        paths += sorted(
            os.path.join(dataset_dir, name)
            for name in os.listdir(dataset_dir)
            if name.startswith("shards.") and name.endswith(".json")
        )
    except OSError:
        pass
    for path in paths:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        shards = data.get("shards") if isinstance(data, dict) else None
        if isinstance(shards, int) and not isinstance(shards, bool) and shards >= 1:
            return shards
    raise FlowFailed(
        f"分片账本里读不到有效分片数（shards）: {dataset_dir}；"
        "请检查 shards.json / shards.<i>.json",
        task_dir,
    )


def _expect_ok(result: CommandResult, what: str, task_dir: str) -> None:
    """普通命令必须退出 0。"""
    if result.returncode != 0:
        raise FlowFailed(f"{what} 失败（退出码 {result.returncode}）", task_dir)


def _parse_verdict(output: str) -> Optional[str]:
    """从命令输出里解析门禁结论（最后一条匹配）。"""
    verdict: Optional[str] = None
    for pattern in _VERDICT_PATTERNS:
        for match in pattern.finditer(output):
            verdict = match.group(1)
    return verdict


def _handle_gate(result: CommandResult, what: str, state: FlowState) -> None:
    """门禁命令：退出 1 + FAIL 如实记为门禁 FAIL；其余非零按流程失败。"""
    if result.returncode == 0:
        return
    verdict = _parse_verdict(result.output)
    if result.returncode == 1 and verdict == "FAIL":
        state.gate_fail = True
        print(
            f"[run_task] {what} 门禁 FAIL（退出码 1 如实向上传递）",
            file=sys.stderr,
        )
        return
    raise FlowFailed(
        f"{what} 失败（退出码 {result.returncode}，未解析到门禁 FAIL 结论）",
        state.task_dir,
    )


def _fuse_dataset(
    executor: Executor,
    dataset_dir: str,
    state: FlowState,
) -> None:
    """就地融合一个数据集，并确认数据集级报告已生成。"""
    result = executor.run_capture(_fuse_argv(dataset_dir))
    _handle_gate(result, "fuse", state)
    if not _complete_report_exists(dataset_dir):
        raise FlowFailed(
            f"fuse 后仍缺数据集级报告: {dataset_dir}/report/metrics.json",
            state.task_dir,
        )


def _run_shards_parallel(
    executor: Executor,
    *,
    exp_root: str,
    dataset: DatasetArg,
    arm: ArmArgs,
    common: CommonArgs,
    shards: int,
    task_dir: str,
) -> None:
    """并行跑 0..N-1 片；任一失败即整体失败（不提前杀其它片）。"""
    argvs = [
        _run_argv(
            arm=arm,
            dataset=dataset,
            target_dir=exp_root,
            common=common,
            shards=shards,
            shard_index=index,
        )
        for index in range(shards)
    ]
    results = executor.run_parallel(argvs)
    failures = [item for item in results if item.returncode != 0]
    if failures:
        details = "；".join(
            f"shard {index} -> 退出码 {item.returncode}"
            for index, item in enumerate(results)
            if item.returncode != 0
        )
        raise FlowFailed(f"分片运行失败（{details}）", task_dir)


def _run_dataset(
    executor: Executor,
    *,
    exp_root: str,
    dataset: DatasetArg,
    arm: ArmArgs,
    common: CommonArgs,
    shards: int | str,
    state: FlowState,
) -> str:
    """跑一个数据集（按需分片 + fuse），返回数据集目录。"""
    before = _child_dirs(os.path.join(exp_root, "runs"))

    if shards == "max":
        # max 需要先跑 shard 0，从账本读有效分片数 N，再补 1..N-1
        result = executor.run_stream(
            _run_argv(
                arm=arm,
                dataset=dataset,
                target_dir=exp_root,
                common=common,
                shards="max",
                shard_index=0,
            )
        )
        _expect_ok(result, f"数据集 {dataset.display} 的 shard 0（--shards max）", state.task_dir)
        dataset_dir = _discover_dataset_dir(exp_root, before, task_dir=state.task_dir)
        if _complete_report_exists(dataset_dir) and not _has_ledger(dataset_dir):
            return dataset_dir  # 有效分片数 = 1，run 直接写成完整产物
        n_shards = _read_ledger_shards(dataset_dir, task_dir=state.task_dir)
        if n_shards > 1:
            print(
                f"[run_task] --shards max -> {n_shards}（数据集 {dataset.display}），"
                f"补齐 shard 1..{n_shards - 1}",
                file=sys.stderr,
            )
            rest = [
                _run_argv(
                    arm=arm,
                    dataset=dataset,
                    target_dir=exp_root,
                    common=common,
                    shards=n_shards,
                    shard_index=index,
                )
                for index in range(1, n_shards)
            ]
            results = executor.run_parallel(rest)
            failures = [item for item in results if item.returncode != 0]
            if failures:
                raise FlowFailed(
                    "分片运行失败（"
                    + "；".join(
                        f"shard {index + 1} -> 退出码 {item.returncode}"
                        for index, item in enumerate(results)
                        if item.returncode != 0
                    )
                    + "）",
                    state.task_dir,
                )
        _fuse_dataset(executor, dataset_dir, state)
        return dataset_dir

    shards_int = int(shards)
    if shards_int > 1:
        _run_shards_parallel(
            executor,
            exp_root=exp_root,
            dataset=dataset,
            arm=arm,
            common=common,
            shards=shards_int,
            task_dir=state.task_dir,
        )
        dataset_dir = _discover_dataset_dir(exp_root, before, task_dir=state.task_dir)
        if _complete_report_exists(dataset_dir) and not _has_ledger(dataset_dir):
            # --shards N 超过对话数，有效分片数为 1：shard 0 已写成完整产物
            return dataset_dir
        _fuse_dataset(executor, dataset_dir, state)
        return dataset_dir

    # 不分片：run 自带完整数据集级 / 任务级产物，无需 fuse
    result = executor.run_stream(
        _run_argv(
            arm=arm,
            dataset=dataset,
            target_dir=exp_root,
            common=common,
        )
    )
    _expect_ok(result, f"数据集 {dataset.display} 的 run", state.task_dir)
    return _discover_dataset_dir(exp_root, before, task_dir=state.task_dir)


def _execute_eval(options: EvalOptions, root: Path) -> FlowState:
    """执行 ``eval`` 全流程（``options.task_dir`` 给定时跳过 ``task init``）。"""
    executor = Executor(root)
    if options.task_dir is not None:
        task_dir = options.task_dir
        print(f"[run_task] 复用任务目录: {task_dir}", file=sys.stderr)
    else:
        task_dir = executor.init_task(
            _task_init_argv(
                name=options.name,
                arms=1,
                platform=options.common.platform,
                report_out=options.report_out,
            )
        )
    state = FlowState(task_dir=task_dir)
    if options.use_evaluate:
        # 官方多数据集路径：一次 evaluate 跑完全部数据集并写任务级报告
        result = executor.run_capture(_evaluate_argv(options=options, task_dir=task_dir))
        _handle_gate(result, "evaluate", state)
        return state
    for dataset in options.datasets:
        directory = _run_dataset(
            executor,
            exp_root=task_dir,
            dataset=dataset,
            arm=options.arm,
            common=options.common,
            shards=options.shards,
            state=state,
        )
        state.datasets.append(ResolvedDataset(dataset, directory))
    if len(options.datasets) > 1:
        result = executor.run_capture(_evaluate_from_runs_argv(task_dir, state.datasets))
        _handle_gate(result, "evaluate", state)
    return state


def _execute_ab(options: AbOptions, root: Path) -> FlowState:
    """执行 ``ab`` 全流程（``options.task_dir`` 给定时跳过 ``task init``）。"""
    executor = Executor(root)
    if options.task_dir is not None:
        task_dir = options.task_dir
        print(f"[run_task] 复用任务目录: {task_dir}", file=sys.stderr)
    else:
        task_dir = executor.init_task(
            _task_init_argv(
                name=options.name,
                arms=2,
                platform=options.common.platform,
                report_out=options.report_out,
            )
        )
    state = FlowState(task_dir=task_dir)
    for label, arm in (("A", options.baseline), ("B", options.candidate)):
        arm_dir = os.path.join(task_dir, label)
        for dataset in options.datasets:
            print(
                f"[run_task] 臂 {label}（{arm.adapter}）数据集 {dataset.display}",
                file=sys.stderr,
            )
            directory = _run_dataset(
                executor,
                exp_root=arm_dir,
                dataset=dataset,
                arm=arm,
                common=options.common,
                shards=options.shards,
                state=state,
            )
            state.arm_datasets.setdefault(label, {})[dataset.key] = ResolvedDataset(
                dataset, directory
            )

    if len(options.datasets) == 1:
        result = executor.run_capture(_compare_argv(task_dir))
        _handle_gate(result, "compare", state)
        return state

    # 多数据集：逐数据集 strict 门禁 + evaluate-compare 融合报告
    for dataset in options.datasets:
        baseline = state.arm_datasets["A"][dataset.key]
        candidate = state.arm_datasets["B"][dataset.key]
        print(
            f"[run_task] 数据集 {dataset.display} 的可比性 strict 门禁",
            file=sys.stderr,
        )
        result = executor.run_capture(
            _compare_argv(
                task_dir,
                baseline=baseline.directory,
                candidate=candidate.directory,
            )
        )
        _handle_gate(result, "compare", state)
    result = executor.run_capture(
        _evaluate_compare_argv(task_dir, state.arm_datasets, options.datasets)
    )
    _handle_gate(result, "evaluate-compare", state)
    return state


# ---------------------------------------------------------------------------
# dry-run：只打印命令序列
# ---------------------------------------------------------------------------


def _display_argv(root: Path, argv: Sequence[str]) -> str:
    """干燥运行的命令展示；``$TASK`` 占位符保持可 shell 展开，密钥参数已脱敏。"""
    text = shlex.join(_cli_argv(root, _redacted_argv(root, argv)))
    # shlex 会把 $TASK / $TASK/A 单引号包起来；占位符要留给 shell 展开
    return re.sub(r"'(\$TASK[^']*)'", r"\1", text)


def _predicted_dataset_dir(exp_ref: str, dataset: DatasetArg) -> str:
    """dry-run 下预测数据集目录（真实目录以 run 输出为准）。"""
    return f"{exp_ref}/runs/{_slug(dataset.key)}"


def _print_run_steps(
    root: Path,
    *,
    exp_ref: str,
    dataset: DatasetArg,
    arm: ArmArgs,
    common: CommonArgs,
    shards: int | str,
) -> None:
    """打印一个数据集的 run / 并行 / fuse 命令。"""
    print(f"# 数据集: {dataset.display}")
    if dataset.path_based:
        pass
    elif dataset.key == "office":
        print(
            "# 注意: office 槽位的实际目录名可能是 office_dataset（用户测试集则为其文件名），"
            "fuse/compare 以真实目录为准"
        )
    if shards == "max":
        print(
            _display_argv(
                root,
                _run_argv(
                    arm=arm,
                    dataset=dataset,
                    target_dir=exp_ref,
                    common=common,
                    shards="max",
                    shard_index=0,
                ),
            )
        )
        print(
            "# --shards max：若 shard 0 的账本显示有效分片数 N>1，"
            "再并行执行 shard 1..N-1（替换 --shard-index）"
        )
        print("# 若产生分片账本，执行:")
        print(_display_argv(root, _fuse_argv(_predicted_dataset_dir(exp_ref, dataset))))
        return
    shards_int = int(shards)
    if shards_int > 1:
        print(f"# 并行执行 {shards_int} 片（Popen 同时启动，全部 wait；任一失败即整体失败）")
        for index in range(shards_int):
            command = _display_argv(
                root,
                _run_argv(
                    arm=arm,
                    dataset=dataset,
                    target_dir=exp_ref,
                    common=common,
                    shards=shards_int,
                    shard_index=index,
                ),
            )
            print(f"{command} &")
        print("wait")
        print(_display_argv(root, _fuse_argv(_predicted_dataset_dir(exp_ref, dataset))))
        return
    print(
        _display_argv(
            root,
            _run_argv(
                arm=arm,
                dataset=dataset,
                target_dir=exp_ref,
                common=common,
            ),
        )
    )


def _print_task_ref(
    root: Path,
    *,
    task_dir: Optional[str],
    name: str,
    arms: int,
    platform: Optional[str],
    report_out: Optional[str],
) -> None:
    """dry-run 的任务目录来源：给了 --task-dir 就直接用，否则 ``TASK=$(task init ...)``。"""
    if task_dir:
        print(f"TASK={shlex.quote(task_dir)}")
        return
    print(
        "TASK=$("
        + _display_argv(
            root,
            _task_init_argv(
                name=name,
                arms=arms,
                platform=platform,
                report_out=report_out,
            ),
        )
        + ")"
    )


def _print_dry_run_eval(options: EvalOptions, root: Path) -> None:
    """``eval --dry-run``。"""
    print("# run_task.py --dry-run：以下命令按顺序执行；$TASK 为任务目录")
    print("# 注: office 槽位与自定义 key 的数据集目录名是预测值，真实运行以 run 落盘目录为准")
    print("# 真实运行时 stdout 只打印任务目录绝对路径；进度 / 自检摘要走 stderr")
    _print_task_ref(
        root,
        task_dir=options.task_dir,
        name=options.name,
        arms=1,
        platform=options.common.platform,
        report_out=options.report_out,
    )
    if options.use_evaluate:
        print("# 多数据集且无分片 / limit：直接走官方 evaluate --dataset 路径")
        print(_display_argv(root, _evaluate_argv(options=options, task_dir="$TASK")))
        return
    for dataset in options.datasets:
        _print_run_steps(
            root,
            exp_ref="$TASK",
            dataset=dataset,
            arm=options.arm,
            common=options.common,
            shards=options.shards,
        )
    if len(options.datasets) > 1:
        print("# 多数据集（分片 / limit）：逐数据集 run 后汇总任务级报告（--from-runs，不重跑）")
        argv = ["evaluate", "--task-dir", "$TASK"]
        for dataset in options.datasets:
            argv += [
                "--from-runs",
                f"{dataset.key}={_predicted_dataset_dir('$TASK', dataset)}",
            ]
        print(_display_argv(root, argv))


def _print_dry_run_ab(options: AbOptions, root: Path) -> None:
    """``ab --dry-run``。"""
    print("# run_task.py --dry-run：以下命令按顺序执行；$TASK 为任务目录")
    print("# 注: office 槽位与自定义 key 的数据集目录名是预测值，真实运行以 run 落盘目录为准")
    print("# 真实运行时 stdout 只打印任务目录绝对路径；进度 / 自检摘要走 stderr")
    _print_task_ref(
        root,
        task_dir=options.task_dir,
        name=options.name,
        arms=2,
        platform=options.common.platform,
        report_out=options.report_out,
    )
    for label, arm in (("A", options.baseline), ("B", options.candidate)):
        print(f"# 臂 {label}（{arm.adapter}）")
        for dataset in options.datasets:
            _print_run_steps(
                root,
                exp_ref=f"$TASK/{label}",
                dataset=dataset,
                arm=arm,
                common=options.common,
                shards=options.shards,
            )
    if len(options.datasets) == 1:
        print("# 单数据集：就地 strict 门禁")
        print(_display_argv(root, _compare_argv("$TASK")))
        return
    print("# 多数据集：逐数据集 strict 门禁，再写跨数据集融合对比报告")
    for dataset in options.datasets:
        print(f"# 数据集 {dataset.display} 的 strict 门禁")
        print(
            _display_argv(
                root,
                _compare_argv(
                    "$TASK",
                    baseline=_predicted_dataset_dir("$TASK/A", dataset),
                    candidate=_predicted_dataset_dir("$TASK/B", dataset),
                ),
            )
        )
    argv = ["evaluate-compare", "--task-dir", "$TASK"]
    for dataset in options.datasets:
        argv += [
            "--baseline-run",
            f"{dataset.key}={_predicted_dataset_dir('$TASK/A', dataset)}",
        ]
        argv += [
            "--candidate-run",
            f"{dataset.key}={_predicted_dataset_dir('$TASK/B', dataset)}",
        ]
    print(_display_argv(root, argv))


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """两种模式共用的参数。"""
    parser.add_argument(
        "--judge",
        required=True,
        choices=("heuristic", "llm", "client"),
        help="判分器；heuristic 只允许与 --limit 同时出现（小样本冒烟），正式评测用 llm/client",
    )
    parser.add_argument(
        "--recall-mode",
        required=True,
        choices=("substring", "overlap", "llm"),
        help=(
            "检索证据命中口径（映射为 membench --evidence-match）：substring=逐字子串，"
            "overlap=token 包含度，llm=逐题语义判定。摘要 / 改写式记忆请用 llm，"
            "否则 evidence_recall 会系统性偏低"
        ),
    )
    parser.add_argument("--platform", default=None, help="平台标签（如 wx / fs），进目录名与元数据")
    parser.add_argument("--commit", default=None, help="commit 标签，记入元数据")
    parser.add_argument("--jobs", type=int, default=None, help="题目级并发数（默认 1）")
    parser.add_argument("--limit", type=int, default=None, help="冒烟：每个数据集只评测前 N 题")
    parser.add_argument(
        "--shards",
        type=_parse_shards,
        default=1,
        help=(
            "分片总数（默认 1 不分片）；>=2 时按对话级并行分片，跑完自动 fuse；"
            "max 会先跑 shard 0，从账本解析有效分片数后再并行补齐"
        ),
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="不导出对话状态快照（默认导出；适配器不支持时警告一次并跳过）",
    )
    parser.add_argument(
        "--evidence-match-threshold",
        type=float,
        default=None,
        help="evidence-match=overlap 时的包含度阈值（缺省用 membench 默认 0.6）",
    )
    parser.add_argument("--llm-base-url", default=None, help="judge=llm / recall-mode=llm 的 OpenAI 兼容 base_url")
    parser.add_argument("--llm-api-key", default=None, help="judge=llm / recall-mode=llm 的 API key")
    parser.add_argument("--llm-model", default=None, help="judge=llm / recall-mode=llm 的模型名")
    parser.add_argument("--llm-timeout", type=float, default=None, help="LLM 请求超时秒数")
    parser.add_argument("--client-llm", default=None, help="judge=client 的内置 LLM 类路径，如 my_agent.llm:LLMClient")
    parser.add_argument("--name", default=None, help="任务名（进目录名）；缺省用 adapter 名 / <基线>_vs_<候选>")
    parser.add_argument(
        "--report-out",
        default=None,
        help="任务目录父目录（缺省 membench output_root()）；相对路径按调用方 cwd 解析",
    )
    parser.add_argument(
        "--task-dir",
        default=None,
        help="复用已存在的任务目录（跳过 task init；eval 需 runs/，ab 需 A/ B/）；"
        "--background 的父子进程共用它，也给续跑 / 接管留口子",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="派生到后台执行：stdout 只打印任务目录一行并立即退出 0，"
        "进度用 scripts/watch_task.py <任务目录> --until-done 定时查看；与 --dry-run 互斥",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的命令序列，不建目录、不跑评测",
    )


def _add_dataset_argument(parser: argparse.ArgumentParser) -> None:
    """``--dataset``（可重复）。"""
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="KEY|PATH",
        help="数据集 key 或 JSONL 路径（可重复；相对路径按调用方 cwd 解析）",
    )


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="run_task.py",
        description=(
            "membench 标准流程编排器：task init → run（分片并行）→ fuse → "
            "A/B compare → verify_artifacts 自检。"
        ),
        epilog=(
            "退出码: 0=成功；1=门禁 FAIL；2=流程失败；3=自检失败。"
            " stdout 只打印任务目录绝对路径（最后一行），其余输出走 stderr。"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser(
        "eval",
        help="单臂评测：建任务目录 → 逐数据集 run（可并行分片）→ fuse → 自检",
        description=(
            "单臂评测。单数据集：run --task-dir（分片则并行 + fuse）；"
            "多数据集且无分片 / limit：evaluate --task-dir --dataset；"
            "多数据集且分片或 limit：逐数据集 run → evaluate --from-runs 汇总。"
        ),
    )
    _add_common_arguments(p_eval)
    p_eval.add_argument("--adapter", required=True, help="适配器名（见 membench adapter list）")
    p_eval.add_argument("--driver-llm", default=None, help="驱动被测系统的 LLM 标识")
    p_eval.add_argument("--adapter-version", default=None, help="适配器实现版本 / commit")
    p_eval.add_argument(
        "--adapter-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="适配器构造参数（可重复），如 --adapter-arg base_url=http://127.0.0.1:8787",
    )
    _add_dataset_argument(p_eval)

    p_ab = sub.add_parser(
        "ab",
        help="A/B 评测：建双臂任务目录 → 两臂逐数据集 run → fuse → compare strict 门禁 → 自检",
        description=(
            "A/B 评测。单数据集：compare --task-dir --require-comparable；"
            "多数据集：逐数据集 strict compare + evaluate-compare 融合报告。"
        ),
    )
    _add_common_arguments(p_ab)
    p_ab.add_argument("--baseline-adapter", required=True, help="基线适配器名")
    p_ab.add_argument("--candidate-adapter", required=True, help="候选适配器名")
    p_ab.add_argument(
        "--baseline-version",
        required=True,
        help="基线侧 adapter 版本 / commit（strict 可比性关键维度；同系统只改读取侧时两侧填同一值）",
    )
    p_ab.add_argument(
        "--candidate-version",
        required=True,
        help="候选侧 adapter 版本 / commit（同上）",
    )
    p_ab.add_argument(
        "--driver-llm-a",
        required=True,
        help="基线侧驱动记忆的 LLM 标识（strict 关键维度；没有驱动 LLM 时填 none）",
    )
    p_ab.add_argument(
        "--driver-llm-b",
        required=True,
        help="候选侧驱动记忆的 LLM 标识（同上）",
    )
    p_ab.add_argument(
        "--adapter-arg-a",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="基线侧适配器构造参数（可重复）",
    )
    p_ab.add_argument(
        "--adapter-arg-b",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="候选侧适配器构造参数（可重复）",
    )
    _add_dataset_argument(p_ab)

    return parser


def _common_from_args(args: argparse.Namespace) -> CommonArgs:
    """从 argparse 结果提取公共参数。"""
    return CommonArgs(
        judge=args.judge,
        recall_mode=args.recall_mode,
        platform=args.platform,
        commit=args.commit,
        jobs=args.jobs,
        limit=args.limit,
        no_snapshot=args.no_snapshot,
        evidence_match_threshold=args.evidence_match_threshold,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
        llm_timeout=args.llm_timeout,
        client_llm=args.client_llm,
    )


def _validate_common(common: CommonArgs) -> None:
    """参数组合校验（开跑前暴露）。"""
    if common.limit is not None and common.limit < 1:
        raise UsageError(f"--limit 必须 >= 1，当前为 {common.limit}")
    if common.jobs is not None and common.jobs < 1:
        raise UsageError(f"--jobs 必须 >= 1，当前为 {common.jobs}")
    if common.judge == "heuristic" and common.limit is None:
        raise UsageError(
            "--judge heuristic 只允许与 --limit 同时出现：正式评测请用 --judge llm/client，"
            "heuristic 仅限小样本冒烟"
        )
    if common.judge == "client" and not common.client_llm:
        raise UsageError("--judge client 需要 --client-llm 'module:ClassName'（agent 内置 LLM 类路径）")


def _eval_options(args: argparse.Namespace) -> EvalOptions:
    """校验并组装 eval 参数。"""
    common = _common_from_args(args)
    _validate_common(common)
    datasets = _collect_datasets(args.dataset)
    report_out = _resolve_caller_path(args.report_out) if args.report_out else None
    arm = ArmArgs(
        adapter=args.adapter,
        driver_llm=args.driver_llm,
        adapter_version=args.adapter_version,
        adapter_args=list(args.adapter_arg),
    )
    name = args.name or args.adapter
    return EvalOptions(
        common=common,
        arm=arm,
        datasets=datasets,
        name=name,
        report_out=report_out,
        shards=args.shards,
        task_dir=_resolve_caller_path(args.task_dir) if args.task_dir else None,
    )


def _ab_options(args: argparse.Namespace) -> AbOptions:
    """校验并组装 ab 参数。"""
    common = _common_from_args(args)
    _validate_common(common)
    datasets = _collect_datasets(args.dataset)
    report_out = _resolve_caller_path(args.report_out) if args.report_out else None
    baseline = ArmArgs(
        adapter=args.baseline_adapter,
        driver_llm=args.driver_llm_a,
        adapter_version=args.baseline_version,
        adapter_args=list(args.adapter_arg_a),
    )
    candidate = ArmArgs(
        adapter=args.candidate_adapter,
        driver_llm=args.driver_llm_b,
        adapter_version=args.candidate_version,
        adapter_args=list(args.adapter_arg_b),
    )
    name = args.name or f"{args.baseline_adapter}_vs_{args.candidate_adapter}"
    return AbOptions(
        common=common,
        baseline=baseline,
        candidate=candidate,
        datasets=datasets,
        name=name,
        report_out=report_out,
        shards=args.shards,
        task_dir=_resolve_caller_path(args.task_dir) if args.task_dir else None,
    )


# ---------------------------------------------------------------------------
# 自检与主入口
# ---------------------------------------------------------------------------


def _print_verification(result: Any) -> None:
    """把自检结果打印到 stderr（stdout 留给任务路径）。"""
    for warning in result.warnings:
        print(
            f"[run_task] 自检提示 [{warning.check}/{warning.code}] {warning.message}",
            file=sys.stderr,
        )
    if result.ok:
        print(f"[run_task] 自检: OK {result.task}", file=sys.stderr)
        return
    for error in result.errors:
        location = f"（{error.path}）" if error.path else ""
        print(
            f"[run_task] 自检 FAIL [{error.check}/{error.code}] {error.message}{location}",
            file=sys.stderr,
        )
    print(
        f"[run_task] 自检失败: {result.task}（{len(result.errors)} 项问题）",
        file=sys.stderr,
    )


def _run_flow(options: EvalOptions | AbOptions, root: Path) -> int:
    """跑完整流程并自检，返回退出码（stdout 仍只打印任务目录）。"""
    if options.common.judge == "heuristic":
        print(f"[run_task] 提示: {_heuristic_smoke_note(options.common)}", file=sys.stderr)

    state: Optional[FlowState] = None
    try:
        if isinstance(options, EvalOptions):
            state = _execute_eval(options, root)
        else:
            state = _execute_ab(options, root)
    except FlowFailed as exc:
        print(f"[run_task] 流程失败: {exc}", file=sys.stderr)
        if exc.task_dir:
            print(exc.task_dir)
        return EXIT_FLOW_FAIL
    except KeyboardInterrupt:
        print("[run_task] 已中断", file=sys.stderr)
        raise

    if state is None:
        print("[run_task] 内部错误: 流程没有返回任务目录", file=sys.stderr)
        return EXIT_FLOW_FAIL
    try:
        verification = verify_task(
            state.task_dir,
            allow_limit=options.common.limit is not None,
            allow_partial=False,
        )
    except Exception as exc:  # 自检自身异常也要有明确退出码与现场
        print(f"[run_task] 自检执行失败: {exc}", file=sys.stderr)
        print(state.task_dir)
        return EXIT_VERIFY_FAIL

    _print_verification(verification)
    if state.gate_fail:
        print("[run_task] A/B / 评测门禁结论: FAIL（退出码 1）", file=sys.stderr)
    print(state.task_dir)
    if not verification.ok:
        return EXIT_VERIFY_FAIL
    if state.gate_fail:
        return EXIT_GATE_FAIL
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口，返回进程退出码。"""
    parser = build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)

    try:
        if args.command == "eval":
            options = _eval_options(args)
        else:
            options = _ab_options(args)
    except UsageError as exc:
        print(f"[run_task] 参数错误: {exc}", file=sys.stderr)
        return EXIT_FLOW_FAIL

    try:
        root = find_membench_root()
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_FLOW_FAIL

    if args.dry_run and args.background:
        print("[run_task] 参数错误: --dry-run 与 --background 互斥", file=sys.stderr)
        return EXIT_FLOW_FAIL

    # 后台派生的子进程也会带 --task-dir；结束时必须把最终状态写回 orchestrator.json
    orchestrator_child = os.environ.get(ORCHESTRATOR_ENV) == "1"
    exit_code = EXIT_FLOW_FAIL
    try:
        if options.task_dir is not None:
            _validate_task_dir(options.task_dir, 1 if isinstance(options, EvalOptions) else 2, root)
        if args.dry_run:
            if isinstance(options, EvalOptions):
                _print_dry_run_eval(options, root)
            else:
                _print_dry_run_ab(options, root)
            exit_code = EXIT_OK
        elif args.background:
            exit_code = _start_background(options, root, raw_argv)
        else:
            exit_code = _run_flow(options, root)
    except UsageError as exc:
        print(f"[run_task] 参数错误: {exc}", file=sys.stderr)
        exit_code = EXIT_FLOW_FAIL
    except FlowFailed as exc:
        print(f"[run_task] 流程失败: {exc}", file=sys.stderr)
        if exc.task_dir:
            print(exc.task_dir)
        exit_code = EXIT_FLOW_FAIL
    finally:
        if orchestrator_child and options.task_dir is not None:
            _finish_orchestrator(options.task_dir, exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
