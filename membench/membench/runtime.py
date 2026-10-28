"""实验 / 任务级运行时数据：命令、日志与环境快照。

产物目录里除了结果（``report/`` / ``metrics.json`` / 快照），还要能回答"这次评测
是怎么跑出来的"：谁在什么环境下执行的、命令参数是什么、控制台输出了什么、退出码
多少。本模块把一次 CLI 调用的这三样东西落到::

    <实验目录>/runtime/
    ├── commands.jsonl      # 一行一次调用（argv / 起止时间 / 耗时 / 退出码 / 日志 / 环境）
    ├── env.json            # 一次性环境快照（python / 平台 / membench 版本与 commit / MEMBENCH_*）
    └── logs/
        └── <YYYYmmdd-HHMMSS>-<command>-<pid>.log   # 本次命令的 stdout+stderr 全量

**与对话级 runtime 的区别**：``<实验目录>/runs/<dataset>/<对话目录>/runtime/`` 是
"该对话的运行时数据"（归适配器使用的运行时文件）；本模块写的是**实验 /
臂目录级**的 runtime，与 ``runs/`` / ``reports/`` 平级，记录的是评测命令本身。

用法（只给 CLI 与编排脚本用）::

    session = runtime.begin("run", argv)     # 开始捕获 stdout+stderr（终端照常显示）
    runtime.set_dir(experiment_dir)          # 命令内部确定实验目录后调用（首次生效）
    session.finish(exit_code)                # 命令结束落盘（CLI 在 finally 里调用）

直接以库方式调用 ``Runner`` / ``evaluate`` 时没有活动的 CLI 会话，:func:`set_dir`
是空操作——库用户不会平白多出一个 ``runtime/`` 目录。

并发：多个分片进程共用同一个实验目录时，日志文件名含 pid、``commands.jsonl`` 用
``O_APPEND`` 单行追加、``env.json`` 以 ``O_EXCL`` 只建一次，全程不做读-改-写，因此
不需要锁。
"""

from __future__ import annotations

import json
import os
import platform as platform_module
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Sequence
from datetime import datetime
from typing import Any, IO, Union

from membench import layout

#: 运行时目录与文件名（落 ``<实验目录>/runtime/`` 下）。
FILE_COMMANDS = "commands.jsonl"
FILE_ENV = "env.json"
DIR_LOGS = "logs"

#: ``commands.jsonl`` 单条记录的结构版本。
COMMANDS_SCHEMA = 1

#: ``env.json`` 的结构版本。
ENV_SCHEMA = 1

#: 参数名里出现这些词即视为密钥（大小写不敏感；``_`` / ``-`` 归一后比较）。
_SECRET_NAME_MARKERS: tuple[str, ...] = (
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "apikey",
    "api-key",
)

#: 参数**值**里出现 ``token=`` / ``secret:`` 这类形式时整段替换（如 URL 查询串）。
_SECRET_VALUE_RE = re.compile(
    r"(?i)(token|secret|password|passwd|api[_-]?key)\s*[=:]"
)

#: 脱敏后的统一占位符。
_MASK = "***"


# ---------------------------------------------------------------------------
# 路径与读取侧（task status / watch_task 共用）
# ---------------------------------------------------------------------------


def runtime_dir(experiment_dir: str) -> str:
    """实验 / 臂目录的运行时目录：``<实验目录>/runtime``。"""
    return os.path.join(experiment_dir, layout.DIR_RUNTIME)


def commands_path(experiment_dir: str) -> str:
    """命令账本路径：``<实验目录>/runtime/commands.jsonl``。"""
    return os.path.join(runtime_dir(experiment_dir), FILE_COMMANDS)


def env_path(experiment_dir: str) -> str:
    """环境快照路径：``<实验目录>/runtime/env.json``。"""
    return os.path.join(runtime_dir(experiment_dir), FILE_ENV)


def logs_dir(experiment_dir: str) -> str:
    """日志目录：``<实验目录>/runtime/logs``。"""
    return os.path.join(runtime_dir(experiment_dir), DIR_LOGS)


def read_commands(experiment_dir: str) -> list[dict[str, Any]]:
    """读 ``commands.jsonl``（按追加顺序；没有文件返回空列表）。

    坏行 / 非对象行跳过——status 与 watch 是诊断视图，一份坏账本不该让整条命令
    连"跑到哪了"都看不到。只读，不创建任何东西。
    """
    path = commands_path(experiment_dir)
    records: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    item = json.loads(text)
                except ValueError:
                    continue
                if isinstance(item, dict):
                    records.append(item)
    except OSError:
        return []
    return records


def summarize(experiment_dir: str, limit: int = 5) -> dict[str, Any]:
    """运行时数据的摘要（``task status`` / ``watch_task`` 用）。

    Returns:
        ``{"dir", "exists", "n_commands", "n_failed", "recent", "last_log"}``：
        ``recent`` 是最近 ``limit`` 条记录（原顺序），``last_log`` 是最后一条有
        日志的记录的**绝对路径**。没有 ``runtime/`` 时 ``exists`` 为 False，其余为
        空值——调用方据此输出「无运行时数据」。
    """
    directory = runtime_dir(experiment_dir)
    if not os.path.isdir(directory):
        return {
            "dir": directory,
            "exists": False,
            "n_commands": 0,
            "n_failed": 0,
            "recent": [],
            "last_log": None,
        }
    commands = read_commands(experiment_dir)
    last_log: Union[str, None] = None
    for record in commands:
        relative = record.get("log")
        if isinstance(relative, str) and relative:
            candidate = os.path.join(experiment_dir, relative)
            if os.path.isfile(candidate):
                last_log = candidate
    return {
        "dir": directory,
        "exists": True,
        "n_commands": len(commands),
        "n_failed": sum(
            1
            for record in commands
            if isinstance(record.get("exit_code"), int)
            and record["exit_code"] != 0
        ),
        "recent": commands[-limit:] if limit > 0 else [],
        "last_log": last_log,
    }


# ---------------------------------------------------------------------------
# 脱敏
# ---------------------------------------------------------------------------


def _normalized_name(name: str) -> str:
    """参数名归一：去掉前导 ``-``，``_`` 换成 ``-``，小写。"""
    return name.strip().lstrip("-").replace("_", "-").lower()


def is_secret_name(name: str) -> bool:
    """参数名是否按命名就会被当作密钥（``--llm-api-key`` / ``token`` / ``*secret*``）。"""
    key = _normalized_name(name)
    return any(marker in key for marker in _SECRET_NAME_MARKERS)


def _redact_payload(text: str) -> str:
    """脱敏 ``KEY=VALUE`` / URL 查询串形式的值（``api_key=abc`` → ``api_key=***``）。"""
    if "=" not in text:
        if _SECRET_VALUE_RE.search(text):
            return _MASK
        return text
    key, _, value = text.partition("=")
    if is_secret_name(key) or _SECRET_VALUE_RE.search(value):
        return f"{key}={_MASK}"
    return text


def redact_argv(argv: Sequence[Any]) -> list[str]:
    """argv 脱敏：密钥类参数的值一律替换为 ``***``。

    覆盖三种形态：``--llm-api-key SECRET``（掩下一个 token）、``--api-key=SECRET``
    （掩 ``=`` 右侧）、以及 ``--adapter-arg api_key=SECRET`` / 裸 ``token=...``
    （掩 ``KEY=VALUE`` 里的值）。参数名命中 :func:`is_secret_name`、或值里出现
    ``token=`` / ``secret:`` 这类模式都算。下一个 token 本身是选项（以 ``-`` 开头）
    时不掩它——那是漏填值的用法，掩掉选项会让 argv 失去可读性。
    """
    redacted: list[str] = []
    mask_next = False
    for raw in argv:
        text = str(raw)
        if mask_next:
            mask_next = False
            if text.startswith("-"):
                redacted.append(text)
            else:
                redacted.append(_MASK)
            continue
        if text.startswith("-"):
            name, sep, value = text.partition("=")
            if is_secret_name(name):
                if sep:
                    redacted.append(f"{name}={_MASK}")
                else:
                    redacted.append(text)
                    mask_next = True
                continue
            if sep:
                redacted.append(f"{name}={_redact_payload(value)}")
                continue
            redacted.append(text)
            continue
        redacted.append(_redact_payload(text))
    return redacted


# ---------------------------------------------------------------------------
# 环境快照
# ---------------------------------------------------------------------------


def _masked_env() -> dict[str, str]:
    """只收 ``MEMBENCH_*`` 与 ``TZ``；密钥命名的变量同样掩值。"""
    env: dict[str, str] = {}
    for key in sorted(os.environ):
        if not (key.startswith("MEMBENCH_") or key == "TZ"):
            continue
        env[key] = _MASK if is_secret_name(key) else os.environ[key]
    return env


def _membench_version() -> Union[str, None]:
    """已安装的 membench 版本（未安装 / 元数据缺失时 None）。"""
    try:
        from importlib import metadata

        return metadata.version("membench")
    except Exception:  # noqa: BLE001 - 环境信息缺失不应影响评测
        return None


def _membench_commit() -> Union[str, None]:
    """源码树当前 commit（能顺着 ``.git`` 找到就返回，否则 None，纯标准库）。

    只看包目录向上的 2 级（仓库内是 ``<repo>/membench/membench`` → ``<repo>/.git``）；
    打包后的 skill 里没有 ``.git``，不会误记宿主项目的 commit。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(3):
        git_dir = os.path.join(here, ".git")
        if os.path.isdir(git_dir):
            try:
                with open(os.path.join(git_dir, "HEAD"), encoding="utf-8") as handle:
                    head = handle.read().strip()
            except OSError:
                return None
            if not head.startswith("ref:"):
                return head or None
            ref_path = os.path.join(git_dir, *head[4:].strip().split("/"))
            try:
                with open(ref_path, encoding="utf-8") as handle:
                    return handle.read().strip() or None
            except OSError:
                return None
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return None


def _env_snapshot() -> dict[str, Any]:
    """一次性环境快照的结构（python / 平台 / membench 版本与 commit / MEMBENCH_*）。"""
    return {
        "schema": ENV_SCHEMA,
        "created_at": _now(),
        "python": platform_module.python_version(),
        "executable": sys.executable,
        "platform": platform_module.platform(),
        "membench_version": _membench_version(),
        "membench_commit": _membench_commit(),
        "env": _masked_env(),
    }


# ---------------------------------------------------------------------------
# 捕获会话
# ---------------------------------------------------------------------------


def _now() -> str:
    """本地时间（带时区偏移，精确到秒）。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _safe_command_name(command: str) -> str:
    """日志文件名里的命令段（只留字母数字与 ``._-``）。"""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(command)).strip("-")
    return cleaned or "membench"


class _Tee:
    """把写往终端的内容同时抄一份进日志文件（终端行为不变）。"""

    def __init__(self, original: Any, handle: IO[str], lock: threading.Lock) -> None:
        self._original = original
        self._handle = handle
        self._lock = lock

    def write(self, data: str) -> int:
        written = self._original.write(data)
        with self._lock:
            try:
                self._handle.write(data)
                self._handle.flush()
            except (OSError, ValueError):
                pass
        return written

    def flush(self) -> None:
        try:
            self._original.flush()
        except (OSError, ValueError):
            pass
        with self._lock:
            try:
                self._handle.flush()
            except (OSError, ValueError):
                pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def rebind(self, handle: IO[str]) -> None:
        """切换日志句柄（跨设备落盘时日志会换到目标文件继续写）。"""
        with self._lock:
            self._handle = handle


class Session:
    """一次 CLI 调用的运行时捕获：临时日志 + 终端 tee + 结束时落盘。"""

    def __init__(self, command: str, argv: Sequence[Any], cwd: str) -> None:
        self.command = command
        self.argv = redact_argv(argv)
        self.cwd = cwd
        self.pid = os.getpid()
        self.started_at = _now()
        self._started = time.monotonic()
        self.experiment_dir: Union[str, None] = None
        self.log_path: Union[str, None] = None
        self._tmp_path: Union[str, None] = None
        self._handle: Union[IO[str], None] = None
        self._lock = threading.Lock()
        self._stdout: Any = None
        self._stderr: Any = None
        self._tee_out: Any = None
        self._tee_err: Any = None
        self._closed = False
        self._open()

    # -- 生命周期 -------------------------------------------------------
    def _open(self) -> None:
        """建临时日志文件并把 stdout / stderr 换成 tee。"""
        fd, path = tempfile.mkstemp(prefix="membench-runtime-", suffix=".log")
        self._tmp_path = path
        self._handle = os.fdopen(fd, "a", encoding="utf-8", errors="replace", buffering=1)
        self._stdout, self._stderr = sys.stdout, sys.stderr
        self._tee_out = _Tee(self._stdout, self._handle, self._lock)
        self._tee_err = _Tee(self._stderr, self._handle, self._lock)
        sys.stdout, sys.stderr = self._tee_out, self._tee_err

    def _restore(self) -> None:
        """恢复原始 stdout / stderr（只恢复一次）。"""
        if self._stdout is not None and sys.stdout is self._tee_out:
            sys.stdout = self._stdout
        if self._stderr is not None and sys.stderr is self._tee_err:
            sys.stderr = self._stderr

    def _log_only(self, text: str) -> None:
        """只写日志、不写终端（记下最终由解释器打印的退出信息）。"""
        with self._lock:
            try:
                self._handle.write(text)  # type: ignore[union-attr]
                self._handle.flush()  # type: ignore[union-attr]
            except (OSError, ValueError, AttributeError):
                pass

    def note_exception(self, exc: BaseException) -> None:
        """把异常的最终信息补进日志（终端上的输出由调用方 / 解释器负责）。

        ``SystemExit`` 的消息与未捕获异常的 traceback 都在 ``main()`` 返回之后才由
        解释器写到 stderr——那时 tee 已恢复，不补这一笔日志里就只剩"跑了一半"。
        """
        if isinstance(exc, SystemExit):
            code = exc.code
            if isinstance(code, str) and code:
                self._log_only(code if code.endswith("\n") else code + "\n")
            return
        if isinstance(exc, KeyboardInterrupt):
            self._log_only("[membench] 已中断（KeyboardInterrupt）\n")
            return
        self._log_only(traceback.format_exc())

    def _relocate(self, target: str) -> None:
        """把临时日志放到 ``target``；跨设备时改为复制 + 切换写句柄。

        实验目录可能在另一块盘上（``MEMBENCH_OUTPUT_DIR`` 指外置卷很常见），
        ``os.replace`` 会因 ``EXDEV`` 失败。此时把已写内容复制过去，再把 tee 的句柄
        换到目标文件——后续输出继续进日志，不会因为跨卷而静默丢日志。
        """
        if self._tmp_path is None or self._handle is None:
            raise OSError("运行时日志临时文件不可用")
        try:
            os.replace(self._tmp_path, target)
            self._tmp_path = None
            return
        except OSError:
            pass
        with open(self._tmp_path, "rb") as source, open(target, "ab") as destination:
            shutil.copyfileobj(source, destination)
        new_handle = open(target, "a", encoding="utf-8", errors="replace", buffering=1)
        old_handle = self._handle
        self._handle = new_handle
        self._tee_out.rebind(new_handle)
        self._tee_err.rebind(new_handle)
        try:
            old_handle.close()
        except OSError:
            pass
        try:
            os.remove(self._tmp_path)
            self._tmp_path = None
        except OSError:
            pass

    def place(self, experiment_dir: str) -> None:
        """把临时日志移进 ``<实验目录>/runtime/logs/``（首次调用生效）。

        移动前写入的内容一并带过去，因此命令一早就打印的内容不会丢；重复调用同
        路径幂等，不同路径忽略——一次命令只记一个实验目录。
        """
        if self._closed or self.experiment_dir is not None:
            return
        if not str(experiment_dir).strip():
            return
        target_dir = logs_dir(os.path.abspath(experiment_dir))
        target = ""
        try:
            os.makedirs(target_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            base = f"{stamp}-{_safe_command_name(self.command)}-{self.pid}"
            target = os.path.join(target_dir, f"{base}.log")
            suffix = 1
            while os.path.exists(target):
                target = os.path.join(target_dir, f"{base}-{suffix}.log")
                suffix += 1
            self._relocate(target)
        except OSError as exc:
            self._log_only(f"[membench] 运行时日志落盘失败: {exc}\n")
            return
        self.experiment_dir = os.path.abspath(experiment_dir)
        self.log_path = target

    # -- 落盘 -----------------------------------------------------------
    def _record(self, exit_code: int) -> dict[str, Any]:
        """本次调用的 ``commands.jsonl`` 记录。"""
        relative = None
        if self.log_path and self.experiment_dir:
            relative = os.path.relpath(self.log_path, self.experiment_dir).replace(os.sep, "/")
        return {
            "schema": COMMANDS_SCHEMA,
            "argv": self.argv,
            "cwd": self.cwd,
            "command": self.command,
            "started_at": self.started_at,
            "finished_at": _now(),
            "duration_ms": int((time.monotonic() - self._started) * 1000),
            "exit_code": int(exit_code),
            "python": platform_module.python_version(),
            "platform": platform_module.platform(),
            "pid": self.pid,
            "log": relative,
            "env": _masked_env(),
        }

    def _append_command(self, exit_code: int) -> None:
        """单行追加（``O_APPEND``）：并发分片不会互相覆盖、也不做读-改-写。"""
        if not self.experiment_dir:
            return
        path = commands_path(self.experiment_dir)
        line = json.dumps(self._record(exit_code), ensure_ascii=False) + "\n"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line.encode("utf-8"))
            finally:
                os.close(fd)
        except OSError:
            pass

    def _write_env(self) -> None:
        """一次性环境快照（``O_EXCL``：并发的第一个进程写，其余直接跳过）。"""
        if not self.experiment_dir:
            return
        path = env_path(self.experiment_dir)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            return
        except OSError:
            return
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(_env_snapshot(), handle, ensure_ascii=False, indent=2, sort_keys=True)
        except OSError:
            pass

    def finish(self, exit_code: int) -> None:
        """结束捕获：恢复终端、按需写 ``commands.jsonl`` 与 ``env.json``。"""
        if self._closed:
            return
        self._closed = True
        self._restore()
        if self._handle is not None:
            try:
                self._handle.flush()
                self._handle.close()
            except OSError:
                pass
            self._handle = None
        try:
            if self.experiment_dir:
                self._append_command(exit_code)
                self._write_env()
        finally:
            if self._tmp_path and os.path.exists(self._tmp_path):
                try:
                    os.remove(self._tmp_path)
                except OSError:
                    pass
                self._tmp_path = None


#: 当前 CLI 会话（进程内唯一；``set_dir`` 委托给它）。
_current: Union[Session, None] = None


def begin(command: str, argv: Sequence[Any], *, cwd: Union[str, None] = None) -> Session:
    """开始一次 CLI 捕获（CLI 入口在解析完参数后调用）。"""
    global _current
    if _current is not None:
        raise RuntimeError("runtime 会话已存在：begin 只能由 CLI 入口调用一次")
    session = Session(command, argv, cwd or os.getcwd())
    _current = session
    return session


def set_dir(experiment_dir: str) -> None:
    """登记本次命令的实验 / 臂目录（没有活动会话时是空操作，库调用不受影响）。"""
    if _current is None:
        return
    _current.place(experiment_dir)


def finish(session: Session, exit_code: int) -> None:
    """结束会话并释放模块级引用。"""
    global _current
    try:
        session.finish(exit_code)
    finally:
        if _current is session:
            _current = None


def exit_code_of(exc: BaseException) -> int:
    """异常对应的进程退出码（用于把失败也记进 ``commands.jsonl``）。"""
    if isinstance(exc, SystemExit):
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    if isinstance(exc, KeyboardInterrupt):
        return 130
    return 1
