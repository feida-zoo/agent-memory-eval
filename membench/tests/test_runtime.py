"""实验 / 任务级运行时数据（``<exp>/runtime/``）的测试。

覆盖：

- 命令捕获：``logs/*.log`` 有完整 stdout+stderr（终端照常显示）、``commands.jsonl``
  字段齐全且 ``exit_code`` 正确、``env.json`` 只收 ``MEMBENCH_*`` / ``TZ`` 且不含密钥；
- argv 脱敏：``--llm-api-key`` / ``--api-key=...`` / ``token=...``；
- 并发：两个进程写同一个实验目录不丢条目（日志按 pid 分文件、commands 单行追加）；
- CLI 面：``run`` 成功后 runtime 三件套存在且日志有控制台输出；``--task-dir`` 写到
  该任务 / 臂目录下；失败命令在任务 runtime 里留下非零 ``exit_code``；
  只读的 ``task status`` 不产生 runtime。
"""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from membench import cli, layout, runtime
from membench.data import SAMPLE_DATASET_PATH

#: 并发用例的子进程脚本：各写各的日志 + 一行 commands 记录到同一个实验目录。
_CONCURRENT_SCRIPT = """
import os
from membench import runtime

session = runtime.begin("run", ["run", "--shard-index", os.environ["SHARD"]])
print("shard", os.environ["SHARD"])
runtime.set_dir(os.environ["EXP_DIR"])
runtime.finish(session, 0)
"""


def _commands(exp: Path) -> list[dict]:
    """读 commands.jsonl（JSON 列表，便于断言字段）。"""
    text = (exp / "runtime" / runtime.FILE_COMMANDS).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _read_json(path: Path) -> dict:
    """读 JSON。"""
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _isolated_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """产物根指向 tmp_path，且不加载本机持久化适配器。"""
    root = tmp_path / "output-root"
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(root))
    monkeypatch.delenv("MEMBENCH_ADAPTER_MODULES", raising=False)
    adapters = tmp_path / "adapters"
    adapters.mkdir()
    monkeypatch.setenv("MEMBENCH_ADAPTERS_DIR", str(adapters))
    return root


# ---------------------------------------------------------------------------
# 捕获与字段
# ---------------------------------------------------------------------------


def test_session_capture_writes_log_commands_and_env(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """stdout+stderr 全量进日志（终端照常显示），commands 字段齐全、exit_code 正确。"""
    exp = tmp_path / "exp"
    session = runtime.begin(
        "run",
        ["run", "--adapter", "mock-strong", "--dataset", "sample.jsonl"],
    )
    try:
        print("hello stdout")
        print("hello stderr", file=sys.stderr)
        runtime.set_dir(str(exp))
        print("after place")
    finally:
        runtime.finish(session, 0)

    terminal = capsys.readouterr()
    assert "hello stdout" in terminal.out and "hello stderr" in terminal.err
    assert "after place" in terminal.out

    commands = _commands(exp)
    assert len(commands) == 1
    record = commands[0]
    for key in (
        "schema",
        "argv",
        "cwd",
        "command",
        "started_at",
        "finished_at",
        "duration_ms",
        "exit_code",
        "python",
        "platform",
        "pid",
        "log",
        "env",
    ):
        assert key in record, f"commands.jsonl 缺字段 {key}"
    assert record["command"] == "run"
    assert record["exit_code"] == 0
    assert record["pid"] == os.getpid()
    assert record["argv"] == ["run", "--adapter", "mock-strong", "--dataset", "sample.jsonl"]
    assert record["log"] == f"runtime/logs/{Path(record['log']).name}"
    log_path = exp / record["log"]
    assert log_path.is_file()
    log_text = log_path.read_text(encoding="utf-8")
    assert "hello stdout" in log_text
    assert "hello stderr" in log_text
    assert "after place" in log_text

    env = _read_json(exp / "runtime" / runtime.FILE_ENV)
    assert env["schema"] == runtime.ENV_SCHEMA
    assert env["python"]
    assert env["platform"]
    assert "env" in env


def test_finish_records_nonzero_exit_code(tmp_path: Path) -> None:
    """失败命令同样落一条记录（exit_code 非零），日志文件名含命令与 pid。"""
    exp = tmp_path / "exp"
    session = runtime.begin("fuse", ["fuse", "--task-dir", "x"])
    try:
        runtime.set_dir(str(exp))
    finally:
        runtime.finish(session, 3)
    record = _commands(exp)[0]
    assert record["exit_code"] == 3
    assert Path(record["log"]).name.endswith(f"-fuse-{os.getpid()}.log")


def test_summarize_reports_counts_and_last_log(tmp_path: Path) -> None:
    """summarize: 命令数 / 非零退出数 / 最近日志；没有 runtime 时 exists=False。"""
    exp = tmp_path / "exp"
    for code in (0, 1):
        session = runtime.begin("run", ["run", str(code)])
        try:
            runtime.set_dir(str(exp))
        finally:
            runtime.finish(session, code)

    summary = runtime.summarize(str(exp), limit=5)
    assert summary["exists"] is True
    assert summary["n_commands"] == 2
    assert summary["n_failed"] == 1
    assert len(summary["recent"]) == 2
    assert summary["last_log"] is not None and Path(summary["last_log"]).is_file()

    empty = runtime.summarize(str(tmp_path / "nope"))
    assert empty == {
        "dir": str(tmp_path / "nope" / "runtime"),
        "exists": False,
        "n_commands": 0,
        "n_failed": 0,
        "recent": [],
        "last_log": None,
    }


def test_place_falls_back_to_copy_across_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """跨设备（EXDEV）时改为复制 + 切换句柄：移动前后写的内容都在日志里。"""
    exp = tmp_path / "exp"
    session = runtime.begin("run", ["run"])
    real_replace = os.replace

    def failing_replace(src: object, dst: object, *args: object, **kwargs: object) -> None:
        if str(dst).endswith(".log"):
            raise OSError(errno.EXDEV, "cross-device link")
        real_replace(src, dst, *args, **kwargs)  # type: ignore[arg-type]

    try:
        print("before place")
        monkeypatch.setattr(os, "replace", failing_replace)
        runtime.set_dir(str(exp))
        monkeypatch.setattr(os, "replace", real_replace)
        print("after place")
    finally:
        runtime.finish(session, 0)

    logs = sorted((exp / "runtime" / "logs").glob("*.log"))
    assert len(logs) == 1
    text = logs[0].read_text(encoding="utf-8")
    assert "before place" in text
    assert "after place" in text
    assert _commands(exp)[0]["exit_code"] == 0
    assert _commands(exp)[0]["log"] == f"runtime/logs/{logs[0].name}"


# ---------------------------------------------------------------------------
# argv 脱敏
# ---------------------------------------------------------------------------


def test_redact_argv_secret_forms() -> None:
    """``--llm-api-key`` / ``--api-key=`` / ``token=`` 三种形态都掩值。"""
    argv = [
        "run",
        "--llm-api-key",
        "sk-live-1",
        "--api-key=sk-live-2",
        "--adapter-arg",
        "api_key=sk-live-3",
        "--dataset",
        "sample.jsonl",
        "--llm-base-url",
        "https://example.test/v1?token=sk-live-4",
        "token=sk-live-5",
        "--password",
        "p@ss",
        "--timeout",
        "10",
    ]
    assert runtime.redact_argv(argv) == [
        "run",
        "--llm-api-key",
        "***",
        "--api-key=***",
        "--adapter-arg",
        "api_key=***",
        "--dataset",
        "sample.jsonl",
        "--llm-base-url",
        "https://example.test/v1?token=***",
        "token=***",
        "--password",
        "***",
        "--timeout",
        "10",
    ]
    # 密钥不出现在任何位置
    text = json.dumps(runtime.redact_argv(argv), ensure_ascii=False)
    for secret in ("sk-live-1", "sk-live-2", "sk-live-3", "sk-live-4", "sk-live-5", "p@ss"):
        assert secret not in text


def test_redact_argv_keeps_missing_value_flag() -> None:
    """密钥选项后面紧跟另一个选项（漏填值）时不掩选项本身。"""
    assert runtime.redact_argv(["--api-key", "--judge", "heuristic"]) == [
        "--api-key",
        "--judge",
        "heuristic",
    ]


def test_env_snapshot_excludes_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """env.json 只收 MEMBENCH_* / TZ；密钥命名的变量连值也不落。"""
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("MEMBENCH_SECRET_TOKEN", "top-secret-value")
    monkeypatch.setenv("TZ", "Asia/Shanghai")
    exp = tmp_path / "exp"
    session = runtime.begin("run", ["run"])
    try:
        runtime.set_dir(str(exp))
    finally:
        runtime.finish(session, 0)

    text = (exp / "runtime" / runtime.FILE_ENV).read_text(encoding="utf-8")
    assert "top-secret-value" not in text
    env = _read_json(exp / "runtime" / runtime.FILE_ENV)["env"]
    assert env["MEMBENCH_OUTPUT_DIR"] == str(tmp_path / "out")
    assert env["MEMBENCH_SECRET_TOKEN"] == "***"
    assert env["TZ"] == "Asia/Shanghai"
    assert all(key.startswith("MEMBENCH_") or key == "TZ" for key in env)


# ---------------------------------------------------------------------------
# 并发
# ---------------------------------------------------------------------------


def test_concurrent_processes_append_without_loss(tmp_path: Path) -> None:
    """两个进程写同一实验目录：commands 两条都在、日志各一份（按 pid 分文件）。"""
    exp = tmp_path / "exp"
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["EXP_DIR"] = str(exp)
    processes = []
    for shard in ("0", "1"):
        child_env = {**env, "SHARD": shard}
        processes.append(
            subprocess.Popen(
                [sys.executable, "-c", _CONCURRENT_SCRIPT],
                env=child_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    for proc in processes:
        _out, err = proc.communicate(timeout=60)
        assert proc.returncode == 0, err

    commands = _commands(exp)
    assert len(commands) == 2
    assert sorted(record["pid"] for record in commands) == sorted(
        proc.pid for proc in processes
    )
    logs = sorted((exp / "runtime" / "logs").glob("*.log"))
    assert len(logs) == 2
    assert len({record["log"] for record in commands}) == 2


# ---------------------------------------------------------------------------
# CLI 面
# ---------------------------------------------------------------------------


def _init_task(base: Path, name: str, arms: int = 1) -> Path:
    """走 CLI ``task init`` 建任务目录（并从 stdout 取回路径）。"""
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert cli.main(["task", "init", "--name", name, "--groups", str(arms),
                         "--report-out", str(base)]) == 0
    return Path(buffer.getvalue().strip())


def test_cli_run_writes_runtime_trio(tmp_path: Path) -> None:
    """run 成功后 <exp>/runtime/ 三件套齐备，日志里有控制台输出。"""
    out = tmp_path / "out"
    assert (
        cli.main(
            [
                "run",
                "--adapter",
                "mock-strong",
                "--dataset",
                str(SAMPLE_DATASET_PATH),
                "--judge",
                "heuristic",
                "--report-out",
                str(out),
            ]
        )
        == 0
    )
    exp = next(p for p in out.iterdir() if p.is_dir())
    runtime_dir = exp / layout.DIR_RUNTIME
    assert (runtime_dir / runtime.FILE_COMMANDS).is_file()
    assert (runtime_dir / runtime.FILE_ENV).is_file()
    log_file = next(Path(runtime.logs_dir(str(exp))).glob("*.log"))
    log_text = log_file.read_text(encoding="utf-8")
    assert "[membench] 完成:" in log_text
    assert "[membench] run_id=" in log_text
    record = _commands(exp)[0]
    assert record["exit_code"] == 0
    assert record["command"] == "run"
    assert record["log"].startswith("runtime/logs/")


def test_cli_run_task_dir_writes_runtime_under_arm(tmp_path: Path) -> None:
    """--task-dir <任务>/A：runtime 落该臂目录；任务根只记 task init 的那条。"""
    task = _init_task(tmp_path / "out", "ab", arms=2)
    arm = task / layout.DIR_GROUP_BASELINE
    assert (
        cli.main(
            [
                "run",
                "--adapter",
                "mock-strong",
                "--dataset",
                str(SAMPLE_DATASET_PATH),
                "--judge",
                "heuristic",
                "--limit",
                "3",
                "--task-dir",
                str(arm),
            ]
        )
        == 0
    )
    arm_commands = _commands(arm)
    assert len(arm_commands) == 1
    assert arm_commands[0]["command"] == "run"
    assert arm_commands[0]["exit_code"] == 0
    assert (arm / layout.DIR_RUNTIME / runtime.FILE_ENV).is_file()

    task_commands = _commands(task)
    assert [record["command"] for record in task_commands] == ["task"]
    assert task_commands[0]["exit_code"] == 0


def test_cli_failure_with_task_dir_records_exit_code(tmp_path: Path) -> None:
    """任务目录里的失败命令也要留痕：exit_code 非零 + 日志含报错原因。"""
    task = _init_task(tmp_path / "out", "boom", arms=1)
    with pytest.raises(SystemExit, match="未知适配器"):
        cli.main(
            [
                "run",
                "--adapter",
                "no-such-adapter",
                "--dataset",
                str(SAMPLE_DATASET_PATH),
                "--judge",
                "heuristic",
                "--task-dir",
                str(task),
            ]
        )
    commands = _commands(task)
    assert [record["command"] for record in commands] == ["task", "run"]
    failed = commands[-1]
    assert failed["exit_code"] == 1
    log_text = (task / failed["log"]).read_text(encoding="utf-8")
    assert "未知适配器" in log_text


def test_task_status_is_read_only(tmp_path: Path) -> None:
    """task status 不产生 / 不追加 runtime（只读命令不记账）。"""
    task = _init_task(tmp_path / "out", "solo", arms=1)
    before = _commands(task)
    log_files = sorted((task / layout.DIR_RUNTIME / "logs").glob("*.log"))
    assert cli.main(["task", "status", str(task)]) == 0
    assert _commands(task) == before
    assert sorted((task / layout.DIR_RUNTIME / "logs").glob("*.log")) == log_files
