"""多线对比（membench.series）测试。

覆盖三块：

1. 命名：conventional 前缀 / 中英文冒号 / 首尾标点 / 24 字截断 / 重名追加 sha8 /
   git 取不到回退短 sha / 显式名优先 / 第一条默认「基线」；
2. 三线对比：结论表的 vs 基线、vs 上一步数值，门禁 PASS/FAIL 与退出码，
   ``--require-comparable`` 的 strict 行为；
3. 多数据集：逐数据集表 + 融合表，办公域槽位不进融合；图表与报告内相对引用无死链。

产物用 ``Runner`` + 测试内的 :class:`_QualityAdapter` 真实跑出来（同一适配器名，
答对率由 quality 控制，模拟"同一系统不同 commit"），只在需要精确数值时才读
``metrics.json`` 反推期望值——测试不写死 mock 的具体准确率。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from membench import cli, layout, series
from membench.compare import check_comparability
from membench.adapters.base import QueryResult
from membench.adapters.mock import MockWeakAdapter
from membench.data import SAMPLE_DATASET_PATH
from membench.runner import Runner
from membench.userdata import builtin_office_dataset_path

#: 三线测试共用的中文 commit 主题（顺序即时间顺序）。
_SUBJECTS = [
    "feat: 优化检索排序",
    "fix: 修复旧记忆覆盖新值",
    "perf(core)：降低查询延迟",
]


class _QualityAdapter(MockWeakAdapter):
    """同一适配器名，答对率由 ``quality`` 控制（模拟同一系统不同 commit 的记忆质量）。

    题目内容的 sha256 决定这一版是否"答对"，因此 quality 越高 accuracy 越高且完全
    确定；适配器名 / 版本 / 驱动模型都可保持一致，跨线只有 commit 变化。
    """

    name = "quality-memory"

    def __init__(self, quality: int) -> None:
        super().__init__()
        self.quality = quality

    def query(self, question: str) -> QueryResult:
        result = super().query(question)
        hit = int(hashlib.sha256(question.encode("utf-8")).hexdigest(), 16) % 100
        if hit < self.quality:
            return result
        return QueryResult(
            answer="（未命中记忆）",
            retrieved=result.retrieved,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            llm_calls=result.llm_calls,
        )


def _git(repo: Path, *args: str) -> str:
    """执行一条 git 命令并返回 stdout（失败即测试失败）。"""
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _make_repo(tmp_path: Path, messages: list[str]) -> tuple[str, list[str]]:
    """建临时 git 仓库并逐个提交，返回 ``(repo, [sha 旧→新])``。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "commit.gpgsign", "false")
    shas: list[str] = []
    for message in messages:
        _git(repo, "commit", "--allow-empty", "-q", "-m", message)
        shas.append(_git(repo, "rev-parse", "HEAD"))
    return str(repo), shas


@pytest.fixture
def git_repo(tmp_path: Path) -> tuple[str, list[str]]:
    """三中文 commit 的临时仓库（没有 git 的环境跳过依赖它的用例）。"""
    if shutil.which("git") is None:
        pytest.skip("需要 git 才能测 commit 命名")
    return _make_repo(tmp_path, _SUBJECTS)


def _run(
    tmp_path: Path,
    name: str,
    quality: int,
    datasets: tuple[str, ...] = (str(SAMPLE_DATASET_PATH),),
    **runner_kwargs: object,
) -> str:
    """用 :class:`_QualityAdapter` 跑一份（或同一实验目录下的多份）数据集产物。"""
    exp = tmp_path / name
    kwargs = {"adapter_version": "v1", "driver_llm": "none", **runner_kwargs}
    for dataset in datasets:
        Runner(
            _QualityAdapter(quality),
            dataset,
            judge="heuristic",
            out_dir=str(exp),
            write_task_reports=False,
            **kwargs,
        ).run()
    return str(exp)


def _metrics(experiment_dir: str, dataset: str = "sample_dataset") -> dict:
    """读数据集级 ``report/metrics.json``。"""
    path = Path(layout.dataset_report_dir(experiment_dir, dataset)) / layout.FILE_METRICS
    return json.loads(path.read_text(encoding="utf-8"))


def _accuracy(experiment_dir: str, dataset: str = "sample_dataset") -> float:
    return float(_metrics(experiment_dir, dataset)["overall"]["accuracy"])


def _pp(diff: float) -> str:
    """与 membench.ratings.format_metric_delta 同口径（比率差值用 pt）。"""
    return f"{diff * 100:+.1f}pt"


def _section_rows(markdown: str, title: str) -> list[list[str]]:
    """取某个 ``## `` 小节里第一张表的数据行（表头与分隔行去掉）。"""
    lines = markdown.splitlines()
    start = lines.index(title)
    rows: list[list[str]] = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(set(cell) <= set("-: ") for cell in cells):
            continue
        rows.append(cells)
    return rows[1:]


def _assert_links_alive(report_path: Path) -> list[str]:
    """报告里的相对引用必须都指向真实文件（图与副本报告）。"""
    text = report_path.read_text(encoding="utf-8")
    refs = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    assert refs, "报告里没有任何相对引用"
    for ref in refs:
        assert not ref.startswith(("http://", "https://")), ref
        assert (report_path.parent / ref).is_file(), f"死链: {ref}"
    return refs


# ---------------------------------------------------------------------------
# 命名
# ---------------------------------------------------------------------------


def test_extract_name_strips_prefix_and_punctuation() -> None:
    assert series.extract_name_from_subject("feat(cli): 优化检索排序") == "优化检索排序"
    assert series.extract_name_from_subject("refactor(core)：统一记忆写入") == "统一记忆写入"
    assert series.extract_name_from_subject("fix: ！修复召回排序！") == "修复召回排序"
    assert series.extract_name_from_subject("   perf:   降低延迟  ") == "降低延迟"
    # 中文标题里的冒号不是 conventional 前缀，正是要展示的优化点，不能删
    assert series.extract_name_from_subject("优化：检索排序") == "优化：检索排序"


def test_extract_name_truncates_to_24() -> None:
    source = "很长的优化说明" * 6
    name = series.extract_name_from_subject(f"feat: {source}")
    assert name == source[: series.LINE_NAME_MAX]
    assert len(name) <= series.LINE_NAME_MAX


def test_line_name_first_is_baseline_and_subjects_resolve(git_repo) -> None:
    repo, shas = git_repo
    assert series.line_name(shas[0], repo, 0) == "基线"
    assert series.line_name(shas[1], repo, 1) == "修复旧记忆覆盖新值"
    assert series.line_name(shas[2], repo, 2) == "降低查询延迟"
    # 只有第一线接受显式基线名；其余线的左侧是 commit，显式名不生效
    assert series.line_name(shas[0], repo, 0, "第一版") == "第一版"
    assert series.line_name(shas[1], repo, 1, "第一版") == "修复旧记忆覆盖新值"


def test_line_name_explicit_name_wins(tmp_path: Path) -> None:
    # 左侧不是 7~40 位十六进制 -> 显式线名，原样使用（不做自动清洗）
    assert series.line_name("我的优化版", str(tmp_path), 3) == "我的优化版"


def test_line_name_falls_back_to_sha_when_git_missing(tmp_path: Path) -> None:
    warned: list[str] = []
    assert series.line_name("deadbeef", str(tmp_path), 1, warn=warned.append) == "deadbeef"
    assert len(warned) == 1
    assert "回退为短 sha" in warned[0]


def test_line_name_english_subject_falls_back_to_sha(tmp_path: Path) -> None:
    repo, shas = _make_repo(tmp_path, ["fix: improve retrieval"])
    warned: list[str] = []
    assert series.line_name(shas[0], repo, 1, warn=warned.append) == shas[0][:8]
    assert any("不含中文" in message for message in warned)


def test_duplicate_names_get_sha_suffix() -> None:
    names = series._dedupe_names(["修复召回", "修复召回"], ["a1b2c3d4e5f6", None])
    assert names == ["修复召回（a1b2c3d4）", "修复召回（第2线）"]


# ---------------------------------------------------------------------------
# LLM 命名：优先 LLM，失败 / 不合规 / 未配置降级代码抽取
# ---------------------------------------------------------------------------

#: 测试里的 LLM 配置（假服务，不发真实请求）。
_NAMING_LLM = series.LLMNaming(base_url="http://127.0.0.1:9/v1", model="fake")


class _FakeNamingLLM:
    """``chat_completion`` 的替身：按顺序返回预设输出，并记录每次请求。"""

    def __init__(self, outputs: list[object]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(
            {
                "base_url": base_url,
                "api_key": api_key,
                "model": model,
                "timeout": timeout,
                "messages": messages,
                "temperature": temperature,
            }
        )
        result = self.outputs.pop(0)
        if isinstance(result, BaseException):
            raise result
        return str(result)


def _fake_naming_llm(monkeypatch, outputs: list[object]) -> _FakeNamingLLM:
    """把 series 的 chat_completion 换成按序返回 outputs 的替身。"""
    fake = _FakeNamingLLM(outputs)
    monkeypatch.setattr(series, "chat_completion", fake)
    return fake


def _counting_naming_llm(monkeypatch) -> list[dict[str, object]]:
    """记录调用并立即报错的替身（用于断言"这条线不该调用 LLM"）。"""
    calls: list[dict[str, object]] = []

    def fake(*args: object, **kwargs: object) -> str:
        calls.append({"args": args, "kwargs": kwargs})
        raise AssertionError("这条线不该调用 LLM")

    monkeypatch.setattr(series, "chat_completion", fake)
    return calls


def test_llm_naming_used_and_report_marks_llm_and_code(
    tmp_path: Path, git_repo, monkeypatch, capsys
) -> None:
    """合法中文名采用并标 llm；模型不听话（英文）时回退代码抽取并标 code。"""
    repo, shas = git_repo
    runs = [_run(tmp_path, f"llm{i}", quality) for i, quality in enumerate((40, 60, 80))]
    specs = [f"{sha}={run}" for sha, run in zip(shas, runs)]
    # 基线不调 LLM，所以只有第 2、3 线两次请求
    fake = _fake_naming_llm(monkeypatch, ["旧值覆盖修复", "improve retrieval"])

    result = series.run_series(
        specs, repo=repo, report_out=str(tmp_path / "out"), llm=_NAMING_LLM
    )

    assert [line["name"] for line in result["lines"]] == [
        "基线",
        "旧值覆盖修复",
        "降低查询延迟",
    ]
    assert [line["name_source"] for line in result["lines"]] == ["基线", "llm", "code"]
    assert len(fake.calls) == 2
    first = fake.calls[0]
    assert (first["base_url"], first["model"]) == (_NAMING_LLM.base_url, "fake")
    assert first["temperature"] == 0
    user_prompt = first["messages"][-1]["content"]  # type: ignore[index]
    assert "修复旧记忆覆盖新值" in user_prompt
    assert "【改动摘要】" in user_prompt and "【改动片段】" in user_prompt
    # 英文输出走回退路径，只告警一次
    assert capsys.readouterr().err.count("LLM 命名失败，回退代码抽取") == 1

    report = Path(result["report_path"]).read_text(encoding="utf-8")
    source_rows = _section_rows(report, "## 来源与产物")
    assert [row[2] for row in source_rows] == ["基线", "llm", "code"]
    conclusion = _section_rows(report, "## 结论表")
    assert [row[1] for row in conclusion] == [
        "基线",
        "旧值覆盖修复",
        "降低查询延迟",
    ]


@pytest.mark.parametrize(
    "bad_output",
    [
        "improve retrieval",  # 全英文
        "很长的优化说明" * 4,  # 超过 24 字
        "「优化检索、排序」",  # 内部带标点（首尾引号会先剥掉）
    ],
)
def test_llm_invalid_output_falls_back_to_code(
    tmp_path: Path, git_repo, monkeypatch, bad_output: str
) -> None:
    """英文 / 超长 / 带标点 → 校验不通过，回退代码抽取并标 code。"""
    repo, shas = git_repo
    run = _run(tmp_path, "bad-name", 40)
    fake = _fake_naming_llm(monkeypatch, [bad_output])
    warned: list[str] = []

    lines = series.load_lines(
        [f"{shas[0]}={run}", f"{shas[1]}={run}"],
        repo=repo,
        warn=warned.append,
        llm=_NAMING_LLM,
    )

    assert [line.name for line in lines] == ["基线", "修复旧记忆覆盖新值"]
    assert [line.name_source for line in lines] == ["基线", "code"]
    assert len(fake.calls) == 1
    assert sum("LLM 命名失败，回退代码抽取" in message for message in warned) == 1


def test_llm_failure_warns_once_and_falls_back(
    tmp_path: Path, git_repo, monkeypatch
) -> None:
    """LLM 抛超时 / 异常：首次失败即短路——后续线不再请求、直接回退；告警只发一次。"""
    repo, shas = git_repo
    run = _run(tmp_path, "llm-error", 40)
    fake = _fake_naming_llm(
        monkeypatch, [TimeoutError("超时"), RuntimeError("服务 500")]
    )
    warned: list[str] = []

    lines = series.load_lines(
        [f"{sha}={run}" for sha in shas],
        repo=repo,
        warn=warned.append,
        llm=_NAMING_LLM,
    )

    assert [line.name for line in lines] == [
        "基线",
        "修复旧记忆覆盖新值",
        "降低查询延迟",
    ]
    assert [line.name_source for line in lines] == ["基线", "code", "code"]
    # 首次失败后 disabled：第三条线不再等一次超时，也不再请求
    assert len(fake.calls) == 1
    assert sum("LLM 命名失败，回退代码抽取" in message for message in warned) == 1


def test_llm_valid_name_strips_quotes_and_whitespace() -> None:
    """首尾引号 / 空白 / 标点剥掉后合法；内部标点仍判不合法。"""
    assert series.clean_llm_name("  “检索排序优化。”  ") == "检索排序优化"
    assert series.clean_llm_name("「旧值覆盖修复」") == "旧值覆盖修复"
    assert series.clean_llm_name("优化  检索") == "优化 检索"
    assert series.clean_llm_name("优化、检索") is None
    assert series.clean_llm_name("improve retrieval") is None
    assert series.clean_llm_name("很长的优化说明" * 4) is None
    assert series.clean_llm_name("") is None


@pytest.mark.parametrize(
    "llm",
    [
        None,
        series.LLMNaming(base_url="http://127.0.0.1:9/v1", model=""),
        series.LLMNaming(base_url="", model="fake"),
        series.LLMNaming(),
    ],
)
def test_no_llm_config_skips_llm(
    tmp_path: Path, git_repo, monkeypatch, llm
) -> None:
    """未配置 / 只给 base_url 或 model 之一：不调 LLM，直接代码抽取（不告警）。"""
    repo, shas = git_repo
    run = _run(tmp_path, "no-llm", 40)
    calls = _counting_naming_llm(monkeypatch)
    warned: list[str] = []

    lines = series.load_lines(
        [f"{shas[0]}={run}", f"{shas[1]}={run}"],
        repo=repo,
        warn=warned.append,
        llm=llm,
    )

    assert calls == []
    assert [line.name_source for line in lines] == ["基线", "code"]
    assert warned == []


def test_explicit_names_and_baseline_never_call_llm(
    tmp_path: Path, git_repo, monkeypatch
) -> None:
    """显式名永远优先、基线也不命名：只有第三条 commit 请求一次 LLM。"""
    repo, shas = git_repo
    run = _run(tmp_path, "explicit", 40)
    fake = _fake_naming_llm(monkeypatch, ["检索优化"])

    lines = series.load_lines(
        [f"{shas[0]}={run}", f"我的优化版={run}", f"{shas[1]}={run}"],
        repo=repo,
        llm=_NAMING_LLM,
    )

    assert [line.name for line in lines] == ["基线", "我的优化版", "检索优化"]
    assert [line.name_source for line in lines] == ["基线", "显式名", "llm"]
    assert len(fake.calls) == 1
    assert "修复旧记忆覆盖新值" in fake.calls[0]["messages"][-1]["content"]  # type: ignore[index]


def test_llm_prompt_truncates_large_diff(tmp_path: Path, monkeypatch) -> None:
    """大改动的 patch 按上限截断，prompt 不会随改动量线性膨胀。"""
    repo = tmp_path / "big-repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "commit.gpgsign", "false")
    big = repo / "big.txt"
    big.write_text("旧内容\n" * 2000, encoding="utf-8")
    _git(repo, "add", "big.txt")
    _git(repo, "commit", "-q", "-m", "feat: 初始化大文件")
    big.write_text("新内容\n" * 2000, encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "feat: 批量替换大文件")
    sha = _git(repo, "rev-parse", "HEAD")
    run = _run(tmp_path, "big-diff", 40)
    fake = _fake_naming_llm(monkeypatch, ["批量替换大文件"])

    series.load_lines(
        [f"{sha}={run}", f"{sha}={run}"],
        repo=str(repo),
        llm=_NAMING_LLM,
    )

    user_prompt = fake.calls[0]["messages"][-1]["content"]  # type: ignore[index]
    assert "…（已截断）" in user_prompt
    assert len(user_prompt) < series._PATCH_MAX + series._STAT_MAX + series._BODY_MAX + 200


def test_cli_series_llm_flags_wire_naming(monkeypatch, tmp_path: Path) -> None:
    """--llm-* 透传为 LLMNaming；--no-llm-naming 强制 None（不构造 LLM 配置）。"""
    captured: dict[str, object] = {}

    def fake_run_series(specs, **kwargs):
        captured.update(kwargs)
        return {"series_dir": str(tmp_path / "series"), "verdict": "PASS"}

    monkeypatch.setattr(cli, "run_series", fake_run_series)
    base = [
        "series",
        "--line",
        "aaaaaaaa=/tmp/a",
        "--line",
        "bbbbbbbb=/tmp/b",
        "--report-out",
        str(tmp_path / "out"),
    ]

    assert (
        cli.main(
            [
                *base,
                "--llm-base-url",
                "http://127.0.0.1:9/v1",
                "--llm-api-key",
                "k",
                "--llm-model",
                "fake",
                "--llm-timeout",
                "12.5",
            ]
        )
        == 0
    )
    llm = captured["llm"]
    assert isinstance(llm, series.LLMNaming)
    assert llm.enabled
    assert (llm.base_url, llm.api_key, llm.model, llm.timeout) == (
        "http://127.0.0.1:9/v1",
        "k",
        "fake",
        12.5,
    )

    captured.clear()
    assert (
        cli.main(
            [
                *base,
                "--llm-base-url",
                "http://127.0.0.1:9/v1",
                "--llm-model",
                "fake",
                "--no-llm-naming",
            ]
        )
        == 0
    )
    assert captured["llm"] is None

    # 不传 --llm-timeout 时与 evaluate 同口径：用默认 90s
    captured.clear()
    assert (
        cli.main(
            [*base, "--llm-base-url", "http://127.0.0.1:9/v1", "--llm-model", "fake"]
        )
        == 0
    )
    assert isinstance(captured["llm"], series.LLMNaming)
    assert captured["llm"].timeout == 90.0


# ---------------------------------------------------------------------------
# 三线对比：数值、门禁、退出码
# ---------------------------------------------------------------------------


def test_series_three_lines_numbers_and_exit_code(tmp_path: Path, git_repo) -> None:
    repo, shas = git_repo
    runs = [_run(tmp_path, f"exp{i}", quality) for i, quality in enumerate((40, 60, 80))]
    specs = [f"{sha}={run}" for sha, run in zip(shas, runs)]
    before = (Path(layout.dataset_report_dir(runs[0], "sample_dataset")) / layout.FILE_METRICS).read_bytes()

    result = series.run_series(specs, repo=repo, report_out=str(tmp_path / "out"))

    assert [line["name"] for line in result["lines"]] == [
        "基线",
        "修复旧记忆覆盖新值",
        "降低查询延迟",
    ]
    assert result["verdict"] == "PASS"
    assert result["dataset_keys"] == ["sample_dataset"]

    # 结论表：名称 / 短 sha / vs 基线 / vs 上一步 与 metrics 反推的期望值一致
    report_path = Path(result["report_path"])
    report = report_path.read_text(encoding="utf-8")
    rows = _section_rows(report, "## 结论表")
    accs = [_accuracy(run) for run in runs]
    assert rows[0][1] == "基线"
    assert rows[0][5] == "—" and rows[0][6] == "—"
    for i in (1, 2):
        assert rows[i][1] == result["lines"][i]["name"]
        assert rows[i][2] == shas[i][:8]
        assert rows[i][3] == f"{accs[i] * 100:.1f}%"
        assert rows[i][5] == _pp(accs[i] - accs[0])
        assert rows[i][6] == _pp(accs[i] - accs[i - 1])
        assert rows[i][7] == result["lines"][i]["verdict"]

    # 提升是真实正向的：quality 40 -> 60 -> 80 的 accuracy 单调上升
    assert accs[0] < accs[1] < accs[2]

    # 图表与副本：图在 reports/ 下，md 用同目录相对文件名引用；副本按线序落 runs/
    assert Path(result["fig_path"]) == report_path.parent / layout.FIG_SERIES_ACCURACY
    assert (report_path.parent / layout.FIG_SERIES_ACCURACY).is_file()
    assert f"]({layout.FIG_SERIES_ACCURACY})" in report
    for i, line in enumerate(result["lines"]):
        copy_dir = Path(layout.series_line_dir(result["series_dir"], i, line["name"], line["commit"]))
        assert line["copy_dir"] == str(copy_dir)
        assert (copy_dir / "sample_dataset" / layout.DIR_REPORT / layout.FILE_REPORT_MD).is_file()
    refs = _assert_links_alive(report_path)
    assert layout.FIG_SERIES_ACCURACY in refs

    # 只读：源产物的数据集级 metrics 一个字节都没变
    after = (Path(layout.dataset_report_dir(runs[0], "sample_dataset")) / layout.FILE_METRICS).read_bytes()
    assert before == after

    # CLI 同口径：退出码 0，实验目录名是 series_<ts>/
    rc = cli.main(
        ["series", "--repo", repo, *[item for spec in specs for item in ("--line", spec)],
         "--report-out", str(tmp_path / "cli_out")]
    )
    assert rc == 0
    created = [path for path in (tmp_path / "cli_out").iterdir() if path.is_dir()]
    assert len(created) == 1 and created[0].name.startswith("series_")
    assert (created[0] / layout.DIR_REPORTS / layout.FILE_REPORT_MD).is_file()


def test_series_regression_line_fails_with_exit_code_one(tmp_path: Path, git_repo) -> None:
    repo, shas = git_repo
    good = _run(tmp_path, "good", 80)
    bad = _run(tmp_path, "bad", 40)
    # 时间顺序：先能答对的版本，再明显劣化的版本
    specs = [f"{shas[2]}={good}", f"{shas[0]}={bad}"]

    rc = cli.main(
        ["series", "--repo", repo, *[item for spec in specs for item in ("--line", spec)],
         "--report-out", str(tmp_path / "out")]
    )

    assert rc == 1
    report_path = next((tmp_path / "out").glob("series_*/reports/report.md"))
    report = report_path.read_text(encoding="utf-8")
    rows = _section_rows(report, "## 结论表")
    assert rows[1][7] == "FAIL"
    assert rows[1][5].startswith("-")  # 相对基线是负值
    assert "FAIL" in report


def test_series_require_comparable_marks_strict_failure(tmp_path: Path, git_repo) -> None:
    repo, shas = git_repo
    base = _run(tmp_path, "base", 40)
    cand = _run(tmp_path, "cand", 60, driver_llm="other-model")
    specs = [f"{shas[0]}={base}", f"{shas[1]}={cand}"]

    # 非 strict：关键维度不一致只降级 WARN（与 compare 同口径），退出码 0
    loose = series.run_series(specs, repo=repo, report_out=str(tmp_path / "loose"))
    assert loose["lines"][1]["verdict"] == "WARN"
    assert loose["lines"][1]["comparability"]["comparable"] is False
    assert loose["verdict"] == "WARN"

    # strict：关键维度不一致计入 FAIL，退出码 1
    strict = series.run_series(
        specs, repo=repo, report_out=str(tmp_path / "strict"), require_comparable=True
    )
    assert strict["lines"][1]["verdict"] == "FAIL"
    assert strict["verdict"] == "FAIL"
    rc = cli.main(
        ["series", "--repo", repo, "--require-comparable",
         *[item for spec in specs for item in ("--line", spec)],
         "--report-out", str(tmp_path / "cli_strict")]
    )
    assert rc == 1


def test_series_tolerates_adapter_version_change(tmp_path: Path, git_repo) -> None:
    """多线场景里 adapter_version 是被测变量：series 按 info 级比较、不判不可比；
    ``compare`` 的直接口径不变（它仍是关键维度）。"""
    repo, shas = git_repo
    base = _run(tmp_path, "av-base", 40, adapter_version="v1")
    cand = _run(tmp_path, "av-cand", 60, adapter_version="v2")
    specs = [f"{shas[0]}={base}", f"{shas[1]}={cand}"]

    result = series.run_series(
        specs, repo=repo, report_out=str(tmp_path / "av"), require_comparable=True
    )
    line = result["lines"][1]
    assert line["comparability"]["comparable"] is True
    assert line["verdict"] in ("PASS", "WARN")

    # compare 的直接调用不受影响：adapter_version 仍是 critical
    direct = check_comparability(_metrics(base), _metrics(cand))
    assert direct["comparable"] is False

    # 降级只作用于显式指定的维度
    downgraded = check_comparability(
        _metrics(base), _metrics(cand), downgrade=("adapter_version",)
    )
    assert downgraded["comparable"] is True
    assert all(
        item["severity"] == "info"
        for item in downgraded["mismatches"]
        if item["dimension"] == "adapter_version"
    )


# ---------------------------------------------------------------------------
# 多数据集：逐数据集表 + 融合表，办公域槽位不进融合
# ---------------------------------------------------------------------------


def test_series_multi_dataset_fusion_excludes_office(tmp_path: Path, git_repo) -> None:
    repo, shas = git_repo
    datasets = (str(SAMPLE_DATASET_PATH), str(builtin_office_dataset_path()))
    runs = [_run(tmp_path, f"multi{i}", quality, datasets=datasets) for i, quality in enumerate((40, 80))]
    assert {path.name for path in (Path(runs[0]) / layout.DIR_RUNS).iterdir()} == {
        "sample_dataset",
        "office_dataset",
    }
    specs = [f"{sha}={run}" for sha, run in zip(shas, runs)]

    result = series.run_series(specs, repo=repo, report_out=str(tmp_path / "out"))

    assert result["dataset_keys"] == ["office_dataset", "sample_dataset"]
    # 融合 accuracy = 基准集（sample）的值：办公域槽位不参与
    sample_acc = _accuracy(runs[0], "sample_dataset")
    office_acc = _accuracy(runs[0], "office_dataset")
    assert sample_acc != pytest.approx(office_acc)
    assert result["lines"][0]["overall"]["accuracy"] == pytest.approx(sample_acc)

    report_path = Path(result["report_path"])
    report = report_path.read_text(encoding="utf-8")
    assert "## 逐数据集 accuracy" in report
    rows = _section_rows(report, "## 逐数据集 accuracy（vs 基线）")
    by_dataset = {row[0]: row for row in rows}
    assert set(by_dataset) == {"office_dataset", "sample_dataset"}
    assert by_dataset["sample_dataset"][1] == f"{sample_acc * 100:.1f}%"
    assert by_dataset["office_dataset"][1] == f"{office_acc * 100:.1f}%"
    assert by_dataset["office_dataset"][2] == (
        f"{_accuracy(runs[1], 'office_dataset') * 100:.1f}% "
        f"({_pp(_accuracy(runs[1], 'office_dataset') - office_acc)})"
    )

    # 逐指标表存在（多数据集只列质量 + 诊断），图仍是同目录相对引用
    assert "## 逐指标表" in report
    assert f"]({layout.FIG_SERIES_ACCURACY})" in report
    _assert_links_alive(report_path)


def test_series_per_dataset_metrics_and_bar_charts(
    tmp_path: Path, git_repo
) -> None:
    """逐数据集关键指标表 + 三张柱状图：表格与图都要有，且引用无死链。

    表的单元格口径与逐指标表一致：值 + （vs 基线差值）；图是"对比看柱更清楚"
    的补充，因此图与表必须同一个数据源（都来自各线各数据集的 metrics.overall）。
    """
    repo, shas = git_repo
    datasets = (str(SAMPLE_DATASET_PATH), str(builtin_office_dataset_path()))
    runs = [
        _run(tmp_path, f"charts{i}", quality, datasets=datasets)
        for i, quality in enumerate((40, 80))
    ]
    specs = [f"{sha}={run}" for sha, run in zip(shas, runs)]

    result = series.run_series(specs, repo=repo, report_out=str(tmp_path / "out"))
    report_path = Path(result["report_path"])
    report = report_path.read_text(encoding="utf-8")

    # 逐数据集关键指标：每个数据集一小节，行是指标、列是各线
    assert "## 逐数据集关键指标（vs 基线）" in report
    for dataset in ("office_dataset", "sample_dataset"):
        assert f"### {dataset}" in report
    section = report.split("## 逐数据集关键指标（vs 基线）", 1)[1]
    section = section.split("## accuracy 走势", 1)[0]
    for metric in ("正确率", "证据召回率", "陈旧率"):
        assert metric in section

    # 三张新图：都真实产出且被报告引用
    reports_dir = report_path.parent
    for name in (
        layout.FIG_SERIES_FUSED,
        layout.FIG_SERIES_PER_DATASET,
        layout.FIG_SERIES_DELTA,
    ):
        assert (reports_dir / name).is_file(), f"{name} 未生成"
        assert f"]({name})" in report, f"{name} 未在报告中引用"
    _assert_links_alive(report_path)


def test_series_single_dataset_has_no_per_dataset_section(
    tmp_path: Path, git_repo
) -> None:
    """单数据集：不重复出"逐数据集"表/图（融合表就是该数据集本身）。"""
    repo, shas = git_repo
    runs = [_run(tmp_path, f"single{i}", quality) for i, quality in enumerate((40, 80))]
    specs = [f"{sha}={run}" for sha, run in zip(shas, runs)]

    result = series.run_series(specs, repo=repo, report_out=str(tmp_path / "out"))
    report_path = Path(result["report_path"])
    report = report_path.read_text(encoding="utf-8")
    assert "## 逐数据集关键指标" not in report
    assert not (report_path.parent / layout.FIG_SERIES_PER_DATASET).exists()
    # 融合层与差值的柱状图仍然有（单数据集同样需要看图）
    assert (report_path.parent / layout.FIG_SERIES_FUSED).is_file()
    assert (report_path.parent / layout.FIG_SERIES_DELTA).is_file()
    _assert_links_alive(report_path)


def test_series_dataset_flag_selects_one(tmp_path: Path, git_repo) -> None:
    repo, shas = git_repo
    datasets = (str(SAMPLE_DATASET_PATH), str(builtin_office_dataset_path()))
    runs = [_run(tmp_path, f"one{i}", quality, datasets=datasets) for i, quality in enumerate((40, 80))]
    specs = [f"{sha}={run}" for sha, run in zip(shas, runs)]

    result = series.run_series(
        specs, repo=repo, dataset="sample_dataset", report_out=str(tmp_path / "out")
    )

    assert result["dataset_keys"] == ["sample_dataset"]
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "## 逐数据集 accuracy" not in report
    assert f"]({layout.FIG_SERIES_ACCURACY})" in report


def test_series_rejects_mismatched_dataset_sets(tmp_path: Path, git_repo) -> None:
    repo, shas = git_repo
    both = _run(tmp_path, "both", 40, datasets=(str(SAMPLE_DATASET_PATH), str(builtin_office_dataset_path())))
    only = _run(tmp_path, "only", 60)
    specs = [f"{shas[0]}={both}", f"{shas[1]}={only}"]

    with pytest.raises(SystemExit) as excinfo:
        series.run_series(specs, repo=repo, report_out=str(tmp_path / "out"))

    assert "数据集不一致" in str(excinfo.value)
    assert not (tmp_path / "out").exists(), "入参不一致不该留下半个 series_* 目录"


def test_cli_series_requires_two_lines(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["series", "--line", "abc1234=/tmp", "--report-out", str(tmp_path / "out")])
    assert "至少需要 2 条" in str(excinfo.value)
    assert not (tmp_path / "out").exists()


def test_series_reuses_group_dir_from_task_manifest(tmp_path: Path, git_repo) -> None:
    """产物来自带 groups.json 的任务：副本目录直接用组的 ASCII 目录名。

    回归：中文线名曾让副本目录退化成 ``01-line1``，即使源任务里明明白白有
    ``baseline`` / ``index-opt``。
    """
    repo, shas = git_repo
    datasets = (str(SAMPLE_DATASET_PATH),)
    runs = [_run(tmp_path, f"grp{i}", quality, datasets=datasets) for i, quality in enumerate((40, 80))]

    # 造两个"有清单"的任务：各自把数据集产物放进 baseline / index-opt 组目录
    task_dirs = []
    for index, (run, group_dir) in enumerate(zip(runs, ("baseline", "index-opt"))):
        task = tmp_path / f"compare_task_{index}"
        group_path = task / group_dir
        group_path.mkdir(parents=True)
        shutil.copytree(Path(run) / layout.DIR_RUNS, group_path / layout.DIR_RUNS)
        layout.write_groups_manifest(
            str(task),
            [
                {"role": layout.ROLE_BASELINE, "label": "基线", "dir": "baseline"},
                {"role": layout.ROLE_CANDIDATE, "label": "索引优化", "dir": "index-opt"},
            ],
        )
        task_dirs.append(str(group_path))

    result = series.run_series(
        [f"基线={task_dirs[0]}", f"索引优化={task_dirs[1]}"],
        repo=repo,
        report_out=str(tmp_path / "out"),
    )
    series_dir = Path(result["series_dir"])
    dirs = sorted(p.name for p in (series_dir / layout.DIR_RUNS).iterdir())
    assert dirs == ["01-baseline", "02-index-opt"]


def test_series_writes_per_line_fused_report(tmp_path: Path, git_repo) -> None:
    """每条线都有一份自己的融合报告（与 evaluate 任务级报告同构，不是只有数据集报告）。

    用户视角：多线报告回答"哪一步涨了多少"，但"这一线到底哪里不行"（评级 / 门禁 /
    逐数据集表 / 最难题型）要看该线自己的任务级报告。缺了它，series 树里每条线只剩
    逐数据集报告，只能手工拼。
    """
    from membench.evaluate import fuse_metrics

    repo, shas = git_repo
    runs = [
        _run(tmp_path, "exp0", 40),
        _run(tmp_path, "exp1", 80),
    ]
    specs = [f"{sha}={run}" for sha, run in zip(shas, runs)]
    result = series.run_series(specs, repo=repo, report_out=str(tmp_path / "out"))

    assert set(result["line_reports"]) == {0, 1}
    for entry in result["lines"]:
        line_dir = Path(entry["copy_dir"])
        reports = line_dir / layout.DIR_REPORTS
        # 与 evaluate 的任务级报告同构：md + metrics.json + 概览图 + summary
        for name in (
            layout.FILE_REPORT_MD,
            layout.FILE_METRICS,
            layout.FIG_DATASET_OVERVIEW,
            layout.FILE_SUMMARY,
        ):
            assert (reports / name).is_file(), f"缺少 reports/{name}"
        assert entry["report_path"] == str(reports / layout.FILE_REPORT_MD)

        text = reports.joinpath(layout.FILE_REPORT_MD).read_text(encoding="utf-8")
        # 抬头点出这是哪一线，并链回跨线主报告
        assert f"「{entry['name']}」这一线的融合报告" in text
        assert "../../../reports/report.md" in text
        # 该线自己的融合指标表：数值与 series 报告用的同口径融合值一致
        expected = fuse_metrics(list(entry["datasets"].values()))
        accuracy = expected["accuracy"]["value"]
        assert f"{accuracy * 100:.1f}%" in text
        # 数据集报告链接指向副本（不是源产物），且不是死链
        assert "../sample_dataset/report/report.md" in text
        _assert_links_alive(reports / layout.FILE_REPORT_MD)

    # 主报告的「来源与产物」表给每线一个入口
    main = Path(result["report_path"]).read_text(encoding="utf-8")
    for line in result["lines"]:
        assert "该线融合报告" in main
        rel = os.path.relpath(line["report_path"], Path(result["report_path"]).parent)
        assert rel.replace(os.sep, "/") in main


def test_series_reuses_single_group_task_name(tmp_path: Path, git_repo) -> None:
    """单组任务（只有 ``task.json``、没有 ``groups.json``）也复用任务名做副本目录。

    回归：此前只认 ``groups.json``，单组任务因此全部退化成 ``01-<commit8>``——明明
    用户已经在 ``task init --name`` 给了 ASCII 名字（``perf-base``），目录却只剩哈希。
    """
    repo, shas = git_repo
    runs = [_run(tmp_path, f"solo{i}", quality) for i, quality in enumerate((40, 80))]

    tasks = []
    for index, (run, name, display) in enumerate(
        zip(runs, ("perf-base", "perf-fast"), ("基线跑", "降延迟"))
    ):
        task = Path(layout.init_task_dir(str(tmp_path / "tasks"), name, "t", groups=1, ts=f"2026010{index + 1}-120000", display_name=display))
        # task init 已建好 runs/：把数据集子树逐个并进去（copytree 不能覆盖已有目录）
        for dataset_dir in (Path(run) / layout.DIR_RUNS).iterdir():
            shutil.copytree(dataset_dir, task / layout.DIR_RUNS / dataset_dir.name)
        tasks.append(str(task))

    result = series.run_series(
        [f"{sha}={target}" for sha, target in zip(shas, tasks)],
        repo=repo,
        report_out=str(tmp_path / "out"),
    )
    series_dir = Path(result["series_dir"])
    dirs = sorted(p.name for p in (series_dir / layout.DIR_RUNS).iterdir())
    assert dirs == ["01-perf-base", "02-perf-fast"]
    # 线名沿用任务显示名（用户给的），不再从 commit 抽取；第一条按约定叫「基线」
    assert [line["name"] for line in result["lines"]] == ["基线", "降延迟"]
    assert [line["name_source"] for line in result["lines"]] == ["基线", "任务清单"]


def test_series_line_dir_falls_back_when_manifest_names_are_not_ascii() -> None:
    """清单里的名字是纯中文（老产物）时顺延下一级，不产出 ``01-default``。"""
    from membench import layout as layout_mod

    series_dir = "/tmp/series-demo"
    # 老产物：组目录名 / 任务名是中文 → slug 退化成 default，顺延到线名
    assert layout_mod.series_line_dir(series_dir, 0, "基线", None, "中文目录") .endswith(
        "01-line1"
    )
    assert layout_mod.series_line_dir(series_dir, 1, "index-opt", None, "索引优化").endswith(
        "02-index-opt"
    )
    # 线名也是中文 → 退 commit8；没有 commit 才退 line<i>
    assert layout_mod.series_line_dir(series_dir, 2, "基线", "abcdef1234", "索引优化").endswith(
        "03-abcdef12"
    )
    assert layout_mod.series_line_dir(series_dir, 3, "基线", None, "索引优化").endswith(
        "04-line4"
    )


def _big_category_dataset(path: Path, *, per_category: int = 12) -> Path:
    """单一题型、题量够进榜（≥ CATEGORY_MIN_N）的数据集：用来验证「提升最多」榜。"""
    from membench.report import CATEGORY_MIN_N

    assert per_category >= CATEGORY_MIN_N
    questions = [
        {
            "qid": f"q{i}",
            "question": f"第 {i} 题：项目截止日是哪天？",
            "gold_answer": "周五",
            "category": "single_hop",
            "evidence": ["项目截止日是周五"],
        }
        for i in range(per_category)
    ]
    record = {
        "dialogue_id": "big",
        "sessions": [
            [{"role": "user", "content": "记录：项目截止日是周五。", "timestamp": None}]
        ],
        "questions": questions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def test_series_report_has_top_improved_categories(tmp_path: Path, git_repo) -> None:
    """series 报告要有「提升最多的题型」榜与逐数据集题型表——与 compare 同一份口径。

    用户视角：series 只是"多几条线"，题型 diff / 提升 TOP 这类结论不该只有 compare 有。
    判定（两侧题量 ≥ CATEGORY_MIN_N、Δ > 0、按 Δ 降序）由
    :func:`membench.evaluate_compare.category_diff_rows` 与 series 共用。
    """
    from membench.evaluate_compare import top_improved_categories, GroupDatasetResult

    repo, shas = git_repo
    dataset = _big_category_dataset(tmp_path / "ds" / "big.jsonl")
    weak = _run(tmp_path, "big-weak", 40, datasets=(str(dataset),))
    strong = _run(tmp_path, "big-strong", 80, datasets=(str(dataset),))
    result = series.run_series(
        [f"{shas[0]}={weak}", f"{shas[1]}={strong}"],
        repo=repo,
        report_out=str(tmp_path / "out"),
    )
    report = Path(result["report_path"]).read_text(encoding="utf-8")

    assert "## 题型 diff（vs 基线）" in report
    top_title = next(
        line
        for line in report.splitlines()
        if line.startswith("### 提升最多的题型")
    )
    top_rows = _section_rows(report, top_title)
    assert top_rows, "题量够、Δ>0 时必须给出提升榜"
    assert top_rows[0][0] == result["lines"][1]["name"]
    assert top_rows[0][2].startswith("单跳")  # category_display('single_hop')
    # 与 compare 的同一份判定对齐：同一对产物的提升榜第一条一致
    baseline_metrics = json.loads(
        (Path(result["lines"][0]["datasets"]["big"].run_dir) / "report" / "metrics.json")
        .read_text(encoding="utf-8")
    )
    candidate_metrics = json.loads(
        (Path(result["lines"][1]["datasets"]["big"].run_dir) / "report" / "metrics.json")
        .read_text(encoding="utf-8")
    )
    from membench.evaluate_compare import DatasetSpec as _Spec

    spec = _Spec(key="big", name="big", path=str(dataset), source="user-specified")
    compare_top = top_improved_categories(
        [
            GroupDatasetResult(
                spec=spec,
                baseline=baseline_metrics,
                candidate=candidate_metrics,
                baseline_dataset_dir="",
                candidate_dataset_dir="",
            )
        ]
    )
    assert compare_top and f"{compare_top[0]['delta'] * 100:+.1f}pt" == top_rows[0][6]

    # 逐数据集题型表：列出该数据集的题型与各线准确率（括号内是 vs 基线的差值）
    assert "### 逐数据集题型准确率（vs 基线）" in report
    assert "| 单跳（single_hop） | 12 |" in report
    assert "(+" in report

    # TOP N 柱状图：只画榜内题型，且报告引用它
    fig = Path(result["report_path"]).parent / layout.FIG_SERIES_CATEGORY_TOP
    assert fig.is_file()
    assert layout.FIG_SERIES_CATEGORY_TOP in report
    _assert_links_alive(Path(result["report_path"]))


def test_series_no_category_chart_without_improvements(tmp_path: Path, git_repo) -> None:
    """没有达标的提升项时不画图、也不写引用（不留死链，与其它图同一规矩）。"""
    repo, shas = git_repo
    runs = [_run(tmp_path, f"flat{i}", 80) for i in range(2)]
    result = series.run_series(
        [f"{sha}={run}" for sha, run in zip(shas, runs)],
        repo=repo,
        report_out=str(tmp_path / "out"),
    )
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "没有满足条件的提升项" in report
    assert layout.FIG_SERIES_CATEGORY_TOP not in report
    assert not (
        Path(result["report_path"]).parent / layout.FIG_SERIES_CATEGORY_TOP
    ).is_file()


def test_series_report_has_efficiency_table(tmp_path: Path, git_repo) -> None:
    """效率成本逐数据集对照（与 evaluate-compare 的「效率成本 diff」同口径）。"""
    repo, shas = git_repo
    weak = _run(tmp_path, "eff-weak", 40)
    strong = _run(tmp_path, "eff-strong", 80)
    result = series.run_series(
        [f"{shas[0]}={weak}", f"{shas[1]}={strong}"],
        repo=repo,
        report_out=str(tmp_path / "out"),
    )
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "## 逐数据集效率成本（vs 基线）" in report
    rows = _section_rows(report, "#### sample_dataset · 效率成本")
    metrics = [row[0] for row in rows]
    assert any("延迟 P95" in m for m in metrics)
    assert any("平均输入 Token" in m for m in metrics)
