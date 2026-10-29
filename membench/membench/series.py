"""series：多线对比（N 个版本按时间顺序逐步对比，线名从 commit 信息提取）。

用户做了多个优化点、每个优化点一个 commit 时，A/B 只能回答"最后一版比基线好多少"，
回答不了"每一步提升多少"。本模块按 ``--line`` 的给出顺序（即时间顺序）读 N 条线的
同一批产物，产出：

1. 结论表：每线的 accuracy / 证据召回率 / ↑vs 基线 / ↑vs 上一步 / 门禁结论；
2. 逐指标表：行=指标、列=各线（另给每线 vs 基线的差值，始终保留基线一列）；
3. 逐数据集 accuracy 表（多数据集时）：单元格为值 + (vs 基线)；
4. 门禁：每条线 vs 基线，复用 compare 的阈值与
   :func:`membench.compare._eval_gates`（accuracy 降 >0.02 / stale 升 >0.02 /
   latency_p95 +30%）；任一 FAIL → 命令退出码 1；
5. 图表：accuracy 随线序的走势（多数据集时逐数据集多条折线 + 融合值一条粗线）。

线名规则（:func:`line_name`）：

- 左侧是 7~40 位十六进制 → 视为 commit，按优先级命名：
  1. **LLM 命名**（配置 :class:`LLMNaming` 时）：从 commit 主题 + 正文、
     ``git show --stat`` 改动摘要与截断后的 patch 抽一个简短中文名；输出必须含
     中文、长度 ≤ :data:`LINE_NAME_MAX` 个字且不含标点，否则视为失败；
  2. **代码抽取**（:func:`extract_name_from_subject`）：去掉 conventional-commit
     前缀（``feat(scope):`` / ``fix：`` 等，中英文冒号都认）、剥首尾标点、折叠
     空白，最后截断到 :data:`LINE_NAME_MAX` 个字；
  LLM 仅在 ``--llm-base-url`` 与 ``--llm-model`` 同时给出（且未开
  ``--no-llm-naming``）时启用；没有配置直接走代码抽取（不告警），调用失败或输出
  不合规时告警一次后降级代码抽取，**都不中断**；
- 第一条（``index == 0``）默认叫「基线」，不走任何命名、也不调 LLM；
- 左侧不是 commit 时就是显式线名，原样使用（永远优先，不调 LLM，且不参与任何
  自动清洗）；
- git 取不到（非 git 目录 / commit 不存在 / git 不可用）或降级后仍不含中文 →
  线名回退短 sha 并在 stderr 告警一次，**不中断**（评测已经跑完，命名失败不该让
  对比出不了报告）；
- 重名时追加 ``（<sha8>）`` 区分（没有 sha 的显式名退 ``（第N线）``）；
- 报告「来源与产物」表的「命名来源」列标出 ``llm`` / ``code`` / ``显式名`` /
  ``基线`` / ``回退短 sha``，便于追溯每条线名是怎么来的。

产物写到 ``out_dir``（完整实验目录，见 :mod:`membench.layout`）::

    <out_dir>/reports/{report.md, fig_series_accuracy.png}
    <out_dir>/runs/<线序>-<slug>/<dataset>/   # 各线的数据集树副本（报告内相对链接）

**只读**：各线的产物目录（实验目录或 ``runs/<dataset>`` 数据集目录）原样不动；本模块
只读 ``<数据集目录>/report/metrics.json`` 并复制副本，不改 A/B（compare）、
evaluate、fuse 的既有语义。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Union

from membench import layout, runtime
from membench.compare import _comparability_gate, _eval_gates, check_comparability
from membench.evaluate import (
    FUSED_METRICS,
    FUSION_CHOICES,
    FUSION_WEIGHTED,
    FUSION_WEIGHT_CAP,
    DatasetResult,
    DatasetSpec,
    _capabilities_from_metrics,
    fuse_metrics,
    is_office_slot_dataset,
)
from membench.judges.llm import chat_completion
from membench.paths import output_root
from membench.ratings import RATE_METRICS, format_metric_value
from membench.report import _md_table, dim_label, fig_series_accuracy, metric_label

#: 第一条（无显式名）线的报告名。
BASELINE_NAME = "基线"

#: 线名的最大字符数（超出截断；重名时的 ``（sha8）`` 后缀不计入）。
LINE_NAME_MAX = 24

#: 逐指标表的行：质量 + 诊断（两种模式都展示），再接效率成本（仅单数据集模式——
#: 多数据集融合不覆盖延迟 / token 这类依赖部署环境的指标，见
#: :func:`membench.evaluate.fuse_metrics`）。
SERIES_METRICS: tuple[str, ...] = FUSED_METRICS + (
    "latency_p50_ms",
    "latency_p95_ms",
    "avg_input_tokens",
    "avg_output_tokens",
    "avg_ingest_latency_ms",
    "avg_llm_calls",
)

#: 门禁阈值：与 compare 的默认阈值口径一致（accuracy 允许下降 / stale 允许上升 /
#: latency_p95 相对上升的告警比例）。series 不另设开关，避免与 compare 两套阈值。
ACC_THRESHOLD = 0.02
STALE_THRESHOLD = 0.02
LATENCY_WARN = 0.30

#: 视为 commit 的左侧写法：7~40 位十六进制（短 sha / 全 sha）。
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")

#: conventional-commit 的 type 词表。只认常见 type：中文标题里的"优化：xxx"不是
#: conventional 前缀，不能被当成前缀删掉（那正是要展示的优化点）。
_CONVENTIONAL_RE = re.compile(
    r"^(?:feat|fix|refactor|perf|docs|test|tests|chore|build|ci|style|revert)"
    r"(?:\([^)]*\))?!?\s*[:：]\s*",
    re.IGNORECASE,
)

#: 首尾要剥掉的标点 / 空白（中英文标点都认，含全角括号与常见符号）。
_EDGE_PUNCT_RE = re.compile(
    r"^[\s\-–—:：,，.。;；!！?？、·…()（）\[\]【】「」『』"
    r"\"'“”‘’<>《》|/\\~`#*_=+]+"
    r"|[\s\-–—:：,，.。;；!！?？、·…()（）\[\]【】「」『』"
    r"\"'“”‘’<>《》|/\\~`#*_=+]+$"
)

_WHITESPACE_RE = re.compile(r"\s+")

#: 中文字符（线名必须含中文才有"一眼看出优化点"的价值，否则回退短 sha）。
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

#: 线名内部不允许出现的标点（prompt 要求"不要标点"；首尾标点先剥掉再查内部，
#: 剩下的标点说明模型没按要求输出，宁可用回代码抽取的 commit 主题）。
_INNER_PUNCT_RE = re.compile(
    r"[-–—:：,，.。;；!！?？、·…()（）\[\]【】「」『』"
    r"\"'“”‘’<>《》|/\\~`#*_=+]"
)

#: LLM 线名请求的 system prompt：只输出一个短语（输出仍由
#: :func:`clean_llm_name` 严格校验，不信任模型自觉）。
_LLM_NAME_PROMPT = (
    "你是一个 git 提交命名助手。根据 commit 主题、正文与改动摘要，用不超过 16 个"
    "汉字概括这次优化的可读名称；不要标点，不要英文，不要 feat / fix 这类前缀，"
    "只输出这个名称本身，不要任何解释。"
)

#: LLM 命名失败 / 输出不合规时的统一告警（每次 series 运行最多一条）。
_LLM_NAME_FALLBACK_WARN = "[membench] ⚠️ LLM 命名失败，回退代码抽取"

#: 进 prompt 前的截断长度（字符）：正文与 stat 只留摘要，patch 用
#: ``--unified=0`` 只留改动行——大 commit 的完整 patch 会把 prompt 撑到几万 token。
_BODY_MAX = 1000
_STAT_MAX = 1500
_PATCH_MAX = 2000


def is_commit_ref(text: str) -> bool:
    """左侧写法是不是 commit（7~40 位十六进制）；否则视为显式线名。"""
    return bool(_COMMIT_RE.match(str(text or "").strip()))


def extract_name_from_subject(subject: str) -> str:
    """从 commit subject 提取报告用短名。

    依次：去 conventional-commit 前缀 → 折叠空白 → 剥首尾标点 → 截断到
    :data:`LINE_NAME_MAX` 个字（截断后再剥一次尾部标点，免得名字以标点收尾）。
    """
    text = _CONVENTIONAL_RE.sub("", str(subject or "").strip())
    text = _WHITESPACE_RE.sub(" ", text).strip()
    text = _EDGE_PUNCT_RE.sub("", text).strip()
    if len(text) > LINE_NAME_MAX:
        text = _EDGE_PUNCT_RE.sub("", text[:LINE_NAME_MAX]).strip()
    return text


def _git_output(
    repo: str, args: Sequence[str], timeout: float = 10
) -> Union[str, None]:
    """跑一条 ``git -C <repo> ...`` 并返回 stdout；任何失败（含 git 不可用）返回 None。

    命名只是报告展示，git 失败不该让一次已经跑完的对比失败——所有调用方都按
    "取不到" 处理，不抛异常。
    """
    try:
        proc = subprocess.run(
            ["git", "-C", repo, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def git_subject(repo: str, commit: str) -> Union[str, None]:
    """``git -C <repo> log -1 --format=%s <commit>`` 的 subject；取不到返回 None。

    非 git 目录 / commit 不存在 / git 不可用都走 None（调用方回退短 sha 并告警），
    不抛异常：命名失败只影响展示名，不该让一次已经跑完的对比失败。
    """
    output = _git_output(repo, ["log", "-1", "--format=%s", str(commit)])
    if output is None:
        return None
    subject = output.strip()
    return subject or None


def _emit(warn: Union[Callable[[str], None], None], message: str) -> None:
    """告警出口：默认写 stderr；调用方可注入去重后的 warn（"告警一次"）。"""
    if warn is None:
        print(message, file=sys.stderr)
    else:
        warn(message)


@dataclass(frozen=True)
class LLMNaming:
    """series 线名的 LLM 配置；``base_url`` 与 ``model`` 都非空才启用。

    没有配置（``enabled`` 为 False）时直接走代码抽取，不告警也不发请求。
    """

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout: float = 90.0

    @property
    def enabled(self) -> bool:
        """``base_url`` 与 ``model`` 都给才尝试 LLM（缺一个就静默降级）。"""
        return bool(self.base_url.strip() and self.model.strip())


def _clip(text: str, limit: int) -> str:
    """进 prompt 前截断超长文本，并标注已截断（避免大 commit 撑爆 token）。"""
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "…（已截断）"


def _llm_name_prompt(subject: str, body: str, stat: str, patch: str) -> str:
    """线名请求的 user 段：主题 / 正文 / 改动摘要 / 截断后的改动片段。"""
    return (
        f"【主题】{subject}\n"
        f"【正文】{_clip(body, _BODY_MAX) or '（无）'}\n"
        f"【改动摘要】\n{_clip(stat, _STAT_MAX) or '（无）'}\n"
        f"【改动片段】\n{_clip(patch, _PATCH_MAX) or '（无）'}"
    )


def _request_llm_name(
    repo: str, commit: str, subject: str, config: LLMNaming
) -> str:
    """对一条 commit 调一次 chat completion，返回模型原始输出（未校验）。

    改动片段用 ``--unified=0`` 只留改动行，再按 :data:`_PATCH_MAX` 截断；git 取
    不到正文 / stat / patch 时用「（无）」占位，仍给模型主题这一条线索。
    """
    body = (_git_output(repo, ["log", "-1", "--format=%b", str(commit)]) or "").strip()
    stat = (
        _git_output(repo, ["show", "--stat", "--format=", "-1", str(commit)]) or ""
    ).strip()
    patch = (
        _git_output(repo, ["show", "--unified=0", "--format=", "-1", str(commit)]) or ""
    ).strip()
    return chat_completion(
        config.base_url,
        config.api_key,
        config.model,
        config.timeout,
        [
            {"role": "system", "content": _LLM_NAME_PROMPT},
            {"role": "user", "content": _llm_name_prompt(subject, body, stat, patch)},
        ],
        temperature=0,
    )


def clean_llm_name(text: str) -> Union[str, None]:
    """清洗并校验 LLM 输出的线名；不合规返回 ``None``（调用方回退代码抽取）。

    依次：折叠空白 → 剥首尾空白 / 引号 / 标点；随后必须**含中文**、长度
    ≤ :data:`LINE_NAME_MAX` 个字、内部不含标点（prompt 已要求"不要标点"，剩下
    标点说明模型没按要求输出，宁可回代码抽取的 commit 主题）。
    """
    name = _WHITESPACE_RE.sub(" ", str(text or "")).strip()
    name = _EDGE_PUNCT_RE.sub("", name).strip()
    if not name or len(name) > LINE_NAME_MAX:
        return None
    if not _CJK_RE.search(name):
        return None
    if _INNER_PUNCT_RE.search(name):
        return None
    return name


@dataclass
class _NamingState:
    """一次 series 运行的 LLM 命名状态（配置 + "失败只告警一次"的标记）。

    ``disabled``：首次失败（异常或输出不合规）后整体短路——同一模型、同一 prompt
    下逐线重试只会让每条线各等一次超时，换不来成功。
    """

    repo: str
    config: Union[LLMNaming, None]
    warn: Union[Callable[[str], None], None]
    llm_warned: bool = False
    disabled: bool = False

    def llm_name(self, commit: str, subject: str) -> Union[str, None]:
        """LLM 线名；未配置 / 已短路 / 调用失败 / 输出不合规都返回 None（由调用方降级）。"""
        if self.config is None or not self.config.enabled or self.disabled:
            return None
        try:
            raw = _request_llm_name(self.repo, commit, subject, self.config)
        except Exception:  # noqa: BLE001 - 命名失败不能中断已跑完的对比
            raw = None
        name = clean_llm_name(raw or "")
        if name:
            return name
        self.disabled = True
        if not self.llm_warned:
            self.llm_warned = True
            _emit(self.warn, _LLM_NAME_FALLBACK_WARN)
        return None


def _resolve_name(
    commit_or_name: str,
    repo: str,
    index: int,
    explicit_first_name: Union[str, None],
    warn: Union[Callable[[str], None], None],
    naming: Union[_NamingState, None] = None,
) -> tuple[str, str]:
    """``line_name`` 的实现：返回 ``(线名, 命名来源)``（来源进报告的可追溯表）。

    命名优先级：基线 / 显式名（都不调 LLM）→ LLM（``naming`` 启用且调用成功）→
    代码抽取（commit subject）→ 短 sha。``naming`` 为 None 时不尝试 LLM。
    """
    text = str(commit_or_name or "").strip()
    if not text:
        raise SystemExit("--line 的左侧（commit 或线名）不能为空")
    if not is_commit_ref(text):
        return text, "显式名"
    if index == 0:
        return explicit_first_name or BASELINE_NAME, "基线"
    subject = git_subject(repo, text)
    if subject is None:
        sha8 = text[:8].lower()
        _emit(
            warn,
            f"[membench] ⚠️ 取不到 commit subject（非 git 目录 / commit 不存在）："
            f"{text} 的线名回退为短 sha {sha8}",
        )
        return sha8, "回退短 sha"
    if naming is not None:
        name = naming.llm_name(text, subject)
        if name:
            return name, "llm"
    name = extract_name_from_subject(subject)
    if name and _CJK_RE.search(name):
        return name, "code"
    sha8 = text[:8].lower()
    _emit(
        warn,
        f"[membench] ⚠️ commit subject 不含中文：{text} 的线名回退为短 sha {sha8}",
    )
    return sha8, "回退短 sha"


def line_name(
    commit_or_name: str,
    repo: Union[str, None] = None,
    index: int = 0,
    explicit_first_name: Union[str, None] = None,
    *,
    warn: Union[Callable[[str], None], None] = None,
) -> str:
    """一条线的报告名（命名规则见模块 docstring）。

    Args:
        commit_or_name: ``--line`` 左侧原文：7~40 位十六进制视为 commit，否则是显式线名。
        repo: git 仓库路径（缺省当前目录），只对 commit 生效。
        index: 线序（从 0 起）；``0`` 且无显式名时叫「基线」。
        explicit_first_name: 显式指定第一线的名字（缺省 :data:`BASELINE_NAME`）；
            只在左侧是 commit 时生效——显式线名永远以左侧为准。
        warn: 告警回调（缺省写 stderr）；命名回退时调用一次。

    Returns:
        报告里用的线名（≤ :data:`LINE_NAME_MAX` 个字，重名后缀另计）。
    """
    return _resolve_name(
        commit_or_name, repo or os.getcwd(), index, explicit_first_name, warn
    )[0]


def _dedupe_names(names: list[str], commits: list[Union[str, None]]) -> list[str]:
    """重名时追加 ``（<sha8>）``（无 sha 的线退 ``（第N线）``），保证线名唯一。

    线名进表格与图例，两条线同名等于报告里分不清谁是谁。后缀加在截断之后，
    因此带后缀的线名可能略超 :data:`LINE_NAME_MAX` 个字——可区分优先。
    """
    counts = Counter(names)
    used: set[str] = set()
    out: list[str] = []
    for index, name in enumerate(names):
        final = name
        if counts[name] > 1:
            sha = commits[index]
            suffix = str(sha)[:8].lower() if sha else f"第{index + 1}线"
            final = f"{name}（{suffix}）"
        if final in used:
            raise SystemExit(
                f"线名重复且无法区分: {final!r}；请把 --line 左侧改成显式线名"
            )
        used.add(final)
        out.append(final)
    return out


@dataclass
class SeriesLine:
    """多线对比里的一条线。

    Attributes:
        index: 线序（0 起，顺序即时间顺序）。
        ref: ``--line`` 左侧原文（commit 或显式线名）。
        name: 报告里的线名。
        name_source: 命名来源（``基线`` / ``llm`` / ``code`` / ``显式名`` / ``回退短 sha``）。
        commit: commit sha（显式线名时为 None）。
        target: ``--line`` 右侧产物目录（绝对路径）。
        datasets: ``{数据集目录名: 该集结果}``（跨线键集合必须一致）。
        copy_dir: 落盘后的副本目录（:func:`run_series` 填）。
    """

    index: int
    ref: str
    name: str
    name_source: str
    commit: Union[str, None]
    target: str
    datasets: dict[str, DatasetResult] = field(default_factory=dict)
    copy_dir: str = ""

    @property
    def sha8(self) -> str:
        """commit 短 sha（8 位）；显式线名为空串。"""
        return str(self.commit)[:8].lower() if self.commit else ""


def _parse_spec(raw: str) -> tuple[str, str]:
    """``<commit-or-name>=<DIR>`` → ``(左侧, 右侧)``；格式不对直接报错。"""
    text = str(raw)
    ref, sep, target = text.partition("=")
    ref, target = ref.strip(), target.strip()
    if not sep or not ref or not target:
        raise SystemExit(
            f"--line 需为 <commit-or-name>=<产物目录> 形式，当前为 {text!r}"
        )
    return ref, target


def _line_datasets(target: str, dataset: Union[str, None]) -> dict[str, str]:
    """一条线的产物目录 -> ``{数据集目录名: 绝对路径}``。

    实验目录取 ``runs/`` 下全部数据集目录；数据集目录入参只取它自己。``dataset``
    给定时按 slug 只取一个，找不到直接报错并列出可选值（认错数据集会把两条线比成
    两个数据集，分数差不可归因）。两者都不是时交给
    :func:`membench.layout.resolve_dataset_dir` 给出统一的可操作报错（对话目录 /
    旧版产物 / 未融合分片 / 没有产物各有文案）。
    """
    if not os.path.isdir(target):
        raise SystemExit(f"产物目录不存在: {target}")
    abspath = os.path.abspath(target)
    found = layout.dataset_dirs(abspath)
    if not found:
        single = layout.resolve_dataset_dir(abspath, dataset)
        key = os.path.basename(single)
        if dataset is not None and key != layout.slug(dataset):
            raise SystemExit(
                f"--dataset {dataset!r} 与产物数据集 {key!r} 不一致: {abspath}"
            )
        return {key: single}
    if dataset is None:
        return {os.path.basename(path): path for path in found}
    wanted = layout.slug(dataset)
    hit = [path for path in found if os.path.basename(path) == wanted]
    if not hit:
        names = "、".join(os.path.basename(path) for path in found)
        raise SystemExit(
            f"--dataset {dataset!r} 不在 {abspath} 的数据集里（可选: {names}）"
        )
    return {os.path.basename(hit[0]): hit[0]}


def _load_dataset_result(dataset_dir: str) -> DatasetResult:
    """读数据集级 ``report/metrics.json``，组装成融合 / 门禁能直接用的结果。

    办公域槽位按**数据集身份**判定（内置 office / 用户测试集），与 evaluate 的
    ``--from-runs`` 汇总口径一致——多线对比里它同样不参与基准融合。
    """
    path = os.path.join(dataset_dir, layout.DIR_REPORT, layout.FILE_METRICS)
    with open(path, encoding="utf-8") as handle:
        metrics = json.load(handle)
    dataset_path = str(metrics.get("dataset_path") or "")
    key = os.path.basename(os.path.normpath(dataset_dir))
    spec = DatasetSpec(
        key=key,
        name=str(metrics.get("dataset") or key),
        path=dataset_path,
        source=str(metrics.get("dataset_source") or "user-specified"),
        is_office_slot=is_office_slot_dataset(dataset_path, dataset_dir),
    )
    return DatasetResult(
        spec=spec,
        run_dir=os.path.abspath(dataset_dir),
        metrics=metrics,
        capabilities=_capabilities_from_metrics(metrics, dataset_path),
    )


def _check_same_datasets(lines: list[SeriesLine]) -> None:
    """每条线必须在同一批数据集上评测；缺 / 多都直接报错。

    少一个数据集时，该线的"融合值"就是另一批题的平均，跨线差值不再可比——
    宁可拒绝，也不能给出一个看起来合理的错误结论。
    """
    reference = lines[0]
    for line in lines[1:]:
        missing = sorted(set(reference.datasets) - set(line.datasets))
        extra = sorted(set(line.datasets) - set(reference.datasets))
        if not missing and not extra:
            continue
        parts = []
        if missing:
            parts.append("缺 " + "、".join(missing))
        if extra:
            parts.append("多 " + "、".join(extra))
        raise SystemExit(
            f"第 {line.index + 1} 线与基线（第 1 线）的数据集不一致（{'；'.join(parts)}）："
            "多线对比要求每条线在同一批数据集上评测——补跑缺的数据集，"
            "或用 --dataset 指定同一个数据集"
        )


def load_lines(
    specs: Sequence[str],
    repo: Union[str, None] = None,
    dataset: Union[str, None] = None,
    *,
    warn: Union[Callable[[str], None], None] = None,
    llm: Union[LLMNaming, None] = None,
) -> list[SeriesLine]:
    """解析并加载全部线（顺序即时间顺序，第一条是基线）。

    Args:
        specs: ``--line`` 原始值列表（``<commit-or-name>=<产物目录>``）。
        repo: git 仓库路径（缺省当前目录），用于取 commit subject。
        dataset: 只比某个数据集（多数据集实验目录时用）；缺省比全部数据集。
        warn: 命名回退的告警回调（缺省写 stderr；调用方传去重函数即"告警一次"）。
        llm: 线名的 LLM 配置；缺省或 :attr:`LLMNaming.enabled` 为 False 时只用
            代码抽取（不告警、不发请求）。

    Returns:
        :class:`SeriesLine` 列表（``index`` 从 0 起、``datasets`` 已加载）。

    Raises:
        SystemExit: ``--line`` 格式不对 / 目录不存在 / 各线数据集集合不一致 /
            数据集产物不完整。
    """
    if not specs:
        raise SystemExit(
            "series 需要至少两条 --line <commit-or-name>=<产物目录>（顺序即时间顺序）"
        )
    repo_path = os.path.abspath(repo or os.getcwd())
    parsed = [_parse_spec(raw) for raw in specs]
    naming = _NamingState(repo_path, llm, warn) if llm and llm.enabled else None
    resolved = [
        _resolve_name(ref, repo_path, index, None, warn, naming)
        for index, (ref, _target) in enumerate(parsed)
    ]
    names = _dedupe_names(
        [name for name, _source in resolved],
        [ref if is_commit_ref(ref) else None for ref, _target in parsed],
    )
    lines: list[SeriesLine] = []
    for index, ((ref, target), (name, source)) in enumerate(zip(parsed, resolved)):
        line = SeriesLine(
            index=index,
            ref=ref,
            name=names[index],
            name_source=source,
            commit=ref if is_commit_ref(ref) else None,
            target=os.path.abspath(target),
            datasets={
                key: _load_dataset_result(path)
                for key, path in _line_datasets(target, dataset).items()
            },
        )
        lines.append(line)
    _check_same_datasets(lines)
    return lines


def _fused_of(line: SeriesLine, fusion: str, cap: int) -> dict[str, dict[str, Any]]:
    """一条线的融合指标（多数据集）；单数据集时也返回同结构（值 = 该集的值）。"""
    return fuse_metrics(list(line.datasets.values()), fusion=fusion, cap=cap)


def _overall_of(line: SeriesLine, fused: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """门禁 / 结论表用的汇总指标。

    单数据集直接取该集 ``overall``（不融合，避免办公域槽位被排除后整行变空）；
    多数据集取 :func:`fuse_metrics` 的融合值（办公域槽位不参与，与 evaluate 同口径）。
    """
    if len(line.datasets) == 1:
        res = next(iter(line.datasets.values()))
        return dict(res.metrics.get("overall") or {})
    return {key: item["value"] for key, item in fused.items() if item.get("value") is not None}


def _comparability(base: SeriesLine, line: SeriesLine) -> dict[str, Any]:
    """一条线 vs 基线的可比性：逐数据集各跑 ``check_comparability`` 再汇总。

    多数据集线的"能否归因"要求每个数据集都可比——某个数据集换了判分器，整条线的
    提升数字就不可归因，不能在别的数据集上平均掉。
    """
    per_dataset: dict[str, dict[str, Any]] = {}
    mismatches: list[dict[str, Any]] = []
    unverifiable: list[dict[str, Any]] = []
    reasons: list[str] = []
    for key in base.datasets:
        comp = check_comparability(
            base.datasets[key].metrics,
            line.datasets[key].metrics,
            # 多线场景里版本号正是被测变量：按 info 级比较（仍记录，不判不可比）
            downgrade=("adapter_version",),
        )
        per_dataset[key] = comp
        mismatches.extend({**item, "dataset": key} for item in comp["mismatches"])
        unverifiable.extend({**item, "dataset": key} for item in comp["unverifiable"])
        reasons.extend(f"[{key}] {reason}" for reason in comp["reasons"])
    return {
        "comparable": not any(item["severity"] == "critical" for item in mismatches),
        "datasets": per_dataset,
        "mismatches": mismatches,
        "unverifiable": unverifiable,
        "reasons": reasons,
    }


def _verdict(gates: list[dict[str, Any]]) -> str:
    """门禁列表的结论（任一 FAIL → FAIL；任一 WARN → WARN；否则 PASS）。"""
    if any(gate["status"] == "FAIL" for gate in gates):
        return "FAIL"
    if any(gate["status"] == "WARN" for gate in gates):
        return "WARN"
    return "PASS"


def _pp(diff: Any) -> str:
    """百分点差值：``+2.3pp`` / ``-1.0pp``；``None`` 返回 ``n/a``。"""
    if diff is None:
        return "n/a"
    return f"{float(diff) * 100:+.1f}pp"


def _fusion_text(fusion: str, cap: int) -> str:
    """融合口径的中文说明（封顶按本次实参，不写死 500）。"""
    if fusion == "equal":
        return "基准集内等权"
    return f"按参与题数加权，单集封顶 {cap} 题"


def _delta(metric: str, value: Any, base_value: Any) -> str:
    """一个指标相对基线的差值单元格：比率型用 pp，其余保留 3 位小数。"""
    if value is None or base_value is None:
        return "n/a"
    diff = float(value) - float(base_value)
    return _pp(diff) if metric in RATE_METRICS else f"{diff:+.3f}"


def _value_cell(metric: str, value: Any) -> str:
    """指标值单元格（比率型按百分比展示；未计算显示「未定义」）。"""
    return format_metric_value(metric, value)


def _gate_status(gates: list[dict[str, Any]], name: str) -> str:
    """取某条门禁的状态（没有该门禁时给 ``—``）。"""
    for gate in gates:
        if gate["name"] == name:
            return str(gate["status"])
    return "—"


def _worst_verdict(entries: list[dict[str, Any]]) -> str:
    """各线结论里最差的那个（报告头与退出码用）。"""
    if any(entry["verdict"] == "FAIL" for entry in entries):
        return "FAIL"
    if any(entry["verdict"] == "WARN" for entry in entries):
        return "WARN"
    return "PASS"


_GATE_NAMES = (
    "comparability",
    "accuracy_regression",
    "stale_rate_regression",
    "latency_p95_regression",
)


def _write_series_report(
    reports_dir: str,
    entries: list[dict[str, Any]],
    *,
    dataset_keys: list[str],
    fusion: str,
    fusion_cap: int,
    base_metrics: dict[str, Any],
) -> str:
    """生成 ``report.md``（中文；图片引用同目录相对文件名），返回报告路径。"""
    multi = len(dataset_keys) > 1
    metrics_list = SERIES_METRICS if not multi else FUSED_METRICS
    base = entries[0]
    worst = _worst_verdict(entries)
    icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(worst, "")

    lines: list[str] = []
    lines.append("# membench 多线对比报告")
    lines.append("")
    lines.append(f"**整体门禁结论： {worst} {icon}**")
    lines.append("")
    lines.append(f"- 线数：{len(entries)}（按时间顺序；第一条为基线）")
    lines.append(f"- 数据集：{'、'.join(dataset_keys)}（{len(dataset_keys)} 个）")
    judge = str(base_metrics.get("judge") or "n/a")
    judge_model = base_metrics.get("judge_model")
    lines.append(
        f"- 判分器：{judge}" + (f"（判分模型 {judge_model}）" if judge_model else "")
    )
    if multi:
        lines.append(
            f"- 融合口径：{_fusion_text(fusion, fusion_cap)}"
            "（办公域槽位不参与基准融合，单列在逐数据集表里）"
        )
    else:
        lines.append("- 融合口径：单数据集（不融合，直接取该数据集的指标）")
    lines.append(
        "- 门禁阈值（与 compare 同口径）：accuracy 下降 > 0.02 或 stale_rate 上升 > 0.02 "
        "判 FAIL；latency_p95 上升 > 30% 记 WARN；--require-comparable 下关键维度"
        "不一致 / 无法核验判 FAIL"
    )
    lines.append("")

    # 1. 结论表
    lines.append("## 结论表")
    lines.append("")
    lines.append(
        "<sub>↑ vs 基线 / ↑ vs 上一步均为 accuracy 差值（百分点，pp）；"
        "基线行没有对比；门禁是每条线 vs 基线的结论。</sub>"
    )
    lines.append("")
    rows = []
    for entry in entries:
        acc = entry["overall"].get("accuracy")
        rows.append(
            [
                str(entry["index"] + 1),
                entry["name"],
                entry["sha8"] or "—",
                _value_cell("accuracy", acc),
                _value_cell("evidence_recall", entry["overall"].get("evidence_recall")),
                "—" if entry["index"] == 0 else _pp(entry["diff_vs_base"]),
                "—" if entry["index"] == 0 else _pp(entry["diff_vs_prev"]),
                "—" if entry["index"] == 0 else entry["verdict"],
            ]
        )
    lines.append(
        _md_table(
            [
                "线序",
                "名称",
                "commit（短 sha）",
                "accuracy",
                "证据召回率",
                "↑ vs 基线",
                "↑ vs 上一步",
                "门禁(vs 基线)",
            ],
            rows,
        )
    )
    lines.append("")

    # 2. 逐指标表（行=指标，列=各线 + 每线 vs 基线的差值；始终保留基线一列）
    lines.append("## 逐指标表")
    lines.append("")
    if multi:
        lines.append(
            f"<sub>多数据集按融合口径汇总（{_fusion_text(fusion, fusion_cap)}；"
            "办公域槽位不参与）；效率成本类指标依赖部署环境，多数据集不融合，"
            "见各数据集级报告。</sub>"
        )
        lines.append("")
    headers = ["指标", base["name"]]
    for entry in entries[1:]:
        headers.extend([entry["name"], f"{entry['name']} vs 基线"])
    metric_rows = []
    for metric in metrics_list:
        base_value = base["overall"].get(metric)
        row = [metric_label(metric), _value_cell(metric, base_value)]
        for entry in entries[1:]:
            value = entry["overall"].get(metric)
            row.extend(
                [_value_cell(metric, value), _delta(metric, value, base_value)]
            )
        metric_rows.append(row)
    lines.append(_md_table(headers, metric_rows))
    lines.append("")

    # 3. 逐数据集 accuracy（多数据集时）
    if multi:
        lines.append("## 逐数据集 accuracy（vs 基线）")
        lines.append("")
        lines.append(
            "<sub>单元格为 accuracy；非基线列括号内是该集相对基线的差值（pp），"
            "不参与融合的办公域槽位同样列出，便于核对业务影响。</sub>"
        )
        lines.append("")
        ds_rows = []
        for key in dataset_keys:
            base_value = base["datasets"][key].metrics.get("overall", {}).get("accuracy")
            row = [key, _value_cell("accuracy", base_value)]
            for entry in entries[1:]:
                value = (
                    entry["datasets"][key].metrics.get("overall", {}).get("accuracy")
                )
                cell = _value_cell("accuracy", value)
                if isinstance(value, (int, float)) and isinstance(
                    base_value, (int, float)
                ):
                    cell = f"{cell} ({_pp(float(value) - float(base_value))})"
                row.append(cell)
            ds_rows.append(row)
        lines.append(
            _md_table(["数据集", base["name"], *[e["name"] for e in entries[1:]]], ds_rows)
        )
        lines.append("")

    # 4. 图表
    lines.append("## accuracy 走势")
    lines.append("")
    lines.append(f"![多线 accuracy 走势]({layout.FIG_SERIES_ACCURACY})")
    lines.append("")

    # 5. 门禁
    lines.append("## 门禁（每条线 vs 基线）")
    lines.append("")
    gate_rows = []
    for entry in entries:
        if entry["index"] == 0:
            gate_rows.append(["1 " + entry["name"], "—", *["—"] * len(_GATE_NAMES)])
            continue
        gate_rows.append(
            [
                f"{entry['index'] + 1} {entry['name']}",
                entry["verdict"],
                *[_gate_status(entry["gates"], name) for name in _GATE_NAMES],
            ]
        )
    lines.append(
        _md_table(
            ["线", "结论", *[dim_label(name) for name in _GATE_NAMES]], gate_rows
        )
    )
    lines.append("")
    details = [
        f"- **{entry['name']}**：{gate['name']} — {gate['detail']}"
        for entry in entries[1:]
        for gate in entry["gates"]
        if gate["status"] != "PASS"
    ]
    lines.append("**未通过项说明**")
    lines.append("")
    if details:
        lines.extend(details)
    else:
        lines.append("（全部通过）")
    lines.append("")
    lines.append("**可比性（vs 基线）**")
    lines.append("")
    for entry in entries[1:]:
        reason = "；".join(entry["comparability"]["reasons"]) or "全部维度一致或可核验"
        state = "可比" if entry["comparability"]["comparable"] else "不可比"
        lines.append(f"- {entry['name']}：{state} — {reason}")
    lines.append("")

    # 6. 来源与产物（可追溯）
    lines.append("## 来源与产物")
    lines.append("")
    source_rows = []
    for entry in entries:
        copies = []
        for key in dataset_keys:
            rel = os.path.relpath(
                os.path.join(entry["copy_dir"], key, layout.DIR_REPORT, layout.FILE_REPORT_MD),
                reports_dir,
            ).replace(os.sep, "/")
            copies.append(f"[{key}]({rel})")
        source_rows.append(
            [
                str(entry["index"] + 1),
                entry["name"],
                entry["name_source"],
                entry["commit"] or "—",
                f"`{entry['target']}`",
                "、".join(copies),
            ]
        )
    lines.append(
        _md_table(
            ["线序", "名称", "命名来源", "commit", "产物目录", "副本（数据集报告）"],
            source_rows,
        )
    )
    lines.append("")

    out_path = os.path.join(reports_dir, layout.FILE_REPORT_MD)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return out_path


def run_series(
    specs: Sequence[str],
    repo: Union[str, None] = None,
    dataset: Union[str, None] = None,
    report_out: Union[str, None] = None,
    require_comparable: bool = False,
    fusion: str = FUSION_WEIGHTED,
    fusion_cap: int = FUSION_WEIGHT_CAP,
    *,
    llm: Union[LLMNaming, None] = None,
) -> dict[str, Any]:
    """多线对比：读 N 条线的产物，写 ``series_<ts>/`` 并返回结果。

    Args:
        specs: ``--line`` 原始值（``<commit-or-name>=<产物目录>``，顺序即时间顺序）。
        repo: 取 commit subject 的 git 仓库（缺省当前目录）。
        dataset: 只比某个数据集（多数据集实验目录时用）；缺省比全部。
        report_out: ``series_<ts>`` 的父目录（缺省 :func:`membench.paths.output_root`）。
        require_comparable: strict 可比性；关键维度不一致 / 无法核验 → 该线门禁 FAIL。
        fusion: 多数据集融合口径（``weighted`` / ``equal``）。
        fusion_cap: 加权口径下的单集权重封顶（题数）。
        llm: 线名的 LLM 配置（commit + 改动 → 简短中文名）；缺省或未同时给出
            base_url / model 时只用代码抽取。显式线名与基线不调 LLM。

    Returns:
        ``{"series_dir", "report_path", "fig_path", "lines", "verdict", "fusion"}``；
        ``lines`` 每项含线名 / 命名来源 / commit / 目录 / 副本 / 汇总指标 /
        逐线门禁 / 结论。``verdict`` 是各线结论里最差的一个（FAIL 时命令退出码应为 1）。

    Raises:
        SystemExit: 线数不足、入参 / 产物不完整、各线数据集不一致、融合口径非法。
    """
    if fusion not in FUSION_CHOICES:
        raise SystemExit(
            f"未知融合口径: {fusion!r}（可选 {' / '.join(FUSION_CHOICES)}）"
        )
    if len(specs) < 2:
        raise SystemExit(
            "series 至少需要 2 条 --line（单条线没有对比意义）："
            "--line <commit-or-name>=<产物目录> --line ..."
        )
    warned: set[str] = set()

    def _warn_once(message: str) -> None:
        if message not in warned:
            warned.add(message)
            print(message, file=sys.stderr)

    lines = load_lines(specs, repo=repo, dataset=dataset, warn=_warn_once, llm=llm)
    dataset_keys = list(lines[0].datasets)
    multi = len(dataset_keys) > 1

    fused_by_line = {
        line.index: _fused_of(line, fusion, fusion_cap) for line in lines
    }
    overall_by_line = {
        line.index: _overall_of(line, fused_by_line[line.index]) for line in lines
    }

    entries: list[dict[str, Any]] = []
    for line in lines:
        gates: list[dict[str, Any]] = []
        comparability: dict[str, Any] = {
            "comparable": True,
            "datasets": {},
            "mismatches": [],
            "unverifiable": [],
            "reasons": [],
        }
        if line.index > 0:
            comparability = _comparability(lines[0], line)
            gates.append(
                _comparability_gate(comparability, require_comparable)
            )
            gates.extend(
                _eval_gates(
                    overall_by_line[0],
                    overall_by_line[line.index],
                    ACC_THRESHOLD,
                    STALE_THRESHOLD,
                    LATENCY_WARN,
                )
            )
        entries.append(
            {
                "index": line.index,
                "name": line.name,
                "name_source": line.name_source,
                "commit": line.commit,
                "sha8": line.sha8,
                "target": line.target,
                "datasets": line.datasets,
                "overall": overall_by_line[line.index],
                "fused": fused_by_line[line.index],
                "comparability": comparability,
                "gates": gates,
                "verdict": _verdict(gates) if gates else "PASS",
                "diff_vs_base": None,
                "diff_vs_prev": None,
                "copy_dir": "",
            }
        )

    base_acc = _accuracy(entries[0]["overall"])
    prev_acc = base_acc
    for entry in entries[1:]:
        acc = _accuracy(entry["overall"])
        entry["diff_vs_base"] = _sub(acc, base_acc)
        entry["diff_vs_prev"] = _sub(acc, prev_acc)
        prev_acc = acc

    series_dir = layout.experiment_path(report_out, layout.series_subdir())
    for line, entry in zip(lines, entries):
        line_dir = layout.series_line_dir(series_dir, line.index, line.name, line.commit)
        for key, res in line.datasets.items():
            shutil.copytree(res.run_dir, os.path.join(line_dir, key))
        entry["copy_dir"] = line_dir
    runtime.set_dir(os.path.abspath(series_dir))

    reports_dir = layout.task_report_dir(series_dir)
    os.makedirs(reports_dir, exist_ok=True)
    fig_path = os.path.join(reports_dir, layout.FIG_SERIES_ACCURACY)
    fig_series_accuracy(
        [entry["name"] for entry in entries],
        [_accuracy(entry["overall"]) for entry in entries],
        fig_path,
        dataset_series={
            key: [
                entry["datasets"][key].metrics.get("overall", {}).get("accuracy")
                for entry in entries
            ]
            for key in dataset_keys
        }
        if multi
        else None,
        headline_label="融合值" if multi else "accuracy",
    )
    report_path = _write_series_report(
        reports_dir,
        entries,
        dataset_keys=dataset_keys,
        fusion=fusion,
        fusion_cap=fusion_cap,
        base_metrics=lines[0].datasets[dataset_keys[0]].metrics,
    )

    verdict = _worst_verdict(entries)
    print(f"[membench] series: {len(entries)} 线 -> {series_dir}")
    for entry in entries:
        acc = entry["overall"].get("accuracy")
        delta = (
            ""
            if entry["index"] == 0
            else f"（vs 基线 {_pp(entry['diff_vs_base'])}，vs 上一步 {_pp(entry['diff_vs_prev'])}）"
        )
        print(
            f"[membench]   {entry['index'] + 1} {entry['name']}"
            f" [{entry['sha8'] or '—'}]: accuracy={_value_cell('accuracy', acc)}"
            f"{delta} {entry['verdict']}"
        )
    print(f"[membench] 整体结论: {verdict}")
    print(f"[membench] report: {report_path}")

    return {
        "series_dir": series_dir,
        "report_path": report_path,
        "fig_path": fig_path,
        "lines": entries,
        "verdict": verdict,
        "fusion": fusion,
        "dataset_keys": dataset_keys,
    }


def _accuracy(overall: dict[str, Any]) -> Union[float, None]:
    """汇总 accuracy（None = 未计算；调用方按 n/a 展示）。"""
    value = overall.get("accuracy")
    return float(value) if isinstance(value, (int, float)) else None


def _sub(value: Union[float, None], base: Union[float, None]) -> Union[float, None]:
    """两个可选值相减（任一为 None 时返回 None）。"""
    if value is None or base is None:
        return None
    return value - base
