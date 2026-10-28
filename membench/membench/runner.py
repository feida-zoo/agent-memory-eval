"""Runner：对单个记忆系统执行评测并写出实验目录下的产物。

流程（ingest 阶段始终串行，print 打印进度）：
    load_dataset -> 每个 dialogue: adapter.reset() -> 逐 session adapter.ingest()
    -> 该对话 ingest 完成后（第一道 query 之前）导出它唯一一份状态快照
    -> 每题 adapter.query() + 文本/检索指标 + judge.grade()
    -> 逐对话聚合写对话级 report/ -> 全对话聚合写数据集级 report/（含图）
    -> write_task_reports 时把数据集级报告复制到 <实验目录>/reports/

分片（``shards>1``）只写本片负责的对话目录与分片账本，**不写**数据集级 ``report/`` 与
任务级 ``reports/``——分片产物不再"看起来像完整实验"；数据集级与任务级产物一律由
``fuse`` 补齐。

并发：query+judge+证据匹配 阶段在 ``jobs>1`` 时用 ThreadPoolExecutor 题目级并发
（对话时序和记忆状态敏感的 ingest 阶段必须保持串行）；收集结果后按原始题目
顺序排序落盘，results.jsonl 的记录顺序与 jobs=1 完全一致。
**jobs>1 要求 adapter.query 与 judge 线程安全**（evidence-match 匹配器自身
线程安全，LLM 口径的逐题调用也随之并发）；内置 mock adapter 和
LLMJudge（urllib 无状态调用）均视为线程安全。

题目级忽略标记（ignore=true）的题整条跳过：不调用 adapter.query、不写入
results.jsonl（不浪费 LLM 调用，metrics 聚合分母口径自然正确）；忽略总数
记入 metrics.json 的 ``n_ignored`` 元信息，report.md 评测配置处注明。

产物结构（见 :mod:`membench.layout`）::

    <实验目录>/runs/<dataset>/report/{results.jsonl,metrics.json,report.md,fig_*.png}
    <实验目录>/runs/<dataset>/<对话目录>/report/{results.jsonl,metrics.json,report.md}
    <实验目录>/runs/<dataset>/<对话目录>/runtime/          # 运行时数据（reset 备份等）
    <实验目录>/runs/<dataset>/<对话目录>/memory_snapshot/  # 快照默认导出（含 manifest.json）
    <实验目录>/runs/<dataset>/shards.json                  # 分片账本（分片评测时；读/写/校验见 layout）
    <实验目录>/reports/                                    # 任务级（数据集级报告副本）

回放任务（``task init --replay``）另有任务级快照池::

    <任务>/snapshots/<sha256 前 12 位>/             # 内容寻址桶（含 manifest.json，来源记在 source 段）
    <任务>/snapshots/entries/<臂>/<数据集>/<对话>.json  # 逐对话条目（真源，各进程各写各的）
    <任务>/snapshots/index.json                    # 派生视图（由条目 + 桶清单重算，fuse/status 刷新）

快照默认导出（``snapshot=True``）：每个对话只有一份，打点在该对话全部 session ingest
完成之后、第一道 query 之前——正是回放时的入口状态。适配器未声明快照能力时打印一次
警告并跳过（metrics 记 ``state_snapshot.role="unsupported"``），不中断评测。

**回放**（``replay``）同样在 ``restore()`` / ``rebuild()`` 之后、第一道 query 之前导出，
但落点是任务级内容寻址池 ``<任务>/snapshots/<sha256 前 12 位>/``（见
:class:`_SnapshotPoolWriter` 与 :mod:`membench.layout` 的池 helper）：状态内容相同的
导出共用同一个桶，A/B 两臂回放同一源时只产生一份；``--no-snapshot`` 关闭，关闭时
既不写桶也不写条目 / 索引。并行分片回放时各进程只写自己对话的条目，池里没有需要
争抢的共享可变状态。
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Union

from membench import layout, runtime
from membench.data.schema import (
    dataset_fingerprint,
    load_dataset,
    question_capabilities,
)
from membench.judges import get_judge
from membench.metrics import aggregate
from membench.metrics.evidence import SubstringMatcher
from membench.metrics.text import bleu1, token_f1
from membench.report import (
    copy_report,
    format_token_usage,
    render_dataset_report,
    render_dialogue_report,
)
from membench import snapshot as snapshot_mod

#: 单份对话快照超过该字节数时告警一次（提示可用 --no-snapshot 关闭）。
SNAPSHOT_WARN_BYTES = 100 * 1024 * 1024


def _write_run_artifacts(
    report_dir: str,
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    figures: bool,
) -> None:
    """把一层级的逐题明细 / 指标 / 报告写到 ``report_dir``。

    对话级与数据集级共用这段落盘逻辑，差别只有 ``figures``（对话级不出图，
    报告也不引用图片）。
    """
    os.makedirs(report_dir, exist_ok=True)
    with open(
        os.path.join(report_dir, layout.FILE_RESULTS), "w", encoding="utf-8"
    ) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(
        os.path.join(report_dir, layout.FILE_METRICS), "w", encoding="utf-8"
    ) as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    if figures:
        render_dataset_report(report_dir, records, metrics)
    else:
        render_dialogue_report(report_dir, records, metrics)


def _judge_result_to_dict(jr: Any) -> dict[str, Any]:
    """把 JudgeResult 转成可 JSON 序列化的 dict。"""
    if dataclasses.is_dataclass(jr) and not isinstance(jr, type):
        return dataclasses.asdict(jr)
    if isinstance(jr, dict):
        return dict(jr)
    try:
        return dict(vars(jr))
    except TypeError:
        return {"correct": bool(getattr(jr, "correct", False))}


def _shard_units(dialogues: list[Any], shard_by: str) -> int:
    """分片上限：dialogue 维度为对话数，question 维度为题目数。"""
    if shard_by == "dialogue":
        return len(dialogues)
    return sum(len(dialogue.questions) for dialogue in dialogues)


def select_shard_questions(
    dialogues: list[Any],
    shards: int,
    shard_index: int,
    limit: Union[int, None],
    shard_by: str = "question",
) -> list[Any]:
    """对加载后的题目做确定性分片与截断（先分片再截断）。

    ``shard_by="question"``：全局题号 ``i``（按 dialogue 顺序、题序展开）满足
    ``i % shards == shard_index`` 的题入选，再取前 ``limit`` 道；不含入选题的
    dialogue 整段跳过（ingest 也省掉）。

    ``shard_by="dialogue"``：全局对话序号 ``d``（按数据集顺序）满足
    ``d % shards == shard_index`` 的**整段对话**入选，保留其全部题目，再取前
    ``limit`` 道。分片之间 ingest 互不重复——适合"几乎每个对话都有题"的
    ingest 密集型数据集（如 BEAM-1M），避免题目级分片造成的 ingest 成本 ×N。

    调用方（:meth:`Runner.run`）保证 ``shards`` 已按数据集上限解析过
    （``max`` / 超限整数在此前收敛），本函数不做上限校验。

    本函数是分片选取口径的**唯一实现**：``fuse`` 用它复算"每一片本该有哪些对话 /
    哪些题"，与账本和实际产物逐项核对覆盖完整性（缺片、重复覆盖都要报）。
    """
    if shard_by == "dialogue":
        out: list[Any] = []
        n_taken = 0
        for d_idx, dialogue in enumerate(dialogues):
            if d_idx % shards != shard_index:
                continue
            questions = dialogue.questions
            if limit is not None:
                if n_taken >= limit:
                    break
                questions = questions[: limit - n_taken]
            if questions:
                out.append(dataclasses.replace(dialogue, questions=questions))
                n_taken += len(questions)
        return out

    selected: dict[str, list[Any]] = {}
    idx = 0
    n_taken = 0
    for dialogue in dialogues:
        for q in dialogue.questions:
            take = idx % shards == shard_index
            idx += 1
            if not take:
                continue
            if limit is not None and n_taken >= limit:
                continue
            selected.setdefault(str(dialogue.dialogue_id), []).append(q)
            n_taken += 1
    out = []
    for dialogue in dialogues:
        qs = selected.get(str(dialogue.dialogue_id))
        if qs:
            out.append(dataclasses.replace(dialogue, questions=qs))
    return out


class _SnapshotPoolWriter:
    """回放任务的内容寻址快照池写入器（``<任务>/snapshots/``）。

    每个对话先导出到池内临时目录（``.tmp-<uuid>``），算完内容哈希（沿用
    :func:`membench.snapshot.dir_sha256` 的口径，排除 ``manifest.json``）后按前 12 位
    定桶：桶首次出现则整体移入，已存在则校验桶内清单的 ``entry.sha256`` 与本次一致
    后丢弃临时目录（复用）；不一致只可能是哈希前缀撞车或桶被改动，直接拒绝。

    落位只靠一次 ``os.replace`` 的成败判"本次创建 / 别人已建好"——不预判目录存在性、
    不读共享索引：两个分片进程同时建同一个桶时，失败的一方按磁盘上的清单校验后记
    复用即可。每个对话另写一个自足的条目文件
    （``snapshots/entries/<臂>/<数据集>/<对话>.json``，见
    :func:`membench.layout.write_snapshot_entry`），对话级分片下各进程只写自己的
    对话 ⇒ **并行回放没有共享可变状态要抢，不需要任何锁**。

    池属于**任务**而不是臂：``out_dir`` 是 ``<任务>/A`` / ``B`` 时池在父任务目录，
    臂键即目录名；单臂任务 / 普通实验目录的臂键是空串。
    """

    def __init__(self, out_dir: str, plan: dict[str, Any], dataset: str) -> None:
        self.pool_root = layout.snapshot_pool_root(out_dir)
        self.arm = layout.snapshot_arm_key(out_dir)
        self.dataset = dataset
        self.plan = plan
        #: 本 run 逐对话的桶登记：``{对话 id: {"bucket": digest12, "reused": bool}}``
        self.buckets: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    def _require_same_bucket(self, bucket: str, state_sha256: str) -> None:
        """桶已存在时必须装的是同一份内容（桶名是内容哈希前 12 位）。

        ``load_manifest`` 负责"桶里没有/读不出清单"的报错；这里只比对内容哈希：
        一致才允许复用，不一致说明前缀撞车或桶被改动/损坏。
        """
        known = (snapshot_mod.load_manifest(bucket).get("entry") or {}).get("sha256")
        if known != state_sha256:
            raise SystemExit(
                f"快照桶 {bucket} 已存在但内容哈希不一致：桶内 {known} vs 本次 "
                f"{state_sha256}——桶名是内容哈希前 12 位，同名不同内容只可能是哈希前缀"
                "撞车或桶被改动；请检查该桶（必要时删除）后重跑，不要硬塞进同一桶"
            )

    def _place_bucket(self, tmp: str, bucket: str, state_sha256: str) -> bool:
        """把临时导出目录整体落成桶；返回 True 表示复用了别的进程已建好的桶。

        成败由 ``os.replace`` 本身裁决（目录整体改名是原子的）：成功 = 本次创建；
        抛 ``OSError``（目标已存在且非空）→ 以磁盘上的桶为准校验内容哈希、删临时目录。
        不先 ``isdir`` 预判——预判与改名之间正是两个分片进程同时建同一个桶的竞态窗口。
        """
        try:
            os.replace(tmp, bucket)
        except OSError:
            if not os.path.isdir(bucket):
                raise
            self._require_same_bucket(bucket, state_sha256)
            return True
        return False

    def export(
        self,
        adapter: Any,
        manifest: dict[str, Any],
        *,
        dialogue_id: str,
        sessions: list[list[dict]],
        n_questions: int,
    ) -> dict[str, Any]:
        """导出一个对话到池、写它的条目文件，返回 ``{bucket, reused, entry}``。"""
        tmp = layout.new_snapshot_tmp_dir(self.pool_root)
        try:
            manifest["source"] = {
                "kind": "replay",
                "replay": self.plan["source"],
                "replay_mode": self.plan["mode"],
                "source_run": self.plan["source_run"],
                "source_manifest_sha256": self.plan.get("manifest_sha256"),
            }
            if self.plan.get("source_pool"):
                # 链式回放：上一轮也是回放任务，记下用的是哪个臂的桶
                manifest["source"]["source_pool"] = self.plan["source_pool"]
                manifest["source"]["source_arm"] = self.plan.get("source_arm")
            entry = snapshot_mod.write_snapshot(
                tmp,
                adapter,
                manifest,
                dialogue_id=dialogue_id,
                sessions=sessions,
                n_questions=n_questions,
            )
            n_bytes, n_files = snapshot_mod.dir_stats(
                tmp, exclude=(snapshot_mod.MANIFEST_NAME,)
            )
            digest = layout.snapshot_bucket_id(entry["sha256"])
            bucket = layout.snapshot_bucket_dir(self.pool_root, digest)
            reused = self._place_bucket(tmp, bucket, entry["sha256"])
            if reused:
                shutil.rmtree(tmp, ignore_errors=True)
            tmp = None
            layout.write_snapshot_entry(
                self.pool_root,
                {
                    "schema": layout.SNAPSHOT_ENTRY_SCHEMA,
                    "digest": digest,
                    "state_sha256": entry["sha256"],
                    "bytes": n_bytes,
                    "n_files": n_files,
                    "dataset": self.dataset,
                    "dialogue_id": str(dialogue_id),
                    "arm": self.arm,
                    "source": self.plan["source"],
                    "source_manifest_sha256": self.plan.get("manifest_sha256"),
                    "reused": reused,
                    "run_id": str(manifest.get("run_id") or ""),
                    "written_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            self.buckets[str(dialogue_id)] = {
                "bucket": digest,
                "reused": reused,
            }
            return {"bucket": digest, "reused": reused, "entry": entry}
        finally:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)

    def summary(self) -> dict[str, Any]:
        """本 run 的池汇总：池路径 / 不同桶数 / 复用数 / 新写数。"""
        digests = {info["bucket"] for info in self.buckets.values()}
        return {
            "snapshots_dir": os.path.abspath(self.pool_root),
            "n_buckets": len(digests),
            "n_reused": sum(1 for info in self.buckets.values() if info["reused"]),
            "n_new": sum(1 for info in self.buckets.values() if not info["reused"]),
        }


class Runner:
    """评测执行器。

    Args:
        adapter: 实现 BaseMemoryAdapter 契约的实例（需有 name 属性）。
        dataset_path: 数据集 JSONL 路径。
        judge: "heuristic" / "llm" 字符串（经 get_judge 解析），
            或直接传入 Judge 实例（例如自定义配置的 LLMJudge）。
        out_dir: **实验目录**（不存在则创建）；本次评测的数据落在它的
            ``runs/<dataset>/`` 下，任务级报告在 ``reports/``。该数据集在本实验
            目录下已有产物时拒绝运行（不静默覆盖上一次的结果）。
        platform: 可选平台标签（如 "wx" / "fs"），用于 run_id 与元数据；
            缺省时 run_id 中用 "default"。
        commit: 可选 commit 标签（如 "a1b2c3"），记入元数据。
        dataset_source: 测试集来源标签（"user-specified" / "user" /
            "builtin-office"），通常来自 userdata.resolve_dataset；
            缺省视为 "user-specified"（直接向 Runner 传了路径）。
        jobs: 题目级并发数（默认 1 串行）。ingest 阶段始终串行；仅
            query+judge 阶段并发，输出顺序与 jobs=1 一致。
            **jobs>1 要求 adapter.query 与 judge 线程安全。**
        limit: 冒烟模式：只评测（分片后的）前 N 道题；None 表示全量。
        shards: 分片总数（默认 1 不分片）；可传 ``"max"`` 按数据集上限自动取最大
            （dialogue=对话数、question=题目数）；超过上限的整数按上限执行，
            越界分片索引安全跳过（不生成空 run）。与 limit 组合时先分片再截断。
        shard_index: 本分片序号（从 0 开始，需 < shards）。
        shard_by: 分片维度（默认 ``"question"``）——``"question"`` 按全局题号
            ``i % shards == shard_index`` 分片；``"dialogue"`` 整段选取
            ``d % shards == shard_index`` 的对话、保留其全部题目，分片间
            ingest 互不重复，适合"几乎每个对话都有题"的 ingest 密集型数据集。
        adapter_meta: 适配器配置自由文本（如 ``"key=value;..."``），原样记入
            metrics.json 的 ``adapter_meta``。
        driver_llm: 驱动被测系统的 LLM 标识（如模型名），记入 metrics.json
            供 compare 的可比性前置校验；缺省 None。
        judge_model: 判分模型标识；缺省 None 时若 judge 对象带非空 ``.model``
            属性则自动取该值。
        adapter_version: 适配器实现版本标识，记入 metrics.json；缺省 None。
        write_task_reports: True（默认）时把数据集级报告复制到实验目录的
            ``reports/``（单数据集 run 的任务级报告 = 数据集级报告内容）；
            evaluate 跑多数据集时传 False，由调用方统一写任务级 reports/。
            分片（``shards>1``）不写，由 ``fuse`` 补齐。
        shared: True 表示 ``out_dir`` 是多方共用的任务目录（CLI ``--task-dir``）：允许
            写进已存在且非空的实验目录，分片归属与进度按分片账本判重——同一个分片
            重复跑直接拒绝。共用目录下只支持 ``shard_by="dialogue"``：题目级分片会让
            多个分片写同一对话目录、互相覆盖。
        snapshot: 默认 True：在每个对话 ingest 完成后（第一道 query 之前）调用适配器
            的 ``snapshot()`` 把状态导出到该对话的 ``memory_snapshot/`` 并写
            ``manifest.json``——之后改读取侧代码可用 ``replay`` 免重跑 ingest。
            适配器未声明快照能力时打印一次警告并跳过（metrics 记
            ``state_snapshot.role="unsupported"``）。回放模式下同样导出，但落任务级
            内容寻址池 ``<任务>/snapshots/<内容哈希前 12 位>/``（相同状态只存一份）；
            ``--no-snapshot``（``snapshot=False``）两种模式都不写快照。
        replay: 回放源：实验目录、数据集目录（``<实验目录>/runs/<dataset>``），或
            **回放任务**（含 ``snapshots/index.json`` 的任务根 / 臂目录
            ``<任务>/A``）。给定时跳过 reset+ingest，按对话 ``restore()``（或
            ``rebuild()``）后只跑 query。清单齐备性 / 数据集指纹 / 对话覆盖 / 能力 /
            模式都在**开跑前**校验，失败即停且不产生任何产物。
        replay_mode: ``auto``（默认，优先 restore，格式不兼容时退回 rebuild）/
            ``restore`` / ``rebuild``。
        replay_verify_source: 给定源数据集目录时，回放结束与之逐题比对
            （answer / retrieved / 判分），结果写进 metrics 与报告——这就是
            "回放误差地板"，读取侧 A/B 的差值必须大于它才有意义。
    """

    def __init__(
        self,
        adapter: Any,
        dataset_path: str,
        judge: Union[str, Any] = "heuristic",
        *,
        out_dir: str,
        platform: Union[str, None] = None,
        commit: Union[str, None] = None,
        dataset_source: Union[str, None] = None,
        jobs: int = 1,
        limit: Union[int, None] = None,
        shards: Union[int, str] = 1,
        shard_index: int = 0,
        shard_by: Union[str, None] = None,
        adapter_meta: Union[str, None] = None,
        driver_llm: Union[str, None] = None,
        judge_model: Union[str, None] = None,
        evidence_matcher: Any = None,
        adapter_version: Union[str, None] = None,
        write_task_reports: bool = True,
        shared: bool = False,
        snapshot: bool = True,
        replay: Union[str, None] = None,
        replay_mode: str = "auto",
        replay_verify_source: Union[str, None] = None,
    ) -> None:
        if jobs < 1:
            raise ValueError(f"jobs 必须 >= 1，当前为 {jobs}")
        if replay_verify_source and not replay:
            raise ValueError("replay_verify_source 只在 replay 模式下有意义（需同时给 replay）")
        if replay_mode not in ("auto", "restore", "rebuild"):
            raise ValueError(f"replay_mode 必须是 auto / restore / rebuild，当前为 {replay_mode!r}")
        # 回放也从恢复后的状态导出快照，但落任务级内容寻址池（同样的状态只存一份），
        # 因此 snapshot 开关在回放下同样生效（--no-snapshot 时不写任何桶与索引）
        self.snapshot = bool(snapshot)
        self.shared = bool(shared)
        self.replay = replay
        self.replay_mode = replay_mode
        self.replay_verify_source = replay_verify_source
        self.write_task_reports = bool(write_task_reports)
        self.jobs = int(jobs)
        if limit is not None and limit < 1:
            raise ValueError(f"limit 必须 >= 1，当前为 {limit}")
        if shards != "max" and (not isinstance(shards, int) or shards < 1):
            raise ValueError(f"shards 必须 >= 1 的整数或 'max'，当前为 {shards!r}")
        if shard_index < 0:
            raise ValueError(f"shard_index 必须 >= 0，当前为 {shard_index}")
        if isinstance(shards, int) and shard_index >= shards:
            raise ValueError(
                f"shard_index 必须满足 0 <= shard_index < shards，当前为 {shard_index}/{shards}"
            )
        self.shard_by = shard_by or "question"
        if self.shard_by not in ("question", "dialogue"):
            raise ValueError(f"shard_by 必须是 'question' 或 'dialogue'，当前为 {self.shard_by!r}")
        self.limit = limit
        self.shards = shards  # int 或 "max"（run() 按数据集上限解析）
        self.shard_index = int(shard_index)
        self.adapter_meta = adapter_meta
        self.adapter = adapter
        self.dataset_path = dataset_path
        self.driver_llm = driver_llm
        self.adapter_version = adapter_version
        self.out_dir = out_dir
        self.platform = platform
        self.commit = commit
        # 显式向 Runner 传 dataset_path 即视为 user-specified
        self.dataset_source = dataset_source or "user-specified"
        if isinstance(judge, str):
            self.judge_name = judge
            self.judge = get_judge(judge)
        else:
            self.judge = judge
            self.judge_name = getattr(judge, "name", judge.__class__.__name__)
        # judge_model 缺省时从 judge 对象的 .model 自动补全（非空才算）
        self.judge_model = judge_model or getattr(self.judge, "model", None) or None
        # 检索命中口径：默认 substring（与历史分数一致），可注入 overlap / llm
        self._evidence_matcher = (
            evidence_matcher if evidence_matcher is not None else SubstringMatcher()
        )

    # ------------------------------------------------------------------
    def _eval_question(
        self,
        dialogue_id: str,
        question: Any,
        ingest_latency_ms: float,
        ingest_llm_calls: int,
        ingest_input_tokens: int,
    ) -> dict[str, Any]:
        """评测单题，返回 SPEC 3.4 的记录行。"""
        result = self.adapter.query(question.question)

        answer = str(getattr(result, "answer", "") or "")
        retrieved = [str(x) for x in (getattr(result, "retrieved", None) or [])]
        evidence = [str(x) for x in (getattr(question, "evidence", None) or [])]
        gold_answer = str(getattr(question, "gold_answer", "") or "")

        # 检索命中口径由匹配器决定（substring / overlap / llm）；
        # abstention 题无 evidence，匹配器返回 (False, 0.0)
        evidence_hit, recall_at_5 = self._evidence_matcher.evidence_hits(
            evidence, retrieved, k=5
        )

        judge_result = self.judge.grade(question, gold_answer, answer)

        return {
            "qid": str(getattr(question, "qid", "")),
            "dialogue_id": dialogue_id,
            "category": str(getattr(question, "category", "")),
            "question": str(getattr(question, "question", "") or ""),
            "gold_answer": gold_answer,
            "prediction": answer,
            "retrieved": retrieved,
            "evidence_hit": evidence_hit,
            "token_f1": float(token_f1(gold_answer, answer)),
            "bleu1": float(bleu1(gold_answer, answer)),
            "recall@5": float(recall_at_5),
            "judge": _judge_result_to_dict(judge_result),
            "latency_ms": float(getattr(result, "latency_ms", 0.0) or 0.0),
            "input_tokens": int(getattr(result, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(result, "output_tokens", 0) or 0),
            "llm_calls": int(getattr(result, "llm_calls", 0) or 0),
            "ingest_latency_ms": float(ingest_latency_ms),
            "ingest_llm_calls": int(ingest_llm_calls),
            "ingest_input_tokens": int(ingest_input_tokens),
        }

    # ------------------------------------------------------------------
    def run(self) -> Union[str, None]:
        """执行评测并落盘，返回数据集产物目录（``<out_dir>/runs/<dataset>``）。

        数据集为空 / 分片索引越界（本进程无事可做）时返回 None、不产生任何产物；
        回放与快照能力的校验、分片账本的核对都在建目录之前完成，失败即 SystemExit、
        同样不留半成品。本实验目录下该数据集已有完整产物时同样在写盘前拒绝（不静默
        覆盖上一次的结果）；分片（``shards>1``）只写本片负责的对话目录与分片账本，
        数据集级与任务级产物由 ``fuse`` 补齐。
        """
        dialogues = load_dataset(self.dataset_path)

        # 分片解析：'max' 取数据集上限；超过上限的整数按上限执行（不报错）
        shard_units = _shard_units(dialogues, self.shard_by)
        unit_label = "对话数" if self.shard_by == "dialogue" else "题目数"
        if self.shards == "max":
            shards = shard_units
            print(f"[membench] --shards max -> {shards}（按数据集{unit_label}）")
        else:
            shards = min(int(self.shards), shard_units)
            if shards < int(self.shards):
                print(
                    f"[membench] --shards {self.shards} 超过数据集上限 {shard_units}"
                    f"（{unit_label}），按 {shards} 分片执行"
                )
        if shards == 0:
            print("[membench] 数据集为空（无可分片的对话/题目），本进程跳过、不产生产物。")
            return None
        if self.shard_index >= shards:
            print(
                f"[membench] 分片索引 {self.shard_index} 超出有效分片数 {shards}，"
                "本进程无事可做，跳过（不产生产物）。"
            )
            return None
        if shards > 1 or self.limit is not None:
            dialogues = select_shard_questions(
                dialogues, shards, self.shard_index, self.limit, self.shard_by
            )

        adapter_name = str(getattr(self.adapter, "name", self.adapter.__class__.__name__))
        dataset_name = os.path.splitext(os.path.basename(self.dataset_path))[0]
        sharded = shards > 1
        # 共用任务目录（CLI --task-dir）下多个分片写同一棵对话树：题目级分片会让多片
        # 写同一对话目录并互相覆盖，只有对话级分片是"各写各的对话"。
        if self.shared and sharded and self.shard_by != "dialogue":
            raise SystemExit(
                "共享任务目录下只支持 --shard-by dialogue：题目级分片会让多个分片写"
                "同一对话目录并互相覆盖；请改用 --shard-by dialogue，"
                "或给各分片独立的 --report-out"
            )
        # 写入前的覆盖拒绝：分片不写数据集级报告，该位置有 metrics 只可能来自不分片的
        # 评测或 fuse 产物——两种情况下这棵树都已有完整结果，不该再往里写分片。
        existing_metrics = os.path.join(
            layout.dataset_report_dir(self.out_dir, dataset_name), layout.FILE_METRICS
        )
        if os.path.isfile(existing_metrics):
            if sharded:
                raise SystemExit(
                    f"数据集 {dataset_name} 在本实验目录下已有完整产物: {existing_metrics}；"
                    "分片只写本片负责的对话目录、数据集级报告由 fuse 补齐，"
                    "请换一个实验目录写分片"
                )
            raise SystemExit(
                f"数据集 {dataset_name} 已在本实验目录下完成评测: {existing_metrics}；"
                "继续会在同一位置静默覆盖上一次的 metrics / 逐题明细，"
                "请换 --report-out 或删除旧产物后重跑"
            )
        # 回放保真校验的源产物：与其它「产物目录」入参同口径（实验目录或数据集目录），
        # 跑前解析——源产物有问题就不该白跑一场评测
        verify_source = (
            layout.resolve_dataset_dir(self.replay_verify_source, dataset_name)
            if self.replay_verify_source
            else None
        )
        platform_tag = self.platform or "default"
        now = datetime.now()
        # run_id 的唯一组装入口在 layout：commit 短哈希避免矩阵评测中同 adapter+platform
        # 同秒撞身份；limit / 分片标记避免冒烟或分片结果被误当全量结果
        # （词表见 layout.run_id_tags，格式见 layout.run_id）
        run_id = layout.run_id(
            adapter_name,
            platform_tag,
            dataset_name,
            now.strftime("%Y%m%d_%H%M%S"),
            commit=self.commit,
            tags=layout.run_id_tags(shards, self.shard_index, self.shard_by, self.limit),
        )
        dialogue_ids = [str(d.dialogue_id) for d in dialogues]
        dataset_dir = os.path.abspath(layout.dataset_dir(self.out_dir, dataset_name))

        # 分片账本：开跑前先核对身份/范围、看本片是否已经跑过——全是读操作，
        # 失败即停，不产生任何产物（账本的读 / 写 / 校验都在 layout）
        ledger_path: Union[str, None] = None
        ledger_header: dict[str, Any] = {}
        ledger_key = layout.shard_key(self.shard_index)
        if sharded:
            ledger_path = layout.shard_ledger_path(self.out_dir, dataset_name)
            ledger_header = layout.shard_ledger_header(
                dataset=dataset_name,
                shards=shards,
                shard_by=self.shard_by,
                limit=self.limit,
                fingerprint=dataset_fingerprint(self.dataset_path),
            )
            ledger = layout.read_shard_ledger(dataset_dir)
            layout.check_shard_ledger(ledger, ledger_header, path=ledger_path)
            known = ((ledger or {}).get("entries") or {}).get(ledger_key)
            if known:
                raise SystemExit(
                    f"分片 {self.shard_index}/{shards} 已经在本实验目录下跑过"
                    f"（账本状态 {known.get('status')!r}）：{ledger_path}；"
                    "同一分片重复跑会覆盖它已写好的对话目录，请换一个实验目录，"
                    "或确认没有进程在跑后删掉账本里这一条再重跑"
                )

        # 回放 / 快照能力的校验全部前移到建目录之前：失败即停，不留半成品产物
        replay_plan: Union[dict, None] = None
        if self.replay:
            replay_plan = snapshot_mod.plan_replay(
                self.replay,
                dataset_name,
                self.adapter,
                dialogue_ids,
                dataset_fingerprint=dataset_fingerprint(self.dataset_path),
                requested_mode=self.replay_mode,
            )
            print(
                f"[membench] 快照回放：{replay_plan['run_mode']}（源 run "
                f"{replay_plan['source_run']}，格式 v{replay_plan['format_version']}，"
                f"{len(dialogues)} 个对话）——写入侧固定，结论只覆盖读取/回答层"
            )
            if replay_plan.get("source_pool"):
                print(
                    f"[membench] 回放源是回放任务: {replay_plan['source_pool']}"
                    f"（臂 {replay_plan['source_arm'] or '单臂'}）"
                )
        snapshot_caps: dict[str, Any] = {}
        # 快照默认导出；适配器没有这个能力就跳过（只打印一次警告，不中断评测），
        # metrics 里如实记 state_snapshot.role="unsupported"
        if self.snapshot:
            snapshot_caps = snapshot_mod.read_capabilities(self.adapter)
            if not snapshot_caps.get("supported"):
                print(
                    "[membench] ⚠️ 该适配器未声明快照能力，本次跳过状态快照"
                    "（metrics 记 state_snapshot.role=unsupported）：适配器实现 "
                    "snapshot() 并让 snapshot_capabilities() 返回 supported=True 后即可导出；"
                    "不需要快照可用 CLI 的 --no-snapshot 关闭"
                )

        os.makedirs(self.out_dir, exist_ok=True)
        experiment_dir = os.path.abspath(self.out_dir)
        # 预检全部通过、目录已建：从这一刻起本次命令的运行时数据归这个实验目录。
        # 没有活动 CLI 会话时 set_dir 是空操作（库调用不会多出 runtime/）。
        runtime.set_dir(experiment_dir)
        # ignore=true 的题整条跳过（不查询、不计分），n_active 为实际评测题数。
        total = sum(len(d.questions) for d in dialogues)
        n_ignored = sum(
            1 for d in dialogues for q in d.questions if getattr(q, "ignore", False)
        )
        n_active = total - n_ignored
        print(f"[membench] run_id={run_id}")
        print(f"[membench] 实验目录={experiment_dir}")
        print(f"[membench] 数据集目录={dataset_dir}")
        print(f"[membench] dialogues={len(dialogues)} questions={total} judge={self.judge_name}")
        if n_ignored:
            print(f"[membench] 已忽略 {n_ignored} 道标记题（ignore=true）")

        shard_started_at: Union[str, None] = None
        if ledger_path is not None:
            # 分片开跑：先把"哪片在跑"写进账本，跑的过程中就能看出进度
            print(
                f"[membench] 分片 {self.shard_index}/{shards}"
                f"（{self.shard_by}）写入 {experiment_dir}"
            )
            shard_started_at = datetime.now().isoformat(timespec="seconds")
            layout.update_shard_ledger(
                dataset_dir,
                ledger_header,
                ledger_key,
                {
                    "status": "running",
                    "dialogues": dialogue_ids,
                    "run_id": run_id,
                    "started_at": shard_started_at,
                },
            )
            print(f"[membench] 分片账本: {ledger_path}（{ledger_key} running）")

        # 适配器可选的 describe() -> dict：记录其配置/版本等自描述信息（不参与评分
        # 与可比性）。失败只警告一次并记 None，不中断评测。放在开跑前，对话级与
        # 数据集级元数据取同一份。
        adapter_describe: Any = None
        describe_fn = getattr(self.adapter, "describe", None)
        if callable(describe_fn):
            try:
                described = describe_fn()
            except Exception as exc:  # noqa: BLE001 - 自描述失败不应中断评测
                print(
                    f"[membench] ⚠️ 适配器 describe() 抛错（{exc}），本次不记录"
                    "自描述（metrics.adapter_describe=null）；评测与评分不受影响"
                )
            else:
                if isinstance(described, dict):
                    try:
                        json.dumps(described, ensure_ascii=False)
                    except (TypeError, ValueError) as exc:
                        # metrics.json 在跑完之后才写：不可序列化的自描述若留到最后
                        # 才炸，等于让整场评测白跑——这里提前拦下。
                        print(
                            "[membench] ⚠️ 适配器 describe() 返回值无法 JSON 序列化"
                            f"（{exc}），本次不记录自描述"
                            "（metrics.adapter_describe=null）"
                        )
                    else:
                        adapter_describe = described
                else:
                    print(
                        "[membench] ⚠️ 适配器 describe() 返回 "
                        f"{type(described).__name__}，契约要求 dict；本次不记录"
                        "自描述（metrics.adapter_describe=null）"
                    )

        # 全流程 token / LLM 调用记账（ingest 按对话真实调用计一次；query 逐题累加）：
        # 与 records 的「每题归属」不同，这里是总量的唯一真相来源——question 分片下
        # 同一对话可能被多个分片各自 ingest，只有各分片自己的计数才是真实成本。
        token_usage: dict[str, int] = {
            "ingest_input_tokens": 0,
            "ingest_llm_calls": 0,
            "query_input_tokens": 0,
            "query_output_tokens": 0,
            "query_llm_calls": 0,
        }
        if replay_plan:
            # 回放跳过 reset + ingest：账目里必须显式标注，避免被读成"写入零成本"
            token_usage["ingest_skipped"] = 1
        # 检索回退数按本次 run 增量统计（matcher 实例可在多数据集间共享）
        match_fallbacks_start = int(getattr(self._evidence_matcher, "fallbacks", 0) or 0)
        # 快照目录登记：普通 run 是本次导出的目录（逐对话 memory_snapshot/）；回放是
        # **源**的逐对话快照目录（数据集目录或源池里的桶），state_snapshot 依据
        snapshot_dirs: dict[str, str] = dict(replay_plan["snapshots"]) if replay_plan else {}
        # 快照导出开关：请求了导出且适配器支持（回放同样导出，只是落任务级内容寻址池）
        export_snapshots = bool(self.snapshot and snapshot_caps.get("supported"))
        # 回放任务的池写入器：out_dir 是臂目录时池在父任务目录（见 layout 的池 helper）
        snapshot_pool: Union[_SnapshotPoolWriter, None] = None
        if replay_plan is not None and export_snapshots:
            snapshot_pool = _SnapshotPoolWriter(self.out_dir, replay_plan, dataset_name)
        snapshot_oversize_warned = False

        def _retrieval_meta(fallbacks: int) -> dict[str, Any]:
            """检索命中口径落账（substring / overlap / llm；阈值与回退数按需）。"""
            meta: dict[str, Any] = {"retrieval_match": self._evidence_matcher.name}
            threshold = getattr(self._evidence_matcher, "threshold", None)
            if threshold is not None:
                meta["retrieval_match_threshold"] = float(threshold)
            if fallbacks:
                meta["retrieval_match_fallbacks"] = fallbacks
            return meta

        def _snapshot_scope(dialogue_id: Union[str, None]) -> Union[dict[str, str], None]:
            """该层级的快照范围：数据集级=全部对话，对话级=自己；该对话没快照则 None。"""
            if dialogue_id is None:
                return snapshot_dirs
            path = snapshot_dirs.get(dialogue_id)
            return {dialogue_id: path} if path else None

        def _snapshot_state(dialogue_id: Union[str, None]) -> Union[dict, None]:
            """state_snapshot 块：数据集级含全部对话，对话级只含自己。

            角色三态：``replayed``（回放）/ ``exported``（本次导出）/ ``unsupported``
            （请求了导出但适配器不支持，如实记"没导"）；没开快照时整块不落——读取方
            据此就能区分"这次没导快照"与"适配器导不了"。回放且导出到池时另给
            ``snapshots_dir`` 与复用/新写计数（对话级给 ``bucket`` / ``reused``）。
            """
            if replay_plan:
                subset = _snapshot_scope(dialogue_id)
                if subset is None:
                    return None
                state = {
                    "role": "replayed",
                    "root": replay_plan["root"],
                    "manifest_sha256": snapshot_mod.manifests_digest(subset),
                    "format_version": replay_plan["format_version"],
                    "source_run": replay_plan["source_run"],
                    "replay_mode": replay_plan["mode"],
                    "n_dialogues": len(subset),
                }
                if snapshot_pool is not None:
                    # 本 run 的导出侧：状态已按内容去重写入任务级池（--no-snapshot 时没有）
                    summary = snapshot_pool.summary()
                    state["snapshots_dir"] = summary["snapshots_dir"]
                    if dialogue_id is None:
                        state["n_buckets"] = summary["n_buckets"]
                        state["n_reused"] = summary["n_reused"]
                        state["n_new"] = summary["n_new"]
                    else:
                        info = snapshot_pool.buckets.get(dialogue_id)
                        if info:
                            state["bucket"] = info["bucket"]
                            state["reused"] = bool(info["reused"])
                return state
            if export_snapshots:
                subset = _snapshot_scope(dialogue_id)
                if subset is None:
                    return None
                return {
                    "role": "exported",
                    "root": dataset_dir,
                    "manifest_sha256": snapshot_mod.manifests_digest(subset),
                    "format_version": snapshot_caps.get("format_version", 1),
                    "n_dialogues": len(subset),
                }
            if self.snapshot:
                # 适配器不支持：快照字段没有值，但对话数与根目录照旧给出
                return {
                    "role": "unsupported",
                    "root": dataset_dir,
                    "manifest_sha256": None,
                    "format_version": None,
                    "n_dialogues": 1 if dialogue_id is not None else len(dialogues),
                }
            return None

        def _compose_metrics(
            subset: list[dict[str, Any]],
            subset_caps: dict[str, bool],
            *,
            n_ignored: int,
            usage: dict[str, int],
            state: Union[dict, None],
            retrieval_meta: dict[str, Any],
            dialogue_id: Union[str, None] = None,
        ) -> dict[str, Any]:
            """组装某一层级的 metrics（数据集级 / 对话级同一套字段与口径）。"""
            out = dict(aggregate(subset, capabilities=subset_caps))
            out.update(
                {
                    "run_id": run_id,
                    "adapter": adapter_name,
                    "dataset": dataset_name,
                    "dataset_path": os.path.abspath(self.dataset_path),
                    "dataset_source": self.dataset_source,
                    "platform": platform_tag,
                    "commit": self.commit,
                    "judge": self.judge_name,
                    "n_ignored": n_ignored,
                    "timestamp": now.isoformat(timespec="seconds"),
                    "limit": self.limit,
                    "shards": shards,
                    "shard_index": self.shard_index,
                    "shard_by": self.shard_by,
                    "adapter_meta": self.adapter_meta,
                    "adapter_describe": adapter_describe,
                    "schema_version": "0.2",
                    "driver_llm": self.driver_llm,
                    "judge_model": self.judge_model,
                    "adapter_version": self.adapter_version,
                    "dataset_fingerprint": dataset_fingerprint(self.dataset_path),
                    # 运行模式：full（默认）/ replay-restore / replay-rebuild；compare
                    # 把 run_mode 当关键维度——回放分不得与全量分被静默对比
                    "run_mode": replay_plan["run_mode"] if replay_plan else "full",
                    # 产物归属：读取方据此定位数据集目录，不必再猜层级
                    "experiment_dir": experiment_dir,
                    "dataset_dir": dataset_dir,
                    # 全流程 token / 调用记账（被测系统上报口径；适配器未上报时为 0，
                    # 报告按「未上报」提示，不当作真实 0 消耗）。这不是指标，不参与
                    # 评级/门禁，只回答"这次评测烧了多少"。
                    "token_usage": usage,
                }
            )
            if dialogue_id is not None:
                out["dialogue_id"] = dialogue_id
            out.update(retrieval_meta)
            if state is not None:
                out["state_snapshot"] = state
            return out

        records: list[dict[str, Any]] = []
        done = 0
        fallbacks_seen = match_fallbacks_start
        for di, dialogue in enumerate(dialogues, 1):
            dialogue_id = str(dialogue.dialogue_id)
            dialogue_path = layout.dialogue_dir(self.out_dir, dataset_name, dialogue_id)
            # runtime/ 始终建出（哪怕没有写入者）：适配器要把自己的运行时数据
            # （日志、追踪、自定义回溯文件等）落进来时不必自己猜目录。
            os.makedirs(
                layout.dialogue_runtime_dir(self.out_dir, dataset_name, dialogue_id),
                exist_ok=True,
            )
            if replay_plan:
                # 回放：恢复该对话的快照（restore 或 rebuild），不 reset、不 ingest
                snapshot_mod.replay_dialogue(self.adapter, replay_plan, dialogue_id)
                ingest_latency_ms = 0.0
                ingest_llm_calls = 0
                ingest_input_tokens = 0
                print(
                    f"[membench] dialogue {di}/{len(dialogues)} {dialogue_id}: "
                    f"从快照恢复（{replay_plan['run_mode']}）"
                )
            else:
                # reset 上下文：只作为「适配器可选提示」暴露（环境变量 + HTTP 桥透传），
                # 框架不读不写适配器的记忆存储——MEMBENCH_RUN_DIR 指本对话目录，
                # 供日志 / 追踪 / 接入侧自定义回溯；框架不内置 reset 前备份。
                os.environ["MEMBENCH_RUN_DIR"] = dialogue_path
                os.environ["MEMBENCH_DIALOGUE_ID"] = dialogue_id
                os.environ["MEMBENCH_DIALOGUE_INDEX"] = str(di)
                self.adapter.reset()
                ingest_latency_ms = 0.0
                ingest_llm_calls = 0
                ingest_input_tokens = 0
                for session in dialogue.sessions:
                    stats = self.adapter.ingest(session)
                    ingest_latency_ms += float(getattr(stats, "latency_ms", 0.0) or 0.0)
                    ingest_llm_calls += int(getattr(stats, "llm_calls", 0) or 0)
                    ingest_input_tokens += int(getattr(stats, "input_tokens", 0) or 0)
            token_usage["ingest_input_tokens"] += ingest_input_tokens
            token_usage["ingest_llm_calls"] += ingest_llm_calls

            if not replay_plan:
                print(
                    f"[membench] dialogue {di}/{len(dialogues)} "
                    f"{dialogue_id}: ingested {len(dialogue.sessions)} session(s), "
                    f"{ingest_latency_ms:.0f} ms"
                )
            active_questions = [
                q for q in dialogue.questions if not getattr(q, "ignore", False)
            ]

            if export_snapshots:
                # 快照打点：该对话 ingest / restore 完成后、第一道 query 之前（回放的入口
                # 状态）。一个对话只导这一份：per-session 导出会得到用不上的中间态。
                manifest = snapshot_mod.new_manifest(
                    run_id=run_id,
                    adapter=adapter_name,
                    adapter_version=self.adapter_version,
                    platform=platform_tag,
                    dataset=dataset_name,
                    dataset_path=os.path.abspath(self.dataset_path),
                    dataset_fingerprint=dataset_fingerprint(self.dataset_path),
                    dialogue_id=dialogue_id,
                    capabilities=snapshot_caps,
                )
                if snapshot_pool is not None:
                    exported = snapshot_pool.export(
                        self.adapter,
                        manifest,
                        dialogue_id=dialogue_id,
                        sessions=dialogue.sessions,
                        n_questions=len(active_questions),
                    )
                    entry = exported["entry"]
                    print(
                        f"[membench] 快照池: {exported['bucket']} "
                        f"（{'复用已有桶' if exported['reused'] else '新写'}，"
                        f"{entry['bytes']} 字节，对话 {dialogue_id}）"
                    )
                    add_bytes = 0 if exported["reused"] else int(entry.get("bytes") or 0)
                else:
                    snapshot_path = layout.dialogue_snapshot_dir(
                        self.out_dir, dataset_name, dialogue_id
                    )
                    entry = snapshot_mod.write_snapshot(
                        snapshot_path,
                        self.adapter,
                        manifest,
                        dialogue_id=dialogue_id,
                        sessions=dialogue.sessions,
                        n_questions=len(active_questions),
                    )
                    snapshot_dirs[dialogue_id] = snapshot_path
                    print(
                        f"[membench] 快照: {snapshot_path}（{entry['bytes']} 字节，"
                        f"sha256 {entry['sha256'][:12]}）"
                    )
                    add_bytes = int(entry.get("bytes") or 0)
                if add_bytes > SNAPSHOT_WARN_BYTES and not snapshot_oversize_warned:
                    snapshot_oversize_warned = True
                    print(
                        f"[membench] ⚠️ 单份快照 {add_bytes / (1 << 20):.0f} MB 超过 "
                        f"{SNAPSHOT_WARN_BYTES >> 20} MB：快照体积随对话数线性增长，"
                        "可用 CLI 的 --no-snapshot 关闭（关掉后改读取侧只能重跑 ingest）"
                    )

            def _eval(q: Any) -> dict[str, Any]:
                # 单题内部计时：latency_ms 是该题 query 调用的墙钟时间，不受并发影响。
                return self._eval_question(
                    dialogue_id,
                    q,
                    ingest_latency_ms,
                    ingest_llm_calls,
                    ingest_input_tokens,
                )

            if self.jobs > 1 and len(active_questions) > 1:
                # 题目级并发；map 保持提交顺序，输出与 jobs=1 一致。
                with ThreadPoolExecutor(max_workers=self.jobs) as pool:
                    dialogue_records = list(pool.map(_eval, active_questions))
            else:
                dialogue_records = [_eval(q) for q in active_questions]

            records.extend(dialogue_records)
            for record in dialogue_records:
                done += 1
                print(
                    f"[membench] [{done}/{n_active}] {record['qid']} "
                    f"({record['category']}) correct={record['judge'].get('correct')} "
                    f"latency={record['latency_ms']:.0f}ms"
                )

            # 逐对话落盘：该对话的题目与指标先独立成文，不等整场跑完
            dialogue_usage: dict[str, int] = {
                "ingest_input_tokens": ingest_input_tokens,
                "ingest_llm_calls": ingest_llm_calls,
                "query_input_tokens": sum(
                    int(r.get("input_tokens", 0) or 0) for r in dialogue_records
                ),
                "query_output_tokens": sum(
                    int(r.get("output_tokens", 0) or 0) for r in dialogue_records
                ),
                "query_llm_calls": sum(
                    int(r.get("llm_calls", 0) or 0) for r in dialogue_records
                ),
                "n_dialogues": 1,
                "n_questions": len(dialogue_records),
            }
            if replay_plan:
                dialogue_usage["ingest_skipped"] = 1
            token_usage["query_input_tokens"] += dialogue_usage["query_input_tokens"]
            token_usage["query_output_tokens"] += dialogue_usage["query_output_tokens"]
            token_usage["query_llm_calls"] += dialogue_usage["query_llm_calls"]
            fallbacks_now = int(getattr(self._evidence_matcher, "fallbacks", 0) or 0)
            dialogue_metrics = _compose_metrics(
                dialogue_records,
                question_capabilities(dialogue.questions),
                n_ignored=sum(
                    1 for q in dialogue.questions if getattr(q, "ignore", False)
                ),
                usage=dialogue_usage,
                state=_snapshot_state(dialogue_id),
                retrieval_meta=_retrieval_meta(fallbacks_now - fallbacks_seen),
                dialogue_id=dialogue_id,
            )
            fallbacks_seen = fallbacks_now
            dialogue_report_dir = layout.dialogue_report_dir(
                self.out_dir, dataset_name, dialogue_id
            )
            _write_run_artifacts(
                dialogue_report_dir, dialogue_records, dialogue_metrics, figures=False
            )
            print(f"[membench] 对话报告: {dialogue_report_dir}")

        # reset 上下文用完即撤（异常中断时残留也无害：下一次 reset 前会被覆写）
        for _key in ("MEMBENCH_RUN_DIR", "MEMBENCH_DIALOGUE_ID", "MEMBENCH_DIALOGUE_INDEX"):
            os.environ.pop(_key, None)

        replay_verify: Union[dict, None] = None
        if verify_source:
            replay_verify = snapshot_mod.verify_replay(verify_source, records)
            if "error" in replay_verify:
                print(f"[membench] ⚠️ 回放保真校验失败：{replay_verify['error']}")
            else:
                print(
                    "[membench] 回放保真校验（对比源 run "
                    f"{replay_verify['source_run']}）：共同题 {replay_verify['n_common']}，"
                    f"答案不一致 {replay_verify['n_answer_diff']}、"
                    f"检索不一致 {replay_verify['n_retrieved_diff']}、"
                    f"判分不一致 {replay_verify['n_correct_diff']}"
                    + (
                        f"（答案一致率 {replay_verify['answer_match_rate']:.1%}）"
                        if replay_verify.get("answer_match_rate") is not None
                        else ""
                    )
                )

        # 数据集级聚合：与逐对话 metrics 同一套函数与字段（只是喂全量记录）
        metrics = _compose_metrics(
            records,
            # 能力标注（has_evidence / has_stale / has_abstention）从题目集扫描
            # 得到（排除 ignore 题），经 aggregate 落账到 metrics.json 顶层，
            # caller 不必再扫 results.jsonl 二次推断。
            question_capabilities(q for d in dialogues for q in d.questions),
            n_ignored=n_ignored,
            usage={
                **token_usage,
                "n_dialogues": len(dialogues),
                "n_questions": n_active,
            },
            state=_snapshot_state(None),
            retrieval_meta=_retrieval_meta(
                int(getattr(self._evidence_matcher, "fallbacks", 0) or 0)
                - match_fallbacks_start
            ),
        )
        if replay_verify is not None:
            metrics["replay_verify"] = replay_verify

        if ledger_path is not None:
            # 分片：数据集级与任务级产物一律由 fuse 补齐——分片产物不是"完整实验"。
            # 上面那份聚合只用于控制台汇总（overall.* 的打印），不落盘。
            layout.update_shard_ledger(
                dataset_dir,
                ledger_header,
                ledger_key,
                {
                    "status": "done",
                    "dialogues": dialogue_ids,
                    "n_questions": n_active,
                    "run_id": run_id,
                    "started_at": shard_started_at,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            print(
                f"[membench] 分片账本: {ledger_path}（{ledger_key} done，"
                f"{len(dialogue_ids)} 个对话 / {n_active} 题）"
            )
            print(
                "[membench] 本分片只落对话目录：数据集级 report/ 与任务级 reports/ "
                "由 fuse 补齐"
            )
        else:
            dataset_report_dir = layout.dataset_report_dir(self.out_dir, dataset_name)
            _write_run_artifacts(dataset_report_dir, records, metrics, figures=True)
            print(f"[membench] 数据集报告: {dataset_report_dir}")

            if self.write_task_reports:
                # 任务级报告 = 数据集级报告内容（md 连带图片 + metrics）；逐题明细留在
                # 数据集目录，不重复落一份可能很大的 results.jsonl。
                task_dir = layout.task_report_dir(self.out_dir)
                copy_report(
                    os.path.join(dataset_report_dir, layout.FILE_REPORT_MD),
                    task_dir,
                    layout.FILE_REPORT_MD,
                    extra_files=[os.path.join(dataset_report_dir, layout.FILE_METRICS)],
                )
                print(f"[membench] 任务级报告: {task_dir}")
        if snapshot_pool is not None:
            pool = snapshot_pool.summary()
            print(
                f"[membench] 快照池: {pool['snapshots_dir']}（{pool['n_buckets']} 个桶，"
                f"复用 {pool['n_reused']} / 新写 {pool['n_new']}）"
                "——同一源回放到另一臂不会新增桶；--replay 传该任务目录或其臂目录可"
                "继续回放（链式）"
            )
        elif export_snapshots:
            print(
                f"[membench] 快照: {len(snapshot_dirs)} 个对话 → {dataset_dir}"
                "（--replay 传该数据集目录或实验目录即可免重跑 ingest）"
            )

        overall = metrics.get("overall", {})

        def _fmt(v: Any) -> str:
            """指标值格式化：None 显式 'n/a'，避免 .3f 报 TypeError。"""
            return f"{float(v):.3f}" if v is not None else "n/a"

        print(
            f"[membench] done. accuracy={_fmt(overall.get('accuracy'))} "
            f"stale_rate={_fmt(overall.get('stale_rate'))} "
            f"evidence_recall={_fmt(overall.get('evidence_recall'))}"
        )
        print(f"[membench] token 消耗（被测系统上报）: {format_token_usage(token_usage)}")
        print(f"[membench] dataset dir: {dataset_dir}")
        return dataset_dir
