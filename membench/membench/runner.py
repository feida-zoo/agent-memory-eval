"""Runner：对单个记忆系统执行评测并写出实验目录下的产物。

流程（ingest 阶段始终串行，print 打印进度）：
    load_dataset -> 每个 dialogue: adapter.reset() -> 逐 session adapter.ingest()
    -> 每题 adapter.query() + 文本/检索指标 + judge.grade()
    -> 逐对话聚合写对话级 report/ -> 全对话聚合写数据集级 report/（含图）
    -> write_task_reports 时把数据集级报告复制到 <实验目录>/reports/

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
    <实验目录>/runs/<dataset>/<对话目录>/memory_snapshot/  # --snapshot 时（含 manifest.json）
    <实验目录>/reports/                                    # 任务级（数据集级报告副本）
"""

from __future__ import annotations

import dataclasses
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Union

from membench import layout
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


def _select_questions(
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
        snapshot: True 时在每个对话 ingest 完成后（第一道 query 之前）调用适配器
            的 ``snapshot()`` 把状态导出到该对话的 ``memory_snapshot/`` 并写
            ``manifest.json``——之后改读取侧代码可用 ``replay`` 免重跑 ingest。
            与 ``replay`` 互斥；适配器未声明快照能力则在开跑前拒绝。
        replay: 实验目录或数据集目录（``<实验目录>/runs/<dataset>``）；给定时
            跳过 reset+ingest，按对话 ``restore()``（或 ``rebuild()``）后只跑
            query。清单齐备性 / 数据集指纹 / 对话覆盖 / 能力 / 模式都在**开跑前**
            校验，失败即停且不产生任何产物。
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
        snapshot: bool = False,
        replay: Union[str, None] = None,
        replay_mode: str = "auto",
        replay_verify_source: Union[str, None] = None,
    ) -> None:
        if jobs < 1:
            raise ValueError(f"jobs 必须 >= 1，当前为 {jobs}")
        if snapshot and replay:
            raise ValueError("snapshot 与 replay 互斥（导快照 / 用快照回放，二选一）")
        if replay_verify_source and not replay:
            raise ValueError("replay_verify_source 只在 replay 模式下有意义（需同时给 replay）")
        if replay_mode not in ("auto", "restore", "rebuild"):
            raise ValueError(f"replay_mode 必须是 auto / restore / rebuild，当前为 {replay_mode!r}")
        self.snapshot = bool(snapshot)
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
        回放与快照能力的校验都在建目录之前完成，失败即 SystemExit、同样不留半成品。
        本实验目录下该数据集已有产物时同样在写盘前拒绝（不静默覆盖上一次的结果）。
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
            dialogues = _select_questions(
                dialogues, shards, self.shard_index, self.limit, self.shard_by
            )

        adapter_name = str(getattr(self.adapter, "name", self.adapter.__class__.__name__))
        dataset_name = os.path.splitext(os.path.basename(self.dataset_path))[0]
        # 写入前的覆盖拒绝：同一实验目录下该数据集已有产物时，本次会静默盖掉上一次的
        # metrics / 逐题明细（分片误用同一 --report-out 尤其容易丢数据），先拦下。
        existing_metrics = os.path.join(
            layout.dataset_report_dir(self.out_dir, dataset_name), layout.FILE_METRICS
        )
        if os.path.isfile(existing_metrics):
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
        snapshot_caps: dict[str, Any] = {}
        if self.snapshot:
            snapshot_caps = snapshot_mod.read_capabilities(self.adapter)
            if not snapshot_caps.get("supported"):
                raise SystemExit(
                    "该适配器未声明快照能力，无法 --snapshot："
                    "请实现 snapshot()（可选 restore()/rebuild()）并让 "
                    "snapshot_capabilities() 返回 supported=True；HTTP 桥同理"
                )

        os.makedirs(self.out_dir, exist_ok=True)
        experiment_dir = os.path.abspath(self.out_dir)
        dataset_dir = os.path.abspath(layout.dataset_dir(self.out_dir, dataset_name))
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

        # 适配器可选的 describe() -> dict：记录其配置/版本等自描述信息；
        # 调用失败记 None，不中断评测。放在开跑前，对话级与数据集级元数据取同一份。
        adapter_describe: Any = None
        describe_fn = getattr(self.adapter, "describe", None)
        if callable(describe_fn):
            try:
                adapter_describe = describe_fn()
            except Exception:  # noqa: BLE001 - 自描述信息失败不应中断评测
                adapter_describe = None

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
        # 快照目录登记：导出时逐对话登记，回放时来自计划；state_snapshot 依据
        snapshot_dirs: dict[str, str] = dict(replay_plan["snapshots"]) if replay_plan else {}

        def _retrieval_meta(fallbacks: int) -> dict[str, Any]:
            """检索命中口径落账（substring / overlap / llm；阈值与回退数按需）。"""
            meta: dict[str, Any] = {"retrieval_match": self._evidence_matcher.name}
            threshold = getattr(self._evidence_matcher, "threshold", None)
            if threshold is not None:
                meta["retrieval_match_threshold"] = float(threshold)
            if fallbacks:
                meta["retrieval_match_fallbacks"] = fallbacks
            return meta

        def _snapshot_state(dialogue_id: Union[str, None]) -> Union[dict, None]:
            """state_snapshot 块：数据集级含全部对话，对话级只含自己。"""
            if not snapshot_dirs:
                return None
            if dialogue_id is None:
                subset = snapshot_dirs
            else:
                path = snapshot_dirs.get(dialogue_id)
                if path is None:
                    return None
                subset = {dialogue_id: path}
            if replay_plan:
                return {
                    "role": "replayed",
                    "root": replay_plan["root"],
                    "manifest_sha256": snapshot_mod.manifests_digest(subset),
                    "format_version": replay_plan["format_version"],
                    "source_run": replay_plan["source_run"],
                    "replay_mode": replay_plan["mode"],
                    "n_dialogues": len(subset),
                }
            return {
                "role": "exported",
                "root": dataset_dir,
                "manifest_sha256": snapshot_mod.manifests_digest(subset),
                "format_version": snapshot_caps.get("format_version", 1),
                "n_dialogues": len(subset),
            }

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
            # runtime/ 始终建出（哪怕没有写入者）：适配器 / 桥要把运行时数据
            # （如 reset 前的记忆备份落 runtime/reset_backup/）时不必自己猜目录。
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
                # 要不要在清空前快照、落到哪里，由接入侧决定。
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

            if self.snapshot:
                # 快照打点：该对话 ingest 完成后、第一道 query 之前（回放的入口状态）
                snapshot_path = layout.dialogue_snapshot_dir(
                    self.out_dir, dataset_name, dialogue_id
                )
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
        if self.snapshot:
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
