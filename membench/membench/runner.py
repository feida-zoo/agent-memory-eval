"""Runner：对单个记忆系统执行评测并写出 run 目录。

流程（ingest 阶段始终串行，print 打印进度）：
    load_dataset -> 每个 dialogue: adapter.reset() -> 逐 session adapter.ingest()
    -> 每题 adapter.query() + 文本/检索指标 + judge.grade()
    -> membench.metrics.aggregate(records) -> 补元数据
    -> 写 results.jsonl / metrics.json -> report.py 生成 report.md 与图表

并发：query+judge 阶段在 ``jobs>1`` 时用 ThreadPoolExecutor 题目级并发
（对话时序和记忆状态敏感的 ingest 阶段必须保持串行）；收集结果后按原始题目
顺序排序落盘，results.jsonl 的记录顺序与 jobs=1 完全一致。
**jobs>1 要求 adapter.query 与 judge 线程安全**；内置 mock adapter 和
LLMJudge（urllib 无状态调用）均视为线程安全。

题目级忽略标记（ignore=true）的题整条跳过：不调用 adapter.query、不写入
results.jsonl（不浪费 LLM 调用，metrics 聚合分母口径自然正确）；忽略总数
记入 metrics.json 的 ``n_ignored`` 元信息，report.md 评测配置处注明。

run 目录结构（SPEC 第 6 节）：
    {out_dir}/{run_id}/results.jsonl, metrics.json, report.md,
    fig_category_accuracy.png, fig_latency_hist.png
"""

from __future__ import annotations

import dataclasses
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Union

from membench import layout
from membench.data.schema import dataset_fingerprint, load_dataset
from membench.judges import get_judge
from membench.metrics import aggregate
from membench.metrics.text import bleu1, recall_at_k, token_f1
from membench.paths import runs_dir
from membench.report import render_run_report


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
        out_dir: run 目录的父目录，默认 "runs"。
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
        shards: 分片总数（默认 1 不分片）；与 limit 组合时先分片再截断。
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
        report_out: 非空时，评测结束后把本次实验产物 materialize 到
            ``<report_out>/eval_<adapter>_<platform>_<ts>/``（报告 + metrics +
            结果 + run 副本），并把 ``experiment_dir`` 写回 metrics.json；
            缺省 None 时不额外落盘。
    """

    def __init__(
        self,
        adapter: Any,
        dataset_path: str,
        judge: Union[str, Any] = "heuristic",
        out_dir: Union[str, None] = None,
        platform: Union[str, None] = None,
        commit: Union[str, None] = None,
        dataset_source: Union[str, None] = None,
        jobs: int = 1,
        limit: Union[int, None] = None,
        shards: int = 1,
        shard_index: int = 0,
        shard_by: Union[str, None] = None,
        adapter_meta: Union[str, None] = None,
        driver_llm: Union[str, None] = None,
        judge_model: Union[str, None] = None,
        adapter_version: Union[str, None] = None,
        report_out: Union[str, None] = None,
    ) -> None:
        if jobs < 1:
            raise ValueError(f"jobs 必须 >= 1，当前为 {jobs}")
        self.jobs = int(jobs)
        if limit is not None and limit < 1:
            raise ValueError(f"limit 必须 >= 1，当前为 {limit}")
        if shards < 1:
            raise ValueError(f"shards 必须 >= 1，当前为 {shards}")
        if not 0 <= shard_index < shards:
            raise ValueError(f"shard_index 必须满足 0 <= shard_index < shards，当前为 {shard_index}/{shards}")
        self.shard_by = shard_by or "question"
        if self.shard_by not in ("question", "dialogue"):
            raise ValueError(f"shard_by 必须是 'question' 或 'dialogue'，当前为 {self.shard_by!r}")
        self.limit = limit
        self.shards = int(shards)
        self.shard_index = int(shard_index)
        self.adapter_meta = adapter_meta
        self.adapter = adapter
        self.dataset_path = dataset_path
        self.driver_llm = driver_llm
        self.adapter_version = adapter_version
        self.report_out = report_out
        # 默认 run 落 paths.runs_dir()（skill 环境 output/runs/，开发环境工程根 runs/）
        self.out_dir = out_dir if out_dir is not None else runs_dir()
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

    # ------------------------------------------------------------------
    def _eval_question(
        self,
        dialogue_id: str,
        question: Any,
        ingest_latency_ms: float,
        ingest_llm_calls: int,
    ) -> dict[str, Any]:
        """评测单题，返回 SPEC 3.4 的记录行。"""
        result = self.adapter.query(question.question)

        answer = str(getattr(result, "answer", "") or "")
        retrieved = [str(x) for x in (getattr(result, "retrieved", None) or [])]
        evidence = [str(x) for x in (getattr(question, "evidence", None) or [])]
        gold_answer = str(getattr(question, "gold_answer", "") or "")

        # evidence_hit: 任一 evidence 子串出现在任一 retrieved 条目中
        evidence_hit = bool(evidence) and any(ev in item for item in retrieved for ev in evidence)

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
            # abstention 题无 evidence，recall@5 无意义，记 0.0（且不调用以免除零）
            "recall@5": float(recall_at_k(evidence, retrieved, 5)) if evidence else 0.0,
            "judge": _judge_result_to_dict(judge_result),
            "latency_ms": float(getattr(result, "latency_ms", 0.0) or 0.0),
            "input_tokens": int(getattr(result, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(result, "output_tokens", 0) or 0),
            "llm_calls": int(getattr(result, "llm_calls", 0) or 0),
            "ingest_latency_ms": float(ingest_latency_ms),
            "ingest_llm_calls": int(ingest_llm_calls),
        }

    # ------------------------------------------------------------------
    def run(self) -> str:
        """执行评测，返回 run 目录路径。"""
        dialogues = load_dataset(self.dataset_path)
        if self.shards > 1 or self.limit is not None:
            dialogues = _select_questions(
                dialogues, self.shards, self.shard_index, self.limit, self.shard_by
            )
        adapter_name = str(getattr(self.adapter, "name", self.adapter.__class__.__name__))
        dataset_name = os.path.splitext(os.path.basename(self.dataset_path))[0]
        platform_tag = self.platform or "default"
        now = datetime.now()
        # commit 短哈希进 run_id，避免矩阵评测中同 adapter+platform 同秒撞目录
        commit_tag = f"_{str(self.commit)[:8]}" if self.commit else ""
        # limit / shard 进 run_id，避免冒烟或分片结果被误当全量结果对比；
        # dialogue 分片用 _dshard 标记，与题目级分片区分
        limit_tag = f"_limit{self.limit}" if self.limit is not None else ""
        if self.shards > 1:
            mode = "dshard" if self.shard_by == "dialogue" else "shard"
            shard_tag = f"_{mode}{self.shard_index}of{self.shards}"
        else:
            shard_tag = ""
        run_id = (
            f"{adapter_name}_{platform_tag}{commit_tag}_"
            f"{dataset_name}{shard_tag}{limit_tag}_{now.strftime('%Y%m%d_%H%M%S')}"
        )
        run_dir = os.path.join(self.out_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        total = sum(len(d.questions) for d in dialogues)
        # ignore=true 的题整条跳过（不查询、不计分），n_active 为实际评测题数。
        n_ignored = sum(
            1 for d in dialogues for q in d.questions if getattr(q, "ignore", False)
        )
        n_active = total - n_ignored
        print(f"[membench] run_id={run_id}")
        print(f"[membench] dialogues={len(dialogues)} questions={total} judge={self.judge_name}")
        if n_ignored:
            print(f"[membench] 已忽略 {n_ignored} 道标记题（ignore=true）")

        records: list[dict[str, Any]] = []
        done = 0
        for di, dialogue in enumerate(dialogues, 1):
            self.adapter.reset()
            ingest_latency_ms = 0.0
            ingest_llm_calls = 0
            for session in dialogue.sessions:
                stats = self.adapter.ingest(session)
                ingest_latency_ms += float(getattr(stats, "latency_ms", 0.0) or 0.0)
                ingest_llm_calls += int(getattr(stats, "llm_calls", 0) or 0)

            print(
                f"[membench] dialogue {di}/{len(dialogues)} "
                f"{dialogue.dialogue_id}: ingested {len(dialogue.sessions)} session(s), "
                f"{ingest_latency_ms:.0f} ms"
            )
            active_questions = [
                q for q in dialogue.questions if not getattr(q, "ignore", False)
            ]

            def _eval(q: Any) -> dict[str, Any]:
                # 单题内部计时：latency_ms 是该题 query 调用的墙钟时间，不受并发影响。
                return self._eval_question(
                    str(dialogue.dialogue_id), q, ingest_latency_ms, ingest_llm_calls
                )

            if self.jobs > 1 and len(active_questions) > 1:
                # 题目级并发；map 保持提交顺序，输出与 jobs=1 一致。
                with ThreadPoolExecutor(max_workers=self.jobs) as pool:
                    dialogue_records = list(pool.map(_eval, active_questions))
            else:
                dialogue_records = [_eval(q) for q in active_questions]

            for record in dialogue_records:
                records.append(record)
                done += 1
                print(
                    f"[membench] [{done}/{n_active}] {record['qid']} "
                    f"({record['category']}) correct={record['judge'].get('correct')} "
                    f"latency={record['latency_ms']:.0f}ms"
                )

        # 适配器可选的 describe() -> dict：记录其配置/版本等自描述信息；
        # 调用失败记 None，不中断评测。
        adapter_describe: Any = None
        describe_fn = getattr(self.adapter, "describe", None)
        if callable(describe_fn):
            try:
                adapter_describe = describe_fn()
            except Exception:  # noqa: BLE001 - 自描述信息失败不应中断评测
                adapter_describe = None

        metrics = dict(aggregate(records))
        metrics.update(
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
                "shards": self.shards,
                "shard_index": self.shard_index,
                "shard_by": self.shard_by,
                "adapter_meta": self.adapter_meta,
                "adapter_describe": adapter_describe,
                "schema_version": "0.2",
                "driver_llm": self.driver_llm,
                "judge_model": self.judge_model,
                "adapter_version": self.adapter_version,
                "dataset_fingerprint": dataset_fingerprint(self.dataset_path),
            }
        )

        results_path = os.path.join(run_dir, "results.jsonl")
        with open(results_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        render_run_report(run_dir, records, metrics)

        if self.report_out:
            exp_dir = layout.experiment_path(
                self.report_out,
                layout.subdir_for(adapter_name, platform_tag, layout.ts_now(now)),
            )
            layout.materialize_run_experiment(exp_dir, run_dir)
            # 实验目录自描述：把归属目录写回 metrics.json（run 与实验目录各存一份）
            metrics["experiment_dir"] = exp_dir
            with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            print(f"[membench] 实验目录: {exp_dir}")

        overall = metrics.get("overall", {})
        print(
            f"[membench] done. accuracy={overall.get('accuracy', 0.0):.3f} "
            f"stale_rate={overall.get('stale_rate', 0.0):.3f} "
            f"evidence_recall={overall.get('evidence_recall', 0.0):.3f}"
        )
        print(f"[membench] run dir: {run_dir}")
        return run_dir
