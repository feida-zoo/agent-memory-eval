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

from membench.data.schema import load_dataset
from membench.judges import get_judge
from membench.metrics import aggregate
from membench.metrics.text import bleu1, recall_at_k, token_f1
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
    ) -> None:
        if jobs < 1:
            raise ValueError(f"jobs 必须 >= 1，当前为 {jobs}")
        self.jobs = int(jobs)
        self.adapter = adapter
        self.dataset_path = dataset_path
        # 默认输出到 skill 的 output/runs/；开发环境（无 SKILL.md）则退到工程根 runs/
        if out_dir is None:
            pkg_dir = os.path.dirname(os.path.abspath(__file__))  # membench/membench/
            proj_root = os.path.dirname(pkg_dir)  # membench/（工程根）
            skill_root = os.path.dirname(proj_root)  # skill 根目录（或开发环境的 project 父目录）
            # 检测是否在 skill 环境：skill 根目录应有 SKILL.md
            if os.path.isfile(os.path.join(skill_root, "SKILL.md")):
                self.out_dir = os.path.join(skill_root, "output", "runs")
            else:
                # 开发/独立环境：退到工程根 runs/
                self.out_dir = os.path.join(proj_root, "runs")
        else:
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
        adapter_name = str(getattr(self.adapter, "name", self.adapter.__class__.__name__))
        dataset_name = os.path.splitext(os.path.basename(self.dataset_path))[0]
        platform_tag = self.platform or "default"
        now = datetime.now()
        # commit 短哈希进 run_id，避免矩阵评测中同 adapter+platform 同秒撞目录
        commit_tag = f"_{str(self.commit)[:8]}" if self.commit else ""
        run_id = (
            f"{adapter_name}_{platform_tag}{commit_tag}_"
            f"{dataset_name}_{now.strftime('%Y%m%d_%H%M%S')}"
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
            }
        )

        results_path = os.path.join(run_dir, "results.jsonl")
        with open(results_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        render_run_report(run_dir, records, metrics)

        overall = metrics.get("overall", {})
        print(
            f"[membench] done. accuracy={overall.get('accuracy', 0.0):.3f} "
            f"stale_rate={overall.get('stale_rate', 0.0):.3f} "
            f"evidence_recall={overall.get('evidence_recall', 0.0):.3f}"
        )
        print(f"[membench] run dir: {run_dir}")
        return run_dir
