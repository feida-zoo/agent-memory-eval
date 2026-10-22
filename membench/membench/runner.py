"""Runner：对单个记忆系统执行评测并写出 run 目录。

流程（ingest 阶段始终串行，print 打印进度）：
    load_dataset -> 每个 dialogue: adapter.reset() -> 逐 session adapter.ingest()
    -> 每题 adapter.query() + 文本/检索指标 + judge.grade()
    -> membench.metrics.aggregate(records) -> 补元数据
    -> 写 results.jsonl / metrics.json -> report.py 生成 report.md 与图表

并发：query+judge+证据匹配 阶段在 ``jobs>1`` 时用 ThreadPoolExecutor 题目级并发
（对话时序和记忆状态敏感的 ingest 阶段必须保持串行）；收集结果后按原始题目
顺序排序落盘，results.jsonl 的记录顺序与 jobs=1 完全一致。
**jobs>1 要求 adapter.query 与 judge 线程安全**（evidence-match 匹配器自身
线程安全，LLM 口径的逐题调用也随之并发）；内置 mock adapter 和
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
from membench.data.schema import (
    dataset_fingerprint,
    load_dataset,
    question_capabilities,
)
from membench.judges import get_judge
from membench.metrics import aggregate
from membench.metrics.evidence import SubstringMatcher
from membench.metrics.text import bleu1, token_f1
from membench.paths import runs_dir
from membench.report import format_token_usage, render_run_report
from membench import snapshot as snapshot_mod


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
        report_out: 非空时，评测结束后把本次实验产物 materialize 到
            ``<report_out>/eval_<adapter>_<platform>_<ts>/``（报告 + metrics +
            结果 + run 副本），并把 ``experiment_dir`` 写回 metrics.json；
            缺省 None 时不额外落盘。
        snapshot_out: 非空时，每个对话 ingest 完成后（query 之前）调用适配器的
            ``snapshot()`` 把状态导出到该目录，并写 ``manifest.json``——之后改
            读取侧代码可用 ``replay`` 免重跑 ingest。与 ``replay`` 互斥。
        replay: 快照根目录；给定时跳过 reset+ingest，按对话 ``restore()``（或
            ``rebuild()``）后只跑 query。数据集指纹不符、快照缺对话、适配器不
            支持快照都会在**开跑前**拒绝。
        replay_mode: ``auto``（默认，优先 restore，格式不兼容时退回 rebuild）/
            ``restore`` / ``rebuild``。
        replay_verify_source: 给定源 run 目录时，回放结束与之逐题比对
            （answer / retrieved / 判分），结果写进 metrics 与报告——这就是
            "回放误差地板"，读取侧 A/B 的差值必须大于它才有意义。
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
        shards: Union[int, str] = 1,
        shard_index: int = 0,
        shard_by: Union[str, None] = None,
        adapter_meta: Union[str, None] = None,
        driver_llm: Union[str, None] = None,
        judge_model: Union[str, None] = None,
        evidence_matcher: Any = None,
        adapter_version: Union[str, None] = None,
        report_out: Union[str, None] = None,
        snapshot_out: Union[str, None] = None,
        replay: Union[str, None] = None,
        replay_mode: str = "auto",
        replay_verify_source: Union[str, None] = None,
    ) -> None:
        if jobs < 1:
            raise ValueError(f"jobs 必须 >= 1，当前为 {jobs}")
        if snapshot_out and replay:
            raise ValueError("snapshot_out 与 replay 互斥（导快照 / 用快照回放，二选一）")
        if replay_verify_source and not replay:
            raise ValueError("replay_verify_source 只在 replay 模式下有意义（需同时给 replay）")
        if replay_mode not in ("auto", "restore", "rebuild"):
            raise ValueError(f"replay_mode 必须是 auto / restore / rebuild，当前为 {replay_mode!r}")
        self.snapshot_out = snapshot_out
        self.replay = replay
        self.replay_mode = replay_mode
        self.replay_verify_source = replay_verify_source
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
        """执行评测，返回 run 目录路径；分片无可执行内容（空数据集/索引越界）时返回 None。"""
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
            print("[membench] 数据集为空（无可分片的对话/题目），本进程跳过、不产生 run 产物。")
            return None
        if self.shard_index >= shards:
            print(
                f"[membench] 分片索引 {self.shard_index} 超出有效分片数 {shards}，"
                "本进程无事可做，跳过（不产生 run 产物）。"
            )
            return None
        if shards > 1 or self.limit is not None:
            dialogues = _select_questions(
                dialogues, shards, self.shard_index, self.limit, self.shard_by
            )
        # 回放跑前校验（失败即停，不留下半成品 run 目录）：清单/指纹/覆盖/模式协商
        replay_plan: Union[dict, None] = None
        if self.replay:
            replay_plan = snapshot_mod.plan_replay(
                self.adapter,
                self.replay,
                dataset_fingerprint=dataset_fingerprint(self.dataset_path),
                dialogue_ids=[str(d.dialogue_id) for d in dialogues],
                requested_mode=self.replay_mode,
            )
            print(
                f"[membench] 快照回放：{replay_plan['run_mode']}（源 run "
                f"{replay_plan['source_run']}，格式 v{replay_plan['format_version']}，"
                f"{len(dialogues)} 个对话）——写入侧固定，结论只覆盖读取/回答层"
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
        if shards > 1:
            mode = "dshard" if self.shard_by == "dialogue" else "shard"
            shard_tag = f"_{mode}{self.shard_index}of{shards}"
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
        # 快照清单（--snapshot-out）：逐对话增量登记，跑完置 complete=true；
        # 覆盖旧清单时只提示不报错（快照可再生，重 ingest 才是贵的）。
        snapshot_manifest: Union[dict, None] = None
        snapshot_root: Union[str, None] = None
        if self.snapshot_out:
            snapshot_root = os.path.abspath(self.snapshot_out)
            caps = snapshot_mod.read_capabilities(self.adapter)
            if not caps.get("supported"):
                raise SystemExit(
                    "该适配器未声明快照能力，无法 --snapshot-out："
                    "请实现 snapshot()（可选 restore()/rebuild()）并让 "
                    "snapshot_capabilities() 返回 supported=True；HTTP 桥同理"
                )
            previous = os.path.join(snapshot_root, snapshot_mod.MANIFEST_NAME)
            if os.path.isfile(previous):
                try:
                    with open(previous, encoding="utf-8") as handle:
                        old = json.load(handle)
                    print(
                        f"[membench] 提示: 覆盖已有快照清单（旧 run_id={old.get('run_id')}，"
                        f"{len(old.get('dialogues') or {})} 个对话）: {previous}"
                    )
                except ValueError:
                    print(f"[membench] 提示: 覆盖无法解析的旧快照清单: {previous}")
            snapshot_manifest = snapshot_mod.new_manifest(
                run_id=run_id,
                adapter=adapter_name,
                adapter_version=self.adapter_version,
                platform=platform_tag,
                dataset=dataset_name,
                dataset_path=os.path.abspath(self.dataset_path),
                dataset_fingerprint=dataset_fingerprint(self.dataset_path),
                shard_by=self.shard_by,
                shards=shards,
                shard_index=self.shard_index,
                capabilities=caps,
            )
            snapshot_mod.save_manifest(snapshot_root, snapshot_manifest)

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
        match_fallbacks_start = int(
            getattr(self._evidence_matcher, "fallbacks", 0) or 0
        )

        for di, dialogue in enumerate(dialogues, 1):
            if replay_plan:
                # 回放：恢复该对话的快照（restore 或 rebuild），不 reset、不 ingest
                snapshot_mod.replay_dialogue(self.adapter, replay_plan, str(dialogue.dialogue_id))
                ingest_latency_ms = 0.0
                ingest_llm_calls = 0
                ingest_input_tokens = 0
                print(
                    f"[membench] dialogue {di}/{len(dialogues)} {dialogue.dialogue_id}: "
                    f"从快照恢复（{replay_plan['run_mode']}）"
                )
            else:
                # reset 上下文：只作为「适配器可选提示」暴露（环境变量 + HTTP 桥透传），
                # 框架不读不写适配器的记忆存储——要不要在清空前快照、落到哪里，由接入侧
                # 决定（见 examples/http_bridge_node 的 BRIDGE_RESET_BACKUP）。
                os.environ["MEMBENCH_RUN_DIR"] = run_dir
                os.environ["MEMBENCH_DIALOGUE_ID"] = str(dialogue.dialogue_id)
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
                    f"{dialogue.dialogue_id}: ingested {len(dialogue.sessions)} session(s), "
                    f"{ingest_latency_ms:.0f} ms"
                )
            active_questions = [
                q for q in dialogue.questions if not getattr(q, "ignore", False)
            ]

            # 快照打点：该对话 ingest 完成后、第一道 query 之前（回放的入口状态）
            if snapshot_manifest is not None and snapshot_root is not None:
                entry = snapshot_mod.write_snapshot(
                    snapshot_root,
                    self.adapter,
                    snapshot_manifest,
                    dialogue_id=str(dialogue.dialogue_id),
                    sessions=dialogue.sessions,
                    n_questions=len(active_questions),
                )
                snapshot_mod.save_manifest(snapshot_root, snapshot_manifest)
                print(
                    f"[membench] 快照: {entry['dir']}/（{entry['bytes']} 字节，"
                    f"sha256 {entry['sha256'][:12]}）"
                )

            def _eval(q: Any) -> dict[str, Any]:
                # 单题内部计时：latency_ms 是该题 query 调用的墙钟时间，不受并发影响。
                return self._eval_question(
                    str(dialogue.dialogue_id),
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

            for record in dialogue_records:
                records.append(record)
                token_usage["query_input_tokens"] += int(record.get("input_tokens", 0) or 0)
                token_usage["query_output_tokens"] += int(record.get("output_tokens", 0) or 0)
                token_usage["query_llm_calls"] += int(record.get("llm_calls", 0) or 0)
                done += 1
                print(
                    f"[membench] [{done}/{n_active}] {record['qid']} "
                    f"({record['category']}) correct={record['judge'].get('correct')} "
                    f"latency={record['latency_ms']:.0f}ms"
                )

        # reset 上下文用完即撤（异常中断时残留也无害：下一次 reset 前会被覆写）
        for _key in ("MEMBENCH_RUN_DIR", "MEMBENCH_DIALOGUE_ID", "MEMBENCH_DIALOGUE_INDEX"):
            os.environ.pop(_key, None)

        # 快照清单收尾：置 complete=true（回放前会校验覆盖是否齐全）
        snapshot_meta: Union[dict, None] = None
        if snapshot_manifest is not None and snapshot_root is not None:
            snapshot_manifest["complete"] = True
            manifest_sha = snapshot_mod.save_manifest(snapshot_root, snapshot_manifest)
            snapshot_meta = {
                "role": "exported",
                "root": snapshot_root,
                "manifest_sha256": manifest_sha,
                "format_version": (snapshot_manifest.get("capabilities") or {}).get(
                    "format_version", 1
                ),
                "n_dialogues": len(snapshot_manifest["dialogues"]),
            }
            print(
                f"[membench] 快照清单: {snapshot_root}/{snapshot_mod.MANIFEST_NAME}"
                f"（{snapshot_meta['n_dialogues']} 个对话，格式 v{snapshot_meta['format_version']}）"
            )
        if replay_plan:
            snapshot_meta = {
                "role": "replayed",
                "root": replay_plan["root"],
                "manifest_sha256": replay_plan["manifest_sha256"],
                "format_version": replay_plan["format_version"],
                "source_run": replay_plan["source_run"],
                "replay_mode": replay_plan["mode"],
                "n_dialogues": len(dialogues),
            }
        replay_verify: Union[dict, None] = None
        if self.replay_verify_source:
            replay_verify = snapshot_mod.verify_replay(self.replay_verify_source, records)
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

        # 适配器可选的 describe() -> dict：记录其配置/版本等自描述信息；
        # 调用失败记 None，不中断评测。
        adapter_describe: Any = None
        describe_fn = getattr(self.adapter, "describe", None)
        if callable(describe_fn):
            try:
                adapter_describe = describe_fn()
            except Exception:  # noqa: BLE001 - 自描述信息失败不应中断评测
                adapter_describe = None

        # 能力标注（has_evidence / has_stale / has_abstention）从题目集扫描
        # 得到（排除 ignore 题），经 aggregate 落账到 metrics.json 顶层，
        # caller 不必再扫 results.jsonl 二次推断。
        capabilities = question_capabilities(
            q for d in dialogues for q in d.questions
        )

        metrics = dict(aggregate(records, capabilities=capabilities))
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
                # 全流程 token / 调用记账（被测系统上报口径；适配器未上报时为 0，
                # 报告按「未上报」提示，不当作真实 0 消耗）。这不是指标，不参与
                # 评级/门禁，只回答"这次评测烧了多少"。
                "token_usage": {
                    **token_usage,
                    "n_dialogues": len(dialogues),
                    "n_questions": n_active,
                },
            }
        )
        # 检索命中口径落账（substring / overlap / llm；阈值与回退数按需）
        retrieval_meta: dict[str, Any] = {"retrieval_match": self._evidence_matcher.name}
        match_threshold = getattr(self._evidence_matcher, "threshold", None)
        if match_threshold is not None:
            retrieval_meta["retrieval_match_threshold"] = float(match_threshold)
        match_fallbacks = (
            int(getattr(self._evidence_matcher, "fallbacks", 0) or 0)
            - match_fallbacks_start
        )
        if match_fallbacks:
            retrieval_meta["retrieval_match_fallbacks"] = match_fallbacks
        metrics.update(retrieval_meta)
        if snapshot_meta is not None:
            metrics["state_snapshot"] = snapshot_meta
        if replay_verify is not None:
            metrics["replay_verify"] = replay_verify

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

        def _fmt(v: Any) -> str:
            """指标值格式化：None 显式 'n/a'，避免 .3f 报 TypeError。"""
            return f"{float(v):.3f}" if v is not None else "n/a"

        print(
            f"[membench] done. accuracy={_fmt(overall.get('accuracy'))} "
            f"stale_rate={_fmt(overall.get('stale_rate'))} "
            f"evidence_recall={_fmt(overall.get('evidence_recall'))}"
        )
        print(f"[membench] token 消耗（被测系统上报）: {format_token_usage(token_usage)}")
        print(f"[membench] run dir: {run_dir}")
        return run_dir
