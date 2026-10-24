"""分片融合：把同一评测的多个分片产物按对话粒度合并成一个完整实验目录。

``membench run --shards N --shard-index I`` 产生的各分片产物只含本片的对话 / 题目，
用 ``membench fuse --runs dir1,dir2,... --out DIR`` 融合；入参既可以是实验目录，也可以
是数据集目录（``<实验目录>/runs/<dataset>``）。

- 校验各分片的 adapter / dataset / shards / shard_by / limit 一致，且 shard_index 恰好
  覆盖 0..N-1；判分与可比性元数据（judge / judge_model / driver_llm / adapter_version /
  adapter_meta / dataset_fingerprint / retrieval_match / run_mode / schema_version）
  也必须一致，防止混片（含"全量分片与 --limit 冒烟分片混融"这种假全量）；
- 在**对话粒度**合并各分片的逐对话 ``report/results.jsonl``：题目级分片下同一对话的
  记录来自多个分片，对话级分片下同一对话只出现在一片；源数据集可读时按数据集里的
  对话 / 题目顺序重排，使融合结果与不分片直接跑逐题对齐；
- 复用 ``metrics.summary.aggregate`` 重新计算数据集级与逐对话指标（能力标注按数据集
  扫描，与 run 口径一致），复用 report 渲染数据集级报告（md + 两图）与逐对话报告；
- 门禁判定（stale 一票否决）对融合结果同样生效：存在 knowledge_update 题
  且 stale_rate 超过阈值（与 evaluate 的办公域门禁同阈值）即 FAIL。

``out_dir`` 是**完整实验目录**：``runs/<dataset>/{<对话>/report/...,report/...}`` +
``reports/``（任务级报告，同 Runner 的 write_task_reports 形态）。metrics 标记
``"fused": true`` 及来源分片列表 ``"fused_from"``，并透传可比性元数据（adapter_version /
judge_model / driver_llm / dataset_fingerprint / retrieval_match / adapter_meta /
adapter_describe / schema_version）与冒烟范围 ``limit``——融合产物可直接参与
``compare --require-comparable``：``--limit`` 冒烟分片的融合产物带 ``limitN`` 与
``fused`` 标记，不会被当成全量融合产物。
各分片的 ``token_usage`` 账本逐片相加（任一分片未落账则融合产物不落账，避免"部分求和"
被读成全量消耗）。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from membench import layout
from membench.compare import _load_metrics, resolve_dataset_dir
from membench.data.schema import load_dataset, question_capabilities
from membench.evaluate import STALE_GATE_THRESHOLD
from membench.metrics import aggregate
from membench.report import copy_report, render_dataset_report, render_dialogue_report


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _err(msg: str) -> ValueError:
    return ValueError(f"fuse 校验失败: {msg}")


def fuse_gate(
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
    threshold: float = STALE_GATE_THRESHOLD,
) -> dict[str, Any]:
    """融合结果的 stale 门禁（一票否决，与 evaluate 办公域门禁同阈值）。

    融合结果中存在 knowledge_update 题时：stale_rate > threshold -> FAIL
    （退出码 1）；无 knowledge_update 题、或 stale_rate 未计算（``None``，
    如数据集缺 gold_stale 标注）-> SKIP——不把"未计算"当 0 判 PASS。

    Returns:
        ``{"status": "PASS"|"FAIL"|"SKIP", "detail": 说明}``。
    """
    if not any(r.get("category") == "knowledge_update" for r in records):
        return {"status": "SKIP", "detail": "融合结果中无 knowledge_update 题，stale 门禁跳过。"}
    stale_raw = metrics.get("overall", {}).get("stale_rate")
    if stale_raw is None:
        return {
            "status": "SKIP",
            "detail": "融合结果 stale_rate 未计算（None，数据缺 stale 标注或分片未落账），stale 门禁跳过。",
        }
    stale_rate = float(stale_raw)
    if stale_rate > threshold:
        return {
            "status": "FAIL",
            "detail": (
                f"融合结果 stale_rate={stale_rate:.3f} > 阈值 {threshold}，"
                "一票否决：评测不达标。"
            ),
        }
    return {
        "status": "PASS",
        "detail": f"融合结果 stale_rate={stale_rate:.3f} <= 阈值 {threshold}。",
    }


def _read_dialogue_records(dataset_dir: str) -> dict[str, list[dict[str, Any]]]:
    """读数据集目录下各对话的逐题明细：``<对话目录>/report/results.jsonl``。

    键是对话**目录名**（``layout.tag_dir_name(dialogue_id)``）；没有落明细的对话
    目录（该片内没有它参与的题）不计入。
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for name in sorted(os.listdir(dataset_dir)):
        path = os.path.join(dataset_dir, name, layout.DIR_REPORT, layout.FILE_RESULTS)
        if os.path.isfile(path):
            out[name] = _load_jsonl(path)
    return out


def _merge_dialogue_records(dataset_dirs: list[str]) -> dict[str, list[dict[str, Any]]]:
    """按对话目录名合并各分片的逐题记录（分片内的原始顺序保持不变）。"""
    merged: dict[str, list[dict[str, Any]]] = {}
    for dataset_dir in dataset_dirs:
        for name, records in _read_dialogue_records(dataset_dir).items():
            merged.setdefault(name, []).extend(records)
    return merged


def _dataset_index(dataset_path: str) -> dict[str, Any] | None:
    """读源数据集建索引：对话 / 题目顺序、逐对话与整集的能力标注、忽略题数。

    数据集不可读（被移动 / 已清理）时返回 None：融合退化为按分片顺序拼接，能力标注
    由 :func:`membench.metrics.aggregate` 从记录自行推断。
    """
    if not dataset_path or not os.path.isfile(dataset_path):
        return None
    try:
        dialogues = load_dataset(dataset_path)
    except (OSError, ValueError):
        return None
    return {
        "dialogue_order": {
            layout.tag_dir_name(str(dialogue.dialogue_id)): i
            for i, dialogue in enumerate(dialogues)
        },
        "qid_order": {
            str(question.qid): i
            for dialogue in dialogues
            for i, question in enumerate(dialogue.questions)
        },
        "caps": {
            str(dialogue.dialogue_id): question_capabilities(dialogue.questions)
            for dialogue in dialogues
        },
        "capabilities": question_capabilities(
            question for dialogue in dialogues for question in dialogue.questions
        ),
        "n_ignored": sum(
            1
            for dialogue in dialogues
            for question in dialogue.questions
            if getattr(question, "ignore", False)
        ),
        "dialogue_n_ignored": {
            layout.tag_dir_name(str(dialogue.dialogue_id)): sum(
                1 for question in dialogue.questions if getattr(question, "ignore", False)
            )
            for dialogue in dialogues
        },
    }


def _dialogue_id(records: list[dict[str, Any]], dir_name: str) -> str:
    """对话 id：取记录里的 ``dialogue_id``；该对话没有题时退回目录名。"""
    for record in records:
        dialogue_id = record.get("dialogue_id")
        if dialogue_id:
            return str(dialogue_id)
    return dir_name


def _write_report_dir(
    report_dir: str,
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    figures: bool,
) -> None:
    """把逐题明细 / 指标 / 报告写到 ``report_dir``（对话级与数据集级共用）。"""
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, layout.FILE_RESULTS), "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(os.path.join(report_dir, layout.FILE_METRICS), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    if figures:
        render_dataset_report(report_dir, records, metrics)
    else:
        render_dialogue_report(report_dir, records, metrics)


def fuse_runs(run_dirs: list[str], out_dir: str) -> dict[str, Any]:
    """校验并融合多个分片产物，在 ``out_dir``（完整实验目录）下写出完整数据集树。

    Args:
        run_dirs: 分片产物目录列表，每项是实验目录或数据集目录。
        out_dir: 融合产物输出目录——完整实验目录（``runs/<dataset>/`` + ``reports/``）。

    Returns:
        ``{"out_dir", "dataset_dir", "metrics", "gate", "n_questions"}``。

    Raises:
        ValueError: adapter / dataset / shards / shard_by 不一致、shard_index 未覆盖
            0..N-1、qid 重复，或各分片都没落逐对话明细等校验失败。
        SystemExit: 入参目录不存在、入参是对话级产物目录、数据集目录缺
            ``report/metrics.json``（产物不完整），或入参是旧版产物结构
            （见 :func:`membench.layout.resolve_dataset_dir`）。
    """
    if not run_dirs:
        raise _err("--runs 为空")

    dataset_dirs: list[str] = []
    metas: list[dict[str, Any]] = []
    for target in run_dirs:
        dataset_dirs.append(resolve_dataset_dir(target))
        metas.append(_load_metrics(target))

    first = metas[0]
    for target, m in zip(run_dirs, metas):
        for key in ("adapter", "dataset"):
            if m.get(key) != first.get(key):
                raise _err(f"{key} 不一致: {first.get(key)!r} vs {m.get(key)!r}（{target}）")
        for key in (
            "judge",
            "judge_model",
            "driver_llm",
            "adapter_version",
            "adapter_meta",
            "dataset_fingerprint",
            "retrieval_match",
            "retrieval_match_threshold",
            "schema_version",
            # 运行模式：全量分片与快照回放分片不得混融（写入侧口径不同，分数不可归因）
            "run_mode",
            # 冒烟截断范围：全量分片与 --limit 分片混融会得到"部分全量"的假完整结果
            "limit",
        ):
            if m.get(key) != first.get(key):
                raise _err(f"{key} 不一致: {first.get(key)!r} vs {m.get(key)!r}（{target}）")
        if int(m.get("shards") or 1) != int(first.get("shards") or 1):
            raise _err(
                f"shards 不一致: {first.get('shards')!r} vs {m.get('shards')!r}（{target}）"
            )
        if (m.get("shard_by") or "question") != (first.get("shard_by") or "question"):
            raise _err(
                f"shard_by 不一致: {first.get('shard_by') or 'question'} vs "
                f"{m.get('shard_by') or 'question'}（{target}）"
            )

    shards = int(first.get("shards") or 1)
    if shards < 2:
        raise _err("各 run 均未分片（shards=1），无需融合")
    indices = sorted(int(m.get("shard_index") or 0) for m in metas)
    if indices != list(range(shards)):
        raise _err(f"shard_index 未完整覆盖 0..{shards - 1}: 实际为 {indices}")

    merged = _merge_dialogue_records(dataset_dirs)
    if not merged:
        # 各分片都没有逐对话明细：融合只会产出空数据集，宁可报错也别给"0 题"的假结果
        raise _err(
            "各分片产物里都没有逐对话明细（<对话目录>/report/results.jsonl），无法融合"
        )
    index = _dataset_index(str(first.get("dataset_path") or ""))
    if index is not None:
        # 按源数据集的对话 / 题目顺序重排：融合结果与不分片直接跑逐题对齐
        merged = dict(
            sorted(
                merged.items(),
                key=lambda item: (
                    index["dialogue_order"].get(item[0], len(index["dialogue_order"])),
                    item[0],
                ),
            )
        )
        unknown = len(index["qid_order"])
        for dialogue_records in merged.values():
            dialogue_records.sort(
                key=lambda r: index["qid_order"].get(str(r.get("qid")), unknown)
            )
    records = [record for dialogue_records in merged.values() for record in dialogue_records]

    # qid 去重校验：同一题出现在多个分片中说明分片口径被破坏。
    seen: set[str] = set()
    for r in records:
        qid = str(r.get("qid", ""))
        if qid in seen:
            raise _err(f"qid 重复出现在多个分片中: {qid}")
        seen.add(qid)

    now = datetime.now()
    dataset_name = str(first.get("dataset") or "unknown")
    experiment_dir = os.path.abspath(out_dir)
    dataset_dir = layout.dataset_dir(experiment_dir, dataset_name)
    # run_id 走 layout 的唯一组装入口：commit、分片与冒烟范围都进身份，另有 "fused"
    # 标出这是融合产物——融合结果（尤其 --limit 冒烟分片的融合）不得与全量结果撞身份。
    # 各分片的 limit 前面已校验一致，这里直接透传。
    run_id = layout.run_id(
        str(first.get("adapter") or "unknown"),
        first.get("platform"),
        dataset_name,
        now.strftime("%Y%m%d_%H%M%S"),
        commit=str(first.get("commit") or "") or None,
        tags=[
            "fused",
            *layout.run_id_tags(
                shards, None, first.get("shard_by") or "question", first.get("limit")
            ),
        ],
    )
    # 数据集级与逐对话共用的身份 / 口径元数据；可比性字段透传，融合产物与普通 run
    # 同等可校验、可对比。
    base: dict[str, Any] = {
        "run_id": run_id,
        "adapter": first.get("adapter"),
        "dataset": first.get("dataset"),
        "dataset_path": first.get("dataset_path"),
        "dataset_source": first.get("dataset_source"),
        "platform": first.get("platform", "default"),
        "commit": first.get("commit"),
        "judge": first.get("judge"),
        "timestamp": now.isoformat(timespec="seconds"),
        # 冒烟范围透传（各分片一致）：冒烟融合产物与全量融合产物必须能被 compare 区分
        "limit": first.get("limit"),
        "shards": shards,
        "shard_index": None,
        "shard_by": first.get("shard_by") or "question",
        "adapter_meta": first.get("adapter_meta"),
        "adapter_describe": first.get("adapter_describe"),
        "schema_version": first.get("schema_version"),
        "driver_llm": first.get("driver_llm"),
        "judge_model": first.get("judge_model"),
        "adapter_version": first.get("adapter_version"),
        "dataset_fingerprint": first.get("dataset_fingerprint"),
        "retrieval_match": first.get("retrieval_match"),
        # 运行模式透传（前面已校验各分片一致）
        "run_mode": first.get("run_mode") or "full",
        "experiment_dir": experiment_dir,
        "dataset_dir": dataset_dir,
    }
    if first.get("retrieval_match_threshold") is not None:
        base["retrieval_match_threshold"] = first["retrieval_match_threshold"]
    match_fallbacks = sum(int(m.get("retrieval_match_fallbacks", 0) or 0) for m in metas)
    if match_fallbacks:
        base["retrieval_match_fallbacks"] = match_fallbacks

    caps = index["capabilities"] if index is not None else None
    n_ignored = (
        int(index["n_ignored"])
        if index is not None
        else sum(int(m.get("n_ignored", 0) or 0) for m in metas)
    )
    metrics = dict(aggregate(records, capabilities=caps))
    metrics.update(base)
    metrics.update(
        {
            "n_ignored": n_ignored,
            "fused": True,
            "fused_from": [str(m.get("run_id", d)) for m, d in zip(metas, run_dirs)],
        }
    )
    # token 记账：各分片各记各的（question 分片下同一对话可能被多片 ingest，
    # 只有分片自己的计数才是真实成本），故直接相加；任一分片未落账则整体不落账
    # （避免"部分求和"被读成全量消耗）。
    if all(isinstance(m.get("token_usage"), dict) for m in metas):

        def _sum_usage(field: str) -> int:
            return sum(int((m.get("token_usage") or {}).get(field, 0) or 0) for m in metas)

        metrics["token_usage"] = {
            field: _sum_usage(field)
            for field in (
                "ingest_input_tokens",
                "ingest_llm_calls",
                "query_input_tokens",
                "query_output_tokens",
                "query_llm_calls",
                "n_dialogues",
                "n_questions",
            )
        }
        if any((m.get("token_usage") or {}).get("ingest_skipped") for m in metas):
            # 回放分片：融合产物同样标"ingest 被跳过"，避免 0 被读成零成本
            metrics["token_usage"]["ingest_skipped"] = 1
    gate = fuse_gate(records, metrics)
    metrics["stale_gate"] = gate

    # 逐对话报告：对话级指标由合并后的记录重新聚合（题目级分片下同一对话的记录来自
    # 多个分片，各片自己的对话级指标都只是子集）。
    for dir_name, dialogue_records in merged.items():
        dialogue_id = _dialogue_id(dialogue_records, dir_name)
        dialogue_metrics = dict(
            aggregate(
                dialogue_records,
                capabilities=index["caps"].get(dialogue_id) if index is not None else None,
            )
        )
        dialogue_metrics.update(base)
        dialogue_metrics["dialogue_id"] = dialogue_id
        dialogue_metrics["n_ignored"] = (
            index["dialogue_n_ignored"].get(dir_name, 0) if index is not None else 0
        )
        _write_report_dir(
            layout.dialogue_report_dir(experiment_dir, dataset_name, dialogue_id),
            dialogue_records,
            dialogue_metrics,
            figures=False,
        )

    dataset_report = layout.dataset_report_dir(experiment_dir, dataset_name)
    _write_report_dir(dataset_report, records, metrics, figures=True)
    copy_report(
        os.path.join(dataset_report, layout.FILE_REPORT_MD),
        layout.task_report_dir(experiment_dir),
        layout.FILE_REPORT_MD,
        extra_files=[os.path.join(dataset_report, layout.FILE_METRICS)],
    )

    return {
        "out_dir": out_dir,
        "dataset_dir": dataset_dir,
        "metrics": metrics,
        "gate": gate,
        "n_questions": len(records),
    }
