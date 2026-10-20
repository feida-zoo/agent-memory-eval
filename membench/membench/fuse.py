"""分片融合：把同一评测的多个分片 run 合并为一个完整 run 目录。

``membench run --shards N --shard-index I`` 产生的各分片 run 可通过
``membench fuse --runs dir1,dir2,... --out DIR`` 融合：

- 校验各 run 的 adapter / dataset / shards / shard_by 一致，且 shard_index 恰好覆盖 0..N-1；
  判分与可比性元数据（judge / judge_model / driver_llm / adapter_version / adapter_meta /
  dataset_fingerprint / retrieval_match / schema_version）也必须一致，防止混片；
- 合并 results.jsonl（按 qid 去重校验，简单拼接保持传入顺序）；
- 复用 ``metrics.summary.aggregate`` 重新计算指标（能力标注优先按数据集扫描，
  与 run 口径一致），复用 report 渲染完整报告；
- 门禁判定（stale 一票否决）对融合结果同样生效：存在 knowledge_update 题
  且 stale_rate 超过阈值（与 evaluate 的办公域门禁同阈值）即 FAIL。

输出目录结构与普通 run 一致（results.jsonl + metrics.json + report.md + 图表），
metadata 标记 ``"fused": true`` 及来源分片列表 ``"fused_from"``，并透传可比性
元数据（adapter_version / judge_model / driver_llm / dataset_fingerprint /
retrieval_match / adapter_meta / adapter_describe / schema_version）——融合产物
可直接参与 ``compare --require-comparable``。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Union

from membench.evaluate import STALE_GATE_THRESHOLD, scan_capabilities
from membench.metrics import aggregate
from membench.report import render_run_report


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
    （退出码 1）；无 knowledge_update 题 -> SKIP。

    Returns:
        ``{"status": "PASS"|"FAIL"|"SKIP", "detail": 说明}``。
    """
    if not any(r.get("category") == "knowledge_update" for r in records):
        return {"status": "SKIP", "detail": "融合结果中无 knowledge_update 题，stale 门禁跳过。"}
    stale_rate = float(metrics.get("overall", {}).get("stale_rate", 0.0))
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


def fuse_runs(run_dirs: list[str], out_dir: str) -> dict[str, Any]:
    """校验并融合多个分片 run，在 ``out_dir`` 下生成完整 run 目录结构。

    Args:
        run_dirs: 分片 run 目录列表（每个含 results.jsonl 与 metrics.json）。
        out_dir: 融合产物输出目录。

    Returns:
        ``{"out_dir", "metrics", "gate", "n_questions"}``。

    Raises:
        ValueError: adapter / dataset / shards 不一致、shard_index 未覆盖
            0..N-1、qid 重复等校验失败。
        FileNotFoundError: 缺少 results.jsonl / metrics.json。
    """
    if not run_dirs:
        raise _err("--runs 为空")

    metas: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for d in run_dirs:
        metrics_path = os.path.join(d, "metrics.json")
        results_path = os.path.join(d, "results.jsonl")
        if not os.path.isfile(metrics_path):
            raise FileNotFoundError(f"缺少 metrics.json: {metrics_path}")
        if not os.path.isfile(results_path):
            raise FileNotFoundError(f"缺少 results.jsonl: {results_path}")
        metas.append(_load_json(metrics_path))
        records.extend(_load_jsonl(results_path))

    first = metas[0]
    for d, m in zip(run_dirs, metas):
        for key in ("adapter", "dataset"):
            if m.get(key) != first.get(key):
                raise _err(f"{key} 不一致: {first.get(key)!r} vs {m.get(key)!r}（{d}）")
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
        ):
            if m.get(key) != first.get(key):
                raise _err(f"{key} 不一致: {first.get(key)!r} vs {m.get(key)!r}（{d}）")
        if int(m.get("shards") or 1) != int(first.get("shards") or 1):
            raise _err(f"shards 不一致: {first.get('shards')!r} vs {m.get('shards')!r}（{d}）")
        if (m.get("shard_by") or "question") != (first.get("shard_by") or "question"):
            raise _err(
                f"shard_by 不一致: {first.get('shard_by') or 'question'} vs "
                f"{m.get('shard_by') or 'question'}（{d}）"
            )

    shards = int(first.get("shards") or 1)
    if shards < 2:
        raise _err("各 run 均未分片（shards=1），无需融合")
    indices = sorted(int(m.get("shard_index") or 0) for m in metas)
    if indices != list(range(shards)):
        raise _err(f"shard_index 未完整覆盖 0..{shards - 1}: 实际为 {indices}")

    # qid 去重校验：同一题出现在多个分片中说明分片口径被破坏。
    seen: set[str] = set()
    for r in records:
        qid = str(r.get("qid", ""))
        if qid in seen:
            raise _err(f"qid 重复出现在多个分片中: {qid}")
        seen.add(qid)

    # 能力标注优先按数据集扫描（与 run 口径一致）；数据集不可读时按 records 推断
    caps = None
    ds_path = str(first.get("dataset_path") or "")
    if ds_path and os.path.isfile(ds_path):
        try:
            caps = scan_capabilities(ds_path)
        except (OSError, ValueError):
            caps = None
    metrics = dict(aggregate(records, capabilities=caps))
    now = datetime.now()
    run_id = (
        f"{first.get('adapter', 'unknown')}_{first.get('platform', 'default')}_"
        f"{first.get('dataset', 'unknown')}_fused_{now.strftime('%Y%m%d_%H%M%S')}"
    )
    metrics.update(
        {
            "run_id": run_id,
            "adapter": first.get("adapter"),
            "dataset": first.get("dataset"),
            "dataset_path": first.get("dataset_path"),
            "dataset_source": first.get("dataset_source"),
            "platform": first.get("platform", "default"),
            "commit": first.get("commit"),
            "judge": first.get("judge"),
            "n_ignored": sum(int(m.get("n_ignored", 0) or 0) for m in metas),
            "timestamp": now.isoformat(timespec="seconds"),
            "limit": None,
            "shards": shards,
            "shard_index": None,
            "shard_by": first.get("shard_by") or "question",
            # 可比性元数据透传：融合产物与普通 run 同等可校验/可对比
            "adapter_meta": first.get("adapter_meta"),
            "adapter_describe": first.get("adapter_describe"),
            "schema_version": first.get("schema_version"),
            "driver_llm": first.get("driver_llm"),
            "judge_model": first.get("judge_model"),
            "adapter_version": first.get("adapter_version"),
            "dataset_fingerprint": first.get("dataset_fingerprint"),
            "retrieval_match": first.get("retrieval_match"),
            "fused": True,
            "fused_from": [str(m.get("run_id", d)) for m, d in zip(metas, run_dirs)],
        }
    )
    if first.get("retrieval_match_threshold") is not None:
        metrics["retrieval_match_threshold"] = first["retrieval_match_threshold"]
    match_fallbacks = sum(int(m.get("retrieval_match_fallbacks", 0) or 0) for m in metas)
    if match_fallbacks:
        metrics["retrieval_match_fallbacks"] = match_fallbacks
    gate = fuse_gate(records, metrics)
    metrics["stale_gate"] = gate

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.jsonl"), "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    render_run_report(out_dir, records, metrics)

    return {
        "out_dir": out_dir,
        "metrics": metrics,
        "gate": gate,
        "n_questions": len(records),
    }
