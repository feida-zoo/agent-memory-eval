"""分片融合：按对话目录 + 分片账本把各片产物拼成一份完整实验。

``membench run --shards N --shard-index I`` 的分片产物只落本片负责的对话目录
（``runs/<dataset>/<对话>/{runtime,memory_snapshot,report}/``），数据集级报告与任务级
报告一律由 fuse 补齐；各片的进度与归属记在分片账本 ``runs/<dataset>/shards.json``
（读 / 写 / 校验见 :func:`membench.layout.read_shard_ledger` 等）。

同一个 :func:`fuse_runs` 支持两种入参形态：

- **跨目录**：``fuse_runs([dir1, dir2, ...], out)``——每项是某个分片的实验目录或数据集
  目录，各片可以各写自己的目录；融合产物落到 ``out``（完整实验目录）；
- **就地**：``fuse_runs([], task_dir)``，等价写法 ``fuse_runs([task_dir], task_dir)``——
  各片本来就写在同一棵树上（CLI ``--task-dir`` 的 ``shared`` 写入，或手工并行写同一
  目录），此时输入树与输出树是同一棵，不新建目录，只补齐数据集级与任务级产物。
  就地模式要求那棵树里有账本，且它位于某实验目录的 ``runs/<dataset>`` 下。

校验（全部在写盘之前，失败即拒绝、不留半成品）：

- **口径一致**：各片的账本 schema / ``dataset`` / ``shards`` / ``shard_by`` / ``limit``
  / 数据集指纹逐项比对；判分与可比性元数据（adapter / platform / commit / judge /
  judge_model / driver_llm / adapter_version / adapter_meta / dataset_fingerprint /
  retrieval_match / schema_version / run_mode）也必须一致——防止混片（含"全量分片与
  ``--limit`` 冒烟分片混融"这种假全量）；
- **账本齐备**：``s0..s{N-1}`` 一片不少、每片 ``status="done"``（还在跑或跑挂的片点名
  拒绝）、同一片不能出现在两个目录里；
- **账本与产物对得上**：账本记的对话就是树里有的对话（缺 / 多都报），对话级
  ``report/metrics.json`` 的 ``shard_index`` / ``run_id`` / ``shards`` / ``shard_by`` /
  ``limit`` / 数据集指纹与账本逐项吻合；
- **覆盖完整**：用 :func:`membench.runner.select_shard_questions` 复算每片"本该有哪些
  对话、哪些题"，与该片的实际产物逐项比对——dialogue 分片下每个对话恰好被一片覆盖
  （缺 / 重都报），question 分片下逐题覆盖完整且无重复。源数据集不可读、或内容与各片
  记录的数据集指纹不一致时降级为按账本核对并打印告警（不与数据集逐项比对）；
- 各片账本记的题数合计必须等于融合产物的逐题明细条数（账本或产物被动过就报错）。

融合产物（``out`` / 就地模式的 ``task_dir`` 下）：

- ``runs/<dataset>/<对话>/{runtime,memory_snapshot,report}/``：对话目录随对话并入
  （就地模式本来就在原位）。**跨目录模式**下逐对话 ``report/`` 由合并后的记录重建
  （question 分片下同一对话的记录来自多个分片，各片自己的对话级报告都只是子集），
  ``report/`` 之外的内容（runtime / 快照）随对话并入——同一对话出现在多片时只取序号
  最小那一片的那一份；**就地模式**不重写对话级产物：那些目录本来就是各片各写各的
  （互不重叠），保留原样既留住了"哪个对话归哪一片"的证据，也让 fuse 可以安全地重复执行；
- ``runs/<dataset>/report/{report.md,metrics.json,results.jsonl,fig_*.png}``：数据集级报告；
  逐题顺序与"不分片跑一遍"一致（源数据集可读时按数据集顺序重排）；
- ``reports/{report.md,metrics.json}``：任务级报告（同 Runner 的 write_task_reports 形态）。

数据集级 metrics 与普通 run **同构**：可比性字段透传、``limit`` 透传、
``token_usage`` 逐片逐对话相加（任一对话副本未落账则整份不落账，避免"部分求和"被读成
全量消耗）、另标 ``fused: true`` 与 ``fused_from``（各片 run_id）、``shard_index: None``；
``run_id`` 走 :func:`membench.layout.run_id`（带 ``fused`` / 分片 / limit 标记），因此
``--limit`` 冒烟分片的融合产物不会被当成全量融合产物，可直接参与
``compare --require-comparable``。

融合**重写**的产物（数据集级 metrics，以及跨目录模式下重建的对话级 metrics）不落
``state_snapshot`` 块：各片的快照按对话归属各自的片目录，融一次不会凭空产生"整场"快照；
快照文件本身随对话目录并入（就地模式本来就在位），能不能回放由 manifest 决定
（``membench.snapshot.plan_replay`` 只读 manifest），不看这段元数据。

门禁判定（stale 一票否决）对融合结果同样生效：存在 knowledge_update 题且 stale_rate
超过阈值（与 evaluate 的办公域门禁同阈值）即 FAIL。
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import Any, Union

from membench import layout
from membench.data.schema import dataset_fingerprint, load_dataset
from membench.evaluate import STALE_GATE_THRESHOLD
from membench.metrics import aggregate
from membench.report import copy_report, render_dataset_report, render_dialogue_report
from membench.runner import select_shard_questions

#: 跨分片必须一致的元数据：身份（adapter / platform / commit）、判分与驱动
#: （judge / judge_model / driver_llm / adapter_version / adapter_meta）、口径
#: （dataset_fingerprint / retrieval_match / schema_version / run_mode）与分片范围
#: （limit / shards / shard_by）。任一项不一致即拒绝融合——混片的分数不可归因。
_FUSED_METADATA_KEYS = (
    "adapter",
    "dataset",
    "platform",
    "commit",
    "judge",
    "judge_model",
    "driver_llm",
    "adapter_version",
    "adapter_meta",
    "dataset_fingerprint",
    "retrieval_match",
    "retrieval_match_threshold",
    "schema_version",
    "run_mode",
    "limit",
    "shards",
    "shard_by",
)

#: 对话级 metrics 必须与账本头部逐项吻合的字段（外加 ``shard_index`` / ``run_id``）。
_LEDGER_HEADER_KEYS = ("dataset", "shards", "shard_by", "limit", "dataset_fingerprint")

#: ``token_usage`` 逐字段求和的字段（与 Runner 落账的字段一致）。
_USAGE_FIELDS = (
    "ingest_input_tokens",
    "ingest_llm_calls",
    "query_input_tokens",
    "query_output_tokens",
    "query_llm_calls",
    "n_dialogues",
    "n_questions",
)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


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


def _brief(items: Any, limit: int = 5) -> str:
    """把一组待点名的对象写成一行（超过 limit 个只列前几个并给出总数）。"""
    names = sorted(str(item) for item in items)
    if len(names) <= limit:
        return "、".join(names)
    return "、".join(names[:limit]) + f" 等 {len(names)} 项"


#: 就地融合下的对话目录前提：各片各写各的对话目录（dialogue 分片）。题目级分片会把同一
#: 对话的题目分散写进同一个对话目录并互相覆盖，只能各片独立成目录后跨目录融合；这条提示
#: 只加在就地模式失败时，避免用户拿着"元数据对不上"去猜原因。
_IN_PLACE_HINT = (
    "；就地融合要求各片各写各的对话目录（--shard-by dialogue）——题目级分片会把同一对话"
    "的题目分散写进同一目录并互相覆盖，请给各片独立的实验目录后用 --runs 跨目录融合"
)


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


# ---------------------------------------------------------------------------
# 分片产物的定位与读取
# ---------------------------------------------------------------------------

def _locate_shard_dir(target: str) -> str:
    """把「产物目录」入参归一成**分片数据集目录**（带分片账本的那种）。

    入参是分片的实验目录或数据集目录；实验目录在其 ``runs/`` 下定位唯一带账本的数据集
    目录。都不是时按统一口径报错：不存在 / 入参是对话目录 / 旧版结构 / 缺 metrics 各有
    专门文案（:func:`membench.layout.resolve_dataset_dir`）；若那其实是一份**完整 run**
    （数据集级 metrics 在），再按分片口径说清为什么不用融合。
    """
    if not os.path.isdir(target):
        raise SystemExit(f"产物目录不存在: {target}")
    abspath = os.path.abspath(target)
    if layout.has_shard_ledger(abspath):
        return abspath
    runs = layout.runs_root(abspath)
    if os.path.isdir(runs):
        found = sorted(
            os.path.join(runs, name)
            for name in os.listdir(runs)
            if layout.has_shard_ledger(os.path.join(runs, name))
        )
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            names = "、".join(os.path.basename(path) for path in found)
            raise SystemExit(
                f"实验目录 {abspath} 下有多个分片数据集目录（{names}），"
                f"无法确定融合哪一个；请把入参指向具体的数据集目录 {runs}/<dataset>"
            )
    dataset_dir = layout.resolve_dataset_dir(abspath)
    metrics = _load_json(os.path.join(dataset_dir, layout.DIR_REPORT, layout.FILE_METRICS))
    if metrics.get("fused"):
        raise _err(f"该产物已经是融合结果（fused=true），无需再融合: {dataset_dir}")
    if int(metrics.get("shards") or 1) < 2:
        raise _err("各 run 均未分片（shards=1），无需融合")
    raise _err(
        f"分片产物缺少账本（{layout.FILE_SHARDS}），无法确认它覆盖了哪些对话: {dataset_dir}；"
        "请用当前版本的分片模式重跑该片"
    )


def _is_own_dataset_dir(dataset_dir: str, out_dir: str) -> bool:
    """该分片数据集目录就是 ``out_dir`` 自己的数据集目录吗（就地融合的判定）。"""
    return os.path.normpath(os.path.dirname(dataset_dir)) == os.path.normpath(
        layout.runs_root(out_dir)
    )


def _read_shard(dataset_dir: str) -> dict[str, Any]:
    """读一个分片树：账本 + 逐对话产物（``<对话目录>/report/{results.jsonl,metrics.json}``）。

    缺账本、对话目录缺 ``report/`` 层级或报告文件不全都在这里当场报错——半成品被当成
    完整的一片刻进融合结果，比直接失败危险得多。
    """
    ledger = layout.read_shard_ledger(dataset_dir)
    if ledger is None:
        raise _err(f"分片产物缺少账本（{layout.FILE_SHARDS}）: {dataset_dir}")
    dialogues: dict[str, dict[str, Any]] = {}
    for name in sorted(os.listdir(dataset_dir)):
        dialogue_dir = os.path.join(dataset_dir, name)
        if not os.path.isdir(dialogue_dir) or name == layout.DIR_REPORT:
            continue
        report_dir = os.path.join(dialogue_dir, layout.DIR_REPORT)
        if not os.path.isdir(report_dir):
            raise _err(
                f"对话目录没有 {layout.DIR_REPORT}/ 层级（产物不完整，可能中途中断）：{dialogue_dir}"
            )
        results_path = os.path.join(report_dir, layout.FILE_RESULTS)
        metrics_path = os.path.join(report_dir, layout.FILE_METRICS)
        if not (os.path.isfile(results_path) and os.path.isfile(metrics_path)):
            raise _err(
                f"对话产物不完整（缺 {layout.DIR_REPORT}/{layout.FILE_RESULTS} 或 "
                f"{layout.DIR_REPORT}/{layout.FILE_METRICS}）：{dialogue_dir}"
            )
        dialogues[name] = {
            "dir": dialogue_dir,
            "records": _load_jsonl(results_path),
            "metrics": _load_json(metrics_path),
        }
    return {"dir": dataset_dir, "ledger": ledger, "dialogues": dialogues}


def _shard_entries(shard: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """账本条目按分片序号展开：``s<i>`` -> 条目，坏键 / 未跑完一律拒绝。

    题数合计与逐题明细是否对得上在 :func:`fuse_runs` 里另行核对（要等各片读齐）。
    """
    entries: dict[int, dict[str, Any]] = {}
    dataset_dir = shard["dir"]
    for key, entry in (shard["ledger"].get("entries") or {}).items():
        index = layout.shard_key_index(key)
        if index is None:
            raise _err(f"分片账本条目键非法（应为 s<i>）: {key!r}（{dataset_dir}）")
        if not isinstance(entry, dict):
            raise _err(f"分片账本条目结构非法（{key}）: {entry!r}（{dataset_dir}）")
        status = str(entry.get("status") or "")
        if status != "done":
            raise _err(
                f"分片 {key} 还没跑完（账本状态 {status or '未记'}）：{dataset_dir}；"
                "中断或正在跑的分片会被当成完整的片融进去，请先跑完再融合"
            )
        entries[index] = entry
    return entries


def _check_ledgers(shards: list[dict[str, Any]]) -> dict[str, Any]:
    """各片账本口径逐项比对，返回参照账本头部（``shards`` 片数由它给出）。"""
    ref = shards[0]
    header = ref["ledger"]
    for shard in shards[1:]:
        for key in _LEDGER_HEADER_KEYS:
            if shard["ledger"].get(key) != header.get(key):
                raise _err(
                    f"{key} 不一致: {header.get(key)!r} vs {shard['ledger'].get(key)!r}"
                    f"（{shard['dir']}）"
                )
    shards_n = int(header.get("shards") or 1)
    if shards_n < 2:
        raise _err(f"分片账本记的分片数为 {shards_n}，无需融合（{ref['dir']}）")
    return header


def _group_by_shard(
    shards: list[dict[str, Any]], shards_n: int
) -> dict[int, dict[str, Any]]:
    """把各片展开成 ``序号 -> {条目, 对话产物}``：覆盖 0..N-1、同片不得两处。"""
    by_index: dict[int, dict[str, Any]] = {}
    for shard in shards:
        entries = _shard_entries(shard)
        for index, entry in entries.items():
            other = by_index.get(index)
            if other is not None:
                raise _err(
                    f"分片 s{index} 同时出现在两个目录里（{other['dir']}、{shard['dir']}），"
                    "无法确定用哪一份"
                )
            by_index[index] = {
                "entry": entry,
                "dir": shard["dir"],
                # 一片写一条账本条目，但它可能写在含多条目的树里（共享任务目录）；
                # 只取该条目录及的对话，别把别的片的产物算进来
                "dialogues": {
                    name: item
                    for name, item in shard["dialogues"].items()
                    if name in set(entry.get("dialogues") or ())
                },
            }
    missing = [index for index in range(shards_n) if index not in by_index]
    if missing:
        have = "、".join(f"s{index}" for index in sorted(by_index)) or "无"
        lack = "、".join(f"s{index}" for index in missing)
        raise _err(
            f"分片序号未完整覆盖 0..{shards_n - 1}：缺 {lack}（账本里只有 {have}）——"
            "少片的融合结果会缺对话 / 缺题，请补齐后再融合"
        )
    return by_index


def _check_dialogues_against_ledger(
    by_index: dict[int, dict[str, Any]],
    shards: list[dict[str, Any]],
    *,
    hint: str = "",
) -> None:
    """账本记的对话 == 树里有的对话（多 / 少都报），对话级元数据与账本逐项吻合。"""
    ledgers = {shard["dir"]: shard["ledger"] for shard in shards}
    for shard in shards:
        ledger_names = {
            str(name)
            for entry in (shard["ledger"].get("entries") or {}).values()
            if isinstance(entry, dict)
            for name in (entry.get("dialogues") or ())
        }
        in_tree = set(shard["dialogues"])
        orphan = in_tree - ledger_names
        if orphan:
            raise _err(
                f"对话目录不在分片账本里（{shard['dir']}）：{_brief(orphan)}——"
                "账本是各片产物的唯一归属记录，对不上就无法判断它属于哪一片"
            )
        absent = ledger_names - in_tree
        if absent:
            raise _err(
                f"分片账本记了对话、树里却没有它的产物（{shard['dir']}）：{_brief(absent)}——"
                "该片没跑完或产物被删过，请重跑该片"
            )

    for index, shard in sorted(by_index.items()):
        entry = shard["entry"]
        ledger = ledgers[shard["dir"]]
        for name, item in sorted(shard["dialogues"].items()):
            metrics = item["metrics"]
            problems = [
                f"{key}={metrics.get(key)!r}（账本 {ledger.get(key)!r}）"
                for key in _LEDGER_HEADER_KEYS
                if metrics.get(key) != ledger.get(key)
            ]
            if metrics.get("shard_index") != index:
                problems.append(
                    f"shard_index={metrics.get('shard_index')!r}（账本条目 s{index}）"
                )
            if str(metrics.get("run_id") or "") != str(entry.get("run_id") or ""):
                problems.append(
                    f"run_id={metrics.get('run_id')!r}（账本 {entry.get('run_id')!r}）"
                )
            if problems:
                raise _err(
                    f"对话级元数据与分片账本不一致（{item['dir']}）："
                    + "；".join(problems)
                    + hint
                )


def _check_shard_metadata(
    by_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """各片对话级元数据：片内自洽 + 跨片一致，返回序号最小那片的代表元数据。

    空片（一道题都没分到、没写对话目录）没有对话级元数据可比，跳过；一片都没有产物时
    报错——融合只会产出空数据集，宁可失败也别给"0 题"的假结果。
    """
    ref_metrics: Union[dict[str, Any], None] = None
    for index, shard in sorted(by_index.items()):
        reps = [item["metrics"] for item in shard["dialogues"].values()]
        if not reps:
            continue
        first = reps[0]
        for metrics in reps[1:]:
            for key in _FUSED_METADATA_KEYS:
                if metrics.get(key) != first.get(key):
                    raise _err(
                        f"同一分片 s{index} 内的对话级元数据不一致（{key}）: "
                        f"{first.get(key)!r} vs {metrics.get(key)!r}（{shard['dir']}）"
                    )
        if ref_metrics is None:
            ref_metrics = first
            continue
        for key in _FUSED_METADATA_KEYS:
            if first.get(key) != ref_metrics.get(key):
                raise _err(
                    f"{key} 不一致: {ref_metrics.get(key)!r} vs {first.get(key)!r}"
                    f"（{shard['dir']}）"
                )
    if ref_metrics is None:
        raise _err(
            "各分片都没有逐对话明细（<对话目录>/report/results.jsonl），无法融合"
        )
    return ref_metrics


# ---------------------------------------------------------------------------
# 覆盖完整性：用数据集复算每片"本该有哪些对话 / 哪些题"
# ---------------------------------------------------------------------------

def _dataset_truth(
    dataset_path: str,
    *,
    shards_n: int,
    shard_by: str,
    limit: Union[int, None],
    fingerprint: Union[dict[str, Any], None],
) -> Union[dict[str, Any], None]:
    """源数据集给出的真值：对话 / 题目顺序 + 每片应有的对话与题集。

    数据集不可读（被移动 / 已清理）、或内容与各片记录的数据集指纹不一致时返回 None 并
    打印告警：融合退化为按账本校验覆盖（仍然拒绝少片 / 未跑完 / 账本与产物对不上），
    但无法与数据集逐项比对。
    """
    if not dataset_path or not os.path.isfile(dataset_path):
        print(
            f"[membench] ⚠️ 源数据集不可读（{dataset_path or '未记路径'}）："
            "只能按分片账本校对覆盖，无法与数据集逐项比对"
        )
        return None
    want = (fingerprint or {}).get("sha256")
    now = dataset_fingerprint(dataset_path).get("sha256")
    if want and now and want != now:
        print(
            f"[membench] ⚠️ 源数据集内容与各分片记录的不一致（指纹 {want} vs {now}）："
            "跳过与数据集的逐题覆盖比对，只按分片账本校对"
        )
        return None
    try:
        dialogues = load_dataset(dataset_path)
    except (OSError, ValueError) as exc:
        print(f"[membench] ⚠️ 源数据集无法解析（{exc}）：只按分片账本校对覆盖")
        return None
    expected: dict[int, dict[str, set[str]]] = {}
    for index in range(shards_n):
        selected = select_shard_questions(dialogues, shards_n, index, limit, shard_by)
        expected[index] = {
            layout.tag_dir_name(str(dialogue.dialogue_id)): {
                str(question.qid)
                for question in dialogue.questions
                if not getattr(question, "ignore", False)
            }
            for dialogue in selected
        }
    return {
        "expected": expected,
        "dialogue_order": {
            layout.tag_dir_name(str(dialogue.dialogue_id)): position
            for position, dialogue in enumerate(dialogues)
        },
        "qid_order": {
            str(question.qid): position
            for dialogue in dialogues
            for position, question in enumerate(dialogue.questions)
        },
    }


def _check_coverage(
    by_index: dict[int, dict[str, Any]], truth: Union[dict[str, Any], None]
) -> None:
    """每片的对话 / 题集与该片"本该有的"逐项比对（覆盖完整性）。"""
    if truth is None:
        return
    for index, shard in sorted(by_index.items()):
        expected = truth["expected"].get(index, {})
        actual_names = set(shard["dialogues"])
        expected_names = set(expected)
        if expected_names - actual_names:
            raise _err(
                f"分片 s{index} 少了本该它负责的对话：{_brief(expected_names - actual_names)}"
                f"（{shard['dir']}）——融合结果会缺对话，请重跑该片"
            )
        if actual_names - expected_names:
            raise _err(
                f"分片 s{index} 多了不该它负责的对话：{_brief(actual_names - expected_names)}"
                f"（{shard['dir']}）——同一个对话不该被两片重复覆盖，"
                "请核对 --shards / --shard-by"
            )
        for name, expected_qids in sorted(expected.items()):
            actual_qids = {
                str(record.get("qid", ""))
                for record in shard["dialogues"][name]["records"]
            }
            if actual_qids == expected_qids:
                continue
            detail = []
            if expected_qids - actual_qids:
                detail.append(f"缺 {_brief(expected_qids - actual_qids)}")
            if actual_qids - expected_qids:
                detail.append(f"多 {_brief(actual_qids - expected_qids)}")
            raise _err(
                f"分片 s{index} 的对话 {name} 题集与该片口径不符（{'；'.join(detail)}）"
                f"（{shard['dir']}）——题目覆盖不完整或重复，请重跑该片"
            )


# ---------------------------------------------------------------------------
# 融合产物
# ---------------------------------------------------------------------------

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


def _copy_dialogue_content(source: str, target: str) -> None:
    """把对话目录里 ``report/`` 之外的内容（runtime / memory_snapshot 等）并入融合树。

    只在**跨目录**融合时用：目标树是新的，对话目录要随对话搬过来，而逐对话 ``report/``
    由合并后的记录重建，不复制（源目录那一份可能只是子集）。
    """
    for name in sorted(os.listdir(source)):
        if name == layout.DIR_REPORT:
            continue
        src = os.path.join(source, name)
        dst = os.path.join(target, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            os.makedirs(target, exist_ok=True)
            shutil.copyfile(src, dst)


def _dialogue_id(records: list[dict[str, Any]], dir_name: str) -> str:
    """对话 id：取记录里的 ``dialogue_id``；该对话没有题时退回目录名。"""
    for record in records:
        dialogue_id = record.get("dialogue_id")
        if dialogue_id:
            return str(dialogue_id)
    return dir_name


def _sum_usage(blocks: list[dict[str, Any]]) -> Union[dict[str, int], None]:
    """把若干 ``token_usage`` 块逐字段相加；任一块缺失即返回 None（不落账）。

    "部分求和"会被读成全量消耗（真实成本只会更高），所以宁可整份不落账。
    """
    if not blocks or not all(isinstance(block, dict) for block in blocks):
        return None

    def _total(field: str) -> int:
        return sum(int(block.get(field, 0) or 0) for block in blocks)

    usage = {field: _total(field) for field in _USAGE_FIELDS}
    if any(block.get("ingest_skipped") for block in blocks):
        # 回放分片：融合产物同样标"ingest 被跳过"，避免 0 被读成零成本
        usage["ingest_skipped"] = 1
    return usage


def _combine_usage(blocks: list[Union[dict[str, Any], None]]) -> Union[dict[str, int], None]:
    """把逐对话的 ``token_usage`` 合计成数据集级；任一对话未落账即整体不落账。"""
    if not blocks or any(block is None for block in blocks):
        return None
    return _sum_usage([block for block in blocks if block is not None])


def _caps_or(blocks: list[dict[str, Any]]) -> dict[str, bool]:
    """能力标注取各片副本的并集（题目被多片分摊时，任一有某题型就认这个能力）。

    与 Runner 的口径一致：Runner 按对话的题目集扫描后落账，融合时逐对话取并集——
    既不凭空多认没有的题型，也不会因分片把题型切成两半而漏掉。
    """
    return {
        key: any(bool(block.get(key)) for block in blocks)
        for key in ("has_evidence", "has_stale", "has_abstention")
    }


def fuse_runs(
    run_dirs: Union[list[str], None],
    out_dir: str,
) -> dict[str, Any]:
    """校验并融合多个分片产物，在 ``out_dir`` 下写出完整数据集树与任务级报告。

    Args:
        run_dirs: 分片产物目录列表，每项是实验目录或数据集目录；传空（或 None）表示
            **就地模式**：各片本来就写在 ``out_dir`` 这棵树里，只补齐数据集级 / 任务级
            产物；给同 ``out_dir`` 里那一个数据集目录也算就地模式。
        out_dir: 融合产物输出目录——跨目录模式下是**完整实验目录**；就地模式下是那棵
            已经写好各片对话目录的树（实验目录，或它的 ``runs/<dataset>``）。

    Returns:
        ``{"out_dir", "experiment_dir", "dataset_dir", "mode", "metrics", "gate",
        "n_questions"}``；``mode`` 为 ``"in_place"`` 或 ``"merge"``。

    Raises:
        ValueError: 账本口径 / 判分与可比性元数据不一致、账本条目不全或未跑完、账本与
            产物对不上、对话或题目覆盖不完整、题数合计与明细不符、qid 重复等校验失败。
        SystemExit: 入参目录不存在、入参是对话级产物目录、产物不完整、旧版产物结构、
            未融合的分片产物，或入参根本不是分片产物。
    """
    targets = [str(target) for target in (run_dirs or []) if str(target).strip()]
    out_path = os.path.abspath(out_dir)
    if not targets:
        # 就地：入参就是那棵已经写好各片对话目录的树
        shard_dirs = [_locate_shard_dir(out_dir)]
        in_place = True
    else:
        shard_dirs = [_locate_shard_dir(target) for target in targets]
        duplicated = sorted({d for d in shard_dirs if shard_dirs.count(d) > 1})
        if duplicated:
            raise _err(
                f"同一个分片目录被给了多次: {_brief(duplicated)}——每片只应给一次"
                "（同一个分片的产物只在一处）"
            )
        in_place = len(shard_dirs) == 1 and _is_own_dataset_dir(shard_dirs[0], out_path)

    shards = [_read_shard(dataset_dir) for dataset_dir in shard_dirs]
    header = _check_ledgers(shards)
    shards_n = int(header["shards"])
    shard_by = str(header.get("shard_by") or "question")
    limit = header.get("limit")
    by_index = _group_by_shard(shards, shards_n)

    _check_dialogues_against_ledger(
        by_index, shards, hint=_IN_PLACE_HINT if in_place else ""
    )
    ref_metrics = _check_shard_metadata(by_index)
    truth = _dataset_truth(
        str(ref_metrics.get("dataset_path") or ""),
        shards_n=shards_n,
        shard_by=shard_by,
        limit=limit,
        fingerprint=header.get("dataset_fingerprint"),
    )
    _check_coverage(by_index, truth)

    # 逐对话合并记录：question 分片下同一对话的记录来自多个分片，按片序拼接后
    # 再按源数据集的题目顺序重排（数据集不可读时保持拼接顺序）
    merged: dict[str, list[dict[str, Any]]] = {}
    copies: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, str] = {}
    for index, shard in sorted(by_index.items()):
        for name, item in sorted(shard["dialogues"].items()):
            merged.setdefault(name, []).extend(item["records"])
            copies.setdefault(name, []).append(item["metrics"])
            sources.setdefault(name, item["dir"])
    total_records = sum(len(records) for records in merged.values())
    counted = sum(int(shard["entry"].get("n_questions", 0) or 0) for shard in by_index.values())
    if counted != total_records:
        raise _err(
            f"各片账本记的题数合计 {counted} 与逐题明细条数 {total_records} 不一致"
            f"（账本或产物被动过）"
        )
    seen: set[str] = set()
    for records in merged.values():
        for record in records:
            qid = str(record.get("qid", ""))
            if qid in seen:
                raise _err(f"qid 重复出现在多个分片中: {qid}")
            seen.add(qid)
    if truth is not None:
        merged = dict(
            sorted(
                merged.items(),
                key=lambda item: (
                    truth["dialogue_order"].get(item[0], len(truth["dialogue_order"])),
                    item[0],
                ),
            )
        )
        unknown = len(truth["qid_order"])
        for records in merged.values():
            records.sort(key=lambda r: truth["qid_order"].get(str(r.get("qid")), unknown))
    records = [record for dialogue_records in merged.values() for record in dialogue_records]

    now = datetime.now()
    dataset_name = str(header["dataset"])
    if in_place:
        # 就地融合：输入树就是输出树（reports/ 必须落在实验根层，故要求 runs/ 层级）
        dataset_dir = shard_dirs[0]
        experiment_dir = os.path.dirname(os.path.dirname(dataset_dir))
        if os.path.basename(os.path.dirname(dataset_dir)) != layout.DIR_RUNS:
            raise _err(
                f"就地融合的入参不是实验目录下的数据集目录（缺 {layout.DIR_RUNS}/ 这一级，"
                f"任务级 reports/ 没有该落的地方）: {dataset_dir}"
            )
    else:
        experiment_dir = out_path
        dataset_dir = layout.dataset_dir(experiment_dir, dataset_name)
    # run_id 走 layout 的唯一组装入口：commit、分片与冒烟范围都进身份，另有 "fused"
    # 标出这是融合产物——融合结果（尤其 --limit 冒烟分片的融合）不得与全量结果撞身份。
    run_id = layout.run_id(
        str(ref_metrics.get("adapter") or "unknown"),
        ref_metrics.get("platform"),
        dataset_name,
        now.strftime("%Y%m%d_%H%M%S"),
        commit=str(ref_metrics.get("commit") or "") or None,
        tags=["fused", *layout.run_id_tags(shards_n, None, shard_by, limit)],
    )
    # 数据集级与逐对话共用的身份 / 口径元数据；可比性字段透传，融合产物与普通 run
    # 同等可校验、可对比
    base: dict[str, Any] = {
        "run_id": run_id,
        "adapter": ref_metrics.get("adapter"),
        "dataset": ref_metrics.get("dataset"),
        "dataset_path": ref_metrics.get("dataset_path"),
        "dataset_source": ref_metrics.get("dataset_source"),
        "platform": ref_metrics.get("platform", "default"),
        "commit": ref_metrics.get("commit"),
        "judge": ref_metrics.get("judge"),
        "timestamp": now.isoformat(timespec="seconds"),
        # 冒烟范围透传（各片一致）：冒烟融合产物与全量融合产物必须能被 compare 区分
        "limit": limit,
        "shards": shards_n,
        "shard_index": None,
        "shard_by": shard_by,
        "adapter_meta": ref_metrics.get("adapter_meta"),
        "adapter_describe": ref_metrics.get("adapter_describe"),
        "schema_version": ref_metrics.get("schema_version"),
        "driver_llm": ref_metrics.get("driver_llm"),
        "judge_model": ref_metrics.get("judge_model"),
        "adapter_version": ref_metrics.get("adapter_version"),
        "dataset_fingerprint": ref_metrics.get("dataset_fingerprint"),
        "retrieval_match": ref_metrics.get("retrieval_match"),
        # 运行模式透传（前面已校验各片一致）
        "run_mode": ref_metrics.get("run_mode") or "full",
        "experiment_dir": experiment_dir,
        "dataset_dir": dataset_dir,
    }
    if ref_metrics.get("retrieval_match_threshold") is not None:
        base["retrieval_match_threshold"] = ref_metrics["retrieval_match_threshold"]

    # 逐对话 token 记账：同一对话各片各记各的（question 分片下同一对话可能被多片
    # ingest，只有各片自己的计数才是真实成本），故直接相加、不去重。
    dialogue_usage = {
        name: _sum_usage([block.get("token_usage") for block in copies[name]])
        for name in merged
    }
    n_ignored = sum(
        int(block.get("n_ignored", 0) or 0)
        for blocks in copies.values()
        for block in blocks
    )

    metrics = dict(
        aggregate(
            records,
            capabilities=_caps_or(
                [block for blocks in copies.values() for block in blocks]
            ),
        )
    )
    metrics.update(base)
    metrics.update(
        {
            "n_ignored": n_ignored,
            "fused": True,
            "fused_from": [
                str(shard["entry"].get("run_id") or shard["dir"])
                for _, shard in sorted(by_index.items())
            ],
        }
    )
    # 检索回退数只记在数据集级：逐对话那份是各对话自己的计数，写进每份对话级
    # metrics 后再融合会被重复累加（同一个数字被算两次）
    fallbacks = sum(
        int(block.get("retrieval_match_fallbacks", 0) or 0)
        for blocks in copies.values()
        for block in blocks
    )
    if fallbacks:
        metrics["retrieval_match_fallbacks"] = fallbacks
    usage = _combine_usage([dialogue_usage[name] for name in merged])
    if usage is not None:
        metrics["token_usage"] = usage
    else:
        print(
            "[membench] ⚠️ 有分片未落账 token_usage：融合产物不落 token 消耗"
            "（部分求和会被读成全量消耗）"
        )
    gate = fuse_gate(records, metrics)
    metrics["stale_gate"] = gate

    if not in_place:
        # 跨目录融合：目标树是新的，逐对话报告必须按合并后的记录重建（question 分片
        # 下同一对话的记录来自多个分片，各片自己的对话级报告都只是子集），对话目录里
        # report/ 之外的内容（runtime / 快照）随对话并入。
        # 就地融合不重写对话级产物：那些目录本来就是各片各写各的（互不重叠），保留
        # 原样既留住了"哪个对话归哪一片"的证据，也让 fuse 可以安全地重复执行。
        for name, dialogue_records in merged.items():
            dialogue_id = _dialogue_id(dialogue_records, name)
            target_dir = os.path.join(dataset_dir, name)
            source_dir = sources[name]
            if os.path.abspath(source_dir) != os.path.abspath(target_dir):
                _copy_dialogue_content(source_dir, target_dir)
            dialogue_metrics = dict(
                aggregate(dialogue_records, capabilities=_caps_or(copies[name]))
            )
            dialogue_metrics.update(base)
            dialogue_metrics["dialogue_id"] = dialogue_id
            dialogue_metrics["n_ignored"] = sum(
                int(block.get("n_ignored", 0) or 0) for block in copies[name]
            )
            if dialogue_usage[name] is not None:
                dialogue_metrics["token_usage"] = dialogue_usage[name]
            _write_report_dir(
                os.path.join(target_dir, layout.DIR_REPORT),
                dialogue_records,
                dialogue_metrics,
                figures=False,
            )

    dataset_report = os.path.join(dataset_dir, layout.DIR_REPORT)
    _write_report_dir(dataset_report, records, metrics, figures=True)
    copy_report(
        os.path.join(dataset_report, layout.FILE_REPORT_MD),
        layout.task_report_dir(experiment_dir),
        layout.FILE_REPORT_MD,
        extra_files=[os.path.join(dataset_report, layout.FILE_METRICS)],
    )

    return {
        "out_dir": out_dir,
        "experiment_dir": experiment_dir,
        "dataset_dir": dataset_dir,
        "mode": "in_place" if in_place else "merge",
        "metrics": metrics,
        "gate": gate,
        "n_questions": len(records),
    }
