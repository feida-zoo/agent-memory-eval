"""membench 命令行入口：run / evaluate / evaluate-compare / fuse / compare /
matrix / series / task / doctor / adapter / user-data / dataset / list-datasets。

用法（SPEC 第 6 节）：
    python -m membench.cli task init --name NAME [--platform P] [--groups 1|2] \
        [--report-out DIR] [--replay]
        # 建任务目录骨架（groups=2 -> A/ B/；groups=1 -> runs/，都带任务级 reports/）并
        # 打印它的绝对路径；--replay 额外建任务级快照池 snapshots/（回放任务的
        # 内容寻址桶 + index.json）；stdout 只有这一行路径，可 TASK=$(...) 接走
    python -m membench.cli task status DIR
        # 进度视图：骨架 / 各组各数据集的分片账本进度（status、题数、对话数、run_id）/
        # 数据集级产物齐备与否 / 快照目录数与体积（回放任务改为快照池：桶数、体积、
        # 各组复用/新写、两组是否逐对话一致）/ 运行时（最近命令、非零退出条数、最近
        # 日志；该组/任务没有 runtime/ 时明确写「无运行时数据」）/ 下一步该跑什么；
        # DIR 是任务目录（含 A/ B/）或单组实验目录
    python -m membench.cli run --adapter mock-strong \
        [--dataset KEY|PATH] --judge heuristic \
        [--platform wx --commit a1b2c3] [--driver-llm gpt-4o] \
        [--adapter-version a1b2c3] \
        [--report-out DIR | --task-dir DIR] \
        # 二选一，见下「--task-dir 语义」
        [--adapter-arg KEY=VALUE]...   # 适配器构造参数，如 base_url=http://127.0.0.1:8787 \
        [--shards N|max --shard-index I --shard-by question|dialogue] \
        [--evidence-match substring|overlap|llm] \
        [--no-snapshot] [--trace-state | --no-trace-state] \
        # --no-snapshot：默认每对话 ingest 后导出状态快照（适配器不支持时警告一次并跳过）
        # --trace-state：失败归因，interleaved 数据集在每个有提问的 as_of 导一份状态到
        #   <对话目录>/trace_state/t<as_of>/（sequence 用末尾 memory_snapshot/）。
        #   缺省=自动：有 interleaved 对话含 safety 题时开启，--no-trace-state 关闭
        # interleaved 数据集（如 gatemem_office）要求 --shard-by dialogue，且不支持 --replay
        # --dataset：key（office / sample / locomo…，按需下载集先 dataset fetch）或 JSONL 路径
    python -m membench.cli run --adapter mock-strong --dataset ... \
        --replay <实验目录|数据集目录|回放任务或它的组目录> \
        [--replay-mode auto|restore|rebuild] \
        [--replay-verify-source <源产物：实验目录或数据集目录>]   # 快照回放：跳过 ingest，只跑 query；
        # 回放同样默认把恢复后的状态导出到任务级快照池 snapshots/（--no-snapshot 关闭）
    python -m membench.cli evaluate --adapter mock-strong \
        [--judge heuristic] [--platform wx --commit a1b2c3] \
        [--driver-llm gpt-4o] [--adapter-version a1b2c3] \
        [--report-out DIR | --task-dir DIR] [--adapter-arg KEY=VALUE]... \
        [--no-snapshot] [--trace-state | --no-trace-state] \
        # --trace-state：失败归因（仅 interleaved 对话生效；sequence 用末尾 memory_snapshot/）；
        #   缺省=自动（有 interleaved 对话含 safety 题时开启）
        [--quick | --beam-10m | --no-beam] [--no-lme-s] [--common] \
        [--fusion weighted|equal]   # 基准集融合口径（默认 weighted：按题数加权、封顶 500）\
        [--evidence-match substring|overlap|llm] \
        [--dataset KEY|PATH]...   # 显式数据集清单（与预设开关互斥） \
        [--from-runs KEY=DIR]...   # 复用已有产物；缺失数据集用 --adapter 补跑
    python -m membench.cli evaluate-compare --baseline-adapter A --candidate-adapter B \
        [--judge heuristic] [--platform wx --commit a1b2c3] \
        [--driver-llm-a gpt-4o] [--adapter-version-a a1b2c3] \
        [--driver-llm-b gpt-4o-mini] [--adapter-version-b d4e5f6] \
        [--adapter-arg-a KEY=VALUE]... [--adapter-arg-b KEY=VALUE]... \
        [--acc-threshold 0.02] [--stale-threshold 0.02] [--latency-warn 0.30] \
        [--report-out DIR | --task-dir DIR] [--no-snapshot] \
        [--trace-state | --no-trace-state] [--tag <name>] \
        [--quick | --beam-10m | --no-beam] [--common] \
        [--fusion weighted|equal]   # 两侧共用（默认 weighted）\
        [--no-lme-s] [--evidence-match substring|overlap|llm] \
        [--dataset KEY|PATH]...   # 两侧使用同一清单（与预设开关互斥） \
        [--baseline-run KEY=DIR]... [--candidate-run KEY=DIR]...  # 两侧复用+补跑
    python -m membench.cli fuse --runs <分片产物目录>,... --out <完整实验目录> \
        | python -m membench.cli fuse --task-dir <任务目录>   # 就地融合（不新建目录）
        # stale 门禁只对办公域槽位一票否决；非办公域数据集超阈值记 INFO（仅诊断）、退出码 0
    python -m membench.cli compare --baseline <产物目录> --candidate <产物目录> \
        [--acc-threshold 0.02] [--stale-threshold 0.02] [--latency-warn 0.30] \
        [--require-comparable] [--tag <name>] [--report-out DIR | --task-dir DIR]
        # --task-dir：就地读该任务目录的 A/（基线）与 B/（候选）（--baseline/--candidate
        # 显式给出时以显式为准），对比报告落它的 reports/，不再新建 compare_* 目录、
        # 也不复制两组；不传 --task-dir 时两组入参必填
    python -m membench.cli matrix --runs "wx@a1b2c3=<产物目录>" "fs@a1b2c3=<产物目录>" ... \
        [--out DIR] [--report-out DIR]
    python -m membench.cli series --line <commit-or-name>=<产物目录> --line ... \
        [--repo PATH] [--dataset NAME] [--report-out DIR] \
        [--require-comparable] [--fusion weighted|equal] [--fusion-cap N] \
        [--llm-base-url URL --llm-model M [--llm-api-key K] [--llm-timeout S]] \
        [--no-llm-naming]
        # 多线对比：--line 顺序即时间顺序，第一条默认叫「基线」。线名优先级：
        # 1) LLM（给了 --llm-base-url 与 --llm-model 时，从 commit 主题+正文+改动摘要
        #    抽一个简短中文名；输出必须含中文、≤24 字、不含标点）；
        # 2) 代码抽取（subject 去 feat(scope): / fix： 等前缀、剥首尾标点、截断 24 字）；
        # 未配置 LLM 直接代码抽取；LLM 失败/输出不合规告警一次后回退代码抽取，
        # git 取不到或不含中文回退短 sha（均不中断）。重名追加（sha8），
        # 显式线名永远优先且不调 LLM；--no-llm-naming 强制代码抽取。
        # 报告「来源与产物」表标出命名来源（llm / code / 显式名 / 基线 / 回退短 sha）。
        # 任一线的门禁 FAIL → 退出码 1
    python -m membench.cli doctor --target http://127.0.0.1:8787 \
        [--timeout 10.0] [--api-key K]
    python -m membench.cli adapter install <adapter.py> | list | remove <name>
    python -m membench.cli user-data install <jsonl路径> | status | delete
    python -m membench.cli dataset info [<key>] | dataset fetch <key> [--out DIR]
    python -m membench.cli list-datasets

产物一律落「一次实验一个自包含子目录」（见 :mod:`membench.layout`）::

    <report-out>/eval_<name>_<platform>_<ts>[_s<i>][_l<limit>]/   # 单组任务目录，例 eval_mock-strong_default_20261024-193000（分片/冒烟带后缀，见 layout.experiment_tags）
    ├── runtime/{commands.jsonl,logs/,env.json}  # 实验级运行时数据（命令 / 全量日志 / 环境快照）
    ├── runs/<dataset>/{report/,shards.json,<对话>/{report/,runtime/,memory_snapshot/,trace_state/}}
    └── reports/                                # 任务级（report.md / metrics.json / evaluate_summary.json / fig_dataset_overview.png）
    <report-out>/compare_<name>_<platform>_<ts>/{A/,B/,runtime/,reports/comparison.md}   # 双组任务目录
    <report-out>/matrix_<ts>/{runtime/,reports/,runs/<label>/}
    <report-out>/series_<ts>/{runtime/,reports/,runs/<线序>-<slug>/<dataset>/}   # 多线对比

回放任务（``task init --replay``）在任务目录根多一个内容寻址快照池 ``snapshots/``：
回放导出的状态按内容哈希去重成桶 ``snapshots/<sha256 前 12 位>/``，每个对话一个
条目文件 ``snapshots/entries/<组>/<数据集>/<对话>.json``（真源；分片并行回放时各
进程只写自己负责的对话，不需要锁），``snapshots/index.json`` 只是由条目 + 桶清单
重算的派生视图（``task status`` / ``fuse`` 收尾刷新）。A/B 两组回放同一源时只产生
一份桶（两组指向同一 digest），不同则各存一份——差异本身就是"回放不保真"的证据。
回放任务本身可以继续作为 ``--replay`` 源（链式回放）。

"产物目录"入参（``--baseline`` / ``--candidate`` / ``--from-runs`` / ``--baseline-run`` /
``--candidate-run`` / matrix 的 label 右值 / ``--replay`` / ``--replay-verify-source``）
接受**实验目录**或**数据集目录**（``<实验目录>/runs/<dataset>``），统一经
:func:`membench.layout.resolve_dataset_dir` 解析。

--report-out 统一语义：实验目录（``eval_*`` / ``compare_*`` / ``matrix_*``）的
父目录；缺省时用 paths.output_root()（MEMBENCH_OUTPUT_DIR > skill 的 output/ >
工程根）。

--task-dir 语义（与 --report-out 互斥，二选一）：把产物写进**已存在的任务目录**
（``task init`` 建的骨架，或它的某个组目录 ``A/`` / ``B/``），而不是新建 ``eval_*`` /
``compare_*`` 目录——同一次评测的多个分片因此能写进同一棵树：

- ``run --task-dir DIR`` / ``evaluate --task-dir DIR``：产物直接落 DIR；共享目录下
  只允许 ``--shard-by dialogue``（题目级分片会让多个分片写同一对话目录并互相覆盖），
  同一分片重复跑直接拒绝；
- ``evaluate-compare --task-dir DIR``：两组分别写 ``DIR/A``、``DIR/B``，对比报告落
  ``DIR/reports/``；
- ``compare --task-dir DIR``：就地读 ``DIR/A`` 与 ``DIR/B``，``comparison.md`` 落
  ``DIR/reports/``；
- ``fuse --task-dir DIR``：就地补齐数据集级与任务级产物（各片本来就写在这一棵树里）。

摄入策略与失败归因（与数据集绑定）：
- ``sequence``（默认，旧数据兼容）：整段对话 ingest 完再提问，``Question.as_of``
  不强制；
- ``interleaved``（如 gatemem_office）：按 ``as_of``（已 ingest 的消息条数）升序
  分组，边 ingest 边提问；要求 ``--shard-by dialogue``、不支持 ``--replay``；
- ``run --trace-state`` 对 interleaved 的每个有提问 as_of 导出状态到
  ``<对话目录>/trace_state/t<as_of>/``；缺省**自动**——本次有 interleaved 对话含
  safety 题时开启（safety 的「真遗忘」要读提问时点的状态），``--no-trace-state``
  关闭。``results.jsonl`` 每题记 ``as_of`` /
  ``evidence_in_state``，metrics 与报告给出失败归因四桶
  （not_written / not_retrieved / wrong_answer / unknown）。

任务目录的共同形态是 ``groups`` 决定的前缀（``eval_*`` / ``compare_*``），骨架由
:func:`membench.layout.init_task_dir` 唯一决定。
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any, Optional, Sequence

from membench import layout, runtime
from membench.adapters import ADAPTERS
from membench.compare import compare_runs
from membench.evaluate import FUSION_CHOICES, FUSION_WEIGHT_CAP
from membench.matrix import compare_matrix
from membench.paths import output_root
from membench.runner import Runner
from membench.series import LLMNaming, run_series
from membench.userdata import (
    delete_user_dataset,
    install_user_dataset,
    resolve_dataset,
    user_dataset_status,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="membench",
        description="membench — Agent 记忆系统评测框架",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = sub.add_parser("run", help="对一个记忆系统运行评测数据集")
    p_run.add_argument("--adapter", required=True, help="适配器名（见 list-datasets 提示 / ADAPTERS 注册表）")
    p_run.add_argument(
        "--adapter-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="适配器构造参数（可重复），如 --adapter-arg base_url=http://127.0.0.1:8787 --adapter-arg timeout=10；"
        "KEY 必须是该适配器构造函数的形参名（不认识会报错并列出可接受的参数名），值按形参注解转换（int/float 转数字，其余按字符串）",
    )
    p_run.add_argument(
        "--dataset",
        default=None,
        metavar="KEY|PATH",
        help="数据集 key（office/sample/locomo/…；按需下载集先 dataset fetch）或 JSONL 路径；"
        "不传时依次回退到已安装的用户测试集、内置 office 数据集"
        "（可用 dataset info 查看全部支持数据集）",
    )
    p_run.add_argument("--judge", default="heuristic", choices=["heuristic", "llm", "client"], help="判分器（默认 heuristic；llm=OpenAI兼容接口；client=agent内置LLM）")
    p_run.add_argument("--platform", default=None, help="平台标签（如 wx / fs），记入元数据与实验目录名")
    p_run.add_argument("--commit", default=None, help="commit 标签（如 a1b2c3），记入元数据")
    p_run.add_argument("--driver-llm", default=None, help="驱动被测系统的 LLM 标识（如 gpt-4o），记入元数据供 compare 可比性校验")
    p_run.add_argument("--adapter-version", default=None, help="适配器实现版本/commit，记入元数据供 compare 可比性校验")
    run_out = p_run.add_mutually_exclusive_group()
    run_out.add_argument("--report-out", default=None, help="实验目录父目录（缺省 output_root()）：产出 <DIR>/eval_<adapter>_<platform>_<ts>/，数据集产物落其 runs/<dataset>/，任务级报告落其 reports/")
    run_out.add_argument(
        "--task-dir",
        default=None,
        help="已存在的任务目录（task init 建的骨架，或它的组目录 A/ / B/）：产物直接写进该目录，多个分片可共用同一棵树"
        "（要求 --shard-by dialogue，同一分片重复跑会被拒绝）；与 --report-out 互斥",
    )
    p_run.add_argument("--llm-base-url", default=None, help="judge=llm 时的 OpenAI 兼容 base_url")
    p_run.add_argument("--llm-api-key", default=None, help="judge=llm 时的 API key")
    p_run.add_argument("--llm-model", default=None, help="judge=llm 时的模型名")
    p_run.add_argument("--llm-timeout", type=float, default=None, help="judge=llm 时的请求超时秒数（默认用 judge 内建值 90s）")
    p_run.add_argument(
        "--evidence-match",
        choices=("substring", "overlap", "llm"),
        default="substring",
        help="检索证据命中口径：substring=逐字子串（默认，历史口径）；overlap=token 包含度容错近原文改写；llm=语义判定（复用 --llm-* 配置，失败回退 substring）",
    )
    p_run.add_argument("--evidence-match-threshold", type=float, default=0.6, help="evidence-match=overlap 时的包含度阈值（默认 0.6）")
    p_run.add_argument("--jobs", type=int, default=1, help="题目级并发数（默认 1；ingest 始终串行；query / judge / 证据匹配在此并发下执行，要求 adapter.query 与 judge 线程安全）")
    p_run.add_argument("--client-llm", default=None, help="judge=client 时的 agent 内置 LLM 类路径，如 'my_agent.llm:LLMClient'")
    p_run.add_argument("--limit", type=int, default=None, help="冒烟模式：只评测（分片后的）前 N 道题，run_id 与 metadata 记录 limit")
    p_run.add_argument("--adapter-meta", default=None, help='适配器配置自由文本（"key=value;..."），原样记入 metadata["adapter_meta"]')
    p_run.add_argument(
        "--no-snapshot",
        action="store_true",
        help="不导出对话状态快照（默认导出：每个对话 ingest 完成后落到它的 memory_snapshot/，之后改读取侧代码可用 --replay 免重跑 ingest；"
        "适配器不支持快照时默认只警告一次并跳过）",
    )
    p_run.add_argument(
        "--trace-state",
        dest="trace_state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="失败归因：对 interleaved 数据集在每个有提问的 as_of ingest 完成后、query 之前"
        "导出状态到 <对话目录>/trace_state/t<as_of>/（sequence 对话用末尾 memory_snapshot/）；"
        "用于判断答错是「没写入 / 写入但没检索到 / 检索到但答错」。默认自动：本次有 "
        "interleaved 对话含 safety 题时开启（safety 的「真遗忘」要读提问时点的状态），"
        "其余情况不开；可用 --no-trace-state 关闭",
    )
    p_run.add_argument(
        "--abstention-recheck",
        dest="abstention_recheck",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="该拒答的题（gold=<ABSTAIN> 或 category=abstention，**不限 judge_spec**）"
        "词表判「未拒答」时，再发一次 YES/NO 复核（默认开启）；"
        "--no-abstention-recheck 关闭可省 LLM 调用，代价是改写式拒答会被记成未拒答",
    )
    p_run.add_argument(
        "--replay",
        default=None,
        help="回放源：实验目录、runs/<dataset> 数据集目录，或回放任务（含 snapshots/index.json 的任务目录 / 其组目录 A/ B/）；"
        "跳过 reset+ingest，按对话恢复快照后只跑 query（写入侧固定，结论只覆盖读取/回答层）。"
        "回放默认把恢复后的状态导出到任务级快照池 snapshots/（按内容去重，--no-snapshot 关闭）；"
        "interleaved（按 as_of 边 ingest 边提问）数据集不支持回放：只有末尾快照，给不出各时点状态",
    )
    p_run.add_argument(
        "--replay-mode",
        choices=("auto", "restore", "rebuild"),
        default="auto",
        help="回放模式（默认 auto：优先 restore，快照格式不兼容时退回 rebuild）",
    )
    p_run.add_argument(
        "--replay-verify-source",
        default=None,
        help="回放时与源产物逐题比对（答案/检索/判分），写进 metrics 与报告——回放误差地板；"
        "入参为实验目录或 runs/<dataset> 数据集目录",
    )
    p_run.add_argument(
        "--shards",
        type=_parse_shards,
        default=1,
        help="数据集分片总数（默认 1 不分片）；可传整数或 max（自动按数据集上限取最大：dialogue=对话数、question=题目数）；超过上限的整数按上限执行，越界分片索引安全跳过",
    )
    p_run.add_argument("--shard-index", type=int, default=0, help="本分片序号（从 0 开始，需 < --shards；与 --limit 组合时先分片再截断）")
    p_run.add_argument("--shard-by", choices=["question", "dialogue"], default="question", help="分片维度：question 按题目（默认）；dialogue 整段选取对话、分片间不重复 ingest，适合几乎每个对话都有题的数据集（如 BEAM）")
    p_run.set_defaults(func=_cmd_run)

    # --- compare ---
    p_cmp = sub.add_parser("compare", help="对比两个产物目录（基线 vs 候选）并输出门禁结论")
    p_cmp.add_argument("--baseline", default=None, help="基线产物目录：实验目录或 runs/<dataset> 数据集目录；--task-dir 时缺省取该任务目录的 A/")
    p_cmp.add_argument("--candidate", default=None, help="候选产物目录：实验目录或 runs/<dataset> 数据集目录；--task-dir 时缺省取该任务目录的 B/")
    p_cmp.add_argument("--acc-threshold", type=float, default=0.02, help="accuracy 允许最大下降（默认 0.02）")
    p_cmp.add_argument("--stale-threshold", type=float, default=0.02, help="stale_rate 允许最大上升（默认 0.02）")
    p_cmp.add_argument("--latency-warn", type=float, default=0.30, help="latency_p95 上升告警比例（默认 0.30）")
    p_cmp.add_argument("--require-comparable", action="store_true", help="strict 可比性：关键维度不一致或无法核验时门禁 FAIL")
    p_cmp.add_argument("--tag", default=None, help="对比实验目录名的 <name> 段（缺省 <基线 adapter>_vs_<候选 adapter>）")
    p_cmp.add_argument("--baseline-name", default=None, help="基线组显示名（缺省 基线）")
    p_cmp.add_argument("--candidate-name", default=None, help="候选组显示名（缺省 候选）")
    cmp_out = p_cmp.add_mutually_exclusive_group()
    cmp_out.add_argument("--report-out", default=None, help="实验目录父目录（缺省 output_root()）：产出 <DIR>/compare_<name>_<platform>_<ts>/，A/ 与 B/ 各为完整评测产物，报告落其 reports/")
    cmp_out.add_argument(
        "--task-dir",
        default=None,
        help="已存在的任务目录：就地读它的 A/（基线）与 B/（候选）（--baseline / --candidate 显式给出时以显式为准），"
        "comparison.md 落它的 reports/——不新建 compare_* 目录、不复制两组；与 --report-out 互斥",
    )
    p_cmp.set_defaults(func=_cmd_compare)

    # --- fuse ---
    p_fuse = sub.add_parser("fuse", help="融合同一评测的多个分片产物（跨目录合并或就地补齐；stale 门禁只对办公域槽位一票否决，非办公域超阈值记 INFO）")
    p_fuse.add_argument("--runs", default=None, help="逗号分隔的分片产物目录列表（实验目录或 runs/<dataset> 数据集目录；需覆盖全部 shard_index）；与 --out 一起用，或用 --task-dir 就地融合")
    p_fuse.add_argument("--out", default=None, help="融合产物输出的完整实验目录：数据集产物落其 runs/<dataset>/，任务级报告落其 reports/")
    p_fuse.add_argument(
        "--task-dir",
        default=None,
        help="就地融合：入参是各分片共同写入的任务目录（或它的 runs/<dataset>），直接在该树里补齐数据集级与任务级产物，"
        "不新建目录、不搬动对话产物（与 --runs/--out 二选一）",
    )
    p_fuse.set_defaults(func=_cmd_fuse)

    # --- matrix ---
    p_mtx = sub.add_parser("matrix", help="多产物目录矩阵对比（platform × commit 维度）")
    p_mtx.add_argument(
        "--runs",
        required=True,
        nargs="+",
        help='带标签的产物目录，形如 "platform@commit=<实验目录或 runs/<dataset> 数据集目录>"（label 也接受任意字符串）',
    )
    p_mtx.add_argument("--out", default=None, help="矩阵实验目录（完整实验目录）：报告落其 reports/，各 label 产物落其 runs/<label>/")
    p_mtx.add_argument("--report-out", default=None, help="实验目录父目录（缺省 output_root()）：产出 <DIR>/matrix_<ts>/（--out 优先于本参数）")
    p_mtx.set_defaults(func=_cmd_matrix)

    # --- series ---
    p_ser = sub.add_parser(
        "series",
        help="多线对比：N 个版本按时间顺序逐步对比（线名优先 LLM 从 commit+改动抽取，第一条为基线）",
        description=(
            "多线对比：做多个优化点、每个优化点一个 commit 时，看每一步提升多少。"
            "--line 按时间顺序重复给出。线名优先级：LLM（给了 --llm-base-url 与 "
            "--llm-model 时，从 commit 主题+正文+改动摘要抽一个简短中文名）→ 代码抽取"
            "（左侧 7~40 位十六进制视为 commit，subject 去掉 feat(scope): / fix： 等 "
            "conventional 前缀、剥首尾标点后截断到 24 字）；第一条默认叫「基线」，"
            "显式线名永远优先。LLM 未配置直接代码抽取；失败或输出不合规时告警一次"
            "并回退代码抽取；git 取不到或不含中文时回退短 sha（均不中断）。"
        ),
    )
    p_ser.add_argument(
        "--line",
        action="append",
        required=True,
        metavar="COMMIT-OR-NAME=DIR",
        help="一条线，按时间顺序重复给出（至少两条）：左侧是 commit（7~40 位十六进制）或显式线名，"
        "右侧是实验目录或 runs/<dataset> 数据集目录",
    )
    p_ser.add_argument("--repo", default=None, help="读取 commit subject 的 git 仓库路径（默认当前目录）")
    p_ser.add_argument("--dataset", default=None, help="只比某个数据集（多数据集实验目录时用它指定）；缺省比全部数据集")
    p_ser.add_argument("--report-out", default=None, help="实验目录父目录（缺省 output_root()）：产出 <DIR>/series_<ts>/")
    p_ser.add_argument(
        "--require-comparable",
        action="store_true",
        help="strict 可比性：每条线 vs 基线的关键维度不一致或无法核验时门禁 FAIL",
    )
    p_ser.add_argument(
        "--fusion",
        choices=list(FUSION_CHOICES),
        default="weighted",
        help="多数据集融合口径（默认 weighted：按参与题数加权、单集封顶 500；办公域槽位不参与）",
    )
    p_ser.add_argument(
        "--fusion-cap",
        type=int,
        default=FUSION_WEIGHT_CAP,
        help=f"加权融合的单数据集权重封顶（题数，默认 {FUSION_WEIGHT_CAP}）",
    )
    p_ser.add_argument(
        "--llm-base-url",
        default=None,
        help="线名 LLM 的 OpenAI 兼容 base_url（与 --llm-model 同时给出才启用）",
    )
    p_ser.add_argument("--llm-api-key", default=None, help="线名 LLM 的 API key")
    p_ser.add_argument(
        "--llm-model",
        default=None,
        help="线名 LLM 的模型名（与 --llm-base-url 同时给出才启用）",
    )
    p_ser.add_argument(
        "--llm-timeout",
        type=float,
        default=None,
        help="线名 LLM 的请求超时秒数（默认 90s）",
    )
    p_ser.add_argument(
        "--no-llm-naming",
        action="store_true",
        help="强制用 commit 信息代码抽取命名，不调 LLM（即使给了 --llm-*）",
    )
    p_ser.set_defaults(func=_cmd_series)

    # --- task ---
    p_task = sub.add_parser("task", help="任务目录管理（init / status）：一次评测的产物根与进度视图")
    task_sub = p_task.add_subparsers(dest="task_command", required=True)
    p_task_init = task_sub.add_parser(
        "init", help="建任务目录骨架并把它的绝对路径打印到 stdout（供 TASK=$(...) 接走）"
    )
    p_task_init.add_argument("--name", required=True, help="任务名（进目录名，如 a_vs_b）")
    p_task_init.add_argument("--platform", default=None, help="平台标签（如 wx / fs），进目录名；缺省 default")
    p_task_init.add_argument(
        "--groups",
        type=int,
        choices=(1, 2),
        default=2,
        help="组数：2（默认）= 双组任务目录 compare_<name>_<platform>_<ts>/（两个组目录各是一个完整评测产物根，"
        "目录名由 --baseline-name / --candidate-name 给）；1 = 单组 eval_<name>_<platform>_<ts>/（含 runs/）",
    )
    p_task_init.add_argument(
        "--baseline-name",
        default=layout.DEFAULT_BASELINE_LABEL,
        help=f"基线组显示名 / 目录名（默认 {layout.DEFAULT_BASELINE_LABEL}，允许中文）",
    )
    p_task_init.add_argument(
        "--candidate-name",
        default="候选",
        help="候选组显示名（进报告与清单，允许中文；默认「候选」）",
    )
    p_task_init.add_argument(
        "--baseline-dir",
        default=None,
        help="基线组目录名（ASCII，缺省 baseline）；与 --baseline-name 分开：目录名不用中文",
    )
    p_task_init.add_argument(
        "--candidate-dir",
        default=None,
        help="候选组目录名（ASCII，缺省 candidate）；中文组名请在这里给英文目录名",
    )
    p_task_init.add_argument("--report-out", default=None, help="任务目录的父目录（缺省 output_root()）")
    p_task_init.add_argument(
        "--replay",
        action="store_true",
        help="回放任务：骨架额外建任务级快照池 snapshots/（回放导出的状态按内容去重成桶，"
        "A/B 回放同一源只存一份；不带本参数时不建 snapshots/）",
    )
    p_task_init.set_defaults(func=_cmd_task_init)
    p_task_status = task_sub.add_parser(
        "status", help="进度视图：骨架 / 分片账本进度 / 数据集级产物 / 快照体积 / 下一步"
    )
    p_task_status.add_argument("dir", metavar="DIR", help="任务目录（含 A/ B/）或单组实验目录")
    p_task_status.set_defaults(func=_cmd_task_status)

    # --- user-data ---
    p_ud = sub.add_parser("user-data", help="用户测试集管理（install / status / delete）")
    ud_sub = p_ud.add_subparsers(dest="user_data_command", required=True)
    p_ud_install = ud_sub.add_parser("install", help="校验并安装用户测试集")
    p_ud_install.add_argument("src", help="用户测试集 JSONL 路径")
    p_ud_install.set_defaults(func=_cmd_user_data_install)
    p_ud_status = ud_sub.add_parser("status", help="查看用户测试集状态")
    p_ud_status.set_defaults(func=_cmd_user_data_status)
    p_ud_delete = ud_sub.add_parser("delete", help="删除用户测试集")
    p_ud_delete.set_defaults(func=_cmd_user_data_delete)

    # --- evaluate ---
    p_ev = sub.add_parser("evaluate", help="全量评估：多数据集依次评测 + 基准集加权融合出基准评级 + 办公域 stale 门禁（office 单列业务结论）")
    p_ev.add_argument("--adapter", default=None, help="适配器名（见 list-datasets 提示 / ADAPTERS 注册表）；--from-runs 全复用时可省略")
    p_ev.add_argument(
        "--adapter-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="适配器构造参数（可重复），如 --adapter-arg base_url=http://127.0.0.1:8787；"
        "KEY 必须是该适配器构造函数的形参名（不认识会报错并列出可接受的参数名），值按形参注解转换（int/float 转数字，其余按字符串）",
    )
    p_ev.add_argument("--judge", default=None, choices=["heuristic", "llm", "client"], help="判分器（默认 heuristic；--from-runs 全复用/纯汇总时以已有 run 判分器为准）")
    p_ev.add_argument("--platform", default=None, help="平台标签（如 wx / fs），记入元数据与实验目录名")
    p_ev.add_argument("--commit", default=None, help="commit 标签（如 a1b2c3），记入元数据")
    p_ev.add_argument("--driver-llm", default=None, help="驱动被测系统的 LLM 标识（如 gpt-4o），记入每个 run 的元数据")
    p_ev.add_argument("--adapter-version", default=None, help="适配器实现版本/commit，记入每个 run 的元数据")
    ev_out = p_ev.add_mutually_exclusive_group()
    ev_out.add_argument("--report-out", default=None, help="实验目录父目录（缺省 output_root()）：产出 <DIR>/eval_<adapter>_<platform>_<ts>/，各数据集落其 runs/<dataset>/，任务级报告落其 reports/")
    ev_out.add_argument(
        "--task-dir",
        default=None,
        help="已存在的任务目录：产物直接写进该目录（各数据集落它的 runs/<dataset>/、报告落它的 reports/），不再新建 eval_* 目录；"
        "与 --report-out 互斥",
    )
    p_ev.add_argument(
        "--no-snapshot",
        action="store_true",
        help="不导出对话状态快照（默认导出：每个对话 ingest 完成后落到它的 memory_snapshot/；适配器不支持快照时默认只警告一次并跳过）",
    )
    p_ev.add_argument(
        "--trace-state",
        dest="trace_state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="失败归因：透传给每个数据集的 run，仅 interleaved 对话生效"
        "（在每个有提问的 as_of 导出 <对话目录>/trace_state/t<as_of>/）；"
        "sequence 对话用末尾 memory_snapshot/。默认自动：有 interleaved 对话含 safety 题时"
        "开启（safety 的「真遗忘」要读提问时点的状态），可用 --no-trace-state 关闭",
    )
    p_ev.add_argument(
        "--abstention-recheck",
        dest="abstention_recheck",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="该拒答的题（gold=<ABSTAIN> 或 category=abstention，**不限 judge_spec**）"
        "词表判「未拒答」时，再发一次 YES/NO 复核（默认开启）；"
        "--no-abstention-recheck 关闭可省 LLM 调用，代价是改写式拒答会被记成未拒答",
    )
    ev_group = p_ev.add_mutually_exclusive_group()
    ev_group.add_argument("--quick", action="store_true", help="快速模式：跳过 BEAM 与 LongMemEval-S（等价 --no-beam --no-lme-s，分钟级回归）")
    ev_group.add_argument("--beam-10m", action="store_true", help="BEAM 槽位使用 beam_10m.jsonl（默认 beam_1m.jsonl）")
    p_ev.add_argument("--no-beam", action="store_true", help="不评测 BEAM 槽位（与 --beam-10m 互斥）")
    p_ev.add_argument("--no-lme-s", action="store_true", help="剔除 LongMemEval-S 变体（默认包含；~268MB、高成本）")
    p_ev.add_argument(
        "--common",
        action="store_true",
        help="公开基准常用集：locomo / longmemeval_oracle / gatemem_office / beam_1m，"
        "不含办公域槽位（office / 用户测试集）与 LongMemEval-S；与 --quick 互斥，"
        "--no-beam 跳过 BEAM、--beam-10m 换 BEAM 全集，--no-lme-s 在本预设下无效果",
    )
    p_ev.add_argument(
        "--fusion",
        choices=("weighted", "equal"),
        default="weighted",
        help="基准集融合口径：weighted=按参与题数加权、单数据集封顶 500（默认）；"
        "equal=基准集内等权（旧宏平均语义）。两种口径都不含办公域槽位（office 单列业务结论），"
        "口径写入 metrics 并参与可比性校验（旧产物缺该字段归一为 legacy，与两者都不可比）",
    )
    p_ev.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="KEY|PATH",
        help="显式指定评测数据集（数据集 key 或 JSONL 路径，可重复，列表中第一个承担办公域槽位）；与预设开关（--quick / --beam-10m / --no-beam / --no-lme-s）互斥",
    )
    p_ev.add_argument(
        "--from-runs",
        action="append",
        default=None,
        metavar="KEY=DIR",
        help="复用已有产物进行汇总（可重复，KEY 为数据集 key、DIR 为实验目录或 runs/<dataset> 数据集目录）；目标清单中缺失的数据集用 --adapter 补跑；与预设开关互斥",
    )
    p_ev.add_argument("--llm-base-url", default=None, help="judge=llm 时的 OpenAI 兼容 base_url")
    p_ev.add_argument("--llm-api-key", default=None, help="judge=llm 时的 API key")
    p_ev.add_argument("--llm-model", default=None, help="judge=llm 时的模型名")
    p_ev.add_argument("--llm-timeout", type=float, default=None, help="judge=llm 时的请求超时秒数（默认用 judge 内建值 90s）")
    p_ev.add_argument(
        "--evidence-match",
        choices=("substring", "overlap", "llm"),
        default="substring",
        help="检索证据命中口径：substring=逐字子串（默认，历史口径）；overlap=token 包含度容错近原文改写；llm=语义判定（复用 --llm-* 配置，失败回退 substring）",
    )
    p_ev.add_argument("--evidence-match-threshold", type=float, default=0.6, help="evidence-match=overlap 时的包含度阈值（默认 0.6）")
    p_ev.add_argument("--jobs", type=int, default=1, help="题目级并发数（默认 1；ingest 始终串行；query / judge / 证据匹配在此并发下执行，要求 adapter.query 与 judge 线程安全）")
    p_ev.add_argument("--client-llm", default=None, help="judge=client 时的 agent 内置 LLM 类路径，如 'my_agent.llm:LLMClient'")
    p_ev.set_defaults(func=_cmd_evaluate)

    # --- evaluate-compare ---
    p_ev_cmp = sub.add_parser(
        "evaluate-compare",
        help="两组（baseline vs candidate）× 多数据集融合 A/B 对比；产出 compare_<name>_<platform>_<ts>/",
    )
    p_ev_cmp.add_argument("--baseline-adapter", default=None, help="基线适配器名；该侧全复用时可省略")
    p_ev_cmp.add_argument("--candidate-adapter", default=None, help="候选适配器名；该侧全复用时可省略")
    p_ev_cmp.add_argument(
        "--adapter-arg-a",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="基线侧适配器构造参数（可重复），如 --adapter-arg-a base_url=http://127.0.0.1:8787；"
        "KEY 必须是该适配器构造函数的形参名，值按形参注解转换（int/float 转数字，其余按字符串）",
    )
    p_ev_cmp.add_argument(
        "--adapter-arg-b",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="候选侧适配器构造参数（可重复），语义同 --adapter-arg-a",
    )
    p_ev_cmp.add_argument(
        "--baseline-run",
        action="append",
        default=None,
        metavar="KEY=DIR",
        help="复用已有的基线产物（可重复，KEY 为数据集 key、DIR 为实验目录或 runs/<dataset> 数据集目录）；缺失数据集用 --baseline-adapter 补跑",
    )
    p_ev_cmp.add_argument(
        "--candidate-run",
        action="append",
        default=None,
        metavar="KEY=DIR",
        help="复用已有的候选产物（可重复，KEY 为数据集 key、DIR 为实验目录或 runs/<dataset> 数据集目录）；缺失数据集用 --candidate-adapter 补跑",
    )
    p_ev_cmp.add_argument("--judge", default=None, choices=["heuristic", "llm", "client"], help="判分器（默认 heuristic；两侧全复用时以已有产物判分器为准）")
    p_ev_cmp.add_argument("--platform", default=None, help="平台标签（如 wx / fs），记入两侧产物元数据与实验目录名")
    p_ev_cmp.add_argument("--commit", default=None, help="commit 标签，记入两侧产物元数据")
    p_ev_cmp.add_argument("--driver-llm-a", default=None, help="基线侧的 driver LLM 标识")
    p_ev_cmp.add_argument("--driver-llm-b", default=None, help="候选侧的 driver LLM 标识")
    p_ev_cmp.add_argument("--adapter-version-a", default=None, help="基线侧 adapter 版本/commit")
    p_ev_cmp.add_argument("--adapter-version-b", default=None, help="候选侧 adapter 版本/commit")
    p_ev_cmp.add_argument("--acc-threshold", type=float, default=0.02, help="accuracy 允许最大下降（融合维度，默认 0.02）")
    p_ev_cmp.add_argument("--stale-threshold", type=float, default=0.02, help="stale_rate 允许最大上升（融合维度，默认 0.02）")
    p_ev_cmp.add_argument("--latency-warn", type=float, default=0.30, help="latency_p95 相对上升告警比例（融合维度，默认 0.30）")
    p_ev_cmp.add_argument("--tag", default=None, help="对比实验目录名的 <name> 段（缺省 <基线 adapter>_vs_<候选 adapter>）")
    ev_cmp_out = p_ev_cmp.add_mutually_exclusive_group()
    ev_cmp_out.add_argument("--report-out", default=None, help="实验目录父目录（缺省 output_root()）：产出 <DIR>/compare_<name>_<platform>_<ts>/，A/ 与 B/ 各为完整评测产物，报告落其 reports/")
    ev_cmp_out.add_argument(
        "--task-dir",
        default=None,
        help="已存在的任务目录：两组分别写它的 A/（基线）与 B/（候选），对比报告落它的 reports/，不再新建 compare_* 目录；"
        "与 --report-out 互斥",
    )
    p_ev_cmp.add_argument(
        "--no-snapshot",
        action="store_true",
        help="不导出对话状态快照（默认导出：每个对话 ingest 完成后落到它的 memory_snapshot/；适配器不支持快照时默认只警告一次并跳过）",
    )
    p_ev_cmp.add_argument(
        "--trace-state",
        dest="trace_state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="失败归因：两侧逐数据集透传给 run，仅 interleaved 对话生效"
        "（在每个有提问的 as_of 导出 <对话目录>/trace_state/t<as_of>/）；"
        "sequence 对话用末尾 memory_snapshot/。默认自动：有 interleaved 对话含 safety 题时"
        "开启（safety 的「真遗忘」要读提问时点的状态），可用 --no-trace-state 关闭",
    )
    p_ev_cmp.add_argument(
        "--abstention-recheck",
        dest="abstention_recheck",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="该拒答的题（gold=<ABSTAIN> 或 category=abstention，**不限 judge_spec**）"
        "词表判「未拒答」时，再发一次 YES/NO 复核（默认开启）；"
        "--no-abstention-recheck 关闭可省 LLM 调用，代价是改写式拒答会被记成未拒答",
    )
    p_ev_cmp.add_argument(
        "--baseline-name",
        default=None,
        help="基线组显示名（进 groups.json 与报告；缺省 基线）",
    )
    p_ev_cmp.add_argument(
        "--candidate-name",
        default=None,
        help="候选组显示名（进 groups.json 与报告；缺省 候选）",
    )
    p_ev_cmp.add_argument(
        "--baseline-dir",
        default=None,
        help="基线组目录名（ASCII，缺省 baseline）",
    )
    p_ev_cmp.add_argument(
        "--candidate-dir",
        default=None,
        help="候选组目录名（ASCII，缺省 candidate）",
    )
    ev_cmp_group = p_ev_cmp.add_mutually_exclusive_group()
    ev_cmp_group.add_argument("--quick", action="store_true", help="快速模式：跳过 BEAM 与 LongMemEval-S（等价 --no-beam --no-lme-s）")
    ev_cmp_group.add_argument("--beam-10m", action="store_true", help="BEAM 槽位使用 beam_10m.jsonl")
    p_ev_cmp.add_argument("--no-beam", action="store_true", help="不评测 BEAM 槽位（与 --beam-10m 互斥）")
    p_ev_cmp.add_argument("--no-lme-s", action="store_true", help="剔除 LongMemEval-S 变体（默认包含）")
    p_ev_cmp.add_argument(
        "--common",
        action="store_true",
        help="公开基准常用集：locomo / longmemeval_oracle / gatemem_office / beam_1m，"
        "不含办公域槽位（office / 用户测试集）与 LongMemEval-S；与 --quick 互斥，"
        "--no-beam 跳过 BEAM、--beam-10m 换 BEAM 全集，--no-lme-s 在本预设下无效果",
    )
    p_ev_cmp.add_argument(
        "--fusion",
        choices=("weighted", "equal"),
        default="weighted",
        help="两侧共用的基准集融合口径：weighted=按参与题数加权、单数据集封顶 500（默认）；"
        "equal=基准集内等权（旧宏平均语义）。两侧必须同口径；办公域槽位不参与基准评级，"
        "单列业务 diff 与红线门禁",
    )
    p_ev_cmp.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="KEY|PATH",
        help="显式指定评测数据集（数据集 key 或 JSONL 路径，可重复，两侧使用同一清单，第一个承担办公域槽位）；与预设开关（--quick / --beam-10m / --no-beam / --no-lme-s）互斥",
    )
    p_ev_cmp.add_argument("--llm-base-url", default=None, help="judge=llm 时的 OpenAI 兼容 base_url")
    p_ev_cmp.add_argument("--llm-api-key", default=None, help="judge=llm 时的 API key")
    p_ev_cmp.add_argument("--llm-model", default=None, help="judge=llm 时的模型名")
    p_ev_cmp.add_argument("--llm-timeout", type=float, default=None, help="judge=llm 时的请求超时秒数（默认 90s）")
    p_ev_cmp.add_argument(
        "--evidence-match",
        choices=("substring", "overlap", "llm"),
        default="substring",
        help="检索证据命中口径（两侧一致）：substring=逐字子串（默认，历史口径）；overlap=token 包含度容错近原文改写；llm=语义判定（复用 --llm-* 配置，失败回退 substring）",
    )
    p_ev_cmp.add_argument("--evidence-match-threshold", type=float, default=0.6, help="evidence-match=overlap 时的包含度阈值（默认 0.6）")
    p_ev_cmp.add_argument("--jobs", type=int, default=1, help="题目级并发数（默认 1；ingest 始终串行；query / judge / 证据匹配在此并发下执行，要求 adapter.query 与 judge 线程安全）")
    p_ev_cmp.add_argument("--client-llm", default=None, help="judge=client 时的 agent 内置 LLM 类路径")
    p_ev_cmp.set_defaults(func=_cmd_evaluate_compare)

    # --- adapter ---
    p_ad = sub.add_parser("adapter", help="业务适配器持久化管理（install / list / remove）")
    ad_sub = p_ad.add_subparsers(dest="adapter_command", required=True)
    p_ad_install = ad_sub.add_parser("install", help="把适配器 .py 落盘到持久化目录，跨 session 可用")
    p_ad_install.add_argument("src", help="适配器 .py 源文件路径")
    p_ad_install.add_argument("--name", default=None, help="注册名（缺省用文件名 stem；源码可推断时以源码为准）")
    p_ad_install.set_defaults(func=_cmd_adapter_install)
    p_ad_list = ad_sub.add_parser("list", help="列出内置与持久化适配器")
    p_ad_list.set_defaults(func=_cmd_adapter_list)
    p_ad_remove = ad_sub.add_parser("remove", help="删除持久化适配器")
    p_ad_remove.add_argument("name", help="安装时的文件名（<name>.py / <name>.json）")
    p_ad_remove.set_defaults(func=_cmd_adapter_remove)

    # --- list-datasets ---
    p_ls = sub.add_parser("list-datasets", help="列出内置数据集")
    p_ls.set_defaults(func=_cmd_list_datasets)

    # --- dataset ---
    p_ds = sub.add_parser("dataset", help="数据集介绍与按需下载（info / fetch）")
    ds_sub = p_ds.add_subparsers(dest="dataset_command", required=True)
    p_ds_info = ds_sub.add_parser("info", help="介绍数据集；不传 name 时概览全部支持的数据集")
    p_ds_info.add_argument("name", nargs="?", default=None, help="数据集 key（可选；未知 key 列出全部可用 key）")
    p_ds_info.set_defaults(func=_cmd_dataset_info)
    p_ds_fetch = ds_sub.add_parser("fetch", help="下载并转换需按需获取的数据集（locomo / longmemeval_* / gatemem_office / beam_*）")
    p_ds_fetch.add_argument("name", help="数据集 key（仅支持 status=fetch 的条目）")
    p_ds_fetch.add_argument("--out", default=None, help="输出目录（默认 MEMBENCH_DATASETS_DIR 或 ~/.membench/datasets/）")
    p_ds_fetch.set_defaults(func=_cmd_dataset_fetch)

    # --- doctor ---
    p_doc = sub.add_parser("doctor", help="探测被测服务是否符合 HttpAdapter 契约（reset/ingest/query 三端点）")
    p_doc.add_argument("--target", required=True, help="被测服务根地址，如 http://127.0.0.1:8787")
    p_doc.add_argument("--timeout", type=float, default=10.0, help="单次请求超时秒数（默认 10.0）")
    p_doc.add_argument("--api-key", default=None, help="非空时给每个请求加 Authorization: Bearer <key>")
    p_doc.set_defaults(func=_cmd_doctor)

    return parser


def _parse_shards(value: str) -> int | str:
    """--shards 解析：>= 1 的整数，或 'max'（按数据集上限自动取最大分片）。"""
    text = str(value).strip().lower()
    if text == "max":
        return "max"
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("--shards 需为 >= 1 的整数或 'max'")
    if n < 1:
        raise argparse.ArgumentTypeError(f"--shards 必须 >= 1 或 'max'，当前为 {n}")
    return n


def _adapter_ctor_spec(target: object) -> tuple[dict[str, inspect.Parameter], bool]:
    """适配器构造形参表（形参名 -> Parameter）与是否接受任意关键字（``**kwargs``）。

    注册表项可以是类，也可以是工厂函数（插件常用 ``lambda: MyAdapter(...)``），
    故内省对象本身而非 ``__init__``。无法内省（C 实现等）时返回 ``({}, True)``：
    不做 key 校验，交给实例化时的 TypeError 兜底。
    """
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return {}, True
    accepted: dict[str, inspect.Parameter] = {}
    variadic_keywords = False
    for param in signature.parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            variadic_keywords = True
        elif param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            accepted[param.name] = param
    return accepted, variadic_keywords


def _adapter_arg_value(param: inspect.Parameter | None, key: str, raw: str) -> object:
    """按构造形参的注解做最小强制转换：``int`` / ``float`` 注解转数字，其余保持字符串。"""
    annotation = param.annotation if param is not None else inspect.Parameter.empty
    # 适配器模块常用 ``from __future__ import annotations``，此时注解是字符串
    label = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    if label in ("int", "float"):
        caster = int if label == "int" else float
        try:
            return caster(raw)
        except ValueError:
            raise SystemExit(f"--adapter-arg {key}={raw} 无法按 {label} 解析")
    return raw


def _parse_adapter_args(
    name: str, target: object, adapter_args: Sequence[str]
) -> dict[str, Any]:
    """解析 ``--adapter-arg KEY=VALUE``：校验 key 是构造形参名，值按注解转换。

    同一个 key 给多次时后者生效。key 不认识 / ``KEY=VALUE`` 格式非法都直接
    SystemExit——参数错误必须在开跑前暴露，而不是等实例化时的 TypeError 堆栈。
    """
    accepted, variadic_keywords = _adapter_ctor_spec(target)
    kwargs: dict[str, Any] = {}
    for item in adapter_args:
        key, sep, raw = str(item).partition("=")
        key = key.strip()
        if not sep or not key:
            raise SystemExit(
                f"--adapter-arg 需要 KEY=VALUE 形式（如 base_url=http://127.0.0.1:8787），"
                f"当前为: {item!r}"
            )
        if key not in accepted and not variadic_keywords:
            names = ", ".join(sorted(accepted)) or "(该适配器不接受构造参数)"
            raise SystemExit(f"适配器 {name} 不认识参数 {key}；可接受的参数: {names}")
        kwargs[key] = _adapter_arg_value(accepted.get(key), key, raw)
    return kwargs


def _make_adapter(name: str, adapter_args: Sequence[str] = ()) -> Any:
    """按注册表名实例化适配器；``adapter_args`` 为 ``KEY=VALUE`` 形式的构造参数。

    构造参数走通用机制（``--adapter-arg``），不给单个适配器开专用开关；必填参数
    没给时把 TypeError 转成可操作的 SystemExit（提示用 ``--adapter-arg`` 传参）。
    """
    if name not in ADAPTERS:
        available = ", ".join(sorted(ADAPTERS)) or "(空)"
        raise SystemExit(f"未知适配器: {name}。可用: {available}")
    target = ADAPTERS[name]
    kwargs = _parse_adapter_args(name, target, adapter_args)
    try:
        return target(**kwargs)
    except TypeError as exc:
        accepted, _ = _adapter_ctor_spec(target)
        names = ", ".join(sorted(accepted)) or "(该适配器不接受构造参数)"
        raise SystemExit(
            f"适配器 {name} 实例化失败: {exc}；构造参数用 --adapter-arg KEY=VALUE 传入，"
            f"{name} 接受: {names}"
        ) from exc


def _make_judge(args: argparse.Namespace):
    """构造判分器；judge=llm 时把 --llm-* 参数传给 LLMJudge，
    judge=client 时用 --client-llm 指定的类路径构造 ClientLLMJudge。"""
    if args.judge == "llm":
        from membench.judges.llm import LLMJudge

        kwargs = {}
        # --llm-timeout 为 None 时使用 LLMJudge 内建默认值（90s）
        if getattr(args, "llm_timeout", None) is not None:
            kwargs["timeout"] = args.llm_timeout
        return LLMJudge(
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            model=args.llm_model,
            **kwargs,
        )
    if args.judge == "client":
        from membench.judges.client import ClientLLMJudge

        if not args.client_llm:
            raise SystemExit(
                "--judge client 需要 --client-llm 'module:ClassName'（agent 内置 LLM 客户端类路径）"
            )
        return ClientLLMJudge(args.client_llm)
    return args.judge  # "heuristic"，由 Runner 经 get_judge 解析


def _make_evidence_matcher(args: argparse.Namespace):
    """构造检索命中口径匹配器（--evidence-match；llm 复用 --llm-* 配置）。"""
    from membench.metrics.evidence import build_evidence_matcher

    try:
        return build_evidence_matcher(
            args.evidence_match,
            threshold=args.evidence_match_threshold,
            llm_base_url=getattr(args, "llm_base_url", None),
            llm_api_key=getattr(args, "llm_api_key", None),
            llm_model=getattr(args, "llm_model", None),
            llm_timeout=getattr(args, "llm_timeout", None) or 90.0,
        )
    except ValueError as exc:
        raise SystemExit(f"[membench] 参数错误: {exc}")


def _discard_empty_experiment(path: str | None) -> None:
    """删掉存在的空目录（异常吞掉，失败收口用）。

    只删**空**目录：任何已有产物的一律原样保留——那是定位问题要看的现场。
    """
    if not path:
        return
    try:
        if os.path.isdir(path) and not os.listdir(path):
            os.rmdir(path)
    except OSError:
        pass


@contextmanager
def _no_leftover_experiment(base_dir: str | None) -> Iterator[None]:
    """失败收口：本命令在 ``base_dir`` 下新建实验目录后失败时，把仍为空的目录删掉。

    实验目录名（``eval_*`` / ``compare_*`` / ``matrix_*``）在命令内部生成，CLI 事前
    拿不到名字，故进入时先记下 ``base_dir`` 的既有条目，出错时只清理**本次新建且仍为
    空**的目录（含本次新建的 ``base_dir`` 本身）——Runner 层"失败不留半成品"的约定
    不会被 CLI 的提前建目录破坏；有任何产物的目录一律保留。
    """
    root = base_dir or output_root()
    existed = os.path.isdir(root)
    before = set(os.listdir(root)) if existed else set()
    try:
        yield
    except BaseException:
        try:
            after = set(os.listdir(root))
        except OSError:
            after = set()
        for name in sorted(after - before):
            _discard_empty_experiment(os.path.join(root, name))
        if not existed:
            _discard_empty_experiment(root)
        raise


def _out_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """``--task-dir`` / ``--report-out`` 二选一映射成下游的输出容器 kwarg。

    ``--task-dir DIR`` 表示"就地写进这个已存在的任务目录"（下游按 ``task_dir`` 语义
    处理：容器就是 DIR，不再新建实验目录）；否则给 ``report_out``（实验目录的父目录）。
    默认路径（两个都没传）只把 ``report_out=None`` 传下去，与历史行为一致。
    """
    if args.task_dir is not None:
        return {"task_dir": args.task_dir}
    return {"report_out": args.report_out}


def _snapshot_kwarg(args: argparse.Namespace) -> dict[str, Any]:
    """``--no-snapshot`` 才把 ``snapshot=False`` 传下去（默认导出快照，不覆盖下游默认值）。"""
    return {"snapshot": False} if args.no_snapshot else {}


def _cmd_run(args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit < 1:
        raise SystemExit(f"--limit 必须 >= 1，当前为 {args.limit}")
    if args.shard_index < 0:
        raise SystemExit(f"--shard-index 必须 >= 0，当前为 {args.shard_index}")
    if args.shards != "max" and args.shard_index >= args.shards:
        raise SystemExit(
            f"--shard-index 必须满足 0 <= I < --shards，当前为 {args.shard_index}/{args.shards}"
        )
    if args.task_dir is not None and not os.path.isdir(args.task_dir):
        raise SystemExit(
            f"任务目录不存在: {args.task_dir}；"
            "请先运行 python -m membench.cli task init --name <NAME> 建任务目录"
        )
    if args.task_dir is not None:
        # 任务目录开跑前就已确定：先登记，参数 / 适配器错误等早期失败也能落进它的
        # runtime（否则一条失败命令在任务目录里查不到任何痕迹）
        runtime.set_dir(os.path.abspath(args.task_dir))
    adapter = _make_adapter(args.adapter, args.adapter_arg)
    judge = _make_judge(args)
    evidence_matcher = _make_evidence_matcher(args)
    dataset_path, dataset_source = resolve_dataset(args.dataset)
    print(f"[membench] 测试集: {dataset_path} (来源: {dataset_source})")
    adapter_name = str(getattr(adapter, "name", adapter.__class__.__name__))
    with (nullcontext() if args.task_dir is not None else _no_leftover_experiment(args.report_out)):
        if args.task_dir is not None:
            # --task-dir：目录已由 task init 建好（存在性上面校验过），本次不新建实验
            # 目录，也就没有"新建目录失败收口"这回事；同片重复跑与分片口径冲突由
            # Runner 按分片账本拒绝。
            experiment_dir = os.path.abspath(args.task_dir)
            print(f"[membench] 任务目录: {experiment_dir}")
        else:
            # 分片 / limit 进实验目录名：同秒并行的两个分片可以共用一个 --report-out。
            # 建目录与后续预检都包在收口里：校验失败不留空 eval_*（含本次新建的根目录）。
            experiment_dir = layout.experiment_path(
                args.report_out,
                layout.eval_subdir(
                    adapter_name,
                    args.platform,
                    layout.ts_now(),
                    layout.experiment_tags(args.shards, args.shard_index, args.limit),
                ),
            )
            print(f"[membench] 实验目录: {experiment_dir}")
        try:
            runner = Runner(
                adapter,
                dataset_path,
                judge=judge,
                out_dir=experiment_dir,
                platform=args.platform,
                commit=args.commit,
                dataset_source=dataset_source,
                jobs=args.jobs,
                limit=args.limit,
                shards=args.shards,
                shard_index=args.shard_index,
                shard_by=args.shard_by,
                adapter_meta=args.adapter_meta,
                driver_llm=args.driver_llm,
                adapter_version=args.adapter_version,
                evidence_matcher=evidence_matcher,
                shared=args.task_dir is not None,
                snapshot=not args.no_snapshot,
                trace_state=args.trace_state,
                abstention_recheck=args.abstention_recheck,
                replay=args.replay,
                replay_mode=args.replay_mode,
                replay_verify_source=args.replay_verify_source,
            )
            dataset_dir = runner.run()
        except ValueError as exc:
            # 参数错误（数据集 schema 校验失败、回放 / 快照参数不合法等）→ 友好退出
            raise SystemExit(f"[membench] 参数错误: {exc}")
    if dataset_dir is None:
        # 分片无事可做（索引越界 / 空数据集）：runner 已打印说明，干净退出且不留空目录
        if args.task_dir is None:
            _discard_empty_experiment(experiment_dir)
        return 0
    print(f"[membench] 完成: {dataset_dir}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """compare：--task-dir 就地对比该任务目录的 A/ 与 B/，否则新建 compare_* 目录并复制两组。"""
    if args.task_dir is not None:
        if not os.path.isdir(args.task_dir):
            raise SystemExit(f"任务目录不存在: {args.task_dir}")
        # 就地：两组由任务目录给定（--baseline/--candidate 显式给出时以显式为准）
        baseline = args.baseline or layout.group_dir(
            args.task_dir, layout.ROLE_BASELINE
        )
        candidate = args.candidate or layout.group_dir(
            args.task_dir, layout.ROLE_CANDIDATE
        )
        for group_path in (baseline, candidate):
            if not os.path.isdir(group_path):
                raise SystemExit(
                    f"任务目录里的组目录不存在: {group_path}；"
                    "--task-dir 就地对比要求两组（见 groups.json）都有产物，"
                    "或显式给 --baseline / --candidate"
                )
        # 输出容器 = 该任务目录（就地写 reports/，不另建 compare_* 目录、不复制两组）
        out_kwargs: dict[str, Any] = {"task_dir": args.task_dir}
        guard: Any = nullcontext()
    else:
        if not args.baseline or not args.candidate:
            raise SystemExit(
                "compare 需要 --baseline DIR --candidate DIR（两组产物目录），"
                "或用 --task-dir DIR 就地对比该任务目录的 A/ 与 B/"
            )
        baseline, candidate = args.baseline, args.candidate
        out_kwargs = {"out_dir": args.report_out}
        guard = _no_leftover_experiment(args.report_out)
    if args.tag:
        try:
            layout.ensure_dir_name(args.tag, what="对比标签（--tag）")
        except ValueError as exc:
            raise SystemExit(str(exc))
    with guard:
        result = compare_runs(
            baseline,
            candidate,
            acc_threshold=args.acc_threshold,
            stale_threshold=args.stale_threshold,
            latency_warn=args.latency_warn,
            require_comparable=args.require_comparable,
            name=args.tag,
            group_labels=(
                [
                    getattr(args, "baseline_name", None) or "基线",
                    getattr(args, "candidate_name", None) or "候选",
                ]
                if (
                    getattr(args, "baseline_name", None)
                    or getattr(args, "candidate_name", None)
                )
                else None
            ),
            **out_kwargs,
        )
    # compare_runs 已打印 verdict / 逐门禁 / 不可比原因 / 报告路径；此处补实验目录便于脚本消费
    print(f"[membench] verdict: {result['verdict']}")
    print(f"[membench] compare_dir: {result['compare_dir']}")
    # FAIL 时以非零码退出，便于 CI 门禁集成
    return 1 if result["verdict"] == "FAIL" else 0


def _cmd_fuse(args: argparse.Namespace) -> int:
    """fuse：融合多个分片产物（跨目录合并 / 就地补齐）。

    退出码只看 stale 门禁的 ``FAIL``（办公域槽位一票否决）；``INFO``（非办公域
    数据集 stale 超阈值，仅诊断）、``PASS``、``SKIP`` 都是 0。
    """
    from membench.fuse import fuse_runs

    if args.task_dir is not None:
        if args.runs or args.out:
            raise SystemExit(
                "--task-dir 与 --runs/--out 互斥：就地融合只需 --task-dir DIR，"
                "跨目录融合请给 --runs <分片目录>,... --out <完整实验目录>"
            )
        run_dirs: list[str] = []
        out_dir = args.task_dir
    else:
        if not args.runs or not args.out:
            raise SystemExit(
                "fuse 需要 --runs <分片产物目录>,... --out <完整实验目录>（跨目录融合），"
                "或 --task-dir <任务目录>（各片本就写在同一棵树里，就地补齐）"
            )
        run_dirs = [d.strip() for d in args.runs.split(",") if d.strip()]
        out_dir = args.out
    try:
        result = fuse_runs(run_dirs, out_dir)
    except ValueError as exc:
        raise SystemExit(f"[membench] {exc}")
    gate = result["gate"]
    print(f"[membench] 融合完成: {result['n_questions']} 题 -> {result['out_dir']}")
    print(f"[membench] 数据集产物: {result['dataset_dir']}")
    print(f"[membench] stale 门禁: {gate['status']} — {gate['detail']}")
    return 1 if gate["status"] == "FAIL" else 0


def _cmd_matrix(args: argparse.Namespace) -> int:
    labeled_runs: list[tuple[str, str]] = []
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"--runs 项缺少 '=' 分隔符: {spec}（形如 label=<产物目录>）")
        label, _, run_dir = spec.partition("=")
        if not label or not run_dir:
            raise SystemExit(f"--runs 项格式非法: {spec}（形如 label=<产物目录>）")
        labeled_runs.append((label, run_dir))
    # 可前置的入参检查要在**建目录之前**做完：失败不该在产物根留空 matrix_<ts>/
    if len(labeled_runs) < 2:
        raise SystemExit(
            f"[membench] matrix 至少需要 2 个 labeled run（当前 {len(labeled_runs)} 个）："
            "形如 --runs 'wx@a1b2c3=<产物目录>' 'fs@a1b2c3=<产物目录>'"
        )

    # --out 优先；否则按 --report-out（缺省 output_root()）建 matrix_<ts>/
    out_dir = args.out or layout.experiment_path(args.report_out, layout.matrix_subdir())
    try:
        result = compare_matrix(labeled_runs, out_dir)
    except ValueError as exc:
        # 入参 / 产物问题（label 目录冲突、源产物不可解析等）→ 友好报错，不留空实验目录
        _discard_empty_experiment(out_dir)
        raise SystemExit(f"[membench] {exc}")
    except BaseException:
        _discard_empty_experiment(out_dir)
        raise
    worst = "PASS"
    for pair in result["pairwise"]:
        if pair["verdict"] == "FAIL":
            worst = "FAIL"
        elif pair["verdict"] == "WARN" and worst != "FAIL":
            worst = "WARN"
    print(f"[membench] matrix verdict: {worst}")
    print(f"[membench] 实验目录: {result['out_dir']}")
    print(f"[membench] report: {result['report_path']}")
    # 任一两两组合判 FAIL 时以非零码退出（与 compare 同口径，可直接接 CI）
    return 1 if worst == "FAIL" else 0


def _cmd_series(args: argparse.Namespace) -> int:
    """series：多线对比；任一线的门禁 FAIL → 退出码 1（与 compare / matrix 同口径）。"""
    if len(args.line) < 2:
        raise SystemExit(
            f"[membench] series 至少需要 2 条 --line（当前 {len(args.line)} 条）："
            "--line <commit-or-name>=<产物目录> --line ..."
        )
    if args.fusion_cap < 1:
        raise SystemExit(f"--fusion-cap 必须 >= 1，当前为 {args.fusion_cap}")
    # 只有 base_url 与 model 都给才尝试 LLM；未配置 / --no-llm-naming 时直接代码抽取
    naming = None
    if not args.no_llm_naming:
        naming = LLMNaming(
            base_url=args.llm_base_url or "",
            api_key=args.llm_api_key or "",
            model=args.llm_model or "",
            timeout=args.llm_timeout or 90.0,
        )
    with _no_leftover_experiment(args.report_out):
        try:
            result = run_series(
                args.line,
                repo=args.repo,
                dataset=args.dataset,
                report_out=args.report_out,
                require_comparable=args.require_comparable,
                fusion=args.fusion,
                fusion_cap=args.fusion_cap,
                llm=naming,
            )
        except ValueError as exc:
            raise SystemExit(f"[membench] {exc}")
    print(f"[membench] series_dir: {result['series_dir']}")
    return 1 if result["verdict"] == "FAIL" else 0


# ---------------------------------------------------------------------------
# task：任务目录骨架与进度视图
# ---------------------------------------------------------------------------

def _task_group_roots(task_dir: str) -> list[tuple[str, str]]:
    """任务目录下的组：``[(组标签, 组根目录)]``。

    双组任务目录（``task init --groups 2``）取 ``groups.json`` 里实际存在的组目录；都不存在
    时把任务目录本身当单组实验根（``runs/`` 就在它下面）。骨架只建了一半（例如只跑
    了 ``A/``）时只报存在的那一个——status 是进度视图，不该因为另一半还没建就拒绝。
    """
    groups = layout.read_groups_manifest(task_dir) or []
    groups = [
        (str(item["dir"]), os.path.join(task_dir, str(item["dir"])))
        for item in groups
        if os.path.isdir(os.path.join(task_dir, str(item["dir"])))
    ]
    return groups or [("", task_dir)]


def _subdirs(path: str) -> list[str]:
    """目录下的子目录（按路径排序；目录不存在时为空）。"""
    if not os.path.isdir(path):
        return []
    return sorted(
        os.path.join(path, name)
        for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    )


def _dataset_dirs_of(group_root: str) -> list[str]:
    """组根 ``runs/`` 下的数据集目录。"""
    return _subdirs(layout.runs_root(group_root))


def _dialogue_dirs_of(dataset_dir: str) -> list[str]:
    """数据集目录下的对话目录（数据集级 ``report/`` 不算）。"""
    return [path for path in _subdirs(dataset_dir) if os.path.basename(path) != layout.DIR_REPORT]


def _read_ledger_for_status(dataset_dir: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """读分片账本：``(账本, 报错文案)``。

    账本损坏时只报告不中断——status 是诊断视图，一份坏账本不该让它连别处的进度都看不了。
    """
    try:
        return layout.read_shard_ledger(dataset_dir), None
    except SystemExit as exc:
        return None, str(exc)


def _snapshot_usage(dataset_dir: str) -> tuple[int, int]:
    """数据集目录下的快照用量：``(快照目录数, 总字节数)``（扫 ``<对话>/memory_snapshot/``）。"""
    count = 0
    total = 0
    for dialogue_dir in _dialogue_dirs_of(dataset_dir):
        snapshot_dir = os.path.join(dialogue_dir, layout.DIR_SNAPSHOT)
        if not os.path.isdir(snapshot_dir):
            continue
        count += 1
        for root, _dirs, files in os.walk(snapshot_dir):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    continue
    return count, total


def _human_bytes(size: int) -> str:
    """字节数的人类可读形式（进度视图用）。"""
    value = float(size)
    for unit in ("B", "KB", "MB"):
        if value < 1024:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _refresh_snapshot_index(pool_root: str, index: dict[str, Any]) -> None:
    """索引缺失 / 过期时按条目与桶重算写回（best effort：写不进去不影响展示）。

    ``index.json`` 是派生视图，status 展示不信任它；但顺手刷新一次，人直接看文件时
    也能看到与展示一致的内容。
    """
    try:
        current = layout.read_snapshot_index(pool_root)
    except SystemExit:
        current = None
    if current == index:
        return
    try:
        layout.write_snapshot_index(pool_root, index)
    except OSError:
        pass


def _snapshot_pool_lines(target: str) -> list[str]:
    """回放任务快照池的进度行：桶数 / 体积 / 各组来源与复用计数 / 两组一致性。

    新产物从条目树 + 各桶清单**重算**（:func:`membench.layout.build_snapshot_index`），
    不信任可能过期的 ``index.json``；缺索引 / 索引过期时顺手刷新一次（best effort）。
    旧产物（只有 ``index.json``、没有条目树）以索引为唯一记录，原样展示、不重算覆盖。
    普通任务（任务根没有 ``snapshots/``）返回空列表——它们的快照用量仍按数据集目录
    统计（见 :func:`_snapshot_usage`）。坏条目 / 坏桶 / 坏索引只报告不中断：status 是
    诊断视图，一份坏数据不该让它连分片进度都看不了。
    """
    task_dir = layout.snapshot_task_root(target)
    if not layout.has_snapshot_pool(task_dir):
        return []
    pool_root = os.path.join(task_dir, layout.DIR_SNAPSHOTS)
    try:
        if layout.has_snapshot_entries(pool_root):
            index = layout.build_snapshot_index(pool_root)
            _refresh_snapshot_index(pool_root, index)
        else:
            # 旧产物：没有条目文件，索引是唯一记录（重算会得到一个空 entries 的索引）
            index = layout.read_snapshot_index(pool_root) or layout.build_snapshot_index(
                pool_root
            )
    except SystemExit as exc:
        return [f"快照池: ⚠️ {exc}"]
    buckets = index["buckets"]
    entries = index["entries"]
    groups = index["groups"]
    total_bytes = sum(
        int(item.get("bytes") or 0)
        for item in buckets.values()
        if isinstance(item, dict)
    )
    lines = [f"快照池: {pool_root}（{len(buckets)} 个桶 / {_human_bytes(total_bytes)}）"]
    for group in sorted(str(name) for name in groups):
        info = groups.get(group) if isinstance(groups.get(group), dict) else {}
        label = f"组 {group}" if group else "单组"
        source = str(info.get("source") or "n/a")
        manifest_sha = str(info.get("source_manifest_sha256") or "")
        lines.append(
            f"  {label}: {int(info.get('n_dialogues') or 0)} 对话"
            f"（复用 {int(info.get('n_reused') or 0)} / 新写 {int(info.get('n_new') or 0)}），"
            f"源 {source}"
            + (f"（源清单 sha256 {manifest_sha[:12]}）" if manifest_sha else "")
        )
    if len(groups) >= 2:
        lines.extend(_group_consistency_lines(entries))
    return lines


def _group_consistency_lines(entries: dict[str, Any]) -> list[str]:
    """两组逐对话比对快照桶：全部指向同一批桶就是"回放保真"，否则列出差异规模。

    组名由用户在 ``groups.json`` 里给（可以是中文），这里不假设固定词表：取
    快照条目里实际出现的组键，两两比对（通常正好两组）。
    """
    by_dataset: dict[str, dict[str, dict[str, str]]] = {}
    for key, digest in entries.items():
        parsed = layout.parse_snapshot_entry_key(key)
        if parsed is None:
            continue
        group, dataset, dialogue_id = parsed
        by_dataset.setdefault(dataset, {}).setdefault(group, {})[dialogue_id] = str(digest)
    lines: list[str] = []
    for dataset in sorted(by_dataset):
        per_group = by_dataset[dataset]
        keys = sorted(per_group)
        if len(keys) < 2:
            continue
        group_a = per_group.get(keys[0]) or {}
        group_b = per_group.get(keys[1]) or {}
        if not group_a or not group_b:
            continue
        all_ids = set(group_a) | set(group_b)
        same = [did for did in all_ids if group_a.get(did) == group_b.get(did)]
        diff = [did for did in all_ids if did in group_a and did in group_b and group_a[did] != group_b[did]]
        only_a = [did for did in group_a if did not in group_b]
        only_b = [did for did in group_b if did not in group_a]
        if len(same) == len(all_ids):
            shared = len({group_a[did] for did in all_ids})
            lines.append(
                f"  数据集 {dataset}: A 组与 B 组快照 {len(same)}/{len(all_ids)} 逐对话一致"
                f"（共用 {shared} 个桶）"
            )
        else:
            lines.append(
                f"  数据集 {dataset}: A 组与 B 组快照有差异——一致 {len(same)}/{len(all_ids)}，"
                f"内容不同 {len(diff)}、仅 A {len(only_a)}、仅 B {len(only_b)}"
                "（差异即回放不保真的证据）"
            )
    return lines


def _format_duration(ms: Any) -> str:
    """耗时的人类可读形式（毫秒 → 秒；非数字返回 ``-``）。"""
    if not isinstance(ms, (int, float)):
        return "-"
    return f"{ms / 1000:.1f}s"


def _print_runtime(exp_root: str, label: str = "") -> None:
    """打印一个实验 / 组目录的运行时概况（只读；没有 ``runtime/`` 时明确说明）。"""
    title = f"运行时（{label}）" if label else "运行时"
    summary = runtime.summarize(exp_root, limit=5)
    if not summary["exists"]:
        print(f"[membench] {title}: 无运行时数据")
        return
    print(
        f"[membench] {title}: {summary['dir']}"
        f"（最近 {len(summary['recent'])} 条命令，非零退出 {summary['n_failed']} 条）"
    )
    for record in summary["recent"]:
        print(
            f"[membench]   {record.get('started_at') or '-'} "
            f"{record.get('command') or '-'} "
            f"exit={record.get('exit_code')} "
            f"{_format_duration(record.get('duration_ms'))}"
        )
    if summary["last_log"]:
        print(f"[membench]   最近日志: {summary['last_log']}")


def _cmd_task_init(args: argparse.Namespace) -> int:
    """task init：建任务目录骨架，stdout 只打印它的绝对路径（供 ``TASK=$(...)`` 接走）。"""
    try:
        layout.ensure_dir_name(args.name, what="任务名（--name）")
    except ValueError as exc:
        raise SystemExit(str(exc))
    task_dir = layout.init_task_dir(
        args.report_out,
        args.name,
        args.platform,
        args.groups,
        group_labels=(
            [args.baseline_name, args.candidate_name] if args.groups == 2 else None
        ),
        group_dirs=(
            [args.baseline_dir or "baseline", args.candidate_dir or "candidate"]
            if args.groups == 2
            else None
        ),
        replay=args.replay,
    )
    # 任务目录从这一刻起就是本次命令的产物根：task init 自己也记一条命令与日志
    runtime.set_dir(task_dir)
    groups = layout.read_groups_manifest(task_dir) or []
    shape = (
        " ".join(f"{item['dir']}/" for item in groups)
        if args.groups == 2
        else f"{layout.DIR_RUNS}/"
    )
    if args.replay:
        shape += f" {layout.DIR_SNAPSHOTS}/"
    first_group = (
        layout.group_dir(task_dir, layout.ROLE_BASELINE) if args.groups == 2 else task_dir
    )
    # 人看的说明全部走 stderr：stdout 只有一行路径，脚本 $() 才能直接接走
    print(f"[membench] 任务目录: {task_dir}", file=sys.stderr)
    print(f"[membench] 骨架: groups={args.groups}（{shape} + {layout.DIR_REPORTS}/）", file=sys.stderr)
    if args.replay:
        print(
            f"[membench] 下一步: 回放同一个源到该任务 —— "
            f"python -m membench.cli run --adapter <适配器> --dataset <数据集> "
            f"--task-dir {first_group} --replay <源：实验目录 / 数据集目录 / 回放任务>"
            "（两组分别 --task-dir <任务>/A、<任务>/B 回放同一源时，状态相同的对话共用"
            "同一个桶；同一组可再加 --shards <N> --shard-index <I> --shard-by dialogue "
            "让多个进程并行回放，各进程只写自己对话的条目）；进度用 "
            f"python -m membench.cli task status {task_dir}",
            file=sys.stderr,
        )
    else:
        print(
            f"[membench] 下一步: 各分片写进同一任务目录 —— "
            f"python -m membench.cli run --adapter <适配器> --dataset <数据集> "
            f"--task-dir {first_group} --shards <N> --shard-index <I> --shard-by dialogue"
            f"（I 取 0..N-1）；进度用 python -m membench.cli task status {task_dir}",
            file=sys.stderr,
        )
    print(task_dir)
    return 0


def _cmd_task_status(args: argparse.Namespace) -> int:
    """task status：任务目录（或单组实验目录）的进度视图，只读、不写任何产物。"""
    target = os.path.abspath(args.dir)
    if not os.path.isdir(target):
        raise SystemExit(f"任务目录不存在: {args.dir}")
    groups = _task_group_roots(target)
    group_labels_list = [label for label, _ in groups if label]
    if group_labels_list:
        skeleton = f"groups={len(group_labels_list)}（{' '.join(f'{label}/' for label in group_labels_list)} + {layout.DIR_REPORTS}/）"
    elif os.path.isdir(layout.runs_root(target)):
        skeleton = f"groups=1（{layout.DIR_RUNS}/ + {layout.DIR_REPORTS}/）"
    else:
        raise SystemExit(
            f"目录里没有任务骨架（缺 runs/ 或 groups.json 里的组目录）: {target}；"
            "请先用 python -m membench.cli task init --name <NAME> 建任务目录"
        )
    print(f"[membench] 任务目录: {target}")
    print(f"[membench] 骨架: {skeleton}")
    is_replay_task = layout.has_snapshot_pool(layout.snapshot_task_root(target))
    for line in _snapshot_pool_lines(target):
        print(f"[membench] {line}")
    if group_labels_list:
        # 任务根的 runtime 记的是 task init / evaluate / fuse / compare 等任务级命令；
        # 各组的 runtime 记写进该组的 run（分片）命令
        _print_runtime(target, "任务")

    next_steps: list[str] = []
    n_datasets = 0
    groups_all_ready = True
    # 组目录名是 ASCII，显示名在 groups.json 里：进度视图两个都给（"组 基线（baseline）"）
    manifest_labels = {
        str(item["dir"]): str(item.get("label") or item["dir"])
        for item in (layout.read_groups_manifest(target) or [])
    }
    for label, group_root in groups:
        display = manifest_labels.get(label)
        where = (
            f"组 {display}（{label}）"
            if label and display and display != label
            else (f"组 {label}" if label else "单组实验目录")
        )
        print(f"[membench] {where}: {group_root}")
        _print_runtime(group_root, f"组 {label}" if label else "")
        datasets = _dataset_dirs_of(group_root)
        if not datasets:
            print("[membench]   数据集: （尚无产物）")
            groups_all_ready = False
            next_steps.append(
                f"{where} 尚无产物：先跑分片，例如 python -m membench.cli run "
                f"--adapter <适配器> --dataset <数据集> --task-dir {group_root} "
                f"--shards 2 --shard-index 0 --shard-by dialogue"
            )
            continue
        group_ready = True
        for dataset_dir in datasets:
            n_datasets += 1
            name = os.path.basename(dataset_dir)
            metrics_path = os.path.join(dataset_dir, layout.DIR_REPORT, layout.FILE_METRICS)
            ready = os.path.isfile(metrics_path)
            group_ready = group_ready and ready
            ledger, error = _read_ledger_for_status(dataset_dir)
            if error is not None:
                print(f"[membench]   数据集 {name}: ⚠️ {error}")
            elif ledger is None:
                print(
                    f"[membench]   数据集 {name}: 未分片（无分片账本 {layout.FILE_SHARDS}）"
                )
            else:
                shard_by = str(ledger.get("shard_by") or "question")
                shards_n = int(ledger.get("shards") or 1)
                entries = {
                    key: entry
                    for key, entry in (ledger.get("entries") or {}).items()
                    if isinstance(entry, dict)
                }
                done = [key for key, entry in entries.items() if entry.get("status") == "done"]
                running = [key for key, entry in entries.items() if entry.get("status") == "running"]
                print(
                    f"[membench]   数据集 {name}: 分片 {len(done)}/{shards_n} 完成"
                    f"（shard_by={shard_by}，limit={ledger.get('limit')}）"
                    + (f"，{len(running)} 片仍在跑（running）" if running else "")
                )
                for key in sorted(
                    entries,
                    key=lambda k: (layout.shard_key_index(k) is None, layout.shard_key_index(k) or 0),
                ):
                    entry = entries[key]
                    n_questions = entry.get("n_questions")
                    print(
                        f"[membench]     {key}: {entry.get('status')} "
                        f"{len(entry.get('dialogues') or [])} 对话 / "
                        f"{n_questions if n_questions is not None else 'n/a'} 题 "
                        f"run_id={entry.get('run_id')} "
                        f"finished_at={entry.get('finished_at') or '-'}"
                    )
                missing = [
                    layout.shard_key(index)
                    for index in range(shards_n)
                    if layout.shard_key(index) not in entries
                ]
                if running:
                    next_steps.append(
                        f"分片 {'、'.join(sorted(running))} 仍在跑（running）：等它结束后再融合"
                    )
                if missing:
                    next_steps.append(
                        f"还有 {len(missing)} 片未完成（{'、'.join(missing)}）：补跑 "
                        f"python -m membench.cli run --task-dir {group_root} "
                        f"--shards {shards_n} --shard-index <I> --shard-by {shard_by}"
                    )
                elif not running and not ready:
                    next_steps.append(
                        f"数据集 {name} 的分片已跑齐：补齐数据集级与任务级产物 "
                        f"python -m membench.cli fuse --task-dir {group_root}"
                    )
            if is_replay_task:
                snapshot_text = "快照: 任务级池（见上）"
            else:
                snapshots, size = _snapshot_usage(dataset_dir)
                snapshot_text = f"快照 {snapshots} 个 / {_human_bytes(size)}"
            print(
                f"[membench]   数据集级产物: {'已齐备' if ready else '缺失'}（{metrics_path}）；"
                f"对话目录 {len(_dialogue_dirs_of(dataset_dir))} 个；"
                f"{snapshot_text}"
            )
        groups_all_ready = groups_all_ready and group_ready

    if n_datasets == 0:
        next_steps.append("任务目录里还没有数据集产物——按上面的示例跑分片后再 status 看进度")
    if len(groups) == 2 and groups_all_ready:
        next_steps.append(
            f"两组数据集级产物已齐备：python -m membench.cli compare --task-dir {target}"
        )
    print("[membench] 下一步:")
    for step in next_steps or ["产物已齐备，无需操作"]:
        print(f"[membench]   - {step}")
    return 0


def _cmd_user_data_install(args: argparse.Namespace) -> int:
    try:
        stats = install_user_dataset(args.src)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"安装失败，测试集校验未通过: {exc}")
    print(f"[membench] 用户测试集已安装: {stats['path']}")
    print(
        f"[membench] dialogues={stats['n_dialogues']} questions={stats['n_questions']} "
        f"source={stats['source']}"
    )
    return 0


def _cmd_user_data_status(args: argparse.Namespace) -> int:
    status = user_dataset_status()
    if not status["exists"]:
        print("[membench] 未安装用户测试集（run 不传 --dataset 时将回退内置 office 数据集）")
        return 0
    print(f"[membench] 用户测试集: {status['path']}")
    meta = status["metadata"]
    if meta:
        print(f"[membench] installed_at={meta.get('installed_at', 'n/a')} "
              f"source={meta.get('source', 'n/a')} "
              f"dialogues={meta.get('n_dialogues', 'n/a')} "
              f"questions={meta.get('n_questions', 'n/a')}")
    else:
        print("[membench] (metadata.json 缺失或损坏)")
    return 0


def _cmd_user_data_delete(args: argparse.Namespace) -> int:
    existed = delete_user_dataset()
    if existed:
        print("[membench] 用户测试集已删除")
    else:
        print("[membench] 本就没有用户测试集，无需删除")
    return 0


def _cmd_adapter_install(args: argparse.Namespace) -> int:
    """adapter install：落盘 + 校验；不要求当前进程已加载该适配器。"""
    from membench.adapters import install_adapter

    try:
        result = install_adapter(args.src, name=args.name)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"安装失败: {exc}")
    print(f"[membench] 适配器已安装: {result['path']}")
    print(f"[membench] 注册名: {result['name']}  installed_at: {result['installed_at']}")
    print("[membench] 下次运行任意 membench 命令时将自动加载，无需再设环境变量")
    return 0


def _cmd_adapter_list(args: argparse.Namespace) -> int:
    """adapter list：表格列出内置适配器与持久化适配器（标注来源）。"""
    from membench.adapters import ADAPTERS, list_installed_adapters

    installed = list_installed_adapters()
    installed_names = {m.get("name", "") for m in installed}
    builtin_names = sorted(n for n in ADAPTERS if n not in installed_names)

    rows: list[tuple[str, str, str, str, str]] = []
    for n in builtin_names:
        cls = ADAPTERS[n]
        rows.append((n, cls.__name__, "-", "-", "内置"))
    for m in installed:
        rows.append(
            (
                str(m.get("name", "")),
                str(m.get("class_name", "") or "?"),
                str(m.get("installed_at", "")),
                str(m.get("source_path", "")),
                "持久化",
            )
        )
    if not rows:
        print("[membench] 暂无可用适配器")
        return 0
    headers = ("注册名", "类名", "安装时间", "来源路径", "来源")
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(5)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))
    return 0


def _cmd_adapter_remove(args: argparse.Namespace) -> int:
    """adapter remove：删除持久化适配器文件。"""
    from membench.adapters import remove_adapter

    if remove_adapter(args.name):
        print(f"[membench] 持久化适配器 '{args.name}' 已删除")
    else:
        print(f"[membench] 持久化目录中没有名为 '{args.name}' 的适配器")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from membench.evaluate import dataset_specs, evaluate, parse_run_kv

    if args.jobs < 1:
        raise SystemExit(f"--jobs 必须 >= 1，当前为 {args.jobs}")
    if getattr(args, "tag", None):
        try:
            layout.ensure_dir_name(args.tag, what="对比标签（--tag）")
        except ValueError as exc:
            raise SystemExit(str(exc))
    if args.beam_10m and args.no_beam:
        raise SystemExit("--beam-10m 与 --no-beam 互斥（BEAM 槽位状态冲突）")
    if args.quick and args.common:
        raise SystemExit(
            "--common 与 --quick 互斥（公开基准常用集与快速模式的数据集清单不同）"
        )
    preset_used = args.quick or args.beam_10m or args.no_beam or args.no_lme_s or args.common
    if args.dataset and preset_used:
        raise SystemExit(
            "--dataset 与预设开关（--quick / --beam-10m / --no-beam / --no-lme-s / --common）"
            "互斥：显式清单时请直接枚举全部数据集"
        )
    if args.from_runs and preset_used:
        raise SystemExit(
            "--from-runs 与预设开关"
            "（--quick / --beam-10m / --no-beam / --no-lme-s / --common）互斥"
        )
    specs = dataset_specs(args.dataset) if args.dataset else None
    from_runs = parse_run_kv(args.from_runs, "--from-runs")
    if args.judge is not None:
        judge = _make_judge(args)
    elif from_runs and (not specs or {s.key for s in specs} <= set(from_runs)):
        judge = None  # 全复用：报告头以已有 run 的元数据为准
    else:
        judge = "heuristic"
    adapter = _make_adapter(args.adapter, args.adapter_arg) if args.adapter else None
    with (nullcontext() if args.task_dir is not None else _no_leftover_experiment(args.report_out)):
        return evaluate(
            adapter,
            judge,
            platform=args.platform,
            commit=args.commit,
            quick=args.quick,
            beam_10m=args.beam_10m,
            no_beam=args.no_beam,
            no_lme_s=args.no_lme_s,
            common=args.common,
            jobs=args.jobs,
            driver_llm=args.driver_llm,
            adapter_version=args.adapter_version,
            evidence_matcher=_make_evidence_matcher(args),
            specs=specs,
            from_runs=from_runs or None,
            fusion=args.fusion,
            **_out_kwargs(args),
            **_snapshot_kwarg(args),
            trace_state=args.trace_state,
            abstention_recheck=args.abstention_recheck,
        )


def _cmd_evaluate_compare(args: argparse.Namespace) -> int:
    """两组（baseline vs candidate）× 多数据集融合 A/B 对比。"""
    from membench.evaluate import dataset_specs, parse_run_kv
    from membench.evaluate_compare import evaluate_compare

    if args.jobs < 1:
        raise SystemExit(f"--jobs 必须 >= 1，当前为 {args.jobs}")
    if args.beam_10m and args.no_beam:
        raise SystemExit("--beam-10m 与 --no-beam 互斥（BEAM 槽位状态冲突）")
    if args.quick and args.common:
        raise SystemExit(
            "--common 与 --quick 互斥（公开基准常用集与快速模式的数据集清单不同）"
        )
    preset_used = args.quick or args.beam_10m or args.no_beam or args.no_lme_s or args.common
    if args.dataset and preset_used:
        raise SystemExit(
            "--dataset 与预设开关（--quick / --beam-10m / --no-beam / --no-lme-s / --common）"
            "互斥：显式清单时请直接枚举全部数据集"
        )
    if (args.baseline_run or args.candidate_run) and preset_used:
        raise SystemExit(
            "--baseline-run / --candidate-run 与预设开关"
            "（--quick / --beam-10m / --no-beam / --no-lme-s / --common）互斥"
        )
    specs = dataset_specs(args.dataset) if args.dataset else None
    baseline_runs = parse_run_kv(args.baseline_run, "--baseline-run")
    candidate_runs = parse_run_kv(args.candidate_run, "--candidate-run")
    if args.judge is not None:
        judge = _make_judge(args)
    else:
        provided_union = set(baseline_runs) | set(candidate_runs)
        target_keys = {s.key for s in specs} if specs else provided_union
        both_covered = bool(target_keys) and (
            target_keys <= set(baseline_runs) and target_keys <= set(candidate_runs)
        )
        judge = None if (provided_union and both_covered) else "heuristic"
    baseline_adapter = (
        _make_adapter(args.baseline_adapter, args.adapter_arg_a) if args.baseline_adapter else None
    )
    candidate_adapter = (
        _make_adapter(args.candidate_adapter, args.adapter_arg_b) if args.candidate_adapter else None
    )
    with (nullcontext() if args.task_dir is not None else _no_leftover_experiment(args.report_out)):
        result = evaluate_compare(
            baseline_adapter,
            candidate_adapter,
            judge,
            platform=args.platform,
            commit=args.commit,
            quick=args.quick,
            beam_10m=args.beam_10m,
            no_beam=args.no_beam,
            no_lme_s=args.no_lme_s,
            common=args.common,
            tag=args.tag,
            jobs=args.jobs,
            driver_llm_baseline=args.driver_llm_a,
            driver_llm_candidate=args.driver_llm_b,
            adapter_version_baseline=args.adapter_version_a,
            adapter_version_candidate=args.adapter_version_b,
            acc_threshold=args.acc_threshold,
            stale_threshold=args.stale_threshold,
            latency_warn=args.latency_warn,
            evidence_matcher=_make_evidence_matcher(args),
            specs=specs,
            baseline_runs=baseline_runs or None,
            candidate_runs=candidate_runs or None,
            fusion=args.fusion,
            **_out_kwargs(args),
            **_snapshot_kwarg(args),
            trace_state=args.trace_state,
            abstention_recheck=args.abstention_recheck,
            baseline_label=args.baseline_name,
            candidate_label=args.candidate_name,
            baseline_dir=args.baseline_dir,
            candidate_dir=args.candidate_dir,
        )
    return 1 if result["verdict"] == "FAIL" else 0


def _cmd_list_datasets(args: argparse.Namespace) -> int:
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    print("内置数据集（membench/data/*.jsonl）:")
    found = False
    if os.path.isdir(data_dir):
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".jsonl"):
                continue
            found = True
            path = os.path.join(data_dir, fname)
            info = ""
            try:
                from membench.data.schema import load_dataset

                dialogues = load_dataset(path)
                n_q = sum(len(d.questions) for d in dialogues)
                info = f"  ({len(dialogues)} dialogues, {n_q} questions)"
            except Exception:
                pass
            print(f"  {path}{info}")
    if not found:
        print("  (未找到内置数据集)")
    print("\n可用适配器: " + (", ".join(sorted(ADAPTERS)) or "(无)"))
    return 0


def _cmd_dataset_info(args: argparse.Namespace) -> int:
    """dataset info：无 name 概览全部，有 name 详细介绍单个数据集。"""
    from membench.data import info as dataset_info

    if args.name is None:
        print(dataset_info.describe_all())
        return 0
    try:
        print(dataset_info.describe(args.name))
    except KeyError as exc:
        print(f"[membench] {exc.args[0]}", file=sys.stderr)
        return 2
    return 0


def _cmd_dataset_fetch(args: argparse.Namespace) -> int:
    """dataset fetch：按需下载并转换 status=fetch 的数据集。

    下载+转换实现集中在 ``membench.data.fetch``（已有完整产物则跳过，幂等），CLI 只做
    参数校验与退出码映射：内置数据集无需下载（提示路径，退出码 0）；
    未知 key / 暂不可用 -> 2（并给出可手动获取的来源）；下载失败 -> 1。
    """
    from membench.data import fetch as dataset_fetch
    from membench.data import info as dataset_info

    try:
        entry = dataset_info.get_dataset_info(args.name)
    except KeyError as exc:
        print(f"[membench] {exc.args[0]}", file=sys.stderr)
        return 2

    if entry.status == "builtin":
        path = dataset_info.builtin_path(entry.key)
        print(f"[membench] {entry.key} 为内置数据集，无需下载: {path}")
        print(
            f"[membench] 运行示例: python -m membench.cli run "
            f"--adapter <你的适配器> --dataset {path}"
        )
        return 0

    if not dataset_fetch.is_fetchable(entry.key):
        label = {"unavailable": "暂不可用"}.get(entry.status, entry.status)
        print(
            f"[membench] {entry.key} {label}，不支持按需下载；"
            f"可手动获取来源: {entry.source}",
            file=sys.stderr,
        )
        return 2

    try:
        out_path = dataset_fetch.fetch_dataset(entry.key, args.out)
    except KeyError as exc:  # 注册表标了 fetch 但下载后端未登记
        print(f"[membench] {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:  # 下载源未确证或下载失败
        print(f"[membench] {exc}", file=sys.stderr)
        return 1
    print(f"[membench] 数据集已就绪: {out_path}")
    print(f"[membench] 运行示例: python -m membench.cli run --adapter <你的适配器> --dataset {out_path}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """doctor：探测被测服务 reset/ingest/query 三端点是否符合 HttpAdapter 契约。"""
    from membench.doctor import check_target

    result = check_target(args.target, timeout=args.timeout, api_key=args.api_key)
    for check in result["checks"]:
        print(f"[membench] doctor {check['name']}: {check['status']} — {check['detail']}")
    return 0 if result["ok"] else 1


#: 会写实验产物、因而需要记运行时数据的命令（``task status`` 等只读命令不记）。
_RUNTIME_COMMANDS = frozenset(
    {"run", "evaluate", "evaluate-compare", "compare", "fuse", "matrix", "series"}
)


def _writes_runtime(args: argparse.Namespace) -> bool:
    """该命令是否值得记运行时数据（task 只记 init，status 是只读视图）。"""
    if args.command == "task":
        return getattr(args, "task_command", None) == "init"
    return args.command in _RUNTIME_COMMANDS


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口，返回进程退出码。"""
    # 在任何子命令解析前加载外部适配器插件（MEMBENCH_ADAPTER_MODULES）。
    from membench.adapters import load_external_adapters

    load_external_adapters()
    parser = _build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)
    session = runtime.begin(args.command, raw_argv) if _writes_runtime(args) else None
    exit_code = 0
    try:
        try:
            exit_code = int(args.func(args))
        except RuntimeError as exc:
            # 如缺少可选依赖（matplotlib）等运行时错误：输出友好提示而非堆栈
            print(f"[membench] {exc}", file=sys.stderr)
            exit_code = 1
        return exit_code
    except BaseException as exc:
        # 失败也要落一条带退出码的命令记录；SystemExit 的文案 / traceback 由解释器
        # 在 main 返回后打印，先补进日志（终端输出不变）
        exit_code = runtime.exit_code_of(exc)
        if session is not None:
            session.note_exception(exc)
        raise
    finally:
        if session is not None:
            runtime.finish(session, exit_code)


if __name__ == "__main__":
    sys.exit(main())
