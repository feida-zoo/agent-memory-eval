#!/usr/bin/env bash
# eval_two_commits.sh — 两个 commit 的 A/B 记忆评测模板脚本
# 用法: ./eval_two_commits.sh <AGENT_REPO> <SHA_A> <SHA_B> <ADAPTER> [DATASET]
# 使用前按你的项目修改 BUILD_CMD。详细说明见 references/compare_workflows.md
#
# 新流程建议先用同目录的 run_task.py 编排标准步骤（task init → 分片 run → fuse →
# compare --require-comparable → verify_artifacts 自检），本脚本保留为「两个 git
# worktree 各自构建」的模板；交付前自检可直接运行：
#   python scripts/verify_artifacts.py <任务目录>
#
# 产物形态：每次 run 新建 <OUT_ROOT>/eval_<adapter>_<platform>_<ts>/ 自包含实验目录，
# 数据集产物在其 runs/<dataset>/（报告在 report/，逐题明细 results.jsonl），
# 对比实验落 <OUT_ROOT>/compare_<name>_<platform>_<ts>/（A/、B/、reports/comparison.md）。
#
# 可选环境变量：
#   JUDGE=llm|heuristic|client   判分器（默认 heuristic）
#   DRIVER_LLM=<模型标识>         驱动被测系统的 LLM（可比性校验维度，建议显式指定）
#   ADAPTER_VERSION=<版本标识>    adapter 实现版本（可比性校验维度）
#   PYTHON=<解释器>               默认 python3
# 两侧都显式落账 DRIVER_LLM 与 ADAPTER_VERSION 后，compare 才可加 --require-comparable
# 做 strict 门禁（缺维度时 strict 判 FAIL）。
set -euo pipefail

AGENT_REPO=${1:?"agent repo 路径"}
SHA_A=${2:?"基线 commit"}
SHA_B=${3:?"候选 commit"}
ADAPTER=${4:?"adapter 名或模块"}
DATASET=${5:-""}
SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MEMBENCH_DIR="$SKILL_DIR/membench"
OUT_ROOT="$SKILL_DIR/output"        # 实验目录父目录（--report-out）
BUILD_CMD="pip install -e ."        # TODO: 改成你的构建命令
JUDGE="${JUDGE:-heuristic}"         # 正式评估建议 JUDGE=llm（需配置 LLM 密钥环境变量）
PYTHON="${PYTHON:-python3}"

export MEMBENCH_USER_DATA_DIR="$SKILL_DIR/user_dataset"

git -C "$AGENT_REPO" worktree add "/tmp/agent_$SHA_A" "$SHA_A"
git -C "$AGENT_REPO" worktree add "/tmp/agent_$SHA_B" "$SHA_B"
trap 'git -C "$AGENT_REPO" worktree remove --force "/tmp/agent_$SHA_A"; git -C "$AGENT_REPO" worktree remove --force "/tmp/agent_$SHA_B"' EXIT

(cd "/tmp/agent_$SHA_A" && eval "$BUILD_CMD")
(cd "/tmp/agent_$SHA_B" && eval "$BUILD_CMD")

DATASET_ARG=()
if [ -n "$DATASET" ]; then
  DATASET_ARG=(--dataset "$DATASET")
fi
META_ARG=()
if [ -n "${DRIVER_LLM:-}" ]; then
  META_ARG+=(--driver-llm "$DRIVER_LLM")
fi
if [ -n "${ADAPTER_VERSION:-}" ]; then
  META_ARG+=(--adapter-version "$ADAPTER_VERSION")
fi

# 空数组在 bash 3.2（macOS 自带）配 set -u 会报 unbound variable，故用 ${arr[@]+...} 惯用法；
# 实验目录名精确到秒，同一秒内的第二次 run 会被拒绝，故失败后重试（真实数据集下几乎不会触发）。
run_eval() {
  local sha=$1 attempt
  for attempt in 1 2 3; do
    if PYTHONPATH="/tmp/agent_$sha:$MEMBENCH_DIR" "$PYTHON" -m membench.cli run \
      --adapter "$ADAPTER" --commit "$sha" --judge "$JUDGE" \
      "${DATASET_ARG[@]+"${DATASET_ARG[@]}"}" "${META_ARG[@]+"${META_ARG[@]}"}" \
      --report-out "$OUT_ROOT"; then
      return 0
    fi
    echo "[eval_two_commits] run($sha) 失败，2s 后重试（实验目录名精确到秒，同一秒内的第二次 run 会被拒绝）" >&2
    sleep 2
  done
  return 1
}

cd "$MEMBENCH_DIR"
run_eval "$SHA_A"
run_eval "$SHA_B"

# 定位两次 run 的数据集目录（<实验目录>/runs/<dataset>）：扫描 OUT_ROOT 下各实验目录的
# runs/<dataset>/report/metrics.json，按元数据 commit 认领、按 timestamp 取最新一次——
# run_id 只记身份，不再是目录名，不能靠目录名或 mtime 猜。
locate_dataset_dir() {
  "$PYTHON" - "$OUT_ROOT" "$1" <<'PY'
import glob
import json
import os
import sys

root, commit = sys.argv[1], sys.argv[2]
found = []
for metrics_path in glob.glob(os.path.join(root, "eval_*", "runs", "*", "report", "metrics.json")):
    try:
        with open(metrics_path, encoding="utf-8") as handle:
            metrics = json.load(handle)
    except (OSError, ValueError):
        continue
    if str(metrics.get("commit") or "") == commit:
        found.append((str(metrics.get("timestamp") or ""), os.path.dirname(os.path.dirname(metrics_path))))
if not found:
    print(f"未找到 commit={commit} 的产物（--report-out {root}）", file=sys.stderr)
    sys.exit(1)
print(sorted(found)[-1][1])
PY
}

DATASET_A=$(locate_dataset_dir "$SHA_A")
DATASET_B=$(locate_dataset_dir "$SHA_B")
echo "[eval_two_commits] baseline=$DATASET_A candidate=$DATASET_B"
PYTHONPATH="$MEMBENCH_DIR" "$PYTHON" -m membench.cli compare \
  --baseline "$DATASET_A" --candidate "$DATASET_B" \
  --report-out "$OUT_ROOT"
