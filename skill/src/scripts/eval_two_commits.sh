#!/usr/bin/env bash
# eval_two_commits.sh — 两个 commit 的 A/B 记忆评测模板脚本
# 用法: ./eval_two_commits.sh <AGENT_REPO> <SHA_A> <SHA_B> <ADAPTER> [DATASET]
# 使用前按你的项目修改 BUILD_CMD。详细说明见 references/compare_workflows.md
set -euo pipefail

AGENT_REPO=${1:?"agent repo 路径"}
SHA_A=${2:?"基线 commit"}
SHA_B=${3:?"候选 commit"}
ADAPTER=${4:?"adapter 名或模块"}
DATASET=${5:-""}
SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MEMBENCH_DIR="$SKILL_DIR/membench"
BUILD_CMD="pip install -e ."        # TODO: 改成你的构建命令
JUDGE="${JUDGE:-heuristic}"         # 正式评估建议 JUDGE=llm（需配置 LLM 密钥环境变量）

export MEMBENCH_USER_DATA_DIR="$SKILL_DIR/user_dataset"

git -C "$AGENT_REPO" worktree add "/tmp/agent_$SHA_A" "$SHA_A"
git -C "$AGENT_REPO" worktree add "/tmp/agent_$SHA_B" "$SHA_B"
trap 'git -C "$AGENT_REPO" worktree remove --force "/tmp/agent_$SHA_A"; git -C "$AGENT_REPO" worktree remove --force "/tmp/agent_$SHA_B"' EXIT

(cd "/tmp/agent_$SHA_A" && eval "$BUILD_CMD")
(cd "/tmp/agent_$SHA_B" && eval "$BUILD_CMD")

DATASET_ARG=(); [ -n "$DATASET" ] && DATASET_ARG=(--dataset "$DATASET")

mkdir -p "$SKILL_DIR/runs"
cd "$MEMBENCH_DIR"
PYTHONPATH="/tmp/agent_$SHA_A:$MEMBENCH_DIR" python -m membench.cli run \
  --adapter "$ADAPTER" --commit "$SHA_A" --judge "$JUDGE" "${DATASET_ARG[@]}" --out "$SKILL_DIR/runs" --report-out "$SKILL_DIR/output"
PYTHONPATH="/tmp/agent_$SHA_B:$MEMBENCH_DIR" python -m membench.cli run \
  --adapter "$ADAPTER" --commit "$SHA_B" --judge "$JUDGE" "${DATASET_ARG[@]}" --out "$SKILL_DIR/runs" --report-out "$SKILL_DIR/output"

RUN_A=$(ls -dt "$SKILL_DIR"/runs/*"${SHA_A:0:8}"* | head -1)
RUN_B=$(ls -dt "$SKILL_DIR"/runs/* | head -1)
echo "[eval_two_commits] baseline=$RUN_A candidate=$RUN_B"
PYTHONPATH="$MEMBENCH_DIR" python -m membench.cli compare \
  --baseline "$RUN_A" --candidate "$RUN_B" --report-out "$SKILL_DIR/output"
