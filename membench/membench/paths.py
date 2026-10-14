"""产物目录解析：统一的 output 根目录与 runs 目录判定。

skill 环境与开发环境的判定（SPEC 0.2）：

- **skill 环境**：membench 工程根（``membench/``）的上一级目录存在 ``SKILL.md``
  ——即 membench 被打包进 skill 包内运行，产物应落在 skill 包的 ``output/``；
- **开发环境**：不存在 ``SKILL.md``（在仓库里直接跑），产物落工程根。

目录：

- :func:`output_root`：实验目录（``eval_*`` / ``ab_*`` / ``matrix_*``）的父目录。
  优先级 ``MEMBENCH_OUTPUT_DIR`` > ``<skill根>/output`` > ``<工程根>/output``。
- :func:`runs_dir`：run 目录的父目录。优先级
  ``MEMBENCH_OUTPUT_DIR`` > ``<skill根>/output/runs`` > ``<工程根>/runs``
  （开发环境保持历史行为，run 落工程根 ``runs/``）。

这两个函数是全工程判定默认产物的唯一入口，避免各处重复实现 skill/开发环境探测
（接入侧反馈 7：约定下沉为可复用 API）。
"""

from __future__ import annotations

import os

#: 覆盖产物根目录的环境变量。
OUTPUT_ENV = "MEMBENCH_OUTPUT_DIR"

#: skill 包标识文件（存在即视为 skill 环境）。
SKILL_MARKER = "SKILL.md"


def package_dir() -> str:
    """membench 包目录（``.../membench/membench``）。"""
    return os.path.dirname(os.path.abspath(__file__))


def project_root() -> str:
    """membench 工程根（``.../membench``，含 pyproject.toml）。"""
    return os.path.dirname(package_dir())


def skill_root() -> str | None:
    """skill 根目录（工程根的上一级，且含 ``SKILL.md``）；非 skill 环境返回 None。"""
    root = os.path.dirname(project_root())
    if os.path.isfile(os.path.join(root, SKILL_MARKER)):
        return root
    return None


def output_root() -> str:
    """实验目录的父目录。"""
    env = os.environ.get(OUTPUT_ENV)
    if env:
        return env
    root = skill_root()
    if root:
        return os.path.join(root, "output")
    return os.path.join(project_root(), "output")


def runs_dir() -> str:
    """run 目录的父目录（``output_root()/runs``，开发环境为工程根 ``runs/``）。"""
    env = os.environ.get(OUTPUT_ENV)
    if env:
        return os.path.join(env, "runs")
    root = skill_root()
    if root:
        return os.path.join(root, "output", "runs")
    return os.path.join(project_root(), "runs")
