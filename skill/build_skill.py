#!/usr/bin/env python3
"""把本仓库的 skill 源与评测工程打包成可分发的 ``.skill``（zip）。

用法::

    python skill/build_skill.py            # 默认 --repo-root . --out skill/agent-memory-eval.skill
    python skill/build_skill.py --list     # 只列出将被打包的文件（含排除项提示），不写盘

组装规则
--------

打包后顶层目录名固定为 ``agent-memory-eval/``，内容来自三处：

1. ``skill/src/**`` → ``agent-memory-eval/**``（原样，含 ``.gitkeep``）；
2. ``<repo>/membench/`` → ``agent-memory-eval/membench/``（按下方排除策略裁剪）；
3. ``<repo>/docs/`` → ``agent-memory-eval/docs/``（报告 md 与配图 png）。

为什么不用 ``git ls-files``：脚本要能在没有 git 的环境（CI 容器、直接拷目录）跑，
因此改用显式排除规则，并把"哪些不该进包"写在代码里而非依赖仓库状态。

排除策略与理由
--------------

- ``*.egg-info/`` / ``__pycache__/`` / ``*.pyc`` / ``.DS_Store``：构建缓存与系统垃圾，
  与源码无关，且随环境变化（会破坏可复现构建）。
- ``runs/`` / ``output/``：上次评测的运行产物与报告，属于运行期数据，不应随源码分发。
- 包内 ``membench/membench/data/`` 下可能残留的**按需下载**数据集文件
  （``locomo.jsonl``、``longmemeval_*.jsonl``、``gatemem_office.jsonl``、
  ``beam_*.jsonl``）：``dataset fetch`` 现在把原始缓存与产物落到
  ``~/.membench/datasets/``（不再写进包内），但用 ``-o`` 显式指到包内（或旧版本
  残留）时仍可能出现。这些文件体积巨大（``longmemeval_s.jsonl`` 约 268MB），
  仓库 ``.gitignore`` 也明确约定"大数据集不入库"；排除规则把它与 ``runs/`` /
  ``output/`` 等其它构建产物一并挡掉。历史包曾把它们打进去导致包体膨胀到 83MB，
  这是必须修掉的 bug；``--check`` 自检会在打包后复查包内是否出现这些文件名，
  出现即报错退出（防止回归）。内置的 ``office_dataset.jsonl`` /
  ``sample_dataset.jsonl`` 则**保留**。

可复现构建
----------

条目按打包路径排序、zip 内统一使用正斜杠与 UTF-8 文件名、时间戳固定为 1980-01-01，
从而同一份源码每次打出的包字节一致。
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import zipfile
from typing import Iterable, Iterator, List, Tuple

#: 包内顶层目录名（固定）。
TOP_DIR = "agent-memory-eval"

#: 固定时间戳，保证可复现构建。
FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)

#: label → 顶层来源目录（相对 repo-root）。
SOURCE_DIRS = ("skill/src", "membench", "docs")

#: 需要整目录排除的目录名（membench 来源）。
EXCLUDE_DIR_NAMES = frozenset({"__pycache__", ".pytest_cache", ".venv", "venv", "node_modules", "runs", "output"})
#: 目录名以这些后缀结尾时整目录排除（覆盖 ``*.egg-info``）。
EXCLUDE_DIR_SUFFIXES = (".egg-info",)
#: 文件名匹配这些 glob 时排除。
EXCLUDE_FILE_GLOBS = ("*.pyc", ".DS_Store")
#: ``membench/membench/data/`` 下按需下载的数据集文件名（glob），必须排除。
EXCLUDE_DATA_GLOBS = (
    "locomo.jsonl",
    "longmemeval_*.jsonl",
    "gatemem_office.jsonl",
    "beam_*.jsonl",
)

#: 包内自检：这些路径前缀下不得出现被排除的数据集文件名。
DATA_DIR_REL = "membench/membench/data"

#: 包内自检：禁止出现的目录名（构建/运行环境产物；不进包）。
#: 例如 .venv/ 会让 skill 包从 ~MB 涨到 ~50 MB（解释器 + site-packages）。
EXCLUDE_DIR_NAMES_FORBIDDEN = ("__pycache__", ".pytest_cache", ".venv", "venv", "node_modules", "runs", "output")


def _posix(*parts: str) -> str:
    """用正斜杠拼接 zip 内路径。"""
    return "/".join(p.strip("/") for p in parts if p)


def _is_excluded_dir(name: str) -> bool:
    return name in EXCLUDE_DIR_NAMES or name.endswith(EXCLUDE_DIR_SUFFIXES)


def _is_excluded_file(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in EXCLUDE_FILE_GLOBS)


def _is_excluded_data(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in EXCLUDE_DATA_GLOBS)


def _walk_files(root: str) -> Iterator[str]:
    """遍历目录下的文件绝对路径（不含目录项）。"""
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            yield os.path.join(dirpath, fname)


def _iter_membench(root: str) -> Iterator[Tuple[str, str, str]]:
    """产出 membench 来源的文件 (src_abs, arc_rel, exclude_reason)。

    ``exclude_reason`` 为空表示入选；非空表示被排除（供 ``--list`` 展示）。
    """
    for dirpath, dirnames, filenames in os.walk(root):
        # 原地裁剪：被排除的目录不再深入。
        kept_dirs = []
        for d in dirnames:
            rel_dir = os.path.relpath(os.path.join(dirpath, d), root)
            if _is_excluded_dir(d):
                kept_dirs.append((d, rel_dir, True))
            else:
                kept_dirs.append((d, rel_dir, False))
        dirnames[:] = [d for d, _rel, excl in kept_dirs if not excl]

        rel_dir = os.path.relpath(dirpath, root)
        for d, rel_d, excl in kept_dirs:
            if excl:
                yield (
                    os.path.join(dirpath, d),
                    os.path.join("membench", rel_d).replace(os.sep, "/"),
                    f"排除目录（{d}）",
                )

        for fname in filenames:
            src = os.path.join(dirpath, fname)
            rel = os.path.relpath(src, root).replace(os.sep, "/")
            arc = _posix("membench", rel)
            if _is_excluded_file(fname):
                yield (src, arc, f"排除文件（{fname}）")
                continue
            in_data = _posix(rel).startswith(DATA_DIR_REL + "/")
            if in_data and _is_excluded_data(fname):
                yield (src, arc, f"排除按需下载数据集（{fname}）")
                continue
            yield (src, arc, "")


def plan(repo_root: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """计算打包计划，返回 (included, excluded)。

    included: ``(src_abs, arcname)`` 列表，已按 arcname 排序。
    excluded: ``(arc_or_path, reason)`` 列表。
    """
    included: List[Tuple[str, str]] = []
    excluded: List[Tuple[str, str]] = []

    src_root = os.path.join(repo_root, "skill", "src")
    if not os.path.isdir(src_root):
        raise SystemExit(f"缺少 skill 源目录: {src_root}")
    for src in _walk_files(src_root):
        rel = os.path.relpath(src, src_root).replace(os.sep, "/")
        included.append((src, _posix(TOP_DIR, rel)))

    mb_root = os.path.join(repo_root, "membench")
    if not os.path.isdir(mb_root):
        raise SystemExit(f"缺少 membench 目录: {mb_root}")
    for src, arc, reason in _iter_membench(mb_root):
        rel = os.path.relpath(src, mb_root).replace(os.sep, "/")
        if reason:
            excluded.append((_posix(TOP_DIR, arc), reason))
        else:
            included.append((src, _posix(TOP_DIR, arc)))

    doc_root = os.path.join(repo_root, "docs")
    if not os.path.isdir(doc_root):
        raise SystemExit(f"缺少 docs 目录: {doc_root}")
    for src in _walk_files(doc_root):
        rel = os.path.relpath(src, doc_root).replace(os.sep, "/")
        included.append((src, _posix(TOP_DIR, "docs", rel)))

    included.sort(key=lambda it: it[1])
    excluded.sort(key=lambda it: it[0])
    return included, excluded


def _fmt_size(num: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(num)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{num} B"


def list_plan(included: List[Tuple[str, str]], excluded: List[Tuple[str, str]]) -> None:
    """打印将被打包的文件清单与排除项提示（不写盘）。"""
    print(f"将打包 {len(included)} 个文件（顶层目录 {TOP_DIR}/）：")
    for _src, arc in included:
        print(f"  + {arc}")
    if excluded:
        print(f"\n排除 {len(excluded)} 项：")
        for arc, reason in excluded:
            print(f"  - {arc}  [{reason}]")


def _resolve_out(repo_root: str, out: str) -> str:
    out = out if os.path.isabs(out) else os.path.join(os.getcwd(), out)
    return os.path.abspath(out)


def build(included: Iterable[Tuple[str, str]], out_path: str) -> Tuple[int, int, int]:
    """写出 zip，返回 (文件数, 压缩前字节, 压缩后字节)。"""
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    raw_total = 0
    count = 0
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in included:
            info = zipfile.ZipInfo(arc, date_time=FIXED_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(src, "rb") as fh:
                data = fh.read()
            zf.writestr(info, data)
            raw_total += len(data)
            count += 1
    packed = os.path.getsize(out_path)
    return count, raw_total, packed


def check_zip(out_path: str) -> List[str]:
    """包内自检：返回违规条目列表。

    - data 目录下出现被排除数据集（locomo / longmemeval_* / gatemem_office / beam_*）；
    - membench 来源下出现被禁止进包的目录（.venv / __pycache__ / runs / output 等）——
      ``skill/src/output/.gitkeep`` 之类的占位文件不在此列。
    """
    bad: List[str] = []
    with zipfile.ZipFile(out_path) as zf:
        for name in zf.namelist():
            parts = name.split("/")
            if f"/{DATA_DIR_REL}/" in f"/{name}" or name.startswith(DATA_DIR_REL + "/"):
                base = parts[-1]
                if _is_excluded_data(base):
                    bad.append(name)
            # membench 来源前缀：membench/<...>；在此范围内才检查禁止目录
            if len(parts) >= 2 and parts[1] == "membench":
                for part in parts[2:-1]:
                    if part in EXCLUDE_DIR_NAMES_FORBIDDEN:
                        bad.append(name)
                        break
    return bad


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把 skill 源与 membench 工程打包成 .skill（zip）。"
    )
    parser.add_argument(
        "--repo-root", default=".", help="仓库根目录（默认当前目录 .）"
    )
    parser.add_argument(
        "--out",
        default=os.path.join("skill", "agent-memory-eval.skill"),
        help="输出包路径（默认 skill/agent-memory-eval.skill）",
    )
    parser.add_argument(
        "--list", action="store_true", help="只列出将被打包的文件（含排除项提示），不写盘"
    )
    args = parser.parse_args(argv)

    repo_root = os.path.abspath(args.repo_root)
    included, excluded = plan(repo_root)

    if args.list:
        list_plan(included, excluded)
        return 0

    out_path = _resolve_out(repo_root, args.out)
    count, raw_total, packed = build(included, out_path)

    bad = check_zip(out_path)
    if bad:
        print("自检失败：包内出现被排除的按需下载数据集：", file=sys.stderr)
        for name in bad:
            print(f"  ! {name}", file=sys.stderr)
        try:
            os.remove(out_path)
        except OSError:
            pass
        return 1

    print(f"打包完成：{count} 个文件")
    print(f"  压缩前: {_fmt_size(raw_total)}")
    print(f"  压缩后: {_fmt_size(packed)}")
    print(f"  输出包: {out_path}")
    print("  自检: 通过（包内无按需下载数据集）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
