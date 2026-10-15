"""build_skill.py 自检回归测试。

覆盖：
- ``.venv/`` / ``node_modules/`` 等构建环境产物不进包；
- ``check_zip`` 能拦下把 .venv 重新塞进 membench 来源的违规情况；
- 正常构建（不含上述产物）能产出包且体积合理（< 5 MB）；
- 占位文件 ``skill/src/output/.gitkeep`` 不会被自检误报为违规。
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import pytest

# 让 build_skill 可作为模块导入
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_skill as bs  # type: ignore[import-not-found]


REPO_ROOT = Path(__file__).resolve().parent.parent  # 仓库根


@pytest.fixture()
def fake_repo(tmp_path: Path) -> Path:
    """最小可用的 repo 结构：skill/src + membench + docs。"""
    (tmp_path / "skill" / "src").mkdir(parents=True)
    (tmp_path / "skill" / "src" / "SKILL.md").write_text("# SKILL\n", encoding="utf-8")
    (tmp_path / "skill" / "src" / "output").mkdir()
    (tmp_path / "skill" / "src" / "output" / ".gitkeep").touch()
    (tmp_path / "membench").mkdir()
    (tmp_path / "membench" / "membench").mkdir()
    (tmp_path / "membench" / "membench" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "membench" / "README.md").write_text("README\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "x.md").write_text("docs\n", encoding="utf-8")
    return tmp_path


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()


def test_venv_excluded_from_pack(fake_repo: Path) -> None:
    """membench/.venv/ 必须被排除（核心回归点：以前漏 .venv 把 50MB 打进包）。"""
    _touch(fake_repo / "membench" / ".venv" / "bin" / "python")
    _touch(fake_repo / "membench" / "site.py")

    included, excluded = bs.plan(str(fake_repo))
    included_arcs = {arc for _, arc in included}
    excluded_arcs = {arc for arc, _ in excluded}

    assert any(".venv" in a for a in excluded_arcs), f".venv 应出现在排除项；实际 {excluded_arcs}"
    assert not any(".venv" in a for a in included_arcs), f".venv 不应入选；实际 {included_arcs}"


def test_node_modules_and_pytest_cache_excluded(fake_repo: Path) -> None:
    """node_modules / .pytest_cache / runs / output 一并排除。"""
    _touch(fake_repo / "membench" / "node_modules" / "x.js")
    _touch(fake_repo / "membench" / ".pytest_cache" / "x")
    _touch(fake_repo / "membench" / "runs" / "x.json")
    _touch(fake_repo / "membench" / "output" / "x.json")

    included, excluded = bs.plan(str(fake_repo))
    excluded_arcs = {arc for arc, _ in excluded}

    for forbidden in ("node_modules", ".pytest_cache", "runs", "output"):
        assert any(forbidden in a for a in excluded_arcs), (
            f"{forbidden} 应被排除；实际 {excluded_arcs}"
        )


def test_skill_src_output_gitkeep_not_flagged(fake_repo: Path) -> None:
    """skill/src/output/.gitkeep 是占位文件，不被 check_zip 误报。"""
    out_path = str(fake_repo / "out.skill")
    included, _ = bs.plan(str(fake_repo))
    bs.build(included, out_path)
    bad = bs.check_zip(out_path)
    assert bad == [], f"占位文件不应被误报；违规 {bad}"


def test_check_zip_catches_venv(fake_repo: Path) -> None:
    """如果 .venv 重新混入（手动绕过 plan），check_zip 必须拦下。"""
    out_path = str(fake_repo / "out.skill")
    _touch(fake_repo / "membench" / ".venv" / "bin" / "python")
    # 模拟有人手动把 .venv 加进 included（绕过 plan）：
    included = [
        (str(fake_repo / "membench" / ".venv" / "bin" / "python"),
         "agent-memory-eval/membench/.venv/bin/python"),
        (str(fake_repo / "skill" / "src" / "SKILL.md"),
         "agent-memory-eval/SKILL.md"),
    ]
    bs.build(included, out_path)
    bad = bs.check_zip(out_path)
    assert any(".venv" in b for b in bad), f"check_zip 必须拦下 .venv；实际 {bad}"


def test_full_build_size_under_5mb() -> None:
    """对真实仓库构建：包大小 < 5 MB（保护性断言，含 .venv 时会 > 30 MB）。"""
    out_path = "/tmp/_test_full_build.skill"
    try:
        included, _ = bs.plan(str(REPO_ROOT))
        bs.build(included, out_path)
        size = os.path.getsize(out_path)
        assert size < 5 * 1024 * 1024, (
            f"skill 包大小 {size / 1024 / 1024:.1f} MB 超过 5 MB——多半是 .venv / "
            "node_modules / runs / output 漏网了"
        )
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)
