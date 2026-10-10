"""适配器持久化注册机制测试（MEMBENCH_ADAPTERS_DIR + adapter install/list/remove）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from membench.adapters import (
    ADAPTERS,
    install_adapter,
    list_installed_adapters,
    load_external_adapters,
    remove_adapter,
)
from membench.cli import main

# 一个最小可用的持久化适配器源码（顶层 ADAPTERS 注册风格）。
_PERSISTENT_SRC = (
    "from membench.adapters import ADAPTERS\n"
    "from membench.adapters.base import BaseMemoryAdapter, QueryResult\n\n"
    "class MybizProdAdapter(BaseMemoryAdapter):\n"
    "    name = 'mybiz-prod'\n"
    "    def reset(self): pass\n"
    "    def ingest(self, messages): return None\n"
    "    def query(self, question):\n"
    "        return QueryResult(answer='', retrieved=[], latency_ms=0.0,\n"
    "                           input_tokens=0, output_tokens=0, llm_calls=0)\n\n"
    "ADAPTERS['mybiz-prod'] = MybizProdAdapter\n"
)


@pytest.fixture()
def persist_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """把持久化目录重定向到 tmp_path，并在测试后清理注册的适配器。"""
    directory = tmp_path / "adapters"
    monkeypatch.setenv("MEMBENCH_ADAPTERS_DIR", str(directory))
    monkeypatch.delenv("MEMBENCH_ADAPTER_MODULES", raising=False)
    yield directory
    ADAPTERS.pop("mybiz-prod", None)
    ADAPTERS.pop("env-override", None)
    for name in list(sys.modules):
        if name.startswith("membench_ext_"):
            sys.modules.pop(name, None)


def _write_src(tmp_path: Path, filename: str = "mybiz_prod.py", content: str = _PERSISTENT_SRC) -> Path:
    src = tmp_path / filename
    src.write_text(content, encoding="utf-8")
    return src


def test_install_persists_files_and_metadata(
    persist_dir: Path, tmp_path: Path
) -> None:
    """install 后 .py 与 .json 落盘，元信息字段正确，返回值含 name/path/installed_at。"""
    src = _write_src(tmp_path)
    result = install_adapter(src)
    assert result["name"] == "mybiz-prod"
    assert Path(result["path"]).is_file()
    assert result["installed_at"]

    py_file = persist_dir / "mybiz_prod.py"
    meta_file = persist_dir / "mybiz_prod.json"
    assert py_file.is_file()
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["name"] == "mybiz-prod"
    assert meta["class_name"] == "MybizProdAdapter"
    assert meta["installed_at"]
    assert meta["source_path"] == str(src.resolve())


def test_install_then_load_registers_adapter(
    persist_dir: Path, tmp_path: Path
) -> None:
    """install 后重新调 load_external_adapters（模拟新进程），适配器出现在 ADAPTERS。"""
    src = _write_src(tmp_path)
    install_adapter(src)
    assert "mybiz-prod" not in ADAPTERS  # install 只落盘，不注册到当前进程
    load_external_adapters()
    assert "mybiz-prod" in ADAPTERS
    assert ADAPTERS["mybiz-prod"].__name__ == "MybizProdAdapter"


def test_install_overwrite_prints_notice(
    persist_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """同名重复 install 覆盖并打印提示。"""
    src = _write_src(tmp_path)
    install_adapter(src)
    install_adapter(src)
    out = capsys.readouterr().out
    assert "覆盖" in out


def test_install_rejects_missing_and_broken_src(
    persist_dir: Path, tmp_path: Path
) -> None:
    """源文件不存在或语法错误时报错。"""
    with pytest.raises(FileNotFoundError):
        install_adapter(tmp_path / "nope.py")
    bad = tmp_path / "bad.py"
    bad.write_text("def broken(:\n", encoding="utf-8")
    with pytest.raises(ValueError, match="语法错误"):
        install_adapter(bad)


def test_adapter_list_shows_persistent(
    persist_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """adapter list 输出包含持久化适配器及其来源标注。"""
    src = _write_src(tmp_path)
    install_adapter(src)
    assert main(["adapter", "list"]) == 0
    out = capsys.readouterr().out
    assert "mybiz-prod" in out
    assert "MybizProdAdapter" in out
    assert "持久化" in out
    assert "内置" in out  # 内置适配器也列出
    assert "mock-strong" in out


def test_remove_stops_loading(
    persist_dir: Path, tmp_path: Path
) -> None:
    """remove 后文件消失、list 为空、加载不再注册。"""
    src = _write_src(tmp_path)
    install_adapter(src)
    assert remove_adapter("mybiz_prod") is True
    assert not (persist_dir / "mybiz_prod.py").exists()
    assert not (persist_dir / "mybiz_prod.json").exists()
    assert remove_adapter("mybiz_prod") is False  # 再删返回 False
    assert list_installed_adapters() == []
    load_external_adapters()
    assert "mybiz-prod" not in ADAPTERS


def test_broken_persistent_adapter_does_not_block_others(
    persist_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """损坏的 .py 只产生 stderr 警告，不阻塞其他持久化适配器加载。"""
    (persist_dir).mkdir(parents=True, exist_ok=True)
    (persist_dir / "aaa_broken.py").write_text("def broken(:\n", encoding="utf-8")
    src = _write_src(tmp_path)
    install_adapter(src)
    load_external_adapters()
    assert "mybiz-prod" in ADAPTERS
    err = capsys.readouterr().err
    assert "aaa_broken.py" in err
    assert "警告" in err


def test_env_modules_override_persistent(
    persist_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同名注册时环境变量方式（后加载）覆盖持久化目录的类。"""
    src = _write_src(tmp_path)
    install_adapter(src)
    # 环境变量插件模块：同名注册另一个类
    (tmp_path / "env_plugin.py").write_text(
        "from membench.adapters import ADAPTERS, MockStrongAdapter\n"
        "ADAPTERS['mybiz-prod'] = MockStrongAdapter\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("MEMBENCH_ADAPTER_MODULES", "env_plugin")
    try:
        load_external_adapters()
        assert ADAPTERS["mybiz-prod"].__name__ == "MockStrongAdapter"
    finally:
        sys.modules.pop("env_plugin", None)


def test_cli_install_list_remove_end_to_end(
    persist_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI 端到端：install 提示下次运行自动加载；remove 删除。"""
    src = _write_src(tmp_path)
    assert main(["adapter", "install", str(src)]) == 0
    out = capsys.readouterr().out
    assert "已安装" in out
    assert "下次运行" in out
    assert (persist_dir / "mybiz_prod.py").is_file()

    assert main(["adapter", "remove", "mybiz_prod"]) == 0
    out = capsys.readouterr().out
    assert "已删除" in out
    assert list_installed_adapters() == []
