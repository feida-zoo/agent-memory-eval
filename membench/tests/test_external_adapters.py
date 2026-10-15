"""外部适配器插件机制（MEMBENCH_ADAPTER_MODULES）的回归测试（修复 5 替代项）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from membench.adapters import ADAPTERS, load_external_adapters


@pytest.fixture(autouse=True)
def _isolate_adapters_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离持久化适配器目录，任何测试不得依赖/受本机 ``~/.membench/adapters`` 影响。

    ``load_external_adapters()`` 第一阶段总会扫描持久化目录；本机若执行过
    ``adapter install``（如接入方持久化的业务适配器），未隔离的"无副作用"
    断言会因额外注册而失败（且仅在先于其他触发加载的用例执行时暴露，
    属顺序依赖缺陷）。
    """
    empty = tmp_path / "adapters"
    empty.mkdir()
    monkeypatch.setenv("MEMBENCH_ADAPTERS_DIR", str(empty))


@pytest.fixture()
def _plugin_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """把 tmp_path 注入 sys.path，并清理已导入的插件模块名。"""
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("my_ext_adapter", "my_ext_register"):
        sys.modules.pop(name, None)
    yield tmp_path
    for name in ("my_ext_adapter", "my_ext_register"):
        sys.modules.pop(name, None)
    ADAPTERS.pop("ext-import", None)
    ADAPTERS.pop("ext-register", None)


def test_import_side_effect_registration(_plugin_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """模块顶层执行 ADAPTERS["name"] = Class 的“导入即注册”风格。"""
    (_plugin_dir / "my_ext_adapter.py").write_text(
        "from membench.adapters import ADAPTERS, MockWeakAdapter\n"
        "ADAPTERS['ext-import'] = MockWeakAdapter\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMBENCH_ADAPTER_MODULES", "my_ext_adapter")
    load_external_adapters()
    assert "ext-import" in ADAPTERS


def test_register_function_registration(_plugin_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """模块提供 register(ADAPTERS) 函数时调用之。"""
    (_plugin_dir / "my_ext_register.py").write_text(
        "from membench.adapters import MockStrongAdapter\n"
        "def register(registry):\n"
        "    registry['ext-register'] = MockStrongAdapter\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMBENCH_ADAPTER_MODULES", "my_ext_register")
    load_external_adapters()
    assert "ext-register" in ADAPTERS


def test_missing_module_system_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """指向不存在模块时 SystemExit 且消息含模块名。"""
    monkeypatch.setenv("MEMBENCH_ADAPTER_MODULES", "no_such_pkg.no_such_module")
    with pytest.raises(SystemExit, match="no_such_pkg.no_such_module"):
        load_external_adapters()


def test_env_not_set_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """无任何外部适配器来源（无环境变量、持久化目录为空）时无副作用。"""
    monkeypatch.delenv("MEMBENCH_ADAPTER_MODULES", raising=False)
    before = dict(ADAPTERS)
    load_external_adapters()
    assert dict(ADAPTERS) == before


def test_persisted_dir_adapters_are_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """持久化目录中的 ``*.py`` 会被 load_external_adapters 注册（隔离目录内自证机制）。"""
    adir = tmp_path / "adapters-with-one"
    adir.mkdir()
    (adir / "biz_persisted.py").write_text(
        "from membench.adapters import ADAPTERS, MockStrongAdapter\n"
        "ADAPTERS['biz-persisted'] = MockStrongAdapter\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMBENCH_ADAPTERS_DIR", str(adir))
    monkeypatch.delenv("MEMBENCH_ADAPTER_MODULES", raising=False)
    try:
        load_external_adapters()
        assert "biz-persisted" in ADAPTERS
    finally:
        ADAPTERS.pop("biz-persisted", None)
