"""适配器注册表（SPEC 第 2 节）。

``ADAPTERS`` 把 CLI / Runner 中使用的适配器名映射到适配器类：
``{"mock-strong": ..., "mock-weak": ..., "transcript": ..., "http": ...}``。
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from .base import BaseMemoryAdapter, IngestStats, QueryResult
from .http import HttpAdapter
from .mock import MockStrongAdapter, MockWeakAdapter
from .transcript import TranscriptAdapter

ADAPTERS: dict[str, type[BaseMemoryAdapter]] = {
    "mock-strong": MockStrongAdapter,
    "mock-weak": MockWeakAdapter,
    "transcript": TranscriptAdapter,
    "http": HttpAdapter,
}


def adapters_dir() -> Path:
    """返回适配器持久化目录。

    默认 ``~/.membench/adapters/``；可用环境变量
    ``MEMBENCH_ADAPTERS_DIR`` 覆盖（便于测试与容器场景）。
    """
    override = os.environ.get("MEMBENCH_ADAPTERS_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".membench" / "adapters"


def _load_persistent_adapters() -> None:
    """按路径加载持久化目录中的全部 ``*.py`` 适配器模块。

    按文件名排序保证确定性；每个模块以 ``membench_ext_<文件名>`` 为名，
    经 ``importlib.util.spec_from_file_location`` 按文件路径加载（不依赖
    sys.path）。单个模块加载失败时打印警告到 stderr 并继续，不阻塞其他
    适配器与整体使用。模块若定义 ``register(ADAPTERS)`` 则调用之。
    """
    directory = adapters_dir()
    if not directory.is_dir():
        return
    for py_file in sorted(directory.glob("*.py")):
        module_name = f"membench_ext_{py_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"无法为 {py_file} 构造模块 spec")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            register = getattr(module, "register", None)
            if callable(register):
                register(ADAPTERS)
        except Exception as exc:  # noqa: BLE001 - 坏适配器不阻塞其他
            sys.modules.pop(module_name, None)
            print(
                f"[membench] 警告：持久化适配器 {py_file.name} 加载失败"
                f"（{type(exc).__name__}: {exc}），已跳过",
                file=sys.stderr,
            )


def load_external_adapters() -> None:
    """两阶段加载外部适配器：先持久化目录，后环境变量插件模块。

    第一阶段：``_load_persistent_adapters()`` 加载持久化目录
    （``~/.membench/adapters/`` 或 ``MEMBENCH_ADAPTERS_DIR``）中经
    ``adapter install`` 落盘的全部 ``*.py`` 适配器；单个失败仅警告不中断。

    第二阶段：读 ``MEMBENCH_ADAPTER_MODULES``（逗号分隔的模块路径，如
    ``"my_pkg.mybiz,my_pkg.other"``），逐个 ``importlib.import_module``。
    约定模块被导入时自行向 ``ADAPTERS`` 注册：模块若定义了
    ``register(ADAPTERS)`` 函数则调用之；否则假定“导入即注册”
    （模块顶层执行 ``ADAPTERS["name"] = Class``）。导入或注册失败抛
    ``SystemExit`` 并指明是哪个模块失败。未设环境变量时无副作用。

    优先级：后加载的覆盖先加载的，故同名注册时环境变量方式优先于
    持久化目录，便于临时/CI 场景覆盖已安装的适配器。
    """
    _load_persistent_adapters()
    spec = os.environ.get("MEMBENCH_ADAPTER_MODULES", "").strip()
    if not spec:
        return
    for module_name in spec.split(","):
        module_name = module_name.strip()
        if not module_name:
            continue
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise SystemExit(
                f"加载外部适配器模块失败: {module_name}（{type(exc).__name__}: {exc}）"
            ) from exc
        register = getattr(module, "register", None)
        if callable(register):
            try:
                register(ADAPTERS)
            except Exception as exc:
                raise SystemExit(
                    f"外部适配器模块 {module_name} 的 register(ADAPTERS) 调用失败"
                    f"（{type(exc).__name__}: {exc}）"
                ) from exc


def _infer_registered_name(source: str) -> Optional[str]:
    """用 ast 从源码推断适配器注册名（尽力而为）。

    依次识别两种约定：
    1. ``register`` 函数体内的 ``registry["name"] = ...`` 赋值；
    2. 模块顶层的 ``ADAPTERS["name"] = ...`` 赋值。
    推断不出时返回 None。
    """
    tree = ast.parse(source)
    # 先看 register 函数体内的下标赋值
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "register":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        if (
                            isinstance(target, ast.Subscript)
                            and isinstance(target.slice, ast.Constant)
                            and isinstance(target.slice.value, str)
                        ):
                            return target.slice.value
    # 再看模块顶层 ADAPTERS["name"] = ...
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "ADAPTERS"
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    return target.slice.value
    return None


def _infer_class_name(source: str) -> str:
    """用 ast 推断适配器类名：取源码中第一个继承 BaseMemoryAdapter 的类。

    找不到时返回空字符串（元信息中记为未知）。
    """
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = ""
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name == "BaseMemoryAdapter":
                    return node.name
    return ""


def install_adapter(src_path: str | Path, name: Optional[str] = None) -> dict[str, str]:
    """把业务适配器源码落盘到持久化目录，跨 session 可用。

    校验 ``src_path`` 存在且可 ``ast.parse``，然后把 ``.py`` 复制到
    持久化目录并写入元信息 ``<name>.json``（含注册名、类名、安装时间、
    原始路径）。注册名与类名尽力从源码推断：注册名看 ``register`` 函数
    或顶层 ``ADAPTERS[...]`` 赋值，推断不到时回退到调用方传入的
    ``name``；``name`` 缺省时用源文件名 stem。同名已存在时覆盖并打印提示。

    Args:
        src_path: 适配器 .py 源文件路径。
        name: 注册名（可选，缺省用文件名 stem）。

    Returns:
        ``{"name", "path", "installed_at"}``。

    Raises:
        FileNotFoundError: 源文件不存在。
        ValueError: 源文件不是 .py 或语法错误无法解析。
    """
    src = Path(src_path).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"适配器源文件不存在: {src}")
    if src.suffix != ".py":
        raise ValueError(f"适配器源文件必须是 .py: {src}")
    source = src.read_text(encoding="utf-8")
    try:
        ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"适配器源码语法错误，无法解析: {src}（{exc}）") from exc

    fallback_name = name or src.stem
    reg_name = _infer_registered_name(source) or fallback_name
    class_name = _infer_class_name(source)

    directory = adapters_dir()
    directory.mkdir(parents=True, exist_ok=True)
    dest_py = directory / f"{fallback_name}.py"
    dest_meta = directory / f"{fallback_name}.json"
    if dest_py.exists():
        print(f"[membench] 已存在同名适配器 '{fallback_name}'，覆盖安装")

    shutil.copyfile(src, dest_py)
    installed_at = datetime.now().isoformat(timespec="seconds")
    meta = {
        "name": reg_name,
        "class_name": class_name,
        "installed_at": installed_at,
        "source_path": str(src.resolve()),
    }
    dest_meta.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"name": reg_name, "path": str(dest_py), "installed_at": installed_at}


def list_installed_adapters() -> list[dict[str, str]]:
    """扫描持久化目录，返回已安装适配器的元信息列表（按注册名排序）。

    目录不存在时返回空列表；元信息 json 损坏的条目跳过并打印警告。
    """
    directory = adapters_dir()
    if not directory.is_dir():
        return []
    result: list[dict[str, str]] = []
    for meta_file in sorted(directory.glob("*.json")):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(
                f"[membench] 警告：适配器元信息 {meta_file.name} 损坏（{exc}），已跳过",
                file=sys.stderr,
            )
            continue
        if isinstance(meta, dict):
            result.append(meta)
    result.sort(key=lambda m: str(m.get("name", "")))
    return result


def remove_adapter(name: str) -> bool:
    """从持久化目录删除名为 ``name`` 的适配器（.py + .json）。

    Args:
        name: 安装时使用的文件名（即 ``<name>.py`` / ``<name>.json``）。

    Returns:
        是否删除过内容（此前存在为 True）。
    """
    directory = adapters_dir()
    existed = False
    for suffix in (".py", ".json"):
        path = directory / f"{name}{suffix}"
        if path.exists():
            path.unlink()
            existed = True
    return existed


def get_adapter(name: str, **kwargs: object) -> BaseMemoryAdapter:
    """按名称实例化适配器。

    Args:
        name: 适配器名（须在 ``ADAPTERS`` 注册表中）。
        **kwargs: 透传给适配器构造函数（如 http 的 base_url、
            transcript 的 transcript_path）。

    Raises:
        ValueError: 未知适配器名。
    """
    if name not in ADAPTERS:
        raise ValueError(
            f"未知适配器 '{name}'，可用：{sorted(ADAPTERS)}"
        )
    return ADAPTERS[name](**kwargs)  # type: ignore[arg-type]


__all__ = [
    "ADAPTERS",
    "adapters_dir",
    "get_adapter",
    "install_adapter",
    "list_installed_adapters",
    "remove_adapter",
    "load_external_adapters",
    "BaseMemoryAdapter",
    "IngestStats",
    "QueryResult",
    "MockWeakAdapter",
    "MockStrongAdapter",
    "TranscriptAdapter",
    "HttpAdapter",
]
