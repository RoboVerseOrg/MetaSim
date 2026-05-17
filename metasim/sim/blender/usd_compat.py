from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Callable

USD_COMPAT_ENV = "METASIM_BLENDER_USD_COMPAT"
USD_RESOLVER_ENV = "METASIM_BLENDER_USD_RESOLVER"


def _adjacent_root_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.blender_root.usda")


def _load_callback(spec: str) -> Callable[[str], str | Path]:
    module_name, sep, func_name = spec.partition(":")
    if not sep or not module_name or not func_name:
        raise ValueError(f"{USD_RESOLVER_ENV} must be 'module:function', got {spec!r}")
    func = getattr(importlib.import_module(module_name), func_name)
    if not callable(func):
        raise TypeError(f"{spec!r} does not resolve to a callable")
    return func


def resolve_usd_for_blender(path: str | Path) -> Path:
    source = Path(path)
    mode = os.environ.get(USD_COMPAT_ENV, "auto").strip().lower()
    if mode in {"off", "false", "0", "none"}:
        return source

    callback_spec = os.environ.get(USD_RESOLVER_ENV)
    if callback_spec:
        resolved = Path(_load_callback(callback_spec)(str(source)))
        if not resolved.is_file():
            raise FileNotFoundError(f"Blender USD resolver returned missing path: {resolved}")
        return resolved

    adjacent_root = _adjacent_root_path(source)
    if adjacent_root.is_file():
        return adjacent_root
    if mode in {"require", "required", "strict"}:
        raise FileNotFoundError(f"Blender-compatible USD root does not exist for {source}: {adjacent_root}")
    return source
