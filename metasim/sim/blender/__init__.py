"""Blender render backend."""

from __future__ import annotations

from typing import Any

__all__ = ["BlenderEnv", "BlenderHandler", "BlenderOfflineRenderCfg", "render_state_sequence"]


def __getattr__(name: str) -> Any:
    if name in {"BlenderEnv", "BlenderHandler"}:
        from .blender import BlenderEnv, BlenderHandler

        return {"BlenderEnv": BlenderEnv, "BlenderHandler": BlenderHandler}[name]
    if name in {"BlenderOfflineRenderCfg", "render_state_sequence"}:
        from .offline import BlenderOfflineRenderCfg, render_state_sequence

        return {
            "BlenderOfflineRenderCfg": BlenderOfflineRenderCfg,
            "render_state_sequence": render_state_sequence,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
