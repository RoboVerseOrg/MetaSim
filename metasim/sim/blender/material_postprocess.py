from __future__ import annotations


def classify_material(*, name: str, mdl_source_asset: str | None, opacity: float | None, has_emissive: bool) -> str | None:
    key = name.lower()
    source = (mdl_source_asset or "").lower()
    if any(token in key for token in ("glass", "window", "acrylic", "mirror")) or "omniglass" in source:
        return "glass"
    if opacity is not None and opacity < 1.0:
        return "transparent"
    if has_emissive:
        return "emissive"
    return None
