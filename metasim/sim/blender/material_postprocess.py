from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaterialFallbackSpec:
    material_class: str
    base_color: tuple[float, float, float, float]
    roughness: float = 0.55
    metallic: float = 0.0


_CLASS_TOKEN_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("glass", ("glass", "window", "acrylic", "mirror", "bottle", "wine_set", "wineglass", "decanter", "jar")),
    ("screen", ("screen", "television", "tv_", "_tv", "display")),
    ("ceramic", ("vase", "ceramic", "porcelain", "ornament", "tea_set", "plate", "pottery")),
    ("floor", ("floor", "parquet", "tile_floor")),
    ("wall", ("wall", "stucco", "paint", "ceiling")),
    ("cabinet", ("cabinet", "drawer", "cupboard", "counter")),
    ("wood", ("wood", "walnut", "oak", "veneer")),
    ("metal", ("metal", "steel", "chrome", "brass", "aluminum", "aluminium")),
    ("fabric", ("fabric", "curtain", "carpet", "rug", "cloth", "linen")),
)


_FALLBACK_SPECS: dict[str, MaterialFallbackSpec] = {
    "wall": MaterialFallbackSpec("wall", (0.84, 0.81, 0.78, 1.0), roughness=0.78),
    "floor": MaterialFallbackSpec("floor", (0.46, 0.36, 0.27, 1.0), roughness=0.62),
    "wood": MaterialFallbackSpec("wood", (0.50, 0.38, 0.26, 1.0), roughness=0.58),
    "cabinet": MaterialFallbackSpec("cabinet", (0.72, 0.66, 0.58, 1.0), roughness=0.58),
    "screen": MaterialFallbackSpec("screen", (0.02, 0.02, 0.02, 1.0), roughness=0.42),
    "glass": MaterialFallbackSpec("glass", (0.70, 0.78, 0.82, 0.38), roughness=0.12),
    "ceramic": MaterialFallbackSpec("ceramic", (0.68, 0.65, 0.60, 1.0), roughness=0.52),
    "transparent": MaterialFallbackSpec("transparent", (0.75, 0.80, 0.82, 0.45), roughness=0.22),
    "emissive": MaterialFallbackSpec("emissive", (1.0, 0.92, 0.74, 1.0), roughness=0.35),
    "metal": MaterialFallbackSpec("metal", (0.56, 0.55, 0.53, 1.0), roughness=0.35, metallic=0.45),
    "fabric": MaterialFallbackSpec("fabric", (0.62, 0.60, 0.56, 1.0), roughness=0.85),
}


def classify_material(*, name: str, mdl_source_asset: str | None, opacity: float | None, has_emissive: bool) -> str | None:
    key = name.lower()
    source = (mdl_source_asset or "").lower()
    if any(token in key for token in ("glass", "window", "acrylic", "mirror", "bottle", "wine_set", "wineglass", "decanter")) or "omniglass" in source:
        return "glass"
    if has_emissive:
        return "emissive"
    if opacity is not None and opacity < 1.0:
        return "transparent"
    return None


def classify_material_from_context(
    *,
    object_name: str,
    material_name: str,
    mdl_source_asset: str | None,
    opacity: float | None,
    has_emissive: bool,
) -> str | None:
    """Classify a material using generic object/material context tokens."""
    explicit = classify_material(
        name=material_name,
        mdl_source_asset=mdl_source_asset,
        opacity=opacity,
        has_emissive=has_emissive,
    )
    if explicit in {"glass", "emissive"}:
        return explicit

    context = f"{object_name} {material_name} {mdl_source_asset or ''}".lower()
    for material_class, tokens in _CLASS_TOKEN_GROUPS:
        if any(token in context for token in tokens):
            return material_class
    return explicit


def fallback_spec_for_class(material_class: str | None) -> MaterialFallbackSpec | None:
    if material_class is None:
        return None
    return _FALLBACK_SPECS.get(material_class)


def material_has_image_texture(material) -> bool:
    node_tree = getattr(material, "node_tree", None)
    if node_tree is None:
        return False
    for node in getattr(node_tree, "nodes", ()) or ():
        node_type = str(getattr(node, "type", "") or getattr(node, "bl_idname", "")).upper()
        if "TEX_IMAGE" in node_type or "SHADERNODETEXIMAGE" in node_type:
            if getattr(node, "image", None) is not None:
                return True
    return False


def _material_node_alpha(material) -> float | None:
    node_tree = getattr(material, "node_tree", None)
    if node_tree is None:
        return None
    for node in getattr(node_tree, "nodes", ()) or ():
        node_type = str(getattr(node, "type", "") or getattr(node, "bl_idname", "")).upper()
        if "BSDF_PRINCIPLED" not in node_type and "SHADERNODEBSDFPRINCIPLED" not in node_type:
            continue
        for socket in getattr(node, "inputs", ()) or ():
            if getattr(socket, "name", "") != "Alpha":
                continue
            try:
                return float(socket.default_value)
            except Exception:
                return None
    return None


def material_needs_fallback(material) -> bool:
    """Return True for default-gray/untextured Blender imports."""
    if material_has_image_texture(material):
        return False
    rgba = getattr(material, "diffuse_color", None)
    if rgba is None:
        return True
    try:
        rgb = tuple(float(value) for value in rgba[:3])
    except Exception:
        return True
    if len(rgb) != 3:
        return True
    spread = max(rgb) - min(rgb)
    mean = sum(rgb) / 3.0
    return spread <= 0.06 and 0.40 <= mean <= 0.98


def material_has_transparent_alpha(material) -> bool:
    return material_source_alpha(material) is not None


def material_source_alpha(material) -> float | None:
    rgba = getattr(material, "diffuse_color", None)
    if rgba is not None:
        try:
            alpha = float(rgba[3])
            if alpha < 1.0:
                return alpha
        except Exception:
            pass
    node_alpha = _material_node_alpha(material)
    if node_alpha is not None and node_alpha < 1.0:
        return node_alpha
    return None


def material_should_apply_class_fallback(material, material_class: str | None) -> bool:
    if material_needs_fallback(material):
        return True
    if material_class in {"glass", "transparent"} and material_has_transparent_alpha(material):
        return True
    return False


def material_should_preserve_class_fallback_surface(material, material_class: str | None) -> bool:
    return (
        material_class in {"glass", "transparent"}
        and material_has_transparent_alpha(material)
        and not material_needs_fallback(material)
    )


def material_render_alpha_for_class_fallback(material, material_class: str | None, fallback_alpha: float) -> float:
    source_alpha = material_source_alpha(material)
    if material_class == "glass" and source_alpha is not None:
        return max(source_alpha, 0.25) if source_alpha < 0.1 else source_alpha
    if material_class == "transparent" and source_alpha is not None:
        return source_alpha
    return fallback_alpha
