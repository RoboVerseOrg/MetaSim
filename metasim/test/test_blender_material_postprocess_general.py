from __future__ import annotations

from types import SimpleNamespace

import pytest

from metasim.sim.blender.material_postprocess import (
    classify_material,
    classify_material_from_context,
    fallback_spec_for_class,
    material_has_transparent_alpha,
    material_needs_fallback,
    material_render_alpha_for_class_fallback,
    material_should_apply_class_fallback,
    material_should_preserve_class_fallback_surface,
)


@pytest.mark.general
def test_classify_material_detects_glass_by_name_and_mdl_source() -> None:
    assert classify_material(name="WindowGlass", mdl_source_asset=None, opacity=1.0, has_emissive=False) == "glass"
    assert classify_material(name="Panel", mdl_source_asset="OmniGlass.mdl", opacity=1.0, has_emissive=False) == "glass"


@pytest.mark.general
def test_classify_material_detects_transparent_and_emissive() -> None:
    assert classify_material(name="Curtain", mdl_source_asset=None, opacity=0.4, has_emissive=False) == "transparent"
    assert classify_material(name="LampShade", mdl_source_asset=None, opacity=1.0, has_emissive=True) == "emissive"


@pytest.mark.general
def test_classify_material_emissive_wins_over_generic_opacity() -> None:
    assert classify_material(name="LampShade", mdl_source_asset=None, opacity=0.4, has_emissive=True) == "emissive"


@pytest.mark.general
@pytest.mark.parametrize(
    ("object_name", "material_name", "expected"),
    [
        ("Root/Meshes/floor/floor_0003/mesh_0004", "Wood_05", "floor"),
        ("Root/Meshes/wall/wall_0012", "WorldGridMaterial", "wall"),
        ("Root/Meshes/window/window_0001", "Glass_02", "glass"),
        ("livingroom_133/television_0000", "ScreenSurface", "screen"),
        ("livingroom_133/television_0000/Meshes/television_0000", "WorldGridMaterial", "screen"),
        ("livingroom_133/wine_set_0005/Meshes/wine_set_0005", "WorldGridMaterial", "glass"),
        ("livingroom_133/vase_0001/Meshes/vase_0001", "WorldGridMaterial", "ceramic"),
        ("kitchen/cabinet_0004", "MI_neutral", "cabinet"),
        ("bedroom/curtain_0000", "Surface_01", "fabric"),
    ],
)
def test_classify_material_from_generic_context(object_name: str, material_name: str, expected: str) -> None:
    assert (
        classify_material_from_context(
            object_name=object_name,
            material_name=material_name,
            mdl_source_asset=None,
            opacity=1.0,
            has_emissive=False,
        )
        == expected
    )
    assert fallback_spec_for_class(expected) is not None


@pytest.mark.general
def test_context_class_wins_over_generic_opacity_transparency() -> None:
    assert (
        classify_material_from_context(
            object_name="livingroom_133/vase_0001/Meshes/vase_0001",
            material_name="WorldGridMaterial",
            mdl_source_asset=None,
            opacity=0.5,
            has_emissive=False,
        )
        == "ceramic"
    )


@pytest.mark.general
def test_material_needs_fallback_only_for_default_gray_untextured_materials() -> None:
    default_gray = SimpleNamespace(diffuse_color=(0.8, 0.8, 0.8, 1.0), node_tree=SimpleNamespace(nodes=[]))
    default_white = SimpleNamespace(diffuse_color=(0.96, 0.96, 0.96, 1.0), node_tree=SimpleNamespace(nodes=[]))
    colored = SimpleNamespace(diffuse_color=(0.8, 0.2, 0.1, 1.0), node_tree=SimpleNamespace(nodes=[]))
    textured = SimpleNamespace(
        diffuse_color=(0.8, 0.8, 0.8, 1.0),
        node_tree=SimpleNamespace(nodes=[SimpleNamespace(type="TEX_IMAGE", image=object())]),
    )

    assert material_needs_fallback(default_gray) is True
    assert material_needs_fallback(default_white) is True
    assert material_needs_fallback(colored) is False
    assert material_needs_fallback(textured) is False


@pytest.mark.general
def test_material_needs_fallback_for_untextured_transparent_preview_material() -> None:
    transparent_glass = SimpleNamespace(
        diffuse_color=(0.901, 0.9355, 0.9632, 0.01),
        node_tree=SimpleNamespace(nodes=[]),
    )

    assert material_has_transparent_alpha(transparent_glass) is True
    assert material_should_apply_class_fallback(transparent_glass, "glass") is True


@pytest.mark.general
def test_material_has_transparent_alpha_reads_principled_alpha_socket() -> None:
    alpha_socket = SimpleNamespace(name="Alpha", default_value=0.01)
    bsdf = SimpleNamespace(type="BSDF_PRINCIPLED", bl_idname="ShaderNodeBsdfPrincipled", inputs=[alpha_socket])
    transparent_glass = SimpleNamespace(
        diffuse_color=(0.901, 0.9355, 0.9632, 1.0),
        node_tree=SimpleNamespace(nodes=[bsdf]),
    )

    assert material_has_transparent_alpha(transparent_glass) is True
    assert material_needs_fallback(transparent_glass) is False
    assert material_should_apply_class_fallback(transparent_glass, "glass") is True
    assert material_should_preserve_class_fallback_surface(transparent_glass, "glass") is True


@pytest.mark.general
def test_material_alpha_does_not_force_generic_context_fallback_when_source_color_exists() -> None:
    alpha_socket = SimpleNamespace(name="Alpha", default_value=0.8)
    bsdf = SimpleNamespace(type="BSDF_PRINCIPLED", bl_idname="ShaderNodeBsdfPrincipled", inputs=[alpha_socket])
    dark_translucent_ornament = SimpleNamespace(
        diffuse_color=(0.1103, 0.1103, 0.1103, 1.0),
        node_tree=SimpleNamespace(nodes=[bsdf]),
    )

    assert material_has_transparent_alpha(dark_translucent_ornament) is True
    assert material_needs_fallback(dark_translucent_ornament) is False
    assert material_should_apply_class_fallback(dark_translucent_ornament, "ceramic") is False
    assert material_should_preserve_class_fallback_surface(dark_translucent_ornament, "ceramic") is False


@pytest.mark.general
def test_default_gray_transparent_glass_gets_full_fallback_instead_of_preserving_gray() -> None:
    alpha_socket = SimpleNamespace(name="Alpha", default_value=0.2)
    bsdf = SimpleNamespace(type="BSDF_PRINCIPLED", bl_idname="ShaderNodeBsdfPrincipled", inputs=[alpha_socket])
    gray_glass = SimpleNamespace(
        diffuse_color=(0.8, 0.8, 0.8, 1.0),
        node_tree=SimpleNamespace(nodes=[bsdf]),
    )

    assert material_should_apply_class_fallback(gray_glass, "glass") is True
    assert material_should_preserve_class_fallback_surface(gray_glass, "glass") is False


@pytest.mark.general
def test_glass_alpha_floor_keeps_near_zero_source_opacity_visible_but_not_milky() -> None:
    alpha_socket = SimpleNamespace(name="Alpha", default_value=0.01)
    bsdf = SimpleNamespace(type="BSDF_PRINCIPLED", bl_idname="ShaderNodeBsdfPrincipled", inputs=[alpha_socket])
    pale_glass = SimpleNamespace(
        diffuse_color=(0.901, 0.9355, 0.9632, 1.0),
        node_tree=SimpleNamespace(nodes=[bsdf]),
    )

    assert material_render_alpha_for_class_fallback(pale_glass, "glass", 0.38) == pytest.approx(0.25)


@pytest.mark.general
def test_glass_alpha_policy_preserves_non_tiny_source_opacity() -> None:
    alpha_socket = SimpleNamespace(name="Alpha", default_value=0.35)
    bsdf = SimpleNamespace(type="BSDF_PRINCIPLED", bl_idname="ShaderNodeBsdfPrincipled", inputs=[alpha_socket])
    transparent_glass = SimpleNamespace(
        diffuse_color=(0.2, 0.3, 0.4, 1.0),
        node_tree=SimpleNamespace(nodes=[bsdf]),
    )

    assert material_render_alpha_for_class_fallback(transparent_glass, "glass", 0.38) == pytest.approx(0.35)
