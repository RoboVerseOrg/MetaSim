from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import metasim.sim.blender.blender as blender_mod
from metasim.sim.blender.blender import BlenderHandler, _scene_usd_xform_matrix


@pytest.mark.general
def test_scene_usd_xform_matches_usd_transform_order() -> None:
    scene_cfg = SimpleNamespace(
        default_position=(1.0, 2.0, 3.0),
        default_orientation=(0.0, 0.0, 0.0, 1.0),
        scale=(2.0, 3.0, 4.0),
    )

    matrix = _scene_usd_xform_matrix(scene_cfg)

    assert tuple(round(value, 6) for value in matrix.to_translation()) == (-1.0, -2.0, 3.0)
    transformed_x = matrix @ blender_mod.Vector((1.0, 0.0, 0.0, 1.0))
    assert tuple(round(value, 6) for value in transformed_x[:3]) == (-3.0, -2.0, 3.0)


@pytest.mark.general
def test_import_scene_applies_scene_transform_once_after_identity_import(monkeypatch: pytest.MonkeyPatch) -> None:
    root = SimpleNamespace(parent=None, matrix_basis=None, matrix_world=None)
    calls: dict[str, object] = {}

    class Scene:
        name = "kujiale_0008"
        default_position = (-7.2, -1.5, 0.0)
        quat = (0.0, 0.0, 0.0, 1.0)
        scale = (1.0, 1.0, 1.0)

        def file_name(self, sim_name: str) -> str:
            assert sim_name == "blender"
            return "third_party/InteriorAgent/kujiale_0008/008.usda"

    def fake_import_usd_visuals(path, **kwargs):
        calls["path"] = Path(path)
        calls["kwargs"] = kwargs
        return SimpleNamespace(root=root, imported=[])

    monkeypatch.setattr(blender_mod, "resolve_usd_for_blender", lambda path: Path(path))
    monkeypatch.setattr(blender_mod, "import_usd_visuals", fake_import_usd_visuals)
    monkeypatch.setattr(blender_mod, "_apply_optional_usd_stage_enhancements", lambda path, imported: None)
    monkeypatch.setattr(blender_mod, "_record_material_class_diagnostics", lambda imported: None)
    monkeypatch.setattr(blender_mod, "_apply_material_class_fallbacks", lambda imported: None)

    handler = SimpleNamespace(scenario=SimpleNamespace(scene=Scene()), _scene_objs=[])
    BlenderHandler._import_scene(handler)

    assert calls["kwargs"]["default_position"] == (0.0, 0.0, 0.0)
    assert calls["kwargs"]["default_orientation"] == (1.0, 0.0, 0.0, 0.0)
    assert calls["kwargs"]["scale"] == (1.0, 1.0, 1.0)
    assert tuple(round(value, 6) for value in root.matrix_basis.to_translation()) == (7.2, 1.5, 0.0)
    assert handler._scene_objs == [root]


@pytest.mark.general
def test_usda_text_reference_xforms_are_loaded_through_blender_root_sublayers(tmp_path: Path) -> None:
    scene = tmp_path / "008.usda"
    scene.write_text(
        """#usda 1.0
def Xform "Root"
{
    def Scope "Meshes"
    {
        def Scope "livingroom"
        {
            def Xform "television_0001" (
                prepend references = @./Meshes/television_0001.usd@
            )
            {
                quatf xformOp:orient = (0.7071068, 0, 0, -0.7071068)
                float3 xformOp:scale = (1.5, 2.0, 2.5)
                double3 xformOp:translate = (10.0, 1.5, 2.0)
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
            }
        }
    }
}
""",
        encoding="utf-8",
    )
    overlay = tmp_path / "008.blender_materials.usda"
    overlay.write_text("#usda 1.0\n", encoding="utf-8")
    root = tmp_path / "008.blender_root.usda"
    root.write_text(
        """#usda 1.0
(
    subLayers = [
        @008.blender_materials.usda@,
        @008.usda@
    ]
)
""",
        encoding="utf-8",
    )

    xforms = blender_mod._usd_reference_xforms_by_name_text_fallback(root)

    assert set(xforms) == {"television_0001"}
    matrix = xforms["television_0001"][0]
    assert tuple(round(value, 6) for value in matrix.to_translation()) == (10.0, 1.5, 2.0)
    assert tuple(round(value, 6) for value in matrix.to_scale()) == (1.5, 2.0, 2.5)


@pytest.mark.general
def test_apply_usd_reference_xforms_uses_text_fallback_when_pxr_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = tmp_path / "scene.usda"
    scene.write_text(
        """#usda 1.0
def Xform "Root"
{
    def Xform "cabinet_0004" (
        prepend references = @./Meshes/cabinet_0004.usd@
    )
    {
        double3 xformOp:translate = (3, 4, 5)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
""",
        encoding="utf-8",
    )
    class FakeObject:
        name = "cabinet_0004"
        parent = None
        matrix_basis = blender_mod.Matrix.Identity(4)

    obj = FakeObject()
    monkeypatch.setattr(
        blender_mod,
        "_usd_reference_xforms_by_name",
        lambda _path: (_ for _ in ()).throw(blender_mod.UsdPythonBindingsUnavailable("missing pxr")),
    )

    blender_mod._apply_usd_reference_xforms(scene, [obj])

    assert tuple(round(value, 6) for value in obj.matrix_basis.to_translation()) == (3.0, 4.0, 5.0)


@pytest.mark.general
def test_material_class_fallbacks_use_each_object_context_for_repeated_generic_material_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMaterial(dict):
        def __init__(self, name: str):
            super().__init__()
            self.name = name
            self.diffuse_color = (0.8, 0.8, 0.8, 1.0)
            self.node_tree = SimpleNamespace(nodes=[])
            self.blend_method = "OPAQUE"

    class FakeSlot:
        def __init__(self, material):
            self.material = material

    class FakeObject:
        def __init__(self, name: str, material, parent=None):
            self.name = name
            self.type = "MESH"
            self.parent = parent
            self.children = []
            self.material_slots = [FakeSlot(material)]
            self.instance_collection = None

    monkeypatch.setattr(
        blender_mod,
        "_repair_material_surface_shader",
        lambda material, rgba=None: setattr(material, "diffuse_color", rgba or material.diffuse_color),
    )

    unrelated = FakeObject("chair_0000", FakeMaterial("WorldGridMaterial"))
    television = FakeObject("television_0001", FakeMaterial("WorldGridMaterial"))

    blender_mod._apply_material_class_fallbacks([unrelated, television])

    assert television.material_slots[0].material["metasim_material_class"] == "screen"
    assert television.material_slots[0].material.diffuse_color[:3] == pytest.approx((0.02, 0.02, 0.02))


@pytest.mark.general
def test_usd_material_specs_keep_duplicate_material_names_path_scoped(tmp_path: Path) -> None:
    pytest.importorskip("pxr")
    scene = tmp_path / "scene.usda"
    scene.write_text(
        """#usda 1.0
def Xform "Root"
{
    def Scope "A"
    {
        def Scope "Looks"
        {
            def Material "WorldGridMaterial"
            {
                def Shader "BlenderPreview"
                {
                    uniform token info:id = "UsdPreviewSurface"
                    color3f inputs:diffuseColor = (0.1, 0.2, 0.3)
                }
            }
        }
    }
    def Scope "B"
    {
        def Scope "Looks"
        {
            def Material "WorldGridMaterial"
            {
                def Shader "BlenderPreview"
                {
                    uniform token info:id = "UsdPreviewSurface"
                    color3f inputs:diffuseColor = (0.8, 0.7, 0.6)
                }
            }
        }
    }
}
""",
        encoding="utf-8",
    )

    specs = blender_mod._usd_material_specs_from_stage(scene)

    assert specs["/Root/A/Looks/WorldGridMaterial"].base_color[:3] == pytest.approx((0.1, 0.2, 0.3))
    assert specs["/Root/B/Looks/WorldGridMaterial"].base_color[:3] == pytest.approx((0.8, 0.7, 0.6))
    assert "WorldGridMaterial" not in specs


@pytest.mark.general
def test_usd_base_color_texture_alias_resolves_without_basecolor_tex_name(tmp_path: Path) -> None:
    texture = tmp_path / "textures" / "wall_albedo.png"
    texture.parent.mkdir()
    texture.write_bytes(b"png")

    resolved = blender_mod._usd_base_color_texture_path(
        {"BaseColor_Texture": "textures/wall_albedo.png"},
        tmp_path,
    )

    assert resolved == texture.resolve()


@pytest.mark.general
def test_apply_usd_material_specs_matches_duplicate_names_by_object_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMaterial:
        users = 1

        def __init__(self, name: str):
            self.name = name
            self.applied = None

    class FakeSlot:
        def __init__(self, material):
            self.material = material

    class FakeObject:
        def __init__(self, name: str, material, parent=None):
            self.name = name
            self.type = "MESH"
            self.parent = parent
            self.children = []
            self.material_slots = [FakeSlot(material)]
            self.instance_collection = None

    def fake_apply(material, spec):
        material.applied = spec.path

    monkeypatch.setattr(blender_mod, "_apply_usd_material_spec_to_material", fake_apply)

    mat_a = FakeMaterial("WorldGridMaterial")
    mat_b = FakeMaterial("WorldGridMaterial")
    obj_a = FakeObject("mesh_0000", mat_a, parent=SimpleNamespace(name="asset_a", parent=None))
    obj_b = FakeObject("mesh_0000.001", mat_b, parent=SimpleNamespace(name="asset_b", parent=None))
    specs = {
        "/Root/Meshes/asset_a/Meshes/asset_a/mesh_0000/Looks/WorldGridMaterial": blender_mod.UsdMaterialSpec(
            name="WorldGridMaterial",
            path="/Root/Meshes/asset_a/Meshes/asset_a/mesh_0000/Looks/WorldGridMaterial",
        ),
        "/Root/Meshes/asset_b/Meshes/asset_b/mesh_0000/Looks/WorldGridMaterial": blender_mod.UsdMaterialSpec(
            name="WorldGridMaterial",
            path="/Root/Meshes/asset_b/Meshes/asset_b/mesh_0000/Looks/WorldGridMaterial",
        ),
    }

    blender_mod._apply_usd_material_specs([obj_a, obj_b], specs)

    assert mat_a.applied == "/Root/Meshes/asset_a/Meshes/asset_a/mesh_0000/Looks/WorldGridMaterial"
    assert mat_b.applied == "/Root/Meshes/asset_b/Meshes/asset_b/mesh_0000/Looks/WorldGridMaterial"


@pytest.mark.general
def test_usd_asset_material_rgba_discovers_adjacent_urdf_sidecar(tmp_path: Path) -> None:
    usd_dir = tmp_path / "asset" / "usd"
    urdf_dir = tmp_path / "asset" / "urdf"
    usd_dir.mkdir(parents=True)
    urdf_dir.mkdir(parents=True)
    usd_path = usd_dir / "URDF_Data.usd"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    (urdf_dir / "URDF_Data.urdf").write_text(
        """<robot name="sidecar">
  <material name="mat0">
    <color rgba="0.26667 0.58039 0.27843 1.0"/>
  </material>
</robot>
""",
        encoding="utf-8",
    )

    colors = blender_mod._asset_material_rgba(SimpleNamespace(usd_path=str(usd_path), extra_resources=[]))

    assert "mat0" in colors
    assert colors["mat0"][1] > colors["mat0"][0]
    assert colors["mat0"][1] > colors["mat0"][2]


@pytest.mark.general
def test_usd_asset_material_rgba_discovers_direct_root_urdf_sidecar(tmp_path: Path) -> None:
    usd_dir = tmp_path / "asset" / "usd"
    usd_dir.mkdir(parents=True)
    usd_path = usd_dir / "thing.usd"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    (tmp_path / "asset" / "thing.urdf").write_text(
        """<robot name="thing">
  <material name="material_0">
    <color rgba="0.1 0.2 0.7 1.0"/>
  </material>
</robot>
""",
        encoding="utf-8",
    )

    colors = blender_mod._asset_material_rgba(SimpleNamespace(usd_path=str(usd_path), extra_resources=[]))

    assert "material_0" in colors
    assert colors["material_0"][2] > colors["material_0"][0]


@pytest.mark.general
def test_asset_material_textures_discovers_mjcf_sidecar_texture(tmp_path: Path) -> None:
    usd_dir = tmp_path / "asset" / "usd"
    mjcf_mesh_dir = tmp_path / "asset" / "mjcf" / "mesh"
    usd_dir.mkdir(parents=True)
    mjcf_mesh_dir.mkdir(parents=True)
    usd_path = usd_dir / "thing.usd"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    texture = mjcf_mesh_dir / "material_0.png"
    texture.write_bytes(b"png")
    (tmp_path / "asset" / "mjcf" / "thing.xml").write_text(
        """<mujoco model="thing">
  <asset>
    <material name="material_0" texture="texture_0_material_0"/>
    <texture name="texture_0_material_0" type="2d" file="mesh/material_0.png"/>
  </asset>
</mujoco>
""",
        encoding="utf-8",
    )

    textures = blender_mod._asset_material_textures(SimpleNamespace(usd_path=str(usd_path), extra_resources=[]))

    assert textures["material_0"] == texture.resolve()


@pytest.mark.general
def test_asset_material_textures_discovers_direct_root_urdf_obj_mtl_texture(tmp_path: Path) -> None:
    usd_dir = tmp_path / "asset" / "usd"
    mesh_dir = tmp_path / "asset" / "mesh"
    usd_dir.mkdir(parents=True)
    mesh_dir.mkdir(parents=True)
    usd_path = usd_dir / "thing.usd"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    texture = mesh_dir / "material_0.png"
    texture.write_bytes(b"png")
    (mesh_dir / "thing.obj").write_text("mtllib material.mtl\nusemtl material_0\n", encoding="utf-8")
    (mesh_dir / "material.mtl").write_text("newmtl material_0\nmap_Kd material_0.png\n", encoding="utf-8")
    (tmp_path / "asset" / "thing.urdf").write_text(
        """<robot name="thing">
  <link name="thing">
    <visual>
      <geometry>
        <mesh filename="mesh/thing.obj"/>
      </geometry>
    </visual>
  </link>
</robot>
""",
        encoding="utf-8",
    )

    textures = blender_mod._asset_material_textures(SimpleNamespace(usd_path=str(usd_path), extra_resources=[]))

    assert textures["material_0"] == texture.resolve()


@pytest.mark.general
def test_apply_imported_material_textures_matches_normalized_names(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    applied: list[tuple[str, Path]] = []
    texture = tmp_path / "material_0.png"
    texture.write_bytes(b"png")

    class FakeMaterial:
        name = "material_0.001"

    class FakeSlot:
        material = FakeMaterial()

    class FakeObject:
        name = "mesh"
        type = "MESH"
        material_slots = [FakeSlot()]
        children = []
        instance_collection = None

    monkeypatch.setattr(blender_mod, "material_has_image_texture", lambda _material: False)
    monkeypatch.setattr(
        blender_mod,
        "_apply_texture_to_material",
        lambda material, texture_path: applied.append((material.name, texture_path)),
    )

    blender_mod._apply_imported_material_textures([FakeObject()], {"material_0": texture})

    assert applied == [("material_0.001", texture)]
