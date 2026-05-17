from __future__ import annotations

import pytest

from metasim.sim.blender.material_postprocess import classify_material


@pytest.mark.general
def test_classify_material_detects_glass_by_name_and_mdl_source() -> None:
    assert classify_material(name="WindowGlass", mdl_source_asset=None, opacity=1.0, has_emissive=False) == "glass"
    assert classify_material(name="Panel", mdl_source_asset="OmniGlass.mdl", opacity=1.0, has_emissive=False) == "glass"


@pytest.mark.general
def test_classify_material_detects_transparent_and_emissive() -> None:
    assert classify_material(name="Curtain", mdl_source_asset=None, opacity=0.4, has_emissive=False) == "transparent"
    assert classify_material(name="LampShade", mdl_source_asset=None, opacity=1.0, has_emissive=True) == "emissive"
