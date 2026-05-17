from __future__ import annotations

from pathlib import Path

import pytest

from metasim.sim.blender.usd_compat import resolve_usd_for_blender


@pytest.mark.general
def test_resolve_usd_for_blender_uses_adjacent_root_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "008.usda"
    root = tmp_path / "008.blender_root.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    root.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.delenv("METASIM_BLENDER_USD_RESOLVER", raising=False)
    monkeypatch.setenv("METASIM_BLENDER_USD_COMPAT", "auto")

    assert resolve_usd_for_blender(source) == root


@pytest.mark.general
def test_resolve_usd_for_blender_off_returns_original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "008.usda"
    root = tmp_path / "008.blender_root.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    root.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.setenv("METASIM_BLENDER_USD_COMPAT", "off")

    assert resolve_usd_for_blender(source) == source


@pytest.mark.general
def test_resolve_usd_for_blender_can_require_generated_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "008.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.delenv("METASIM_BLENDER_USD_RESOLVER", raising=False)
    monkeypatch.setenv("METASIM_BLENDER_USD_COMPAT", "require")

    with pytest.raises(FileNotFoundError, match="Blender-compatible USD root"):
        resolve_usd_for_blender(source)
