"""Asset preparation for the SuperDex backend.

SuperDex (``superdex.physics``, Meta's "Mochi" engine) has two asset constraints that the rest of
MetaSim does not:

* every dynamic rigid body needs a **closed triangle surface mesh** (implicit primitives are
  static-only, and the runtime URDF loader silently drops ``<box>``/``<sphere>``/``<cylinder>``
  collision primitives), and
* collision meshes must be **watertight** so a signed-distance field can be baked; most URDF
  collision meshes in the wild (including the RoboVerse Franka) are open shells.

This module turns MetaSim assets into what SuperDex accepts, without touching the source files:
primitives become trimesh surfaces, and URDFs are rewritten into a per-process cache directory with
every collision geometry replaced by its convex hull (a closed surface by construction) and every
``package://`` / relative path made absolute. Visual geometry is also parsed here so the optional
offscreen renderer can draw the same links the physics engine moves.

Everything here is pure Python + numpy + trimesh; ``superdex`` itself is only imported by the
handler, so these helpers are unit-testable without the engine.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import xml.etree.ElementTree as _ET  # element construction only; parsing goes through defusedxml
from dataclasses import dataclass, field

import numpy as np
from loguru import logger as log

from metasim.utils.xml_safe import ET  # defused parser for untrusted URDF

try:
    import trimesh
except ImportError as _exc:  # pragma: no cover - reported by the handler with an install hint
    trimesh = None
    _TRIMESH_IMPORT_ERROR = _exc
else:
    _TRIMESH_IMPORT_ERROR = None


def _require_trimesh():
    if trimesh is None:
        raise ImportError(
            "The SuperDex backend needs `trimesh` to prepare collision meshes: python -m pip install trimesh"
        ) from _TRIMESH_IMPORT_ERROR


def default_cache_dir() -> str:
    """Directory that baked (hull) URDFs and meshes are written to.

    ``$METASIM_SUPERDEX_CACHE`` overrides the default ``<tmp>/metasim_superdex_cache``. Baked
    files are content-addressed, so the directory can be shared across runs and processes.
    """
    return os.environ.get("METASIM_SUPERDEX_CACHE", os.path.join(tempfile.gettempdir(), "metasim_superdex_cache"))


def _as_vec(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(v) for v in text.split()], dtype=np.float64)


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def origin_to_matrix(origin: ET.Element | None) -> np.ndarray:
    """URDF ``<origin xyz rpy>`` element -> homogeneous 4x4 transform."""
    mat = np.eye(4)
    if origin is None:
        return mat
    mat[:3, :3] = _rpy_to_matrix(_as_vec(origin.get("rpy"), (0.0, 0.0, 0.0)))
    mat[:3, 3] = _as_vec(origin.get("xyz"), (0.0, 0.0, 0.0))
    return mat


def resolve_mesh_path(filename: str, urdf_dir: str) -> str:
    """Resolve a URDF mesh ``filename`` (``package://``, relative or absolute) to an absolute path."""
    if filename.startswith("package://"):
        rel = filename[len("package://") :]
        # ROS-style: package://<pkg>/<path>. Try the path as-is relative to the URDF dir first
        # (RoboVerse assets are laid out that way), then drop the leading package segment.
        for cand in (os.path.join(urdf_dir, rel), os.path.join(urdf_dir, *rel.split("/")[1:])):
            if os.path.exists(cand):
                return os.path.abspath(cand)
        return os.path.abspath(os.path.join(urdf_dir, rel))
    if filename.startswith("file://"):
        filename = filename[len("file://") :]
    if os.path.isabs(filename):
        return filename
    return os.path.abspath(os.path.join(urdf_dir, filename))


def _load_mesh(path: str):
    """Load a mesh file as a single ``trimesh.Trimesh`` (scenes are concatenated)."""
    _require_trimesh()
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):  # pragma: no cover - ``force="mesh"`` normally prevents this
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    return mesh


def geometry_to_trimesh(geometry: ET.Element, urdf_dir: str):
    """URDF ``<geometry>`` element (mesh or primitive) -> ``trimesh.Trimesh`` in the geometry frame."""
    _require_trimesh()
    mesh_el = geometry.find("mesh")
    if mesh_el is not None:
        mesh = _load_mesh(resolve_mesh_path(mesh_el.get("filename", ""), urdf_dir))
        scale = _as_vec(mesh_el.get("scale"), (1.0, 1.0, 1.0))
        if not np.allclose(scale, 1.0):
            mesh = mesh.copy()
            mesh.apply_scale(scale)
        return mesh
    box = geometry.find("box")
    if box is not None:
        return trimesh.creation.box(extents=_as_vec(box.get("size"), (1.0, 1.0, 1.0)))
    sphere = geometry.find("sphere")
    if sphere is not None:
        return trimesh.creation.icosphere(subdivisions=3, radius=float(sphere.get("radius", 0.5)))
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        return trimesh.creation.cylinder(
            radius=float(cylinder.get("radius", 0.5)), height=float(cylinder.get("length", 1.0)), sections=48
        )
    raise ValueError(f"Unsupported URDF geometry: {[c.tag for c in geometry]}")


def watertight_hull(mesh):
    """Return a closed surface for ``mesh``: the mesh itself if already watertight, else its convex hull."""
    if mesh.is_watertight and mesh.volume > 0:
        return mesh
    hull = mesh.convex_hull
    if not hull.is_watertight:  # degenerate (flat) input
        raise ValueError("collision mesh is degenerate: its convex hull is not a closed surface")
    return hull


def _content_key(*parts: object) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


@dataclass
class VisualGeom:
    """One visual geometry of a link: a mesh in the link frame plus an optional RGBA colour."""

    mesh: object  # trimesh.Trimesh
    link_from_geom: np.ndarray  # 4x4
    color: tuple[float, float, float, float] | None = None


@dataclass
class BakedUrdf:
    """Result of :func:`bake_urdf`."""

    path: str
    """Absolute path of the rewritten URDF (hull collision meshes, absolute paths)."""
    link_names: list[str] = field(default_factory=list)
    joint_names: list[str] = field(default_factory=list)
    """Joint names in document order (fixed joints included)."""
    joint_types: dict[str, str] = field(default_factory=dict)
    link_masses: dict[str, float] = field(default_factory=dict)
    visuals: dict[str, list[VisualGeom]] = field(default_factory=dict)
    """Visual geometry per link, for the offscreen renderer."""
    collisions: dict[str, list[tuple[str, np.ndarray]]] = field(default_factory=dict)
    """Per link: (absolute hull mesh path, link_from_geom 4x4) for every baked collision."""


def _material_colors(root: ET.Element) -> dict[str, tuple[float, float, float, float]]:
    colors = {}
    for mat in root.iter("material"):
        name, color = mat.get("name"), mat.find("color")
        if name and color is not None and color.get("rgba"):
            rgba = _as_vec(color.get("rgba"), (0.8, 0.8, 0.8, 1.0))
            colors[name] = tuple(float(v) for v in rgba)
    return colors


def bake_urdf(
    urdf_path: str, scale: tuple[float, float, float] = (1.0, 1.0, 1.0), cache_dir: str | None = None
) -> BakedUrdf:
    """Rewrite ``urdf_path`` so SuperDex can load it, and collect the metadata the handler needs.

    * every ``<collision>`` geometry (mesh **or** primitive) is replaced by a watertight hull mesh
      written to ``cache_dir`` and referenced by absolute path;
    * every ``<visual>`` mesh path is made absolute (SuperDex records them on the prefab);
    * ``scale`` (MetaSim ``BaseObjCfg.scale``) is baked into the collision hulls and returned
      visuals so a scaled asset does not need a scaled source file.

    The output is content-addressed on the source path, its mtime and ``scale``, so repeated
    launches reuse the cache.
    """
    _require_trimesh()
    urdf_path = os.path.abspath(urdf_path)
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    cache_dir = cache_dir or default_cache_dir()
    scale_arr = np.asarray(scale, dtype=np.float64)
    key = _content_key(urdf_path, os.path.getmtime(urdf_path), tuple(np.round(scale_arr, 6)))
    out_dir = os.path.join(cache_dir, f"{os.path.splitext(os.path.basename(urdf_path))[0]}_{key}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(urdf_path))

    urdf_dir = os.path.dirname(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    colors = _material_colors(root)
    baked = BakedUrdf(path=out_path)
    rebuild = not os.path.isfile(out_path)

    for link in root.iter("link"):
        name = link.get("name", "")
        baked.link_names.append(name)
        inertial = link.find("inertial/mass")
        if inertial is not None and inertial.get("value") is not None:
            baked.link_masses[name] = float(inertial.get("value"))

        # --- collisions -> hull meshes -------------------------------------------------------
        baked.collisions[name] = []
        for idx, col in enumerate(link.findall("collision")):
            geometry = col.find("geometry")
            if geometry is None:
                continue
            hull_path = os.path.join(out_dir, f"{name}_collision{idx}_hull.obj")
            if rebuild or not os.path.isfile(hull_path):
                mesh = geometry_to_trimesh(geometry, urdf_dir).copy()
                if not np.allclose(scale_arr, 1.0):
                    mesh.apply_scale(scale_arr)
                hull = watertight_hull(mesh)
                hull.export(hull_path)
            for child in list(geometry):
                geometry.remove(child)
            _ET.SubElement(geometry, "mesh", {"filename": hull_path})
            origin = col.find("origin")
            if origin is not None and not np.allclose(scale_arr, 1.0):
                origin.set("xyz", " ".join(f"{v:.9g}" for v in _as_vec(origin.get("xyz"), (0, 0, 0)) * scale_arr))
            baked.collisions[name].append((hull_path, origin_to_matrix(origin)))

        # --- visuals: absolute paths + renderer metadata ----------------------------------------
        visuals: list[VisualGeom] = []
        for vis in link.findall("visual"):
            geometry = vis.find("geometry")
            if geometry is None:
                continue
            mesh_el = geometry.find("mesh")
            if mesh_el is not None:
                mesh_el.set("filename", resolve_mesh_path(mesh_el.get("filename", ""), urdf_dir))
            color = None
            mat = vis.find("material")
            if mat is not None:
                rgba = mat.find("color")
                if rgba is not None and rgba.get("rgba"):
                    color = tuple(float(v) for v in _as_vec(rgba.get("rgba"), (0.8, 0.8, 0.8, 1.0)))
                elif mat.get("name") in colors:
                    color = colors[mat.get("name")]
            try:
                mesh = geometry_to_trimesh(geometry, urdf_dir).copy()
            except Exception as exc:  # a broken visual must not block physics; it is only drawn
                log.warning(f"[superdex] skipping visual geometry of link '{name}' in {urdf_path}: {exc}")
                continue
            if not np.allclose(scale_arr, 1.0):
                mesh.apply_scale(scale_arr)
            link_from_geom = origin_to_matrix(vis.find("origin"))
            if not np.allclose(scale_arr, 1.0):
                link_from_geom[:3, 3] *= scale_arr
            visuals.append(VisualGeom(mesh=mesh, link_from_geom=link_from_geom, color=color))
        baked.visuals[name] = visuals

    for joint in root.iter("joint"):
        jname = joint.get("name", "")
        baked.joint_names.append(jname)
        baked.joint_types[jname] = joint.get("type", "fixed")
        origin = joint.find("origin")
        if origin is not None and not np.allclose(scale_arr, 1.0):
            origin.set("xyz", " ".join(f"{v:.9g}" for v in _as_vec(origin.get("xyz"), (0, 0, 0)) * scale_arr))

    if rebuild:
        tree.write(out_path)
    return baked


def primitive_trimesh(cfg):
    """MetaSim primitive cfg (cube / sphere / cylinder) -> closed ``trimesh.Trimesh`` centred at the origin."""
    _require_trimesh()
    # Imported lazily to keep this module importable without the full scenario package graph.
    from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveCylinderCfg, PrimitiveSphereCfg

    if isinstance(cfg, PrimitiveCubeCfg):
        return trimesh.creation.box(extents=np.asarray(cfg.size, dtype=np.float64))
    if isinstance(cfg, PrimitiveSphereCfg):
        return trimesh.creation.icosphere(subdivisions=3, radius=float(cfg.radius))
    if isinstance(cfg, PrimitiveCylinderCfg):
        return trimesh.creation.cylinder(radius=float(cfg.radius), height=float(cfg.height), sections=48)
    raise TypeError(f"not a supported primitive cfg: {type(cfg).__name__}")


def mesh_to_arrays(mesh) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a trimesh into the (coordinates, connectivity) arrays ``create_tri_mesh_shape`` wants."""
    coords = np.ascontiguousarray(mesh.vertices, dtype=np.float32).ravel()
    conn = np.ascontiguousarray(mesh.faces, dtype=np.int32).ravel()
    return coords, conn
