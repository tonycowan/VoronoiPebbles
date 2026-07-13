"""
Surface selection policies for mesh patterns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass(slots=True)
class SurfaceSelection:
    """Faces chosen for pattern generation."""

    face_indices: np.ndarray
    submesh: trimesh.Trimesh

    @property
    def face_count(self) -> int:
        return len(self.face_indices)

    @property
    def area(self) -> float:
        return float(self.submesh.area)


def _face_height_extents(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    face_vertices = mesh.vertices[mesh.faces]
    return face_vertices[:, :, 2].min(axis=1), face_vertices[:, :, 2].max(axis=1)


def _radial_normal_component(mesh: trimesh.Trimesh) -> np.ndarray:
    centers = mesh.triangles_center
    normals = mesh.face_normals

    xy = centers[:, :2]
    xy_norm = np.linalg.norm(xy, axis=1, keepdims=True)
    xy_unit = np.divide(
        xy,
        xy_norm,
        out=np.zeros_like(xy),
        where=xy_norm > 1e-9,
    )

    outward = np.column_stack([xy_unit[:, 0], xy_unit[:, 1], np.zeros(len(centers))])
    return np.einsum("ij,ij->i", normals, outward)


@dataclass(slots=True)
class OuterSideSelector:
    """
    Select the outer shell of a lampshade-like mesh, excluding the top cap.
    """

    radial_threshold: float = 0.2
    top_z_min: float = 235.0
    top_radius_max: float = 40.0
    bottom_border: float = 5.0
    top_border: float = 20.0
    clip_faces_at_borders: bool = False

    def select(self, mesh: trimesh.Trimesh) -> SurfaceSelection:
        radial = _radial_normal_component(mesh)
        outer = radial > self.radial_threshold

        centers = mesh.triangles_center
        height = centers[:, 2]
        radius = np.linalg.norm(centers[:, :2], axis=1)

        top_cap = outer & (height > self.top_z_min) & (radius < self.top_radius_max)
        mask = outer & ~top_cap

        face_min_z, face_max_z = _face_height_extents(mesh)

        if self.clip_faces_at_borders and self.bottom_border > 0:
            bottom_z = mesh.bounds[0][2]
            mask &= face_min_z >= bottom_z + self.bottom_border

        if self.clip_faces_at_borders and self.top_border > 0:
            top_z = mesh.bounds[1][2]
            mask &= face_max_z <= top_z - self.top_border

        face_indices = np.where(mask)[0]
        submesh = mesh.submesh([face_indices], append=True, repair=False)

        return SurfaceSelection(
            face_indices=face_indices,
            submesh=submesh,
        )
