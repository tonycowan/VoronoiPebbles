"""
High-level mesh pattern pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from .borders import filter_seeds_by_borders, seed_reach_margin
from .boundary_seeds import BorderSeedSet, build_border_seed_rings, combine_pattern_and_border_seeds
from .gallery_paths import creation_path, ensure_gallery_dirs, next_run_number
from .local_voronoi import characteristic_seed_spacing, partner_search_radius
from .radial_cutters import build_local_radial_pebble_cutters
from .perforate import perforate_mesh
from .sample import poisson_disk_on_surface
from .selector import OuterSideSelector, SurfaceSelection


@dataclass(slots=True)
class PatternResult:
    mesh: trimesh.Trimesh
    selection: SurfaceSelection
    seeds: np.ndarray
    seed_face_ids: np.ndarray
    normals: np.ndarray
    seed_set: BorderSeedSet
    search_radius: float
    typical_spacing: float


@dataclass(slots=True)
class PatternPipeline:
    """
    Generate pebble perforations on a mesh surface.
    """

    min_spacing: float = 10.0
    gap: float = 2.0
    skirt_depth: float = 4.0
    mesh_subdivide: int = 1
    bottom_border: float = 5.0
    top_border: float = 20.0
    seed: int | None = 42
    spacing_metric: str = "euclidean"
    margin: float = 10.0
    perpendicular_half_length: float = 50.0
    outer_cut_margin: float = 0.08
    outer_boundary_backoff: float = 1.0
    cut_depth_margin: float = 1.0
    selector: OuterSideSelector | None = None
    perforation_batch_size: int = 40

    def __post_init__(self) -> None:
        if self.selector is None:
            self.selector = OuterSideSelector(
                bottom_border=self.bottom_border,
                top_border=self.top_border,
                clip_faces_at_borders=False,
            )

    def build_seed_set(
        self,
        mesh: trimesh.Trimesh,
        selection: SurfaceSelection,
        seeds: np.ndarray,
        seed_face_ids: np.ndarray,
    ) -> tuple[BorderSeedSet, float, float]:
        """
        Build the combined pattern + boundary seed set used for Voronoi and cutters.
        """

        normals = selection.submesh.face_normals[seed_face_ids]
        boundary_seeds, boundary_normals = build_border_seed_rings(
            selection.submesh,
            mesh.bounds,
            bottom_border=self.bottom_border,
            top_border=self.top_border,
            spacing=self.min_spacing,
        )
        seed_set = combine_pattern_and_border_seeds(
            seeds,
            normals,
            boundary_seeds,
            boundary_normals,
        )
        search_radius = partner_search_radius(
            seed_set.all_seeds,
            min_spacing=self.min_spacing,
            margin=self.margin,
        )
        typical_spacing = characteristic_seed_spacing(
            seed_set.pattern_seeds,
            min_spacing=self.min_spacing,
        )
        return seed_set, search_radius, typical_spacing

    def run(self, mesh: trimesh.Trimesh) -> PatternResult:
        selection = self.selector.select(mesh)
        seeds, seed_face_ids = poisson_disk_on_surface(
            selection.submesh,
            self.min_spacing,
            seed=self.seed,
            spacing_metric=self.spacing_metric,
        )
        seeds, seed_face_ids = filter_seeds_by_borders(
            seeds,
            seed_face_ids,
            mesh.bounds,
            bottom_border=self.bottom_border,
            top_border=self.top_border,
            reach_margin=seed_reach_margin(self.min_spacing),
        )
        normals = selection.submesh.face_normals[seed_face_ids]
        seed_set, search_radius, typical_spacing = self.build_seed_set(
            mesh,
            selection,
            seeds,
            seed_face_ids,
        )

        return PatternResult(
            mesh=mesh,
            selection=selection,
            seeds=seeds,
            seed_face_ids=seed_face_ids,
            normals=normals,
            seed_set=seed_set,
            search_radius=search_radius,
            typical_spacing=typical_spacing,
        )

    def build_perforated_mesh(self, result: PatternResult) -> trimesh.Trimesh:
        """
        Cut through-holes into the source mesh.
        """

        target_mesh = result.mesh
        for _ in range(self.mesh_subdivide):
            target_mesh = target_mesh.subdivide()

        cutters, cutter_stats = build_local_radial_pebble_cutters(
            result.mesh,
            result.selection.submesh,
            result.seed_set,
            search_radius=result.search_radius,
            perpendicular_half_length=self.perpendicular_half_length,
            gap=self.gap,
            outer_margin=self.outer_cut_margin,
            outer_backoff=self.outer_boundary_backoff,
            cut_depth_margin=self.cut_depth_margin,
        )
        if not cutters:
            raise ValueError("No pebble cutters were generated")

        print(f"  pebble cutters: {len(cutters):,}")
        print(f"  cutter failures: {cutter_stats['failures']:,}")
        print("  performing boolean subtraction (radial toward-axis)...")

        return perforate_mesh(
            target_mesh,
            cutters,
            batch_size=self.perforation_batch_size,
        )

    def export_creation(
        self,
        mesh: trimesh.Trimesh,
        *,
        shape: str,
        pattern: str,
        run_number: int | None = None,
    ) -> Path:
        """
        Export a creation into gallery/creations.

        Files are named <shape>.<pattern>.<run>.stl
        """

        ensure_gallery_dirs()
        if run_number is None:
            run_number = next_run_number(shape, pattern)

        output_path = creation_path(
            shape,
            pattern,
            run_number=run_number,
        )
        mesh.export(output_path)
        return output_path
