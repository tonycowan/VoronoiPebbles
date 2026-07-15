"""
Export intermediate tessellation debug artifacts (Voronoi + shrunk views).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import trimesh

from mesh_patterns.borders import filter_seeds_by_borders, seed_reach_margin
from mesh_patterns.boundary_seeds import BorderSeedSet
from mesh_patterns.export import (
    seed_cloud_scene,
    seed_marker_spheres,
    shrunk_boundary_mesh,
    shrunk_preview_scene,
)
from mesh_patterns.gallery_paths import CREATIONS_DIR, ensure_gallery_dirs, next_run_number
from mesh_patterns.local_voronoi import (
    local_rounded_shrunk_boundary_loops_for_pattern,
    local_shrunk_boundary_loops_for_pattern,
    local_voronoi_boundary_loops_for_pattern,
)
from mesh_patterns.pipeline import PatternPipeline, PatternResult
from mesh_patterns.sample import poisson_disk_on_surface
from mesh_patterns.selector import SurfaceSelection


@dataclass(slots=True)
class TessellationDebugConfig:
    min_spacing: float = 10.0
    gap: float = 2.0
    bottom_border: float = 5.0
    top_border: float = 20.0
    seed: int | None = 42
    spacing_metric: str = "euclidean"
    margin: float = 10.0
    perpendicular_half_length: float = 50.0
    marker_radius: float = 1.5
    voronoi_line_radius: float = 0.25
    voronoi_line_lift: float = 0.35
    outer_line_radius: float = 0.18
    shrunk_line_radius: float = 0.28
    pebble_line_radius: float = 0.32
    rounding_distance: float = 1.0
    rounding_fullness: float = 1.0
    spline_samples: int = 8
    max_overhang_degrees: float = 55.0
    include_outer_voronoi: bool = True


@dataclass(slots=True)
class TessellationSeedSet:
    selection: SurfaceSelection
    seed_set: BorderSeedSet
    search_radius: float
    typical_spacing: float


def tessellation_from_result(result: PatternResult) -> TessellationSeedSet:
    """
    Reuse the exact seed set prepared for cutting.
    """

    return TessellationSeedSet(
        selection=result.selection,
        seed_set=result.seed_set,
        search_radius=result.search_radius,
        typical_spacing=result.typical_spacing,
    )


def build_tessellation_seed_set(
    mesh: trimesh.Trimesh,
    config: TessellationDebugConfig,
    *,
    selection: SurfaceSelection | None = None,
) -> TessellationSeedSet:
    pipeline = PatternPipeline(
        min_spacing=config.min_spacing,
        gap=config.gap,
        bottom_border=config.bottom_border,
        top_border=config.top_border,
        seed=config.seed,
        spacing_metric=config.spacing_metric,
        margin=config.margin,
        perpendicular_half_length=config.perpendicular_half_length,
    )
    pipeline.selector.clip_faces_at_borders = False

    if selection is None:
        selection = pipeline.selector.select(mesh)

    seeds, seed_face_ids = poisson_disk_on_surface(
        selection.submesh,
        config.min_spacing,
        seed=config.seed,
        spacing_metric=config.spacing_metric,
    )
    seeds, seed_face_ids = filter_seeds_by_borders(
        seeds,
        seed_face_ids,
        mesh.bounds,
        bottom_border=config.bottom_border,
        top_border=config.top_border,
        reach_margin=seed_reach_margin(config.min_spacing),
    )
    seed_set, search_radius, typical_spacing = pipeline.build_seed_set(
        mesh,
        selection,
        seeds,
        seed_face_ids,
    )

    return TessellationSeedSet(
        selection=selection,
        seed_set=seed_set,
        search_radius=search_radius,
        typical_spacing=typical_spacing,
    )


def export_voronoi_artifact(
    mesh: trimesh.Trimesh,
    shape: str,
    tessellation: TessellationSeedSet,
    config: TessellationDebugConfig,
    *,
    run_number: int | None = None,
) -> tuple[Path, Path]:
    ensure_gallery_dirs()
    if run_number is None:
        run_number = next_run_number(shape, "voronoi")

    basename = f"{shape}.voronoi.{run_number:03d}"
    selection = tessellation.selection
    seed_set = tessellation.seed_set

    loops, stats = local_voronoi_boundary_loops_for_pattern(
        seed_set,
        selection.submesh,
        min_spacing=config.min_spacing,
        margin=config.margin,
        perpendicular_half_length=config.perpendicular_half_length,
        search_radius=tessellation.search_radius,
    )

    print("\nVoronoi tessellation report")
    print(f"  spacing metric: {config.spacing_metric}")
    print(f"  target spacing: {config.min_spacing:.1f} mm")
    print(f"  typical nearest-neighbor spacing: {tessellation.typical_spacing:.2f} mm")
    print(f"  partner search radius: {tessellation.search_radius:.2f} mm")
    print(f"  pattern seeds: {seed_set.pattern_count:,}")
    print(f"  boundary ring seeds: {seed_set.boundary_count:,}")
    print(f"  cell boundaries: {stats['boundary_count']:,}")
    print(f"  missing boundaries: {stats['missing_boundaries']:,}")
    print(f"  avg partners per seed (search sphere): {stats['avg_partners']:,}")
    print(f"  avg active partners per seed (cell edges): {stats['avg_active_partners']:,}")

    markers = seed_marker_spheres(
        seed_set.pattern_seeds,
        seed_set.pattern_normals,
        radius=config.marker_radius,
    )
    outlines = shrunk_boundary_mesh(
        loops,
        selection.submesh,
        tube_radius=config.voronoi_line_radius,
        lift=config.voronoi_line_lift,
    )
    combined = trimesh.util.concatenate([selection.submesh, markers, outlines])

    stl_path = CREATIONS_DIR / f"{basename}.stl"
    glb_path = CREATIONS_DIR / f"{basename}.glb"
    combined.export(stl_path)

    scene = seed_cloud_scene(
        selection.submesh,
        seed_set.pattern_seeds,
        seed_set.pattern_normals,
        marker_radius=config.marker_radius,
    )
    outlines.visual.face_colors = (30, 90, 220, 255)
    scene.add_geometry(outlines, geom_name="voronoi")
    scene.export(glb_path)

    print(f"  exported voronoi artifact: {glb_path}")
    return stl_path, glb_path


def export_shrunk_artifact(
    mesh: trimesh.Trimesh,
    shape: str,
    tessellation: TessellationSeedSet,
    config: TessellationDebugConfig,
    *,
    run_number: int | None = None,
) -> tuple[Path, Path]:
    ensure_gallery_dirs()
    if run_number is None:
        run_number = next_run_number(shape, "shrunk")

    basename = f"{shape}.shrunk.{run_number:03d}"
    selection = tessellation.selection
    seed_set = tessellation.seed_set

    outer_loops = None
    if config.include_outer_voronoi:
        outer_loops, _ = local_voronoi_boundary_loops_for_pattern(
            seed_set,
            selection.submesh,
            min_spacing=config.min_spacing,
            margin=config.margin,
            perpendicular_half_length=config.perpendicular_half_length,
            search_radius=tessellation.search_radius,
        )

    shrunk_loops, stats = local_shrunk_boundary_loops_for_pattern(
        seed_set,
        selection.submesh,
        min_spacing=config.min_spacing,
        margin=config.margin,
        perpendicular_half_length=config.perpendicular_half_length,
        gap=config.gap,
        search_radius=tessellation.search_radius,
    )

    print("\nShrunk Voronoi report")
    print(f"  minimum pebble gap: {config.gap:.1f} mm")
    print(f"  inset per side: {stats['inset_per_side']:.1f} mm")
    print(f"  pattern seeds: {seed_set.pattern_count:,}")
    print(f"  outer cell boundaries: {len(outer_loops) if outer_loops else 0:,}")
    print(f"  shrunk boundaries: {stats['boundary_count']:,}")
    print(f"  shrink failures: {stats['shrink_failures']:,}")

    markers = seed_marker_spheres(
        seed_set.pattern_seeds,
        seed_set.pattern_normals,
        radius=config.marker_radius,
    )
    parts = [selection.submesh, markers]
    if outer_loops:
        parts.append(
            shrunk_boundary_mesh(
                outer_loops,
                selection.submesh,
                tube_radius=config.outer_line_radius,
                lift=0.30,
            )
        )
    parts.append(
        shrunk_boundary_mesh(
            shrunk_loops,
            selection.submesh,
            tube_radius=config.shrunk_line_radius,
            lift=0.45,
        )
    )
    combined = trimesh.util.concatenate(parts)

    stl_path = CREATIONS_DIR / f"{basename}.stl"
    glb_path = CREATIONS_DIR / f"{basename}.glb"
    combined.export(stl_path)

    scene = shrunk_preview_scene(
        selection.submesh,
        seed_set.pattern_seeds,
        seed_set.pattern_normals,
        shrunk_loops,
        outer_loops=outer_loops,
        marker_radius=config.marker_radius,
        outer_line_radius=config.outer_line_radius,
        shrunk_line_radius=config.shrunk_line_radius,
    )
    scene.export(glb_path)

    print(f"  exported shrunk artifact: {glb_path}")
    return stl_path, glb_path


def export_pebble_cutter_artifact(
    mesh: trimesh.Trimesh,
    shape: str,
    tessellation: TessellationSeedSet,
    config: TessellationDebugConfig,
    *,
    run_number: int | None = None,
) -> tuple[Path, Path]:
    """
    Export the exact shrunk Voronoi outlines used for boolean cutting.
    """

    ensure_gallery_dirs()
    if run_number is None:
        run_number = next_run_number(shape, "pebble_preview")

    basename = f"{shape}.pebble_preview.{run_number:03d}"
    selection = tessellation.selection
    seed_set = tessellation.seed_set

    loops, stats = local_shrunk_boundary_loops_for_pattern(
        seed_set,
        selection.submesh,
        min_spacing=config.min_spacing,
        margin=config.margin,
        perpendicular_half_length=config.perpendicular_half_length,
        gap=config.gap,
        search_radius=tessellation.search_radius,
    )

    print("\nPebble cutter preview report")
    print(f"  minimum pebble gap: {config.gap:.1f} mm")
    print(f"  inset per side: {stats['inset_per_side']:.1f} mm")
    print(f"  pattern seeds: {seed_set.pattern_count:,}")
    print(f"  cutter outlines: {stats['boundary_count']:,}")
    print(f"  shrink failures: {stats['shrink_failures']:,}")

    markers = seed_marker_spheres(
        seed_set.pattern_seeds,
        seed_set.pattern_normals,
        radius=config.marker_radius,
    )
    outlines = shrunk_boundary_mesh(
        loops,
        selection.submesh,
        tube_radius=config.pebble_line_radius,
        lift=0.50,
        follow_surface=False,
    )
    combined = trimesh.util.concatenate([selection.submesh, markers, outlines])

    stl_path = CREATIONS_DIR / f"{basename}.stl"
    glb_path = CREATIONS_DIR / f"{basename}.glb"
    combined.export(stl_path)

    scene = seed_cloud_scene(
        selection.submesh,
        seed_set.pattern_seeds,
        seed_set.pattern_normals,
        marker_radius=config.marker_radius,
    )
    outlines.visual.face_colors = (220, 120, 30, 255)
    scene.add_geometry(outlines, geom_name="pebble_cutters")
    scene.export(glb_path)

    print(f"  exported pebble cutter preview: {glb_path}")
    return stl_path, glb_path


def export_rounded_pebble_preview_artifact(
    mesh: trimesh.Trimesh,
    shape: str,
    tessellation: TessellationSeedSet,
    config: TessellationDebugConfig,
    *,
    run_number: int | None = None,
) -> tuple[Path, Path]:
    """
    Export rounded shrunk Voronoi outlines used for rounded pebble cutting.
    """

    ensure_gallery_dirs()
    if run_number is None:
        run_number = next_run_number(shape, "rounded_pebble_preview")

    basename = f"{shape}.rounded_pebble_preview.{run_number:03d}"
    selection = tessellation.selection
    seed_set = tessellation.seed_set

    loops, stats = local_rounded_shrunk_boundary_loops_for_pattern(
        seed_set,
        selection.submesh,
        min_spacing=config.min_spacing,
        margin=config.margin,
        perpendicular_half_length=config.perpendicular_half_length,
        gap=config.gap,
        rounding_distance=config.rounding_distance,
        spline_samples=config.spline_samples,
        rounding_fullness=config.rounding_fullness,
        max_overhang_degrees=config.max_overhang_degrees,
        search_radius=tessellation.search_radius,
    )

    print("\nRounded pebble cutter preview report")
    print(f"  minimum pebble gap: {config.gap:.1f} mm")
    print(f"  rounding distance: {config.rounding_distance:.1f} mm")
    print(f"  rounding fullness: {config.rounding_fullness:.2f}")
    print(f"  max overhang: {config.max_overhang_degrees:.1f} deg")
    print(f"  inset per side: {stats['inset_per_side']:.1f} mm")
    print(f"  pattern seeds: {seed_set.pattern_count:,}")
    print(f"  cutter outlines: {stats['boundary_count']:,}")
    print(f"  rounding failures: {stats['shrink_failures']:,}")

    markers = seed_marker_spheres(
        seed_set.pattern_seeds,
        seed_set.pattern_normals,
        radius=config.marker_radius,
    )
    outlines = shrunk_boundary_mesh(
        loops,
        selection.submesh,
        tube_radius=config.pebble_line_radius,
        lift=0.50,
        follow_surface=False,
    )
    combined = trimesh.util.concatenate([selection.submesh, markers, outlines])

    stl_path = CREATIONS_DIR / f"{basename}.stl"
    glb_path = CREATIONS_DIR / f"{basename}.glb"
    combined.export(stl_path)

    scene = seed_cloud_scene(
        selection.submesh,
        seed_set.pattern_seeds,
        seed_set.pattern_normals,
        marker_radius=config.marker_radius,
    )
    outlines.visual.face_colors = (90, 180, 255, 255)
    scene.add_geometry(outlines, geom_name="rounded_pebble_cutters")
    scene.export(glb_path)

    print(f"  exported rounded pebble cutter preview: {glb_path}")
    return stl_path, glb_path


def export_tessellation_artifacts(
    mesh: trimesh.Trimesh,
    shape: str,
    config: TessellationDebugConfig,
    *,
    voronoi: bool = True,
    shrunk: bool = True,
    pebble_preview: bool = False,
    rounded_pebble_preview: bool = False,
    tessellation: TessellationSeedSet | None = None,
    selection: SurfaceSelection | None = None,
) -> dict[str, tuple[Path, Path]]:
    if tessellation is None:
        tessellation = build_tessellation_seed_set(mesh, config, selection=selection)
    exports: dict[str, tuple[Path, Path]] = {}

    if voronoi:
        exports["voronoi"] = export_voronoi_artifact(mesh, shape, tessellation, config)
    if shrunk:
        exports["shrunk"] = export_shrunk_artifact(mesh, shape, tessellation, config)
    if pebble_preview:
        exports["pebble_preview"] = export_pebble_cutter_artifact(
            mesh,
            shape,
            tessellation,
            config,
        )
    if rounded_pebble_preview:
        exports["rounded_pebble_preview"] = export_rounded_pebble_preview_artifact(
            mesh,
            shape,
            tessellation,
            config,
        )

    return exports
