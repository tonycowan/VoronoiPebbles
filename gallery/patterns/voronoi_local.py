"""
Test local perpendicular-bisector Voronoi cells on a small seed cluster.

Usage:
    python gallery/patterns/voronoi_local.py gallery/shapes/CurvyLamp.stl
    python gallery/patterns/voronoi_local.py gallery/shapes/CurvyLamp.stl --seed-count 10 --margin 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

from mesh_patterns import PatternPipeline
from mesh_patterns.borders import filter_seeds_by_borders, seed_reach_margin
from mesh_patterns.export import (
    line_polylines_to_mesh,
    seed_marker_spheres,
)
from mesh_patterns.gallery_paths import CREATIONS_DIR, ensure_gallery_dirs, next_run_number
from mesh_patterns.local_voronoi import (
    build_local_voronoi_cells,
    characteristic_seed_spacing,
    max_seed_neighbor_distance,
    partner_sphere_mesh,
    select_cluster_seed_indices,
)
from mesh_patterns.sample import poisson_disk_on_surface

PATTERN = "voronoi_local"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a small local Voronoi bisector test on mesh seeds.",
    )
    parser.add_argument("mesh", type=Path, help="Path to the target mesh STL")
    parser.add_argument("--min-spacing", type=float, default=10.0)
    parser.add_argument("--spacing-metric", choices=("surface", "euclidean"), default="euclidean")
    parser.add_argument("--bottom-border", type=float, default=5.0)
    parser.add_argument("--top-border", type=float, default=20.0)
    parser.add_argument(
        "--seed-count",
        type=int,
        default=0,
        help="Number of seeds to test (0 = all seeds)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=10.0,
        help="Added to max nearest-neighbor distance for partner search sphere",
    )
    parser.add_argument(
        "--perpendicular-half-length",
        type=float,
        default=50.0,
        help="Length of each perpendicular bisector side in mm",
    )
    parser.add_argument("--marker-radius", type=float, default=1.5)
    parser.add_argument("--line-radius", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def _colored_mesh(mesh: trimesh.Trimesh, color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    copy = mesh.copy()
    copy.visual.face_colors = color
    return copy


def main() -> None:
    args = parse_args()
    mesh_path = args.mesh.resolve()
    shape = mesh_path.stem

    mesh = trimesh.load(mesh_path, force="mesh")
    pipeline = PatternPipeline(
        min_spacing=args.min_spacing,
        bottom_border=args.bottom_border,
        top_border=args.top_border,
        seed=args.seed,
    )

    selection = pipeline.selector.select(mesh)
    seeds, seed_face_ids = poisson_disk_on_surface(
        selection.submesh,
        args.min_spacing,
        seed=args.seed,
        spacing_metric=args.spacing_metric,
    )
    seeds, seed_face_ids = filter_seeds_by_borders(
        seeds,
        seed_face_ids,
        mesh.bounds,
        bottom_border=args.bottom_border,
        top_border=args.top_border,
        reach_margin=seed_reach_margin(args.min_spacing),
    )
    normals = selection.submesh.face_normals[seed_face_ids]

    if args.seed_count <= 0:
        cluster_indices = np.arange(len(seeds), dtype=np.int64)
    else:
        cluster_indices = select_cluster_seed_indices(seeds, count=args.seed_count)
    typical_spacing = characteristic_seed_spacing(seeds, min_spacing=args.min_spacing)
    max_neighbor = max_seed_neighbor_distance(seeds)
    search_radius = typical_spacing + args.margin

    cells = build_local_voronoi_cells(
        cluster_indices,
        seeds,
        normals,
        selection.submesh,
        margin=args.margin,
        perpendicular_half_length=args.perpendicular_half_length,
        min_spacing=args.min_spacing,
        search_radius=search_radius,
    )

    chord_segments: list[np.ndarray] = []
    full_perpendiculars: list[np.ndarray] = []
    clipped_perpendiculars: list[np.ndarray] = []
    boundary_loops: list[np.ndarray] = []

    for cell in cells:
        for bisector in cell.bisectors:
            chord_segments.append(bisector.chord_3d)
            full_perpendiculars.append(bisector.full_perpendicular_3d)
            if bisector.clipped_perpendicular_3d is not None:
                clipped_perpendiculars.append(bisector.clipped_perpendicular_3d)

        if cell.boundary_loop_3d is not None:
            boundary_loops.append(cell.boundary_loop_3d)

    print("\nLocal Voronoi bisector test")
    print(f"  seeds in cluster: {len(cluster_indices)}")
    print(f"  typical nearest-neighbor spacing: {typical_spacing:.2f} mm")
    print(f"  max nearest-neighbor spacing: {max_neighbor:.2f} mm")
    print(f"  partner search radius: {search_radius:.2f} mm")
    print(f"  perpendicular half-length: {args.perpendicular_half_length:.1f} mm")
    for cell in cells:
        print(
            f"  seed {cell.seed_index:3d}: "
            f"{len(cell.partner_indices)} partners, "
            f"{len(cell.bisectors)} bisectors, "
            f"boundary={'yes' if cell.boundary_loop_3d is not None else 'no'}"
        )

    ensure_gallery_dirs()
    run_number = next_run_number(shape, PATTERN)
    basename = f"{shape}.{PATTERN}.{run_number:03d}"

    markers = seed_marker_spheres(
        seeds[cluster_indices],
        normals[cluster_indices],
        radius=args.marker_radius,
    )

    partner_spheres = [
        partner_sphere_mesh(
            seeds[int(cell.seed_index)],
            cell.search_radius,
        )
        for cell in cells
    ]

    chords = line_polylines_to_mesh(chord_segments, radius=args.line_radius * 0.8)
    full_perps = line_polylines_to_mesh(full_perpendiculars, radius=args.line_radius)
    clipped_perps = line_polylines_to_mesh(
        clipped_perpendiculars,
        radius=args.line_radius * 1.1,
    )
    boundaries = line_polylines_to_mesh(boundary_loops, radius=args.line_radius * 1.3)

    parts = [
        selection.submesh,
        markers,
        _colored_mesh(trimesh.util.concatenate(partner_spheres), (180, 180, 180, 40)),
        _colored_mesh(chords, (220, 180, 40, 255)),
        _colored_mesh(full_perps, (40, 90, 220, 200)),
        _colored_mesh(clipped_perps, (30, 160, 220, 255)),
        _colored_mesh(boundaries, (30, 180, 70, 255)),
    ]
    combined = trimesh.util.concatenate([part for part in parts if len(part.vertices) > 0])
    stl_path = CREATIONS_DIR / f"{basename}.stl"
    combined.export(stl_path)

    scene = trimesh.Scene()
    scene.add_geometry(
        selection.submesh,
        geom_name="mesh",
    )
    scene.add_geometry(markers, geom_name="seeds")
    if partner_spheres:
        spheres = trimesh.util.concatenate(partner_spheres)
        spheres.visual.face_colors = (180, 180, 180, 40)
        scene.add_geometry(spheres, geom_name="partner_spheres")
    if len(chords.vertices):
        chords.visual.face_colors = (220, 180, 40, 255)
        scene.add_geometry(chords, geom_name="chords")
    if len(full_perps.vertices):
        full_perps.visual.face_colors = (40, 90, 220, 200)
        scene.add_geometry(full_perps, geom_name="full_perpendiculars")
    if len(clipped_perps.vertices):
        clipped_perps.visual.face_colors = (30, 160, 220, 255)
        scene.add_geometry(clipped_perps, geom_name="clipped_perpendiculars")
    if len(boundaries.vertices):
        boundaries.visual.face_colors = (30, 180, 70, 255)
        scene.add_geometry(boundaries, geom_name="cell_boundaries")

    glb_path = CREATIONS_DIR / f"{basename}.glb"
    scene.export(glb_path)

    print(f"\nExported test geometry: {stl_path}")
    print(f"Exported colored preview: {glb_path}")
    print("  yellow = seed-to-partner chords")
    print("  light blue = full perpendiculars")
    print("  cyan = clipped perpendicular edges")
    print("  green = closed cell boundary")

    if args.show:
        scene.show()


if __name__ == "__main__":
    main()
