"""
Visualize Poisson seed placement on a mesh surface.

Usage:
    python gallery/patterns/points.py gallery/shapes/CurvyLamp.stl
    python gallery/patterns/points.py gallery/shapes/CurvyLamp.stl --spacing-metric euclidean
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from mesh_patterns import PatternPipeline
from mesh_patterns.borders import filter_seeds_by_borders, seed_reach_margin
from mesh_patterns.export import seed_cloud_scene, seed_marker_spheres
from mesh_patterns.gallery_paths import CREATIONS_DIR, ensure_gallery_dirs, next_run_number
from mesh_patterns.pebble_shapes import _tangent_basis, _project_to_local
from mesh_patterns.sample import poisson_disk_on_surface, tangent_plane_distance

PATTERN = "points"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a seed-cloud debug view for mesh Poisson sampling.",
    )
    parser.add_argument(
        "mesh",
        type=Path,
        help="Path to the target mesh STL",
    )
    parser.add_argument(
        "--min-spacing",
        type=float,
        default=10.0,
        help="Minimum center-to-center spacing between seeds in mm",
    )
    parser.add_argument(
        "--spacing-metric",
        choices=("surface", "euclidean"),
        default="surface",
        help="Distance metric used during Poisson-disk acceptance",
    )
    parser.add_argument(
        "--bottom-border",
        type=float,
        default=5.0,
        help="Unpatterned margin in mm to leave at the base of the shape",
    )
    parser.add_argument(
        "--top-border",
        type=float,
        default=20.0,
        help="Unpatterned margin in mm to leave at the top of the shape",
    )
    parser.add_argument(
        "--marker-radius",
        type=float,
        default=1.5,
        help="Radius of seed marker spheres in mm",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for point scattering",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive preview window after export",
    )
    return parser.parse_args()


def _band_labels() -> tuple[list[float], list[str]]:
    return [5.0, 80.0, 120.0, 160.0, 200.0, 235.0], [
        "base",
        "lower",
        "mid",
        "upper",
        "neck",
    ]


def print_seed_report(
    seeds: np.ndarray,
    normals: np.ndarray,
    *,
    min_spacing: float,
    spacing_metric: str,
    surface_area: float,
) -> None:
    bins, labels = _band_labels()
    z_band = np.digitize(seeds[:, 2], bins) - 1
    radii = np.linalg.norm(seeds[:, :2], axis=1)

    tree = cKDTree(seeds)
    nn_3d = tree.query(seeds, k=2)[0][:, 1]

    nn_tangent = np.empty(len(seeds))
    for index, (seed, normal) in enumerate(zip(seeds, normals)):
        tangent_u, tangent_v = _tangent_basis(normal)
        local = _project_to_local(seeds, seed, tangent_u, tangent_v)
        distances = np.linalg.norm(local - local[index], axis=1)
        nn_tangent[index] = np.sort(distances[distances > 1e-6])[0]

    violations = 0
    for left in range(len(seeds)):
        for right in range(left + 1, len(seeds)):
            if spacing_metric == "surface":
                if tangent_plane_distance(seeds[left], seeds[right], normals[left]) < min_spacing:
                    violations += 1
                    continue
                if tangent_plane_distance(seeds[right], seeds[left], normals[right]) < min_spacing:
                    violations += 1
            else:
                if np.linalg.norm(seeds[left] - seeds[right]) < min_spacing:
                    violations += 1

    print("\nSeed cloud report")
    print(f"  spacing metric: {spacing_metric}")
    print(f"  target spacing: {min_spacing:.1f} mm")
    print(f"  seed count: {len(seeds):,}")
    print(f"  selected surface area: {surface_area:,.1f} mm^2")
    print(f"  spacing violations: {violations:,}")
    print(f"  3D nn distance: min={nn_3d.min():.2f} mean={nn_3d.mean():.2f} mm")
    print(
        "  tangent nn distance: "
        f"min={nn_tangent.min():.2f} mean={nn_tangent.mean():.2f} mm"
    )
    print("\n  band     seeds   radius   3d_nn   tan_nn")
    for band_index, label in enumerate(labels):
        mask = z_band == band_index
        if not mask.any():
            continue
        print(
            f"  {label:<8} {mask.sum():5d} "
            f"{radii[mask].mean():7.0f} "
            f"{nn_3d[mask].mean():7.2f} "
            f"{nn_tangent[mask].mean():7.2f}"
        )


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

    print_seed_report(
        seeds,
        normals,
        min_spacing=args.min_spacing,
        spacing_metric=args.spacing_metric,
        surface_area=selection.area,
    )

    ensure_gallery_dirs()
    run_number = next_run_number(shape, PATTERN)
    basename = f"{shape}.{PATTERN}.{run_number:03d}"

    markers = seed_marker_spheres(
        seeds,
        normals,
        radius=args.marker_radius,
    )
    combined = trimesh.util.concatenate([mesh, markers])
    stl_path = CREATIONS_DIR / f"{basename}.stl"
    combined.export(stl_path)

    glb_path = CREATIONS_DIR / f"{basename}.glb"
    scene = seed_cloud_scene(
        selection.submesh,
        seeds,
        normals,
        marker_radius=args.marker_radius,
    )
    scene.export(glb_path)

    csv_path = CREATIONS_DIR / f"{basename}.csv"
    np.savetxt(
        csv_path,
        np.column_stack([seeds, normals, seed_face_ids]),
        delimiter=",",
        header="x,y,z,nx,ny,nz,face_id",
        comments="",
    )

    print(f"\nExported mesh + markers: {stl_path}")
    print(f"Exported colored preview: {glb_path}")
    print(f"Exported seed table: {csv_path}")

    if args.show:
        scene.show()


if __name__ == "__main__":
    main()
