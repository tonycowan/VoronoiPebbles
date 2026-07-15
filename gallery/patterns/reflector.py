"""
Generate reflector perforations on a mesh surface.

Square through-cuts are oriented so light from a point source on the lamp's
vertical axis (offset below the top) reflects off one face of each hole and
exits horizontally. Holes near the light height become horizontal square tunnels.

Usage:
    python gallery/patterns/reflector.py gallery/shapes/PlainLamp.stl
    python gallery/patterns/reflector.py gallery/shapes/PlainLamp.stl \\
        --min-spacing 8 --hole-size 1.5 --light-source-offset 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import trimesh

from mesh_patterns import ReflectorPipeline
from mesh_patterns.debug_artifacts import export_light_source_debug_artifact
from mesh_patterns.gallery_paths import next_run_number
from mesh_patterns.twinkle_cutters import light_source_position

PATTERN = "reflector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a reflector perforation pattern to a mesh.",
    )
    parser.add_argument(
        "mesh",
        type=Path,
        help="Path to the target mesh STL",
    )
    parser.add_argument(
        "--min-spacing",
        type=float,
        default=8.0,
        help="Minimum center-to-center spacing between holes in mm",
    )
    parser.add_argument(
        "--hole-size",
        type=float,
        default=1.5,
        help="Side length of each square bore in mm",
    )
    parser.add_argument(
        "--light-source-offset",
        type=float,
        default=30.0,
        help=(
            "Distance in mm below the top of the shade where the nominal light "
            "source sits on the central axis"
        ),
    )
    parser.add_argument(
        "--spacing-metric",
        choices=("surface", "euclidean"),
        default="euclidean",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for point scattering",
    )
    parser.add_argument(
        "--light-marker-diameter",
        type=float,
        default=40.0,
        help="Diameter in mm of the red light-source debug sphere",
    )
    parser.add_argument(
        "--add-light-debug",
        action="store_true",
        help="Also export a copy of the perforated mesh with a red light-source sphere",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_path = args.mesh.resolve()
    shape = mesh_path.stem

    mesh = trimesh.load(mesh_path, force="mesh")

    pipeline = ReflectorPipeline(
        min_spacing=args.min_spacing,
        hole_size=args.hole_size,
        light_source_offset=args.light_source_offset,
        bottom_border=args.bottom_border,
        top_border=args.top_border,
        seed=args.seed,
        spacing_metric=args.spacing_metric,
        outer_cut_margin=0.5,
    )

    result = pipeline.run(mesh)
    run_number = next_run_number(shape, PATTERN)

    print("Building reflector cutters...")
    perforated = pipeline.build_perforated_mesh(result)
    output_path = pipeline.export_creation(
        perforated,
        shape=shape,
        pattern=PATTERN,
        run_number=run_number,
    )

    if args.add_light_debug:
        light = light_source_position(
            mesh,
            light_source_offset=args.light_source_offset,
        )
        export_light_source_debug_artifact(
            perforated,
            light,
            shape,
            pattern=f"{PATTERN}_light",
            run_number=run_number,
            light_diameter=args.light_marker_diameter,
        )

    print(f"Loaded mesh: {mesh_path}")
    print(f"Bottom border: {args.bottom_border:.1f} mm")
    print(f"Top border: {args.top_border:.1f} mm")
    print(f"Selected surface faces: {result.selection.face_count:,}")
    print(f"Selected surface area: {result.selection.area:,.1f} mm^2")
    print(f"Seed points: {len(result.seeds):,}")
    print(f"Hole size: {args.hole_size:.2f} mm")
    print(f"Light source offset below top: {args.light_source_offset:.1f} mm")
    print(f"Perforated mesh faces: {len(perforated.faces):,}")
    print(f"Exported: {output_path}")


if __name__ == "__main__":
    main()
