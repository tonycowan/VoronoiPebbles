"""
Preview export helpers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .voronoi import VoronoiBoundaryGraph


def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normal / np.linalg.norm(normal)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, normal)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])

    tangent_u = np.cross(normal, helper)
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(normal, tangent_u)
    return tangent_u, tangent_v


def circle_polyline(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    segments: int = 24,
) -> np.ndarray:
    """
    Build a closed 3D polyline approximating a circle in a tangent plane.
    """

    tangent_u, tangent_v = _tangent_basis(normal)
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)

    points = (
        center
        + radius * np.cos(angles)[:, None] * tangent_u
        + radius * np.sin(angles)[:, None] * tangent_v
    )
    return np.vstack([points, points[:1]])


def nearest_normals(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
) -> np.ndarray:
    """
    Return outward-facing normals at the closest surface locations.
    """

    _, _, face_ids = mesh.nearest.on_surface(points)
    return mesh.face_normals[face_ids]


def loops_to_path3d(loops: list[np.ndarray]) -> trimesh.path.Path3D:
    """
    Convert closed polyline loops into a Path3D entity.
    """

    entities = []
    vertices: list[np.ndarray] = []
    offset = 0

    for loop in loops:
        if len(loop) < 2:
            continue

        entities.append(trimesh.path.entities.Line(np.arange(offset, offset + len(loop))))
        vertices.append(loop)
        offset += len(loop)

    if not vertices:
        return trimesh.path.Path3D()

    return trimesh.path.Path3D(
        entities=entities,
        vertices=np.vstack(vertices),
    )


def line_segments_to_mesh(
    segments: list[np.ndarray],
    *,
    radius: float = 0.2,
) -> trimesh.Trimesh:
    """
    Convert individual line segments into a thin tube mesh for STL preview.
    """

    meshes: list[trimesh.Trimesh] = []

    for segment in segments:
        segment = np.asarray(segment, dtype=np.float64)
        if segment.shape != (2, 3):
            continue

        start, end = segment
        if np.allclose(start, end):
            continue

        meshes.append(
            trimesh.creation.cylinder(
                radius=radius,
                segment=(start, end),
                sections=6,
            )
        )

    if not meshes:
        return trimesh.Trimesh()

    return trimesh.util.concatenate(meshes)


def line_loops_to_mesh(
    loops: list[np.ndarray],
    *,
    radius: float = 0.2,
) -> trimesh.Trimesh:
    """
    Convert polyline loops into a thin tube mesh for STL preview.
    """

    meshes: list[trimesh.Trimesh] = []

    for loop in loops:
        for start, end in zip(loop[:-1], loop[1:]):
            segment = trimesh.creation.cylinder(
                radius=radius,
                segment=(start, end),
                sections=6,
            )
            meshes.append(segment)

    if not meshes:
        return trimesh.Trimesh()

    return trimesh.util.concatenate(meshes)


def export_paths_as_stl(
    path: trimesh.path.Path3D,
    filename: str | Path,
    *,
    radius: float = 0.2,
) -> Path:
    """
    Export a Path3D object to STL via thin tube geometry.
    """

    loops: list[np.ndarray] = []
    for entity in path.entities:
        if isinstance(entity, trimesh.path.entities.Line):
            loops.append(path.vertices[entity.points])

    mesh = line_loops_to_mesh(loops, radius=radius)
    output = Path(filename)
    mesh.export(output)
    return output


def seed_marker_spheres(
    seeds: np.ndarray,
    normals: np.ndarray,
    *,
    radius: float = 1.5,
    lift: float = 0.2,
) -> trimesh.Trimesh:
    """
    Build small spheres marking seed locations on a mesh surface.
    """

    if len(seeds) == 0:
        return trimesh.Trimesh()

    unit_normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    centers = seeds + unit_normals * lift
    meshes = [
        trimesh.creation.icosphere(
            subdivisions=1,
            radius=radius,
        ).apply_translation(center)
        for center in centers
    ]
    return trimesh.util.concatenate(meshes)


def _lift_points_to_surface(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    *,
    lift: float,
) -> np.ndarray:
    """
    Lift points above the mesh using the local surface normal at each point.
    """

    if len(points) == 0:
        return points

    _, _, face_ids = mesh.nearest.on_surface(points)
    normals = mesh.face_normals[face_ids]
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    return points + normals * lift


def line_polylines_to_mesh(
    polylines: list[np.ndarray],
    *,
    radius: float = 0.2,
) -> trimesh.Trimesh:
    """
    Convert connected polyline spans into a thin tube mesh for STL preview.
    """

    meshes: list[trimesh.Trimesh] = []

    for polyline in polylines:
        polyline = np.asarray(polyline, dtype=np.float64)
        if len(polyline) < 2:
            continue

        for start, end in zip(polyline[:-1], polyline[1:]):
            if np.allclose(start, end):
                continue

            meshes.append(
                trimesh.creation.cylinder(
                    radius=radius,
                    segment=(start, end),
                    sections=6,
                )
            )

    if not meshes:
        return trimesh.Trimesh()

    return trimesh.util.concatenate(meshes)


def voronoi_boundary_mesh(
    graph: VoronoiBoundaryGraph,
    mesh: trimesh.Trimesh,
    *,
    tube_radius: float = 0.25,
    lift: float = 0.35,
    weld_junctions: bool = True,
) -> trimesh.Trimesh:
    """
    Build raised tube geometry for a welded Voronoi boundary graph.
    """

    if graph.edge_count == 0:
        return trimesh.Trimesh()

    lifted = _lift_points_to_surface(mesh, graph.junctions, lift=lift)
    meshes: list[trimesh.Trimesh] = []

    for start, end in graph.edges:
        segment = lifted[[int(start), int(end)]]
        if np.allclose(segment[0], segment[1]):
            continue

        meshes.append(
            trimesh.creation.cylinder(
                radius=tube_radius,
                segment=(segment[0], segment[1]),
                sections=6,
            )
        )

    if weld_junctions:
        degree = np.zeros(len(lifted), dtype=np.int64)
        for start, end in graph.edges:
            degree[int(start)] += 1
            degree[int(end)] += 1

        for index in np.where(degree >= 2)[0]:
            sphere = trimesh.creation.icosphere(
                subdivisions=1,
                radius=tube_radius * 1.15,
            )
            sphere.apply_translation(lifted[index])
            meshes.append(sphere)

    if not meshes:
        return trimesh.Trimesh()

    return trimesh.util.concatenate(meshes)


def voronoi_preview_scene(
    mesh: trimesh.Trimesh,
    seeds: np.ndarray,
    normals: np.ndarray,
    graph: VoronoiBoundaryGraph,
    *,
    marker_radius: float = 1.5,
    mesh_color: tuple[int, int, int, int] = (200, 200, 200, 120),
    seed_color: tuple[int, int, int, int] = (220, 40, 40, 255),
    voronoi_color: tuple[int, int, int, int] = (30, 90, 220, 255),
    tube_radius: float = 0.25,
    lift: float = 0.35,
) -> trimesh.Scene:
    """
    Build a colored scene with mesh, seeds, and Voronoi outlines.
    """

    scene = seed_cloud_scene(
        mesh,
        seeds,
        normals,
        marker_radius=marker_radius,
        mesh_color=mesh_color,
        seed_color=seed_color,
    )

    outlines = voronoi_boundary_mesh(
        graph,
        mesh,
        tube_radius=tube_radius,
        lift=lift,
    )
    outlines.visual.face_colors = voronoi_color
    scene.add_geometry(outlines, geom_name="voronoi")
    return scene


def voronoi_boundary_mesh_from_polylines(
    polylines: list[np.ndarray],
    mesh: trimesh.Trimesh,
    *,
    tube_radius: float = 0.25,
    lift: float = 0.35,
) -> trimesh.Trimesh:
    """
    Build raised tube geometry from explicit polylines.
    """

    if not polylines:
        return trimesh.Trimesh()

    unique_points: dict[tuple[float, float, float], np.ndarray] = {}
    point_order: list[tuple[float, float, float]] = []
    edges: list[tuple[int, int]] = []
    index_lookup: dict[tuple[float, float, float], int] = {}

    for polyline in polylines:
        indices: list[int] = []
        for point in polyline:
            key = tuple(np.round(point, decimals=6))
            if key not in unique_points:
                unique_points[key] = point
                point_order.append(key)
                index_lookup[key] = len(point_order) - 1
            indices.append(index_lookup[key])

        for start, end in zip(indices[:-1], indices[1:]):
            if start != end:
                edges.append((start, end))

    graph = VoronoiBoundaryGraph(
        junctions=np.asarray([unique_points[key] for key in point_order]),
        edges=np.asarray(edges, dtype=np.int64),
    )
    return voronoi_boundary_mesh(
        graph,
        mesh,
        tube_radius=tube_radius,
        lift=lift,
    )


def voronoi_preview_scene_from_polylines(
    mesh: trimesh.Trimesh,
    seeds: np.ndarray,
    normals: np.ndarray,
    polylines: list[np.ndarray],
    *,
    marker_radius: float = 1.5,
    mesh_color: tuple[int, int, int, int] = (200, 200, 200, 120),
    seed_color: tuple[int, int, int, int] = (220, 40, 40, 255),
    voronoi_color: tuple[int, int, int, int] = (30, 90, 220, 255),
    tube_radius: float = 0.25,
    lift: float = 0.35,
) -> trimesh.Scene:
    """
    Build a colored scene from explicit Voronoi polylines.
    """

    scene = seed_cloud_scene(
        mesh,
        seeds,
        normals,
        marker_radius=marker_radius,
        mesh_color=mesh_color,
        seed_color=seed_color,
    )

    outlines = voronoi_boundary_mesh_from_polylines(
        polylines,
        mesh,
        tube_radius=tube_radius,
        lift=lift,
    )
    outlines.visual.face_colors = voronoi_color
    scene.add_geometry(outlines, geom_name="voronoi")
    return scene


def voronoi_boundary_mesh_legacy(
    segments: list[tuple[np.ndarray, np.ndarray]],
    *,
    tube_radius: float = 0.25,
    lift: float = 0.35,
) -> trimesh.Trimesh:
    """
    Backwards-compatible segment-based Voronoi mesh export.
    """

    _ = lift
    lifted_segments: list[np.ndarray] = []
    for endpoints, normal in segments:
        points = np.asarray(endpoints, dtype=np.float64)
        if points.shape != (2, 3):
            continue
        offset = normal / np.linalg.norm(normal) * lift
        lifted_segments.append(points + offset)

    return line_segments_to_mesh(lifted_segments, radius=tube_radius)


def voronoi_preview_scene_from_segments(
    mesh: trimesh.Trimesh,
    seeds: np.ndarray,
    normals: np.ndarray,
    segments: list[tuple[np.ndarray, np.ndarray]],
    *,
    marker_radius: float = 1.5,
    mesh_color: tuple[int, int, int, int] = (200, 200, 200, 120),
    seed_color: tuple[int, int, int, int] = (220, 40, 40, 255),
    voronoi_color: tuple[int, int, int, int] = (30, 90, 220, 255),
    tube_radius: float = 0.25,
    lift: float = 0.35,
) -> trimesh.Scene:
    """
    Backwards-compatible segment-based Voronoi scene export.
    """

    scene = seed_cloud_scene(
        mesh,
        seeds,
        normals,
        marker_radius=marker_radius,
        mesh_color=mesh_color,
        seed_color=seed_color,
    )

    outlines = voronoi_boundary_mesh_legacy(
        segments,
        tube_radius=tube_radius,
        lift=lift,
    )
    outlines.visual.face_colors = voronoi_color
    scene.add_geometry(outlines, geom_name="voronoi")
    return scene


def seed_cloud_scene(
    mesh: trimesh.Trimesh,
    seeds: np.ndarray,
    normals: np.ndarray,
    *,
    marker_radius: float = 1.5,
    mesh_color: tuple[int, int, int, int] = (200, 200, 200, 120),
    seed_color: tuple[int, int, int, int] = (220, 40, 40, 255),
) -> trimesh.Scene:
    """
    Build a colored scene for inspecting seed placement on a mesh.
    """

    surface = mesh.copy()
    surface.visual.face_colors = mesh_color

    markers = seed_marker_spheres(
        seeds,
        normals,
        radius=marker_radius,
    )
    markers.visual.face_colors = seed_color

    scene = trimesh.Scene()
    scene.add_geometry(surface, geom_name="surface")
    scene.add_geometry(markers, geom_name="seeds")
    return scene


def _segment_max_off_surface(
    mesh: trimesh.Trimesh,
    start: np.ndarray,
    end: np.ndarray,
    *,
    samples: int = 5,
) -> float:
    """
    Maximum distance between a straight chord and the mesh surface.
    """

    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    points = start + np.linspace(0.0, 1.0, samples)[:, None] * (end - start)
    nearest, distances, _face_ids = mesh.nearest.on_surface(points)
    return float(np.max(np.linalg.norm(points - nearest, axis=1)))


def _segment_crosses_hollow_interior(
    start: np.ndarray,
    end: np.ndarray,
    *,
    radial_margin: float = 3.0,
) -> bool:
    """
    Detect chords that cut toward the central void of a lampshade-like mesh.
    """

    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    midpoint = 0.5 * (start + end)

    start_radius = float(np.linalg.norm(start[:2]))
    end_radius = float(np.linalg.norm(end[:2]))
    mid_radius = float(np.linalg.norm(midpoint[:2]))

    return mid_radius + radial_margin < min(start_radius, end_radius)


def _surface_polyline_between(
    mesh: trimesh.Trimesh,
    start: np.ndarray,
    end: np.ndarray,
    *,
    samples: int = 8,
) -> np.ndarray:
    """
    Sample a path between two points and snap each sample to the mesh surface.
    """

    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    points = start + np.linspace(0.0, 1.0, samples)[:, None] * (end - start)
    nearest, _distances, _face_ids = mesh.nearest.on_surface(points)
    return np.asarray(nearest, dtype=np.float64)


def surface_resample_loop(
    loop: np.ndarray,
    mesh: trimesh.Trimesh,
    *,
    samples_per_edge: int = 8,
    max_straight_deviation: float = 1.5,
) -> np.ndarray:
    """
    Replace loop chords with surface-following polylines where needed.
    """

    loop = np.asarray(loop, dtype=np.float64)
    if len(loop) < 2:
        return loop

    closed = np.allclose(loop[0], loop[-1])
    vertices = loop[:-1] if closed else loop
    if len(vertices) < 2:
        return loop

    dense_parts: list[np.ndarray] = []
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        if np.allclose(start, end):
            continue

        deviation = _segment_max_off_surface(mesh, start, end)
        crosses_interior = _segment_crosses_hollow_interior(start, end)
        if deviation > max_straight_deviation or crosses_interior:
            edge = _surface_polyline_between(
                mesh,
                start,
                end,
                samples=samples_per_edge,
            )
        else:
            edge = _surface_polyline_between(
                mesh,
                start,
                end,
                samples=max(3, samples_per_edge // 2),
            )

        if dense_parts:
            edge = edge[1:]
        dense_parts.append(edge)

    if not dense_parts:
        return loop

    result = np.vstack(dense_parts)
    if closed:
        result = np.vstack([result, result[0:1]])
    return result


def shrunk_boundary_mesh(
    loops: list[np.ndarray],
    mesh: trimesh.Trimesh,
    *,
    tube_radius: float = 0.25,
    lift: float = 0.45,
    follow_surface: bool = True,
    samples_per_edge: int = 8,
) -> trimesh.Trimesh:
    """
    Build raised tube geometry for closed shrunk-cell boundary loops.
    """

    if not loops:
        return trimesh.Trimesh()

    lifted_loops: list[np.ndarray] = []
    for loop in loops:
        if follow_surface:
            loop = surface_resample_loop(
                loop,
                mesh,
                samples_per_edge=samples_per_edge,
            )
        lifted = _lift_points_to_surface(mesh, loop, lift=lift)
        lifted_loops.append(lifted)

    return line_polylines_to_mesh(lifted_loops, radius=tube_radius)


def shrunk_preview_scene(
    mesh: trimesh.Trimesh,
    seeds: np.ndarray,
    normals: np.ndarray,
    loops: list[np.ndarray],
    *,
    outer_graph: VoronoiBoundaryGraph | None = None,
    outer_loops: list[np.ndarray] | None = None,
    marker_radius: float = 1.5,
    mesh_color: tuple[int, int, int, int] = (200, 200, 200, 120),
    seed_color: tuple[int, int, int, int] = (220, 40, 40, 255),
    outer_color: tuple[int, int, int, int] = (120, 120, 220, 180),
    shrunk_color: tuple[int, int, int, int] = (30, 170, 70, 255),
    outer_line_radius: float = 0.18,
    shrunk_line_radius: float = 0.28,
    outer_lift: float = 0.30,
    shrunk_lift: float = 0.45,
) -> trimesh.Scene:
    """
    Build a scene with seeds, optional outer Voronoi, and shrunk boundaries.
    """

    scene = seed_cloud_scene(
        mesh,
        seeds,
        normals,
        marker_radius=marker_radius,
        mesh_color=mesh_color,
        seed_color=seed_color,
    )

    if outer_loops:
        outer = shrunk_boundary_mesh(
            outer_loops,
            mesh,
            tube_radius=outer_line_radius,
            lift=outer_lift,
        )
        outer.visual.face_colors = outer_color
        scene.add_geometry(outer, geom_name="voronoi")
    elif outer_graph is not None and outer_graph.edge_count > 0:
        outer = voronoi_boundary_mesh(
            outer_graph,
            mesh,
            tube_radius=outer_line_radius,
            lift=outer_lift,
            weld_junctions=True,
        )
        outer.visual.face_colors = outer_color
        scene.add_geometry(outer, geom_name="voronoi")

    shrunk = shrunk_boundary_mesh(
        loops,
        mesh,
        tube_radius=shrunk_line_radius,
        lift=shrunk_lift,
    )
    shrunk.visual.face_colors = shrunk_color
    scene.add_geometry(shrunk, geom_name="shrunk")
    return scene


def pebble_preview_paths(
    mesh: trimesh.Trimesh,
    seeds: np.ndarray,
    radii: np.ndarray,
    *,
    seed_face_ids: np.ndarray | None = None,
    segments: int = 24,
) -> trimesh.path.Path3D:
    """
    Build circular hole previews on a mesh surface.
    """

    if seed_face_ids is None:
        normals = nearest_normals(mesh, seeds)
    else:
        normals = mesh.face_normals[seed_face_ids]

    loops = [
        circle_polyline(center, normal, radius, segments=segments)
        for center, normal, radius in zip(seeds, normals, radii)
    ]
    return loops_to_path3d(loops)
