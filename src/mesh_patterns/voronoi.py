"""
Voronoi partitions and boundary extraction on mesh surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import Voronoi, cKDTree


@dataclass(slots=True)
class SurfaceVoronoi:
    """Nearest-seed partition data for a mesh surface."""

    seeds: np.ndarray
    vertex_seed_ids: np.ndarray
    submesh: trimesh.Trimesh

    @property
    def cell_count(self) -> int:
        return len(self.seeds)


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _quantize(point: np.ndarray, decimals: int = 4) -> tuple[float, float, float]:
    return tuple(np.round(point, decimals=decimals))


def _segment_key(
    start: np.ndarray,
    end: np.ndarray,
    *,
    decimals: int = 4,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    first = _quantize(start, decimals=decimals)
    second = _quantize(end, decimals=decimals)
    return (first, second) if first <= second else (second, first)


def _bisector_value(seed_a: np.ndarray, seed_b: np.ndarray, point: np.ndarray) -> float:
    """
    Signed squared-distance difference between two seeds at a point.

    Zero on the perpendicular bisector plane of the seed pair.
    """

    return (
        np.dot(point - seed_a, point - seed_a)
        - np.dot(point - seed_b, point - seed_b)
    )


def _edge_bisector_point(
    start: np.ndarray,
    end: np.ndarray,
    seed_a: np.ndarray,
    seed_b: np.ndarray,
) -> np.ndarray | None:
    """
    Return the point on an edge where the bisector of two seeds crosses it.
    """

    direction = end - start
    value_start = _bisector_value(seed_a, seed_b, start)
    value_end = _bisector_value(seed_a, seed_b, end)

    if np.isclose(value_start, 0.0) and np.isclose(value_end, 0.0):
        return None

    if np.isclose(value_start, 0.0):
        return start
    if np.isclose(value_end, 0.0):
        return end

    if value_start * value_end > 0:
        return None

    parameter = value_start / (value_start - value_end)
    if parameter < -1e-9 or parameter > 1.0 + 1e-9:
        return None

    return start + parameter * direction


def _bisector_segment_in_triangle(
    vertices: np.ndarray,
    seed_a: np.ndarray,
    seed_b: np.ndarray,
) -> np.ndarray | None:
    """
    Intersect a seed-pair bisector plane with a triangle.
    """

    crossings: list[np.ndarray] = []
    for start, end in (
        (vertices[0], vertices[1]),
        (vertices[1], vertices[2]),
        (vertices[2], vertices[0]),
    ):
        point = _edge_bisector_point(start, end, seed_a, seed_b)
        if point is None:
            continue

        duplicate = any(np.linalg.norm(point - existing) < 1e-6 for existing in crossings)
        if not duplicate:
            crossings.append(point)

    if len(crossings) != 2:
        return None

    return np.asarray(crossings)


def vertex_voronoi(
    submesh: trimesh.Trimesh,
    seeds: np.ndarray,
) -> SurfaceVoronoi:
    """
    Assign each mesh vertex to its nearest seed in 3D Euclidean distance.
    """

    unique_vertices = np.unique(submesh.faces.reshape(-1))
    vertex_seed_ids = np.full(submesh.vertices.shape[0], -1, dtype=np.int64)
    _, nearest = cKDTree(seeds).query(submesh.vertices[unique_vertices])
    vertex_seed_ids[unique_vertices] = nearest

    return SurfaceVoronoi(
        seeds=seeds,
        vertex_seed_ids=vertex_seed_ids,
        submesh=submesh,
    )


def face_voronoi(
    submesh: trimesh.Trimesh,
    seeds: np.ndarray,
) -> SurfaceVoronoi:
    """
    Backwards-compatible alias for vertex-based nearest-seed assignment.
    """

    return vertex_voronoi(submesh, seeds)


def _triangle_plane(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    origin = vertices[0]
    normal = np.cross(vertices[1] - origin, vertices[2] - origin)
    normal_length = np.linalg.norm(normal)
    if normal_length <= 1e-12:
        return origin, np.array([0.0, 0.0, 1.0])
    return origin, normal / normal_length


def _point_in_triangle(
    point: np.ndarray,
    vertices: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> bool:
    origin, normal = _triangle_plane(vertices)
    projected = point - np.dot(point - origin, normal) * normal
    v0, v1, v2 = vertices
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = projected - v0
    dot00 = np.dot(v0v2, v0v2)
    dot01 = np.dot(v0v2, v0v1)
    dot02 = np.dot(v0v2, v0p)
    dot11 = np.dot(v0v1, v0v1)
    dot12 = np.dot(v0v1, v0p)
    denominator = dot00 * dot11 - dot01 * dot01
    if abs(denominator) <= 1e-12:
        return False

    inverse = 1.0 / denominator
    u = (dot11 * dot02 - dot01 * dot12) * inverse
    v = (dot00 * dot12 - dot01 * dot02) * inverse
    return (
        u >= -tolerance
        and v >= -tolerance
        and (u + v) <= 1.0 + tolerance
    )


def _triangle_voronoi_vertex(
    vertices: np.ndarray,
    seeds: np.ndarray,
) -> np.ndarray | None:
    """
    Solve for the point inside a triangle equidistant from three seeds.
    """

    origin, normal = _triangle_plane(vertices)
    seed_a, seed_b, seed_c = seeds
    matrix = np.vstack(
        [
            2.0 * (seed_b - seed_a),
            2.0 * (seed_c - seed_a),
            normal,
        ]
    )
    rhs = np.array(
        [
            np.dot(seed_b, seed_b) - np.dot(seed_a, seed_a),
            np.dot(seed_c, seed_c) - np.dot(seed_a, seed_a),
            np.dot(normal, origin),
        ]
    )

    try:
        vertex = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return None

    if not _point_in_triangle(vertex, vertices):
        return None

    return vertex


def _point_on_segment(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    tolerance: float = 0.05,
) -> bool:
    segment = end - start
    length = np.linalg.norm(segment)
    if length <= 1e-12:
        return np.linalg.norm(point - start) <= tolerance

    parameter = np.dot(point - start, segment) / (length * length)
    if parameter < -1e-6 or parameter > 1.0 + 1e-6:
        return False

    projection = start + parameter * segment
    return np.linalg.norm(point - projection) <= tolerance


def _split_segment_at_interior_vertices(
    segment: np.ndarray,
    interior_vertices: list[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Split a bisector chord at interior Voronoi junctions lying on it.
    """

    if len(interior_vertices) == 0:
        return [(segment[0], segment[1])]

    start, end = segment
    direction = end - start
    parameters = []
    for vertex in interior_vertices:
        denom = np.dot(direction, direction)
        if denom <= 1e-12:
            continue
        parameter = np.dot(vertex - start, direction) / denom
        if 1e-6 < parameter < 1.0 - 1e-6 and _point_on_segment(
            vertex,
            start,
            end,
            tolerance=0.08,
        ):
            parameters.append((parameter, vertex))

    if not parameters:
        return [(start, end)]

    parameters.sort(key=lambda item: item[0])
    ordered = [start] + [vertex for _t, vertex in parameters] + [end]
    return [
        (ordered[index], ordered[index + 1])
        for index in range(len(ordered) - 1)
        if np.linalg.norm(ordered[index] - ordered[index + 1]) > 1e-9
    ]


def _seed_pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _triangle_edge_index(
    vertex_ids: np.ndarray,
    edge: tuple[int, int],
) -> int | None:
    """
    Return the local triangle edge index for a mesh edge key.
    """

    ordered = _edge_key(int(edge[0]), int(edge[1]))
    for index, (left, right) in enumerate(((0, 1), (1, 2), (2, 0))):
        if _edge_key(int(vertex_ids[left]), int(vertex_ids[right])) == ordered:
            return index

    return None


def _crossing_on_triangle_edge(
    point: np.ndarray,
    vertices: np.ndarray,
    edge_index: int,
    *,
    tolerance: float = 1e-4,
) -> bool:
    left, right = ((0, 1), (1, 2), (2, 0))[edge_index]
    return _point_on_segment(
        point,
        vertices[left],
        vertices[right],
        tolerance=tolerance,
    )


def _triangle_seed_ball_radius(
    centroid: np.ndarray,
    seeds: np.ndarray,
    seed_tree: cKDTree,
) -> float:
    """
    Estimate the local Voronoi neighborhood radius for a triangle.
    """

    _, nearest_index = seed_tree.query(centroid, k=1)
    _, neighbor_distances = seed_tree.query(seeds[int(nearest_index)], k=2)
    spacing = float(neighbor_distances[1])
    return spacing * 2.25


def _voronoi_neighbor_pairs(seeds: np.ndarray) -> set[tuple[int, int]]:
    """
    Return seed pairs that share a 3D Voronoi face.
    """

    if len(seeds) < 2:
        return set()

    voronoi = Voronoi(seeds)
    neighbor_pairs: set[tuple[int, int]] = set()
    for left, right in voronoi.ridge_points:
        neighbor_pairs.add(_seed_pair_key(int(left), int(right)))

    return neighbor_pairs


def _triangle_relevant_seed_indices(
    vertices: np.ndarray,
    vertex_seed_ids: np.ndarray,
    seeds: np.ndarray,
    seed_tree: cKDTree,
) -> np.ndarray:
    """
    Return seeds whose Voronoi cells can meet inside a triangle.
    """

    centroid = vertices.mean(axis=0)
    radius = _triangle_seed_ball_radius(centroid, seeds, seed_tree)
    relevant = seed_tree.query_ball_point(centroid, radius)
    relevant = np.unique(
        np.concatenate(
            [
                np.asarray(relevant, dtype=np.int64),
                np.unique(vertex_seed_ids.astype(np.int64)),
            ]
        )
    )
    if len(relevant) > 16:
        distances = np.linalg.norm(seeds[relevant] - centroid, axis=1)
        nearest = np.argsort(distances)[:16]
        relevant = relevant[nearest]
    return relevant


def _triangle_relevant_seed_pairs(
    vertices: np.ndarray,
    vertex_seed_ids: np.ndarray,
    seeds: np.ndarray,
    seed_tree: cKDTree,
    neighbor_pairs: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    Return Voronoi-neighbor seed pairs that can cross a triangle.
    """

    relevant_seed_ids = _triangle_relevant_seed_indices(
        vertices,
        vertex_seed_ids,
        seeds,
        seed_tree,
    )
    if len(relevant_seed_ids) < 2:
        return []

    relevant_pairs: list[tuple[int, int]] = []
    for left_index, seed_a in enumerate(relevant_seed_ids):
        for seed_b in relevant_seed_ids[left_index + 1 :]:
            pair_key = _seed_pair_key(int(seed_a), int(seed_b))
            if pair_key in neighbor_pairs:
                relevant_pairs.append((int(seed_a), int(seed_b)))

    return relevant_pairs


def _triangle_relevant_seed_triples(
    relevant_seed_ids: np.ndarray,
    vertex_seed_ids: np.ndarray,
    neighbor_pairs: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    """
    Return likely triple junctions from Voronoi-neighbor seed triples.
    """

    candidate_triples: list[tuple[int, int, int]] = []
    unique_vertex_seeds = np.unique(vertex_seed_ids)
    if len(unique_vertex_seeds) == 3:
        candidate_triples.append(
            (
                int(unique_vertex_seeds[0]),
                int(unique_vertex_seeds[1]),
                int(unique_vertex_seeds[2]),
            )
        )

    if len(relevant_seed_ids) < 3:
        return candidate_triples

    for left in range(len(relevant_seed_ids)):
        for middle in range(left + 1, len(relevant_seed_ids)):
            for right in range(middle + 1, len(relevant_seed_ids)):
                seed_a = int(relevant_seed_ids[left])
                seed_b = int(relevant_seed_ids[middle])
                seed_c = int(relevant_seed_ids[right])
                if (
                    _seed_pair_key(seed_a, seed_b) in neighbor_pairs
                    and _seed_pair_key(seed_a, seed_c) in neighbor_pairs
                    and _seed_pair_key(seed_b, seed_c) in neighbor_pairs
                ):
                    candidate_triples.append((seed_a, seed_b, seed_c))

    return candidate_triples


def _mesh_edge_crossings(
    voronoi: SurfaceVoronoi,
) -> dict[tuple[tuple[int, int], tuple[int, int]], np.ndarray]:
    """
    Compute each mesh-edge / seed-pair Voronoi crossing once.

    A single mesh edge can host several different bisector crossings when
    adjacent triangles contain more than two Voronoi regions.
    """

    mesh = voronoi.submesh
    seeds = voronoi.seeds
    crossings: dict[tuple[tuple[int, int], tuple[int, int]], np.ndarray] = {}
    seed_tree = cKDTree(seeds)
    neighbor_pairs = _voronoi_neighbor_pairs(seeds)

    for face in mesh.faces:
        vertex_ids = face.astype(np.int64)
        vertices = mesh.vertices[vertex_ids]
        relevant_pairs = _triangle_relevant_seed_pairs(
            vertices,
            voronoi.vertex_seed_ids[vertex_ids],
            seeds,
            seed_tree,
            neighbor_pairs,
        )
        if not relevant_pairs:
            continue

        for seed_a_id, seed_b_id in relevant_pairs:
                chord = _bisector_segment_in_triangle(
                    vertices,
                    seeds[seed_a_id],
                    seeds[seed_b_id],
                )
                if chord is None:
                    continue

                pair_key = _seed_pair_key(seed_a_id, seed_b_id)
                for point in chord:
                    for edge_index in range(3):
                        if not _crossing_on_triangle_edge(
                            point,
                            vertices,
                            edge_index,
                        ):
                            continue

                        left, right = ((0, 1), (1, 2), (2, 0))[edge_index]
                        edge_key = _edge_key(
                            int(vertex_ids[left]),
                            int(vertex_ids[right]),
                        )
                        crossing_key = (edge_key, pair_key)
                        existing = crossings.get(crossing_key)
                        if existing is None:
                            crossings[crossing_key] = np.asarray(point, dtype=np.float64)
                        elif np.linalg.norm(existing - point) > 1e-6:
                            crossings[crossing_key] = 0.5 * (existing + point)

    return crossings


def _interior_voronoi_vertices(
    voronoi: SurfaceVoronoi,
) -> dict[tuple[int, int, int], np.ndarray]:
    """
    Compute triple-seed Voronoi junctions once per seed triple.
    """

    mesh = voronoi.submesh
    seeds = voronoi.seeds
    vertices_by_triple: dict[tuple[int, int, int], np.ndarray] = {}
    seed_tree = cKDTree(seeds)
    neighbor_pairs = _voronoi_neighbor_pairs(seeds)

    for face in mesh.faces:
        vertex_ids = face.astype(np.int64)
        vertices = mesh.vertices[vertex_ids]
        vertex_seed_ids = voronoi.vertex_seed_ids[vertex_ids]
        relevant_seed_ids = _triangle_relevant_seed_indices(
            vertices,
            vertex_seed_ids,
            seeds,
            seed_tree,
        )
        vertex_seed_ids = voronoi.vertex_seed_ids[vertex_ids]
        for seed_a, seed_b, seed_c in _triangle_relevant_seed_triples(
            relevant_seed_ids,
            vertex_seed_ids,
            neighbor_pairs,
        ):
            triple = np.asarray([seed_a, seed_b, seed_c], dtype=np.int64)
            triple_key = tuple(sorted(int(seed_id) for seed_id in triple))
            if triple_key in vertices_by_triple:
                continue

            interior_vertex = _triangle_voronoi_vertex(
                vertices,
                seeds[triple],
            )
            if interior_vertex is not None:
                vertices_by_triple[triple_key] = interior_vertex

    return vertices_by_triple


def _snap_to_triangle_crossings(
    point: np.ndarray,
    vertex_ids: np.ndarray,
    vertices: np.ndarray,
    edge_crossings: dict[tuple[tuple[int, int], tuple[int, int]], np.ndarray],
    seed_a: int,
    seed_b: int,
    *,
    tolerance: float = 0.08,
) -> np.ndarray:
    """
    Snap a bisector endpoint onto the shared crossing for its seed pair.
    """

    pair_key = _seed_pair_key(seed_a, seed_b)
    for left, right in ((0, 1), (1, 2), (2, 0)):
        edge_key = _edge_key(int(vertex_ids[left]), int(vertex_ids[right]))
        crossing = edge_crossings.get((edge_key, pair_key))
        if crossing is None:
            continue

        if np.linalg.norm(point - crossing) <= tolerance:
            return crossing

        edge_start = vertices[left]
        edge_end = vertices[right]
        if _point_on_segment(point, edge_start, edge_end, tolerance=tolerance):
            return crossing

    return point


def _triangle_voronoi_segments(
    vertex_ids: np.ndarray,
    vertices: np.ndarray,
    vertex_seed_ids: np.ndarray,
    seeds: np.ndarray,
    seed_tree: cKDTree,
    neighbor_pairs: set[tuple[int, int]],
    edge_crossings: dict[tuple[tuple[int, int], tuple[int, int]], np.ndarray],
    interior_vertices: dict[tuple[int, int, int], np.ndarray],
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """
    Build full bisector spans across a triangle, welded to shared edge crossings.
    """

    relevant_seed_ids = _triangle_relevant_seed_indices(
        vertices,
        vertex_seed_ids,
        seeds,
        seed_tree,
    )
    relevant_pairs = _triangle_relevant_seed_pairs(
        vertices,
        vertex_seed_ids,
        seeds,
        seed_tree,
        neighbor_pairs,
    )
    if not relevant_pairs:
        return []

    interior_vertex_list: list[np.ndarray] = []
    for seed_a, seed_b, seed_c in _triangle_relevant_seed_triples(
        relevant_seed_ids,
        vertex_seed_ids,
        neighbor_pairs,
    ):
        triple_key = tuple(sorted((seed_a, seed_b, seed_c)))
        interior_vertex = interior_vertices.get(triple_key)
        if interior_vertex is None:
            interior_vertex = _triangle_voronoi_vertex(
                vertices,
                seeds[np.asarray([seed_a, seed_b, seed_c], dtype=np.int64)],
            )
        if interior_vertex is None:
            continue

        duplicate = any(
            np.linalg.norm(interior_vertex - existing) < 1e-6
            for existing in interior_vertex_list
        )
        if not duplicate:
            interior_vertex_list.append(interior_vertex)

    segments: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for seed_a_id, seed_b_id in relevant_pairs:
            chord = _bisector_segment_in_triangle(
                vertices,
                seeds[seed_a_id],
                seeds[seed_b_id],
            )
            if chord is None:
                continue

            start = _snap_to_triangle_crossings(
                chord[0],
                vertex_ids,
                vertices,
                edge_crossings,
                seed_a_id,
                seed_b_id,
            )
            end = _snap_to_triangle_crossings(
                chord[1],
                vertex_ids,
                vertices,
                edge_crossings,
                seed_a_id,
                seed_b_id,
            )
            welded = np.asarray([start, end])
            for split_start, split_end in _split_segment_at_interior_vertices(
                welded,
                interior_vertex_list,
            ):
                segments.append(
                    (
                        seed_a_id,
                        seed_b_id,
                        np.asarray(split_start, dtype=np.float64),
                        np.asarray(split_end, dtype=np.float64),
                    )
                )

    return segments


def _collect_raw_bisector_segments(
    voronoi: SurfaceVoronoi,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """
    Collect Voronoi boundary segments using shared mesh-edge crossings.
    """

    mesh = voronoi.submesh
    seeds = voronoi.seeds
    edge_crossings = _mesh_edge_crossings(voronoi)
    interior_vertices = _interior_voronoi_vertices(voronoi)
    seed_tree = cKDTree(seeds)
    neighbor_pairs = _voronoi_neighbor_pairs(seeds)
    raw_segments: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    seen_segments: set[
        tuple[tuple[float, float, float], tuple[float, float, float]]
    ] = set()

    for face in mesh.faces:
        vertex_ids = face.astype(np.int64)
        vertices = mesh.vertices[vertex_ids]
        vertex_seed_ids = voronoi.vertex_seed_ids[vertex_ids]

        for seed_a_id, seed_b_id, start, end in _triangle_voronoi_segments(
            vertex_ids,
            vertices,
            vertex_seed_ids,
            seeds,
            seed_tree,
            neighbor_pairs,
            edge_crossings,
            interior_vertices,
        ):
            segment_key = _segment_key(start, end)
            if segment_key in seen_segments:
                continue
            seen_segments.add(segment_key)
            raw_segments.append((seed_a_id, seed_b_id, start, end))

    return raw_segments


def _canonical_junction_points(
    points: np.ndarray,
    *,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge nearby endpoints into shared Voronoi junction points.
    """

    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.int64)

    count = len(points)
    parents = np.arange(count, dtype=np.int64)

    def find(index: int) -> int:
        root = index
        while parents[root] != root:
            root = parents[root]
        while parents[index] != index:
            parent = parents[index]
            parents[index] = root
            index = parent
        return root

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parents[root_right] = root_left

    tree = cKDTree(points)
    for index, neighbors in enumerate(tree.query_ball_point(points, tolerance)):
        for neighbor in neighbors:
            if neighbor > index:
                union(index, int(neighbor))

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)

    canonical = np.empty((len(groups), 3), dtype=np.float64)
    mapping = np.empty(count, dtype=np.int64)
    for canonical_id, (_root, members) in enumerate(groups.items()):
        canonical[canonical_id] = points[members].mean(axis=0)
        for member in members:
            mapping[member] = canonical_id

    return canonical, mapping


def _trace_maximal_polylines(
    edges: list[tuple[int, int]],
) -> list[list[int]]:
    """
    Merge graph edges into maximal polylines with shared junction indices.
    """

    if not edges:
        return []

    adjacency: dict[int, list[int]] = {}
    edge_set: set[tuple[int, int]] = set()
    for start, end in edges:
        edge = _edge_key(start, end)
        edge_set.add(edge)
        adjacency.setdefault(start, []).append(end)
        adjacency.setdefault(end, []).append(start)

    used_edges: set[tuple[int, int]] = set()
    polylines: list[list[int]] = []

    for start, end in edges:
        edge = _edge_key(start, end)
        if edge in used_edges:
            continue

        used_edges.add(edge)
        chain = [start, end]

        previous, current = start, end
        while True:
            candidates = [
                candidate
                for candidate in adjacency.get(current, [])
                if candidate != previous
                and _edge_key(current, candidate) in edge_set
                and _edge_key(current, candidate) not in used_edges
            ]
            if len(candidates) != 1:
                break

            previous, current = current, candidates[0]
            used_edges.add(_edge_key(previous, current))
            chain.append(current)

        previous, current = end, start
        prefix: list[int] = []
        while True:
            candidates = [
                candidate
                for candidate in adjacency.get(current, [])
                if candidate != previous
                and _edge_key(current, candidate) in edge_set
                and _edge_key(current, candidate) not in used_edges
            ]
            if len(candidates) != 1:
                break

            previous, current = current, candidates[0]
            used_edges.add(_edge_key(previous, current))
            prefix.append(current)

        prefix.reverse()
        polylines.append(prefix + chain)

    return polylines


@dataclass(slots=True)
class VoronoiBoundaryGraph:
    """Connected Voronoi boundary edges referencing shared junction points."""

    junctions: np.ndarray
    edges: np.ndarray

    @property
    def edge_count(self) -> int:
        return len(self.edges)


def _refine_junction_points(
    junctions: np.ndarray,
    edges: list[tuple[int, int]],
    *,
    tolerance: float,
) -> VoronoiBoundaryGraph:
    """
    Collapse nearby junctions and remap edges onto the welded graph.
    """

    if len(junctions) == 0:
        return VoronoiBoundaryGraph(
            junctions=np.empty((0, 3), dtype=np.float64),
            edges=np.empty((0, 2), dtype=np.int64),
        )

    refined_junctions, remap = _canonical_junction_points(junctions, tolerance=tolerance)
    remapped_edges: list[tuple[int, int]] = []
    for start, end in edges:
        left = int(remap[start])
        right = int(remap[end])
        if left == right:
            continue
        remapped_edges.append((left, right))

    unique_edges = {
        _edge_key(start, end): (start, end) for start, end in remapped_edges
    }
    edge_array = np.asarray(list(unique_edges.values()), dtype=np.int64)
    return VoronoiBoundaryGraph(junctions=refined_junctions, edges=edge_array)


def _build_canonical_junctions(
    voronoi: SurfaceVoronoi,
    *,
    merge_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge all shared crossings and triple points into one junction set.

    Returns the welded junction coordinates, the source point cloud, and a
    remap from each source point to its welded junction index.
    """

    edge_crossings = _mesh_edge_crossings(voronoi)
    interior_vertices = _interior_voronoi_vertices(voronoi)
    points = list(edge_crossings.values()) + list(interior_vertices.values())
    if not points:
        empty = np.empty((0, 3), dtype=np.float64)
        return empty, empty, np.empty(0, dtype=np.int64)

    point_array = np.asarray(points, dtype=np.float64)
    junctions, remap = _canonical_junction_points(
        point_array,
        tolerance=merge_tolerance,
    )
    return junctions, point_array, remap


def _junction_index_for_point(
    point: np.ndarray,
    source_points: np.ndarray,
    remap: np.ndarray,
) -> int:
    matches = np.where(
        np.linalg.norm(source_points - point, axis=1) <= 1e-6
    )[0]
    if len(matches) == 0:
        raise ValueError("Voronoi segment endpoint is missing from canonical junctions.")
    return int(remap[int(matches[0])])


def voronoi_boundary_closure_stats(
    graph: VoronoiBoundaryGraph,
    polylines: list[np.ndarray],
) -> dict[str, int]:
    """
    Summarize how connected the welded Voronoi boundary graph is.
    """

    if len(graph.junctions) == 0:
        return {
            "junction_count": 0,
            "edge_count": 0,
            "dangling_junctions": 0,
            "junction_nodes": 0,
            "closed_polylines": 0,
            "open_polylines": 0,
        }

    degree = np.zeros(len(graph.junctions), dtype=np.int64)
    for start, end in graph.edges:
        degree[int(start)] += 1
        degree[int(end)] += 1

    closed_polylines = 0
    open_polylines = 0
    for polyline in polylines:
        if np.linalg.norm(polyline[0] - polyline[-1]) <= 0.05:
            closed_polylines += 1
        else:
            open_polylines += 1

    return {
        "junction_count": len(graph.junctions),
        "edge_count": graph.edge_count,
        "dangling_junctions": int(np.sum(degree == 1)),
        "junction_nodes": int(np.sum(degree >= 3)),
        "closed_polylines": closed_polylines,
        "open_polylines": open_polylines,
    }


def _bridge_nearby_junctions(
    graph: VoronoiBoundaryGraph,
    *,
    tolerance: float = 0.2,
) -> VoronoiBoundaryGraph:
    """
    Connect dangling junctions that should meet but float slightly apart.
    """

    if graph.edge_count == 0:
        return graph

    junctions = np.asarray(graph.junctions, dtype=np.float64)
    degree = np.zeros(len(junctions), dtype=np.int64)
    for start, end in graph.edges:
        degree[int(start)] += 1
        degree[int(end)] += 1

    dangling_indices = np.where(degree == 1)[0]
    if len(dangling_indices) == 0:
        return graph

    tree = cKDTree(junctions)
    extra_edges: list[tuple[int, int]] = []
    for left, right in tree.query_pairs(tolerance):
        if degree[left] == 1 and degree[right] >= 1:
            extra_edges.append((int(left), int(right)))
        elif degree[right] == 1 and degree[left] >= 1:
            extra_edges.append((int(left), int(right)))

    if not extra_edges:
        return graph

    merged_edges = {
        _edge_key(int(start), int(end)): (int(start), int(end))
        for start, end in graph.edges
    }
    for start, end in extra_edges:
        merged_edges[_edge_key(start, end)] = (start, end)
    edge_array = np.asarray(list(merged_edges.values()), dtype=np.int64)
    return VoronoiBoundaryGraph(junctions=junctions, edges=edge_array)


def euclidean_voronoi_boundary_graph(
    voronoi: SurfaceVoronoi,
    *,
    merge_tolerance: float = 0.12,
    bridge_tolerance: float = 0.2,
    raw_segments: list[tuple[int, int, np.ndarray, np.ndarray]] | None = None,
) -> VoronoiBoundaryGraph:
    """
    Build a welded Voronoi boundary graph with shared junction coordinates.
    """

    raw_segments = raw_segments or _collect_raw_bisector_segments(voronoi)
    if not raw_segments:
        return VoronoiBoundaryGraph(
            junctions=np.empty((0, 3), dtype=np.float64),
            edges=np.empty((0, 2), dtype=np.int64),
        )

    endpoints: list[np.ndarray] = []
    for _seed_a, _seed_b, start, end in raw_segments:
        endpoints.append(np.asarray(start, dtype=np.float64))
        endpoints.append(np.asarray(end, dtype=np.float64))

    endpoint_array = np.asarray(endpoints, dtype=np.float64)
    junctions, remap = _canonical_junction_points(
        endpoint_array,
        tolerance=merge_tolerance,
    )

    edges: set[tuple[int, int]] = set()
    for segment_index, (_seed_a, _seed_b, start, end) in enumerate(raw_segments):
        if np.linalg.norm(start - end) <= 1e-9:
            continue

        start_id = int(remap[segment_index * 2])
        end_id = int(remap[segment_index * 2 + 1])
        if start_id == end_id:
            continue
        edges.add(_edge_key(start_id, end_id))

    edge_array = np.asarray(list(edges), dtype=np.int64)
    graph = VoronoiBoundaryGraph(junctions=junctions, edges=edge_array)
    return _bridge_nearby_junctions(graph, tolerance=bridge_tolerance)


def euclidean_voronoi_boundary_polylines(
    voronoi: SurfaceVoronoi,
    *,
    merge_tolerance: float = 0.12,
) -> list[np.ndarray]:
    """
    Extract connected Voronoi boundary polylines on the mesh surface.
    """

    graph = euclidean_voronoi_boundary_graph(
        voronoi,
        merge_tolerance=merge_tolerance,
    )
    if graph.edge_count == 0:
        return []

    polyline_indices = _trace_maximal_polylines(
        [tuple(edge) for edge in graph.edges]
    )
    return [
        graph.junctions[np.asarray(indices)]
        for indices in polyline_indices
        if len(indices) >= 2
    ]


def euclidean_voronoi_boundary_segments(
    voronoi: SurfaceVoronoi,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Extract true 3D Euclidean Voronoi boundaries on a mesh surface.

    Each returned segment is one straight span of a connected boundary polyline.
    """

    mesh = voronoi.submesh
    polylines = euclidean_voronoi_boundary_polylines(voronoi)

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for polyline in polylines:
        for start, end in zip(polyline[:-1], polyline[1:]):
            midpoint = 0.5 * (start + end)
            _, _, face_id = mesh.nearest.on_surface([midpoint])
            normal = np.asarray(mesh.face_normals[int(face_id[0])], dtype=np.float64)
            normal = normal / np.linalg.norm(normal)
            segments.append((np.asarray([start, end]), normal))

    return segments


def tessellation_boundary_segments(
    voronoi: SurfaceVoronoi,
    *,
    include_border: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Extract exact Euclidean Voronoi boundaries on the mesh surface.
    """

    _ = include_border
    return euclidean_voronoi_boundary_segments(voronoi)


def cell_boundary_segments(voronoi: SurfaceVoronoi) -> list[np.ndarray]:
    """
    Extract open line segments that separate adjacent Voronoi cells.
    """

    return [
        segment
        for segment, _normal in tessellation_boundary_segments(voronoi)
    ]


def _trace_edge_loop(edges: set[tuple[int, int]]) -> list[int] | None:
    """
    Trace a closed vertex loop from a set of undirected edges.
    """

    if not edges:
        return None

    adjacency: dict[int, list[int]] = {}
    for v1, v2 in edges:
        adjacency.setdefault(v1, []).append(v2)
        adjacency.setdefault(v2, []).append(v1)

    start = next(iter(edges))[0]
    loop = [start]
    previous = start
    current = adjacency[start][0]
    visited: set[tuple[int, int]] = set()

    while True:
        visited.add(_edge_key(previous, current))
        loop.append(current)

        neighbors = [
            candidate
            for candidate in adjacency.get(current, [])
            if candidate != previous
            and _edge_key(current, candidate) in edges
            and _edge_key(current, candidate) not in visited
        ]

        if not neighbors:
            return None

        previous, current = current, neighbors[0]

        if current == start:
            return loop


def seed_cell_boundaries(
    voronoi: SurfaceVoronoi,
) -> dict[int, np.ndarray]:
    """
    Extract a boundary loop for each Voronoi cell using vertex ownership.
    """

    mesh = voronoi.submesh
    vertex_seeds = voronoi.vertex_seed_ids
    per_seed_edges: dict[int, set[tuple[int, int]]] = {
        seed_id: set() for seed_id in range(len(voronoi.seeds))
    }

    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_id, face in enumerate(mesh.faces):
        for v1, v2 in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_to_faces.setdefault(_edge_key(int(v1), int(v2)), []).append(face_id)

    for edge, faces in edge_to_faces.items():
        v1, v2 = edge
        seed_a = int(vertex_seeds[v1])
        seed_b = int(vertex_seeds[v2])
        if seed_a == seed_b:
            if len(faces) == 1:
                per_seed_edges[seed_a].add(edge)
            continue

        per_seed_edges[seed_a].add(edge)
        per_seed_edges[seed_b].add(edge)

    boundaries: dict[int, np.ndarray] = {}
    for seed_id, edges in per_seed_edges.items():
        loop = _trace_edge_loop(edges)
        if loop is None or len(loop) < 3:
            continue
        boundaries[seed_id] = mesh.vertices[np.asarray(loop)]

    return boundaries


def cell_boundary_loops(voronoi: SurfaceVoronoi) -> list[np.ndarray]:
    """
    Return connected Voronoi boundary polylines.
    """

    return euclidean_voronoi_boundary_polylines(voronoi)


def pebble_radii(
    seeds: np.ndarray,
    gap: float,
    *,
    fill: float = 0.42,
) -> np.ndarray:
    """
    Estimate pebble hole radii from seed spacing.
    """

    if len(seeds) == 0:
        return np.asarray([], dtype=np.float64)

    tree = cKDTree(seeds)
    nearest, _ = tree.query(seeds, k=2)
    spacing = nearest[:, 1]

    radii = spacing * fill - gap * 0.5
    return np.maximum(radii, gap)
