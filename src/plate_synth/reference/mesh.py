from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import GeometryDescription


@dataclass(frozen=True)
class ReferenceMesh:
    """First-order triangular mesh of one normalized plate geometry."""

    points: np.ndarray       # (N, 2)
    triangles: np.ndarray    # (T, 3), zero-based
    target_edge_length: float
    min_edge_length: float
    mean_edge_length: float
    max_edge_length: float
    target_area: float
    mesh_area: float
    relative_area_error: float
    boundary_hausdorff_approx: float

    @property
    def n_vertices(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_triangles(self) -> int:
        return int(self.triangles.shape[0])


def _edge_statistics(points: np.ndarray, triangles: np.ndarray) -> tuple[float, float, float]:
    edges = np.vstack(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        )
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    return float(np.min(lengths)), float(np.mean(lengths)), float(np.max(lengths))


def _boundary_edges(triangles: np.ndarray) -> np.ndarray:
    edges = np.vstack(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        )
    )
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    if boundary.size == 0:
        raise RuntimeError("mesh has no boundary edges")
    return np.ascontiguousarray(boundary, dtype=np.int64)


def _triangle_mesh_area(points: np.ndarray, triangles: np.ndarray) -> float:
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    twice = np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )
    return 0.5 * float(np.sum(twice))


def _point_to_segments_min_distance(
    query: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    """Distance from each query point to a piecewise-linear boundary."""
    q = np.asarray(query, dtype=np.float64)
    a = np.asarray(seg_a, dtype=np.float64)
    b = np.asarray(seg_b, dtype=np.float64)
    edge = b - a
    edge2 = np.sum(edge * edge, axis=1)
    result = np.empty(q.shape[0], dtype=np.float64)
    for start in range(0, q.shape[0], chunk_size):
        p = q[start : start + chunk_size]
        pa = p[:, None, :] - a[None, :, :]
        t = np.sum(pa * edge[None, :, :], axis=2) / np.maximum(edge2[None, :], 1e-30)
        t = np.clip(t, 0.0, 1.0)
        closest = a[None, :, :] + t[:, :, None] * edge[None, :, :]
        d2 = np.sum((p[:, None, :] - closest) ** 2, axis=2)
        result[start : start + p.shape[0]] = np.sqrt(np.min(d2, axis=1))
    return result


def _geometry_quality(
    points: np.ndarray,
    triangles: np.ndarray,
    target_boundary: np.ndarray,
    target_area: float,
) -> tuple[float, float, float]:
    """Return mesh area, relative area error and symmetric boundary distance.

    The Hausdorff value is a dense polyline approximation: all Phase-2 target
    boundary points are measured against the Gmsh boundary segments, and mesh
    boundary vertices plus edge midpoints are measured against the dense target
    polyline. This catches both contour downsampling and mesher geometry drift.
    """
    mesh_area = _triangle_mesh_area(points, triangles)
    rel_area = abs(mesh_area - target_area) / max(abs(target_area), 1e-30)

    boundary_edges = _boundary_edges(triangles)
    mesh_a = points[boundary_edges[:, 0]]
    mesh_b = points[boundary_edges[:, 1]]
    mesh_mid = 0.5 * (mesh_a + mesh_b)
    mesh_queries = np.vstack((mesh_a, mesh_b, mesh_mid))

    target = np.asarray(target_boundary, dtype=np.float64)
    target_a = target
    target_b = np.roll(target, -1, axis=0)

    target_to_mesh = _point_to_segments_min_distance(target, mesh_a, mesh_b)
    mesh_to_target = _point_to_segments_min_distance(mesh_queries, target_a, target_b)
    hausdorff = max(float(np.max(target_to_mesh)), float(np.max(mesh_to_target)))
    return mesh_area, rel_area, hausdorff


def _orient_triangles_ccw(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    tri = np.asarray(triangles, dtype=np.int64).copy()
    a = points[tri[:, 0]]
    b = points[tri[:, 1]]
    c = points[tri[:, 2]]
    signed_twice_area = (
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )
    flip = signed_twice_area < 0.0
    tri[flip, 1], tri[flip, 2] = tri[flip, 2].copy(), tri[flip, 1].copy()
    if np.any(np.abs(signed_twice_area) < 1e-14):
        raise RuntimeError("Gmsh produced a degenerate triangle")
    return tri


def _mesh_boundary_points(
    boundary: np.ndarray,
    target_edge_length: float,
) -> np.ndarray:
    """Downsample a dense Phase-2 contour for the mesh CAD boundary.

    Phase-2 may use 720+ samples for an accurate SDF, but feeding every sample
    to Gmsh would force hundreds of tiny boundary edges irrespective of the
    requested mesh size. We resample by arc length while explicitly retaining
    sharp corners detected from the dense polyline.
    """
    points = np.asarray(boundary, dtype=np.float64)
    next_points = np.roll(points, -1, axis=0)
    seg = next_points - points
    seg_len = np.linalg.norm(seg, axis=1)
    perimeter = float(np.sum(seg_len))
    if perimeter <= 0.0:
        raise ValueError("geometry boundary has zero perimeter")

    desired_spacing = 0.70 * float(target_edge_length)
    n_uniform = max(16, int(np.ceil(perimeter / desired_spacing)))
    cumulative = np.concatenate(([0.0], np.cumsum(seg_len)))
    targets = np.linspace(0.0, perimeter, n_uniform, endpoint=False)
    uniform_idx = np.searchsorted(cumulative, targets, side="right") - 1
    uniform_idx = np.clip(uniform_idx, 0, points.shape[0] - 1)

    prev_seg = points - np.roll(points, 1, axis=0)
    next_seg = np.roll(points, -1, axis=0) - points
    prev_norm = np.linalg.norm(prev_seg, axis=1)
    next_norm = np.linalg.norm(next_seg, axis=1)
    denom = np.maximum(prev_norm * next_norm, 1e-30)
    cos_turn = np.sum(prev_seg * next_seg, axis=1) / denom
    turn = np.arccos(np.clip(cos_turn, -1.0, 1.0))
    corner_idx = np.flatnonzero(turn > 0.15)

    indices = np.unique(np.concatenate((uniform_idx, corner_idx))).astype(np.int64)
    return np.ascontiguousarray(points[indices], dtype=np.float64)


def mesh_geometry(
    geometry: GeometryDescription,
    target_edge_length: float = 0.055,
    *,
    gmsh_verbosity: int = 0,
) -> ReferenceMesh:
    """Mesh a Phase-2 geometry using Gmsh and quantify approximation error."""
    if target_edge_length <= 0.0:
        raise ValueError("target_edge_length must be positive")

    try:
        import gmsh
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Phase 3 meshing requires the optional 'gmsh' package. "
            "Install requirements-phase3.txt."
        ) from exc

    dense_boundary = np.asarray(geometry.boundary_xy, dtype=np.float64)
    if dense_boundary.ndim != 2 or dense_boundary.shape[1] != 2 or dense_boundary.shape[0] < 3:
        raise ValueError("geometry boundary must have shape (N>=3, 2)")
    boundary = _mesh_boundary_points(dense_boundary, target_edge_length)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", float(gmsh_verbosity > 0))
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", float(target_edge_length))
        gmsh.option.setNumber("Mesh.MeshSizeMax", float(target_edge_length))
        gmsh.model.add("plate_reference")

        point_tags: list[int] = []
        for x, y in boundary:
            point_tags.append(
                gmsh.model.geo.addPoint(float(x), float(y), 0.0, float(target_edge_length))
            )

        line_tags: list[int] = []
        for i, start in enumerate(point_tags):
            end = point_tags[(i + 1) % len(point_tags)]
            line_tags.append(gmsh.model.geo.addLine(start, end))

        loop = gmsh.model.geo.addCurveLoop(line_tags)
        gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=np.int64)
        coords = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)[:, :2]

        element_types, _, element_nodes = gmsh.model.mesh.getElements(dim=2)
        triangle_tags: np.ndarray | None = None
        for element_type, connectivity in zip(element_types, element_nodes):
            _, dim, _, n_nodes, _, _ = gmsh.model.mesh.getElementProperties(element_type)
            if dim == 2 and n_nodes == 3:
                triangle_tags = np.asarray(connectivity, dtype=np.int64).reshape(-1, 3)
                break
        if triangle_tags is None or triangle_tags.size == 0:
            raise RuntimeError("Gmsh did not produce first-order triangles")

        sort_order = np.argsort(node_tags)
        sorted_tags = node_tags[sort_order]
        positions = np.searchsorted(sorted_tags, triangle_tags)
        if np.any(positions >= sorted_tags.size) or np.any(
            sorted_tags[np.minimum(positions, sorted_tags.size - 1)] != triangle_tags
        ):
            raise RuntimeError("Gmsh element connectivity references an unknown node")
        triangles = sort_order[positions]
        triangles = _orient_triangles_ccw(coords, triangles)

        h_min, h_mean, h_max = _edge_statistics(coords, triangles)
        mesh_area, rel_area, hausdorff = _geometry_quality(
            coords,
            triangles,
            dense_boundary,
            float(geometry.area),
        )
        return ReferenceMesh(
            points=np.ascontiguousarray(coords, dtype=np.float64),
            triangles=np.ascontiguousarray(triangles, dtype=np.int32),
            target_edge_length=float(target_edge_length),
            min_edge_length=h_min,
            mean_edge_length=h_mean,
            max_edge_length=h_max,
            target_area=float(geometry.area),
            mesh_area=float(mesh_area),
            relative_area_error=float(rel_area),
            boundary_hausdorff_approx=float(hausdorff),
        )
    finally:
        gmsh.finalize()
