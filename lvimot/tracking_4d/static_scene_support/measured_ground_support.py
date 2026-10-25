"""Conservative horizontal ground candidates from estimated geometry only.

This layer has no road semantic claim. A flat sidewalk can satisfy the same
geometry. Reference road labels must only be used by a separate evaluator.
"""
from __future__ import annotations

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[3]))


import numpy as np
from scipy.spatial import cKDTree


def horizontal_ground_candidates(points, estimated_trajectory):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    path = np.asarray(estimated_trajectory, dtype=np.float64).reshape(-1, 3)
    if len(points) < 16 or not len(path):
        return np.zeros(len(points), dtype=bool), {'status': 'insufficient_geometry'}
    distance_to_path, nearest_pose = cKDTree(path[:, :2]).query(points[:, :2])
    height_above_pose = points[:, 2] - path[nearest_pose, 2]
    near_route = (distance_to_path <= 3.5) & np.isfinite(height_above_pose)
    if np.count_nonzero(near_route) < 30:
        return np.zeros(len(points), dtype=bool), {'status': 'insufficient_near_route_ground'}
    floor_offset = float(np.percentile(height_above_pose[near_route], 30))
    plausible = np.abs(height_above_pose - floor_offset) <= .22
    indices = np.flatnonzero(plausible)
    tree = cKDTree(points)
    result = np.zeros(len(points), dtype=bool)
    for start in range(0, len(indices), 8192):
        chosen = indices[start:start + 8192]
        distances, neighbors = tree.query(points[chosen], k=16, distance_upper_bound=1.25)
        finite = np.isfinite(distances)
        counts = finite.sum(axis=1)
        safe = np.minimum(neighbors, len(points) - 1)
        local = points[safe] - points[chosen, None, :]
        weights = finite.astype(float)
        mean = np.sum(local * weights[:, :, None], axis=1) / np.maximum(counts[:, None], 1)
        centered = (local - mean[:, None, :]) * weights[:, :, None]
        covariance = np.einsum('nki,nkj->nij', centered, centered) / np.maximum(counts[:, None, None], 1)
        values, vectors = np.linalg.eigh(covariance)
        normal_z = np.abs(vectors[:, 2, 0])
        planarity = values[:, 0] / np.maximum(values[:, 1], 1e-9)
        result[chosen] = (counts >= 7) & (normal_z >= np.cos(np.deg2rad(15))) & (planarity <= .12)
    return result, {
        'status': 'GEOMETRIC_GROUND_CANDIDATE_NOT_SEMANTIC_ROAD',
        'estimated_floor_offset_from_ego_m': floor_offset,
        'height_tolerance_m': .22, 'normal_max_slope_deg': 15,
        'neighborhood_radius_m': 1.25, 'minimum_neighbors': 7,
        'point_count': len(points), 'candidate_points': int(result.sum()),
        'ground_truth_or_reference_used': False,
        'limits': ['Flat sidewalks and other low horizontal surfaces can be included.',
                   'Large grade changes, banked roads and long-term trajectory drift are not validated.'],
    }
