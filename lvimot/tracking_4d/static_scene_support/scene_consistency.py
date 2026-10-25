"""Evaluation-only geometry consistency against the SAME archived LiDAR scans.

The reference is constructed with CARLA ground-truth ego poses AFTER LVIMOT
estimation, and must never be supplied to the estimator, mapper or tracker.
This measures point placement/coverage, NOT independent CARLA mesh accuracy.
"""
from __future__ import annotations

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[3]))


from typing import Dict, Iterable

import numpy as np
from scipy.spatial import cKDTree
from lvimot.output.evaluation.map_geometry import _reference_pose


def build_gt_posed_lidar_reference(frames: Iterable[Dict], calibration, last_gt_pose: Dict,
                                   local_radius_m: float = 60.0,
                                   voxel_m: float = 0.35) -> np.ndarray:
    origin = np.array([last_gt_pose['location'][k] for k in ('x', 'y', 'z')], dtype=float)
    extr = calibration.lidar_transform
    R_e_l, t_e_l = _reference_pose(extr)
    kept = []
    for item in frames:
        gt = item.get('ground_truth_ego_pose')
        if gt is None:
            continue
        lidar = np.asarray(item.get('lidar'), dtype=float)
        if lidar.ndim != 2 or lidar.shape[1] < 3:
            continue
        xyz = lidar[:, :3]
        good = np.isfinite(xyz).all(axis=1)
        if not np.any(good):
            continue
        R_w_e, t_w_e = _reference_pose(gt)
        world = ((xyz[good] @ R_e_l.T + t_e_l) @ R_w_e.T + t_w_e)
        nearby = np.linalg.norm(world[:, :2] - origin[:2], axis=1) <= local_radius_m
        if np.any(nearby):
            kept.append(world[nearby])
    if not kept:
        return np.empty((0, 3), dtype=np.float32)
    cloud = np.vstack(kept)
    voxels = np.floor(cloud / max(float(voxel_m), 0.05)).astype(np.int32)
    _, first = np.unique(voxels, axis=0, return_index=True)
    return cloud[first].astype(np.float32)


def evaluate_same_sensor_geometry(estimated_world_xyz, gt_posed_lidar_xyz,
                                  center_xyz, radius_m: float = 60.0) -> Dict:
    """Both directions are reported to expose poor coverage and extra points."""
    e = np.asarray(estimated_world_xyz, dtype=float).reshape(-1, 3)
    r = np.asarray(gt_posed_lidar_xyz, dtype=float).reshape(-1, 3)
    center = np.asarray(center_xyz, dtype=float).reshape(3)
    e = e[np.isfinite(e).all(axis=1)]
    r = r[np.isfinite(r).all(axis=1)]
    e = e[np.linalg.norm(e[:, :2]-center[:2], axis=1) <= radius_m]
    r = r[np.linalg.norm(r[:, :2]-center[:2], axis=1) <= radius_m]
    out = {
        'reference': 'same archived LiDAR positioned with CARLA ground-truth ego poses',
        'gt_used_for_estimation': False,
        'independent_carla_mesh_validation': False,
        'scene_geometry_certified': False,
        'estimated_points_evaluated': len(e),
        'reference_points_evaluated': len(r),
        'status': 'INSUFFICIENT_EVIDENCE',
        'limitations': [
            'Both maps originate from the same LiDAR observations; this is NOT independent scene-mesh ground truth.',
            'GT poses and calibration are applied only to the evaluation reference, AFTER sensor-only estimation.',
            'The reference can include dynamic objects excluded by the static LVIMOT mapper.',
            'Coverage depends on the 60 m rolling map radius and voxel density.',
        ],
    }
    if len(e) < 40 or len(r) < 40:
        return out
    a = cKDTree(r).query(e)[0]
    b = cKDTree(e).query(r)[0]
    out.update({
        'status': 'MEASURED_NOT_CERTIFIED',
        'estimate_to_reference_median_m': round(float(np.median(a)), 4),
        'estimate_to_reference_p95_m': round(float(np.percentile(a, 95)), 4),
        'estimate_within_0_5m_pct': round(100.0*float(np.mean(a <= 0.5)), 2),
        'estimate_within_1m_pct': round(100.0*float(np.mean(a <= 1.0)), 2),
        'reference_to_estimate_median_m': round(float(np.median(b)), 4),
        'reference_within_1m_pct': round(100.0*float(np.mean(b <= 1.0)), 2),
    })
    return out
