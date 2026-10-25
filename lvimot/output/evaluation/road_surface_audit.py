"""Evaluation-only road-height proxy control for saved CARLA captures.

The road reference is independently supplied by CARLA OpenDRIVE, but the
LiDAR control uses the EXACT SAME sensor scans placed using held-out GT poses.
It establishes what the imperfect road-height test reports even with exact
poses. Neither this control nor its GT poses enter SLAM, tracking, or mapping.

This is NOT a road semantic segmentation score or independent full-3D mesh test.
"""
from __future__ import annotations

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[3]))


from typing import Dict
import numpy as np
from lvimot.output.evaluation.map_geometry import compare_road_geometry


def evaluate_road_height_proxy_control(
    estimated_world_xyz,
    gt_posed_lidar_xyz,
    carla_road_xyz,
    center_xyz,
    local_radius_m: float = 45.0,
) -> Dict:
    estimated = compare_road_geometry(
        estimated_world_xyz, carla_road_xyz, center_xyz,
        local_radius_m=local_radius_m,
    )
    control = compare_road_geometry(
        gt_posed_lidar_xyz, carla_road_xyz, center_xyz,
        local_radius_m=local_radius_m,
    )
    estimate_pct = estimated.get('road_agreement_pct')
    control_pct = control.get('road_agreement_pct')
    difference = None
    if estimate_pct is not None and control_pct is not None:
        if np.isfinite(float(estimate_pct)) and np.isfinite(float(control_pct)):
            difference = round(float(estimate_pct) - float(control_pct), 2)
    return {
        'status': 'DIAGNOSTIC_CONTROL_NOT_SEMANTIC_ROAD_CERTIFICATION',
        'uses_gt_for_estimation': False,
        'gt_used_only_for_evaluation_control': True,
        'reference': 'independent CARLA OpenDRIVE road surface; same archived LiDAR with GT poses is a control',
        'estimated_map_height_proxy': estimated,
        'gt_posed_same_sensor_height_proxy': control,
        'estimated_minus_gt_control_percentage_points': difference,
        'comparison_limitations': [
            'GT-posed LiDAR is not an independent scene mesh and is evaluation-only.',
            'A point near the road elevation can be sidewalk, curb, or building: neither score is semantic road precision.',
            'The static rolling map and GT-posed raw scans have different voxel sampling, coverage and dynamic-object filtering.',
            'A low proxy score even with exact GT poses exposes a metric limitation, not necessarily a localization fault.',
            'No automatic full-map certification or pass threshold can be inferred from this control.',
        ],
        'can_certify_full_3d_map': False,
    }
