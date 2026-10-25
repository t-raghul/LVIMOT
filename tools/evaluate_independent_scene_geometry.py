"""Score rolling and accumulated LVIMOT geometry against independent CARLA depth.

The primary map is placed with the first recorded pose only. A fitted rigid
alignment is an additional diagnostic and never changes the sensor map.
Reference semantics measure category coverage, not invented mapper labels.
"""
from __future__ import annotations

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[1]))


import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import sys as _lvimot_sys
from pathlib import Path as _lvimot_Path

from tools.validate_carla_depth_reference import _read_ply_xyz, _write_ply, _voxel_centroids, _compare, _rigid_icp, _matrix_from_pose


SEMANTIC_NAMES = {1: 'buildings', 2: 'fences', 5: 'poles', 6: 'road_lines', 7: 'roads',
                  8: 'sidewalks', 9: 'vegetation', 10: 'parked_vehicles', 11: 'walls',
                  12: 'traffic_signs', 14: 'ground', 15: 'bridges', 22: 'terrain'}


def category_coverage(estimated, reference, labels):
    distances = cKDTree(estimated).query(reference)[0]
    result = {}
    for value in np.unique(labels):
        selected = distances[labels == value]
        result[SEMANTIC_NAMES.get(int(value), 'semantic_%d' % value)] = {
            'reference_voxels': int(len(selected)),
            'median_nearest_map_distance_m': float(np.median(selected)),
            'within_0_5m_pct': float(100 * np.mean(selected <= .5)),
            'within_1m_pct': float(100 * np.mean(selected <= 1)),
            'within_2m_pct': float(100 * np.mean(selected <= 2)),
        }
    road = np.isin(labels, [6, 7])
    if road.any():
        xy = np.floor(reference[road, :2] / .5).astype(np.int64)
        covered = distances[road] <= 1
        result['sampled_road_xy_area'] = {
            'grid_cell_size_m': .5,
            'reference_observed_area_m2': float(len(np.unique(xy, axis=0)) * .25),
            'area_with_map_support_within_1m_m2': float(len(np.unique(xy[covered], axis=0)) * .25),
            'semantic_road_precision': None,
            'precision_not_available_reason': 'LVIMOT has no independently validated per-point semantic road classifier. This reports reference road coverage and geometry only.',
        }
    return result


def one_scope(estimate, reference, labels):
    e = _voxel_centroids(estimate, .25)
    initial = _compare(e, reference)
    result = {'initial_pose_placement': initial,
              'reference_semantic_coverage': category_coverage(e, reference, labels)}
    fitted, rotation, translation, used = _rigid_icp(e, reference)
    result['posthoc_rigid_fit_diagnostic_only'] = {
        'comparison': _compare(fitted, reference), 'matched_points_used': used,
        'row_vector_rotation': rotation.tolist(), 'row_vector_translation_m': translation.tolist(),
        'estimator_or_mapper_modified': False,
    }
    return result


def evaluate(capture, rolling_world, reference_directory, output,
             accumulated_local=None, protocol_path=None):
    manifest = json.loads((reference_directory / 'reference_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format') != 'CARLA_INDEPENDENT_RENDERED_360_REFERENCE_V1':
        raise ValueError('A full independent 360-degree reference is required')
    frames = [json.loads(line) for line in (capture / 'frames.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    first = frames[0]['ground_truth_ego_pose_evaluation_only']
    last = frames[-1]['ground_truth_ego_pose_evaluation_only']
    center = np.array([last['location'][k] for k in ('x', 'y', 'z')])
    gt_xyz = np.array([[row['ground_truth_ego_pose_evaluation_only']['location'][k] for k in ('x', 'y', 'z')] for row in frames])
    data = np.load(str(reference_directory / 'CARLA_independent_360_reference_semantics.npz'), allow_pickle=False)
    reference, labels = data['xyz'].astype(float), data['semantic']
    estimated = _read_ply_xyz(rolling_world)
    radius = 60.0
    eroi = np.linalg.norm(estimated - center, axis=1) <= radius
    rroi = np.linalg.norm(reference - center, axis=1) <= radius
    if min(int(eroi.sum()), int(rroi.sum())) < 40:
        raise RuntimeError('Insufficient common rolling-map geometry')
    output.mkdir(parents=True, exist_ok=True)
    rolling = one_scope(estimated[eroi], reference[rroi], labels[rroi])
    result = {
        'mode': 'INDEPENDENT_OBSERVED_SCENE_GEOMETRY_EVALUATION_ONLY',
        'reference_manifest': str((reference_directory / 'reference_manifest.json').resolve()),
        'reference_source': manifest['reference_source'], 'ground_truth_used_by_estimator': False,
        'alignment': 'First recorded pose placement only is primary. No fitted transform enters LVIMOT.',
        'rolling_scope': {'center_carla_xyz_m': center.tolist(), 'radius_3d_m': radius, 'metrics': rolling},
        'trajectory_observation': {'frames': len(frames),
                                   'gt_traversed_distance_m_evaluation_only': float(np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1).sum()),
                                   'gt_start_to_end_m_evaluation_only': float(np.linalg.norm(gt_xyz[-1] - gt_xyz[0]))},
        'limits': manifest['limits'],
    }
    if accumulated_local is not None:
        r, t = _matrix_from_pose(first)
        accumulated = _read_ply_xyz(accumulated_local) @ r.T + t
        _write_ply(output / 'LVIMOT_accumulated_CARLA_world_EVALUATION_ONLY.ply', accumulated)
        result['accumulated_scope'] = {'description': 'All integrated sensor geometry along this recorded run; no rolling eviction; no loop closure',
                                      'metrics': one_scope(accumulated, reference, labels),
                                      'mapped_bounds_xyz_m': [accumulated.min(axis=0).tolist(), accumulated.max(axis=0).tolist()]}
    if protocol_path is not None:
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        target = protocol['independent_scene_geometry_initial_pose']
        comparison = rolling['initial_pose_placement']
        a, b = comparison['estimate_to_independent_reference'], comparison['independent_reference_to_estimate']
        checks = {
            'estimate_within_1m': a['within_1m_pct'] >= target['estimate_within_1m_min_pct'],
            'reference_within_1m': b['within_1m_pct'] >= target['reference_within_1m_min_pct'],
            'estimate_median': a['median_m'] <= target['estimate_median_max_m'],
            'estimate_p95': a['p95_m'] <= target['estimate_p95_max_m'],
        }
        result['predeclared_local_geometry_gates'] = {'checks': checks, 'passed': all(checks.values()),
                                                     'protocol_file': str(protocol_path.resolve())}
    path = output / 'INDEPENDENT_SCENE_GEOMETRY_REPORT.json'
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('Rolling initial placement:', json.dumps(rolling['initial_pose_placement']))
    print('Saved independent geometry report:', path)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--rolling-world', type=Path, required=True)
    parser.add_argument('--reference-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--accumulated-local', type=Path)
    parser.add_argument('--protocol', type=Path)
    args = parser.parse_args()
    evaluate(args.capture, args.rolling_world, args.reference_directory, args.output,
             args.accumulated_local, args.protocol)
