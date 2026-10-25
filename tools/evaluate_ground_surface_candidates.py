"""Evaluate sensor-only horizontal ground candidates against CARLA semantics."""
from __future__ import annotations

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[1]))


import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from lvimot.tracking_4d.static_scene_support.measured_ground_support import horizontal_ground_candidates
import sys as _lvimot_sys
from pathlib import Path as _lvimot_Path

from tools.validate_carla_depth_reference import _read_ply_xyz, _matrix_from_pose, _write_ply


def evaluate(capture, trace_directory, reference_directory):
    local = _read_ply_xyz(trace_directory / 'live_map_latest.ply')
    trace = [json.loads(row) for row in (trace_directory / 'frame_trace.jsonl').read_text().splitlines()]
    # Core poses are [roll, pitch, yaw, x, y, z].
    poses = np.array([row['sensor_only']['pose'][3:6] for row in trace])
    selected, definition = horizontal_ground_candidates(local, poses)
    # Everything above this line consumes estimated geometry only.
    first = trace[0]['evaluation_only']['carla_ego_pose']
    last = trace[-1]['evaluation_only']['carla_ego_pose']['location']
    center = np.array([last[k] for k in ('x', 'y', 'z')])
    rotation, translation = _matrix_from_pose(first)
    world = local[selected] @ rotation.T + translation
    world = world[np.linalg.norm(world - center, axis=1) <= 60]
    reference_data = np.load(str(reference_directory / 'CARLA_independent_360_reference_semantics.npz'))
    reference, labels = reference_data['xyz'], reference_data['semantic']
    roi = np.linalg.norm(reference - center, axis=1) <= 60
    reference, labels = reference[roi], labels[roi]
    if not len(world):
        raise RuntimeError('No horizontal ground candidate points')
    distance, nearest = cKDTree(reference).query(world)
    nearest_labels = labels[nearest]
    road = np.isin(labels, [6, 7])
    road_distance = cKDTree(world).query(reference[road])[0]
    by_threshold = {}
    for threshold in (.25, .5, 1.0, 2.0):
        hit = (distance <= threshold) & np.isin(nearest_labels, [6, 7])
        by_threshold[str(threshold)] = {
            'candidate_semantic_road_precision_pct': float(100 * np.mean(hit)),
            'reference_road_geometric_recall_pct': float(100 * np.mean(road_distance <= threshold)),
            'unmatched_or_nonroad_candidates_count': int(np.count_nonzero(~hit)),
        }
    result = {
        'mode': 'INDEPENDENT_SEMANTIC_AUDIT_OF_GEOMETRIC_GROUND_CANDIDATES',
        'classifier': definition, 'candidate_world_points_in_60m_scope': len(world),
        'alignment': 'First recorded pose placement after classification; no fitted alignment',
        'thresholds_m': by_threshold,
        'nearest_reference_label_counts': {str(int(v)): int(np.count_nonzero(nearest_labels == v)) for v in np.unique(nearest_labels)},
        'semantic_road_certified': False,
        'interpretation': 'Precision audits how often geometric ground candidates are actually CARLA roads/road lines. Sidewalks are false road labels here. They remain valid 3D geometry.',
        'ground_truth_used_for_classification': False,
        'limits': ['Finite reference ray sampling and semantic boundaries affect nearest-label attribution.',
                   'This does not add invented road semantics to the mapper or flatten structures.'],
    }
    folder = trace_directory / 'independent_geometry'
    folder.mkdir(exist_ok=True)
    _write_ply(folder / 'LVIMOT_horizontal_ground_candidates_EVALUATION_PLACEMENT.ply', world)
    (folder / 'GROUND_CANDIDATE_SEMANTIC_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(by_threshold))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--trace-directory', type=Path, required=True)
    parser.add_argument('--reference-directory', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.capture, args.trace_directory, args.reference_directory)
