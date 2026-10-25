"""Evaluation-only check: 2D-box LiDAR support versus true 3D actor-box hits.

A LiDAR return projected inside a vehicle's image box may hit a background
building. This audit uses held-out CARLA boxes only to diagnose that ambiguity.
It does not modify the detector, tracker, estimator, or existing MOT protocol.
"""

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

from lvimot.inputs.sensor_calibration import CarlaCalibration
from lvimot.inputs.sensor_recording import iter_archive
from lvimot.temporal_fusion.pipeline import CarlaLVIMOTCore
import sys as _lvimot_sys
from pathlib import Path as _lvimot_Path

from tools.validate_carla_depth_reference import _matrix_from_pose
from tools.analyze_recorded_trace import best


def audit(capture, trace_directory):
    cal = CarlaCalibration(str(capture / 'calibration.json'))
    r_el, t_el = _matrix_from_pose(cal.lidar_transform)
    trace = {r['frame']: r for r in [json.loads(line) for line in (trace_directory / 'frame_trace.jsonl').read_text().splitlines()]}
    rows = []
    for item in iter_archive(capture):
        eligible = trace[item['frame']]['evaluation_only']['eligible_gt']
        if not eligible or item['ground_truth_ego_pose'] is None:
            continue
        r_we, t_we = _matrix_from_pose(item['ground_truth_ego_pose'])
        world_points = (item['lidar'][:, :3] @ r_el.T + t_el) @ r_we.T + t_we
        uv, projection_valid = cal.project_lidar(item['lidar'][:, :3])
        depths = cal.lidar_to_opencv(item['lidar'][:, :3])[:, 2]
        actors = {int(obj['actor_id']): obj for obj in item['ground_truth_objects']}
        for obj in eligible:
            actor = actors[int(obj['actor_id'])]
            corners = np.asarray(actor['bounding_box']['world_corners'], dtype=float).reshape(-1, 3)
            center = corners.mean(axis=0)
            r_wo, _ = _matrix_from_pose({'location': {'x': 0, 'y': 0, 'z': 0}, 'rotation': actor['rotation']})
            local_corners = (corners - center) @ r_wo
            half = np.max(np.abs(local_corners), axis=0)
            local_points = (world_points - center) @ r_wo
            exact = np.all(np.abs(local_points) <= half + 0.02, axis=1)
            tolerant = np.all(np.abs(local_points) <= half + 0.15, axis=1)
            _, rgb = best(obj.get('bbox'), trace[item['frame']]['sensor_only']['camera_detections'])
            full_depths = central_depths = []
            if rgb is not None:
                x1, y1, x2, y2 = rgb['bbox']
                bw, bh = x2 - x1, y2 - y1
                full = projection_valid & (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
                central = projection_valid & (uv[:, 0] >= x1 + .08 * bw) & (uv[:, 0] <= x2 - .08 * bw) & (uv[:, 1] >= y1 + .08 * bh) & (uv[:, 1] <= y2 - .18 * bh)
                full_depths = np.round(np.sort(depths[full]), 3).tolist()
                central_depths = np.round(np.sort(depths[central]), 3).tolist()
            rows.append({
                'frame': item['frame'], 'actor_evaluation_only': int(obj['actor_id']),
                'image_box_support': int(obj['lidar_support']),
                'actual_actor_box_support_2cm_tolerance': int(exact.sum()),
                'actual_actor_box_support_15cm_tolerance': int(tolerant.sum()),
                'range_m': float(np.linalg.norm(center - t_we)),
                'rgb_full_box_depths_m': full_depths,
                'rgb_central_box_depths_m': central_depths,
            })
    result = {
        'mode': 'HELD_OUT_GT_GEOMETRIC_OBSERVABILITY_AUDIT_ONLY',
        'uses_gt_for_estimation': False, 'observations': rows,
        'original_image_box_eligible_observations': len(rows),
        'with_at_least_3_true_3d_returns': sum(r['actual_actor_box_support_15cm_tolerance'] >= 3 for r in rows),
        'with_at_least_4_true_3d_returns': sum(r['actual_actor_box_support_15cm_tolerance'] >= 4 for r in rows),
        'warning': 'Existing legacy MOTA is unchanged. Counting projected background returns as vehicle support overstates metric 3D observability.',
    }
    path = trace_directory / 'geometric_observability_audit.json'
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for row in rows:
        print(row['frame'], row['actor_evaluation_only'], 'image box', row['image_box_support'],
              'true 3D', row['actual_actor_box_support_15cm_tolerance'])
    print({k: v for k, v in result.items() if k != 'observations'})
    return result


def audit_unmatched_outputs(capture, trace_directory):
    calibration = CarlaCalibration(str(capture / 'calibration.json'))
    helper = CarlaLVIMOTCore(calibration, detector_mode='lidar')
    static_file = capture / 'CARLA_STATIC_VEHICLES_EVALUATION_ONLY.json'
    static_objects = json.loads(static_file.read_text(encoding='utf-8'))['vehicles'] if static_file.is_file() else []
    trace = {r['frame']: r for r in [json.loads(line) for line in (trace_directory / 'frame_trace.jsonl').read_text().splitlines()]}
    rows = []
    try:
        for item in iter_archive(capture):
            current = trace[item['frame']]['evaluation_only']
            if not current['scored_tracks']:
                continue
            all_objects = list(item['ground_truth_objects']) + list(static_objects)
            boxes = helper._project_gt_camera_boxes(all_objects, item['ground_truth_ego_pose'], item['image'].shape)
            all_regions = [{'actor_id': key, 'bbox': value} for key, value in boxes.items()]
            support = helper._gt_lidar_support_by_bbox(boxes, item['lidar'])
            all_lidar = helper._gt_objects_current_lidar(all_objects, item['ground_truth_ego_pose'])
            actors = {int(obj['actor_id']): obj for obj in all_lidar}
            for output in current['scored_tracks']:
                overlap, _ = best(output.get('bbox'), current['eligible_gt'])
                if overlap >= .25:
                    continue
                overlap, region = best(output.get('bbox'), all_regions)
                actor_id = int(region['actor_id']) if region is not None and overlap >= .25 else None
                position_error = None
                if actor_id is not None:
                    position_error = float(np.linalg.norm(np.asarray(output['position']) - np.asarray(actors[actor_id]['location'])))
                rows.append({
                    'frame': item['frame'], 'track_id': output['track_id'],
                    'matches_other_archived_carla_actor': actor_id,
                    'gt_image_iou_evaluation_only': float(overlap),
                    'gt_actor_image_box_lidar_support': support.get(actor_id) if actor_id is not None else None,
                    'position_error_to_actor_m_evaluation_only': position_error,
                    'camera_only_predicted': output.get('camera_only_predicted'),
                    'estimated_camera_support': output.get('camera_support'),
                    'bbox': output.get('bbox'),
                })
    finally:
        helper.close()
    result = {'mode': 'EVALUATION_ONLY_UNMATCHED_OUTPUT_AUDIT', 'rows': rows,
              'unmatched_outputs': len(rows),
              'matching_other_archived_actor_box': sum(row['matches_other_archived_carla_actor'] is not None for row in rows)}
    (trace_directory / 'unmatched_output_audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for row in rows:
        print(row)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--trace-directory', type=Path, required=True)
    args = parser.parse_args()
    audit(args.capture, args.trace_directory)
    audit_unmatched_outputs(args.capture, args.trace_directory)
