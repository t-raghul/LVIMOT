"""Separate, explicitly defined whole-scene MOT evaluation of saved sensor tracks.

The original NPC-only MOTA remains unchanged in validation_summary.json. This
additional protocol includes CARLA's static parked vehicles and requires actual
3D actor-box LiDAR hits, rather than background hits inside a 2D rectangle.
Known actors without sufficient current measured depth define ignore regions;
only an RGB match plus a nearby 3D hypothesis can be ignored. No estimator runs.
"""
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

from lvimot.inputs.sensor_calibration import CarlaCalibration
from lvimot.temporal_fusion.pipeline import CarlaLVIMOTCore
from lvimot.inputs.sensor_recording import iter_archive
from tools.analyze_recorded_trace import iou
import sys as _lvimot_sys
from pathlib import Path as _lvimot_Path

from tools.validate_carla_depth_reference import _matrix_from_pose


def evaluate(capture: Path, trace_directory: Path, static_reference: Path):
    reference = json.loads(static_reference.read_text(encoding='utf-8'))
    if reference.get('format') != 'CARLA_STATIC_VEHICLE_EVALUATION_REFERENCE_V1':
        raise ValueError('Unexpected evaluation reference format')
    calibration = CarlaCalibration(str(capture / 'calibration.json'))
    helper = CarlaLVIMOTCore(calibration, detector_mode='lidar')
    trace = {r['frame']: r for r in [json.loads(line) for line in (trace_directory / 'frame_trace.jsonl').read_text().splitlines()]}
    r_el, t_el = _matrix_from_pose(calibration.lidar_transform)
    tracked, ground_truth, details = [], [], []
    ignored_count = 0
    try:
        for item in iter_archive(capture):
            objects = list(item['ground_truth_objects']) + list(reference['vehicles'])
            boxes = helper._project_gt_camera_boxes(objects, item['ground_truth_ego_pose'], item['image'].shape)
            current_gt = helper._gt_objects_current_lidar(objects, item['ground_truth_ego_pose'])
            raw_objects = {int(obj['actor_id']): obj for obj in objects}
            r_we, t_we = _matrix_from_pose(item['ground_truth_ego_pose'])
            points = item['lidar'][:, :3]
            world_points = (points @ r_el.T + t_el) @ r_we.T + t_we
            uv, valid = calibration.project_lidar(points)
            camera_depth = calibration.lidar_to_opencv(points)[:, 2]
            valid &= (camera_depth > .1) & (camera_depth <= calibration.lidar_range)
            eligible, ignore_regions = [], []
            for obj in current_gt:
                actor_id = int(obj['actor_id'])
                box = boxes.get(actor_id)
                if box is None:
                    continue
                width, height = box[2] - box[0], box[3] - box[1]
                if width < 8 or height < 8 or width * height < 100:
                    continue
                if np.linalg.norm(np.asarray(obj['location'])[:2]) > calibration.lidar_range:
                    continue
                original = raw_objects[actor_id]
                corners = np.asarray(original['bounding_box']['world_corners'], dtype=float).reshape(-1, 3)
                center = corners.mean(axis=0)
                rotation, _ = _matrix_from_pose({'location': {'x': 0, 'y': 0, 'z': 0}, 'rotation': original['rotation']})
                half = np.max(np.abs((corners - center) @ rotation), axis=0)
                relative = (world_points - center) @ rotation
                in_box = np.all(np.abs(relative) <= half + .15, axis=1)
                in_image = valid & (uv[:, 0] >= box[0]) & (uv[:, 0] <= box[2]) & (uv[:, 1] >= box[1]) & (uv[:, 1] <= box[3])
                support = int(np.count_nonzero(in_box & in_image))
                evaluated = dict(obj, bbox=list(box), actual_3d_lidar_support=support)
                if support >= 3:
                    eligible.append(evaluated)
                else:
                    ignore_regions.append(evaluated)
            state = trace[item['frame']]['sensor_only']
            outputs = helper._tracks_current_lidar(state['published_tracks'], np.asarray(state['pose'], dtype=float))
            retained, ignored = [], []
            for output in outputs:
                box = output.get('bbox')
                if box is None or box[2] - box[0] < 8 or box[3] - box[1] < 8 or (box[2] - box[0]) * (box[3] - box[1]) < 100:
                    continue
                matches_eligible = any(iou(box, gt['bbox']) >= .25 for gt in eligible)
                ignored_gt = None
                if not matches_eligible:
                    for gt in ignore_regions:
                        distance = float(np.linalg.norm(np.asarray(output['position']) - np.asarray(gt['location'])))
                        if iou(box, gt['bbox']) >= .25 and distance <= 5.0:
                            ignored_gt = gt
                            break
                if ignored_gt is None:
                    retained.append(output)
                else:
                    ignored.append({'track_id': output['track_id'], 'known_actor_id': ignored_gt['actor_id'],
                                    'actual_current_3d_returns': ignored_gt['actual_3d_lidar_support']})
            tracked.append(retained)
            ground_truth.append(eligible)
            ignored_count += len(ignored)
            details.append({'frame': item['frame'],
                            'eligible_actors': [{'actor_id': x['actor_id'], 'actual_3d_lidar_support': x['actual_3d_lidar_support']} for x in eligible],
                            'ignored_known_actor_outputs': ignored})
        metrics = helper.evaluator.evaluate_tracking_sequence(tracked, ground_truth)
        from lvimot.output.evaluation.metrics import CarlaEvaluator
        strict_metrics = CarlaEvaluator(require_3d_match=True).evaluate_tracking_sequence(tracked, ground_truth)
    finally:
        helper.close()
    result = {
        'protocol': 'FULL_SCENE_MEASURED_3D_SUPPORT_V1',
        'comparison_with_legacy': 'Separate protocol; compare baseline and modified tracks using this same protocol. Do not substitute it for historical NPC-only MOTA.',
        'ground_truth': 'Recorded CARLA NPC actors plus independently queried original Town10HD_Opt static vehicle environment objects',
        'static_reference': str(static_reference.resolve()),
        'minimum_actual_3d_returns_in_camera_view': 3,
        'actor_box_tolerance_m': .15, 'minimum_bbox_pixels': [8, 8, 100],
        'ignore_policy': 'Known actor image IoU >= 0.25 and 3D distance <= 5 m, but fewer than 3 current actor-box returns; all unrecognized outputs still count as false positives.',
        'gt_used_for_estimation': False, 'ignored_outputs': ignored_count,
        'tracking': metrics, 'frame_visibility': details,
        'strict_3d_tracking_diagnostic': {
            'definition': 'Same eligible objects/ignore policy, but image-box overlap cannot override the 4 m planar/2.5 m vertical spatial gates.',
            'tracking': strict_metrics,
            'legacy_or_predeclared_protocol_replaced': False,
        },
    }
    path = trace_directory / 'scene_tracking_evaluation.json'
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: metrics[key] for key in ('total_gt_objects', 'total_matches', 'false_positives', 'false_negatives', 'id_switches', 'mota', 'position_rmse', 'velocity_rmse')}))
    print('Ignored known-actor outputs:', ignored_count)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True, type=Path)
    parser.add_argument('--trace-directory', required=True, type=Path)
    parser.add_argument('--static-reference', required=True, type=Path)
    args = parser.parse_args()
    evaluate(args.capture, args.trace_directory, args.static_reference)
