"""Replay an LVIMOT raw-sensor archive OFFLINE; does not connect to CARLA.

Usage (Windows, from CARLA_LVIMOT-main):
  .venv_live38\\Scripts\\python.exe tools\\replay_saved_capture.py --capture outputs_carla\\live\\sensor_recordings\\capture_YYYY...

For true reproduction of camera+LiDAR MOT, keep --detector camera_lidar.
Use --detector lidar for GPU-free localization/map debugging (MOT then differs).
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

ROOT = Path(__file__).resolve().parents[1]


from lvimot.inputs.sensor_calibration import CarlaCalibration
from lvimot.temporal_fusion.pipeline import CarlaLVIMOTCore
from lvimot.inputs.sensor_recording import iter_archive
from lvimot.output.map_export import export_live_map, write_binary_ply
from lvimot.output.live_dashboard import CarlaLiveVisualizer
from lvimot.output.evaluation.map_geometry import align_local_map_to_carla, compare_road_geometry, _svg_overlay
from lvimot.tracking_4d.static_scene_support.scene_consistency import build_gt_posed_lidar_reference, evaluate_same_sensor_geometry
from lvimot.output.evaluation.road_surface_audit import evaluate_road_height_proxy_control


def read_binary_ply_xyz(path):
    with Path(path).open('rb') as f:
        header = bytearray()
        while not header.endswith(b'end_header\n'):
            ch = f.read(1)
            if not ch or len(header) > 8192:
                raise ValueError('Invalid PLY header: ' + str(path))
            header.extend(ch)
        lines = header.decode('ascii').splitlines()
        count = next(int(line.split()[2]) for line in lines if line.startswith('element vertex '))
        if not any(line == 'format binary_little_endian 1.0' for line in lines):
            raise ValueError('Only binary little-endian PLY reference supported')
        vertices = np.frombuffer(f.read(count * 12), dtype='<f4').reshape(-1, 3)
        if len(vertices) != count:
            raise ValueError('PLY reference truncated')
        return vertices.copy()


def run_replay(capture, detector='camera_lidar', model='models/yolov8n.pt', device='auto', output=None):
    root = Path(capture).resolve()
    if not root.is_dir():
        raise FileNotFoundError('No archive folder: ' + str(root))
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    cfg = manifest.get('settings', {})
    calibration = CarlaCalibration(str(root / manifest['calibration']))
    out = Path(output).resolve() if output else root / 'offline_replay'
    out.mkdir(parents=True, exist_ok=True)
    core = CarlaLVIMOTCore(
        calibration,
        detector_mode=detector,
        yolo_model=model,
        yolo_device=device,
        yolo_confidence=float(cfg.get('yolo_confidence', 0.06)),
        yolo_image_size=int(cfg.get('yolo_image_size', 640)),
    )
    visualizer = CarlaLiveVisualizer()
    core.mapper_4d.enable_accumulated_map(out / 'accumulated_maps')
    total = 0
    first_gt = None
    last_gt = None
    last_pose = None
    evaluation_scans = []  # GT NEVER enters the sensor-only estimator.
    try:
        for item in iter_archive(root):
            gt = item['ground_truth_ego_pose']
            if gt is not None:
                if first_gt is None:
                    first_gt = gt
                last_gt = gt
            result = core.process(
                frame_id=int(item['frame']), timestamp=float(item['timestamp']),
                image=item['image'], lidar=item['lidar'], imu_data=item['imu'],
                ground_truth_ego_pose=gt,
                ground_truth_objects=item['ground_truth_objects'],
            )
            last_pose = visualizer._pose(result)
            if gt is not None:
                evaluation_scans.append({"ground_truth_ego_pose": gt, "lidar": item["lidar"]})
            total += 1
            if total % 30 == 0:
                print('Offline replay frames:', total, flush=True)
        if not total:
            raise RuntimeError('Capture archive is empty')
        export_data = export_live_map(core.mapper_4d, last_pose, visualizer, out)
        eval_summary = core.evaluation_summary() if first_gt is not None else None
        map_validation = None
        ref_path = root / 'CARLA_reference_road_surface.ply'
        if ref_path.is_file() and first_gt is not None and last_gt is not None:
            road = read_binary_ply_xyz(ref_path)
            world_pts = align_local_map_to_carla(core.mapper_4d.get_static_map_points(), first_gt)
            center = np.array([last_gt['location'][k] for k in 'xyz'], dtype=float)
            map_validation = compare_road_geometry(world_pts, road, center)
            map_validation['coordinate_alignment'] = 'first CARLA ego pose (evaluation only)'
            write_binary_ply(out/'replay_aligned_CARLA_frame_EVALUATION_ONLY.ply', world_pts)
            _svg_overlay(world_pts, road, center, out/'CARLA_vs_replayed_LVIMOT.svg')
        scene_consistency = None
        road_height_control = None
        if first_gt is not None and last_gt is not None and evaluation_scans:
            reference = build_gt_posed_lidar_reference(evaluation_scans, calibration, last_gt)
            # Alignment is evaluation-only and does not change the exported
            # raw sensor-estimated local map or any optimizer state.
            placed = align_local_map_to_carla(core.mapper_4d.get_static_map_points(), first_gt)
            center = np.array([last_gt['location'][k] for k in 'xyz'], dtype=float)
            scene_consistency = evaluate_same_sensor_geometry(placed, reference, center)
            if ref_path.is_file():
                road_height_control = evaluate_road_height_proxy_control(
                    placed, reference, read_binary_ply_xyz(ref_path), center
                )
                (out/'ROAD_HEIGHT_PROXY_GT_CONTROL_EVALUATION_ONLY.json').write_text(
                    json.dumps(road_height_control, indent=2)+'\n', encoding='utf-8')
            if len(reference):
                write_binary_ply(out/'GT_POSED_LIDAR_REFERENCE_EVALUATION_ONLY.ply', reference)
            (out/'SCENE_GEOMETRY_CONSISTENCY_EVALUATION_ONLY.json').write_text(
                json.dumps(scene_consistency, indent=2)+'\n', encoding='utf-8')
        loc = (eval_summary or {}).get('localization') or {}
        ate = float(loc.get('ate_rmse', float('inf')))
        rpe = float(loc.get('rpe_rmse', float('inf')))
        loc_pass = bool(np.isfinite(ate) and np.isfinite(rpe)
                        and ate <= 2.0 and rpe <= 0.5)
        # Road-height scoring is a mixed-scene proxy, not a semantic road
        # mask. Even near-perfect LiDAR placed with GT can score ~50%.
        # Do not turn a corrected trajectory into false map certification.
        summary = {
            'mode': 'OFFLINE_SENSOR_REPLAY_NO_CARLA', 'processed_frames': total,
            'detector_mode': detector, 'map_export': export_data,
            'evaluation': eval_summary, 'map_validation': map_validation,
            'same_sensor_scene_consistency_evaluation_only': scene_consistency,
            'road_height_proxy_gt_control_evaluation_only': road_height_control,
            'localization_only_gate_passed': loc_pass,
            'overall_verdict': 'MAP_AND_MOT_NOT_CERTIFIED',
            'gt_used_by_estimator': False,
        }
        (out/'replay_results.json').write_text(json.dumps(summary, indent=2, default=str)+'\n', encoding='utf-8')
        print('REPLAY COMPLETED: ' + str(out / 'replay_results.json'), flush=True)
        return summary
    finally:
        shutdown = getattr(core, 'close', None)
        if callable(shutdown):
            shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True)
    parser.add_argument('--detector', choices=['camera_lidar','lidar'], default='camera_lidar')
    parser.add_argument('--model', default='models/yolov8n.pt')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output')
    opts = parser.parse_args()
    run_replay(opts.capture, opts.detector, opts.model, opts.device, opts.output)
