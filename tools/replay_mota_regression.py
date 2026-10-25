"""Offline sensor replay with stage evidence, source hashes and optional regression gates.

Use --project to replay an unmodified baseline project, then rerun normally with
--reference BASELINE/summary.json. --no-gt exercises the evaluation boundary.
--disable-traffic-context is a causal ablation, not an original-ZIP baseline.
Output must be outside the read-only capture. This tool never connects to CARLA.
"""
from __future__ import annotations

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[1]))


import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]



def clean(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--disable-tiles', action='store_true')
    parser.add_argument('--disable-traffic-context', action='store_true')
    parser.add_argument('--project', help='Alternative baseline project root; defaults to this project')
    parser.add_argument('--minimum-mota', type=float, help='Fail unless original-protocol MOTA is at least this fraction')
    parser.add_argument('--no-gt', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--reference', help='Baseline summary.json from the identical capture')
    args = parser.parse_args()
    project = Path(args.project).resolve() if args.project else ROOT

    capture, out = Path(args.capture).resolve(), Path(args.output).resolve()
    if out == capture or capture in out.parents:
        parser.error('Choose an output directory outside the original capture.')
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    if args.minimum_mota is not None and (args.no_gt or not 0 <= args.minimum_mota <= 1):
        parser.error('--minimum-mota requires GT evaluation and a value from 0 to 1')
    manifest_path = capture / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    settings = manifest.get('settings', {})
    model_path = project / settings.get('yolo_model', 'models/yolov8n.pt')
    if not model_path.is_file():
        raise FileNotFoundError('Local model required; automatic downloads are disabled: ' + str(model_path))
    out.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(out / 'yolo_config')
    import numpy as np
    import cv2
    import torch
    np.random.seed(42)
    cv2.setRNGSeed(42)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    sys.path.insert(0, str(project if (project / 'lvimot').is_dir() else project / 'src'))
    if (project / 'lvimot').is_dir():
        from lvimot.inputs.sensor_calibration import CarlaCalibration
    else:
        from carla_calibration import CarlaCalibration
    if (project / 'lvimot').is_dir():
        from lvimot.temporal_fusion.pipeline import CarlaLVIMOTCore
    else:
        from carla_pipeline_core import CarlaLVIMOTCore
    if (project / 'lvimot').is_dir():
        from lvimot.inputs.sensor_recording import iter_archive
    else:
        from carla_raw_sensor_archive import iter_archive
    core = CarlaLVIMOTCore(
        CarlaCalibration(str(capture / manifest['calibration'])),
        detector_mode='camera_lidar', yolo_model=str(model_path),
        yolo_confidence=float(settings.get('yolo_confidence', 0.08)),
        yolo_image_size=int(settings.get('yolo_image_size', 640)), yolo_device='cpu')
    if args.disable_tiles:
        core.camera_lidar_detector.config = replace(core.camera_lidar_detector.config, small_object_tiles=False)
    if args.disable_traffic_context:
        core.camera_lidar_detector.config = replace(core.camera_lidar_detector.config, traffic_context_enabled=False)
    active = {}
    detect, update = core.camera_lidar_detector.detect, core.object_tracker.update

    def recorded_detect(image, cloud):
        proposals = detect(image, cloud)
        active['sensor_proposals'] = clean(proposals)
        return proposals

    def recorded_update(observations, timestamp, **kwargs):
        active['tracker_input'] = clean({'observations': observations, 'timestamp': timestamp, **kwargs})
        return update(observations, timestamp, **kwargs)

    core.camera_lidar_detector.detect = recorded_detect
    core.object_tracker.update = recorded_update
    count, start = 0, time.perf_counter()
    input_digest = hashlib.sha256()
    timings = []
    try:
        with (out / 'trace.jsonl').open('w', encoding='utf-8') as handle:
            for item in iter_archive(capture):
                if args.limit is not None and count >= args.limit:
                    break
                active.clear()
                # Fingerprint estimator inputs independently of evaluation labels.
                input_digest.update(item['image'].tobytes())
                input_digest.update(item['lidar'].tobytes())
                input_digest.update(json.dumps(clean([item['frame'], item['timestamp'], item['imu']]), sort_keys=True).encode())
                result = core.process(
                    frame_id=item['frame'], timestamp=item['timestamp'], image=item['image'],
                    lidar=item['lidar'], imu_data=item['imu'],
                    ground_truth_ego_pose=None if args.no_gt else item['ground_truth_ego_pose'],
                    ground_truth_objects=None if args.no_gt else item['ground_truth_objects'])
                row = {
                    'frame': item['frame'], 'timestamp': item['timestamp'],
                    'camera': core.camera_lidar_detector.last_camera_detections,
                    'detector_diagnostics': core.camera_lidar_detector.last_diagnostics,
                    'tracks': result['tracking']['active_tracks'],
                    'state': result['state_estimation'], 'graph': result['graph'],
                    'evaluation_gt': core.ground_truth_history[-1] if not args.no_gt else [],
                    'evaluation_tracks': core.tracking_history[-1] if not args.no_gt else [],
                    **active,
                }
                handle.write(json.dumps(clean(row), separators=(',', ':')) + '\n')
                timings.append(float(result['performance']['total_ms']))
                count += 1
                if count % 20 == 0:
                    print('Offline frames:', count, flush=True)
        if not count:
            raise RuntimeError('Capture contains no readable frames')
        summary = clean({
            'mode': 'OFFLINE_SENSOR_REPLAY_NO_CARLA', 'frames': count,
            'seconds': time.perf_counter() - start, 'capture': str(capture),
            'evaluation': core.evaluation_summary(), 'no_gt': args.no_gt,
            'tiles_enabled': not args.disable_tiles,
            'traffic_context_enabled': bool(core.camera_lidar_detector.last_diagnostics.get('traffic_context_active', False)),
            'traffic_context_sha256': (hashlib.sha256(model_path.with_suffix('.traffic_context.npz').read_bytes()).hexdigest()
                                       if model_path.with_suffix('.traffic_context.npz').is_file() and not args.disable_traffic_context else None),
            'gt_used_by_estimator': False,
            'static_map_sha256': hashlib.sha256(core.mapper_4d.get_static_map_points().tobytes()).hexdigest(),
            'sensor_inputs_sha256': input_digest.hexdigest(),
            'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'calibration_sha256': hashlib.sha256((capture / manifest['calibration']).read_bytes()).hexdigest(),
            'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
            'source_hashes': {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted((project / ('lvimot' if (project / 'lvimot').is_dir() else 'src')).rglob('*.py'))},
            'runtime': {'torch': torch.__version__, 'numpy': np.__version__,
                        'opencv': cv2.__version__, 'device': 'cpu', 'torch_threads': 2,
                        'opencv_threads': 1, 'median_frame_ms': float(np.median(timings)),
                        'p95_frame_ms': float(np.percentile(timings, 95))},
        })
        passed = True
        if args.reference:
            reference = json.loads(Path(args.reference).read_text(encoding='utf-8'))
            if (summary['sensor_inputs_sha256'] != reference.get('sensor_inputs_sha256')
                    or summary['model_sha256'] != reference.get('model_sha256')
                    or summary['calibration_sha256'] != reference.get('calibration_sha256')):
                raise ValueError('Reference must use identical sensor frames and model weights')
            if args.no_gt:
                if summary['traffic_context_sha256'] != reference.get('traffic_context_sha256'):
                    raise ValueError('GT-off comparison requires identical context classifier weights')
                checks = {'static_map_identical': summary['static_map_sha256'] == reference['static_map_sha256']}
                ref_trace = Path(args.reference).parent / 'trace.jsonl'
                with ref_trace.open(encoding='utf-8') as left, (out / 'trace.jsonl').open(encoding='utf-8') as right:
                    pairs = [(json.loads(a), json.loads(b)) for a, b in zip(left, right)]
                checks['states_and_tracks_identical'] = len(pairs) == count and all(
                    a['state'] == b['state'] and a['tracks'] == b['tracks'] for a, b in pairs)
                checks['camera_and_tracker_inputs_identical'] = len(pairs) == count and all(
                    a['camera'] == b['camera'] and a['tracker_input'] == b['tracker_input'] for a, b in pairs)
            else:
                old, new = reference['evaluation'], summary['evaluation']
                checks = {
                    'same_gt_observations': old['tracking']['total_gt_objects'] == new['tracking']['total_gt_objects'],
                    'mota_not_lower': new['tracking']['mota'] >= old['tracking']['mota'],
                    'ate_stable_within_1cm': new['localization']['ate_rmse'] <= old['localization']['ate_rmse'] + 0.01,
                    'rpe_stable_within_5mm': new['localization']['rpe_rmse'] <= old['localization']['rpe_rmse'] + 0.005,
                }
            passed = all(checks.values())
            summary['regression_checks'] = checks
            summary['regression_passed'] = passed
        if args.minimum_mota is not None:
            measured = summary['evaluation']['tracking'].get('mota')
            summary['acceptance_target'] = args.minimum_mota
            summary['acceptance_passed'] = bool(measured is not None and measured >= args.minimum_mota)
            passed = passed and summary['acceptance_passed']
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        tracking = summary['evaluation']['tracking']
        print(json.dumps({k: tracking[k] for k in ('mota', 'false_positives', 'false_negatives', 'id_switches')})
              if tracking else 'Ground truth disabled; sensor-only replay completed', flush=True)
        return 0 if passed else 1
    finally:
        core.close()


if __name__ == '__main__':
    raise SystemExit(main())
