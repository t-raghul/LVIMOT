"""Reproducible sensor-only replay with cached RGB inference and diagnostic traces.

RGB inference caches contain image detections only. Simulator ground truth is
kept in separate evaluation fields and is never part of a cache key or detector
input. No CARLA server is contacted by this program.
"""
from __future__ import annotations

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


from lvimot.inputs.sensor_calibration import CarlaCalibration
from lvimot.output.live_dashboard import CarlaLiveVisualizer
from lvimot.output.map_export import export_live_map, write_binary_ply
from lvimot.output.evaluation.map_geometry import align_local_map_to_carla
from lvimot.temporal_fusion.pipeline import CarlaLVIMOTCore
from lvimot.inputs.sensor_recording import iter_archive
from tools.validate_carla_depth_reference import _compare, _read_ply_xyz, _voxel_centroids


def clean(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


class RGBInferenceCache:
    """Cache the expensive RGB network output while recalculating LiDAR depth."""

    def __init__(self, detector, weights: Path, directory: Path):
        import ultralytics
        self.detector = detector
        self.original = detector._infer_boxes
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        config = asdict(detector.config)
        config['model_path'] = weights.name
        self.signature = {
            'format': 'LVIMOT_RGB_INFERENCE_CACHE_V1',
            'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
            'ultralytics_version': ultralytics.__version__,
            'detector_config': config,
        }
        self.digest = hashlib.sha256(json.dumps(self.signature, sort_keys=True).encode()).hexdigest()
        self.hits = 0
        self.misses = 0
        self.last_proposals = []
        original_detect = detector.detect

        def infer(image):
            image_digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
            key = hashlib.sha256((self.digest + image_digest).encode()).hexdigest()
            path = self.directory / (key + '.json')
            if path.is_file():
                cached = json.loads(path.read_text(encoding='utf-8'))
                if cached.get('signature_digest') != self.digest or cached.get('image_sha256') != image_digest:
                    raise ValueError('RGB inference cache provenance mismatch: ' + str(path))
                boxes = cached['camera_detections']
                self.hits += 1
                detector.last_diagnostics['rgb_inference_source'] = 'verified_saved_RGB_inference'
            else:
                boxes = clean(self.original(image))
                path.write_text(json.dumps({
                    'signature': self.signature, 'signature_digest': self.digest,
                    'image_sha256': image_digest, 'camera_detections': boxes,
                    'ground_truth_used': False,
                }, indent=2) + '\n', encoding='utf-8')
                self.misses += 1
            return boxes

        def detect(image, lidar):
            proposals = original_detect(image, lidar)
            self.last_proposals = clean(proposals)
            return proposals

        detector._infer_boxes = infer
        detector.detect = detect


def track_snapshot(track):
    names = ('track_id', 'hits', 'consecutive_hits', 'missed', 'score',
             'camera_support', 'camera_confidence', 'sparse_consistent_hits',
             'published_with_3d', 'camera_only_predicted', 'camera_only_bridge_count',
             'last_measurement_timestamp', 'last_timestamp', 'bbox')
    result = {name: clean(getattr(track, name, None)) for name in names}
    result['position'] = clean(track.position)
    result['velocity'] = clean(track.velocity)
    return result


def run(capture: Path, output: Path, cache_directory: Path,
        reference_path: Path = None, evaluate_gt: bool = True,
        yolo_image_size: int = None, yolo_model: str = None):
    capture = capture.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((capture / 'manifest.json').read_text(encoding='utf-8'))
    settings = manifest.get('settings') or {}
    expected_frames = sum(bool(line.strip()) for line in (capture / manifest['frames']).read_text(encoding='utf-8').splitlines())
    calibration = CarlaCalibration(str(capture / manifest['calibration']))
    weights = (ROOT / str(yolo_model or settings.get('yolo_model', 'models/yolov8n.pt'))).resolve()
    if not weights.is_file():
        raise FileNotFoundError('Local YOLO weights are required: ' + str(weights))
    core = CarlaLVIMOTCore(
        calibration, detector_mode='camera_lidar', yolo_model=str(weights),
        yolo_device=str(settings.get('yolo_device', 'auto')),
        yolo_confidence=float(settings.get('yolo_confidence', 0.08)),
        yolo_image_size=int(yolo_image_size or settings.get('yolo_image_size', 640)),
    )
    cache = RGBInferenceCache(core.camera_lidar_detector, weights, cache_directory)
    core.mapper_4d.enable_accumulated_map(output / 'accumulated_maps')
    visualizer = CarlaLiveVisualizer()
    first_gt = last_gt = last_pose = None
    count = 0
    timings = []
    started = time.perf_counter()
    try:
        with (output / 'frame_trace.jsonl').open('w', encoding='utf-8') as trace:
            for item in iter_archive(capture):
                gt = item['ground_truth_ego_pose']
                if gt is not None:
                    first_gt = first_gt or gt
                    last_gt = gt
                result = core.process(
                    frame_id=int(item['frame']), timestamp=float(item['timestamp']),
                    image=item['image'], lidar=item['lidar'], imu_data=item['imu'],
                    ground_truth_ego_pose=(gt if evaluate_gt else None),
                    ground_truth_objects=(item['ground_truth_objects'] if evaluate_gt else None),
                    include_lidar_points=True,
                )
                last_pose = visualizer._pose(result)
                timings.append(result['performance'])
                row = {
                    'frame': item['frame'], 'timestamp': item['timestamp'],
                    'sensor_only': {
                        'camera_detections': core.camera_lidar_detector.last_camera_detections,
                        'lidar_proposals': cache.last_proposals,
                        'detector_diagnostics': core.camera_lidar_detector.last_diagnostics,
                        'track_hypotheses': [track_snapshot(t) for t in core.object_tracker.tracks],
                        'published_tracks': result['tracking']['active_tracks'],
                        'pose': result['state_estimation']['pose'],
                        'velocity': result['state_estimation']['velocity'],
                        'sensor_integrity': result['sensor_integrity'],
                        'performance': result['performance'],
                        'map_summary': result['mapping_4d'],
                    },
                    'evaluation_only': {
                        'eligible_gt': core.ground_truth_history[-1] if evaluate_gt and gt is not None else [],
                        'scored_tracks': core.tracking_history[-1] if evaluate_gt and gt is not None else [],
                        'carla_ego_pose': gt,
                    },
                }
                trace.write(json.dumps(clean(row), allow_nan=False) + '\n')
                count += 1
                if count % 30 == 0:
                    print('Recorded validation: {}/{}'.format(count, expected_frames), flush=True)
        if not count:
            raise RuntimeError('Empty recording')
        exports = export_live_map(core.mapper_4d, last_pose, visualizer, output)
        summary = {
            'mode': 'OFFLINE_RECORDED_SENSOR_VALIDATION_NO_CARLA',
            'frames': count, 'gt_used_by_estimator': False,
            'evaluation_enabled': bool(evaluate_gt),
            'evaluation': core.evaluation_summary() if evaluate_gt else None,
            'map_export': exports, 'map_summary': core.mapper_4d.export_summary(),
            'runtime_limits': asdict(core.limits),
            'rgb_cache': {'hits': cache.hits, 'network_inferences': cache.misses,
                          'signature': cache.signature},
            'elapsed_wall_seconds': time.perf_counter() - started,
        }
        for stage in ('total_ms', 'mapping_ms', 'rss_mb'):
            values = np.asarray([x[stage] for x in timings], dtype=float)
            summary[stage] = {'mean': float(np.mean(values)), 'p95': float(np.percentile(values, 95))}
        if first_gt is not None and last_gt is not None:
            placed = align_local_map_to_carla(core.mapper_4d.get_static_map_points(), first_gt)
            write_binary_ply(output / 'replay_aligned_CARLA_frame_EVALUATION_ONLY.ply', placed)
            if reference_path is not None:
                reference = _read_ply_xyz(reference_path)
                center = np.array([last_gt['location'][k] for k in ('x', 'y', 'z')])
                e = placed[np.linalg.norm(placed[:, :2] - center[:2], axis=1) <= 60.0]
                r = reference[np.linalg.norm(reference[:, :2] - center[:2], axis=1) <= 60.0]
                summary['independent_depth_comparison_initial_pose'] = _compare(
                    _voxel_centroids(e, 0.25), _voxel_centroids(r, 0.25))
                summary['independent_depth_reference_file'] = str(reference_path.resolve())
                summary['reference_limits'] = 'Forward depth camera coverage differs from 360-degree LiDAR; both directions are diagnostic, not certification.'
        (output / 'validation_summary.json').write_text(json.dumps(clean(summary), indent=2) + '\n', encoding='utf-8')
        print('Saved diagnostic validation: ' + str(output), flush=True)
        return summary
    finally:
        core.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-directory', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--without-gt-evaluation', action='store_true')
    parser.add_argument('--yolo-image-size', type=int,
                        help='Explicit RGB inference resolution override; included in cache provenance')
    parser.add_argument('--yolo-model', help='Explicit local weights override; SHA recorded in cache provenance')
    args = parser.parse_args()
    run(args.capture, args.output, args.cache_directory, args.reference,
        evaluate_gt=not args.without_gt_evaluation, yolo_image_size=args.yolo_image_size,
        yolo_model=args.yolo_model)
