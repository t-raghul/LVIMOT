"""Capture an independent CARLA rendered-depth reference and score LVIMOT's map.

This is an evaluation-only tool. It reads the saved route's CARLA poses solely
to place a depth camera and align the completed sensor-estimated map after the
fact. It never feeds CARLA geometry or GT into LVIMOT estimation, tracking, or
mapping. It re-renders only the recorded observed region; it cannot validate
untraversed parts of Town10HD_Opt.

Run from the project root while a CARLA 0.9.13 server is available:
  .venv_live38\\Scripts\\python.exe tools\\validate_carla_depth_reference.py --capture outputs_carla\\live\\sensor_recordings\\capture_20261020_085053_922129
"""
from __future__ import annotations

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[1]))


import argparse
import json
import math
import queue
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.spatial import cKDTree


def _matrix_from_pose(pose: Dict) -> Tuple[np.ndarray, np.ndarray]:
    loc = pose['location']
    rot = pose['rotation']
    pitch, yaw, roll = map(math.radians, [rot.get('pitch', 0.0), rot.get('yaw', 0.0), rot.get('roll', 0.0)])
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    rotation = np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp, -cp * sr, cp * cr],
    ], dtype=np.float64)
    translation = np.array([loc['x'], loc['y'], loc['z']], dtype=np.float64)
    return rotation, translation


def _camera_transform(carla, ego_pose: Dict, camera_extrinsic: Dict):
    ego_rotation, ego_translation = _matrix_from_pose(ego_pose)
    camera_rotation, camera_translation = _matrix_from_pose(camera_extrinsic)
    rotation = ego_rotation @ camera_rotation
    translation = ego_rotation @ camera_translation + ego_translation
    pitch = math.asin(float(np.clip(rotation[2, 0], -1.0, 1.0)))
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    roll = math.atan2(float(-rotation[2, 1]), float(rotation[2, 2]))
    return carla.Transform(
        carla.Location(x=float(translation[0]), y=float(translation[1]), z=float(translation[2])),
        carla.Rotation(pitch=math.degrees(pitch), yaw=math.degrees(yaw), roll=math.degrees(roll)),
    )


def _read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open('r', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _read_ply_xyz(path: Path) -> np.ndarray:
    with path.open('rb') as stream:
        header = bytearray()
        while not header.endswith(b'end_header\n'):
            ch = stream.read(1)
            if not ch or len(header) > 8192:
                raise ValueError('Invalid PLY header: ' + str(path))
            header.extend(ch)
        lines = header.decode('ascii').splitlines()
        if 'format binary_little_endian 1.0' not in lines:
            raise ValueError('Only binary little-endian PLY is supported: ' + str(path))
        count = next(int(line.split()[2]) for line in lines if line.startswith('element vertex '))
        vertices = np.frombuffer(stream.read(count * 12), dtype='<f4')
        if vertices.size != count * 3:
            raise ValueError('PLY vertex data is truncated: ' + str(path))
        return vertices.reshape(-1, 3).astype(np.float64)


def _write_ply(path: Path, xyz: np.ndarray) -> None:
    points = np.asarray(xyz, dtype='<f4').reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment independent CARLA rendered depth reference; evaluation only\n'
              'element vertex {}\nproperty float x\nproperty float y\nproperty float z\nend_header\n'.format(len(points)))
    with path.open('wb') as stream:
        stream.write(header.encode('ascii'))
        stream.write(points.tobytes(order='C'))


def _voxel_centroids(points: np.ndarray, voxel_m: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return points
    keys = np.floor(points / float(voxel_m)).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros((len(unique), 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    np.add.at(sums, inverse, points)
    return sums / counts[:, None]


def _dist_summary(distances: np.ndarray) -> Dict:
    d = np.asarray(distances, dtype=np.float64)
    return {
        'count': int(len(d)),
        'median_m': round(float(np.median(d)), 4),
        'p95_m': round(float(np.percentile(d, 95)), 4),
        'within_0_25m_pct': round(100.0 * float(np.mean(d <= 0.25)), 2),
        'within_0_5m_pct': round(100.0 * float(np.mean(d <= 0.5)), 2),
        'within_1m_pct': round(100.0 * float(np.mean(d <= 1.0)), 2),
        'within_2m_pct': round(100.0 * float(np.mean(d <= 2.0)), 2),
    }


def _compare(estimated: np.ndarray, reference: np.ndarray) -> Dict:
    if len(estimated) < 40 or len(reference) < 40:
        return {'status': 'INSUFFICIENT_EVIDENCE', 'estimated_points': int(len(estimated)),
                'reference_points': int(len(reference))}
    est_to_ref = cKDTree(reference).query(estimated, k=1)[0]
    ref_to_est = cKDTree(estimated).query(reference, k=1)[0]
    return {
        'status': 'MEASURED_NOT_CERTIFIED',
        'estimated_point_count': int(len(estimated)),
        'reference_point_count': int(len(reference)),
        'estimate_to_independent_reference': _dist_summary(est_to_ref),
        'independent_reference_to_estimate': _dist_summary(ref_to_est),
    }


def _rigid_icp(source: np.ndarray, target: np.ndarray, iterations: int = 12):
    """Return a diagnostic best-fit rigid alignment using trimmed ICP.

    The returned transform is evaluation-only and is reported separately from
    the first-pose placement. It is never applied to LVIMOT state or exports.
    """
    original = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    current = original.copy()
    tree = cKDTree(np.asarray(target, dtype=np.float64).reshape(-1, 3))
    total_rotation = np.eye(3, dtype=np.float64)
    total_translation = np.zeros(3, dtype=np.float64)
    used = 0
    for _ in range(int(iterations)):
        distances, indices = tree.query(current, k=1)
        valid = np.isfinite(distances) & (distances <= 8.0)
        if int(valid.sum()) < 40:
            break
        cutoff = float(np.quantile(distances[valid], 0.75))
        valid &= distances <= cutoff
        if int(valid.sum()) < 40:
            break
        a = current[valid]
        b = np.asarray(target, dtype=np.float64)[indices[valid]]
        ac, bc = a.mean(axis=0), b.mean(axis=0)
        u, _, vt = np.linalg.svd((a - ac).T @ (b - bc))
        rotation = u @ vt
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vt
        translation = bc - ac @ rotation
        current = current @ rotation + translation
        total_translation = total_translation @ rotation + translation
        total_rotation = total_rotation @ rotation
        used = int(valid.sum())
    return current, total_rotation, total_translation, used


def _finalize_evaluation(capture: Path, output: Path, reference: np.ndarray,
                         center: np.ndarray, max_range_m: float,
                         run_info: Dict) -> Dict:
    output.mkdir(parents=True, exist_ok=True)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    near_ref = np.linalg.norm(reference[:, :2] - center[:2], axis=1) <= float(max_range_m)
    reference = _voxel_centroids(reference[near_ref], 0.25)
    estimated_path = output / 'replay_aligned_CARLA_frame_EVALUATION_ONLY.ply'
    if not estimated_path.is_file():
        raise FileNotFoundError('Run tools\\replay_saved_capture.py first; missing ' + str(estimated_path))
    # This replay artifact is already placed in the CARLA world frame using
    # the first GT pose after estimation. Applying that transform again would
    # rotate/translate the map twice.
    estimated_world = _read_ply_xyz(estimated_path)
    near_est = np.linalg.norm(estimated_world[:, :2] - center[:2], axis=1) <= float(max_range_m)
    estimated = _voxel_centroids(estimated_world[near_est], 0.25)

    first_pose_comparison = _compare(estimated, reference)
    posthoc, rotation, translation, icp_matches = _rigid_icp(estimated, reference)
    posthoc_comparison = _compare(posthoc, reference)
    _write_ply(output / 'CARLA_rendered_depth_reference_EVALUATION_ONLY.ply', reference)
    overlay_path = output / 'CARLA_depth_vs_LVIMOT_overlay_EVALUATION_ONLY.svg'
    _overlay_svg(estimated, reference, center, overlay_path, float(max_range_m))
    posthoc_overlay_path = output / 'CARLA_depth_vs_LVIMOT_posthoc_rigid_overlay_EVALUATION_ONLY.svg'
    _overlay_svg(posthoc, reference, center, posthoc_overlay_path, float(max_range_m))
    result = {
        'mode': 'INDEPENDENT_CARLA_RENDERED_DEPTH_EVALUATION_ONLY',
        'map': 'Carla/Maps/Town10HD_Opt',
        'capture_frames': int(run_info.get('capture_frames', 0)),
        'rendered_frames': int(run_info.get('rendered_frames', 0)),
        'camera_resolution': run_info.get('camera_resolution'),
        'pixel_stride': run_info.get('pixel_stride'),
        'reference_max_range_m': float(max_range_m),
        'excluded_carla_semantic_labels': run_info.get('excluded_carla_semantic_labels'),
        'evaluation_region': 'recorded route, final ego neighborhood; observed depth-camera visibility only',
        'radius_xy_m': float(max_range_m),
        'voxel_centroid_size_m': 0.25,
        'alignment_comparisons': {
            'initial_pose_placement': {
                'description': 'Replay map already placed using first CARLA ego pose after estimation; no drift correction.',
                'comparison': first_pose_comparison,
                'overlay_svg': overlay_path.name,
            },
            'posthoc_rigid_alignment_evaluation_only': {
                'description': 'Trimmed point-cloud ICP diagnostic after initial placement; reports possible alignment benefit separately and does not alter LVIMOT map.',
                'comparison': posthoc_comparison,
                'matched_points_used_by_icp': icp_matches,
                'rotation_row_vector_convention': rotation.tolist(),
                'translation_m_row_vector_convention': translation.tolist(),
                'overlay_svg': posthoc_overlay_path.name,
            },
        },
        'ground_truth_used_by_estimator': False,
        'independent_reference': 'CARLA rendered depth camera with CARLA semantic labels; not LiDAR-derived',
        'artifacts': {
            'reference_ply': 'CARLA_rendered_depth_reference_EVALUATION_ONLY.ply',
            'initial_pose_map_overlay_svg': overlay_path.name,
            'posthoc_rigid_alignment_overlay_svg': posthoc_overlay_path.name,
            'estimated_map_source': estimated_path.name,
        },
        'limitations': [
            'This validates only geometry visible in the recorded route and depth camera field of view, not the whole Town10HD_Opt map.',
            'Depth is a rendered sensor measurement, with finite resolution and quantization; it is not raw Unreal triangle-mesh truth.',
            'The reference is independently rendered but shares the CARLA world and route poses used for evaluation; neither enters estimation.',
            'Static/dynamic separation excludes semantic tags 4, 10, and 11; rare parked-object differences may remain.',
            'The ICP result is a post-hoc evaluation diagnostic; use the initial-pose metrics to assess accumulated drift.',
            'No automatic pass threshold is imposed; interpret bidirectional accuracy and coverage together with the overlay.',
        ],
        'status': 'MEASURED_NOT_CERTIFIED' if first_pose_comparison.get('status') == 'MEASURED_NOT_CERTIFIED' else first_pose_comparison.get('status'),
    }
    report_path = output / 'CARLA_DEPTH_REFERENCE_VALIDATION_EVALUATION_ONLY.json'
    report_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('CARLA depth reference evaluation: ' + str(report_path), flush=True)
    return result


def _overlay_svg(estimated: np.ndarray, reference: np.ndarray, center: np.ndarray,
                 path: Path, radius_m: float, limit: int = 25000) -> None:
    size = 1000
    def layer(points: np.ndarray, color: str, stride_limit: int) -> str:
        inside = points[np.linalg.norm(points[:, :2] - center[:2], axis=1) <= radius_m]
        if len(inside) > stride_limit:
            inside = inside[::int(math.ceil(len(inside) / stride_limit))][:stride_limit]
        circles = []
        for x, y, _ in inside:
            px = size / 2 + (x - center[0]) * size / (2 * radius_m)
            py = size / 2 - (y - center[1]) * size / (2 * radius_m)
            circles.append('<circle cx="{:.2f}" cy="{:.2f}" r="1.2" fill="{}"/>'.format(px, py, color))
        return ''.join(circles)
    body = layer(reference, '#35d07f', limit) + layer(estimated, '#67c8ff', limit)
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1060" width="1000">'
           '<rect width="100%" height="100%" fill="#111b24"/>'
           '<text x="18" y="28" fill="white" font-size="17">Independent CARLA depth (green) | LVIMOT estimated map (blue)</text>'
           '<g transform="translate(0,40)">{}</g>'.format(body) +
           '<text x="18" y="1040" fill="white" font-size="13">{} m across; same CARLA world frame; initial first-pose placement; observed region only; evaluation-only GT</text>'.format(2 * radius_m) +
           '</svg>')
    path.write_text(svg, encoding='utf-8')


def _capture_reference(capture: Path, output: Path, host: str, port: int,
                       timeout_s: float, stride: int, max_range_m: float):
    import carla

    manifest = json.loads((capture / 'manifest.json').read_text(encoding='utf-8'))
    calibration = json.loads((capture / manifest['calibration']).read_text(encoding='utf-8'))
    cam = calibration['camera']
    width, height = int(cam['width']), int(cam['height'])
    intrinsics = np.asarray(cam['intrinsic'], dtype=np.float64)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])

    frames = [item for item in _read_jsonl(capture / manifest['frames'])
              if item.get('ground_truth_ego_pose_evaluation_only') is not None]
    if not frames:
        raise RuntimeError('Capture has no evaluation-only CARLA poses')
    gt_poses = [item['ground_truth_ego_pose_evaluation_only'] for item in frames]
    center_pose = gt_poses[-1]
    center = np.array([center_pose['location'][k] for k in ('x', 'y', 'z')], dtype=np.float64)

    client = carla.Client(host, int(port))
    client.set_timeout(float(timeout_s))
    current_map = client.get_world().get_map().name
    if not current_map.endswith('Town10HD_Opt'):
        client.load_world('Town10HD_Opt')
    world = client.get_world()
    settings = world.get_settings()
    old_sync, old_delta = settings.synchronous_mode, settings.fixed_delta_seconds
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = float(manifest.get('settings', {}).get('fixed_delta', 0.1))
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    depth_bp = blueprint_library.find('sensor.camera.depth')
    semantic_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    for bp in (depth_bp, semantic_bp):
        bp.set_attribute('image_size_x', str(width))
        bp.set_attribute('image_size_y', str(height))
        bp.set_attribute('fov', str(float(cam['fov'])))
        bp.set_attribute('sensor_tick', '0.0')

    queue_depth, queue_sem = queue.Queue(), queue.Queue()
    depth_sensor = None
    sem_sensor = None
    try:
        initial_transform = _camera_transform(carla, gt_poses[0], cam['transform'])
        depth_sensor = world.spawn_actor(depth_bp, initial_transform)
        sem_sensor = world.spawn_actor(semantic_bp, initial_transform)
        depth_sensor.listen(queue_depth.put)
        sem_sensor.listen(queue_sem.put)
    except Exception:
        for sensor in (depth_sensor, sem_sensor):
            if sensor is not None:
                try:
                    sensor.stop()
                except Exception:
                    pass
                sensor.destroy()
        restored = world.get_settings()
        restored.synchronous_mode = old_sync
        restored.fixed_delta_seconds = old_delta
        world.apply_settings(restored)
        raise
    clouds = []
    u = np.arange(0, width, max(1, int(stride)), dtype=np.float64)
    v = np.arange(0, height, max(1, int(stride)), dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    uu, vv = uu.reshape(-1), vv.reshape(-1)
    camera_xyz_scale = np.column_stack([np.ones_like(uu), (uu - cx) / fx, (cy - vv) / fy])
    frame_number = 0
    try:
        for item, pose in zip(frames, gt_poses):
            transform = _camera_transform(carla, pose, cam['transform'])
            depth_sensor.set_transform(transform)
            sem_sensor.set_transform(transform)
            target_frame = world.tick()
            deadline = time.monotonic() + float(timeout_s)
            depth_image = sem_image = None
            while depth_image is None or sem_image is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('CARLA camera sensors did not answer frame {}'.format(target_frame))
                if depth_image is None:
                    candidate = queue_depth.get(timeout=remaining)
                    if int(candidate.frame) >= int(target_frame):
                        depth_image = candidate
                if sem_image is None:
                    candidate = queue_sem.get(timeout=max(0.01, deadline - time.monotonic()))
                    if int(candidate.frame) >= int(target_frame):
                        sem_image = candidate
            depth_bgra = np.frombuffer(depth_image.raw_data, dtype=np.uint8).reshape(height, width, 4)[::stride, ::stride]
            sem_bgra = np.frombuffer(sem_image.raw_data, dtype=np.uint8).reshape(height, width, 4)[::stride, ::stride]
            r = depth_bgra[:, :, 2].astype(np.uint32)
            g = depth_bgra[:, :, 1].astype(np.uint32)
            b = depth_bgra[:, :, 0].astype(np.uint32)
            depth_m = ((r + g * 256 + b * 65536).astype(np.float64) / 16777215.0) * 1000.0
            labels = sem_bgra[:, :, 2]
            # CARLA 0.9.13 CityObjectLabel values come from its installed API.
            # Exclude people, vehicles, generic dynamic actors, and sky; keep
            # walls, terrain, and static geometry in the scene reference.
            excluded = np.array([
                int(carla.CityObjectLabel.Pedestrians),
                int(carla.CityObjectLabel.Vehicles),
                int(carla.CityObjectLabel.Dynamic),
                int(carla.CityObjectLabel.Sky),
            ], dtype=np.uint8)
            static = ~np.isin(labels, excluded)
            keep = static.reshape(-1) & np.isfinite(depth_m.reshape(-1))
            keep &= (depth_m.reshape(-1) >= 1.5) & (depth_m.reshape(-1) <= float(max_range_m))
            if np.any(keep):
                camera_xyz = camera_xyz_scale[keep] * depth_m.reshape(-1)[keep, None]
                camera_matrix = np.asarray(transform.get_matrix(), dtype=np.float64)
                world_xyz = camera_xyz @ camera_matrix[:3, :3].T + camera_matrix[:3, 3]
                clouds.append(world_xyz)
            frame_number += 1
            if frame_number % 20 == 0 or frame_number == len(frames):
                print('CARLA depth reference frames: {}/{}'.format(frame_number, len(frames)), flush=True)
    finally:
        depth_sensor.stop()
        sem_sensor.stop()
        depth_sensor.destroy()
        sem_sensor.destroy()
        restored = world.get_settings()
        restored.synchronous_mode = old_sync
        restored.fixed_delta_seconds = old_delta
        world.apply_settings(restored)

    reference = np.vstack(clouds) if clouds else np.empty((0, 3), dtype=np.float64)
    return _finalize_evaluation(capture, output, reference, center, max_range_m, {
        'capture_frames': len(frames), 'rendered_frames': frame_number,
        'camera_resolution': [width, height], 'pixel_stride': int(stride),
        'excluded_carla_semantic_labels': {
            'Pedestrians': int(carla.CityObjectLabel.Pedestrians),
            'Vehicles': int(carla.CityObjectLabel.Vehicles),
            'Dynamic': int(carla.CityObjectLabel.Dynamic),
            'Sky': int(carla.CityObjectLabel.Sky),
        },
    })


def _evaluate_existing_reference(capture: Path, output: Path, max_range_m: float) -> Dict:
    manifest = json.loads((capture / 'manifest.json').read_text(encoding='utf-8'))
    calibration = json.loads((capture / manifest['calibration']).read_text(encoding='utf-8'))
    frames = [item for item in _read_jsonl(capture / manifest['frames'])
              if item.get('ground_truth_ego_pose_evaluation_only') is not None]
    if not frames:
        raise RuntimeError('Capture has no evaluation-only CARLA poses')
    last = frames[-1]['ground_truth_ego_pose_evaluation_only']['location']
    center = np.array([last[k] for k in ('x', 'y', 'z')], dtype=np.float64)
    reference_path = output / 'CARLA_rendered_depth_reference_EVALUATION_ONLY.ply'
    if not reference_path.is_file():
        raise FileNotFoundError('No saved independent reference cloud: ' + str(reference_path))
    reference = _read_ply_xyz(reference_path)
    return _finalize_evaluation(capture, output, reference, center, max_range_m, {
        'capture_frames': len(frames), 'rendered_frames': len(frames),
        'camera_resolution': [int(calibration['camera']['width']), int(calibration['camera']['height'])],
        'pixel_stride': 2,
        'excluded_carla_semantic_labels': {
            'Pedestrians': 4, 'Vehicles': 10, 'Dynamic': 20, 'Sky': 13,
        },
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True, type=Path)
    parser.add_argument('--replay-output', type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=2000, type=int)
    parser.add_argument('--timeout', default=20.0, type=float)
    parser.add_argument('--pixel-stride', default=2, type=int)
    parser.add_argument('--max-range-m', default=60.0, type=float)
    parser.add_argument('--evaluate-existing-reference', action='store_true',
                        help='Reuse the previously captured CARLA depth PLY; do not connect to or tick CARLA.')
    args = parser.parse_args()
    capture = args.capture.resolve()
    output = args.replay_output.resolve() if args.replay_output else capture / 'offline_replay_workmode_20261020'
    if not capture.is_dir():
        raise SystemExit('Capture directory not found: ' + str(capture))
    if args.pixel_stride < 1 or args.max_range_m <= 0:
        raise SystemExit('pixel-stride and max-range-m must be positive')
    if args.evaluate_existing_reference:
        _evaluate_existing_reference(capture, output, args.max_range_m)
    else:
        _capture_reference(capture, output, args.host, args.port, args.timeout,
                           args.pixel_stride, args.max_range_m)


if __name__ == '__main__':
    main()
