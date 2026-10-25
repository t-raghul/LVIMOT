"""Render an independent 360-degree CARLA scene reference along a saved route.

Four overlapping depth/semantic views are placed at the recorded LiDAR origin.
Only evaluation reference data is written. Original sensors, estimated poses,
tracking, and map state are never modified. Existing actors/processes remain.
All sensor images and exact capture transforms are saved for offline reuse.
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
import queue
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from lvimot.output.evaluation.independent_reference import decode_depth_bgra, depth_camera_points, sensor_field_mask, matrix_to_pose, voxel_reference
from tools.validate_carla_depth_reference import _matrix_from_pose, _write_ply


def selected_keyframes(frames, spacing_m=2.0):
    selected = [frames[0]]
    first = frames[0]['ground_truth_ego_pose_evaluation_only']['location']
    previous = np.array([first[k] for k in ('x', 'y', 'z')], dtype=float)
    for frame in frames[1:-1]:
        loc = frame['ground_truth_ego_pose_evaluation_only']['location']
        current = np.array([loc[k] for k in ('x', 'y', 'z')])
        if np.linalg.norm(current - previous) >= spacing_m:
            selected.append(frame)
            previous = current
    if frames[-1]['frame'] != selected[-1]['frame']:
        selected.append(frames[-1])
    return selected


def _carla_transform(carla, rotation, translation):
    pose = matrix_to_pose(rotation, translation)
    return carla.Transform(carla.Location(**pose['location']), carla.Rotation(**pose['rotation']))


def _exact_image(stream, target_frame, timeout):
    deadline = time.monotonic() + timeout
    while True:
        image = stream.get(timeout=max(.01, deadline - time.monotonic()))
        if image.frame == target_frame:
            return image
        if image.frame > target_frame or time.monotonic() >= deadline:
            raise RuntimeError('Reference camera frame mismatch: expected %d, received %d' % (target_frame, image.frame))


def capture_reference(capture: Path, output: Path, host='127.0.0.1', port=2000,
                      spacing_m=2.0, radius_m=60.0):
    import carla
    manifest = json.loads((capture / 'manifest.json').read_text(encoding='utf-8'))
    calibration = json.loads((capture / manifest['calibration']).read_text(encoding='utf-8'))
    frames = [json.loads(line) for line in (capture / manifest['frames']).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not frames or any(not row.get('ground_truth_ego_pose_evaluation_only') for row in frames):
        raise ValueError('A complete evaluation-only recorded route is required')
    keys = selected_keyframes(frames, spacing_m)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'rendered_frames').mkdir()
    client = carla.Client(host, int(port))
    client.set_timeout(20.0)
    world = client.get_world()
    carla_map = world.get_map()
    if not carla_map.name.endswith('Town10HD_Opt'):
        raise RuntimeError('Reference requires original Town10HD_Opt; refusing to load another world')
    original_settings = world.get_settings()
    actors_before = set(int(actor.id) for actor in world.get_actors())
    live_vehicle_count = len(world.get_actors().filter('vehicle.*'))
    excluded = {int(carla.CityObjectLabel.Pedestrians), int(carla.CityObjectLabel.Dynamic), int(carla.CityObjectLabel.Sky)}
    if live_vehicle_count:
        excluded.add(int(carla.CityObjectLabel.Vehicles))
    sensors = []
    streams = []
    clouds, semantics, records = [], [], []
    r_el, t_el = _matrix_from_pose(calibration['lidar']['transform'])
    view_rotations = [_matrix_from_pose({'location': {'x': 0, 'y': 0, 'z': 0}, 'rotation': {'yaw': yaw, 'pitch': -10, 'roll': 0}})[0] for yaw in (0, 90, 180, 270)]
    started = time.monotonic()
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(manifest.get('settings', {}).get('fixed_delta', .1))
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        library = world.get_blueprint_library()
        for rotation in view_rotations:
            pair, pair_streams = [], []
            for sensor_name in ('sensor.camera.depth', 'sensor.camera.semantic_segmentation'):
                bp = library.find(sensor_name)
                for name, value in {'image_size_x': '512', 'image_size_y': '256', 'fov': '100', 'sensor_tick': '0.0'}.items():
                    bp.set_attribute(name, value)
                if bp.has_attribute('lens_k'):
                    bp.set_attribute('lens_k', '0.0')
                actor = world.spawn_actor(bp, carla.Transform(carla.Location(z=3)))
                sensors.append(actor)
                stream = queue.Queue()
                actor.listen(stream.put)
                pair.append(actor)
                pair_streams.append(stream)
            streams.append((rotation, pair, pair_streams))
        for index, frame in enumerate(keys):
            if time.monotonic() - started > 180:
                raise TimeoutError('Independent reference exceeded its bounded 180-second budget')
            r_we, t_we = _matrix_from_pose(frame['ground_truth_ego_pose_evaluation_only'])
            r_wl, t_wl = r_we @ r_el, r_we @ t_el + t_we
            expected = []
            transform_commands = []
            for rotation, pair, _ in streams:
                transform = _carla_transform(carla, r_wl @ rotation, t_wl)
                expected.append(np.asarray(transform.get_matrix(), dtype=float))
                for sensor in pair:
                    transform_commands.append(carla.command.ApplyTransform(int(sensor.id), transform))
            responses = client.apply_batch_sync(transform_commands, True)
            if any(response.has_error() for response in responses):
                raise RuntimeError('Could not apply all reference camera transforms atomically')
            target = int(world.get_snapshot().frame)
            for view, (rotation, _, pair_streams) in enumerate(streams):
                depth_image, label_image = [_exact_image(stream, target, 20.0) for stream in pair_streams]
                depth_transform = np.asarray(depth_image.transform.get_matrix(), dtype=float)
                label_transform = np.asarray(label_image.transform.get_matrix(), dtype=float)
                if not np.allclose(depth_transform, expected[view], atol=2e-4) or not np.allclose(label_transform, depth_transform, atol=2e-4):
                    mismatch = {'target_carla_frame': target, 'route_frame': frame['frame'], 'view': view,
                                'expected_world_matrix': expected[view].tolist(),
                                'depth_world_matrix': depth_transform.tolist(),
                                'semantic_world_matrix': label_transform.tolist()}
                    (output / 'transform_mismatch.json').write_text(json.dumps(mismatch, indent=2) + '\n')
                    raise RuntimeError('Rendered sensor transform differs from requested pose; details saved in transform_mismatch.json')
                bgra = np.frombuffer(depth_image.raw_data, dtype=np.uint8).reshape(256, 512, 4)
                label_bgra = np.frombuffer(label_image.raw_data, dtype=np.uint8).reshape(256, 512, 4)
                depth = decode_depth_bgra(bgra)
                labels = label_bgra[:, :, 2]
                point_camera, point_labels = depth_camera_points(depth, labels, 100.0, stride=2)
                point_lidar = point_camera @ rotation.T
                keep = sensor_field_mask(point_lidar, radius_m,
                                         float(calibration['lidar'].get('lower_fov', -30)),
                                         float(calibration['lidar'].get('upper_fov', 10)))
                keep &= ~np.isin(point_labels, np.array(sorted(excluded), dtype=np.uint8))
                if np.any(keep):
                    points_world = point_camera[keep] @ depth_transform[:3, :3].T + depth_transform[:3, 3]
                    clouds.append(points_world)
                    semantics.append(point_labels[keep])
                name = 'route_%06d_view_%d.npz' % (frame['frame'], view)
                np.savez_compressed(str(output / 'rendered_frames' / name),
                                    depth_m=depth.astype(np.float32), semantic=labels,
                                    camera_world_matrix=depth_transform,
                                    carla_frame=np.int64(target), recorded_route_frame=np.int64(frame['frame']))
                records.append({'recorded_route_frame': frame['frame'], 'view': view,
                                'carla_reference_frame': int(target), 'data_file': 'rendered_frames/' + name,
                                'world_matrix': depth_transform.tolist()})
            if (index + 1) % 5 == 0 or index + 1 == len(keys):
                print('Independent 360 reference keyposes: %d/%d' % (index + 1, len(keys)), flush=True)
    finally:
        if sensors:
            client.apply_batch([carla.command.DestroyActor(int(sensor.id)) for sensor in sensors])
        world.apply_settings(original_settings)
        restored = world.get_settings()
        restoration = {
            'synchronous_mode': restored.synchronous_mode,
            'fixed_delta_seconds': restored.fixed_delta_seconds,
            'no_rendering_mode': restored.no_rendering_mode,
            'external_actor_ids_preserved': actors_before.issubset({int(actor.id) for actor in world.get_actors()}),
        }
        (output / 'server_restoration.json').write_text(json.dumps(restoration, indent=2) + '\n')
    if not clouds:
        raise RuntimeError('Independent reference contains no static geometry')
    points, labels = voxel_reference(np.vstack(clouds), np.concatenate(semantics), .25)
    _write_ply(output / 'CARLA_independent_360_reference.ply', points)
    np.savez_compressed(str(output / 'CARLA_independent_360_reference_semantics.npz'), xyz=points, semantic=labels)
    semantic_counts = {str(int(label)): int(np.count_nonzero(labels == label)) for label in np.unique(labels)}
    record = {
        'format': 'CARLA_INDEPENDENT_RENDERED_360_REFERENCE_V1',
        'reference_source': 'Original CARLA rendered depth and semantic buffers, independent of recorded LiDAR',
        'map': carla_map.name, 'opendrive_sha256': hashlib.sha256(carla_map.to_opendrive().encode()).hexdigest(),
        'gt_used_by_estimator': False, 'reference_only': True,
        'recorded_capture': str(capture.resolve()), 'recorded_route_frames': len(frames),
        'reference_keyposes': len(keys), 'views_per_pose': 4, 'images': records,
        'camera': {'width': 512, 'height': 256, 'horizontal_fov_deg': 100, 'pitch_deg': -10, 'pixel_stride': 2, 'lens_k': 0},
        'sensor_origin': 'recorded calibrated LiDAR origin', 'ray_range_m': radius_m,
        'vertical_field_deg': [calibration['lidar'].get('lower_fov', -30), calibration['lidar'].get('upper_fov', 10)],
        'voxel_size_m': .25, 'reference_voxels': len(points), 'semantic_counts': semantic_counts,
        'excluded_semantic_tags': sorted(excluded), 'existing_live_vehicle_actors': live_vehicle_count,
        'static_parked_vehicles_included': not bool(live_vehicle_count),
        'server_restoration': restoration,
        'limits': ['Observed route and configured LiDAR field/range only.',
                   'Rendered depth is an independent surface measurement, not exported triangle mesh.',
                   'Finite ray sampling and semantics affect coverage; no whole-town claim is made.'],
    }
    (output / 'reference_manifest.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print('Independent reference voxels:', len(points), 'saved:', output, flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--spacing-m', type=float, default=2.0)
    parser.add_argument('--radius-m', type=float, default=60.0)
    args = parser.parse_args()
    capture_reference(args.capture, args.output, args.host, args.port, args.spacing_m, args.radius_m)
