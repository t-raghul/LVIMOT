"""CARLA snapshots must preserve measured geometry and sparse map support."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


from lvimot.mapping_4d.static_map.rolling_voxel_map import Carla4DEnvironmentMapper
from lvimot.inputs.sensor_calibration import CarlaCalibration
from lvimot.temporal_fusion.pipeline import CarlaLVIMOTCore
from lvimot.temporal_fusion.runtime_limits import DEFAULT_LIMITS


def _calibration(tmp_path):
    path = tmp_path / 'snapshot_calibration.json'
    zero = {'location': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'rotation': {}}
    path.write_text(json.dumps({
        'camera': {'width': 320, 'height': 180, 'fov': 90,
                   'intrinsic': [[160, 0, 160], [0, 160, 90], [0, 0, 1]],
                   'transform': zero},
        'lidar': {'channels': 64, 'range': 100, 'transform': zero},
        'imu': {'transform': zero}, 'lidar_to_camera_sensor': np.eye(4).tolist(),
    }))
    return CarlaCalibration(str(path))


def _core(tmp_path):
    limits = replace(DEFAULT_LIMITS, max_lidar_planar_candidates=800,
                     max_lidar_planar_queries=250, max_visual_features=100)
    return CarlaLVIMOTCore(_calibration(tmp_path), limits=limits)


def test_snapshot_is_not_warped_and_sparse_odd_frame_is_mapped(tmp_path):
    core = _core(tmp_path)
    rng = np.random.default_rng(41)
    points = np.column_stack([rng.uniform(3, 20, 700), rng.uniform(-8, 8, 700),
                              rng.normal(-1.5, 0.1, 700), np.ones(700)])
    image = rng.integers(0, 256, (180, 320, 3), dtype=np.uint8)
    imu = {'accelerometer': {'x': 0, 'y': 0, 'z': 9.81},
           'gyroscope': {'x': 0, 'y': 0, 'z': 0}, 'compass': np.pi / 2}
    calls = []
    def invalid_interpolation(scan, *args):
        calls.append(True)
        warped = scan.copy()
        warped[:, :3] += 7.0
        return warped
    core.lidar_deskewer.deskew = invalid_interpolation
    try:
        core.process(0, 0.0, image, points, imu)
        second_scan = points.copy()
        second_scan[:, 0] -= 0.1
        result = core.process(1, 0.1, image, second_scan, imu, include_lidar_points=True)
        np.testing.assert_array_equal(result['lidar']['points'], second_scan[:, :3])
        assert calls == []
        assert result['performance']['lidar_deskew_applied'] is False
        assert result['performance']['map_static_updated'] is True
        assert result['performance']['map_selected_input_points'] == len(points)
    finally:
        core.close()


def test_sparse_support_survives_and_dense_input_is_bounded():
    core = CarlaLVIMOTCore.__new__(CarlaLVIMOTCore)
    core.limits = DEFAULT_LIMITS
    core.mapper_4d = Carla4DEnvironmentMapper(static_voxel_size=0.2)
    rng = np.random.default_rng(42)
    sparse = rng.uniform(-40, 40, (3000, 4))
    selected, info = core._mapping_scan(sparse)
    np.testing.assert_array_equal(selected, sparse[:, :3])
    assert info['sparse_full_rate'] is True
    dense = rng.uniform(-40, 40, (80000, 4))
    selected, info = core._mapping_scan(dense)
    assert len(selected) <= DEFAULT_LIMITS.map_max_input_points
    assert info['sparse_full_rate'] is False
    assert np.all(selected.min(axis=0) < -35)
    assert np.all(selected.max(axis=0) > 35)


def test_evaluation_ground_truth_cannot_change_snapshot_estimation(tmp_path):
    a, b = _core(tmp_path), _core(tmp_path)
    rng = np.random.default_rng(43)
    points = np.column_stack([rng.uniform(3, 25, 650), rng.uniform(-7, 7, 650),
                              rng.normal(-1.6, 0.02, 650), np.ones(650)])
    image = rng.integers(0, 256, (180, 320, 3), dtype=np.uint8)
    imu = {'accelerometer': {'x': 0.2, 'y': 0, 'z': 9.81},
           'gyroscope': {'x': 0, 'y': 0, 'z': 0}, 'compass': np.pi / 2}
    try:
        for frame in range(3):
            gt = {'location': {'x': 500 + 10 * frame, 'y': -300 + frame, 'z': 17},
                  'rotation': {'pitch': 0, 'yaw': 120 + frame, 'roll': 0}}
            x = a.process(frame, frame * 0.1, image, points, imu)
            y = b.process(frame, frame * 0.1, image, points, imu,
                          ground_truth_ego_pose=gt, ground_truth_objects=[])
            np.testing.assert_array_equal(x['state_estimation']['pose'], y['state_estimation']['pose'])
            np.testing.assert_array_equal(a.mapper_4d.get_static_map_points(), b.mapper_4d.get_static_map_points())
    finally:
        a.close()
        b.close()
