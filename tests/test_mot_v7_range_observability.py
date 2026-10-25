import os
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np


from lvimot.temporal_fusion.pipeline import CarlaLVIMOTCore
from lvimot.temporal_fusion.runtime_limits import DEFAULT_LIMITS


class FakeCalibration:
    width = 200
    height = 200
    lidar_range = 100.0
    K = np.array([[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]])
    def get_camera_matrix(self):
        return self.K.copy()

    lidar_transform = {
        "location": {"x": 0.0, "y": 0.0, "z": 0.0},
        "rotation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    }
    camera_transform = {
        "location": {"x": 1.5, "y": 0.0, "z": 1.6},
        "rotation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    }

    def lidar_to_opencv(self, points):
        p = np.asarray(points, dtype=np.float64)
        return np.column_stack([p[:, 1], -p[:, 2], p[:, 0]])

    def project_lidar(self, points):
        c = self.lidar_to_opencv(points)
        valid = c[:, 2] > 0.1
        uv = np.full((len(c), 2), np.nan, dtype=np.float64)
        uv[valid, 0] = 100.0 + 100.0 * c[valid, 0] / c[valid, 2]
        uv[valid, 1] = 100.0 + 100.0 * c[valid, 1] / c[valid, 2]
        return uv, valid


def test_default_pipeline_range_comes_from_dataset_calibration():
    core = CarlaLVIMOTCore(FakeCalibration(), detector_mode="camera_lidar")
    assert core.object_sensor_range == 100.0
    assert core.camera_lidar_detector.config.max_range == 100.0
    assert core.sensor_object_detector.max_range == 100.0


def test_explicit_range_cap_still_overrides_larger_sensor_range():
    limits = replace(DEFAULT_LIMITS, object_max_range=60.0)
    core = CarlaLVIMOTCore(FakeCalibration(), limits=limits, detector_mode="camera_lidar")
    assert core.object_sensor_range == 60.0
    assert core.camera_lidar_detector.config.max_range == 60.0


def test_camera_lidar_gt_observability_uses_real_point_support_not_legacy_55m_center_gate():
    core = CarlaLVIMOTCore.__new__(CarlaLVIMOTCore)
    core.calibration = FakeCalibration()
    core.detector_mode = "camera_lidar"
    core.object_sensor_range = 100.0
    core.camera_lidar_detector = SimpleNamespace(config=SimpleNamespace(min_lidar_points=4))

    gt = [{
        "actor_id": 7,
        "type": "vehicle.test",
        "location": [80.0, 0.0, 0.0],
        "velocity": [0.0, 0.0, 0.0],
    }]
    boxes = {7: [80.0, 80.0, 120.0, 120.0]}
    rng = np.random.default_rng(8)
    lidar = np.column_stack([
        rng.normal(80.0, 0.25, 20),
        rng.normal(0.0, 0.7, 20),
        rng.normal(0.0, 0.5, 20),
        np.ones(20),
    ])

    support = core._gt_lidar_support_by_bbox(boxes, lidar)
    assert support[7] >= 4
    kept = core._filter_gt_observable_current_lidar(gt, boxes, lidar)
    assert len(kept) == 1
    assert kept[0]["actor_id"] == 7
    assert kept[0]["lidar_support"] >= 4


def test_camera_lidar_support_respects_effective_sensor_range():
    core = CarlaLVIMOTCore.__new__(CarlaLVIMOTCore)
    core.calibration = FakeCalibration()
    core.detector_mode = "camera_lidar"
    core.object_sensor_range = 55.0
    core.camera_lidar_detector = SimpleNamespace(config=SimpleNamespace(min_lidar_points=4))

    gt = [{"actor_id": 7, "type": "vehicle.test", "location": [80.0, 0.0, 0.0]}]
    boxes = {7: [80.0, 80.0, 120.0, 120.0]}
    lidar = np.array([
        [80.0, -0.4, 0.0, 1.0],
        [80.0, 0.0, 0.0, 1.0],
        [80.0, 0.4, 0.0, 1.0],
        [80.0, 0.0, 0.4, 1.0],
    ], dtype=np.float64)
    kept = core._filter_gt_observable_current_lidar(gt, boxes, lidar)
    assert kept == []
