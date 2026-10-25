from types import SimpleNamespace

import numpy as np

from lvimot.preprocessing.camera_lidar_detection import CameraLiDARDetectionConfig
from lvimot.temporal_fusion.pipeline import CarlaLVIMOTCore


class FakeCalibration:
    lidar_range = 100.0

    @staticmethod
    def lidar_to_opencv(xyz):
        # OpenCV X=right, Y=down, Z=forward for this synthetic test.
        xyz = np.asarray(xyz, dtype=np.float64)
        return np.column_stack([xyz[:, 1], xyz[:, 2], xyz[:, 0]])

    @staticmethod
    def project_lidar(xyz):
        xyz = np.asarray(xyz, dtype=np.float64)
        z = np.maximum(xyz[:, 0], 1e-6)
        uv = np.column_stack([100.0 + 100.0 * xyz[:, 1] / z, 100.0 + 100.0 * xyz[:, 2] / z])
        return uv, xyz[:, 0] > 0.1


def test_v8_detector_defaults_keep_native_resolution_for_1242px_camera():
    cfg = CameraLiDARDetectionConfig()
    assert cfg.image_size == 1280
    assert cfg.confidence == 0.12
    assert cfg.min_eval_bbox_area_px > 0


def test_v8_gt_observability_excludes_tiny_boxes_but_keeps_supported_boxes():
    core = CarlaLVIMOTCore.__new__(CarlaLVIMOTCore)
    core.calibration = FakeCalibration()
    core.detector_mode = "camera_lidar"
    core.object_sensor_range = 100.0
    cfg = CameraLiDARDetectionConfig(min_lidar_points=4)
    core.camera_lidar_detector = SimpleNamespace(config=cfg)

    gt = [
        {"actor_id": 1, "type": "vehicle.test", "location": [80.0, 0.0, 0.0]},
        {"actor_id": 2, "type": "vehicle.test", "location": [60.0, 0.0, 0.0]},
    ]
    boxes = {
        1: [97.0, 95.0, 103.0, 106.0],  # 6 x 11 = 66 px^2 -> too small
        2: [90.0, 90.0, 110.0, 110.0],  # 20 x 20 = 400 px^2
    }
    lidar = np.array([
        [80.0, 0.0, 0.0, 1.0],
        [80.0, 0.2, 0.0, 1.0],
        [80.0, -0.2, 0.0, 1.0],
        [80.0, 0.0, 0.2, 1.0],
        [60.0, 0.0, 0.0, 1.0],
        [60.0, 0.2, 0.0, 1.0],
        [60.0, -0.2, 0.0, 1.0],
        [60.0, 0.0, 0.2, 1.0],
    ], dtype=np.float64)

    kept = core._filter_gt_observable_current_lidar(gt, boxes, lidar)
    assert [int(x["actor_id"]) for x in kept] == [2]
