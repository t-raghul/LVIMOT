import os
import sys

import numpy as np


from lvimot.preprocessing.lidar_planar_features import CarlaLiDARPlanarFeatureExtractor
from lvimot.tracking_4d.multimodal_object_tracking.kalman_tracker import CarlaMultiObjectTracker
from lvimot.preprocessing.camera_lidar_detection import CarlaCameraLiDARObjectDetector, CameraLiDARDetectionConfig
from lvimot.temporal_fusion.runtime_limits import DEFAULT_LIMITS


def test_vectorized_planar_extractor_finds_dense_plane():
    xs, ys = np.meshgrid(np.linspace(2.0, 12.0, 80), np.linspace(-4.0, 4.0, 60))
    z = 0.015 * np.sin(xs) + 0.01 * np.cos(ys)
    pts = np.column_stack([xs.ravel(), ys.ravel(), z.ravel(), np.ones(xs.size)])
    extractor = CarlaLiDARPlanarFeatureExtractor(
        voxel_size=0.12,
        search_radius=0.45,
        min_neighbors=8,
        max_neighbors=24,
    )
    features = extractor.extract(pts)
    assert len(features) > 100
    assert all(np.isfinite(f["normal"]).all() for f in features[:50])
    assert all(float(f["planarity"]) > 0.15 for f in features[:50])


def test_tracker_can_create_weak_candidate_but_publish_only_stronger_track():
    tracker = CarlaMultiObjectTracker(
        distance_threshold=3.0,
        max_missed=2,
        min_hits=1,
        use_ego_compensation=False,
        new_track_min_score=0.45,
        publish_min_score=0.50,
    )
    observations = [
        {
            "location": [10.0, 0.0, 0.0],
            "dimensions": {"length": 4.0, "width": 1.8, "height": 1.5},
            "type": "vehicle",
            "class": "vehicle",
            "score": 0.46,
        },
        {
            "location": [20.0, 0.0, 0.0],
            "dimensions": {"length": 4.0, "width": 1.8, "height": 1.5},
            "type": "vehicle",
            "class": "vehicle",
            "score": 0.90,
        },
    ]
    published = tracker.update(observations, 0.0)
    assert len(tracker.tracks) == 2
    assert len(published) == 1
    assert published[0]["position"][0] > 15.0


class _Boxes:
    xyxy = np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32)
    cls = np.array([2], dtype=np.float32)
    conf = np.array([0.9], dtype=np.float32)


class _Result:
    boxes = _Boxes()


class _Model:
    def __init__(self):
        self.kwargs = None

    def predict(self, **kwargs):
        self.kwargs = kwargs
        return [_Result()]


class _Calibration:
    def lidar_to_opencv(self, points):
        p = np.asarray(points, dtype=np.float64)
        return np.column_stack([p[:, 1], -p[:, 2], p[:, 0]])

    def project_lidar(self, points):
        p = np.asarray(points, dtype=np.float64)
        uv = np.column_stack([15.0 + 0.0 * p[:, 0], 15.0 + 0.0 * p[:, 0]])
        return uv, p[:, 0] > 0.1


def test_explicit_cpu_detector_disables_fp16():
    model = _Model()
    detector = CarlaCameraLiDARObjectDetector(
        _Calibration(),
        CameraLiDARDetectionConfig(device="cpu", half=True, min_lidar_points=4, max_range=100.0),
        model=model,
    )
    lidar = np.column_stack([
        np.full(8, 10.0),
        np.linspace(-0.2, 0.2, 8),
        np.linspace(-0.2, 0.2, 8),
        np.ones(8),
    ])
    detector.detect(np.zeros((32, 32, 3), dtype=np.uint8), lidar)
    assert model.kwargs["device"] == "cpu"
    assert model.kwargs["half"] is False


def test_v11_default_optimizer_budget_is_bounded():
    assert DEFAULT_LIMITS.optimizer_max_nfev <= 3
