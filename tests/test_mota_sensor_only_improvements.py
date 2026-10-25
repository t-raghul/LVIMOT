"""Regression checks for conservative MOTA-oriented sensor-only changes.

These tests exercise error mechanisms. They are not a measured MOTA gain on a
recorded CARLA benchmark; use the raw saved replay for such comparisons.
"""
import sys
from pathlib import Path

import numpy as np

from lvimot.preprocessing.camera_lidar_detection import CarlaCameraLiDARObjectDetector, CameraLiDARDetectionConfig
from lvimot.tracking_4d.multimodal_object_tracking.kalman_tracker import CarlaMultiObjectTracker


class _Cal:
    def lidar_to_opencv(self, points):
        p = np.asarray(points)
        return np.column_stack((p[:, 1], -p[:, 2], p[:, 0]))

    def project_lidar(self, points):
        c = self.lidar_to_opencv(points)
        out = np.column_stack((100.0 + 100.0 * c[:, 0] / c[:, 2], 100.0 + 100.0 * c[:, 1] / c[:, 2]))
        return out, c[:, 2] > 0.1


class _Box:
    xyxy = np.asarray([[45, 40, 155, 160]], dtype=np.float32)
    cls = np.asarray([2], dtype=np.float32)
    conf = np.asarray([0.7], dtype=np.float32)


class _Predictor:
    def predict(self, **kwargs):
        from types import SimpleNamespace
        return [SimpleNamespace(boxes=_Box())]


def _obs(x, y=0, bbox=(100, 90, 160, 145), confidence=0.55):
    return {
        'location': [float(x), float(y), 0.0],
        'bbox': list(bbox),
        'camera_evidence': True,
        'camera_confidence': confidence,
        'camera_support': 20,
        'depth_spread_m': 0.5,
        'score': 0.67,
        'class': 'vehicle', 'type': 'vehicle',
        'dimensions': {'length': 4.5, 'width': 1.9, 'height': 1.6},
    }


def test_three_measured_foreground_points_recovered_before_dense_wall():
    d = np.array([29.9, 30.0, 30.2, 54.9, 55.0, 55.1, 55.2, 55.3, 55.4, 55.5])
    selected = CarlaCameraLiDARObjectDetector._depth_cluster(d, np.arange(len(d)))
    assert sorted(selected.tolist()) == [0, 1, 2]


def test_two_foreground_points_do_not_become_vehicle_when_wall_is_dense():
    d = np.array([29.9, 30.0, 54.9, 55.0, 55.1, 55.2, 55.3, 55.4])
    selected = CarlaCameraLiDARObjectDetector._depth_cluster(d, np.arange(len(d)))
    assert selected.size == 0


def test_flat_ground_plane_is_not_published_as_vehicle():
    detector = CarlaCameraLiDARObjectDetector(_Cal(), CameraLiDARDetectionConfig(max_range=55), model=_Predictor())
    xs, ys = np.meshgrid(np.linspace(7, 13, 23), np.linspace(-4, 4, 23))
    xyz = np.column_stack((xs.ravel(), ys.ravel(), np.zeros(xs.size)))
    proposals = detector.detect(np.zeros((200, 200, 3), dtype=np.uint8), xyz)
    assert proposals == []
    assert detector.last_diagnostics['extended_ground_rejections'] >= 1


def test_depth_spread_outlier_cannot_create_track():
    tr = CarlaMultiObjectTracker(min_hits=2)
    obs = _obs(14)
    obs['depth_spread_m'] = 8.0
    assert tr.update([obs], 0.0) == []
    assert tr.next_track_id == 0


def test_high_overlap_with_far_wall_does_not_spawn_ghost_id():
    tr = CarlaMultiObjectTracker(min_hits=2)
    for i in range(3):
        tr.update([_obs(15 + i*.1)], i*.1)
    assert tr.next_track_id == 1
    tr.update([_obs(55.0)], 0.3)
    assert tr.next_track_id == 1
    assert len(tr.tracks) == 1


def test_3d_mahalanobis_rejects_large_jump_despite_good_camera_box():
    tr = CarlaMultiObjectTracker(min_hits=2)
    for i in range(3):
        tr.update([_obs(15 + i*.1)], i*.1)
    costs = tr._build_cost_matrix([_obs(65.0)], [np.array([65.0, 0.0, 0.0])])
    assert np.isinf(costs[0, 0])


def test_two_distinct_simultaneous_vehicles_remain_separate():
    tr = CarlaMultiObjectTracker(min_hits=2)
    first = _obs(15, bbox=(100, 80, 150, 140))
    second = _obs(20, y=3.5, bbox=(300, 80, 350, 140))
    tr.update([first, second], 0.0)
    out = tr.update([_obs(15.2, bbox=(102, 80, 152, 140)), _obs(20.2, y=3.5, bbox=(302, 80, 352, 140))], 0.1)
    assert len(out) == 2
    assert len({t['track_id'] for t in out}) == 2


def test_missed_frame_keeps_identity_without_publishing_blind_prediction():
    tr = CarlaMultiObjectTracker(min_hits=2, max_missed=3, publish_max_missed=0)
    tr.update([_obs(15)], 0.0)
    assert len(tr.update([_obs(15.1)], 0.1)) == 1
    assert tr.update([], 0.2) == []
    out = tr.update([_obs(15.3)], 0.3)
    assert len(out) == 1
    assert out[0]['track_id'] == 0
