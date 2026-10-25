import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

from lvimot.temporal_fusion.pipeline import CarlaLVIMOTCore
from lvimot.temporal_fusion.runtime_limits import DEFAULT_LIMITS
from lvimot.tracking_4d.multimodal_object_tracking.kalman_tracker import CarlaMultiObjectTracker


def _core_stub():
    core = CarlaLVIMOTCore.__new__(CarlaLVIMOTCore)
    core.limits = DEFAULT_LIMITS
    core.object_sensor_range = 100.0
    core._previous_lidar_range_signature = None
    core._last_image_hash = None
    core._last_lidar_hash = None
    core._stationary_streak = 0
    core._stationary_release_streak = 0
    core._stationary_active = False
    core._last_stationary_lidar_metric = None
    core._current_lidar_stationary_overlap = 0.0
    return core


def _synthetic_static_cloud():
    az = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    elev = np.deg2rad(np.linspace(-12.0, 8.0, 16))
    pts = []
    for e in elev:
        for a in az:
            # Range varies with bearing so translating the sensor changes a
            # broad portion of the angular range signature.
            r = 25.0 + 6.0 * np.sin(2.0 * a) + 2.0 * np.cos(3.0 * e)
            ce = np.cos(e)
            pts.append([r * ce * np.cos(a), r * ce * np.sin(a), r * np.sin(e), 1.0])
    return np.asarray(pts, dtype=np.float64)


def test_v19_raw_lidar_signature_is_stable_but_rejects_ego_translation():
    cloud = _synthetic_static_cloud()
    core = _core_stub()
    metric0, overlap0 = core._lidar_stationary_metric(cloud)
    assert metric0 is None
    assert overlap0 == 0.0

    same = cloud.copy()
    same[:, :3] += np.array([0.005, -0.003, 0.002])
    metric_same, overlap_same = core._lidar_stationary_metric(same)
    assert overlap_same > 0.5
    assert metric_same is not None
    assert metric_same < DEFAULT_LIMITS.stationary_lidar_range_delta_m

    moved = cloud.copy()
    moved[:, 0] -= 0.8
    metric_move, overlap_move = core._lidar_stationary_metric(moved)
    assert overlap_move > 0.3
    assert metric_move is not None
    assert metric_move > metric_same


def test_v19_stationary_accepts_both_carla_acceleration_conventions():
    for accel in (
        np.array([0.0, 0.0, 9.80665]),
        np.array([0.0, 0.0, 0.0]),
    ):
        core = _core_stub()
        core._current_lidar_stationary_overlap = 0.85
        state = None
        for i in range(DEFAULT_LIMITS.stationary_confirm_frames):
            state = core._sensor_integrity_state(
                image_hash=bytes([i + 1]),
                lidar_hash=bytes([20 + i]),
                imu_motion={"gyro": np.zeros(3), "acceleration": accel},
                lidar_stationary_metric=0.03,
            )
        assert state is not None
        assert state["imu_quiet"] is True
        assert state["lidar_stationary"] is True
        assert state["stationary_active"] is True


def _obs(x, score=0.45, support=10):
    return {
        "location": [float(x), 0.0, 0.0],
        "dimensions": {"length": 4.2, "width": 1.8, "height": 1.5},
        "type": "vehicle",
        "class": "vehicle",
        "score": float(score),
        "camera_confidence": 0.10,
        "camera_evidence": True,
        "camera_support": int(support),
        "bbox": [100.0, 80.0, 160.0, 130.0],
    }


def test_v19_tracker_keeps_supported_weak_vehicle_and_bridges_one_gap():
    tracker = CarlaMultiObjectTracker(
        distance_threshold=2.75,
        max_missed=2,
        min_hits=2,
        new_track_min_score=0.38,
        publish_min_score=0.40,
        publish_max_missed=1,
        gap_publish_min_hits=4,
        min_camera_support=4,
    )
    assert tracker.update([_obs(10.0)], 0.0) == []
    assert len(tracker.update([_obs(10.1)], 0.1)) == 1
    tracker.update([_obs(10.2)], 0.2)
    tracker.update([_obs(10.3)], 0.3)
    # One missed detector frame is bridged only after the track is established.
    bridged = tracker.update([], 0.4)
    assert len(bridged) == 1
    # A second miss exceeds the publication bridge even though the internal
    # track is retained for reassociation.
    assert tracker.update([], 0.5) == []
