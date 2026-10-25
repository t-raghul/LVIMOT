import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

from lvimot.tracking_4d.multimodal_object_tracking.kalman_tracker import CarlaMultiObjectTracker


def _obs(x, bbox, score=0.27, support=15, conf=0.17, fallback=False, allow_new=True):
    return {
        "location": [float(x), 0.0, 0.0],
        "dimensions": {"length": 4.2, "width": 1.8, "height": 1.5},
        "type": "vehicle",
        "class": "vehicle",
        "score": float(score),
        "camera_confidence": float(conf) if not fallback else 0.0,
        "camera_evidence": not fallback,
        "camera_support": int(support),
        "bbox": [float(v) for v in bbox],
        "allow_new_track": bool(allow_new),
        "lidar_fallback": bool(fallback),
    }


def _tracker():
    return CarlaMultiObjectTracker(
        distance_threshold=2.75,
        max_missed=3,
        min_hits=2,
        new_track_min_score=0.20,
        publish_min_score=0.42,
        publish_max_missed=0,
        min_camera_support=6,
        fast_confirm_hits=2,
        fast_confirm_min_score=0.42,
        fast_confirm_min_support=8,
        weak_confirm_min_score=0.22,
        weak_confirm_min_camera_confidence=0.06,
        weak_confirm_min_support=10,
        weak_confirm_hits=2,
        static_suppress_min_hits=5,
        static_suppress_min_confidence=0.85,
        fallback_min_hits=3,
        fallback_min_support=18,
    )


def test_v22_distant_weak_camera_box_confirms_with_repeated_lidar_support():
    tr = _tracker()
    assert tr.update([_obs(78.0, [518, 187, 534, 199])], 0.0) == []
    out = tr.update([_obs(77.7, [517, 187, 534, 200])], 0.1)
    assert len(out) == 1
    assert out[0]["hits"] == 2
    assert out[0]["score"] < 0.42  # published through temporal LiDAR-backed confirmation


def test_v22_weak_candidate_without_lidar_support_does_not_publish():
    tr = _tracker()
    assert tr.update([_obs(78.0, [518, 187, 534, 199], support=5)], 0.0) == []
    assert tr.update([_obs(77.8, [517, 187, 534, 199], support=5)], 0.1) == []


def test_v22_lidar_fallback_uses_3d_geometry_even_with_bad_projected_bbox():
    tr = _tracker()
    for i in range(3):
        tr.update([_obs(40.0 + 0.1 * i, [100+i, 80, 160+i, 135], score=0.60, support=20, conf=0.5)], i * 0.1)
    # Current LiDAR supports the established track but the projected cluster bbox
    # is deliberately unrelated. Fallback association must rely on 3-D geometry.
    out = tr.update([
        _obs(40.3, [500, 20, 540, 60], score=0.65, support=30, fallback=True, allow_new=False)
    ], 0.3)
    assert len(out) == 1
    assert out[0]["track_id"] == 0
    assert out[0]["lidar_fallback"] is True


def test_v22_static_world_hypothesis_is_hidden_only_in_stationary_mot_mode():
    tr = _tracker()
    out = []
    for i in range(7):
        out = tr.update([
            _obs(50.94 + 0.01 * (i % 2), [430, 189, 462, 206], score=0.75, support=25, conf=0.5)
        ], i * 0.1, suppress_static_publication=True)
    assert out == []
    assert len(tr.tracks) == 1  # kept internally / can later become dynamic
    assert tr.tracks[0].motion_state.value == "STATIC"


def test_v22_moving_track_remains_published_in_stationary_ego_scene():
    tr = _tracker()
    out = []
    for i in range(7):
        out = tr.update([
            _obs(80.0 - 1.0 * i, [520-i*3, 187, 538-i*3, 202], score=0.50, support=15, conf=0.18)
        ], i * 0.1, suppress_static_publication=True)
    assert len(out) == 1
    assert out[0]["motion_state"] == "DYNAMIC"
