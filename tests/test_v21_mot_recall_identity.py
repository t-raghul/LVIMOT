import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

from lvimot.tracking_4d.multimodal_object_tracking.kalman_tracker import CarlaMultiObjectTracker, KalmanTrack


def _obs(x, bbox, score=0.56, support=14, allow_new=True, fallback=False):
    return {
        "location": [float(x), 0.0, 0.0],
        "dimensions": {"length": 4.2, "width": 1.8, "height": 1.5},
        "type": "vehicle",
        "class": "vehicle",
        "score": float(score),
        "camera_confidence": 0.14 if not fallback else 0.0,
        "camera_evidence": not fallback,
        "camera_support": int(support),
        "bbox": [float(v) for v in bbox],
        "allow_new_track": bool(allow_new),
        "lidar_fallback": bool(fallback),
    }


def _v21_tracker():
    return CarlaMultiObjectTracker(
        distance_threshold=2.75,
        max_missed=3,
        min_hits=2,
        new_track_min_score=0.40,
        publish_min_score=0.42,
        publish_max_missed=0,
        gap_publish_min_hits=5,
        min_camera_support=6,
        fast_confirm_hits=2,
        fast_confirm_min_score=0.42,
        fast_confirm_min_support=8,
        fallback_min_hits=3,
        fallback_min_support=18,
    )


def test_v21_strong_second_observation_can_confirm_track():
    tracker = _v21_tracker()
    assert tracker.update([_obs(10.0, [100, 80, 160, 130])], 0.0) == []
    out = tracker.update([_obs(10.2, [102, 80, 162, 130])], 0.1)
    assert len(out) == 1
    assert out[0]["track_id"] == 0
    assert out[0]["hits"] == 2


def test_v21_weak_two_hit_candidate_waits_for_third_observation():
    tracker = _v21_tracker()
    weak = lambda x, dx=0: _obs(x, [200 + dx, 90, 255 + dx, 140], score=0.415, support=7)
    assert tracker.update([weak(20.0)], 0.0) == []
    assert tracker.update([weak(20.1, 1)], 0.1) == []
    # A third supported observation may publish once the normal score/support
    # floor is met; here we strengthen the measurement slightly.
    out = tracker.update([_obs(20.2, [202, 90, 257, 140], score=0.47, support=9)], 0.2)
    assert len(out) == 1
    assert out[0]["track_id"] == 0


def test_v21_duplicate_unmatched_proposal_does_not_spawn_third_identity():
    tracker = _v21_tracker()
    a0 = _obs(12.0, [100, 80, 160, 135], score=0.62, support=16)
    b0 = _obs(26.0, [300, 80, 360, 135], score=0.60, support=15)
    tracker.update([a0, b0], 0.0)
    tracker.update([
        _obs(12.1, [101, 80, 161, 135], score=0.61, support=15),
        _obs(26.1, [301, 80, 361, 135], score=0.60, support=15),
    ], 0.1)
    assert tracker.next_track_id == 2

    # Third proposal is a duplicate of A. It can remain a detector candidate,
    # but must not create a new identity after A has already been matched.
    out = tracker.update([
        _obs(12.2, [102, 80, 162, 135], score=0.61, support=16),
        _obs(12.5, [104, 81, 164, 136], score=0.52, support=10),
        _obs(26.2, [302, 80, 362, 135], score=0.60, support=15),
    ], 0.2)
    assert tracker.next_track_id == 2
    assert len(out) == 2
    assert {item["track_id"] for item in out} == {0, 1}


def test_v21_lidar_fallback_supports_existing_track_but_never_creates_one():
    tracker = _v21_tracker()
    for i in range(3):
        out = tracker.update([
            _obs(10.0 + 0.1 * i, [100 + i, 80, 160 + i, 135], score=0.58, support=15),
            _obs(25.0 + 0.1 * i, [300 + i, 80, 360 + i, 135], score=0.57, support=15),
        ], i * 0.1)
    assert len(out) == 2
    ids = {item["track_id"] for item in out}

    # Camera sees A; current LiDAR geometry supports B. The fallback updates B
    # but carries allow_new_track=False.
    out = tracker.update([
        _obs(10.3, [103, 80, 163, 135], score=0.58, support=15),
        _obs(25.3, [303, 80, 363, 135], score=0.62, support=30, allow_new=False, fallback=True),
    ], 0.3)
    assert len(out) == 2
    assert {item["track_id"] for item in out} == ids
    assert sum(bool(item["lidar_fallback"]) for item in out) == 1

    empty = _v21_tracker()
    assert empty.update([
        _obs(18.0, [200, 80, 260, 135], score=0.7, support=40, allow_new=False, fallback=True)
    ], 0.0) == []
    assert empty.next_track_id == 0


def test_v21_missed_prediction_does_not_reintegrate_old_time_interval():
    obs = _obs(0.0, [100, 80, 160, 135], score=0.60, support=16)
    tr = KalmanTrack(0, obs, 0.0, world_position=np.array([0.0, 0.0, 0.0]))
    tr.state[3] = 1.0
    tracker = _v21_tracker()
    tracker.tracks = [tr]
    tracker.next_track_id = 1

    tracker.update([], 0.1)
    x1 = float(tracker.tracks[0].position[0])
    tracker.update([], 0.2)
    x2 = float(tracker.tracks[0].position[0])
    # With the V20 bug this second step used dt=0.2 after already consuming
    # 0.1 s, yielding about 0.3 m total. V21 advances the state timestamp, so
    # the two predictions are ~0.1 + 0.1 m.
    assert 0.08 <= x1 <= 0.12
    assert 0.18 <= x2 <= 0.22


def test_v21_two_car_stress_keeps_two_ids_through_camera_gap_and_duplicates():
    tracker = _v21_tracker()
    published_id_sets = []
    for frame in range(12):
        a = _obs(12.0 + 0.08 * frame, [100 + frame, 80, 160 + frame, 135], score=0.60, support=16)
        b = _obs(26.0 - 0.05 * frame, [300 - frame, 82, 360 - frame, 137], score=0.57, support=14)
        observations = [a, b]
        if frame in {5, 6}:
            observations = [a, _obs(
                26.0 - 0.05 * frame,
                [300 - frame, 82, 360 - frame, 137],
                score=0.64,
                support=32,
                allow_new=False,
                fallback=True,
            )]
        if frame in {8, 10}:
            observations.insert(1, _obs(
                12.3 + 0.08 * frame,
                [103 + frame, 81, 163 + frame, 136],
                score=0.50,
                support=10,
            ))
        out = tracker.update(observations, frame * 0.1)
        if out:
            published_id_sets.append({item["track_id"] for item in out})
            assert len(out) <= 2
    assert tracker.next_track_id == 2
    assert published_id_sets
    assert published_id_sets[-1] == {0, 1}
