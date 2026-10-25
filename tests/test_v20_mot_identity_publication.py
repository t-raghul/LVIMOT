import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

from lvimot.temporal_fusion.runtime_limits import DEFAULT_LIMITS
from lvimot.tracking_4d.multimodal_object_tracking.kalman_tracker import CarlaMultiObjectTracker
from lvimot.preprocessing.camera_lidar_detection import CarlaCameraLiDARObjectDetector


def _obs(x, bbox, score=0.58, support=12):
    return {
        "location": [float(x), 0.0, 0.0],
        "dimensions": {"length": 4.2, "width": 1.8, "height": 1.5},
        "type": "vehicle",
        "class": "vehicle",
        "score": float(score),
        "camera_confidence": 0.16,
        "camera_evidence": True,
        "camera_support": int(support),
        "bbox": [float(v) for v in bbox],
    }


def _tracker():
    return CarlaMultiObjectTracker(
        distance_threshold=DEFAULT_LIMITS.tracker_distance_threshold,
        max_missed=DEFAULT_LIMITS.tracker_max_missed,
        min_hits=DEFAULT_LIMITS.tracker_min_hits,
        new_track_min_score=DEFAULT_LIMITS.tracker_new_track_min_score,
        publish_min_score=DEFAULT_LIMITS.tracker_publish_min_score,
        publish_max_missed=DEFAULT_LIMITS.tracker_publish_max_missed,
        gap_publish_min_hits=DEFAULT_LIMITS.tracker_gap_publish_min_hits,
        min_camera_support=DEFAULT_LIMITS.tracker_min_camera_support,
    )


def test_v20_camera_geometry_preserves_identity_when_partial_lidar_center_jumps():
    tracker = _tracker()
    # Same image object, but the LiDAR foreground surface centre shifts by more
    # than the nominal 2.75 m gate. The RGB box should preserve one identity.
    assert tracker.update([_obs(20.0, [100, 80, 180, 140])], 0.0) == []
    # V22 may publish a strongly LiDAR-supported camera track after the second
    # consistent observation; the purpose of this regression remains identity
    # preservation across the later LiDAR-centre jump.
    second = tracker.update([_obs(21.0, [102, 80, 182, 140])], 0.1)
    assert len(second) == 1
    assert second[0]["track_id"] == 0
    published = tracker.update([_obs(23.2, [104, 81, 184, 141])], 0.2)
    assert len(published) == 1
    assert published[0]["track_id"] == 0
    assert tracker.next_track_id == 1


def test_v20_missed_tracks_stay_internal_but_are_not_published():
    tracker = _tracker()
    for i in range(4):
        tracker.update([_obs(15.0 + 0.1 * i, [200 + i, 90, 260 + i, 145])], i * 0.1)
    assert len(tracker.get_tracks()) == 1
    assert tracker.update([], 0.4) == []
    # Internal state is preserved for reassociation.
    assert len(tracker.tracks) == 1
    reacquired = tracker.update([_obs(15.5, [205, 90, 265, 145])], 0.5)
    assert len(reacquired) == 1
    assert reacquired[0]["track_id"] == 0


def test_v20_duplicate_publications_are_suppressed_conservatively():
    tracker = _tracker()
    a = _obs(12.0, [300, 100, 370, 155], score=0.62, support=16)
    b = _obs(12.4, [304, 102, 372, 156], score=0.54, support=10)
    # Force two internal tracks as if duplicate low-confidence proposals had
    # survived detector NMS, then mature both.
    tracker._create_track(a, 0.0, np.array(a["location"], dtype=np.float64))
    tracker._create_track(b, 0.0, np.array(b["location"], dtype=np.float64))
    for tr in tracker.tracks:
        tr.hits = 4
        tr.score = max(tr.score, 0.55)
        tr.camera_support = 12
        tr.missed = 0
    published = tracker.get_tracks()
    assert len(published) == 1


def test_v20_detector_duplicate_suppression_keeps_distinct_adjacent_cars():
    detector = CarlaCameraLiDARObjectDetector.__new__(CarlaCameraLiDARObjectDetector)
    dup1 = {"bbox": [10, 10, 80, 60], "location": [15.0, 1.0, 0.0], "score": 0.8, "camera_support": 20}
    dup2 = {"bbox": [12, 11, 81, 61], "location": [15.4, 1.1, 0.0], "score": 0.7, "camera_support": 12}
    other = {"bbox": [90, 10, 150, 60], "location": [18.0, -2.0, 0.0], "score": 0.75, "camera_support": 18}
    kept = detector._suppress_duplicate_proposals([dup1, dup2, other])
    assert len(kept) == 2
    assert dup1 in kept
    assert other in kept
