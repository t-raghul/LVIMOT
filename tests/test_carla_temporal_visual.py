import os
import sys
import numpy as np


from lvimot.tracking_4d.static_scene_support.feature_correspondences import CarlaTemporalVisualTracker


def test_temporal_visual_tracking():
    tracker = CarlaTemporalVisualTracker(max_hamming_distance=30, ratio_threshold=0.8)
    pts_0 = np.array([[100, 100], [200, 150], [300, 200], [400, 250], [500, 300]], dtype=np.float64)
    np.random.seed(42)
    desc_0 = np.random.randint(0, 256, size=(5, 32), dtype=np.uint8)

    tracks_0 = tracker.update(pts_0, desc_0, frame_id=0, timestamp=0.0)
    assert len(tracks_0) == 5

    pts_1 = pts_0 + np.array([2.0, 1.0])
    desc_1 = desc_0.copy()
    desc_1[0, 0] ^= 1

    tracks_1 = tracker.update(pts_1, desc_1, frame_id=1, timestamp=0.1)
    assert len(tracks_1) == 5
