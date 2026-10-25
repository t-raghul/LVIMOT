import os
import sys

import numpy as np


from lvimot.temporal_fusion.runtime_limits import DEFAULT_LIMITS
from lvimot.temporal_fusion.sliding_window_factor_graph import CarlaSlidingWindowEstimator
from lvimot.tracking_4d.static_scene_support.feature_correspondences import CarlaTemporalLiDARMatcher


def _feat(x, y, z=0.0):
    return {
        "point": np.array([x, y, z], dtype=np.float64),
        "center": np.array([x, y, z], dtype=np.float64),
        "normal": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "planarity": 0.6,
    }


def test_v12_runtime_profile_is_bounded_for_live_target():
    assert DEFAULT_LIMITS.max_visual_features <= 700
    assert DEFAULT_LIMITS.max_planar_features_for_temporal <= 600
    assert DEFAULT_LIMITS.max_planar_landmarks <= 800
    assert DEFAULT_LIMITS.max_lidar_planar_candidates == 16000
    assert DEFAULT_LIMITS.sliding_window_size == 4
    assert DEFAULT_LIMITS.optimizer_max_nfev == 2
    assert DEFAULT_LIMITS.map_point_stride >= 4


def test_v12_temporal_lidar_vectorized_match_preserves_identity():
    matcher = CarlaTemporalLiDARMatcher(max_center_distance=0.5, max_features=32, max_landmarks=32)
    prev = [_feat(5.0, 0.0), _feat(10.0, 2.0)]
    curr = [_feat(5.05, 0.02), _feat(10.03, 2.01)]
    matches = matcher.match_planar_features(prev, curr)
    assert set(matches) == {(0, 0), (1, 1)}


def test_v12_optimizer_accepts_two_evaluation_budget():
    estimator = CarlaSlidingWindowEstimator(window_size=4, optimizer_max_nfev=2)
    assert estimator.optimizer_max_nfev == 2
