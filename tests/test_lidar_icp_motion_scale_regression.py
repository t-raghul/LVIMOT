"""Synthetic offline LiDAR motion regression, no CARLA or GT estimator inputs."""
from types import SimpleNamespace

import numpy as np

from lvimot.temporal_fusion.pipeline import CarlaLVIMOTCore
from lvimot.temporal_fusion.runtime_limits import DEFAULT_LIMITS


def _repeatable_urban_geometry(seed=42):
    rng = np.random.default_rng(seed)
    sections = []
    for side in (-1, 1):
        x = rng.uniform(-25, 40, 9000)
        z = rng.uniform(0.2, 12, 9000)
        y = side * 8 + rng.normal(0, 0.05, len(x)) + 0.3 * np.cos(x * 1.7) * np.exp(-z / 3)
        sections.append(np.column_stack([x, y, z]))
    for ox, oy in [(6, 4), (18, -5), (30, 4), (-16, 4)]:
        t = rng.uniform(0, 2 * np.pi, 1200)
        rad = rng.normal(0.22, 0.015, len(t))
        z = rng.uniform(0.1, 8, len(t))
        sections.append(np.column_stack([ox + rad * np.cos(t), oy + rad * np.sin(t), z]))
    return np.concatenate(sections)


def _measure_shift(shift):
    matcher = CarlaLVIMOTCore.__new__(CarlaLVIMOTCore)
    matcher.limits = DEFAULT_LIMITS
    matcher._previous_motion_lidar_xyz = None
    points = _repeatable_urban_geometry()
    first = matcher._lidar_translation_motion_metric(points)
    assert first[0] is None
    moving = points.copy()
    moving[:, 0] -= shift
    output = matcher._lidar_translation_motion_metric(moving)
    return output, matcher._last_lidar_motion_transform


def test_static_scan_cannot_create_false_motion():
    reported, state = _measure_shift(0.0)
    assert reported[0] < 0.01
    assert abs(state['delta_t'][0]) < 0.01


def test_motion_estimator_handles_nearest_neighbor_false_zero_minimum():
    """Previously a full 1 m translation was estimated as ~0.02 m."""
    reported, state = _measure_shift(1.0)
    assert reported[0] is not None
    assert abs(float(state['delta_t'][0]) - 1.0) < 0.15
    assert state['point_rmse_m'] < 0.25


def test_motion_estimator_recovers_medium_translation():
    reported, state = _measure_shift(0.6)
    assert abs(float(state['delta_t'][0]) - 0.6) < 0.15
