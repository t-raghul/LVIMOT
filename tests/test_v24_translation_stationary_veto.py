import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

from lvimot.temporal_fusion.pipeline import CarlaLVIMOTCore, cv2
from lvimot.temporal_fusion.runtime_limits import DEFAULT_LIMITS


def _bare_core():
    core = CarlaLVIMOTCore.__new__(CarlaLVIMOTCore)
    core.limits = DEFAULT_LIMITS
    core._previous_motion_gray = None
    core._previous_motion_lidar_xyz = None
    core._last_image_hash = None
    core._last_lidar_hash = None
    core._stationary_streak = 0
    core._stationary_release_streak = 0
    core._stationary_active = False
    core._last_stationary_lidar_metric = None
    core._current_lidar_stationary_overlap = 1.0
    core._current_object_range = lambda: 100.0
    return core


@pytest.mark.skipif(cv2 is None, reason="OpenCV required")
def test_visual_flow_veto_detects_constant_velocity_translation():
    core = _bare_core()
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    for y in range(20, 230, 24):
        for x in range(20, 310, 24):
            cv2.circle(image, (x, y), 2, (255, 255, 255), -1)
    first = core._visual_background_motion_metric(image)
    assert first[0] is None

    M = np.float32([[1, 0, 2.0], [0, 1, 0.0]])
    moved = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    median_px, moving_fraction, tracks = core._visual_background_motion_metric(moved)
    assert tracks >= DEFAULT_LIMITS.stationary_visual_min_tracks
    assert median_px is not None and median_px > DEFAULT_LIMITS.stationary_visual_flow_threshold_px
    assert moving_fraction > DEFAULT_LIMITS.stationary_visual_motion_fraction_threshold


@pytest.mark.skipif(cv2 is None, reason="OpenCV required")
def test_visual_flow_ignores_small_independently_moving_patch():
    core = _bare_core()
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    for y in range(20, 230, 24):
        for x in range(20, 310, 24):
            cv2.circle(image, (x, y), 2, (255, 255, 255), -1)
    # Add a textured moving-object patch occupying a minority of the image.
    for y in range(90, 130, 8):
        for x in range(130, 170, 8):
            cv2.circle(image, (x, y), 2, (180, 180, 180), -1)
    core._visual_background_motion_metric(image)

    moved = image.copy()
    moved[88:134, 128:176] = 0
    patch = image[88:134, 128:176]
    M = np.float32([[1, 0, 8.0], [0, 1, 0.0]])
    shifted = cv2.warpAffine(patch, M, (patch.shape[1], patch.shape[0]))
    moved[88:134, 128:176] = shifted
    median_px, _moving_fraction, tracks = core._visual_background_motion_metric(moved)
    assert tracks >= DEFAULT_LIMITS.stationary_visual_min_tracks
    assert median_px is not None and median_px < DEFAULT_LIMITS.stationary_visual_flow_threshold_px


def _synthetic_lidar(seed=7, n=5000):
    rng = np.random.default_rng(seed)
    # Vertical/static roadside structure; avoid a dominant translation-ambiguous ground plane.
    x = rng.uniform(5.0, 70.0, n)
    y = rng.uniform(-25.0, 25.0, n)
    z = rng.uniform(-0.5, 5.0, n)
    return np.column_stack([x, y, z, np.ones(n)])


def test_lidar_icp_translation_veto_detects_scan_translation():
    core = _bare_core()
    previous = _synthetic_lidar()
    assert core._lidar_translation_motion_metric(previous)[0] is None
    current = previous.copy()
    current[:, 0] -= 0.50
    translation, rotation, used = core._lidar_translation_motion_metric(current)
    assert used >= 60
    assert translation is not None and translation > 0.30
    assert rotation is not None and rotation < 0.03


def test_sensor_integrity_constant_velocity_motion_cannot_enter_stationary():
    core = _bare_core()
    imu = {"gyro": np.zeros(3), "acceleration": np.array([0.0, 0.0, 9.80665])}
    # Five genuinely zero-motion confirmations should activate stationary mode.
    state = None
    for _ in range(DEFAULT_LIMITS.stationary_confirm_frames):
        state = core._sensor_integrity_state(
            b"img", b"lidar", imu, 0.0,
            visual_motion_px=0.02,
            visual_motion_fraction=0.01,
            visual_motion_tracks=120,
            lidar_translation_m=0.005,
            lidar_rotation_rad=0.001,
            lidar_motion_correspondences=500,
        )
    assert state is not None and state["stationary_active"] is True

    # A single strong translation observation must release immediately even
    # though gyro/acceleration still look exactly like constant-speed driving.
    state = core._sensor_integrity_state(
        b"img2", b"lidar2", imu, 0.01,
        visual_motion_px=1.2,
        visual_motion_fraction=0.75,
        visual_motion_tracks=120,
        lidar_translation_m=0.45,
        lidar_rotation_rad=0.001,
        lidar_motion_correspondences=500,
    )
    assert state["translation_motion_veto"] is True
    assert state["stationary_candidate"] is False
    assert state["stationary_active"] is False
