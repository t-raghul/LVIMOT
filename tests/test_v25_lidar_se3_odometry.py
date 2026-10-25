import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

from scipy.spatial.transform import Rotation

from lvimot.temporal_fusion.sliding_window_factor_graph import LiDARRelativePoseFactor
from lvimot.temporal_fusion.pipeline import CarlaLVIMOTCore
from lvimot.temporal_fusion.runtime_limits import DEFAULT_LIMITS
from lvimot.tracking_4d.static_scene_support.feature_correspondences import CarlaTemporalLiDARMatcher


def test_relative_pose_factor_zero_at_measurement():
    R_i = Rotation.from_euler("xyz", [0.01, -0.02, 0.03]).as_matrix()
    p_i = np.array([1.0, 2.0, 0.3])
    delta_R = Rotation.from_euler("xyz", [0.005, 0.002, 0.04]).as_matrix()
    delta_t = np.array([0.45, -0.03, 0.01])
    R_j = R_i @ delta_R
    p_j = p_i + R_i @ delta_t
    factor = LiDARRelativePoseFactor(delta_R, delta_t)
    residual = factor.compute_residual(R_i, p_i, R_j, p_j)
    assert np.allclose(residual, 0.0, atol=1e-8)


def _feature(center, normal, planarity=0.95):
    n = np.asarray(normal, dtype=np.float64)
    n /= np.linalg.norm(n)
    c = np.asarray(center, dtype=np.float64)
    return {"center": c, "point": c.copy(), "normal": n, "planarity": float(planarity)}


def test_planar_se3_registration_recovers_current_to_previous_transform():
    rng = np.random.default_rng(25)
    core = CarlaLVIMOTCore.__new__(CarlaLVIMOTCore)
    core.limits = DEFAULT_LIMITS
    core.lidar_tracker = CarlaTemporalLiDARMatcher(
        max_center_distance=1.0,
        min_normal_cos=0.80,
        max_planarity_diff=0.30,
        max_features=600,
    )
    core._previous_pose_for_lidar = np.zeros(6, dtype=np.float64)

    true_R = Rotation.from_euler("xyz", [0.004, -0.006, 0.025]).as_matrix()
    true_t = np.array([0.42, -0.04, 0.015], dtype=np.float64)

    prev_features = []
    curr_features = []
    for _ in range(120):
        p_prev = np.array(
            [rng.uniform(5.0, 35.0), rng.uniform(-12.0, 12.0), rng.uniform(-0.5, 4.0)]
        )
        n_prev = rng.normal(size=3)
        n_prev /= np.linalg.norm(n_prev)
        # p_prev = true_R @ p_curr + true_t
        p_curr = true_R.T @ (p_prev - true_t)
        n_curr = true_R.T @ n_prev
        prev_features.append(_feature(p_prev, n_prev))
        curr_features.append(_feature(p_curr, n_curr))

    core._previous_planar_ego = prev_features
    # Slightly imperfect IMU/graph prediction, still inside matcher gate.
    predicted_R = Rotation.from_euler("xyz", [0.006, -0.004, 0.020]).as_matrix()
    predicted_t = np.array([0.36, -0.02, 0.01])
    pose_prediction = np.hstack([Rotation.from_matrix(predicted_R).as_euler("xyz"), predicted_t])

    result = core._lidar_se3_relative_measurement(curr_features, pose_prediction)
    assert result["accepted"] is True
    assert result["matches"] >= DEFAULT_LIMITS.lidar_se3_min_matches
    assert result["rmse_m"] is not None and result["rmse_m"] < 0.02
    assert np.linalg.norm(result["delta_t"] - true_t) < 0.04
    rot_err = Rotation.from_matrix(true_R.T @ result["delta_R"]).magnitude()
    assert rot_err < 0.01


def test_factor_graph_uses_lidar_relative_pose():
    from lvimot.temporal_fusion.sliding_window_factor_graph import CarlaFactorGraph

    fg = CarlaFactorGraph(robust_loss="linear")
    fg.add_keyframe(0, 0.0)
    fg.add_keyframe(1, 0.1)
    fg.add_pose_prior(0, np.zeros(6), weight_rotation=100.0, weight_position=100.0)
    delta_R = Rotation.from_euler("xyz", [0.0, 0.0, 0.02]).as_matrix()
    delta_t = np.array([0.45, -0.02, 0.01])
    fg.add_lidar_relative_factor(0, 1, delta_R, delta_t, weight_translation=15.0, weight_rotation=20.0)
    poses = {0: np.zeros(6), 1: np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.0])}
    velocities = {0: np.zeros(3), 1: np.zeros(3)}
    res = fg.optimize(poses, velocities, max_nfev=20)
    p1 = np.asarray(res["optimized_poses"][1])
    assert np.linalg.norm(p1[3:] - delta_t) < 0.03
    assert abs(p1[2] - 0.02) < 0.01


def test_v251_motion_icp_seed_survives_road_plane_degeneracy():
    """Road-only normals must not overwrite the full-rank V24 rigid ICP motion."""
    core = CarlaLVIMOTCore.__new__(CarlaLVIMOTCore)
    core.limits = DEFAULT_LIMITS
    core.lidar_tracker = CarlaTemporalLiDARMatcher(
        max_center_distance=1.0,
        min_normal_cos=0.80,
        max_planarity_diff=0.30,
        max_features=600,
    )
    core._previous_pose_for_lidar = np.zeros(6, dtype=np.float64)

    true_R = Rotation.from_euler("xyz", [0.0, 0.0, 0.01]).as_matrix()
    true_t = np.array([0.48, -0.03, 0.0], dtype=np.float64)
    core._last_lidar_motion_transform = {
        "accepted": True,
        "delta_R": true_R.copy(),
        "delta_t": true_t.copy(),
        "translation_m": float(np.linalg.norm(true_t)),
        "rotation_rad": float(Rotation.from_matrix(true_R).magnitude()),
        "point_rmse_m": 0.08,
        "correspondences": 500,
    }

    prev_features = []
    curr_features = []
    # All normals are vertical: point-to-plane alone cannot observe x/y motion.
    for x in np.linspace(6.0, 36.0, 8):
        for y in np.linspace(-8.0, 8.0, 6):
            p_prev = np.array([x, y, 0.0], dtype=np.float64)
            p_curr = true_R.T @ (p_prev - true_t)
            prev_features.append(_feature(p_prev, [0.0, 0.0, 1.0]))
            curr_features.append(_feature(p_curr, true_R.T @ np.array([0.0, 0.0, 1.0])))

    core._previous_planar_ego = prev_features
    # Deliberately bad tangent-direction graph/IMU prediction. V25's plane-only
    # solve could preserve/invent this because the road normal cannot constrain x/y.
    pose_prediction = np.array([0.0, 0.0, 0.0, 0.05, 0.30, 0.0], dtype=np.float64)
    result = core._lidar_se3_relative_measurement(
        curr_features,
        pose_prediction,
        expected_translation_m=float(np.linalg.norm(true_t[:2])),
        expected_rotation_rad=float(Rotation.from_matrix(true_R).magnitude()),
    )

    assert result["accepted"] is True
    assert result["source"] == "v24_rigid_icp_validated_by_planes"
    assert np.linalg.norm(np.asarray(result["delta_t"]) - true_t) < 1e-10
    rot_err = Rotation.from_matrix(true_R.T @ np.asarray(result["delta_R"])).magnitude()
    assert rot_err < 1e-10


def test_v251_rejects_bad_rigid_icp_point_quality():
    core = CarlaLVIMOTCore.__new__(CarlaLVIMOTCore)
    core.limits = DEFAULT_LIMITS
    core.lidar_tracker = CarlaTemporalLiDARMatcher(
        max_center_distance=1.0,
        min_normal_cos=0.80,
        max_planarity_diff=0.30,
        max_features=600,
    )
    core._previous_pose_for_lidar = np.zeros(6, dtype=np.float64)
    true_t = np.array([0.4, 0.0, 0.0], dtype=np.float64)
    core._last_lidar_motion_transform = {
        "accepted": True,
        "delta_R": np.eye(3),
        "delta_t": true_t.copy(),
        "translation_m": 0.4,
        "rotation_rad": 0.0,
        "point_rmse_m": 0.9,
        "correspondences": 500,
    }
    prev_features = []
    curr_features = []
    for x in np.linspace(5.0, 25.0, 6):
        for y in np.linspace(-6.0, 6.0, 6):
            p_prev = np.array([x, y, 0.0])
            prev_features.append(_feature(p_prev, [0.0, 0.0, 1.0]))
            curr_features.append(_feature(p_prev - true_t, [0.0, 0.0, 1.0]))
    core._previous_planar_ego = prev_features
    result = core._lidar_se3_relative_measurement(
        curr_features,
        np.zeros(6),
        expected_translation_m=0.4,
        expected_rotation_rad=0.0,
    )
    assert result["accepted"] is False
    assert result["reason"] == "point_quality"


def test_v251_motion_metric_retains_full_current_to_previous_transform():
    rng = np.random.default_rng(251)
    core = CarlaLVIMOTCore.__new__(CarlaLVIMOTCore)
    core.limits = DEFAULT_LIMITS
    core.object_sensor_range = 100.0
    core._previous_motion_lidar_xyz = None
    core._last_lidar_motion_transform = None

    # Structured static roadside geometry with height diversity.
    prev = np.column_stack([
        rng.uniform(5.0, 45.0, 2500),
        rng.uniform(-15.0, 15.0, 2500),
        rng.uniform(0.0, 5.0, 2500),
    ])
    true_R = Rotation.from_euler("xyz", [0.002, -0.003, 0.012]).as_matrix()
    true_t = np.array([0.46, -0.025, 0.008], dtype=np.float64)
    curr = (true_R.T @ (prev - true_t).T).T

    first = core._lidar_translation_motion_metric(prev)
    assert first[0] is None
    translation, rotation, used = core._lidar_translation_motion_metric(curr)
    assert used >= 60
    assert translation is not None and abs(translation - np.linalg.norm(true_t[:2])) < 0.08
    assert rotation is not None and abs(rotation - Rotation.from_matrix(true_R).magnitude()) < 0.02
    seed = core._last_lidar_motion_transform
    assert seed is not None and seed["accepted"] is True
    assert np.linalg.norm(np.asarray(seed["delta_t"]) - true_t) < 0.08
    rot_err = Rotation.from_matrix(true_R.T @ np.asarray(seed["delta_R"])).magnitude()
    assert rot_err < 0.02


def test_v253_high_confidence_motion_seed_gets_strong_translation_weight():
    core = CarlaLVIMOTCore.__new__(CarlaLVIMOTCore)
    core.limits = DEFAULT_LIMITS
    core.lidar_tracker = CarlaTemporalLiDARMatcher(
        max_center_distance=1.0, min_normal_cos=0.80, max_planarity_diff=0.30, max_features=600
    )
    core._previous_pose_for_lidar = np.zeros(6, dtype=np.float64)
    true_t = np.array([0.42, 0.0, 0.0], dtype=np.float64)
    core._last_lidar_motion_transform = {
        "accepted": True,
        "delta_R": np.eye(3),
        "delta_t": true_t.copy(),
        "translation_m": float(np.linalg.norm(true_t)),
        "rotation_rad": 0.0,
        "point_rmse_m": 0.03,
        "correspondences": 450,
    }
    prev_features, curr_features = [], []
    for x in np.linspace(5.0, 35.0, 8):
        for y in np.linspace(-8.0, 8.0, 6):
            q = np.array([x, y, 0.0], dtype=np.float64)
            prev_features.append(_feature(q, [0.0, 0.0, 1.0]))
            curr_features.append(_feature(q - true_t, [0.0, 0.0, 1.0]))
    core._previous_planar_ego = prev_features
    out = core._lidar_se3_relative_measurement(
        curr_features, np.zeros(6), expected_translation_m=0.42, expected_rotation_rad=0.0
    )
    assert out["accepted"] is True
    assert out["weight_translation"] >= 30.0
    assert np.allclose(out["delta_t"], true_t, atol=1e-10)
