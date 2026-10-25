import os
import sys
import numpy as np


from lvimot.tracking_4d.static_scene_support.feature_correspondences import CarlaTemporalVisualTracker
from lvimot.preprocessing.lidar_object_measurements import CarlaLiDARObjectDetector
from lvimot.preprocessing.imu_preintegration import IMUPreintegrator
from lvimot.output.evaluation.metrics import CarlaEvaluator


def test_visual_tracker_caps_large_feature_set():
    rng = np.random.default_rng(10)
    tracker = CarlaTemporalVisualTracker(max_features=400, max_tracks=500)
    pts = rng.uniform([0, 0], [1242, 375], size=(2500, 2))
    desc = rng.integers(0, 256, size=(2500, 32), dtype=np.uint8)
    out0 = tracker.update(pts, desc, 0, 0.0)
    assert len(out0) <= 500
    out1 = tracker.update(pts + 1.0, desc, 1, 0.1)
    assert len(out1) <= 500
    assert len(tracker.tracks) <= 500


def test_lidar_object_detector_is_sensor_only_and_bounded():
    rng = np.random.default_rng(1)
    cluster = rng.normal(loc=[10.0, 2.0, 0.0], scale=[0.7, 0.4, 0.4], size=(300, 3))
    intensity = np.ones((len(cluster), 1))
    detector = CarlaLiDARObjectDetector(max_proposals=5, min_points=5)
    proposals = detector.detect(np.hstack([cluster, intensity]))
    assert len(proposals) <= 5
    assert len(proposals) >= 1
    assert 'actor_id' not in proposals[0]


def test_imu_rotation_increment_is_orthonormal():
    p = IMUPreintegrator()
    m0 = {'dt': 0.1, 'timestamp': 0.0, 'acceleration': np.zeros(3), 'gyro': np.array([0.2, -0.1, 0.3])}
    m1 = {'dt': 0.1, 'timestamp': 0.1, 'acceleration': np.zeros(3), 'gyro': np.array([0.2, -0.1, 0.3])}
    R = p.integrate(m0, m1)['delta_R']
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)


def test_localization_reports_raw_and_aligned_ate():
    evaluator = CarlaEvaluator()
    gt = [np.array([0, 0, 0, i, 0, 0], dtype=float) for i in range(4)]
    est = [np.array([0, 0, 0, i + 100, 50, 0], dtype=float) for i in range(4)]
    result = evaluator.evaluate_localization(est, gt)
    assert result['raw_ate_rmse'] > 100
    assert result['aligned_ate_rmse'] < 1e-9
    assert result['rpe_rmse'] < 1e-9


def test_core_two_frame_sensor_only_smoke(tmp_path):
    import json
    from lvimot.inputs.sensor_calibration import CarlaCalibration
    from lvimot.temporal_fusion.pipeline import CarlaLVIMOTCore
    from lvimot.temporal_fusion.runtime_limits import PipelineLimits

    calibration = {
        'camera': {
            'width': 320,
            'height': 180,
            'fov': 90.0,
            'intrinsic': [[160.0, 0.0, 160.0], [0.0, 160.0, 90.0], [0.0, 0.0, 1.0]],
            'transform': {'location': {'x': 1.5, 'y': 0.0, 'z': 1.5}, 'rotation': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}},
        },
        'lidar': {
            'channels': 64,
            'range': 100.0,
            'transform': {'location': {'x': 0.0, 'y': 0.0, 'z': 1.8}, 'rotation': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}},
        },
        'imu': {
            'transform': {'location': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'rotation': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}},
        },
        'lidar_to_camera_sensor': np.eye(4).tolist(),
    }
    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps(calibration))
    limits = PipelineLimits(max_visual_features=100, max_visual_tracks=120, max_planar_features_for_temporal=100, max_planar_landmarks=150, max_object_proposals=10)
    core = CarlaLVIMOTCore(CarlaCalibration(str(path)), limits=limits, voxel_size=0.3)

    image = np.zeros((180, 320, 3), dtype=np.uint8)
    # Add corners for FAST/BRIEF.
    image[40:140:20, 40:280:20] = 255
    rng = np.random.default_rng(4)
    ground = np.column_stack([rng.uniform(2, 20, 500), rng.uniform(-8, 8, 500), rng.normal(-1.2, 0.03, 500), np.ones(500)])
    obj = np.column_stack([rng.normal(10, 0.5, 120), rng.normal(2, 0.3, 120), rng.normal(0, 0.3, 120), np.ones(120)])
    lidar = np.vstack([ground, obj]).astype(np.float32)
    imu0 = {'accelerometer': {'x': 0.0, 'y': 0.0, 'z': 9.80665}, 'gyroscope': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'timestamp': 0.0}
    imu1 = {'accelerometer': {'x': 0.0, 'y': 0.0, 'z': 9.80665}, 'gyroscope': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'timestamp': 0.1}
    r0 = core.process(0, 0.0, image, lidar, imu0)
    r1 = core.process(1, 0.1, image, lidar, imu1)
    assert r0['estimator']['ground_truth_used'] is False
    assert r1['estimator']['ground_truth_used'] is False
    assert r1['performance']['rss_mb'] >= 0.0


def test_bev_detector_rejects_ground_and_keeps_compact_objects():
    rng = np.random.default_rng(42)
    ground = np.column_stack([
        rng.uniform(2.0, 45.0, 5000),
        rng.uniform(-18.0, 18.0, 5000),
        rng.normal(-1.7, 0.025, 5000),
        np.ones(5000),
    ])
    objects = []
    for cx, cy in [(12.0, 3.0), (22.0, -5.0), (35.0, 7.0)]:
        objects.append(np.column_stack([
            rng.normal(cx, 0.7, 300),
            rng.normal(cy, 0.4, 300),
            rng.uniform(-1.25, 0.75, 300),
            np.ones(300),
        ]))
    detector = CarlaLiDARObjectDetector(max_proposals=10, min_points=10)
    proposals = detector.detect(np.vstack([ground, *objects]))
    assert 2 <= len(proposals) <= 4
    assert all(float(p.get('score', 0.0)) > 0.0 for p in proposals)


def test_core_tracker_confirmation_and_history_are_bounded():
    from lvimot.tracking_4d.multimodal_object_tracking.kalman_tracker import CarlaMultiObjectTracker
    from lvimot.output.evaluation.metrics import CarlaEvaluator

    tracker = CarlaMultiObjectTracker(
        distance_threshold=3.5,
        max_missed=2,
        min_hits=2,
        history_size=8,
        new_track_min_score=0.2,
        use_ego_compensation=False,
    )
    tracked = []
    gts = []
    for f in range(20):
        observations = []
        truth = []
        for actor_id, base in enumerate(([10.0, 2.0, 0.0], [20.0, -4.0, 0.0], [30.0, 6.0, 0.0]), start=1):
            p = np.asarray(base, dtype=float) + np.array([0.15 * f, 0.0, 0.0])
            observations.append({
                'location': p,
                'dimensions': {'length': 4.2, 'width': 1.8, 'height': 1.5},
                'rotation': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
                'type': 'object',
                'score': 0.9,
            })
            truth.append({'actor_id': actor_id, 'location': p, 'velocity': [1.5, 0.0, 0.0]})
        tracked.append(tracker.update(observations, f * 0.1))
        gts.append(truth)
    result = CarlaEvaluator(distance_threshold=3.0).evaluate_tracking_sequence(tracked, gts)
    assert result['mota'] >= 0.90
    assert len(tracker.tracks) == 3
    assert all(len(t.history) <= 8 for t in tracker.tracks)
