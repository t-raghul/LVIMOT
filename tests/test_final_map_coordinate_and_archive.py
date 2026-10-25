"""No CARLA runtime required: verify the exact bug fixes and archive."""
import json
from pathlib import Path

import numpy as np

from lvimot.temporal_fusion.pipeline import CarlaLVIMOTCore
from lvimot.mapping_4d.static_map.rolling_voxel_map import Carla4DEnvironmentMapper
from lvimot.inputs.sensor_recording import RawSensorArchive, iter_archive
from lvimot.output.live_run import _evaluation_only_motion_failure


def test_internal_pose_carla_rotation_matches_any_3d_pose():
    for rpy in ([0.1, -0.2, 0.8], [-0.13, 0.22, -2.1], [0, 0, 1.2]):
        v = np.array([*rpy, 4., 5., 6.], dtype=float)
        pose = CarlaLVIMOTCore.pose_dict_from_vector(v)
        R_core = CarlaLVIMOTCore.euler_to_rot(rpy)
        R_map = Carla4DEnvironmentMapper._rotation_matrix(pose['rotation'])
        np.testing.assert_allclose(R_core, R_map, atol=1e-12)
        roundtrip = CarlaLVIMOTCore.pose_vector_from_dict(pose)
        np.testing.assert_allclose(roundtrip, v, atol=1e-12)


def test_carla_gt_nonflat_rotations_evaluate_consistently():
    pose = {'location': {'x': 14., 'y': 15., 'z': 16.},
            'rotation': {'roll': 5., 'pitch': 10., 'yaw': 67.}}
    v = CarlaLVIMOTCore.pose_vector_from_dict(pose)
    actual = CarlaLVIMOTCore._carla_world_rotation(pose['rotation'])
    np.testing.assert_allclose(CarlaLVIMOTCore.euler_to_rot(v[:3]), actual, atol=1e-12)
    np.testing.assert_allclose(CarlaLVIMOTCore._rotation_from_carla_dict(pose['rotation']), actual, atol=1e-12)


def test_archive_roundtrip_with_gt_separation(tmp_path):
    import cv2
    calib = tmp_path/'calibration.json'
    calib.write_text('{"camera":{},"lidar":{},"imu":{}}')
    live = tmp_path/'live'
    live.mkdir()
    RawSensorArchive.preflight(live, calib)
    assert not any(p.name.startswith('lvimot_archive_preflight_') for p in live.iterdir())
    arch = RawSensorArchive(live, calib, {'seed':42})
    image = np.full((36, 42, 3), 155, dtype=np.uint8)
    points = np.random.default_rng(5).random((123, 4)).astype('float32')
    imu = {'accelerometer':{'x':0.0,'y':0.0,'z':9.81},'gyroscope':{'x':0.,'y':0.,'z':0.},'timestamp':2.3}
    gt = {'location':{'x':10,'y':22,'z':0},'rotation':{'pitch':0,'roll':0,'yaw':30}}
    try:
        arch.write_frame(0, 2.3, 11, image, points, imu, gt_pose=gt)
        arch.write_frame(1, 2.4, 12, image, points+1, imu)
    finally:
        arch.close()
    frames = list(iter_archive(arch.path))
    assert len(frames) == 2
    np.testing.assert_array_equal(frames[0]['lidar'], points)
    np.testing.assert_array_equal(frames[1]['lidar'], points+1)
    np.testing.assert_array_equal(frames[0]['image'], image)
    assert frames[0]['ground_truth_ego_pose'] == gt
    assert frames[1]['ground_truth_ego_pose'] is None
    assert 'ground_truth_ego_pose_evaluation_only' not in str(np.load(arch.path/'frame_000000_lidar.npy'))
    assert json.loads(arch.manifest_path.read_text())['gt_for_evaluation_only']


def test_evaluation_only_mismatch_guard():
    static = [np.r_[np.zeros(3), np.zeros(3)] for _ in range(35)]
    no_gt_move = [np.r_[np.zeros(3), np.zeros(3)] for _ in range(35)]
    assert _evaluation_only_motion_failure(static, no_gt_move) is None
    far = [np.r_[np.zeros(3), [i*.9, 0, 0]] for i in range(35)]
    bad_est = [np.r_[np.zeros(3), [i*.03, 0, 0]] for i in range(35)]
    result = _evaluation_only_motion_failure(bad_est, far)
    assert result and 'underestimation' in result
    good_est = [np.r_[np.zeros(3), [i*.86, 0, 0]] for i in range(35)]
    assert _evaluation_only_motion_failure(good_est, far) is None


def test_gt_motion_diagnosis_cannot_select_live_estimator_frames():
    import ast
    import inspect
    from lvimot.output.live_run import run_live
    tree = ast.parse(inspect.getsource(run_live))
    for loop in (n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))):
        assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                       and n.func.id == '_evaluation_only_motion_failure'
                       for n in ast.walk(loop))


def test_low_quality_icp_must_not_overrule_imu_motion():
    from lvimot.tracking_4d.static_scene_support.feature_correspondences import CarlaTemporalLiDARMatcher
    from lvimot.temporal_fusion.runtime_limits import DEFAULT_LIMITS
    core = CarlaLVIMOTCore.__new__(CarlaLVIMOTCore)
    core.limits = DEFAULT_LIMITS
    core.lidar_tracker = CarlaTemporalLiDARMatcher(min_normal_cos=.8)
    core._previous_pose_for_lidar = np.zeros(6)
    core._last_lidar_motion_transform = {
        'accepted':True, 'delta_R':np.eye(3),
        'delta_t':np.array([0.04, 0, 0]),
        'point_rmse_m':0.34, 'correspondences':500,
    }
    previous = []
    current = []
    for x in np.linspace(5, 40, 10):
        for y in np.linspace(-7, 7, 6):
            q = np.array([x,y,0.],dtype=float)
            previous.append({'center':q,'normal':np.array([0.,0.,1.]),'planarity':.95})
            current.append({'center':q-np.array([.04,0.,0.]),'normal':np.array([0.,0.,1.]),'planarity':.95})
    core._previous_planar_ego = previous
    result = core._lidar_se3_relative_measurement(current, np.zeros(6),
                                                  expected_translation_m=.04)
    assert not result['accepted']
    assert result['reason'] == 'point_quality'
