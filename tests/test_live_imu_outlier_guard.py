"""Offline-only tests: no CARLA connection, sensors, GUI, or GPU needed."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1] / 'src'

from lvimot.temporal_fusion.pipeline import CarlaLVIMOTCore
from lvimot.preprocessing.imu_preintegration import IMUPreintegrator


def core_without_sensors():
    core = CarlaLVIMOTCore.__new__(CarlaLVIMOTCore)
    core._last_imu_motion = {
        'acceleration': np.array([0.0, 0.0, 9.80665]),
        'gyro': np.zeros(3), 'timestamp': 0.0,
    }
    core._last_pose = np.zeros(6)
    core._last_velocity = np.zeros(3)
    core._last_timestamp = 0.0
    return core


def test_corrupted_6767_acceleration_is_rejected_before_preintegration():
    core = core_without_sensors()
    motion = {'acceleration': np.array([-4681.8, -4880.6, 10.0]),
              'gyro': np.zeros(3), 'timestamp': 0.2}
    cleaned, rejected, reason = core._sanitize_imu_motion(motion)
    assert rejected
    assert 'acceleration=' in reason
    assert np.allclose(cleaned['acceleration'], [0, 0, 9.80665])
    assert not np.shares_memory(cleaned['acceleration'], core._last_imu_motion['acceleration'])
    assert core._predict_state(None, 0.2)[0][3:].tolist() == [0.0, 0.0, 0.0]
    # Simulate next sane measurement: no leftover 700 m/s velocity impulse.
    core._last_imu_motion = cleaned
    valid = {'acceleration': np.array([0.0, 0.0, 9.80665]),
             'gyro': np.zeros(3), 'timestamp': 0.3}
    integrated = IMUPreintegrator().integrate(cleaned, valid)
    predicted, velocity = core._predict_state(integrated, 0.3)
    assert np.linalg.norm(predicted[3:]) < 0.001
    assert np.linalg.norm(velocity) < 0.001


def test_normal_vehicle_acceleration_is_accepted():
    core = core_without_sensors()
    sample = {'acceleration': np.array([2.0, 0.0, 9.81]),
              'gyro': np.array([0.0, 0.0, 0.2])}
    cleaned, rejected, reason = core._sanitize_imu_motion(sample)
    assert not rejected and not reason
    assert cleaned is sample


def test_nonfinite_and_excessive_spin_rejected():
    core = core_without_sensors()
    for accel, gyro in [([np.nan,0,0],[0,0,0]),
                        ([0,0,9.81],[0,0,20])]:
        cleaned, rejected, _ = core._sanitize_imu_motion({
            'acceleration': np.array(accel), 'gyro': np.array(gyro)})
        assert rejected
        assert np.isfinite(cleaned['acceleration']).all()
        assert np.isfinite(cleaned['gyro']).all()
