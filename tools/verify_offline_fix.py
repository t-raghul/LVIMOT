"""One-command LVIMOT preflight without starting CARLA or loading YOLO/CUDA.

From CARLA_LVIMOT-main on Windows:
    .venv_live38\\Scripts\\python.exe tools\\verify_offline_fix.py
"""
from __future__ import annotations

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[1]))


import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


from lvimot.mapping_4d.static_map.rolling_voxel_map import Carla4DEnvironmentMapper
from lvimot.output.live_run import _verify_map_export_before_carla
from lvimot.output.live_dashboard import CarlaLiveVisualizer
from lvimot.temporal_fusion.pipeline import CarlaLVIMOTCore
from lvimot.inputs.sensor_recording import RawSensorArchive
from lvimot.temporal_fusion.runtime_limits import DEFAULT_LIMITS
from lvimot.temporal_fusion.sensor_inertial_anchor import SensorInertialAnchor


def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    cfg = ROOT/'config'/'carla_fast_mapping.json'
    if not cfg.is_file():
        raise FileNotFoundError('Calibration missing at ' + str(cfg))
    out = ROOT/'outputs_carla'/'live'
    out.mkdir(parents=True, exist_ok=True)
    RawSensorArchive.preflight(out, cfg)
    _verify_map_export_before_carla(out, CarlaLiveVisualizer())
    pose = np.array([.13, -.09, .43, 3., 4., 5.])
    pd = CarlaLVIMOTCore.pose_dict_from_vector(pose)
    R1 = CarlaLVIMOTCore.euler_to_rot(pose[:3])
    R2 = Carla4DEnvironmentMapper._rotation_matrix(pd['rotation'])
    np.testing.assert_allclose(R1, R2, atol=1e-10)
    core = CarlaLVIMOTCore.__new__(CarlaLVIMOTCore)
    core.limits = DEFAULT_LIMITS
    core.object_sensor_range = 100.0
    core._previous_motion_lidar_xyz = None
    core._last_lidar_motion_transform = None
    rng = np.random.default_rng(42)
    cloud = np.column_stack([
        rng.uniform(5, 40, 4000), rng.uniform(-15, 15, 4000),
        rng.uniform(.5, 6, 4000),
    ])
    core._lidar_translation_motion_metric(cloud)
    estimated, _, _ = core._lidar_translation_motion_metric(cloud - np.array([.6,0,0]))
    if estimated is None or abs(estimated-.6) >= .2:
        raise RuntimeError('LiDAR motion registration preflight FAILED: ' + str(estimated))
    # Independent compass-heading and IMU metric-translation self-test.
    anchor = SensorInertialAnchor()
    for k in range(11):
        accel = np.array([2.0, 0.0, 9.80665])
        motion = {'acceleration': accel, 'gyro': np.zeros(3)}
        state = anchor.update(k * .1, motion, {'compass': 1.2}, np.empty((0,4)))
    if abs(state['pose'][3] - 1.0) > .02:
        raise RuntimeError('Independent sensor inertial odometry preflight FAILED')
    print('PRE-RUN OFFLINE CHECKS PASSED: recording, PLY export, 3D rotation, LiDAR scale, IMU+compass.')
    print('CARLA map accuracy has NOT been validated by these offline checks.')


if __name__ == '__main__':
    main()
