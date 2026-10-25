import numpy as np
from lvimot.temporal_fusion.sensor_inertial_anchor import SensorInertialAnchor


def _pkt(t, ax=0.0, ay=0.0, compass=0.0):
    motion={'acceleration': np.array([ax, ay, 9.80665], dtype=float),
            'gyro': np.zeros(3, dtype=float)}
    return t,motion,{'compass':compass},np.empty((0,4),dtype=float)


def test_constant_acceleration_distance_without_ground_truth():
    a=SensorInertialAnchor()
    for i in range(11):
        t,motion,imu,lidar=_pkt(i*.1, ax=2.0)
        result=a.update(t,motion,imu,lidar)
    assert abs(result['pose'][3]-1.0)<.015
    assert abs(result['velocity'][0]-2.0)<.015
    assert np.allclose(result['pose'][4:6],0)


def test_compass_heading_tracks_sensor_without_gyro_drift():
    a=SensorInertialAnchor()
    for i in range(11):
        t,motion,imu,lidar=_pkt(i*.1, ax=0.0, compass=3.13-.1*i/10)
        result=a.update(t,motion,imu,lidar)
    assert abs(result['pose'][2]+.1)<1e-8
    assert result['compass_used']


def test_road_height_is_derived_from_lidar_not_carla_map():
    a=SensorInertialAnchor()
    cloud=np.tile(np.array([[14.0, 0.0, -2.8, 0.8]]), (25,1))
    p=_pkt(0.)
    z0=a.update(p[0],p[1],p[2],cloud)['pose'][5]
    cloud[:,2]=-2.2
    p=_pkt(.1)
    z1=a.update(p[0],p[1],p[2],cloud)['pose'][5]
    assert abs(z0)<1e-8
    assert abs(z1+.6)<1e-8


def test_bad_sensor_heading_uses_gyro_fallback():
    a=SensorInertialAnchor()
    for k in range(3):
        t,m,imu,lidar=_pkt(k*.1, ax=1., compass=float('nan'))
        m['gyro'][2]=0.2
        result=a.update(t,m,imu,lidar)
    assert abs(result['pose'][2]-.04)<1e-8
    assert not result['compass_used']
