"""Sensor-only IMU/compass trajectory anchor for short CARLA captures.

The compass is included in CARLA's IMU packet. No vehicle/world GT is read.
This is a *regularizing measurement* for the factor graph, not GT mapping.
The integrated position can drift; validation must still compare with held-out GT.
"""
from __future__ import annotations

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))

import math
import numpy as np


class SensorInertialAnchor:
    def __init__(self):
        self.pose = np.zeros(6, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.last_timestamp = None
        self.last_acceleration = None
        self.compass_start = None
        self.last_yaw = 0.0
        self.first_ground_z = None
        self.last_ground_z = None

    @staticmethod
    def ground_lidar_height(lidar):
        """Measure near-road plane height in LiDAR coordinates, without map/GT."""
        p = np.asarray(lidar, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] < 3:
            return None
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        mask = (x > 10.) & (x < 25.) & (np.abs(y) < 4.) & (z < -1.0) & (z > -6.)
        vals = z[mask & np.isfinite(z)]
        if len(vals) < 10:
            return None
        # Nearby front road points dominate the lower robust quantile.
        return float(np.percentile(vals, 25.0))

    def update(self, timestamp, motion, imu_data, lidar):
        stamp = float(timestamp)
        accel = np.asarray(motion['acceleration'], dtype=np.float64).reshape(3)
        gyro = np.asarray(motion['gyro'], dtype=np.float64).reshape(3)
        compass = imu_data.get('compass') if isinstance(imu_data, dict) else None
        try:
            compass = float(compass)
        except (TypeError, ValueError):
            compass = None
        if compass is not None and not math.isfinite(compass):
            compass = None
        if self.compass_start is None and compass is not None:
            self.compass_start = compass
        old_yaw = float(self.pose[2])
        if self.compass_start is not None and compass is not None:
            # Signed shortest relative heading; CARLA IMU heading convention.
            yaw = math.atan2(math.sin(compass-self.compass_start),
                             math.cos(compass-self.compass_start))
        elif self.last_timestamp is not None:
            yaw = old_yaw + float(gyro[2])*max(0.0, stamp-self.last_timestamp)
        else:
            yaw = 0.0
        if self.last_timestamp is None:
            self.last_timestamp = stamp
            self.last_acceleration = accel.copy()
            self.pose[2] = yaw
        else:
            dt = float(np.clip(stamp-self.last_timestamp, 0.0, .25))
            acc = .5*(self.last_acceleration+accel)
            heading = .5*(old_yaw+yaw)
            c, s = math.cos(heading), math.sin(heading)
            accel_xy = np.array([c*acc[0]-s*acc[1],
                                 s*acc[0]+c*acc[1]], dtype=np.float64)
            prev_velocity = self.velocity[:2].copy()
            self.pose[3:5] += prev_velocity*dt+.5*accel_xy*dt*dt
            self.velocity[:2] += accel_xy*dt
            # Prevent a stationary car from accumulating tiny signed reverse
            # velocity from IMU noise. Does not prohibit meaningful reversing.
            if -0.15 < self.velocity[0] < 0.0:
                self.velocity[0] = 0.0
            self.pose[2] = yaw
            self.last_timestamp = stamp
            self.last_acceleration = accel.copy()
        height = self.ground_lidar_height(lidar)
        if height is not None:
            if self.first_ground_z is None:
                self.first_ground_z = height
            # Sensor-ground plane, never CARLA's GT world elevation.
            self.last_ground_z = height
            self.pose[5] = float(np.clip(self.first_ground_z-height, -5., 5.))
        self.last_yaw = yaw
        return {'pose': self.pose.copy(), 'velocity': self.velocity.copy(),
                'compass_used': bool(self.compass_start is not None and compass is not None),
                'ground_height_used': bool(height is not None)}
