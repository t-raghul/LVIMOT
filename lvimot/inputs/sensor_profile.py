"""Explicit acquisition settings for a 64-channel CARLA mapping run."""

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))

from copy import deepcopy

import numpy as np


DEFAULT_LIDAR_POINTS_PER_SECOND = 560000


def with_lidar_profile(calibration_data, fixed_delta):
    data = deepcopy(calibration_data)
    delta = float(fixed_delta)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('fixed_delta must be finite and positive')
    lidar = data.setdefault('lidar', {})
    points_per_second = int(lidar.get('points_per_second', DEFAULT_LIDAR_POINTS_PER_SECOND))
    if points_per_second <= 0:
        raise ValueError('LiDAR points_per_second must be positive')
    lidar['points_per_second'] = points_per_second
    lidar['rotation_frequency'] = 1.0 / delta
    lidar['sensor_tick'] = delta
    lidar['acquisition_model'] = 'CARLA_ray_cast_snapshot'
    return data
