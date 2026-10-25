import sys
from pathlib import Path

import pytest

from lvimot.inputs.sensor_profile import with_lidar_profile


def test_64_channel_mapping_profile_has_explicit_density_and_full_scan():
    original = {'camera': {'width': 828}, 'lidar': {'channels': 64, 'range': 100}}
    result = with_lidar_profile(original, .1)
    assert result['lidar']['points_per_second'] == 560000
    assert result['lidar']['rotation_frequency'] == 10
    assert result['lidar']['sensor_tick'] == .1
    assert result['lidar']['acquisition_model'] == 'CARLA_ray_cast_snapshot'
    assert 'points_per_second' not in original['lidar']
    assert result['camera'] == original['camera']


def test_explicit_user_density_is_preserved():
    assert with_lidar_profile({'lidar': {'points_per_second': 900000}}, .05)['lidar']['points_per_second'] == 900000
    assert with_lidar_profile({'lidar': {'points_per_second': 900000}}, .05)['lidar']['rotation_frequency'] == 20


def test_invalid_sensor_profile_is_rejected_before_carla():
    with pytest.raises(ValueError):
        with_lidar_profile({'lidar': {}}, 0)
    with pytest.raises(ValueError):
        with_lidar_profile({'lidar': {'points_per_second': -1}}, .1)
