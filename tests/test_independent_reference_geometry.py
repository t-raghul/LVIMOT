import math
import sys
from pathlib import Path

import numpy as np

from lvimot.mapping_4d.static_map.rolling_voxel_map import Carla4DEnvironmentMapper
from lvimot.output.evaluation.independent_reference import decode_depth_bgra, depth_camera_points, sensor_field_mask, matrix_to_pose


def test_depth_buffer_byte_order_and_pinhole_axes():
    value = int(round(25 / 1000 * 16777215))
    pixel = np.array([value // 65536, (value // 256) % 256, value % 256, 255], dtype=np.uint8)
    buffer = np.tile(pixel, (4, 8, 1))
    depth = decode_depth_bgra(buffer)
    np.testing.assert_allclose(depth, 25, atol=.0001)
    points, labels = depth_camera_points(depth, np.full((4, 8), 7, dtype=np.uint8), 90, stride=1)
    image = points.reshape(4, 8, 3)
    np.testing.assert_allclose(image[2, 4], [25, 0, 0], atol=.0001)
    assert image[2, 7, 1] > 0
    assert image[0, 4, 2] > 0
    assert set(labels) == {7}


def test_four_depth_views_cover_full_lidar_vertical_and_horizontal_field():
    directions = []
    for elevation in (-29, -20, -10, 0, 9):
        for yaw in range(-180, 180, 5):
            e, a = math.radians(elevation), math.radians(yaw)
            directions.append([30 * math.cos(e) * math.cos(a), 30 * math.cos(e) * math.sin(a), 30 * math.sin(e)])
    points = np.array(directions)
    covered = np.zeros(len(points), dtype=bool)
    focal = 512 / (2 * math.tan(math.radians(50)))
    for yaw in (0, 90, 180, 270):
        rotation = Carla4DEnvironmentMapper._rotation_matrix({'yaw': yaw, 'pitch': -10})
        local = points @ rotation
        forward = local[:, 0]
        u = 256 + focal * local[:, 1] / np.where(np.abs(forward) < 1e-8, 1e-8, forward)
        v = 128 - focal * local[:, 2] / np.where(np.abs(forward) < 1e-8, 1e-8, forward)
        covered |= (forward > 0) & (u >= 0) & (u < 512) & (v >= 0) & (v < 256)
        pose = matrix_to_pose(rotation, [1, 2, 3])
        np.testing.assert_allclose(Carla4DEnvironmentMapper._rotation_matrix(pose['rotation']), rotation, atol=1e-12)
    assert np.all(covered)
    assert np.all(sensor_field_mask(points, 60))
    assert not np.any(sensor_field_mask(points * 3, 60))
