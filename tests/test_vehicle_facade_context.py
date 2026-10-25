import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from lvimot.preprocessing.camera_lidar_detection import CarlaCameraLiDARObjectDetector as Detector


def facade():
    yz = np.array(np.meshgrid(np.arange(-10, 10, .25), np.arange(-3, 4, .25))).reshape(2, -1).T
    return np.column_stack([np.full(len(yz), 30), yz])


def test_extended_vertical_facade_is_not_vehicle_depth():
    points = facade()
    assert Detector._belongs_to_extended_facade(np.array([30, 0, 0]), points, cKDTree(points))


def test_real_vehicle_offset_in_front_of_wall_is_retained():
    points = facade()
    assert not Detector._belongs_to_extended_facade(np.array([28, 0, 0]), points, cKDTree(points))


def test_vehicle_sized_plane_is_not_rejected_as_building():
    points = facade()
    points = points[np.abs(points[:, 1]) <= 4]
    assert not Detector._belongs_to_extended_facade(np.array([30, 0, 0]), points, cKDTree(points))
