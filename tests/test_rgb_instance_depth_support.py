"""RGB masks restrict measured LiDAR support without inventing metric depth."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lvimot.preprocessing.camera_lidar_detection import CarlaCameraLiDARObjectDetector, CameraLiDARDetectionConfig


class Calibration:
    def lidar_to_opencv(self, points):
        return np.column_stack([points[:, 1], -points[:, 2], points[:, 0]])

    def project_lidar(self, points):
        c = self.lidar_to_opencv(points)
        return np.column_stack([100 + 100 * c[:, 0] / c[:, 2],
                                100 + 100 * c[:, 1] / c[:, 2]]), c[:, 2] > 0


class MaskModel:
    def predict(self, **kwargs):
        box = SimpleNamespace(xyxy=np.array([[60, 60, 140, 140]]),
                              cls=np.array([2]), conf=np.array([.9]))
        masks = SimpleNamespace(xy=[np.array([[90, 85], [110, 85], [110, 115], [90, 115]])])
        return [SimpleNamespace(boxes=box, masks=masks)]


def test_instance_silhouette_excludes_rectangle_background():
    detector = CarlaCameraLiDARObjectDetector(Calibration(), CameraLiDARDetectionConfig(), MaskModel())
    vehicle = np.array([[10, -.2, -.2], [10.1, .1, .2], [10.2, .2, .1], [10.3, -.1, -.1]])
    wall = np.column_stack([np.full(30, 40), np.full(30, 10), np.linspace(-1, 1, 30)])
    result = detector.detect(np.zeros((200, 200, 3), dtype=np.uint8), np.vstack([vehicle, wall]))
    assert len(result) == 1
    assert result[0]['camera_support'] == 4
    assert result[0]['location'][0] < 11
    assert result[0]['depth_pixel_support'] == 'RGB_INSTANCE_MASK'


def test_insufficient_instance_returns_cannot_fall_back_to_background_rectangle():
    detector = CarlaCameraLiDARObjectDetector(Calibration(), CameraLiDARDetectionConfig(), MaskModel())
    two_returns = np.array([[10, -.2, -.2], [10.1, .1, .2]])
    wall = np.column_stack([np.full(30, 40), np.full(30, 10), np.linspace(-1, 1, 30)])
    assert detector.detect(np.zeros((200, 200, 3), dtype=np.uint8), np.vstack([two_returns, wall])) == []
