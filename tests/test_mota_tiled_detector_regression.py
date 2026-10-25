"""Behavioral regressions for wide-camera small-object recovery.

The fake model controls semantic boxes only. It cannot supply a metric depth.
Recorded tracker regressions are in test_mota_recorded_tracker_regression.py.
"""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

from lvimot.preprocessing.camera_lidar_detection import CarlaCameraLiDARObjectDetector, CameraLiDARDetectionConfig


def box(x1, x2, confidence=0.6, polygon=None):
    result = {'bbox': [x1, 90., x2, 115.], 'class_id': 2,
              'class': 'vehicle', 'type': 'vehicle', 'camera_confidence': confidence}
    if polygon is not None:
        result['segmentation_polygon'] = polygon
    return result


def scripted(views, config=None):
    config = replace(config or CameraLiDARDetectionConfig(image_size=640), traffic_context_enabled=False)
    detector = CarlaCameraLiDARObjectDetector(None, config)
    calls = []
    iterator = iter(views)

    def infer(image):
        calls.append(image.shape)
        return next(iterator)

    detector._infer_single_view = infer
    return detector, calls


def test_recovered_boxes_use_original_camera_coordinates():
    detector, calls = scripted([[], [box(10., 30.)], [box(10., 30.)], [box(10., 30.)]])
    result = detector._infer_boxes(np.zeros((250, 828, 3), dtype=np.uint8))
    assert [item['bbox'][0] for item in result] == [10., 217., 424.]
    assert calls == [(250, 828, 3)] + [(250, 414, 3)] * 3
    assert detector.last_diagnostics['camera_tile_additions'] == 3


def test_full_frame_observation_wins_over_more_confident_crop():
    primary = box(220., 250., confidence=0.4)
    detector, _ = scripted([[primary], [], [box(13., 43., confidence=0.95)], []])
    result = detector._infer_boxes(np.zeros((250, 828, 3), dtype=np.uint8))
    assert result == [primary]
    assert result[0]['camera_confidence'] == 0.4


def test_overlapping_crops_cannot_duplicate_a_new_object():
    detector, _ = scripted([[], [box(220., 250., 0.7)], [box(13., 43., 0.8)], []])
    result = detector._infer_boxes(np.zeros((250, 828, 3), dtype=np.uint8))
    assert len(result) == 1
    assert result[0]['bbox'] == [220., 90., 250., 115.]
    assert result[0]['camera_confidence'] == 0.8


def test_instance_mask_polygon_moves_with_recovered_camera_box():
    polygon = [[10., 90.], [30., 90.], [30., 115.]]
    detection = box(10., 30., polygon=polygon)
    detector, _ = scripted([[], [], [detection], []])
    result = detector._infer_boxes(np.zeros((250, 828, 3), dtype=np.uint8))
    assert result[0]['segmentation_polygon'] == [[217., 90.], [237., 90.], [237., 115.]]
    assert detection['bbox'][0] == 10.
    assert detection['segmentation_polygon'] == polygon


@pytest.mark.parametrize('shape,config', [
    ((250, 828, 3), CameraLiDARDetectionConfig(image_size=640, small_object_tiles=False)),
    ((250, 828, 3), CameraLiDARDetectionConfig(image_size=1280)),
    ((480, 640, 3), CameraLiDARDetectionConfig(image_size=320)),
])
def test_unneeded_or_disabled_tiles_do_not_add_model_calls(shape, config):
    primary = box(100., 140.)
    detector, calls = scripted([[primary]], config)
    assert detector._infer_boxes(np.zeros(shape, dtype=np.uint8)) == [primary]
    assert len(calls) == 1


def test_recovered_boxes_obey_stricter_camera_confidence_floor():
    config = CameraLiDARDetectionConfig(image_size=640, confidence=0.5)
    detector, _ = scripted([[], [box(10., 30., 0.49)], [], []], config)
    assert detector._infer_boxes(np.zeros((250, 828, 3), dtype=np.uint8)) == []


def test_tile_budget_preserves_primary_boxes_and_caps_additions():
    primary = box(50., 70., 0.3)
    config = CameraLiDARDetectionConfig(image_size=640, max_detections=1)
    detector, _ = scripted([[primary], [box(100., 120., 0.6)], [box(100., 120., 0.9)], [box(100., 120., 0.7)]], config)
    result = detector._infer_boxes(np.zeros((250, 828, 3), dtype=np.uint8))
    assert len(result) == 2
    assert result[0] == primary
    assert result[1]['bbox'][0] == 307.


def test_full_primary_budget_does_not_block_recovery_before_lidar_validation():
    primary = box(50., 70., 0.3)
    config = CameraLiDARDetectionConfig(image_size=640, max_detections=1)
    detector, calls = scripted([[primary], [box(100., 120., 0.7)], [], []], config)
    result = detector._infer_boxes(np.zeros((250, 828, 3), dtype=np.uint8))
    assert result == [primary, box(100., 120., 0.7)]
    assert len(calls) == 4


def test_empty_lidar_cannot_become_a_metric_object_from_a_crop():
    detector, _ = scripted([[], [box(10., 30.)], [], []])
    result = detector.detect(np.zeros((250, 828, 3), dtype=np.uint8), np.empty((0, 4)))
    assert result == []
    assert detector.last_diagnostics['camera_detections'] == 1
    assert detector.last_diagnostics['proposals'] == 0
