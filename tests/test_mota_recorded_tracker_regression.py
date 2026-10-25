"""Replay real measured tracker inputs without YOLO, CARLA or external files.

Fixtures contain sensor-only measurements and estimated poses. Ground truth
remains a separate field, used only after every tracker output is final.
This is a tracker regression, not a substitute for the full raw sensor replay.
"""
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from lvimot.inputs.sensor_calibration import CarlaCalibration
from lvimot.temporal_fusion.pipeline import CarlaLVIMOTCore


@pytest.mark.parametrize('name,expected', [
    ('baseline', (91, 82, 54, 2)),
    ('fixed', (99, 74, 55, 3)),
])
def test_recorded_tracker_retains_metrics_identity_and_sensor_coordinates(name, expected):
    fixture = ROOT / 'tests' / 'fixtures' / ('mota_latest_' + name + '_tracker.json.gz')
    with gzip.open(fixture, 'rt', encoding='utf-8') as handle:
        data = json.load(handle)
    core = CarlaLVIMOTCore(CarlaCalibration(str(ROOT / 'config' / 'carla_fast_mapping.json')),
                          detector_mode='camera_lidar')
    outputs, labels = [], []
    try:
        cfg = core.camera_lidar_detector.config
        for frame in data['frames']:
            # This call contains no actor identity, GT position or GT box.
            tracks = core.object_tracker.update(**frame['tracker_input'])
            local = core._tracks_current_lidar(tracks, np.asarray(frame['pose']))
            scored = []
            for item in local:
                bbox = item.get('bbox')
                if bbox is None:
                    continue
                width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if width >= cfg.min_eval_bbox_width_px and height >= cfg.min_eval_bbox_height_px and width * height >= cfg.min_eval_bbox_area_px:
                    scored.append(item)
            assert [t['track_id'] for t in scored] == [t['track_id'] for t in frame['expected_tracks']]
            for actual, saved in zip(scored, frame['expected_tracks']):
                np.testing.assert_allclose(actual['position'], saved['position'], atol=1e-10, rtol=0)
                np.testing.assert_allclose(actual['velocity'], saved['velocity'], atol=1e-10, rtol=0)
            outputs.append(scored)
            labels.append(frame['evaluation_gt'])
        result = core.evaluator.evaluate_tracking_sequence(outputs, labels)
        assert len(outputs) == 114
        assert result['total_gt_objects'] == 173
        assert (result['total_matches'], result['false_negatives'], result['false_positives'], result['id_switches']) == expected
    finally:
        core.close()
