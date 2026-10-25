"""Sparse physical evidence must confirm without substituting background depth."""
import sys
from pathlib import Path

import numpy as np

from lvimot.preprocessing.camera_lidar_detection import CarlaCameraLiDARObjectDetector
from lvimot.tracking_4d.multimodal_object_tracking.kalman_tracker import CarlaMultiObjectTracker
from lvimot.tracking_4d.multimodal_object_tracking.kalman_tracker import KalmanTrack
from lvimot.output.evaluation.metrics import CarlaEvaluator


def _tracker():
    return CarlaMultiObjectTracker(
        use_ego_compensation=False, min_hits=3,
        new_track_min_score=.20, publish_min_score=.42,
        min_camera_support=6, sparse_min_support=3,
        sparse_min_camera_confidence=.25, sparse_confirm_hits=2,
        sparse_min_bbox_iou=.30, weak_confirm_min_score=.22,
        camera_bridge_max_frames=3, camera_bridge_min_iou=.3,
        camera_bridge_min_confidence=.3,
    )


def _observation(x, score=.30, support=4):
    return {'location': [x, 0, 0], 'bbox': [100, 80, 160, 120],
            'score': score, 'camera_confidence': .40,
            'camera_evidence': True, 'camera_support': support,
            'type': 'vehicle', 'class': 'vehicle',
            'dimensions': {'length': 4.2, 'width': 1.8, 'height': 1.5}}


def test_two_foreground_returns_cannot_be_replaced_by_background_wall():
    depths = np.array([32.44, 32.54, 53.65, 53.66, 54.09])
    chosen = CarlaCameraLiDARObjectDetector._depth_cluster(depths, np.arange(len(depths)))
    assert len(chosen) == 0


def test_real_foreground_cluster_is_kept_with_background_present():
    depths = np.array([30.0, 30.1, 30.2, 30.4, 31.0, 55.0, 55.1, 55.2])
    chosen = CarlaCameraLiDARObjectDetector._depth_cluster(depths, np.arange(len(depths)))
    assert len(chosen) >= 3
    assert np.max(depths[chosen]) < 32


def test_sparse_two_measurement_confirmation_uses_sparse_quality_floor():
    tracker = _tracker()
    assert tracker.update([_observation(30.0)], 0.0) == []
    result = tracker.update([_observation(30.1)], 0.1)
    assert len(result) == 1
    assert result[0]['score'] < tracker.publish_min_score
    assert result[0]['position_source'] == 'CURRENT_3D_LIDAR'


def test_confirmed_sparse_track_can_reacquire_after_one_miss():
    tracker = _tracker()
    tracker.update([_observation(30.0)], 0.0)
    assert len(tracker.update([_observation(30.1)], 0.1)) == 1
    assert tracker.update([], 0.2) == []
    result = tracker.update([_observation(30.3)], 0.3)
    assert len(result) == 1
    assert result[0]['track_id'] == 0


def test_low_confidence_rgb_only_bridges_an_already_metric_confirmed_track():
    tracker = _tracker()
    camera = {'bbox': [100, 80, 160, 120], 'class': 'vehicle', 'camera_confidence': .12}
    assert tracker.update([], 0.0, camera_detections=[camera]) == []
    tracker.update([_observation(30)], 0.1)
    assert len(tracker.update([_observation(30.1)], 0.2)) == 1
    result = tracker.update([], 0.3, camera_detections=[camera])
    assert len(result) == 1
    assert result[0]['camera_only_predicted'] is True
    assert tracker.update([], 0.6, camera_detections=[camera]) == []


def test_motp_reports_spatial_error_rather_than_camera_association_cost():
    tracker = [[{'track_id': 1, 'position': [3, 0, 0], 'bbox': [0, 0, 20, 20]}]]
    truth = [[{'actor_id': 7, 'location': [0, 0, 0], 'bbox': [0, 0, 20, 20]}]]
    result = CarlaEvaluator().evaluate_tracking_sequence(tracker, truth)
    assert result['motp'] == 3.0
    assert result['association_cost_mean'] < .01


def test_eight_and_nine_point_tracks_have_a_confirmation_path():
    for support in (8, 9):
        tracker = _tracker()
        tracker.update([_observation(30, support=support)], 0.0)
        result = tracker.update([_observation(30.1, support=support)], .1)
        assert len(result) == 1


def test_position_measurements_infer_velocity_without_counting_noise_twice():
    track = KalmanTrack(0, _observation(0), 0)
    velocities = []
    for i in range(1, 101):
        track.predict(.1)
        jitter = .2 if i % 2 else -.2
        track.update(_observation(2 * i * .1 + jitter), i * .1)
        velocities.append(float(track.velocity[0]))
    assert abs(np.mean(velocities[-40:]) - 2) < .15
    assert np.std(velocities[-40:]) < .3
    assert np.min(np.linalg.eigvalsh(track.P)) > 0
