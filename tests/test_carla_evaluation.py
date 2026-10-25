import os
import sys
import numpy as np


from lvimot.output.evaluation.metrics import CarlaEvaluator


def test_strict_3d_evaluation_rejects_correct_image_box_at_wrong_metric_depth():
    tracks = [[{'track_id': 1, 'position': [40, 0, 0], 'bbox': [0, 0, 20, 20]}]]
    truth = [[{'actor_id': 3, 'location': [10, 0, 0], 'bbox': [0, 0, 20, 20]}]]
    legacy = CarlaEvaluator().evaluate_tracking_sequence(tracks, truth)
    strict = CarlaEvaluator(require_3d_match=True).evaluate_tracking_sequence(tracks, truth)
    assert legacy['total_matches'] == 1
    assert strict['total_matches'] == 0
    assert strict['false_positives'] == 1 and strict['false_negatives'] == 1


def test_tracking_evaluation():
    evaluator = CarlaEvaluator(distance_threshold=2.0)
    gts = [[{'actor_id': 1, 'location': [10.0, 0.0, 0.0], 'velocity': [5.0, 0.0, 0.0]}]]
    tracks = [[{'track_id': 101, 'position': [10.1, 0.05, 0.0], 'velocity': [4.9, 0.0, 0.0]}]]
    res = evaluator.evaluate_tracking_sequence(tracks, gts)
    assert res['mota'] == 1.0


def test_localization_evaluation():
    evaluator = CarlaEvaluator()
    gt = np.array([[0, 0, 0], [1, 0, 0]])
    est = gt + np.array([0.05, 0, 0])
    res = evaluator.evaluate_localization(est, gt)
    assert abs(res['ate_rmse'] - 0.05) < 1e-4


def test_tracking_no_matches_reports_na_not_zero():
    evaluator = CarlaEvaluator(distance_threshold=2.0)
    gts = [[{'actor_id': 1, 'location': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}]]
    tracks = [[{'track_id': 5, 'position': [20.0, 20.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}]]
    res = evaluator.evaluate_tracking_sequence(tracks, gts)
    assert res['total_matches'] == 0
    assert res['position_rmse'] is None
    assert res['velocity_rmse'] is None
    assert res['motp'] is None


def test_tracking_uses_bev_distance_with_vertical_gate():
    evaluator = CarlaEvaluator(distance_threshold=2.0, vertical_threshold=1.5)
    gts = [[{'actor_id': 1, 'location': [10.0, 0.0, 1.0], 'velocity': [1.0, 0.0, 0.0]}]]
    tracks = [[{'track_id': 2, 'position': [10.5, 0.2, 1.4], 'velocity': [1.1, 0.0, 0.0]}]]
    res = evaluator.evaluate_tracking_sequence(tracks, gts)
    assert res['total_matches'] == 1
    assert res['mota'] == 1.0
    assert res['position_rmse'] is not None
