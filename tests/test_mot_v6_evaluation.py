import os
import sys


from lvimot.output.evaluation.metrics import CarlaEvaluator


def test_mota_is_undefined_when_no_eligible_gt_exists():
    evaluator = CarlaEvaluator()
    res = evaluator.evaluate_tracking_sequence(
        [[{'track_id': 1, 'position': [10.0, 0.0, 0.0], 'velocity': [0, 0, 0], 'bbox': [10, 10, 20, 20]}]],
        [[]],
    )
    assert res['mota'] is None
    assert res['evaluation_valid'] is False
    assert res['false_positives'] == 1


def test_camera_iou_can_associate_when_global_3d_frames_disagree():
    evaluator = CarlaEvaluator(distance_threshold=2.0, iou_threshold=0.25)
    tracks = [[{
        'track_id': 10,
        'position': [50.0, -14.0, -1.5],
        'velocity': [0.0, 0.0, 0.0],
        'bbox': [100.0, 80.0, 180.0, 160.0],
    }]]
    gts = [[{
        'actor_id': 7,
        'location': [5.0, 2.0, 0.0],
        'velocity': [0.0, 0.0, 0.0],
        'bbox': [105.0, 82.0, 178.0, 158.0],
    }]]
    res = evaluator.evaluate_tracking_sequence(tracks, gts)
    assert res['total_matches'] == 1
    assert res['camera_iou_matches'] == 1
    assert res['mota'] == 1.0
    assert res['position_rmse'] is not None
    assert res['position_rmse'] > 10.0


def test_bev_fallback_still_works_without_camera_boxes():
    evaluator = CarlaEvaluator(distance_threshold=2.0)
    tracks = [[{'track_id': 2, 'position': [10.5, 0.2, 1.4], 'velocity': [1.1, 0.0, 0.0]}]]
    gts = [[{'actor_id': 1, 'location': [10.0, 0.0, 1.0], 'velocity': [1.0, 0.0, 0.0]}]]
    res = evaluator.evaluate_tracking_sequence(tracks, gts)
    assert res['total_matches'] == 1
    assert res['bev_matches'] == 1
    assert res['mota'] == 1.0
