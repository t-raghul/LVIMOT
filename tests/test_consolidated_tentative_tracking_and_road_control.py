"""Critical regressions for consolidated FAST saved-sensor repair."""
import sys
from pathlib import Path
import numpy as np

from lvimot.tracking_4d.multimodal_object_tracking.kalman_tracker import CarlaMultiObjectTracker
from lvimot.output.evaluation.road_surface_audit import evaluate_road_height_proxy_control


def tracker():
    return CarlaMultiObjectTracker(
        distance_threshold=3, min_hits=2, new_track_min_score=.20,
        publish_min_score=.42, min_camera_support=6, fast_confirm_hits=2,
        fast_confirm_min_support=8, sparse_min_support=3,
        sparse_min_camera_confidence=.25, sparse_confirm_hits=2,
        sparse_min_bbox_iou=.30, camera_bridge_max_frames=3,
        camera_bridge_min_iou=.30, camera_bridge_min_confidence=.30,
    )


def obs(x=20., n=4, box=(40, 60, 100, 110), score=.48):
    return {
        'location': [float(x), 0., 0.], 'class': 'vehicle', 'type': 'vehicle',
        'dimensions': {'length': 4.5, 'width': 1.8, 'height': 1.6},
        'camera_evidence': True, 'camera_confidence': .70,
        'camera_support': n, 'num_points': n,
        'score': score, 'bbox': list(box),
    }


def cam(box=(42, 60, 102, 110)):
    return {'bbox': list(box), 'class': 'vehicle', 'camera_confidence': .78}


def test_tentative_lidar_seed_survives_camera_only_gap_and_requires_second_real_depth():
    t = tracker()
    assert t.update([obs()], 0.) == []
    original_meas = t.tracks[0].last_measurement_timestamp
    assert t.update([], .1, camera_detections=[cam()]) == []
    assert len(t.tracks) == 1
    assert t.tracks[0].last_measurement_timestamp == original_meas
    assert t.tracks[0].hits == 1
    got = t.update([obs(20.2, box=(43, 60, 103, 110))], .2)
    assert len(got) == 1
    assert got[0]['track_id'] == 0
    assert not got[0]['camera_only_predicted']


def test_cannot_publish_single_sparse_3d_seed_using_camera_boxes_only():
    t = tracker()
    assert t.update([obs()], 0.) == []
    for ts in [.1, .2, .3, .4, .5]:
        assert t.update([], ts, camera_detections=[cam()]) == []
    assert t.next_track_id == 1


def test_camera_only_has_no_way_to_seed_metric_3d():
    t = tracker()
    for ts in [0., .1, .2, .3]:
        assert t.update([], ts, camera_detections=[cam()]) == []
    assert t.next_track_id == 0


def test_depth_and_class_incompatible_camera_cannot_extend_tentative_track():
    t = tracker()
    t.update([obs()], 0.)
    cam_other = {'bbox': [350, 70, 380, 110], 'class': 'vehicle', 'camera_confidence': .8}
    t.update([], .1, camera_detections=[cam_other])
    assert t.tracks[0].missed == 1


def test_road_control_does_not_mislabel_sidewalk_as_road_precision():
    grid = np.array([(float(x), float(y), 0.) for x in range(-15,16) for y in range(-2,3)])
    sidewalk = np.array([(float(x), 12., 0.) for x in range(-15, 16)])
    raw = np.vstack((grid, sidewalk))
    report = evaluate_road_height_proxy_control(raw, raw, grid, [0,0,0])
    assert report['estimated_minus_gt_control_percentage_points'] == 0
    assert report['gt_used_only_for_evaluation_control'] is True
    assert report['can_certify_full_3d_map'] is False
    assert report['status'] == 'DIAGNOSTIC_CONTROL_NOT_SEMANTIC_ROAD_CERTIFICATION'
    assert report['gt_posed_same_sensor_height_proxy']['road_agreement_pct'] < 100


def test_road_control_handles_empty_reference():
    cloud = np.zeros((0, 3))
    report = evaluate_road_height_proxy_control(cloud, cloud, cloud, [0,0,0])
    assert report['estimated_minus_gt_control_percentage_points'] is None
    assert report['can_certify_full_3d_map'] is False
