import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'

from lvimot.tracking_4d.multimodal_object_tracking.kalman_tracker import CarlaMultiObjectTracker


def tracker():
    return CarlaMultiObjectTracker(
        distance_threshold=2.75, min_hits=3,
        new_track_min_score=0.20, publish_min_score=0.42,
        min_camera_support=6, fast_confirm_hits=2,
        fast_confirm_min_support=8, weak_confirm_min_support=10,
        sparse_min_support=4, sparse_min_camera_confidence=0.25,
        sparse_confirm_hits=2, sparse_min_bbox_iou=0.30,
    )


def obs(x, box=(100, 80, 160, 140), points=4, conf=0.65,
        camera=True, score=0.48, new=True):
    return {
        'location':[float(x),0.0,0.0],
        'dimensions': {'length':4.2,'width':1.8,'height':1.5},
        'bbox':list(box), 'score':score,
        'camera_confidence':conf, 'camera_evidence':camera,
        'camera_support':points, 'num_points':points,
        'type':'vehicle', 'class':'vehicle', 'allow_new_track':new,
    }


def test_four_return_vehicle_confirms_after_two_consistent_rgb_and_lidar_frames():
    t=tracker()
    assert t.update([obs(30.0)], 0.0)==[]
    got=t.update([obs(30.2,(102,80,162,140))],0.1)
    assert len(got)==1
    assert got[0]['camera_support']==4 and got[0]['track_id']==0


def test_sparse_single_frame_ghost_is_not_published():
    t=tracker()
    assert t.update([obs(30.0)],0.0)==[]
    assert t.update([],0.1)==[]
    assert t.update([],0.2)==[]


def test_sparse_nonoverlapping_rgb_boxes_cannot_confirm():
    t=tracker()
    assert t.update([obs(30.0)],0.0)==[]
    assert t.update([obs(30.1,(290,80,350,140))],0.1)==[]


def test_sparse_camera_without_lidar_cannot_spawn():
    t=tracker()
    assert t.update([obs(30,points=0)],0.0)==[]
    assert t.next_track_id==0
    assert t.update([obs(30,points=3)],0.1)==[]
    assert t.next_track_id==0


def test_sparse_low_confidence_or_lidar_fallback_cannot_spawn():
    t=tracker()
    assert t.update([obs(30,conf=0.1)],0.0)==[]
    assert t.update([obs(30, camera=False, new=False)],0.1)==[]
    assert t.next_track_id==0


def test_existing_normal_tracked_vehicle_stays_confirmed_with_sparse_returns():
    t=tracker()
    t.update([obs(30.0,points=11)],0.0)
    assert len(t.update([obs(30.1,points=11)],0.1))==1
    assert len(t.update([obs(30.3,points=4)],0.2))==1
    assert len(t.update([obs(30.4,points=4)],0.3))==1


def test_sparse_mode_does_not_change_lidar_only_tracker_by_default():
    t=CarlaMultiObjectTracker(min_camera_support=6,min_hits=2)
    t.update([obs(30)],0.0)
    t.update([obs(30.1)],0.1)
    assert t.next_track_id==0


def test_five_to_six_return_vehicle_confirms_with_two_consistent_views():
    t=tracker()
    assert t.update([obs(44.0,points=5,conf=0.46)],0.0)==[]
    got=t.update([obs(44.3,(104,80,164,140),points=6,conf=0.34)],0.1)
    assert len(got)==1
    assert got[0]['track_id']==0

def test_sparse_inconsistent_depth_measurement_is_not_published():
    t=tracker()
    t.update([obs(31,points=4,conf=0.7)],0.0)
    # A 3-D jump must not count as stable sparse confirmation, even when boxes overlap.
    assert t.update([obs(36.2,points=4,conf=0.7)],0.1)==[]
