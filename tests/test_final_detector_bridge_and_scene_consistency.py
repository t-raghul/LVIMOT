import sys
from pathlib import Path
import numpy as np

from lvimot.preprocessing.camera_lidar_detection import CarlaCameraLiDARObjectDetector
from lvimot.tracking_4d.multimodal_object_tracking.kalman_tracker import CarlaMultiObjectTracker
from lvimot.tracking_4d.static_scene_support.scene_consistency import evaluate_same_sensor_geometry, build_gt_posed_lidar_reference


def test_depth_cluster_rejects_background_outlier_and_keeps_three_real_returns():
    depths=np.array([26.6,27.2,27.9,84.4],dtype=float)
    chosen=CarlaCameraLiDARObjectDetector._depth_cluster(depths,np.arange(4))
    assert len(chosen)==3 and 3 not in chosen


def test_three_incoherent_sparse_depths_are_rejected():
    depths=np.array([14.,28.,42.,60.])
    assert len(CarlaCameraLiDARObjectDetector._depth_cluster(depths,np.arange(4)))==0


def tracker():
    return CarlaMultiObjectTracker(distance_threshold=3, min_hits=2,
       new_track_min_score=.20,publish_min_score=.42,
       min_camera_support=6,fast_confirm_hits=2,fast_confirm_min_support=8,
       sparse_min_support=3,sparse_min_camera_confidence=.25,
       sparse_confirm_hits=2,sparse_min_bbox_iou=.30,
       camera_bridge_max_frames=3,camera_bridge_min_iou=.3,camera_bridge_min_confidence=.3)


def obj(pos, n=5,box=(50,50,110,90)):
    return {'location':[float(pos),0.,0.], 'type':'vehicle','class':'vehicle',
       'dimensions':{'length':4.5,'width':1.8,'height':1.6},
       'bbox':list(box),'camera_support':n,'num_points':n,'camera_evidence':True,
       'camera_confidence':.75,'score':.52}


def cam(box=(53,50,113,90), confidence=.80):
    return {'bbox':list(box),'class':'vehicle','camera_confidence':confidence}


def test_camera_box_bridges_published_3d_track_without_fake_3d_measurement():
    t=tracker()
    assert not t.update([obj(20)],0.0)
    assert len(t.update([obj(20.1)],0.1))==1
    predicted=t.update([],0.2,camera_detections=[cam()])
    assert len(predicted)==1 and predicted[0]['camera_only_predicted']
    assert predicted[0]['position_source']=='PREDICTED_3D_WITH_CURRENT_RGB'
    assert t.tracks[0].last_measurement_timestamp == .1
    assert len(t.update([],0.3,camera_detections=[cam((56,50,116,90))]))==1
    assert len(t.update([],0.4,camera_detections=[cam((57,50,117,90))]))==1
    assert t.update([],0.5,camera_detections=[cam((57,50,117,90))])==[]


def test_camera_only_cannot_create_new_3d_track():
    t=tracker()
    for f in range(4):
        assert t.update([],f*.1,camera_detections=[cam()])==[]
    assert t.next_track_id==0


def test_low_confidence_or_unmatched_box_does_not_bridge():
    t=tracker();t.update([obj(20)],0);t.update([obj(20.1)],.1)
    assert t.update([], .2,camera_detections=[cam(confidence=.08)])==[]
    assert t.update([], .3,camera_detections=[cam((400,20,460,60))])==[]


def test_sparse_three_point_track_needs_multiple_camera_and_lidar_frames():
    t=tracker(); assert t.update([obj(30,n=3)],0)==[]
    assert len(t.update([obj(30.2,n=3,box=(52,50,112,90))],.1))==1


def test_scene_metric_reports_as_same_sensor_not_mesh_certification():
    pts=np.arange(60).reshape(20,3)*.01
    pts=np.vstack([pts,pts+np.array([0,2,0])])
    result=evaluate_same_sensor_geometry(pts,pts.copy(),[0,0,0])
    assert result['status']=='MEASURED_NOT_CERTIFIED'
    assert result['estimate_to_reference_median_m']==0
    assert result['independent_carla_mesh_validation'] is False
    assert result['scene_geometry_certified'] is False


def test_high_confidence_three_point_vehicle_requires_three_consistent_frames_when_score_low():
    t=tracker()
    low=obj(30,n=3)
    low['score']=.30
    assert t.update([low],0)==[]
    low2=obj(30.2,n=3,box=(52,50,112,90));low2['score']=.30
    assert t.update([low2],.1)==[]
    low3=obj(30.4,n=3,box=(54,50,114,90));low3['score']=.30
    got=t.update([low3],.2)
    assert len(got)==1 and got[0]['track_id']==0


def test_camera_bridge_remains_short_even_at_confidence_decay():
    t=tracker()
    first=obj(21,n=10);first['score']=.43
    second=obj(21.1,n=10);second['score']=.43
    t.update([first],0)
    assert len(t.update([second],.1))==1
    assert len(t.update([], .2, camera_detections=[cam(confidence=.35)]))==1
    assert len(t.update([], .3, camera_detections=[cam(confidence=.35)]))==1
    assert t.update([], .5, camera_detections=[cam(confidence=.35)])==[]
