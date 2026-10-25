"""Compact foreground layers and explicitly named custom vehicle models."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from lvimot.preprocessing.camera_lidar_detection import CarlaCameraLiDARObjectDetector as Detector
from lvimot.preprocessing.camera_lidar_detection import CameraLiDARDetectionConfig


@pytest.mark.parametrize('count',[4,5,6,7])
def test_compact_measured_foreground_wins_over_dense_wall(count):
    depths=np.r_[np.linspace(40.,41.,60),np.linspace(20.,21.,count)]
    order=np.random.RandomState(12).permutation(len(depths))
    indices=order.copy()
    selected=Detector._depth_cluster(depths,indices)
    assert set(selected)==set(range(60,60+count))
    assert len(selected)==count


def test_two_nearby_returns_cannot_replace_wall_with_vehicle_depth():
    depths=np.r_[20.,20.2,np.linspace(40.,41.,60)]
    assert len(Detector._depth_cluster(depths,np.arange(len(depths))))==0


def test_broad_foreground_does_not_pass_compact_layer_shortcut():
    depths=np.r_[20.,21.,22.,23.1,np.linspace(40.,41.,60)]
    selected=Detector._depth_cluster(depths,np.arange(len(depths)))
    assert set(selected)!=set(range(4))


def test_compactness_does_not_join_separated_foreground_objects():
    depths=np.r_[20.,20.1,20.2,20.3,25.,25.1,25.2,25.3,np.linspace(40.,41.,60)]
    selected=Detector._depth_cluster(depths,np.arange(len(depths)))
    assert set(selected)==set(range(4))


class Model:
    def __init__(self,names=None):
        if names is not None:self.names=names

    def predict(self,**kwargs):
        boxes=SimpleNamespace(xyxy=np.array([[2,3,20,25],[30,3,50,25]]),
                              cls=np.array([0,2]),conf=np.array([.8,.7]))
        return [SimpleNamespace(boxes=boxes)]


def infer(model):
    detector=Detector(None,CameraLiDARDetectionConfig(device='cpu'),model)
    return detector._infer_single_view(np.zeros((64,64,3),dtype=np.uint8))


def test_single_class_vehicle_zero_is_accepted_by_explicit_model_name():
    detections=infer(Model({0:'vehicle'}))
    assert len(detections)==1
    assert detections[0]['class_id']==0
    assert detections[0]['class']=='vehicle'


def test_named_coco_person_zero_is_never_treated_as_vehicle():
    detections=infer(Model({0:'person',2:'car'}))
    assert [d['class_id'] for d in detections]==[2]


def test_missing_names_preserves_legacy_coco_class_mapping():
    assert [d['class_id'] for d in infer(Model())]==[2]


def test_unknown_or_out_of_range_named_classes_are_ignored():
    assert infer(Model(['unknown']))==[]


def test_named_class_mapping_normalizes_case_and_whitespace():
    assert [d['class'] for d in infer(Model({0:' TRUCK ',2:'BICYCLE'}))]==['vehicle','bicycle']
