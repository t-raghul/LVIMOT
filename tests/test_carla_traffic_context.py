"""Traffic-context inference is image-only and requires repeated rejection."""
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

from lvimot.tracking_4d.multimodal_object_tracking.traffic_context import CarlaTrafficContextScorer
from lvimot.preprocessing.camera_lidar_detection import CarlaCameraLiDARObjectDetector,CameraLiDARDetectionConfig


def parameters(tmp_path,**overrides):
    model=tmp_path/'features.pt';model.write_bytes(b'controlled-feature-model')
    values={'mean':np.zeros(2),'scale':np.ones(2),'weight':np.ones(2),'bias':np.asarray(0.),
            'schema_version':np.asarray(1),'feature_model_sha256':np.asarray(hashlib.sha256(model.read_bytes()).hexdigest()),
            'reject_threshold':np.asarray(.35),'temporal_confirmations':np.asarray(2)}
    values.update(overrides)
    path=tmp_path/'context.npz';np.savez(path,**values)
    return path,model


def scorer(tmp_path,scores):
    path,model=parameters(tmp_path)
    result=CarlaTrafficContextScorer(path,model)
    stream=iter(scores)
    result.score=lambda image,detections:np.asarray(next(stream),dtype=float)
    return result


def detection(box=None,confidence=.7):
    return {'bbox':box or [10.,10.,30.,30.],'class':'vehicle','camera_confidence':confidence}


def test_single_uncertain_image_observation_is_retained(tmp_path):
    context=scorer(tmp_path,[[.1]])
    result=context.filter(np.zeros((64,64,3),np.uint8),[detection()])
    assert len(result)==1
    assert result[0]['traffic_probability']==.1


def test_two_overlapping_nontraffic_images_are_required_for_rejection(tmp_path):
    context=scorer(tmp_path,[[.1],[.2]])
    image=np.zeros((64,64,3),np.uint8)
    assert len(context.filter(image,[detection()]))==1
    assert context.filter(image,[detection([11.,10.,31.,30.])])==[]


def test_positive_current_image_immediately_restores_semantic_candidate(tmp_path):
    context=scorer(tmp_path,[[.1],[.2],[.8]])
    image=np.zeros((64,64,3),np.uint8)
    context.filter(image,[detection()]);context.filter(image,[detection()])
    assert len(context.filter(image,[detection()]))==1


def test_spatially_unrelated_negative_box_cannot_remove_new_vehicle(tmp_path):
    context=scorer(tmp_path,[[.1],[.1]])
    image=np.zeros((100,100,3),np.uint8)
    context.filter(image,[detection()])
    assert len(context.filter(image,[detection([60.,60.,90.,90.])]))==1


def test_empty_frame_and_explicit_reset_clear_temporal_rejection(tmp_path):
    context=scorer(tmp_path,[[.1],[],[.1],[.1]])
    image=np.zeros((64,64,3),np.uint8)
    context.filter(image,[detection()]);context.filter(image,[])
    assert len(context.filter(image,[detection()]))==1
    context.reset()
    assert len(context.filter(image,[detection()]))==1


def test_filter_preserves_input_boxes_confidence_and_metadata(tmp_path):
    context=scorer(tmp_path,[[.8]])
    original=detection();original['actor_id']='arbitrary-untrusted-value'
    result=context.filter(np.zeros((64,64,3),np.uint8),[original])
    assert 'traffic_probability' not in original
    assert result[0]['bbox']==original['bbox']
    assert result[0]['camera_confidence']==original['camera_confidence']


def test_different_feature_model_is_rejected(tmp_path):
    path,model=parameters(tmp_path)
    model.write_bytes(b'different')
    with pytest.raises(ValueError,match='original feature-model'):
        CarlaTrafficContextScorer(path,model)


@pytest.mark.parametrize('overrides',[
    {'scale':np.zeros(2)}, {'weight':np.array([np.nan,0.])},
    {'schema_version':np.asarray(2)}, {'temporal_confirmations':np.asarray(1)},
    {'reject_threshold':np.asarray(1.1)},
])
def test_invalid_parameters_fail_explicitly(tmp_path,overrides):
    path,model=parameters(tmp_path,**overrides)
    with pytest.raises(ValueError):CarlaTrafficContextScorer(path,model)


def test_context_uses_real_image_pixels_and_keeps_crop_in_bounds():
    image=np.zeros((40,50,3),np.uint8);image[0:25,0:25]=[10,80,200]
    patches=CarlaTrafficContextScorer._crops(image,[[0,0,20,20]])
    assert patches[0].shape==(128,128,3)
    np.testing.assert_array_equal(patches[0][20,20],[10,80,200])
    with pytest.raises(ValueError):CarlaTrafficContextScorer._crops(image,[[60,0,70,20]])


def test_no_boxes_do_not_load_model(tmp_path):
    path,model=parameters(tmp_path)
    context=CarlaTrafficContextScorer(path,model)
    assert context.score(np.zeros((64,64,3),np.uint8),[]).shape==(0,)
    assert context._model is None


def test_context_can_be_disabled_without_loading_sidecar(tmp_path):
    bad=tmp_path/'bad.npz';bad.write_bytes(b'not an archive')
    cfg=CameraLiDARDetectionConfig(traffic_context_enabled=False,traffic_context_path=str(bad))
    detector=CarlaCameraLiDARObjectDetector(None,cfg)
    boxes=[detection()]
    assert detector._filter_traffic_context(np.zeros((64,64,3),np.uint8),boxes) is boxes


def test_explicit_missing_context_file_fails_instead_of_silent_fallback(tmp_path):
    cfg=CameraLiDARDetectionConfig(traffic_context_path=str(tmp_path/'missing.npz'))
    detector=CarlaCameraLiDARObjectDetector(None,cfg)
    with pytest.raises(FileNotFoundError):detector._filter_traffic_context(np.zeros((64,64,3),np.uint8),[])


def test_absent_optional_sidecar_preserves_original_semantic_boxes(tmp_path):
    cfg=CameraLiDARDetectionConfig(model_path=str(tmp_path/'model.pt'))
    detector=CarlaCameraLiDARObjectDetector(None,cfg)
    boxes=[detection()]
    assert detector._filter_traffic_context(np.zeros((64,64,3),np.uint8),boxes) is boxes
