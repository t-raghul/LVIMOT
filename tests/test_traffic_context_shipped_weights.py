"""Exercise the real shipped image model on four training-only crop fixtures.

These check inference reproducibility, not performance on unseen images.
Full raw replay evidence is provided separately in docs/mota_high_20261021.
"""
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]

from lvimot.tracking_4d.multimodal_object_tracking.traffic_context import CarlaTrafficContextScorer


@pytest.fixture(scope='module')
def actual_predictions(tmp_path_factory):
    torch=pytest.importorskip('torch')
    previous=os.environ.get('YOLO_CONFIG_DIR')
    os.environ['YOLO_CONFIG_DIR']=str(tmp_path_factory.mktemp('traffic_context_yolo'))
    try:
        pytest.importorskip('ultralytics')
        old_threads=torch.get_num_threads();torch.set_num_threads(2)
        try:
            scorer=CarlaTrafficContextScorer(ROOT/'models/yolov8n.traffic_context.npz',ROOT/'models/yolov8n.pt',device='cpu')
            with np.load(ROOT/'tests/fixtures/traffic_context_training_crops.npz',allow_pickle=False) as fixture:
                patches=fixture['patches'].copy();scores=fixture['scores'].copy();labels=fixture['labels'].copy()
            measured=[float(scorer.score(image,[{'bbox':[21.3334,21.3334,106.6666,106.6666]}])[0]) for image in patches]
            return measured,scores,labels
        finally:torch.set_num_threads(old_threads)
    finally:
        if previous is None:os.environ.pop('YOLO_CONFIG_DIR',None)
        else:os.environ['YOLO_CONFIG_DIR']=previous


@pytest.mark.parametrize('index',range(4))
def test_real_shipped_model_reproduces_crop_score(actual_predictions,index):
    measured,expected,labels=actual_predictions
    assert measured[index]==pytest.approx(float(expected[index]),abs=5e-5)
    assert int(measured[index]>=.35)==int(labels[index])
