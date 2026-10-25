"""Experimental image-only traffic-context scoring for CARLA recordings.

The supervised target is a traffic participant versus a static scene mesh or
background. It is specific to the training domain. It never reads simulator
actor IDs, poses, maps, evaluation labels, or recording paths during inference.
"""

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[3]))

import hashlib
from pathlib import Path

import cv2
import numpy as np


class CarlaTrafficContextScorer:
    def __init__(self, parameters_path, feature_model_path, device='cpu'):
        with np.load(parameters_path,allow_pickle=False) as parameters:
            self.mean=parameters['mean'].copy()
            self.scale=parameters['scale'].copy()
            self.weight=parameters['weight'].copy()
            self.bias=float(parameters['bias'])
            version=int(parameters['schema_version'])
            expected_hash=str(parameters['feature_model_sha256'])
            self.threshold=float(parameters['reject_threshold'])
            confirmations=int(parameters['temporal_confirmations'])
        if (self.mean.ndim!=1 or self.mean.shape!=self.scale.shape
                or self.mean.shape!=self.weight.shape or np.any(self.scale<=0)
                or not all(np.isfinite(v).all() for v in [self.mean,self.scale,self.weight,np.asarray(self.bias)])):
            raise ValueError('Invalid traffic-context classifier parameters')
        if version!=1 or confirmations!=2 or not 0.<self.threshold<1.:
            raise ValueError('Unsupported traffic-context classifier schema')
        if hashlib.sha256(Path(feature_model_path).read_bytes()).hexdigest()!=expected_hash:
            raise ValueError('Traffic-context classifier requires its original feature-model weights')
        self.feature_model_path=str(feature_model_path)
        self.device=device
        self._model=None
        self._previous=[]

    @staticmethod
    def _iou(a,b):
        x=max(0.,min(a[2],b[2])-max(a[0],b[0]))
        y=max(0.,min(a[3],b[3])-max(a[1],b[1]))
        intersection=x*y
        area_a=max(0.,a[2]-a[0])*max(0.,a[3]-a[1])
        area_b=max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
        union=area_a+area_b-intersection
        return intersection/union if union>0 else 0.

    def reset(self):
        self._previous=[]

    def filter(self,image,detections):
        """Reject only after two overlapping image observations score low.

        A single uncertain crop does not remove a vehicle. A positive current
        observation is always retained. This changes semantic eligibility only;
        all downstream metric positions still require actual measured LiDAR.
        """
        scores=self.score(image,detections)
        current=[];kept=[]
        for detection,score in zip(detections,scores):
            item=dict(detection)
            item['traffic_probability']=float(score)
            previous_negative=any(self._iou(item['bbox'],old['bbox'])>=.3
                                  and old['traffic_probability']<self.threshold
                                  for old in self._previous)
            if score>=self.threshold or not previous_negative:
                kept.append(item)
            current.append(item)
        self._previous=current
        return kept

    @staticmethod
    def _crops(image,boxes):
        height,width=image.shape[:2]
        patches=[]
        for box in boxes:
            x1,y1,x2,y2=np.asarray(box,dtype=np.float64)
            if not np.isfinite([x1,y1,x2,y2]).all() or x2<=x1 or y2<=y1:
                raise ValueError('Invalid image box for traffic context')
            dx=.25*(x2-x1);dy=.25*(y2-y1)
            x1,x2=max(0,int(np.floor(x1-dx))),min(width,int(np.ceil(x2+dx)))
            y1,y2=max(0,int(np.floor(y1-dy))),min(height,int(np.ceil(y2+dy)))
            if x2<=x1 or y2<=y1:raise ValueError('Traffic-context box is outside image')
            patches.append(cv2.resize(image[y1:y2,x1:x2],(128,128),interpolation=cv2.INTER_LINEAR))
        return patches

    def features(self,image,boxes):
        if not boxes:return np.empty((0,len(self.mean)),dtype=np.float32)
        if self._model is None:
            from ultralytics import YOLO
            from lvimot.output.compute_runtime import configure_detector_threads
            configure_detector_threads()
            self._model=YOLO(self.feature_model_path)
        patches=self._crops(image,boxes);rows=[]
        for start in range(0,len(patches),16):
            embeddings=self._model.embed(patches[start:start+16],imgsz=128,
                device=self.device,half=False,embed=[8,15,21],verbose=False)
            rows.extend(x.detach().cpu().numpy() for x in embeddings)
        features=np.asarray(rows,dtype=np.float32)
        if features.shape!=(len(boxes),len(self.mean)) or not np.isfinite(features).all():
            raise ValueError('Unexpected traffic-context image embeddings')
        return features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-8)

    def score(self,image,detections):
        features=self.features(image,[d['bbox'] for d in detections])
        logits=((features-self.mean)/self.scale)@self.weight+self.bias
        return 1./(1.+np.exp(-np.clip(logits,-40.,40.)))
