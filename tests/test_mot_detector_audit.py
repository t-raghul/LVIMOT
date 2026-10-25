import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('mot_audit', Path(__file__).resolve().parents[1] / 'tools' / 'audit_saved_mot_detector.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def test_bbox_iou():
 assert mod.bbox_iou([0,0,2,2],[0,0,2,2])==1
 assert mod.bbox_iou([0,0,2,2],[2,2,4,4])==0
 assert mod.bbox_iou([0,0,2,2],[1,1,3,3]) == 1/7
 assert mod.bbox_iou(None,[0,0,1,1]) ==0

def test_best_overlap():
 a,b=mod.best_overlap([0,0,2,2],[{'bbox':[10,10,12,12]},{'bbox':[0,0,2,2],'score':0.8}])
 assert a==1 and b['score']==0.8
 assert mod.best_overlap([0,0,2,2],[])==(0,None)

def test_outcomes():
 assert mod.diagnostic_outcome(0,.9)=='NO_MATCHING_YOLO_2D_BOX'
 assert mod.diagnostic_outcome(.5,.2)=='YOLO_BOX_WITHOUT_MATCHING_LIDAR_3D_PROPOSAL'
 assert mod.diagnostic_outcome(.5,.5)=='MATCHING_CAMERA_AND_LIDAR_PROPOSAL_AVAILABLE'
