"""A separate measured road layer: image semantics, current LiDAR, estimated pose.

The full scene map is never filtered or flattened by this module. Simulator
labels, map waypoints and evaluation references are not accepted as inputs.
"""

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[3]))

from pathlib import Path
import time

import cv2
import numpy as np

from lvimot.mapping_4d.static_map.rolling_voxel_map import Carla4DEnvironmentMapper


class MeasuredRoadSurfaceLayer:
    def __init__(self,calibration,model_path,device='cpu',voxel_size=.25,radius=60.,capacity=150000,confidence=.70):
        self.calibration=calibration
        self.model_path=Path(model_path)
        self.device='cuda:0' if str(device) in {'0','cuda','cuda:0'} else str(device)
        self.confidence=float(confidence)
        self.model=None
        self.mapper=Carla4DEnvironmentMapper(static_voxel_size=voxel_size,max_map_distance=radius,max_static_voxels=capacity)
        self.frames_integrated=0
        self.last_diagnostics={}

    def _road_pixels(self,image):
        import torch
        if self.model is None:
            self.model=torch.jit.load(str(self.model_path),map_location=self.device).eval()
        rgb=cv2.cvtColor(cv2.resize(image,(832,256),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
        tensor=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).unsqueeze(0).to(self.device,dtype=torch.float32)/255.
        mean=tensor.new_tensor([.485,.456,.406]).view(1,3,1,1)
        std=tensor.new_tensor([.229,.224,.225]).view(1,3,1,1)
        with torch.inference_mode():
            logits=self.model((tensor-mean)/std)
            logits=torch.nn.functional.interpolate(logits,size=image.shape[:2],mode='bilinear',align_corners=False)
            probability=torch.softmax(logits,dim=1)[0,0]
            selected=(logits.argmax(dim=1)[0]==0)&(probability>=self.confidence)
            mask=selected.to(torch.uint8).cpu().numpy()
        # Avoid mixing sidewalk/curb pixels across the segmentation boundary.
        return cv2.erode(mask,np.ones((3,3),np.uint8),iterations=1).astype(bool)

    @staticmethod
    def _measured_surface(points):
        """Reject elevated/vertical returns without changing any measured XYZ."""
        points=np.asarray(points,dtype=np.float64)
        if len(points)<40:return np.zeros(len(points),bool),None
        near=(points[:,0]>=3.)&(points[:,0]<=30.)&(np.abs(points[:,1])<=12.)
        pool=points[near]
        if len(pool)<40:return np.zeros(len(points),bool),None
        design=np.column_stack([pool[:,:2],np.ones(len(pool))])
        beta=np.array([0.,0.,np.percentile(pool[:,2],40)])
        for _ in range(5):
            selected=np.abs(pool[:,2]-design@beta)<=.20
            if np.count_nonzero(selected)<30:return np.zeros(len(points),bool),None
            beta=np.linalg.lstsq(design[selected],pool[selected,2],rcond=None)[0]
        residual=pool[selected,2]-design[selected]@beta
        if not np.isfinite(beta).all() or np.linalg.norm(beta[:2])>.30 or np.sqrt(np.mean(residual**2))>.12:
            return np.zeros(len(points),bool),None
        distance=np.abs(points[:,2]-points[:,:2]@beta[:2]-beta[2])
        tolerance=.15+.002*np.linalg.norm(points[:,:2],axis=1)
        return distance<=tolerance,beta

    def update(self,image,lidar_points,frame_id,timestamp,estimated_pose,lidar_to_ego,tracks):
        start=time.perf_counter()
        mask=self._road_pixels(image)
        xyz=np.asarray(lidar_points,dtype=np.float64)[:,:3]
        uv,valid=self.calibration.project_lidar(xyz)
        valid &= np.isfinite(xyz).all(axis=1)&np.isfinite(uv).all(axis=1)
        pixels=np.rint(np.nan_to_num(uv,nan=-1e6,posinf=-1e6,neginf=-1e6)).astype(np.int64)
        valid &= (pixels[:,0]>=0)&(pixels[:,0]<image.shape[1])&(pixels[:,1]>=0)&(pixels[:,1]<image.shape[0])
        visible=np.flatnonzero(valid)
        ids=visible[mask[pixels[visible,1],pixels[visible,0]]]
        consistent,plane=self._measured_surface(xyz[ids])
        measured=xyz[ids[consistent]]
        if len(measured):
            ego_points=lidar_to_ego(measured)
            self.mapper.add_frame_data(frame_id,timestamp,ego_points,[],[],estimated_pose,
                update_static=True,update_planar=False,evict_static=(frame_id%30==0),mask_tracks=tracks)
            self.frames_integrated+=1
        self.last_diagnostics={'status':'MEASURED_CAMERA_LIDAR_ROAD' if len(measured) else 'INSUFFICIENT_ROAD_SUPPORT',
            'visible_lidar_points':len(visible),'semantic_road_points':len(ids),'retained_measured_road_points':len(measured),
            'confidence_threshold':self.confidence,'ground_truth_used':False,'points_flattened':False,
            'camera_field_of_view_only':True,'measured_plane':plane.tolist() if plane is not None else None,
            'milliseconds':(time.perf_counter()-start)*1000}
        return dict(self.last_diagnostics)

    def points(self):return self.mapper.get_static_map_points()

    def summary(self):
        return {'enabled':True,'source':'Cityscapes road semantics + current measured LiDAR + LVIMOT estimated pose',
            'voxel_count':len(self.mapper.static_voxel_map),'frames_integrated':self.frames_integrated,
            'voxel_size_m':self.mapper.static_voxel_size,'radius_m':self.mapper.max_map_distance,
            'ground_truth_used':False,'full_scene_map_preserved':True,'last_frame':dict(self.last_diagnostics)}
