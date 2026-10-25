"""Independent, evaluation-only CARLA road geometry comparison.

The CARLA HD-map geometry is NEVER passed to the LVIMOT estimator, tracker,
mapper or optimizer. Only call this after estimation has completed.

This is a road-surface agreement check, not a claim that an entire 3D map or
4D tracking history has been verified by the OpenDRIVE reference.
"""
from __future__ import annotations

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[3]))


import json
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree

from lvimot.output.map_export import write_binary_ply


def _reference_pose(anchor: Dict):
    xyz = anchor['location']
    rot = anchor['rotation']
    translation = np.asarray([xyz['x'], xyz['y'], xyz['z']], dtype=float)
    pitch, yaw, roll = map(math.radians, [rot.get('pitch', 0.0), rot.get('yaw', 0.0), rot.get('roll', 0.0)])
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    R = np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp, -cp * sr, cp * cr],
    ], dtype=float)
    return R, translation


def align_local_map_to_carla(xyz: np.ndarray, anchor: Dict) -> np.ndarray:
    """Rigidly place local estimated coordinates at CARLA's first GT pose.

    The anchor is evaluation-only: this makes coordinates comparable, but
    does NOT correct drift accumulated by the sensor estimator.
    """
    R, translation = _reference_pose(anchor)
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    return (xyz @ R.T + translation).astype(np.float32)


def road_reference_from_carla_map(carla_map, center_xyz, radius_m=70.0, sampling_m=2.0):
    """Independently sample drivable CARLA road surfaces from OpenDRIVE waypoints.

    This is GT road geometry, not another LiDAR scan. Points are sampled
    across the lane at several lateral offsets, not just the centerline.
    """
    cx, cy = float(center_xyz[0]), float(center_xyz[1])
    pts = []
    for wp in carla_map.generate_waypoints(float(sampling_m)):
        try:
            loc, yaw = wp.transform.location, math.radians(float(wp.transform.rotation.yaw))
            if (float(loc.x)-cx)**2 + (float(loc.y)-cy)**2 > radius_m**2:
                continue
            if str(wp.lane_type).lower().find('driving') < 0:
                continue
            half = max(0.6, min(3.0, float(wp.lane_width) * 0.45))
            for offset in (-half, -0.5*half, 0.0, 0.5*half, half):
                x = float(loc.x) - math.sin(yaw)*offset
                y = float(loc.y) + math.cos(yaw)*offset
                pts.append((x,y,float(loc.z)))
        except (AttributeError, ValueError, TypeError):
            continue
    return np.asarray(pts, dtype=np.float32).reshape(-1, 3)


def compare_road_geometry(map_world_xyz, road_xyz, center_xyz, local_radius_m=45.0):
    """GT road agreement for road-height map returns in final local area.

    A matched return is within 1.25m horizontally and 0.60m vertically of
    an independently sampled CARLA road surface. Unobserved road is not
    counted as a mapping failure, and sidewalks/vegetation are excluded.
    """
    mapped = np.asarray(map_world_xyz, dtype=float).reshape(-1, 3)
    ref = np.asarray(road_xyz, dtype=float).reshape(-1, 3)
    center = np.asarray(center_xyz, dtype=float).reshape(3)
    result = {
        'comparison': 'LVIMOT road-height returns vs CARLA OpenDRIVE road surface',
        'reference_is_independent': True,
        'uses_gt_for_estimation': False,
        'scope': 'final local neighborhood, observed road surface only',
        'radius_m': float(local_radius_m), 'road_match_radius_xy_m': 1.25,
        'road_height_tolerance_m': 0.6,
        'map_points_in_area': 0, 'reference_road_samples': 0,
        'candidate_road_height_map_points': 0, 'road_agreement_pct': None,
        'median_road_xy_error_m': None, 'road_match_count': 0,
        'status': 'INSUFFICIENT_EVIDENCE',
    }
    if not len(mapped) or not len(ref):
        result['reason'] = 'Map or CARLA drivable-road reference is empty'
        return result
    in_map_area = np.linalg.norm(mapped[:,:2]-center[:2],axis=1) <= local_radius_m
    in_ref_area = np.linalg.norm(ref[:,:2]-center[:2],axis=1) <= local_radius_m
    mapped = mapped[in_map_area]
    ref = ref[in_ref_area]
    result['map_points_in_area'] = len(mapped)
    result['reference_road_samples'] = len(ref)
    if len(mapped) < 40 or len(ref) < 40:
        result['reason'] = 'Too few measured map points or reference road samples'
        return result

    tree = cKDTree(ref[:,:2])
    xy_dist, indices = tree.query(mapped[:,:2], k=1)
    z_diff = np.abs(mapped[:,2]-ref[indices,2])
    # Classification uses reference height, so every measured return near
    # expected road elevation can be tested (does not accept tall buildings).
    road_height = z_diff <= 0.6
    candidates = int(road_height.sum())
    result['candidate_road_height_map_points'] = candidates
    if candidates < 40:
        result['reason'] = 'Not enough road-height observations; check pose alignment or map drift'
        return result
    agreed = xy_dist[road_height] <= 1.25
    nmatch = int(agreed.sum())
    result['road_match_count'] = nmatch
    result['road_agreement_pct'] = round(100.0*nmatch/candidates, 2)
    result['median_road_xy_error_m'] = (round(float(np.median(xy_dist[road_height])),3))
    # CRITICAL: ground-height != semantic driving road. Sidewalks, curbs,
    # parking lots and building bases at the same elevation are counted by
    # this historical proxy. This can score ~50% even with perfect GT poses.
    # Never interpret road_agreement_pct as full-map accuracy or use it alone
    # to certify a map. Need independent per-point road semantics / mesh GT.
    result['road_metric_warning'] = (
        'Mixed-scene height proxy, NOT semantic road precision. ' 
        'Ground-height LiDAR on sidewalks or building bases lowers this score ' 
        'even when the estimated 3D coordinates are correct.'
    )
    result['can_certify_full_3d_map'] = False
    # Do not turn arbitrary percentages into 'validated'. That threshold must
    # be defined in an experiment protocol before collecting results.
    result['status'] = 'MEASURED_NOT_CERTIFIED'
    return result


def _svg_overlay(map_world, reference, center, path: Path, radius=45.0):
    """Write an inspectable top-down overlay without GUI or matplotlib dependency."""
    width=850; half=radius
    pts=np.asarray(map_world,dtype=float).reshape(-1,3)
    ref=np.asarray(reference,dtype=float).reshape(-1,3)
    center=np.asarray(center,dtype=float).reshape(3)
    def dots(points,color,limit):
        if not len(points): return ''
        inside=points[np.linalg.norm(points[:,:2]-center[:2],axis=1)<half]
        if len(inside)>limit: inside=inside[::max(1, int(math.ceil(len(inside)/limit)))][:limit]
        chunks=[]
        for p in inside:
            x=width/2+(p[0]-center[0])*width/(2*half)
            y=width/2-(p[1]-center[1])*width/(2*half)
            chunks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.1" fill="{color}"/>')
        return ''.join(chunks)
    # Height is a visual category ONLY; it is NOT a semantic road label.
    # Normal building/vegetation points need not overlap OpenDRIVE lanes.
    low = pts[np.abs(pts[:, 2]-center[2]) <= .65]
    tall = pts[np.abs(pts[:, 2]-center[2]) > .65]
    body = (dots(ref,'#50d98c',9000) + dots(tall,'#ed6b5a',4500)
            + dots(low,'#65cfff',6500))
    html=(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {width+60}" width="{width}">'
          '<rect width="100%" height="100%" fill="#101b24"/>'
          '<text x="15" y="25" fill="white" font-size="16">CARLA road: green | LVIMOT low-height: cyan | other 3D scene: orange</text>'
          f'<g transform="translate(0,45)">{body}</g>'
          f'<text x="15" y="{width+44}" fill="white" font-size="12">{2*radius:.0f}m across | CARLA is evaluation-only | low-height is NOT road segmentation</text>'
          '</svg>')
    path.write_text(html,encoding='utf-8')


def validate_against_carla(world, mapper, first_ground_truth_pose: Dict,
                           latest_ground_truth_pose: Dict, output_dir: Path) -> Dict:
    """Run once AFTER estimation; export independently sourced CARLA comparison."""
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    last=latest_ground_truth_pose['location']
    center=np.array([last['x'],last['y'],last['z']], dtype=np.float64)
    road=road_reference_from_carla_map(world.get_map(),center, radius_m=70.0)
    raw=mapper.get_static_map_points()
    aligned=align_local_map_to_carla(raw, first_ground_truth_pose)
    write_binary_ply(output_dir/'LVIMOT_aligned_CARLA_frame_EVALUATION_ONLY.ply',aligned)
    write_binary_ply(output_dir/'CARLA_reference_road_surface.ply',road)
    metric=compare_road_geometry(aligned,road,center)
    road_layer=getattr(mapper,'road_surface_layer',None)
    if road_layer is not None:
        measured=align_local_map_to_carla(road_layer.points(),first_ground_truth_pose)
        road_metric=compare_road_geometry(measured,road,center)
        road_metric.update({'comparison':'Camera-labelled, measured LiDAR road layer vs CARLA OpenDRIVE',
            'classification_source':'Pretrained Cityscapes image semantics and measured LiDAR surface support',
            'scope':'camera-observed road surface only, final local neighborhood',
            'road_metric_warning':'Predicted road labels and finite OpenDRIVE sampling can be wrong. This does not certify the full scene or unobserved road.',
            'retained_full_scene_map':True,'road_layer_voxels':len(measured),
            'fraction_of_full_scene_voxels':float(len(measured)/max(1,len(raw)))})
        metric['measured_road_surface']=road_metric
        write_binary_ply(output_dir/'LVIMOT_road_surface_aligned_EVALUATION_ONLY.ply',measured)
    metric['coordinate_alignment']='first GT ego pose, applied only AFTER map estimation; no drift correction'
    metric['raw_estimated_frame_map']='live_map_latest.ply'
    metric['aligned_eval_map']='LVIMOT_aligned_CARLA_frame_EVALUATION_ONLY.ply'
    metric['CARLA_reference_map']='CARLA_reference_road_surface.ply'
    metric['visual_overlay']='CARLA_vs_LVIMOT_road_overlay.svg'
    _svg_overlay(aligned,road,center,output_dir/metric['visual_overlay'])
    (output_dir/'CARLA_MAP_VALIDATION.json').write_text(json.dumps(metric,indent=2)+'\n',encoding='utf-8')
    return metric
