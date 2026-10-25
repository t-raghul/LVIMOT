"""Regression checks for the complete offline scene; no CARLA/YOLO dependency."""
import json
import os
import sys
from pathlib import Path

import numpy as np


from lvimot.mapping_4d.multiobject_trajectory_management import IndividualObjectScene, objects_to_local
from lvimot.mapping_4d.static_map.rolling_voxel_map import Carla4DEnvironmentMapper
from lvimot.output.offline_map import export_offline_4d_scene


def fgo(tid, x, speed):
    return {'track_id': tid, 'position': [float(x), 1., 0.], 'velocity': [float(speed), 0., 0.],
            'dimensions': {'length': 4.2, 'width': 1.9, 'height': 1.7},
            'rotation': {'yaw': 10.}, 'class': 'car', 'source': 'temporal_fusion_fgo',
            'motion_state': 'STATIC' if speed == 0 else 'DYNAMIC'}


def sample_mapper(tmp_path):
    mapper = Carla4DEnvironmentMapper()
    mapper.enable_accumulated_map(tmp_path/'archive')
    pose = {'location': {'x':0.,'y':0.,'z':0.},'rotation':{'yaw':0.,'pitch':0.,'roll':0.}}
    for frame in range(15):
        now = frame*0.2
        tracks = [fgo(7, frame*0.5, 2.5), fgo(8, 10., 0.)]
        points = np.array([[2.,2.,.0],[3.,4.,.1],[20.+frame,3.,.1]], dtype=float)
        mapper.add_frame_data(frame, now, points, [], tracks, pose,
                              update_static=True, update_planar=False)
    return mapper


def test_parked_box_persists_for_live_display_after_missing_measurement():
    scene = IndividualObjectScene()
    for frame in range(15):
        scene.update(frame, frame*.2, [fgo(5, 8., 0.)])
    scene.update(20, 4., [])
    assert scene.current(6.) == []  # old one-second live-only behavior preserved
    historical = scene.current(6., include_historical=True)
    assert len(historical)==1
    assert historical[0]['motion_status']=='PARKED_CANDIDATE'
    assert historical[0]['draw_box'] is True
    localized=objects_to_local(historical,[0,0,0,0,0,0],radius=60)
    assert len(localized)==1 and localized[0]['draw_box']
    assert len(localized[0]['box_corners_local'])==8


def test_moving_trajectory_retained_with_no_fake_last_box():
    scene = IndividualObjectScene()
    scene.update(0, 0., [fgo(7, 2., 3.)])
    scene.update(1, .2, [fgo(7, 3., 3.)])
    old=scene.current(10.,include_historical=True)
    assert len(old)==1 and old[0]['motion_status']=='HISTORICAL_PATH'
    assert old[0]['draw_box'] is False
    assert len(old[0]['trajectory'])==2


def test_complete_offline_ply_and_json_with_parked_boxes_and_trajectory(tmp_path):
    mapper = sample_mapper(tmp_path)
    try:
        result = export_offline_4d_scene(mapper,tmp_path/'output')
        assert result['offline_scene_counts']['object_ids']==2
        assert result['offline_scene_counts']['parked_candidates']==1
        assert result['offline_scene_counts']['moving_paths']==1
        assert result['offline_scene_counts']['static_voxels']>0
        assert result['offline_scene_counts']['unified_ply_vertices']>result['offline_scene_counts']['static_voxels']
        scene=json.loads(Path(result['offline_scene_json']).read_text())
        assert scene['ground_truth_used'] is False
        assert set(scene['objects'])=={'7','8'}
        assert scene['objects']['8']['last_status']=='PARKED_CANDIDATE'
        assert len(scene['objects']['7']['trajectory'])==15
        assert len(scene['objects']['8']['history'])==15
        assert Path(result['offline_scene_html']).is_file()
        assert '<canvas' in Path(result['offline_scene_html']).read_text()
        assert 'https://' not in Path(result['offline_scene_html']).read_text()
        with Path(result['offline_scene_ply']).open('rb') as f:
            header=b''
            while not header.endswith(b'end_header\n'):
                header+=f.readline()
            size=Path(result['offline_scene_ply']).stat().st_size-f.tell()
        assert ('element vertex '+str(result['offline_scene_counts']['unified_ply_vertices'])).encode() in header
        assert size==result['offline_scene_counts']['unified_ply_vertices']*15
        assert mapper.object_archive.count_objects()==2
        summary=mapper.object_archive.summary()
        assert summary['total_observations']==30
        assert summary['ground_truth_used'] is False
    finally:
        mapper.close()


def test_status_transition_removes_parked_box_and_keeps_original_id(tmp_path):
    mapper=sample_mapper(tmp_path)
    try:
        pose={'location':{'x':0.,'y':0.,'z':0.},'rotation':{'yaw':0.}}
        mapper.add_frame_data(16,3.2,np.empty((0,3)),[],[fgo(8,11.,2.)],pose,
                              update_static=False,update_planar=False)
        result=export_offline_4d_scene(mapper,tmp_path/'output')
        scene=json.loads(Path(result['offline_scene_json']).read_text())
        assert set(scene['objects'])=={'7','8'}
        assert scene['objects']['8']['last_status']=='MOVING'
        assert scene['objects']['8']['trajectory'][-1]['position'][0]==11.
        assert result['offline_scene_counts']['parked_candidates']==0
    finally:
        mapper.close()


def test_retains_history_even_when_ram_scene_evicts_id(tmp_path):
    mapper=Carla4DEnvironmentMapper(max_dynamic_history=8)
    mapper.individual_object_scene.max_objects=1
    mapper.enable_accumulated_map(tmp_path/'archive')
    pose={'location':{'x':0.,'y':0.,'z':0.},'rotation':{}}
    try:
        for i in range(4):
            mapper.add_frame_data(i,float(i),np.empty((0,3)),[],[fgo(i,i+1,0.)],pose,
                                  update_static=False,update_planar=False)
        assert len(mapper.individual_object_scene.objects)==1
        result=export_offline_4d_scene(mapper,tmp_path/'output')
        assert result['offline_scene_counts']['object_ids']==4
        assert len(json.loads(Path(result['offline_scene_json']).read_text())['objects'])==4
    finally:
        mapper.close()


def test_complete_driven_road_archive_survives_rolling_eviction(tmp_path):
    from types import SimpleNamespace
    mapper=Carla4DEnvironmentMapper(max_map_distance=4., static_voxel_size=.25)
    road_mapper=Carla4DEnvironmentMapper(max_map_distance=4., static_voxel_size=.25)
    road_layer=SimpleNamespace(mapper=road_mapper, points=road_mapper.get_static_map_points,
                               summary=lambda: {'enabled': True})
    mapper.road_surface_layer=road_layer
    mapper.enable_accumulated_map(tmp_path/'run')
    pose={'location': {'x':0,'y':0,'z':0}, 'rotation':{'yaw':0}}
    try:
        for frame,x in enumerate((1., 2., 10.)):
            pose['location']['x']=x
            pts=np.array([[0.,0.,0.]], dtype=float)
            mapper.add_frame_data(frame,frame*.2,pts,[],[],pose,
                                  update_static=True,update_planar=False)
            road_mapper.add_frame_data(frame,frame*.2,pts,[],[],pose,
                                       update_static=True,update_planar=False,
                                       evict_static=True)
        assert len(road_mapper.static_voxel_map)<3  # local view only
        out=export_offline_4d_scene(mapper,tmp_path/'output')
        assert out['offline_scene_counts']['road_voxels']==3  # complete route
        assert out['offline_scene_counts']['static_voxels']==3
    finally:
        mapper.close()



def _write_xyz_ply(path, xyz):
    pts = np.asarray(xyz, dtype='<f4').reshape(-1, 3)
    header = ('ply\nformat binary_little_endian 1.0\n'
              'element vertex {}\n'
              'property float x\nproperty float y\nproperty float z\nend_header\n'.format(len(pts)))
    with Path(path).open('wb') as stream:
        stream.write(header.encode('ascii'))
        stream.write(pts.tobytes(order='C'))


def test_offline_html_can_embed_matched_independent_360_reference(tmp_path):
    mapper = sample_mapper(tmp_path)
    try:
        live = tmp_path / 'output'
        live.mkdir(parents=True, exist_ok=True)
        first = {'location': {'x': 20.0, 'y': -10.0, 'z': 1.0},
                 'rotation': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}}
        capture = live / 'sensor_recordings' / 'capture_test'
        capture.mkdir(parents=True)
        (capture / 'manifest.json').write_text(json.dumps({
            'format': 'LVIMOT_RAW_SENSOR_ARCHIVE_V1', 'frames': 'frames.jsonl'
        }), encoding='utf-8')
        (capture / 'frames.jsonl').write_text(json.dumps({
            'ground_truth_ego_pose_evaluation_only': first
        }) + '\n', encoding='utf-8')
        (live / 'live_results.json').write_text(json.dumps({
            'ground_truth_evaluation_enabled': True,
            'raw_sensor_recording': r'outputs_carla\live\sensor_recordings\capture_test'
        }), encoding='utf-8')
        folder = capture / 'original_carla_map'
        folder.mkdir()
        reference = np.array([[0., 0., 0.], [1., 2., .1], [4., 5., 0.]], dtype=float)
        _write_xyz_ply(folder / 'CARLA_independent_360_reference.ply', reference)
        (folder / 'reference_manifest.json').write_text(json.dumps({
            'format': 'CARLA_INDEPENDENT_RENDERED_360_REFERENCE_V1',
            'recorded_capture': str(capture.resolve()),
            'reference_only': True,
            'gt_used_by_estimator': False
        }), encoding='utf-8')
        result = export_offline_4d_scene(mapper, live,
                    first_gt_pose_evaluation_only=first)
        page = Path(result['offline_scene_html']).read_text(encoding='utf-8')
        manifest = json.loads(Path(result['offline_scene_manifest']).read_text(encoding='utf-8'))
        assert 'id="showCarla" type="checkbox" checked' in page
        assert 'id="showLVIMOT" type="checkbox" checked' in page
        assert '[[-20.0,10.0,-1.0],[-19.0,12.0,-0.9],[-16.0,15.0,-1.0]]' in page
        assert manifest['counts']['carla_reference_available'] is True
        assert manifest['counts']['carla_reference_points_sampled'] == 3
        assert manifest['counts']['carla_reference_type'] == 'independent_rendered_360_geometry'
        assert manifest['counts']['carla_reference_aligned_for_display'] is True
        assert manifest['files']['carla_reference_evaluation_only'].endswith('CARLA_independent_360_reference.ply')
    finally:
        mapper.close()
