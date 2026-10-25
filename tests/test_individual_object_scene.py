"""Test read-only individual-object 4D map history and visual geometry."""
import os
import sys
import numpy as np


from lvimot.mapping_4d.multiobject_trajectory_management import IndividualObjectScene, objects_to_local
from lvimot.mapping_4d.static_map.rolling_voxel_map import Carla4DEnvironmentMapper
from lvimot.temporal_fusion.pipeline import CarlaLVIMOTCore


def fgo_track(tid, x, speed, state='DYNAMIC'):
    return {'track_id': tid, 'position': [x, 4., 0.], 'velocity': [speed, 0., 0.],
            'dimensions': {'length': 4., 'width': 2., 'height': 1.6},
            'rotation': {'yaw': 0.}, 'class': 'car',
            'motion_state': state, 'source': 'temporal_fusion_fgo'}


def test_moving_path_and_unique_id():
    scene = IndividualObjectScene()
    for i in range(8):
        scene.update(i, i * .2, [fgo_track(7, i*.6, 3.)])
    data = scene.current(1.4)[0]
    assert data['track_id'] == 7
    assert data['motion_status'] == 'MOVING'
    assert data['history_length'] == 8
    assert len(data['trajectory']) == 8
    assert np.allclose(data['position'], [4.2, 4., 0.])


def test_stopped_then_parked_candidate_then_moving_preserves_id():
    scene = IndividualObjectScene()
    scene.update(0, 0., [fgo_track(9, 0., 2.)])
    scene.update(1, .2, [fgo_track(9, .4, .1, 'STATIC')])
    assert scene.current(.2)[0]['motion_status'] == 'STOPPED'
    for i in range(2, 16):
        scene.update(i, i*.2, [fgo_track(9, .4, 0., 'STATIC')])
    parked = scene.current(3.)[0]
    assert parked['motion_status'] == 'PARKED_CANDIDATE'
    before = len(parked['trajectory'])
    scene.update(16, 3.2, [fgo_track(9, 1.0, 3., 'DYNAMIC')])
    after = scene.current(3.2)[0]
    assert after['track_id'] == 9
    assert after['motion_status'] == 'MOVING'
    assert len(after['trajectory']) == before+1


def test_stationary_does_not_make_fake_path():
    scene = IndividualObjectScene()
    for i in range(25):
        scene.update(i, i*.2, [fgo_track(10, 3., 0., 'STATIC')])
    obj = scene.current(4.8)[0]
    assert obj['motion_status'] == 'PARKED_CANDIDATE'
    assert len(obj['trajectory']) == 1
    assert obj['history_length'] == 25


def test_missing_fgo_state_never_replaced_by_mot_and_stale_hides():
    estimated = {'objects_current': {1: {'position': [1., 2., 3.], 'velocity': [0., 0., 0.]}}}
    mot = [{'track_id': 1, 'position': [999., 999., 999.], 'velocity': [99., 0., 0.],
            'motion_state': 'STATIC', 'class': 'car'},
           {'track_id': 2, 'position': [50., 0., 0.], 'velocity': [10., 0., 0.]}]
    fused = CarlaLVIMOTCore._fgo_trajectory_tracks(estimated, mot)
    assert len(fused) == 1
    assert fused[0]['position'] == [1., 2., 3.]
    assert fused[0]['velocity'] == [0., 0., 0.]
    scene = IndividualObjectScene()
    scene.update(0, 0., fused)
    scene.update(1, .3, [])
    assert scene.current(.3)[0]['motion_status'] == 'LAST_SEEN'
    assert len(scene.current(.3)[0]['trajectory']) == 1
    assert scene.current(2.) == []


def test_3d_box_corners_and_local_rotation():
    scene = IndividualObjectScene()
    scene.update(0, 0., [fgo_track(5, 11., 0., 'STATIC')])
    pose = [0., 0., 0., 10., 4., 0.]
    boxes = objects_to_local(scene.current(0.), pose)
    assert len(boxes) == 1
    assert len(boxes[0]['box_corners_local']) == 8
    assert np.allclose(boxes[0]['position_local'], [1., 0., 0.])
    assert len(boxes[0]['trajectory_local']) == 1
    assert np.isclose(boxes[0]['box_corners_local'][0][2], -.8)


def test_mapper_uses_stationary_fgo_state_for_scene_without_changing_static_mask():
    mapper = Carla4DEnvironmentMapper()
    pose = {'location': {'x': 0, 'y': 0, 'z': 0}, 'rotation': {}}
    mapper.add_frame_data(0, 0., np.array([[6., 0., 0.], [8., 0., 0.]]), [],
                          [fgo_track(3, 6., 0., 'STATIC')], pose,
                          update_static=False, update_planar=False)
    assert mapper.individual_object_tracks(0.)[0]['track_id'] == 3
    assert mapper.export_summary()['total_individual_object_ids'] == 1
    assert len(mapper.static_voxel_map) == 0
    assert 3 not in mapper.dynamic_object_tubes  # Legacy dynamic tubes unaffected.


def test_history_bounded_and_two_ids():
    scene = IndividualObjectScene(max_history=8)
    for i in range(20):
        scene.update(i, i * .1, [fgo_track(1, i*.2, 3.), fgo_track(2, 4., 0.)])
    assert len(scene.objects[1]['history']) == 8
    assert len(scene.objects[1]['path']) == 8
    assert {i['track_id'] for i in scene.current(1.9)} == {1, 2}


def test_nonfinite_positions_ignored():
    scene = IndividualObjectScene()
    scene.update(0, 0., [fgo_track(1, float('nan'), 1.), fgo_track(2, 5., 2.)])
    assert [obj['track_id'] for obj in scene.current(0.)] == [2]


def test_legacy_mot_only_is_never_labeled_as_fgo_scene():
    mapper = Carla4DEnvironmentMapper()
    pose = {'location': [0., 0., 0.], 'rotation': {}}
    mot_only = fgo_track(12, 5., 1.)
    mot_only.pop('source')
    mapper.add_frame_data(0, 0., np.empty((0, 3)), [], [mot_only], pose,
                          update_static=False, update_planar=False)
    assert mapper.individual_object_tracks(0.) == []


def test_dashboard_preview_contains_individual_object_boxes_and_history(tmp_path, monkeypatch):
    import json
    from lvimot.output.live_dashboard import CarlaLiveVisualizer
    monkeypatch.setenv('LVIMOT_LIVE_MAP_SNAPSHOT', str(tmp_path / 'map.json'))
    monkeypatch.setenv('LVIMOT_LIVE_CAMERA_SNAPSHOT', str(tmp_path / 'camera.png'))
    vis = CarlaLiveVisualizer(bev_size=180, meters=40)
    scene = IndividualObjectScene()
    scene.update(0, 0.0, [fgo_track(21, 5., 1.1)])
    scene.update(1, 0.2, [fgo_track(21, 6., 1.1)])
    result = {'frame': 1, 'timestamp': 0.2,
              'state_estimation': {'pose': [0., 0., 0., 0., 0., 0.],
                                   'velocity': [0., 0., 0.]},
              'tracking': {'active_tracks': []},
              'individual_object_tracks': scene.current(.2),
              'mapping_4d': {'total_static_voxels': 0}}
    view = vis.render(np.zeros((60, 120, 3), dtype=np.uint8),
                      result, np.empty((0, 3)))
    assert view.shape[0] >= 60
    stored = json.loads((tmp_path / 'map.json').read_text())
    assert len(stored['individual_objects_local']) == 1
    obj = stored['individual_objects_local'][0]
    assert obj['track_id'] == 21
    assert len(obj['box_corners_local']) == 8
    assert len(obj['trajectory_local']) == 2
    assert obj['motion_status'] == 'MOVING'


def test_individual_tracks_written_to_map_export_json(tmp_path):
    import json
    from lvimot.output.map_export import export_live_map
    from lvimot.output.live_dashboard import CarlaLiveVisualizer
    mapper = Carla4DEnvironmentMapper()
    pose = {'location': {'x': 0., 'y': 0., 'z': 0.}, 'rotation': {}}
    mapper.add_frame_data(0, 0.0, np.empty((0, 3)), [],
                          [fgo_track(91, 4., 0., 'STATIC')], pose,
                          update_static=False, update_planar=False)
    # Visualization/export maintains the stationary box even without
    # dynamic tubes for the historical MOVING-only output.
    summary = export_live_map(mapper, np.zeros(6), CarlaLiveVisualizer(),
                              tmp_path, include_accumulated=False)
    value = json.loads((tmp_path / 'live_dynamic_tracks_latest.json').read_text())
    assert summary['dynamic_tracks_json']
    assert '91' in value['individual_object_tracks']
    assert value['individual_object_tracks']['91']['last_status'] == 'STOPPED'
    assert value['individual_object_tracks']['91']['source'] == 'temporal_fusion_fgo'
