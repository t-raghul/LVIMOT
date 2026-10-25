"""Estimated ego export, legacy recovery, bounded RAM and independent layers."""
import json
from pathlib import Path

import numpy as np

from lvimot.mapping_4d.static_map.rolling_voxel_map import Carla4DEnvironmentMapper
from lvimot.output.offline_map import export_offline_4d_scene, load_estimated_ego_history


def payload(path):
    page = Path(path).read_text(encoding='utf-8')
    return json.loads(page.split('<script id="payload" type="application/json">')[1].split('</script>')[0])


def test_all_estimated_frames_persist_beyond_ram_limit_without_object_tracks(tmp_path):
    mapper = Carla4DEnvironmentMapper(max_dynamic_history=2)
    mapper.enable_accumulated_map(tmp_path / 'archive')
    try:
        for frame in range(7):
            pose = {'location': {'x': frame * 1.25, 'y': -frame, 'z': .5},
                    'rotation': {'yaw': frame * 12., 'roll': 0., 'pitch': 0.}}
            mapper.add_frame_data(frame, 100. + frame * .2, np.empty((0, 3)), [], [], pose,
                                  update_static=False, update_planar=False,
                                  estimated_ego_velocity=[6.25, -5., 0.])
        assert len(mapper.estimated_ego_history) == 2
        result = export_offline_4d_scene(mapper, tmp_path / 'export')
        scene = json.loads(Path(result['offline_scene_json']).read_text())
        embedded = payload(result['offline_scene_html'])
        assert scene['objects'] == {}
        assert scene['ground_truth_used'] is False
        assert len(scene['ego_history']) == 7
        assert embedded['ego_history'] == scene['ego_history']
        assert embedded['ego_history'][-1]['position'] == [7.5, -6., .5]
        assert embedded['ego_history'][-1]['yaw'] == 72.
        assert embedded['ego_history'][-1]['velocity'] == [6.25, -5., 0.]
        assert all(row['ground_truth_used'] is False for row in embedded['ego_history'])
        assert result['offline_scene_counts']['ego_observations'] == 7
    finally:
        mapper.close()


def test_legacy_recovery_reads_only_estimated_state_and_uses_radians_to_degrees(tmp_path):
    row = {'frame': 9, 'timestamp': 17.2,
           'state_estimation': {'pose': [0., 0., np.pi / 2, 2., 3., 4.], 'velocity': [1., 0., 0.]},
           'ground_truth_ego_pose_evaluation_only': {'location': {'x': 999., 'y': 999., 'z': 999.}}}
    gt_only = {'frame': 10, 'timestamp': 17.4, 'ground_truth_ego_pose_evaluation_only': row['ground_truth_ego_pose_evaluation_only']}
    (tmp_path / 'live_frames.jsonl').write_text(json.dumps(row) + '\n' + json.dumps(gt_only) + '\n')
    history = load_estimated_ego_history(tmp_path, {})
    assert len(history) == 1
    assert history[0]['position'] == [2., 3., 4.]
    assert history[0]['yaw'] == 90.
    assert history[0]['frame_id'] == 9
    assert history[0]['ground_truth_used'] is False
    # An explicitly present empty scene history never gets filled from other runs.
    assert load_estimated_ego_history(tmp_path, {'ego_history': []}) == []


def test_missing_estimated_history_does_not_fabricate_ego(tmp_path):
    assert load_estimated_ego_history(tmp_path, {}) == []
