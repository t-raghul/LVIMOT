"""Independent, matched, FULL 360 CARLA scene versus LVIMOT's offline 4D cloud."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]



from lvimot.output.evaluation.map_geometry import align_local_map_to_carla
from lvimot.output.offline_map import (
    _offline_html,
    _read_binary_ply_xyz,
    _recorded_capture_path,
    _recorded_first_gt_pose,
    _reference_ply_candidates,
    _reference_world_to_local,
    _sample_carla_reference_local,
)
from tools.regenerate_offline_html import regenerate, _classify_lvimot_layers


def pose(x=35.0, y=-14.0, z=2.5, yaw=73.0, pitch=-1.0, roll=2.0):
    return {'location': {'x': x, 'y': y, 'z': z},
            'rotation': {'yaw': yaw, 'pitch': pitch, 'roll': roll}}


def write_xyz_ply(path, xyz):
    pts = np.asarray(xyz, dtype='<f4').reshape(-1, 3)
    header = ('ply\nformat binary_little_endian 1.0\n'
              'element vertex {}\n'
              'property float x\nproperty float y\nproperty float z\nend_header\n'.format(len(pts)))
    Path(path).write_bytes(header.encode('ascii') + pts.tobytes())


def write_xyzrgb_ply(path, xyz, rgb):
    pts = np.asarray(xyz, dtype='<f4').reshape(-1, 3)
    colors = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    dtype = np.dtype([('xyz', '<f4', (3,)), ('rgb', 'u1', (3,))])
    data = np.empty(len(pts), dtype=dtype)
    data['xyz'] = pts
    data['rgb'] = colors
    header = ('ply\nformat binary_little_endian 1.0\n'
              'element vertex {}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'.format(len(pts)))
    Path(path).write_bytes(header.encode('ascii') + data.tobytes())


def create_run(live, first=None, capture_name='capture_20261022_161253_989623'):
    first = first if first is not None else pose()
    live.mkdir(parents=True, exist_ok=True)
    capture = live / 'sensor_recordings' / capture_name
    capture.mkdir(parents=True, exist_ok=True)
    (capture / 'manifest.json').write_text(json.dumps({
        'format': 'LVIMOT_RAW_SENSOR_ARCHIVE_V1', 'frames': 'frames.jsonl'
    }), encoding='utf-8')
    (capture / 'frames.jsonl').write_text(json.dumps({
        'frame': 0, 'ground_truth_ego_pose_evaluation_only': first
    }) + '\n', encoding='utf-8')
    (live / 'live_results.json').write_text(json.dumps({
        'ground_truth_evaluation_enabled': True,
        'raw_sensor_recording': 'outputs_carla\\live\\sensor_recordings\\' + capture_name,
    }), encoding='utf-8')
    return capture


def write_matching_360_reference(capture, world_points, suffix='original_carla_map', recorded_capture=None):
    folder = capture / suffix
    folder.mkdir(parents=True, exist_ok=True)
    ply = folder / 'CARLA_independent_360_reference.ply'
    write_xyz_ply(ply, world_points)
    meta = {
        'format': 'CARLA_INDEPENDENT_RENDERED_360_REFERENCE_V1',
        'reference_only': True,
        'gt_used_by_estimator': False,
        'recorded_capture': str((recorded_capture or capture).resolve()),
        'map': 'Carla/Maps/TM/Town10HD_Opt',
    }
    (folder / 'reference_manifest.json').write_text(json.dumps(meta), encoding='utf-8')
    return ply


def test_inverse_matches_existing_evaluation_transform():
    local = np.asarray([[0.0, 0.0, 0.0], [10.0, -2.0, 1.0], [-14.0, 9.5, 0.2]], dtype=float)
    first = pose()
    world = align_local_map_to_carla(local, first)
    recovered = _reference_world_to_local(world, first)
    np.testing.assert_allclose(recovered, local, atol=1e-5)


def test_only_opendrive_road_is_not_misrepresented_as_full_carla_map(tmp_path):
    live = tmp_path / 'live'
    first = pose()
    create_run(live, first)
    write_xyz_ply(live / 'CARLA_reference_road_surface.ply', np.array([[35.0, -14.0, 2.5]]))
    assert _reference_ply_candidates(live) == []
    ref, points, aligned = _sample_carla_reference_local(live, first)
    assert ref is None
    assert points == [] and aligned is False
    page = _offline_html({'static': [[0, 0, 0]], 'road': [], 'carla': []}, {}, {})
    assert 'id="showCarla" type="checkbox" disabled' in page
    assert 'Independent CARLA 360° reference not found' in page
    assert 'id="showLVIMOT" type="checkbox" checked' in page


def test_mismatched_360_reference_is_rejected(tmp_path):
    live = tmp_path / 'live'
    correct = create_run(live)
    wrong = live / 'sensor_recordings' / 'capture_wrong'
    wrong.mkdir(parents=True)
    write_matching_360_reference(correct, [[35, -14, 2.5], [36, -14, 2.5]], recorded_capture=wrong)
    assert _reference_ply_candidates(live) == []
    write_matching_360_reference(correct, [[35, -14, 2.5], [36, -14, 2.5]], suffix='original_carla_map_attempt_2')
    assert len(_reference_ply_candidates(live)) == 1
    assert _reference_ply_candidates(live)[0].parent.name == 'original_carla_map_attempt_2'


def test_recorded_first_gt_pose_comes_from_named_run_only(tmp_path):
    live = tmp_path / 'live'
    correct = create_run(live, pose(x=35.0))
    other = live / 'sensor_recordings' / 'capture_20261022_170000_000000'
    other.mkdir(parents=True)
    (other / 'manifest.json').write_text(json.dumps({
        'format': 'LVIMOT_RAW_SENSOR_ARCHIVE_V1', 'frames': 'frames.jsonl'
    }), encoding='utf-8')
    (other / 'frames.jsonl').write_text(json.dumps({
        'frame': 0, 'ground_truth_ego_pose_evaluation_only': pose(x=555.0)
    }) + '\n', encoding='utf-8')
    assert _recorded_capture_path(live) == correct
    assert _recorded_first_gt_pose(live)['location']['x'] == 35.0
    (live / 'live_results.json').write_text(json.dumps({
        'ground_truth_evaluation_enabled': True,
        'raw_sensor_recording': r'outputs_carla\live\sensor_recordings\capture_nonexistent',
    }), encoding='utf-8')
    assert _recorded_first_gt_pose(live) is None
    assert _reference_ply_candidates(live) == []


def test_regeneration_embeds_full_3d_reference_and_preserves_input_cloud(tmp_path):
    live = tmp_path / 'live'
    first = pose(x=35, y=-14, z=2.5, yaw=90, pitch=0, roll=0)
    capture = create_run(live, first)
    local = np.asarray([[0, 0, 0], [1, 2, 0], [1.5, -3, .1], [1.2, 2.2, 3.7]], dtype=float)
    world = align_local_map_to_carla(local, first)
    write_matching_360_reference(capture, world)
    # Deliberately put an incompatible road-only file in the live folder.
    write_xyz_ply(live / 'CARLA_reference_road_surface.ply', [[500, 500, 0]])
    write_xyzrgb_ply(live / 'offline_4d_map.ply', local,
                     [[154, 172, 187], [101, 199, 111], [154, 172, 187], [154, 172, 187]])
    (live / 'offline_4d_scene.json').write_text(json.dumps({
        'ground_truth_used': False,
        'objects': {'7': {'history': [], 'trajectory': []}}
    }), encoding='utf-8')
    before_ply = (live / 'offline_4d_map.ply').read_bytes()
    before_scene = (live / 'offline_4d_scene.json').read_bytes()
    html_path = regenerate(live, require_full_reference=True)
    page = html_path.read_text(encoding='utf-8')
    payload = json.loads(page.split('<script id="payload" type="application/json">')[1].split('</script>')[0])
    np.testing.assert_allclose(payload['layers']['carla'], local, atol=0.001)
    assert len(payload['layers']['static']) == 3
    assert len(payload['layers']['road']) == 1
    assert payload['stats']['carla_reference_type'] == 'independent_rendered_360_geometry'
    assert 'id="showCarla" type="checkbox" checked' in page
    assert 'id="showLVIMOT" type="checkbox" checked' in page
    assert "if (!el('showVehicles').checked) return;" in page
    assert 'id="showEgo" type="checkbox" checked' in page
    assert (live / 'offline_4d_map.ply').read_bytes() == before_ply
    assert (live / 'offline_4d_scene.json').read_bytes() == before_scene


def test_strict_regeneration_rejects_road_only_and_keeps_previous_html(tmp_path):
    live = tmp_path / 'live'
    create_run(live)
    write_xyz_ply(live / 'CARLA_reference_road_surface.ply', [[40, 22, 0]])
    write_xyzrgb_ply(live / 'offline_4d_map.ply', [[0, 0, 0]], [[154, 172, 187]])
    (live / 'offline_4d_scene.json').write_text(json.dumps({'objects': {}}), encoding='utf-8')
    (live / 'offline_4d_map.html').write_text('ORIGINAL_HTML', encoding='utf-8')
    with pytest.raises(RuntimeError, match='FULL CARLA 360 REFERENCE MISSING'):
        regenerate(live, require_full_reference=True)
    assert (live / 'offline_4d_map.html').read_text(encoding='utf-8') == 'ORIGINAL_HTML'


def test_lvimot_layer_classification_preserves_distinct_road_layer():
    xyz = np.array([[0, 0, 0], [0, 0, 1], [2, 0, 0]], dtype=np.float32)
    rgb = np.array([[154, 172, 187], [245, 181, 68], [101, 199, 111]], dtype=np.uint8)
    static, road = _classify_lvimot_layers(xyz, rgb)
    assert len(static) == 1
    assert len(road) == 1


def test_full_reference_is_rejected_without_valid_provenance_manifest(tmp_path):
    live = tmp_path / 'live'
    capture = create_run(live)
    folder = capture / 'original_carla_map'
    folder.mkdir()
    write_xyz_ply(folder / 'CARLA_independent_360_reference.ply', [[0, 0, 0], [0, 1, 0]])
    assert _reference_ply_candidates(live) == []
    page = _offline_html({'static': [], 'road': [], 'carla': []}, {}, {})
    assert 'id="showCarla" type="checkbox" disabled' in page
