"""Offline synthetic checks only: NEVER starts CARLA or any simulator."""
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from lvimot.output.evaluation.map_geometry import (
    align_local_map_to_carla, road_reference_from_carla_map,
    compare_road_geometry, validate_against_carla,
)
from lvimot.output.map_export import export_live_map
from lvimot.mapping_4d.static_map.rolling_voxel_map import Carla4DEnvironmentMapper
from lvimot.output.live_dashboard import CarlaLiveVisualizer


def _gt_pose(x=100., y=200., z=0., yaw=0.):
    return {'location': {'x': x, 'y': y, 'z': z},
            'rotation': {'roll': 0., 'pitch': 0., 'yaw': yaw}}


class FakeRoad:
    def generate_waypoints(self, spacing):
        return [SimpleNamespace(
            lane_type='Driving', lane_width=3.6,
            transform=SimpleNamespace(location=SimpleNamespace(x=float(x),y=200.,z=0.),
                                      rotation=SimpleNamespace(yaw=0.)))
                for x in range(60, 141, 2)]


class FakeWorld:
    def get_map(self):
        return FakeRoad()


def test_gt_alignment_is_evaluation_only():
    raw = np.array([[0., 0., 0.], [3., 4., 1.]])
    source = raw.copy()
    aligned = align_local_map_to_carla(raw, _gt_pose(yaw=90.))
    np.testing.assert_allclose(aligned[0], [100,200,0], atol=1e-4)
    np.testing.assert_allclose(aligned[1], [96,203,1], atol=1e-4)
    np.testing.assert_array_equal(raw, source)


def test_reference_road_agreement_pass_and_large_offset_fail():
    ref = road_reference_from_carla_map(FakeRoad(), [100,200,0], radius_m=50)
    assert len(ref) >= 150
    good = ref.copy()
    ok = compare_road_geometry(good, ref, [100,200,0])
    assert ok['status'] == 'MEASURED_NOT_CERTIFIED'
    assert ok['road_agreement_pct'] == 100.0
    assert ok['median_road_xy_error_m'] == 0
    # A known 9-m map displacement must not be accepted as accurate.
    bad = good.copy()
    bad[:,1] += 9.
    failed = compare_road_geometry(bad, ref, [100,200,0])
    assert failed['road_agreement_pct'] is not None
    assert failed['road_agreement_pct'] < 10.


def test_export_and_reference_do_not_modify_mapper(tmp_path):
    mapper = Carla4DEnvironmentMapper()
    ref = road_reference_from_carla_map(FakeRoad(), [100,200,0], radius_m=60)
    # Include also some non-road structures, all in LOCAL estimated frame.
    raw = (ref - np.array([100., 200., 0.])).astype(np.float32)
    mapper.static_voxel_map = {tuple(np.floor(p * 4).astype(int)): p for p in raw}
    original = mapper.get_static_map_points().copy()
    vis = CarlaLiveVisualizer(meters=30)
    exported = export_live_map(mapper, np.zeros(6), vis, tmp_path)
    assert Path(exported['raw_ply']).exists()
    assert exported['raw_voxels'] == len(original)
    result = validate_against_carla(FakeWorld(), mapper,
                                    _gt_pose(), _gt_pose(), tmp_path)
    assert result['status'] == 'MEASURED_NOT_CERTIFIED'
    assert result['road_agreement_pct'] > 75.
    assert (tmp_path/'CARLA_reference_road_surface.ply').exists()
    assert (tmp_path/'LVIMOT_aligned_CARLA_frame_EVALUATION_ONLY.ply').exists()
    assert (tmp_path/'CARLA_vs_LVIMOT_road_overlay.svg').exists()
    assert json.loads((tmp_path/'CARLA_MAP_VALIDATION.json').read_text())['reference_is_independent']
    np.testing.assert_array_equal(mapper.get_static_map_points(), original)
