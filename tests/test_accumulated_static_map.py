"""A bounded live map must not discard the disk record of earlier geometry."""
import sys
from pathlib import Path

import numpy as np
import pytest

from lvimot.mapping_4d.static_map.rolling_voxel_map import Carla4DEnvironmentMapper
from lvimot.mapping_4d.static_map.accumulated_map import AccumulatedStaticMap


def _read(path):
    with path.open('rb') as stream:
        while stream.readline() != b'end_header\n':
            pass
        return np.frombuffer(stream.read(), dtype='<f4').reshape(-1, 3)


def test_accumulated_map_retains_visited_geometry_after_live_eviction(tmp_path):
    mapper = Carla4DEnvironmentMapper(static_voxel_size=.25, max_map_distance=10, max_static_voxels=20)
    mapper.enable_accumulated_map(tmp_path / 'maps')
    local = np.array([[2 + i * .5, 0, 0] for i in range(8)])
    for frame, x in enumerate((0., 100.)):
        pose = {'rotation': {}, 'location': {'x': x, 'y': 0, 'z': 0}}
        mapper.add_frame_data(frame, frame * .1, local, [], [], pose, evict_static=True)
    assert len(mapper.get_static_map_points()) == 8
    assert mapper.get_static_map_points()[:, 0].min() > 100
    try:
        result = mapper.accumulated_map.export(tmp_path / 'accumulated.ply')
        all_points = _read(tmp_path / 'accumulated.ply')
        assert len(all_points) == 16
        assert all_points[:, 0].min() < 10
        assert all_points[:, 0].max() > 100
        assert result['ground_truth_used'] is False
        assert result['loop_closure_applied'] is False
    finally:
        mapper.close()


def test_full_live_dictionary_accepts_new_nearby_surfaces(tmp_path):
    mapper = Carla4DEnvironmentMapper(static_voxel_size=.25, max_map_distance=30, max_static_voxels=5)
    first = np.array([[10 + i, 0, 0] for i in range(5)])
    mapper._update_static_voxels(first, np.zeros(3))
    second = np.array([[1 + i * .5, 0, 0] for i in range(4)])
    mapper._update_static_voxels(second, np.zeros(3))
    points = mapper.get_static_map_points()
    assert len(points) <= 5
    assert sum(points[:, 0] < 4) == 4


def test_archive_averages_per_frame_support_and_refuses_run_mixing(tmp_path):
    path = tmp_path / 'map.sqlite3'
    archive = AccumulatedStaticMap(path, voxel_size=1.0)
    try:
        archive.add([[.1, .1, .1], [.3, .1, .1]], 0)
        archive.add([[.6, .1, .1]], 1)
        archive.export(tmp_path / 'map.ply')
        np.testing.assert_allclose(_read(tmp_path / 'map.ply'), [[.4, .1, .1]], atol=1e-6)
        with pytest.raises(FileExistsError):
            AccumulatedStaticMap(path, voxel_size=1.0)
    finally:
        archive.close()
        archive.close()
