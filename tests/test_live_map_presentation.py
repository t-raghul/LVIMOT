"""Map-view cleanup must not mutate the SLAM mapper or invent distant walls."""
from pathlib import Path

import numpy as np

from lvimot.output.map_export import (
    cleaned_layers_and_outlines,
    export_live_map,
    points_local_to_world,
    write_binary_ply,
)
from lvimot.output.live_dashboard import CarlaLiveVisualizer
from lvimot.mapping_4d.static_map.rolling_voxel_map import Carla4DEnvironmentMapper


def test_cleanup_removes_islands_and_keeps_supported_wall():
    wall = np.asarray([[float(x) * 0.35, 2.0, 1.0] for x in range(-20, 21)])
    speckle = np.asarray([[17.0, 17.0, 0.5]])
    original = np.vstack([wall, speckle])
    pristine = original.copy()
    g, s, gl, sl, stats = cleaned_layers_and_outlines(
        np.empty((0, 3)), original, meters=30, cell_size_m=0.35
    )
    assert len(s) >= 35
    assert len(s) < len(original)
    assert sl
    assert stats["removed_display_islands"] >= 1
    np.testing.assert_array_equal(original, pristine)


def test_local_to_world_is_inverse_heading_rotation():
    yaw = np.pi / 2.0
    pose = [0.0, 0.0, yaw, 15.0, 18.0, 3.0]
    pt = points_local_to_world([[4.0, 2.0, -1.0]], pose)[0]
    np.testing.assert_allclose(pt, [13.0, 22.0, 2.0], atol=1e-5)


def test_ply_export_raw_points_not_filtered(tmp_path):
    mapper = Carla4DEnvironmentMapper()
    raw = np.asarray([[0.0, 0.0, -1.0], [1.0, 2.0, 2.5], [25.0, 23.0, -1.0]])
    mapper.static_voxel_map = {tuple(map(int, p * 4)): p.astype(np.float32) for p in raw}
    vis = CarlaLiveVisualizer(meters=30)
    out = export_live_map(mapper, [0, 0, 0, 0, 0, 0], vis, tmp_path)
    assert out["raw_voxels"] == len(raw)
    raw_ply = Path(out["raw_ply"])
    assert raw_ply.is_file() and raw_ply.stat().st_size > 100
    with raw_ply.open("rb") as f:
        header = f.read(200)
    assert b"element vertex 3\n" in header
    assert Path(out["dynamic_tracks_json"]).is_file()


def test_snapshot_has_separate_lines_and_true_raw_preview(tmp_path, monkeypatch):
    monkeypatch.setenv("LVIMOT_LIVE_MAP_SNAPSHOT", str(tmp_path / "snapshot.json"))
    monkeypatch.setenv("LVIMOT_LIVE_CAMERA_SNAPSHOT", str(tmp_path / "cam.png"))
    vis = CarlaLiveVisualizer(meters=25)
    import json
    local_raw = np.asarray([[x * 0.35, 3.0, 1.0] for x in range(-15, 16)])
    vis._write_snapshot({"frame": 7}, np.zeros(6), np.empty((0, 3)), local_raw,
                        [], {}, structure_lines=[[[0.0, 3.0], [1.0, 3.0]]], raw_local=local_raw)
    doc = json.loads((tmp_path / "snapshot.json").read_text())
    assert doc["structure_outlines_local"]
    assert len(doc["raw_map_points_local"]) == 31
    assert doc["frame"] == 7
