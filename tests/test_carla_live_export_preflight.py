"""Checks the exact live export path without running CARLA."""
from pathlib import Path

from lvimot.output.live_run import _verify_map_export_before_carla, run_live
from lvimot.output.live_dashboard import CarlaLiveVisualizer


def test_live_entrypoint_has_path_import_and_export_preflight(tmp_path):
    assert isinstance(run_live.__globals__["Path"], type(Path))
    _verify_map_export_before_carla(tmp_path, CarlaLiveVisualizer())
    assert not list(tmp_path.glob(".lvimot_map_preflight_*"))
    assert not list(tmp_path.glob("*.ply"))


def test_live_export_preflight_rejects_unwritable_target(tmp_path):
    target = tmp_path / 'not_a_directory'
    target.write_text('exists')
    try:
        _verify_map_export_before_carla(target, CarlaLiveVisualizer())
    except (OSError, RuntimeError):
        return
    raise AssertionError('Expected a failure before capture')
