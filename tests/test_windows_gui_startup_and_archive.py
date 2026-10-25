"""Checks for Windows GUI startup, robust output decoding, and 4D export wiring.

These do not need a display, Windows, CARLA, CUDA, or model downloads.
"""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUI = ROOT / 'lvimot/output/live_gui.py'
LIVE = ROOT / 'lvimot/output/live_run.py'
LAUNCHER = ROOT / 'run_lvimot_gui.bat'


def test_console_output_survives_windows_undefined_charmap_byte():
    child = subprocess.Popen(
        [sys.executable, '-c',
         "import sys;sys.stdout.buffer.write(bytes([102,114,97,109,101,32,49,10,143,10,102,114,97,109,101,32,50,10]))"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace', bufsize=1,
    )
    try:
        assert child.stdout is not None
        output = child.stdout.read()
        assert child.wait(timeout=10) == 0
        assert 'frame 1' in output and 'frame 2' in output
        assert '\ufffd' in output
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_gui_backend_reader_uses_utf8_with_replacement():
    tree = ast.parse(GUI.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == 'LVIMOTLiveGUI')
    start = next(node for node in cls.body
                 if isinstance(node, ast.FunctionDef) and node.name == '_start_backend')
    popens = [n for n in ast.walk(start)
              if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute)
              and n.func.attr == 'Popen']
    assert len(popens) == 1
    keywords = {item.arg: item.value for item in popens[0].keywords}
    assert isinstance(keywords['encoding'], ast.Constant)
    assert keywords['encoding'].value == 'utf-8'
    assert isinstance(keywords['errors'], ast.Constant)
    assert keywords['errors'].value == 'replace'
    source = GUI.read_text(encoding='utf-8')
    assert 'env["PYTHONIOENCODING"] = "utf-8"' in source
    assert 'env["PYTHONUTF8"] = "1"' in source


def test_launcher_does_not_install_packages_or_import_torch_before_gui():
    source = LAUNCHER.read_text(encoding='ascii').lower()
    lines = [s.strip() for s in source.splitlines()
             if s.strip() and not s.lstrip().startswith('rem')]
    assert not any('call setup_cuda_runtime.bat' in line for line in lines)
    assert not any('import torch' in line for line in lines)
    assert 'set "pythonioencoding=utf-8"' in source
    assert any('"%vpy%" -u lvimot\\output\\live_gui.py' in line for line in lines)


def test_live_runner_writes_read_only_offline_scene_after_gt_validation():
    source = LIVE.read_text(encoding='utf-8')
    start = source.index('def run_live(')
    eval_boundary = source.index('last_validation = validate_against_carla(', start)
    export_boundary = source.index('offline_scene_latest = export_offline_4d_scene(', eval_boundary)
    result_boundary = source.index('summary = {', export_boundary)
    assert eval_boundary < export_boundary < result_boundary
    assert "'offline_scene_html'" in source[export_boundary:result_boundary]
    assert '"offline_4d_scene_latest": offline_scene_latest' in source


def test_object_archive_only_receives_fgo_measurements():
    mapper_source = (ROOT / 'lvimot/mapping_4d/static_map/rolling_voxel_map.py').read_text(encoding='utf-8')
    assert 'if tr.get("source") == "temporal_fusion_fgo"' in mapper_source
    assert "self.object_archive.append(tid, state['history'][-1])" in mapper_source
    assert 'road_mapper.enable_accumulated_map(' in mapper_source
