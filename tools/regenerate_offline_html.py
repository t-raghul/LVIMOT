"""Regenerate an honest offline CARLA-vs-LVIMOT 3D comparison for one run.

Examples:
  python tools/regenerate_offline_html.py --output-dir outputs_carla/live
  python tools/regenerate_offline_html.py --output-dir outputs_carla/live --capture-if-missing --require-full-reference

A complete CARLA scene comparison requires a separately rendered, independent
360-degree depth reference of THIS recorded route. The OpenDRIVE road-only PLY
is deliberately not accepted as a full CARLA map. Ground truth is used here
only AFTER estimation; no estimator inputs or evaluation scores are changed.
"""
from __future__ import annotations

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[1]))


import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


from lvimot.output.offline_map import (
    _first_existing_path,
    _offline_html,
    _recorded_capture_path,
    _recorded_first_gt_pose,
    _reference_ply_candidates,
    _sample_carla_reference_local,
    _sample_points,
    load_estimated_ego_history,
)


_STATIC_RGB = np.array([154, 172, 187], dtype=np.int16)
_ROAD_RGB = np.array([101, 199, 111], dtype=np.int16)


def _read_binary_ply_xyzrgb(path: Path):
    with Path(path).open('rb') as stream:
        header = bytearray()
        while not header.endswith(b'end_header\n'):
            ch = stream.read(1)
            if not ch or len(header) > 16384:
                raise ValueError('Invalid PLY header: ' + str(path))
            header.extend(ch)
        lines = header.decode('ascii').splitlines()
        if 'format binary_little_endian 1.0' not in lines:
            raise ValueError('Only binary little-endian PLY is supported: ' + str(path))
        count = next(int(line.split()[2]) for line in lines if line.startswith('element vertex '))
        dtype = np.dtype([('xyz', '<f4', (3,)), ('rgb', 'u1', (3,))])
        vertices = np.frombuffer(stream.read(count * dtype.itemsize), dtype=dtype)
        if len(vertices) != count:
            raise ValueError('PLY vertex data is truncated: ' + str(path))
        return vertices['xyz'].astype(np.float32), vertices['rgb'].astype(np.uint8)


def _classify_lvimot_layers(xyz: np.ndarray, rgb: np.ndarray):
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    if not len(xyz):
        return [], []
    # The unified PLY also has colored vehicle boxes and path annotations.
    # These are not scene-map voxels and must NOT become fake road geometry.
    rgb16 = rgb.astype(np.int16)
    static_mask = np.all(rgb16 == _STATIC_RGB[None, :], axis=1)
    road_mask = np.all(rgb16 == _ROAD_RGB[None, :], axis=1)
    return _sample_points(xyz[static_mask], 23000), _sample_points(xyz[road_mask], 16000)


def _capture_full_reference_for_exact_run(output_dir: Path) -> Path:
    """Render reference from CARLA's original world for this recorded route only.

    CARLA must be running Town10HD_Opt and LVIMOT must already be stopped.
    This is evaluation-only. No live estimator or mapping code is invoked.
    """
    capture = _recorded_capture_path(output_dir)
    if capture is None:
        raise RuntimeError('Cannot find the exact recorded capture named by live_results.json. '
                           'Do not substitute a reference captured from another run.')
    if _recorded_first_gt_pose(output_dir) is None:
        raise RuntimeError('The exact recording lacks the first evaluation-only CARLA ego pose.')

    # Import only when reference capture is requested: no CARLA needed to
    # regenerate from an already captured 360 PLY.

    from tools.capture_independent_scene_reference import capture_reference

    destination = capture / 'original_carla_map'
    if destination.exists():
        number = 2
        while (capture / ('original_carla_map_attempt_%d' % number)).exists():
            number += 1
        destination = capture / ('original_carla_map_attempt_%d' % number)

    print('[CARLA 360] Rendering the independent 3D scene from this recorded route.', flush=True)
    print('[CARLA 360] CARLA 0.9.13 Town10HD_Opt must already be running.', flush=True)
    capture_reference(capture, destination)
    reference = _first_existing_path(_reference_ply_candidates(output_dir))
    if reference is None:
        raise RuntimeError('Independent CARLA 360 capture finished without a valid, matching reference.')
    return reference


def regenerate(output_dir: Path, capture_if_missing=False, require_full_reference=False, html_output=None) -> Path:
    output_dir = Path(output_dir)
    scene_json = output_dir / 'offline_4d_scene.json'
    scene_ply = output_dir / 'offline_4d_map.ply'
    html_path = Path(html_output) if html_output is not None else output_dir / 'offline_4d_map.html'
    if not scene_json.is_file():
        raise FileNotFoundError('Missing ' + str(scene_json))
    if not scene_ply.is_file():
        raise FileNotFoundError('Missing ' + str(scene_ply))

    first_gt_pose = _recorded_first_gt_pose(output_dir)
    reference = _first_existing_path(_reference_ply_candidates(output_dir))
    if reference is None and capture_if_missing:
        _capture_full_reference_for_exact_run(output_dir)
        reference = _first_existing_path(_reference_ply_candidates(output_dir))
    if require_full_reference and reference is None:
        raise RuntimeError('FULL CARLA 360 REFERENCE MISSING: Only a road reference is present. '
                           'A true CARLA-vs-LVIMOT scene comparison needs the matching '
                           'CARLA_independent_360_reference.ply. Start CARLA 0.9.13 Town10HD_Opt '
                           'and run regenerate_current_offline_map.bat to capture it.')
    if require_full_reference and first_gt_pose is None:
        raise RuntimeError('Cannot align maps: first CARLA evaluation pose is missing from the exact saved recording.')

    # Load the complete, colored sensor-only estimate from the saved final run.
    # No re-estimation or modification of .ply/.json/metrics is performed.
    scene = json.loads(scene_json.read_text(encoding='utf-8'))
    xyz, rgb = _read_binary_ply_xyzrgb(scene_ply)
    sample_static, sample_road = _classify_lvimot_layers(xyz, rgb)

    reference_path, carla_sample, carla_aligned = _sample_carla_reference_local(
        output_dir, first_gt_pose
    )
    if require_full_reference and (not carla_aligned or not carla_sample):
        raise RuntimeError('Matching independent 360 reference exists but cannot be aligned or contains no points.')

    stats = {
        'static_points_sampled': len(sample_static),
        'road_points_sampled': len(sample_road),
        'object_ids': len(scene.get('objects', {})),
        'carla_reference_available': reference_path is not None,
        'carla_reference_points_sampled': len(carla_sample),
        'carla_reference_aligned_for_display': bool(carla_aligned),
        'carla_reference_type': ('independent_rendered_360_geometry' if carla_aligned else 'unavailable'),
        'source': 'regenerated_from_existing_exports',
    }
    ego_history = load_estimated_ego_history(output_dir, scene)
    stats['ego_observations'] = len(ego_history)
    page = _offline_html({'static': sample_static, 'road': sample_road, 'carla': carla_sample},
                         scene.get('objects', {}), stats, ego_history)
    # Replace HTML only after all validation + rendering completed.
    html_path.parent.mkdir(parents=True, exist_ok=True)
    temp = html_path.with_name(html_path.name + '.tmp')
    try:
        temp.write_text(page, encoding='utf-8')
        os.replace(str(temp), str(html_path))
    finally:
        if temp.exists():
            temp.unlink()
    print('[LVIMOT] %s static samples and %s measured road samples' %
          (len(sample_static), len(sample_road)), flush=True)
    print('[CARLA 360] %s independent 3D samples; source: %s' %
          (len(carla_sample), reference_path if reference_path else 'MISSING'), flush=True)
    return html_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path,
                        help='Folder with the final run offline_4d_map.ply, offline_4d_scene.json and live_results.json')
    parser.add_argument('--capture-if-missing', action='store_true',
                        help='Render the exact recorded route using a running CARLA server if 360 reference is absent')
    parser.add_argument('--require-full-reference', action='store_true',
                        help='Fail rather than create a misleading road-only CARLA map')
    parser.add_argument('--html-output', type=Path,
                        help='Save a standalone HTML copy here while reading the original completed run')
    args = parser.parse_args()
    try:
        html = regenerate(args.output_dir, args.capture_if_missing, args.require_full_reference, args.html_output)
    except Exception as error:
        print('[CARLA/LVIMOT MAP COMPARISON ERROR] %s' % error, file=sys.stderr, flush=True)
        return 2
    print('[COMPLETE] %s' % html, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
