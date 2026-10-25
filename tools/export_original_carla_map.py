"""Export the exact completed run's CARLA reference and refresh its HTML map.

An existing matching 360 degree reference is reused without connecting to CARLA.
If it is absent, CARLA 0.9.13 Town10HD_Opt must be running and LVIMOT stopped.
Only post-run reference capture and HTML presentation are performed here.
"""
from __future__ import annotations

if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[1]))

import argparse
import os
from pathlib import Path
import sys
import webbrowser

from lvimot.output.offline_map import _recorded_capture_path
from tools.regenerate_offline_html import regenerate


ROOT = Path(__file__).resolve().parents[1]


def refresh_reference_viewer(output_dir, capture_if_missing=True, html_output=None):
    """Bind the reference to live_results.json and update the saved viewer.

    The regenerator verifies the reference manifest and first evaluation pose,
    aligns the reference for display, and preserves the estimated scene data.
    It replaces HTML only after the complete comparison has been generated.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    capture = _recorded_capture_path(output_dir)
    if capture is None:
        raise RuntimeError(
            'Cannot resolve the exact recording named by live_results.json. '
            'Keep that run\'s sensor_recordings folder beside its saved outputs '
            'and enable GT recording for the independent reference comparison. '
            'A different or newer route cannot be substituted.'
        )
    print('[EXACT RECORDED ROUTE] %s' % capture, flush=True)
    print('[COMPARISON] Refreshing the HTML with the matching independent CARLA 360 reference.', flush=True)
    return regenerate(
        output_dir,
        capture_if_missing=bool(capture_if_missing),
        require_full_reference=True,
        html_output=html_output,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs_carla' / 'live',
                        help='Completed run containing live_results.json and offline 4D exports')
    parser.add_argument('--no-capture', action='store_true',
                        help='Use an existing matching reference only; never connect to CARLA')
    parser.add_argument('--no-open', action='store_true',
                        help='Save the corrected HTML without opening a browser')
    args = parser.parse_args(argv)
    try:
        html = refresh_reference_viewer(args.output_dir, capture_if_missing=not args.no_capture)
    except Exception as error:
        print('[CARLA/LVIMOT COMPARISON ERROR] %s' % error, file=sys.stderr, flush=True)
        print('If the matching reference is missing, keep CARLA 0.9.13 Town10HD_Opt '
              'running and stop LVIMOT before retrying.', file=sys.stderr, flush=True)
        return 2
    print('[COMPLETE] %s' % html, flush=True)
    if not args.no_open:
        try:
            if os.name == 'nt':
                os.startfile(str(Path(html).resolve()))
            else:
                webbrowser.open(Path(html).resolve().as_uri())
        except OSError as error:
            print('[VIEWER] HTML was saved; open it manually: %s (%s)' % (html, error), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
