"""Compare complete sensor outputs with evaluation GT enabled and disabled."""

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def verify(enabled, disabled):
    def trace(folder):
        return [json.loads(row) for row in (folder / 'frame_trace.jsonl').read_text().splitlines()]
    a, b = trace(enabled), trace(disabled)
    if not a or len(a) != len(b) or [x['frame'] for x in a] != [x['frame'] for x in b]:
        raise ValueError('Matching nonempty full-recording traces are required')
    disabled_summary = json.loads((disabled / 'validation_summary.json').read_text())
    if disabled_summary['evaluation_enabled']:
        raise ValueError('The comparison recording must actually disable evaluation')
    difference = max(float(np.max(np.abs(np.array(x['sensor_only']['pose']) - np.array(y['sensor_only']['pose']))))
                     for x, y in zip(a, b))
    tracks = sum(x['sensor_only']['published_tracks'] == y['sensor_only']['published_tracks'] for x, y in zip(a, b))
    hashes = {}
    for filename in ('live_map_latest.ply', 'live_map_accumulated.ply'):
        hashes[filename] = hashlib.sha256((enabled / filename).read_bytes()).digest() == hashlib.sha256((disabled / filename).read_bytes()).digest()
    result = {
        'frames': len(a), 'ground_truth_disabled_replay_frames': len(b),
        'maximum_pose_component_difference': difference,
        'identical_published_tracking_frames': tracks,
        'byte_identical_ply_files': hashes,
        'passed_exact_comparison': difference == 0 and tracks == len(a) and all(hashes.values()),
        'adversarial_ground_truth_regression_test': 'test_evaluation_ground_truth_cannot_change_snapshot_estimation',
        'interpretation': 'Full recorded estimates, tracks and maps are compared; evaluation-only aligned PLY files are excluded.',
    }
    (enabled / 'GT_BOUNDARY_RECORDING_CHECK.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enabled', type=Path, required=True)
    parser.add_argument('--disabled', type=Path, required=True)
    args = parser.parse_args()
    if not verify(args.enabled, args.disabled)['passed_exact_comparison']:
        raise SystemExit(2)
