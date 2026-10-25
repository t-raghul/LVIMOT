"""Plot first-pose trajectory placement and measured/predicted object histories."""

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import sys as _lvimot_sys
from pathlib import Path as _lvimot_Path

from tools.validate_carla_depth_reference import _matrix_from_pose


def render(capture, trace_directory):
    trace = [json.loads(row) for row in (trace_directory / 'frame_trace.jsonl').read_text().splitlines()]
    archive = [json.loads(row) for row in (capture / 'frames.jsonl').read_text().splitlines()]
    rotation, translation = _matrix_from_pose(archive[0]['ground_truth_ego_pose_evaluation_only'])
    estimated = np.array([row['sensor_only']['pose'][3:6] for row in trace]) @ rotation.T + translation
    truth = np.array([[row['ground_truth_ego_pose_evaluation_only']['location'][k] for k in ('x', 'y', 'z')] for row in archive])
    seconds = np.array([row['timestamp'] for row in trace]) - trace[0]['timestamp']
    error = np.linalg.norm(estimated - truth, axis=1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(truth[:, 0], truth[:, 1], color='#555', lw=4, label='CARLA ego GT (evaluation)')
    axes[0, 0].plot(estimated[:, 0], estimated[:, 1], color='#0076bb', lw=2, ls='--', label='Sensor-only estimate')
    axes[0, 0].set(title='Ego trajectory in common CARLA coordinates', xlabel='World X (m)', ylabel='World Y (m)')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_aspect('equal', adjustable='datalim')
    axes[0, 1].plot(seconds, error, color='#0076bb')
    axes[0, 1].set(title='Trajectory position error; first-pose placement only', xlabel='Recorded simulation time (s)', ylabel='3D error (m)')
    actors = {}
    eligible_ids = {obj['actor_id'] for row in trace for obj in row['evaluation_only']['eligible_gt']}
    for row in archive:
        for obj in row['ground_truth_objects_evaluation_only']:
            if obj['actor_id'] in eligible_ids:
                corners = np.asarray(obj['bounding_box']['world_corners'])
                actors.setdefault(obj['actor_id'], []).append(corners.mean(0))
    for actor, points in actors.items():
        points = np.asarray(points)
        axes[1, 0].plot(points[:, 0], points[:, 1], ls='--', lw=2, label='NPC %s GT (includes occluded)' % actor)
    tracks = {}
    measured_counts, predicted_counts = [], []
    for row in trace:
        m, p = 0, 0
        for track in row['sensor_only']['published_tracks']:
            position = np.asarray(track['position']) @ rotation.T + translation
            tracks.setdefault(track['track_id'], []).append(position)
            if track.get('camera_only_predicted'):
                p += 1
            else:
                m += 1
        measured_counts.append(m)
        predicted_counts.append(p)
    for track_id, points in tracks.items():
        points = np.asarray(points)
        axes[1, 0].plot(points[:, 0], points[:, 1], marker='.', ms=3, lw=1, alpha=.75, label='Estimated track %d' % track_id)
    axes[1, 0].set(title='4D object histories (false tracks remain visible)', xlabel='World X (m)', ylabel='World Y (m)')
    axes[1, 0].legend(fontsize=6, ncol=2)
    axes[1, 1].step(seconds, measured_counts, where='post', label='Current measured 3D support')
    axes[1, 1].step(seconds, predicted_counts, where='post', label='Short RGB-only prediction bridge')
    axes[1, 1].set(title='Published evidence state over time', xlabel='Recorded simulation time (s)', ylabel='Published objects')
    axes[1, 1].legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=.2)
    fig.suptitle('LVIMOT trajectory and 4D tracking evidence — no fitted trajectory correction', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    output = trace_directory / 'independent_geometry' / 'TRAJECTORY_AND_4D_TRACKING.png'
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--trace-directory', type=Path, required=True)
    args = parser.parse_args()
    render(args.capture, args.trace_directory)
