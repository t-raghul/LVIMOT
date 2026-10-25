"""Render reproducible static 3D and top-view comparison of two world-frame PLYs."""
from __future__ import annotations

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
from scipy.spatial import cKDTree

import sys as _lvimot_sys
from pathlib import Path as _lvimot_Path

from tools.validate_carla_depth_reference import _read_ply_xyz, _voxel_centroids


def render(capture: Path, estimated_file: Path, reference_file: Path, output: Path, title: str):
    frames = [json.loads(line) for line in (capture / 'frames.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    loc = frames[-1]['ground_truth_ego_pose_evaluation_only']['location']
    center = np.array([loc[k] for k in ('x', 'y', 'z')])
    e, r = _read_ply_xyz(estimated_file), _read_ply_xyz(reference_file)
    e = _voxel_centroids(e[np.linalg.norm(e - center, axis=1) <= 60.0], 0.25)
    r = _voxel_centroids(r[np.linalg.norm(r - center, axis=1) <= 60.0], 0.25)
    de, dr = cKDTree(r).query(e)[0], cKDTree(e).query(r)[0]
    fig = plt.figure(figsize=(17, 10), facecolor='#f5f7fa')
    fig.suptitle(title + '\nFirst recorded pose placement only; independent CARLA depth reference; observed region', fontsize=15)
    axes = [fig.add_subplot(2, 3, i) for i in (1, 2, 3)]
    axes[0].scatter(r[:, 0], r[:, 1], c=r[:, 2], cmap='terrain', vmin=-1, vmax=15, s=0.5, rasterized=True)
    axes[0].set_title('CARLA rendered static geometry ({:,} voxels)'.format(len(r)))
    colored = axes[1].scatter(e[:, 0], e[:, 1], c=np.minimum(de, 3.0), cmap='turbo', vmin=0, vmax=3, s=2, rasterized=True)
    axes[1].set_title('LVIMOT ({:,} voxels); color = distance to CARLA'.format(len(e)))
    fig.colorbar(colored, ax=axes[1], label='Nearest surface distance (m; clipped at 3 m)', shrink=0.8)
    axes[2].scatter(r[::3, 0], r[::3, 1], c='#a3aab4', s=0.5, label='CARLA depth', rasterized=True)
    axes[2].scatter(e[:, 0], e[:, 1], c='#0074bd', s=1.2, label='LVIMOT', rasterized=True)
    axes[2].set_title('Common-coordinate overlay')
    axes[2].legend(loc='upper right', markerscale=3)
    for ax in axes:
        ax.set_xlim(center[0] - 60, center[0] + 60)
        ax.set_ylim(center[1] - 60, center[1] + 60)
        ax.set_aspect('equal')
        ax.set_xlabel('CARLA world X (m)')
        ax.set_ylabel('CARLA world Y (m)')
        ax.grid(alpha=0.15)
    for index, points, color, name in ((4, r, '#5d8a67', 'Independent CARLA 3D'), (5, e, '#0074bd', 'LVIMOT estimated 3D')):
        ax = fig.add_subplot(2, 3, index, projection='3d')
        shown = points[::max(1, int(np.ceil(len(points) / 18000)))]
        ax.scatter(shown[:, 0], shown[:, 1], shown[:, 2], c=color, s=0.8, alpha=0.65, rasterized=True)
        ax.set_xlim(center[0] - 60, center[0] + 60)
        ax.set_ylim(center[1] - 60, center[1] + 60)
        ax.set_zlim(-2, 18)
        ax.set_box_aspect((1, 1, 0.35))
        ax.view_init(elev=28, azim=-65)
        ax.set_title(name)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax = fig.add_subplot(2, 3, 6)
    thresholds = np.linspace(0, 3, 151)
    ax.plot(thresholds, [100 * np.mean(de <= t) for t in thresholds], label='LVIMOT → CARLA')
    ax.plot(thresholds, [100 * np.mean(dr <= t) for t in thresholds], label='CARLA → LVIMOT')
    ax.axvline(1, color='#777', ls='--', lw=1)
    ax.set(xlabel='Nearest surface distance threshold (m)', ylabel='Points within threshold (%)', ylim=(0, 100))
    ax.set_title('Accuracy and coverage diagnostic')
    ax.grid(alpha=0.25); ax.legend()
    fig.text(0.02, 0.015, 'Scope: 60 m sphere at final pose. Independent rendered scene; no global ICP fitting. GT is evaluation-only.', fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--estimated', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--title', default='LVIMOT versus original CARLA geometry')
    args = parser.parse_args()
    render(args.capture, args.estimated, args.reference, args.output, args.title)
