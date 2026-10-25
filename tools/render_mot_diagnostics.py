"""Inspect RGB boxes and measured LiDAR returns at recorded MOT misses."""

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[1]))

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lvimot.inputs.sensor_calibration import CarlaCalibration
from tools.analyze_recorded_trace import best


def render(capture, trace_directory, output, selected=None, actor=None):
    calibration = CarlaCalibration(str(capture / 'calibration.json'))
    rows = {row['frame']: row for row in [json.loads(line) for line in (trace_directory / 'frame_trace.jsonl').read_text().splitlines()]}
    selected = selected or [38, 41, 83, 85, 90, 96]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor='white')
    for ax, frame in zip(axes.ravel(), selected):
        row = rows[frame]
        choices = row['evaluation_only']['eligible_gt']
        truth = next((g for g in choices if g.get('actor_id') == actor), choices[0])
        gt_box = np.asarray(truth['bbox'])
        sensor = row['sensor_only']
        _, cam = best(gt_box, sensor['camera_detections'])
        _, proposal = best(gt_box, sensor['lidar_proposals'])
        image = cv2.cvtColor(cv2.imread(str(capture / ('frame_%06d_camera.png' % frame))), cv2.COLOR_BGR2RGB)
        cloud = np.load(str(capture / ('frame_%06d_lidar.npy' % frame)))
        uv, valid = calibration.project_lidar(cloud[:, :3])
        depths = calibration.lidar_to_opencv(cloud[:, :3])[:, 2]
        x1, y1, x2, y2 = gt_box
        left, right = max(0, int(x1 - 65)), min(image.shape[1], int(x2 + 65))
        top, bottom = max(0, int(y1 - 50)), min(image.shape[0], int(y2 + 50))
        ax.imshow(image)
        keep = valid & (uv[:, 0] >= left) & (uv[:, 0] <= right) & (uv[:, 1] >= top) & (uv[:, 1] <= bottom)
        ax.scatter(uv[keep, 0], uv[keep, 1], c=depths[keep], cmap='turbo', vmin=0, vmax=60, s=10, edgecolors='black', linewidths=0.2)
        for box, color, style, label in ((gt_box, '#00ff55', '-', 'GT evaluation'),
                                          (cam['bbox'] if cam else None, '#ffe600', '--', 'YOLO'),
                                          (proposal['bbox'] if proposal else None, '#ff3cf1', ':', 'LiDAR proposal')):
            if box is not None:
                xa, ya, xb, yb = box
                ax.plot([xa, xb, xb, xa, xa], [ya, ya, yb, yb, ya], color=color, ls=style, lw=1.7, label=label)
        title = 'Frame %d | actor %s | GT returns %s' % (frame, truth.get('actor_id'), truth.get('lidar_support'))
        title += '\nRGB conf %.2f | proposal returns %s' % ((cam or {}).get('camera_confidence', 0), (proposal or {}).get('camera_support', 'none'))
        ax.set_title(title, fontsize=10)
        ax.set(xlim=(left, right), ylim=(bottom, top))
        ax.set_aspect('equal')
        ax.axis('off')
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3)
    fig.suptitle('Recorded sparse-vehicle diagnosis — dots are actual projected LiDAR returns (color = depth)', fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(output, dpi=130)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True, type=Path)
    parser.add_argument('--trace-directory', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--frames', nargs=6, type=int)
    parser.add_argument('--actor', type=int)
    args = parser.parse_args()
    render(args.capture, args.trace_directory, args.output, args.frames, args.actor)
