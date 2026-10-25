"""Lossless, bounded-per-frame live CARLA sensor recording for offline LVIMOT replay.

The archive is independent of the estimator. Ground truth is stored in the
metadata for evaluation only, never in the LiDAR/IMU/camera inputs.
"""
from __future__ import annotations

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class RawSensorArchive:
    def __init__(self, live_dir, calibration_file, run_settings=None):
        root = Path(live_dir) / 'sensor_recordings'
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.path = root / ('capture_' + stamp)
        self.path.mkdir(exist_ok=False)
        self.frames_path = self.path / 'frames.jsonl'
        self._file = self.frames_path.open('w', encoding='utf-8', buffering=1)
        cal = Path(calibration_file)
        self.calibration_path = self.path / 'calibration.json'
        self.calibration_path.write_bytes(cal.read_bytes())
        self.manifest_path = self.path / 'manifest.json'
        self.manifest_path.write_text(json.dumps({
            'format': 'LVIMOT_RAW_SENSOR_ARCHIVE_V1',
            'calibration': 'calibration.json',
            'frames': 'frames.jsonl',
            'settings': run_settings or {},
            'gt_for_evaluation_only': True,
        }, indent=2) + '\n', encoding='utf-8')
        self.count = 0

    def write_frame(self, frame_id, timestamp, carla_frame, image, lidar, imu,
                    gt_pose=None, gt_objects=None):
        i = int(frame_id)
        base = 'frame_{:06d}'.format(i)
        camera = np.ascontiguousarray(image)
        cloud = np.ascontiguousarray(lidar)
        if camera.ndim != 3 or camera.shape[2] != 3 or camera.dtype != np.uint8:
            raise ValueError('Sensor archive requires HxWx3 uint8 camera frame')
        if cloud.ndim != 2 or cloud.shape[1] < 3 or not np.isfinite(cloud[:, :3]).all():
            raise ValueError('Sensor archive requires finite Nx3+ LiDAR scan')
        if cv2 is None:
            raise RuntimeError('OpenCV is needed to record lossless RGB PNG frames')
        ok, encoded = cv2.imencode('.png', camera, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        if not ok:
            raise RuntimeError('Camera PNG encode failed')
        # Never leave a partially written frame with the final name.
        lidar_name = base + '_lidar.npy'
        img_name = base + '_camera.png'
        lidar_tmp = self.path / (lidar_name + '.tmp')
        image_tmp = self.path / (img_name + '.tmp')
        with lidar_tmp.open('wb') as handle:
            np.save(handle, cloud, allow_pickle=False)
        image_tmp.write_bytes(encoded.tobytes())
        os.replace(str(lidar_tmp), str(self.path / lidar_name))
        os.replace(str(image_tmp), str(self.path / img_name))
        record = {
            'frame': i, 'carla_frame': int(carla_frame), 'timestamp': float(timestamp),
            'lidar': lidar_name, 'camera': img_name,
            'imu': _json_clean(imu),
            'ground_truth_ego_pose_evaluation_only': _json_clean(gt_pose),
            'ground_truth_objects_evaluation_only': _json_clean(gt_objects),
        }
        self._file.write(json.dumps(record, allow_nan=False) + '\n')
        self._file.flush()
        self.count += 1

    def close(self):
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    @classmethod
    def preflight(cls, live_dir, calibration_file):
        """Offline archive roundtrip; rejects missing OpenCV/disk permissions early."""
        with tempfile.TemporaryDirectory(prefix='lvimot_archive_preflight_', dir=str(live_dir)) as td:
            obj = cls(td, calibration_file)
            try:
                image = np.zeros((16, 16, 3), dtype=np.uint8)
                image[:, :, 0] = 127
                lidar = np.array([[1., 2., 3., 0.2], [5., 6., 7., 0.8]], dtype=np.float32)
                imu = {'accelerometer': {'x': 0., 'y': 0., 'z': 9.81},
                       'gyroscope': {'x': 0., 'y': 0., 'z': 0.}, 'timestamp': 1.0}
                obj.write_frame(0, 1.0, 5, image, lidar, imu)
                restored = next(iter_archive(obj.path))
                np.testing.assert_array_equal(restored['lidar'], lidar)
                if restored['image'].shape != image.shape or restored['imu'] != imu:
                    raise RuntimeError('sensor archive roundtrip mismatch')
            finally:
                obj.close()


def _json_clean(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError('Unsupported sensor record type ' + str(type(obj)))


def iter_archive(path):
    root = Path(path)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format') != 'LVIMOT_RAW_SENSOR_ARCHIVE_V1':
        raise ValueError('Unsupported sensor archive format')
    with (root / manifest['frames']).open('r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            cloud = np.load(str(root / item['lidar']), allow_pickle=False)
            if cv2 is None:
                raise RuntimeError('OpenCV is needed to replay saved camera PNG')
            image = cv2.imread(str(root / item['camera']), cv2.IMREAD_COLOR)
            if image is None:
                raise IOError('Missing/corrupt camera frame ' + item['camera'])
            yield {
                'frame': item['frame'], 'carla_frame': item['carla_frame'],
                'timestamp': item['timestamp'], 'lidar': cloud, 'image': image,
                'imu': item['imu'],
                'ground_truth_ego_pose': item.get('ground_truth_ego_pose_evaluation_only'),
                'ground_truth_objects': item.get('ground_truth_objects_evaluation_only'),
            }
