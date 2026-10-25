"""Pure geometry for independent rendered reference data; evaluation use only."""

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[3]))

import math

import numpy as np


def decode_depth_bgra(bgra):
    image = np.asarray(bgra, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError('Expected an HxWx4 BGRA depth buffer')
    values = image.astype(np.uint32)
    return ((values[:, :, 2] + values[:, :, 1] * 256 + values[:, :, 0] * 65536).astype(float) / 16777215.0) * 1000.0


def depth_camera_points(depth, semantic, horizontal_fov_deg, stride=2):
    depth = np.asarray(depth, dtype=float)
    labels = np.asarray(semantic, dtype=np.uint8)
    if labels.shape != depth.shape or depth.ndim != 2:
        raise ValueError('Depth and semantic images must have identical HxW shape')
    height, width = depth.shape
    focal = width / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))
    y, x = np.meshgrid(np.arange(0, height, stride), np.arange(0, width, stride), indexing='ij')
    d = depth[::stride, ::stride].reshape(-1)
    points = np.column_stack([d, (x.reshape(-1) - width / 2) * d / focal,
                              (height / 2 - y.reshape(-1)) * d / focal])
    return points, labels[::stride, ::stride].reshape(-1)


def sensor_field_mask(points_lidar, maximum_range_m, lower_fov_deg=-30.0, upper_fov_deg=10.0):
    points = np.asarray(points_lidar, dtype=float).reshape(-1, 3)
    distance = np.linalg.norm(points, axis=1)
    elevation = np.degrees(np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], axis=1)))
    return np.isfinite(points).all(axis=1) & (distance >= 1.5) & (distance <= maximum_range_m) & (elevation >= lower_fov_deg) & (elevation <= upper_fov_deg)


def matrix_to_pose(rotation, translation):
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    translation = np.asarray(translation, dtype=float).reshape(3)
    pitch = math.asin(float(np.clip(rotation[2, 0], -1, 1)))
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    roll = math.atan2(float(-rotation[2, 1]), float(rotation[2, 2]))
    return {'location': dict(zip(('x', 'y', 'z'), translation.tolist())),
            'rotation': dict(zip(('pitch', 'yaw', 'roll'), map(math.degrees, (pitch, yaw, roll))))}


def voxel_reference(points, labels, voxel_size=.25):
    xyz = np.asarray(points, dtype=float).reshape(-1, 3)
    semantics = np.asarray(labels, dtype=np.uint8).reshape(-1)
    if len(xyz) != len(semantics):
        raise ValueError('Every reference point needs a semantic label')
    if not len(xyz):
        return xyz.astype(np.float32), semantics
    keys = np.floor(xyz / voxel_size).astype(np.int64)
    unique, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    sums = np.zeros((len(unique), 3), dtype=float)
    np.add.at(sums, inverse, xyz)
    counts = np.bincount(inverse, minlength=len(unique))
    return (sums / counts[:, None]).astype(np.float32), semantics[first]
