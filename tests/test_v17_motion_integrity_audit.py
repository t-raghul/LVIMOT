import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

from lvimot.output.evaluation.dataset_audit import AuditThresholds, audit_sequence


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_dirs(seq: Path):
    for sub in ["image_02", "velodyne", "imu", "poses", "labels"]:
        (seq / sub).mkdir(parents=True, exist_ok=True)


def _checker(h=120, w=160):
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, 12):
        for x in range(0, w, 12):
            if ((x // 12) + (y // 12)) % 2 == 0:
                image[y:y+5, x:x+5] = 220
            else:
                image[y:y+5, x:x+5] = 70
    return image


def _lidar_grid(shift_x=0.0):
    xs, ys = np.meshgrid(np.linspace(3, 30, 36), np.linspace(-10, 10, 28))
    z = np.zeros_like(xs)
    intensity = np.ones_like(xs)
    pts = np.stack([xs + shift_x, ys, z, intensity], axis=-1).reshape(-1, 4)
    return pts.astype(np.float32)


def _write_sequence_frame(seq: Path, i: int, image, pose_x, lidar_shift, actor_x):
    assert cv2.imwrite(str(seq / "image_02" / f"{i:06d}.png"), image)
    _lidar_grid(lidar_shift).tofile(seq / "velodyne" / f"{i:06d}.bin")
    _write_json(
        seq / "imu" / f"{i:06d}.json",
        {
            "accelerometer": {"x": 0.0, "y": 0.0, "z": 9.80665},
            "gyroscope": {"x": 0.0, "y": 0.0, "z": 0.0},
        },
    )
    _write_json(
        seq / "poses" / f"{i:06d}.json",
        {
            "location": {"x": float(pose_x), "y": 0.0, "z": 0.0},
            "rotation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        },
    )
    _write_json(
        seq / "labels" / f"{i:06d}.json",
        {
            "objects": [
                {
                    "actor_id": 10,
                    "type": "vehicle.test",
                    "location": {"x": float(actor_x), "y": 0.0, "z": 0.0},
                    "velocity": {"x": 1.0 if i > 0 else 0.0, "y": 0.0, "z": 0.0},
                }
            ]
        },
    )


def test_stationary_ego_with_moving_object_is_not_camera_motion(tmp_path):
    seq = tmp_path / "sequences" / "0004"
    _make_dirs(seq)
    base = _checker()
    n = 12
    for i in range(n):
        image = base.copy()
        # Textured moving traffic patch on a stationary background.
        x = 20 + 3 * i
        cv2.rectangle(image, (x, 45), (x + 35, 85), (120, 120, 120), -1)
        for yy in range(48, 84, 6):
            for xx in range(x + 3, x + 34, 6):
                cv2.circle(image, (xx, yy), 1, (255, 255, 255), -1)
        _write_sequence_frame(seq, i, image, pose_x=0.0, lidar_shift=0.0, actor_x=0.2 * i)
    (seq / "timestamps.txt").write_text(
        "\n".join(str(0.1 * i) for i in range(n)) + "\n", encoding="utf-8"
    )

    result = audit_sequence(str(tmp_path), "0004", AuditThresholds())
    assert result["sensor_consensus_motion"] in {"stationary", "mostly_stationary"}
    assert result["rgb_motion"]["ego_motion_ranges"] == []
    assert result["object_motion_audit_only"]["moving_ranges"]
    assert result["mot_active_candidate_ranges"]
    assert result["cross_sensor"]["rgb_lidar_move_pose_still_frames"] == []


def test_global_image_and_lidar_shift_produce_consensus_ego_motion(tmp_path):
    seq = tmp_path / "sequences" / "0006"
    _make_dirs(seq)
    base = _checker()
    n = 8
    for i in range(n):
        matrix = np.float32([[1, 0, 2 * i], [0, 1, 0]])
        image = cv2.warpAffine(base, matrix, (base.shape[1], base.shape[0]), borderMode=cv2.BORDER_REFLECT)
        _write_sequence_frame(seq, i, image, pose_x=0.6 * i, lidar_shift=0.8 * i, actor_x=0.0)
    (seq / "timestamps.txt").write_text(
        "\n".join(str(0.1 * i) for i in range(n)) + "\n", encoding="utf-8"
    )

    thresholds = AuditThresholds(lidar_global_still_overlap=0.95)
    result = audit_sequence(str(tmp_path), "0006", thresholds)
    assert result["pose"]["moving_ranges"]
    assert result["rgb_motion"]["ego_motion_ranges"]
    assert result["lidar_motion"]["ego_motion_ranges"]
    assert result["sensor_consensus_motion"] in {"moving", "intermittent_motion"}
    assert result["sensor_consensus"]["moving_ranges"]


def test_perceptually_static_camera_is_reported_without_calling_it_corrupt(tmp_path):
    seq = tmp_path / "sequences" / "0003"
    _make_dirs(seq)
    base = _checker(96, 128)
    n = 6
    for i in range(n):
        image = base.copy()
        # Tiny renderer-like pixel noise that defeats exact equality but should
        # not be mistaken for camera motion.
        image[0, i % image.shape[1], 0] = np.uint8(min(255, int(image[0, i % image.shape[1], 0]) + 1))
        _write_sequence_frame(seq, i, image, pose_x=0.0, lidar_shift=0.0, actor_x=0.0)
    (seq / "timestamps.txt").write_text(
        "\n".join(str(0.1 * i) for i in range(n)) + "\n", encoding="utf-8"
    )

    result = audit_sequence(str(tmp_path), "0003", AuditThresholds())
    assert result["duplicates"]["rgb_pixel_runs"] == []
    assert result["rgb_motion"]["visual_freeze_runs"]
    assert result["duplicates"]["suspect_rgb_runs"] == []
    assert result["sensor_consensus_motion"] == "stationary"
