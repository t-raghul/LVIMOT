import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

from lvimot.temporal_fusion.sliding_window_factor_graph import VelocityPriorFactor
from lvimot.output.evaluation.dataset_audit import AuditThresholds, _runs_from_equal_flags, audit_sequence


def test_repeat_flag_runs_are_inclusive_frames():
    assert _runs_from_equal_flags([False, True, True, False], min_len=3) == [(1, 3)]
    assert _runs_from_equal_flags([True, True, True], min_len=3) == [(0, 3)]


def test_velocity_prior_zero_constraint():
    factor = VelocityPriorFactor(np.zeros(3), weight=10.0)
    residual = factor.compute_residual(np.array([0.1, -0.2, 0.3]))
    assert np.allclose(residual, [1.0, -2.0, 3.0])


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_png(path: Path, image: np.ndarray, compression: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(path),
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, int(compression)],
    )
    assert ok


def _make_sequence_dirs(seq: Path):
    for sub in ["image_02", "velodyne", "imu", "poses", "labels"]:
        (seq / sub).mkdir(parents=True, exist_ok=True)


def _write_common_non_rgb(seq: Path, frame_count: int, positions):
    for i in range(frame_count):
        points = np.array([[float(i), 0.0, 0.0, 1.0]], dtype=np.float32)
        points.tofile(seq / "velodyne" / f"{i:06d}.bin")
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
                "location": {"x": float(positions[i]), "y": 0.0, "z": 0.0},
                "rotation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            },
        )
        _write_json(seq / "labels" / f"{i:06d}.json", {"frame": i})
    (seq / "timestamps.txt").write_text(
        "\n".join(f"{0.1*i:.1f}" for i in range(frame_count)) + "\n",
        encoding="utf-8",
    )


def test_dataset_audit_detects_decoded_pixel_repeats_even_when_png_bytes_differ(tmp_path):
    seq = tmp_path / "sequences" / "0001"
    _make_sequence_dirs(seq)

    frame0 = np.zeros((24, 32, 3), dtype=np.uint8)
    frame0[:, :, 0] = 10
    frozen = np.zeros((24, 32, 3), dtype=np.uint8)
    frozen[:, :, 1] = 120

    _write_png(seq / "image_02" / "000000.png", frame0, compression=3)
    # Same decoded pixels, deliberately different PNG compression levels.
    _write_png(seq / "image_02" / "000001.png", frozen, compression=0)
    _write_png(seq / "image_02" / "000002.png", frozen, compression=9)
    _write_png(seq / "image_02" / "000003.png", frozen, compression=1)

    _write_common_non_rgb(seq, 4, positions=[0.0, 0.1, 0.2, 0.3])

    result = audit_sequence(
        str(tmp_path),
        "0001",
        AuditThresholds(pose_step_m=0.02, pose_step_deg=0.2, frozen_run_min=3),
    )

    assert result["ego_motion"] == "moving"
    assert result["duplicates"]["rgb_file_byte_runs"] == []
    assert result["duplicates"]["rgb_pixel_runs"] == [(1, 3)]
    assert result["duplicates"]["rgb_runs"] == [(1, 3)]
    assert result["duplicates"]["suspect_rgb_runs"] == [(1, 3)]
    assert result["cross_sensor"]["rgb_repeat_while_pose_moves_frames"] == [2, 3]
    assert "suspect frozen RGB run(s)" in result["issues"]


def test_dataset_audit_reports_mostly_stationary_ranges_instead_of_whole_sequence_moving(tmp_path):
    seq = tmp_path / "sequences" / "0003"
    _make_sequence_dirs(seq)

    frame_count = 20
    for i in range(frame_count):
        image = np.full((16, 24, 3), 40 if i >= 3 else (i + 1) * 10, dtype=np.uint8)
        _write_png(seq / "image_02" / f"{i:06d}.png", image, compression=(i % 9))

    positions = [0.0, 0.10, 0.20, 0.30] + [0.30] * (frame_count - 4)
    _write_common_non_rgb(seq, frame_count, positions=positions)

    result = audit_sequence(
        str(tmp_path),
        "0003",
        AuditThresholds(
            pose_step_m=0.02,
            pose_step_deg=0.2,
            frozen_run_min=3,
            mostly_stationary_fraction=0.20,
        ),
    )

    assert result["ego_motion"] == "mostly_stationary"
    assert result["pose"]["moving_ranges"] == [(0, 3)]
    assert result["pose"]["stationary_ranges"] == [(4, 19)]
    assert result["duplicates"]["rgb_pixel_runs"] == [(3, 19)]
    assert result["usable_stationary_localization_ranges"] == [(4, 19)]


def test_dataset_audit_detects_near_identical_pixel_run(tmp_path):
    seq = tmp_path / "sequences" / "0004"
    _make_sequence_dirs(seq)

    base = np.full((20, 20, 3), 100, dtype=np.uint8)
    images = [base.copy() for _ in range(4)]
    # One one-level pixel change per frame: visually identical and below the
    # near-repeat thresholds, but not exact pixel equality.
    images[1][0, 0, 0] = 101
    images[2][0, 0, 0] = 100
    images[3][0, 0, 0] = 101
    for i, image in enumerate(images):
        _write_png(seq / "image_02" / f"{i:06d}.png", image, compression=i)

    _write_common_non_rgb(seq, 4, positions=[0.0, 0.0, 0.0, 0.0])

    result = audit_sequence(
        str(tmp_path),
        "0004",
        AuditThresholds(
            frozen_run_min=3,
            rgb_near_mean_abs_diff=0.01,
            rgb_near_changed_fraction=0.01,
        ),
    )

    assert result["duplicates"]["rgb_pixel_runs"] == []
    assert result["duplicates"]["rgb_near_runs"] == [(0, 3)]
    assert result["ego_motion"] == "stationary"
