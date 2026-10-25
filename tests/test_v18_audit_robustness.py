import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

from lvimot.output.evaluation.dataset_audit import AuditThresholds, _label_object_motion_audit, _lidar_motion_audit


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_gt_motion_audit_rejects_low_speed_jitter_and_requires_persistence(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    # Five frames of sub-threshold physics jitter, then a sustained real motion.
    xs = [0.00, 0.01, 0.00, 0.01, 0.00, 0.20, 0.40, 0.60, 0.80]
    vs = [0.10, 0.15, 0.12, 0.10, 0.12, 1.0, 1.0, 1.0, 1.0]
    files = []
    for i, (x, v) in enumerate(zip(xs, vs)):
        p = labels / f"{i:06d}.json"
        _write_json(
            p,
            {
                "objects": [
                    {
                        "actor_id": 7,
                        "location": {"x": x, "y": 0.0, "z": 0.0},
                        "velocity": {"x": v, "y": 0.0, "z": 0.0},
                    }
                ]
            },
        )
        files.append(str(p))

    out = _label_object_motion_audit(files, AuditThresholds())
    assert not any(out["moving"][:4])
    # Sustained movement survives the 3-transition persistence filter.
    assert all(out["moving"][4:])


def _static_cloud():
    xs, ys = np.meshgrid(np.linspace(5.0, 35.0, 50), np.linspace(-12.0, 12.0, 40))
    z = np.zeros_like(xs)
    intensity = np.ones_like(xs)
    return np.stack([xs, ys, z, intensity], axis=-1).reshape(-1, 4).astype(np.float32)


def test_lidar_moving_object_does_not_become_global_ego_motion(tmp_path):
    files = []
    base = _static_cloud()
    rng = np.random.default_rng(3)
    for i in range(5):
        # Static background dominates. A small object cluster moves laterally.
        cluster = np.column_stack(
            [
                np.full(80, 18.0, dtype=np.float32),
                np.full(80, -4.0 + 0.6 * i, dtype=np.float32),
                rng.normal(0.5, 0.08, 80).astype(np.float32),
                np.ones(80, dtype=np.float32),
            ]
        )
        pts = np.vstack([base, cluster]).astype(np.float32)
        p = tmp_path / f"{i:06d}.bin"
        pts.tofile(p)
        files.append(str(p))

    out = _lidar_motion_audit(
        files,
        AuditThresholds(lidar_sample_stride=1, lidar_audit_max_points=4000),
    )
    assert not any(out["ego_moving"])
