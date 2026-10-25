r"""Evaluation-only MOT detector audit for a previously recorded LVIMOT Fast run.

Runs YOLO+LiDAR detection ONLY on frames with CARLA-evaluation-eligible actors.
Does NOT start CARLA, rerun localization, update a map, or give GT to detector.

Windows usage from CARLA_LVIMOT-main:
.venv_live38\Scripts\python.exe audit_saved_mot_detector.py --capture "outputs_carla\live\sensor_recordings\capture_20261020_085053_922129"

For an environment lacking YOLO, --geometry-only verifies GT observability and
writes a report without inference. Only inference mode can diagnose YOLO misses.
"""
from __future__ import annotations

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[1]))


import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]


from lvimot.inputs.sensor_calibration import CarlaCalibration
from lvimot.temporal_fusion.pipeline import CarlaLVIMOTCore
from lvimot.inputs.sensor_recording import iter_archive


def bbox_iou(a, b):
    """Intersection-over-union of 2D boxes; robust to missing boxes."""
    if a is None or b is None or len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = (float(x) for x in a)
    bx1, by1, bx2, by2 = (float(x) for x in b)
    if not (ax2 > ax1 and ay2 > ay1 and bx2 > bx1 and by2 > by1):
        return 0.0
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    overlap = ix * iy
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    denom = a_area + b_area - overlap
    return float(overlap / denom) if denom > 0 else 0.0


def best_overlap(reference_bbox, candidates: List[Dict]):
    """Return best IoU and corresponding candidate without modifying inputs."""
    if not candidates:
        return 0.0, None
    ranked = sorted(
        ((bbox_iou(reference_bbox, c.get("bbox")), c) for c in candidates),
        key=lambda item: item[0], reverse=True,
    )
    return ranked[0]


def diagnostic_outcome(camera_iou: float, proposal_iou: float, threshold=0.25):
    """Classifies failures by stage, NOT by inferred ground-truth identity."""
    if camera_iou < threshold:
        return "NO_MATCHING_YOLO_2D_BOX"
    if proposal_iou < threshold:
        return "YOLO_BOX_WITHOUT_MATCHING_LIDAR_3D_PROPOSAL"
    return "MATCHING_CAMERA_AND_LIDAR_PROPOSAL_AVAILABLE"


def audit(capture: Path, model: str, device: str, geometry_only: bool = False):
    capture = Path(capture).resolve()
    if not (capture / "manifest.json").is_file():
        raise FileNotFoundError("Missing raw sensor capture manifest: " + str(capture))
    manifest = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
    settings = dict(manifest.get("settings") or {})
    calibration = CarlaCalibration(str(capture / manifest.get("calibration", "calibration.json")))
    # Explicitly avoid model auto-downloads; the correct YOLO weights must
    # exist in this project, so a missing file is an immediate cheap error.
    if not geometry_only and not Path(model).is_file():
        raise FileNotFoundError("No local YOLO weights: " + model)
    core = CarlaLVIMOTCore(
        calibration,
        detector_mode="camera_lidar",
        yolo_model=model,
        yolo_confidence=float(settings.get("yolo_confidence", 0.06)),
        yolo_image_size=int(settings.get("yolo_image_size", 640)),
        yolo_device=device,
    )
    out = capture / "offline_replay"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    inspected = 0
    eligible_frames = 0
    frame_count = 0
    try:
        for item in iter_archive(capture):
            frame_count += 1
            gt = item.get("ground_truth_ego_pose")
            actors = item.get("ground_truth_objects") or []
            if gt is None or not actors:
                continue
            image = item["image"]
            cloud = item["lidar"]
            gt_boxes = core._project_gt_camera_boxes(actors, gt, image.shape)
            gt_in_lidar = core._gt_objects_current_lidar(actors, gt)
            eligible = core._filter_gt_observable_current_lidar(gt_in_lidar, gt_boxes, cloud)
            if not eligible:
                continue
            eligible_frames += 1
            camera_detections = []
            proposals = []
            diagnostics = {}
            if not geometry_only:
                # The ONLY detector inputs are original camera and LiDAR arrays.
                # CARLA evaluation objects/boxes remain outside the detector.
                proposals = core.camera_lidar_detector.detect(image, cloud)
                camera_detections = list(core.camera_lidar_detector.last_camera_detections)
                diagnostics = dict(core.camera_lidar_detector.last_diagnostics)
                inspected += 1
            for obj in eligible:
                actor_id = int(obj.get("actor_id", -1))
                box = gt_boxes[actor_id]
                cam_iou, cam = best_overlap(box, camera_detections)
                proposal_iou, prop = best_overlap(box, proposals)
                row = {
                    "frame": int(item["frame"]),
                    "actor_id_evaluation_only": actor_id,
                    "gt_bbox_evaluation_only": [round(float(x), 3) for x in box],
                    "range_m_evaluation_only": round(float(obj.get("evaluation_range_m", 0)), 3),
                    "lidar_returns_in_gt_box_evaluation_only": int(obj.get("lidar_support", 0)),
                    "camera_boxes_detected": len(camera_detections) if not geometry_only else None,
                    "lidar_3d_proposals_detected": len(proposals) if not geometry_only else None,
                    "best_camera_bbox_iou_evaluation_only": round(cam_iou, 4) if not geometry_only else None,
                    "best_lidar_proposal_bbox_iou_evaluation_only": round(proposal_iou, 4) if not geometry_only else None,
                    "best_camera_confidence": round(float(cam.get("camera_confidence", 0.0)), 4) if cam is not None else None,
                    "best_lidar_proposal_score": round(float(prop.get("score", 0.0)), 4) if prop is not None else None,
                    "best_lidar_proposal_point_support": int(prop.get("camera_support", 0)) if prop is not None else None,
                    "detector_diagnostics": diagnostics,
                    "stage_outcome": (
                        diagnostic_outcome(cam_iou, proposal_iou)
                        if not geometry_only else "GEOMETRY_ONLY_YOLO_NOT_RUN"
                    ),
                }
                rows.append(row)
        outcome = Counter(row["stage_outcome"] for row in rows)
        summary = {
            "mode": "OFFLINE_MOT_DETECTOR_AUDIT_EVALUATION_ONLY",
            "complete": not geometry_only,
            "raw_recording_frames": frame_count,
            "frames_with_eligible_gt_vehicles": eligible_frames,
            "frames_processed_by_yolo": inspected,
            "eligible_gt_vehicle_observations": len(rows),
            "unique_evaluation_actor_ids": sorted({row["actor_id_evaluation_only"] for row in rows}),
            "stage_outcomes": dict(outcome),
            "iou_threshold": 0.25,
            "detector_settings": {
                "weights": str(model),
                "device": str(device),
                "confidence": float(settings.get("yolo_confidence", 0.06)),
                "image_size": int(settings.get("yolo_image_size", 640)),
            },
            "cautions": [
                "Ground-truth actor boxes are used ONLY to select diagnostic frames and score detector outputs, never for detector inputs.",
                "The GT box includes occlusion and depth effects; 2D IoU below 0.25 is not absolute proof of detection failure.",
                "The scan is evaluated without tracker state; downstream MOT identity and persistence require separate diagnosis.",
                "MOTA is NOT recalculated by this detector-only audit.",
            ],
            "per_frame_jsonl": "MOT_DETECTOR_AUDIT_FRAMES.jsonl",
        }
        with (out / "MOT_DETECTOR_AUDIT_FRAMES.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
        (out / "MOT_DETECTOR_AUDIT.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print("MOT AUDIT COMPLETED:", out / "MOT_DETECTOR_AUDIT.json", flush=True)
        print("GT-eligible observations:", len(rows), "YOLO frames:", inspected, flush=True)
        print("Stage outcomes:", dict(outcome), flush=True)
        return summary
    finally:
        close = getattr(core, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", required=True)
    p.add_argument("--model", default="models/yolov8n.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--geometry-only", action="store_true")
    args = p.parse_args()
    audit(Path(args.capture), args.model, args.device, args.geometry_only)
