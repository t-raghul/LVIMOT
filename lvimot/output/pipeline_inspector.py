"""Read-only, frame-aligned explanatory telemetry for the LVIMOT dashboard.

All measurements are derived from CarlaLVIMOTCore.process() output.  This
module NEVER feeds data back into sensor processing, localization or MOT.
Metadata is shared between the live runner and Tk dashboard.
Python 3.8 compatible; standard library only.
"""
from __future__ import annotations

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


import math
from typing import Any, Dict


PIPELINE_STAGES = (
    {
        "id": "inputs",
        "title": "Inputs",
        "short": "Camera · LiDAR · IMU",
        "purpose": "Receive the three live sensor streams that drive the project.",
        "algorithm": "CARLA sensor capture + same-frame synchronization",
        "files": ["lvimot/output/live_run.py", "lvimot/inputs/synchronized_sensors.py"],
        "input": "CARLA world tick, RGB camera frame, LiDAR scan and IMU reading",
        "output": "One synchronized sensor packet: image + point cloud + IMU sample",
    },
    {
        "id": "det2d",
        "title": "Preprocessing — 2D Object Detection",
        "short": "YOLOv8n",
        "purpose": "Find vehicle boxes in the RGB image.",
        "algorithm": "YOLOv8n object detector",
        "files": ["lvimot/preprocessing/camera_lidar_detection.py"],
        "input": "Current RGB image",
        "output": "2D vehicle detections: boxes, classes and confidence scores",
    },
    {
        "id": "visual",
        "title": "Preprocessing — Visual Features",
        "short": "FAST + BRIEF",
        "purpose": "Find repeatable image points and describe them so they can be matched across frames.",
        "algorithm": "FAST keypoints + BRIEF descriptors + Hamming-distance matching",
        "files": ["lvimot/preprocessing/visual_features.py", "lvimot/preprocessing/visual_features.py", "lvimot/tracking_4d/static_scene_support/feature_correspondences.py"],
        "input": "Current RGB image and previous visual history",
        "output": "Visual keypoints, descriptors and short visual tracks",
    },
    {
        "id": "lidar_snapshot",
        "title": "Preprocessing — LiDAR Snapshot",
        "short": "Scan ingest",
        "purpose": "Read the current LiDAR scan and prepare a clean point cloud for later geometry steps.",
        "algorithm": "LiDAR scan ingest and deskew preparation",
        "files": ["lvimot/preprocessing/lidar_snapshot.py", "lvimot/temporal_fusion/pipeline.py"],
        "input": "Raw LiDAR points",
        "output": "Live LiDAR point cloud / snapshot for the current frame",
    },
    {
        "id": "lidar_geometry",
        "title": "Preprocessing — LiDAR Planar Features + Depth",
        "short": "Planar support",
        "purpose": "Extract usable flat-surface geometry and depth support from the LiDAR scan.",
        "algorithm": "Voxelized planar-feature extraction + LiDAR-to-camera projection depth support",
        "files": ["lvimot/preprocessing/lidar_planar_features.py", "lvimot/preprocessing/sensor_projection.py", "lvimot/tracking_4d/static_scene_support/feature_correspondences.py"],
        "input": "LiDAR snapshot and calibration",
        "output": "Planar LiDAR features, stable correspondences and projected depth support",
    },
    {
        "id": 'lvimot.preprocessing.imu_preintegration',
        "title": "Preprocessing — IMU Pre-Integration",
        "short": "Motion summary",
        "purpose": "Summarize short-term motion between frames using acceleration and gyroscope readings.",
        "algorithm": "IMU preprocessing + pre-integration",
        "files": ["lvimot/preprocessing/imu_preintegration.py", "lvimot/preprocessing/imu_preintegration.py"],
        "input": "Accelerometer and gyroscope sample",
        "output": "Short-term motion estimate used by temporal fusion",
    },
    {
        "id": "static_scene",
        "title": "4D Tracking — Static Scene Support",
        "short": "Background support",
        "purpose": "Keep only useful static scene evidence and reject unwanted background structure.",
        "algorithm": "Visual + LiDAR feature support, geometric filtering and façade rejection",
        "files": ["lvimot/tracking_4d/static_scene_support/feature_correspondences.py", "lvimot/tracking_4d/static_scene_support/feature_correspondences.py", "lvimot/tracking_4d/static_scene_support/scene_consistency.py", "lvimot/tracking_4d/static_scene_support/measured_ground_support.py"],
        "input": "Visual features and LiDAR planar/depth support",
        "output": "Static feature correspondences for localization",
    },
    {
        "id": "mot",
        "title": "4D Tracking — Multimodal Object Tracking",
        "short": "Association + Kalman",
        "purpose": "Match vehicle observations over time and keep a stable ID for each moving object.",
        "algorithm": "Association cost matrix (IoU + 3D distance + shape) + Hungarian assignment + Kalman prediction",
        "files": ["lvimot/tracking_4d/multimodal_object_tracking/association.py", "lvimot/tracking_4d/multimodal_object_tracking/kalman_tracker.py", "lvimot/temporal_fusion/static_dynamic_classification.py"],
        "input": "2D detections, LiDAR-supported object measurements and previous tracks",
        "output": "Confirmed tracked vehicles with position and motion",
    },
    {
        "id": "static_dynamic",
        "title": "Temporal Fusion — Static / Dynamic",
        "short": "Motion check",
        "purpose": "Judge whether the ego vehicle is really stopped or moving before fusion uses a stationary hold.",
        "algorithm": "IMU quietness check + visual-motion check + LiDAR motion check",
        "files": ["lvimot/temporal_fusion/pipeline.py", "lvimot/temporal_fusion/static_dynamic_classification.py", "lvimot/temporal_fusion/runtime_limits.py"],
        "input": "IMU motion, visual motion, LiDAR motion and current tracking context",
        "output": "Static/dynamic decision and motion-validity flags",
    },
    {
        "id": "fgo",
        "title": "Temporal Fusion — Sliding Window Factor Graph Optimization",
        "short": "FGO",
        "purpose": "Combine the short recent history of IMU, LiDAR, visual and object evidence to estimate ego pose and object states.",
        "algorithm": "Sliding-window factor graph optimization with least-squares error minimization",
        "files": ["lvimot/temporal_fusion/sliding_window_factor_graph.py", "lvimot/temporal_fusion/sliding_window_factor_graph.py", "lvimot/temporal_fusion/sliding_window_factor_graph.py", "lvimot/tracking_4d/multimodal_object_tracking/multimodal_features.py"],
        "input": "IMU pre-integration, LiDAR relative motion, static feature correspondences and confirmed tracks",
        "output": "Estimated ego pose + velocity and tracked object states",
    },
    {
        "id": "marginalization",
        "title": "Temporal Fusion — Marginalization",
        "short": "Bounded history",
        "purpose": "Keep the optimization window small by removing old states while preserving their useful effect.",
        "algorithm": "Sliding-window marginalization / bounded history maintenance",
        "files": ["lvimot/temporal_fusion/sliding_window_factor_graph.py"],
        "input": "Current factor-graph window",
        "output": "Bounded optimization window with old information compressed",
    },
    {
        "id": "static_map",
        "title": "4D Mapping — Static Map",
        "short": "Rolling voxel map",
        "purpose": "Build the local static map around the ego vehicle.",
        "algorithm": "Rolling voxel-map update using LiDAR points and estimated ego pose",
        "files": ["lvimot/mapping_4d/static_map/rolling_voxel_map.py"],
        "input": "Estimated ego pose and filtered static LiDAR geometry",
        "output": "Rolling static voxel map",
    },
    {
        "id": "trajectory",
        "title": "4D Mapping — Multiobject Trajectory Management",
        "short": "Object paths",
        "purpose": "Store the path history of tracked dynamic objects.",
        "algorithm": "Per-track history management",
        "files": ["lvimot/mapping_4d/static_map/rolling_voxel_map.py"],
        "input": "FGO tracked-object states (world X/Y/Z and Vx/Vy/Vz); MOT supplies ID/class/size only",
        "output": "Individual paths built from FGO object positions and velocities; no direct MOT trajectories",
    },
    {
        "id": "output",
        "title": "Output",
        "short": "GUI + exports",
        "purpose": "Show the live result and save the main outputs.",
        "algorithm": "Read-only GUI rendering + JSON / PLY export",
        "files": ["lvimot/output/live_gui.py", "lvimot/output/live_dashboard.py", "lvimot/output/map_export.py"],
        "input": "Estimated ego pose, tracks, static map and synchronized image",
        "output": "Live GUI, 4D map, and exported JSON / PLY files",
    },
)


def _dict(value: Any) -> Dict:
    return value if isinstance(value, dict) else {}


def _finite(value: Any):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _fmt(value: Any, unit: str = "", precision: int = 2) -> str:
    if value is None:
        return "Not reported"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    number = _finite(value)
    if number is not None:
        return (str(int(number)) if precision == 0 else ("{:, .".replace(' ', '') + str(precision) + "f}").format(number)) + unit
    return str(value)[:96]


def _vec(value: Any, unit: str = "", limit: int = 3) -> str:
    if not isinstance(value, (list, tuple)) or not value:
        return "Not reported"
    return ", ".join(_fmt(v, precision=2) for v in value[:limit]) + unit


def build_pipeline_snapshot(result: Dict) -> Dict:
    result = _dict(result)
    session = _dict(result.get("live_session"))
    live_sync = _dict(result.get("live_sync"))
    perf = _dict(result.get("performance"))
    image = _dict(result.get("image"))
    camera = _dict(result.get("camera"))
    lidar = _dict(result.get("lidar"))
    imu = _dict(result.get("imu"))
    integrity = _dict(result.get("sensor_integrity"))
    temporal_visual = _dict(result.get("temporal_visual"))
    temporal_lidar = _dict(result.get("temporal_lidar"))
    objects = _dict(result.get("objects"))
    detector = _dict(objects.get("detector_diagnostics"))
    odom = _dict(result.get("lidar_odometry"))
    graph = _dict(result.get("graph"))
    pose_state = _dict(result.get("state_estimation"))
    mapping = _dict(result.get("mapping_4d"))
    tracking = _dict(result.get("tracking"))
    tracks = tracking.get("active_tracks") if isinstance(tracking.get("active_tracks"), list) else []
    frame = result.get("carla_frame", result.get("frame"))

    metrics = {
        "inputs": {
            "RGB frame": _fmt(result.get("frame"), precision=0),
            "CARLA frame": _fmt(frame, precision=0),
            "Image size": f"{image.get('width', 'Not reported')} × {image.get('height', 'Not reported')}",
            "LiDAR raw points": _fmt(lidar.get("raw_points"), precision=0),
            "IMU sample": "Available" if imu else "Not reported",
            "Timestamp skew": _fmt((_finite(live_sync.get("timestamp_skew_s")) or 0.0) * 1000.0 if live_sync.get("timestamp_skew_s") is not None else None, " ms"),
        },
        "det2d": {
            "2D detections": _fmt(detector.get("camera_detections"), precision=0),
            "Detector model": str(objects.get("detector_mode", "camera_lidar")),
            "Detector wall time": _fmt(perf.get("detector_wall_ms"), " ms"),
            "Detector wait": _fmt(perf.get("detector_wait_ms"), " ms"),
        },
        "visual": {
            "FAST raw": _fmt(camera.get("fast_keypoints_raw"), precision=0),
            "FAST retained": _fmt(camera.get("fast_keypoints_used"), precision=0),
            "BRIEF keypoints": _fmt(camera.get("brief_keypoints"), precision=0),
            "Visual tracks": _fmt(len(temporal_visual.get("active_tracks")) if isinstance(temporal_visual.get("active_tracks"), list) else None, precision=0),
            "Feature processing": _fmt(perf.get("camera_features_ms"), " ms"),
        },
        "lidar_snapshot": {
            "Raw points": _fmt(lidar.get("raw_points"), precision=0),
            "Deskewed points": _fmt(lidar.get("deskewed_points"), precision=0),
            "LiDAR processing": _fmt(perf.get("lidar_features_ms"), " ms"),
        },
        "lidar_geometry": {
            "Planar features": _fmt(lidar.get("planar_features"), precision=0),
            "Temporal planar used": _fmt(lidar.get("temporal_features_used"), precision=0),
            "Stable correspondences": _fmt(temporal_lidar.get("stable_landmarks"), precision=0),
            "Projected depth points": _fmt(detector.get("projection_valid_points"), precision=0),
            "LiDAR points in range": _fmt(detector.get("points_within_range"), precision=0),
            "ICP accepted": _fmt(odom.get("accepted")),
        },
        'lvimot.preprocessing.imu_preintegration': {
            "Accel magnitude": _fmt(integrity.get("accel_norm_m_s2"), " m/s²"),
            "Gyro magnitude": _fmt(integrity.get("gyro_norm_rad_s"), " rad/s"),
            "Outlier rejected": _fmt(integrity.get("imu_sample_rejected")),
            "Preintegration": "Available" if imu.get("preintegration") is not None else "Not available",
            "IMU processing": _fmt(perf.get("imu_ms"), " ms"),
        },
        "static_scene": {
            "Visual tracks": _fmt(len(temporal_visual.get("active_tracks")) if isinstance(temporal_visual.get("active_tracks"), list) else None, precision=0),
            "Stable LiDAR correspondences": _fmt(temporal_lidar.get("stable_landmarks"), precision=0),
            "Projected depth support": _fmt(detector.get("projection_valid_points"), precision=0),
            "Sensor proposals": _fmt(objects.get("sensor_proposals"), precision=0),
        },
        "mot": {
            "Sensor proposals": _fmt(objects.get("sensor_proposals"), precision=0),
            "Published tracks": _fmt(len(tracks), precision=0),
            "Track IDs": ", ".join(str(t.get("track_id", "?")) for t in tracks[:10]) if tracks else "None published",
            "Tracker processing": _fmt(perf.get("tracking_ms"), " ms"),
            "Hungarian matches": _fmt(perf.get("tracking_association_matches"), precision=0),
            "Association valid pairs": _fmt(perf.get("tracking_association_valid_pairs"), precision=0),
            "Current RGB track bridges": _fmt(perf.get("tracking_camera_bridges"), precision=0),
            "Facade proposals rejected": _fmt(detector.get("extended_facade_rejections"), precision=0),
            "Ground proposals rejected": _fmt(detector.get("extended_ground_rejections"), precision=0),
            "3-point foreground proposals": _fmt(detector.get("sparse_foreground_proposals"), precision=0),
        },
        "static_dynamic": {
            "Ego stationary": _fmt(integrity.get("stationary_active")),
            "Stationary streak": _fmt(integrity.get("stationary_streak"), precision=0),
            "Visual motion": _fmt(integrity.get("visual_motion_px"), " px"),
            "LiDAR translation": _fmt(integrity.get("lidar_translation_m"), " m"),
            "Motion veto": _fmt(integrity.get("translation_motion_veto")),
        },
        "fgo": {
            "Pose [roll,pitch,yaw,x,y,z]": _vec(pose_state.get("pose"), limit=6),
            "Velocity [vx,vy,vz]": _vec(pose_state.get("velocity"), " m/s"),
            "Window keyframes": _fmt(graph.get("window_keyframes"), precision=0),
            "Visual factors": _fmt(graph.get("visual_factors"), precision=0),
            "LiDAR factors": _fmt(graph.get("lidar_factors"), precision=0),
            "LiDAR relative factors": _fmt(graph.get("lidar_relative_factors"), precision=0),
            "IMU factors": _fmt(graph.get("imu_factors"), precision=0),
            "Dynamic factors": _fmt(graph.get("dynamic_factors"), precision=0),
            "FGO object states": _fmt(graph.get("object_states"), precision=0),
            "Optimization time": _fmt(perf.get("graph_opt_ms"), " ms"),
        },
        "marginalization": {
            "Window keyframes": _fmt(graph.get("window_keyframes"), precision=0),
            "Configured window": _fmt(session.get("sliding_window_size"), precision=0),
            "Bounded history": "Yes" if graph else "Not reported",
        },
        "static_map": {
            "Static voxels": _fmt(mapping.get("total_static_voxels"), precision=0),
            "Planar landmarks": _fmt(mapping.get("total_planar_landmarks"), precision=0),
            "Voxel size": _fmt(mapping.get("rolling_map_voxel_size_m"), " m"),
            "Map radius": _fmt(mapping.get("rolling_map_radius_m"), " m"),
            "Max in memory": _fmt(mapping.get("rolling_map_voxel_capacity"), precision=0),
            "Map processing": _fmt(perf.get("mapping_ms"), " ms"),
        },
        "trajectory": {
            "Dynamic actors stored": _fmt(mapping.get("total_dynamic_actors"), precision=0),
            "Published MOT tracks": _fmt(len(tracks), precision=0),
            "FGO tracks mapped this frame": _fmt(mapping.get("fgo_trajectory_tracks_this_frame"), precision=0),
            "Trajectory source": str(mapping.get("trajectory_source") or "Not reported"),
            "Object IDs": ", ".join(str(x) for x in (mapping.get("dynamic_actor_ids") or [])[:10]) if mapping.get("dynamic_actor_ids") else "None",
        },
        "output": {
            "Live GUI": "Enabled",
            "Published tracks": _fmt(len(tracks), precision=0),
            "Static voxels": _fmt(mapping.get("total_static_voxels"), precision=0),
            "Camera stream": "Enabled" if session.get("visual_stream_enabled") is True else "Disabled",
            "Pipeline time": _fmt(perf.get("total_ms"), " ms"),
        },
    }

    stage_status = {s["id"]: "COMPLETE" for s in PIPELINE_STAGES}
    if not session:
        stage_status["inputs"] = "UNAVAILABLE"
    if not result.get("camera"):
        stage_status["visual"] = "UNAVAILABLE"
        stage_status["det2d"] = "UNAVAILABLE"
    if not result.get("lidar"):
        stage_status["lidar_snapshot"] = "UNAVAILABLE"
        stage_status["lidar_geometry"] = "UNAVAILABLE"
        stage_status["static_scene"] = "UNAVAILABLE"
        stage_status["static_map"] = "UNAVAILABLE"
    if not result.get("graph"):
        stage_status["fgo"] = "UNAVAILABLE"
        stage_status["marginalization"] = "UNAVAILABLE"
    if integrity.get("imu_sample_rejected"):
        stage_status['lvimot.preprocessing.imu_preintegration'] = "WARNING"
    if integrity.get("stale_rgb"):
        stage_status["det2d"] = "WARNING"
        stage_status["visual"] = "WARNING"
    if integrity.get("stale_lidar"):
        stage_status["lidar_snapshot"] = "WARNING"
        stage_status["lidar_geometry"] = "WARNING"

    return {
        "version": 2,
        "frame": result.get("frame"),
        "carla_frame": frame,
        "timestamp": result.get("timestamp"),
        "snapshot_kind": "completed_frame",
        "stage_metrics": metrics,
        "stage_status": stage_status,
        "notes": (
            "This is a completed-frame snapshot, not an in-progress execution trace. "
            "Some preprocessing branches run in parallel. "
            "No inspector value is fed back into LVIMOT."
        ),
    }
