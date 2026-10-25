# LVIMOT architecture and paths

The runtime has exactly the six diagram sections below. `tests/`, `tools/`,
`scripts/`, `config/`, `models/`, and `docs/` support those sections. They are
not additional estimator stages. The project root has no Python source files.

| Diagram section | Responsibility and implementation |
| --- | --- |
| INPUTS | `lvimot/inputs/synchronized_sensors.py`: synchronized live camera, LiDAR and IMU. `sensor_calibration.py`, `sensor_profile.py`: calibration and timing/profile. `sensor_recording.py`, `recorded_sensors.py`: recordings and saved sequences. `carla_api.py`: optional CARLA API discovery. `recorded_detections.py`: legacy recorded annotation reader, outside sensor measurement generation. |
| PREPROCESSING | `lvimot/preprocessing/visual_features.py`: FAST + BRIEF. `lidar_snapshot.py`: scan handling; no invented per-point timing. `lidar_planar_features.py`: planar feature extraction. `sensor_projection.py`: LiDAR/ego/camera transforms and 3D box projection. `imu_preintegration.py`: IMU normalization and SO(3) preintegration. `camera_lidar_detection.py`: YOLOv8n boxes with measured LiDAR depth, including extended façade/ground rejection. `lidar_object_measurements.py`: sensor geometry proposals. |
| 4D TRACKING | `lvimot/tracking_4d/static_scene_support/feature_correspondences.py`: visual and LiDAR temporal correspondences. `measured_ground_support.py` and `scene_consistency.py`: geometric scene support and consistency diagnostics. `multimodal_object_tracking/kalman_tracker.py`: association costs, Hungarian assignment, Kalman prediction/update and confirmed IDs. `association.py`: retained association helper. `multimodal_features.py`: cross-sensor feature construction. `traffic_context.py`: separate foreground-support model; it is neither YOLO nor SegFormer. |
| TEMPORAL FUSION | `lvimot/temporal_fusion/pipeline.py`: sensor-only orchestration and integrity gates. `static_dynamic_classification.py`: motion classification. `sliding_window_factor_graph.py`: factor definitions, factor graph optimization, sliding window, marginalization and current estimates of ego pose/velocity plus tracked object states. `sensor_inertial_anchor.py`: IMU/compass support. `runtime_limits.py`: bounded computation/history settings. |
| 4D MAPPING | `lvimot/mapping_4d/static_map/rolling_voxel_map.py`: rolling full-scene voxel map from estimated poses; records every processed estimated ego pose. `accumulated_map.py`: disk-backed route geometry. `measured_road_surface.py`: SegFormer-B0 labels select actual LiDAR returns; no flattening/filling. `multiobject_trajectory_management.py`: IndividualObjectScene, persistent object states/trajectories, and SQLite lifetime object + estimated ego histories. |
| OUTPUT | `lvimot/output/live_gui.py`: live GUI. `live_dashboard.py`: rendering and bounded worker process. `cli.py`, `live_run.py`: recorded/live entry points. `offline_map.py`: JSON/PLY/standalone HTML export. `map_export.py`, `recorded_visualization.py`: map/recorded presentation. `pipeline_inspector.py`: read-only architecture telemetry. `compute_runtime.py`: runtime selection. `evaluation/`: metrics, independent reference geometry, road audits, dataset/preflight checks and recorded IMU diagnostics. |

## Data flow and estimation boundary

The synchronized RGB, LiDAR and IMU packet enters preprocessing and tracking.
Static correspondences, inertial motion and confirmed tracks enter the sliding
window optimizer. Its accepted current ego pose places the measured map points;
its object states populate individual trajectories. The new ego archive receives
that same accepted estimated pose, timestamp, frame ID, heading and velocity.
It is a display/export sink and has no path back into tracking or optimization.

The HTML slider selects the last archived estimate at or before its time. It
does not extrapolate. The path contains only past accepted estimates; missing
frame IDs or intervals longer than one second break the drawn line. The complete
ego history stays on disk when the bounded live history evicts older samples.
With no disk archive, the mapper deliberately exports only its bounded history.

GT data stays in separate evaluation fields. The legacy GT motion-mismatch
diagnostic is now computed after the live loop. It cannot stop or correct the
sensor estimator. The independent CARLA 360 degree reference is displayed only
for its matching recorded capture and aligned for comparison after estimation.
The sensor map, object positions and estimated ego history remain in their
estimated local coordinate frame.

## Model names

The preprocessing object detector is **YOLOv8n** (`models/yolov8n.pt`). Its
foreground context artifact is a separate model (`yolov8n.traffic_context.npz`).
The static-map road selector is **SegFormer-B0**, not “SegFormer 80”. Its source
is `nvidia/segformer-b0-finetuned-cityscapes-768-768` at revision
`d3b7801ed329668d5bff04cd33365fa37f538c3b`.

The included configuration has encoder widths `[32, 64, 160, 256]`, depths
`[2, 2, 2, 2]`, decoder width 256, and 19 Cityscapes classes. The TorchScript
artifact SHA256 matches the packaged manifest:
`52a4948db6ce858ca76ae77064d45bb17c6a33b193253bc52837f446851ce734`.
The detector and segmenter weights, APIs and roles remain distinct.

## Consolidations and compatibility

FAST and BRIEF share `visual_features.py`. IMU preprocessing and preintegration
share `imu_preintegration.py`. The three projection/transform implementations
share `sensor_projection.py`. Visual and LiDAR correspondence classes share
`feature_correspondences.py`. Factors, graph and sliding-window classes share
`sliding_window_factor_graph.py`. Dashboard rendering and its worker share
`live_dashboard.py`. Their class APIs and algorithm implementations are retained.

All imports use package paths. Direct scripts bootstrap the project root before
importing dependencies. Windows launchers, GUI backend invocation, worker process
pickling, replayers, evaluators, inspector file references and source-inspection
tests use the new paths. Replayers that accept `--project` also retain an explicit
legacy `src/` import branch for comparisons against original projects. There are
no compatibility stub modules under old filenames. See `RENAMED_FILES.json` for
the complete mapping. Old manual dataset diagnostics are preserved in `tests/`.
