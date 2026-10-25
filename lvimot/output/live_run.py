
# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))

import json
import os
import random
import time
import signal
import traceback
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from lvimot.inputs.carla_api import load_carla
from lvimot.inputs.sensor_calibration import CarlaCalibration
from lvimot.temporal_fusion.pipeline import CarlaLVIMOTCore
from lvimot.inputs.synchronized_sensors import CarlaLiveSensorAdapter
from lvimot.output.live_dashboard import CarlaLiveVisualizer
from lvimot.output.map_export import export_live_map, write_binary_ply
from lvimot.inputs.sensor_profile import with_lidar_profile
from lvimot.output.evaluation.map_geometry import validate_against_carla
from lvimot.output.offline_map import export_offline_4d_scene
from lvimot.output.pipeline_inspector import build_pipeline_snapshot
from lvimot.inputs.sensor_recording import RawSensorArchive
from lvimot.temporal_fusion.runtime_limits import DEFAULT_LIMITS


def _locrot(transform_dict):
    loc = transform_dict.get("location", transform_dict)
    rot = transform_dict.get("rotation", {})
    return loc, rot


def _to_carla_transform(carla, transform_dict):
    loc, rot = _locrot(transform_dict)
    return carla.Transform(
        carla.Location(
            x=float(loc.get("x", 0.0)),
            y=float(loc.get("y", 0.0)),
            z=float(loc.get("z", 0.0)),
        ),
        carla.Rotation(
            pitch=float(rot.get("pitch", 0.0)),
            yaw=float(rot.get("yaw", 0.0)),
            roll=float(rot.get("roll", 0.0)),
        ),
    )


def _snapshot_actor(snapshot, actor_id):
    if snapshot is None:
        return None
    try:
        return snapshot.find(int(actor_id))
    except Exception:
        return None


def _gt_pose(actor, snapshot=None):
    actor_snapshot = _snapshot_actor(snapshot, actor.id)
    if actor_snapshot is not None:
        t = actor_snapshot.get_transform()
    else:
        t = actor.get_transform()
    return {
        "location": {
            "x": float(t.location.x),
            "y": float(t.location.y),
            "z": float(t.location.z),
        },
        "rotation": {
            "pitch": float(t.rotation.pitch),
            "yaw": float(t.rotation.yaw),
            "roll": float(t.rotation.roll),
        },
    }


def _world_vertices(actor, transform):
    """Evaluation-only CARLA 3-D bounding-box vertices in world coordinates."""
    try:
        vertices = actor.bounding_box.get_world_vertices(transform)
    except Exception:
        return []
    return [
        [float(v.x), float(v.y), float(v.z)]
        for v in vertices
    ]


def _gt_objects(world, ego_id, snapshot=None):
    """Read CARLA actor ground truth for post-estimation evaluation only.

    When the live synchronizer returns a delayed GPU-camera frame, ``snapshot``
    is the buffered world snapshot for that same CARLA frame.  This keeps
    optional evaluation aligned without exposing simulator state to the
    estimator.
    """
    output = []
    actors = list(world.get_actors().filter("vehicle.*")) + list(
        world.get_actors().filter("walker.pedestrian.*")
    )
    for actor in actors:
        if int(actor.id) == int(ego_id):
            continue

        actor_snapshot = _snapshot_actor(snapshot, actor.id)
        if snapshot is not None and actor_snapshot is None:
            # The actor did not exist in the synchronized evaluation frame.
            continue

        if actor_snapshot is not None:
            t = actor_snapshot.get_transform()
            v = actor_snapshot.get_velocity()
        else:
            t = actor.get_transform()
            v = actor.get_velocity()

        ext = actor.bounding_box.extent
        world_corners = _world_vertices(actor, t)
        output.append(
            {
                "actor_id": int(actor.id),
                "type": actor.type_id,
                "location": {
                    "x": float(t.location.x),
                    "y": float(t.location.y),
                    "z": float(t.location.z),
                },
                "velocity": {
                    "x": float(v.x),
                    "y": float(v.y),
                    "z": float(v.z),
                },
                "dimensions": [
                    float(2.0 * ext.x),
                    float(2.0 * ext.y),
                    float(2.0 * ext.z),
                ],
                "rotation": {
                    "pitch": float(t.rotation.pitch),
                    "yaw": float(t.rotation.yaw),
                    "roll": float(t.rotation.roll),
                },
                "bounding_box": {
                    "world_corners": world_corners,
                },
            }
        )
    return output


def _choose_blueprint(rng, blueprints):
    candidates = list(blueprints)
    if not candidates:
        return None
    return candidates[rng.randrange(len(candidates))]


def _configure_vehicle_blueprint(bp, rng, role_name=None):
    if role_name and bp.has_attribute("role_name"):
        bp.set_attribute("role_name", str(role_name))
    if bp.has_attribute("color"):
        values = list(bp.get_attribute("color").recommended_values)
        if values:
            bp.set_attribute("color", values[rng.randrange(len(values))])
    if bp.has_attribute("driver_id"):
        values = list(bp.get_attribute("driver_id").recommended_values)
        if values:
            bp.set_attribute("driver_id", values[rng.randrange(len(values))])
    return bp


def _spawn_ego(carla, world, rng):
    library = world.get_blueprint_library()
    blueprints = list(library.filter("vehicle.*"))
    if not blueprints:
        raise RuntimeError("No vehicle blueprint found in CARLA world")
    spawn_points = list(world.get_map().get_spawn_points())
    if not spawn_points:
        raise RuntimeError("No vehicle spawn points available in CARLA map")
    rng.shuffle(spawn_points)

    for spawn in spawn_points:
        bp = _configure_vehicle_blueprint(_choose_blueprint(rng, blueprints), rng, "hero")
        vehicle = world.try_spawn_actor(bp, spawn)
        if vehicle is not None:
            return vehicle, spawn_points
    raise RuntimeError("Could not spawn ego vehicle at any CARLA map spawn point")


def _spawn_traffic(world, traffic_manager, rng, spawn_points, count):
    count = max(0, int(count))
    if count == 0:
        return []

    library = world.get_blueprint_library()
    blueprints = list(library.filter("vehicle.*"))
    points = list(spawn_points)
    rng.shuffle(points)
    actors = []
    tm_port = int(traffic_manager.get_port())

    for spawn in points:
        if len(actors) >= count:
            break
        bp = _configure_vehicle_blueprint(_choose_blueprint(rng, blueprints), rng, "autopilot")
        actor = world.try_spawn_actor(bp, spawn)
        if actor is None:
            continue
        actor.set_autopilot(True, tm_port)
        actors.append(actor)
    return actors


def _compact_live_record(result: Dict) -> Dict:
    tracks = result.get("tracking", {}).get("active_tracks", [])
    return {
        "frame": result.get("frame"),
        "carla_frame": result.get("carla_frame"),
        "timestamp": result.get("timestamp"),
        "state_estimation": result.get("state_estimation", {}),
        "tracks": [
            {
                "track_id": t.get("track_id"),
                "class": t.get("class"),
                "position": t.get("position"),
                "velocity": t.get("velocity"),
                "speed": t.get("speed"),
                "motion_state": t.get("motion_state"),
                "hits": t.get("hits"),
                "missed": t.get("missed"),
                "bbox": t.get("bbox"),
            }
            for t in tracks
        ],
        "mapping_4d": result.get("mapping_4d", {}),
        "sensor_integrity": result.get("sensor_integrity", {}),
        "evaluation_frame": result.get("objects", {}).get("evaluation", {}),
        "performance": result.get("performance", {}),
        "live_sync": result.get("live_sync", {}),
        # Read-only, compact per-frame intermediate values.  This field never
        # enters the estimator and is consumed only by the dashboard.
        "pipeline_inspector": build_pipeline_snapshot(result),
    }


def _print_preflight(client, world, calibration, fixed_delta, traffic_vehicles):
    try:
        server_version = client.get_server_version()
    except Exception:
        server_version = "unknown"
    try:
        client_version = client.get_client_version()
    except Exception:
        client_version = "unknown"
    map_name = getattr(world.get_map(), "name", "unknown")

    print("\n" + "=" * 70)
    print("LVIMOT LIVE V1 - CARLA PREFLIGHT")
    print("=" * 70)
    print(f"CARLA client: {client_version}")
    print(f"CARLA server: {server_version}")
    print(f"Map:          {map_name}")
    print(f"Fixed dt:     {fixed_delta:.3f} s ({1.0 / fixed_delta:.1f} Hz)")
    print(f"Camera:       {calibration.width} x {calibration.height}, FOV {calibration.fov:.1f} deg")
    print(f"LiDAR:        {calibration.lidar_channels} ch, {calibration.lidar_range:.1f} m")
    print('LiDAR rays:   {} / second; complete snapshot per tick'.format(calibration.data['lidar']['points_per_second']))
    print(f"NPC vehicles: {int(traffic_vehicles)} requested")
    print("Estimator:    RGB + LiDAR + IMU only")
    print("CARLA GT:     post-estimation evaluation only")
    print("=" * 70)



def _verify_map_export_before_carla(output_dir, visualizer):
    """Verify real PLY and map export code before the CARLA run spends time.

    This small synthetic test does NOT start CARLA, change mapper state or
    write into the user-visible map files. It catches missing imports,
    output-folder permissions, and raw-export regressions before capture.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    known_points = np.array([
        [1.0, 2.0, 0.0],
        [2.0, 2.0, 0.5],
        [3.0, 2.0, 1.0],
    ], dtype=np.float64)

    class _SyntheticMapper:
        dynamic_object_tubes = {}

        def get_static_map_points(self):
            return known_points.copy()

    try:
        with tempfile.TemporaryDirectory(prefix='.lvimot_map_preflight_', dir=str(out)) as tmp:
            result = export_live_map(
                _SyntheticMapper(), np.zeros(6, dtype=np.float64),
                visualizer, Path(tmp)
            )
            exported = Path(result['raw_ply'])
            if result['raw_voxels'] != len(known_points) or not exported.is_file():
                raise RuntimeError('Raw PLY export reported success but no correct file exists')
            raw = exported.read_bytes()
            expected_header = b'element vertex 3\n'
            if (expected_header not in raw or b'end_header\n' not in raw or
                    len(raw) < len(known_points) * 12):
                raise RuntimeError('Raw PLY export was incomplete or malformed')
    except Exception as exc:
        raise RuntimeError(
            'Map export preflight FAILED before capture; no CARLA frames processed: '
            + str(exc)
        ) from exc
    print('[MAP EXPORT PREFLIGHT PASS] PLY writer and output folder verified', flush=True)


def _evaluation_only_motion_failure(estimates, truth, min_frames=30):
    """Conservatively stop a bad validation run; GT never reaches estimation.

    Only fires when GT shows substantial actual travel (>15m) with
    substantial net displacement (>10m), but the sensor-estimated path is
    less than 35% of that travel. Stationary/spinning setups cannot trigger.
    """
    if len(estimates) < min_frames or len(truth) != len(estimates):
        return None
    ep = np.asarray(estimates, dtype=np.float64)[:, 3:6]
    gp = np.asarray(truth, dtype=np.float64)[:, 3:6]
    if not np.isfinite(ep).all() or not np.isfinite(gp).all():
        return 'Nonfinite localization trajectory'
    gt_travel = float(np.linalg.norm(np.diff(gp, axis=0), axis=1).sum())
    gt_net = float(np.linalg.norm(gp[-1] - gp[0]))
    est_travel = float(np.linalg.norm(np.diff(ep, axis=0), axis=1).sum())
    if gt_travel > 15.0 and gt_net > 10.0 and est_travel < 0.35 * gt_travel:
        return ('Severe LiDAR/IMU odometry underestimation: '
                'estimated {:.1f}m vs CARLA {:.1f}m traveled; '
                'raw sensor archive saved for offline replay'.format(est_travel, gt_travel))
    return None


def run_live(
    carla_install: str,
    calibration_file: str,
    host: str = "127.0.0.1",
    port: int = 2000,
    seconds: float = 120.0,
    wall_seconds: Optional[float] = None,
    autopilot: bool = False,
    evaluate_ground_truth: bool = False,
    no_display: bool = False,
    output_root: str = "outputs_carla",
    detector_mode: str = "camera_lidar",
    yolo_model: str = "models/yolov8n.pt",
    yolo_confidence: float = 0.06,
    yolo_image_size: int = 1280,
    yolo_device: str = "auto",
    traffic_vehicles: int = 12,
    fixed_delta: float = 0.1,
    seed: int = 42,
    sensor_timeout: float = 3.0,
    sliding_window_size: int = 4,
    map_voxel_size: float = 0.25,
    map_radius: float = 60.0,
    map_max_in_memory: int = 300000,
):
    fixed_delta = float(fixed_delta)
    if not np.isfinite(fixed_delta) or fixed_delta <= 0.0:
        raise ValueError("--fixed-delta must be a positive number of seconds")

    # All file writers are checked before CARLA starts or GPU models load.
    output_dir = os.path.join(output_root, "live")
    os.makedirs(output_dir, exist_ok=True)
    RawSensorArchive.preflight(output_dir, calibration_file)
    print('[SENSOR ARCHIVE PREFLIGHT PASS] Raw LiDAR + RGB + IMU can be recorded', flush=True)

    carla = load_carla(carla_install)
    calibration = CarlaCalibration(calibration_file)
    calibration.data = with_lidar_profile(calibration.data, fixed_delta)
    limits = replace(
        DEFAULT_LIMITS,
        sliding_window_size=max(2, int(sliding_window_size)),
        map_voxel_size_m=max(0.05, float(map_voxel_size)),
        map_rolling_radius_m=max(5.0, float(map_radius)),
        max_map_static_voxels=max(1000, int(map_max_in_memory)),
    )
    core = CarlaLVIMOTCore(
        calibration,
        limits=limits,
        detector_mode=detector_mode,
        yolo_model=yolo_model,
        yolo_confidence=yolo_confidence,
        yolo_image_size=yolo_image_size,
        yolo_device=yolo_device,
        road_surface_enabled=True,
    )
    visualizer = CarlaLiveVisualizer()
    rng = random.Random(int(seed))
    core.mapper_4d.enable_accumulated_map(Path(output_dir) / 'accumulated_maps')

    client = carla.Client(host, int(port))
    # Avoid killing an otherwise live session on a single >10s Unreal tick stall.
    client.set_timeout(max(20.0, float(sensor_timeout) + 5.0))
    world = client.get_world()
    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(8000 + max(0, int(port)-2000))
    all_actors: List = []
    sensor_adapter: Optional[CarlaLiveSensorAdapter] = None

    output_dir = os.path.join(output_root, "live")
    os.makedirs(output_dir, exist_ok=True)
    _verify_map_export_before_carla(output_dir, visualizer)
    jsonl_path = os.path.join(output_dir, "live_frames.jsonl")
    summary_path = os.path.join(output_dir, "live_results.json")
    # Export is display/review-only, kept separate from the live sensor estimator.
    # Stable rolling-map files remain available after the GUI is stopped.
    map_export_latest = None
    last_map_pose = None
    first_gt_pose = None
    latest_gt_pose = None
    last_validation = None
    stop_requested = False
    archive = None
    presentation = None
    live_cycle_times = []
    dashboard_only = os.environ.get('LVIMOT_DASHBOARD_ONLY','0') == '1'
    archive_path = None
    early_validation_stop = None
    connection_lost = False
    wall_start = time.monotonic()
    previous_signal_handlers = {}

    def _graceful_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print('[LVIMOT] Graceful stop requested; saving full PLY before exit', flush=True)

    # Windows GUI sends CTRL_BREAK_EVENT. Unlike the old handler, this keeps
    # Python running long enough to save the full map and report.
    for signame in ('SIGBREAK', 'SIGINT', 'SIGTERM'):
        signum = getattr(signal, signame, None)
        if signum is not None:
            try:
                previous_signal_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, _graceful_stop)
            except (ValueError, OSError):
                pass

    _print_preflight(client, world, calibration, fixed_delta, traffic_vehicles)

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_delta
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        try:
            traffic_manager.set_random_device_seed(int(seed))
        except Exception:
            pass

        vehicle, spawn_points = _spawn_ego(carla, world, rng)
        all_actors.append(vehicle)

        if autopilot:
            vehicle.set_autopilot(True, traffic_manager.get_port())

        traffic = _spawn_traffic(
            world,
            traffic_manager,
            rng,
            spawn_points,
            traffic_vehicles,
        )
        all_actors.extend(traffic)

        bp = world.get_blueprint_library()

        cam_bp = bp.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(calibration.width))
        cam_bp.set_attribute("image_size_y", str(calibration.height))
        cam_bp.set_attribute("fov", str(calibration.fov))
        cam_bp.set_attribute("sensor_tick", str(fixed_delta))
        camera = world.spawn_actor(
            cam_bp,
            _to_carla_transform(carla, calibration.camera_transform),
            attach_to=vehicle,
        )
        all_actors.append(camera)

        lidar_bp = bp.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", str(calibration.lidar_channels))
        lidar_bp.set_attribute("range", str(calibration.lidar_range))
        lidar_bp.set_attribute("rotation_frequency", str(1.0 / fixed_delta))
        lidar_bp.set_attribute('sensor_tick', str(fixed_delta))
        lidar_cfg = calibration.data.get("lidar", {})
        if "points_per_second" in lidar_cfg:
            lidar_bp.set_attribute("points_per_second", str(int(lidar_cfg["points_per_second"])))
        if "upper_fov" in lidar_cfg:
            lidar_bp.set_attribute("upper_fov", str(float(lidar_cfg["upper_fov"])))
        if "lower_fov" in lidar_cfg:
            lidar_bp.set_attribute("lower_fov", str(float(lidar_cfg["lower_fov"])))
        lidar_sensor = world.spawn_actor(
            lidar_bp,
            _to_carla_transform(carla, calibration.lidar_transform),
            attach_to=vehicle,
        )
        all_actors.append(lidar_sensor)

        imu_bp = bp.find("sensor.other.imu")
        imu_bp.set_attribute("sensor_tick", str(fixed_delta))
        imu_sensor = world.spawn_actor(
            imu_bp,
            _to_carla_transform(carla, calibration.imu_transform),
            attach_to=vehicle,
        )
        all_actors.append(imu_sensor)

        sensor_adapter = CarlaLiveSensorAdapter(
            world,
            camera,
            lidar_sensor,
            imu_sensor,
            timeout_s=sensor_timeout,
            max_timestamp_skew_s=max(1e-4, fixed_delta * 0.05),
        )

        max_frames = max(1, int(np.ceil(float(seconds) / fixed_delta)))
        print("\n" + "=" * 70)
        print("LVIMOT LIVE V1")
        print("RGB + LiDAR + IMU -> localization + MOT + factor graph + 4D map")
        print(f"Spawned traffic vehicles: {len(traffic)}")
        print("CARLA ground truth is never an estimator input")
        print("Press Q or ESC in the visualization window to stop")
        print("=" * 70)

        archive = RawSensorArchive(
            output_dir, calibration_file,
            run_settings={
                'seed': int(seed), 'fixed_delta': fixed_delta,
                'detector_mode': detector_mode,
                'yolo_confidence': float(yolo_confidence),
                'yolo_image_size': int(yolo_image_size),
                'yolo_model': str(yolo_model),
                'yolo_device': str(yolo_device),
                'autopilot': bool(autopilot),
                'evaluate_ground_truth': bool(evaluate_ground_truth),
                'effective_lidar_attributes': dict(lidar_sensor.attributes),
                'carla_map': world.get_map().name,
                'carla_server_version': client.get_server_version(),
                'map_policy': {'voxel_size_m': core.mapper_4d.static_voxel_size,
                               'rolling_radius_m': core.mapper_4d.max_map_distance,
                               'accumulated_map_enabled': True},
            },
        )
        archive_path = str(archive.path)
        archive.calibration_path.write_text(json.dumps(calibration.data, indent=2) + '\n', encoding='utf-8')
        print('[RAW SENSOR RECORDING] {}'.format(archive_path), flush=True)
        processed_frames = 0
        if dashboard_only:
            from lvimot.output.live_dashboard import DashboardPresentation
            presentation = DashboardPresentation(visualizer)
        with open(jsonl_path, "w", encoding="utf-8") as log_file:
            for live_idx in range(max_frames):
                if stop_requested:
                    print('[LVIMOT] Graceful stop: ending acquisition', flush=True)
                    break
                if wall_seconds is not None and (time.monotonic() - wall_start) >= float(wall_seconds):
                    print('[LVIMOT] Wall-clock run limit reached', flush=True)
                    break
                cycle_started = time.perf_counter()
                tick_started = cycle_started
                try:
                    live_frame = sensor_adapter.tick()
                except (RuntimeError, TimeoutError) as tick_exc:
                    message = str(tick_exc)
                    if isinstance(tick_exc, TimeoutError) or any(
                        term in message.lower()
                        for term in ("time-out", "timeout", "connection failed", "connection refused", "waiting for the simulator")
                    ):
                        connection_lost = True
                        print("[CARLA WORLD TICK LOST] {}".format(message), flush=True)
                        print("[LVIMOT] Preserving map and frame logs; skipping unsafe native actor cleanup.", flush=True)
                        break
                    raise

                tick_ms = (time.perf_counter()-tick_started)*1000.
                # Evaluation-only values are intentionally read after the live
                # RGB/LiDAR/IMU packet has already been assembled.  The core
                # consumes them only after estimator outputs are finalized.
                gt_pose = (
                    _gt_pose(vehicle, live_frame.world_snapshot)
                    if evaluate_ground_truth
                    else None
                )
                gt_objects = (
                    _gt_objects(world, vehicle.id, live_frame.world_snapshot)
                    if evaluate_ground_truth
                    else None
                )
                if gt_pose is not None:
                    if first_gt_pose is None:
                        first_gt_pose = gt_pose
                    latest_gt_pose = gt_pose

                # Persist ORIGINAL synchronized sensors before any estimator
                # operation can fail. GT is in metadata, not sensor arrays.
                archive_started = time.perf_counter()
                archive.write_frame(
                    live_idx, live_frame.timestamp, live_frame.carla_frame,
                    live_frame.image, live_frame.lidar, live_frame.imu,
                    gt_pose=gt_pose, gt_objects=gt_objects,
                )
                archive_ms = (time.perf_counter()-archive_started)*1000.
                result = core.process(
                    frame_id=live_idx,
                    timestamp=live_frame.timestamp,
                    image=live_frame.image,
                    lidar=live_frame.lidar,
                    imu_data=live_frame.imu,
                    ground_truth_ego_pose=gt_pose,
                    ground_truth_objects=gt_objects,
                    include_lidar_points=False,
                )
                if result.get("sensor_integrity", {}).get("imu_sample_rejected", False):
                    print("[IMU OUTLIER REJECTED] frame={} {}".format(
                        live_idx,
                        result["sensor_integrity"].get("imu_rejection_reason", "")),
                        flush=True)
                result["carla_frame"] = int(live_frame.carla_frame)
                last_map_pose = visualizer._pose(result)
                result["live_session"] = {
                    "ego_spawned": True,
                    "npc_spawned": int(len(traffic)),
                    "sensors_attached": True,
                    "autopilot": bool(autopilot),
                    "ground_truth_evaluation": bool(evaluate_ground_truth),
                    "calibration_file": str(calibration_file),
                    "camera_fov": float(calibration.fov),
                    "visual_stream_enabled": bool(not no_display and cv2 is not None),
                    "sliding_window_size": int(core.sliding_window.window_size),
                    "map_voxel_size_m": float(core.mapper_4d.static_voxel_size),
                    "map_radius_m": float(core.mapper_4d.max_map_distance),
                    "map_max_in_memory": int(core.mapper_4d.max_static_voxels),
                }
                result["live_sync"] = {
                    "timestamp_skew_s": float(live_frame.timestamp_skew_s),
                    "world_frame_at_return": live_frame.world_frame_at_return,
                    "simulation_lag_frames": int(live_frame.simulation_lag_frames),
                    **sensor_adapter.diagnostics(),
                }
                processed_frames += 1

                # Full raw export happens at frame 1, every 30 frames,
                # and on graceful stop. A display-cleaning error cannot lose raw PLY.
                if live_idx in (0, 1) or (live_idx > 0 and live_idx % 30 == 0):
                    try:
                        map_export_latest = export_live_map(
                            core.mapper_4d, last_map_pose, visualizer, Path(output_dir), include_accumulated=False
                        )
                        print('[MAP EXPORTED] frame={} voxels={} file={}'.format(
                            live_idx, map_export_latest['raw_voxels'],
                            map_export_latest['raw_ply']), flush=True)
                        for warning in map_export_latest.get('warnings', []):
                            print('[MAP EXPORT WARNING] '+warning, flush=True)
                    except Exception as export_exc:
                        print('[MAP EXPORT ERROR] {}'.format(export_exc), flush=True)
                        traceback.print_exc()
                        # Last-resort independent exporter, no visualizer needed.
                        try:
                            emergency = Path(output_dir) / 'live_map_EMERGENCY.ply'
                            write_binary_ply(emergency, core.mapper_4d.get_static_map_points())
                            print('[MAP EXPORTED EMERGENCY] {}'.format(emergency), flush=True)
                        except Exception:
                            traceback.print_exc()

                display_started = time.perf_counter()
                road_points = core.road_surface_layer.points() if core.road_surface_layer is not None else None
                if presentation is not None:
                    presentation.submit(live_frame.image,result,core.mapper_4d.get_static_map_points(),road_points)
                elif not no_display and cv2 is not None:
                    display = visualizer.render(
                        live_frame.image,
                        result,
                        core.mapper_4d.get_static_map_points(),
                        road_points=road_points,
                    )
                    cv2.imshow("LVIMOT LIVE V1: CARLA | 4D MAP", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        stop_requested = True
                perf = result['performance']
                perf['sensor_tick_ms'] = tick_ms
                perf['raw_archive_ms'] = archive_ms
                perf['presentation_submit_or_render_ms'] = (time.perf_counter()-display_started)*1000.
                perf['live_total_ms'] = (time.perf_counter()-cycle_started)*1000.
                perf['end_to_end_fps'] = 1000./max(perf['live_total_ms'],1e-6)
                live_cycle_times.append(perf['live_total_ms'])
                log_file.write(json.dumps(_json_safe(_compact_live_record(result)))+'\n')
                log_file.flush()
                stationary = bool(result.get('sensor_integrity',{}).get('stationary_active',False))
                print(
                    f"frame {live_idx:05d} / CARLA {live_frame.carla_frame:06d} | "
                    f"{perf['live_total_ms']:7.0f} ms | "
                    f"RSS {perf.get('rss_mb', 0):6.0f} MB | "
                    f"tracks {len(result.get('tracking', {}).get('active_tracks', [])):2d} | "
                    f"map {result.get('mapping_4d', {}).get('total_static_voxels', 0):6d} | "
                    f"stationary={stationary} | sync={live_frame.timestamp_skew_s*1000.:.2f} ms | "
                    f"camlag={live_frame.simulation_lag_frames}f | pump={sensor_adapter.diagnostics().get('world_ticks_last_call',0)}",
                    flush=True)

        # Persist FULL rolling map, even when Stop is clicked early.
        if presentation is not None:
            presentation.close()
        # CARLA reference is handled separately and only after estimation.
        if processed_frames and last_map_pose is not None:
            try:
                map_export_latest = export_live_map(
                    core.mapper_4d, last_map_pose, visualizer, Path(output_dir)
                )
                print('[MAP EXPORTED FINAL] {}'.format(map_export_latest['raw_ply']), flush=True)
            except Exception as export_exc:
                print('[MAP EXPORT ERROR FINAL] {}'.format(export_exc), flush=True)
                try:
                    write_binary_ply(Path(output_dir)/'live_map_EMERGENCY.ply',
                                     core.mapper_4d.get_static_map_points())
                except Exception:
                    traceback.print_exc()

        # This is the actual independent comparison with CARLA's OpenDRIVE
        # road geometry. It NEVER feeds reference data into the estimator.
        if (not connection_lost and processed_frames and evaluate_ground_truth and
                first_gt_pose is not None and latest_gt_pose is not None):
            try:
                last_validation = validate_against_carla(
                    world, core.mapper_4d, first_gt_pose,
                    latest_gt_pose, Path(output_dir)
                )
                print('[CARLA MAP VALIDATION] {} road agreement={}%, report={}'.format(
                    last_validation.get('status'),
                    last_validation.get('road_agreement_pct'),
                    Path(output_dir)/'CARLA_MAP_VALIDATION.json'), flush=True)
            except Exception as exc:
                print('[CARLA MAP VALIDATION ERROR] '+str(exc), flush=True)
                traceback.print_exc()
        # Keep the CARLA reference in the same immutable recording directory
        # so replay never needs a second simulator run to retrieve it.
        if archive is not None and last_validation is not None:
            reference = Path(output_dir) / 'CARLA_reference_road_surface.ply'
            if reference.is_file():
                import shutil
                shutil.copy2(str(reference), str(archive.path / reference.name))
        evaluation = core.evaluation_summary() if evaluate_ground_truth else None
        if last_validation is not None and evaluation is not None:
            # Evaluation ONLY: CARLA ground truth must never change the sensor
            # estimator or mapper. Publish clear independent accuracy gates,
            # rather than treating a plausible overlay as map certification.
            location = evaluation.get('localization') or {}
            tracking = evaluation.get('tracking') or {}
            ate = float(location.get('ate_rmse', float('inf')))
            rpe = float(location.get('rpe_rmse', float('inf')))
            road_check = last_validation.get('measured_road_surface') or last_validation
            road_pct = road_check.get('road_agreement_pct')
            road_n = int(road_check.get('candidate_road_height_map_points', 0))
            last_validation['road_check_protocol'] = ('measured_camera_lidar_road_surface'
                if 'measured_road_surface' in last_validation else 'legacy_mixed_scene_height_proxy')
            mota = tracking.get('mota')
            thresholds = {
                'localization_ate_rmse_max_m': 2.0,
                'localization_rpe_rmse_max_m': 0.5,
                'road_agreement_min_pct': 85.0,
                'min_road_height_candidates': 100,
                'tracking_mota_min': 0.5,
            }
            gates = {
                'localization': bool(np.isfinite(ate) and np.isfinite(rpe)
                                     and ate <= 2.0 and rpe <= 0.5),
                'local_road_surface': bool(road_pct is not None and
                                           float(road_pct) >= 85.0 and road_n >= 100),
                'dynamic_tracking': bool(mota is not None and
                                         np.isfinite(float(mota)) and float(mota) >= 0.5),
            }
            last_validation['thresholds_for_preliminary_checks'] = thresholds
            last_validation['measured_ate_rmse_m'] = round(ate, 4)
            last_validation['measured_rpe_rmse_m'] = round(rpe, 4)
            last_validation['measured_mota'] = mota
            last_validation['checks_passed'] = gates
            last_validation['overall_verdict'] = (
                'PRELIMINARY_CHECKS_PASSED_NOT_FULL_4D_CERTIFICATION'
                if all(gates.values()) and early_validation_stop is None
                else 'FAIL_NOT_VALIDATED'
            )
            last_validation['early_stop_reason'] = early_validation_stop
            last_validation['limitations'] = (
                'The OpenDRIVE reference is only a sampled drivable-road '
                'surface. It cannot validate all 3D buildings, unmapped '
                'areas or complete 4D trajectories. Independent CARLA '
                'localization and MOT evaluations are reported separately.'
            )
            (Path(output_dir) / 'CARLA_MAP_VALIDATION.json').write_text(
                json.dumps(_json_safe(last_validation), indent=2) + '\n',
                encoding='utf-8'
            )
            print('[MAP VALIDATION VERDICT] {}'.format(
                last_validation['overall_verdict']), flush=True)
            print('[ROAD AGREEMENT] measured surface={}%; full-scene height proxy={}%; reference thresholds unchanged'.format(
                road_pct,last_validation.get('road_agreement_pct')),flush=True)
            if visualizer.snapshot_path.is_file():
                try:
                    snapshot=json.loads(visualizer.snapshot_path.read_text(encoding='utf-8'))
                    snapshot['road_surface_agreement_pct']=road_pct
                    snapshot['legacy_scene_height_agreement_pct']=last_validation.get('road_agreement_pct')
                    temporary=visualizer.snapshot_path.with_suffix('.json.tmp')
                    temporary.write_text(json.dumps(snapshot,separators=(',',':')),encoding='utf-8')
                    os.replace(str(temporary),str(visualizer.snapshot_path))
                except Exception as display_error:
                    print('[ROAD DISPLAY WARNING] '+str(display_error),flush=True)
        # Export the complete read-only 4D scene after estimation and
        # independent CARLA evaluation, not inside the live state pipeline.
        offline_scene_latest = None
        if processed_frames:
            try:
                offline_scene_latest = export_offline_4d_scene(
                    core.mapper_4d, Path(output_dir),
                    first_gt_pose_evaluation_only=(first_gt_pose if evaluate_ground_truth
                                                  and last_validation is not None else None),
                )
                print('[OFFLINE 4D MAP EXPORTED] {}'.format(
                    offline_scene_latest['offline_scene_html']), flush=True)
            except Exception as offline_exc:
                print('[OFFLINE 4D MAP EXPORT ERROR] {}'.format(
                    offline_exc), flush=True)
                traceback.print_exc()

        # Ground-truth diagnosis is post-run reporting only. It cannot stop,
        # correct or select frames for the sensor-only estimator.
        post_run_motion_failure = (_evaluation_only_motion_failure(
            core.estimated_pose_history, core.ground_truth_pose_history)
            if evaluate_ground_truth else None)
        summary = {
            "mode": "live_carla",
            "version": "LVIMOT_LIVE_V1",
            "processed_frames": int(processed_frames),
            "runtime": dict(core.compute_runtime),
            "performance": {'end_to_end_fps': float(1000.*processed_frames/max(sum(live_cycle_times),1.)),
                'median_live_frame_ms':float(np.median(live_cycle_times)) if live_cycle_times else None,
                'active_frame_processing_seconds':float(sum(live_cycle_times)/1000.),
                'fps_includes_simulator_wait_recording_and_presentation_submit':True,
                'dashboard':presentation.summary() if presentation is not None else None},
            "estimator_inputs": ["RGB", "LiDAR", "IMU"],
            "estimator_ground_truth_used": False,
            "ground_truth_evaluation_enabled": bool(evaluate_ground_truth),
            "evaluation": evaluation,
            "mapping_4d": core.mapper_4d.export_summary(),
            "map_export_latest": map_export_latest,
            "offline_4d_scene_latest": offline_scene_latest,
            "carla_reference_map_validation": last_validation,
            "wall_seconds_limit": wall_seconds,
            "live_sync": sensor_adapter.diagnostics(),
            "spawned_traffic_vehicles": int(len(traffic)),
            "fixed_delta_seconds": float(fixed_delta),
            "frame_log": jsonl_path,
            "raw_sensor_recording": archive_path,
            "raw_recorded_frames": archive.count if archive is not None else 0,
            "validation_early_stop_reason": None,
            "post_run_motion_failure": post_run_motion_failure,
            "carla_connection_lost": bool(connection_lost),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(summary), f, indent=2)

        print("\n" + "=" * 70)
        print("LVIMOT LIVE V1 STOPPED (CARLA CONNECTION LOST)" if connection_lost else "LVIMOT LIVE V1 COMPLETE")
        print(f"Processed frames: {processed_frames}")
        print(f"Results: {summary_path}")
        if evaluation is not None:
            localization = evaluation.get("localization", {})
            tracking = evaluation.get("tracking", {})
            print(
                f"Live GT evaluation: ATE {localization.get('raw_ate_rmse', 0.0):.4f} m | "
                f"RPE {localization.get('rpe_rmse', 0.0):.4f} m | "
                f"MOTA {tracking.get('mota')}"
            )
        print("=" * 70)
        return summary
    except BaseException:
        # Print the real Python-side failure *before* touching CARLA actors.
        # CARLA 0.9.13 can terminate the interpreter during sensor teardown,
        # which otherwise masks the original exception and leaves only
        # Windows fast-fail 0xC0000409.
        print("\n[LVIMOT LIVE ERROR BEFORE CLEANUP]", flush=True)
        traceback.print_exc()
        raise
    finally:
        if presentation is not None:
            presentation.close()
        try:
            core.close()
        except Exception:
            traceback.print_exc()
        if archive is not None:
            archive.close()
        for signum, handler in previous_signal_handlers.items():
            try:
                signal.signal(signum, handler)
            except Exception:
                pass
        # Disable Python callbacks first, but deliberately do not call the
        # native CARLA sensor.stop() method here.  CARLA 0.9.13 on Windows
        # can fast-fail during native unsubscribe/teardown.  The sensor actors
        # are destroyed together with the other actors in the batch below.
        if sensor_adapter is not None:
            try:
                sensor_adapter.close()
            except Exception:
                traceback.print_exc()

        # Destroy actors through one client batch instead of repeated native
        # Actor.destroy()/Sensor.stop() calls from Python.
        if not connection_lost:
            try:
                actor_ids = [int(a.id) for a in reversed(all_actors)]
                if actor_ids:
                    client.apply_batch([carla.command.DestroyActor(actor_id) for actor_id in actor_ids])
            except Exception:
                traceback.print_exc()

            try:
                traffic_manager.set_synchronous_mode(False)
            except Exception:
                pass
            try:
                world.apply_settings(original_settings)
            except Exception:
                pass
        else:
            print("[LVIMOT] CARLA server disconnected; native actor destruction and world reset skipped.", flush=True)

        if (not no_display) and cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        if connection_lost:
            # The CARLA 0.9.13 native bindings can fatal-abort during process
            # interpreter teardown after the server disappears. All file
            # writers and archives are closed above. Bypass native destructors
            # here; the next run must restart/reconnect CARLA explicitly.
            print("[LVIMOT] CARLA disconnected; frame log and saved map retained. Restart CARLA before another run.", flush=True)
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(3)


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    # Handles np.bool_, np.integer, np.floating and other NumPy scalar
    # subclasses that can appear only after temporal state exists.
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def run_from_namespace(args):
    calibration_file = args.calibration_file
    if calibration_file is None:
        if args.carla_root:
            calibration_file = os.path.join(
                args.carla_root,
                "calibration",
                'lvimot.inputs.sensor_calibration.json',
            )
        else:
            raise ValueError(
                "Live mode requires --calibration-file or --carla-root pointing "
                "to the dataset calibration root"
            )
    return run_live(
        carla_install=args.carla_install,
        calibration_file=calibration_file,
        host=args.host,
        port=args.port,
        seconds=args.seconds,
        wall_seconds=getattr(args, 'wall_seconds', None),
        autopilot=args.autopilot,
        evaluate_ground_truth=args.evaluate_ground_truth,
        no_display=args.no_display,
        output_root=args.output_root,
        detector_mode=args.detector_mode,
        yolo_model=args.yolo_model,
        yolo_confidence=args.yolo_confidence,
        yolo_image_size=args.yolo_image_size,
        yolo_device=args.yolo_device,
        traffic_vehicles=args.traffic_vehicles,
        fixed_delta=args.fixed_delta,
        seed=args.seed,
        sensor_timeout=args.sensor_timeout,
        sliding_window_size=getattr(args, 'sliding_window_size', 4),
        map_voxel_size=getattr(args, 'map_voxel_size', 0.25),
        map_radius=getattr(args, 'map_radius', 60.0),
        map_max_in_memory=getattr(args, 'map_max_in_memory', 300000),
    )
