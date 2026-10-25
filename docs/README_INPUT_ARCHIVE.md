> **21 October 2026 Individual Object 4D Map extension:** the 4D Map tab now shows per-car movement trails, stopped/parked-candidate 3D boxes, ID, speed and motion status. Core MOT/FGO/road logic is unchanged. Details: [individual object map guide](docs/INDIVIDUAL_OBJECT_4D_MAP.md).

> **21 October 2026 FPS and road update:** GPU inference, bounded CPU pools, faster unchanged FGO/map arithmetic, and a separate dashboard process. A 120-second Fast/8-NPC live check processed 132 frames versus 59 in the supplied run. A separate camera/LiDAR road layer scored 91.30%; the full-scene score remains visible at 54.88%. See [FPS and road results](FPS_ROAD_RESULTS_20261021.md). Start with **run_lvimot_gui.bat**, **Fast**, **Low**, and **Show CARLA spectator window** unchecked. The GPU runtime is installed separately and excluded from the ZIP.

> **21 October 2026 MOTA update:** The three requested raw replays achieve 51.45%, 52.57%, and 50.96% under the original NPC-only rules. See [the complete results](MOTA_REPLAY_RESULTS_20261021.md) and [model card](docs/mota_high_20261021/TRAFFIC_CONTEXT_MODEL_CARD.md). Keep `models/yolov8n.pt` and `models/yolov8n.traffic_context.npz` together.

# LVIMOT — CARLA 0.9.13 (Windows)

**Goal:** use RGB + LiDAR + IMU to estimate an ego trajectory, track vehicles and generate an observed-area 3-D/4-D map resembling the **independent original CARLA scene**. CARLA ground truth is **evaluation-only**, never an input to the estimator or map.

## Start

1. Keep your existing `.venv_live38` alongside this source, or double-click `setup_live_carla_env.bat` once to install Python dependencies (requires Python 3.8 x64 and internet).
2. Double-click `run_lvimot_gui.bat`; choose **Fast**. Start CARLA once, then click **START LVIMOT**.
3. Let LVIMOT finish, then find `live_map_latest.ply` and `live_map_clean_latest.ply` in `outputs_carla/live/`. The saved sensor route is under `outputs_carla/live/sensor_recordings/capture_*/`.

## Get CARLA's original map for comparison

- **To view CARLA's actual town:** start `start_carla_0913.bat`. LVIMOT uses CARLA's built-in **Town10HD_Opt** world (in the CARLA installation, not this repository).
- **To obtain an independent 3-D reference PLY of the area you just drove:** after a completed recorded LVIMOT run, **leave CARLA running**, stop LVIMOT and double-click **`export_original_carla_map.bat`**. The output folder opens automatically. Its `CARLA_independent_360_reference.ply` is rendered from CARLA's original world geometry, not reconstructed from LVIMOT LiDAR.
- Compare that file with `outputs_carla/live/live_map_latest.ply` in a viewer such as CloudCompare. For numerical scoring use `tools/evaluate_independent_scene_geometry.py`; see tool help.

**Important:** The exported reference covers the **recorded route only**, not the whole CARLA town. Do not run the reference export simultaneously with LVIMOT: it temporarily changes CARLA simulation settings, then restores them. CARLA must be running with Town10HD_Opt.

## Folder structure

- `src/`: live pipeline, localization, 3-D/4-D mapping, tracking, GUI, dataset research modules.
- `config/`: camera/LiDAR calibration and acceptance test criteria.
- `models/`: YOLO nano detector and optional segmentation weights.
- `tests/`: automated regression checks.
- `docs/`: historical, measured CARLA validation report and environment verification receipts.
- `outputs_carla/`: generated maps, captured sensor data, CARLA reference maps. Intentionally empty until you run the program.
- `tools/`: optional offline audits, plots, geometry evaluation and research diagnostics.
- Root Python scripts: live recording replay and independent CARLA-map export.

## Measured status (historical run)

The previously recorded 39 m route had approximately **0.17 m median LVIMOT-to-independent-CARLA surface distance**, 98.39% of assessed LVIMOT points within 1 m of reference and 95.16% reverse coverage within 1 m. These figures are **not a guarantee** for any new run. Vehicle tracking did **not** meet its acceptance target. Full-town reconstruction and hours-long reliability are **not** validated.

Read `docs/VALIDATION_2026-10-20.md` for exact evaluation details. The original captures and figures were deliberately omitted from this source-only ZIP.

## Requirements

- CARLA 0.9.13 WindowsNoEditor (default install path is `F:\Softwares\CARLA_0.9.13\WindowsNoEditor`, editable in the GUI).
- Python 3.8 x64; use `.venv_live38` on Windows.
- Dependencies in `requirements.txt`; the historical measured environment is recorded in `docs/verification_20261020/requirements_verified_python38.txt`.
- Do not delete your existing CARLA installation or virtual environment when updating source files.
