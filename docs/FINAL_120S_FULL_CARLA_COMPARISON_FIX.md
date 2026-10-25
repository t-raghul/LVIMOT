# Final run — full 3D CARLA-versus-LVIMOT viewer

Target: **120-second limit, seed 42, Fast, 8 NPC, 105 processed frames**.

## What was wrong

The earlier `CARLA map` HTML checkbox was displaying `CARLA_reference_road_surface.ply` (OpenDRIVE drivable-road samples). It was not the CARLA world's full observed-area geometry. Since LVIMOT's `offline_4d_map.ply` contains walls, trees, ground and measured road, showing only purple road lines made the comparison incomplete and misleading.

## What's corrected

- `CARLA map` means **independent rendered 360-degree CARLA static surface reference**, generated from original CARLA world depth cameras at the saved route's recorded poses. The saved source has to be in the **same run**: `outputs_carla/live/sensor_recordings/<capture from live_results.json>/original_carla_map*/CARLA_independent_360_reference.ply`, with its `reference_manifest.json` naming that capture.
- `LVIMOT map` shows LVIMOT's estimated static map in gray-blue and camera/LiDAR measured road in green, parsed from `offline_4d_map.ply`. The object's path overlays are independent but hide when LVIMOT is turned off.
- Both checkboxes independently show/hide their map layers; combined visibility allows a proper colored overlay.
- CARLA world reference points use the inverse of the **exact first GT ego pose** for *display only*. This does not alter localization, 4D map estimation, objects, road scores, MOTA or any raw recorded data.
- A reference belonging to another capture is rejected. A road-only reference is **never** accepted as CARLA's full 3D map, and an unavailable full reference is clearly labeled as such.
- Regeneration uses a larger independent CARLA sample budget and auto fits the cloud to the view.

## How to update existing final HTML

Keep the complete **current** source project and recorded outputs. Copy the replacement files into the same project root, retaining `src/`, `tools/` and `tests/` directories. Then:

1. **Start CARLA 0.9.13** and ensure Town10HD_Opt is loaded. **Do not run LVIMOT while capturing the independent reference.**
2. Double-click **`regenerate_current_offline_map.bat`**.
3. The batch script validates the saved `live_results.json`, reads the referenced exact recording, and **captures the missing independent 360° reference** if needed.
4. It then generates and opens `outputs_carla/live/offline_4d_map.html` showing both full 3D clouds with independently selectable checkboxes.

If the independent 360 reference already exists for the exact run with its `reference_manifest.json`, CARLA need not be running and regeneration uses that file immediately. No LVIMOT rerun is required in either case.

If CARLA is unavailable, the script **refuses to present only road geometry as a full CARLA comparison**. It will not overwrite an existing HTML with false source data. Run CARLA and retry. If the **exact saved route files** are missing, their geometry cannot be regenerated from `live_results.json` alone; the script reports that error explicitly.

## Truthfulness and scope

CARLA ground truth and rendered reference geometry are kept strictly outside the estimator. CARLA's reference samples represent **only the recorded route's observed area** at LiDAR field/range limits, not all of Town10HD_Opt. Coverage and occlusions differ between the separate sensors. Two point clouds may not coincide everywhere even with valid map alignment, and a visual overlay is not proof of full 3D reconstruction quality.

The final run's reported **54.27% MOTA, 0.696m ego ATE RMSE, and 91.43% camera/LiDAR measured-road agreement** remain unchanged. That 91.43% is based on an OpenDRIVE *road-only evaluation* and is not a full-scene 3D accuracy metric.
