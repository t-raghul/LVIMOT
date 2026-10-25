# Final-run offline CARLA/LVIMOT reference alignment

**Target run:** 2026-10-22, 120-second wall limit, seed 42, Fast preset, 8 NPC vehicles, 105 processed frames.

## Corrected display frame

- `offline_4d_map.ply` and `offline_4d_scene.json` are kept in the LVIMOT **estimated local** coordinate frame.
- `CARLA_reference_road_surface.ply` contains independent CARLA world-frame road points.
- Before drawing the CARLA reference in the offline HTML, the renderer now applies the inverse of the exact **first ground-truth ego pose** transformation used by `carla_map_validation.align_local_map_to_carla`:

  `carla_reference_in_local = (reference_world - first_pose_world_translation) @ first_pose_world_rotation`

- This rigid transformation is **for visualization/evaluation only**. It does not improve or change the estimated trajectories, drift, MOT, road map, point-cloud PLYs, or acceptance metrics.
- If the exact saved run's first ground-truth pose is unavailable, the CARLA checkbox is disabled instead of drawing knowingly unmatched coordinate systems.
- Regeneration picks the exact sensor recording named by that run's `live_results.json`; it does not use the newest capture directory from a different run.
- For future live runs, the reference is converted during final HTML export after the ground-truth evaluation stage has completed.

## Apply to the existing final run (no CARLA rerun)

Unzip the replacement archive in your existing LVIMOT project root, retaining `src/` and `tools/` paths. Then double-click `regenerate_current_offline_map.bat`, or run:

```bat
.venv_live38\Scripts\python.exe tools\regenerate_offline_html.py --output-dir outputs_carla\live
```

Required in `outputs_carla\live`:
- `offline_4d_map.ply`
- `offline_4d_scene.json`
- `CARLA_reference_road_surface.ply`
- `live_results.json`
- `sensor_recordings\capture_20261022_161253_989623\manifest.json`
- `sensor_recordings\capture_20261022_161253_989623\frames.jsonl`

The final pose is drawn in local map coordinates, including vehicles and paths. Files other than the regenerated HTML are not modified. The same run's performance metrics remain unchanged.

Note: physical geometries will not coincide everywhere: the reference represents CARLA drivable-road samples, while LVIMOT includes walls, vegetation and other scene geometry. The reported run has 0.696 m ego ATE RMSE and 91.43% measured-road agreement, so small positional differences remain physically plausible.
