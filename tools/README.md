# Optional LVIMOT diagnostics

These scripts are useful when archived runs exist under `outputs_carla/live/sensor_recordings/`. They are not needed to start the live GUI.

- `evaluate_independent_scene_geometry.py`: numerical comparison of estimated PLY and independently captured CARLA surfaces.
- `evaluate_ground_surface_candidates.py`: classify road/ground candidates and evaluate their semantic labels.
- `evaluate_recorded_scene_tracking.py`, `audit_saved_mot_detector.py`, `audit_geometric_observability.py`: diagnose vehicle tracking.
- `render_*.py`, `visualize_output.py`: scientific plots and visualizations.
- `summarize_lvimot_acceptance.py`, `verify_recorded_gt_boundary.py`: evaluate project acceptance and ground-truth isolation.
- `analyze_recorded_trace.py`, `export_carla_evaluation_reference.py`: debug data and road-only reference exports.

Run scripts from the project root using `python tools/<script>.py --help`. For simple map generation use `run_lvimot_gui.bat`. To obtain a comparison reference use `export_original_carla_map.bat`.
## FPS and measured road regression

Run `replay_fps_road.py --capture CAPTURE --output OUTPUT --device cuda:0 --road --render --async-preview` for the complete recorded estimator, disk-backed map and dashboard. Omit `--render` for core-only diagnosis. `--no-gt` removes evaluator labels from every core call. Outputs include per-frame state, track and stage traces, complete scene and road arrays, PLYs and scores. Choose an output folder outside the raw recording. This tool never connects to CARLA.

Use `evaluate_measured_road.py --capture CAPTURE --replay OUTPUT` when the original capture has `original_carla_map/CARLA_independent_360_reference_semantics.npz`. It independently checks predicted road points and coverage at 0.25/0.5/1m, after estimation. It does not change the original full-scene score.

Run automated regressions with `python -m pytest -q tests`. The 13 added safeguards cover exact voxel arithmetic, FGO cache lifetime, supported road geometry, unchanged measured heights, conservative abstention, complete model assets, and bounded spawned dashboard rendering. See the main FPS/road results document for measured runs and limits.

