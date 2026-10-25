# Changes applied to uploaded LVIMOT.zip (2026-10-22)

This patch was applied to the actual `LVIMOT.zip` supplied by the user, not to an older project archive.

## Windows GUI startup

`run_lvimot_gui.bat` previously tried installing CUDA PyTorch and then imported PyTorch to test GPU availability before displaying the GUI. The launcher no longer downloads packages or imports PyTorch during normal startup. It still loads an already installed `.cuda_runtime` directory if present. To install the optional CUDA runtime manually, run `setup_cuda_runtime.bat` **once** while the GUI is closed. A missing `.venv_live38` still triggers first-time environment setup.

## GUI output decoding

The subprocess reader in `src/lvimot_live_gui.py` now requests UTF-8 with `errors='replace'` and sets `PYTHONIOENCODING=utf-8` / `PYTHONUTF8=1` for the backend. A Windows-native log byte such as `0x8f` can no longer abort the output reader due to `UnicodeDecodeError`. A replacement character in the display preserves subsequent lines.

## Offline 4D map exporter

The uploaded project contained an HTML exporter with colored CARLA/LVIMOT map toggles, but the live runner **did not call it**. `src/carla_live_main.py` now invokes `export_offline_4d_scene` once **after** the run's estimation and CARLA-only post-estimation reference check. Output files are saved in `outputs_carla/live/`: `offline_4d_map.html`, `offline_4d_map.ply`, `offline_4d_scene.json`, `offline_4d_map_manifest.json`. The result is included under `offline_4d_scene_latest` in `live_results.json`.

The `Carla4DEnvironmentMapper` now attaches and closes disk-backed sensor-only object histories and measured road geometry, so complete-route 4D export does not silently truncate older tracks/road segments due to rolling live-map eviction. `IndividualObjectScene.current(..., include_historical=True)` allows historical parked candidates to keep their last measured boxes while stale moving tracks retain paths without invented boxes. Default live/current behavior remains unchanged.

The CARLA reference layer is evaluation-only and is not an input to localization, MOT, FGO or map integration. If a separate 360-degree CARLA reference is captured **after** the HTML was generated, re-run `python tools/regenerate_offline_html.py --output-dir outputs_carla/live` to embed it. If there is no CARLA reference in the output directory, the CARLA toggle will have no points to show. The map colors are those already included in the user's uploaded archive.

## Checks

- Windows log-decoding subprocess regression test
- Static checks for launcher/GPU startup and live offline export wiring
- Colored HTML toggle + measured road/object archive tests
- Object scene and accumulated static-map tests

These tests are offline. They do not certify a full Windows/CARLA simulator run, solve existing localization/tracking metric failures, or establish a measured improvement in FPS. The runner still depends on CARLA 0.9.13, the Python 3.8 venv, and the project's installed model dependencies.
