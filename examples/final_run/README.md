# Final-run example (197 frames)

These files are copied unchanged from the `outputs_carla/` directory in the uploaded
`LVIMOT_organized-final.zip` archive. They describe the **197-frame** final CARLA run.

- `live_results.json`: saved run metrics and diagnostics.
- `offline_4d_map.html`: standalone HTML viewer; open locally in a browser.

The raw sensor captures, complete JSON/PLY map exports, and an independently rendered
full-3D CARLA reference are **not present** in the uploaded archive, so they are
not fabricated or included here. The viewer is an example, not a guarantee of
independent full-scene acceptance.

**Do not confuse this example with the historical 105-frame validation described
in `docs/VALIDATION_20261022.md`.** That report covers a different run.

Selected values read from this `live_results.json`:

| Metric | Value |
| --- | ---: |
| Processed frames | 197 |
| Ground-truth object observations | 256 |
| Correct matched observations | 173 |
| False positives / false negatives / ID switches | 9 / 83 / 3 |
| MOTA | 62.89% |
| Localization ATE RMSE | 1.716 m |
| Localization RPE RMSE | 0.0553 m |
| Measured road-layer agreement (limited scope) | 91.38% |
| End-to-end processing rate | 1.10 FPS |

MOTA is an evaluation metric, not the percentage of distinct cars detected.
The road agreement covers the camera-observed measured LiDAR road layer and
**does not certify** the complete 3D map.
