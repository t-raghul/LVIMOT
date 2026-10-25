# GitHub publication changes

Prepared from the uploaded `LVIMOT_organized-final.zip` for public viewing.

- Added a clear public-facing README and GitHub upload instructions.
- Copied the uploaded final 197-frame `live_results.json` and standalone
  `offline_4d_map.html` to `examples/final_run/` without altering their data.
- Excluded `outputs_carla/` from the public repository; raw recordings and
  large PLYs were not present in the uploaded final ZIP.
- Moved local GUI settings to `lvimot_gui_config.example.json`, with a blank
  CARLA installation path; the real GUI config is ignored by Git.
- Removed a machine-specific CARLA installation path from GUI/CLI defaults,
  the Windows CARLA helper and two manual dataset inspection scripts.
- Excluded two archived pytest XML reports that included workstation names
  and paths; retained the summarized test text reports.
- Excluded Python bytecode caches, regenerated the file hash manifest and
  added `.gitattributes` and third-party model license notices.

The algorithm implementations, included pretrained model weights, recorded
final result data and original architecture image are otherwise unchanged.
No new live CARLA or Linux run was performed for this packaging step.

## Packaging checks in this environment

- The new ZIP passed CRC integrity verification and the file SHA-256 manifest
  matched every packaged file.
- All included Python source files parsed successfully.
- A local `python -m pytest -q` run reported **266 passed, 7 skipped,
  6 failed**. All six failures were import errors because this packaging
  environment does **not** have the `ultralytics` dependency installed.
  This is not a claim that the project is bug-free or that a full CARLA run
  was retested. See `requirements.txt` for that dependency.
