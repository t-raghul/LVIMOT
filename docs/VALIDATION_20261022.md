# LVIMOT cleaned architecture and estimated ego map — 22 October 2026

Built from `LVIMOT_CURRENT_FIXED_20261022.zip`.
Input SHA256: `3c48d74190f83bedf8bd7eb8c692939c3d16a68faf06e940ee2e66070ee88073`.

## Delivered behavior

The offline HTML time slider updates a red ego heading marker and its past path
from accepted estimated poses. All new processed frames, including frames with
no objects or map integration, are recorded with frame ID, timestamp, position,
yaw and estimated velocity. SQLite keeps the complete route beyond live RAM
limits. Older runs recover only their saved `state_estimation` values from
`live_frames.jsonl`. CARLA GT is never a fallback for missing ego history.

Confirmed vehicle ID text is green. CARLA map, LVIMOT map, vehicles and ego have
independent checkboxes. The original independent 360 degree CARLA comparison,
map alignment safeguards, 3D/top views and full-scene map remain available.
Static map geometry represents the completed route at every slider time.

`examples/final_run/offline_4d_map.html` is regenerated from the supplied final
105-frame run. It embeds 105 estimated ego observations, 21,888 static samples,
12,158 measured road samples and 30,599 independent CARLA reference samples.
The source final JSON/PLY and metrics were read without modification. The example
JSON contains the recovered ego history; its PLY is a complete unchanged copy.

The old GT-derived live early-stop diagnosis is now evaluated after the live
loop. GT cannot change which sensor frames are processed through that check.
The diagnostic remains visible in the post-run report.

## Structure and removal audit

The six runtime sections are INPUTS, PREPROCESSING, 4D TRACKING, TEMPORAL FUSION,
4D MAPPING and OUTPUT. See `ARCHITECTURE.md` for algorithm-level paths and
`RENAMED_FILES.json` for all 67 path mappings. The source diagram is preserved;
its “SegFormer 80” label is corrected in the architecture documentation to
**SegFormer-B0**. YOLOv8n and its separate foreground-context model retain their
own identities. The packaged road model hash matches its pinned manifest.

| Python file category | Before | After |
| --- | ---: | ---: |
| Runtime source | 56 | 49 |
| Root scripts | 6 | 0 |
| Tools | 18 | 24 |
| Tests and manual diagnostics | 70 | 71 |
| **Total** | **150** | **144** |

Eight obsolete file boundaries were removed by six documented consolidations;
all original algorithms/classes/functions remain. A package entry point and
one regression-test file were added. All original non-cache files are accounted
for, and model files are byte-identical to the input. No functional source module
was discarded as unused. The input archive's **45 compiled cache files** were
omitted; newly generated caches/build/test temporary files are excluded as well.
The original README is retained as provenance in `README_INPUT_ARCHIVE.md`.
Detailed reasons and paths are in `CLEANUP_AUDIT_20261022.json`.

## Executed checks

- Python **3.8.10**: **276 passed, 3 skipped**, zero failures/errors.
- Python **3.11.8**: **276 passed, 3 skipped**, zero failures/errors.
- All **144 Python files** passed source compilation; Python 3.8 syntax and
  actual Python 3.8 compilation were checked.
- **142 isolated imports** passed: all 71 package/tool modules in both runtimes.
- **36 direct/module entry-point checks** passed from outside the project folder.
  These include CLI, GUI help, replayers, evaluators and manual IMU entry points.
  Windows launcher paths and startup behavior are covered by source regressions.
- **8 offline JavaScript behavior checks** passed using an executed viewer script:
  time range, forward/reverse marker movement, green IDs, independent controls,
  and empty-history behavior. These use a simulated canvas/DOM, not a browser
  screenshot or a claim of complete browser compatibility.
- Three complete **105-frame offline CPU replays** ran with the real local
  YOLOv8n and SegFormer-B0 artifacts, full static/road mapping, recording archives
  and the actual Windows dashboard worker process: original source with GT
  evaluation, cleaned source with GT evaluation, and cleaned source without GT.
- Original versus cleaned: **105/105 identical estimated states and published
  tracking frames**, maximum pose-component difference **0**. Rolling static/road
  arrays and their PLY files were byte-identical. Full accumulated static/road
  voxel records matched exactly. SQLite quick checks passed.
- Cleaned GT enabled versus disabled: the same exact state/track/map comparison
  passed for **105/105 frames**. Both new ego archives contain all 105 estimates,
  and their positions/velocities match the optimizer's saved states exactly.
- Original-protocol replay MOTA stayed **54.2683%** in original and cleaned runs.
  This is a preservation check on one recording, not a general accuracy claim.
- The independent semantic road evaluator ran on the cleaned replay output.
  Import/entry-point checks exposed and resolved its argument-parsing issue.

The complete evidence, replay traces, summaries and test XML are in
`validation_20261022/`. Different replay timings reflect cache/system conditions;
this release makes no new speed-improvement claim.

## Checks that could not establish live/platform behavior

Three legacy tests skip because the old CARLA sequence `0006` is absent:
`test_carla_camera_features`, `test_carla_complete`, and
`test_carla_multimodal_features`. Their source and manual IMU diagnostics remain.
The diagnostics requiring those datasets were not treated as passed.

No new live CARLA server run was performed. Sensor callbacks, newly spawned
actors, new rendered reference capture, GUI driving controls, CUDA execution,
long routes and long-duration resource behavior remain unvalidated for this
release. Existing recorded data was replayed completely without contacting
CARLA. GUI help/imports and the dashboard worker were exercised; a live Tk driving
session was not. Linux paths/launcher were updated and Python 3.8 compatibility
checked, but no Linux execution was available. No absence-of-bugs claim is made.

CARLA binaries, Python/CUDA environments and raw sensor recordings are external
to the source ZIP, as in the input archive. The included standalone final HTML
can be opened without them. The checksum manifest records SHA256 for every other packaged file, and the
outer verification receipt records ZIP CRC/hash verification and packaged smoke
results after creation.
