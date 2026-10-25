# Third-party models and licensing

The LVIMOT source archive contains third-party pretrained model artifacts.
Their rights and restrictions are separate from the authorship of the LVIMOT
project's own source code.

- **SegFormer-B0**: `models/segformer_road/` includes the exported model,
  upstream metadata and the complete `LICENSE`. The included NVIDIA Source
  Code License for SegFormer limits use to **non-commercial research or
  evaluation** and requires preserving its notices. See
  `models/segformer_road/MODEL_CARD.md` for the source and revision.
- **YOLOv8n**: `models/yolov8n.pt` is a third-party pretrained model associated
  with Ultralytics. Check the applicable upstream model/code licensing and
  obligations before redistribution or reuse, especially commercial use.
  Upstream: https://github.com/ultralytics/ultralytics
- **CARLA**: CARLA 0.9.13 simulator binaries and assets are **not bundled**.
  Install CARLA separately and follow its licensing terms.

**No project-wide license is granted by this repository at this stage.**
Public GitHub visibility allows viewing and downloading, but does not by
itself grant permission to reuse, modify or redistribute the original project
source. The project owner should choose a source-code license deliberately,
after reviewing the third-party components. Do not assume the NVIDIA model
can be relicensed for unrestricted commercial use.
