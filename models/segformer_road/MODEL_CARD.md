# Measured road layer model

Source: https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-768-768

Pinned revision: d3b7801ed329668d5bff04cd33365fa37f538c3b.

Architecture: SegFormer-B0, pretrained and fine-tuned externally on Cityscapes; no training on these LVIMOT captures.

License: NVIDIA Source Code License for SegFormer, research/evaluation only. The complete LICENSE is included.

The original weights were exported using Transformers 4.44.2 and CPU PyTorch 2.4.1 to a TorchScript logits module. The runtime needs PyTorch, NumPy and OpenCV. It does not need Transformers or an internet connection. The input is fixed at 256x832 RGB with ImageNet normalization. Full-precision logits matched the export example exactly. See model_manifest.json for source and artifact hashes.

Class 0 is road. Runtime selects pixels whose most likely class is road with probability >=0.70, erodes the boundary by one pixel, projects the current real LiDAR into that image, and rejects returns inconsistent with measured local surface support. All retained XYZ values stay as measured. A fitted plane only decides which points have support; it never changes their height. Steep or poorly supported surfaces cause abstention. Frames without current image/map updates do not fabricate new road observations.

This layer covers road seen jointly by the front camera and LiDAR. It does not cover all of the 360-degree scene. Sharp crests, slopes, darkness, domain differences and uncertain road boundaries can reduce recall or cause wrong labels. The full scene map remains available separately. Ground truth, actor poses and OpenDRIVE references are only used after estimation for evaluation.

On the supplied 59-frame capture, OpenDRIVE road-layer agreement is 90.96%. Against the separately recorded semantic point reference, 97.16% of retained points match a road-labelled reference within 0.5m; observed-reference road recall is 56.64%. These point-sample measurements do not certify full road coverage or a complete 3-D map.
