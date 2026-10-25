from __future__ import annotations

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class CameraLiDARDetectionConfig:
    model_path: str = "models/yolov8n.pt"
    confidence: float = 0.12
    iou: float = 0.45
    image_size: int = 1280
    max_detections: int = 12
    min_lidar_points: int = 4
    max_range: float = 55.0
    device: str = "auto"
    half: Optional[bool] = False
    min_eval_bbox_width_px: float = 8.0
    min_eval_bbox_height_px: float = 8.0
    min_eval_bbox_area_px: float = 100.0
    small_object_tiles: bool = True
    tile_min_confidence: float = 0.25
    traffic_context_enabled: bool = True
    traffic_context_path: Optional[str] = None


class CarlaCameraLiDARObjectDetector:
    """GT-free camera + LiDAR traffic-object detector.

    A pretrained COCO detector supplies *semantic 2-D boxes* from the RGB
    camera.  Raw LiDAR points are projected into each box and a robust near
    depth cluster provides the 3-D measurement used by MOT.  No CARLA actor
    pose, actor ID, semantic LiDAR, or simulator ground truth is consumed.

    The Ultralytics dependency is imported lazily so the rest of LVIMOT and
    the unit tests do not require it.  A test double may be passed as ``model``.
    """

    # COCO vehicle-like classes used by yolov8n/YOLO11 COCO weights.
    COCO_CLASS_MAP = {
        1: "bicycle",
        2: "vehicle",      # car
        3: "motorcycle",
        5: "vehicle",      # bus
        7: "vehicle",      # truck
    }

    def __init__(
        self,
        calibration,
        config: Optional[CameraLiDARDetectionConfig] = None,
        model=None,
    ):
        self.calibration = calibration
        self.config = config or CameraLiDARDetectionConfig()
        self._model = model
        self._model_load_error: Optional[Exception] = None
        self.last_camera_detections: List[Dict] = []
        self.last_diagnostics: Dict = {}
        self._traffic_context = None
        self._traffic_context_checked = False

    def _filter_traffic_context(self, image, detections):
        if self.config.traffic_context_enabled and not self._traffic_context_checked:
            sidecar = (Path(self.config.traffic_context_path) if self.config.traffic_context_path
                       else Path(self.config.model_path).with_suffix('.traffic_context.npz'))
            if sidecar.is_file():
                from lvimot.tracking_4d.multimodal_object_tracking.traffic_context import CarlaTrafficContextScorer
                device, _, _ = self._resolve_runtime()
                self._traffic_context = CarlaTrafficContextScorer(sidecar, self.config.model_path, device=device)
            elif self.config.traffic_context_path:
                raise FileNotFoundError('Missing configured traffic-context classifier: '+str(sidecar))
            self._traffic_context_checked = True
        if not self.config.traffic_context_enabled or self._traffic_context is None:
            return detections
        kept = self._traffic_context.filter(image, detections)
        self.last_diagnostics.update({'traffic_context_active': True,
            'traffic_context_candidates': len(detections), 'traffic_context_rejections': len(detections)-len(kept)})
        return kept

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if self._model_load_error is not None:
            raise RuntimeError(self._dependency_error_message()) from self._model_load_error
        try:
            from ultralytics import YOLO
            from lvimot.output.compute_runtime import configure_detector_threads
            configure_detector_threads()
            self._model = YOLO(self.config.model_path)
            return self._model
        except Exception as exc:  # import error, missing model, download failure, etc.
            self._model_load_error = exc
            raise RuntimeError(self._dependency_error_message()) from exc

    def _dependency_error_message(self) -> str:
        return (
            "The GT-free MOT detector requires Ultralytics YOLO. Install it with: "
            "python -m pip install ultralytics ; then rerun. "
            f"Model: {self.config.model_path!r}. The model is downloaded once by Ultralytics "
            "if it is not already cached."
        )

    @staticmethod
    def _as_numpy(value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def _resolve_runtime(self):
        requested = str(self.config.device).strip().lower()
        cuda_available = False
        try:
            import torch
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False

        if requested in {"", "auto", "none"}:
            device = 0 if cuda_available else "cpu"
        elif requested in {"cuda", "cuda:0", "gpu", "0"}:
            device = 0 if cuda_available else "cpu"
        else:
            device = self.config.device

        if self.config.half is None:
            half = bool(cuda_available and device != "cpu")
        else:
            half = bool(self.config.half and cuda_available and device != "cpu")
        return device, half, cuda_available

    def _infer_boxes(self, image: np.ndarray) -> List[Dict]:
        """Keep full-frame boxes and recover small objects in a wide image.

        Resizing an 828x250 camera frame to width 640 removes useful pixels
        from distant vehicles. Three overlapping half-width views let the
        SAME model inspect them at a larger scale. Full-frame boxes win every
        overlap, so a crop cannot replace an established semantic observation.
        The extra boxes still need current-frame measured LiDAR and normal
        tracker confirmation; a tile never supplies or predicts a depth.
        """
        primary = self._infer_single_view(image)
        self.last_diagnostics.update({
            "camera_full_frame_detections": len(primary),
            "camera_tile_views": 0,
            "camera_tile_additions": 0,
        })
        height, width = image.shape[:2]
        # max_detections remains the final 3-D proposal budget. Give semantic
        # recovery its own equally sized allowance: background RGB boxes can
        # fill the full-frame budget and then fail the physical LiDAR gates.
        # Dropping tile boxes before those gates would lose genuine vehicles.
        limit = 2 * int(self.config.max_detections)
        if (not self.config.small_object_tiles or height <= 0
                or width < 2 * height or width <= int(self.config.image_size)):
            return self._filter_traffic_context(image, primary)

        tile_width = width // 2
        extra = []
        confidence_floor = max(float(self.config.confidence), float(self.config.tile_min_confidence))
        for left in (0, width // 4, width - tile_width):
            tile = np.ascontiguousarray(image[:, left:left + tile_width])
            self.last_diagnostics["camera_tile_views"] += 1
            for detection in self._infer_single_view(tile):
                if float(detection["camera_confidence"]) < confidence_floor:
                    continue
                item = dict(detection)
                x1, y1, x2, y2 = item["bbox"]
                item["bbox"] = [x1 + left, y1, x2 + left, y2]
                polygon = item.get("segmentation_polygon")
                if polygon is not None:
                    shifted = np.asarray(polygon, dtype=np.float64).copy()
                    shifted[:, 0] += left
                    item["segmentation_polygon"] = shifted.tolist()
                if any(self._bbox_iou(item["bbox"], old["bbox"]) >= 0.25 for old in primary):
                    continue
                extra.append(item)

        merged = list(primary)
        extra.sort(key=lambda d: -float(d["camera_confidence"]))
        for item in extra:
            if len(merged) >= limit:
                break
            if any(self._bbox_iou(item["bbox"], old["bbox"]) >= 0.25 for old in merged):
                continue
            merged.append(item)
        self.last_diagnostics["camera_tile_additions"] = len(merged) - len(primary)
        return self._filter_traffic_context(image, merged)

    def _infer_single_view(self, image: np.ndarray) -> List[Dict]:
        model = self._ensure_model()
        device, half, cuda_available = self._resolve_runtime()
        kwargs = {
            "source": image,
            "conf": float(self.config.confidence),
            "iou": float(self.config.iou),
            "imgsz": int(self.config.image_size),
            "verbose": False,
            "device": device,
            "half": half,
        }
        self.last_diagnostics.update({
            "inference_device": str(device),
            "cuda_available": bool(cuda_available),
            "fp16": bool(half),
        })
        results = model.predict(**kwargs)
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = self._as_numpy(getattr(boxes, "xyxy", None))
        cls = self._as_numpy(getattr(boxes, "cls", None))
        conf = self._as_numpy(getattr(boxes, "conf", None))
        if xyxy is None or cls is None or conf is None:
            return []
        xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
        cls = np.asarray(cls, dtype=np.int64).reshape(-1)
        conf = np.asarray(conf, dtype=np.float64).reshape(-1)

        masks = getattr(result, 'masks', None)
        polygons = masks.xy if masks is not None else None
        detections = []
        names = getattr(model, 'names', None)
        semantic_classes = {'vehicle': 'vehicle', 'car': 'vehicle', 'bus': 'vehicle',
                            'truck': 'vehicle', 'bicycle': 'bicycle', 'motorcycle': 'motorcycle'}
        for detection_index, (bbox, class_id, score) in enumerate(zip(xyxy, cls, conf)):
            if names is None:
                mapped = self.COCO_CLASS_MAP.get(int(class_id))
            else:
                name = (names.get(int(class_id), '') if isinstance(names, dict)
                        else names[int(class_id)] if 0 <= int(class_id) < len(names) else '')
                mapped = semantic_classes.get(str(name).strip().lower())
            if mapped is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "class_id": int(class_id),
                    "class": mapped,
                    "type": mapped,
                    "camera_confidence": float(score),
                }
            )
            if polygons is not None and detection_index < len(polygons):
                polygon = np.asarray(polygons[detection_index], dtype=np.float64).reshape(-1, 2)
                if len(polygon) >= 3 and np.isfinite(polygon).all():
                    detections[-1]['segmentation_polygon'] = polygon.tolist()
        detections.sort(key=lambda d: -d["camera_confidence"])
        return detections[: int(self.config.max_detections)]

    @staticmethod
    def _depth_cluster(depths: np.ndarray, indices: np.ndarray, bin_size: float = 0.75) -> np.ndarray:
        """Return indices belonging to the most plausible foreground depth mode."""
        if len(indices) == 0:
            return indices
        d = np.asarray(depths[indices], dtype=np.float64)
        finite = np.isfinite(d) & (d > 0.1)
        if not np.any(finite):
            return np.empty(0, dtype=np.int64)
        indices = indices[finite]
        d = d[finite]
        if len(indices) <= 4:
            # With only four returns, a single wall/background outlier can
            # corrupt the entire 3-D center. Keep three mutually close
            # depths when possible; otherwise refuse the ambiguous cluster.
            order = np.argsort(d)
            sorted_d = d[order]
            if len(indices) < 3:
                return np.empty(0, dtype=np.int64)
            spans = sorted_d[2:] - sorted_d[:-2]
            best = int(np.argmin(spans))
            if float(spans[best]) > 2.8:
                return np.empty(0, dtype=np.int64)
            # Four consistent returns are retained; three are allowed only
            # when they form a coherent depth cluster.
            if len(indices) == 4 and float(sorted_d[-1]-sorted_d[0]) <= 2.8:
                return indices
            return indices[order[best:best+3]]

        # A true distant vehicle may provide only three tightly spaced LiDAR
        # returns, while a wall behind its image box contributes many.  Recover
        # those three real foreground returns ONLY when their depths agree and
        # they are separated from the remainder by an unambiguous gap.  Never
        # infer a depth from RGB alone or accept a two-return foreground.
        order = np.argsort(d)
        sorted_d = d[order]
        # Recover any compact, measured foreground layer, rather than only
        # the special case of exactly three returns. A 4-7-return vehicle can
        # otherwise lose to a denser wall, followed by an abstention despite
        # the vehicle's coherent depth being present in the current scan.
        gaps = np.flatnonzero(np.diff(sorted_d) >= 2.8) + 1
        if len(gaps):
            first_end = int(gaps[0])
            span_limit = 1.35 if first_end == 3 else 2.8
            if first_end >= 3 and float(sorted_d[first_end - 1] - sorted_d[0]) <= span_limit:
                return indices[order[:first_end]]
        if len(sorted_d) >= 5:
            foreground_span = float(sorted_d[2] - sorted_d[0])
            separation = float(sorted_d[3] - sorted_d[2])
            if foreground_span <= 1.35 and separation >= 2.8:
                return indices[order[:3]]

        bins = np.floor(d / float(bin_size)).astype(np.int32)
        unique, counts = np.unique(bins, return_counts=True)
        # Favor dense *nearer* modes. This suppresses wall/building points
        # behind a vehicle that happen to fall inside its 2-D box.
        centers = (unique.astype(np.float64) + 0.5) * float(bin_size)
        score = counts.astype(np.float64) / np.sqrt(np.maximum(centers, 1.0))
        # Discard one-point modes unless there is nothing better.
        score = np.where(counts >= 2, score, score * 0.15)
        best_bin = int(unique[int(np.argmax(score))])
        best_center = (best_bin + 0.5) * float(bin_size)
        # Keep a physically useful depth thickness around the visible surface.
        keep = np.abs(d - best_center) <= 2.25
        chosen = indices[keep]
        if len(chosen) < 3:
            order = np.argsort(d)
            sorted_d = d[order]
            spans = sorted_d[2:] - sorted_d[:-2]
            best = int(np.argmin(spans))
            if float(spans[best]) > 2.8:
                return np.empty(0, dtype=np.int64)
            chosen = indices[order[best:best + 3]]
        # Limit broad depth mixtures within the selected mode.  Untrimmed
        # background geometry creates unstable 3-D centres and ghost tracks.
        if len(chosen) > 4 and float(np.ptp(depths[chosen])) > 3.0:
            center_depth = float(np.median(depths[chosen]))
            compact = chosen[np.abs(depths[chosen] - center_depth) <= 1.5]
            if len(compact) >= 3:
                chosen = compact

        # A dense wall behind a sparse vehicle must not become that vehicle's
        # metric location. Two mutually close nearer returns establish an
        # unresolved foreground: abstain rather than substituting background
        # depth. An already anchored track may still use its guarded RGB bridge.
        closest_chosen = float(np.min(depths[chosen]))
        nearer = np.sort(d[d < closest_chosen - 2.8])
        if len(nearer) >= 2 and bool(np.any(np.diff(nearer) <= 1.5)):
            return np.empty(0, dtype=np.int64)
        return chosen

    @staticmethod
    def _robust_box(points: np.ndarray):
        pts = np.asarray(points, dtype=np.float64)
        center = np.median(pts, axis=0)
        p05 = np.percentile(pts, 5.0, axis=0)
        p95 = np.percentile(pts, 95.0, axis=0)
        span = np.maximum(p95 - p05, np.array([0.3, 0.3, 0.3]))
        # Raw CARLA LiDAR axes are X forward, Y right, Z up.  Partial visible
        # surfaces under-estimate vehicle extent, so only use dimensions for
        # tracker shape consistency, not for evaluation.
        length = float(max(span[0], span[1]))
        width = float(min(span[0], span[1]))
        height = float(span[2])
        return center, length, width, height, p05, p95

    @staticmethod
    def _belongs_to_extended_facade(center, cloud, tree):
        """Reject a proposed surface lying on a much larger vertical wall.

        RGB windows can resemble tiny cars. A local planar facade continuing
        more than 13.5 m sideways and 3 m vertically is incompatible with the
        bounded vehicle shapes used by this detector. An offset real vehicle
        is retained even when its neighborhood also contains a building.
        """
        nearby = tree.query_ball_point(center, 9.0)
        if len(nearby) < 24:
            return False
        points = cloud[nearby]
        origin = np.median(points, axis=0)
        centered = points - origin
        _, vectors = np.linalg.eigh(centered.T @ centered / len(points))
        normal = vectors[:, 0]
        if abs(float(normal[2])) > .35:
            return False
        residuals = np.abs(centered @ normal)
        plane = points[residuals <= .35]
        if len(plane) < 24 or len(plane) < .65 * len(points):
            return False
        if abs(float((np.asarray(center) - origin) @ normal)) > .35:
            return False
        horizontal = np.array([-normal[1], normal[0], 0.0])
        horizontal /= max(float(np.linalg.norm(horizontal)), 1e-9)
        across = plane @ horizontal
        width = float(np.percentile(across, 98) - np.percentile(across, 2))
        height = float(np.percentile(plane[:, 2], 98) - np.percentile(plane[:, 2], 2))
        return width >= 13.5 and height >= 3.0

    @staticmethod
    def _belongs_to_extended_ground(center, cloud, tree):
        """Reject a ground patch masquerading as a vehicle depth surface.

        The candidate must lie nearly on a broad horizontal plane; a real
        vehicle standing above the road is not rejected just because its
        neighborhood contains ground returns.  No semantic CARLA data used.
        """
        nearby = tree.query_ball_point(center, 5.0)
        if len(nearby) < 36:
            return False
        points = np.asarray(cloud[nearby], dtype=np.float64)
        origin = np.median(points, axis=0)
        centered = points - origin
        covariance = centered.T @ centered / max(1, len(points))
        _, basis = np.linalg.eigh(covariance)
        normal = basis[:, 0]
        if abs(float(normal[2])) < 0.93:
            return False
        residuals = np.abs(centered @ normal)
        plane = points[residuals <= 0.18]
        if len(plane) < 36 or len(plane) < 0.70 * len(points):
            return False
        if abs(float((np.asarray(center) - origin) @ normal)) > 0.18:
            return False
        ranges = np.percentile(plane[:, :2], 95, axis=0) - np.percentile(plane[:, :2], 5, axis=0)
        return bool(min(ranges) >= 3.5)

    @staticmethod
    def _bbox_iou(a, b) -> float:
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        if len(a) != 4 or len(b) != 4:
            return 0.0
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        return float(inter / union) if union > 0.0 else 0.0

    def _suppress_duplicate_proposals(self, proposals: List[Dict]) -> List[Dict]:
        ranked = sorted(
            proposals,
            key=lambda p: (float(p.get("score", 0.0)), int(p.get("camera_support", 0))),
            reverse=True,
        )
        kept: List[Dict] = []
        for proposal in ranked:
            p = np.asarray(proposal.get("location", [0.0, 0.0, 0.0]), dtype=np.float64)
            duplicate = False
            for other in kept:
                q = np.asarray(other.get("location", [0.0, 0.0, 0.0]), dtype=np.float64)
                if self._bbox_iou(proposal.get("bbox"), other.get("bbox")) >= 0.80 and float(np.linalg.norm(p - q)) <= 4.0:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(proposal)
        return kept

    def detect(self, image: np.ndarray, lidar_points: np.ndarray) -> List[Dict]:
        self.last_diagnostics = {
            "camera_detections": 0,
            "max_range_m": float(self.config.max_range),
            "raw_lidar_points": int(len(lidar_points)) if hasattr(lidar_points, "__len__") else 0,
            "projection_valid_points": 0,
            "points_within_range": 0,
            "proposals": 0,
        }
        detections = self._infer_boxes(image)
        self.last_camera_detections = [dict(d) for d in detections]
        self.last_diagnostics["camera_detections"] = int(len(detections))
        if not detections:
            return []

        pts = np.asarray(lidar_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
            return []
        xyz = pts[:, :3]
        finite_xyz = np.isfinite(xyz).all(axis=1)

        camera_xyz = self.calibration.lidar_to_opencv(xyz)
        depth = camera_xyz[:, 2]
        uv, valid_projection = self.calibration.project_lidar(xyz)
        valid_projection_mask = finite_xyz & valid_projection & np.isfinite(uv).all(axis=1)
        self.last_diagnostics["projection_valid_points"] = int(np.count_nonzero(valid_projection_mask))
        valid = valid_projection_mask & (depth > 0.1) & (depth <= float(self.config.max_range))
        self.last_diagnostics["points_within_range"] = int(np.count_nonzero(valid))
        from scipy.spatial import cKDTree
        facade_cloud = xyz[finite_xyz]
        facade_tree = cKDTree(facade_cloud)
        self.last_diagnostics['extended_facade_rejections'] = 0
        self.last_diagnostics['extended_ground_rejections'] = 0
        self.last_diagnostics['sparse_foreground_proposals'] = 0

        proposals: List[Dict] = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            bw = max(x2 - x1, 1.0)
            bh = max(y2 - y1, 1.0)

            # A pretrained RGB instance mask identifies visible object pixels.
            # It is never a depth measurement: every retained 3-D point still
            # comes from this frame's actual LiDAR. The mask prevents a wall
            # inside the surrounding image rectangle becoming the vehicle.
            polygon = det.get('segmentation_polygon')
            instance_support = None
            if polygon is not None:
                import cv2
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [np.rint(np.asarray(polygon)).astype(np.int32)], 1)
                pixels = np.rint(np.nan_to_num(uv, nan=-1e6, posinf=-1e6, neginf=-1e6)).astype(np.int64)
                in_image = (pixels[:, 0] >= 0) & (pixels[:, 0] < mask.shape[1]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < mask.shape[0])
                instance_support = np.zeros(len(xyz), dtype=bool)
                candidates = np.flatnonzero(valid & in_image)
                instance_support[candidates] = mask[pixels[candidates, 1], pixels[candidates, 0]] != 0

            # Use a central region to estimate foreground depth and avoid road
            # pixels at the very bottom of a 2-D box.
            cx1 = x1 + 0.08 * bw
            cx2 = x2 - 0.08 * bw
            cy1 = y1 + 0.08 * bh
            cy2 = y2 - 0.18 * bh
            central = (
                valid
                & (uv[:, 0] >= cx1)
                & (uv[:, 0] <= cx2)
                & (uv[:, 1] >= cy1)
                & (uv[:, 1] <= cy2)
            )
            idx = np.flatnonzero(central)
            if instance_support is not None:
                # A segmentation boundary can miss one of a real vehicle's
                # LiDAR returns.  Allow a third point from the central box only
                # if TWO mask points already agree in depth and the third has
                # essentially the same depth.  This does not substitute wall
                # points at a different depth for a vehicle.
                masked = np.flatnonzero(instance_support)
                idx = masked
                if len(masked) == 2 and abs(float(depth[masked[0]] - depth[masked[1]])) <= 0.9:
                    pool = np.flatnonzero(central & ~instance_support)
                    near = pool[np.abs(depth[pool] - float(np.median(depth[masked]))) <= 0.9]
                    if len(near):
                        xy_mid = np.mean(uv[masked], axis=0)
                        candidate = near[int(np.argmin(np.linalg.norm(uv[near] - xy_mid, axis=1)))]
                        idx = np.r_[masked, candidate]
            elif len(idx) < self.config.min_lidar_points:
                full = (
                    valid
                    & (uv[:, 0] >= x1)
                    & (uv[:, 0] <= x2)
                    & (uv[:, 1] >= y1)
                    & (uv[:, 1] <= y2)
                )
                idx = np.flatnonzero(full)
            if len(idx) < self.config.min_lidar_points:
                # A tightly projected, high-confidence distant vehicle can
                # have three LiDAR returns. Do not hallucinate depth from a
                # camera-only box; require all three real measured returns.
                if len(idx) < 3 or float(det["camera_confidence"]) < 0.25:
                    continue

            selected_idx = self._depth_cluster(depth, idx)
            if len(selected_idx) < self.config.min_lidar_points:
                if (len(selected_idx) < 3
                        or float(det["camera_confidence"]) < 0.25
                        or float(np.ptp(depth[selected_idx])) > 2.8):
                    continue
            selected = xyz[selected_idx]
            azimuth_span = float(np.ptp(np.arctan2(selected[:, 1], selected[:, 0])))
            center, length, width, height, pmin, pmax = self._robust_box(selected)
            if self._belongs_to_extended_facade(center, facade_cloud, facade_tree):
                self.last_diagnostics['extended_facade_rejections'] += 1
                continue
            if self._belongs_to_extended_ground(center, facade_cloud, facade_tree):
                self.last_diagnostics['extended_ground_rejections'] += 1
                continue
            if len(selected_idx) == 3:
                self.last_diagnostics['sparse_foreground_proposals'] += 1
            cls_name = det["class"]
            if cls_name == "motorcycle":
                length = float(np.clip(max(length, 1.6), 1.6, 3.2))
                width = float(np.clip(max(width, 0.55), 0.55, 1.2))
                height = float(np.clip(max(height, 1.0), 1.0, 2.2))
            elif cls_name == "bicycle":
                length = float(np.clip(max(length, 1.4), 1.4, 2.4))
                width = float(np.clip(max(width, 0.45), 0.45, 1.0))
                height = float(np.clip(max(height, 1.0), 1.0, 2.0))
            else:
                # Camera semantics make the dimensions a tracker stabilizer rather
                # than an object classifier.  Prevent sparse visible surfaces from
                # producing wildly different sizes on successive frames.
                length = float(np.clip(max(length, 3.2), 3.2, 9.0))
                width = float(np.clip(max(width, 1.4), 1.4, 3.2))
                height = float(np.clip(max(height, 1.2), 1.2, 3.8))

            r_xy = float(np.linalg.norm(center[:2]))
            if not np.isfinite(r_xy) or r_xy < 1.0 or r_xy > float(self.config.max_range):
                continue

            point_support = float(np.clip(len(selected_idx) / 18.0, 0.0, 1.0))
            # V19 recall repair: distant CARLA vehicles can have weak COCO
            # confidence while still carrying strong, geometrically coherent
            # LiDAR support.  Let LiDAR contribute enough to preserve those
            # candidates; temporal confirmation in the tracker suppresses noise.
            score = float(np.clip(0.55 * det["camera_confidence"] + 0.45 * point_support, 0.0, 1.0))
            proposals.append(
                {
                    "location": center,
                    "dimensions": {"length": length, "width": width, "height": height},
                    "rotation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                    "type": det["type"],
                    "class": det["class"],
                    "score": score,
                    "camera_confidence": float(det["camera_confidence"]),
                    "camera_evidence": True,
                    "camera_support": int(len(selected_idx)),
                    "num_points": int(len(selected_idx)),
                    "sparse_depth_measured": bool(len(selected_idx) == 3),
                    'depth_azimuth_span_rad': azimuth_span,
                    'depth_m': float(np.median(depth[selected_idx])),
                    'depth_spread_m': float(np.ptp(depth[selected_idx])),
                    'depth_quality': ('SPARSE_MEASURED' if len(selected_idx) < self.config.min_lidar_points else 'MEASURED'),
                    'depth_pixel_support': 'RGB_INSTANCE_MASK' if instance_support is not None else 'RGB_BOX_REGION',
                    "bbox": list(det["bbox"]),
                    "min_corner": pmin,
                    "max_corner": pmax,
                }
            )

        proposals = self._suppress_duplicate_proposals(proposals)
        proposals.sort(key=lambda p: -float(p.get("score", 0.0)))
        proposals = proposals[: int(self.config.max_detections)]
        self.last_diagnostics["proposals"] = int(len(proposals))
        return proposals
