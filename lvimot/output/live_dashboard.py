from __future__ import annotations

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


import json
import math
import os
from pathlib import Path
from typing import Dict

import numpy as np

from lvimot.output.map_export import cleaned_layers_and_outlines
from lvimot.mapping_4d.multiobject_trajectory_management import objects_to_local

try:
    import cv2
except ImportError:
    cv2 = None


class CarlaLiveVisualizer:
    """Processed live camera + cleaned rolling spatial 4-D-map visualizer.

    Important separation:
    - The estimator/tracker/map are not modified here.
    - This class only visualizes estimator outputs.
    - Short display-only track persistence is labelled PREDICTED and never fed
      back into MOT or evaluation.
    - The dashboard preview files are display outputs only.
    """

    def __init__(
        self,
        bev_size=620,
        meters=60.0,
        max_snapshot_points=5000,
        map_cell_size_m=0.35,
        display_track_hold_frames=5,
        display_candidate_hold_frames=None,
        preview_width=960,
    ):
        self.bev_size = int(bev_size)
        self.meters = float(meters)
        self.max_snapshot_points = max(800, int(max_snapshot_points))
        self.map_cell_size_m = float(np.clip(map_cell_size_m, 0.15, 1.0))
        self.display_track_hold_frames = max(0, int(display_track_hold_frames))
        if display_candidate_hold_frames is None:
            display_candidate_hold_frames = os.environ.get(
                "LVIMOT_CANDIDATE_HOLD_FRAMES", "4"
            )
        try:
            display_candidate_hold_frames = int(display_candidate_hold_frames)
        except Exception:
            display_candidate_hold_frames = 4
        self.display_candidate_hold_frames = max(0, min(12, display_candidate_hold_frames))
        self.preview_width = max(480, int(preview_width))

        map_env = os.environ.get("LVIMOT_LIVE_MAP_SNAPSHOT", "").strip()
        if map_env:
            self.snapshot_path = Path(map_env)
        else:
            self.snapshot_path = Path("outputs_carla") / "live" / "live_map_snapshot.json"
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        camera_env = os.environ.get("LVIMOT_LIVE_CAMERA_SNAPSHOT", "").strip()
        if camera_env:
            self.camera_snapshot_path = Path(camera_env)
        else:
            self.camera_snapshot_path = Path("outputs_carla") / "live" / "live_camera_preview.png"
        self.camera_snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        # Display-only cache.  It prevents the camera overlay from blinking off
        # immediately when one detector frame is missed.  These cached boxes do
        # NOT become tracker outputs; they are drawn amber as PREDICTED.
        self._display_track_cache: Dict[int, Dict] = {}

        # Display-only fused-candidate cache.  The actual detector/fusion output
        # is untouched.  A candidate that disappears for only a few frames is
        # retained as CAND HOLD so the live frontend visualization does not
        # flicker, especially for distant/static vehicles with sparse LiDAR.
        self._display_candidate_cache: Dict[int, Dict] = {}
        self._next_display_candidate_id = 1

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _pose(result):
        est = result.get("state_estimation", {}) or {}
        pose = np.asarray(est.get("pose", np.zeros(6)), dtype=np.float64).reshape(-1)
        if pose.size < 6:
            padded = np.zeros(6, dtype=np.float64)
            padded[: pose.size] = pose
            pose = padded
        return pose

    @staticmethod
    def _velocity(result):
        est = result.get("state_estimation", {}) or {}
        vel = np.asarray(est.get("velocity", np.zeros(3)), dtype=np.float64).reshape(-1)
        if vel.size < 3:
            padded = np.zeros(3, dtype=np.float64)
            padded[: vel.size] = vel
            vel = padded
        return vel

    @staticmethod
    def _world_to_heading_up(points_xy, ego_x, ego_y, yaw):
        pts = np.asarray(points_xy, dtype=np.float64)
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        dx = pts[:, 0] - float(ego_x)
        dy = pts[:, 1] - float(ego_y)
        c = math.cos(float(yaw))
        s = math.sin(float(yaw))
        forward = c * dx + s * dy
        lateral = -s * dx + c * dy
        return np.column_stack([forward, lateral])

    # ------------------------------------------------------------------
    # Clean map rendering
    # ------------------------------------------------------------------

    def _bounded_local_map(self, map_points, pose):
        pts = np.asarray(map_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
            return np.empty((0, 3), dtype=np.float64)

        local_xy = self._world_to_heading_up(pts[:, :2], pose[3], pose[4], pose[2])
        z_rel = pts[:, 2] - pose[5]
        dist2 = np.einsum("ij,ij->i", local_xy, local_xy)
        keep = (
            np.isfinite(local_xy).all(axis=1)
            & np.isfinite(z_rel)
            & (dist2 <= self.meters * self.meters)
            & (z_rel >= -4.0)
            & (z_rel <= 6.0)
        )
        return np.column_stack([local_xy[keep], z_rel[keep]])

    @staticmethod
    def _estimate_ground_height(local):
        if len(local) < 30:
            return -1.0
        z = np.asarray(local[:, 2], dtype=np.float64)
        z = z[np.isfinite(z) & (z >= -4.0) & (z <= 2.0)]
        if len(z) < 20:
            return float(np.median(local[:, 2])) if len(local) else -1.0
        bins = np.arange(-4.0, 2.05, 0.15)
        hist, edges = np.histogram(z, bins=bins)
        idx = int(np.argmax(hist))
        return float(0.5 * (edges[idx] + edges[idx + 1]))

    def _clean_map_cells(self, local):
        """Convert noisy 3-D points into a cleaner 2-D local occupancy map.

        The underlying mapper is unchanged.  For display only, points are
        collapsed into 35 cm XY cells, isolated single-cell speckle is removed,
        and cells are split into ground/road versus vertical structure.
        """
        pts = np.asarray(local, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
            empty = np.empty((0, 3), dtype=np.float64)
            return empty, empty, {"ground_z": None, "raw_points": 0, "clean_cells": 0}

        cell = self.map_cell_size_m
        ix = np.floor((pts[:, 0] + self.meters) / cell).astype(np.int32)
        iy = np.floor((pts[:, 1] + self.meters) / cell).astype(np.int32)
        grid_n = int(math.ceil((2.0 * self.meters) / cell)) + 2
        valid = (ix >= 0) & (ix < grid_n) & (iy >= 0) & (iy < grid_n)
        pts = pts[valid]
        ix = ix[valid]
        iy = iy[valid]
        if len(pts) == 0:
            empty = np.empty((0, 3), dtype=np.float64)
            return empty, empty, {"ground_z": None, "raw_points": 0, "clean_cells": 0}

        keys = ix.astype(np.int64) * int(grid_n) + iy.astype(np.int64)
        unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        n_cells = len(unique_keys)

        # Mean height in each XY cell.  Forward/lateral use the cell centre so
        # nearby 3-D voxels appear as a clean surface instead of random speckle.
        sum_z = np.bincount(inverse, weights=pts[:, 2], minlength=n_cells)
        mean_z = sum_z / np.maximum(counts, 1)
        ux = (unique_keys // int(grid_n)).astype(np.int32)
        uy = (unique_keys % int(grid_n)).astype(np.int32)
        forward = (ux.astype(np.float64) + 0.5) * cell - self.meters
        lateral = (uy.astype(np.float64) + 0.5) * cell - self.meters

        # Count the same eight occupied neighbours in array operations.
        # This avoids eight Python hash lookups for every displayed cell.
        lo_x, lo_y = int(ux.min()), int(uy.min())
        nx, ny = int(ux.max())-lo_x+1, int(uy.max())-lo_y+1
        if nx * ny <= 2000000:
            occupancy = np.zeros((nx+2, ny+2), dtype=np.uint8)
            ax, ay = ux-lo_x+1, uy-lo_y+1
            occupancy[ax, ay] = 1
            counts_grid = np.zeros((nx, ny), dtype=np.int16)
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    if da or db:
                        counts_grid += occupancy[1+da:1+da+nx, 1+db:1+db+ny]
            neighbour_count = counts_grid[ax-1, ay-1]
        else:
            occupied = set(zip(ux.tolist(), uy.tolist()))
            neighbour_count = np.asarray([sum((a+da,b+db) in occupied
                for da in (-1,0,1) for db in (-1,0,1) if da or db)
                for a,b in zip(ux.tolist(),uy.tolist())], dtype=np.int16)

        ground_z = self._estimate_ground_height(pts)
        is_ground = np.abs(mean_z - ground_z) <= 0.32

        # Ground needs spatial support; structures may be thin (poles, walls),
        # so repeated vertical occupancy is also enough to keep them.
        keep_ground = is_ground & (neighbour_count >= 2)
        keep_structure = (~is_ground) & (
            (counts >= 2) | (neighbour_count >= 2)
        )

        ground = np.column_stack([forward[keep_ground], lateral[keep_ground], mean_z[keep_ground]])
        structure = np.column_stack(
            [forward[keep_structure], lateral[keep_structure], mean_z[keep_structure]]
        )

        # Conservative fallback for unusually sparse maps.
        if len(ground) + len(structure) < min(120, max(20, n_cells // 20)):
            keep = neighbour_count >= 1
            fallback = np.column_stack([forward[keep], lateral[keep], mean_z[keep]])
            gmask = np.abs(fallback[:, 2] - ground_z) <= 0.32 if len(fallback) else np.zeros(0, dtype=bool)
            ground = fallback[gmask]
            structure = fallback[~gmask]

        # Do not decimate BEFORE finding connected outlines. Removing regular
        # cells here would create artificial gaps and ragged contour lines.
        # The final dashboard point payload is bounded after contours are made.

        stats = {
            "ground_z": float(ground_z),
            "raw_points": int(len(pts)),
            "clean_cells": int(len(ground) + len(structure)),
            "ground_cells": int(len(ground)),
            "structure_cells": int(len(structure)),
            "cell_size_m": float(cell),
        }
        return ground, structure, stats

    # ------------------------------------------------------------------
    # Track display continuity
    # ------------------------------------------------------------------

    @staticmethod
    def _valid_bbox(box):
        if box is None or len(box) != 4:
            return None
        try:
            values = [float(v) for v in box]
        except Exception:
            return None
        if not np.isfinite(values).all():
            return None
        x1, y1, x2, y2 = values
        if x2 <= x1 or y2 <= y1:
            return None
        return values


    @staticmethod
    def _bbox_iou(a, b):
        if a is None or b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return float(inter / union) if union > 1e-9 else 0.0

    @staticmethod
    def _bbox_center_distance(a, b):
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        acx, acy = 0.5 * (ax1 + ax2), 0.5 * (ay1 + ay2)
        bcx, bcy = 0.5 * (bx1 + bx2), 0.5 * (by1 + by2)
        return math.hypot(acx - bcx, acy - bcy)

    @staticmethod
    def _draw_dashed_rect(image, box, color, thickness=1, dash=8, gap=5):
        if cv2 is None:
            return
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        for x in range(x1, x2, dash + gap):
            cv2.line(image, (x, y1), (min(x + dash, x2), y1), color, thickness)
            cv2.line(image, (x, y2), (min(x + dash, x2), y2), color, thickness)
        for y in range(y1, y2, dash + gap):
            cv2.line(image, (x1, y), (x1, min(y + dash, y2)), color, thickness)
            cv2.line(image, (x2, y), (x2, min(y + dash, y2)), color, thickness)

    def _update_display_candidate_cache(self, result):
        frame = int(result.get("frame", 0) or 0)
        current = []
        for obj in result.get("multimodal_objects", []) or []:
            box = self._valid_bbox(obj.get("projected_bbox"))
            if box is None:
                continue
            current.append({
                "bbox": box,
                "class": obj.get("class", obj.get("label", "candidate")),
            })

        cache_ids = list(self._display_candidate_cache.keys())
        unmatched_cache = set(cache_ids)
        matched_current = set()

        # Greedy association for display only.  Strong IoU wins; otherwise use
        # a conservative centre-distance gate scaled by box size.
        pairs = []
        for ci, cand in enumerate(current):
            cb = cand["bbox"]
            cw = max(1.0, cb[2] - cb[0])
            ch = max(1.0, cb[3] - cb[1])
            for cache_id in cache_ids:
                item = self._display_candidate_cache[cache_id]
                age = frame - int(item.get("last_frame", frame))
                if age < 0 or age > self.display_candidate_hold_frames + 1:
                    continue
                pb = item.get("bbox")
                if pb is None:
                    continue
                iou = self._bbox_iou(cb, pb)
                dist = self._bbox_center_distance(cb, pb)
                pw = max(1.0, pb[2] - pb[0])
                ph = max(1.0, pb[3] - pb[1])
                gate = max(28.0, 1.25 * max(cw, ch, pw, ph))
                if iou >= 0.08 or dist <= gate:
                    score = 4.0 * iou - dist / max(gate, 1.0)
                    pairs.append((score, ci, cache_id))

        for _score, ci, cache_id in sorted(pairs, reverse=True):
            if ci in matched_current or cache_id not in unmatched_cache:
                continue
            cand = current[ci]
            item = self._display_candidate_cache[cache_id]
            item.update({
                "bbox": cand["bbox"],
                "class": cand["class"],
                "last_frame": frame,
                "age": 0,
                "status": "ACTIVE",
            })
            matched_current.add(ci)
            unmatched_cache.remove(cache_id)

        for ci, cand in enumerate(current):
            if ci in matched_current:
                continue
            cache_id = self._next_display_candidate_id
            self._next_display_candidate_id += 1
            self._display_candidate_cache[cache_id] = {
                "display_candidate_id": cache_id,
                "bbox": cand["bbox"],
                "class": cand["class"],
                "last_frame": frame,
                "age": 0,
                "status": "ACTIVE",
            }

        for cache_id in list(self._display_candidate_cache):
            item = self._display_candidate_cache[cache_id]
            age = frame - int(item.get("last_frame", frame))
            if age <= 0:
                item["status"] = "ACTIVE"
                item["age"] = 0
            elif age <= self.display_candidate_hold_frames:
                item["status"] = "HELD"
                item["age"] = int(age)
            else:
                self._display_candidate_cache.pop(cache_id, None)

        return list(self._display_candidate_cache.values())

    def _update_display_track_cache(self, result):
        frame = int(result.get("frame", 0) or 0)
        active = result.get("tracking", {}).get("active_tracks", []) or []
        active_ids = set()

        for track in active:
            try:
                tid = int(track.get("track_id"))
            except Exception:
                continue
            box = self._valid_bbox(track.get("bbox"))
            if box is None:
                continue
            active_ids.add(tid)
            self._display_track_cache[tid] = {
                "track_id": tid,
                "class": track.get("class", "object"),
                "bbox": box,
                "last_frame": frame,
                "speed": float(track.get("speed", 0.0) or 0.0),
                "motion_state": track.get("motion_state", "-"),
                "position": track.get("position"),
                "status": "ACTIVE",
            }

        for tid in list(self._display_track_cache):
            item = self._display_track_cache[tid]
            age = frame - int(item.get("last_frame", frame))
            if tid in active_ids:
                item["status"] = "ACTIVE"
                continue
            if age <= self.display_track_hold_frames:
                item["status"] = "PREDICTED"
                item["age"] = int(age)
            else:
                self._display_track_cache.pop(tid, None)

        return list(self._display_track_cache.values())

    def _local_tracks(self, result, pose, display_tracks):
        current_by_id = {}
        for track in result.get("tracking", {}).get("active_tracks", []) or []:
            try:
                current_by_id[int(track.get("track_id"))] = track
            except Exception:
                pass

        output = []
        c = math.cos(float(pose[2]))
        s = math.sin(float(pose[2]))
        for display in display_tracks:
            tid = display.get("track_id")
            track = current_by_id.get(tid)
            position = None
            if track is not None:
                position = track.get("position")
            if position is None:
                position = display.get("position")
            p = np.asarray(position if position is not None else [], dtype=np.float64).reshape(-1)
            if p.size < 2:
                continue
            dx = float(p[0]) - float(pose[3])
            dy = float(p[1]) - float(pose[4])
            forward = c * dx + s * dy
            lateral = -s * dx + c * dy
            if not np.isfinite(forward) or not np.isfinite(lateral):
                continue
            if forward * forward + lateral * lateral > self.meters * self.meters:
                continue
            output.append(
                {
                    "track_id": tid,
                    "class": display.get("class", "object"),
                    "forward": float(forward),
                    "lateral": float(lateral),
                    "speed": float(display.get("speed", 0.0) or 0.0),
                    "motion_state": display.get("motion_state", "-"),
                    "status": display.get("status", "ACTIVE"),
                    "age": int(display.get("age", 0) or 0),
                }
            )
        return output

    # ------------------------------------------------------------------
    # Snapshot streams for the dashboard
    # ------------------------------------------------------------------

    def _write_snapshot(self, result, pose, ground, structure, local_tracks, stats, ground_lines=None, structure_lines=None, raw_local=None, road_local=None, individual_objects_local=None):
        combined = np.vstack([ground, structure]) if len(ground) + len(structure) else np.empty((0, 3))
        raw = np.asarray(raw_local if raw_local is not None else combined, dtype=np.float64).reshape(-1, 3)
        raw_stride = max(1, int(math.ceil(len(raw) / 4500.0)))
        snapshot = {
            "frame": int(result.get("frame", 0) or 0),
            "carla_frame": int(result.get("carla_frame", 0) or 0),
            "timestamp": float(result.get("timestamp", 0.0) or 0.0),
            "meters": float(self.meters),
            "pose": [float(v) for v in pose[:6]],
            "map_voxels": int((result.get("mapping_4d", {}) or {}).get("total_static_voxels", 0) or 0),
            "map_points_local": combined.astype(np.float32).tolist(),
            "raw_map_points_local": raw[::raw_stride].astype(np.float32).tolist(),
            "ground_points_local": ground.astype(np.float32).tolist(),
            "structure_points_local": structure.astype(np.float32).tolist(),
            "ground_outlines_local": ground_lines or [],
            "structure_outlines_local": structure_lines or [],
            "map_view_format": "cleaned-connected-contours-v2",
            "map_render_stats": stats,
            "tracks_local": local_tracks,
            # Unique-ID FGO trajectories, stationary boxes, and motion status.
            "individual_objects_local": individual_objects_local or [],
            "measured_road_points_local": (np.asarray(road_local,dtype=np.float32).reshape(-1,3).tolist() if road_local is not None else []),
            "measured_road_layer": result.get('mapping_4d',{}).get('road_surface',{}),
        }
        temp_path = self.snapshot_path.with_suffix(self.snapshot_path.suffix + ".tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                # dumps uses the C JSON encoder; dump iterates the deeply
                # nested point lists in Python. The JSON content is identical.
                handle.write(json.dumps(snapshot, separators=(",", ":")))
            os.replace(str(temp_path), str(self.snapshot_path))
        except Exception:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass

    def _write_camera_preview(self, camera):
        if cv2 is None or camera is None or np.asarray(camera).size == 0:
            return
        image = np.asarray(camera)
        h, w = image.shape[:2]
        if w > self.preview_width:
            scale = self.preview_width / float(w)
            image = cv2.resize(
                image,
                (self.preview_width, max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            return
        temp_path = self.camera_snapshot_path.with_suffix(self.camera_snapshot_path.suffix + ".tmp")
        try:
            with temp_path.open("wb") as handle:
                handle.write(encoded.tobytes())
            os.replace(str(temp_path), str(self.camera_snapshot_path))
        except Exception:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Camera overlay
    # ------------------------------------------------------------------

    def _draw_camera(self, image, result, pose, vel, display_tracks, display_candidates):
        camera = np.asarray(image).copy()
        if cv2 is None:
            return camera

        # Fused camera+LiDAR candidates.  ACTIVE is a real current-frame
        # candidate.  HELD is display-only continuity for a very short miss and
        # is deliberately dashed/dim so it cannot be mistaken for current data.
        for item in display_candidates:
            box = self._valid_bbox(item.get("bbox"))
            if box is None:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            if item.get("status") == "ACTIVE":
                color = (255, 255, 0)
                cv2.rectangle(camera, (x1, y1), (x2, y2), color, 2)
                label = "CAND"
            else:
                color = (150, 150, 0)
                self._draw_dashed_rect(camera, box, color, thickness=1)
                label = f"CAND HOLD +{int(item.get('age', 1))}f"
            cls = str(item.get("class", "candidate"))
            if cls and cls not in ("candidate", "object", "None"):
                label += f" {cls}"
            cv2.putText(
                camera, label, (x1, min(camera.shape[0] - 8, y2 + 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA
            )

        # Confirmed track = solid green.  A one-to-three frame display bridge
        # is amber and explicitly labelled PRED so a missed detector frame does
        # not look like a tracker ID was deleted.
        for item in display_tracks:
            box = self._valid_bbox(item.get("bbox"))
            if box is None:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            active = item.get("status") == "ACTIVE"
            color = (0, 220, 0) if active else (0, 175, 255)
            thickness = 2 if active else 1
            cv2.rectangle(camera, (x1, y1), (x2, y2), color, thickness)
            label = f"ID {item.get('track_id', '?')} {item.get('class', 'object')}"
            if not active:
                label += f" PRED +{int(item.get('age', 1))}f"
            cv2.putText(
                camera,
                label,
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

        perf = result.get("performance", {}) or {}
        text = [
            "LVIMOT LIVE - RGB/LIDAR/IMU ESTIMATE",
            f"frame {result.get('frame', 0)}  speed {np.linalg.norm(vel):.2f} m/s",
            f"pose x={pose[3]:.2f} y={pose[4]:.2f} yaw={np.rad2deg(pose[2]):.1f} deg",
            f"published tracks {len(result.get('tracking', {}).get('active_tracks', []))}  map {(result.get('mapping_4d', {}) or {}).get('total_static_voxels', 0)} voxels",
            f"core {perf.get('total_ms', 0):.0f} ms  RSS {perf.get('rss_mb', 0):.0f} MB",
            f"cyan=fused candidate (hold {self.display_candidate_hold_frames}f dashed) | green=published track | amber=track hold",
            "GT is evaluation only",
        ]
        for i, line in enumerate(text):
            cv2.putText(
                camera,
                line,
                (18, 28 + 23 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return camera

    # ------------------------------------------------------------------
    # BEV rendering
    # ------------------------------------------------------------------

    def _draw_bev(self, ground, structure, local_tracks, road=None, individual_objects_local=None):
        if cv2 is None:
            return np.empty((0, 0, 3), dtype=np.uint8)

        bev = np.zeros((self.bev_size, self.bev_size, 3), dtype=np.uint8)
        center = self.bev_size // 2
        scale = self.bev_size / (2.0 * self.meters)

        for radius_m in (10, 20, 30, 40, 50, 60):
            radius_px = int(round(radius_m * scale))
            if radius_px <= center:
                cv2.circle(bev, (center, center), radius_px, (30, 48, 65), 1)
        cv2.line(bev, (0, center), (self.bev_size - 1, center), (30, 48, 65), 1)
        cv2.line(bev, (center, 0), (center, self.bev_size - 1), (30, 48, 65), 1)

        def draw_cells(points, color, radius):
            pts = np.asarray(points, dtype=np.float64)
            if len(pts) == 0:
                return
            px = np.rint(center + pts[:, 1] * scale).astype(np.int32)
            py = np.rint(center - pts[:, 0] * scale).astype(np.int32)
            valid = (px >= 0) & (px < self.bev_size) & (py >= 0) & (py < self.bev_size)
            for x, y in zip(px[valid], py[valid]):
                cv2.circle(bev, (int(x), int(y)), radius, color, -1)

        # Ground/road is intentionally dim; structures/obstacles are bright.
        draw_cells(ground, (75, 88, 100), 1)
        draw_cells(structure, (180, 190, 200), 1)
        if road is not None:
            draw_cells(road,(95,180,75),1)

        # Moving-object lines + stationary-object 3D-box footprints are only
        # presentation. They never feed measurements back to estimator/MOT.
        for obj in individual_objects_local or []:
            status = obj.get('motion_status', 'LAST_SEEN')
            color = ((220, 210, 30) if status == 'MOVING' else
                     (120, 180, 230) if status == 'STOPPED' else
                     (110, 220, 110) if status == 'PARKED_CANDIDATE' else (110, 110, 130))
            path = obj.get('trajectory_local', [])
            segment = []
            previous = None
            for xyz in path:
                if len(xyz) < 2:
                    continue
                current = np.asarray(xyz[:2], dtype=float)
                if np.dot(current, current) > self.meters ** 2:
                    if len(segment) >= 2:
                        cv2.polylines(bev, [np.asarray(segment, dtype=np.int32)], False, color, 2)
                    segment, previous = [], None
                    continue
                if previous is not None and np.linalg.norm(current - previous) > 8.0:
                    if len(segment) >= 2:
                        cv2.polylines(bev, [np.asarray(segment, dtype=np.int32)], False, color, 2)
                    segment = []
                segment.append([int(round(center + current[1] * scale)),
                                int(round(center - current[0] * scale))])
                previous = current
            if len(segment) >= 2:
                cv2.polylines(bev, [np.asarray(segment, dtype=np.int32)], False, color, 2)
            corners = obj.get('box_corners_local', [])
            if len(corners) >= 4:
                box = np.asarray([[round(center + c[1] * scale), round(center - c[0] * scale)]
                                  for c in corners[:4]], dtype=np.int32)
                cv2.polylines(bev, [box], True, color, 2)
            xyz = obj.get('position_local', [])
            if len(xyz) >= 2:
                x, y = int(round(center+xyz[1]*scale)), int(round(center-xyz[0]*scale))
                cv2.putText(bev, 'ID {} {}'.format(obj.get('track_id', '?'), status),
                            (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        triangle = np.asarray(
            [[center, center - 10], [center - 7, center + 8], [center + 7, center + 8]],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(bev, triangle, (0, 220, 160))

        for track in local_tracks:
            x = int(round(center + float(track["lateral"]) * scale))
            y = int(round(center - float(track["forward"]) * scale))
            if 0 <= x < self.bev_size and 0 <= y < self.bev_size:
                predicted = track.get("status") != "ACTIVE"
                color = (0, 175, 255) if predicted else (0, 165, 255)
                cv2.circle(bev, (x, y), 6 if not predicted else 5, color, -1)
                suffix = " P" if predicted else ""
                cv2.putText(
                    bev,
                    f"ID {track.get('track_id', '')}{suffix}",
                    (x + 8, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                )

        cv2.putText(bev, "CLEAN ROLLING 4D LOCAL MAP", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(
            bev,
            "dark=ground | green=measured road | light=structures",
            (15, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 190, 200),
            1,
        )
        return bev

    # ------------------------------------------------------------------
    # Public renderer
    # ------------------------------------------------------------------

    def render(self, image, result, map_points, road_points=None):
        pose = self._pose(result)
        vel = self._velocity(result)

        raw_local = self._bounded_local_map(map_points, pose)
        road_local = self._bounded_local_map(road_points,pose) if road_points is not None else np.empty((0,3))
        if len(road_local)>2500:
            road_local=road_local[::int(math.ceil(len(road_local)/2500.))]
        ground, structure, map_stats = self._clean_map_cells(raw_local)
        ground, structure, ground_lines, structure_lines, filtered_stats = cleaned_layers_and_outlines(
            ground, structure, self.meters, self.map_cell_size_m
        )
        map_stats.update(filtered_stats)
        map_stats["ground_cells"] = int(len(ground))
        map_stats["structure_cells"] = int(len(structure))
        map_stats["clean_cells"] = int(len(ground) + len(structure))
        # Bound JSON/GUI work while retaining the connected outlines extracted
        # from *all* supported cells, not from a decimated point cloud.
        total = len(ground) + len(structure)
        if total > self.max_snapshot_points:
            step = max(1, int(math.ceil(total / float(self.max_snapshot_points))))
            ground_display = ground[::step]
            structure_display = structure[::step]
        else:
            ground_display, structure_display = ground, structure
        display_tracks = self._update_display_track_cache(result)
        display_candidates = self._update_display_candidate_cache(result)
        local_tracks = self._local_tracks(result, pose, display_tracks)
        individual_objects_local = objects_to_local(
            result.get("individual_object_tracks", []) or [], pose, self.meters)

        self._write_snapshot(result, pose, ground_display, structure_display, local_tracks, map_stats, ground_lines, structure_lines, raw_local, road_local, individual_objects_local)

        if cv2 is None:
            return image

        camera = self._draw_camera(image, result, pose, vel, display_tracks, display_candidates)
        self._write_camera_preview(camera)
        bev = self._draw_bev(ground_display, structure_display, local_tracks,road_local,individual_objects_local)

        h = max(camera.shape[0], bev.shape[0])
        cam_scale = h / camera.shape[0]
        camera = cv2.resize(camera, (int(camera.shape[1] * cam_scale), h))
        if bev.shape[0] != h:
            bev = cv2.resize(bev, (h, h))
        return np.hstack([camera, bev])


"""Bounded dashboard rendering in a separate CPU process.

Only preview packets may be replaced. Every sensor frame still reaches the
estimator, recorder and evaluator. No CUDA objects cross the process boundary.
"""


import copy
import multiprocessing
import queue
import signal
import threading
import time


def _render_worker(visualizer, packets, completed, last_frame, errors):
    # The parent handles the GUI's graceful stop and then flushes this worker.
    # Windows console break events can otherwise interrupt both processes.
    for name in ('SIGINT', 'SIGBREAK'):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), signal.SIG_IGN)
            except ValueError:  # The unit-test thread cannot install handlers.
                pass
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass
    while True:
        packet = packets.get()
        if packet is None:
            return
        try:
            image, result, points, road = packet
            visualizer.render(image, result, points, road_points=road)
            completed.value += 1
            last_frame.value = int(result.get('frame', -1))
        except Exception as exc:
            try:
                errors.put_nowait(str(exc))
            except queue.Full:
                pass


class DashboardPresentation:
    def __init__(self, visualizer, use_process=True):
        self.visualizer = visualizer
        context = multiprocessing.get_context('spawn')
        self._packets = context.Queue(maxsize=1) if use_process else queue.Queue(maxsize=1)
        self._errors = context.Queue(maxsize=1) if use_process else queue.Queue(maxsize=1)
        self._completed = context.Value('i', 0)
        self._last_frame = context.Value('q', -1)
        self.replaced_previews = 0
        self.error = None
        self._stopping = False
        args = (visualizer, self._packets, self._completed, self._last_frame, self._errors)
        if use_process:
            self._worker = context.Process(target=_render_worker, args=args, name='lvimot-dashboard', daemon=True)
        else:
            self._worker = threading.Thread(target=_render_worker, args=args, name='lvimot-dashboard', daemon=True)
        self._worker.start()

    def submit(self, image, result, map_points, road_points=None):
        if self._stopping:
            return
        display = {key: copy.deepcopy(result[key]) for key in (
            'frame', 'carla_frame', 'timestamp', 'tracking', 'performance',
            'mapping_4d', 'individual_object_tracks') if key in result}
        state = result.get('state_estimation', {})
        display['state_estimation'] = copy.deepcopy({key: state[key] for key in ('pose', 'velocity') if key in state})
        # Retain fused candidates for the existing cyan camera overlays.
        display['multimodal_objects'] = copy.deepcopy([
            {key: obj[key] for key in ('projected_bbox', 'class', 'label') if key in obj}
            for obj in result.get('multimodal_objects', [])])
        packet = (image.copy(), display, map_points.copy(),
                  None if road_points is None else road_points.copy())
        try:
            self._packets.put_nowait(packet)
        except queue.Full:
            try:
                self._packets.get(timeout=0.01)
                self.replaced_previews += 1
            except queue.Empty:
                pass
            try:
                self._packets.put_nowait(packet)
            except queue.Full:
                self.replaced_previews += 1

    def summary(self):
        try:
            self.error = self._errors.get_nowait()
        except queue.Empty:
            pass
        return {'completed_previews': self._completed.value,
                'replaced_pending_previews': self.replaced_previews,
                'last_preview_frame': self._last_frame.value,
                'error': self.error, 'estimator_frames_dropped': 0}

    def close(self):
        if self._stopping:
            return
        self._stopping = True
        deadline = time.monotonic() + 30.
        while self._worker.is_alive() and time.monotonic() < deadline:
            try:
                self._packets.put(None, timeout=0.2)
                break
            except queue.Full:
                continue
        self._worker.join(timeout=max(0., deadline-time.monotonic()))
        if self._worker.is_alive():
            self.error = 'Dashboard did not finish within 30 seconds'
            if hasattr(self._worker, 'terminate'):
                self._worker.terminate()
                self._worker.join(timeout=5.)
        if getattr(self._worker, 'exitcode', 0) not in (None, 0):
            self.error = 'Dashboard worker exited with code {}'.format(self._worker.exitcode)
