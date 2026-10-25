
# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[3]))

from collections import deque
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path

import numpy as np
from lvimot.mapping_4d.static_map.accumulated_map import AccumulatedStaticMap
from lvimot.mapping_4d.multiobject_trajectory_management import IndividualObjectScene


class Carla4DEnvironmentMapper:
    """Bounded-memory static 3D map plus dynamic-object tubes."""

    def __init__(
        self,
        static_voxel_size: float = 0.25,
        max_map_distance: float = 60.0,
        max_static_voxels: int = 300000,
        max_planar_landmarks: int = 20000,
        max_dynamic_history: int = 500,
    ):
        self.static_voxel_size = float(static_voxel_size)
        self.max_map_distance = float(max_map_distance)
        self.max_static_voxels = int(max_static_voxels)
        self.max_planar_landmarks = int(max_planar_landmarks)
        self.max_dynamic_history = int(max_dynamic_history)
        self.static_voxel_map: Dict[Tuple[int, int, int], np.ndarray] = {}
        self.static_planar_landmarks = deque(maxlen=self.max_planar_landmarks)
        self.dynamic_object_tubes: Dict[int, deque] = {}
        # FGO-only object scene used for display/export, never feedback to MOT/FGO.
        self.individual_object_scene = IndividualObjectScene(max_history=self.max_dynamic_history)
        self.accumulated_map = None
        self.object_archive = None
        self.estimated_ego_history = deque(maxlen=self.max_dynamic_history)

    def enable_accumulated_map(self, directory):
        if self.accumulated_map is not None:
            raise RuntimeError('An accumulated map is already attached to this mapper')
        folder = Path(directory) / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        folder.mkdir(parents=True, exist_ok=False)
        self.accumulated_map = AccumulatedStaticMap(folder / 'static_map.sqlite3', self.static_voxel_size)
        # Disk-backed ID histories survive the bounded in-memory 4D scene.
        # The archive accepts only validated FGO snapshots; never used by MOT.
        from lvimot.output.offline_map import ObjectTrackArchive
        self.object_archive = ObjectTrackArchive(folder / 'object_tracks.sqlite3')
        road_layer = getattr(self, 'road_surface_layer', None)
        road_mapper = getattr(road_layer, 'mapper', None)
        if road_mapper is not None and road_mapper is not self and road_mapper.accumulated_map is None:
            # Keep the whole driven road, even after live rolling-map eviction.
            road_mapper.enable_accumulated_map(folder / 'road')
        return self.accumulated_map.summary()

    def close(self):
        if self.object_archive is not None:
            self.object_archive.close()
        if self.accumulated_map is not None:
            self.accumulated_map.close()
        road_layer = getattr(self, 'road_surface_layer', None)
        road_mapper = getattr(road_layer, 'mapper', None)
        if road_mapper is not None and road_mapper is not self:
            road_mapper.close()

    @staticmethod
    def _rotation_matrix(rotation: Dict[str, float]) -> np.ndarray:
        pitch = np.deg2rad(float(rotation.get("pitch", 0.0)))
        yaw = np.deg2rad(float(rotation.get("yaw", 0.0)))
        roll = np.deg2rad(float(rotation.get("roll", 0.0)))
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cr, sr = np.cos(roll), np.sin(roll)
        return np.array(
            [
                [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
                [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
                [sp, -cp * sr, cp * cr],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _vector3(value, default=(0.0, 0.0, 0.0)) -> np.ndarray:
        if value is None:
            return np.asarray(default, dtype=np.float64)
        if isinstance(value, dict):
            return np.array(
                [float(value.get("x", 0.0)), float(value.get("y", 0.0)), float(value.get("z", 0.0))],
                dtype=np.float64,
            )
        return np.asarray(value, dtype=np.float64).reshape(3)

    @staticmethod
    def _dimensions(value) -> Dict[str, float]:
        if isinstance(value, dict):
            return {
                "length": float(value.get("length", 4.5)),
                "width": float(value.get("width", 1.8)),
                "height": float(value.get("height", 1.5)),
            }
        arr = np.asarray(value if value is not None else [4.5, 1.8, 1.5], dtype=np.float64).reshape(-1)
        padded = np.pad(arr[:3], (0, max(0, 3 - len(arr))), constant_values=1.0)
        return {"length": float(padded[0]), "width": float(padded[1]), "height": float(padded[2])}

    def separate_points(
        self,
        points: np.ndarray,
        dynamic_objects: List[Dict],
        R_world_ego: np.ndarray,
        t_world_ego: np.ndarray,
    ):
        points = np.asarray(points, dtype=np.float64)
        if len(points) == 0:
            return np.empty((0, 3)), np.empty((0, 3))
        pts_w = points[:, :3] @ np.asarray(R_world_ego, dtype=np.float64).T + np.asarray(t_world_ego, dtype=np.float64)
        if not dynamic_objects:
            return pts_w, np.empty((0, 3))

        dynamic_mask = np.zeros(len(pts_w), dtype=bool)
        for obj in dynamic_objects:
            center = self._vector3(obj.get("position", obj.get("location")))
            dims = self._dimensions(obj.get("dimensions"))
            half = np.array([dims["length"] / 2 + 0.2, dims["width"] / 2 + 0.2, dims["height"] / 2 + 0.2])
            radius2 = float(np.dot(half, half))
            delta = pts_w - center
            candidate = np.flatnonzero(np.einsum("ij,ij->i", delta, delta) <= radius2)
            if candidate.size == 0:
                continue
            R = self._rotation_matrix(obj.get("rotation", {}))
            local = (pts_w[candidate] - center) @ R
            inside = np.all(np.abs(local) <= half, axis=1)
            dynamic_mask[candidate[inside]] = True
        return pts_w[~dynamic_mask], pts_w[dynamic_mask]

    def _evict_distant_voxels(self, ego_position: np.ndarray, margin: float = 1.35):
        """Keep the live map spatially bounded around the current ego pose.

        A fixed global dictionary eventually reaches its cap and then stops
        mapping new road.  A rolling local map is more appropriate for the
        live visualization target and gives bounded RAM over long drives.
        """
        if not self.static_voxel_map:
            return
        ego = np.asarray(ego_position, dtype=np.float64)
        max_d2 = float((self.max_map_distance * margin) ** 2)
        keys = list(self.static_voxel_map)
        centers = (np.asarray(keys, dtype=np.float64) + .5) * self.static_voxel_size
        delta = centers - ego
        remove = np.flatnonzero(np.einsum('ij,ij->i', delta, delta) > max_d2)
        for index in remove:
            self.static_voxel_map.pop(keys[int(index)], None)

    def _reserve_voxel_capacity(self, keys, ego_position):
        if isinstance(keys, np.ndarray):
            keys = list(map(tuple, keys.tolist()))
        new_count = sum(key not in self.static_voxel_map for key in keys)
        overflow = len(self.static_voxel_map) + new_count - self.max_static_voxels
        if overflow <= 0 or not self.static_voxel_map:
            return
        # Never silently freeze a full map. Preserve the nearest live context;
        # older distant observations remain available in the disk archive.
        existing = list(self.static_voxel_map)
        xyz = np.asarray(list(self.static_voxel_map.values()), dtype=float)
        delta = xyz - np.asarray(ego_position, dtype=float)
        d2 = np.einsum('ij,ij->i', delta, delta)
        amount = min(len(existing), max(overflow, self.max_static_voxels // 10))
        remove = np.argpartition(d2, len(d2) - amount)[-amount:]
        for index in remove:
            self.static_voxel_map.pop(existing[int(index)], None)

    def _update_static_voxels(self, points_world: np.ndarray, ego_position: np.ndarray):
        if len(points_world) == 0:
            return
        delta = points_world - ego_position
        keep = np.einsum("ij,ij->i", delta, delta) <= self.max_map_distance ** 2
        pts = points_world[keep]
        if len(pts) == 0:
            return
        keys_arr = np.floor(pts / self.static_voxel_size).astype(np.int64)
        keys, inverse = np.unique(keys_arr, axis=0, return_inverse=True)
        sums = np.zeros((len(keys), 3), dtype=np.float64)
        np.add.at(sums, inverse, pts)
        counts = np.bincount(inverse, minlength=len(keys)).astype(np.float64)
        representatives = sums / np.maximum(counts[:, None], 1.0)
        self._update_grouped_voxels(keys, representatives, ego_position)

    def _update_grouped_voxels(self, keys, representatives, ego_position):
        """Apply the same per-voxel update with batched arithmetic."""
        key_list = list(map(tuple, keys.tolist()))
        self._reserve_voxel_capacity(key_list, ego_position)
        order = np.argsort(np.sum((representatives - ego_position) ** 2, axis=1))
        ordered_keys = [key_list[int(i)] for i in order]
        points = representatives[order]
        existing = np.asarray([key in self.static_voxel_map for key in ordered_keys], dtype=bool)
        allowed = existing.copy()
        slots = max(0, self.max_static_voxels-len(self.static_voxel_map))
        allowed[np.flatnonzero(~existing)[:slots]] = True
        updated = points.astype(np.float32)
        ids = np.flatnonzero(existing)
        if len(ids):
            old = np.asarray([self.static_voxel_map[ordered_keys[int(i)]] for i in ids], dtype=np.float32)
            # Keep the old float32 multiplication before the float64 addition.
            # Promoting old values early would alter rounding near voxel edges.
            updated[ids] = (0.7*old + 0.3*points[ids]).astype(np.float32)
        self.static_voxel_map.update((key, row.copy())
            for key,row,keep in zip(ordered_keys,updated,allowed) if keep)

    def add_frame_data(
        self,
        frame_id: int,
        timestamp: float,
        lidar_points: np.ndarray,
        planar_features: List[Dict],
        active_tracks: List[Dict],
        ego_pose: Dict,
        update_static: bool = True,
        update_planar: bool = True,
        evict_static: bool = False,
        mask_tracks: List[Dict] = None,
        estimated_ego_velocity=None,
    ):
        R_w_e = self._rotation_matrix(ego_pose["rotation"])
        t_w_e = self._vector3(ego_pose["location"])
        # This pose is the optimized sensor estimate supplied by the core.
        # Store every accepted processed frame, including frames with no objects
        # or map updates. This display archive never feeds the estimator.
        yaw = float(ego_pose.get('rotation', {}).get('yaw', 0.0))
        velocity = (self._vector3(estimated_ego_velocity).tolist()
                    if estimated_ego_velocity is not None else None)
        if not np.isfinite(np.r_[t_w_e, yaw, float(timestamp)]).all():
            raise ValueError('Nonfinite estimated ego observation')
        if velocity is not None and not np.isfinite(velocity).all():
            raise ValueError('Nonfinite estimated ego velocity')
        ego_state = {'frame_id': int(frame_id), 'timestamp': float(timestamp),
                     'position': t_w_e.tolist(), 'yaw': yaw, 'velocity': velocity,
                     'source': 'temporal_fusion_fgo', 'ground_truth_used': False}
        self.estimated_ego_history.append(ego_state)
        if self.object_archive is not None:
            self.object_archive.append_ego(ego_state)
        # Include stationary FGO objects in the individual-object display history.
        fgo_tracks = [tr for tr in active_tracks if tr.get("source") == "temporal_fusion_fgo"]
        self.individual_object_scene.update(frame_id, timestamp, fgo_tracks)
        if self.object_archive is not None:
            for tr in fgo_tracks:
                try:
                    tid = int(tr['track_id'])
                    state = self.individual_object_scene.objects.get(tid)
                    if state is not None and state['frame_id'] == int(frame_id) and state['timestamp'] == float(timestamp):
                        # Only the scene's finite, non-extrapolated observation.
                        self.object_archive.append(tid, state['history'][-1])
                except (KeyError, TypeError, ValueError):
                    continue
        dynamic_tracks = [
            t for t in active_tracks if t.get("motion_state") in {"DYNAMIC", "STOPPED_DYNAMIC", "UNCERTAIN"}
        ]

        for track in dynamic_tracks:
            tid = int(track["track_id"])
            tube = self.dynamic_object_tubes.setdefault(tid, deque(maxlen=self.max_dynamic_history))
            tube.append(
                {
                    "frame_id": int(frame_id),
                    "timestamp": float(timestamp),
                    "position": self._vector3(track.get("position")).tolist(),
                    "velocity": self._vector3(track.get("velocity")).tolist(),
                    "speed": float(track.get("speed", np.linalg.norm(self._vector3(track.get("velocity"))))),
                    "dimensions": self._dimensions(track.get("dimensions")),
                    "rotation": track.get("rotation", {}),
                    "class": track.get("class", "object"),
                    "motion_state": track.get("motion_state", "UNCERTAIN"),
                    "source": track.get("source", "legacy_direct_tracks"),
                }
            )

        if bool(update_static):
            # Preserve reliable MOT masking even when no FGO object state has
            # been created yet. MOT points NEVER enter the trajectory history.
            masking_source = active_tracks if mask_tracks is None else mask_tracks
            masking_objects = [
                item for item in masking_source
                if item.get("motion_state") in {"DYNAMIC", "STOPPED_DYNAMIC", "UNCERTAIN"}
            ]
            static_pts, _ = self.separate_points(lidar_points, masking_objects, R_w_e, t_w_e)
            if self.accumulated_map is not None:
                delta = static_pts - t_w_e
                local = np.einsum('ij,ij->i', delta, delta) <= self.max_map_distance ** 2
                self.accumulated_map.add(static_pts[local], frame_id)
            self._update_static_voxels(static_pts, t_w_e)
        if bool(evict_static):
            self._evict_distant_voxels(t_w_e)

        # Store only a bounded representative subset of planar landmarks.
        if bool(update_planar) and planar_features:
            step = max(1, len(planar_features) // 256)
            for feat in planar_features[::step][:256]:
                c = self._vector3(feat.get("center"))
                n = self._vector3(feat.get("normal"), default=(0.0, 0.0, 1.0))
                c_w = R_w_e @ c + t_w_e
                n_w = R_w_e @ n
                n_w /= max(np.linalg.norm(n_w), 1e-6)
                self.static_planar_landmarks.append(
                    {
                        "center": c_w.tolist(),
                        "normal": n_w.tolist(),
                        "planarity": float(feat.get("planarity", 0.0)),
                        "frame_id": int(frame_id),
                        "timestamp": float(timestamp),
                    }
                )

    def individual_object_tracks(self, timestamp: float) -> List[Dict]:
        return self.individual_object_scene.current(timestamp)

    def get_static_map_points(self) -> np.ndarray:
        if not self.static_voxel_map:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(list(self.static_voxel_map.values()), dtype=np.float64)

    def query_dynamic_objects_at_time(self, query_timestamp: float) -> List[Dict]:
        results = []
        for tid, tube_deque in self.dynamic_object_tubes.items():
            tube = list(tube_deque)
            if not tube:
                continue
            timestamps = np.asarray([snap["timestamp"] for snap in tube], dtype=np.float64)
            if query_timestamp <= timestamps[0]:
                snap = tube[0]
                pos = np.asarray(snap["position"]) + np.asarray(snap["velocity"]) * (query_timestamp - timestamps[0])
            elif query_timestamp >= timestamps[-1]:
                snap = tube[-1]
                pos = np.asarray(snap["position"]) + np.asarray(snap["velocity"]) * (query_timestamp - timestamps[-1])
            else:
                idx = int(np.searchsorted(timestamps, query_timestamp))
                s0, s1 = tube[idx - 1], tube[idx]
                alpha = (query_timestamp - s0["timestamp"]) / max(s1["timestamp"] - s0["timestamp"], 1e-6)
                pos = (1.0 - alpha) * np.asarray(s0["position"]) + alpha * np.asarray(s1["position"])
                snap = s1
            results.append(
                {
                    "track_id": tid,
                    "timestamp": float(query_timestamp),
                    "interpolated_position": pos.tolist(),
                    "velocity": snap["velocity"],
                    "class": snap["class"],
                    "dimensions": snap["dimensions"],
                    "rotation": snap["rotation"],
                }
            )
        return results

    def export_summary(self) -> Dict:
        result = {
            "total_static_voxels": len(self.static_voxel_map),
            "total_planar_landmarks": len(self.static_planar_landmarks),
            "total_dynamic_actors": len(self.dynamic_object_tubes),
            "total_individual_object_ids": len(self.individual_object_scene.objects),
            "dynamic_actor_ids": list(self.dynamic_object_tubes.keys()),
            "dynamic_tube_lengths": {int(k): len(v) for k, v in self.dynamic_object_tubes.items()},
            "memory_bounded": True,
            'rolling_map_radius_m': self.max_map_distance,
            'rolling_map_voxel_size_m': self.static_voxel_size,
            'rolling_map_voxel_capacity': self.max_static_voxels,
            'accumulated_map': self.accumulated_map.summary() if self.accumulated_map is not None else None,
        }
        road_layer=getattr(self,'road_surface_layer',None)
        if road_layer is not None:
            result['road_surface']=road_layer.summary()
        return result
