
# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))

import numpy as np
from typing import Dict, Optional, Tuple, Union


class PosePriorFactor:
    def __init__(self, pose: np.ndarray, weight_rotation: float = 50.0, weight_position: float = 50.0):
        self.pose = np.asarray(pose, dtype=np.float64).reshape(6)
        self.weight_rotation = float(weight_rotation)
        self.weight_position = float(weight_position)

    def compute_residual(self, current_pose: np.ndarray) -> np.ndarray:
        current = np.asarray(current_pose, dtype=np.float64).reshape(6)
        residual = current - self.pose
        # Wrap Euler-angle residuals into [-pi, pi].
        residual[:3] = (residual[:3] + np.pi) % (2.0 * np.pi) - np.pi
        residual[:3] *= self.weight_rotation
        residual[3:] *= self.weight_position
        return residual


class VelocityPriorFactor:
    """Unary velocity prior used by sensor-only stationary mode."""

    def __init__(self, velocity: np.ndarray, weight: float = 20.0):
        self.velocity = np.asarray(velocity, dtype=np.float64).reshape(3)
        self.weight = float(weight)

    def compute_residual(self, current_velocity: np.ndarray) -> np.ndarray:
        current = np.asarray(current_velocity, dtype=np.float64).reshape(3)
        return self.weight * (current - self.velocity)


class IMUPreintegrationFactor:
    def __init__(
        self,
        delta_p: np.ndarray,
        delta_v: np.ndarray,
        delta_R: np.ndarray,
        dt: float,
        gravity: Optional[np.ndarray] = None,
        weight_p: float = 10.0,
        weight_v: float = 5.0,
        weight_R: float = 20.0
    ):
        self.delta_p = np.asarray(delta_p, dtype=np.float64)
        self.delta_v = np.asarray(delta_v, dtype=np.float64)
        self.delta_R = np.asarray(delta_R, dtype=np.float64)
        self.dt = float(dt)
        self.gravity = (
            gravity
            if gravity is not None
            else np.array([0.0, 0.0, -9.80665], dtype=np.float64)
        )
        self.weight_p = weight_p
        self.weight_v = weight_v
        self.weight_R = weight_R

    @staticmethod
    def log_so3(R: np.ndarray) -> np.ndarray:
        trace = np.trace(R)
        cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        if abs(theta) < 1e-6:
            return 0.5 * np.array(
                [
                    R[2, 1] - R[1, 2],
                    R[0, 2] - R[2, 0],
                    R[1, 0] - R[0, 1]
                ],
                dtype=np.float64
            )

        sin_theta = np.sin(theta)

        if abs(sin_theta) < 1e-8:
            return np.zeros(3, dtype=np.float64)

        factor = theta / (2.0 * sin_theta)

        return factor * np.array(
            [
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1]
            ],
            dtype=np.float64
        )

    def compute_residual(
        self,
        R_i: np.ndarray,
        p_i: np.ndarray,
        v_i: np.ndarray,
        R_j: np.ndarray,
        p_j: np.ndarray,
        v_j: np.ndarray
    ) -> np.ndarray:
        dt = self.dt
        g = self.gravity

        r_p = (
            R_i.T
            @ (p_j - p_i - v_i * dt - 0.5 * g * (dt ** 2))
            - self.delta_p
        )

        r_v = (
            R_i.T
            @ (v_j - v_i - g * dt)
            - self.delta_v
        )

        R_err = self.delta_R.T @ (R_i.T @ R_j)
        r_R = self.log_so3(R_err)

        return np.hstack(
            [
                self.weight_p * r_p,
                self.weight_v * r_v,
                self.weight_R * r_R
            ]
        )


class LiDARRelativePoseFactor:
    """Binary SE(3) relative-pose constraint from scan registration.

    ``delta_R`` and ``delta_t`` map a point expressed in the current ego frame
    into the previous ego frame:

        p_prev = delta_R @ p_curr + delta_t

    For ego poses represented as ego->world transforms, the predicted transform
    is ``R_i.T @ R_j`` and ``R_i.T @ (p_j - p_i)``.
    """

    def __init__(
        self,
        delta_R: np.ndarray,
        delta_t: np.ndarray,
        weight_translation: float = 8.0,
        weight_rotation: float = 12.0,
    ):
        self.delta_R = np.asarray(delta_R, dtype=np.float64).reshape(3, 3)
        self.delta_t = np.asarray(delta_t, dtype=np.float64).reshape(3)
        self.weight_translation = float(weight_translation)
        self.weight_rotation = float(weight_rotation)

    @staticmethod
    def _log_so3(R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        trace = float(np.trace(R))
        cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        theta = float(np.arccos(cos_theta))
        vee = np.array(
            [
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ],
            dtype=np.float64,
        )
        if theta < 1e-7:
            return 0.5 * vee
        sin_theta = float(np.sin(theta))
        if abs(sin_theta) < 1e-8:
            return np.zeros(3, dtype=np.float64)
        return (theta / (2.0 * sin_theta)) * vee

    def compute_residual(
        self,
        R_i: np.ndarray,
        p_i: np.ndarray,
        R_j: np.ndarray,
        p_j: np.ndarray,
    ) -> np.ndarray:
        R_i = np.asarray(R_i, dtype=np.float64).reshape(3, 3)
        R_j = np.asarray(R_j, dtype=np.float64).reshape(3, 3)
        p_i = np.asarray(p_i, dtype=np.float64).reshape(3)
        p_j = np.asarray(p_j, dtype=np.float64).reshape(3)

        R_pred = R_i.T @ R_j
        t_pred = R_i.T @ (p_j - p_i)
        R_err = self.delta_R.T @ R_pred
        r_rot = self._log_so3(R_err)
        r_trans = t_pred - self.delta_t
        return np.hstack(
            [
                self.weight_translation * r_trans,
                self.weight_rotation * r_rot,
            ]
        )


class LiDARPlanarFactor:
    def __init__(
        self,
        point_lidar: np.ndarray,
        plane_normal_w: np.ndarray,
        plane_center_w: np.ndarray,
        weight: float = 5.0
    ):
        self.point_lidar = np.asarray(point_lidar, dtype=np.float64)
        self.plane_normal_w = np.asarray(
            plane_normal_w,
            dtype=np.float64
        )

        self.plane_normal_w /= max(
            np.linalg.norm(self.plane_normal_w),
            1e-6
        )

        self.plane_center_w = np.asarray(
            plane_center_w,
            dtype=np.float64
        )

        self.weight = weight

    def compute_residual(
        self,
        R_ego: np.ndarray,
        t_ego: np.ndarray
    ) -> float:
        p_w = R_ego @ self.point_lidar + t_ego

        dist = float(
            np.dot(
                self.plane_normal_w,
                p_w - self.plane_center_w
            )
        )

        return self.weight * dist


class VisualReprojectionFactor:
    def __init__(
        self,
        measurement_uv: np.ndarray,
        K: np.ndarray,
        landmark_3d_w: np.ndarray,
        weight: float = 1.0,
        camera_translation_ego: Optional[np.ndarray] = None,
        camera_rotation_ego: Optional[np.ndarray] = None,
        use_carla_camera_axes: bool = False,
    ):
        self.z_uv = np.asarray(measurement_uv, dtype=np.float64)
        self.K = np.asarray(K, dtype=np.float64)
        self.landmark_3d_w = np.asarray(landmark_3d_w, dtype=np.float64)
        self.weight = float(weight)
        self.camera_translation_ego = (
            None if camera_translation_ego is None
            else np.asarray(camera_translation_ego, dtype=np.float64).reshape(3)
        )
        self.camera_rotation_ego = (
            None if camera_rotation_ego is None
            else np.asarray(camera_rotation_ego, dtype=np.float64).reshape(3, 3)
        )
        self.use_carla_camera_axes = bool(use_carla_camera_axes)

    def compute_residual(
        self,
        R_cam_w: np.ndarray,
        t_cam_w: np.ndarray
    ) -> np.ndarray:
        # Backward-compatible path: callers may provide world->camera directly.
        if self.camera_translation_ego is None or self.camera_rotation_ego is None:
            p_c = R_cam_w @ self.landmark_3d_w + t_cam_w
        else:
            # In the factor graph R_cam_w/t_cam_w are passed as world->ego.
            p_ego = R_cam_w @ self.landmark_3d_w + t_cam_w
            p_carla_cam = self.camera_rotation_ego.T @ (p_ego - self.camera_translation_ego)
            if self.use_carla_camera_axes:
                # CARLA camera: X forward, Y right, Z up.
                # OpenCV: X right, Y down, Z forward.
                p_c = np.array([p_carla_cam[1], -p_carla_cam[2], p_carla_cam[0]], dtype=np.float64)
            else:
                p_c = p_carla_cam

        if p_c[2] <= 0.1 or not np.isfinite(p_c).all():
            return np.array([100.0, 100.0], dtype=np.float64) * self.weight
        uv_h = self.K @ p_c
        predicted = uv_h[:2] / uv_h[2]
        return self.weight * (predicted - self.z_uv[:2])


class DynamicObjectFactor:
    def __init__(
        self,
        obs_local: np.ndarray,
        dt_from_origin: float,
        weight_pos: float = 5.0
    ):
        self.obs_local = np.asarray(
            obs_local,
            dtype=np.float64
        )

        self.dt = float(dt_from_origin)
        self.weight_pos = weight_pos

    def compute_residual(
        self,
        R_ego: np.ndarray,
        t_ego: np.ndarray,
        obj_p0: np.ndarray,
        obj_vel: np.ndarray
    ) -> np.ndarray:
        p_obj_pred = (
            obj_p0
            + obj_vel * self.dt
        )

        p_obj_meas = (
            R_ego @ self.obs_local
            + t_ego
        )

        return self.weight_pos * (
            p_obj_pred - p_obj_meas
        )


class MarginalizationFactor:
    """
    Square-root representation of a marginalized quadratic prior.

    The stored quadratic form is:

        0.5 * dx.T @ H_prior @ dx + b_prior.T @ dx

    The least-squares residual is constructed as:

        r = sqrt_info @ dx + residual_offset

    such that:

        sqrt_info.T @ sqrt_info = H_prior
        sqrt_info.T @ residual_offset = b_prior
    """

    def __init__(
        self,
        linearization_point: np.ndarray,
        H_prior: np.ndarray,
        b_prior: np.ndarray
    ):
        self.x_lin = np.asarray(
            linearization_point,
            dtype=np.float64
        ).copy()

        H = np.asarray(
            H_prior,
            dtype=np.float64
        )

        b = np.asarray(
            b_prior,
            dtype=np.float64
        ).reshape(-1)

        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError(
                "H_prior must be a square matrix."
            )

        if H.shape[0] != len(self.x_lin):
            raise ValueError(
                "H_prior dimension must match linearization_point."
            )

        if len(b) != len(self.x_lin):
            raise ValueError(
                "b_prior dimension must match linearization_point."
            )

        H = 0.5 * (H + H.T)

        eigenvalues, eigenvectors = np.linalg.eigh(H)

        max_eigenvalue = max(
            float(np.max(np.abs(eigenvalues))),
            1.0
        )

        tolerance = 1e-10 * max_eigenvalue

        keep = eigenvalues > tolerance

        if np.any(keep):
            values = np.maximum(
                eigenvalues[keep],
                0.0
            )

            vectors = eigenvectors[:, keep]

            self.sqrt_info = (
                np.sqrt(values)[:, None]
                * vectors.T
            )

            self.residual_offset = np.linalg.lstsq(
                self.sqrt_info.T,
                b,
                rcond=None
            )[0]
        else:
            self.sqrt_info = np.zeros(
                (0, len(self.x_lin)),
                dtype=np.float64
            )

            self.residual_offset = np.zeros(
                0,
                dtype=np.float64
            )

        self.H_prior = (
            self.sqrt_info.T
            @ self.sqrt_info
        )

        self.b_prior = (
            self.sqrt_info.T
            @ self.residual_offset
        )

    def compute_residual(
        self,
        current_x: np.ndarray
    ) -> np.ndarray:
        current_x = np.asarray(
            current_x,
            dtype=np.float64
        )

        if len(current_x) != len(self.x_lin):
            raise ValueError(
                "Current state dimension does not match "
                "the marginalization prior."
            )

        dx = current_x - self.x_lin

        if self.sqrt_info.shape[0] == 0:
            return np.zeros(
                0,
                dtype=np.float64
            )

        return (
            self.sqrt_info @ dx
            + self.residual_offset
        )



import numpy as np
from typing import Dict, List, Optional, Tuple

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix, csr_matrix
from scipy.optimize._numdiff import group_columns



class CarlaFactorGraph:
    def __init__(
        self,
        robust_loss: str = "cauchy",
        loss_scale: float = 1.0,
    ):
        self.robust_loss = robust_loss
        self.loss_scale = loss_scale

        self.pose_prior_factors: List[Dict] = []
        self.velocity_prior_factors: List[Dict] = []
        self.imu_factors: List[Dict] = []
        self.lidar_factors: List[Dict] = []
        self.lidar_relative_factors: List[Dict] = []
        self.visual_factors: List[Dict] = []
        self.dynamic_factors: List[Dict] = []
        self.marginalization_factors: List[
            MarginalizationFactor
        ] = []

        self.keyframe_ids: List[int] = []
        self.timestamps: Dict[int, float] = {}

    def clear(self):
        self.pose_prior_factors.clear()
        self.velocity_prior_factors.clear()
        self.imu_factors.clear()
        self.lidar_factors.clear()
        self.lidar_relative_factors.clear()
        self.visual_factors.clear()
        self.dynamic_factors.clear()
        self.marginalization_factors.clear()
        self.keyframe_ids.clear()
        self.timestamps.clear()

    def add_keyframe(
        self,
        frame_id: int,
        timestamp: float,
    ):
        if frame_id not in self.keyframe_ids:
            self.keyframe_ids.append(frame_id)
            self.timestamps[frame_id] = float(timestamp)

    def add_pose_prior(
        self,
        frame_id: int,
        pose: np.ndarray,
        weight_rotation: float = 50.0,
        weight_position: float = 50.0,
    ):
        self.pose_prior_factors.append({
            "frame_id": int(frame_id),
            "factor": PosePriorFactor(pose, weight_rotation, weight_position),
        })

    def add_velocity_prior(
        self,
        frame_id: int,
        velocity: np.ndarray,
        weight: float = 20.0,
    ):
        self.velocity_prior_factors.append({
            "frame_id": int(frame_id),
            "factor": VelocityPriorFactor(velocity, weight),
        })

    def add_imu_factor(
        self,
        frame_i: int,
        frame_j: int,
        delta_p: np.ndarray,
        delta_v: np.ndarray,
        delta_R: np.ndarray,
        dt: float,
    ):
        factor = IMUPreintegrationFactor(
            delta_p,
            delta_v,
            delta_R,
            dt,
        )

        self.imu_factors.append(
            {
                "frame_i": frame_i,
                "frame_j": frame_j,
                "factor": factor,
            }
        )

    def add_lidar_factor(
        self,
        frame_id: int,
        point_lidar: np.ndarray,
        plane_normal_w: np.ndarray,
        plane_center_w: np.ndarray,
        weight: float = 5.0,
    ):
        factor = LiDARPlanarFactor(
            point_lidar,
            plane_normal_w,
            plane_center_w,
            weight=weight,
        )

        self.lidar_factors.append(
            {
                "frame_id": frame_id,
                "factor": factor,
            }
        )

    def add_lidar_relative_factor(
        self,
        frame_i: int,
        frame_j: int,
        delta_R: np.ndarray,
        delta_t: np.ndarray,
        weight_translation: float = 8.0,
        weight_rotation: float = 12.0,
    ):
        factor = LiDARRelativePoseFactor(
            delta_R,
            delta_t,
            weight_translation=weight_translation,
            weight_rotation=weight_rotation,
        )
        self.lidar_relative_factors.append(
            {
                "frame_i": int(frame_i),
                "frame_j": int(frame_j),
                "factor": factor,
            }
        )

    def add_visual_factor(
        self,
        frame_id: int,
        measurement_uv: np.ndarray,
        K: np.ndarray,
        landmark_3d_w: np.ndarray,
        weight: float = 1.0,
        camera_translation_ego: Optional[np.ndarray] = None,
        camera_rotation_ego: Optional[np.ndarray] = None,
        use_carla_camera_axes: bool = False,
    ):
        factor = VisualReprojectionFactor(
            measurement_uv,
            K,
            landmark_3d_w,
            weight=weight,
            camera_translation_ego=camera_translation_ego,
            camera_rotation_ego=camera_rotation_ego,
            use_carla_camera_axes=use_carla_camera_axes,
        )

        self.visual_factors.append(
            {
                "frame_id": frame_id,
                "factor": factor,
            }
        )

    def add_dynamic_object_factor(
        self,
        frame_id: int,
        track_id: int,
        obs_local: np.ndarray,
        dt_from_origin: float,
        weight_pos: float = 5.0,
    ):
        factor = DynamicObjectFactor(
            obs_local,
            dt_from_origin,
            weight_pos=weight_pos,
        )

        self.dynamic_factors.append(
            {
                "frame_id": frame_id,
                "track_id": track_id,
                "factor": factor,
            }
        )

    def add_marginalization_factor(
        self,
        factor: MarginalizationFactor,
    ):
        self.marginalization_factors.append(factor)

    @staticmethod
    def euler_to_rot(
        rpy: np.ndarray,
    ) -> np.ndarray:
        roll, pitch, yaw = rpy

        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        return np.array(
            [
                [
                    cp * cy,
                    cy * sp * sr - sy * cr,
                    -cy * sp * cr - sy * sr,
                ],
                [
                    cp * sy,
                    sy * sp * sr + cy * cr,
                    -sy * sp * cr + cy * sr,
                ],
                [
                    sp,
                    -cp * sr,
                    cp * cr,
                ],
            ],
            dtype=np.float64,
        )

    def _pack_states(
        self,
        ego_poses: Dict[int, np.ndarray],
        ego_velocities: Dict[int, np.ndarray],
        object_states: Dict[int, np.ndarray],
    ) -> Tuple[np.ndarray, Dict]:
        vec_list = []

        mapping = {
            "ego_poses": {},
            "ego_velocities": {},
            "object_states": {},
        }

        offset = 0

        for fid in self.keyframe_ids:
            p = np.asarray(
                ego_poses.get(
                    fid,
                    np.zeros(6, dtype=np.float64),
                ),
                dtype=np.float64,
            )

            mapping["ego_poses"][fid] = (
                offset,
                offset + 6,
            )

            vec_list.append(p)
            offset += 6

            v = np.asarray(
                ego_velocities.get(
                    fid,
                    np.zeros(3, dtype=np.float64),
                ),
                dtype=np.float64,
            )

            mapping["ego_velocities"][fid] = (
                offset,
                offset + 3,
            )

            vec_list.append(v)
            offset += 3

        for tid in sorted(object_states.keys()):
            ostate = np.asarray(
                object_states[tid],
                dtype=np.float64,
            )

            if len(ostate) != 6:
                raise ValueError(
                    "Dynamic object state must have 6 elements."
                )

            mapping["object_states"][tid] = (
                offset,
                offset + 6,
            )

            vec_list.append(ostate)
            offset += 6

        if vec_list:
            flat_vector = np.concatenate(vec_list)
        else:
            flat_vector = np.empty(
                0,
                dtype=np.float64,
            )

        return flat_vector, mapping

    @staticmethod
    def _state_keys(
        mapping: Dict,
    ) -> List[Tuple]:
        keys = []

        pose_items = sorted(
            mapping["ego_poses"].items(),
            key=lambda item: item[1][0],
        )

        velocity_items = sorted(
            mapping["ego_velocities"].items(),
            key=lambda item: item[1][0],
        )

        object_items = sorted(
            mapping["object_states"].items(),
            key=lambda item: item[1][0],
        )

        for fid, (start, end) in pose_items:
            for _ in range(end - start):
                keys.append(("pose", fid))

        for fid, (start, end) in velocity_items:
            for _ in range(end - start):
                keys.append(("velocity", fid))

        for tid, (start, end) in object_items:
            for _ in range(end - start):
                keys.append(("object", tid))

        return keys

    @staticmethod
    def _indices_for_key(
        mapping: Dict,
        key: Tuple,
    ) -> np.ndarray:
        kind, identifier = key

        if kind == "pose":
            start, end = mapping["ego_poses"][identifier]
        elif kind == "velocity":
            start, end = mapping["ego_velocities"][identifier]
        elif kind == "object":
            start, end = mapping["object_states"][identifier]
        else:
            raise KeyError(f"Unknown state key: {kind}")

        return np.arange(
            start,
            end,
            dtype=int,
        )

    def _unpack_states(
        self,
        flat_vector: np.ndarray,
        mapping: Dict,
    ) -> Tuple[Dict, Dict, Dict]:
        ego_poses = {}
        ego_velocities = {}
        object_states = {}

        for fid, (s, e) in mapping["ego_poses"].items():
            ego_poses[fid] = flat_vector[s:e].copy()

        for fid, (s, e) in mapping["ego_velocities"].items():
            ego_velocities[fid] = flat_vector[s:e].copy()

        for tid, (s, e) in mapping["object_states"].items():
            object_states[tid] = flat_vector[s:e].copy()

        return (
            ego_poses,
            ego_velocities,
            object_states,
        )

    def _prior_indices(
        self,
        factor: MarginalizationFactor,
        mapping: Dict,
    ) -> Optional[np.ndarray]:
        keys = getattr(
            factor,
            "state_keys",
            None,
        )

        if keys is None:
            prior_dim = len(factor.x_lin)

            if prior_dim > sum(
                end - start
                for group in (
                    mapping["ego_poses"],
                    mapping["ego_velocities"],
                    mapping["object_states"],
                )
                for start, end in group.values()
            ):
                return None

            return np.arange(
                prior_dim,
                dtype=int,
            )

        indices = []

        for key in keys:
            if key[0] == "pose":
                if key[1] not in mapping["ego_poses"]:
                    return None
            elif key[0] == "velocity":
                if key[1] not in mapping["ego_velocities"]:
                    return None
            elif key[0] == "object":
                if key[1] not in mapping["object_states"]:
                    return None
            else:
                return None

            indices.extend(
                self._indices_for_key(
                    mapping,
                    key,
                ).tolist()
            )

        indices = np.asarray(
            indices,
            dtype=int,
        )

        if len(indices) != len(factor.x_lin):
            return None

        return indices

    def compute_all_residuals(
        self,
        flat_vector: np.ndarray,
        mapping: Dict,
    ) -> np.ndarray:
        (
            ego_poses,
            ego_velocities,
            object_states,
        ) = self._unpack_states(
            flat_vector,
            mapping,
        )

        residuals = []
        rotations = {fid: self.euler_to_rot(pose[:3]) for fid,pose in ego_poses.items()}
        cache = getattr(self, '_optimization_residual_cache', None)
        def residual(factor, *arguments):
            if cache is None:
                return factor.compute_residual(*arguments)
            key = (id(factor), tuple((np.asarray(arg).shape, np.asarray(arg).dtype.str,
                                     np.asarray(arg).tobytes()) for arg in arguments))
            if key not in cache:
                if len(cache) >= 4096:
                    cache.clear()
                cache[key] = factor.compute_residual(*arguments)
            return cache[key]

        for prior_item in self.pose_prior_factors:
            fid = prior_item["frame_id"]
            if fid in ego_poses:
                r = residual(prior_item["factor"], ego_poses[fid])
                residuals.extend(np.asarray(r).reshape(-1).tolist())

        for velocity_item in self.velocity_prior_factors:
            fid = velocity_item["frame_id"]
            if fid in ego_velocities:
                r = residual(velocity_item["factor"], ego_velocities[fid])
                residuals.extend(np.asarray(r).reshape(-1).tolist())

        for imu_item in self.imu_factors:
            fi = imu_item["frame_i"]
            fj = imu_item["frame_j"]

            if fi in ego_poses and fj in ego_poses:
                pose_i = ego_poses[fi]
                pose_j = ego_poses[fj]

                v_i = ego_velocities[fi]
                v_j = ego_velocities[fj]

                R_i, p_i = (
                    rotations[fi],
                    pose_i[3:],
                )

                R_j, p_j = (
                    rotations[fj],
                    pose_j[3:],
                )

                r = residual(imu_item["factor"],
                    R_i,
                    p_i,
                    v_i,
                    R_j,
                    p_j,
                    v_j,
                )

                residuals.extend(
                    np.asarray(r).reshape(-1).tolist()
                )

        for lidar_rel_item in self.lidar_relative_factors:
            fi = lidar_rel_item["frame_i"]
            fj = lidar_rel_item["frame_j"]
            if fi in ego_poses and fj in ego_poses:
                pose_i = ego_poses[fi]
                pose_j = ego_poses[fj]
                R_i = rotations[fi]
                R_j = rotations[fj]
                r = residual(lidar_rel_item["factor"],
                    R_i, pose_i[3:], R_j, pose_j[3:]
                )
                residuals.extend(np.asarray(r).reshape(-1).tolist())

        for lidar_item in self.lidar_factors:
            fid = lidar_item["frame_id"]

            if fid in ego_poses:
                pose = ego_poses[fid]

                R, p = (
                    rotations[fid],
                    pose[3:],
                )

                r = residual(lidar_item["factor"],
                    R,
                    p,
                )

                residuals.append(float(r))

        for vis_item in self.visual_factors:
            fid = vis_item["frame_id"]

            if fid in ego_poses:
                pose = ego_poses[fid]

                R_e, p_e = (
                    rotations[fid],
                    pose[3:],
                )

                R_c = R_e.T
                t_c = -R_c @ p_e

                r = residual(vis_item["factor"],
                    R_c,
                    t_c,
                )

                residuals.extend(
                    np.asarray(r).reshape(-1).tolist()
                )

        for dyn_item in self.dynamic_factors:
            fid = dyn_item["frame_id"]
            tid = dyn_item["track_id"]

            if fid in ego_poses and tid in object_states:
                pose = ego_poses[fid]
                ostate = object_states[tid]

                R, p = (
                    rotations[fid],
                    pose[3:],
                )

                r = residual(dyn_item["factor"],
                    R,
                    p,
                    ostate[:3],
                    ostate[3:],
                )

                residuals.extend(
                    np.asarray(r).reshape(-1).tolist()
                )

        for marg_factor in self.marginalization_factors:
            indices = self._prior_indices(
                marg_factor,
                mapping,
            )

            if indices is None:
                continue

            current_x = flat_vector[indices]

            r = residual(marg_factor,
                current_x
            )

            residuals.extend(
                np.asarray(r).reshape(-1).tolist()
            )

        if not residuals:
            return np.zeros(
                1,
                dtype=np.float64,
            )

        return np.asarray(
            residuals,
            dtype=np.float64,
        )

    def _residual_structure(
        self,
        mapping: Dict,
    ) -> Tuple[int, List[np.ndarray]]:
        dependencies = []

        def pose_indices(fid):
            if fid not in mapping["ego_poses"]:
                return np.empty(0, dtype=int)
            return self._indices_for_key(
                mapping,
                ("pose", fid),
            )

        def velocity_indices(fid):
            if fid not in mapping["ego_velocities"]:
                return np.empty(0, dtype=int)
            return self._indices_for_key(
                mapping,
                ("velocity", fid),
            )

        def object_indices(tid):
            if tid not in mapping["object_states"]:
                return np.empty(0, dtype=int)
            return self._indices_for_key(
                mapping,
                ("object", tid),
            )

        row_count = 0

        for item in self.pose_prior_factors:
            fid = item["frame_id"]
            if fid in mapping["ego_poses"]:
                dependencies.append((row_count, row_count + 6, pose_indices(fid)))
                row_count += 6

        for item in self.velocity_prior_factors:
            fid = item["frame_id"]
            if fid in mapping["ego_velocities"]:
                dependencies.append((row_count, row_count + 3, velocity_indices(fid)))
                row_count += 3

        for item in self.imu_factors:
            fi = item["frame_i"]
            fj = item["frame_j"]

            if (
                fi in mapping["ego_poses"]
                and fj in mapping["ego_poses"]
            ):
                cols = np.unique(
                    np.concatenate(
                        [
                            pose_indices(fi),
                            velocity_indices(fi),
                            pose_indices(fj),
                            velocity_indices(fj),
                        ]
                    )
                )
                dependencies.append(
                    (row_count, row_count + 9, cols)
                )
                row_count += 9

        for item in self.lidar_relative_factors:
            fi = item["frame_i"]
            fj = item["frame_j"]
            if fi in mapping["ego_poses"] and fj in mapping["ego_poses"]:
                cols = np.unique(np.concatenate([pose_indices(fi), pose_indices(fj)]))
                dependencies.append((row_count, row_count + 6, cols))
                row_count += 6

        for item in self.lidar_factors:
            fid = item["frame_id"]

            if fid in mapping["ego_poses"]:
                dependencies.append(
                    (
                        row_count,
                        row_count + 1,
                        pose_indices(fid),
                    )
                )
                row_count += 1

        for item in self.visual_factors:
            fid = item["frame_id"]

            if fid in mapping["ego_poses"]:
                dependencies.append(
                    (
                        row_count,
                        row_count + 2,
                        pose_indices(fid),
                    )
                )
                row_count += 2

        for item in self.dynamic_factors:
            fid = item["frame_id"]
            tid = item["track_id"]

            if (
                fid in mapping["ego_poses"]
                and tid in mapping["object_states"]
            ):
                cols = np.unique(
                    np.concatenate(
                        [
                            pose_indices(fid),
                            object_indices(tid),
                        ]
                    )
                )

                dependencies.append(
                    (
                        row_count,
                        row_count + 3,
                        cols,
                    )
                )
                row_count += 3

        for factor in self.marginalization_factors:
            indices = self._prior_indices(
                factor,
                mapping,
            )

            if indices is not None:
                prior_dim = len(
                    factor.compute_residual(
                        factor.x_lin
                    )
                )

                dependencies.append(
                    (
                        row_count,
                        row_count + prior_dim,
                        indices,
                    )
                )

                row_count += prior_dim

        return row_count, dependencies

    def build_jacobian_sparsity(
        self,
        mapping: Dict,
        state_size: int,
    ) -> csr_matrix:
        row_count, dependencies = (
            self._residual_structure(mapping)
        )

        sparsity = lil_matrix(
            (row_count, state_size),
            dtype=np.int8,
        )

        for row_start, row_end, cols in dependencies:
            if len(cols) == 0:
                continue

            sparsity[
                row_start:row_end,
                cols,
            ] = 1

        return sparsity.tocsr()

    def numerical_jacobian(
        self,
        flat_vector: np.ndarray,
        mapping: Dict,
        relative_step: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(
            flat_vector,
            dtype=np.float64,
        )

        r0 = self.compute_all_residuals(
            x,
            mapping,
        )

        m = len(r0)
        n = len(x)

        sparsity = self.build_jacobian_sparsity(
            mapping,
            n,
        )

        groups = group_columns(
            sparsity,
        )

        J = np.zeros(
            (m, n),
            dtype=np.float64,
        )

        for group_id in range(
            int(np.max(groups)) + 1
            if len(groups)
            else 0
        ):
            columns = np.flatnonzero(
                groups == group_id
            )

            if len(columns) == 0:
                continue

            xp = x.copy()
            xm = x.copy()
            steps = {}

            for index in columns:
                step = relative_step * max(
                    1.0,
                    abs(x[index]),
                )

                xp[index] += step
                xm[index] -= step
                steps[index] = step

            rp = self.compute_all_residuals(
                xp,
                mapping,
            )

            rm = self.compute_all_residuals(
                xm,
                mapping,
            )

            for index in columns:
                step = steps[index]

                row_indices = (
                    sparsity[:, index]
                    .nonzero()[0]
                )

                if len(row_indices):
                    J[
                        row_indices,
                        index,
                    ] = (
                        rp[row_indices]
                        - rm[row_indices]
                    ) / (2.0 * step)

        return r0, J

    def build_marginalization_system(
        self,
        flat_vector: np.ndarray,
        mapping: Dict,
        eliminated_indices: np.ndarray,
        residuals: Optional[np.ndarray] = None,
        jacobian: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        # Reuse the optimizer's final residual/Jacobian when available.
        if residuals is None or jacobian is None:
            r, J = self.numerical_jacobian(
                flat_vector,
                mapping,
            )
        else:
            r = np.asarray(
                residuals,
                dtype=np.float64,
            ).reshape(-1)

            if hasattr(jacobian, "toarray"):
                J = jacobian.toarray()
            else:
                J = np.asarray(
                    jacobian,
                    dtype=np.float64,
                )

            if J.ndim != 2 or J.shape[1] != len(flat_vector):
                raise ValueError(
                    "Cached Jacobian does not match the current state vector."
                )

            if J.shape[0] != len(r):
                raise ValueError(
                    "Cached Jacobian and residual dimensions do not match."
                )

        all_indices = np.arange(
            len(flat_vector)
        )

        keep_mask = np.ones(
            len(flat_vector),
            dtype=bool,
        )

        keep_mask[
            np.asarray(
                eliminated_indices,
                dtype=int,
            )
        ] = False

        remaining_indices = all_indices[
            keep_mask
        ]

        m = np.asarray(
            eliminated_indices,
            dtype=int,
        )

        r_idx = remaining_indices

        # Form only the blocks required by the Schur complement.
        Jm = J[:, m]
        Jr = J[:, r_idx]

        Hmm = Jm.T @ Jm
        Hmr = Jm.T @ Jr
        Hrm = Hmr.T
        Hrr = Jr.T @ Jr

        gm = Jm.T @ r
        gr = Jr.T @ r

        diagonal = np.diag(Hmm)

        if len(diagonal):
            scale = max(
                1.0,
                float(
                    np.max(
                        np.abs(
                            diagonal
                        )
                    )
                ),
            )
        else:
            scale = 1.0

        regularization = (
            1e-8 * scale
        )

        Hmm_reg = Hmm + (
            regularization
            * np.eye(
                len(m),
                dtype=np.float64,
            )
        )

        try:
            Hmm_inv_Hmr = np.linalg.solve(
                Hmm_reg,
                Hmr,
            )

            Hmm_inv_gm = np.linalg.solve(
                Hmm_reg,
                gm,
            )
        except np.linalg.LinAlgError:
            Hmm_inv_Hmr = (
                np.linalg.pinv(
                    Hmm_reg
                )
                @ Hmr
            )

            Hmm_inv_gm = (
                np.linalg.pinv(
                    Hmm_reg
                )
                @ gm
            )

        H_prior = (
            Hrr
            - Hrm @ Hmm_inv_Hmr
        )

        b_prior = (
            gr
            - Hrm @ Hmm_inv_gm
        )

        H_prior = 0.5 * (
            H_prior + H_prior.T
        )

        x_remaining = flat_vector[
            remaining_indices
        ].copy()

        return (
            x_remaining,
            H_prior,
            b_prior,
        )

    def retain_keyframes(
        self,
        remaining_fids: List[int],
    ):
        remaining_set = set(
            remaining_fids
        )

        self.pose_prior_factors = [
            item for item in self.pose_prior_factors
            if item["frame_id"] in remaining_set
        ]

        self.velocity_prior_factors = [
            item for item in self.velocity_prior_factors
            if item["frame_id"] in remaining_set
        ]

        self.imu_factors = [
            item
            for item in self.imu_factors
            if (
                item["frame_i"] in remaining_set
                and item["frame_j"] in remaining_set
            )
        ]

        self.lidar_relative_factors = [
            item
            for item in self.lidar_relative_factors
            if (
                item["frame_i"] in remaining_set
                and item["frame_j"] in remaining_set
            )
        ]

        self.lidar_factors = [
            item
            for item in self.lidar_factors
            if item["frame_id"] in remaining_set
        ]

        self.visual_factors = [
            item
            for item in self.visual_factors
            if item["frame_id"] in remaining_set
        ]

        self.dynamic_factors = [
            item
            for item in self.dynamic_factors
            if item["frame_id"] in remaining_set
        ]

        self.keyframe_ids = [
            fid
            for fid in self.keyframe_ids
            if fid in remaining_set
        ]

        self.timestamps = {
            fid: self.timestamps[fid]
            for fid in self.keyframe_ids
            if fid in self.timestamps
        }

    def optimize(
        self,
        initial_poses: Dict[int, np.ndarray],
        initial_velocities: Dict[int, np.ndarray],
        initial_objects: Optional[
            Dict[int, np.ndarray]
        ] = None,
        max_nfev: int = 25,
    ) -> Dict:
        initial_objects = (
            initial_objects or {}
        )

        x0, mapping = self._pack_states(
            initial_poses,
            initial_velocities,
            initial_objects,
        )

        if len(x0) == 0:
            return {
                "success": False,
                "cost": 0.0,
                "optimized_poses": {},
                "optimized_velocities": {},
                "optimized_objects": {},
            }

        def loss_func(x):
            return self.compute_all_residuals(
                x,
                mapping,
            )

        jac_sparsity = (
            self.build_jacobian_sparsity(
                mapping,
                len(x0),
            )
        )

        # Numerical differentiation repeatedly asks factors about identical
        # states. Reuse exact byte-identical inputs within this optimization;
        # the objective, finite differences and solver settings are unchanged.
        self._optimization_residual_cache = {}
        try:
            res = least_squares(
                loss_func,
                x0,
                jac="2-point",
                jac_sparsity=jac_sparsity,
                method="trf",
                loss=self.robust_loss,
                f_scale=self.loss_scale,
                ftol=1e-3,
                xtol=1e-3,
                gtol=1e-3,
                max_nfev=max_nfev,
            )
        finally:
            self._optimization_residual_cache = None

        (
            opt_poses,
            opt_vels,
            opt_objs,
        ) = self._unpack_states(
            res.x,
            mapping,
        )

        return {
            "success": bool(res.success),
            "status": int(res.status),
            "cost": float(res.cost),
            "optimized_poses": opt_poses,
            "optimized_velocities": opt_vels,
            "optimized_objects": opt_objs,
            "optimized_vector": res.x,
            "mapping": mapping,
            "optimized_residuals": np.asarray(
                res.fun, dtype=np.float64
            ).copy(),
            "optimized_jacobian": res.jac.copy()
            if hasattr(res.jac, "copy")
            else np.asarray(
                res.jac, dtype=np.float64
            ).copy(),
        }




import numpy as np
from typing import Dict, List, Optional



class CarlaSlidingWindowEstimator:
    def __init__(
        self,
        window_size: int = 8,
        robust_loss: str = "cauchy",
        loss_scale: float = 1.0,
        max_lidar_factors_per_frame: int = 48,
        max_visual_factors_per_frame: int = 40,
        max_object_factors_per_frame: int = 24,
        max_objects: int = 40,
        optimizer_max_nfev: int = 10,
    ):
        self.window_size = int(window_size)
        self.robust_loss = robust_loss
        self.loss_scale = loss_scale
        self.max_lidar_factors_per_frame = int(max_lidar_factors_per_frame)
        self.max_visual_factors_per_frame = int(max_visual_factors_per_frame)
        self.max_object_factors_per_frame = int(max_object_factors_per_frame)
        self.max_objects = int(max_objects)
        self.optimizer_max_nfev = max(2, int(optimizer_max_nfev))

        self.factor_graph = CarlaFactorGraph(
            robust_loss=robust_loss,
            loss_scale=loss_scale,
        )

        self.estimated_poses: Dict[
            int,
            np.ndarray,
        ] = {}

        self.estimated_velocities: Dict[
            int,
            np.ndarray,
        ] = {}

        self.estimated_objects: Dict[
            int,
            np.ndarray,
        ] = {}

        self.object_reference_timestamps: Dict[int, float] = {}

        self.current_prior: Optional[
            MarginalizationFactor
        ] = None

    def add_keyframe_data(
        self,
        frame_id: int,
        timestamp: float,
        init_pose: np.ndarray,
        init_vel: np.ndarray,
        imu_preintegration: Optional[Dict] = None,
        inertial_pose_prior: Optional[np.ndarray] = None,
        inertial_pose_weight_position: float = 45.0,
        inertial_pose_weight_rotation: float = 25.0,
        planar_features: Optional[List[Dict]] = None,
        lidar_relative_pose: Optional[Dict] = None,
        visual_observations: Optional[List[Dict]] = None,
        object_observations: Optional[List[Dict]] = None,
        prev_frame_id: Optional[int] = None,
        stationary_pose_prior: Optional[np.ndarray] = None,
        stationary_velocity_prior: Optional[np.ndarray] = None,
        stationary_pose_weight: float = 20.0,
        stationary_velocity_weight: float = 20.0,
    ):
        self.factor_graph.add_keyframe(
            frame_id,
            timestamp,
        )

        self.estimated_poses[
            frame_id
        ] = np.asarray(
            init_pose,
            dtype=np.float64,
        ).copy()

        self.estimated_velocities[
            frame_id
        ] = np.asarray(
            init_vel,
            dtype=np.float64,
        ).copy()

        if len(self.factor_graph.keyframe_ids) == 1 and not self.factor_graph.pose_prior_factors:
            self.factor_graph.add_pose_prior(frame_id, np.asarray(init_pose, dtype=np.float64))

        if inertial_pose_prior is not None:
            # Independent IMU + compass absolute-initial-frame metric prior.
            # Keeps weak LiDAR/visual correspondences from compressing travel.
            self.factor_graph.add_pose_prior(
                frame_id, np.asarray(inertial_pose_prior, dtype=np.float64),
                weight_rotation=float(inertial_pose_weight_rotation),
                weight_position=float(inertial_pose_weight_position),
            )

        if stationary_pose_prior is not None:
            self.factor_graph.add_pose_prior(
                frame_id,
                np.asarray(stationary_pose_prior, dtype=np.float64),
                weight_rotation=float(stationary_pose_weight),
                weight_position=float(stationary_pose_weight),
            )

        if stationary_velocity_prior is not None:
            self.factor_graph.add_velocity_prior(
                frame_id,
                np.asarray(stationary_velocity_prior, dtype=np.float64),
                weight=float(stationary_velocity_weight),
            )

        if (
            prev_frame_id is not None
            and imu_preintegration is not None
        ):
            dt = imu_preintegration.get(
                "dt",
                0.1,
            )

            dp = imu_preintegration.get(
                "delta_p",
                np.zeros(3),
            )

            dv = imu_preintegration.get(
                "delta_v",
                np.zeros(3),
            )

            dR = imu_preintegration.get(
                "delta_R",
                np.eye(3),
            )

            self.factor_graph.add_imu_factor(
                prev_frame_id,
                frame_id,
                dp,
                dv,
                dR,
                dt,
            )

        if (
            prev_frame_id is not None
            and lidar_relative_pose is not None
            and bool(lidar_relative_pose.get("accepted", False))
        ):
            self.factor_graph.add_lidar_relative_factor(
                prev_frame_id,
                frame_id,
                np.asarray(lidar_relative_pose.get("delta_R", np.eye(3)), dtype=np.float64),
                np.asarray(lidar_relative_pose.get("delta_t", np.zeros(3)), dtype=np.float64),
                weight_translation=float(lidar_relative_pose.get("weight_translation", 8.0)),
                weight_rotation=float(lidar_relative_pose.get("weight_rotation", 12.0)),
            )

        if planar_features:
            for feat in planar_features[: self.max_lidar_factors_per_frame]:
                self.factor_graph.add_lidar_factor(
                    frame_id=frame_id,
                    point_lidar=feat.get(
                        "point",
                        feat.get(
                            "center",
                            np.zeros(3),
                        ),
                    ),
                    plane_normal_w=feat.get(
                        "normal",
                        np.array(
                            [0.0, 0.0, 1.0]
                        ),
                    ),
                    plane_center_w=feat.get(
                        "center",
                        np.zeros(3),
                    ),
                    weight=3.0,
                )

        if visual_observations:
            for obs in visual_observations[: self.max_visual_factors_per_frame]:
                measurement_uv = obs.get(
                    "measurement_uv",
                    obs.get(
                        "uv",
                        np.zeros(2),
                    ),
                )

                K = obs.get(
                    "K",
                    np.eye(3),
                )

                landmark_3d_w = obs.get(
                    "landmark_3d_w",
                    obs.get(
                        "landmark",
                        np.zeros(3),
                    ),
                )

                weight = obs.get(
                    "weight",
                    1.0,
                )

                self.factor_graph.add_visual_factor(
                    frame_id=frame_id,
                    measurement_uv=measurement_uv,
                    K=K,
                    landmark_3d_w=landmark_3d_w,
                    weight=weight,
                    camera_translation_ego=obs.get("camera_translation_ego"),
                    camera_rotation_ego=obs.get("camera_rotation_ego"),
                    use_carla_camera_axes=bool(obs.get("use_carla_camera_axes", False)),
                )

        if object_observations:
            object_observations = list(object_observations)[: self.max_object_factors_per_frame]
            for obj in object_observations:
                tid = obj.get(
                    "track_id",
                    0,
                )

                if tid not in self.estimated_objects and len(self.estimated_objects) < self.max_objects:
                    # obs_local is in the ego frame; the factor predicts
                    # WORLD positions. Initialize its state in WORLD coordinates.
                    local = np.asarray(obj.get("location", np.zeros(3)), dtype=np.float64).reshape(3)
                    pose = self.estimated_poses[frame_id]
                    R = self.factor_graph.euler_to_rot(pose[:3])
                    p0 = R @ local + pose[3:6]
                    v0 = np.asarray(obj.get("velocity", np.zeros(3)), dtype=np.float64).reshape(3)
                    self.estimated_objects[tid] = np.hstack([p0, v0])
                    self.object_reference_timestamps[tid] = float(timestamp)

                if tid not in self.estimated_objects:
                    continue

                loc = obj.get(
                    "location",
                    np.zeros(3),
                )

                dt_obj = float(timestamp) - self.object_reference_timestamps.get(tid, float(timestamp))

                self.factor_graph.add_dynamic_object_factor(
                    frame_id=frame_id,
                    track_id=tid,
                    obs_local=loc,
                    dt_from_origin=dt_obj,
                    weight_pos=4.0,
                )

        if (
            len(
                self.factor_graph.keyframe_ids
            )
            >= self.window_size
        ):
            self.optimize_and_marginalize()

    def marginalize_oldest_state(
        self,
        optimized_vector: np.ndarray,
        mapping: Dict,
        optimized_residuals: Optional[np.ndarray] = None,
        optimized_jacobian: Optional[np.ndarray] = None,
    ):
        if len(
            self.factor_graph.keyframe_ids
        ) < 2:
            return

        oldest_fid = (
            self.factor_graph.keyframe_ids[0]
        )

        remaining_fids = (
            self.factor_graph.keyframe_ids[1:]
        )

        pose_start, pose_end = mapping[
            "ego_poses"
        ][oldest_fid]

        vel_start, vel_end = mapping[
            "ego_velocities"
        ][oldest_fid]

        eliminated_indices = np.concatenate(
            [
                np.arange(
                    pose_start,
                    pose_end,
                    dtype=int,
                ),
                np.arange(
                    vel_start,
                    vel_end,
                    dtype=int,
                ),
            ]
        )

        (
            x_remaining,
            H_prior,
            b_prior,
        ) = self.factor_graph.build_marginalization_system(
            optimized_vector,
            mapping,
            eliminated_indices,
            residuals=optimized_residuals,
            jacobian=optimized_jacobian,
        )

        self.current_prior = MarginalizationFactor(
            linearization_point=x_remaining,
            H_prior=H_prior,
            b_prior=b_prior,
        )

        remaining_keys = []

        for fid in remaining_fids:
            remaining_keys.append(("pose", fid))
            remaining_keys.append(("velocity", fid))

        for tid in sorted(
            self.estimated_objects.keys()
        ):
            remaining_keys.append(("object", tid))

        self.current_prior.state_keys = (
            remaining_keys
        )

        self.factor_graph.retain_keyframes(
            remaining_fids
        )

        self.factor_graph.marginalization_factors.clear()

        self.factor_graph.add_marginalization_factor(
            self.current_prior
        )

        self.estimated_poses = {
            fid: self.estimated_poses[fid]
            for fid in remaining_fids
            if fid in self.estimated_poses
        }

        self.estimated_velocities = {
            fid: self.estimated_velocities[fid]
            for fid in remaining_fids
            if fid in self.estimated_velocities
        }

        active_tids = {int(item["track_id"]) for item in self.factor_graph.dynamic_factors}
        self.estimated_objects = {
            tid: state for tid, state in self.estimated_objects.items()
            if tid in active_tids
        }
        self.object_reference_timestamps = {
            tid: ts for tid, ts in self.object_reference_timestamps.items()
            if tid in self.estimated_objects
        }

    def optimize_and_marginalize(self) -> Dict:
        res = self.factor_graph.optimize(
            initial_poses=self.estimated_poses,
            initial_velocities=self.estimated_velocities,
            initial_objects=self.estimated_objects,
            max_nfev=self.optimizer_max_nfev,
        )

        if "optimized_vector" not in res or not np.isfinite(float(res.get("cost", np.inf))):
            return res

        self.estimated_poses.update(
            res["optimized_poses"]
        )

        self.estimated_velocities.update(
            res["optimized_velocities"]
        )

        self.estimated_objects.update(
            res["optimized_objects"]
        )

        if (
            len(
                self.factor_graph.keyframe_ids
            )
            >= self.window_size
        ):
            self.marginalize_oldest_state(
                res["optimized_vector"],
                res["mapping"],
                res.get("optimized_residuals"),
                res.get("optimized_jacobian"),
            )

        return res

    def diagnostics(self) -> Dict:
        fg = self.factor_graph
        return {
            "window_keyframes": int(len(fg.keyframe_ids)),
            "pose_prior_factors": int(len(fg.pose_prior_factors)),
            "velocity_prior_factors": int(len(fg.velocity_prior_factors)),
            "imu_factors": int(len(fg.imu_factors)),
            "lidar_factors": int(len(fg.lidar_factors)),
            "lidar_relative_factors": int(len(fg.lidar_relative_factors)),
            "visual_factors": int(len(fg.visual_factors)),
            "dynamic_factors": int(len(fg.dynamic_factors)),
            "marginalization_factors": int(len(fg.marginalization_factors)),
            "object_states": int(len(self.estimated_objects)),
            "optimizer_max_nfev": int(self.optimizer_max_nfev),
        }

    def get_current_estimate(
        self,
        frame_id: int,
    ) -> Optional[Dict]:
        if frame_id not in self.estimated_poses:
            return None

        timestamp = float(self.factor_graph.timestamps.get(frame_id, 0.0))
        objects_current = {}
        for tid, state in self.estimated_objects.items():
            state = np.asarray(state, dtype=np.float64).reshape(-1)
            if state.size != 6 or not np.isfinite(state).all():
                continue
            dt = timestamp - float(self.object_reference_timestamps.get(tid, timestamp))
            pos = state[:3] + state[3:6] * dt
            if np.isfinite(pos).all():
                objects_current[int(tid)] = {
                    "position": pos.tolist(),
                    "velocity": state[3:6].tolist(),
                    "reference_timestamp": float(self.object_reference_timestamps.get(tid, timestamp)),
                    "timestamp": timestamp,
                    "source": "temporal_fusion_fgo",
                }

        return {
            "frame_id": frame_id,
            "objects_current": objects_current,
            "pose": self.estimated_poses[
                frame_id
            ].tolist(),
            "velocity": self.estimated_velocities.get(
                frame_id,
                np.zeros(3),
            ).tolist(),
            "objects": {
                k: v.tolist()
                for k, v in self.estimated_objects.items()
            },
        }