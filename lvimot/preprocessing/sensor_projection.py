
# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))

import numpy as np


class CarlaBoundingBoxProjector:

    def __init__(self, calibration):
        self.calibration = calibration
        self.K = calibration.get_camera_matrix()

        if self.K is None:
            raise ValueError(
                "Camera intrinsic matrix is missing."
            )

    def _rotation_matrix(self, rotation):
        """
        CARLA rotation convention.

        rotation contains degrees:
            roll
            pitch
            yaw
        """

        roll = np.deg2rad(
            float(rotation.get("roll", 0.0))
        )
        pitch = np.deg2rad(
            float(rotation.get("pitch", 0.0))
        )
        yaw = np.deg2rad(
            float(rotation.get("yaw", 0.0))
        )

        cr = np.cos(roll)
        sr = np.sin(roll)

        cp = np.cos(pitch)
        sp = np.sin(pitch)

        cy = np.cos(yaw)
        sy = np.sin(yaw)

        R = np.array([
            [
                cp * cy,
                cy * sp * sr - sy * cr,
                -cy * sp * cr - sy * sr
            ],
            [
                cp * sy,
                sy * sp * sr + cy * cr,
                -sy * sp * cr + cy * sr
            ],
            [
                sp,
                -cp * sr,
                cp * cr
            ]
        ], dtype=np.float64)

        return R

    def world_to_ego(self, points, ego_pose):
        """
        Transform world coordinates into the ego-vehicle frame.
        """

        location = ego_pose["location"]

        ego_translation = np.array([
            float(location["x"]),
            float(location["y"]),
            float(location["z"])
        ], dtype=np.float64)

        R_world_ego = self._rotation_matrix(
            ego_pose["rotation"]
        )

        points = np.asarray(
            points,
            dtype=np.float64
        )

        # Inverse rigid transform:
        #
        # p_ego = R^T (p_world - t)

        return (
            R_world_ego.T
            @ (points - ego_translation).T
        ).T

    def ego_to_camera(self, points):
        """
        Transform ego coordinates into CARLA camera
        sensor coordinates.

        Camera transform is relative to the ego vehicle.
        """

        camera_transform = self.calibration.camera_transform

        location = camera_transform["location"]

        camera_translation = np.array([
            float(location["x"]),
            float(location["y"]),
            float(location["z"])
        ], dtype=np.float64)

        camera_rotation = camera_transform.get(
            "rotation",
            {}
        )

        R_ego_camera = self._rotation_matrix(
            camera_rotation
        )

        points = np.asarray(
            points,
            dtype=np.float64
        )

        return (
            R_ego_camera.T
            @ (points - camera_translation).T
        ).T

    def carla_camera_to_opencv(self, points):
        """
        CARLA camera coordinates:

            X = forward
            Y = right
            Z = up

        OpenCV camera coordinates:

            X = right
            Y = down
            Z = forward
        """

        points = np.asarray(
            points,
            dtype=np.float64
        )

        return np.column_stack([
            points[:, 1],
            -points[:, 2],
            points[:, 0]
        ])

    def project_world_corners(
        self,
        world_corners,
        ego_pose,
        image_shape
    ):
        """
        Project world-space CARLA bounding-box corners
        into image pixels.

        Returns:
            projected
            valid
            bbox
        """

        corners = np.asarray(
            world_corners,
            dtype=np.float64
        )

        if corners.ndim != 2 or corners.shape[1] != 3:
            raise ValueError(
                "world_corners must have shape (N, 3)."
            )

        # --------------------------------------------------
        # WORLD -> EGO
        # --------------------------------------------------

        ego_points = self.world_to_ego(
            corners,
            ego_pose
        )

        # --------------------------------------------------
        # EGO -> CAMERA
        # --------------------------------------------------

        camera_points = self.ego_to_camera(
            ego_points
        )

        # --------------------------------------------------
        # CARLA CAMERA -> OPENCV
        # --------------------------------------------------

        cv_points = self.carla_camera_to_opencv(
            camera_points
        )

        depth = cv_points[:, 2]

        valid = (
            np.isfinite(cv_points).all(axis=1)
            & (depth > 0)
        )

        projected = np.full(
            (len(corners), 2),
            np.nan,
            dtype=np.float64
        )

        if not np.any(valid):
            return projected, valid, None

        points = cv_points[valid]

        homogeneous = (
            self.K @ points.T
        ).T

        uv = np.column_stack([
            homogeneous[:, 0] / homogeneous[:, 2],
            homogeneous[:, 1] / homogeneous[:, 2]
        ])

        projected[valid] = uv

        # --------------------------------------------------
        # Enclosing image bounding box
        # --------------------------------------------------

        height = image_shape[0]
        width = image_shape[1]

        x1 = np.min(uv[:, 0])
        y1 = np.min(uv[:, 1])
        x2 = np.max(uv[:, 0])
        y2 = np.max(uv[:, 1])

        # Reject boxes that don't intersect image.
        if (
            x2 < 0
            or y2 < 0
            or x1 >= width
            or y1 >= height
        ):
            return projected, valid, None

        x1 = max(0.0, min(float(width - 1), x1))
        y1 = max(0.0, min(float(height - 1), y1))
        x2 = max(0.0, min(float(width - 1), x2))
        y2 = max(0.0, min(float(height - 1), y2))

        if x2 <= x1 or y2 <= y1:
            return projected, valid, None

        bbox = [
            float(x1),
            float(y1),
            float(x2),
            float(y2)
        ]

        return projected, valid, bbox




import numpy as np


class CarlaLiDARCameraProjector:

    def __init__(self, calibration):
        """
        LiDAR -> camera image projection for CARLA.

        Parameters
        ----------
        calibration : CarlaCalibration
            CARLA calibration object.
        """

        self.calibration = calibration
        self.K = calibration.get_camera_matrix()

        if self.K is None:
            raise ValueError("Camera intrinsic matrix is missing.")

    def project(self, lidar_points, image_shape):
        """
        Project CARLA LiDAR points onto the camera image.

        Parameters
        ----------
        lidar_points : numpy.ndarray
            LiDAR points with shape (N, 3) or (N, 4).

        image_shape : tuple
            Image shape: (height, width, channels).

        Returns
        -------
        projected_points : numpy.ndarray
            Image coordinates with shape (M, 2).

        depths : numpy.ndarray
            Camera depth for each projected point.

        valid_lidar_points : numpy.ndarray
            Original LiDAR XYZ points corresponding to
            projected points.

        valid_indices : numpy.ndarray
            Indices of the original LiDAR points.
        """

        points = np.asarray(
            lidar_points,
            dtype=np.float64
        )

        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(
                "LiDAR points must have shape (N, >=3)."
            )

        xyz = points[:, :3]

        # --------------------------------------------------
        # Remove invalid points
        # --------------------------------------------------

        finite = np.isfinite(xyz).all(axis=1)

        if not np.any(finite):
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0,), dtype=np.int64)
            )

        original_indices = np.nonzero(finite)[0]
        valid_xyz = xyz[finite]

        # --------------------------------------------------
        # LiDAR -> OpenCV camera coordinates
        # --------------------------------------------------

        camera_points = self.calibration.lidar_to_opencv(
            valid_xyz
        )

        # --------------------------------------------------
        # Keep points in front of camera
        # --------------------------------------------------

        positive_depth = camera_points[:, 2] > 0

        if not np.any(positive_depth):
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0,), dtype=np.int64)
            )

        camera_points = camera_points[positive_depth]
        valid_xyz = valid_xyz[positive_depth]
        original_indices = original_indices[positive_depth]

        # --------------------------------------------------
        # Perspective projection
        # --------------------------------------------------

        homogeneous = (
            self.K @ camera_points.T
        ).T

        depths = camera_points[:, 2]

        u = (
            homogeneous[:, 0]
            / depths
        )

        v = (
            homogeneous[:, 1]
            / depths
        )

        # --------------------------------------------------
        # Keep points inside image
        # --------------------------------------------------

        height = image_shape[0]
        width = image_shape[1]

        inside = (
            (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )

        projected_points = np.column_stack(
            (
                u[inside],
                v[inside]
            )
        )

        depths = depths[inside]
        valid_lidar_points = valid_xyz[inside]
        valid_indices = original_indices[inside]

        return (
            projected_points,
            depths,
            valid_lidar_points,
            valid_indices
        )




import numpy as np


class CarlaLiDARTransform:

    def __init__(self, calibration):
        self.calibration = calibration

    def rotation_matrix(self, rotation):
        roll = np.deg2rad(float(rotation.get("roll", 0.0)))
        pitch = np.deg2rad(float(rotation.get("pitch", 0.0)))
        yaw = np.deg2rad(float(rotation.get("yaw", 0.0)))

        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)

        return np.array([
            [
                cp * cy,
                cy * sp * sr - sy * cr,
                -cy * sp * cr - sy * sr
            ],
            [
                cp * sy,
                sy * sp * sr + cy * cr,
                -sy * sp * cr + cy * sr
            ],
            [
                sp,
                -cp * sr,
                cp * cr
            ]
        ], dtype=np.float64)

    def lidar_to_ego(self, points):
        """
        LiDAR sensor coordinates -> ego vehicle coordinates.

        CARLA calibration contains the LiDAR sensor transform
        relative to the ego vehicle.
        """

        points = np.asarray(points, dtype=np.float64)

        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(
                "points must have shape (N, >=3)"
            )

        lidar_transform = self.calibration.lidar_transform

        translation = np.array([
            float(lidar_transform["location"]["x"]),
            float(lidar_transform["location"]["y"]),
            float(lidar_transform["location"]["z"])
        ])

        rotation = lidar_transform.get(
            "rotation",
            {}
        )

        R = self.rotation_matrix(rotation)

        xyz = points[:, :3]

        # LiDAR local -> ego coordinates
        ego = (
            R @ xyz.T
        ).T + translation

        return ego

    def ego_to_world(self, points, ego_pose):
        """
        Ego vehicle coordinates -> world coordinates.
        """

        points = np.asarray(
            points,
            dtype=np.float64
        )

        location = np.array([
            float(ego_pose["location"]["x"]),
            float(ego_pose["location"]["y"]),
            float(ego_pose["location"]["z"])
        ])

        R = self.rotation_matrix(
            ego_pose["rotation"]
        )

        world = (
            R @ points.T
        ).T + location

        return world

    def lidar_to_world(self, points, ego_pose):
        """
        LiDAR sensor coordinates -> world coordinates.
        """

        ego_points = self.lidar_to_ego(points)

        return self.ego_to_world(
            ego_points,
            ego_pose
        )
