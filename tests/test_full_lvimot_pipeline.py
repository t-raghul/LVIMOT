import os
import sys
import numpy as np


from lvimot.tracking_4d.multimodal_object_tracking.kalman_tracker import CarlaMultiObjectTracker
from lvimot.temporal_fusion.static_dynamic_classification import CarlaMotionClassifier
from lvimot.tracking_4d.static_scene_support.feature_correspondences import CarlaTemporalVisualTracker
from lvimot.tracking_4d.static_scene_support.feature_correspondences import CarlaTemporalLiDARMatcher
from lvimot.temporal_fusion.sliding_window_factor_graph import CarlaSlidingWindowEstimator
from lvimot.mapping_4d.static_map.rolling_voxel_map import Carla4DEnvironmentMapper
from lvimot.output.evaluation.metrics import CarlaEvaluator


def world_to_ego(point_world, ego_pose):
    point_world = np.asarray(point_world, dtype=np.float64)

    loc = ego_pose["location"]
    translation = np.array(
        [
            float(loc["x"]),
            float(loc["y"]),
            float(loc["z"]),
        ],
        dtype=np.float64,
    )

    rotation = ego_pose["rotation"]

    pitch = np.deg2rad(float(rotation.get("pitch", 0.0)))
    yaw = np.deg2rad(float(rotation.get("yaw", 0.0)))
    roll = np.deg2rad(float(rotation.get("roll", 0.0)))

    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cr, sr = np.cos(roll), np.sin(roll)

    R_world_ego = np.array(
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

    return R_world_ego.T @ (point_world - translation)


def make_tracker_observations(world_objects, ego_pose):
    observations = []

    for obj in world_objects:
        item = dict(obj)
        item["location"] = world_to_ego(
            obj["location"],
            ego_pose,
        )
        observations.append(item)

    return observations


def test_full_lvimot_system_simulation():
    classifier = CarlaMotionClassifier()

    tracker = CarlaMultiObjectTracker(
        distance_threshold=5.0,
        max_missed=3,
        use_ego_compensation=True,
        motion_classifier=classifier,
    )

    vis_tracker = CarlaTemporalVisualTracker(
        max_hamming_distance=40,
        ratio_threshold=0.85,
    )

    lidar_matcher = CarlaTemporalLiDARMatcher(
        max_center_distance=1.0,
    )

    sliding_window = CarlaSlidingWindowEstimator(
        window_size=5,
    )

    mapper_4d = Carla4DEnvironmentMapper(
        static_voxel_size=0.5,
    )

    evaluator = CarlaEvaluator()

    all_tracks = []
    all_gts = []
    all_est = []
    all_gt_poses = []

    dt = 0.1

    for f in range(10):
        t = f * dt
        ego_x = 10.0 * t

        ego_pose = {
            "location": {
                "x": ego_x,
                "y": 0.0,
                "z": 0.0,
            },
            "rotation": {
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
            },
        }

        gt_pose = np.array(
            [
                0.0,
                0.0,
                0.0,
                ego_x,
                0.0,
                0.0,
            ]
        )

        all_gt_poses.append(gt_pose)

        world_obs = [
            {
                "actor_id": 1,
                "type": "vehicle",
                "location": [
                    ego_x + 15.0,
                    0.0,
                    0.0,
                ],
                "velocity": [
                    15.0,
                    0.0,
                    0.0,
                ],
                "dimensions": {
                    "length": 4.5,
                    "width": 1.8,
                    "height": 1.5,
                },
                "rotation": {
                    "pitch": 0,
                    "yaw": 0,
                    "roll": 0,
                },
            },
            {
                "actor_id": 2,
                "type": "vehicle",
                "location": [
                    30.0,
                    5.0,
                    0.0,
                ],
                "velocity": [
                    0.0,
                    0.0,
                    0.0,
                ],
                "dimensions": {
                    "length": 4.2,
                    "width": 1.8,
                    "height": 1.4,
                },
                "rotation": {
                    "pitch": 0,
                    "yaw": 0,
                    "roll": 0,
                },
            },
        ]

        # Ground truth remains in world coordinates for evaluation.
        all_gts.append(world_obs)

        # The production CARLA pipeline converts world -> ego
        # before calling the tracker. The synthetic test now follows
        # the same contract.
        tracker_observations = make_tracker_observations(
            world_obs,
            ego_pose,
        )

        active = tracker.update(
            tracker_observations,
            timestamp=t,
            ego_pose=ego_pose,
        )

        all_tracks.append(active)

        pts = np.array(
            [
                [100 + f * 2, 200],
                [300 + f * 2, 250],
            ],
            dtype=np.float64,
        )

        np.random.seed(100)

        desc = np.random.randint(
            0,
            256,
            size=(2, 32),
            dtype=np.uint8,
        )

        vis_tracker.update(
            pts,
            desc,
            frame_id=f,
            timestamp=t,
        )

        planar = [
            {
                "center": np.array(
                    [
                        ego_x + 10.0,
                        10.0,
                        2.0,
                    ]
                ),
                "normal": np.array(
                    [
                        0.0,
                        1.0,
                        0.0,
                    ]
                ),
                "planarity": 0.05,
            }
        ]

        lidar_matcher.update(
            planar,
            frame_id=f,
            timestamp=t,
        )

        imu_preint = (
            {
                "dt": dt,
                "delta_p": np.array(
                    [
                        1.0,
                        0.0,
                        0.0,
                    ]
                ),
                "delta_v": np.zeros(3),
                "delta_R": np.eye(3),
            }
            if f > 0
            else None
        )

        init_est = (
            gt_pose
            + np.random.normal(
                0,
                0.02,
                size=6,
            )
        )

        sliding_window.add_keyframe_data(
            f,
            t,
            init_est,
            np.array(
                [
                    10.0,
                    0.0,
                    0.0,
                ]
            ),
            imu_preintegration=imu_preint,
            planar_features=planar,
            object_observations=tracker_observations,
            prev_frame_id=f - 1 if f > 0 else None,
        )

        all_est.append(init_est)

        lidar_pts = np.array(
            [
                [ego_x + 5.0, 10.0, 0.0],
                [ego_x + 15.0, -10.0, 0.0],
            ]
        )

        mapper_4d.add_frame_data(
            f,
            t,
            lidar_pts,
            planar,
            active,
            ego_pose,
        )

    mot_res = evaluator.evaluate_tracking_sequence(
        all_tracks,
        all_gts,
    )

    assert mot_res["mota"] >= 0.95

    loc_res = evaluator.evaluate_localization(
        all_est,
        all_gt_poses,
    )

    assert loc_res["ate_rmse"] < 0.2