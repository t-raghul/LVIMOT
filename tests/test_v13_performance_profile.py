import os
import sys

import numpy as np


from lvimot.preprocessing.lidar_planar_features import CarlaLiDARPlanarFeatureExtractor
from lvimot.temporal_fusion.runtime_limits import DEFAULT_LIMITS
from lvimot.mapping_4d.static_map.rolling_voxel_map import Carla4DEnvironmentMapper


def test_v13_runtime_profile_targets_cpu_bottlenecks():
    assert DEFAULT_LIMITS.max_lidar_planar_candidates == 16000
    assert DEFAULT_LIMITS.max_lidar_planar_queries <= 6000
    assert DEFAULT_LIMITS.map_point_stride >= 8
    assert DEFAULT_LIMITS.map_static_update_interval >= 2
    assert DEFAULT_LIMITS.map_evict_interval >= 20


def test_v13_planar_queries_keep_dense_support_neighbors():
    # Dense plane: a small query subset should still find planarity because
    # neighborhoods are searched against the complete voxel support cloud.
    xs, ys = np.meshgrid(np.linspace(-2.0, 2.0, 60), np.linspace(-2.0, 2.0, 60))
    pts = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    extractor = CarlaLiDARPlanarFeatureExtractor(
        voxel_size=0.08,
        search_radius=0.24,
        min_neighbors=8,
        max_neighbors=20,
        max_candidate_points=2000,
        max_query_points=500,
    )
    features = extractor.extract(pts)
    assert len(features) > 100
    assert all(float(f["planarity"]) > 0.15 for f in features[:50])


def test_v13_mapper_can_skip_static_but_keep_dynamic_tubes():
    mapper = Carla4DEnvironmentMapper(max_dynamic_history=10)
    pose = {"location": {"x": 0.0, "y": 0.0, "z": 0.0}, "rotation": {}}
    track = {
        "track_id": 1,
        "position": [5.0, 0.0, 0.0],
        "velocity": [1.0, 0.0, 0.0],
        "motion_state": "DYNAMIC",
        "class": "vehicle",
    }
    mapper.add_frame_data(
        frame_id=1, timestamp=0.1, lidar_points=np.ones((100, 3)),
        planar_features=[], active_tracks=[track], ego_pose=pose,
        update_static=False, update_planar=False, evict_static=False,
    )
    assert len(mapper.static_voxel_map) == 0
    assert len(mapper.dynamic_object_tubes[1]) == 1
