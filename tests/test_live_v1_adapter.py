import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np


from lvimot.inputs.synchronized_sensors import _SensorFrameQueue
from lvimot.output.live_run import _gt_objects, _compact_live_record


class _FakeSensor:
    def __init__(self):
        self.callback = None
        self.stopped = False

    def listen(self, callback):
        self.callback = callback

    def stop(self):
        self.stopped = True

    def emit(self, frame, timestamp, payload=None):
        data = SimpleNamespace(frame=frame, timestamp=timestamp, payload=payload)
        self.callback(data)


def test_live_sensor_queue_requires_exact_frame():
    sensor = _FakeSensor()
    q = _SensorFrameQueue(sensor, lambda d: d.payload, maxsize=4)
    sensor.emit(10, 1.0, "old")
    sensor.emit(11, 1.1, "wanted")
    ts, value = q.get_exact(11, timeout=0.1)
    assert ts == 1.1
    assert value == "wanted"


def test_live_sensor_queue_fails_on_future_frame():
    sensor = _FakeSensor()
    q = _SensorFrameQueue(sensor, lambda d: d.payload, maxsize=4)
    sensor.emit(12, 1.2, "future")
    try:
        q.get_exact(11, timeout=0.1)
        assert False, "expected synchronization failure"
    except RuntimeError as exc:
        assert "advanced to frame 12" in str(exc)


class _Vec:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class _Rot:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll


class _Transform:
    def __init__(self):
        self.location = _Vec(10.0, 2.0, 0.5)
        self.rotation = _Rot(0.0, 15.0, 0.0)


class _BoundingBox:
    def __init__(self):
        self.extent = _Vec(2.0, 1.0, 0.8)

    def get_world_vertices(self, transform):
        return [_Vec(float(i), float(i + 1), float(i + 2)) for i in range(8)]


class _Actor:
    def __init__(self, actor_id, type_id="vehicle.test"):
        self.id = actor_id
        self.type_id = type_id
        self.bounding_box = _BoundingBox()

    def get_transform(self):
        return _Transform()

    def get_velocity(self):
        return _Vec(1.0, 0.0, 0.0)


class _ActorList(list):
    def filter(self, pattern):
        if pattern == "vehicle.*":
            return [a for a in self if a.type_id.startswith("vehicle.")]
        if pattern == "walker.pedestrian.*":
            return [a for a in self if a.type_id.startswith("walker.pedestrian.")]
        return []


class _World:
    def __init__(self, actors):
        self._actors = _ActorList(actors)

    def get_actors(self):
        return self._actors


def test_live_gt_objects_include_world_box_corners_and_exclude_ego():
    ego = _Actor(1)
    target = _Actor(2)
    objects = _gt_objects(_World([ego, target]), ego_id=1)
    assert len(objects) == 1
    obj = objects[0]
    assert obj["actor_id"] == 2
    assert len(obj["bounding_box"]["world_corners"]) == 8
    assert obj["dimensions"] == [4.0, 2.0, 1.6]


def test_live_log_is_compact_and_keeps_estimator_state():
    result = {
        "frame": 3,
        "carla_frame": 1003,
        "timestamp": 12.3,
        "state_estimation": {"pose": [0, 0, 0, 1, 2, 3], "velocity": [1, 0, 0]},
        "tracking": {"active_tracks": [{"track_id": 7, "position": [1, 2, 3], "velocity": [0, 0, 0]}]},
        "mapping_4d": {"total_static_voxels": 10},
        "sensor_integrity": {"stationary_active": False},
        "objects": {"evaluation": {"eligible_gt_objects": 2}},
        "performance": {"total_ms": 400},
        "live_sync": {"timestamp_skew_s": 0.0},
        "temporal_visual": {"active_tracks": list(range(1000))},
    }
    compact = _compact_live_record(result)
    assert compact["frame"] == 3
    assert compact["carla_frame"] == 1003
    assert compact["evaluation_frame"]["eligible_gt_objects"] == 2
    assert "temporal_visual" not in compact
    assert compact["tracks"][0]["track_id"] == 7
