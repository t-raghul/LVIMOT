import sys
import os
import numpy as np


from lvimot.inputs.recorded_sensors import CarlaReader
from lvimot.output.evaluation.recorded_imu_motion import CarlaIMUMotion


SEQUENCE = "0006"
DATA_ROOT = os.environ.get("CARLA_DATA_ROOT", "../CARLA_LVIMOT")

reader = CarlaReader(DATA_ROOT, SEQUENCE)
motion = CarlaIMUMotion(reader)

frames = [50, 75, 100, 125, 150, 175, 200, 225, 250]

print("=" * 120)
print("DETAILED CARLA IMU / GRAVITY INSPECTION")
print("=" * 120)
print()

for k in frames:

    pose = reader.load_pose(k)

    rotation = pose["rotation"]

    raw = motion.get_raw_acceleration(k)

    gravity = motion.gravity_in_body_frame(k)

    corrected = motion.corrected_acceleration(k)

    R = motion.rotation_matrix(rotation)

    corrected_world = R @ corrected

    velocity = motion.get_velocity(k)

    print(f"FRAME {k}")
    print("-" * 120)

    print(
        "Rotation:",
        f"roll={rotation['roll']:.6f}°",
        f"pitch={rotation['pitch']:.6f}°",
        f"yaw={rotation['yaw']:.6f}°"
    )

    print("\nRaw IMU acceleration [body]:")
    print(raw)

    print("\nGravity [body]:")
    print(gravity)

    print("\nCorrected acceleration [body]:")
    print(corrected)

    print("\nCorrected acceleration [world]:")
    print(corrected_world)

    print("\nGT velocity [world]:")
    print(velocity)

    print()

print("=" * 120)
