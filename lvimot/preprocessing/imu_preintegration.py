
# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))

import numpy as np


class IMUPreintegrator:
    """Lightweight two-sample IMU preintegration.

    Acceleration is treated as body-frame specific force. Gravity is therefore
    not subtracted here; it is handled by the IMU factor in the world frame.
    Rotation uses the SO(3) exponential map rather than I + [w dt]x.
    """

    def __init__(self, min_dt: float = 1e-4, max_dt: float = 1.0):
        self.min_dt = float(min_dt)
        self.max_dt = float(max_dt)

    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        x, y, z = np.asarray(v, dtype=np.float64)
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)

    @classmethod
    def exp_so3(cls, rotation_vector: np.ndarray) -> np.ndarray:
        w = np.asarray(rotation_vector, dtype=np.float64)
        theta = float(np.linalg.norm(w))
        K = cls._skew(w)
        if theta < 1e-10:
            return np.eye(3, dtype=np.float64) + K + 0.5 * (K @ K)
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / (theta * theta)
        return np.eye(3, dtype=np.float64) + A * K + B * (K @ K)

    def _dt(self, motion_k, motion_k1) -> float:
        t0 = motion_k.get("timestamp")
        t1 = motion_k1.get("timestamp")
        if t0 is not None and t1 is not None:
            dt = float(t1) - float(t0)
        else:
            dt = float(motion_k.get("dt", motion_k1.get("dt", 0.1)))
        if not np.isfinite(dt) or dt <= 0:
            dt = float(motion_k.get("dt", 0.1))
        return float(np.clip(dt, self.min_dt, self.max_dt))

    def integrate(self, motion_k, motion_k1):
        dt = self._dt(motion_k, motion_k1)
        a_k = np.asarray(motion_k["acceleration"], dtype=np.float64)
        a_k1 = np.asarray(motion_k1["acceleration"], dtype=np.float64)
        w_k = np.asarray(motion_k["gyro"], dtype=np.float64)
        w_k1 = np.asarray(motion_k1["gyro"], dtype=np.float64)

        acceleration = 0.5 * (a_k + a_k1)
        angular_velocity = 0.5 * (w_k + w_k1)
        delta_theta = angular_velocity * dt
        delta_R = self.exp_so3(delta_theta)
        delta_v = acceleration * dt
        delta_p = 0.5 * acceleration * (dt ** 2)

        return {
            "dt": dt,
            "delta_p": delta_p,
            "delta_v": delta_v,
            "delta_theta": delta_theta,
            "delta_R": delta_R,
        }




import numpy as np


class CarlaIMUPreprocessor:
    """
    Converts CARLA IMU JSON measurements into the motion format
    expected by the existing LVIMOT IMU preintegrator.
    """

    def __init__(self, dt=0.1):
        self.dt = dt

    def process(self, imu_data):
        """
        Convert one CARLA IMU measurement into LVIMOT motion format.
        """

        acceleration = imu_data["accelerometer"]
        gyroscope = imu_data["gyroscope"]

        accel = np.array(
            [
                acceleration["x"],
                acceleration["y"],
                acceleration["z"]
            ],
            dtype=np.float64
        )

        gyro = np.array(
            [
                gyroscope["x"],
                gyroscope["y"],
                gyroscope["z"]
            ],
            dtype=np.float64
        )

        return {
            "dt": self.dt,
            "acceleration": accel,
            "gyro": gyro,

            # CARLA does not provide KITTI-style vf/vl/vu here.
            # These are kept as None until velocity/orientation
            # information is incorporated from the CARLA pose.
            "velocity": None,
            "rotation": None,

            "timestamp": imu_data.get("timestamp"),
            "dataset_frame": imu_data.get("dataset_frame"),
            "carla_frame": imu_data.get("carla_frame")
        }

    def remove_gravity(self, acceleration, gravity=9.80665):
        """
        Remove gravity from the Z component.

        This is useful for analysing the acceleration signal.
        The raw acceleration is preserved separately.
        """

        acceleration = np.asarray(
            acceleration,
            dtype=np.float64
        ).copy()

        acceleration[2] -= gravity

        return acceleration


if __name__ == "__main__":

    import sys


    from lvimot.inputs.recorded_sensors import CarlaReader
    from lvimot.preprocessing.imu_preintegration import IMUPreintegrator

    import argparse
    parser = argparse.ArgumentParser(description='Inspect recorded IMU preintegration; evaluation only.')
    parser.add_argument('--carla-root', required=True)
    parser.add_argument('--sequence', default='0001')
    args = parser.parse_args()
    sequence = args.sequence

    reader = CarlaReader(
        args.carla_root,
        sequence
    )

    imu_processor = CarlaIMUPreprocessor(
        dt=0.1
    )

    preintegrator = IMUPreintegrator()

    print("=" * 60)
    print("CARLA IMU PREINTEGRATION TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # Frame k
    # ---------------------------------------------------------

    imu_data_k = reader.load_imu(0)

    motion_k = imu_processor.process(
        imu_data_k
    )

    # ---------------------------------------------------------
    # Frame k+1
    # ---------------------------------------------------------

    imu_data_k1 = reader.load_imu(1)

    motion_k1 = imu_processor.process(
        imu_data_k1
    )

    # ---------------------------------------------------------
    # Print measurements
    # ---------------------------------------------------------

    print("\nFrame 0")
    print("Acceleration:", motion_k["acceleration"])
    print("Gyroscope:", motion_k["gyro"])

    print("\nFrame 1")
    print("Acceleration:", motion_k1["acceleration"])
    print("Gyroscope:", motion_k1["gyro"])

    print("\nDelta time:", motion_k["dt"])

    # ---------------------------------------------------------
    # Preintegrate
    # ---------------------------------------------------------

    result = preintegrator.integrate(
        motion_k,
        motion_k1
    )

    print("\nIMU PREINTEGRATION RESULT")
    print("-" * 60)

    print("Delta position:")
    print(result["delta_p"])

    print("\nDelta velocity:")
    print(result["delta_v"])

    print("\nDelta rotation:")
    print(result["delta_theta"])

    print("\nDelta rotation matrix:")
    print(result["delta_R"])

    print("\nCARLA IMU PREINTEGRATION TEST PASSED")
