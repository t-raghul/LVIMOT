import numpy as np

from lvimot.inputs.recorded_sensors import CarlaReader
from lvimot.output.evaluation.recorded_imu_motion import CarlaIMUMotion


# ============================================================
# Configuration
# ============================================================

SEQUENCE = "0006"
CARLA_ROOT = "../CARLA_LVIMOT"


# ============================================================
# CARLA rotation matrix
# ============================================================

def rotation_matrix_from_carla_euler(roll, pitch, yaw):
    """
    CARLA Transform::GetMatrix-compatible rotation matrix.

    CARLA stores roll, pitch and yaw in degrees.
    """

    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

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
    ])


# ============================================================
# CARLA vector dictionary -> numpy
# ============================================================

def vec3(data):
    return np.array(
        [
            data["x"],
            data["y"],
            data["z"],
        ],
        dtype=float,
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Reader
    # --------------------------------------------------------

    reader = CarlaReader(
        carla_root=CARLA_ROOT,
        sequence=SEQUENCE,
    )

    # --------------------------------------------------------
    # Existing CARLA IMU motion processor
    #
    # IMPORTANT:
    # This class operates directly using a frame number.
    # --------------------------------------------------------

    motion_processor = CarlaIMUMotion(
        reader
    )

    n = reader.num_frames()

    print("=" * 115)
    print("CARLA IMU VALIDATION")
    print("=" * 115)

    print(f"Sequence         : {SEQUENCE}")
    print(f"Number of frames : {n}")

    print()
    print("Ground truth     : CARLA ego.get_acceleration()")
    print("IMU acceleration : Existing CarlaIMUMotion correction")
    print("Comparison       : World frame")

    print("=" * 115)

    # --------------------------------------------------------
    # Error storage
    # --------------------------------------------------------

    errors = []
    abs_errors = []
    squared_errors = []

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print(
        f"{'Frame':>5} "
        f"{'GT ax':>10} "
        f"{'GT ay':>10} "
        f"{'GT az':>10} "
        f"{'IMU ax':>10} "
        f"{'IMU ay':>10} "
        f"{'IMU az':>10} "
        f"{'Err x':>10} "
        f"{'Err y':>10} "
        f"{'Err z':>10} "
        f"{'|Err|':>10}"
    )

    print("-" * 115)

    # ========================================================
    # Frame loop
    # ========================================================

    for k in range(n):

        # ----------------------------------------------------
        # Load pose
        # ----------------------------------------------------

        pose = reader.load_pose(k)

        # ----------------------------------------------------
        # Existing IMU processing
        #
        # This internally:
        #
        # 1. reads the raw IMU
        # 2. calculates gravity in body frame
        # 3. removes gravity
        # 4. transforms corrected acceleration to world frame
        # ----------------------------------------------------

        imu_world = np.asarray(
            motion_processor.corrected_acceleration_world(k),
            dtype=float,
        )

        # ----------------------------------------------------
        # Ground-truth CARLA acceleration
        # ----------------------------------------------------

        gt_body = vec3(
            pose["acceleration"]
        )

        # ----------------------------------------------------
        # Ego rotation
        # ----------------------------------------------------

        rotation = pose["rotation"]

        R = rotation_matrix_from_carla_euler(
            rotation["roll"],
            rotation["pitch"],
            rotation["yaw"],
        )

        # ----------------------------------------------------
        # CARLA get_acceleration()
        #
        # Convert from ego/body frame to world frame.
        # ----------------------------------------------------

        gt_world = R @ gt_body

        # ----------------------------------------------------
        # Error
        # ----------------------------------------------------

        error = imu_world - gt_world

        error_norm = np.linalg.norm(
            error
        )

        errors.append(error)
        abs_errors.append(np.abs(error))
        squared_errors.append(error ** 2)

        # ----------------------------------------------------
        # Print selected frames
        # ----------------------------------------------------

        if (
            k < 20
            or k % 25 == 0
            or error_norm > 2.0
        ):

            print(
                f"{k:5d} "
                f"{gt_world[0]:10.4f} "
                f"{gt_world[1]:10.4f} "
                f"{gt_world[2]:10.4f} "
                f"{imu_world[0]:10.4f} "
                f"{imu_world[1]:10.4f} "
                f"{imu_world[2]:10.4f} "
                f"{error[0]:10.4f} "
                f"{error[1]:10.4f} "
                f"{error[2]:10.4f} "
                f"{error_norm:10.4f}"
            )

    # ========================================================
    # Convert errors to numpy arrays
    # ========================================================

    errors = np.asarray(errors)
    abs_errors = np.asarray(abs_errors)
    squared_errors = np.asarray(squared_errors)

    # ========================================================
    # Statistics
    # ========================================================

    mean_error = np.mean(
        errors,
        axis=0
    )

    mae = np.mean(
        abs_errors,
        axis=0
    )

    rmse = np.sqrt(
        np.mean(
            squared_errors,
            axis=0
        )
    )

    overall_rmse = np.sqrt(
        np.mean(
            np.sum(
                errors ** 2,
                axis=1
            )
        )
    )

    error_norms = np.linalg.norm(
        errors,
        axis=1
    )

    # ========================================================
    # Results
    # ========================================================

    print()
    print("=" * 115)
    print("VALIDATION RESULTS")
    print("=" * 115)

    print()
    print(f"Number of frames: {n}")

    print()
    print("Mean error [x, y, z]:")
    print(mean_error)

    print()
    print("MAE [x, y, z]:")
    print(mae)

    print()
    print("RMSE [x, y, z]:")
    print(rmse)

    print()
    print(
        f"Overall RMSE: "
        f"{overall_rmse:.6f} m/s²"
    )

    print()
    print(
        f"Maximum error magnitude: "
        f"{np.max(error_norms):.6f} m/s²"
    )

    print(
        f"Mean error magnitude: "
        f"{np.mean(error_norms):.6f} m/s²"
    )

    # ========================================================
    # Percentiles
    # ========================================================

    print()
    print("Error magnitude percentiles:")

    for percentile in [50, 75, 90, 95, 99]:

        value = np.percentile(
            error_norms,
            percentile
        )

        print(
            f"  {percentile:2d}th percentile: "
            f"{value:.6f} m/s²"
        )

    # ========================================================
    # Worst frames
    # ========================================================

    worst_indices = np.argsort(
        error_norms
    )[-10:][::-1]

    print()
    print("10 frames with largest error:")

    for idx in worst_indices:

        print(
            f"  Frame {idx:4d}: "
            f"{error_norms[idx]:.6f} m/s²"
        )

    print()
    print("=" * 115)
    print("CARLA IMU VALIDATION COMPLETE")
    print("=" * 115)


if __name__ == "__main__":
    main()
