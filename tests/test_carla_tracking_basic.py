import json
import glob
import os

from lvimot.tracking_4d.multimodal_object_tracking.kalman_tracker import CarlaMultiObjectTracker


SEQUENCE = "0006"

OUTPUT_DIR = (
    f"outputs_carla/sequence_{SEQUENCE}"
)


def load_frame(path):

    with open(path, "r") as f:
        return json.load(f)


def main():

    pattern = os.path.join(
        OUTPUT_DIR,
        "frame_*.json"
    )

    files = sorted(glob.glob(pattern))

    print("Frames found:", len(files))

    tracker = CarlaMultiObjectTracker(
        distance_threshold=5.0,
        max_missed=5,
    )

    total_tracks = 0

    for frame_number, path in enumerate(files):

        data = load_frame(path)

        timestamp = data["timestamp"]

        observations = data.get(
            "multimodal_objects",
            []
        )

        tracks = tracker.update(
            observations,
            timestamp,
        )

        total_tracks += len(tracks)

        print(
            f"Frame {data['frame']:3d} | "
            f"observations={len(observations):2d} | "
            f"active_tracks={len(tracks):2d}"
        )

        if frame_number >= 99:
            break

    print()
    print("TRACKING TEST COMPLETE")
    print("Frames processed:", min(100, len(files)))
    print("Total track observations:", total_tracks)

    print()
    print("FINAL ACTIVE TRACKS:")

    for track in tracker.get_tracks():
        print(track)


if __name__ == "__main__":
    main()
