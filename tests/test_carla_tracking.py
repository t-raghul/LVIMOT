import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from lvimot.tracking_4d.multimodal_object_tracking.kalman_tracker import CarlaMultiObjectTracker


OUTPUT_ROOT = Path("outputs_carla/sequence_0006")
NUM_FRAMES = 100

# Evaluation-only threshold.
# This does NOT affect the tracker.
EVAL_DISTANCE_THRESHOLD = 5.0


def load_frame(frame):
    path = OUTPUT_ROOT / f"frame_{frame:06d}.json"

    with open(path) as f:
        return json.load(f)


def distance(a, b):
    return float(
        np.linalg.norm(
            np.asarray(a, dtype=np.float64)
            - np.asarray(b, dtype=np.float64)
        )
    )


def evaluate_frame(tracks, detections):
    """
    Evaluation-only matching between tracker tracks and CARLA actor IDs.

    actor_id is used ONLY here for evaluation.
    It is never passed into the tracker.
    """

    if not tracks or not detections:
        return [], set(range(len(tracks))), set(range(len(detections)))

    cost = np.full(
        (len(tracks), len(detections)),
        np.inf,
        dtype=np.float64,
    )

    for i, track in enumerate(tracks):

        for j, detection in enumerate(detections):

            if track["class"] != detection["type"]:
                continue

            d = distance(
                track["position"],
                detection["location"],
            )

            if d <= EVAL_DISTANCE_THRESHOLD:
                cost[i, j] = d

    rows, cols = linear_sum_assignment(
        np.where(
            np.isfinite(cost),
            cost,
            1e9,
        )
    )

    matches = []

    matched_tracks = set()
    matched_detections = set()

    for row, col in zip(rows, cols):

        if not np.isfinite(cost[row, col]):
            continue

        matches.append(
            {
                "track_index": row,
                "track_id": tracks[row]["track_id"],
                "actor_id": detections[col]["actor_id"],
                "distance": float(cost[row, col]),
            }
        )

        matched_tracks.add(row)
        matched_detections.add(col)

    unmatched_tracks = (
        set(range(len(tracks))) - matched_tracks
    )

    unmatched_detections = (
        set(range(len(detections))) - matched_detections
    )

    return (
        matches,
        unmatched_tracks,
        unmatched_detections,
    )


def main():

    tracker = CarlaMultiObjectTracker(
        distance_threshold=5.0,
        max_missed=5,
    )

    total_observations = 0

    # --------------------------------------------------
    # Evaluation statistics
    # --------------------------------------------------

    total_matches = 0
    total_unmatched_tracks = 0
    total_unmatched_detections = 0

    position_errors = []

    # Track -> actor identity established by evaluation.
    track_actor_map = {}

    # Actual identity switches.
    id_switches = 0

    # Track history.
    track_stats = {}

    print(f"Frames found: 500")
    print(f"Frames evaluated: {NUM_FRAMES}")
    print()

    for frame in range(NUM_FRAMES):

        data = load_frame(frame)

        detections = data.get(
            "multimodal_objects",
            []
        )

        tracks = tracker.update(
            detections,
            timestamp=data["timestamp"],
        )

        total_observations += len(detections)

        (
            matches,
            unmatched_tracks,
            unmatched_detections,
        ) = evaluate_frame(
            tracks,
            detections,
        )

        total_matches += len(matches)
        total_unmatched_tracks += len(
            unmatched_tracks
        )
        total_unmatched_detections += len(
            unmatched_detections
        )

        print(
            f"Frame {frame:3d} | "
            f"observations={len(detections):2d} | "
            f"active_tracks={len(tracks):2d} | "
            f"matches={len(matches):2d}"
        )

        # --------------------------------------------------
        # Process valid evaluation matches
        # --------------------------------------------------

        for match in matches:

            track_id = match["track_id"]
            actor_id = match["actor_id"]
            error = match["distance"]

            position_errors.append(error)

            # ----------------------------------------------
            # Identity consistency
            # ----------------------------------------------

            previous_actor = track_actor_map.get(
                track_id
            )

            if previous_actor is None:

                track_actor_map[track_id] = actor_id

            elif previous_actor != actor_id:

                id_switches += 1

                print(
                    f"    ID SWITCH: "
                    f"Track {track_id} "
                    f"{previous_actor} -> {actor_id}"
                )

                track_actor_map[track_id] = actor_id

            # ----------------------------------------------
            # Track statistics
            # ----------------------------------------------

            if track_id not in track_stats:

                track_stats[track_id] = {
                    "actor_id": actor_id,
                    "matches": 0,
                    "errors": [],
                }

            track_stats[track_id]["matches"] += 1
            track_stats[track_id]["errors"].append(
                error
            )

            print(
                f"    Track {track_id:2d}"
                f" -> Actor {actor_id:3d}"
                f" | error={error:.3f} m"
            )

        # --------------------------------------------------
        # Report unmatched tracks
        # --------------------------------------------------

        for track_index in sorted(unmatched_tracks):

            track = tracks[track_index]

            print(
                f"    Track {track['track_id']:2d}"
                f" -> UNMATCHED"
                f" | missed={track['missed']}"
            )

        # --------------------------------------------------
        # Report unmatched detections
        # --------------------------------------------------

        for detection_index in sorted(
            unmatched_detections
        ):

            detection = detections[detection_index]

            print(
                f"    Actor {detection['actor_id']:3d}"
                f" -> UNMATCHED DETECTION"
            )

    # ======================================================
    # Final statistics
    # ======================================================

    print()
    print("=" * 60)
    print("TRACKING EVALUATION")
    print("=" * 60)

    print()
    print(f"Frames evaluated            : {NUM_FRAMES}")
    print(f"Total observations          : {total_observations}")
    print(f"Total valid matches         : {total_matches}")
    print(
        f"Unmatched tracker outputs   : "
        f"{total_unmatched_tracks}"
    )
    print(
        f"Unmatched detections        : "
        f"{total_unmatched_detections}"
    )

    print()
    print(f"Tracks created              : {tracker.next_track_id}")
    print(f"Identity switches           : {id_switches}")

    # --------------------------------------------------
    # Position error
    # --------------------------------------------------

    if position_errors:

        errors = np.asarray(
            position_errors,
            dtype=np.float64,
        )

        print()
        print("POSITION ERROR")
        print(
            f"Mean                        : "
            f"{np.mean(errors):.4f} m"
        )
        print(
            f"Median                      : "
            f"{np.median(errors):.4f} m"
        )
        print(
            f"RMSE                        : "
            f"{np.sqrt(np.mean(errors ** 2)):.4f} m"
        )
        print(
            f"P95                         : "
            f"{np.percentile(errors, 95):.4f} m"
        )
        print(
            f"Maximum                     : "
            f"{np.max(errors):.4f} m"
        )

    # --------------------------------------------------
    # Per-track statistics
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("PER-TRACK STATISTICS")
    print("=" * 60)

    for track_id in sorted(track_stats):

        stats = track_stats[track_id]

        errors = np.asarray(
            stats["errors"],
            dtype=np.float64,
        )

        print()
        print(
            f"Track {track_id}"
            f" | Actor {stats['actor_id']}"
        )

        print(
            f"  Matches             : "
            f"{stats['matches']}"
        )

        print(
            f"  Mean position error : "
            f"{np.mean(errors):.4f} m"
        )

        print(
            f"  RMSE                : "
            f"{np.sqrt(np.mean(errors ** 2)):.4f} m"
        )

        print(
            f"  Maximum error       : "
            f"{np.max(errors):.4f} m"
        )

    print()
    print("=" * 60)
    print("TRACKING TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
