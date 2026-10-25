import sys
from pathlib import Path

import numpy as np

from lvimot.tracking_4d.static_scene_support.measured_ground_support import horizontal_ground_candidates


def test_ground_classifier_preserves_vertical_structures_and_does_not_label_roofs_ground():
    xy = np.array(np.meshgrid(np.arange(-4, 4, .25), np.arange(-4, 4, .25))).reshape(2, -1).T
    ground = np.column_stack([xy, np.full(len(xy), -.5)])
    roof = np.column_stack([xy, np.full(len(xy), 3)])
    wall = np.array([[3, y, z] for y in np.arange(-4, 4, .25) for z in np.arange(0, 3, .25)])
    points = np.vstack([ground, roof, wall])
    original = points.copy()
    selected, details = horizontal_ground_candidates(points, np.array([[0, 0, 0], [1, 0, 0]]))
    assert selected[:len(ground)].mean() > .9
    assert not selected[len(ground):].any()
    assert np.array_equal(points, original)
    assert details['ground_truth_or_reference_used'] is False
