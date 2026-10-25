import os
import sys

import numpy as np


from lvimot.temporal_fusion.pipeline import CarlaLVIMOTCore


def test_corner_array_accepts_list_of_xyz_dicts():
    corners = [
        {"x": 1.0, "y": 2.0, "z": 3.0},
        {"x": 4.0, "y": 5.0, "z": 6.0},
        {"x": 7.0, "y": 8.0, "z": 9.0},
        {"x": 10.0, "y": 11.0, "z": 12.0},
    ]
    arr = CarlaLVIMOTCore._corner_array(corners)
    assert arr.shape == (4, 3)
    assert np.allclose(arr[1], [4.0, 5.0, 6.0])


def test_corner_array_accepts_numeric_key_dict():
    corners = {
        "0": {"x": 1.0, "y": 2.0, "z": 3.0},
        "1": {"x": 4.0, "y": 5.0, "z": 6.0},
        "2": {"x": 7.0, "y": 8.0, "z": 9.0},
        "3": {"x": 10.0, "y": 11.0, "z": 12.0},
    }
    arr = CarlaLVIMOTCore._corner_array(corners)
    assert arr.shape == (4, 3)
    assert np.allclose(arr[-1], [10.0, 11.0, 12.0])


def test_corner_array_accepts_column_dict():
    corners = {
        "x": [1.0, 4.0, 7.0, 10.0],
        "y": [2.0, 5.0, 8.0, 11.0],
        "z": [3.0, 6.0, 9.0, 12.0],
    }
    arr = CarlaLVIMOTCore._corner_array(corners)
    assert arr.shape == (4, 3)
    assert np.allclose(arr[2], [7.0, 8.0, 9.0])


def test_corner_array_accepts_regular_list_of_lists():
    corners = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    arr = CarlaLVIMOTCore._corner_array(corners)
    assert arr.shape == (4, 3)
    assert np.allclose(arr[0], [1, 2, 3])
