"""Display-only map cleanup, contour extraction, and raw-map PLY export.

This module must never be used to change LVIMOT localization, tracking, or
4-D map data. All filters operate on copies of the mapper's output.
"""
from __future__ import annotations

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def cleaned_layers_and_outlines(
    ground: np.ndarray,
    structure: np.ndarray,
    meters: float,
    cell_size_m: float,
    min_ground_component: int = 7,
    min_structure_component: int = 4,
    max_contours: int = 160,
) -> Tuple[np.ndarray, np.ndarray, List[List[List[float]]], List[List[List[float]]], Dict]:
    """Suppress small *display* islands and extract local XY cluster boundaries.

    The outlines follow supported occupancy cells. To smooth pixelated contours,
    at most a single 3x3 grid closing is used for visual outlines only; this
    can bridge a gap no larger than about two cells, so lines are NOT surveyed
    walls or driveable road boundaries. Only clusters with measured support
    are outlined. The 3-D mapper is never modified.
    """
    ground = np.asarray(ground, dtype=np.float64).reshape(-1, 3)
    structure = np.asarray(structure, dtype=np.float64).reshape(-1, 3)
    stats = {"removed_display_islands": 0, "ground_outline_count": 0, "structure_outline_count": 0,
             "outline_gap_limit_m": float(2 * cell_size_m)}
    if cv2 is None:
        return ground, structure, [], [], stats

    scale = 1.0 / max(0.05, float(cell_size_m))
    half = float(meters)
    n = max(8, int(np.ceil(2.0 * half * scale)) + 3)

    def layer(points: np.ndarray, min_pixels: int):
        if len(points) == 0:
            return points, [], 0
        indices = np.floor((points[:, :2] + half) * scale).astype(np.int32)
        valid = np.all((indices >= 0) & (indices < n), axis=1)
        indices = indices[valid]
        data = points[valid]
        if len(data) == 0:
            return data, [], len(points)
        mask = np.zeros((n, n), dtype=np.uint8)
        # rows = forward, columns = lateral; keep a consistent transform.
        mask[indices[:, 0], indices[:, 1]] = 1
        count, labels, cc_stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        allow = np.zeros(count, dtype=bool)
        if count > 1:
            allow[1:] = cc_stats[1:, cv2.CC_STAT_AREA] >= int(min_pixels)
        kept_pixels = (allow[labels] & (mask > 0)).astype(np.uint8)
        keep_pt = allow[labels[indices[:, 0], indices[:, 1]]]
        removed = int(len(data) - int(keep_pt.sum()) + len(points) - len(data))
        filtered = data[keep_pt]
        if len(filtered) == 0:
            return filtered, [], removed
        # For visualization of *lines*, connect only immediate supported
        # neighbours. This is not written back into the occupancy map.
        if len(filtered) >= 10:
            kernel = np.ones((3, 3), dtype=np.uint8)
            outline_mask = cv2.morphologyEx(kept_pixels, cv2.MORPH_CLOSE, kernel)
        else:
            outline_mask = kept_pixels
        contours, _ = cv2.findContours(outline_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.arcLength(x, True), reverse=True)
        outlines = []
        for contour in contours[:max_contours]:
            if len(contour) < 2:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter < 4.0:
                continue
            simpler = cv2.approxPolyDP(contour, epsilon=0.65, closed=True)
            if len(simpler) < 2:
                continue
            poly = []
            for pos in simpler.reshape(-1, 2):
                iy, ix = int(pos[0]), int(pos[1])
                forward = (ix + 0.5) / scale - half
                lateral = (iy + 0.5) / scale - half
                poly.append([round(forward, 3), round(lateral, 3)])
            outlines.append(poly)
        return filtered, outlines, removed

    clean_ground, ground_lines, removed_g = layer(ground, min_ground_component)
    clean_structure, structure_lines, removed_s = layer(structure, min_structure_component)
    stats["removed_display_islands"] = int(removed_g + removed_s)
    stats["ground_outline_count"] = len(ground_lines)
    stats["structure_outline_count"] = len(structure_lines)
    return clean_ground, clean_structure, ground_lines, structure_lines, stats


def points_local_to_world(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Convert GUI heading-up (forward, lateral, relative z) to map-world XYZ."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    p = np.asarray(pose, dtype=np.float64).reshape(-1)
    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.float32)
    yaw, x, y, z = float(p[2]), float(p[3]), float(p[4]), float(p[5])
    c, s = np.cos(yaw), np.sin(yaw)
    f, l = pts[:, 0], pts[:, 1]
    return np.column_stack([x + c * f - s * l, y + s * f + c * l, z + pts[:, 2]]).astype(np.float32)


def write_binary_ply(path: Path, xyz: np.ndarray, colors: Optional[np.ndarray] = None) -> int:
    """Atomic binary PLY with original map-world Cartesian coordinates."""
    path = Path(path)
    pts = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    rgb = None
    if colors is not None:
        rgb = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)[valid]
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        'ply', 'format binary_little_endian 1.0',
        'comment LVIMOT sensor-estimated rolling local map (NOT ground truth)',
        'element vertex {}'.format(len(pts)),
        'property float x', 'property float y', 'property float z',
    ]
    if rgb is not None:
        header += ['property uchar red', 'property uchar green', 'property uchar blue']
    header += ['end_header']
    tmp = path.with_name(path.name + '.tmp')
    try:
        with tmp.open('wb') as fp:
            fp.write(('\n'.join(header) + '\n').encode('ascii'))
            if rgb is None:
                fp.write(np.asarray(pts, dtype='<f4').tobytes(order='C'))
            else:
                packed = np.empty(len(pts), dtype=[('xyz', '<f4', (3,)), ('rgb', 'u1', (3,))])
                packed['xyz'] = pts
                packed['rgb'] = rgb
                fp.write(packed.tobytes(order='C'))
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            tmp.unlink()
    return len(pts)


def export_live_map(mapper, pose, visualizer, export_dir: Path, include_accumulated: bool = True) -> Dict:
    """Save REAL mapper voxels first. Clean display and tracks are best effort.

    A failed clean filter must never prevent the raw point-cloud export.
    The mapper itself is not changed by any step here.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    raw_path = export_dir / 'live_map_latest.ply'
    # NEVER downsample this raw export or rely on the GUI's 5000-point snapshot.
    n_raw = write_binary_ply(raw_path, mapper.get_static_map_points())
    result = {'raw_voxels': n_raw, 'raw_ply': str(raw_path),
              'clean_points': None, 'clean_ply': None,
              'dynamic_tracks_json': None, 'warnings': []}
    road_layer=getattr(mapper,'road_surface_layer',None)
    if road_layer is not None:
        road_path=export_dir/'live_road_surface_latest.ply'
        road_count=write_binary_ply(road_path,road_layer.points())
        result['measured_road_surface']={'ply':str(road_path),'measured_voxels':road_count,**road_layer.summary()}

    try:
        raw = mapper.get_static_map_points()
        local = visualizer._bounded_local_map(raw, pose)
        ground, structure, _ = visualizer._clean_map_cells(local)
        ground, structure, _, _, _ = cleaned_layers_and_outlines(
            ground, structure, visualizer.meters, visualizer.map_cell_size_m
        )
        clean_local = (np.vstack([ground, structure])
                       if len(ground) + len(structure)
                       else np.empty((0, 3)))
        ground_color = np.tile(np.array([[95, 125, 145]], dtype=np.uint8), (len(ground), 1))
        structure_color = np.tile(np.array([[220, 230, 240]], dtype=np.uint8), (len(structure), 1))
        colors = np.vstack([ground_color, structure_color])
        clean_path = export_dir / 'live_map_clean_latest.ply'
        result['clean_points'] = write_binary_ply(
            clean_path, points_local_to_world(clean_local, pose), colors
        )
        result['clean_ply'] = str(clean_path)
    except Exception as exc:
        result['warnings'].append('Clean view unavailable: ' + str(exc))

    try:
        tubes_path = export_dir / 'live_dynamic_tracks_latest.json'
        tubes = {str(key): list(tube) for key, tube in mapper.dynamic_object_tubes.items()}
        def safe(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {str(k): safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [safe(v) for v in value]
            return value
        temp = tubes_path.with_name(tubes_path.name + '.tmp')
        try:
            with temp.open('w', encoding='utf-8') as out:
                json.dump({'coordinate_frame': 'LVIMOT estimated local world',
                           'description': 'Timestamped tracking history, not static PLY',
                           'dynamic_object_tubes': safe(tubes),
                           'individual_object_tracks': safe(mapper.individual_object_scene.export())}, out, indent=2)
            os.replace(str(temp), str(tubes_path))
        finally:
            if temp.exists():
                temp.unlink()
        result['dynamic_tracks_json'] = str(tubes_path)
    except Exception as exc:
        result['warnings'].append('Dynamic tracks export unavailable: ' + str(exc))
    accumulated = getattr(mapper, 'accumulated_map', None)
    radius = getattr(mapper, 'max_map_distance', None)
    result['map_scope'] = {'live': 'rolling local map', 'radius_m': float(radius) if radius is not None else None}
    if accumulated is not None:
        result['map_scope']['accumulated'] = 'all integrated observations along this run; sensor-estimated; no loop closure'
        if include_accumulated:
            try:
                result['accumulated_map'] = accumulated.export(export_dir / 'live_map_accumulated.ply')
            except Exception as exc:
                result['warnings'].append('Accumulated map export unavailable: ' + str(exc))
    return result
