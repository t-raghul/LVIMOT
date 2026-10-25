"""Read-only 4D object scene history built exclusively from FGO object states.

This module never influences perception, tracking, FGO, map masking or MOT scoring.
The input track IDs/classes/boxes come from the confirmed MOT frontend; XYZ and
velocity are the FGO world estimates. Missing estimates are never invented.
"""

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))

from collections import deque
import math
from typing import Dict, Iterable, List

import numpy as np


class IndividualObjectScene:
    """Maintain bounded per-ID history, stationary boxes and moving trails."""

    def __init__(self, max_history=500, max_objects=64, move_speed=0.70,
                 stop_speed=0.30, parked_after_s=2.0, hold_s=1.2):
        self.max_history = max(8, int(max_history))
        self.max_objects = max(1, int(max_objects))
        self.move_speed = float(move_speed)
        self.stop_speed = float(stop_speed)
        self.parked_after_s = float(parked_after_s)
        self.hold_s = float(hold_s)
        self.objects: Dict[int, Dict] = {}

    @staticmethod
    def _xyz(value):
        if isinstance(value, dict):
            value = [value.get('x', 0), value.get('y', 0), value.get('z', 0)]
        xyz = np.asarray(value, dtype=np.float64).reshape(3)
        return xyz.tolist() if np.isfinite(xyz).all() else None

    @staticmethod
    def _dims(value):
        if isinstance(value, dict):
            v = [value.get('length', 4.5), value.get('width', 1.9), value.get('height', 1.6)]
        else:
            v = value if value is not None else [4.5, 1.9, 1.6]
        try:
            xyz = np.asarray(v, dtype=np.float64).reshape(-1)
            if len(xyz) < 3 or not np.isfinite(xyz[:3]).all():
                raise ValueError('bad dimensions')
            return dict(zip(('length', 'width', 'height'),
                            [float(np.clip(n, 0.5, 18.0)) for n in xyz[:3]]))
        except (ValueError, TypeError):
            return {'length': 4.5, 'width': 1.9, 'height': 1.6}

    @staticmethod
    def _rotation(value):
        if not isinstance(value, dict):
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
        return {key: float(value.get(key, 0.0) or 0.0)
                if math.isfinite(float(value.get(key, 0.0) or 0.0)) else 0.0
                for key in ('yaw', 'pitch', 'roll')}

    def update(self, frame_id: int, timestamp: float, tracks: Iterable[Dict]):
        """Only append finite FGO states; retain the last valid observation on gaps."""
        frame_id, timestamp = int(frame_id), float(timestamp)
        for tr in tracks:
            try:
                tid = int(tr['track_id'])
                pos = self._xyz(tr.get('position'))
                vel = self._xyz(tr.get('velocity'))
            except (ValueError, TypeError, KeyError):
                continue
            if pos is None or vel is None or tid < 0:
                continue
            old = self.objects.get(tid)
            if old is not None and timestamp <= old['timestamp']:
                continue
            speed = float(np.linalg.norm(vel))
            if old is None:
                old = {
                    'track_id': tid,
                    'first_timestamp': timestamp,
                    'first_frame': frame_id,
                    'stationary_since': None,
                    'is_moving': speed >= self.move_speed,
                    'history': deque(maxlen=self.max_history),
                    'path': deque(maxlen=self.max_history),
                }
                self.objects[tid] = old

            # Hysteresis avoids flickering when FGO velocities are noisy.
            moving = bool(old['is_moving'])
            if speed >= self.move_speed:
                moving = True
            elif speed <= self.stop_speed:
                moving = False
            was_moving = bool(old['is_moving'])
            old['is_moving'] = moving
            if was_moving and not moving and old['path']:
                # Preserve the observed stop location once, not every idle frame.
                last = np.asarray(old['path'][-1]['position'])
                if np.linalg.norm(np.asarray(pos) - last) >= 0.15:
                    old['path'].append({'timestamp': timestamp, 'position': pos})
            if moving:
                old['stationary_since'] = None
                status = 'MOVING'
            else:
                if old['stationary_since'] is None:
                    old['stationary_since'] = timestamp
                stationary_time = timestamp - float(old['stationary_since'])
                # "Parked" is only a candidate without semantic parking context.
                status = 'PARKED_CANDIDATE' if stationary_time >= self.parked_after_s else 'STOPPED'

            snap = {
                'frame_id': frame_id, 'timestamp': timestamp,
                'position': pos, 'velocity': vel, 'speed': speed,
                'motion_status': status,
                'class': str(tr.get('class', 'vehicle')),
                'dimensions': self._dims(tr.get('dimensions')),
                'rotation': self._rotation(tr.get('rotation')),
                'source': 'temporal_fusion_fgo',
            }
            old.update(snap)
            old['history'].append(snap.copy())
            # Store measured FGO trajectory samples, never extrapolated segments.
            # A stationary car keeps a box but does not grow a fake moving path.
            if not old['path']:
                old['path'].append({'timestamp': timestamp, 'position': pos})
            elif moving:
                last = np.asarray(old['path'][-1]['position'])
                if np.linalg.norm(np.asarray(pos) - last) >= 0.15:
                    old['path'].append({'timestamp': timestamp, 'position': pos})

        if len(self.objects) > self.max_objects:
            order = sorted(self.objects, key=lambda k: self.objects[k]['timestamp'])
            for tid in order[:len(self.objects) - self.max_objects]:
                self.objects.pop(tid, None)

    def current(self, timestamp: float, max_display=32, path_points=140,
                include_historical=False) -> List[Dict]:
        """Return live objects, optionally including historical parked boxes/paths."""
        result = []
        newest = sorted(self.objects.values(), key=lambda item: item['timestamp'], reverse=True)
        for obj in newest:
            age = max(0.0, float(timestamp) - float(obj['timestamp']))
            if age > self.hold_s and not include_historical:
                continue
            is_old = age > self.hold_s
            parked = obj['motion_status'] == 'PARKED_CANDIDATE'
            # A last measured parked candidate remains visible in historical
            # mode, but an old moving track is only a path, never a fake box.
            draw_box = not is_old or parked
            status = ('PARKED_CANDIDATE' if parked else 'HISTORICAL_PATH') if is_old else (
                obj['motion_status'] if age < 0.0001 else 'LAST_SEEN')
            path = list(obj['path'])
            if len(path) > path_points:
                step = max(1, math.ceil(len(path) / path_points))
                path = path[::step]
                if path[-1] != obj['path'][-1]:
                    path.append(obj['path'][-1])
            result.append({
                'track_id': obj['track_id'], 'class': obj['class'],
                'position': obj['position'], 'velocity': obj['velocity'],
                'speed': obj['speed'], 'dimensions': obj['dimensions'],
                'rotation': obj['rotation'],
                'motion_status': status,
                'last_known_motion_status': obj['motion_status'],
                'draw_box': draw_box,
                'age_s': round(age, 3), 'source': obj['source'],
                'trajectory': path, 'history_length': len(obj['history']),
            })
            if len(result) >= max_display:
                break
        return result

    def export(self) -> Dict:
        return {str(tid): {
            'track_id': tid, 'class': obj['class'],
            'last_status': obj['motion_status'],
            'position': obj['position'], 'velocity': obj['velocity'],
            'speed': obj['speed'], 'dimensions': obj['dimensions'],
            'rotation': obj['rotation'], 'source': obj['source'],
            'history': list(obj['history']), 'trajectory': list(obj['path']),
        } for tid, obj in self.objects.items()}


def objects_to_local(objects: Iterable[Dict], pose, radius: float = 60.0) -> List[Dict]:
    """Project FGO world points/3-D bounding-box corners to ego heading-up XYZ."""
    p = np.asarray(pose, dtype=float).reshape(-1)
    if len(p) < 6:
        return []
    yaw, tx, ty, tz = map(float, (p[2], p[3], p[4], p[5]))
    c, s = math.cos(yaw), math.sin(yaw)

    def local(xyz):
        x, y, z = [float(a) for a in xyz]
        dx, dy = x - tx, y - ty
        return [c * dx + s * dy, -s * dx + c * dy, z - tz]

    output = []
    for obj in objects:
        try:
            center = local(obj['position'])
            if center[0] ** 2 + center[1] ** 2 > (radius + 15.0) ** 2:
                continue
            dims = obj['dimensions']
            length, width, height = (float(dims[k]) for k in ('length', 'width', 'height'))
            ang = math.radians(float(obj.get('rotation', {}).get('yaw', 0.0)))
            ca, sa = math.cos(ang), math.sin(ang)
            world_center = np.asarray(obj['position'], dtype=float)
            corners = []
            for zsign in (-1, 1):
                for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    dx, dy = a * length / 2, b * width / 2
                    w = world_center + np.array([ca*dx - sa*dy, sa*dx + ca*dy, zsign*height/2])
                    corners.append(local(w))
            path = [local(item['position']) for item in obj.get('trajectory', [])]
            output.append({
                'track_id': obj['track_id'], 'class': obj.get('class', 'vehicle'),
                'position_local': center, 'velocity': obj['velocity'],
                'speed': obj['speed'], 'motion_status': obj['motion_status'],
                'draw_box': bool(obj.get('draw_box', True)),
                'age_s': obj.get('age_s', 0.0),
                'box_corners_local': corners if obj.get('draw_box', True) else [],
                'trajectory_local': path, 'source': obj.get('source', ''),
                'history_length': obj.get('history_length', 0),
            })
        except (KeyError, ValueError, TypeError, IndexError):
            continue
    return output


import json
import sqlite3
from pathlib import Path


class ObjectTrackArchive:
    """Append-only, disk-backed lifetime history for every observed FGO track."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError('Refusing to overwrite old 4D object archive: ' + str(self.path))
        self.db = sqlite3.connect(str(self.path))
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=NORMAL')
        self.db.execute('CREATE TABLE observations (track_id INTEGER NOT NULL, frame_id INTEGER NOT NULL, timestamp REAL NOT NULL, state_json TEXT NOT NULL, PRIMARY KEY(track_id,frame_id)) WITHOUT ROWID')
        self.db.execute('CREATE TABLE ego_poses (frame_id INTEGER PRIMARY KEY, timestamp REAL NOT NULL, state_json TEXT NOT NULL)')
        self.db.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)')
        self.db.executemany('INSERT INTO metadata VALUES (?,?)', [
            ('source', 'temporal_fusion_fgo'),
            ('coordinate_frame', 'LVIMOT estimated local world'),
            ('ground_truth_used', 'false'),
        ])
        self.db.commit()
        self.pending = 0

    def append(self, track_id, state):
        if self.db is None:
            raise RuntimeError('Object archive is closed')
        self.db.execute('INSERT OR REPLACE INTO observations VALUES (?,?,?,?)',
                        (int(track_id), int(state['frame_id']), float(state['timestamp']),
                         json.dumps(state, separators=(',', ':'), allow_nan=False)))
        self.pending += 1
        if self.pending >= 32:
            self.flush()

    def append_ego(self, state):
        if self.db is None:
            raise RuntimeError('Object archive is closed')
        self.db.execute('INSERT OR REPLACE INTO ego_poses VALUES (?,?,?)',
                        (int(state['frame_id']), float(state['timestamp']),
                         json.dumps(state, separators=(',', ':'), allow_nan=False)))
        self.pending += 1
        if self.pending >= 32:
            self.flush()

    def ego_history(self):
        self.flush()
        return [json.loads(row[0]) for row in self.db.execute(
            'SELECT state_json FROM ego_poses ORDER BY timestamp,frame_id')]

    def flush(self):
        if self.db is not None:
            self.db.commit()
            self.pending = 0

    def count_objects(self):
        return int(self.db.execute('SELECT COUNT(DISTINCT track_id) FROM observations').fetchone()[0])

    def tracks(self):
        self.flush()
        result = {}
        cursor = self.db.execute('SELECT track_id,state_json FROM observations ORDER BY track_id,timestamp,frame_id')
        for tid, raw in cursor:
            state = json.loads(raw)
            entry = result.setdefault(str(tid), {'track_id': tid, 'source': 'temporal_fusion_fgo',
                                                  'history': []})
            entry['history'].append(state)
        for entry in result.values():
            history = entry['history']
            last = history[-1]
            entry.update({k: last[k] for k in ('position', 'velocity', 'speed', 'dimensions',
                         'rotation', 'class', 'motion_status')})
            entry['last_status'] = last['motion_status']
            entry['trajectory'] = _trajectory_from_history(history)
            entry['last_frame'] = last['frame_id']
            entry['last_timestamp'] = last['timestamp']
        return result

    def summary(self):
        self.flush()
        return {'database': str(self.path.resolve()), 'total_ids': self.count_objects(),
                'total_observations': int(self.db.execute('SELECT COUNT(*) FROM observations').fetchone()[0]),
                'source': 'temporal_fusion_fgo', 'ground_truth_used': False}

    def close(self):
        if self.db is not None:
            self.flush()
            self.db.close()
            self.db = None


def _trajectory_from_history(history, min_distance=0.15):
    """Observed positions only; include stops but do not interpolate gaps."""
    path = []
    last_moving = False
    for state in history:
        position = state['position']
        moving = state['motion_status'] == 'MOVING'
        if not path:
            path.append({'timestamp': state['timestamp'], 'position': position,
                         'frame_id': state['frame_id']})
        elif moving or (last_moving and not moving):
            if np.linalg.norm(np.asarray(position) - np.asarray(path[-1]['position'])) >= min_distance:
                path.append({'timestamp': state['timestamp'], 'position': position,
                             'frame_id': state['frame_id']})
        last_moving = moving
    return path


