"""Disk-backed, sensor-estimated static geometry across the complete driven route.

The live mapper may evict old geometry from RAM. This separate voxel archive
retains integrated observations on disk and streams a final PLY without loading
the complete map into memory. No simulator pose or geometry is accepted here.
The archive follows the estimated trajectory and does not perform loop closure.
"""
from __future__ import annotations

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[3]))


import json
import os
import sqlite3
from pathlib import Path

import numpy as np


class AccumulatedStaticMap:
    def __init__(self, database_path, voxel_size=0.25, commit_interval=10):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError('Refusing to mix separate map runs: ' + str(self.path))
        self.voxel_size = float(voxel_size)
        if not np.isfinite(self.voxel_size) or self.voxel_size <= 0:
            raise ValueError('Accumulated map voxel size must be positive')
        self.commit_interval = max(1, int(commit_interval))
        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('PRAGMA synchronous=NORMAL')
        self.connection.execute('PRAGMA cache_size=-4096')
        self.connection.execute('CREATE TABLE voxels (ix INTEGER, iy INTEGER, iz INTEGER, sx REAL, sy REAL, sz REAL, n INTEGER, PRIMARY KEY(ix,iy,iz)) WITHOUT ROWID')
        self.connection.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)')
        self.connection.executemany('INSERT INTO metadata VALUES (?,?)', [
            ('coordinate_frame', 'LVIMOT estimated local world; sensor-only'),
            ('ground_truth_used', 'false'), ('loop_closure_applied', 'false'),
            ('voxel_size_m', str(self.voxel_size)),
        ])
        self.connection.commit()
        self.frames_integrated = 0
        self.input_points = 0
        self.last_frame = None
        self.committed_voxel_count = 0

    def add(self, points_world, frame_id):
        if self.connection is None:
            raise RuntimeError('Accumulated map archive is closed')
        points = np.asarray(points_world, dtype=float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            return
        keys, inverse = np.unique(np.floor(points / self.voxel_size).astype(np.int64), axis=0, return_inverse=True)
        sums = np.zeros((len(keys), 3), dtype=float)
        np.add.at(sums, inverse, points)
        counts = np.bincount(inverse, minlength=len(keys))
        representatives = sums / counts[:, None]
        rows = [(int(k[0]), int(k[1]), int(k[2]), float(p[0]), float(p[1]), float(p[2]), 1)
                for k, p in zip(keys, representatives)]
        self.connection.executemany(
            'INSERT INTO voxels VALUES (?,?,?,?,?,?,?) ON CONFLICT(ix,iy,iz) DO UPDATE SET sx=sx+excluded.sx, sy=sy+excluded.sy, sz=sz+excluded.sz, n=n+1', rows)
        self.frames_integrated += 1
        self.input_points += int(len(points))
        self.last_frame = int(frame_id)
        if self.frames_integrated % self.commit_interval == 0:
            self.flush()

    def flush(self):
        if self.connection is not None:
            self.committed_voxel_count = int(self.connection.execute('SELECT COUNT(*) FROM voxels').fetchone()[0])
            self.connection.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',
                                    ('integration_summary', json.dumps(self.summary())))
            self.connection.commit()

    def summary(self):
        return {
            'database': str(self.path.resolve()), 'voxel_size_m': self.voxel_size,
            'frames_integrated': self.frames_integrated, 'input_points_integrated': self.input_points,
            'last_frame': self.last_frame, 'last_committed_voxel_count': self.committed_voxel_count,
            'scope': 'all integrated sensor observations along the driven route; disk-backed',
            'coordinate_frame': 'LVIMOT estimated local world',
            'ground_truth_used': False, 'loop_closure_applied': False,
        }

    def export(self, ply_path):
        if self.connection is None:
            raise RuntimeError('Accumulated map archive is closed')
        self.flush()
        path = Path(ply_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = self.committed_voxel_count
        temporary = path.with_name(path.name + '.tmp')
        header = ('ply\nformat binary_little_endian 1.0\n'
                  'comment LVIMOT accumulated estimated map; GT-free; no loop closure\n'
                  'element vertex %d\nproperty float x\nproperty float y\nproperty float z\nend_header\n' % count)
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        try:
            with temporary.open('wb') as stream:
                stream.write(header.encode('ascii'))
                cursor = self.connection.execute('SELECT sx/n, sy/n, sz/n FROM voxels ORDER BY ix,iy,iz')
                while True:
                    batch = cursor.fetchmany(8192)
                    if not batch:
                        break
                    points = np.asarray(batch, dtype='<f4').reshape(-1, 3)
                    lo = np.minimum(lo, points.min(axis=0))
                    hi = np.maximum(hi, points.max(axis=0))
                    stream.write(points.tobytes(order='C'))
            os.replace(str(temporary), str(path))
        finally:
            if temporary.exists():
                temporary.unlink()
        result = self.summary()
        result.update({'ply': str(path.resolve()), 'voxel_count': count,
                       'bounds_xyz_m': [lo.tolist(), hi.tolist()] if count else None})
        path.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        return result

    def close(self):
        if self.connection is not None:
            self.flush()
            self.connection.close()
            self.connection = None
