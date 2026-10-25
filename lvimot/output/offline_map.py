"""Persistent, evaluation-free, offline 4D map objects and combined scene export.

The object archive accepts only measured FGO states forwarded by the existing
mapper. This file NEVER supplies state to MOT, FGO or static-point masking.
A standalone HTML map, colored PLY and structured JSON are generated without
CARLA ground truth or any web dependencies.
"""
from __future__ import annotations

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


import json
import math
import os
import sqlite3
from pathlib import Path

import numpy as np


from lvimot.mapping_4d.multiobject_trajectory_management import ObjectTrackArchive


def box_corners(obj):
    center = np.asarray(obj['position'], dtype=np.float64).reshape(3)
    dims = obj['dimensions']
    l, w, h = [float(dims[k]) for k in ('length', 'width', 'height')]
    yaw = math.radians(float(obj.get('rotation', {}).get('yaw', 0.0)))
    co, si = math.cos(yaw), math.sin(yaw)
    output = []
    for zs in (-1, 1):
        for xs, ys in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            x, y = xs * l / 2.0, ys * w / 2.0
            output.append((center + np.asarray([co * x - si * y, si * x + co * y, zs * h / 2.0])).tolist())
    return output


_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def _sample_line(a, b, spacing=0.20):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / spacing)) + 1)
    return np.linspace(a, b, num=n, endpoint=True, dtype=np.float32)


def _object_points(objects):
    """Colored geometry-only annotation of observed paths + last valid boxes."""
    geometry = []
    colors = []
    nparked = 0
    moving_paths = 0
    for tid, obj in sorted(objects.items(), key=lambda item: int(item[0])):
        state = obj['last_status']
        if state == 'PARKED_CANDIDATE':
            color = (135, 225, 158)
            nparked += 1
        elif state == 'STOPPED':
            color = (245, 181, 68)
        else:
            color = (73, 217, 245)
        corners = box_corners(obj)
        # Last *measured* box, not a future or GT-predicted position.
        for a, b in _EDGES:
            section = _sample_line(corners[a], corners[b])
            geometry.append(section)
            colors.append(np.tile(np.asarray(color, dtype=np.uint8), (len(section), 1)))
        trajectory = [sample['position'] for sample in obj['trajectory']]
        if len(trajectory) >= 2:
            moving_paths += 1
        for a, b in zip(trajectory, trajectory[1:]):
            # Never create a line across a giant detection gap (false path).
            if np.linalg.norm(np.asarray(b) - np.asarray(a)) > 8.0:
                continue
            section = _sample_line(a, b)
            geometry.append(section)
            colors.append(np.tile(np.asarray((36, 224, 245), dtype=np.uint8), (len(section), 1)))
    if geometry:
        return np.vstack(geometry), np.vstack(colors), nparked, moving_paths
    return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), 0, 0


def _write_json_atomic(path, content):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    try:
        temp.write_text(json.dumps(content, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(str(temp), str(path))
    finally:
        if temp.exists():
            temp.unlink()


def _write_unified_ply(path, mapper, road, overlay, overlay_colors, road_acc=None):
    """Stream the WHOLE disk-backed route map into one colored sensor-only PLY."""
    acc = getattr(mapper, 'accumulated_map', None)
    if acc is not None and acc.connection is not None:
        acc.flush()
        count = int(acc.committed_voxel_count)
    else:
        count = len(mapper.get_static_map_points())
    if road_acc is not None and road_acc.connection is not None:
        road_acc.flush()
        road_count = int(road_acc.committed_voxel_count)
    else:
        road_count = len(road)
    total = count + road_count + len(overlay)
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment LVIMOT offline 4D geometry: gray static, green measured road, cyan paths, amber/green object boxes\n'
              'comment Object IDs, timestamps and statuses are in offline_4d_scene.json\n'
              'comment Sensor-estimated LVIMOT local world; no CARLA ground truth\n'
              'element vertex %d\n' % total)
    header += 'property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'
    packet_type = np.dtype([('xyz', '<f4', (3,)), ('rgb', 'u1', (3,))])

    def packet(points, color):
        values = np.empty(len(points), dtype=packet_type)
        values['xyz'] = np.asarray(points, dtype='<f4').reshape(-1, 3)
        values['rgb'] = color
        return values.tobytes(order='C')

    temp = path.with_name(path.name + '.tmp')
    try:
        with temp.open('wb') as output:
            output.write(header.encode('ascii'))
            if acc is not None and acc.connection is not None:
                cursor = acc.connection.execute('SELECT sx/n,sy/n,sz/n FROM voxels ORDER BY ix,iy,iz')
                while True:
                    rows = cursor.fetchmany(8192)
                    if not rows:
                        break
                    output.write(packet(rows, (154, 172, 187)))
            else:
                output.write(packet(mapper.get_static_map_points(), (154, 172, 187)))
            if road_acc is not None and road_acc.connection is not None:
                cursor = road_acc.connection.execute('SELECT sx/n,sy/n,sz/n FROM voxels ORDER BY ix,iy,iz')
                while True:
                    rows = cursor.fetchmany(8192)
                    if not rows:
                        break
                    output.write(packet(rows, (101, 199, 111)))
            elif len(road):
                output.write(packet(road, (101, 199, 111)))
            if len(overlay):
                output.write(packet(overlay, overlay_colors))
        os.replace(str(temp), str(path))
    finally:
        if temp.exists():
            temp.unlink()
    return total, count, road_count


def _recorded_capture_path(export_dir):
    """Get the exact sensor recording for THIS completed run, never the newest.

    This identifier is read only by post-estimation evaluation/visualization.
    """
    output = Path(export_dir)
    results_path = output / 'live_results.json'
    if not results_path.is_file():
        return None
    try:
        results = json.loads(results_path.read_text(encoding='utf-8'))
        if not results.get('ground_truth_evaluation_enabled', False):
            return None
        recording = str(results.get('raw_sensor_recording') or '').replace('\\', '/')
        capture_name = recording.split('/')[-1]
        if not capture_name.startswith('capture_') or capture_name in ('capture_', '.', '..'):
            return None
        capture = output / 'sensor_recordings' / capture_name
        manifest = json.loads((capture / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('format') != 'LVIMOT_RAW_SENSOR_ARCHIVE_V1':
            return None
        if manifest.get('frames') != 'frames.jsonl':
            return None
        if not (capture / 'frames.jsonl').is_file():
            return None
        return capture
    except (OSError, ValueError, TypeError, KeyError, UnicodeError):
        return None


def _independent_reference_matches_capture(reference_dir, capture):
    """Never use a 360 reference rendered for a different driving route."""
    reference_dir = Path(reference_dir)
    manifest_path = reference_dir / 'reference_manifest.json'
    reference_path = reference_dir / 'CARLA_independent_360_reference.ply'
    if not (manifest_path.is_file() and reference_path.is_file()
            and reference_path.stat().st_size > 100):
        return False
    try:
        meta = json.loads(manifest_path.read_text(encoding='utf-8'))
        if meta.get('format') != 'CARLA_INDEPENDENT_RENDERED_360_REFERENCE_V1':
            return False
        if not meta.get('reference_only') or meta.get('gt_used_by_estimator') is not False:
            return False
        # The reference exporter writes an absolute path to its source capture.
        return Path(meta['recorded_capture']).resolve() == Path(capture).resolve()
    except (OSError, ValueError, TypeError, KeyError, UnicodeError):
        return False


def _reference_ply_candidates(export_dir):
    """Whole-scene observed-area CARLA reference candidates for this exact run.

    CARLA_reference_road_surface.ply is an OpenDRIVE road-only sampling,
    NOT CARLA's full 3D scene. It must never be displayed as 'CARLA map'.
    """
    output = Path(export_dir)
    capture = _recorded_capture_path(output)
    if capture is None:
        return []
    dirs = []
    # Standard output and retries for the matching raw recording.
    dirs.extend(sorted(capture.glob('original_carla_map*')))
    # Allow copies next to live_results.json only when provenance agrees.
    dirs.append(output)
    return [folder / 'CARLA_independent_360_reference.ply'
            for folder in dirs if folder.is_dir()
            and _independent_reference_matches_capture(folder, capture)]


def _first_existing_path(paths):
    for path in paths:
        p = Path(path)
        if p.is_file():
            return p
    return None


def _read_binary_ply_xyz(path):
    path = Path(path)
    with path.open('rb') as stream:
        header = bytearray()
        while not header.endswith(b'end_header\n'):
            ch = stream.read(1)
            if not ch or len(header) > 16384:
                raise ValueError('Invalid PLY header: ' + str(path))
            header.extend(ch)
        lines = header.decode('ascii').splitlines()
        if 'format binary_little_endian 1.0' not in lines:
            raise ValueError('Only binary little-endian PLY is supported: ' + str(path))
        count = next(int(line.split()[2]) for line in lines if line.startswith('element vertex '))
        vertices = np.frombuffer(stream.read(count * 12), dtype='<f4')
        if vertices.size != count * 3:
            raise ValueError('PLY vertex data is truncated: ' + str(path))
        return vertices.reshape(-1, 3).astype(np.float32)


def _sample_points(points, max_points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if not len(points):
        return []
    step = max(1, int(math.ceil(len(points) / float(max_points))))
    return points[::step].astype(float).round(3).tolist()


def _reference_world_to_local(points, first_gt_pose):
    """Evaluation/viewer ONLY: inverse of align_local_map_to_carla.

    CARLA point clouds are in the simulator world, whereas the accumulated
    sensor map, measured road, and FGO object histories are in the LVIMOT
    estimated local world.  Correcting only the display reference leaves
    every sensor-estimated state, output PLY, and acceptance metric untouched.
    """
    from lvimot.output.evaluation.map_geometry import _reference_pose

    rotation, translation = _reference_pose(first_gt_pose)
    world = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(world).all():
        raise ValueError('Non-finite CARLA reference coordinates')
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError('Invalid first evaluation-only CARLA ego pose')
    return ((world - translation) @ rotation).astype(np.float32)


def _recorded_first_gt_pose(export_dir):
    """Use first ego GT pose of the exact saved run for viewer alignment only."""
    capture = _recorded_capture_path(export_dir)
    if capture is None:
        return None
    try:
        with (capture / 'frames.jsonl').open('r', encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                pose = json.loads(line).get('ground_truth_ego_pose_evaluation_only')
                if isinstance(pose, dict) and 'location' in pose and 'rotation' in pose:
                    return pose
                break
    except (OSError, ValueError, TypeError, KeyError, UnicodeError):
        return None
    return None


def _sample_carla_reference_local(export_dir, first_gt_pose, maximum=35000):
    """Produce comparable, non-estimator CARLA reference points for the HTML."""
    reference = _first_existing_path(_reference_ply_candidates(export_dir))
    if reference is None or first_gt_pose is None:
        return reference, [], False
    world = _read_binary_ply_xyz(reference)
    # Transform AFTER sampling to minimize overhead on large reference PLYs.
    world_sample = np.asarray(_sample_points(world, maximum), dtype=np.float32)
    if world_sample.size == 0:
        return reference, [], False
    local = _reference_world_to_local(world_sample, first_gt_pose)
    return reference, local.astype(float).round(3).tolist(), True


def load_estimated_ego_history(export_dir, scene):
    """Recover legacy runs exclusively from their saved estimated frame states.

    Raw sensor GT poses and CARLA reference alignment are never candidates.
    Old runs with no saved estimate remain empty and are reported honestly.
    """
    if 'ego_history' in scene:
        return scene['ego_history']
    path = Path(export_dir) / 'live_frames.jsonl'
    history = []
    if path.is_file():
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                state = row.get('state_estimation', {})
                pose = np.asarray(state.get('pose', []), dtype=float).reshape(-1)
                now = row.get('timestamp')
                if pose.size != 6 or now is None or not np.isfinite(np.r_[pose, float(now)]).all():
                    continue
                velocity = state.get('velocity')
                if velocity is not None:
                    velocity = np.asarray(velocity, dtype=float).reshape(3).tolist()
                    if not np.isfinite(velocity).all():
                        velocity = None
                history.append({'frame_id': int(row.get('frame', state.get('frame_id', len(history)))),
                                'timestamp': float(now), 'position': pose[3:].tolist(),
                                'yaw': float(np.degrees(pose[2])), 'velocity': velocity,
                                'source': 'saved_state_estimation', 'ground_truth_used': False})
    return sorted(history, key=lambda item: (item['timestamp'], item['frame_id']))


def _offline_html(point_sets, objects, stats, ego_history=None):
    """Browser-offline 3D/BEV viewer. Embedded samples, no CDN or local fetch."""
    payload = json.dumps({'layers': point_sets, 'objects': objects, 'stats': stats, 'ego_history': ego_history or []}, separators=(',', ':'), allow_nan=False)
    payload = payload.replace('<', '\\u003c')
    return '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LVIMOT Offline 4D Map</title>
<style>
* { box-sizing: border-box; }
html, body { width: 100%; height: 100%; margin: 0; }
body { color: #111; background: #fff; font: 14px Arial, Helvetica, sans-serif; }
header { height: 48px; padding: 13px 16px; border-bottom: 1px solid #bbb; font-size: 17px; }
main { display: grid; grid-template-columns: minmax(0,1fr) 210px; height: calc(100% - 48px); }
canvas { display: block; width: 100%; height: 100%; background: #fff; cursor: grab; touch-action: none; }
canvas:active { cursor: grabbing; }
aside { border-left: 1px solid #bbb; padding: 19px 15px; background: #fafafa; }
label { display: block; margin: 0 0 9px; }
select { display: block; background: #fff; color: #111; border: 1px solid #888; border-radius: 0; width: 100%; padding: 6px; margin-bottom: 22px; font: inherit; }
.checkbox { display: flex; align-items: center; gap: 7px; margin-bottom: 14px; }
.checkbox input { width: 15px; height: 15px; margin: 0; accent-color: #111; }
.section-gap { margin-bottom: 25px; }
.timer { display: flex; align-items: center; gap: 8px; }
.timer input { width: 100%; min-width: 0; accent-color: #111; }
.timer output { min-width: 45px; text-align: right; white-space: nowrap; font-size: 12px; }
@media(max-width:640px) { main { grid-template-columns:minmax(0,1fr) 155px; } aside { padding:12px 8px; } }
</style>
</head>
<body>
<header>LVIMOT Offline 4D Map</header>
<main>
<canvas id="view" aria-label="3D point cloud map"></canvas>
<aside>
<label for="mode">View</label>
<select id="mode"><option value="3d">3D</option><option value="bev">Top view</option></select>
<label class="checkbox" for="showLVIMOT"><input id="showLVIMOT" type="checkbox" checked>LVIMOT map</label>
<label class="checkbox" for="showCarla"><input id="showCarla" type="checkbox" __CARLA_TOGGLE_STATE__>CARLA map</label>
<div id="carlaMissing" style="__CARLA_MISSING_STYLE__font-size:12px;color:#555;margin:-6px 0 12px">Independent CARLA 360° reference not found for this run.</div>
<label class="checkbox section-gap" for="showVehicles"><input id="showVehicles" type="checkbox" checked>Vehicles and paths</label>
<label class="checkbox" for="showEgo"><input id="showEgo" type="checkbox" checked>Ego position and path</label>
<label for="time">Time</label>
<div class="timer"><input id="time" type="range" min="0" max="1000" value="1000"><output id="timeValue">0.0 s</output></div>
</aside>
</main>
<script id="payload" type="application/json">__SCENE_PAYLOAD__</script>
<script>
'use strict';
const data = JSON.parse(document.getElementById('payload').textContent);
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const el = id => document.getElementById(id);
let yaw = .6, pitch = .52, zoom = 1, mouse = null, offsetX = 0, offsetY = 0;
const colors = {
  static: '#9aa6b2',
  road: '#57b95f',
  carla: '#7d6df2',
  path: '#24d0f5',
  movingBox: '#24d0f5',
  stoppedBox: '#f5b544',
  parkedBox: '#57b95f',
  label: '#238636',
  ego: '#c62828',
  egoPath: '#9c2727'
};
const egoHistory = (data.ego_history || []).slice().sort((a,b)=>a.timestamp-b.timestamp || a.frame_id-b.frame_id);
const allStates = [...egoHistory.map(h=>h.timestamp), ...Object.values(data.objects || {}).flatMap(o => (o.history || []).map(h => h.timestamp))];
let tmin = allStates.reduce((a,b) => Math.min(a,b), Infinity);
let tmax = allStates.reduce((a,b) => Math.max(a,b), -Infinity);
if (!Number.isFinite(tmin)) { tmin = 0; tmax = 0; }
// Fit the actually rendered spatial layers once. Panning/orbit controls remain.
const fitPoints = [...(data.layers.static || []), ...(data.layers.road || []), ...(data.layers.carla || []), ...egoHistory.map(h=>h.position)];
let mapSpan = 160;
if (fitPoints.length) {
  let xmin = Infinity, ymin = Infinity, xmax = -Infinity, ymax = -Infinity;
  for (const p of fitPoints) {
    if (!Number.isFinite(p[0]) || !Number.isFinite(p[1])) continue;
    xmin = Math.min(xmin, p[0]); ymin = Math.min(ymin, p[1]);
    xmax = Math.max(xmax, p[0]); ymax = Math.max(ymax, p[1]);
  }
  if (Number.isFinite(xmin)) {
    offsetX = (xmin+xmax)/2; offsetY = (ymin+ymax)/2;
    mapSpan = Math.max(45, Math.hypot(xmax-xmin,ymax-ymin)*1.15);
  }
}
const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
function corners(o) {
  const p = o.position, d = o.dimensions;
  const a = (o.rotation?.yaw || 0) * Math.PI / 180;
  const c = Math.cos(a), s = Math.sin(a), r = [];
  for (const z of [-.5,.5]) {
    for (const [x,y] of [[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]) {
      const X = x * d.length, Y = y * d.width;
      r.push([p[0]+c*X-s*Y, p[1]+s*X+c*Y, p[2]+z*d.height]);
    }
  }
  return r;
}
function project(p, W, H) {
  const x = p[0]-offsetX, y = p[1]-offsetY, z = p[2];
  const c = Math.cos(yaw), s = Math.sin(yaw);
  let u = c*x-s*y, v = s*x+c*y;
  if (el('mode').value === '3d') v = v*Math.sin(pitch)-z*Math.cos(pitch);
  const q = Math.min(W,H)/(mapSpan/zoom);
  return [W/2+u*q,H/2-v*q];
}
function line(a,b,color,W,H,weight=2,dashed=false) {
  const A=project(a,W,H),B=project(b,W,H);
  ctx.strokeStyle=color;ctx.lineWidth=weight;ctx.setLineDash(dashed?[4,3]:[]);
  ctx.beginPath();ctx.moveTo(...A);ctx.lineTo(...B);ctx.stroke();ctx.setLineDash([]);
}
function dots(points,color,W,H,size=1.25,alpha=1) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle=color;
  for (const p of points || []) {
    const [x,y] = project(p,W,H);
    if (x>=0 && x<W && y>=0 && y<H) ctx.fillRect(x,y,size,size);
  }
  ctx.restore();
}
function drawEgo(t,W,H) {
  if (!el('showEgo').checked) return;
  const history=egoHistory.filter(h=>h.timestamp<=t);
  if (!history.length) return;
  // Samples only up to this time; no interpolation or forward prediction.
  for (let i=1;i<history.length;i++) {
    const a=history[i-1], b=history[i];
    if (b.frame_id!==a.frame_id+1 || b.timestamp-a.timestamp>1 || b.timestamp<=a.timestamp) continue;
    line(a.position,b.position,colors.egoPath,W,H,2.2);
  }
  const snap=history[history.length-1];
  // Show the last accepted estimate, also during stops or sparse sampling.
  const a=snap.yaw*Math.PI/180, c=Math.cos(a), s=Math.sin(a), p=snap.position;
  const vertices=[[2,0],[-1.3,-.9],[-1.3,.9]].map(([x,y])=>
    project([p[0]+c*x-s*y,p[1]+s*x+c*y,p[2]],W,H));
  ctx.fillStyle=colors.ego;ctx.strokeStyle='#fff';ctx.lineWidth=1.2;
  ctx.beginPath();ctx.moveTo(...vertices[0]);
  ctx.lineTo(...vertices[1]);ctx.lineTo(...vertices[2]);ctx.closePath();ctx.fill();ctx.stroke();
  const [x,y]=project(p,W,H);
  ctx.font='bold 12px Arial, Helvetica, sans-serif';ctx.fillStyle=colors.ego;
  ctx.fillText('EGO',x+9,y+15);
}
function draw() {
  const W=canvas.clientWidth,H=canvas.clientHeight;
  if (!W || !H) return;
  const ratio=window.devicePixelRatio || 1;
  canvas.width=Math.round(W*ratio);canvas.height=Math.round(H*ratio);
  ctx.setTransform(ratio,0,0,ratio,0,0);
  ctx.clearRect(0,0,W,H);
  if (el('showCarla').checked) dots(data.layers.carla || [], colors.carla, W, H, 1.35, 0.68);
  if (el('showLVIMOT').checked) {
    dots(data.layers.static || [], colors.static, W, H, 1.15, 0.78);
    dots(data.layers.road || [], colors.road, W, H, 1.5, 0.95);
  }
  const t=tmin+(tmax-tmin)*Number(el('time').value)/1000;
  el('timeValue').textContent=(t-tmin).toFixed(1)+' s';
  drawEgo(t,W,H);
  if (!el('showVehicles').checked) return;
  for (const [id,o] of Object.entries(data.objects || {})) {
    const history=(o.history || []).filter(h=>h.timestamp<=t);
    if (!history.length) continue;
    const snap=history[history.length-1],stale=t-snap.timestamp>.5;
    const positions=(o.trajectory || []).filter(p=>p.timestamp<=t);
    for(let i=1;i<positions.length;i++) {
      const a=positions[i-1].position,b=positions[i].position;
      if(Math.hypot(a[0]-b[0],a[1]-b[1])<8) line(a,b,colors.path,W,H,2);
    }
    // Keep genuinely persistent parked candidates, not invented stale moving boxes.
    if(stale && snap.motion_status!=='PARKED_CANDIDATE') continue;
    const cs=corners(snap),isParked=snap.motion_status==='PARKED_CANDIDATE';
    const isStopped=snap.motion_status==='STOPPED';
    const boxColor = isParked ? colors.parkedBox : (isStopped ? colors.stoppedBox : colors.movingBox);
    for(const [a,b] of edges) line(cs[a],cs[b],boxColor,W,H,isParked?2.3:1.7,isStopped);
    const [tx,ty]=project(snap.position,W,H);
    ctx.fillStyle=colors.label;ctx.font='12px Arial, Helvetica, sans-serif';
    ctx.fillText('ID '+id,tx+7,ty-8);
  }
}
function redraw(){requestAnimationFrame(draw)}
for(const id of ['mode','showLVIMOT','showCarla','showVehicles','showEgo','time']) el(id).addEventListener('input',redraw);
canvas.addEventListener('pointerdown',e=>{mouse=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener('pointermove',e=>{
  if(!mouse) return;
  yaw+=(e.clientX-mouse[0])*.009;
  pitch=Math.min(1.4,Math.max(.15,pitch+(e.clientY-mouse[1])*.005));
  mouse=[e.clientX,e.clientY];redraw();
});
canvas.addEventListener('pointerup',()=>{mouse=null});
canvas.addEventListener('pointercancel',()=>{mouse=null});
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.2,Math.min(10,zoom*Math.exp(-e.deltaY*.001)));redraw()},{passive:false});
window.addEventListener('resize',redraw);redraw();
</script>
</body>
</html>'''.replace('__SCENE_PAYLOAD__', payload).replace('__CARLA_TOGGLE_STATE__', 'checked' if point_sets.get('carla') else 'disabled').replace('__CARLA_MISSING_STYLE__', 'display:none;' if point_sets.get('carla') else '')


def export_offline_4d_scene(mapper, export_dir, first_gt_pose_evaluation_only=None):
    """Automatically export one complete offline scene and an offline HTML viewer.

    The colored PLY is viewable in common point-cloud tools; true semantic
    object IDs/frames/3-D boxes are carried in companion JSON/SQLite files.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    archive = getattr(mapper, 'object_archive', None)
    objects = archive.tracks() if archive is not None else mapper.individual_object_scene.export()
    ego_history = (archive.ego_history() if archive is not None else
                   list(getattr(mapper, 'estimated_ego_history', [])))
    road_layer = getattr(mapper, 'road_surface_layer', None)
    road = np.asarray(road_layer.points(), dtype=np.float32).reshape(-1, 3) if road_layer is not None else np.empty((0, 3), np.float32)
    overlay, overlay_colors, nparked, npaths = _object_points(objects)
    road_acc = getattr(getattr(road_layer, 'mapper', None), 'accumulated_map', None)
    scene_path = export_dir / 'offline_4d_scene.json'
    ply_path = export_dir / 'offline_4d_map.ply'
    html_path = export_dir / 'offline_4d_map.html'
    manifest_path = export_dir / 'offline_4d_map_manifest.json'
    meta = {'coordinate_frame': 'LVIMOT estimated local world', 'estimator_inputs': ['RGB', 'LiDAR', 'IMU'],
            'ground_truth_used': False, 'object_state_source': 'temporal_fusion_fgo_only',
            'parked_definition': 'speed <= 0.30 m/s continuously for >= 2 s; candidate only',
            'not_detected_vehicles': 'Cars never detected by MOT/FGO remain anonymous LiDAR geometry, not invented boxes',
            'layers': {'static': 'accumulated complete route, gray', 'measured_road': 'green',
                       'observed_parked_boxes': 'green', 'stopped_boxes': 'amber', 'motion_paths': 'cyan', 'confirmed_vehicle_ids': 'green',
                       'estimated_ego_marker': 'red', 'estimated_ego_path': 'dark red'},
            'ego_state_source': 'temporal_fusion_fgo_only', 'ego_history': ego_history,
            'objects': objects}
    _write_json_atomic(scene_path, meta)
    count, nstatic, nroad = _write_unified_ply(ply_path, mapper, road, overlay, overlay_colors, road_acc)
    # Tiny source-sampled layers, only for the embedded viewer; PLY remains complete.
    sample_static = []
    acc = getattr(mapper, 'accumulated_map', None)
    if acc is not None and acc.connection is not None:
        stride = max(1, int(math.ceil(nstatic / 16000.0)))
        cursor = acc.connection.execute('SELECT sx/n,sy/n,sz/n FROM voxels ORDER BY ix,iy,iz')
        for i, xyz in enumerate(cursor):
            if i % stride == 0:
                sample_static.append([round(float(v), 3) for v in xyz])
    else:
        pts = mapper.get_static_map_points()
        stride = max(1, int(math.ceil(len(pts) / 16000.0)))
        sample_static = pts[::stride].astype(float).round(3).tolist()
    if road_acc is not None and road_acc.connection is not None:
        road_sample = []
        road_step = max(1, math.ceil(nroad / 12000.0))
        cursor = road_acc.connection.execute('SELECT sx/n,sy/n,sz/n FROM voxels ORDER BY ix,iy,iz')
        for i, xyz in enumerate(cursor):
            if i % road_step == 0:
                road_sample.append([round(float(v), 3) for v in xyz])
    else:
        road_step = max(1, math.ceil(len(road) / 12000.0))
        road_sample = road[::road_step].astype(float).round(3).tolist()

    # The reference PLY is in CARLA world. It MUST be transformed to the
    # estimated local map frame before sharing the same HTML canvas.
    # This parameter is passed only after independent evaluation; it never
    # enters LVIMOT mapping, FGO, localization or the exported sensor-only PLY.
    carla_reference_path, carla_sample, carla_aligned = _sample_carla_reference_local(
        export_dir, first_gt_pose_evaluation_only
    )

    stats = {'ego_observations': len(ego_history), 'static_voxels': nstatic, 'road_voxels': nroad, 'object_ids': len(objects),
             'parked_candidates': nparked, 'moving_paths': npaths, 'unified_ply_vertices': count,
             'carla_reference_available': carla_reference_path is not None,
             'carla_reference_points_sampled': len(carla_sample),
             'carla_reference_type': ('independent_rendered_360_geometry' if carla_aligned else 'unavailable'),
           'carla_reference_aligned_for_display': bool(carla_aligned)}
    html_path.write_text(_offline_html({'static': sample_static, 'road': road_sample, 'carla': carla_sample}, objects, stats, ego_history), encoding='utf-8')
    paths = {'scene_ply': str(ply_path), 'scene_json': str(scene_path), 'offline_viewer': str(html_path)}
    if carla_reference_path is not None:
        paths['carla_reference_evaluation_only'] = str(carla_reference_path)
    if archive is not None:
        paths['object_archive_sqlite'] = str(archive.path)
    if acc is not None:
        paths['static_archive_sqlite'] = str(acc.path)
    if road_acc is not None:
        paths['measured_road_archive_sqlite'] = str(road_acc.path)
    manifest = {'status': 'EXPORTED', 'coordinate_frame': meta['coordinate_frame'],
                'ground_truth_used_for_estimation': False, 'files': paths,
                'counts': stats, 'limitations': meta['not_detected_vehicles'],
              'reference_type': 'independent_360_geometry_only',
              'reference_display_alignment': ('inverse_first_evaluation_gt_ego_pose'
                  if carla_aligned else 'unavailable_not_displayed')}
    _write_json_atomic(manifest_path, manifest)
    return {'offline_scene_ply': str(ply_path), 'offline_scene_json': str(scene_path),
            'offline_scene_html': str(html_path), 'offline_scene_manifest': str(manifest_path),
            'offline_scene_counts': stats}


def link_carla_reference_evaluation_only(export_dir):
    """After independent validation, link CARLA reference, never fuse into map."""
    output = Path(export_dir)
    ref = _first_existing_path(_reference_ply_candidates(output))
    manifest_path = output / 'offline_4d_map_manifest.json'
    if ref is None or not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest.setdefault('evaluation_only', {})['carla_reference'] = str(ref)
    manifest.setdefault('files', {})['carla_reference_evaluation_only'] = str(ref)
    _write_json_atomic(manifest_path, manifest)
    return True
