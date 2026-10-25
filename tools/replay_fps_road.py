"""Measured raw replay including the live mapper archive and GUI rendering."""

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[1]))

import argparse
import cProfile
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--capture',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--limit',type=int);p.add_argument('--no-gt',action='store_true');p.add_argument('--profile',action='store_true')
    p.add_argument('--render',action='store_true');p.add_argument('--road',action='store_true');p.add_argument('--no-archive',action='store_true')
    p.add_argument('--async-preview',action='store_true')
    a=p.parse_args();project=Path(a.project).resolve();out=Path(a.output).resolve()
    capture=Path(a.capture).resolve()
    if out==capture or capture in out.parents:p.error('Output must be outside the original sensor capture')
    if a.limit is not None and a.limit<1:p.error('--limit must be positive')
    out.mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(out/'yolo_config');os.environ['LVIMOT_LIVE_MAP_SNAPSHOT']=str(out/'live_map_snapshot.json')
    os.environ['LVIMOT_LIVE_CAMERA_SNAPSHOT']=str(out/'live_camera_preview.png')

    import cv2
    import numpy as np
    import torch
    torch.set_num_threads(a.threads);cv2.setNumThreads(1);np.random.seed(42);cv2.setRNGSeed(42)
    sys.path.insert(0, str(project if (project / 'lvimot').is_dir() else project / 'src'))
    if (project / 'lvimot').is_dir():
        from lvimot.inputs.sensor_calibration import CarlaCalibration
    else:
        from carla_calibration import CarlaCalibration
    if (project / 'lvimot').is_dir():
        from lvimot.temporal_fusion.pipeline import CarlaLVIMOTCore
    else:
        from carla_pipeline_core import CarlaLVIMOTCore
    if (project / 'lvimot').is_dir():
        from lvimot.inputs.sensor_recording import iter_archive
    else:
        from carla_raw_sensor_archive import iter_archive
    if (project / 'lvimot').is_dir():
        from lvimot.output.live_dashboard import CarlaLiveVisualizer
    else:
        from carla_live_visualizer import CarlaLiveVisualizer
    if (project / 'lvimot').is_dir():
        from lvimot.output.map_export import write_binary_ply
    else:
        from carla_map_presentation import write_binary_ply
    if (project / 'lvimot').is_dir():
        from lvimot.output.evaluation.map_geometry import align_local_map_to_carla,compare_road_geometry
    else:
        from carla_map_validation import align_local_map_to_carla,compare_road_geometry

    from tools.validate_carla_depth_reference import _read_ply_xyz
    cap=Path(a.capture);manifest=json.loads((cap/'manifest.json').read_text());settings=manifest['settings']
    core=CarlaLVIMOTCore(CarlaCalibration(str(cap/'calibration.json')),detector_mode='camera_lidar',
        yolo_model=str(project/'models/yolov8n.pt'),yolo_confidence=settings['yolo_confidence'],
        yolo_image_size=settings['yolo_image_size'],yolo_device=a.device,**({'road_surface_enabled':True} if a.road else {}))
    if not a.no_archive:core.mapper_4d.enable_accumulated_map(out/'accumulated_maps')
    viewer=CarlaLiveVisualizer();rows=[];first=None;last=None
    presentation=None
    if a.async_preview:
        if (project / 'lvimot').is_dir():
            from lvimot.output.live_dashboard import DashboardPresentation
        else:
            from carla_live_presentation import DashboardPresentation
        presentation=DashboardPresentation(viewer)
    profiler=cProfile.Profile();start=time.perf_counter()
    def clean(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [clean(v) for v in value]
        return value
    try:
        with (out/'trace.jsonl').open('w') as trace:
            for index,item in enumerate(iter_archive(cap)):
                if a.limit and index>=a.limit:break
                first=first or item['ground_truth_ego_pose'];last=item['ground_truth_ego_pose']
                if a.profile and index==3:profiler.enable()
                frame_start=time.perf_counter()
                result=core.process(item['frame'],item['timestamp'],item['image'],item['lidar'],item['imu'],
                    ground_truth_ego_pose=None if a.no_gt else item['ground_truth_ego_pose'],
                    ground_truth_objects=None if a.no_gt else item['ground_truth_objects'])
                rendered=0.
                if a.render:
                    t=time.perf_counter()
                    road_layer=getattr(core,'road_surface_layer',None)
                    road_points=road_layer.points() if road_layer is not None else None
                    if presentation is not None:
                        presentation.submit(item['image'],result,core.mapper_4d.get_static_map_points(),road_points)
                    else:
                        viewer.render(item['image'],result,core.mapper_4d.get_static_map_points(),**({'road_points':road_points} if road_points is not None else {}))
                    rendered=(time.perf_counter()-t)*1000
                row=clean({'frame':item['frame'],'timestamp':item['timestamp'],'state':result['state_estimation'],
                    'tracks':result['tracking']['active_tracks'],'performance':result['performance'],
                    'render_ms':rendered,'loop_ms':(time.perf_counter()-frame_start)*1000,
                    'detector':core.camera_lidar_detector.last_diagnostics,
                    'evaluation_gt':core.ground_truth_history[-1] if not a.no_gt else [],
                    'evaluation_tracks':core.tracking_history[-1] if not a.no_gt else []})
                trace.write(json.dumps(row,separators=(',',':'))+'\n');trace.flush();rows.append(row)
                if (index+1)%20==0:print('frames',index+1,'seconds',round(time.perf_counter()-start,1),flush=True)
        if a.profile:profiler.disable();profiler.dump_stats(str(out/'profile.pstats'))
        if presentation is not None:presentation.close()
        points=core.mapper_4d.get_static_map_points();np.save(out/'static_map.npy',points);write_binary_ply(out/'live_map_latest.ply',points)
        result={'frames':len(rows),'seconds':time.perf_counter()-start,'device_requested':a.device,'torch':torch.__version__,
            'evaluation':clean(core.evaluation_summary()),'source_hashes':{str(f.relative_to(project)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((project/('lvimot' if (project/'lvimot').is_dir() else 'src')).rglob('*.py'))},
            'static_map_sha256':hashlib.sha256(points.tobytes()).hexdigest(),
            'median_loop_ms':float(np.median([r['loop_ms'] for r in rows[3:]])),
            'median_core_ms':float(np.median([r['performance']['total_ms'] for r in rows[3:]])),
            'median_render_ms':float(np.median([r['render_ms'] for r in rows[3:]]))}
        if presentation is not None:result['presentation']=presentation.summary()
        reference_path=cap/'CARLA_reference_road_surface.ply'
        if not reference_path.exists():reference_path=cap.parent.parent/'CARLA_reference_road_surface.ply'
        if reference_path.exists():
            reference=_read_ply_xyz(reference_path);center=np.array([last['location'][k] for k in ['x','y','z']])
            result['legacy_road']=compare_road_geometry(align_local_map_to_carla(points,first),reference,center)
            road_layer=getattr(core,'road_surface_layer',None)
            if road_layer is not None:
                road_points=road_layer.points();np.save(out/'road_map.npy',road_points);write_binary_ply(out/'measured_road_surface.ply',road_points)
                result['road_surface']=road_layer.summary()
                result['measured_road_agreement']=compare_road_geometry(align_local_map_to_carla(road_points,first),reference,center)
                result['measured_road_agreement'].update({'comparison':'Measured camera-labelled LiDAR road layer vs CARLA OpenDRIVE',
                    'scope':'camera-observed road surface only, final local neighborhood',
                    'road_metric_warning':'Predicted labels and sampled references can be wrong; this does not certify unobserved road or the full scene.'})
        (out/'summary.json').write_text(json.dumps(clean(result),indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ['source_hashes','evaluation']}),flush=True)
    finally:
        if presentation is not None:presentation.close()
        core.close()

if __name__ == '__main__':
    main()
