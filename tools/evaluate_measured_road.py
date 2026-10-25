"""Evaluate road-layer labels against an independent semantic point reference."""

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--capture',required=True);p.add_argument('--replay',required=True)
    a=p.parse_args()
    sys.path.insert(0, str(Path(a.project) if (Path(a.project) / 'lvimot').is_dir() else Path(a.project) / 'src'))
    if (Path(a.project) / 'lvimot').is_dir():
        from lvimot.output.evaluation.map_geometry import align_local_map_to_carla
    else:
        from carla_map_validation import align_local_map_to_carla
    cap=Path(a.capture);out=Path(a.replay)
    records=[json.loads(s) for s in (cap/'frames.jsonl').read_text().splitlines()]
    first=records[0]['ground_truth_ego_pose_evaluation_only'];last=records[-1]['ground_truth_ego_pose_evaluation_only']['location']
    center=np.array([last[k] for k in ['x','y','z']])
    reference=np.load(cap/'original_carla_map/CARLA_independent_360_reference_semantics.npz',allow_pickle=False)
    xyz=reference['xyz'];labels=reference['semantic'];roi=np.linalg.norm(xyz[:,:2]-center[:2],axis=1)<=45.
    xyz,labels=xyz[roi],labels[roi]
    tree=cKDTree(xyz)
    full=align_local_map_to_carla(np.load(out/'static_map.npy'),first)
    road=align_local_map_to_carla(np.load(out/'road_map.npy'),first)
    full=full[np.linalg.norm(full[:,:2]-center[:2],axis=1)<=45.]
    road=road[np.linalg.norm(road[:,:2]-center[:2],axis=1)<=45.]
    result={'protocol':'INDEPENDENT_SEMANTIC_POINT_REFERENCE_V1','evaluation_only':True,'scope':'final45m local area',
        'road_labels':[6,7],'full_scene_points':len(full),'predicted_road_points':len(road),
        'reference_road_points':int(np.isin(labels,[6,7]).sum()),'thresholds':{}}
    distance,nearest=tree.query(road)
    reference_road=xyz[np.isin(labels,[6,7])]
    observed_distance=cKDTree(full).query(reference_road)[0]
    road_distance=cKDTree(road).query(reference_road)[0]
    for threshold in [.25,.5,1.]:
        correct=(distance<=threshold)&np.isin(labels[nearest],[6,7])
        observed=observed_distance<=threshold
        result['thresholds'][str(threshold)]={'predicted_road_precision_pct':float(100*np.mean(correct)),
            'all_reference_road_recall_pct':float(100*np.mean(road_distance<=threshold)),
            'observed_reference_road_recall_pct':float(100*np.mean(road_distance[observed]<=threshold)) if observed.any() else None,
            'observed_reference_road_points':int(observed.sum()),
            'nearest_nonroad_or_unmatched_points':int((~correct).sum())}
    result['limits']=['Camera sees only the front sector; the full map retains other visible surfaces.',
        'Reference sampling density, semantic boundaries and estimated-pose error affect nearest-point labels.']
    (out/'semantic_road_evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
