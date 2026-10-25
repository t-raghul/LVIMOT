"""Build an evidence-based acceptance table without running CARLA or estimators."""
from __future__ import annotations

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize(trace_directory, protocol_path):
    validation = load(trace_directory / 'validation_summary.json')
    geometry_path = trace_directory / 'independent_geometry' / 'INDEPENDENT_SCENE_GEOMETRY_REPORT.json'
    scene_path = trace_directory / 'scene_tracking_evaluation.json'
    geometry = load(geometry_path)
    scene = load(scene_path)
    protocol = load(protocol_path)
    localization = validation['evaluation']['localization']
    tracking = scene['tracking']
    initial = geometry['rolling_scope']['metrics']['initial_pose_placement']
    comparison = initial['estimate_to_independent_reference']
    completeness = initial['independent_reference_to_estimate']
    observation = geometry['trajectory_observation']
    rows = []

    def check(name, value, target, minimum=False, unit=''):
        valid = isinstance(value, (int, float))
        passed = valid and (value >= target if minimum else value <= target)
        rows.append({'measure': name, 'value': value, 'target': target,
                     'relation': '>=' if minimum else '<=', 'unit': unit,
                     'passed': bool(passed)})

    loc = protocol['localization']
    geo = protocol['independent_scene_geometry_initial_pose']
    mot = protocol['tracking_full_scene_measured_support_protocol']
    evidence = protocol['minimum_evidence']
    check('Localization ATE RMSE', localization.get('ate_rmse'), loc['ate_rmse_max_m'], unit='m')
    check('Localization RPE RMSE', localization.get('rpe_rmse'), loc['rpe_rmse_max_m'], unit='m')
    check('Map points within 1 m of independent CARLA', comparison['within_1m_pct'], geo['estimate_within_1m_min_pct'], True, '%')
    check('Independent observed CARLA surfaces within 1 m of map', completeness['within_1m_pct'], geo['reference_within_1m_min_pct'], True, '%')
    check('Map surface median distance', comparison['median_m'], geo['estimate_median_max_m'], unit='m')
    check('Map surface 95th percentile distance', comparison['p95_m'], geo['estimate_p95_max_m'], unit='m')
    check('Whole-scene measured-support MOTA', tracking['mota'], mot['mota_min'], True)
    check('Matched object position RMSE', tracking.get('position_rmse'), mot['position_rmse_max_m'], unit='m')
    check('Matched object velocity RMSE', tracking.get('velocity_rmse'), mot['velocity_rmse_max_m_s'], unit='m/s')
    check('Recorded frames', validation['frames'], evidence['frames'], True)
    check('Observed route length', observation['gt_traversed_distance_m_evaluation_only'], evidence['traversed_distance_m'], True, 'm')
    check('Eligible tracking observations', tracking['total_gt_objects'], evidence['eligible_tracking_observations'], True)
    check('Unique eligible vehicles', tracking['unique_gt_actors'], evidence['unique_gt_vehicles'], True)
    files = [trace_directory / 'validation_summary.json', geometry_path, scene_path, protocol_path]
    export_ok = not validation['map_export'].get('warnings')
    safe = validation.get('gt_used_by_estimator') is False and geometry.get('ground_truth_used_by_estimator') is False
    verdict = all(row['passed'] for row in rows) and export_ok and safe
    result = {
        'verdict': 'OBSERVED_REGION_ENGINEERING_CHECKS_PASS' if verdict else 'FULL_PROJECT_ACCEPTANCE_NOT_REACHED',
        'scope': protocol['scope'], 'checks': rows,
        'export_warnings': validation['map_export'].get('warnings'),
        'evaluation_only_gt_boundary_reported': safe,
        'legacy_npc_only_tracking_preserved': validation['evaluation']['tracking'],
        'whole_scene_tracking_protocol': scene['protocol'],
        'sources_sha256': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        'limits': ['Finite independent rendered surfaces in the observed region, not a complete CARLA mesh or entire town.',
                   'A pass is limited to these recorded observations and predeclared engineering thresholds.',
                   'No long-route loop-closure/global-drift validation is implied.'],
    }
    (trace_directory / 'ACCEPTANCE_REPORT.json').write_text(json.dumps(result, indent=2) + '\n')
    md = ['# LVIMOT recorded acceptance checks', '', '**' + result['verdict'] + '**', '',
          '| Measure | Observed | Required | Result |', '|---|---:|---:|---|']
    for row in rows:
        value = ('%.4f' % row['value']) if isinstance(row['value'], float) else str(row['value'])
        md.append('| %s | %s %s | %s %s %s | %s |' % (
            row['measure'], value, row['unit'], row['relation'], row['target'], row['unit'],
            'PASS' if row['passed'] else 'FAIL'))
    md.extend(['', 'Primary geometry uses first-pose placement only. Post-hoc fitting is a separate diagnostic.', '',
               'The legacy NPC-only score is preserved in ACCEPTANCE_REPORT.json. The additional whole-scene protocol includes static vehicle labels and measured 3D support.', '',
               '## Limits', ''] + ['- ' + line for line in result['limits']])
    (trace_directory / 'ACCEPTANCE_REPORT.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(result['verdict'])
    for row in rows:
        print(('PASS' if row['passed'] else 'FAIL'), row['measure'], row['value'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-directory', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path('config/validation_protocol.json'))
    args = parser.parse_args()
    summarize(args.trace_directory, args.protocol)
