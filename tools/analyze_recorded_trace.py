"""Attribute recorded MOT misses to sensing or publication, evaluation only."""
from __future__ import annotations

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[1]))


import argparse
import json
from collections import Counter
from pathlib import Path


def iou(a, b):
    if a is None or b is None or len(a) != 4 or len(b) != 4:
        return 0.0
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def best(box, candidates):
    ranked = [(iou(box, item.get('bbox')), index, item) for index, item in enumerate(candidates)]
    if not ranked:
        return 0.0, None
    overlap, _, item = max(ranked, key=lambda row: (row[0], -row[1]))
    return overlap, item


def analyze(directory: Path):
    rows = []
    no_gt_outputs = []
    for line in (directory / 'frame_trace.jsonl').read_text(encoding='utf-8').splitlines():
        frame = json.loads(line)
        sensor = frame['sensor_only']
        evaluated = frame['evaluation_only']
        if not evaluated['eligible_gt'] and evaluated['scored_tracks']:
            no_gt_outputs.append({'frame': frame['frame'], 'tracks': evaluated['scored_tracks']})
        for gt in evaluated['eligible_gt']:
            bbox = gt.get('bbox')
            camera_iou, camera = best(bbox, sensor['camera_detections'])
            proposal_iou, proposal = best(bbox, sensor['lidar_proposals'])
            track_iou, track = best(bbox, sensor['track_hypotheses'])
            output_iou, published = best(bbox, evaluated['scored_tracks'])
            if output_iou >= 0.25:
                reason = 'MATCHED_RGB_BOX'
            elif camera_iou < 0.25:
                reason = 'NO_MATCHING_RGB_BOX'
            elif proposal_iou < 0.25:
                reason = 'RGB_WITHOUT_CURRENT_3D_PROPOSAL'
            elif track_iou < 0.25:
                reason = 'PROPOSAL_WITHOUT_TRACK'
            else:
                reason = 'INTERNAL_TRACK_NOT_PUBLISHED'
            row = {
                'frame': frame['frame'], 'actor_evaluation_only': gt.get('actor_id'),
                'gt_lidar_support_evaluation_only': gt.get('lidar_support'),
                'rgb_iou': round(camera_iou, 4),
                'camera_confidence': camera.get('camera_confidence') if camera else None,
                'proposal_iou': round(proposal_iou, 4),
                'proposal_support': proposal.get('camera_support') if proposal else None,
                'proposal_score': proposal.get('score') if proposal else None,
                'internal_track_iou': round(track_iou, 4),
                'internal_track': track if track_iou >= 0.25 else None,
                'published_iou': round(output_iou, 4),
                'published_track_id': published.get('track_id') if output_iou >= 0.25 else None,
                'stage': reason,
            }
            rows.append(row)
    result = {
        'mode': 'READ_ONLY_EVALUATION_OF_SENSOR_TRACE',
        'source_trace': str(directory / 'frame_trace.jsonl'),
        'gt_used_by_estimator': False,
        'stage_counts': dict(Counter(row['stage'] for row in rows)),
        'eligible_observations': rows,
        'outputs_on_frames_without_eligible_gt': no_gt_outputs,
    }
    (directory / 'mot_stage_diagnosis.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(result['stage_counts'])
    for row in rows:
        if row['stage'] != 'MATCHED_RGB_BOX':
            track = row['internal_track'] or {}
            print('frame', row['frame'], 'actor', row['actor_evaluation_only'], row['stage'],
                  'support', row['proposal_support'], 'score', row['proposal_score'],
                  'track', {k: track.get(k) for k in ('track_id', 'hits', 'consecutive_hits', 'score', 'sparse_consistent_hits', 'camera_support')})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    analyze(parser.parse_args().directory)
