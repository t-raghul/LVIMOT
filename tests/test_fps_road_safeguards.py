"""Speed changes preserve arithmetic; road inference never invents geometry."""
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

from lvimot.mapping_4d.static_map.rolling_voxel_map import Carla4DEnvironmentMapper
from lvimot.temporal_fusion.sliding_window_factor_graph import CarlaFactorGraph
from lvimot.output.live_dashboard import DashboardPresentation
from lvimot.mapping_4d.static_map.measured_road_surface import MeasuredRoadSurfaceLayer


def test_batched_voxel_blend_matches_legacy_float32_arithmetic():
    rng=np.random.RandomState(8)
    mapper=Carla4DEnvironmentMapper(max_static_voxels=500)
    keys=np.array([[i,0,0] for i in range(100)],dtype=np.int64)
    initial=rng.randn(70,3).astype(np.float32)
    mapper.static_voxel_map={tuple(k):v.copy() for k,v in zip(keys[:70],initial)}
    points=keys*.25+rng.uniform(.01,.24,(100,3))
    expected={k:v.copy() for k,v in mapper.static_voxel_map.items()}
    for i in np.argsort(np.sum(points**2,axis=1)):
        key=tuple(int(v) for v in keys[i]);point=points[i]
        expected[key]=((.7*expected[key]+.3*point).astype(np.float32) if key in expected else point.astype(np.float32).copy())
    mapper._update_grouped_voxels(keys,points,np.zeros(3))
    assert list(mapper.static_voxel_map)==list(expected)
    for key in expected:np.testing.assert_array_equal(mapper.static_voxel_map[key],expected[key])


def test_voxel_batch_respects_capacity_even_when_eviction_removes_current_keys():
    mapper=Carla4DEnvironmentMapper(max_static_voxels=10)
    mapper.static_voxel_map={(i,0,0):np.array([i,0,0],np.float32) for i in range(10)}
    keys=np.array([[i,0,0] for i in range(15)])
    mapper._update_grouped_voxels(keys,keys.astype(float),np.zeros(3))
    assert len(mapper.static_voxel_map)==10
    assert all(np.isfinite(p).all() for p in mapper.static_voxel_map.values())


def test_stored_voxel_rows_do_not_retain_shared_work_buffers():
    mapper=Carla4DEnvironmentMapper()
    keys=np.array([[0,0,0],[1,0,0]])
    mapper._update_grouped_voxels(keys,np.array([[.1,.1,.1],[.4,.1,.1]]),np.zeros(3))
    assert all(point.base is None for point in mapper.static_voxel_map.values())


def test_factor_cache_exact_states_and_lifetime():
    graph=CarlaFactorGraph();graph.add_keyframe(0,0.)
    graph.add_pose_prior(0,np.zeros(6));graph.add_velocity_prior(0,np.zeros(3))
    x,mapping=graph._pack_states({0:np.arange(6,dtype=float)*.01},{0:np.ones(3)}, {})
    raw=graph.compute_all_residuals(x,mapping)
    graph._optimization_residual_cache={}
    cached=graph.compute_all_residuals(x,mapping)
    np.testing.assert_array_equal(raw,cached)
    size=len(graph._optimization_residual_cache)
    graph.compute_all_residuals(x.copy(),mapping)
    assert len(graph._optimization_residual_cache)==size
    perturbed=x.copy();perturbed[0]+=1e-12
    graph.compute_all_residuals(perturbed,mapping)
    assert len(graph._optimization_residual_cache)>size
    graph._optimization_residual_cache=None
    graph.optimize({0:np.zeros(6)},{0:np.zeros(3)},max_nfev=2)
    assert graph._optimization_residual_cache is None


class BlockingViewer:
    def __init__(self):
        self.started=threading.Event();self.release=threading.Event();self.received=[]
    def render(self,image,result,points,road_points=None):
        if result['frame']==0:
            self.started.set();assert self.release.wait(3.)
        self.received.append((result['frame'],image.copy(),points.copy(),result['tracking']))


def test_presentation_is_bounded_freezes_inputs_and_flushes_latest_frame():
    viewer=BlockingViewer();worker=DashboardPresentation(viewer,use_process=False)
    image=np.ones((8,8,3),np.uint8);points=np.ones((3,3))
    record={'frame':0,'tracking':{'active_tracks':[{'track_id':1}]}}
    worker.submit(image,record,points)
    assert viewer.started.wait(3.)
    image[:]=0;points[:]=0;record['tracking']['active_tracks'][0]['track_id']=999
    worker.submit(image,{'frame':1},points);worker.submit(image,{'frame':2,'tracking':{}},points)
    viewer.release.set();worker.close()
    assert [r[0] for r in viewer.received]==[0,2]
    assert np.all(viewer.received[0][1]==1) and np.all(viewer.received[0][2]==1)
    assert viewer.received[0][3]['active_tracks'][0]['track_id']==1
    assert worker.summary()['estimator_frames_dropped']==0
    assert worker.summary()['replaced_pending_previews']==1
    worker.close()


def test_surface_support_rejects_roof_points_and_preserves_measured_heights():
    x,y=np.meshgrid(np.linspace(4,25,15),np.linspace(-4,4,10))
    ground=np.column_stack([x.ravel(),y.ravel(),-.02*x.ravel()-2.2])
    elevated=ground[:30]+np.array([0,0,2.])
    raw=np.vstack([ground,elevated]);before=raw.copy()
    mask,plane=MeasuredRoadSurfaceLayer._measured_surface(raw)
    assert plane is not None
    assert mask[:len(ground)].all() and not mask[len(ground):].any()
    np.testing.assert_array_equal(raw,before)


@pytest.mark.parametrize('count',[0,2,39])
def test_insufficient_road_support_abstains(count):
    selected,plane=MeasuredRoadSurfaceLayer._measured_surface(np.ones((count,3)))
    assert not selected.any() and plane is None


def test_steep_surface_is_not_flattened_into_a_road():
    x,y=np.meshgrid(np.linspace(3,10,15),np.linspace(-4,4,15))
    cloud=np.column_stack([x.ravel(),y.ravel(),.6*x.ravel()-2.2])
    selected,plane=MeasuredRoadSurfaceLayer._measured_surface(cloud)
    assert not selected.any() and plane is None

class PacketViewer:
    def __init__(self):
        self.records=[]
    def render(self,image,result,points,road_points=None):
        self.records.append(result)


def test_presentation_retains_fused_candidates_and_only_compact_estimate():
    viewer=PacketViewer();worker=DashboardPresentation(viewer,use_process=False)
    record={'frame':12,'state_estimation':{'pose':np.zeros(6),'velocity':np.ones(3),'large_diagnostic':np.ones(1000)},
            'multimodal_objects':[{'projected_bbox':[1,2,20,30],'class':'vehicle','label':'car','points':np.ones((100,3))}]}
    worker.submit(np.zeros((32,32,3),np.uint8),record,np.empty((0,3)))
    worker.close()
    assert set(viewer.records[0]['state_estimation'])=={'pose','velocity'}
    assert viewer.records[0]['multimodal_objects']==[{'projected_bbox':[1,2,20,30],'class':'vehicle','label':'car'}]


def test_spawned_dashboard_writes_preview_and_flushes_last_frame(tmp_path,monkeypatch):
    import json
    from lvimot.output.live_dashboard import CarlaLiveVisualizer
    monkeypatch.setenv('LVIMOT_LIVE_MAP_SNAPSHOT',str(tmp_path/'map.json'))
    monkeypatch.setenv('LVIMOT_LIVE_CAMERA_SNAPSHOT',str(tmp_path/'camera.png'))
    worker=DashboardPresentation(CarlaLiveVisualizer(bev_size=240))
    cloud=np.column_stack([np.linspace(3,12,80),np.zeros(80),np.zeros(80)])
    worker.submit(np.zeros((128,320,3),np.uint8),{'frame':7,'tracking':{'active_tracks':[]}},cloud,cloud)
    worker.close()
    assert worker.summary()['error'] is None
    assert worker.summary()['last_preview_frame']==7
    assert (tmp_path/'camera.png').stat().st_size>100
    snapshot=json.loads((tmp_path/'map.json').read_text())
    assert snapshot['frame']==7 and len(snapshot['measured_road_points_local'])>0


def test_road_model_manifest_and_research_license_are_shipped():
    import hashlib
    folder=Path(__file__).resolve().parents[1]/'models/segformer_road'
    model=folder/'road_segformer_b0.ts'
    assert hashlib.sha256(model.read_bytes()).hexdigest()=='52a4948db6ce858ca76ae77064d45bb17c6a33b193253bc52837f446851ce734'
    assert 'research or evaluation purposes only' in (folder/'LICENSE').read_text(encoding='utf-8')
