"""Read original CARLA static vehicle boxes and sensor defaults for evaluation.

This program only queries the server. It does not tick, load a world, change
settings, spawn actors, enable objects, or feed data to LVIMOT estimation.
"""

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
from pathlib import Path


def export(output, host='127.0.0.1', port=2000):
    import carla
    client = carla.Client(host, int(port))
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    if not carla_map.name.endswith('Town10HD_Opt'):
        raise RuntimeError('Expected original Town10HD_Opt; current map is ' + carla_map.name)
    vehicles = []
    for obj in world.get_environment_objects(carla.CityObjectLabel.Vehicles):
        bbox = obj.bounding_box
        location, rotation = bbox.location, bbox.rotation
        corners = bbox.get_world_vertices(carla.Transform())
        vehicles.append({
            'actor_id': -(int(obj.id) + 1), 'environment_object_id': int(obj.id),
            'name': obj.name, 'type': 'vehicle.static_environment',
            'location': {k: float(getattr(location, k)) for k in ('x', 'y', 'z')},
            'rotation': {k: float(getattr(rotation, k)) for k in ('pitch', 'yaw', 'roll')},
            'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'dimensions': [float(2 * bbox.extent.x), float(2 * bbox.extent.y), float(2 * bbox.extent.z)],
            'bounding_box': {'world_corners': [[float(v.x), float(v.y), float(v.z)] for v in corners]},
            'reference_source': 'CARLA original static environment object; evaluation only',
        })
    bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    attrs = {key: bp.get_attribute(key).as_int() for key in ('points_per_second', 'channels')}
    attrs.update({key: bp.get_attribute(key).as_float() for key in ('range', 'rotation_frequency', 'upper_fov', 'lower_fov', 'dropoff_general_rate')})
    result = {
        'format': 'CARLA_STATIC_VEHICLE_EVALUATION_REFERENCE_V1',
        'map': carla_map.name, 'opendrive_sha256': hashlib.sha256(carla_map.to_opendrive().encode()).hexdigest(),
        'server_version': client.get_server_version(), 'vehicles': vehicles,
        'default_lidar_blueprint_attributes': attrs,
        'ground_truth_used_by_estimator': False, 'server_world_modified': False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('Saved', len(vehicles), 'independent static vehicle labels:', output)
    print('Default LiDAR blueprint:', attrs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    args = parser.parse_args()
    export(args.output, args.host, args.port)
