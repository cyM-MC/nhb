"""Road inputs, explicit facility metadata and a stylized spatial hazard."""

import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


DEFAULT_FACILITIES = Path(__file__).parent / 'data' / 'facilities.csv'


def validate_facilities(facilities):
    required = {'name', 'facility_type', 'attraction_size', 'prestige', 'servers', 'service_minutes'}
    if not required <= set(facilities) or facilities.empty:
        raise ValueError(f"Facilities require {sorted(required)} and at least one row")
    f = facilities.copy().sort_values('name').reset_index(drop=True)
    if f['name'].isna().any() or f['name'].duplicated().any() or (f['name'].str.strip() == '').any():
        raise ValueError("Facility names must be nonempty and unique")
    if not f['facility_type'].isin(['core', 'clinic']).all():
        raise ValueError("facility_type must be core or clinic")
    for col in ('attraction_size', 'prestige', 'servers', 'service_minutes'):
        f[col] = pd.to_numeric(f[col], errors='raise')
        if not np.isfinite(f[col]).all() or (f[col] <= 0).any():
            raise ValueError(f"{col} must be finite and positive")
    if (f['prestige'] > 1).any() or (f['servers'] % 1 != 0).any():
        raise ValueError("Prestige must be <=1; servers must be integers")
    f['servers'] = f['servers'].astype(int)
    return f


def load_facilities(path=DEFAULT_FACILITIES, region='Japan'):
    f = pd.read_csv(path)
    if 'region' not in f:
        raise ValueError("Facility input requires a region column")
    f = validate_facilities(f.loc[f['region'] == region])
    for col, bound in [('lat', 90), ('lon', 180)]:
        if col not in f or not np.isfinite(f[col]).all() or (f[col].abs() > bound).any():
            raise ValueError(f"Invalid facility {col}")
    return f


def build_real_network(place_name, graph_path=None):
    """OSMnx free-flow seconds; preserve disconnected components for access metrics."""
    import osmnx as ox

    if graph_path is not None and Path(graph_path).exists():
        graph = ox.load_graphml(graph_path)
    else:
        ox.settings.use_cache = True
        graph = ox.graph_from_place(place_name, network_type='drive', retain_all=True)
        graph = ox.routing.add_edge_speeds(graph, fallback=50)
        graph = ox.routing.add_edge_travel_times(graph)
        if graph_path is not None:
            Path(graph_path).parent.mkdir(parents=True, exist_ok=True)
            ox.save_graphml(graph, graph_path)
    validate_graph(graph)
    return graph


def validate_graph(graph):
    if not graph.is_directed() or not graph.number_of_nodes() or not graph.number_of_edges():
        raise ValueError("Road input must be a nonempty directed graph with edges")
    for _, data in graph.nodes(data=True):
        if not (-180 <= float(data['x']) <= 180 and -90 <= float(data['y']) <= 90):
            raise ValueError("Road coordinates must be unprojected longitude/latitude")
    for _, _, data in graph.edges(data=True):
        time = float(data['travel_time'])
        if not math.isfinite(time) or time < 0:
            raise ValueError("Baseline road travel_time must be finite non-negative seconds")


def anchor_facilities(graph, facilities, max_snap_m=2000):
    """Fail on distant snaps; silently dropping facilities changes the experiment."""
    import osmnx as ox

    nodes, distances = ox.distance.nearest_nodes(
        graph, facilities['lon'].to_numpy(), facilities['lat'].to_numpy(), return_dist=True)
    f = facilities.copy()
    f['node_id'], f['snap_distance_m'] = nodes, distances
    too_far = f.loc[f['snap_distance_m'] > max_snap_m, 'name'].tolist()
    if too_far:
        raise ValueError(f"Facilities are >{max_snap_m} m from the graph: {too_far}")
    return f


def haversine_distance(lat1, lon1, lat2, lon2):
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2)**2 + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2)
    return 6371 * 2 * math.asin(math.sqrt(min(1, max(0, a))))


def apply_disaster_friction(graph, epicenter_lat, epicenter_lon, radius_km=4.5,
                            severity=3.0, closure_intensity=None):
    """Return a damaged copy; midpoint exposure is an approximation, not flood depth.

    t_damaged = t_baseline * exp(severity * intensity).
    Optional closures remove roads at/above a fixed intensity only when severity>0.
    Closures and severity are separate model controls.
    """
    if not math.isfinite(radius_km) or radius_km <= 0 or not math.isfinite(severity) or not 0 <= severity <= 20:
        raise ValueError("Positive radius and severity in [0,20] required")
    if closure_intensity is not None and not 0 < closure_intensity <= 1:
        raise ValueError("closure_intensity must be in (0,1]")
    damaged = graph.copy()
    edges = list(damaged.edges(keys=True, data=True)) if damaged.is_multigraph() else [
        (u, v, None, d) for u, v, d in damaged.edges(data=True)]
    for u, v, key, data in edges:
        lat = (graph.nodes[u]['y'] + graph.nodes[v]['y']) / 2
        lon = (graph.nodes[u]['x'] + graph.nodes[v]['x']) / 2
        intensity = max(0.0, 1 - haversine_distance(epicenter_lat, epicenter_lon, lat, lon) / radius_km)
        baseline = data.get('baseline_travel_time', data['travel_time'])
        data['baseline_travel_time'] = baseline
        data['hazard_intensity'] = intensity if severity > 0 else 0.0
        data['travel_time'] = baseline * math.exp(severity * intensity)
        if severity > 0 and closure_intensity is not None and intensity >= closure_intensity:
            if key is None:
                damaged.remove_edge(u, v)
            else:
                damaged.remove_edge(u, v, key)
    return damaged


def travel_time_matrix(graph, origins, facilities):
    """One reverse Dijkstra per distinct destination, respecting one-way roads."""
    reverse = graph.reverse(copy=False)
    matrix = np.full((len(origins), len(facilities)), np.inf)
    destinations = {}
    for j, node in enumerate(facilities['node_id']):
        if node not in graph:
            raise ValueError(f"Facility node absent from graph: {node}")
        destinations.setdefault(node, []).append(j)
    for node, columns in destinations.items():
        lengths = nx.single_source_dijkstra_path_length(reverse, node, weight='travel_time')
        values = np.array([lengths.get(origin, np.inf) for origin in origins])
        matrix[:, columns] = values[:, None]
    return matrix


def build_demo():
    """Small deterministic synthetic directed network: runs entirely offline."""
    graph = nx.MultiDiGraph(crs='EPSG:4326')
    side = 9
    for y in range(side):
        for x in range(side):
            graph.add_node(y * side + x, x=139 + (x - 4) * .008, y=35 + (y - 4) * .008)
    for y in range(side):
        for x in range(side):
            u = y * side + x
            for dx, dy in ((1, 0), (0, 1)):
                if x + dx < side and y + dy < side:
                    v = (y + dy) * side + x + dx
                    graph.add_edge(u, v, travel_time=60.0, length=700.0)
                    graph.add_edge(v, u, travel_time=65.0, length=700.0)
    rows = []
    for name, node, kind in [('Core_A', 40, 'core'), ('Core_B', 10, 'core'),
                              ('Clinic_A', 8, 'clinic'), ('Clinic_B', 72, 'clinic'),
                              ('Clinic_C', 80, 'clinic'), ('Clinic_D', 28, 'clinic')]:
        rows.append(dict(name=name, node_id=node, facility_type=kind,
                         lat=graph.nodes[node]['y'], lon=graph.nodes[node]['x'],
                         attraction_size=1000 if kind == 'core' else 150,
                         prestige=.9 if kind == 'core' else .25,
                         servers=6 if kind == 'core' else 2, service_minutes=10.0,
                         region='Demo', source_status='synthetic'))
    return graph, validate_facilities(pd.DataFrame(rows))
