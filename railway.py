"""Local railway observations: user-disjoint routing checks and empirical demand.

The trajectories are mobility observations, not medical choices. No individual
identifiers are copied to experiment outputs.
"""

from collections import defaultdict
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from shapely import from_wkb

from environment import haversine_distance, validate_graph


def summarize_trips(records):
    records = records.sort_values(['uid', 'tid', 'timestamp'], kind='stable')
    groups = records.groupby(['uid', 'tid'], sort=False)
    trips = groups.agg(start_node=('nid', 'first'), end_node=('nid', 'last'),
                       departure=('timestamp', 'first'), arrival=('timestamp', 'last'),
                       records=('nid', 'size')).reset_index()
    trips['duration_sec'] = (trips.arrival - trips.departure).dt.total_seconds()
    previous = groups.nid.shift()
    # Repeated observations at one station are not independent traversals.
    visits = records.loc[records.nid.ne(previous)].copy()
    groups = visits.groupby(['uid', 'tid'], sort=False)
    visits['next_node'] = groups.nid.shift(-1)
    visits['duration_sec'] = (groups.timestamp.shift(-1) - visits.timestamp).dt.total_seconds()
    return trips, visits


def prepare_railway(data_dir, cache_dir, sample_modulus=100):
    """Select about 1% of users by stable hash, then read all their records.

    Speed estimates are station-first-observation intervals (include dwell),
    not timetable running times. The held-out users never estimate edge times.
    """
    data_dir, cache = Path(data_dir), Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    paths = [data_dir / 'japan_railway_network.csv', data_dir / 'railway.csv.gz']
    signature = {'version': 1, 'sample_modulus': sample_modulus,
                 'inputs': {p.name: {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in paths}}
    if (cache / 'signature.json').exists() and json.loads((cache / 'signature.json').read_text()) == signature:
        return nx.read_graphml(cache / 'network.graphml', node_type=int), pd.read_csv(
            cache / 'demand.csv', parse_dates=['departure']), json.loads((cache / 'validation.json').read_text())
    links = pd.read_csv(paths[0])
    if links.linkid.duplicated().any():
        raise ValueError('Railway linkid must be unique')
    graph = nx.MultiDiGraph(crs='EPSG:4326')
    station_groups = defaultdict(set)
    for row in links.itertuples():
        coords = list(from_wkb(row.geom).coords)
        for prefix, xy in [('source', coords[0]), ('target', coords[-1])]:
            node = int(getattr(row, prefix + '_station_code'))
            group = int(getattr(row, prefix + '_station_group'))
            graph.add_node(node, x=float(xy[0]), y=float(xy[1]),
                           station_name=getattr(row, prefix + '_station_name'))
            station_groups[group].add(node)
        u, v = int(row.source_station_code), int(row.target_station_code)
        for a, b in [(u, v), (v, u)]:
            graph.add_edge(a, b, key=int(row.linkid), linkid=int(row.linkid),
                           length=float(row.length), travel_time=max(30., float(row.length) / (30 / 3.6) + 60),
                           time_source='assumed_30kmh_plus_60s', edge_type='rail')
    selected, total, node_matches, link_matches = [], 0, 0, 0
    for i, chunk in enumerate(pd.read_csv(paths[1], chunksize=250000)):
        total += len(chunk)
        node_matches += int(chunk.nid.isin(graph.nodes).sum())
        link_matches += int(chunk.lid.isin(links.linkid).sum())
        hashes = pd.util.hash_array(chunk.uid.to_numpy())
        part = chunk.loc[hashes % sample_modulus == 0].copy()
        part['heldout'] = (hashes[hashes % sample_modulus == 0] // sample_modulus) % 5 == 0
        selected.append(part)
        if i % 20 == 0:
            print(f'Railway audit: {total:,} records scanned', flush=True)
    records = pd.concat(selected, ignore_index=True)
    records['timestamp'] = pd.to_datetime(records.timestamp, errors='raise')
    if records[['uid', 'tid', 'nid', 'lid', 'timestamp']].isna().any().any():
        raise ValueError('Missing railway identifiers or timestamps')
    trips, visits = summarize_trips(records)
    user_split = records.groupby('uid').heldout.first()
    trips['heldout'] = trips.uid.map(user_split)
    train = visits.loc[~visits.heldout & visits.duration_sec.between(30, 1800) & visits.next_node.notna()]
    observed = train.groupby(['nid', 'next_node']).duration_sec.agg(['median', 'size'])
    calibrated = 0
    for u, v, key, edge in graph.edges(keys=True, data=True):
        if (u, v) in observed.index and observed.loc[(u, v), 'size'] >= 5:
            edge['travel_time'] = float(observed.loc[(u, v), 'median'])
            edge['observations'] = int(observed.loc[(u, v), 'size'])
            edge['time_source'] = 'training_user_median_station_interval'
            calibrated += 1
    for nodes in station_groups.values():
        for u in nodes:
            for v in nodes:
                if u == v:
                    continue
                length = haversine_distance(graph.nodes[u]['y'], graph.nodes[u]['x'],
                                            graph.nodes[v]['y'], graph.nodes[v]['x']) * 1000
                graph.add_edge(u, v, length=length, travel_time=120 + length / 1.2,
                               edge_type='transfer', time_source='assumed_walk_plus_120s')
    validate_graph(graph)
    eligible = trips.loc[trips.start_node.isin(graph) & trips.end_node.isin(graph)
                         & trips.duration_sec.between(60, 14400) & trips.start_node.ne(trips.end_node)].copy()
    heldout = eligible.loc[eligible.heldout].head(5000)
    checks = []
    for origin, group in heldout.groupby('start_node'):
        times = nx.single_source_dijkstra_path_length(graph, origin, weight='travel_time')
        for row in group.itertuples():
            predicted = times.get(row.end_node, np.inf)
            checks.append((row.duration_sec, predicted))
    check = np.asarray(checks, dtype=float).reshape(-1, 2)
    finite = np.isfinite(check[:, 1])
    errors = check[finite, 1] - check[finite, 0]
    nodes = graph.nodes
    tokyo = eligible.start_node.map(lambda n: 35.5 <= nodes[n]['y'] <= 35.85 and 139.5 <= nodes[n]['x'] <= 139.95)
    # One earliest eligible Tokyo trip per training user; no user counted twice.
    demand = eligible.loc[tokyo & ~eligible.heldout].sort_values('departure').drop_duplicates('uid').copy()
    if demand.empty or len(errors) == 0:
        raise ValueError('No usable Tokyo demand or held-out trips')
    demand.to_csv(cache / 'demand.csv', index=False)
    report = dict(records=total, node_match_share=node_matches / total, link_match_share=link_matches / total,
                  sampled_users=int(records.uid.nunique()), sampled_trips=len(trips),
                  training_users=int((~user_split).sum()), heldout_users=int(user_split.sum()),
                  nodes=graph.number_of_nodes(), edges=graph.number_of_edges(),
                  calibrated_directed_rail_edges=calibrated, directed_rail_edges=2 * len(links),
                  tokyo_demand_users=len(demand), heldout_trips=len(check),
                  heldout_reachable_share=float(finite.mean()),
                  heldout_mae_min=float(np.abs(errors).mean() / 60),
                  heldout_bias_min=float(errors.mean() / 60),
                  heldout_median_observed_min=float(np.median(check[finite, 0]) / 60),
                  heldout_median_predicted_min=float(np.median(check[finite, 1]) / 60),
                  date_min=str(records.timestamp.min()), date_max=str(records.timestamp.max()),
                  limitations='Observed mobility only; bidirectional rail assumed; unobserved edge times, transfers, walking, clinical parameters and disasters are hypothetical.')
    nx.write_graphml(graph, cache / 'network.graphml')
    (cache / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    (cache / 'signature.json').write_text(json.dumps(signature, indent=2) + '\n')
    return graph, demand, report
