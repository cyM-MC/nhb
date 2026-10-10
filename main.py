"""Reproducible paper simulation: run, analyse and optionally plot from one CLI."""

import argparse
from dataclasses import replace
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
from pathlib import Path
import platform
import shutil
from zoneinfo import ZoneInfo

import networkx as nx
import numpy as np
import pandas as pd

from analysis import RUN_KEYS, paired_effects, summarize_run
from config import ExperimentConfig, REGIONS
from environment import (DEFAULT_FACILITIES, anchor_facilities, apply_disaster_friction,
                         build_demo, build_real_network, load_facilities, travel_time_matrix)
from simulation import make_population, run_scenario


ROOT = Path(__file__).resolve().parent


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def origin_distribution(graph, path=None):
    """Default is node-uniform; optional external population weights avoid that proxy."""
    if path is None:
        return np.array(sorted(graph.nodes, key=str)), None
    weights = pd.read_csv(path, dtype={'node_id': str})
    if not {'node_id', 'weight'} <= set(weights) or weights.empty or weights['node_id'].duplicated().any():
        raise ValueError("Origins CSV requires unique node_id and weight rows")
    node_map = {str(node): node for node in graph}
    if not weights['node_id'].isin(node_map).all():
        raise ValueError("Origins include nodes absent from the road graph")
    p = weights['weight'].to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("Origin weights must be finite non-negative with positive total")
    return np.array([node_map[node] for node in weights['node_id']]), p / p.sum()


def run_experiment(config, output_dir, facility_path=DEFAULT_FACILITIES,
                   graph_path=None, origins_path=None):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    if config.region == 'Demo':
        graph, facilities = build_demo()
        nx.write_graphml(graph, output / 'network.graphml')
    else:
        import osmnx as ox
        if graph_path is not None and not Path(graph_path).is_file():
            raise FileNotFoundError(graph_path)
        # Reuse a baseline snapshot locally; record the exact graph used in each run.
        cache = ROOT / 'cache' / f"{config.region.replace(' ', '_')}_drive_all_v2.graphml"
        graph = build_real_network(REGIONS[config.region]['place'], graph_path or cache)
        facilities = anchor_facilities(graph, load_facilities(facility_path, config.region))
        ox.save_graphml(graph, output / 'network.graphml')
    facilities.to_csv(output / 'facilities_input.csv', index=False)
    facilities['service_minutes'] = facilities['service_minutes'] / config.service_rate_multiplier
    facilities.to_csv(output / 'facilities.csv', index=False)
    (output / 'config.json').write_text(json.dumps(config.to_dict(), indent=2) + '\n')
    versions = {}
    for package in ('numpy', 'pandas', 'networkx', 'matplotlib', 'osmnx'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    source_dir = output / 'source'
    source_dir.mkdir()
    for source in sorted(ROOT.glob('*.py')):
        shutil.copy2(source, source_dir / source.name)
    if origins_path is not None:
        shutil.copy2(origins_path, output / 'origins.csv')
    manifest = {
        'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'python': platform.python_version(), 'packages': versions,
        'graph_sha256': file_hash(output / 'network.graphml'),
        'facilities_sha256': file_hash(output / 'facilities.csv'),
        'facilities_input_sha256': file_hash(output / 'facilities_input.csv'),
        'origins_sha256': file_hash(origins_path) if origins_path else None,
        'origin_sampling': 'external_weights' if origins_path else 'uniform_road_nodes',
        'source_sha256': {p.name: file_hash(p) for p in sorted(source_dir.glob('*.py'))},
        'network_bounds': [min(d['x'] for _, d in graph.nodes(data=True)),
                           min(d['y'] for _, d in graph.nodes(data=True)),
                           max(d['x'] for _, d in graph.nodes(data=True)),
                           max(d['y'] for _, d in graph.nodes(data=True))],
        'interpretation': 'Stylized simulation; Monte Carlo uncertainty; no empirical calibration',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    nodes, weights = origin_distribution(graph, origins_path)
    populations, noise = {}, {}
    for seed in config.seeds:
        population, choice_rng = make_population(nodes, config, seed, weights)
        populations[seed] = population
        noise[seed] = choice_rng.gumbel(size=(config.n_agents, len(facilities)))
    origins = np.array(sorted(set(np.concatenate([p['start_node'].to_numpy() for p in populations.values()])), key=str))
    origin_index = {node: i for i, node in enumerate(origins)}
    population_indices = {seed: [origin_index[node] for node in p['start_node']] for seed, p in populations.items()}
    summaries, load_frames = [], []
    first = True
    for severity in config.severity_levels:
        damaged = apply_disaster_friction(graph, *config.hazard_center, config.radius_km,
                                          severity, config.closure_intensity)
        print(f"Severity={severity}: routing to {len(facilities)} facilities", flush=True)
        matrix = travel_time_matrix(damaged, origins, facilities)
        for seed, population in populations.items():
            times = matrix[population_indices[seed]]
            for stress in config.stress_levels:
                for scenario in config.scenarios:
                    decisions = run_scenario(population, times, facilities, config, stress, scenario, noise[seed])
                    for col, value in [('region', config.region), ('seed', seed), ('severity', severity), ('stress', stress)]:
                        decisions[col] = value
                    # Stream individual outcomes; do not hold all grid runs in RAM.
                    decisions.to_csv(output / 'decisions.csv', index=False, mode='w' if first else 'a', header=first)
                    first = False
                    rows, loads = summarize_run(decisions, facilities, config)
                    summaries.extend(rows)
                    for key in RUN_KEYS:
                        loads[key] = decisions.iloc[0][key]
                    load_frames.append(loads)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / 'run_summary.csv', index=False)
    pd.concat(load_frames, ignore_index=True).to_csv(output / 'facility_loads.csv', index=False)
    paired_effects(summary, config).to_csv(output / 'paired_effects.csv', index=False)
    print(f"Saved experiment: {output.resolve()}", flush=True)
    return output


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--config', type=Path, help='JSON with any ExperimentConfig fields')
    cli.add_argument('--region', choices=REGIONS)
    cli.add_argument('--agents', type=int)
    cli.add_argument('--seeds', nargs='+', type=int)
    cli.add_argument('--stress', nargs='+', type=float)
    cli.add_argument('--severity', nargs='+', type=float)
    cli.add_argument('--gamma', type=float)
    cli.add_argument('--temperature', type=float)
    cli.add_argument('--friction', type=float)
    cli.add_argument('--exposure', type=float)
    cli.add_argument('--radius', type=float)
    cli.add_argument('--service-rate', type=float, help='Multiply all service rates, independently of demand')
    cli.add_argument('--epicenter', nargs=2, type=float, metavar=('LAT', 'LON'))
    cli.add_argument('--closure', type=float, help='Optional road-closure intensity in (0,1]')
    cli.add_argument('--facilities', type=Path, default=DEFAULT_FACILITIES)
    cli.add_argument('--graph', type=Path, help='Existing unprojected OSMnx GraphML snapshot')
    cli.add_argument('--origins', type=Path, help='CSV: node_id, weight')
    cli.add_argument('--output', type=Path)
    cli.add_argument('--plots', action='store_true')
    cli.add_argument('--maps', action='store_true', help='Also reconstruct representative time-optimal routes')
    cli.add_argument('--plot-only', type=Path, help='Replot a completed output directory')
    return cli


def main():
    args = parser().parse_args()
    if args.plot_only:
        from visual import plot_paper_figures
        plot_paper_figures(args.plot_only, maps=args.maps)
        return
    config = ExperimentConfig(**json.loads(args.config.read_text())) if args.config else ExperimentConfig()
    aliases = {'agents': 'n_agents', 'stress': 'stress_levels', 'severity': 'severity_levels',
               'radius': 'radius_km', 'closure': 'closure_intensity', 'service_rate': 'service_rate_multiplier'}
    overrides = {aliases.get(k, k): getattr(args, k) for k in
                 ('region', 'agents', 'seeds', 'stress', 'severity', 'gamma', 'temperature',
                  'friction', 'exposure', 'radius', 'epicenter', 'closure', 'service_rate') if getattr(args, k) is not None}
    config = replace(config, **overrides)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    output = args.output or ROOT / 'results' / f"{config.region.replace(' ', '_')}_{stamp}"
    run_experiment(config, output, args.facilities, args.graph, args.origins)
    if args.plots or args.maps:
        from visual import plot_paper_figures
        plot_paper_figures(output, maps=args.maps)


if __name__ == '__main__':
    main()
