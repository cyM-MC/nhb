"""Paper figures from saved runs; no fixed regional lists or hand-picked success cases."""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np
import pandas as pd

from analysis import bootstrap_mean
from config import ExperimentConfig
from decision import rationality_weight
from environment import apply_disaster_friction


COLORS = {'Baseline': '#c44e52', 'Full': '#4c72b0', 'Friction': '#dd8452',
          'Exposure': '#55a868', 'Fastest': '#8172b3'}


def save_figure(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(output / f'{name}.svg', bbox_inches='tight')
    plt.close(fig)


def get_edge_frequencies(decisions, graph, facility_map):
    """Reconstruct the same travel-time routes, preserving parallel-edge keys.

    Compute reverse routes once per chosen destination, aggregate repeated origins.
    The selected geometry is the fastest parallel edge, not an arbitrary edge.
    """
    flows = {}
    reverse = graph.reverse(copy=False)
    for name, rows in decisions.loc[decisions['reachable']].groupby('chosen_facility'):
        target = facility_map[name]
        if target not in graph:
            raise ValueError(f"Unknown destination: {target}")
        _, paths = nx.single_source_dijkstra(reverse, target, weight='travel_time')
        for start, count in rows['start_node'].value_counts().items():
            if start not in paths:
                raise ValueError(f"Saved reachable route cannot be reconstructed: {start}->{target}")
            path = list(reversed(paths[start]))
            for u, v in zip(path, path[1:]):
                key = (min(graph[u][v], key=lambda k: (graph[u][v][k]['travel_time'], str(k)))
                       if graph.is_multigraph() else None)
                edge = (u, v, key)
                flows[edge] = flows.get(edge, 0) + int(count)
    return flows


def edge_line(graph, u, v, key):
    data = graph[u][v][key] if graph.is_multigraph() else graph[u][v]
    if 'geometry' in data:
        return list(data['geometry'].coords)
    return [(graph.nodes[u]['x'], graph.nodes[u]['y']), (graph.nodes[v]['x'], graph.nodes[v]['y'])]


def plot_maps(output, config, facilities, severity, stress):
    if config.region == 'Demo':
        graph = nx.read_graphml(output / 'network.graphml', node_type=int, force_multigraph=True)
    else:
        import osmnx as ox
        graph = ox.load_graphml(output / 'network.graphml')
    graph = apply_disaster_friction(graph, *config.hazard_center, config.radius_km,
                                    severity, config.closure_intensity)
    selected = []
    for chunk in pd.read_csv(output / 'decisions.csv', chunksize=50000):
        selected.append(chunk.loc[(chunk['seed'] == config.seeds[0])
                                  & (chunk['severity'] == severity) & (chunk['stress'] == stress)
                                  & chunk['scenario'].isin(['Baseline', 'Full'])])
    rows = pd.concat(selected, ignore_index=True)
    facility_map = dict(zip(facilities['name'], facilities['node_id']))
    flows = {scenario: get_edge_frequencies(rows.loc[rows['scenario'] == scenario], graph, facility_map)
             for scenario in ('Baseline', 'Full')}
    max_flow = max((max(f.values(), default=0) for f in flows.values()), default=0) or 1
    edges = (graph.edges(keys=True) if graph.is_multigraph() else
             ((u, v, None) for u, v in graph.edges()))
    background = [edge_line(graph, u, v, k) for u, v, k in edges]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, scenario in zip(axes, ('Baseline', 'Full')):
        ax.add_collection(LineCollection(background, colors='#dddddd', linewidths=.3))
        flow = flows[scenario]
        ax.add_collection(LineCollection([edge_line(graph, *e) for e in flow],
                                        colors=COLORS[scenario],
                                        linewidths=[.5 + 4 * n / max_flow for n in flow.values()], alpha=.7))
        counts = rows.loc[rows['scenario'] == scenario, 'chosen_facility'].value_counts()
        for kind, marker, color in [('core', 's', '#c44e52'), ('clinic', '^', '#55a868')]:
            f = facilities.loc[facilities['facility_type'] == kind]
            ax.scatter(f['lon'], f['lat'], s=30 + f['name'].map(counts).fillna(0) * 2,
                       marker=marker, color=color, edgecolor='white', label=kind)
        ax.scatter(config.hazard_center[1], config.hazard_center[0], marker='x', c='black', label='Hazard centre')
        ax.autoscale()
        ax.set_aspect(1 / math.cos(math.radians(config.hazard_center[0])))
        ax.set_title(f'{scenario}; seed={config.seeds[0]}')
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend(frameon=False)
    fig.suptitle(f'Time-optimal routes, severity={severity}, stress={stress}; widths share one scale')
    save_figure(fig, output, 'routes')


def plot_paper_figures(output_dir, maps=False):
    output = Path(output_dir)
    config = ExperimentConfig(**json.loads((output / 'config.json').read_text()))
    summary = pd.read_csv(output / 'run_summary.csv')
    loads = pd.read_csv(output / 'facility_loads.csv')
    facilities = pd.read_csv(output / 'facilities.csv')
    all_runs = summary.loc[summary['subgroup'] == 'All']
    severity, stress = max(config.severity_levels), sorted(config.stress_levels)[len(config.stress_levels) // 2]
    rng = np.random.default_rng(config.analysis_seed)
    severities = sorted(config.severity_levels)
    fig, axes = plt.subplots(len(severities), 2, figsize=(11, 3.2 * len(severities)), squeeze=False)
    for row, value in enumerate(severities):
        for col, (metric, label) in enumerate([('mean_access_min', 'Mean access time (min), reachable'),
                                               ('within_target_share', 'Share within access target, all agents')]):
            ax = axes[row, col]
            for scenario, color in COLORS.items():
                means, lows, highs = [], [], []
                for s in sorted(config.stress_levels):
                    values = all_runs.loc[(all_runs['severity'] == value) & (all_runs['stress'] == s)
                                          & (all_runs['scenario'] == scenario), metric]
                    mean, low, high, _ = bootstrap_mean(values, rng, config.bootstrap_samples)
                    means.append(mean)
                    lows.append(low)
                    highs.append(high)
                ax.plot(sorted(config.stress_levels), means, color=color, marker='o', label=scenario)
                ax.fill_between(sorted(config.stress_levels), lows, highs, color=color, alpha=.12)
            ax.set(xlabel='Exogenous stress', ylabel=label, title=f'Severity={value}')
            ax.spines[['top', 'right']].set_visible(False)
            if col == 1:
                ax.set_ylim(-.03, 1.03)
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=8)
    save_figure(fig, output, 'stress_severity_outcomes')

    reference = loads.loc[(loads['severity'] == severity) & (loads['stress'] == stress)]
    fig, ax = plt.subplots(figsize=(max(9, len(facilities) * .5), 4))
    x = np.arange(len(facilities))
    for offset, scenario in [(-.2, 'Baseline'), (.2, 'Full')]:
        means = reference.loc[reference['scenario'] == scenario].groupby('name')['demand_capacity_ratio'].mean()
        ax.bar(x + offset, facilities['name'].map(means), width=.4, color=COLORS[scenario], label=scenario)
    ax.axhline(1, color='grey', linestyle='--', linewidth=1)
    ax.set_xticks(x, facilities['name'], rotation=45, ha='right')
    ax.set(ylabel='Demand / service capacity in departure window', title=f'Severity={severity}, stress={stress}; mean over seeds')
    ax.legend(frameon=False)
    save_figure(fig, output, 'facility_demand')

    effects = pd.read_csv(output / 'paired_effects.csv')
    effects = effects.loc[(effects['severity'] == severity) & (effects['stress'] == stress)
                          & (effects['subgroup'] == 'All') & (effects['metric'] == 'mean_access_min')]
    fig, ax = plt.subplots(figsize=(7, 4))
    for y, (_, effect) in enumerate(effects.iterrows()):
        ax.plot([effect['ci_lower'], effect['ci_upper']], [y, y], color='#333333')
        ax.scatter(effect['delta'], y, color=COLORS.get(effect['comparison'], '#333333'))
    ax.set_yticks(np.arange(len(effects)), effects['comparison'])
    ax.axvline(0, color='grey', linestyle='--')
    ax.set(xlabel='Change in mean access time (min); 95% seed-bootstrap interval',
           title=f'Paired contrasts; severity={severity}, stress={stress}')
    save_figure(fig, output, 'paired_access_effects')

    fig, ax = plt.subplots(figsize=(6, 4))
    s = np.linspace(0, max(config.threshold_high, max(config.stress_levels)) + 2, 200)
    for threshold in (config.threshold_low, (config.threshold_low + config.threshold_high) / 2, config.threshold_high):
        ax.plot(s, rationality_weight(threshold, s, config.logistic_slope), label=f'Threshold={threshold:g}')
    ax.set(xlabel='Stress', ylabel='Rationality weight', title='Assumed cognitive mechanism')
    ax.legend(frameon=False)
    save_figure(fig, output, 'assumed_mechanism')
    if maps:
        plot_maps(output, config, facilities, severity, stress)
    print(f'Figures saved in {output.resolve()}')
