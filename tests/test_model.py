import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from analysis import bootstrap_mean, paired_effects, summarize_run
from config import ExperimentConfig
from decision import choose_facilities, rationality_weight
from environment import (apply_disaster_friction, build_demo, load_facilities,
                         travel_time_matrix, validate_facilities)
from main import origin_distribution, run_experiment
from simulation import make_population, queue_waits, run_scenario
from visual import get_edge_frequencies, plot_paper_figures


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.graph, self.facilities = build_demo()
        self.config = ExperimentConfig(region='Demo', n_agents=25, seeds=(1, 2, 3),
                                       stress_levels=(3.5, 8.5), severity_levels=(0, 3),
                                       temperature=0, bootstrap_samples=50)

    def test_rationality_is_monotonic_and_bounded(self):
        weights = rationality_weight(5, np.array([-1e100, 0, 5, 10, 1e100]))
        self.assertTrue(np.all(np.diff(weights) <= 0))
        self.assertEqual(weights[2], .5)
        self.assertTrue(((weights >= 0) & (weights <= 1)).all())

    def test_types_do_not_depend_on_facility_size(self):
        facilities = validate_facilities(pd.DataFrame([
            dict(name='Core', facility_type='core', attraction_size=450, prestige=1, servers=1, service_minutes=10),
            dict(name='Clinic', facility_type='clinic', attraction_size=100, prestige=1, servers=1, service_minutes=10)]))
        times = np.full((1, 2), 100.0)
        base, _ = choose_facilities(times, facilities, [3], 100, [False], self.config)
        nudge, _ = choose_facilities(times, facilities, [3], 100, [False], self.config, friction=.1)
        self.assertEqual(facilities.iloc[base[0]]['name'], 'Core')
        self.assertEqual(facilities.iloc[nudge[0]]['name'], 'Clinic')

    def test_core_requirement_and_inaccessible(self):
        times = np.full((2, len(self.facilities)), np.inf)
        clinic = np.flatnonzero(self.facilities['facility_type'] == 'clinic')[0]
        times[:, clinic] = 0
        choices, _ = choose_facilities(times, self.facilities, [3, 3], 8, [True, False], self.config)
        self.assertEqual(choices.tolist(), [-1, clinic])

    def test_fastest_and_colocated(self):
        times = np.array([[60, 0, 10, 100, 1000, 2000]], dtype=float)
        choices, alpha = choose_facilities(times, self.facilities, [3], 8, [False], self.config, fastest=True)
        self.assertEqual(choices[0], 1)
        self.assertEqual(alpha[0], 1)

    def test_invalid_times_and_noise(self):
        times = np.zeros((1, len(self.facilities)))
        times[0, 0] = -1
        with self.assertRaises(ValueError):
            choose_facilities(times, self.facilities, [3], 3, [False], self.config)
        with self.assertRaises(ValueError):
            choose_facilities(np.zeros_like(times), self.facilities, [3], 3, [False], replace(self.config, temperature=.1))

    def test_hazard_copy_and_zero_control(self):
        zero = apply_disaster_friction(self.graph, 35, 139, severity=0, closure_intensity=.5)
        damaged = apply_disaster_friction(self.graph, 35, 139, severity=3)
        repeated = apply_disaster_friction(damaged, 35, 139, severity=3)
        self.assertEqual(self.graph.number_of_edges(), zero.number_of_edges())
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            self.assertEqual(zero[u][v][k]['travel_time'], data['travel_time'])
            self.assertEqual(repeated[u][v][k]['travel_time'], damaged[u][v][k]['travel_time'])
            self.assertGreaterEqual(damaged[u][v][k]['travel_time'], data['travel_time'])
        closed = apply_disaster_friction(self.graph, 35, 139, severity=3, closure_intensity=.5)
        self.assertLess(closed.number_of_edges(), self.graph.number_of_edges())

    def test_reverse_routing_matches_directed_shortest_paths(self):
        graph = nx.MultiDiGraph()
        graph.add_edge(0, 1, travel_time=100)
        graph.add_edge(0, 1, travel_time=3)
        graph.add_edge(1, 2, travel_time=4)
        graph.add_node(3)
        f = pd.DataFrame({'node_id': [0, 2, 2]})
        result = travel_time_matrix(graph, [0, 1, 3], f)
        np.testing.assert_array_equal(result, [[0, 7, 7], [np.inf, 4, 4], [np.inf, np.inf, np.inf]])

    def test_routing_cost_scales_with_facilities_not_agents(self):
        with patch('environment.nx.single_source_dijkstra_path_length', wraps=nx.single_source_dijkstra_path_length) as dijkstra:
            travel_time_matrix(self.graph, list(self.graph) * 10, self.facilities)
            self.assertEqual(dijkstra.call_count, self.facilities['node_id'].nunique())

    def test_queue_by_arrival_and_id_not_row_order(self):
        f = self.facilities.iloc[:1].copy().reset_index(drop=True)
        f['servers'], f['service_minutes'] = 1, 10
        arrivals = np.array([5, 0, 0, np.nan])
        ids = np.array([2, 1, 0, 3])
        choices = np.array([0, 0, 0, -1])
        waits = queue_waits(arrivals, choices, f, ids)
        np.testing.assert_allclose(waits, [15, 10, 0, np.nan])
        order = [2, 3, 0, 1]
        permuted = queue_waits(arrivals[order], choices[order], f, ids[order])
        np.testing.assert_allclose(permuted, waits[order])
        f['servers'] = 2
        np.testing.assert_allclose(queue_waits(arrivals, choices, f, ids), [5, 0, 0, np.nan])

    def test_no_route_not_zero_time_or_success(self):
        population, rng = make_population(list(self.graph), self.config, 1)
        times = np.full((len(population), len(self.facilities)), np.inf)
        result = run_scenario(population, times, self.facilities, self.config, 3.5,
                              self.config.scenarios[0], rng.gumbel(size=times.shape))
        self.assertTrue(result['access_min'].isna().all())
        self.assertFalse(result['within_target'].any())
        for k, v in [('region', 'Demo'), ('seed', 1), ('severity', 0), ('stress', 3.5)]:
            result[k] = v
        rows, loads = summarize_run(result, self.facilities, self.config)
        self.assertEqual(rows[0]['unreachable_share'], 1)
        self.assertTrue(np.isnan(rows[0]['mean_access_min']))
        self.assertTrue(np.isnan(rows[0]['demand_capacity_cv']))
        self.assertEqual(len(loads), len(self.facilities))
        self.assertEqual(loads['patients'].sum(), 0)

    def test_saved_route_uses_fastest_parallel_edge(self):
        graph = nx.MultiDiGraph()
        graph.add_edge(1, 0, key=0, travel_time=100, length=1)
        graph.add_edge(1, 2, key=0, travel_time=9, length=2)
        graph.add_edge(1, 2, key=1, travel_time=1, length=2)
        graph.add_edge(2, 0, key=0, travel_time=1, length=2)
        rows = pd.DataFrame(dict(start_node=[1], chosen_facility=['h'], reachable=[True]))
        self.assertEqual(get_edge_frequencies(rows, graph, {'h': 0}), {(1, 2, 1): 1, (2, 0, 0): 1})

    def test_seed_reproducibility(self):
        a, rng_a = make_population(list(self.graph), self.config, 23)
        b, rng_b = make_population(list(self.graph), self.config, 23)
        assert_frame_equal(a, b)
        np.testing.assert_equal(rng_a.gumbel(size=(25, 6)), rng_b.gumbel(size=(25, 6)))

    def test_external_origin_weights(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'origins.csv'
            pd.DataFrame({'node_id': [0, 1], 'weight': [0, 10]}).to_csv(path, index=False)
            nodes, weights = origin_distribution(self.graph, path)
            pop, _ = make_population(nodes, self.config, 1, weights)
            self.assertTrue((pop['start_node'] == 1).all())

    def test_single_seed_interval_is_not_fabricated(self):
        mean, low, high, n = bootstrap_mean([4], np.random.default_rng(0), 50)
        self.assertEqual((mean, n), (4, 1))
        self.assertTrue(np.isnan(low) and np.isnan(high))

    def test_config_and_inputs_validation(self):
        for args in [dict(seeds=(1, 1)), dict(radius_km=0), dict(gamma=-1),
                     dict(closure_intensity=0), dict(n_agents=0), dict(stress_levels=())]:
            with self.assertRaises(ValueError):
                ExperimentConfig(**args)
        japan = load_facilities(region='Japan')
        self.assertEqual(japan.set_index('name').loc['Hiroo_Hosp', 'facility_type'], 'core')
        bad = self.facilities.copy()
        bad.loc[0, 'servers'] = 0
        with self.assertRaises(ValueError):
            validate_facilities(bad)

    def test_end_to_end_paired_noop_reproducibility_and_plots(self):
        config = replace(self.config, n_agents=12, friction=1, exposure=1,
                         stress_levels=(6,), severity_levels=(0,), temperature=.05)
        with tempfile.TemporaryDirectory() as d:
            output = run_experiment(config, Path(d) / 'run1')
            second = run_experiment(config, Path(d) / 'run2')
            for file in ('decisions.csv', 'run_summary.csv', 'paired_effects.csv'):
                self.assertEqual((output / file).read_bytes(), (second / file).read_bytes())
            decisions = pd.read_csv(output / 'decisions.csv')
            self.assertEqual(len(decisions), config.n_agents * len(config.seeds) * 5)
            for _, group in decisions.groupby('seed'):
                base = group.loc[group['scenario'] == 'Baseline'].drop(columns='scenario').reset_index(drop=True)
                for scenario in ('Full', 'Friction', 'Exposure'):
                    other = group.loc[group['scenario'] == scenario].drop(columns='scenario').reset_index(drop=True)
                    assert_frame_equal(base, other)
            effects = pd.read_csv(output / 'paired_effects.csv')
            null_effects = effects.loc[effects['comparison'] != 'Fastest', 'delta'].dropna()
            np.testing.assert_allclose(null_effects, 0, atol=1e-12)
            summary = pd.read_csv(output / 'run_summary.csv')
            shuffled = paired_effects(summary.sample(frac=1, random_state=1), config)
            assert_frame_equal(shuffled, paired_effects(summary, config))
            with self.assertRaises(ValueError):
                paired_effects(pd.concat([summary, summary.iloc[:1]]), config)
            with self.assertRaises(FileExistsError):
                run_experiment(config, output)
            plot_paper_figures(output, maps=True)
            self.assertEqual(len(list(output.glob('*.svg'))), 5)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['origin_sampling'], 'uniform_road_nodes')


if __name__ == '__main__':
    unittest.main()
