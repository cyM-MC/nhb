"""Exercise installed OSMnx against a local graph; no live network dependency."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from config import ExperimentConfig
from environment import anchor_facilities, build_demo, build_real_network
from main import run_experiment

try:
    import osmnx as ox
except ImportError:
    ox = None


@unittest.skipIf(ox is None, 'Optional OSMnx dependency is not installed')
class OSMAdapterTests(unittest.TestCase):
    def test_graphml_roundtrip_anchor_and_runner(self):
        graph, facilities = build_demo()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'input.graphml'
            ox.save_graphml(graph, path)
            restored = build_real_network('unused', path)
            snapped = anchor_facilities(restored, facilities.drop(columns='node_id'))
            np.testing.assert_array_equal(snapped['node_id'].to_numpy(), facilities['node_id'].to_numpy())
            np.testing.assert_allclose(snapped['snap_distance_m'], 0, atol=1e-5)
            distant = facilities.copy()
            distant.loc[0, 'lat'] = 0
            with self.assertRaises(ValueError):
                anchor_facilities(restored, distant)
            facilities['region'] = 'Japan'
            facility_path = Path(d) / 'facilities.csv'
            facilities.to_csv(facility_path, index=False)
            config = ExperimentConfig(region='Japan', n_agents=5, seeds=(1, 2),
                                      stress_levels=(6,), severity_levels=(0,),
                                      epicenter=(35, 139), bootstrap_samples=20, service_rate_multiplier=2)
            output = run_experiment(config, Path(d) / 'output', facility_path, path)
            raw = pd.read_csv(output / 'facilities_input.csv')
            used = pd.read_csv(output / 'facilities.csv')
            np.testing.assert_allclose(raw['service_minutes'], 10)
            np.testing.assert_allclose(used['service_minutes'], 5)


if __name__ == '__main__':
    unittest.main()
