from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from config import ExperimentConfig
from simulation import adaptive_choices


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig(region='Demo', recommendation_compliance=1,
                                       recommendation_min_gain_min=5)
        self.facilities = pd.DataFrame({'facility_type': ['core', 'clinic'],
                                        'servers': [1, 1], 'service_minutes': [30., 10.]})
        self.population = pd.DataFrame({'agent_id': [0, 1, 2], 'departure_min': [0., 1., 100.],
                                        'requires_core': [False, False, False]})
        self.times = np.array([[60., 120.]] * 3)

    def test_recommendation_reduces_forecast_cost_without_future_knowledge(self):
        baseline = np.zeros(3, dtype=int)
        choices, accepted, gain = adaptive_choices(self.population, self.times, self.facilities, baseline, self.config)
        self.assertEqual(choices.tolist(), [0, 1, 0])
        self.assertTrue(accepted[1])
        self.assertGreaterEqual(gain[1], 5)
        short = adaptive_choices(self.population.iloc[:2], self.times[:2], self.facilities, baseline[:2], self.config)
        np.testing.assert_array_equal(choices[:2], short[0])

    def test_clinical_constraint_and_zero_compliance(self):
        baseline = np.zeros(3, dtype=int)
        for population, config in [(self.population.assign(requires_core=True), self.config),
                                    (self.population, replace(self.config, recommendation_compliance=0))]:
            choices, accepted, _ = adaptive_choices(population, self.times, self.facilities, baseline, config)
            np.testing.assert_array_equal(choices, baseline)
            self.assertFalse(accepted.any())


if __name__ == '__main__':
    unittest.main()
