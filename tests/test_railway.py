import unittest

import pandas as pd

from railway import summarize_trips


class RailwayTests(unittest.TestCase):
    def test_trip_keys_and_repeated_station_intervals(self):
        records = pd.DataFrame({'uid': [1, 1, 1, 2, 2], 'tid': [3]*5,
                                'nid': [10, 10, 11, 12, 13],
                                'timestamp': pd.to_datetime(['2012-06-01 08:00:00', '2012-06-01 08:01:00',
                                                             '2012-06-01 08:02:00', '2012-06-01 09:00:00',
                                                             '2012-06-01 09:03:00'])})
        trips, visits = summarize_trips(records.iloc[::-1])
        self.assertEqual(len(trips), 2)
        self.assertEqual(trips.duration_sec.tolist(), [120, 180])
        self.assertEqual(visits.loc[visits.next_node.notna(), 'duration_sec'].tolist(), [120, 180])


if __name__ == '__main__':
    unittest.main()
