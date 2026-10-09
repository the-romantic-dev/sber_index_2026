"""Behavior checks for clustering, screening and economic interpretation."""
from copy import deepcopy
from pathlib import Path
import tempfile
import json
import unittest

import numpy as np
import pandas as pd

from solution.clustering import fit_main_months
from solution.config import load_config
from solution.economics import describe_groups, within_region_r2
from solution.study import candidate_status
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


class SolutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / 'config.json')
        cls.enterClassContext(threadpool_limits(limits=cls.config['runtime']['threads']))

    def stable_diagnostics(self):
        """A valid candidate to isolate the constraint tested below."""
        monthly = pd.DataFrame([dict(
            min_size=40, max_share=.6, seed_ARI_min=1.,
            group_Jaccard_min=.9, SW_full=.4,
        )])
        return monthly, pd.DataFrame([dict(ARI=.9)])

    def test_identical_months_keep_groups_and_have_symmetric_graphs(self):
        first = np.random.default_rng(7).normal(size=(40, 7))
        result = fit_main_months(
            np.stack([first, first, first + .01]),
            np.array(['2024-01', '2024-02', '2024-03']), 2, self.config,
        )
        self.assertEqual(result['labels'].shape, (3, 40))
        np.testing.assert_array_equal(result['labels'][0], result['labels'][1])
        self.assertEqual(result['monthly'].loc[1, 'switch_rate'], 0.)
        for graph in result['graphs']:
            self.assertEqual((graph - graph.T).nnz, 0)
            self.assertFalse(graph.diagonal().any())
            self.assertTrue(np.isfinite(graph.data).all())

    def test_requested_three_groups_are_used_in_every_month(self):
        first = np.random.default_rng(8).normal(size=(60, 7))
        result = fit_main_months(
            np.stack([first, first]), np.array(['2023-01', '2023-02']), 3, self.config,
        )
        self.assertTrue(result['monthly'].k.eq(3).all())
        self.assertTrue(all(len(np.unique(labels)) == 3 for labels in result['labels']))
        np.testing.assert_array_equal(result['labels'][0], result['labels'][1])

    def test_graph_neighbors_are_taken_from_config(self):
        first = np.random.default_rng(9).normal(size=(60, 7))
        config = deepcopy(self.config)
        config['graph']['neighbors'] = 4
        small = fit_main_months(first[None], ['2023-01'], 2, config)
        large = fit_main_months(first[None], ['2023-01'], 2, self.config)
        self.assertLess(small['graphs'][0].nnz, large['graphs'][0].nnz)

    def test_economic_association_does_not_depend_on_group_numbers(self):
        y = np.array([0., 1., 10., 11., 20., 21., 2., 3., 12., 13., 22., 23.])
        labels = np.tile([0, 0, 1, 1, 2, 2], 2)
        regions = np.repeat([1, 2], 6)
        renamed = np.array([9, 2, 7])[labels]
        score = within_region_r2(y, labels, regions)
        self.assertAlmostEqual(score, within_region_r2(y, renamed, regions))
        self.assertGreater(score, .99)

    def test_regional_membership_alone_explains_no_within_region_variation(self):
        regions = np.repeat([1, 2], 6)
        score = within_region_r2(np.arange(12.), np.repeat([0, 1], 6), regions)
        self.assertAlmostEqual(score, 0.)

    def test_small_group_is_rejected_even_with_high_silhouette(self):
        monthly, sensitivity = self.stable_diagnostics()
        monthly.loc[0, 'min_size'] = 2
        monthly.loc[0, 'SW_full'] = .8
        result = candidate_status(monthly, sensitivity, 100, .5, True, self.config['screening'])
        self.assertFalse(result['accepted'])
        self.assertIn('small_groups', result['reasons'])

    def test_economic_overlap_and_missingness_block_a_stable_candidate(self):
        monthly, sensitivity = self.stable_diagnostics()
        thresholds = self.config['screening']
        self.assertTrue(candidate_status(monthly, sensitivity, 100, .5, True, thresholds)['accepted'])
        self.assertIn('economic_overlap', candidate_status(monthly, sensitivity, 100, .1, True, thresholds)['reasons'])
        self.assertIn('economic_missingness', candidate_status(monthly, sensitivity, 100, .5, False, thresholds)['reasons'])

    def test_nonfinite_diagnostics_cannot_be_admitted(self):
        monthly, sensitivity = self.stable_diagnostics()
        monthly.loc[0, 'SW_full'] = np.nan
        result = candidate_status(monthly, sensitivity, 100, .5, True, self.config['screening'])
        self.assertFalse(result['accepted'])
        self.assertIn('invalid_diagnostics', result['reasons'])

    def test_sparse_sector_observations_are_retained_but_not_used_to_name_a_group(self):
        labels = np.repeat([0, 1], 50)
        observed = pd.DataFrame(dict(
            wage_total_rub=np.repeat([100., 200.], 50),
            employer_intensity=np.repeat([10., 20.], 50),
            employment_share_A=np.r_[np.full(10, .9), np.full(40, np.nan), np.full(50, .1)],
            employment_share_C=np.full(100, np.nan),
            group_share_public_services=np.full(100, np.nan),
        ))
        profiles, separation, supported = describe_groups(
            labels, observed, np.repeat([10., 20.], 50), np.full((100, 2), .5),
            ['food', 'other'], 2, 2023, self.config['economics'],
        )
        self.assertTrue(supported)
        self.assertGreater(separation, .25)
        self.assertEqual(profiles.loc[0, 'employment_share_A_n'], 10)
        self.assertAlmostEqual(profiles.loc[0, 'employment_share_A'], .9)
        self.assertNotIn('сельского/лесного', profiles.loc[0, 'description'])
        self.assertTrue(profiles.employment_share_C.isna().all())

    def test_config_paths_are_relative_to_the_json_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(dict(data_dir='inputs', output_dir='outputs')), encoding='utf-8')
            config = load_config(path)
            self.assertEqual(config['data_dir'], path.parent.resolve() / 'inputs')
            self.assertEqual(config['output_dir'], path.parent.resolve() / 'outputs')


if __name__ == '__main__':
    unittest.main()
