"""Checks for the external-context block: deflation, scaling, boom flag, research labels."""
import unittest

import numpy as np
import pandas as pd

from solution.context import (
    annual_growth, defense_boom, half_year_acceleration, real_totals,
    research_sequences, spending_spikes, urban_scaling,
)

MONTHS = np.array([f'{year}-{month:02}' for year in [2023, 2024] for month in range(1, 13)])


class ContextTest(unittest.TestCase):
    def test_deflation_removes_regional_price_growth(self):
        cpi = pd.DataFrame([
            dict(region_code=code, date=month, price_index_dec2022_100=100 * (1.1 if month >= '2024' else 1.0))
            for code in [1, 2] for month in MONTHS
        ])
        total = np.where(np.arange(24)[:, None] >= 12, 110., 100.) * np.ones((24, 3))
        real = real_totals(total, MONTHS, np.array([1, 2, 2]), cpi)
        np.testing.assert_allclose(annual_growth(real), 0, atol=1e-12)
        np.testing.assert_allclose(annual_growth(total), .1)

    def test_scaling_recovers_known_exponent(self):
        rng = np.random.default_rng(1)
        population = np.exp(rng.uniform(7, 13, 400))
        per_capita = 1000 * population ** .2 * np.exp(rng.normal(0, .01, 400))
        shares = np.full((400, 2), .5)
        table = urban_scaling(per_capita, shares, population, ['a', 'b'], rng, 50)
        self.assertAlmostEqual(table.loc[0, 'beta'], 1.2, places=2)
        self.assertTrue((table.low <= table.beta).all() and (table.beta <= table.high).all())

    def test_half_year_acceleration_in_points(self):
        total = np.ones((24, 1))
        total[18:] = 1.1
        self.assertAlmostEqual(half_year_acceleration(total)[0], 10.)

    def test_boom_flag_needs_industrial_share_and_top_quartile(self):
        quality = pd.DataFrame(dict(
            workers_C_2021=[30, 30, 30, 30, 5], workers_total_2021=[100] * 5,
            wage_C_rub_2021=[100.] * 5, wage_C_rub_2024=[200., 120., 110., 105., 300.],
            wage_C_rub_2023=[np.nan] * 5, wage_total_rub_2021=[100.] * 5,
            wage_total_rub_2024=[100.] * 5, wage_total_rub_2023=[100.] * 5,
        ))
        table, _ = defense_boom(quality)
        self.assertEqual(table.boom.tolist(), [True, False, False, False, False])

    def test_spikes_are_zero_for_a_smooth_series(self):
        total = np.exp(np.linspace(0, 1, 24))[:, None] * np.ones((24, 4))
        spikes = spending_spikes(total)
        self.assertTrue(np.isnan(spikes[0]).all() and np.isnan(spikes[-1]).all())
        np.testing.assert_allclose(spikes[1:-1], 0, atol=1e-12)

    def test_research_labels_cover_requested_months(self):
        ids = np.array([10, 11, 12])
        months = MONTHS[12:]
        inputs = dict(
            research_static=pd.DataFrame(dict(a=[0, 1, 1, 2], b=[0, None, 1, 1]), index=[10, 11, 12, 13]).astype('Int64'),
            research_monthly=pd.DataFrame([
                dict(territory_id=i, month=m, label=i % 2, method='dyn') for i in ids for m in months
            ]),
        )
        sequences = research_sequences(ids, months, inputs)
        self.assertEqual(sorted(sequences), ['a', 'dyn'])
        self.assertEqual(sequences['a'].shape, (12, 3))
        np.testing.assert_array_equal(sequences['dyn'][0], [0, 1, 0])


if __name__ == '__main__':
    unittest.main()
