"""Temporal atlas contracts independent of the real consumption panel."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import sparse

from solution import atlas


class AtlasTest(unittest.TestCase):
    def inputs(self):
        cube = np.random.default_rng(3).normal(size=(6, 8, 7))
        before = np.tile(np.arange(8) % 2, (3, 1))
        after = before.copy()
        after[:, 0] = 1
        graph = sparse.csr_matrix(np.ones((8, 8)) - np.eye(8))
        results = dict(smoothed=cube, months_all=np.array([
            '2023-10', '2023-11', '2023-12', '2024-01', '2024-02', '2024-03',
        ]), ids=np.arange(8), selected_k=2,
            training_runs={2: dict(labels=before, graphs=[graph] * 3)},
            study_runs={2: dict(labels=after)})
        return results, [np.array([[0, 1, .8]])] * 3, dict(random_seed=42)

    def test_full_period_labels_graphs_and_shared_projections(self):
        self.assertTrue(callable(getattr(atlas, 'temporal_views', None)))
        results, edges, config = self.inputs()
        with tempfile.TemporaryDirectory() as directory:
            views = atlas.temporal_views(results, edges, Path(directory), config)
        labels = views['labels_by_k'][2]
        self.assertEqual(labels.shape, (6, 8))
        self.assertEqual(views['months'][2:4].tolist(), ['2023-12', '2024-01'])
        self.assertEqual(np.count_nonzero(labels[2] != labels[3]), 1)
        self.assertEqual(len(views['graph_edges']), 6)
        self.assertEqual(len(views['graph_edges'][0]), 28)
        for key in ['pca_xy', 'tsne_xy']:
            self.assertEqual(views[key].shape, (6, 8, 2))
            self.assertTrue(np.isfinite(views[key]).all())
        self.assertEqual(len(views['pca_variance']), 2)

    def test_projection_cache_reuses_coordinates_when_only_labels_change(self):
        self.assertTrue(callable(getattr(atlas, 'temporal_views', None)))
        results, edges, config = self.inputs()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            first = atlas.temporal_views(results, edges, destination, config)
            results['study_runs'][2]['labels'][:, 1] = 0
            with patch('solution.atlas.TSNE', side_effect=AssertionError('cache not used')):
                second = atlas.temporal_views(results, edges, destination, config)
            np.testing.assert_array_equal(first['tsne_xy'], second['tsne_xy'])
            self.assertEqual(second['labels_by_k'][2][3, 1], 0)
            results['smoothed'][0, 0, 0] += .2
            with patch('solution.atlas.TSNE') as tsne:
                tsne.return_value.fit_transform.return_value = np.zeros((48, 2))
                atlas.temporal_views(results, edges, destination, config)
                tsne.assert_called_once()

    def test_nonfinite_features_and_misaligned_labels_are_rejected(self):
        self.assertTrue(callable(getattr(atlas, 'temporal_views', None)))
        results, edges, config = self.inputs()
        with tempfile.TemporaryDirectory() as directory:
            results['smoothed'][0, 0, 0] = np.nan
            with self.assertRaises(ValueError):
                atlas.temporal_views(results, edges, Path(directory), config)
            results['smoothed'][0, 0, 0] = 0
            results['study_runs'][2]['labels'] = np.zeros((2, 8), dtype=int)
            with self.assertRaises(ValueError):
                atlas.temporal_views(results, edges, Path(directory), config)

    def test_new_group_ids_are_preserved_when_older_groups_disappear(self):
        results, edges, config = self.inputs()
        results['study_runs'][2]['labels'] += 2
        with tempfile.TemporaryDirectory() as directory, patch('solution.atlas.TSNE') as tsne:
            tsne.return_value.fit_transform.return_value = np.zeros((48, 2))
            views = atlas.temporal_views(results, edges, Path(directory), config)
        np.testing.assert_array_equal(views['labels_by_k'][2][3:], results['study_runs'][2]['labels'])


if __name__ == '__main__':
    unittest.main()
