"""Full integration run; compare every partition with the pre-refactor notebook."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from solution.config import load_config
from solution.pipeline import run


def main():
    """Проверить результат реального расчёта на исходной панели."""
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / 'config.json')
    results = run(config)
    reference = np.load(root / 'tests/fixtures/reference_labels.npz')
    np.testing.assert_array_equal(results['ids'], reference['ids'])
    for method, labels in results['sequences'].items():
        assert all(adjusted_rand_score(old, new) == 1
                   for old, new in zip(reference[method], labels)), method
    for k, result in results['study_runs'].items():
        labels = np.concatenate([results['training_runs'][k]['labels'], result['labels']])
        assert labels.shape == (24, len(results['ids']))
        assert all(adjusted_rand_score(old, new) == 1
                   for old, new in zip(reference[f'study_k{k}'], labels)), k
    assert results['selected_k'] == 3
    assert results['flows'].groupby(['method', 'month'])['count'].sum().eq(len(results['ids'])).all()
    assert results['external_validation_all'].within_region_R2.between(0, 1).all()
    assert np.isfinite(results['smoothed']).all()
    summary = dict(partitions_match_reference=True, selected_k=results['selected_k'],
                   studied_k=list(results['study_runs']), methods=len(results['sequences']))
    (config['output_dir'] / 'verification.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('PIPELINE AND ALL REFERENCE PARTITIONS PASS')


if __name__ == '__main__':
    main()
