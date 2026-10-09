"""Загрузка панели 2023–2024 и справочника муниципалитетов."""
import numpy as np
import pandas as pd


def load_consumer(data_dir):
    """Загрузить raw_cube (24, n_primary=1774, 7), total (24, n_all), shares (24, n_all, 6)."""
    raw = pd.read_parquet(data_dir / 'raw/sberindex/consumption.parquet')
    spending = raw.pivot(
        index=['territory_id', 'date'],
        columns='category',
        values='value',
    )
    categories = sorted(set(spending.columns) - {'Все категории'})
    if len(categories) != 5 or 'Все категории' not in spending:
        raise ValueError('Five observed spending categories and total required')
    parts = spending[categories].to_numpy()
    observed_total = spending['Все категории'].to_numpy()
    full = np.column_stack([parts, observed_total - parts.sum(axis=1)])
    if not np.isfinite(full).all() or (full <= 0).any():
        raise ValueError('Расходы и остаток должны быть конечными и положительными')
    observed_shares = full / observed_total[:, None]

    months = np.array(sorted(raw.date.unique()))
    expected_months = np.array([
        f'{year}-{month:02}'
        for year in [2023, 2024]
        for month in range(1, 13)
    ])
    if not np.array_equal(months, expected_months):
        raise ValueError('January 2023 to December 2024 monthly panel required')

    ids_all = np.sort(raw.territory_id.unique())
    panel_index = pd.MultiIndex.from_product(
        [ids_all, months], names=['territory_id', 'date'],
    )
    total = (
        spending['Все категории'].reindex(panel_index).to_numpy()
        .reshape(len(ids_all), 24).T
    )
    shares = (
        pd.DataFrame(observed_shares, index=spending.index)
        .reindex(panel_index).to_numpy()
    )
    shares = shares.reshape(len(ids_all), 24, 6).transpose(1, 0, 2)
    complete = np.isfinite(total).all(axis=0)

    municipalities = (
        pd.read_parquet(data_dir / 'processed/municipalities/mo.parquet')
        .set_index('territory_id')
    )
    if not municipalities.index.is_unique:
        raise ValueError('ID муниципалитетов должны быть уникальными')
    municipalities = municipalities.reindex(ids_all)
    if municipalities.region_code.isna().any():
        raise ValueError('Every consumption municipality requires an observed region')

    regions_all = municipalities.region_code.to_numpy()
    metropolitan = np.isin(regions_all, [77, 78])
    primary = complete & ~metropolitan
    ids = ids_all[primary]

    log_shares = np.log(shares)
    raw_cube = np.concatenate([
        log_shares - log_shares.mean(axis=2, keepdims=True),
        np.log(total)[:, :, None],
    ], axis=2)[:, primary]

    coverage = municipalities[['name', 'region_name', 'region_code']].copy()
    coverage['complete'] = complete
    coverage['metropolitan'] = metropolitan
    coverage['observed_months'] = np.isfinite(total).sum(axis=0)
    return dict(
        ids=ids,
        months=months,
        raw_cube=raw_cube,
        regions=regions_all[primary],
        mo=municipalities.loc[ids],
        categories=categories + ['Прочее'],
        coverage=coverage,
        ids_all=ids_all,
        total=total,
        shares=shares,
    )
