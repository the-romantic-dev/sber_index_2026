"""Наблюдаемые экономические профили и проверка связи вне выбора K."""
import numpy as np
import pandas as pd


def within_region_r2(y, labels, regions):
    """Доля внутрирегиональной дисперсии, объяснённая категориальным эффектом групп."""
    _, region = np.unique(regions, return_inverse=True)
    _, group = np.unique(labels, return_inverse=True)
    counts = np.bincount(region)
    centered_y = y - (np.bincount(region, weights=y) / counts)[region]
    indicators = np.eye(group.max() + 1)[group]
    for column in range(indicators.shape[1]):
        indicators[:, column] -= (
            np.bincount(region, weights=indicators[:, column]) / counts
        )[region]
    denominator = centered_y @ centered_y
    if denominator <= 0:
        return np.nan
    fitted = indicators @ np.linalg.lstsq(indicators, centered_y, rcond=None)[0]
    return float(np.clip(1 - np.sum((centered_y - fitted) ** 2) / denominator, 0, 1))


def load_economic_data(data_dir, ids, year):
    """Наблюдаемые зарплата, интенсивность и три отраслевые доли; пропуски сохранены."""
    filename = 'sector_features.parquet' if year == 2023 else 'sector_features_2024.parquet'
    sector = pd.read_parquet(data_dir / 'processed/economic_features' / filename).reindex(ids)
    quality = pd.read_parquet(data_dir / 'processed/economic_features/quality.parquet').reindex(ids)
    columns = ['employer_intensity', 'employment_share_A', 'employment_share_C', 'group_share_public_services']
    observed = sector[[f'{column}_{year}' for column in columns]].copy()
    observed.columns = columns
    observed.insert(0, 'wage_total_rub', quality[f'wage_total_rub_{year}'])
    if not observed.index.is_unique or np.isinf(observed.to_numpy()).any():
        raise ValueError('Экономические наблюдения должны иметь уникальные ID и конечные значения')
    if (observed.iloc[:, :2] < 0).any().any():
        raise ValueError('Зарплаты и интенсивность занятости не могут быть отрицательными')
    for column in columns[1:]:
        if not observed[column].dropna().between(0, 1).all():
            raise ValueError('Недопустимая наблюдаемая отраслевая доля: ' + column)
    return observed


def describe_groups(labels, observed, annual_spending, annual_shares, categories, k, year, settings):
    """Медианы, покрытие и описания групп; минимальное попарное отличие в IQR."""
    medians = observed.groupby(labels).median()
    counts = observed.groupby(labels).count()
    sizes = pd.Series(labels).value_counts().sort_index()
    supported = (counts.ge(settings['min_observations']) & counts.div(sizes, axis=0).ge(settings['min_coverage']))
    iqr = observed.quantile(.75) - observed.quantile(.25)
    distances = []
    for left in medians.index:
        for right in medians.index:
            if left >= right:
                continue
            available = supported.loc[left] & supported.loc[right] & iqr.gt(0)
            distance = ((medians.loc[left] - medians.loc[right]).abs() / iqr)[available]
            distances.append(distance.max() if len(distance) else np.nan)
    separation = min(distances) if distances and np.isfinite(distances).all() else np.nan
    rows = []
    baseline = observed.median()
    titles = dict(wage_total_rub='зарплата', employer_intensity='интенсивность занятости',
                  employment_share_A='доля сельского/лесного хозяйства',
                  employment_share_C='доля обработки', group_share_public_services='доля публичных услуг')
    for group in medians.index:
        mask = labels == group
        spending = float(np.median(annual_spending[mask]))
        spending_level = 'выше' if spending >= np.median(annual_spending) else 'ниже'
        traits = []
        for column, title in titles.items():
            deviation = (medians.loc[group, column] - baseline[column]) / iqr[column] if iqr[column] > 0 else 0
            if supported.loc[group, column] and abs(deviation) >= settings['description_iqr']:
                traits.append(f'{title} {"выше" if deviation > 0 else "ниже"} медианы выборки')
        description = f'Расходы {spending_level} медианы выборки; ' + ('; '.join(traits) if traits else 'резких наблюдаемых экономических отличий нет')
        row = dict(k=k, year=year, group=int(group), size=int(sizes[group]), description=description,
                   spending_rub=spending, economic_supported=bool(supported.loc[group].sum() >= settings['min_supported_features']))
        for column in observed:
            row[column] = medians.loc[group, column]
            row[column + '_n'] = int(counts.loc[group, column])
            row[column + '_coverage'] = counts.loc[group, column] / sizes[group]
        for column, title in enumerate(categories):
            row['basket_' + title] = float(np.median(annual_shares[mask, column]))
        rows.append(row)
    return pd.DataFrame(rows), separation, bool(supported.sum(axis=1).ge(settings['min_supported_features']).all())


def external_validation(study_runs, economic_observed, regions, config):
    """Категориальный R² внутри регионов и перестановочные p с общей поправкой."""
    external_rows = []
    for k, run in study_runs.items():
        for column in list(economic_observed.columns)[2:]:
            observed = economic_observed[column].notna().to_numpy()
            y = economic_observed[column].to_numpy()[observed]
            groups = run['labels'][-1][observed]
            observed_regions = regions[observed]
            region_rows = [np.flatnonzero(observed_regions == region) for region in np.unique(observed_regions)]
            score = within_region_r2(y, groups, observed_regions)
            rng = np.random.default_rng(config['random_seed'])
            extreme = 0
            for _ in range(config['economics']['permutations']):
                shuffled = groups.copy()
                for rows in region_rows:
                    shuffled[rows] = rng.permutation(groups[rows])
                extreme += within_region_r2(y, shuffled, observed_regions) >= score
            p_value = (1 + extreme) / (config['economics']['permutations'] + 1) if np.isfinite(score) else np.nan
            external_rows.append(dict(k=k, feature=column, n=observed.sum(), within_region_R2=score,
                permutation_p=p_value, p_bonferroni_all=min(1., 3 * len(study_runs) * p_value) if np.isfinite(p_value) else np.nan))
    external_validation_all = pd.DataFrame(external_rows)

    return external_validation_all
