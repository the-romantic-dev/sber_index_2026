"""Исследование K при одной модели; выбор только по 2023, проверка на 2024."""
import numpy as np
import pandas as pd
from .clustering import fit_main_months
from .economics import load_economic_data, describe_groups


def candidate_status(monthly, sensitivity, n, economic_separation, economic_supported, thresholds):
    """Вернуть допуск и причины отказа по заранее заданным ограничениям."""
    required = ['min_size', 'max_share', 'seed_ARI_min', 'group_Jaccard_min', 'SW_full']
    finite = np.isfinite(monthly[required].to_numpy()).all()
    minimum_size = max(thresholds['min_group_size'], np.ceil(thresholds['min_group_fraction'] * n))
    failures = {
        'invalid_diagnostics': not finite or not np.isfinite(sensitivity.ARI.to_numpy()).all(),
        'small_groups': monthly.min_size.min() < minimum_size,
        'dominant_group': monthly.max_share.max() > thresholds['max_group_share'],
        'seed_instability': monthly.seed_ARI_min.min() < thresholds['min_seed_ari'],
        'sensitivity_instability': sensitivity.ARI.min() < thresholds['min_sensitivity_ari'],
        'group_instability': monthly.group_Jaccard_min.min() < thresholds['min_group_jaccard'],
        'weak_geometry': monthly.SW_full.mean() < thresholds['min_silhouette'],
        'economic_missingness': not economic_supported,
        'economic_overlap': (
            not np.isfinite(economic_separation)
            or economic_separation < thresholds['min_economic_separation']
        ),
    }
    reasons = [name for name, failed in failures.items() if failed]
    return dict(accepted=not reasons, reasons=reasons)


def run_cluster_study(panel, smoothed, data_dir, config):
    """Рассчитать все K и оба года; вернуть метрики, профили, назначения и выбранный K."""
    ids, months_all = panel["ids"], panel["months"]
    months = months_all[12:]
    economic_by_year = {year: load_economic_data(data_dir, ids, year) for year in [2023, 2024]}
    primary_positions = pd.Index(panel['ids_all']).get_indexer(ids)
    training_runs, study_runs, study_rows, profile_tables = {}, {}, [], []
    candidate_counts = list(config['study']['counts'])
    for k in candidate_counts:
        print(f'K={k}: 2023 → 2024', flush=True)
        training_runs[k] = fit_main_months(smoothed[:12], months_all[:12], k=k, config=config)
        study_runs[k] = fit_main_months(smoothed[12:], months, k=k, config=config,
                                        initial_labels=training_runs[k]['labels'][-1])
        for year, run, window in [(2023, training_runs[k], slice(0, 12)), (2024, study_runs[k], slice(12, 24))]:
            profiles, separation, supported = describe_groups(
                run['labels'][-1], economic_by_year[year],
                panel['total'][window, primary_positions].mean(axis=0),
                panel['shares'][window, primary_positions].mean(axis=0), panel['categories'], k, year, config['economics'],
            )
            profile_tables.append(profiles)
            status = candidate_status(run['monthly'], run['sensitivity'], len(ids), separation, supported, config['screening'])
            metrics = run['monthly']
            study_rows.append(dict(k=k, year=year, accepted=status['accepted'], reasons=';'.join(status['reasons']),
                min_size=metrics.min_size.min(), max_share=metrics.max_share.max(),
                seed_ARI_min=metrics.seed_ARI_min.min(), sensitivity_ARI_min=run['sensitivity'].ARI.min(),
                group_Jaccard_min=metrics.group_Jaccard_min.min(), SW_full=metrics.SW_full.mean(),
                switch_rate=metrics.switch_rate.iloc[1:].mean(), economic_separation=separation,
                economic_supported=supported, **metrics[['CH', 'S_Dbw', 'AVI', 'AVU', 'MQ', 'Q']].mean().to_dict()))
        if k == candidate_counts[-1] and study_rows[-2]['accepted'] and k < config['study']['max_count']:
            candidate_counts.extend(range(k + 1, min(config['study']['max_count'], k + config['study']['extension_step']) + 1))

    cluster_study = pd.DataFrame(study_rows)
    cluster_profiles = pd.concat(profile_tables, ignore_index=True)
    admitted = cluster_study.query('year == 2023 and accepted').k.tolist()
    selected_k = max(admitted) if admitted else None
    if selected_k is None:
        raise ValueError('Ни один K не прошёл критерии:\n' + cluster_study.to_string(index=False))

    return dict(training_runs=training_runs, study_runs=study_runs, cluster_study=cluster_study,
        cluster_profiles=cluster_profiles, economic_by_year=economic_by_year, selected_k=selected_k)
