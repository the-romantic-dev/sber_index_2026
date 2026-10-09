"""Таблицы, массивы, графы и русский отчёт с экономическими профилями."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from . import METHOD


def clean_json(value):
    """Привести массивы и скаляры к JSON, сохранив пропуски как null."""
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_tables(results, output_dir, config):
    """Экспортировать таблицы, признаки, назначения и графы полного расчёта."""
    panel = results['panel']
    sequences = results['sequences']
    columns = {
        f'monthly_{method}_{month}': pd.Series(labels[index], index=panel['ids'])
        for method, labels in sequences.items()
        for index, month in enumerate(results['months'])
    }
    assignments = pd.concat([panel['coverage'], pd.DataFrame(columns)], axis=1)
    assignments.to_csv(output_dir / 'atlas_municipalities.csv', encoding='utf-8-sig')
    tables = {
        'atlas_flows.csv': results['flows'],
        'comparison.csv': results['comparison'],
        'monthly_comparison.csv': results['monthly_comparison_table'],
        'main_monthly_metrics.csv': results['main']['monthly'],
        'main_sensitivity.csv': results['main']['sensitivity'],
        'forecast_errors.csv': results['forecast_errors'],
        'seed_stability.csv': results['baseline_seed_checks'],
        'cluster_count_study.csv': results['cluster_study'],
        'cluster_economic_profiles.csv': results['cluster_profiles'],
        'external_validation_2024.csv': results['external_validation'],
        'external_validation_all_k_2024.csv': results['external_validation_all'],
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False)
    results['economic_profiles'].to_csv(output_dir / 'economic_profiles_2024.csv')
    results['economic_counts'].to_csv(output_dir / 'economic_observed_counts_2024.csv')
    results['economic_observed'].to_parquet(output_dir / 'economic_observed_2024.parquet')
    np.savez_compressed(
        output_dir / 'dynamic.npz', ids=panel['ids'], months=results['months'],
        months_all=panel['months'], consumer_z=results['consumer_z'],
        smoothed=results['smoothed'], **sequences,
    )
    graph_dir = output_dir / 'monthly_graphs'
    graph_dir.mkdir(exist_ok=True)
    for month, graph in zip(results['months'], results['main']['graphs']):
        sparse.save_npz(graph_dir / f'{METHOD}_{month}.npz', graph)
    monthly, sensitivity, labels = [], [], {}
    for year, runs in [(2023, results['training_runs']), (2024, results['study_runs'])]:
        for k, result in runs.items():
            monthly.append(result['monthly'].assign(year=year, k=k))
            sensitivity.append(result['sensitivity'].assign(year=year, k=k))
    for k, result in results['study_runs'].items():
        labels[f'k{k}'] = np.concatenate([
            results['training_runs'][k]['labels'], result['labels']
        ])
    pd.concat(monthly).to_csv(output_dir / 'cluster_study_monthly.csv', index=False)
    pd.concat(sensitivity).to_csv(output_dir / 'cluster_study_sensitivity.csv', index=False)
    np.savez_compressed(
        output_dir / 'cluster_study_labels.npz', ids=panel['ids'],
        months=panel['months'], **labels,
    )
    (output_dir / 'run_config.json').write_text(
        json.dumps(clean_json(config), ensure_ascii=False, indent=2), encoding='utf-8'
    )


def write_cluster_report(results, output_dir, config):
    """Сформировать подробный отчёт для каждого K, включая отвергнутые варианты."""
    ids = results['ids']
    selected_k = results['selected_k']
    study_runs = results['study_runs']
    cluster_study = results['cluster_study']
    cluster_profiles = results['cluster_profiles']
    external_validation_all = results['external_validation_all']
    thresholds, economics = config['screening'], config['economics']
    ceiling = config['study']['max_count']
    reason_titles = dict(invalid_diagnostics='неполные диагностические метрики', small_groups='слишком маленькие группы', dominant_group='одна группа доминирует',
        seed_instability='неустойчивость seed', sensitivity_instability='чувствительность к настройкам',
        group_instability='неустойчивость отдельных групп', weak_geometry='слабая геометрическая разделимость',
        economic_missingness='недостаточно наблюдаемых экономических показателей', economic_overlap='близкие экономические профили')
    report = ['# Исследование числа групп', '', f'Выбран K={selected_k} по 2023. Проверены K={min(study_runs)}…{max(study_runs)} на {len(ids)} МО.', '',
        f'Меняется только K. Одна потребительская модель, EWMA, адаптивный граф {config['graph']['neighbors']} соседей. Критерии зафиксированы в config.json.', '',
        f"Допуск: min_size≥{max(thresholds['min_group_size'], int(np.ceil(thresholds['min_group_fraction'] * len(ids))))}, "
        f"max_share≤{100 * thresholds['max_group_share']:g}%, seed ARI≥{thresholds['min_seed_ari']:.2f}, "
        f"sensitivity ARI≥{thresholds['min_sensitivity_ari']:.2f}, group Jaccard≥{thresholds['min_group_jaccard']:.2f}, "
        f"SW≥{thresholds['min_silhouette']:.2f}; для каждого профиля ≥{economics['min_supported_features']} показателей "
        f"с n≥{economics['min_observations']} и покрытием≥{100 * economics['min_coverage']:g}%; "
        f"попарное отличие≥{thresholds['min_economic_separation']:.2f} общего IQR.", '',
        'Критерии являются рабочими ограничениями; они не устанавливают истинное число экономических типов. '
        '2024 не использован для перенастройки K. Выбор полного состава МО использует наличие всех 24 месяцев.', '',
        '| K | Допуск 2023 | Допуск 2024 | SW 2023 | SW 2024 | ARI чувств. 2023 | Jaccard 2023 | Причины отказа 2023 |',
        '|---|---|---|---|---|---|---|---|']
    for k in study_runs:
        a = cluster_study.query('k == @k and year == 2023').iloc[0]
        b = cluster_study.query('k == @k and year == 2024').iloc[0]
        reasons = ', '.join(reason_titles[name] for name in a.reasons.split(';') if name)
        report.append(f'| {k} | {"да" if a.accepted else "нет"} | {"да" if b.accepted else "нет"} | {a.SW_full:.3f} | {b.SW_full:.3f} | {a.sensitivity_ARI_min:.3f} | {a.group_Jaccard_min:.3f} | {reasons or "—"} |')
    strict_candidates = cluster_study.query('year == 2023 and accepted and sensitivity_ARI_min >= .80').k.tolist()
    strict_k = max(strict_candidates) if strict_candidates else None
    report += ['', f"При дополнительном пороге ARI чувствительности ≥0.80 допуск сохраняется до K={strict_k}. "
        'Поэтому выбранный вариант — допустимый компромисс детализации и устойчивости; более грубое разбиение остаётся консервативным ориентиром.', '',
        '![Сравнение числа групп](cluster_count_comparison.png)']
    report += ['', 'Номера групп локальны для каждого K; разбиения разных K не обязательно вложены друг в друга.', '', 'Профили ниже относятся к декабрю каждого года. Расходы — медиана среднегодовых безналичных расходов жителей, '
        'а остальные значения — медианы исходной годовой статистики среди муниципалитетов группы. '
        'Доля отрасли рассчитана для работников обследуемых организаций без МСП. Зарплата работников не равна доходам жителей.', '',
        f"Названия основаны на отклонении медианы группы от общей медианы ≥{economics['description_iqr']:g} IQR при достаточном покрытии. "
        'Это описания выборки: абсолютные различия могут отражать региональную структуру и пропуски. '
        'Проверка связи внутри регионов приведена после профилей. Доли корзины, числа наблюдений и покрытие каждого показателя — в cluster_economic_profiles.csv.']
    for k in study_runs:
        report += ['', f'## K={k}', '']
        for year in [2023, 2024]:
            status = cluster_study.query('k == @k and year == @year').iloc[0]
            report += [f'### {year}', '', 'Допуск: ' + ('да' if status.accepted else '; '.join(reason_titles[name] for name in status.reasons.split(';'))), '',
                '| Группа | МО | Расходы, ₽ | Зарплата, ₽ | Работники / 1000 | Сельское/лесное, % | Обработка, % | Публичные услуги, % | Описание |',
                '|---|---|---|---|---|---|---|---|---|']
            for row in cluster_profiles.query('k == @k and year == @year').itertuples():
                report.append(f'| {row.group + 1} | {row.size} | {row.spending_rub:.0f} | {row.wage_total_rub:.0f} | {row.employer_intensity:.1f} | {100 * row.employment_share_A:.1f} | {100 * row.employment_share_C:.1f} | {100 * row.group_share_public_services:.1f} | {row.description} |')
    report += ['', '## Связь с экономикой внутри регионов · 2024', '',
        'R² — доля остаточной дисперсии после эффектов региона, объяснённая категориальным эффектом групп. '
        f"{economics['permutations']} перестановок внутри регионов. Поправка на все проверенные K и 3 отраслевых показателя. "
        'Это дополнительная проверка связи на тех же МО, а не доказательство причинности.', '',
        '| K | Показатель | n | R² | p с поправкой |', '|---|---|---|---|---|']
    for row in external_validation_all.itertuples():
        report.append(f'| {row.k} | {row.feature} | {row.n} | {row.within_region_R2:.3f} | {row.p_bonferroni_all:.3f} |')
    if max(study_runs) == config['study']['max_count'] and cluster_study.query('year == 2023 and k == @ceiling').accepted.any():
        report += ['', f'Верхняя граница K={ceiling} прошла: предел допустимой детализации пока не установлен.']
    (output_dir / 'cluster_count_report.md').write_text('\n'.join(report), encoding='utf-8')
