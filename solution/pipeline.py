"""Полный расчёт: данные → признаки → исследование K → сравнение → материалы."""
import os
import time

import pandas as pd
from threadpoolctl import threadpool_limits

from . import METHOD
from .clustering import baseline_sequences, transition_flows
from .data import load_consumer
from .economics import external_validation
from .features import forecasts, ewma
from .metrics import monthly_comparison
from .study import run_cluster_study


def run(config):
    """Рассчитать проект и сохранить материалы в config['output_dir'].

    Возвращает словарь: panel и признаки, результаты исследования K, main,
    sequences, comparison, flows и экономическая проверка. Ноутбук использует
    эти же объекты для таблиц; все числовые параметры приходят из config.json.
    """
    started = time.perf_counter()
    output_dir = config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(output_dir / '.mpl-cache'))
    from .atlas import write_atlas
    from .plotting import save_figures
    from .reporting import save_tables, write_cluster_report

    with threadpool_limits(limits=config['runtime']['threads']):
        panel = load_consumer(config['data_dir'])
        forecast_errors, alpha, consumer_z, _ = forecasts(
            panel['raw_cube'], panel['regions'], config
        )
        smoothed = ewma(consumer_z, alpha)
        study = run_cluster_study(panel, smoothed, config['data_dir'], config)
        k = study['selected_k']
        months = panel['months'][12:]
        main = study['study_runs'][k]
        sequences = {METHOD: main['labels']}
        sequences.update({
            f'adaptive_temporal_k{count}': result['labels']
            for count, result in study['study_runs'].items() if count != k
        })
        baselines, seed_checks = baseline_sequences(panel, consumer_z, smoothed, months, k, config)
        sequences.update(baselines)
        monthly = monthly_comparison(months, consumer_z[12:], sequences, config)
        comparison = monthly.groupby('method', sort=False).agg(
            SW_full_mean=('SW_full', 'mean'), CH_mean=('CH', 'mean'),
            S_Dbw_mean=('S_Dbw', 'mean'), AVI_mean=('AVI', 'mean'),
            AVU_mean=('AVU', 'mean'), MQ_mean=('MQ', 'mean'),
            min_group_size=('min_size', 'min'), max_group_share=('max_share', 'max'),
            mean_switch_rate=('switch_rate', 'mean'),
        ).reset_index()
        flows = pd.DataFrame([
            dict(method=method, **flow)
            for method, labels in sequences.items()
            for index in range(1, len(months))
            for flow in transition_flows(labels[index - 1], labels[index], months[index])
        ])
        observed = study['economic_by_year'][2024].copy()
        observed.columns = [f'{column}_2024' for column in observed]
        external = external_validation(study['study_runs'], observed, panel['regions'], config)
        results = dict(
            panel=panel, ids=panel['ids'], months=months, months_all=panel['months'],
            forecast_errors=forecast_errors, alpha=alpha, consumer_z=consumer_z,
            smoothed=smoothed, main=main, sequences=sequences,
            baseline_seed_checks=seed_checks, monthly_comparison_table=monthly,
            comparison=comparison, flows=flows, economic_observed=observed,
            economic_profiles=observed.groupby(main['labels'][-1]).median(),
            economic_counts=observed.groupby(main['labels'][-1]).count(),
            external_validation_all=external,
            external_validation=external.query('k == @k').copy(),
            **study,
        )
        if config.get('context', {}).get('enabled', False):
            if (config['data_dir'] / 'context').is_dir():
                from . import context
                results['context'] = context.run_context(results, config)
            else:
                print('Внешний контекст пропущен: нет data/context (см. data/README.md)', flush=True)
        graph_xy, edges = save_figures(results, output_dir, config)
        save_tables(results, output_dir, config)
        write_cluster_report(results, output_dir, config)
        write_atlas(results, graph_xy, edges, output_dir, config)
    print(f'Готово: {len(panel["ids"])} МО, K={k}, alpha={alpha}; '
          f'{time.perf_counter() - started:.1f} с. Результаты: {output_dir}')
    return results
