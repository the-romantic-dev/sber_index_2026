"""Внешний контекст групп 2023–2024 и сравнение с методами исследовательского модуля.

Модуль работает после выбора K и не меняет графы, группы и критерии отбора.
Используются только официальные и открытые научные источники из data/context
(описание — data/README.md и data/sources.csv):

* ИПЦ регионов (Банк России по данным Росстата) — реальные темпы трат;
* индекс доступности рынков, дорожные расстояния и индекс мобильности СберИндекса;
* БДМО Росстата (уже в quality.parquet) — оборонно-промышленный рост зарплат;
* отраслевые экспортные МО, ночные огни VIIRS, климат и расстояние до центра из справочника;
* события с официальными источниками (ЦБ, МЧС, указы Президента, органы власти субъектов);
* национальная типология и метки методов исследовательского модуля, включая нейросети.
"""
from html import escape

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.preprocessing import RobustScaler

from .economics import within_region_r2
from .metrics import monthly_comparison

RESEARCH_NAMES = {
    'research_kmeans_v5': 'Исследование · k-средних, национальные типы (K=5)',
    'research_anchored_real': 'Исследование · якорная динамика, реальные траты',
    'research_archetypes': 'Литература · архетипы PCHA → k-средних',
    'research_ntf': 'Литература · тензорная NTF → k-средних',
    'research_nmf': 'Литература · NMF «образы жизни» → k-средних',
    'research_scaling': 'Литература · остатки закона скейлинга',
    'research_kshape': 'Литература · k-Shape по форме ряда',
    'research_dgi': 'Нейросеть · DGI (графовый энкодер)',
    'research_gae': 'Нейросеть · GAE',
    'research_vgae': 'Нейросеть · VGAE',
    'research_dmon': 'Нейросеть · DMoN (пулинг сообществ)',
    'research_evolvegcn': 'Нейросеть · EvolveGCN-O (временной граф)',
    'research_gconvgru': 'Нейросеть · GConvGRU (временной граф)',
}
OWN_NAMES = {
    'adaptive_temporal_spectral': 'Решение · EWMA spectral, локальный масштаб',
    'adaptive_spectral': 'Решение · spectral, локальный масштаб',
    'kmeans': 'Решение · KMeans',
    'spectral': 'Решение · spectral, общий масштаб',
    'temporal_spectral': 'Решение · EWMA spectral, общий масштаб',
    'anchored_k2': 'Решение · фиксированные прототипы, K=2',
    'anchored_k5': 'Решение · фиксированные прототипы, K=5',
}
EVENT_TITLES_SHORT = {
    'devaluation_2023': 'Девальвация, август 2023',
    'flood_orenburg_2024': 'Паводок, Оренбургская обл.',
    'flood_kurgan_tyumen_2024': 'Паводок, Курганская и Тюменская обл.',
    'ukaz_644_2024': 'Указ № 644, август 2024',
    'toropets_2024': 'Торопец, сентябрь 2024',
}


def load_inputs(data_dir):
    """Прочитать файлы data/context и наборы хакатона; отсутствие файла — ошибка конфигурации."""
    context = data_dir / 'context'
    raw = data_dir / 'raw/sberindex'
    return dict(
        cpi=pd.read_parquet(context / 'regional_cpi.parquet'),
        mobility=pd.read_csv(context / 'mobility_index_szfo.csv').set_index('territory_id'),
        viirs=pd.read_parquet(context / 'viirs_annual.parquet'),
        export=pd.read_csv(context / 'export_mo.csv').set_index('territory_id'),
        coverage=pd.read_csv(context / 'coverage_official.csv'),
        regions=pd.read_csv(context / 'regions_official.csv').set_index('region_key'),
        national=pd.read_csv(context / 'national_types.csv').set_index('territory_id'),
        research_static=pd.read_parquet(context / 'research_labels.parquet'),
        research_monthly=pd.read_parquet(context / 'research_labels_monthly.parquet'),
        events=pd.read_csv(context / 'events.csv', dtype={'territory_ids': str}).fillna(''),
        market_access=pd.read_parquet(raw / 'market_access.parquet').set_index('territory_id'),
        connection_path=raw / 'connection.parquet',
        mo=pd.read_parquet(data_dir / 'processed/municipalities/mo.parquet').set_index('territory_id'),
        quality=pd.read_parquet(data_dir / 'processed/economic_features/quality.parquet'),
    )


def real_totals(total, months, region_codes, cpi):
    """Траты (24, n) в ценах декабря 2022 по ИПЦ региона; без ИПЦ — NaN."""
    index = cpi.pivot(index='date', columns='region_code', values='price_index_dec2022_100')
    index = index.reindex(index=list(months))
    deflator = index.reindex(columns=region_codes).to_numpy() / 100
    return total / deflator


def annual_growth(values):
    """Рост суммы 2024 к сумме 2023 для массива (24, n)."""
    return values[12:].sum(axis=0) / values[:12].sum(axis=0) - 1


def half_year_acceleration(total):
    """Разность годовых темпов: (II пол. 2024 / II пол. 2023) − (I пол. 2024 / I пол. 2023), п.п."""
    first = total[12:18].sum(axis=0) / total[0:6].sum(axis=0)
    second = total[18:24].sum(axis=0) / total[6:12].sum(axis=0)
    return 100 * (second - first)


def defense_boom(quality):
    """Индекс оборонно-промышленного роста по БДМО: доля обработки 2021 и опережение её зарплат.

    shC — доля обработки (раздел C) в занятости 2021; dwC — рост зарплаты в обработке 2021→2024
    (если 2024 скрыт — 2023) сверх роста средней зарплаты МО. Флаг: shC ≥ 0.2 и dwC
    в верхней четверти среди таких МО.
    """
    share = quality['workers_C_2021'] / quality['workers_total_2021']
    later = quality['wage_C_rub_2024'].combine_first(quality['wage_C_rub_2023'])
    total_later = np.where(
        quality['wage_C_rub_2024'].notna(), quality['wage_total_rub_2024'], quality['wage_total_rub_2023']
    )
    excess = np.log(later / quality['wage_C_rub_2021']) - np.log(total_later / quality['wage_total_rub_2021'])
    industrial = share >= 0.2
    threshold = excess[industrial].quantile(0.75)
    flag = (industrial & (excess >= threshold)).where(share.notna() & excess.notna())
    return pd.DataFrame(dict(shC=share, dwC=excess, boom=flag)), float(threshold)


def urban_scaling(per_capita, shares, population, categories, rng, n_boot):
    """Показатель β в законе «траты МО ∝ население^β» по категориям; 95 % бутстрэп-интервал."""
    rows = []
    channels = {'Все категории': per_capita}
    channels.update({name: per_capita * shares[:, column] for column, name in enumerate(categories)})
    for name, values in channels.items():
        ok = np.isfinite(values) & np.isfinite(population) & (values > 0) & (population > 0)
        x, y = np.log(population[ok]), np.log(values[ok] * population[ok])
        beta = np.polyfit(x, y, 1)[0]
        draws = [np.polyfit(x[s], y[s], 1)[0] for s in rng.integers(0, ok.sum(), (n_boot, ok.sum()))]
        rows.append(dict(category=name, beta=beta, low=np.quantile(draws, .025),
                         high=np.quantile(draws, .975), n=int(ok.sum())))
    return pd.DataFrame(rows)


def permutation_r2(y, labels, regions, permutations, seed):
    """Внутрирегиональный R² групп для числового показателя и перестановочное p внутри регионов."""
    observed = np.isfinite(y)
    y, labels, regions = y[observed], labels[observed], regions[observed]
    if len(y) < 20 or len(np.unique(labels)) < 2:
        return np.nan, np.nan, int(observed.sum())
    score = within_region_r2(y, labels, regions)
    rows = [np.flatnonzero(regions == region) for region in np.unique(regions)]
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(permutations):
        shuffled = labels.copy()
        for index in rows:
            shuffled[index] = rng.permutation(labels[index])
        extreme += within_region_r2(y, shuffled, regions) >= score
    return score, (1 + extreme) / (permutations + 1), int(observed.sum())


def national_space(cube, scaler_months):
    """Общее пространство без вычитания регионального уровня (те же веса блоков)."""
    scaler = RobustScaler().fit(cube[:scaler_months].reshape(-1, cube.shape[2]))
    scaled = scaler.transform(cube.reshape(-1, cube.shape[2])).reshape(cube.shape)
    scaled[:, :, :6] /= np.sqrt(6)
    return scaled / np.sqrt(2)


def research_sequences(ids, months, inputs):
    """Метки исследовательских методов в форме (12, n); МО без метки не допускаются."""
    sequences = {}
    static = inputs['research_static'].reindex(ids)
    for column in static:
        values = static[column]
        if values.isna().any():
            continue
        sequences[column] = np.tile(values.to_numpy(int), (len(months), 1))
    monthly = inputs['research_monthly']
    for method, table in monthly.groupby('method'):
        wide = table.pivot(index='territory_id', columns='month', values='label').reindex(index=ids)
        if not set(months) <= set(wide.columns) or wide[list(months)].isna().any().any():
            continue
        sequences[method] = wide[list(months)].to_numpy(int).T
    return sequences


def spatial_check(ids, labels, graph, connection_path, rng, max_km=150):
    """Дорожные расстояния: соседи в графе и пары одной группы против случайных пар."""
    roads = pd.read_parquet(connection_path, columns=['territory_id_x', 'territory_id_y', 'distance', 'type'])
    roads = roads[roads['type'] == 'highway']
    position = pd.Series(np.arange(len(ids)), index=ids)
    roads = roads[roads.territory_id_x.isin(ids) & roads.territory_id_y.isin(ids)]
    a = position[roads.territory_id_x].to_numpy()
    b = position[roads.territory_id_y].to_numpy()
    distance = np.full((len(ids), len(ids)), np.nan, dtype=np.float32)
    distance[a, b] = roads.distance.to_numpy()
    distance[b, a] = roads.distance.to_numpy()
    graph = graph.tocoo()
    upper = graph.row < graph.col
    edge_distance = distance[graph.row[upper], graph.col[upper]]
    left, right = rng.integers(0, len(ids), (2, 200000))
    keep = left != right
    random_distance = distance[left[keep], right[keep]]
    same = labels[left[keep]] == labels[right[keep]]
    rows = [
        dict(pairs='рёбра графа сходства (декабрь 2024)', n=int(np.isfinite(edge_distance).sum()),
             median_km=np.nanmedian(edge_distance), share_within_km=np.nanmean(edge_distance <= max_km)),
        dict(pairs='пары одной группы', n=int(np.isfinite(random_distance[same]).sum()),
             median_km=np.nanmedian(random_distance[same]), share_within_km=np.nanmean(random_distance[same] <= max_km)),
        dict(pairs='случайные пары МО', n=int(np.isfinite(random_distance).sum()),
             median_km=np.nanmedian(random_distance), share_within_km=np.nanmean(random_distance <= max_km)),
    ]
    return pd.DataFrame(rows).assign(threshold_km=max_km)


def spending_spikes(total):
    """Отклонение log трат месяца от среднего двух соседних месяцев минус медиана МО, (24, n)."""
    log_total = np.log(total)
    spike = np.full_like(log_total, np.nan)
    inner = log_total[1:-1] - (log_total[:-2] + log_total[2:]) / 2
    spike[1:-1] = inner - np.nanmedian(inner, axis=1, keepdims=True)
    return spike


def event_checks(events, ids, months_all, labels_all, total):
    """Смена группы и всплеск трат в месяц события и следующий за ним."""
    switches = np.vstack([np.zeros(labels_all.shape[1], bool), labels_all[1:] != labels_all[:-1]])
    base_rate = switches[1:].mean()
    spikes = spending_spikes(total)
    position = pd.Series(np.arange(len(ids)), index=ids)
    rows = []
    for event in events.itertuples():
        month = list(months_all).index(event.month)
        window = [m for m in [month, month + 1] if m < len(months_all)]
        if event.territory_ids:
            members = [int(x) for x in event.territory_ids.split(';') if int(x) in position.index]
            columns = position[members].to_numpy()
        else:
            members, columns = [], np.arange(len(ids))
        if not len(columns):
            continue
        switched = switches[np.ix_(window, columns)].any(axis=0).mean()
        expected = 1 - (1 - base_rate) ** len(window)
        spike = np.nanmean(spikes[month, columns]) if 0 < month < len(months_all) - 1 else np.nan
        rank = (np.nanmean([(np.abs(spikes[1:-1, c]) >= abs(spikes[month, c])).mean() for c in columns])
                if np.isfinite(spike) else np.nan)
        rows.append(dict(event=event.event, title=event.title, source=event.source, month=event.month,
                         n_mo=len(columns) if members else 'все', switched_share=switched,
                         expected_switch_share=expected, spike_pct=100 * spike, spike_month_rank=rank))
    monthly_rate = pd.DataFrame(dict(month=months_all[1:], switch_rate=switches[1:].mean(axis=1)))
    return pd.DataFrame(rows), monthly_rate


def group_context(results, inputs, config):
    """Сводка внешних показателей по декабрьским группам и их внутрирегиональная связь."""
    panel = results['panel']
    ids = panel['ids']
    labels = results['main']['labels'][-1]
    regions = panel['regions']
    positions = pd.Index(panel['ids_all']).get_indexer(ids)
    total = panel['total'][:, positions]
    mo = inputs['mo'].reindex(ids)
    quality = inputs['quality'].reindex(ids)
    real = real_totals(total, panel['months'], mo.region_code.to_numpy(), inputs['cpi'])
    boom, threshold = defense_boom(inputs['quality'])
    viirs = inputs['viirs'].query('year == 2024').set_index('territory_id').reindex(ids)
    population = quality['population_average_2024'].to_numpy()
    numeric = pd.DataFrame({
        'Население 2024, тыс.': population / 1000,
        'Номинальный рост трат 2024/2023, %': 100 * annual_growth(total),
        'Реальный рост трат (ИПЦ региона), %': 100 * annual_growth(real),
        'Ускорение во II пол. 2024, п.п.': half_year_acceleration(total),
        'Индекс доступности рынков': inputs['market_access'].market_access.reindex(ids).to_numpy(),
        'Расстояние до центра региона, км': mo.dist_capital_km.to_numpy(),
        'Температура года, °C': mo.clim_T2M.to_numpy(),
        'Ночные огни VIIRS на жителя (log)': np.log(viirs.rad_masked.to_numpy() / population),
        'Индекс мобильности 2024, км (СЗФО)': inputs['mobility'].mobility_km_2024.reindex(ids).to_numpy(),
    }, index=ids)
    flags = pd.DataFrame({
        'Оборонно-промышленный рост зарплат (БДМО)': boom.boom.reindex(ids).astype(float).to_numpy(),
        'Экспортная отрасль': inputs['export'].export_sector.reindex(ids).notna().astype(float).to_numpy(),
        'Агломерация': inputs['national'].agglomeration.reindex(ids).astype(float).to_numpy(),
    }, index=ids)
    settings = config['context']
    rows = []
    for column in list(numeric) + list(flags):
        values = (numeric[column] if column in numeric else flags[column]).to_numpy(float)
        r2, p, n = permutation_r2(values, labels, regions, settings['permutations'], config['random_seed'])
        row = dict(indicator=column, kind='доля МО' if column in flags else 'медиана', n=n,
                   within_region_R2=r2, permutation_p=p)
        for group in np.unique(labels):
            chosen = values[labels == group]
            chosen = chosen[np.isfinite(chosen)]
            row[f'group_{group + 1}'] = (chosen.mean() if column in flags else np.median(chosen)) if len(chosen) else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    sectors = pd.crosstab(inputs['export'].export_sector.reindex(ids).fillna('—').to_numpy(), labels + 1)
    sectors = sectors.drop(index='—', errors='ignore')
    per_mo = numeric.join(flags).assign(group=labels + 1)
    per_mo['export_sector'] = inputs['export'].export_sector.reindex(ids)
    return table, sectors, per_mo, dict(boom_threshold=threshold, boom_n=int(boom.boom.sum()))


def two_axes(results, inputs):
    """Сопоставление групп (положение внутри региона) с национальной типологией."""
    ids = results['panel']['ids']
    labels = results['main']['labels'][-1] + 1
    national = inputs['national'].reindex(ids)
    by_type = pd.crosstab(national.national_type.to_numpy(), labels)
    by_within = pd.crosstab(national.within_region_type.to_numpy(), labels)
    known = national.national_type.notna().to_numpy()
    agreement = dict(
        ARI_national=ARI(national.national_type[known], labels[known]),
        ARI_within_region=ARI(national.within_region_type[known], labels[known]),
    )
    return by_type, by_within, agreement


def compare_research(results, inputs, config):
    """ICVI исследовательских методов и методов решения в двух общих пространствах 2024."""
    panel = results['panel']
    months = results['months']
    research = research_sequences(panel['ids'], months, inputs)
    own = {k: v for k, v in results['sequences'].items() if not k.startswith('adaptive_temporal_k')}
    regional = monthly_comparison(months, results['consumer_z'][12:], research, config)
    national = national_space(panel['raw_cube'], config['features']['scaler_months'])
    national = monthly_comparison(months, national[12:], {**own, **research}, config)
    summary = []
    for space, table in [('внутри региона (пространство решения)', pd.concat([results['monthly_comparison_table'][
            results['monthly_comparison_table'].method.isin(own)], regional])), ('национальное (без вычитания региона)', national)]:
        agg = table.groupby('method', sort=False).agg(
            K=('k', 'median'), SW=('SW_full', 'mean'), CH=('CH', 'mean'), S_Dbw=('S_Dbw', 'mean'),
            AVI=('AVI', 'mean'), AVU=('AVU', 'mean'), MQ=('MQ', 'mean'), switch_rate=('switch_rate', 'mean'),
        ).reset_index()
        summary.append(agg.assign(space=space))
    summary = pd.concat(summary, ignore_index=True)
    summary['source'] = np.where(summary.method.str.startswith('research_'), 'исследовательский модуль', 'решение')
    summary['title'] = summary.method.map({**OWN_NAMES, **RESEARCH_NAMES}).fillna(summary.method)
    return summary, research


def table_html(frame, digits=2, percent=()):
    """Простая HTML-таблица без внешних библиотек."""
    head = ''.join(f'<th>{escape(str(c))}</th>' for c in frame.columns)
    body = []
    for row in frame.itertuples(index=False):
        cells = []
        for column, value in zip(frame.columns, row):
            if isinstance(value, (float, np.floating)):
                text = '—' if not np.isfinite(value) else (
                    f'{100 * value:.1f} %' if column in percent else f'{value:,.{digits}f}'.replace(',', ' '))
            else:
                text = escape(str(value))
            cells.append(f'<td>{text}</td>')
        body.append('<tr>' + ''.join(cells) + '</tr>')
    return f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def context_html(context):
    """Раздел презентации «Контекст 2023–2024»."""
    groups = [c for c in context['groups'].columns if c.startswith('group_')]
    table = context['groups'][['indicator', 'kind', *groups, 'within_region_R2', 'permutation_p', 'n']].copy()
    table.columns = ['Показатель', 'Сводка', *[f'Группа {g.split("_")[1]}' for g in groups], 'R² внутри регионов', 'p', 'n']
    table['p'] = [f'{v:.3f}' if np.isfinite(v) else '—' for v in table['p']]
    table['n'] = table['n'].astype(int).astype(str)
    for column in table.columns[2:2 + len(groups)]:
        table[column] = [f'{100 * v:.0f} %' if kind == 'доля МО' and np.isfinite(v) else v
                         for v, kind in zip(table[column], table['Сводка'])]
    axes = context['axes_type'].reset_index().rename(columns={'index': 'Национальный тип'})
    axes.columns = ['Национальный тип', *[f'Группа {c}' for c in axes.columns[1:]]]
    comparison = context['comparison'].copy()
    comparison['K'] = comparison['K'].round().astype(int).astype(str)
    blocks = []
    for space, table_space in comparison.groupby('space', sort=False):
        best = table_space.sort_values('SW', ascending=False).head(10)
        best = best[['title', 'source', 'K', 'SW', 'CH', 'S_Dbw', 'AVI', 'AVU', 'MQ']].rename(
            columns=dict(title='Метод', source='Источник'))
        blocks.append(f'<p><strong>Пространство: {escape(space)}</strong></p>' + table_html(best, 3))
    comparison_html = ''.join(blocks)
    scaling = context['scaling'].rename(columns=dict(category='Категория', beta='β', low='95 % от', high='до', n='МО'))
    events = context['events'][['title', 'month', 'n_mo', 'switched_share', 'expected_switch_share', 'spike_pct', 'source']]
    events = events.rename(columns=dict(title='Событие', month='Месяц', n_mo='МО', switched_share='Сменили группу',
                                        expected_switch_share='Ожидаемо при обычной смене', spike_pct='Всплеск трат, %',
                                        source='Источник'))
    coverage = context['coverage'].rename(columns=dict(group='Группа МО', MO='МО', pop_mln='Население, млн', pop_share='Доля населения'))
    spatial = context['spatial'].rename(columns=dict(pairs='Пары', n='n', median_km='Медиана, км',
                                                    share_within_km='Ближе порога', threshold_km='Порог, км'))
    a = context['agreement']
    sectors = context['sectors'].reset_index()
    sectors.columns = ['Отрасль', *[f'Группа {c}' for c in sectors.columns[1:]]]
    return f'''
    <section class="presentation" aria-labelledby="context-title">
      <h2 id="context-title">Контекст 2023–2024: экономика, санкции, события</h2>
      <p>Блок не участвует в построении графов и выборе K. Источники — официальная статистика
      (Росстат, Банк России, СберИндекс), нормативные акты и научные данные VIIRS.</p>
      <h3>Две оси типологии</h3>
      <p>Группы решения описывают положение МО <strong>внутри своего региона</strong>: региональный уровень вычтен.
      Национальная типология исследовательского модуля описывает уровень и структуру трат между регионами.
      Согласие с национальными типами: ARI = {a["ARI_national"]:.2f}; с внутрирегиональной осью: ARI = {a["ARI_within_region"]:.2f}.</p>
      {table_html(axes, 0)}
      <h3>Чем различаются группы во внешних данных</h3>
      {table_html(table, 2)}
      <p class="small">R² — доля внутрирегионального разброса показателя, объяснённая группами; p — перестановки
      меток внутри регионов. Оборонно-промышленный рост: доля обработки 2021 ≥ 20 % и опережающий рост зарплат
      в обработке (верхняя четверть, БДМО). Экспортные отрасли:</p>
      {table_html(sectors, 0)}
      <h3>Закон скейлинга: траты растут быстрее населения</h3>
      {table_html(scaling, 2)}
      <p class="small">β &gt; 1: крупные МО тратят на жителя больше (Bettencourt et al., 2007; Sobolevsky et al., 2015).
      Сильнее всего от размера зависит общепит, слабее всего — продовольствие.</p>
      <h3>Сравнение с нейросетями и подходами из литературы</h3>
      {comparison_html}
      <p class="small">Лучшие 10 методов в каждом пространстве по SW; SW и AVU систематически выше при K=2. Метки исследовательских методов получены в
      отдельном репозитории (GPU: DGI, GAE, VGAE, DMoN, EvolveGCN-O, GConvGRU) и оценены здесь теми же ICVI на тех же месяцах.
      Полная таблица — context/research_comparison.csv.</p>
      <h3>События и переходы между группами</h3>
      {table_html(events, 2, percent=('Сменили группу', 'Ожидаемо при обычной смене'))}
      <h3>География сходства (дорожные расстояния СберИндекса)</h3>
      {table_html(spatial, 2, percent=('Ближе порога',))}
      <h3>Кого нет в данных</h3>
      {table_html(coverage, 3, percent=('Доля населения',))}
      <p class="small">В данных СберИндекса нет восьми субъектов со средним уровнем реагирования (Указ Президента № 757
      от 19.10.2022) и всех ЗАТО. Выводы относятся к остальной территории страны.</p>
    </section>'''


def run_context(results, config):
    """Рассчитать контекстный блок, сохранить таблицы в output_dir/context и вернуть словарь для атласа."""
    settings = config['context']
    inputs = load_inputs(config['data_dir'])
    rng = np.random.default_rng(config['random_seed'])
    out = config['output_dir'] / 'context'
    out.mkdir(parents=True, exist_ok=True)
    panel = results['panel']
    ids = panel['ids']
    positions = pd.Index(panel['ids_all']).get_indexer(ids)
    total = panel['total'][:, positions]
    shares = panel['shares'][:, positions]
    population = inputs['quality'].reindex(ids)

    groups, sectors, per_mo, boom_info = group_context(results, inputs, config)
    by_type, by_within, agreement = two_axes(results, inputs)
    scaling = pd.concat([
        urban_scaling(total[window].mean(axis=0) * 12, shares[window].mean(axis=0),
                      population[f'population_average_{year}'].to_numpy(), panel['categories'],
                      rng, settings['bootstrap']).assign(year=year)
        for year, window in [(2023, slice(0, 12)), (2024, slice(12, 24))]
    ])
    comparison, research = compare_research(results, inputs, config)
    labels_all = np.concatenate([results['training_runs'][results['selected_k']]['labels'], results['main']['labels']])
    events, switch_by_month = event_checks(inputs['events'], ids, panel['months'], labels_all, total)
    spatial = spatial_check(ids, results['main']['labels'][-1], results['main']['graphs'][-1],
                            inputs['connection_path'], rng, settings['road_km'])

    tables = {
        'group_context.csv': groups, 'export_sectors_by_group.csv': sectors.reset_index(),
        'axes_national_types.csv': by_type.reset_index(), 'axes_within_region.csv': by_within.reset_index(),
        'urban_scaling.csv': scaling, 'research_comparison.csv': comparison, 'events.csv': events,
        'switch_rate_by_month.csv': switch_by_month, 'spatial_check.csv': spatial,
        'coverage_official.csv': inputs['coverage'], 'regions_official.csv': inputs['regions'].reset_index(),
    }
    for name, table in tables.items():
        table.to_csv(out / name, index=False)
    per_mo.to_csv(out / 'mo_context.csv')
    (out / 'summary.json').write_text(pd.Series(dict(**agreement, **boom_info)).to_json(force_ascii=False, indent=2),
                                      encoding='utf-8')
    context = dict(groups=groups, sectors=sectors, axes_type=by_type, axes_within=by_within, agreement=agreement,
                   scaling=scaling.query('year == 2024').drop(columns='year'), comparison=comparison,
                   events=events, spatial=spatial, coverage=inputs['coverage'], research=research,
                   per_mo=per_mo, national=inputs['national'].reindex(ids))
    context['html'] = context_html(context)
    return context
