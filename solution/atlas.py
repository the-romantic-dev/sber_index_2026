"""Интерактивная презентация с картой, временными группами и экономическими профилями."""
import base64
import hashlib
import json
from html import escape
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn import __version__ as sklearn_version
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

from .reporting import clean_json

TEMPLATES = Path(__file__).parent / 'templates'


def temporal_views(results, edges, output_dir, config):
    """Общие проекции и согласованные группы/графы за оба года, без повторной кластеризации."""
    cube = np.asarray(results['smoothed'])
    months = np.asarray(results['months_all'])
    n = len(results['ids'])
    if cube.ndim != 3 or cube.shape[:2] != (len(months), n) or not np.isfinite(cube).all():
        raise ValueError('Признаки атласа должны быть конечными и соответствовать месяцам и МО')
    labels_by_k = {
        k: np.concatenate([results['training_runs'][k]['labels'], run['labels']])
        for k, run in results['study_runs'].items()
    }
    if any(labels.shape != cube.shape[:2] or not np.isfinite(labels).all()
           or (labels < 0).any() or (labels != np.floor(labels)).any()
           or any(len(np.unique(month)) != k for month in labels)
           for k, labels in labels_by_k.items()):
        raise ValueError('Группы атласа должны соответствовать всем месяцам и МО')
    training = results['training_runs'][results['selected_k']]
    if len(training['graphs']) + len(edges) != len(months):
        raise ValueError('Граф атласа требуется для каждого месяца')
    training_edges = []
    for graph in training['graphs']:
        upper = sparse.triu(graph, k=1).tocoo()
        training_edges.append(np.column_stack([upper.row, upper.col, upper.data]))

    signature = hashlib.sha256(cube.tobytes() + (
        f'{cube.shape}:{len(training["labels"])}:{config["random_seed"]}:{sklearn_version}:pca-tsne600-v1'
    ).encode()).hexdigest()
    cache_path = output_dir / 'atlas_projections.npz'
    projections = None
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if str(cache['signature']) == signature:
                projections = {key: cache[key] for key in ['pca_xy', 'pca_variance', 'tsne_xy']}
    if projections is None:
        with threadpool_limits(limits=config.get('runtime', {}).get('threads', 1)):
            pca = PCA(2, random_state=config['random_seed']).fit(
                cube[:len(training['labels'])].reshape(-1, cube.shape[-1]))
            pca_xy = pca.transform(cube.reshape(-1, cube.shape[-1])).reshape(*cube.shape[:2], 2)
            print(f'Атлас: общее t-SNE для {len(months)} месяцев, {n} МО', flush=True)
            tsne_xy = TSNE(n_components=2, perplexity=min(30, len(months) * n - 1),
                           max_iter=600, init='pca', random_state=config['random_seed'], n_jobs=1
                           ).fit_transform(cube.reshape(-1, cube.shape[-1])).reshape(*cube.shape[:2], 2)
        projections = dict(pca_xy=np.round(pca_xy, 4),
                           pca_variance=np.round(100 * pca.explained_variance_ratio_, 1),
                           tsne_xy=np.round(tsne_xy, 3))
        np.savez_compressed(cache_path, signature=signature, **projections)
    return dict(months=months, labels_by_k=labels_by_k,
                graph_edges=[np.round(month, 4) for month in [*training_edges, *edges]],
                **projections)


def presentation_summary(results, config, output_dir):
    """Краткие результаты и методология для начала интерактивного лендинга."""
    k = results['selected_k']
    study = results['cluster_study']
    counts = sorted(results['study_runs'])
    check = study.query('k == @k and year == 2024').iloc[0]
    profiles = results['cluster_profiles'].query('k == @k and year == 2024')
    rows = []
    for row in profiles.itertuples():
        rows.append(f'<tr><td>{row.group + 1}</td><td>{row.size}</td>'
                    f'<td>{row.spending_rub:,.0f}</td><td>{row.wage_total_rub:,.0f}</td>'
                    f'<td>{escape(row.description)}</td></tr>')
    plot = base64.b64encode((output_dir / 'cluster_count_comparison.png').read_bytes()).decode('ascii')
    neighbors = config['graph']['neighbors']
    alpha = results['alpha']
    return f'''
    <section class="presentation" aria-labelledby="result-title">
      <h2 id="result-title">Экономические профили муниципалитетов</h2>
      <div class="summary-grid">
        <p><strong>{len(results['ids'])}</strong><span>МО в полной панели</span></p>
        <p><strong>{k}</strong><span>групп выбрано по 2023</span></p>
        <p><strong>{min(counts)}–{max(counts)}</strong><span>исследованный диапазон K</span></p>
        <p><strong>{check.sensitivity_ARI_min:.2f}</strong><span>минимальная ARI в 2024</span></p>
      </div>
      <p>Выбран наиболее подробный вариант, который проходит ограничения размеров,
      устойчивости и экономической различимости на 2023. На 2024 он
      {'проходит те же ограничения' if check.accepted else 'не проходит все ограничения; причины приведены в отчёте'}.</p>
      <div class="table-scroll"><table>
        <thead><tr><th>Группа</th><th>МО</th><th>Расходы, ₽</th><th>Зарплата, ₽</th><th>Экономическое описание</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table></div>
      <p class="small">Группы на декабрь 2024. Расходы — медиана среднегодовых безналичных расходов,
      зарплата — медиана муниципальных наблюдений. Названия описательные; отраслевые
      показатели охватывают обследуемые организации без МСП. Зарплаты работников не равны доходам жителей.</p>
      <h3>Как устроен расчёт</h3>
      <ol>
        <li>Шесть долей корзины переводятся в CLR, расходы — в логарифм; получаем семь признаков.</li>
        <li>Учитываем региональный уровень, масштабируем по 2023 и сглаживаем EWMA с α={alpha:g}.</li>
        <li>Ежемесячный граф: {neighbors} соседей, адаптивный вес exp(−d² / (σᵢσⱼ)),
        симметризация максимумом. Спектральные координаты кластеризуем KMeans.</li>
        <li>Согласуем номера групп между месяцами. Выбираем K на 2023;
        зарплата и отраслевая занятость 2024 служат отложенной экономической проверкой.</li>
      </ol>
      <h3>Где заканчивается допустимая детализация</h3>
      <img class="study-chart" src="data:image/png;base64,{plot}" alt="Сравнение разделимости, устойчивости и экономических различий при разных K">
      <p class="small">Красные линии — заданные пороги допуска. Более строгое требование
      устойчивости может уменьшить число допустимых групп. Проверки не устанавливают
      причинность или истинное число локальных экономических типов.</p>
      <nav aria-label="Материалы исследования">
        <a href="#explore">Исследовать карту и переходы</a>
        <a href="cluster_count_report.md">Все варианты и экономические профили</a>
      </nav>
    </section>'''


def write_atlas(results, graph_xy, edges, output_dir, config):
    """Сохранить atlas_data.json и автономный atlas.html без внешних библиотек."""
    panel = results['panel']
    geometry_path = config['data_dir'] / 'processed/municipal_geometry/teammate_data_map.json'
    views = temporal_views(results, edges, output_dir, config)
    atlas = dict(
        ids=panel['ids_all'], dynamic_ids=panel['ids'],
        names=panel['coverage']['name'].tolist(),
        regions=panel['coverage']['region_name'].tolist(),
        total=panel['total'], categories=panel['categories'], shares=panel['shares'],
        economic_labels=[
            'Начисленная зарплата 2024, ₽', 'Обследуемые работники на 1000 жителей',
            'Доля работников сельского/лесного хозяйства', 'Доля работников обработки',
            'Доля работников управления, образования и здоровья',
        ],
        economic_observed=results['economic_observed'].to_numpy(),
        geo=json.loads(geometry_path.read_text(encoding='utf-8')),
        graph_xy=graph_xy, comparison=results['comparison'].to_dict('records'),
        selected_k=results['selected_k'], cluster_study=results['cluster_study'].to_dict('records'),
        cluster_profiles=results['cluster_profiles'].query('year == 2024').to_dict('records'),
        **views,
    )
    data = json.dumps(clean_json(atlas), ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    (output_dir / 'atlas_data.json').write_text(data, encoding='utf-8')
    html = (TEMPLATES / 'atlas.html').read_text(encoding='utf-8')
    css = (TEMPLATES / 'atlas.css').read_text(encoding='utf-8')
    js = (TEMPLATES / 'atlas.js').read_text(encoding='utf-8')
    html = html.replace('{ATLAS_CSS}', css).replace('{ATLAS_JS}', js)
    html = html.replace('{SUMMARY}', presentation_summary(results, config, output_dir))
    html = html.replace('__DATA__', data.replace('<', r'\u003c'))
    (output_dir / 'atlas.html').write_text(html, encoding='utf-8')
