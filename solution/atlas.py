"""Интерактивная презентация с картой, временными группами и экономическими профилями."""
import base64
import json
from html import escape
from pathlib import Path

from . import METHOD
from .reporting import clean_json

TEMPLATES = Path(__file__).parent / 'templates'


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
    atlas = dict(
        ids=panel['ids_all'], dynamic_ids=panel['ids'], months=results['months'],
        default_method=METHOD, names=panel['coverage']['name'].tolist(),
        regions=panel['coverage']['region_name'].tolist(), sequences=results['sequences'],
        total=panel['total'][12:], categories=panel['categories'], shares=panel['shares'][12:],
        economic_labels=[
            'Начисленная зарплата 2024, ₽', 'Обследуемые работники на 1000 жителей',
            'Доля работников сельского/лесного хозяйства', 'Доля работников обработки',
            'Доля работников управления, образования и здоровья',
        ],
        economic_observed=results['economic_observed'].to_numpy(),
        geo=json.loads(geometry_path.read_text(encoding='utf-8')),
        graph_xy=graph_xy, graph_edges=edges, comparison=results['comparison'].to_dict('records'),
        selected_k=results['selected_k'], cluster_study=results['cluster_study'].to_dict('records'),
        cluster_profiles=results['cluster_profiles'].query('year == 2024').to_dict('records'),
    )
    data = json.dumps(clean_json(atlas), ensure_ascii=False, allow_nan=False)
    (output_dir / 'atlas_data.json').write_text(data, encoding='utf-8')
    html = (TEMPLATES / 'atlas.html').read_text(encoding='utf-8')
    css = (TEMPLATES / 'atlas.css').read_text(encoding='utf-8')
    js = (TEMPLATES / 'atlas.js').read_text(encoding='utf-8')
    html = html.replace('{ATLAS_CSS}', css).replace('{ATLAS_JS}', js)
    html = html.replace('{SUMMARY}', presentation_summary(results, config, output_dir))
    html = html.replace('__DATA__', data.replace('<', r'\u003c'))
    (output_dir / 'atlas.html').write_text(html, encoding='utf-8')
