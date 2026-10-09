"""Интерактивная презентация с картой, временными группами и экономическими профилями."""
import base64
import json

import pandas as pd
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


ATLAS_RESEARCH = ['research_kmeans_v5', 'research_anchored_real', 'research_archetypes',
                  'research_dgi', 'research_dmon', 'research_evolvegcn', 'research_gconvgru']


def context_research(results):
    """Метки исследовательских методов для переключателя карты (если контекст рассчитан)."""
    context = results.get('context')
    if context is None:
        return {}
    return {name: context['research'][name] for name in ATLAS_RESEARCH if name in context['research']}


def context_layers(results, context):
    """Слои контекстного блока: методы исследования на карте и внешние показатели в карточке МО."""
    from .context import RESEARCH_NAMES
    research = {name: context['research'][name] for name in ATLAS_RESEARCH if name in context['research']}
    national = context['national']
    group_names = {}
    for name in ['research_kmeans_v5', 'research_anchored_real']:
        if name in research:
            frame = pd.DataFrame(dict(label=research[name][-1], type=national.national_type.to_numpy()))
            mapping = frame.dropna().groupby('label').type.agg(lambda values: values.mode().iat[0])
            group_names[name] = {int(label): title for label, title in mapping.items()}
    per_mo = context['per_mo']
    columns = ['Население 2024, тыс.', 'Реальный рост трат (ИПЦ региона), %', 'Ускорение во II пол. 2024, п.п.',
               'Индекс доступности рынков', 'Индекс мобильности 2024, км (СЗФО)', 'Ночные огни VIIRS на жителя (log)']
    rows = []
    for position, territory in enumerate(results['ids']):
        boom = per_mo['Оборонно-промышленный рост зарплат (БДМО)'].iat[position]
        rows.append([
            national.national_type.iat[position], national.within_region_type.iat[position],
            per_mo['export_sector'].fillna('нет').iat[position],
            None if pd.isna(boom) else ('да' if boom else 'нет'),
            *[per_mo[column].iat[position] for column in columns],
        ])
    return dict(
        method_names=RESEARCH_NAMES, group_names=group_names,
        context_labels=['Национальный тип (исследование)', 'Положение внутри региона (исследование)',
                        'Экспортная отрасль', 'Оборонно-промышленный рост зарплат', *columns],
        context_mo=rows,
    )


def write_atlas(results, graph_xy, edges, output_dir, config):
    """Сохранить atlas_data.json и автономный atlas.html без внешних библиотек."""
    panel = results['panel']
    geometry_path = config['data_dir'] / 'processed/municipal_geometry/teammate_data_map.json'
    atlas = dict(
        ids=panel['ids_all'], dynamic_ids=panel['ids'], months=results['months'],
        default_method=METHOD, names=panel['coverage']['name'].tolist(),
        regions=panel['coverage']['region_name'].tolist(), sequences={**results['sequences'], **(context_research(results))},
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
    context = results.get('context')
    if context is not None:
        atlas.update(context_layers(results, context))
    data = json.dumps(clean_json(atlas), ensure_ascii=False, allow_nan=False)
    (output_dir / 'atlas_data.json').write_text(data, encoding='utf-8')
    html = (TEMPLATES / 'atlas.html').read_text(encoding='utf-8')
    css = (TEMPLATES / 'atlas.css').read_text(encoding='utf-8')
    js = (TEMPLATES / 'atlas.js').read_text(encoding='utf-8')
    html = html.replace('{ATLAS_CSS}', css).replace('{ATLAS_JS}', js)
    html = html.replace('{SUMMARY}', presentation_summary(results, config, output_dir))
    html = html.replace('{CONTEXT}', context['html'] if context is not None else '')
    html = html.replace('__DATA__', data.replace('<', r'\u003c'))
    (output_dir / 'atlas.html').write_text(html, encoding='utf-8')
