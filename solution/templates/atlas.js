const D=__DATA__;
const $ = id => document.getElementById(id);
const colors = ['#47876e', '#bf8051', '#667ba8', '#a685b0', '#b7aa53', '#658891', '#d06b73', '#8b9461', '#826d5f', '#a35eb0', '#5d92bb', '#d6a357', '#717477', '#da89ab', '#55a89c', '#b1ad77', '#8050bc', '#ca976b', '#467054', '#748cba'];
const names = {
    adaptive_temporal_spectral: 'Основной · EWMA spectral · локальный масштаб',
    adaptive_spectral: 'Spectral · локальный масштаб',
    kmeans: 'KMeans',
    spectral: 'Spectral · общий масштаб',
    temporal_spectral: 'EWMA spectral · общий масштаб',
    anchored_k2: 'Фиксированные прототипы · 2',
    anchored_k5: 'Фиксированные прототипы · 5'
};
names.adaptive_temporal_spectral += ` · K=${D.selected_k}`;
const positions = new Map(D.ids.map((id, index) => [id, index]));
const dynamic = new Map(D.dynamic_ids.map((id, index) => [id, index]));
const svgNamespace = 'http://www.w3.org/2000/svg';
let selected = null;
let timer = null;
let graphPositions = [];

D.months.forEach((month, index) => $('month').add(new Option(month, index)));
Object.keys(D.sequences).forEach(method => {
    $('method').add(new Option((D.method_names || {})[method] || names[method] || `${method.startsWith('anchored_') ? 'Фиксированные прототипы' : 'Основной'} · K=${method.match(/k(\d+)/)?.[1]}`, method));
});
$('method').value = D.default_method;

$('map').setAttribute(
    'viewBox',
    Array.isArray(D.geo.viewBox) ? D.geo.viewBox.join(' ') : D.geo.viewBox
);
D.geo.features.forEach(feature => {
    const path = document.createElementNS(svgNamespace, 'path');
    path.setAttribute('d', feature.d);
    path.setAttribute('fill-rule', D.geo.fillRule || 'evenodd');
    path.dataset.id = feature.id;
    path.addEventListener('click', () => {
        if (positions.has(feature.id)) choose(feature.id);
    });
    $('map').appendChild(path);
});

function label(id) {
    const row = dynamic.get(id);
    const method = $('method').value;
    const monthIndex = +$('month').value;
    if (row === undefined) return null;
    return D.sequences[method][monthIndex][row];
}

function groupName(method, group) {
    const custom = (D.group_names || {})[method];
    return custom && custom[group] !== undefined ? custom[group] : 'Группа ' + (group + 1);
}

function esc(value) {
    return String(value ?? '').replace(/[&<>"']/g, character => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    }[character]));
}

function fmt(value) {
    return value === null || value === undefined
        ? 'нет наблюдения'
        : Number(value).toLocaleString('ru-RU', {maximumFractionDigits: 2});
}

function table(rows) {
    return '<table>' + rows.map(row =>
        '<tr><td>' + esc(row[0]) + '</td><td>' + esc(row[1]) + '</td></tr>'
    ).join('') + '</table>';
}

function choose(id) {
    selected = id;
    render();
    card();
}

function card() {
    if (selected === null) return;
    const row = positions.get(selected);
    const dynamicRow = dynamic.get(selected);
    const monthIndex = +$('month').value;
    if (row === undefined) return;

    $('name').textContent = D.names[row];
    const group = label(selected);
    let html = '<p>' + esc(D.regions[row]) + ' · ID ' + selected + '</p>' + table([
        ['Месяц', D.months[monthIndex]],
        ['Группа выбранного метода', group === null ? 'нет назначения' : groupName($('method').value, group)],
        ['Средний безналичный расход', fmt(D.total[monthIndex][row]) + ' ₽']
    ]);

    html += '<h3>Наблюдаемая корзина</h3>' + table(D.categories.map((category, column) => [
        category,
        fmt(D.shares[monthIndex][row][column] === null
            ? null
            : 100 * D.shares[monthIndex][row][column]) + ' %'
    ]));

    if (dynamicRow !== undefined) {
        html += '<h3>Экономическая проверка · годовые данные 2024</h3>' + table(
            D.economic_labels.map((title, column) => [title, fmt(D.economic_observed[dynamicRow][column])])
        );
    }
    if (dynamicRow !== undefined && D.context_mo) {
        html += '<h3>Контекст · официальные данные и исследование</h3>' + table(
            D.context_labels.map((title, column) => {
                const value = D.context_mo[dynamicRow][column];
                return [title, typeof value === 'number' ? fmt(value) : (value ?? 'нет наблюдения')];
            })
        );
    }
    html += '<p class="small">Годовые показатели используются только для проверки групп. '
        + 'Отраслевые доли относятся к обследуемым работникам без МСП; '
        + 'доли показаны от 0 до 1. Зарплата работников не равна доходу жителей. '
        + 'Неизвестные значения не заменены нулём.</p>';
    $('details').innerHTML = html;
}

function drawGraph() {
    const canvas = $('graph');
    const bounds = canvas.getBoundingClientRect();
    const pixelRatio = window.devicePixelRatio || 1;
    canvas.width = bounds.width * pixelRatio;
    canvas.height = bounds.height * pixelRatio;
    const context = canvas.getContext('2d');
    context.scale(pixelRatio, pixelRatio);

    const xCoordinates = D.graph_xy.map(point => point[0]);
    const yCoordinates = D.graph_xy.map(point => point[1]);
    const minX = Math.min(...xCoordinates);
    const maxX = Math.max(...xCoordinates);
    const minY = Math.min(...yCoordinates);
    const maxY = Math.max(...yCoordinates);
    const scale = Math.min(
        (bounds.width - 40) / (maxX - minX),
        (bounds.height - 40) / (maxY - minY)
    );
    graphPositions = D.graph_xy.map(point => [
        20 + (point[0] - minX) * scale,
        20 + (point[1] - minY) * scale
    ]);

    context.strokeStyle = 'rgba(65,95,78,.08)';
    context.lineWidth = .5;
    context.beginPath();
    D.graph_edges[+$('month').value].forEach(([source, target, weight]) => {
        context.moveTo(...graphPositions[source]);
        context.lineTo(...graphPositions[target]);
    });
    context.stroke();

    graphPositions.forEach((point, row) => {
        const id = D.dynamic_ids[row];
        const group = label(id);
        context.fillStyle = group === null ? '#d5ded7' : colors[group % colors.length];
        context.beginPath();
        context.arc(...point, id === selected ? 5 : 2.3, 0, Math.PI * 2);
        context.fill();
    });
}

function render() {
    const method = $('method').value;
    const k = method === D.default_method ? D.selected_k : Number(method.match(/^adaptive_temporal_k(\d+)$/)?.[1]);
    const rows = D.cluster_study.filter(row => row.k === k);
    $('study').innerHTML = rows.length ? '<p>Отбор K по 2023: ' + (rows[0].accepted ? 'прошёл' : 'не прошёл')
        + '. Проверка 2024: ' + (rows[1].accepted ? 'прошёл' : 'не прошёл')
        + '. Описания ниже — декабрь 2024, годовая статистика.</p>'
        + table(D.cluster_profiles.filter(row => row.k === k).map(row => ['Группа ' + (row.group + 1), row.description])) : '';
    const counts = new Map();
    D.ids.forEach(id => {
        const group = label(id);
        if (group !== null) counts.set(group, (counts.get(group) || 0) + 1);
    });
    $('legend').textContent = [...counts]
        .sort((left, right) => left[0] - right[0])
        .map(([group, count]) => groupName(method, group) + ': ' + count + ' МО')
        .join(' · ') + ' · Серый: нет назначения';

    document.querySelectorAll('#map path').forEach(path => {
        const id = +path.dataset.id;
        const group = label(id);
        path.setAttribute('fill', group === null ? '#d5ded7' : colors[group % colors.length]);
        path.classList.toggle('selected', id === selected);
    });
    const graphVisible = $('mode').value === 'graph';
    $('map').style.display = graphVisible ? 'none' : '';
    $('graph').style.display = graphVisible ? '' : 'none';
    if (graphVisible) drawGraph();
    card();
}

$('graph').addEventListener('click', event => {
    const bounds = $('graph').getBoundingClientRect();
    const clickX = event.clientX - bounds.left;
    const clickY = event.clientY - bounds.top;
    let closestRow = -1;
    let squaredDistance = 144;
    graphPositions.forEach((point, row) => {
        const distance = (point[0] - clickX) ** 2 + (point[1] - clickY) ** 2;
        if (distance < squaredDistance) {
            squaredDistance = distance;
            closestRow = row;
        }
    });
    if (closestRow >= 0) choose(D.dynamic_ids[closestRow]);
});

['month', 'method', 'mode'].forEach(id => $(id).addEventListener('change', render));
window.addEventListener('resize', render);
$('search').addEventListener('input', () => {
    const query = $('search').value.toLocaleLowerCase().trim();
    $('searchResults').replaceChildren();
    if (!query) return;
    D.ids.map((id, index) => ({id, index}))
        .filter(({id, index}) =>
            String(id) === query || D.names[index].toLocaleLowerCase().includes(query)
        )
        .slice(0, 15)
        .forEach(({id, index}) => {
            const button = document.createElement('button');
            button.textContent = D.names[index] + ' · ' + D.regions[index];
            button.onclick = () => choose(id);
            $('searchResults').appendChild(button);
        });
});

$('play').onclick = () => {
    if (timer) {
        clearInterval(timer);
        timer = null;
        $('play').textContent = '▶ По месяцам';
    } else {
        timer = setInterval(() => {
            $('month').value = (+$('month').value + 1) % D.months.length;
            render();
        }, 900);
        $('play').textContent = '⏸ Остановить';
    }
};
render();
