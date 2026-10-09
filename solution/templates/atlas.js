const D = __DATA__;
const $ = id => document.getElementById(id);
const svgNamespace = 'http://www.w3.org/2000/svg';
const positions = new Map(D.ids.map((id, index) => [id, index]));
const dynamic = new Map(D.dynamic_ids.map((id, index) => [id, index]));
const monthNames = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь'];
const shortMonths = ['янв', 'фев', 'мар', 'апр', 'май', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек'];
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const groupingCache = new Map();
const canvasPositions = new Map();
let selected = null;
let frame = null;
let discreteKey = '';
let flowKey = '';
let colors, neutral;

function esc(value) {
    return String(value ?? '').replace(/[&<>"']/g, character => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[character]));
}
function fmt(value) {
    return value === null || value === undefined ? 'нет наблюдения'
        : Number(value).toLocaleString('ru-RU', {maximumFractionDigits: 2});
}
function date(index) {
    return monthNames[Number(D.months[index].slice(5)) - 1] + ' ' + D.months[index].slice(0, 4);
}
function shortDate(index) {
    return shortMonths[Number(D.months[index].slice(5)) - 1] + ' ' + D.months[index].slice(2, 4);
}
function groupColor(group) { return group === null ? neutral.empty : colors[group % colors.length]; }
function badge(group) {
    return `<span class="group-label"><i class="dot" style="--group:${groupColor(group)}" aria-hidden="true"></i>Группа ${group + 1}</span>`;
}
function resolveColors() {
    const probe = document.createElement('span');
    document.body.appendChild(probe);
    const resolve = variable => { probe.style.color = `var(${variable})`; return getComputedStyle(probe).color; };
    colors = Array.from({length: 20}, (_, i) => resolve(`--group-${i + 1}`));
    neutral = Object.fromEntries(['ink', 'quiet', 'line', 'paper', 'empty'].map(key => [key, resolve(`--${key}`)]));
    probe.remove();
}
resolveColors();

Object.keys(D.labels_by_k).map(Number).sort((a, b) => a - b).forEach(k => {
    $('k').add(new Option(String(k) + (k === D.selected_k ? ' · выбрано моделью' : ''), k));
});
$('k').value = D.selected_k;
$('month').max = D.months.length - 1;
$('coverage').textContent = `${D.months.length} месяца · ${fmt(D.dynamic_ids.length)} МО`;
$('flowsDescription').textContent = `Все ${D.months.length - 1} соседних интервала · ${date(0)} — ${date(D.months.length - 1)}`;
$('timelineTicks').innerHTML = [...new Set([0, 6, 12, D.months.length - 1])].filter(i => i < D.months.length)
    .map(i => `<span>${shortDate(i)}</span>`).join('');
$('pcaVariance').textContent = `PC1: ${fmt(D.pca_variance[0])}% · PC2: ${fmt(D.pca_variance[1])}% дисперсии`;

const mapPaths = D.geo.features.map(feature => {
    const path = document.createElementNS(svgNamespace, 'path');
    path.setAttribute('d', feature.d);
    path.setAttribute('fill-rule', D.geo.fillRule || 'evenodd');
    path.dataset.id = feature.id;
    const title = document.createElementNS(svgNamespace, 'title');
    title.textContent = D.names[positions.get(feature.id)] || String(feature.id);
    path.appendChild(title);
    path.addEventListener('click', () => { if (positions.has(feature.id)) choose(feature.id); });
    $('map').appendChild(path);
    return path;
});
$('map').setAttribute('viewBox', Array.isArray(D.geo.viewBox) ? D.geo.viewBox.join(' ') : D.geo.viewBox);

function grouping(k) {
    if (groupingCache.has(k)) return groupingCache.get(k);
    const labels = D.labels_by_k[k];
    // Labels can introduce new IDs when a group disappears; keep the saved identities.
    const ids = [...new Set(labels.flat())].sort((a, b) => a - b);
    const size = Math.max(...ids) + 1;
    const counts = labels.map(month => {
        const row = Array(size).fill(0);
        month.forEach(group => row[group]++);
        return row;
    });
    const matrices = labels.slice(1).map((month, index) => {
        const matrix = Array.from({length: size}, () => Array(size).fill(0));
        month.forEach((group, row) => matrix[labels[index][row]][group]++);
        return matrix;
    });
    const flows = matrices.map(matrix => ids.flatMap(source => ids
        .filter(target => matrix[source][target] > 0)
        .map(target => [source, target, matrix[source][target]])));
    const value = {labels, ids, counts, matrices, flows};
    groupingCache.set(k, value);
    return value;
}

const domains = {};
for (const [id, frames] of Object.entries({graph: [D.graph_xy], pca: D.pca_xy, tsne: D.tsne_xy})) {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    frames.forEach(points => points.forEach(([x, y]) => {
        minX = Math.min(minX, x); maxX = Math.max(maxX, x);
        minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }));
    const padX = (maxX - minX) * .06 || 1, padY = (maxY - minY) * .06 || 1;
    domains[id] = [minX - padX, maxX + padX, minY - padY, maxY + padY];
}

function drawCanvas(id, position, k) {
    const canvas = $(id), width = canvas.clientWidth, height = canvas.clientHeight;
    if (!width || !height) return;
    const pixelRatio = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * pixelRatio) || canvas.height !== Math.round(height * pixelRatio)) {
        canvas.width = Math.round(width * pixelRatio); canvas.height = Math.round(height * pixelRatio);
    }
    const context = canvas.getContext('2d');
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, width, height);
    const previous = Math.floor(position), next = Math.min(previous + 1, D.months.length - 1);
    const fraction = position - previous, current = Math.round(position);
    const labels = grouping(k).labels;
    const pad = id === 'graph' ? 12 : 44;
    const [minX, maxX, minY, maxY] = domains[id];
    const scaleX = (width - pad * 2) / (maxX - minX), scaleY = (height - pad * 2) / (maxY - minY);
    const scale = Math.min(scaleX, scaleY);
    const from = id === 'graph' ? D.graph_xy : D[id + '_xy'][previous];
    const to = id === 'graph' ? from : D[id + '_xy'][next];
    const points = from.map(([x, y], row) => {
        x += (to[row][0] - x) * fraction; y += (to[row][1] - y) * fraction;
        return id === 'graph'
            ? [width / 2 + (x - (minX + maxX) / 2) * scale, height / 2 - (y - (minY + maxY) / 2) * scale]
            : [pad + (x - minX) * scaleX, height - pad - (y - minY) * scaleY];
    });
    canvasPositions.set(id, points);
    if (id === 'graph') {
        context.strokeStyle = neutral.quiet; context.lineWidth = .55;
        [[previous, 1 - fraction], [next, fraction]].forEach(([month, opacity]) => {
            if (!opacity) return;
            context.globalAlpha = .075 * opacity; context.beginPath();
            D.graph_edges[month].forEach(([source, target]) => {
                context.moveTo(...points[source]); context.lineTo(...points[target]);
            });
            context.stroke();
        });
        const row = dynamic.get(selected);
        if (row !== undefined) {
            context.globalAlpha = .8; context.lineWidth = 1.2;
            D.graph_edges[current].forEach(([source, target, weight]) => {
                if (source !== row && target !== row) return;
                context.lineWidth = .6 + weight * 1.8;
                context.beginPath(); context.moveTo(...points[source]); context.lineTo(...points[target]); context.stroke();
            });
        }
    } else {
        context.globalAlpha = 1; context.strokeStyle = neutral.line; context.lineWidth = 1;
        context.strokeRect(pad, pad, width - pad * 2, height - pad * 2);
        context.fillStyle = neutral.quiet; context.font = '11px system-ui';
        for (let i = 0; i < 3; i++) {
            const ratio = i / 2, x = pad + ratio * (width - pad * 2), y = height - pad - ratio * (height - pad * 2);
            context.textAlign = i === 0 ? 'left' : i === 2 ? 'right' : 'center';
            context.fillText((minX + ratio * (maxX - minX)).toFixed(0), x, height - pad + 16);
            context.textAlign = 'right'; context.fillText((minY + ratio * (maxY - minY)).toFixed(0), pad - 5, y + 4);
        }
        context.textAlign = 'center'; context.fillText(id === 'pca' ? 'PC1' : 't-SNE 1', width / 2, height - 6);
        context.save(); context.translate(11, height / 2); context.rotate(-Math.PI / 2);
        context.fillText(id === 'pca' ? 'PC2' : 't-SNE 2', 0, 0); context.restore();
    }
    const rgb = colors.map(color => color.match(/[\d.]+/g).slice(0, 3).map(Number));
    context.globalAlpha = .78;
    points.forEach((point, row) => {
        const a = labels[previous][row], b = labels[next][row];
        context.fillStyle = a === b ? groupColor(a) : `rgb(${rgb[a % colors.length].map((v, i) => Math.round(v + (rgb[b % colors.length][i] - v) * fraction)).join(',')})`;
        context.beginPath(); context.arc(...point, D.dynamic_ids[row] === selected ? 4.8 : 1.9, 0, Math.PI * 2); context.fill();
    });
    context.globalAlpha = .45; context.lineWidth = .7;
    const changedFrom = fraction > 0 ? previous : Math.max(0, current - 1);
    const changedTo = fraction > 0 ? next : current;
    if (changedTo > changedFrom) points.forEach((point, row) => {
        if (labels[changedFrom][row] !== labels[changedTo][row]) {
            context.strokeStyle = groupColor(labels[changedTo][row]);
            context.beginPath(); context.arc(...point, 3.3, 0, Math.PI * 2); context.stroke();
        }
    });
    const row = dynamic.get(selected);
    if (row !== undefined) { context.globalAlpha = 1; context.strokeStyle = neutral.ink; context.lineWidth = 1.8; context.beginPath(); context.arc(...points[row], 6, 0, Math.PI * 2); context.stroke(); }
    context.globalAlpha = 1;
}

function svgElement(tag, attributes, parent) {
    const element = document.createElementNS(svgNamespace, tag);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    parent.appendChild(element);
    return element;
}
function drawFlows(k) {
    const width = $('flows').clientWidth;
    const key = `${k}:${width}`;
    if (!width || key === flowKey) return;
    flowKey = key; discreteKey = ''; $('flows').replaceChildren();
    const {ids, counts, flows} = grouping(k);
    const columns = Math.min(13, Math.max(4, Math.floor(width / 60)));
    const height = Math.max(210, ids.length * 26 + 52), top = 34, gap = 12;
    const scale = (height - top - 24 - (ids.length - 1) * gap) / D.dynamic_ids.length;
    let start = 0;
    while (start < D.months.length - 1) {
        const end = Math.min(D.months.length, start + columns);
        const step = (width - 56) / (end - start - 1);
        const svg = svgElement('svg', {viewBox: `0 0 ${width} ${height}`, height, role: 'img', 'aria-label': `Переходы ${date(start)} — ${date(end - 1)}`}, $('flows'));
        svgElement('rect', {x: 0, y: 26, width: step, height: height - 26, fill: colors[0], opacity: 0, rx: 4,
            'data-interval-start': start, 'data-interval-end': end - 1}, svg);
        const nodes = [];
        for (let month = start; month < end; month++) {
            let y = top;
            nodes.push(new Map(ids.map(group => {
                const node = {y, h: counts[month][group] * scale, out: y, into: y};
                y += node.h + gap; return [group, node];
            })));
        }
        for (let month = start; month < end - 1; month++) {
            const x0 = 28 + (month - start) * step + 4, x1 = 28 + (month + 1 - start) * step - 4;
            flows[month].forEach(([source, target, count]) => {
                const from = nodes[month - start].get(source), to = nodes[month + 1 - start].get(target);
                const y0 = from.out, y1 = to.into, h = count * scale;
                from.out += h; to.into += h;
                const path = svgElement('path', {d: `M${x0} ${y0} C${x0 + step * .4} ${y0} ${x1 - step * .4} ${y1} ${x1} ${y1} L${x1} ${y1 + h} C${x1 - step * .4} ${y1 + h} ${x0 + step * .4} ${y0 + h} ${x0} ${y0 + h} Z`,
                    fill: groupColor(source), opacity: source === target ? .14 : .74}, svg);
                svgElement('title', {}, path).textContent = `${shortDate(month)} → ${shortDate(month + 1)} · группа ${source + 1} → ${target + 1}: ${fmt(count)} МО`;
            });
        }
        for (let month = start; month < end; month++) {
            const x = 28 + (month - start) * step;
            svgElement('text', {x, y: 17, 'text-anchor': 'middle'}, svg).textContent = shortDate(month);
            ids.forEach(group => {
                const node = nodes[month - start].get(group);
                const rect = svgElement('rect', {x: x - 4, y: node.y, width: 8, height: node.h, rx: 1.5, fill: groupColor(group)}, svg);
                svgElement('title', {}, rect).textContent = `${date(month)} · группа ${group + 1}: ${fmt(counts[month][group])} МО`;
            });
        }
        start = end - 1;
    }
}

function transitionDetails(k, interval) {
    const {ids, counts, flows, matrices} = grouping(k);
    const sources = ids.filter(group => counts[interval][group] > 0);
    const targets = ids.filter(group => counts[interval + 1][group] > 0);
    const title = `${date(interval)} → ${date(interval + 1)}`;
    $('transitionPeriod').textContent = title;
    $('transitionMatrix').setAttribute('aria-label', title);
    const changed = flows[interval].reduce((sum, [source, target, count]) => sum + (source === target ? 0 : count), 0);
    $('switchCount').textContent = `Сменили группу: ${fmt(changed)} МО · ${fmt(100 * changed / D.dynamic_ids.length)}%`;
    $('transitionMatrix').querySelector('thead').innerHTML = '<tr><th scope="col">Из / в</th>' + targets.map(group =>
        `<th scope="col" style="--group:${groupColor(group)}">${badge(group)}</th>`).join('') + '</tr>';
    $('transitionMatrix').querySelector('tbody').innerHTML = sources.map(source =>
        `<tr><th scope="row" style="--group:${groupColor(source)}">${badge(source)}</th>` + targets.map(target =>
            `<td${source === target ? ' class="stayed"' : ''}>${fmt(matrices[interval][source][target])}</td>`).join('') + '</tr>').join('');
    document.querySelectorAll('[data-interval-start]').forEach(band => {
        const start = Number(band.dataset.intervalStart), end = Number(band.dataset.intervalEnd);
        const width = band.ownerSVGElement.viewBox.baseVal.width, step = (width - 56) / (end - start);
        band.setAttribute('x', 28 + (interval - start) * step); band.setAttribute('width', step);
        band.setAttribute('opacity', interval >= start && interval < end ? .06 : 0);
    });
}

function table(rows) {
    return '<table>' + rows.map(([key, value]) => `<tr><td>${esc(key)}</td><td>${esc(value)}</td></tr>`).join('') + '</table>';
}
function card(k, month) {
    if (selected === null) return;
    const row = positions.get(selected), dynamicRow = dynamic.get(selected);
    $('name').textContent = D.names[row];
    const group = dynamicRow === undefined ? null : grouping(k).labels[month][dynamicRow];
    let html = `<p>${esc(D.regions[row])} · ID ${selected}</p><p>${group === null ? 'Нет назначения в группу' : badge(group)}</p>`;
    const rows = [['Месяц', D.months[month]], ['Средний безналичный расход', fmt(D.total[month][row]) + ' ₽']];
    if (dynamicRow !== undefined) rows.push(['Связи узла в этом месяце', D.graph_edges[month].filter(([s, t]) => s === dynamicRow || t === dynamicRow).length]);
    html += table(rows) + '<h3>Наблюдаемая корзина</h3>' + table(D.categories.map((category, column) => [category,
        D.shares[month][row][column] === null ? 'нет наблюдения' : fmt(100 * D.shares[month][row][column]) + ' %']));
    if (dynamicRow !== undefined) {
        html += '<h3>Экономическая проверка · годовые данные 2024</h3>' + table(D.economic_labels.map((title, column) => [title, fmt(D.economic_observed[dynamicRow][column])]));
        if (D.context_mo) html += '<h3>Контекст · официальные данные и исследование</h3>' + table(D.context_labels.map((title, column) => {
            const value = D.context_mo[dynamicRow][column];
            return [title, typeof value === 'number' ? fmt(value) : (value ?? 'нет наблюдения')];
        }));
        html += '<p class="small">Годовые данные 2024 служат проверкой групп и не меняются при выборе месяца. Отраслевые доли относятся к обследуемым работникам без МСП; значения от 0 до 1. Зарплата работников не равна доходам жителей.</p>';
    }
    $('details').innerHTML = html;
}
function choose(id) {
    selected = id;
    $('municipalityCard').hidden = false; $('exploreLayout').classList.add('has-selection');
    $('searchResults').replaceChildren(); discreteKey = ''; scheduleRender();
}

function render() {
    frame = null;
    const value = Number($('month').value), position = reducedMotion.matches ? Math.round(value) : value;
    const month = Math.round(position), k = Number($('k').value), fraction = position - Math.floor(position);
    const period = fraction > .02 && fraction < .98 ? `${date(Math.floor(position))} → ${date(Math.ceil(position))}` : date(month);
    if ($('period').textContent !== period) $('period').textContent = period;
    const mapVisible = $('mode').value === 'map';
    $('map').toggleAttribute('hidden', !mapVisible); $('plots').hidden = mapVisible;
    drawFlows(k);
    if (!mapVisible) ['graph', 'pca', 'tsne'].forEach(id => drawCanvas(id, position, k));
    const interval = fraction > 0 ? Math.floor(position) : Math.max(0, month - 1);
    const key = `${k}:${month}:${interval}:${selected}:${mapVisible}`;
    if (key === discreteKey) return;
    discreteKey = key;
    const {ids, counts, labels} = grouping(k);
    $('legend').innerHTML = ids.filter(group => counts[month][group]).map(group =>
        `<span>${badge(group)} · ${fmt(counts[month][group])}</span>`).join('') + '<span><i class="ring" aria-hidden="true"></i>Смена группы</span>'
        + (mapVisible ? '<span><i class="dot" style="--group:var(--empty)" aria-hidden="true"></i>Нет назначения</span>' : '');
    const statuses = D.cluster_study.filter(row => row.k === k);
    const status = year => statuses.find(row => row.year === year)?.accepted ? 'пройден' : 'не пройден';
    $('study').textContent = `K=${k} · отбор по 2023: ${status(2023)} · проверка 2024: ${status(2024)}. Номера групп сохраняются во времени; состав групп может меняться.`;
    if (mapVisible) mapPaths.forEach(path => {
        const row = dynamic.get(Number(path.dataset.id));
        path.setAttribute('fill', row === undefined ? neutral.empty : groupColor(labels[month][row]));
        path.classList.toggle('selected', Number(path.dataset.id) === selected);
    });
    transitionDetails(k, interval); card(k, month);
}
function scheduleRender() { if (frame === null) frame = requestAnimationFrame(render); }

['graph', 'pca', 'tsne'].forEach(id => $(id).addEventListener('click', event => {
    const rect = $(id).getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
    let closest = -1, squaredDistance = 14 * 14;
    (canvasPositions.get(id) || []).forEach((point, row) => {
        const distance = (point[0] - x) ** 2 + (point[1] - y) ** 2;
        if (distance < squaredDistance) { closest = row; squaredDistance = distance; }
    });
    if (closest >= 0) choose(D.dynamic_ids[closest]);
}));
$('month').addEventListener('input', scheduleRender);
$('month').addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    $('month').value = Math.max(0, Math.min(D.months.length - 1, Math.round(Number($('month').value)) + (event.key === 'ArrowRight' ? 1 : -1)));
    scheduleRender();
});
['k', 'mode'].forEach(id => $(id).addEventListener('change', scheduleRender));
$('closeDetails').addEventListener('click', () => {
    selected = null; $('municipalityCard').hidden = true; $('exploreLayout').classList.remove('has-selection');
    discreteKey = ''; scheduleRender(); $('search').focus();
});
$('search').addEventListener('input', () => {
    const query = $('search').value.toLocaleLowerCase().trim();
    $('searchResults').replaceChildren();
    if (!query) return;
    D.ids.map((id, index) => ({id, index})).filter(({id, index}) => String(id) === query || D.names[index].toLocaleLowerCase().includes(query))
        .slice(0, 15).forEach(({id, index}) => {
            const button = document.createElement('button'); button.type = 'button';
            button.textContent = D.names[index] + ' · ' + D.regions[index];
            button.addEventListener('click', () => choose(id)); $('searchResults').appendChild(button);
        });
});
new ResizeObserver(() => { flowKey = ''; discreteKey = ''; scheduleRender(); }).observe($('exploreLayout'));
window.addEventListener('resize', () => { flowKey = ''; discreteKey = ''; scheduleRender(); });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    resolveColors(); flowKey = ''; discreteKey = ''; scheduleRender();
});
reducedMotion.addEventListener('change', scheduleRender);
render();
