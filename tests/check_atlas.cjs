// Browser integration check; pass the generated atlas path as the first argument.
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { createRequire } = require('node:module');
const requireRuntime = process.env.NODE_PATH ? createRequire(path.join(process.env.NODE_PATH, 'package.json')) : require;
const { chromium } = requireRuntime('playwright');

let browser;
(async () => {
    browser = await chromium.launch({headless: true, channel: process.env.ATLAS_BROWSER || 'msedge'});
    const page = await browser.newPage({viewport: {width: 1280, height: 1100}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const requests = [];
    page.on('request', request => requests.push(request.url()));
    await page.goto(pathToFileURL(path.resolve(process.argv[2] || 'notebook_results/atlas.html')).href);
    assert.equal(await page.locator('#month[type="range"]').count(), 1, 'a single timeline slider is required');
    assert.equal(await page.locator('#play, [data-action="play"], [data-action="previous"], [data-action="next"]').count(), 0);
    assert.equal(await page.locator('#k option').count(), 9);
    assert.equal(await page.locator('#month').getAttribute('max'), '23');
    async function month(value) {
        await page.locator('#month').evaluate((element, value) => {
            element.value = value;
            element.dispatchEvent(new Event('input', {bubbles: true}));
        }, value);
        await page.evaluate(() => new Promise(requestAnimationFrame));
    }
    const timings = [];
    for (let k = 2; k <= 10; k++) {
        await page.selectOption('#k', String(k));
        for (let m = 1; m < 24; m++) {
            await month(m);
            const matrix = await page.locator('#transitionMatrix td').allTextContents();
            assert.equal(matrix.reduce((sum, value) => sum + Number(value.replace(/\s/g, '')), 0), 1774);
            assert.equal(matrix.length, k * k);
        }
        const intervals = await page.locator('[data-interval-start]').evaluateAll(elements => elements.flatMap(element =>
            Array.from({length: Number(element.dataset.intervalEnd) - Number(element.dataset.intervalStart)}, (_, i) => Number(element.dataset.intervalStart) + i)));
        assert.deepEqual(intervals, Array.from({length: 23}, (_, i) => i));
    }
    await page.selectOption('#k', '3');
    await month(12);
    assert.equal(await page.locator('#transitionPeriod').textContent(), 'Декабрь 2023 → Январь 2024');
    const pictures = await page.locator('#pca, #tsne').evaluateAll(canvases => canvases.map(canvas => canvas.toDataURL()));
    await month(12.25);
    assert.equal(await page.locator('#transitionPeriod').textContent(), 'Январь 2024 → Февраль 2024');
    const moved = await page.locator('#pca, #tsne').evaluateAll(canvases => canvases.map(canvas => canvas.toDataURL()));
    assert(pictures.every((picture, i) => picture !== moved[i]), 'both projections must move between months');
    await page.locator('#search').fill('Усть-Коксинский');
    await page.locator('#searchResults button').first().click();
    await page.waitForFunction(() => document.getElementById('name').textContent.length > 0);
    const name = await page.locator('#name').textContent();
    await page.selectOption('#k', '5');
    await month(23);
    assert.equal(await page.locator('#name').textContent(), name);
    assert.match(await page.locator('#details').textContent(), /21/);
    await page.selectOption('#mode', 'map');
    await page.locator('#map').waitFor({state: 'visible'});
    assert(await page.locator('#map').isVisible());
    await month(0);
    assert.match(await page.locator('#details').textContent(), /2023-01/);
    await page.selectOption('#mode', 'plots');
    await page.locator('#plots').waitFor({state: 'visible'});
    await page.locator('#closeDetails').click();
    for (const width of [1280, 360, 320]) {
        await page.setViewportSize({width, height: 1100});
        await page.evaluate(() => new Promise(requestAnimationFrame));
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `overflow at ${width}px`);
        const labelsFit = await page.locator('#flows svg text').evaluateAll(elements => elements.every(element => {
            const box = element.getBBox(), width = element.ownerSVGElement.viewBox.baseVal.width;
            return box.x >= 0 && box.x + box.width <= width + 1;
        }));
        assert(labelsFit, `flow labels clipped at ${width}px`);
    }
    await page.setViewportSize({width: 1280, height: 1100});
    await page.emulateMedia({reducedMotion: 'reduce'});
    await month(5);
    assert.equal(await page.locator('#period').textContent(), 'Июнь 2023');
    await page.locator('#month').focus();
    await page.keyboard.press('ArrowRight');
    assert(Number(await page.locator('#month').inputValue()) > 5);
    const rapid = await page.locator('#month').evaluate(async element => {
        for (const value of [2, 20, 8, 22, 4]) {
            element.value = value;
            element.dispatchEvent(new Event('input', {bubbles: true}));
        }
        await new Promise(requestAnimationFrame);
        return element.value;
    });
    assert.equal(Number(rapid), 4);
    await page.emulateMedia({reducedMotion: 'no-preference'});
    for (let i = 0; i < 20; i++) {
        timings.push(await page.locator('#month').evaluate(async (element, value) => {
            const start = performance.now(); element.value = value;
            element.dispatchEvent(new Event('input', {bubbles: true}));
            await new Promise(requestAnimationFrame);
            return performance.now() - start;
        }, i + .35));
    }
    await page.screenshot({path: 'notebook_results/atlas-desktop.png', fullPage: true});
    await page.setViewportSize({width: 360, height: 3000});
    await page.screenshot({path: 'notebook_results/atlas-mobile.png', fullPage: true});
    assert.deepEqual(errors, []);
    assert(requests.every(url => url.startsWith('file:') || url.startsWith('data:')), 'atlas must work offline');
    console.log(`PASS: K=2–10, all 23 intervals, 24-month slider, shared projections, selected MO, map, reduced motion, keyboard, narrow widths; scrub mean ${(timings.reduce((a,b)=>a+b,0)/timings.length).toFixed(1)}ms, max ${Math.max(...timings).toFixed(1)}ms.`);
    await browser.close();
})().catch(async error => {console.error(error); await browser?.close(); process.exitCode = 1;});
