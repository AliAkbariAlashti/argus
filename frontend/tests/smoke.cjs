// Run against an isolated local Argus instance. AI output is intercepted only
// inside this test; no fixture answers or provider keys are saved by the app.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');

(async () => {
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}),
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const base = process.env.ARGUS_URL || 'http://127.0.0.1:8090';
    await page.goto(base);
    await page.waitForSelector('.fleet-tile');
    await page.locator('.fleet-tile').first().click();
    await page.waitForSelector('#viewer-stream.visible');
    const sources = await (await page.request.get(`${base}/api/cameras`)).json();
    const frame = await (await page.request.get(`${base}/api/cameras/${sources[0].id}/snapshot`)).body();
    const snapshot = `data:image/jpeg;base64,${frame.toString('base64')}`;
    const event = (type, value) => `event: ${type}\ndata: ${JSON.stringify(value)}\n\n`;
    await page.route('**/api/chat/stream', route => route.fulfill({
      contentType: 'text/event-stream',
      body: event('status', { text: 'Test fixture' })
        + event('token', { text: 'Fixture answer: ' })
        + event('token', { text: 'camera frame received.' })
        + event('answer', { answer: 'Fixture answer: camera frame received.', cameras_used: [sources[0].id], snapshot, elapsed_seconds: 1 })
        + event('done', {}),
    }));
    await page.locator('#chat-input').fill('Describe this test frame');
    await page.locator('#chat-send').click();
    await page.getByText('Answer complete · 1s · Evidence attached').waitFor();
    await page.locator('.msg-snapshot').last().click();
    assert(await page.locator('#evidence-dialog').isVisible());
    await page.keyboard.press('Escape');
    assert(!await page.locator('#evidence-dialog').isVisible());
    await page.route('**/api/chat/stream', route => route.fulfill({ status: 429, contentType: 'application/json', body: JSON.stringify({ detail: 'An answer is already in progress. Please wait.' }) }));
    await page.locator('#chat-input').fill('Retry this question');
    await page.locator('#chat-send').click();
    await page.getByText('Answer unavailable. Your question is ready to retry.').waitFor();
    assert.equal(await page.locator('#chat-input').inputValue(), 'Retry this question');
    await page.locator('.rail-btn[data-view="alerts"]').click();
    await page.locator('#events-live-state').waitFor();
    await page.locator('#events-live-toggle').click();
    assert.match(await page.locator('#events-live-state').innerText(), /PAUSED/);
    await page.locator('#events-live-toggle').click();
    assert.match(await page.locator('#events-live-state').innerText(), /LIVE/);
    await page.locator('.rail-btn[data-view="cpu"]').click();
    await page.locator('#cpu-health').waitFor();
    assert.match(await page.locator('#cpu-health').innerText(), /OpenCV|Health unavailable/);
    for (const view of ['rules', 'directory', 'health', 'runtime']) {
      await page.locator(`.rail-btn[data-view="${view}"]`).click();
      assert(await page.locator(`#view-${view}`).isVisible());
    }
    await page.setViewportSize({ width: 390, height: 844 });
    for (const view of ['dashboard', 'chat', 'runtime', 'alerts', 'directory']) {
      await page.locator(`.rail-btn[data-view="${view}"]`).click();
      assert(!await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), `${view} overflows horizontally`);
    }
    assert.deepEqual(errors, []);
    console.log('Browser checks passed: sources, streamed answer, evidence dialog, retry, navigation, mobile layout.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
