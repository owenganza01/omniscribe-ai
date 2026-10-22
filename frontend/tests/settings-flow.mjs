// Browser test for the Settings panel and its localStorage persistence in frontend/index.html.
//
// Uses a MOCKED /api/process-meeting route, so it never reaches the backend or Groq.
// Uses Chromium's fake microphone, so no real microphone or speech is needed.
//
// Run from the project root (requires Node 20+ and Microsoft Edge or Chrome installed):
//   npm install --no-save playwright-core
//   node frontend/tests/settings-flow.mjs
//
// Optional environment variables (same as results-flow.mjs):
//   BROWSER_PATH     path to msedge.exe or chrome.exe (defaults to Edge on Windows)
//   PLAYWRIGHT_CORE  absolute path to a playwright-core install, if not in node_modules
//   FRONTEND_INDEX   path to a different index.html to test (defaults to frontend/index.html)

import { createRequire } from 'node:module';
import path from 'node:path';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');

const PAGE_URL = pathToFileURL(path.resolve(process.env.FRONTEND_INDEX || 'frontend/index.html')).href;
const BROWSER_PATH = process.env.BROWSER_PATH || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const KEY = 'omniscribe.settings';

let posts = [];
let completed = 0;

function successBody() {
  return {
    success: true,
    meeting_name: 'Untitled meeting',
    transcript_segment: 'Settings test transcript.',
    summary: '[MOCK AI ANALYSIS] Placeholder summary.',
    action_items: ['[MOCK] Placeholder action item'],
    qa_items: [],
  };
}

async function mockApi(page) {
  await page.route('http://localhost:8000/api/process-meeting', async (route) => {
    posts.push(route.request().postDataBuffer().toString('latin1'));
    await route.fulfill({
      status: 200, contentType: 'application/json',
      headers: { 'Access-Control-Allow-Origin': '*' }, body: JSON.stringify(successBody()),
    });
    completed++;
  });
}

// Reads one text field from a multipart/form-data body
function formValue(body, name) {
  const m = body.match(new RegExp(`name="${name}"\\r\\n\\r\\n([^\\r]*)\\r\\n`));
  return m ? m[1] : null;
}

async function waitForPosts(page, n) {
  const deadline = Date.now() + 15000;
  while (completed < n) {
    if (Date.now() > deadline) throw new Error(`expected ${n} POST(s), got ${completed}`);
    await page.waitForTimeout(50);
  }
  await page.waitForFunction(() => document.getElementById('processing').hidden === true, null, { timeout: 15000 });
}

async function recordAndStop(page, ms = 1000) {
  await page.waitForTimeout(700); // the page ignores taps within 600 ms
  await page.click('#recBtn', { force: true });
  await page.waitForTimeout(ms);
  await page.click('#recBtn', { force: true });
}

async function openSettings(page) {
  await page.click('#settingsBtn');
  await page.waitForFunction(() => document.getElementById('settingsDialog').open);
}

async function closeWithDone(page) {
  await page.click('#settingsDone');
  await page.waitForFunction(() => !document.getElementById('settingsDialog').open);
}

const checked = (page, id) => page.getAttribute(`#${id}`, 'aria-checked');
const stored = (page) => page.evaluate((k) => localStorage.getItem(k), KEY);

const scenarios = [];
function scenario(name, fn, opts = {}) { scenarios.push({ name, fn, opts }); }

scenario('new user: Vault is OFF, Web is ON, the request sends those values, and nothing is written on load', async (page) => {
  assert.equal(await stored(page), null, 'settings were written on page load');
  assert.equal(await page.textContent('#settingsSummary'), 'Vault saving off · Web fallback on');
  await openSettings(page);
  assert.equal(await checked(page, 'vaultSwitch'), 'false');
  assert.equal(await checked(page, 'webSwitch'), 'true');
  assert.equal(await checked(page, 'setAutoDetect'), 'false');
  assert.equal(await checked(page, 'setQuiet'), 'false');
  assert.equal(await checked(page, 'setLive'), 'false');
  assert.equal(await page.isChecked('input[name="answerStyle"][value="balanced"]'), true);
  assert.equal(await page.isChecked('input[name="delivery"][value="panel"]'), true);
  await closeWithDone(page);
  await recordAndStop(page);
  await waitForPosts(page, 1);
  assert.equal(formValue(posts[0], 'is_internal_context'), 'false');
  assert.equal(formValue(posts[0], 'is_web_fallback'), 'true');
  assert.equal(await stored(page), null, 'recording wrote settings without a user change');
});

const SAVED_VAULT_ON = JSON.stringify({ version: 1, values: { vault: true } });
scenario('existing saved Vault ON is kept, sent, and not rewritten on load', async (page) => {
  assert.equal(await stored(page), SAVED_VAULT_ON, 'saved settings were rewritten on load');
  assert.equal(await page.textContent('#settingsSummary'), 'Vault saving on · Web fallback on');
  await recordAndStop(page);
  await waitForPosts(page, 1);
  assert.equal(formValue(posts[0], 'is_internal_context'), 'true');
  assert.equal(formValue(posts[0], 'is_web_fallback'), 'true');
}, { storage: SAVED_VAULT_ON });

scenario('changing Vault and Web in the panel persists across reload and changes the request', async (page) => {
  await openSettings(page);
  await page.click('#vaultSwitch');
  await page.click('#webSwitch');
  await page.keyboard.press('Escape');
  await page.waitForFunction(() => !document.getElementById('settingsDialog').open);
  assert.deepEqual(JSON.parse(await stored(page)), { version: 1, values: { vault: true, web: false } });
  await page.reload();
  await page.waitForTimeout(300);
  assert.equal(await page.textContent('#settingsSummary'), 'Vault saving on · Web fallback off');
  await openSettings(page);
  assert.equal(await checked(page, 'vaultSwitch'), 'true');
  assert.equal(await checked(page, 'webSwitch'), 'false');
  await closeWithDone(page);
  await recordAndStop(page);
  await waitForPosts(page, 1);
  assert.equal(formValue(posts[0], 'is_internal_context'), 'true');
  assert.equal(formValue(posts[0], 'is_web_fallback'), 'false');
});

scenario('upload sends the saved Vault and Web values in the same request format', async (page) => {
  await openSettings(page);
  await page.click('#webSwitch');
  await closeWithDone(page);
  const wav = Buffer.concat([Buffer.from('RIFF'), Buffer.alloc(4), Buffer.from('WAVEfmt '), Buffer.alloc(64)]);
  await page.setInputFiles('#fileInput', { name: 'call.wav', mimeType: 'audio/wav', buffer: wav });
  await waitForPosts(page, 1);
  assert.ok(posts[0].includes('name="file"; filename="call.wav"'));
  assert.equal(formValue(posts[0], 'meeting_name'), 'Untitled meeting');
  assert.equal(formValue(posts[0], 'is_internal_context'), 'false');
  assert.equal(formValue(posts[0], 'is_web_fallback'), 'false');
  assert.equal(await page.evaluate(() => document.getElementById('results').classList.contains('open')), true);
});

scenario('malformed JSON in storage falls back to defaults without errors and is not overwritten on load', async (page) => {
  assert.equal(await stored(page), 'not json {');
  assert.equal(await page.textContent('#settingsSummary'), 'Vault saving off · Web fallback on');
  assert.deepEqual(page.pageErrors, []);
}, { storage: 'not json {' });

const PARTLY_INVALID = JSON.stringify({ version: 1, values: {
  vault: 'yes', web: false, answerStyle: 'loud', delivery: 'sound', name: 'Tom', nicknames: 'Tommy',
  liveTranscription: true, quietMode: 1, unknownKey: 42,
} });
scenario('invalid stored values are ignored one by one; valid ones are kept', async (page) => {
  await openSettings(page);
  assert.equal(await checked(page, 'vaultSwitch'), 'false', 'invalid vault value was used');
  assert.equal(await checked(page, 'webSwitch'), 'false', 'valid web value was lost');
  assert.equal(await page.isChecked('input[name="answerStyle"][value="balanced"]'), true);
  assert.equal(await page.isChecked('input[name="delivery"][value="sound"]'), true);
  assert.equal(await page.inputValue('#setName'), 'Tom');
  assert.equal(await page.inputValue('#setNicknames'), '');
  assert.equal(await checked(page, 'setLive'), 'false');
  assert.equal(await checked(page, 'setQuiet'), 'false');
  assert.deepEqual(page.pageErrors, []);
}, { storage: PARTLY_INVALID });

scenario('names and Coming soon preferences save, reload, and trigger nothing', async (page) => {
  await openSettings(page);
  await page.fill('#setName', '  Tom  ');
  await page.fill('#setNicknames', ' Tommy , T,, tommy ');
  await page.fill('#setMishearings', 'Thom, Tum');
  await page.click('#setAutoDetect');
  await page.click('#setQuiet');
  // The style radios are visually hidden behind their labels, so click the label as a user would
  await page.click('label:has(input[name="answerStyle"][value="detailed"])');
  await page.click('label:has(input[name="delivery"][value="vibration"])');
  await closeWithDone(page);
  const values = JSON.parse(await stored(page)).values;
  assert.deepEqual(values, {
    name: 'Tom', nicknames: ['Tommy', 'T'], mishearings: ['Thom', 'Tum'],
    autoDetect: true, quietMode: true, answerStyle: 'detailed', delivery: 'vibration',
  });
  await page.reload();
  await page.waitForTimeout(300);
  await openSettings(page);
  assert.equal(await page.inputValue('#setName'), 'Tom');
  assert.equal(await page.inputValue('#setNicknames'), 'Tommy, T');
  assert.equal(await page.inputValue('#setMishearings'), 'Thom, Tum');
  assert.equal(await checked(page, 'setAutoDetect'), 'true');
  assert.equal(await checked(page, 'setQuiet'), 'true');
  assert.equal(await page.isChecked('input[name="answerStyle"][value="detailed"]'), true);
  assert.equal(await page.isChecked('input[name="delivery"][value="vibration"]'), true);
  await closeWithDone(page);
  await recordAndStop(page);
  await waitForPosts(page, 1);
  const calls = await page.evaluate(() => window.__sideEffects);
  assert.deepEqual(calls, { vibrate: 0, notification: 0, audio: 0 }, 'a Coming soon preference triggered a side effect');
  assert.equal(formValue(posts[0], 'is_internal_context'), 'false', 'Coming soon settings changed the request');
});

// Regression: the hidden Answer style radios once sat outside the scrolling area, so selecting one
// scrolled the whole dialog and pushed the footer up. Short screens showed it most clearly.
scenario('choosing an answer style or delivery never scrolls the dialog or moves the footer', async (page) => {
  for (const vp of [{ width: 375, height: 667 }, { width: 1366, height: 680 }]) {
    await page.setViewportSize(vp);
    await openSettings(page);
    const layout = () => page.evaluate(() => {
      const d = document.getElementById('settingsDialog');
      const box = d.getBoundingClientRect();
      const foot = d.querySelector('.sheet-foot').getBoundingClientRect();
      return { dialogScroll: d.scrollTop, footTop: Math.round(foot.top - box.top), footHeight: Math.round(foot.height), gapBelow: Math.round(box.bottom - foot.bottom) };
    });
    const before = await layout();
    for (const [name, value] of [['answerStyle', 'short'], ['answerStyle', 'balanced'], ['answerStyle', 'detailed'],
      ['delivery', 'sound'], ['delivery', 'vibration'], ['delivery', 'panel']]) {
      await page.$eval('#settingsDialog .overflow-y-auto', (e) => e.scrollTo(0, 450));
      await page.click(`label:has(input[name="${name}"][value="${value}"])`);
      await page.waitForTimeout(150);
      assert.equal(await page.isChecked(`input[name="${name}"][value="${value}"]`), true);
      assert.deepEqual(await layout(), before, `${vp.width}x${vp.height}: selecting ${name}=${value} moved the layout`);
    }
    await closeWithDone(page);
  }
});

scenario('Live transcription is disabled and stays off', async (page) => {
  await openSettings(page);
  assert.equal(await page.isDisabled('#setLive'), true);
  await page.click('#setLive', { force: true });
  assert.equal(await checked(page, 'setLive'), 'false');
  assert.equal(await stored(page), null);
});

scenario('Vault and Web are locked during a recording and unlocked afterwards', async (page) => {
  await page.click('#recBtn', { force: true });
  await page.waitForTimeout(800);
  await openSettings(page);
  assert.equal(await page.isDisabled('#vaultSwitch'), true);
  assert.equal(await page.isDisabled('#webSwitch'), true);
  assert.equal(await page.isVisible('#lockNote'), true);
  await page.click('#vaultSwitch', { force: true });
  assert.equal(await checked(page, 'vaultSwitch'), 'false', 'locked switch changed');
  assert.equal(await page.isDisabled('#setAutoDetect'), false, 'non-request settings should stay editable');
  await closeWithDone(page);
  await page.waitForTimeout(400);
  await page.click('#recBtn', { force: true });
  await waitForPosts(page, 1);
  assert.equal(formValue(posts[0], 'is_internal_context'), 'false');
  await openSettings(page);
  assert.equal(await page.isDisabled('#vaultSwitch'), false);
  assert.equal(await page.isVisible('#lockNote'), false);
});

scenario('panel closes with Escape, Done and the close button; focus returns to the opener', async (page) => {
  await openSettings(page);
  await page.keyboard.press('Escape');
  await page.waitForFunction(() => !document.getElementById('settingsDialog').open);
  assert.equal(await page.evaluate(() => document.activeElement.id), 'settingsBtn');
  await openSettings(page);
  await page.click('#settingsClose');
  await page.waitForFunction(() => !document.getElementById('settingsDialog').open);
  await page.click('#summarySettingsBtn');
  await page.waitForFunction(() => document.getElementById('settingsDialog').open);
  await closeWithDone(page);
  assert.equal(await page.evaluate(() => document.activeElement.id), 'summarySettingsBtn');
});

scenario('blocked storage: settings still work for this visit and the panel says they are not saved', async (page) => {
  assert.deepEqual(page.pageErrors, []);
  await openSettings(page);
  await page.click('#vaultSwitch');
  assert.ok((await page.textContent('#storageNote')).includes('not saving settings'));
  await closeWithDone(page);
  await recordAndStop(page);
  await waitForPosts(page, 1);
  assert.equal(formValue(posts[0], 'is_internal_context'), 'true');
}, { blockStorage: true });

async function main() {
  const browser = await chromium.launch({
    executablePath: BROWSER_PATH,
    headless: true,
    args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--autoplay-policy=no-user-gesture-required'],
  });
  let failures = 0;
  try {
    for (const { name, fn, opts } of scenarios) {
      const context = await browser.newContext({ viewport: { width: 420, height: 900 } });
      // Count calls that a Coming soon preference must never make in Phase 2
      await context.addInitScript(() => {
        window.__sideEffects = { vibrate: 0, notification: 0, audio: 0 };
        try { navigator.vibrate = () => { window.__sideEffects.vibrate++; return true; }; } catch (_) {}
        if (window.Notification) window.Notification.requestPermission = () => { window.__sideEffects.notification++; return Promise.resolve('denied'); };
        const play = HTMLMediaElement.prototype.play;
        HTMLMediaElement.prototype.play = function () { window.__sideEffects.audio++; return play.call(this); };
      });
      if (opts.storage !== undefined) {
        // Seed storage once, before the first load only, so reloads see what the page saved
        await context.addInitScript(([k, v]) => {
          if (!sessionStorage.getItem('__seeded')) { localStorage.setItem(k, v); sessionStorage.setItem('__seeded', '1'); }
        }, [KEY, opts.storage]);
      }
      if (opts.blockStorage) {
        await context.addInitScript(() => {
          Object.defineProperty(window, 'localStorage', { get() { throw new Error('storage blocked'); } });
        });
      }
      const page = await context.newPage();
      page.pageErrors = [];
      page.on('pageerror', (e) => page.pageErrors.push(e.message));
      posts = [];
      completed = 0;
      await mockApi(page);
      await page.goto(PAGE_URL);
      await page.waitForTimeout(300);
      try {
        await fn(page);
        console.log(`PASS  ${name}`);
      } catch (err) {
        failures++;
        console.log(`FAIL  ${name}\n      ${err.message.split('\n')[0]}`);
      }
      await context.close();
    }
  } finally {
    await browser.close();
  }
  console.log(failures === 0
    ? `\nAll ${scenarios.length} settings scenarios passed (mocked API, fake microphone).`
    : `\n${failures} of ${scenarios.length} settings scenarios failed.`);
  process.exitCode = failures === 0 ? 0 : 1;
}

main().catch((err) => { console.error(err); process.exit(1); });
