// Browser test for the recording-to-results flow in frontend/index.html.
//
// Uses a MOCKED /api/process-meeting route, so it never reaches the backend or Groq.
// Uses Chromium's fake microphone, so no real microphone or speech is needed.
//
// Run from the project root (requires Node 20+ and Microsoft Edge or Chrome installed):
//   npm install --no-save playwright-core
//   node frontend/tests/results-flow.mjs
//
// Optional environment variables:
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
const SENTENCE = 'Hello, this is a test recording for OmniScribe AI. We are checking whether the application can accurately convert speech into text.';

// Per-scenario state. `script` holds one scripted response per POST, in order.
let script = [];
let postCount = 0;
let completedCount = 0;

function successBody(transcript) {
  return {
    success: true,
    meeting_name: 'Untitled meeting',
    transcript_segment: transcript,
    summary: '[MOCK AI ANALYSIS] Placeholder summary.',
    action_items: ['[MOCK] Placeholder action item'],
    qa_items: [{ question: '[MOCK] Placeholder question', answer: '[MOCK] Placeholder answer', source: 'none' }],
  };
}

async function mockApi(page) {
  await page.route('http://localhost:8000/api/process-meeting', async (route) => {
    postCount++;
    const step = script.shift() || { status: 200, body: successBody(`Mock transcript ${postCount}.`) };
    if (step.delayMs) await new Promise((r) => setTimeout(r, step.delayMs));
    const headers = { 'Access-Control-Allow-Origin': '*' };
    if (step.text !== undefined) {
      await route.fulfill({ status: step.status, contentType: 'text/plain', headers, body: step.text });
    } else {
      await route.fulfill({ status: step.status, contentType: 'application/json', headers, body: JSON.stringify(step.body) });
    }
    completedCount++;
  });
}

// Waits until the mocked API has answered `expected` POSTs in total, then until the spinner is hidden
async function waitUntilSettled(page, expected) {
  const deadline = Date.now() + 15000;
  while (completedCount < expected) {
    if (Date.now() > deadline) throw new Error(`expected ${expected} completed POST(s), got ${completedCount}`);
    await page.waitForTimeout(50);
  }
  await page.waitForFunction(() => document.getElementById('processing').hidden === true, null, { timeout: 15000 });
}

async function tapRecord(page) {
  await page.click('#recBtn', { force: true }); // the button pulses while recording, so skip the stability check
}

async function recordAndStop(page, ms = 1200) {
  await page.waitForTimeout(700); // human tap spacing; the page ignores taps within 600 ms
  await tapRecord(page);
  await page.waitForTimeout(ms);
  await tapRecord(page);
}

// Waits for the open-results transition and the scroll-to-results animation to finish.
// If results never become visible, the caller's assertions report the real state.
async function waitForResultsToSettle(page) {
  await page.waitForFunction(() => {
    const r = document.getElementById('results');
    const s = document.getElementById('scroller').getBoundingClientRect();
    const rect = r.getBoundingClientRect();
    return r.classList.contains('open') && rect.height > 0 && rect.bottom > s.top && rect.top < s.bottom
      && getComputedStyle(r).opacity === '1';
  }, null, { timeout: 4000 }).catch(() => {});
}

async function settledResults(page) {
  await waitForResultsToSettle(page);
  return resultsState(page);
}

function resultsState(page) {
  return page.evaluate(() => {
    const r = document.getElementById('results');
    const s = document.getElementById('scroller').getBoundingClientRect();
    const rect = r.getBoundingClientRect();
    return {
      open: r.classList.contains('open'),
      height: Math.round(rect.height),
      inViewport: rect.height > 0 && rect.bottom > s.top && rect.top < s.bottom,
      transcript: document.getElementById('transcript').textContent,
      notice: document.getElementById('resultsNotice').textContent,
      source: document.getElementById('resSource').textContent,
    };
  });
}

function errorState(page) {
  return page.evaluate(() => ({
    hidden: document.getElementById('errorBanner').hidden,
    title: document.getElementById('errorTitle').textContent,
    message: document.getElementById('errorMessage').textContent,
  }));
}

const scenarios = [];
function scenario(name, fn) { scenarios.push({ name, fn }); }

scenario('single recording shows real transcript, mock labels, and visible results', async (page) => {
  script = [{ status: 200, body: successBody(SENTENCE) }];
  await recordAndStop(page);
  await waitUntilSettled(page, 1);
  assert.equal(postCount, 1, 'one recording must produce exactly one POST');
  const r = await settledResults(page);
  assert.equal(r.open, true);
  assert.ok(r.height > 0, `results collapsed to ${r.height}px`);
  assert.ok(r.inViewport, 'results rendered but not inside the visible scroller');
  assert.ok(r.transcript.includes(SENTENCE), 'transcript does not contain the returned text');
  assert.ok(r.notice.includes('mock placeholders'), 'notice must mark analysis as mock');
  assert.ok(r.source.startsWith('live-mic-'), `unexpected source label: ${r.source}`);
  assert.equal((await errorState(page)).hidden, true);
});

scenario('second recording replaces results; failed third keeps them and shows server message', async (page) => {
  script = [
    { status: 200, body: successBody('Second transcript.') },
    { status: 500, body: { detail: 'Speech-to-text service error: simulated failure' } },
  ];
  await recordAndStop(page);
  await waitUntilSettled(page, 1);
  assert.ok((await resultsState(page)).transcript.includes('Second transcript.'));

  await recordAndStop(page);
  await waitUntilSettled(page, 2);
  const r = await settledResults(page);
  assert.equal(r.open, true, 'failed request must not close existing results');
  assert.ok(r.height > 0, 'failed request must not collapse existing results');
  assert.ok(r.transcript.includes('Second transcript.'), 'failed request must not erase the last good transcript');
  const e = await errorState(page);
  assert.equal(e.hidden, false);
  assert.ok(e.message.includes('simulated failure'), `server detail not shown: ${e.message}`);
});

scenario('non-JSON error body is shown instead of "Backend Connection Failed"', async (page) => {
  script = [{ status: 502, text: 'Bad gateway from proxy' }];
  await recordAndStop(page);
  await waitUntilSettled(page, 1);
  const e = await errorState(page);
  assert.ok(e.message.includes('Bad gateway from proxy'), `raw error body lost: ${e.message}`);
  assert.notEqual(e.title, 'Backend Connection Failed');
});

scenario('double-tap on stop submits once and does not start a new recording', async (page) => {
  script = [{ status: 200, body: successBody('Double tap transcript.') }];
  await tapRecord(page);
  await page.waitForTimeout(1200);
  await tapRecord(page); // stop
  await tapRecord(page); // second tap, fired immediately
  await page.waitForTimeout(1500);
  await waitUntilSettled(page, 1);
  await page.waitForTimeout(500);
  assert.equal(postCount, 1, `expected 1 POST, got ${postCount}`);
  assert.equal(await page.getAttribute('#recBtn', 'aria-pressed'), 'false', 'second tap started a new recording');
});

scenario('processing indicator shows during the request and hides when it finishes', async (page) => {
  script = [{ status: 200, body: successBody('Slow transcript.'), delayMs: 1500 }];
  await recordAndStop(page, 800);
  await page.waitForFunction(() => document.getElementById('processing').hidden === false, null, { timeout: 5000 });
  assert.equal(await page.isVisible('#processing'), true);
  await waitUntilSettled(page, 1);
  assert.equal(await page.isVisible('#processing'), false);
});

scenario('starting a new recording does not clear existing results', async (page) => {
  script = [{ status: 200, body: successBody('Kept transcript.') }];
  await recordAndStop(page);
  await waitUntilSettled(page, 1);
  await page.waitForTimeout(700);
  await tapRecord(page); // start a second recording, do not stop yet
  await page.waitForTimeout(1000);
  const r = await settledResults(page);
  assert.equal(r.open, true, 'starting a new recording hid the previous results');
  assert.ok(r.transcript.includes('Kept transcript.'));
  await tapRecord(page); // stop the second recording
  await waitUntilSettled(page, 2);
});

scenario('sample button is ignored while recording (no duplicate or lost submission)', async (page) => {
  script = [{ status: 200, body: successBody('Recording transcript.') }];
  await tapRecord(page);
  await page.waitForTimeout(800);
  await page.click('#sampleBtn');
  await page.waitForTimeout(300);
  assert.equal(postCount, 0, 'sample was sent while a recording was active');
  await tapRecord(page);
  await waitUntilSettled(page, 1);
  assert.equal(postCount, 1);
  assert.ok((await resultsState(page)).source.startsWith('live-mic-'), 'the recording, not the sample, was submitted');
});

scenario('Clear hides results', async (page) => {
  script = [{ status: 200, body: successBody('To be cleared.') }];
  await recordAndStop(page);
  await waitUntilSettled(page, 1);
  await page.click('#clearBtn');
  await page.waitForTimeout(900);
  assert.equal((await resultsState(page)).open, false);
});

async function main() {
  const browser = await chromium.launch({
    executablePath: BROWSER_PATH,
    headless: true,
    args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--autoplay-policy=no-user-gesture-required'],
  });
  let failures = 0;
  try {
    for (const { name, fn } of scenarios) {
      const context = await browser.newContext({ viewport: { width: 420, height: 900 } });
      const page = await context.newPage();
      script = [];
      postCount = 0;
      completedCount = 0;
      await mockApi(page);
      await page.goto(PAGE_URL);
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
    ? `\nAll ${scenarios.length} browser scenarios passed (mocked API, fake microphone).`
    : `\n${failures} of ${scenarios.length} scenarios failed.`);
  process.exitCode = failures === 0 ? 0 : 1;
}

main().catch((err) => { console.error(err); process.exit(1); });
