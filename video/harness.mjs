// Offline, frame-accurate renderer for the visualizzer project.
// The page runs on a virtual clock: time only moves when we step it, so every
// video frame is rendered exactly 1/fps after the previous one no matter how long
// the GPU takes, and the audio data for that instant is injected directly.
import { chromium } from 'playwright-core';
import fs from 'node:fs';

export const INIT = `(() => {
  let now = 0;
  const base = 1767225600000;
  performance.now = () => now;
  Date.now = () => base + now;
  let s = 20261111 >>> 0;
  Math.random = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
  const raf = [];
  window.requestAnimationFrame = (cb) => { raf.push(cb); return raf.length; };
  window.cancelAnimationFrame = () => {};
  const timers = []; let tid = 1;
  window.setTimeout = (fn, d = 0, ...a) => { timers.push({ id: tid, t: now + (+d || 0), fn, a, every: 0 }); return tid++; };
  window.setInterval = (fn, d = 0, ...a) => { timers.push({ id: tid, t: now + (+d || 0), fn, a, every: Math.max(+d || 0, 1) }); return tid++; };
  window.clearTimeout = window.clearInterval = (id) => { const i = timers.findIndex((t) => t.id === id); if (i >= 0) timers.splice(i, 1); };
  Object.defineProperty(window, 'devicePixelRatio', { get: () => (window.__dpr || 1) });
  window.__step = (dt) => {
    now += dt;
    for (let guard = 0; guard < 1000; guard++) {
      timers.sort((x, y) => x.t - y.t);
      const t = timers[0];
      if (!t || t.t > now) break;
      if (t.every) t.t += t.every; else timers.shift();
      try { typeof t.fn === 'function' ? t.fn(...t.a) : null; } catch (e) { console.error(e); }
    }
    const cbs = raf.splice(0);
    for (const cb of cbs) { try { cb(now); } catch (e) { console.error('raf', e); } }
  };
})();`;

export async function openVisualizer({ width = 1920, height = 1080, dpr = 2, url = process.env.VIZ_URL || 'http://127.0.0.1:8765/index.html' } = {}) {
  const browser = await chromium.launch({
    channel: 'chrome', headless: true,
    args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-gpu-rasterization',
           '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--autoplay-policy=no-user-gesture-required'],
  });
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  page.on('console', (m) => { if (m.type() === 'error') console.error('[page]', m.text().slice(0, 200)); });
  page.on('pageerror', (e) => console.error('[pageerror]', String(e).slice(0, 200)));
  await page.addInitScript(INIT);
  await page.addInitScript(`window.__dpr = ${dpr};`);
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.visualizerApp);
  // older versions only build the visualizers on Start: start with a stubbed capture
  await page.evaluate(async () => {
    const app = window.visualizerApp;
    if (!app.visualizers) {
      app.audioCapture.start = async () => { app.audioCapture.isCapturing = true; };
      await app.start();
    }
  });
  await page.waitForFunction(() => window.visualizerApp.visualizers, null, { timeout: 30000 });
  await page.evaluate(() => {
    const app = window.visualizerApp;
    const v = app.visualizers;
    // hide the app's own UI chrome — the video is the visuals
    for (const sel of ['#uiOverlay', '.header', '.info-panel', '#exitFullscreenBtn']) {
      document.querySelectorAll(sel).forEach((el) => (el.style.display = 'none'));
    }
    document.body.style.cursor = 'none';
    app.isRunning = true;
    if (app.autoModeCheckbox) app.autoModeCheckbox.checked = false;
    app.audioCapture.isCapturing = true;
    window.__frame = null;
    app.audioCapture.getAudioData = () => window.__frame && {
      frequencyData: window.__frame.f, timeData: window.__frame.t,
      bufferLength: 1024, sampleRate: 44100, fftSize: 2048,
    };
    // lock maximum quality: no adaptive down-scaling
    const po = v.performanceOptimizer;
    if (po) {
      po.adaptQuality = () => {};
      po.qualityLevel = 'high'; po.particleCountMultiplier = 1; po.effectIntensityMultiplier = 1; po.resolutionMultiplier = 1;
    }
    window.__setFrame = (b64) => {
      const bin = atob(b64); const u = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
      window.__frame = { f: u.subarray(0, 1024), t: u.subarray(1024, 2048) };
    };
    window.__maxQuality = () => {
      const tv = v.fluidVisualizer && v.fluidVisualizer.threeVisualizer;
      if (!tv || !tv.renderer) return false;
      tv._updateQuality = () => {};
      tv.renderScale = 1.0; tv.maxPixelRatio = window.__dpr;
      tv.resize();
      const gl = tv.renderer.getContext();
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      return { size: [tv.renderer.domElement.width, tv.renderer.domElement.height],
               gpu: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : 'unknown' };
    };
  });
  return { browser, page };
}

export async function setMode(page, mode) {
  await page.evaluate((m) => window.visualizerApp.visualizers.setVisualizer(m), mode);
  // the 3D engine is imported lazily: step until it is up, then force max quality
  for (let i = 0; i < 400; i++) {
    const q = await page.evaluate(() => window.__maxQuality());
    if (q) return q;
    await page.evaluate(() => window.__step(1000 / 60));
    await new Promise((r) => setTimeout(r, 10));
  }
  return null;
}

export function frameReader(path) {
  const fd = fs.openSync(path, 'r');
  const buf = Buffer.alloc(2048);
  return (i) => { fs.readSync(fd, buf, 0, 2048, i * 2048); return buf.toString('base64'); };
}
