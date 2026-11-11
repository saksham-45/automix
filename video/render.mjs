// Render the automix × visualizzer video.
// usage: node render.mjs <frames.bin> <mix.json> <audio.wav> <out.mp4> [startSec] [durSec] [metalFrames.bin] [subtitle] [credit]
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import { openVisualizer, setMode, frameReader } from './harness.mjs';

const [bin, mixJson, audio, out, startArg, durArg, metalBin, subtitle, credit] = process.argv.slice(2);
const FPS = 60;
const mix = JSON.parse(fs.readFileSync(mixJson, 'utf8'));
const markers = mix.markers;
const duration = markers[markers.length - 1].end_sec;
const t0 = +(startArg || 0);
const t1 = durArg ? Math.min(duration, t0 + +durArg) : duration;

// visual mode per track; it changes on each transition's switch (its midpoint)
const ROTATION = [
  ['liquidGeometry', 'Chrome Morph'], ['shader_neonVortex', 'Neon Vortex'], ['mercuryOrbs', 'Mercury Orbs'],
  ['shader_electricStorm', 'Electric Storm'], ['liquidMetal', 'Liquid Metal'], ['shader_hypnoticSpiral', 'Hypnotic Spiral'],
  ['metallicNebula', 'Metallic Nebula'], ['liquidGeometry', 'Chrome Morph'],
];
const METAL = new Set(['liquidGeometry', 'mercuryOrbs', 'liquidMetal', 'metallicNebula']);
const modes = [{ at: 0, mode: ROTATION[0][0], label: ROTATION[0][1] }];
for (const m of markers) {
  if (m.type !== 'transition') continue;
  const r = ROTATION[m.to % ROTATION.length];
  modes.push({ at: (m.start_sec + m.end_sec) / 2, mode: r[0], label: r[1] });
}
const modeAt = (t) => modes.filter((x) => x.at <= t).pop();

const hud = {
  markers, duration, introSec: 5, outroSec: 10, modes, subtitle: subtitle || '', credit: credit || '',
  tracks: mix.tracks.map((t) => ({ title: t.title, artist: t.artist })),
};

const readOther = frameReader(bin);
const readMetal = metalBin ? frameReader(metalBin) : readOther;
const { browser, page } = await openVisualizer({ width: 1920, height: 1080, dpr: 2 });
await page.addScriptTag({ content: fs.readFileSync(new URL('./hud.js', import.meta.url), 'utf8') });
await page.evaluate((d) => window.__hudInit(d), hud);

const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error',
  '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'mjpeg', '-i', '-',
  '-ss', String(t0), '-t', String(t1 - t0), '-i', audio,
  '-map', '0:v', '-map', '1:a',
  '-c:v', 'libx264', '-preset', 'slow', '-crf', '14', '-pix_fmt', 'yuv420p', '-profile:v', 'high',
  '-r', String(FPS), '-movflags', '+faststart',
  '-c:a', 'aac', '-b:a', '320k', '-shortest', out], { stdio: ['pipe', 'inherit', 'inherit'] });

const cdp = await page.context().newCDPSession(page);
let cur = null;
// warm-up: let the visuals settle before the first recorded frame
const f0 = Math.round(t0 * FPS), f1 = Math.round(t1 * FPS);
const warm = Math.max(0, f0 - 120);
const started = Date.now();
for (let f = warm; f < f1; f++) {
  const t = f / FPS;
  const m = modeAt(t);
  if (m.mode !== cur) { await setMode(page, m.mode); cur = m.mode; }
  await page.evaluate(([b, dt, tt]) => { window.__setFrame(b); window.__hud(tt); window.__step(dt); }, [(METAL.has(m.mode) ? readMetal : readOther)(f), 1000 / FPS, t]);
  if (f < f0) continue;
  const { data } = await cdp.send('Page.captureScreenshot', { format: 'jpeg', quality: 95, optimizeForSpeed: false });
  if (!ff.stdin.write(Buffer.from(data, 'base64'))) await new Promise((r) => ff.stdin.once('drain', r));
  if ((f - f0) % 600 === 0) {
    const done = (f - f0) / (f1 - f0), el = (Date.now() - started) / 1000;
    console.log(`${(t / 60).toFixed(2)} min  ${(done * 100).toFixed(1)}%  ${((f - f0) / el).toFixed(1)} fps  eta ${done > 0 ? ((el / done - el) / 60).toFixed(0) : '?'} min  mode ${m.label}`);
  }
}
ff.stdin.end();
await new Promise((r) => ff.on('close', r));
await browser.close();
console.log('done', out, `${((Date.now() - started) / 60000).toFixed(1)} min`);
