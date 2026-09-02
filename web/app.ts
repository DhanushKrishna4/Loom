// Drives the engine from JS, one token per frame.
//
// The engine deliberately has no internal generate() loop: wasm runs on the
// thread that called it, so looping inside would freeze the tab for the whole
// generation -- no repaint, no input, no stop button. Everything here is built
// around calling decodeStep() exactly once per animation frame.

import init, { Engine } from './pkg/nano_infer_wasm.js';
import type { DecodeResult, TensorSummary } from './pkg/nano_infer_wasm.js';
import * as cache from './idb';
import * as viz from './viz';
import { startHero, revealOnScroll } from './hero';

// Chrome-first, before anything that can throw: the hero and the reveals are
// pure presentation, and a model that fails to load should still leave a page
// that looks finished rather than a half-painted one.
const heroCanvas = document.getElementById('heroCanvas');
if (heroCanvas instanceof HTMLCanvasElement) startHero(heroCanvas);
revealOnScroll();

// `?gguf=<url>` points at a local copy, which saves a 469 MB download when you
// already have the file on disk.
const DEFAULT_URL =
  new URLSearchParams(location.search).get('gguf') ||
  'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf';

// `?seed=<n>` reproduces a specific generation. Sampling is seeded PCG32, so
// the same seed, prompt and parameters give the same tokens back -- which is
// only useful if the seed can travel, hence the URL. Written back on every run
// (replaceState, not pushState: a shared link should not stack history entries).
const URL_SEED = new URLSearchParams(location.search).get('seed');

function shareCurrentSeed(seed: number): void {
  const u = new URL(location.href);
  u.searchParams.set('seed', String(seed));
  history.replaceState(null, '', u);
}

/**
 * Look up an element that the page is required to contain.
 *
 * Throws rather than returning null: every id here is in this file's own
 * markup, so a miss is a typo at author time, not a runtime condition worth
 * threading null checks through the whole module for.
 */
const $ = <T extends HTMLElement = HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el as T;
};

/** Debug hook, so the console can drive the engine directly. */
declare global {
  interface Window {
    nanoInfer?: Engine;
  }
}

// Schedule the next decode step.
//
// NOT requestAnimationFrame. rAF does not fire in a hidden tab, so switching
// away mid-generation stops it dead and it never resumes -- you come back to a
// half-written answer and a spinner. Found by driving this page in a browser
// whose tab reported `visibilityState: "hidden"`; it never produced a token.
//
// setTimeout(0) yields to the event loop just as well, still lets the browser
// repaint and handle the Stop button between tokens, and keeps running when
// hidden (throttled to ~1/s in the background, which is the browser's policy
// and the right one). At ~100 ms per token, frame alignment buys nothing.
const nextTick = (fn: () => void): void => { setTimeout(fn, 0); };
/** Narrow a caught value to something printable. `catch` binds `unknown`. */
/** The module-level engine, narrowed for callers that must handle its absence. */
const globalEngine = (): Engine | null => engine;

const msg = (e: unknown): string => (e instanceof Error ? e.message : String(e));

const fmtBytes = (n: number): string => {
  const u = ['B', 'KiB', 'MiB', 'GiB'];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
};

let engine: Engine | null = null;
let running = false;

function status(msg: string, isError = false): void {
  const el = $('loadStatus');
  el.textContent = msg;
  el.classList.toggle('error', isError);
}

async function build(bytes: Uint8Array, source = 'file'): Promise<void> {
  const maxSeq = parseInt($<HTMLSelectElement>('maxseq').value, 10);
  status(`parsing ${fmtBytes(bytes.length)}…`);
  // Yield once so the message paints before the parse blocks the thread.
  await new Promise((r) => setTimeout(r, 0));

  const t0 = performance.now();
  try {
    engine = new Engine(bytes, maxSeq);
  } catch (e) {
    status(`could not load: ${e}`, true);
    throw e;
  }
  const ms = performance.now() - t0;

  // Debug hook: lets the console drive the engine directly, which is how the
  // background-tab throttling question below got settled.
  window.nanoInfer = engine;

  const s = engine.stats();
  $('modelInfo').textContent =
    `${engine.description}\nweights ${fmtBytes(s.weightBytes)} · ` +
    `kv cache ${fmtBytes(s.cacheBytes)} · workspace ${fmtBytes(s.workspaceBytes)} · ` +
    `context ${engine.maxSeq} of ${engine.contextLength}`;
  status(`ready in ${(ms / 1000).toFixed(1)} s (from ${source})`);
  $('gen').hidden = false;
  $('viz').hidden = false;
  $('quant').hidden = false;
  populateTensors();
  $('headMax').textContent = `head ${engine.nHeads - 1}`;
  $<HTMLProgressElement>('loadBar').hidden = true;
  redrawViz(null);
}

const fmtAge = (ms: number): string => {
  const s = (Date.now() - ms) / 1000;
  if (s < 90) return 'just now';
  if (s < 5400) return `${Math.round(s / 60)} min ago`;
  if (s < 172800) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
};

async function refreshCacheUI(): Promise<void> {
  const el = $('cacheStatus');
  const btn = $<HTMLButtonElement>('clearCache');
  try {
    const rows = await cache.list();
    if (!rows.length) {
      const q = await cache.quota();
      el.textContent = q
        ? `nothing cached · ${fmtBytes(q.quota - q.usage)} of storage available`
        : 'nothing cached';
      btn.hidden = true;
      return;
    }
    const total = rows.reduce((a, r) => a + r.bytes, 0);
    const newest = Math.max(...rows.map((r) => r.storedAt));
    el.textContent =
      `cached: ${rows.length} model${rows.length > 1 ? 's' : ''}, ` +
      `${fmtBytes(total)}, last stored ${fmtAge(newest)}`;
    btn.hidden = false;
  } catch (e) {
    el.textContent = `cache unavailable: ${msg(e)}`;
    btn.hidden = true;
  }
}

/** Download with a progress bar, returning a Blob. */
async function download(url: string): Promise<Blob> {
  const bar = $<HTMLProgressElement>('loadBar');
  bar.hidden = false;
  bar.value = 0;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length')) || 0;
  const chunks = [];
  let got = 0;
  if (!res.body) throw new Error('response had no body to stream');
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (total) bar.value = got / total;
    status(`downloading ${fmtBytes(got)}${total ? ` of ${fmtBytes(total)}` : ''}…`);
  }
  return new Blob(chunks);
}

async function loadFromUrl(url: string): Promise<void> {
  let blob = null;
  const hit = await cache.get(url).catch(() => null);

  if (hit) {
    status(`reading ${fmtBytes(hit.bytes)} from cache…`);
    blob = hit.blob;
  } else {
    blob = await download(url);
    // Ask before storing, not after: without persistence a 469 MB cache is
    // "best effort" and can be evicted under disk pressure, which silently
    // turns an instant revisit back into a download.
    await cache.requestPersistence().catch(() => false);
    status(`caching ${fmtBytes(blob.size)}…`);
    const stored = await cache.put(url, blob).catch(() => false);
    if (!stored) {
      status('could not cache (quota) — the model still loads, but will not persist');
    }
    await refreshCacheUI();
  }

  // Blob -> ArrayBuffer -> wasm. This is the peak-memory moment: the blob's
  // bytes and the wasm copy exist at once.
  const bytes = new Uint8Array(await blob.arrayBuffer());
  await build(bytes, hit ? 'cache' : 'network');
}

$<HTMLInputElement>('file').addEventListener('change', async () => {
  const f = $<HTMLInputElement>('file').files?.[0];
  if (!f) return;
  status(`reading ${f.name}…`);
  // A hand-picked file is not cached: it is already on disk, and its path is
  // not a stable key.
  await build(new Uint8Array(await f.arrayBuffer()), 'file');
});

$('fetch').addEventListener('click', async () => {
  const url = $<HTMLInputElement>('url').value || DEFAULT_URL;
  try {
    await loadFromUrl(url);
  } catch (e) {
    status(`load failed: ${msg(e)}`, true);
    $<HTMLProgressElement>('loadBar').hidden = true;
  }
});

$<HTMLButtonElement>('clearCache').addEventListener('click', async () => {
  await cache.clear();
  await refreshCacheUI();
  status('cache cleared');
});

let selected: viz.HeadSelection = { layer: 0, head: 0 };
let promptPieces: string[] = [];

function redrawViz(result: DecodeResult | null): void {
  const engine = globalEngine();
  if (!engine) return;
  const s = engine.stats();

  viz.drawCache($<HTMLCanvasElement>('kv'), s.cacheUsed, s.cacheCapacity, s.cacheBytes);

  if (result) {
    viz.drawPerf($<HTMLCanvasElement>('perf'), [
      { label: 'matmul', ms: result.matmulMs, color: opColour('--op-matmul', '#ffae57') },
      { label: 'attention', ms: result.attentionMs, color: opColour('--op-attn', '#4fe0bc') },
      { label: 'elementwise', ms: result.otherMs, color: opColour('--op-elem', '#7aa2c9') },
      { label: 'overhead', ms: result.overheadMs, color: opColour('--op-overhead', '#56666a') },
    ]);
  }

  if (!$<HTMLInputElement>('capture').checked) return;
  // One view for all 336 distributions. Fetching them per head would be 336
  // boundary crossings per token.
  const attn = engine.attentionView();
  viz.drawHeads($<HTMLCanvasElement>('heads'), attn,
    engine.nLayers, engine.nHeads, engine.maxSeq, s.cacheUsed, selected);
  viz.drawDistribution($<HTMLCanvasElement>('dist'),
    engine.attention(selected.layer, selected.head), promptPieces);
}

$<HTMLInputElement>('capture').addEventListener('change', () => {
  engine?.setCaptureAttention($<HTMLInputElement>('capture').checked);
  redrawViz(null);
});

$<HTMLCanvasElement>('heads').addEventListener('click', (e: MouseEvent) => {
  const eng = engine;
  if (!eng) return;
  const r = $<HTMLCanvasElement>('heads').getBoundingClientRect();
  const head = Math.min(eng.nHeads - 1,
    Math.floor(((e.clientX - r.left) / r.width) * eng.nHeads));
  const layer = Math.min(eng.nLayers - 1,
    Math.floor(((e.clientY - r.top) / r.height) * eng.nLayers));
  selected = { layer, head };
  $('headSel').textContent = `layer ${layer} · head ${head}`;
  redrawViz(null);
});

// ------------------------------------------- quantisation explorer ----

let tensors: TensorSummary[] = [];

function populateTensors(): void {
  const engine = globalEngine();
  if (!engine) return;
  tensors = engine.tensorList();
  const sel = $<HTMLSelectElement>('qtensor');
  sel.innerHTML = '';
  for (const t of tensors) {
    const o = document.createElement('option');
    o.value = t.name;
    o.textContent = `${t.name}  —  ${t.format}  ${t.rows}×${t.cols}  ${t.bitsPerWeight.toFixed(2)} bits/wt`;
    o.disabled = !t.inspectable;
    sel.appendChild(o);
  }
  // Default to an ffn_down: it is the only tensor with rows long enough for a
  // k-quant, so it is where the interesting packing lives.
  const kq = tensors.find((t) => t.inspectable && t.format.endsWith('_K'));
  sel.value = (kq || tensors.find((t) => t.inspectable) || tensors[0]).name;
  drawQuant();
}

function drawQuant(): void {
  const engine = globalEngine();
  if (!engine || !tensors.length) return;
  const name = $<HTMLSelectElement>('qtensor').value;
  const info = tensors.find((t) => t.name === name);
  if (!info || !info.inspectable) {
    $('qmeta').textContent = `${name}: ${info ? info.format : '?'} — no decoder, so nothing to show`;
    return;
  }

  const blockInput = $<HTMLInputElement>('qblock');
  blockInput.max = String(info.blocks - 1);
  let b = parseInt(blockInput.value, 10);
  if (!Number.isFinite(b) || b < 0) b = 0;
  if (b > info.blocks - 1) { b = info.blocks - 1; blockInput.value = String(b); }

  let v;
  try {
    v = engine.inspectBlock(name, b);
  } catch (e) {
    $('qmeta').textContent = String(e);
    return;
  }

  const quants = v.quants, values = v.values, scales = v.scales, mins = v.mins;
  const group = v.group, lo = v.quantMin, hi = v.quantMax;

  viz.drawQuants($<HTMLCanvasElement>('qquants'), quants, group, lo, hi);
  viz.drawValues($<HTMLCanvasElement>('qvalues'), values, scales, mins, lo, hi);

  $('qrange').textContent = `quant range ${lo}…${hi} (${Math.ceil(Math.log2(hi - lo + 1))} bits)`;
  $('qmeta').textContent =
    `${info.format} · block ${b} of ${info.blocks} · ${quants.length} weights in ` +
    `${(info.bytes / info.blocks).toFixed(0)} bytes · ${info.bitsPerWeight.toFixed(2)} bits/weight · ` +
    `${scales.length} scale group${scales.length > 1 ? 's' : ''} of ${group}`;

  // Per-group scales, and what each one costs in resolution. `step` is the gap
  // between adjacent representable values, so any weight is stored to within
  // half of it -- the honest answer to "what did quantising cost here".
  const affine = mins.some((m) => m !== 0);
  // Array.from first: `scales` is a Float32Array, and a typed array's `.map()`
  // returns another typed array — so mapping to strings coerces every result
  // back to a number and yields a row of NaN.
  const rows = Array.from(scales).map((sc, g) => {
    const slice = values.subarray(g * group, (g + 1) * group);
    let amax = 0;
    for (const v of slice) amax = Math.max(amax, Math.abs(v));
    const step = Math.abs(sc);
    return `<tr><td>${g}</td><td>${sc.toExponential(3)}</td>` +
      (affine ? `<td>${mins[g].toExponential(3)}</td>` : '') +
      `<td>${step.toExponential(2)}</td><td>${(amax ? (step / 2 / amax) * 100 : 0).toFixed(2)}%</td></tr>`;
  }).join('');
  $('qscales').innerHTML =
    `<table class="scales"><thead><tr><th>group</th><th>scale</th>` +
    (affine ? '<th>min</th>' : '') +
    `<th>step</th><th>max error vs peak</th></tr></thead><tbody>${rows}</tbody></table>`;
}

$<HTMLSelectElement>('qtensor').addEventListener('change', () => { $<HTMLInputElement>('qblock').value = '0'; drawQuant(); });
$<HTMLInputElement>('qblock').addEventListener('change', drawQuant);

function applySampling(): void {
  const engine = globalEngine();
  if (!engine) return;
  const seed = parseInt($<HTMLInputElement>('seed').value, 10);
  // Every path that changes sampling comes through here, so publishing the seed
  // here means the URL is correct no matter how the value was changed.
  if (Number.isFinite(seed)) shareCurrentSeed(seed);
  engine.setSampling(
    parseFloat($<HTMLInputElement>('temp').value),
    parseInt($<HTMLInputElement>('topk').value, 10),
    parseFloat($<HTMLInputElement>('topp').value),
    parseFloat($<HTMLInputElement>('pen').value),
    64,
    seed,
  );
}

/** Read a per-op bar colour from CSS so the bar follows the active theme. */
function opColour(name: string, fallback: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

function showStats(extra = ''): void {
  const engine = globalEngine();
  if (!engine) return;
  const s = engine.stats();
  $('stats').textContent =
    `${s.tokensPerSecond.toFixed(2)} tok/s now (${s.averageTokensPerSecond.toFixed(2)} avg) · ` +
    // Not the same as prefill: the first decode step lands before any character
    // does, so this is the latency a reader actually experiences.
    (s.timeToFirstToken > 0 ? `first token ${(s.timeToFirstToken / 1000).toFixed(2)} s · ` : '') +
    `prefill ${s.prefillTokens} tok in ${(s.prefillMs / 1000).toFixed(2)} s · ` +
    `decode ${s.decodeTokens} tok · ` +
    `kv ${s.cacheUsed}/${s.cacheCapacity} (${fmtBytes(s.cacheBytes)})${extra}`;
}

$<HTMLButtonElement>('run').addEventListener('click', async () => {
  // Bind once: `engine` is module-level `let`, so a narrowing at the top does
  // not survive into the closures below.
  const eng = engine;
  if (!eng || running) return;
  running = true;
  $<HTMLButtonElement>('run').disabled = true;
  $<HTMLButtonElement>('stop').disabled = false;

  const out = $('out');
  out.textContent = '';
  applySampling();

  const raw = $<HTMLTextAreaElement>('prompt').value;
  const useChat = $<HTMLInputElement>('chat').checked;
  const text = useChat ? eng.chatPrompt(raw) : raw;
  const tokens = eng.tokenize(text, useChat);
  // Per-position strings, so the distribution chart can name what a head is
  // looking at rather than just its index.
  promptPieces = Array.from(tokens, (t) => eng.detokenize(new Uint32Array([t])));

  const budget = parseInt($<HTMLInputElement>('ntok').value, 10);
  try {
    eng.prefill(tokens);
  } catch (e) {
    out.textContent = `error: ${e}`;
    running = false;
    $<HTMLButtonElement>('run').disabled = false;
    $<HTMLButtonElement>('stop').disabled = true;
    return;
  }

  let produced = 0;
  const step = () => {
    if (!running || produced >= budget) return finish();
    let r;
    try {
      r = eng.decodeStep();
    } catch (e) {
      out.textContent += `\n\nerror: ${e}`;
      return finish();
    }
    if (r.isEos) return finish();
    out.textContent += r.text;
    promptPieces.push(r.text);
    produced++;
    showStats();
    redrawViz(r);
    // One token per turn of the event loop. This is the whole point: the
    // browser gets to repaint and process the Stop button between every token.
    nextTick(step);
  };

  const finish = () => {
    running = false;
    $<HTMLButtonElement>('run').disabled = false;
    $<HTMLButtonElement>('stop').disabled = true;
    showStats(' · done');
  };

  nextTick(step);
});

$<HTMLButtonElement>('stop').addEventListener('click', () => { running = false; });

$('clear').addEventListener('click', () => {
  if (!engine) return;
  engine.reset();
  $('out').textContent = '';
  $('stats').textContent = '';
});

await init();
await refreshCacheUI();

// A shared `?seed=` link has to actually reproduce the run, which means the
// value must reach the input before the first generation reads it. Rejecting
// non-finite input here rather than letting NaN through to the engine, where it
// would silently become a different seed.
if (URL_SEED !== null) {
  const n = parseInt(URL_SEED, 10);
  if (Number.isFinite(n) && n >= 0) $<HTMLInputElement>('seed').value = String(n);
}

// Preselect the most recently cached model, so a revisit is one click and no
// download. Falling back to the CDN URL only when nothing is cached -- otherwise
// the field would point at a 469 MB download while a local copy sat unused.
const rows = await cache.list().catch(() => []);
const newest = rows.sort((a, b) => b.storedAt - a.storedAt)[0];
$<HTMLInputElement>('url').value = newest ? newest.url : DEFAULT_URL;
status(newest
  ? `cached model ready (${fmtBytes(newest.bytes)}) — press Fetch to load it`
  : 'pick a .gguf file, or fetch one from a URL');
