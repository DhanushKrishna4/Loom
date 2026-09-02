// The three live visualisations.
//
// All canvas, no DOM. A [24 x 14] attention grid is 336 cells redrawn every
// token; as DOM nodes that is 336 style recalculations per token competing with
// the decode loop for the same thread. Canvas draws it in one pass and the
// figures below stay pixel-cheap even at 2048 positions.

const css = (name: string): string => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** Device-pixel-ratio aware sizing, so nothing is blurry on a retina display. */
interface Box { ctx: CanvasRenderingContext2D; w: number; h: number }

function fit(canvas: HTMLCanvasElement): Box | null {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  // Refuse to resize to nothing. A draw that happens before layout reports
  // clientHeight 0, and writing canvas.height = 0 is not merely a no-op -- it
  // discards the backing store, and for a canvas sized by its intrinsic aspect
  // ratio it collapses the element for good. Bail instead; the next draw, after
  // layout, will size it correctly.
  if (w <= 0 || h <= 0) return null;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

/**
 * Perceptually ordered ramp for attention weights.
 *
 * Attention is overwhelmingly concentrated: most heads put nearly all their mass
 * on one or two positions, so a linear ramp renders as one bright dot on black
 * and tells you nothing about the rest. The caller passes normalised values and
 * applies its own gamma; this only maps [0,1] to colour.
 */
function heat(v: number): string {
  const t = Math.max(0, Math.min(1, v));
  // dark slate -> amber -> white
  const r = Math.round(255 * Math.min(1, t * 1.9));
  const g = Math.round(255 * Math.max(0, Math.min(1, t * 1.5 - 0.25)));
  const b = Math.round(255 * Math.max(0, t * t * 0.9) + 26 * (1 - t));
  return `rgb(${r},${g},${b})`;
}

/**
 * Attention heatmap: a [layer x head] grid, each cell shaded by how sharply that
 * head is attending.
 *
 * "Sharply" rather than "how much": every distribution sums to 1, so the useful
 * signal is concentration, not magnitude. A head attending equally to everything
 * is doing something very different from one locked onto a single token, and the
 * sum cannot tell them apart.
 */
export interface HeadSelection { layer: number; head: number }

export function drawHeads(
  canvas: HTMLCanvasElement,
  attn: Float32Array | null,
  nLayers: number,
  nHeads: number,
  maxSeq: number,
  len: number,
  selected: HeadSelection | null,
): void {
  const box = fit(canvas);
  if (!box) return;
  const { ctx, w, h } = box;
  ctx.clearRect(0, 0, w, h);
  if (!attn || !len) return;

  const cw = w / nHeads;
  const ch = h / nLayers;
  for (let l = 0; l < nLayers; l++) {
    for (let hd = 0; hd < nHeads; hd++) {
      const base = (l * nHeads + hd) * maxSeq;
      // Max probability = how concentrated this head is. 1/len is uniform.
      let peak = 0;
      for (let i = 0; i < len; i++) peak = Math.max(peak, attn[base + i]);
      // Rescale against uniform so short contexts are not all bright.
      //
      // No gamma correction: measured across all 336 heads on a real prompt,
      // this quantity already spans the range (min 0.06, quartiles 0.29 / 0.48 /
      // 0.75, max 1.00), so a curve would only distort a well-spread signal. An
      // earlier 0.55 washed the whole grid out.
      const norm = len > 1 ? (peak - 1 / len) / (1 - 1 / len) : 1;
      ctx.fillStyle = heat(Math.max(0, norm));
      ctx.fillRect(hd * cw, l * ch, Math.ceil(cw) - 0.5, Math.ceil(ch) - 0.5);
    }
  }

  if (selected) {
    ctx.strokeStyle = css('--accent') || '#e08b52';
    ctx.lineWidth = 2;
    ctx.strokeRect(selected.head * cw, selected.layer * ch, cw, ch);
  }
}

/** The selected head's full distribution over the context, as a bar chart. */
export function drawDistribution(
  canvas: HTMLCanvasElement,
  weights: Float32Array | number[] | null,
  tokens: string[],
): void {
  const box = fit(canvas);
  if (!box) return;
  const { ctx, w, h } = box;
  ctx.clearRect(0, 0, w, h);
  if (!weights || !weights.length) return;

  const n = weights.length;
  const bw = w / n;
  let peak = 0;
  for (const v of weights) peak = Math.max(peak, v);
  if (peak <= 0) return;

  ctx.fillStyle = css('--accent') || '#e08b52';
  for (let i = 0; i < n; i++) {
    const bh = (weights[i] / peak) * (h - 14);
    ctx.fillRect(i * bw, h - bh, Math.max(1, bw - 0.5), bh);
  }

  // Label the strongest position, which is the one worth reading.
  let argmax = 0;
  for (let i = 1; i < n; i++) if (weights[i] > weights[argmax]) argmax = i;
  ctx.fillStyle = css('--dim') || '#888';
  ctx.font = '10px ui-monospace, monospace';
  const label = tokens && tokens[argmax] !== undefined
    ? `${argmax}: ${JSON.stringify(tokens[argmax]).slice(0, 18)} (${(weights[argmax] * 100).toFixed(0)}%)`
    : `${argmax} (${(weights[argmax] * 100).toFixed(0)}%)`;
  ctx.fillText(label, 2, 10);
}

/** KV cache occupancy: positions used against allocated. */
export function drawCache(
  canvas: HTMLCanvasElement,
  used: number,
  capacity: number,
  bytes: number,
): void {
  const box = fit(canvas);
  if (!box) return;
  const { ctx, w, h } = box;
  ctx.clearRect(0, 0, w, h);

  const barH = h - 16;
  ctx.fillStyle = css('--line') || '#333';
  ctx.fillRect(0, 0, w, barH);

  const frac = capacity ? used / capacity : 0;
  ctx.fillStyle = css('--accent') || '#e08b52';
  ctx.fillRect(0, 0, w * frac, barH);

  // Tick every 256 positions, so the scale is readable rather than implied.
  ctx.fillStyle = css('--bg') || '#000';
  for (let p = 256; p < capacity; p += 256) {
    ctx.fillRect((p / capacity) * w, 0, 1, barH);
  }

  ctx.fillStyle = css('--dim') || '#888';
  ctx.font = '10px ui-monospace, monospace';
  const mb = (bytes / (1024 * 1024)).toFixed(1);
  ctx.fillText(`${used} / ${capacity} positions · ${mb} MiB allocated · ${(frac * 100).toFixed(1)}% used`, 0, h - 3);
}

/**
 * Where the last token's time went, as a stacked bar.
 *
 * Fed from the engine's own per-op timings rather than wall clock around the
 * call, so "overhead" genuinely means sampling plus detokenisation plus the
 * boundary crossing, and not "everything I forgot to instrument".
 */
export interface PerfSlice { label: string; ms: number; color: string }

export function drawPerf(canvas: HTMLCanvasElement, parts: PerfSlice[]): void {
  const box = fit(canvas);
  if (!box) return;
  const { ctx, w, h } = box;
  ctx.clearRect(0, 0, w, h);

  const total = parts.reduce((a, p) => a + p.ms, 0);
  if (total <= 0) return;

  const barH = h - 16;
  let x = 0;
  for (const p of parts) {
    const pw = (p.ms / total) * w;
    ctx.fillStyle = p.color;
    ctx.fillRect(x, 0, pw, barH);
    x += pw;
  }

  ctx.font = '10px ui-monospace, monospace';
  ctx.fillStyle = css('--dim') || '#888';
  const legend = parts
    .filter((p) => p.ms / total > 0.005)
    .map((p) => `${p.label} ${((p.ms / total) * 100).toFixed(0)}%`)
    .join('  ·  ');
  ctx.fillText(`${total.toFixed(1)} ms/token — ${legend}`, 0, h - 3);
}

// ---------------------------------------------------------------- quant ----

/** Diverging ramp for signed quants: blue for negative, amber for positive. */
function signed(v: number): string {
  const t = Math.max(-1, Math.min(1, v));
  if (t >= 0) return `rgb(${Math.round(60 + 195 * t)},${Math.round(50 + 110 * t)},${Math.round(60 - 30 * t)})`;
  return `rgb(${Math.round(60 + 30 * t)},${Math.round(70 + 40 * t)},${Math.round(70 - 150 * t)})`;
}

/** The block's stored integers, one cell each, laid out in rows of `group`. */
export function drawQuants(
  canvas: HTMLCanvasElement,
  quants: Int32Array | number[],
  group: number,
  lo: number,
  hi: number,
): void {
  const box = fit(canvas);
  if (!box) return;
  const { ctx, w, h } = box;
  ctx.clearRect(0, 0, w, h);
  if (!quants || !quants.length) return;

  const rows = Math.ceil(quants.length / group);
  const cw = w / group;
  const ch = h / rows;
  // Normalise against the format's range, not the block's, so the picture is
  // comparable between blocks and shows when a block wastes its range.
  const span = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
  for (let i = 0; i < quants.length; i++) {
    const r = Math.floor(i / group);
    const c = i % group;
    // Q4_K's quants are unsigned 0..15, so centre them for display.
    const centred = lo >= 0 ? (quants[i] - (hi + lo) / 2) / ((hi - lo) / 2 || 1) : quants[i] / span;
    ctx.fillStyle = signed(centred);
    ctx.fillRect(c * cw, r * ch, Math.ceil(cw) - 0.5, Math.ceil(ch) - 0.5);
  }
}

/**
 * Reconstructed values against the grid of levels they can occupy.
 *
 * The point of the picture: every weight lands exactly on a line. The spacing
 * between lines is the resolution the format bought, and it is visible rather
 * than asserted.
 */
export function drawValues(
  canvas: HTMLCanvasElement,
  values: Float32Array,
  scales: Float32Array,
  mins: Float32Array,
  lo: number,
  hi: number,
): void {
  const box = fit(canvas);
  if (!box) return;
  const { ctx, w, h } = box;
  ctx.clearRect(0, 0, w, h);
  if (!values || !values.length) return;

  let vmin = Infinity;
  let vmax = -Infinity;
  for (const v of values) { vmin = Math.min(vmin, v); vmax = Math.max(vmax, v); }
  const pad = (vmax - vmin) * 0.08 || 1e-6;
  vmin -= pad; vmax += pad;
  const y = (v: number): number => h - ((v - vmin) / (vmax - vmin)) * h;

  // The representable levels for the first group. Drawing every group's grid
  // would be unreadable; the first is representative and the table below has
  // the rest.
  const nLevels = hi - lo + 1;
  if (nLevels <= 64) {
    ctx.strokeStyle = css('--line') || '#333';
    ctx.lineWidth = 1;
    for (let q = lo; q <= hi; q++) {
      const level = scales[0] * q - mins[0];
      if (level < vmin || level > vmax) continue;
      ctx.beginPath();
      ctx.moveTo(0, y(level));
      ctx.lineTo(w, y(level));
      ctx.stroke();
    }
  }

  ctx.fillStyle = css('--accent') || '#e08b52';
  const dx = w / values.length;
  for (let i = 0; i < values.length; i++) {
    ctx.fillRect(i * dx, y(values[i]) - 1, Math.max(1, dx - 0.4), 2);
  }
}
