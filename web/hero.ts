/**
 * The hero panel's attention matrix.
 *
 * This draws a causal attention matrix, which is the shape the engine actually
 * computes: row `q` is a query position, column `k` a key position, and the
 * upper triangle does not exist because a token cannot attend to its own
 * future. Rows fill in one at a time because that is how decoding works -- one
 * token, one row, the cache one entry longer.
 *
 * The structure is synthetic but not arbitrary. Real maps from this model show
 * three things, and all three are here:
 *   - a bright column at position 0 (the "attention sink" -- the first token
 *     soaks up probability mass that a head has nowhere better to put). The
 *     engine's own test asserts that at position 0 every head attends to itself
 *     with weight exactly 1.
 *   - a strong diagonal band, because heads attend locally.
 *   - diffuse mid-range structure.
 * Each row is normalised to sum to 1, so brightness reads as concentration --
 * the same property the live heatmap in section 3 visualises.
 */

const css = (name: string): string =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** Deterministic hash -> [0,1). Seeded so the field is stable across resizes. */
function noise(x: number, y: number): number {
  const n = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return n - Math.floor(n);
}

export function startHero(canvas: HTMLCanvasElement): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const PAD = 14;

  let n = 0;           // the matrix is n x n, so the diagonal is a true 45 degrees
  let cell = 0;        // css px per cell, including its gap
  let ox = 0, oy = 0;  // top-left of the matrix, centred in the panel
  let w = 0, h = 0;
  let field = new Float32Array(0);

  function build(): void {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect();
    w = Math.max(1, Math.round(r.width));
    h = Math.max(1, Math.round(r.height));
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);

    const side = Math.max(40, Math.min(w, h) - PAD * 2);
    n = Math.max(12, Math.min(46, Math.round(side / 13)));
    cell = side / n;
    ox = (w - side) / 2;
    oy = (h - side) / 2;

    field = new Float32Array(n * n);
    for (let q = 0; q < n; q++) {
      let sum = 0;
      const base = q * n;
      for (let k = 0; k <= q; k++) {
        const sink = k === 0 ? 0.9 : 0;                        // attention sink
        const d = q - k;
        const local = Math.exp(-(d * d) / 26) * 1.15;          // local band
        const diffuse = noise(k, q) * noise(q * 0.37, k * 0.71) * 0.5;
        const v = sink + local + diffuse;
        field[base + k] = v;
        sum += v;
      }
      if (sum > 0) for (let k = 0; k <= q; k++) field[base + k] /= sum;
    }
  }

  // The visible row count sweeps 0 -> n, holds, then restarts: a generation
  // running to completion and a new one beginning.
  const PERIOD = 9000;
  const HOLD = 0.24;
  let raf = 0;

  function draw(t: number): void {
    const accent = css('--accent') || '#4fe0bc';
    const line = css('--line') || '#1a252a';
    ctx!.clearRect(0, 0, w, h);

    let progress = 1;
    if (!reduced) {
      const p = (t % PERIOD) / PERIOD;
      progress = Math.min(1, p / (1 - HOLD));
      progress = 1 - Math.pow(1 - progress, 2.2);   // decelerate as context fills
    }
    const visible = progress * n;
    const g = Math.max(1, cell * 0.14);             // gap scales with the cell
    const s = cell - g;

    // The empty upper triangle is drawn as a faint grid so the causal mask is
    // visible as a *shape*, not just an absence.
    ctx!.strokeStyle = line;
    ctx!.lineWidth = 1;
    ctx!.globalAlpha = 0.5;
    ctx!.beginPath();
    ctx!.moveTo(ox, oy);
    ctx!.lineTo(ox + n * cell, oy + n * cell);
    ctx!.stroke();
    ctx!.globalAlpha = 1;

    for (let q = 0; q < n; q++) {
      if (q > visible) break;
      const age = visible - q;
      const fresh = age < 2.5 ? 1 + (2.5 - age) * 0.55 : 1;   // newest rows glow
      const y = oy + q * cell;
      const base = q * n;

      for (let k = 0; k <= q; k++) {
        const v = field[base + k];
        if (v <= 0.0005) continue;
        // sqrt so the long tail of small weights stays visible; a linear ramp
        // would show only the diagonal and the sink.
        ctx!.globalAlpha = Math.min(0.95, Math.sqrt(v) * 1.75 * fresh);
        ctx!.fillStyle = accent;
        ctx!.fillRect(ox + k * cell, y, s, s);
      }
    }
    ctx!.globalAlpha = 1;

    if (!reduced) raf = requestAnimationFrame(draw);
  }

  function restart(): void {
    build();
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(draw);
  }

  restart();

  // ResizeObserver rather than window.resize: the panel's size depends on its
  // own content reflowing, which a window resize event does not always follow.
  if ('ResizeObserver' in window) {
    let pending = 0;
    new ResizeObserver(() => {
      clearTimeout(pending);
      pending = setTimeout(restart, 120) as unknown as number;
    }).observe(canvas);
  } else {
    addEventListener('resize', restart);
  }

  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', restart);
}

/** Fade sections in as they arrive. Silently does nothing if unsupported. */
export function revealOnScroll(): void {
  const els = document.querySelectorAll<HTMLElement>('.reveal');
  if (!('IntersectionObserver' in window) || matchMedia('(prefers-reduced-motion: reduce)').matches) {
    els.forEach((e) => e.classList.add('in'));
    return;
  }
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          e.target.classList.add('in');
          io.unobserve(e.target);
        }
      }
    },
    { rootMargin: '0px 0px -8% 0px', threshold: 0.05 },
  );
  els.forEach((e) => io.observe(e));
}
