// Interactive widgets for the "Convolutional Tsetlin machines" lesson.
// Widgets, each mounted on a <div> with a matching id:
//   #viz-shapes   – the eight Circle/Cross images: eight position-specific rules vs two window rules
//   #viz-slide    – slide a 3×3 window over an image, match a rule on every patch, OR the results
//   #viz-learn    – train a convolutional Tsetlin machine on "Circle or Cross, anywhere in a 6×6 image"
//   #viz-position – thermometer-coded window positions make a rule location-sensitive
// The learning code (CTM below) is a line-by-line port of the lesson's Python check script, with the same
// seeded random numbers, so the widget reproduces the numbers quoted in the lesson exactly.

const CTM = (function () {
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = Math.imul(a ^ (a >>> 15), a | 1);
      t = (t + Math.imul(t ^ (t >>> 7), t | 61)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // The chapter's two shapes, as offsets from the centre pixel. Class 0 = Circle, class 1 = Cross.
  const CIRCLE = [[-1, 0], [0, -1], [0, 1], [1, 0]];
  const CROSS = CIRCLE.concat([[0, 0]]);
  const SHAPES = [CIRCLE, CROSS];

  // An image is {h, w, px: Uint8Array(h*w)} with 1 = black.
  function blank(h, w) { return { h, w, px: new Uint8Array(h * w) }; }
  function drawShape(img, shape, r, c) { for (const [dr, dc] of shape) img.px[(r + dr) * img.w + c + dc] = 1; }

  // n images size×size with one shape and `noise` extra black pixels outside the shape's 3×3 box.
  // label "shape": Circle (0) or Cross (1) anywhere; "half": a Circle in the upper (0) or lower (1) half.
  function makeDataset(n, size, noise, rand, cols, label = "shape") {
    const X = [], y = [];
    const centres = []; for (let i = 1; i < size - 1; i++) centres.push(i);
    cols = cols || centres;
    for (let i = 0; i < n; i++) {
      const cls = rand() < 0.5 ? 0 : 1;
      let shape, r;
      if (label === "shape") { shape = SHAPES[cls]; r = centres[Math.floor(rand() * centres.length)]; }
      else {
        shape = CIRCLE;
        const half = Math.floor(centres.length / 2);
        r = centres[Math.floor(rand() * half) + (cls ? half : 0)];
      }
      const c = cols[Math.floor(rand() * cols.length)];
      const img = blank(size, size);
      drawShape(img, shape, r, c);
      let k = 0;
      while (k < noise) {
        const rr = Math.floor(rand() * size), cc = Math.floor(rand() * size);
        if ((Math.abs(rr - r) <= 1 && Math.abs(cc - c) <= 1) || img.px[rr * size + cc]) continue;
        img.px[rr * size + cc] = 1;
        k++;
      }
      X.push(img); y.push(cls);
    }
    return { X, y };
  }

  // Slide a win×win window (top-left corner at column x, row y). Each patch: its pixel features, optionally
  // thermometer-coded position (X≥1 … , Y≥1 …), then the negation of every feature.
  function patches(img, win, pos) {
    const out = [], where = [];
    for (let y = 0; y + win <= img.h; y++) {
      for (let x = 0; x + win <= img.w; x++) {
        const f = [];
        for (let i = 0; i < win; i++) for (let j = 0; j < win; j++) f.push(img.px[(y + i) * img.w + x + j]);
        if (pos) {
          for (let t = 1; t <= img.w - win; t++) f.push(x >= t ? 1 : 0);
          for (let t = 1; t <= img.h - win; t++) f.push(y >= t ? 1 : 0);
        }
        const lit = new Uint8Array(2 * f.length);
        f.forEach((v, k) => { lit[k] = v; lit[f.length + k] = 1 - v; });
        out.push(lit); where.push([x, y]);
      }
    }
    return { lits: out, where };
  }

  class Machine {
    // n rules per class; rules 0…n−1 vote for class 0, rules n…2n−1 for class 1.
    // Memory depth = 2·half positions per literal (1 … 2·half); Memorized = upper half. Start at half.
    constructor(L, n, T, forget, half, rand) {
      Object.assign(this, { L, n, T, F: forget, M: 1 - forget, N: half, rand });
      this.state = Array.from({ length: 2 * n }, () => new Int16Array(L).fill(half));
      this.cls = Array.from({ length: 2 * n }, (_, j) => (j < n ? 0 : 1));
    }
    // does rule j match patch literals p? (no memorized literal may be False)
    matches(j, p) {
      const s = this.state[j], N = this.N;
      for (let k = 0; k < this.L; k++) if (s[k] > N && !p[k]) return false;
      return true;
    }
    matchTable(P) { return this.state.map((_, j) => P.map((p) => this.matches(j, p))); }
    outputs(P) { return this.matchTable(P).map((row) => row.some(Boolean)); }       // OR over patches
    vote(P) {                                                                        // v for class 0
      const out = this.outputs(P);
      let v = 0;
      out.forEach((o, j) => { if (o) v += this.cls[j] === 0 ? 1 : -1; });
      return v;
    }
    predict(P) { return this.vote(P) >= 0 ? 0 : 1; }                                 // tie → first class
    update(P, y, log) {
      const { rand, T, L, N } = this;
      const m = this.matchTable(P), out = m.map((row) => row.some(Boolean));
      let v = 0;
      out.forEach((o, j) => { if (o) v += this.cls[j] === y ? 1 : -1; });
      const raw = v;
      v = Math.max(-T, Math.min(T, v));
      const prob = (T - v) / (2 * T);
      if (log) { log.v = raw; log.prob = prob; log.out = out; log.m = m; log.fb = []; }
      for (let j = 0; j < this.state.length; j++) {
        if (rand() > prob) { if (log) log.fb.push({ type: "none" }); continue; }
        const s = this.state[j];
        if (this.cls[j] === y) {
          if (out[j]) {                                   // Recognize Feedback on one random matching patch
            const idx = []; m[j].forEach((b, i) => { if (b) idx.push(i); });
            const pick = idx[Math.floor(rand() * idx.length)], p = P[pick];
            for (let k = 0; k < L; k++) {
              const u = rand();
              if (p[k]) { if (u <= this.M && s[k] < 2 * N) s[k]++; }
              else if (u <= this.F && s[k] > 1) s[k]--;
            }
            if (log) log.fb.push({ type: "recognize", patch: pick, of: idx.length });
          } else {                                          // Erase Feedback: no patch needed
            for (let k = 0; k < L; k++) if (rand() <= this.F && s[k] > 1) s[k]--;
            if (log) log.fb.push({ type: "erase" });
          }
        } else if (out[j]) {                                // Reject Feedback on one random matching patch
          const idx = []; m[j].forEach((b, i) => { if (b) idx.push(i); });
          const pick = idx[Math.floor(rand() * idx.length)], p = P[pick];
          for (let k = 0; k < L; k++) if (!p[k] && s[k] <= N) s[k]++;
          if (log) log.fb.push({ type: "reject", patch: pick, of: idx.length });
        } else if (log) log.fb.push({ type: "none" });
      }
    }
    // memorized literals of rule j as a picture: per pixel "on" / "off" / "any" / "both", plus extra literals
    picture(j, win) {
      const s = this.state[j], half = this.L / 2, N = this.N;
      const cells = [];
      for (let k = 0; k < win * win; k++) {
        const on = s[k] > N, off = s[half + k] > N;
        cells.push(on && off ? "both" : on ? "on" : off ? "off" : "any");
      }
      const extra = [];
      for (let k = win * win; k < half; k++) extra.push([s[k] > N, s[half + k] > N]);
      return { cells, extra };
    }
  }

  function accuracy(tm, Ps, y) {
    let ok = 0;
    Ps.forEach((P, i) => { if (tm.predict(P) === y[i]) ok++; });
    return ok / Ps.length;
  }

  function shuffledOrder(n, rand) {
    const order = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) { const k = Math.floor(rand() * (i + 1)); [order[i], order[k]] = [order[k], order[i]]; }
    return order;
  }

  // One experiment exactly as in the Python script: data, machine and training share one RNG.
  function experiment({ seed = 1, win = 3, task = "shape", holdout = false, pos = false, n = 4, T = 4, forget = 0.7, half = 5 }) {
    const rand = mulberry32(seed);
    let tr, te;
    if (task === "shape") {
      tr = makeDataset(300, 6, 2, rand, holdout ? [1, 2] : null);
      te = makeDataset(300, 6, 2, rand, holdout ? [3, 4] : null);
    } else {
      tr = makeDataset(300, 6, 2, rand, null, "half");
      te = makeDataset(300, 6, 2, rand, null, "half");
    }
    const Ptr = tr.X.map((im) => patches(im, win, pos).lits), Pte = te.X.map((im) => patches(im, win, pos).lits);
    const tm = new Machine(Ptr[0][0].length, n, T, forget, half, rand)   // memory depth 10, as in the book;
    return { rand, tr, te, Ptr, Pte, tm, win, pos, history: [] };
  }

  function trainEpoch(ex) {
    for (const i of shuffledOrder(ex.Ptr.length, ex.rand)) ex.tm.update(ex.Ptr[i], ex.tr.y[i]);
    ex.history.push([accuracy(ex.tm, ex.Ptr, ex.tr.y), accuracy(ex.tm, ex.Pte, ex.te.y)]);
  }

  return { mulberry32, CIRCLE, CROSS, SHAPES, blank, drawShape, makeDataset, patches, Machine, accuracy, shuffledOrder, experiment, trainEpoch };
})();

if (typeof module !== "undefined") module.exports = CTM;

if (typeof document !== "undefined") document.addEventListener("DOMContentLoaded", function () {
  const { colors, onThemeChange, linear, el, svg, axes, ticks, fmt } = window.Viz;
  const textStyle = "fill: var(--viz-text)";
  const pct = (a) => (100 * a).toFixed(1) + "%";

  // ---------------------------------------------------------------- Shared helpers
  function control(parent, html) {
    const div = document.createElement("div");
    div.className = "viz-controls";
    div.innerHTML = html;
    parent.appendChild(div);
    return div;
  }
  function readout(parent) {
    const div = document.createElement("div");
    div.className = "viz-readout";
    div.style.whiteSpace = "pre-line";
    parent.appendChild(div);
    return div;
  }
  function hint(parent, text) {
    const div = document.createElement("div");
    div.className = "viz-hint";
    div.innerHTML = text;
    parent.appendChild(div);
    return div;
  }
  function svgIn(root, W, H, maxW) {
    const s = svg(W, H);
    s.style.maxWidth = (maxW || W) + "px";
    s.style.margin = "0 auto";
    s.style.display = "block";
    root.appendChild(s);
    return s;
  }
  const label = (g, x, y, txt, opts = {}) => {
    const t = el("text", Object.assign({ x, y, "text-anchor": "middle" }, opts), g);
    t.textContent = txt;
    return t;
  };
  const isNarrow = (root) => root.clientWidth < 480;

  // Draw a binary image; returns the cell size. opts: {cell, onClick(r, c)}
  function drawImage(g, img, x0, y0, cell, opts = {}) {
    const c = colors();
    for (let r = 0; r < img.h; r++) for (let q = 0; q < img.w; q++) {
      const on = img.px[r * img.w + q];
      const rect = el("rect", { x: x0 + q * cell, y: y0 + r * cell, width: cell, height: cell,
        fill: on ? c.text : c.bg, "fill-opacity": on ? 0.82 : 1, stroke: c.border }, g);
      if (opts.onClick) { rect.style.cursor = "pointer"; rect.addEventListener("click", () => opts.onClick(r, q)); }
    }
  }

  // Draw a rule as a little picture: must be black = dark cell with green border, must be white = light cell
  // with a green ring, don't care = faint grey, contradiction (both) = red.
  function drawStencil(g, cells, win, x0, y0, cell, opts = {}) {
    const c = colors();
    cells.forEach((kind, k) => {
      const r = Math.floor(k / win), q = k % win, x = x0 + q * cell, y = y0 + r * cell;
      const fill = kind === "on" ? c.text : kind === "both" ? c.grad : c.bg;
      const rect = el("rect", { x, y, width: cell, height: cell, fill, "fill-opacity": kind === "on" ? 0.82 : kind === "both" ? 0.5 : 1,
        stroke: c.border }, g);
      if (kind === "any") el("rect", { x: x + 1, y: y + 1, width: cell - 2, height: cell - 2, fill: c.muted, "fill-opacity": 0.12 }, g);
      if (kind === "on" || kind === "off") {
        el("rect", { x: x + 1.5, y: y + 1.5, width: cell - 3, height: cell - 3, fill: "none", stroke: c.opt, "stroke-width": Math.max(1.5, cell / 10) }, g);
        if (kind === "off") el("circle", { cx: x + cell / 2, cy: y + cell / 2, r: cell * 0.2, fill: "none", stroke: c.opt, "stroke-width": Math.max(1.2, cell / 14) }, g);
      }
      if (opts.onClick) { rect.style.cursor = "pointer"; rect.addEventListener("click", () => opts.onClick(k)); }
    });
    el("rect", { x: x0, y: y0, width: win * cell, height: win * cell, fill: "none", stroke: c.model, "stroke-width": 2 }, g);
  }

  function stencilLegend(parent) {
    return hint(parent, "Rule pictures: <b>filled cell, green border</b> = pixel must be black (x<sub>i,j</sub> memorized) · " +
      "<b>empty cell with a green ring</b> = pixel must be white (¬x<sub>i,j</sub> memorized) · <b>grey</b> = don't care (both forgotten). " +
      "(In dark mode, black pixels are drawn light.)");
  }

  // ---------------------------------------------------------------- Widget 0: the eight images
  const BOOK_CENTRES = [[2, 1], [1, 1], [1, 2], [2, 2]];
  function bookImage(cls, i) {
    const img = CTM.blank(4, 4);
    CTM.drawShape(img, CTM.SHAPES[cls], BOOK_CENTRES[i][0], BOOK_CENTRES[i][1]);
    return img;
  }
  const RULE_CELLS = [  // R5 (Circle) and R6 (Cross) from the chapter, as 3×3 pictures
    ["any", "on", "any", "on", "off", "on", "any", "on", "any"],
    ["any", "on", "any", "on", "on", "on", "any", "on", "any"],
  ];

  function shapesWidget(root) {
    const ctrl = control(root, `<button data-m="brute">Without a window: one rule per position</button><button data-m="window">With a 3×3 window: one rule per shape</button>`);
    let mode = "brute", s = null;
    const out = readout(root);
    stencilLegend(root);

    function draw() {
      if (s) s.remove();
      const narrow = isNarrow(root);
      const perRow = narrow ? 2 : 4, cell = narrow ? 20 : 18, gap = narrow ? 32 : 26;
      const imgW = 4 * cell, colW = imgW + gap, left = narrow ? 70 : 64;
      const rowsPerClass = 4 / perRow;
      const blockH = 4 * cell + 14 + 4 * (cell * 0.8) + 30;
      const W = left + perRow * colW, H = 2 * rowsPerClass * blockH + 10;
      s = svg(W, H);
      s.style.maxWidth = W + "px"; s.style.margin = "0 auto"; s.style.display = "block";
      root.insertBefore(s, ctrl);
      const c = colors();
      for (let cls = 0; cls < 2; cls++) {
        for (let i = 0; i < 4; i++) {
          const row = cls * rowsPerClass + Math.floor(i / perRow), col = i % perRow;
          const x = left + col * colW, y = 10 + row * blockH;
          if (col === 0 && (i === 0)) label(s, 30, y + 2 * cell + 4, cls ? "Cross" : "Circle", { style: textStyle + "; font-weight: 600" });
          const img = bookImage(cls, i);
          drawImage(s, img, x, y, cell);
          const sc = cell * 0.8, sy = y + 4 * cell + 14;
          if (mode === "brute") {
            // the position-specific rule: the 3×3 shape stencil placed at its exact pixels in a 4×4 grid
            const cells = Array(16).fill("any");
            const [r0, c0] = BOOK_CENTRES[i];
            RULE_CELLS[cls].forEach((k, idx) => { if (k !== "any") cells[(r0 - 1 + Math.floor(idx / 3)) * 4 + c0 - 1 + (idx % 3)] = k; });
            drawStencil(s, cells, 4, x + (imgW - 4 * sc) / 2, sy, sc);
            label(s, x + imgW / 2, sy + 4 * sc + 13, `rule ${cls ? "C" : "R"}${i + 1}`, {});
          } else {
            const [r0, c0] = BOOK_CENTRES[i];
            el("rect", { x: x + (c0 - 1) * cell, y: y + (r0 - 1) * cell, width: 3 * cell, height: 3 * cell, fill: "none", stroke: c.model, "stroke-width": 3 }, s);
            drawStencil(s, RULE_CELLS[cls], 3, x + (imgW - 3 * sc) / 2, sy, sc);
            label(s, x + imgW / 2, sy + 4 * sc + 13, cls ? "R6 (same)" : "R5 (same)", {});
          }
        }
      }
      out.textContent = mode === "brute"
        ? "Each rule looks at all 16 pixels, so a shape in a new place needs a new rule: 4 for Circle + 4 for Cross = 8 rules.\nA bigger image would need even more (one per possible position)."
        : "Each rule looks only through the 3×3 window (orange). The same two rules, R5 and R6, cover all eight images:\nthe rule doesn't care where the window is, only what it sees inside it.";
      ctrl.querySelectorAll("button").forEach((b) => (b.style.borderColor = b.dataset.m === mode ? "var(--viz-model)" : ""));
    }
    ctrl.addEventListener("click", (e) => { if (e.target.dataset.m) { mode = e.target.dataset.m; draw(); } });
    onThemeChange(draw);
    window.addEventListener("resize", draw);
    draw();
  }

  // ---------------------------------------------------------------- Widget 1: slide the window
  function bookExample() {           // the input image of the chapter's Figure 4.3
    const img = CTM.blank(4, 4);
    [[1, 1], [2, 0], [2, 2], [3, 1]].forEach(([r, c]) => (img.px[r * 4 + c] = 1));
    return img;
  }
  function slideWidget(root) {
    const PRESETS = {
      book: () => bookExample(),
      cross: () => bookImage(1, 2),
      big: () => { const im = CTM.blank(6, 6); CTM.drawShape(im, CTM.CIRCLE, 3, 4); CTM.drawShape(im, CTM.CROSS, 2, 1); im.px[0] = 1; im.px[5 * 6 + 2] = 1; return im; },
      seq: () => { const im = CTM.blank(3, 8); [[0, 1], [1, 0], [1, 2], [2, 1], [0, 5], [1, 4], [1, 5], [1, 6], [2, 5]].forEach(([r, c]) => (im.px[r * 8 + c] = 1)); return im; },
    };
    const ctrl1 = control(root, `Image: <button data-i="book">Chapter's example (4×4)</button><button data-i="cross">A Cross (4×4)</button><button data-i="big">Bigger image (6×6)</button><button data-i="seq">1-D sequence (3 × 8)</button><button data-i="clear">Clear</button>`);
    const ctrl2 = control(root, `Rule: <button data-r="0">R5: Circle</button><button data-r="1">R6: Cross</button><button data-r="empty">Empty rule</button>
      <button data-a="prev">◀ window</button><button data-a="next">window ▶</button><button data-a="play">▶ Slide</button>`);
    hint(root, "Click pixels in the image to paint them. Click cells of the patch (middle) to edit the rule: must be black → must be white → don't care. Click the match map to move the window.");
    hint(root, "In the patch, the rule's literals are drawn on top: <b>small square</b> = must be black, <b>ring</b> = must be white; <b>green</b> = satisfied in this patch, <b>red</b> = violated. A green dot in the image marks every window where the whole rule matches.");
    const out = readout(root);
    let img = PRESETS.book(), rule = RULE_CELLS[0].slice(), cur = 0, timer = null, s = null;

    function nPos() { return { nx: img.w - 2, ny: img.h - 2 }; }
    function matchAt(x, y) {
      for (let k = 0; k < 9; k++) {
        const v = img.px[(y + Math.floor(k / 3)) * img.w + x + (k % 3)];
        if ((rule[k] === "on" && !v) || (rule[k] === "off" && v)) return false;
      }
      return true;
    }
    function stop() { clearInterval(timer); timer = null; }

    function draw() {
      if (s) s.remove();
      const c = colors(), narrow = isNarrow(root);
      const { nx, ny } = nPos();
      const total = nx * ny;
      cur = Math.min(cur, total - 1);
      const cx = cur % nx, cy = Math.floor(cur / nx);
      // layout
      const imgCell = Math.min(narrow ? 30 : 36, (narrow ? 150 : 200) / Math.max(img.w, img.h * 1.0));
      const pCell = narrow ? 34 : 40, mCell = Math.min(narrow ? 34 : 40, (narrow ? 150 : 160) / Math.max(nx, ny));
      let W, H, ix, iy, px, py, mx, my;
      if (!narrow) {
        ix = 12; iy = 30; px = ix + img.w * imgCell + 40; py = 30; mx = px + 3 * pCell + 48; my = 30;
        W = mx + nx * mCell + 12; H = Math.max(iy + img.h * imgCell, py + 3 * pCell + 20, my + ny * mCell) + 26;
        W = Math.max(W, 560);
      } else {
        ix = 10; iy = 30; px = ix + img.w * imgCell + 30; py = 30;
        W = Math.max(340, px + 3 * pCell + 10);
        mx = (W - nx * mCell) / 2; my = Math.max(iy + img.h * imgCell, py + 3 * pCell + 20) + 40;
        H = my + ny * mCell + 22;
      }
      s = svg(W, H);
      s.style.maxWidth = W + "px"; s.style.margin = "0 auto"; s.style.display = "block";
      root.insertBefore(s, ctrl1);
      label(s, ix + (img.w * imgCell) / 2, 18, `image (${img.h}×${img.w})`, { style: textStyle });
      drawImage(s, img, ix, iy, imgCell, { onClick: (r, q) => { stop(); img.px[r * img.w + q] ^= 1; draw(); } });
      // every window that matches gets a green dot at its centre
      for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++)
        if (matchAt(x, y)) el("circle", { cx: ix + (x + 1.5) * imgCell, cy: iy + (y + 1.5) * imgCell, r: imgCell * 0.16, fill: c.opt, stroke: c.bg, "stroke-width": 1.5 }, s);
      el("rect", { x: ix + cx * imgCell, y: iy + cy * imgCell, width: 3 * imgCell, height: 3 * imgCell, fill: "none", stroke: c.model, "stroke-width": 3 }, s);

      // the current patch, with the rule drawn on top of it
      label(s, px + 1.5 * pCell, 18, `patch at (x=${cx}, y=${cy})`, { style: textStyle });
      const patch = CTM.blank(3, 3);
      for (let k = 0; k < 9; k++) patch.px[k] = img.px[(cy + Math.floor(k / 3)) * img.w + cx + (k % 3)];
      drawImage(s, patch, px, py, pCell);
      for (let k = 0; k < 9; k++) {
        const r = Math.floor(k / 3), q = k % 3, x = px + q * pCell, y = py + r * pCell, v = patch.px[k];
        const hit = el("rect", { x, y, width: pCell, height: pCell, fill: "transparent" }, s);
        hit.style.cursor = "pointer";
        hit.addEventListener("click", () => { stop(); rule[k] = rule[k] === "on" ? "off" : rule[k] === "off" ? "any" : "on"; draw(); });
        if (rule[k] === "any") continue;
        const ok = rule[k] === "on" ? v === 1 : v === 0, col = ok ? c.opt : c.grad;
        if (rule[k] === "on") el("rect", { x: x + pCell * 0.3, y: y + pCell * 0.3, width: pCell * 0.4, height: pCell * 0.4, fill: col, stroke: c.bg, "stroke-width": 1.5, "pointer-events": "none" }, s);
        else el("circle", { cx: x + pCell / 2, cy: y + pCell / 2, r: pCell * 0.22, fill: "none", stroke: col, "stroke-width": 4, "pointer-events": "none" }, s);
      }
      el("rect", { x: px, y: py, width: 3 * pCell, height: 3 * pCell, fill: "none", stroke: c.model, "stroke-width": 2.5 }, s);
      const here = matchAt(cx, cy);
      label(s, px + 1.5 * pCell, py + 3 * pCell + 18, here ? "rule matches ✓" : "no match ✗", { style: `fill: ${here ? c.opt : c.grad}; font-weight: 600` });

      // the match map: one cell per window position
      label(s, mx + (nx * mCell) / 2, my - 12, "match at each position", { style: textStyle });
      for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
        const m = matchAt(x, y), i = y * nx + x;
        const rect = el("rect", { x: mx + x * mCell, y: my + y * mCell, width: mCell, height: mCell, fill: m ? c.opt : c.bg, "fill-opacity": m ? 0.35 : 1, stroke: i === cur ? c.model : c.border, "stroke-width": i === cur ? 3 : 1 }, s);
        rect.style.cursor = "pointer";
        rect.addEventListener("click", () => { stop(); cur = i; draw(); });
        label(s, mx + (x + 0.5) * mCell, my + (y + 0.5) * mCell + 4, m ? "T" : "F", { style: textStyle + "; pointer-events: none" });
      }
      // readout: the OR over all positions
      const all = [];
      for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) all.push(matchAt(x, y));
      const any = all.some(Boolean), nm = all.filter(Boolean).length;
      const lits = [];
      rule.forEach((k, idx) => { const n = `x${Math.floor(idx / 3) + 1},${(idx % 3) + 1}`; if (k === "on") lits.push(n); if (k === "off") lits.push("¬" + n); });
      const chain = total <= 8 ? all.map((b) => (b ? "True" : "False")).join(" OR ") : `OR over ${total} windows`;
      out.textContent = `rule: if ${lits.length ? lits.join(" ∧ ") : "(empty: always True)"}\n` +
        `${total} window positions · matches at ${nm} · ${chain} = ${any ? "True" : "False"}\n` +
        `→ the rule's single output for this image: ${any ? "True (it votes)" : "False (no vote)"}`;
    }

    ctrl1.addEventListener("click", (e) => {
      const i = e.target.dataset.i; if (!i) return;
      stop(); img = i === "clear" ? CTM.blank(img.h, img.w) : PRESETS[i](); cur = 0; draw();
    });
    ctrl2.addEventListener("click", (e) => {
      const r = e.target.dataset.r, a = e.target.dataset.a;
      if (r !== undefined) { stop(); rule = r === "empty" ? Array(9).fill("any") : RULE_CELLS[+r].slice(); draw(); }
      const { nx, ny } = nPos();
      if (a === "next") { stop(); cur = (cur + 1) % (nx * ny); draw(); }
      if (a === "prev") { stop(); cur = (cur + nx * ny - 1) % (nx * ny); draw(); }
      if (a === "play") {
        if (timer) { stop(); return; }
        cur = -1;
        timer = setInterval(() => { cur++; if (cur >= nx * ny - 1) { cur = nx * ny - 1; stop(); } draw(); }, 650);
      }
    });
    onThemeChange(draw);
    window.addEventListener("resize", draw);
    draw();
  }

  // ---------------------------------------------------------------- Accuracy chart (shared)
  function drawChart(s, W, H, history, epochs) {
    const c = colors(), M = { l: 40, r: 10, t: 14, b: 32 };
    const x = linear(0, epochs, M.l, W - M.r), y = linear(0.4, 1, H - M.b, M.t);
    axes(s, x, y, { xDomain: [0, epochs], yDomain: [0.4, 1], xTicks: ticks(0, epochs, 5), yTicks: [0.4, 0.6, 0.8, 1], xLabel: "epoch" });
    const line = (k, col, dash) => {
      if (!history.length) return;
      const d = history.map((h, i) => (i ? "L" : "M") + x(i + 1) + "," + y(Math.max(0.4, h[k]))).join("");
      el("path", { d, fill: "none", stroke: col, "stroke-width": 2.5, "stroke-dasharray": dash || "" }, s);
      history.forEach((h, i) => el("circle", { cx: x(i + 1), cy: y(Math.max(0.4, h[k])), r: 2.5, fill: col }, s));
    };
    line(0, c.muted, "5 4");
    line(1, c.opt);
    label(s, W - M.r - 4, H - M.b - 26, "test accuracy", { "text-anchor": "end", style: `fill: ${c.opt}; font-weight: 600` });
    label(s, W - M.r - 4, H - M.b - 10, "training accuracy (dashed)", { "text-anchor": "end" });
  }

  // Draw one rule card: stencil + caption; returns nothing. fb = feedback record (or null)
  function ruleCard(g, tm, j, win, x, y, cell, opts) {
    const c = colors(), pic = tm.picture(j, win);
    if (opts.selected) el("rect", { x: x - 5, y: y - 5, width: win * cell + 10, height: win * cell + 10, fill: "none", stroke: c.model, "stroke-width": 2, "stroke-dasharray": "4 3", rx: 4 }, g);
    drawStencil(g, pic.cells, win, x, y, cell);
    const hit = el("rect", { x: x - 5, y: y - 5, width: win * cell + 10, height: win * cell + 10, fill: "transparent" }, g);
    hit.style.cursor = "pointer";
    hit.addEventListener("click", opts.onClick);
    if (opts.fb) {
      const t = opts.fb.type, col = t === "erase" ? c.grad : t === "none" ? c.muted : c.opt;
      const name = { recognize: "Recognize", erase: "Erase", reject: "Reject", none: "no feedback" }[t];
      label(g, x + (win * cell) / 2, y + win * cell + 14, name, { style: `fill: ${col}; font-size: 11px; font-weight: ${t === "none" ? 400 : 600}` });
    }
    return pic;
  }

  // ---------------------------------------------------------------- Widget 2: learning
  function learnWidget(root) {
    const ctrl1 = control(root, `Window: <button data-w="3">3×3 (convolution)</button><button data-w="6">6×6 = whole image (standard TM)</button>`);
    const ctrl1b = control(root, `Training shapes: <button data-h="0">anywhere</button><button data-h="1">left half only (test: right half)</button>`);
    const ctrl2 = control(root, `<label>Forget Value <input type="range" min="0.1" max="0.9" step="0.1" value="0.7"> <span class="fv"></span></label>
      <label>rules per class <select><option>2</option><option selected>4</option><option>8</option></select></label>
      <label>Voting Margin T <input type="range" min="1" max="8" step="1" value="4"> <span class="tv"></span></label>
      <label>seed <input type="number" min="1" max="999" value="1" style="width:4em"></label>`);
    const ctrl3 = control(root, `<button data-a="step">Step: one image</button><button data-a="epoch">Train 1 epoch</button><button data-a="all">Train to 20 epochs</button><button data-a="test">Show a test image</button><button data-a="reset">Reset</button>`);
    const [fIn, tIn] = ctrl2.querySelectorAll("input[type=range]"), nSel = ctrl2.querySelector("select"), seedIn = ctrl2.querySelector("input[type=number]");
    hint(root, "Click a rule picture to see where it matches in the image (green dots). After a step, the orange box is the patch that rule was given feedback on, chosen at random among its matches.");
    stencilLegend(root);
    const out = readout(root);
    const EPOCHS = 20;
    let win = 3, holdout = false, ex, order, ptr, last, sel, view, s = null, chart = null, busy = null;

    function reset() {
      clearTimeout(busy); busy = null;
      ex = CTM.experiment({ seed: +seedIn.value || 1, win, holdout, n: +nSel.value, T: +tIn.value, forget: +fIn.value });
      order = null; ptr = 0; last = null; sel = 0;
      view = { img: ex.tr.X[0], y: ex.tr.y[0], what: "first training image" };
      draw();
    }
    function step() {
      if (ex.history.length >= EPOCHS) return;
      if (!order) { order = CTM.shuffledOrder(ex.Ptr.length, ex.rand); ptr = 0; }
      const i = order[ptr++], log = {};
      ex.tm.update(ex.Ptr[i], ex.tr.y[i], log);
      last = log; view = { img: ex.tr.X[i], y: ex.tr.y[i], what: `training image #${i + 1}` };
      const fbRule = log.fb.findIndex((f) => f.type === "recognize" || f.type === "reject");
      if (fbRule >= 0) sel = fbRule;
      if (ptr >= order.length) finishEpoch();
    }
    function finishEpoch() {
      ex.history.push([CTM.accuracy(ex.tm, ex.Ptr, ex.tr.y), CTM.accuracy(ex.tm, ex.Pte, ex.te.y)]);
      order = null;
    }
    function epoch() {
      if (ex.history.length >= EPOCHS) return;
      while (true) { const before = ex.history.length; step(); if (ex.history.length > before) break; }
      last = null;
    }

    function draw() {
      if (s) s.remove();
      if (chart) chart.remove();
      const c = colors(), narrow = isNarrow(root), tm = ex.tm, n2 = tm.cls.length;
      fIn.nextElementSibling.textContent = (+fIn.value).toFixed(1);
      tIn.nextElementSibling.textContent = tIn.value;
      [...ctrl1.querySelectorAll("button"), ...ctrl1b.querySelectorAll("button")].forEach((b) => {
        const on = (b.dataset.w && +b.dataset.w === win) || (b.dataset.h && (b.dataset.h === "1") === holdout);
        b.style.borderColor = on ? "var(--viz-model)" : "";
      });
      const imgCell = narrow ? 26 : 32;
      const perRow = narrow ? Math.min(4, n2 / 2) : Math.min(4, n2 / 2);
      const sc = win === 3 ? (narrow ? 15 : 17) : (narrow ? 8 : 9);
      const cardW = win * sc + (narrow ? 22 : 30), cardH = win * sc + 34;
      const rowsPerClass = Math.ceil(n2 / 2 / perRow);
      const imgBlockW = 6 * imgCell + 30;
      let W, H, ix, iy, rx, ry;
      if (!narrow) {
        ix = 10; iy = 30; rx = ix + imgBlockW + 64; ry = 30;
        W = Math.max(600, rx + perRow * cardW); H = Math.max(iy + 6 * imgCell + 30, ry + 2 * rowsPerClass * cardH + 24);
      } else {
        W = 340; ix = (W - 6 * imgCell) / 2; iy = 30; rx = 64; ry = iy + 6 * imgCell + 50;
        H = ry + 2 * rowsPerClass * cardH + 16;
      }
      s = svg(W, H);
      s.style.maxWidth = W + "px"; s.style.margin = "0 auto"; s.style.display = "block";
      root.insertBefore(s, ctrl1);
      const name = holdout ? "" : "";
      label(s, ix + 3 * imgCell, 18, `${view.what}: ${view.y ? "Cross" : "Circle"}${name}`, { style: textStyle });
      drawImage(s, view.img, ix, iy, imgCell);
      // where the selected rule matches
      const P = CTM.patches(view.img, win, false);
      P.lits.forEach((p, i) => {
        if (!tm.matches(sel, p)) return;
        const [x, y] = P.where[i];
        el("circle", { cx: ix + (x + win / 2) * imgCell, cy: iy + (y + win / 2) * imgCell, r: imgCell * 0.17, fill: c.opt, stroke: c.bg, "stroke-width": 1.5 }, s);
      });
      if (last && last.fb[sel] && last.fb[sel].patch !== undefined) {
        const [x, y] = P.where[last.fb[sel].patch];
        el("rect", { x: ix + x * imgCell, y: iy + y * imgCell, width: win * imgCell, height: win * imgCell, fill: "none", stroke: c.model, "stroke-width": 3 }, s);
      }
      // rule cards, Circle rules then Cross rules
      for (let cls = 0; cls < 2; cls++) {
        const y0 = ry + cls * rowsPerClass * cardH;
        label(s, rx - 10, y0 + (win * sc) / 2 + 4, cls ? "Cross" : "Circle", { "text-anchor": "end", style: textStyle + "; font-weight: 600" });
        for (let q = 0; q < n2 / 2; q++) {
          const j = cls * (n2 / 2) + q, col = q % perRow, row = Math.floor(q / perRow);
          ruleCard(s, tm, j, win, rx + col * cardW + 6, y0 + row * cardH, sc, {
            selected: j === sel, fb: last ? last.fb[j] : null, onClick: () => { sel = j; draw(); },
          });
        }
      }
      // chart
      const CW = narrow ? 340 : 600, CH = 180;
      chart = svg(CW, CH);
      chart.style.maxWidth = CW + "px"; chart.style.margin = "8px auto 0"; chart.style.display = "block";
      root.insertBefore(chart, ctrl1);
      drawChart(chart, CW, CH, ex.history, EPOCHS);
      // readout
      const ep = ex.history.length, h = ex.history[ep - 1];
      let txt = `epochs trained: ${ep}/${EPOCHS}` + (h ? ` · training accuracy ${pct(h[0])} · test accuracy ${pct(h[1])}` : "") +
        (holdout ? "\n(training shapes only in the left half; test shapes only in the right half)" : "");
      if (order && ptr) txt += `\nepoch ${ep + 1}: image ${ptr} of ${order.length}`;
      if (last) {
        const nf = last.fb.filter((f) => f.type !== "none").length;
        txt += `\nvote sum for the true class v = ${last.v}, so each rule gets feedback with probability (T − v)/2T = ${fmt(last.prob, 3)} → ${nf} of ${n2} rules did.`;
        const f = last.fb[sel];
        if (f && f.patch !== undefined) txt += `\nselected rule: ${f.type === "recognize" ? "Recognize" : "Reject"} Feedback on 1 patch, picked at random from its ${f.of} matching patch${f.of > 1 ? "es" : ""} (orange box).`;
      }
      const tmOut = tm.outputs(P.lits).filter(Boolean).length;
      if (!last) txt += `\nOn this image ${tmOut} of ${n2} rules fire; prediction: ${tm.predict(P.lits) ? "Cross" : "Circle"} (vote sum ${tm.vote(P.lits)} for Circle).`;
      out.textContent = txt;
    }

    function run() {
      // train in chunks so the page stays responsive
      if (busy || ex.history.length >= EPOCHS) return;
      const tick = () => { epoch(); draw(); busy = ex.history.length < EPOCHS ? setTimeout(tick, 30) : null; };
      busy = setTimeout(tick, 0);
    }
    const onSet = (e) => {
      if (e.target.dataset.w) { win = +e.target.dataset.w; reset(); }
      if (e.target.dataset.h) { holdout = e.target.dataset.h === "1"; reset(); }
    };
    ctrl1.addEventListener("click", onSet);
    ctrl1b.addEventListener("click", onSet);
    ctrl2.addEventListener("change", reset);
    ctrl2.addEventListener("input", (e) => { if (e.target.type === "range") reset(); });
    ctrl3.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (!a || (busy && a !== "reset")) return;
      if (a === "step") { step(); draw(); }
      if (a === "epoch") { epoch(); draw(); }
      if (a === "all") run();
      if (a === "reset") reset();
      if (a === "test") {
        const k = Math.floor(Math.random() * ex.te.X.length);
        last = null; view = { img: ex.te.X[k], y: ex.te.y[k], what: `test image #${k + 1}` };
        const P = CTM.patches(view.img, win, false).lits;
        const firing = ex.tm.outputs(P).findIndex((o, j) => o && ex.tm.cls[j] === view.y);
        if (firing >= 0) sel = firing;
        draw();
      }
    });
    onThemeChange(draw);
    window.addEventListener("resize", draw);
    reset();
  }

  // ---------------------------------------------------------------- Widget 3: position encoding
  function positionWidget(root) {
    const ctrl1 = control(root, `<label>window x from <select data-k="x0"></select> to <select data-k="x1"></select></label>
      <label>window y from <select data-k="y0"></select> to <select data-k="y1"></select></label>
      <button data-a="any">Anywhere</button><button data-a="top">Upper half</button><button data-a="bottom">Lower half</button>`);
    const ctrl2 = control(root, `<button data-a="learn">Learn "upper or lower half?" from data</button><span class="lres"></span>`);
    const sels = {};
    ctrl1.querySelectorAll("select").forEach((sEl) => {
      for (let v = 0; v <= 3; v++) sEl.insertAdjacentHTML("beforeend", `<option>${v}</option>`);
      sels[sEl.dataset.k] = sEl;
    });
    sels.x1.value = 3; sels.y1.value = 3;
    hint(root, "Click a pixel to move the Circle there. The rule is the full 3×3 Circle picture plus position literals; the dashed box is the image region the position literals allow.");
    const out = readout(root);
    const learnBox = document.createElement("div");
    root.appendChild(learnBox);
    let centre = [1, 1], s = null, learned = null, ls = null;

    const ring = ["off", "on", "off", "on", "off", "on", "off", "on", "off"];
    function posLits() {
      const x0 = +sels.x0.value, x1 = Math.max(x0, +sels.x1.value), y0 = +sels.y0.value, y1 = Math.max(y0, +sels.y1.value);
      const lits = [];
      if (x0 > 0) lits.push(`X≥${x0}`);
      if (x1 < 3) lits.push(`¬X≥${x1 + 1}`);
      if (y0 > 0) lits.push(`Y≥${y0}`);
      if (y1 < 3) lits.push(`¬Y≥${y1 + 1}`);
      return { x0, x1, y0, y1, lits };
    }
    function therm(g, x, y, value, name, cell) {
      const c = colors();
      label(g, x - 8, y + cell * 0.7, name, { "text-anchor": "end", style: textStyle });
      for (let t = 1; t <= 3; t++) {
        const on = value >= t;
        el("rect", { x: x + (t - 1) * cell, y, width: cell, height: cell, fill: on ? c.data : c.bg, "fill-opacity": on ? 0.3 : 1, stroke: c.border }, g);
        label(g, x + (t - 0.5) * cell, y + cell * 0.68, on ? "T" : "F", { style: textStyle + "; font-size: 12px" });
        label(g, x + (t - 0.5) * cell, y - 4, `≥${t}`, { style: "font-size: 10px" });
      }
      label(g, x + 3 * cell + 8, y + cell * 0.7, `= ${value}`, { "text-anchor": "start", style: textStyle });
    }

    function draw() {
      if (s) s.remove();
      const c = colors(), narrow = isNarrow(root);
      const img = CTM.blank(6, 6);
      CTM.drawShape(img, CTM.CIRCLE, centre[0], centre[1]);
      const cell = narrow ? 30 : 34, { x0, x1, y0, y1, lits } = posLits();
      const wx = centre[1] - 1, wy = centre[0] - 1;                         // window holding the circle
      let W, H, ix = 10, iy = 28, tx, ty, mx, my;
      const mc = narrow ? 28 : 30;
      if (!narrow) { tx = ix + 6 * cell + 80; ty = 50; mx = tx + 20; my = ty + 100; W = 600; H = Math.max(iy + 6 * cell + 12, my + 4 * mc + 10); }
      else { W = 340; ix = (W - 6 * cell) / 2; tx = 110; ty = iy + 6 * cell + 62; mx = tx - 40; my = ty + 100; H = my + 4 * mc + 12; }
      s = svg(W, H);
      s.style.maxWidth = W + "px"; s.style.margin = "0 auto"; s.style.display = "block";
      root.insertBefore(s, ctrl1);
      label(s, ix + 3 * cell, 16, "image (6×6)", { style: textStyle });
      drawImage(s, img, ix, iy, cell, { onClick: (r, q) => { centre = [Math.min(4, Math.max(1, r)), Math.min(4, Math.max(1, q))]; draw(); } });
      // allowed region = union of all windows whose corner lies in the allowed range
      el("rect", { x: ix + x0 * cell + 2, y: iy + y0 * cell + 2, width: (x1 - x0 + 3) * cell - 4, height: (y1 - y0 + 3) * cell - 4,
        fill: c.model, "fill-opacity": 0.07, stroke: c.model, "stroke-width": 2.5, "stroke-dasharray": "7 5" }, s);
      el("rect", { x: ix + wx * cell, y: iy + wy * cell, width: 3 * cell, height: 3 * cell, fill: "none", stroke: c.data, "stroke-width": 3 }, s);
      // thermometers for the window that holds the circle
      label(s, tx + 50, ty - 30, `window holding the Circle: (x=${wx}, y=${wy})`, { style: textStyle });
      therm(s, tx + 20, ty - 8, wx, "X", 28);
      therm(s, tx + 20, ty + 34, wy, "Y", 28);
      // which positions does the full rule (pattern AND position) match?
      label(s, mx + 2 * mc, my - 8, "pattern ∧ position, per window", { "text-anchor": "start", style: "font-size: 11px" });
      const P = CTM.patches(img, 3, true);
      let fires = false;
      P.lits.forEach((p, i) => {
        const [x, y] = P.where[i];
        const pat = ring.every((k, idx) => (k === "on" ? p[idx] : !p[idx]));
        const inPos = x >= x0 && x <= x1 && y >= y0 && y <= y1;
        if (pat && inPos) fires = true;
        el("rect", { x: mx + x * mc, y: my + y * mc, width: mc, height: mc, fill: pat && inPos ? c.opt : inPos ? c.model : c.bg,
          "fill-opacity": pat && inPos ? 0.45 : inPos ? 0.1 : 1, stroke: c.border }, s);
        label(s, mx + (x + 0.5) * mc, my + (y + 0.5) * mc + 4, pat ? (inPos ? "✓" : "✗") : "·", { style: `fill: ${pat ? (inPos ? c.opt : c.grad) : c.muted}; font-weight: 600` });
      });
      const ruleTxt = `if [Circle picture]${lits.map((l) => " ∧ " + l).join("")}`;
      out.textContent = `rule: ${ruleTxt}\n` +
        `The pattern is found at window (x=${wx}, y=${wy}); its position literals there: X = ${wx} → [${[1, 2, 3].map((t) => (wx >= t ? "T" : "F")).join(" ")}], Y = ${wy} → [${[1, 2, 3].map((t) => (wy >= t ? "T" : "F")).join(" ")}]\n` +
        `→ rule output: ${fires ? "True: right pattern, allowed place" : "False: " + "the Circle is outside the allowed region"}`;
    }

    function drawLearned() {
      if (ls) ls.remove();
      if (!learned) return;
      const c = colors(), narrow = isNarrow(root), tm = learned.tm, n2 = tm.cls.length;
      const sc = 14, mc = 9, cardW = narrow ? 130 : 120, cardH = 3 * sc + 30;
      const perRow = narrow ? 2 : 4;
      const rowsPerClass = Math.ceil(n2 / 2 / perRow);
      const left = narrow ? 56 : 70;
      const W = narrow ? 340 : Math.max(600, left + perRow * cardW), H = 2 * rowsPerClass * cardH + 40;
      ls = svg(W, H);
      ls.style.maxWidth = W + "px"; ls.style.margin = "8px auto 0"; ls.style.display = "block";
      learnBox.appendChild(ls);
      label(ls, W / 2, 14, narrow ? "learned rules: pattern + allowed window positions" : "learned rules: pattern (left) + where it may be (right; shaded = allowed window positions)", { style: "font-size: 11px" });
      for (let cls = 0; cls < 2; cls++) {
        const y0 = 30 + cls * rowsPerClass * cardH;
        label(ls, left - 8, y0 + 1.5 * sc + 4, cls ? "Lower" : "Upper", { "text-anchor": "end", style: textStyle + "; font-weight: 600" });
        for (let q = 0; q < n2 / 2; q++) {
          const j = cls * (n2 / 2) + q, x = left + (q % perRow) * cardW, y = y0 + Math.floor(q / perRow) * cardH;
          const pic = tm.picture(j, 3);
          drawStencil(ls, pic.cells, 3, x, y, sc);
          // allowed positions: thermometer literals X≥1..3 (extra 0..2), Y≥1..3 (extra 3..5)
          const ok = (v, base) => [1, 2, 3].every((t) => { const [inc, neg] = pic.extra[base + t - 1]; return (!inc || v >= t) && (!neg || v < t); });
          for (let yy = 0; yy < 4; yy++) for (let xx = 0; xx < 4; xx++) {
            const a = ok(xx, 0) && ok(yy, 3);
            el("rect", { x: x + 3 * sc + 10 + xx * mc, y: y + 3 + yy * mc, width: mc, height: mc, fill: a ? c.model : c.bg, "fill-opacity": a ? 0.55 : 1, stroke: c.border }, ls);
          }
          const names = [];
          pic.extra.forEach(([inc, neg], k) => { const nm = (k < 3 ? "X" : "Y") + "≥" + ((k % 3) + 1); if (inc) names.push(nm); if (neg) names.push("¬" + nm); });
          label(ls, x + (3 * sc + 10 + 4 * mc) / 2, y + 3 * sc + 16, names.length ? names.join(" ") : "(no position literal)", { style: "font-size: 10px" });
        }
      }
    }

    function learn() {
      const lres = ctrl2.querySelector(".lres");
      lres.textContent = " training…";
      setTimeout(() => {
        const a = CTM.experiment({ seed: 1, task: "half", pos: true });
        for (let e = 0; e < 20; e++) CTM.trainEpoch(a);
        const b = CTM.experiment({ seed: 1, task: "half", pos: false });
        for (let e = 0; e < 20; e++) CTM.trainEpoch(b);
        learned = a;
        lres.textContent = ` test accuracy with position literals ${pct(a.history[19][1])} · without ${pct(b.history[19][1])} (seed 1, 20 epochs)`;
        drawLearned();
      }, 20);
    }

    ctrl1.addEventListener("change", draw);
    ctrl1.addEventListener("click", (e) => {
      const a = e.target.dataset.a; if (!a) return;
      const set = { any: [0, 3, 0, 3], top: [0, 3, 0, 1], bottom: [0, 3, 2, 3] }[a];
      [sels.x0.value, sels.x1.value, sels.y0.value, sels.y1.value] = set;
      draw();
    });
    ctrl2.addEventListener("click", (e) => { if (e.target.dataset.a === "learn") learn(); });
    onThemeChange(() => { draw(); drawLearned(); });
    window.addEventListener("resize", () => { draw(); drawLearned(); });
    draw();
  }

  const mounts = { "viz-shapes": shapesWidget, "viz-slide": slideWidget, "viz-learn": learnWidget, "viz-position": positionWidget };
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
