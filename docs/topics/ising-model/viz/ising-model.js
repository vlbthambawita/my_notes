// Interactive widgets for "The Ising model".
// Widgets, each mounted on a <div> with a matching id:
//   #viz-spins      – click spins on a 4×4 grid; bonds turn green/red and the energy moves on a ladder
//   #viz-boltzmann  – all 16 states of a 2×2 grid with their Boltzmann probabilities at temperature T
//   #viz-flip       – one spin and its four neighbours: ΔE and the Metropolis flip probability
//   #viz-lattice    – a live 64×64 Metropolis simulation with temperature and field sliders
//   #viz-sweep      – measure |m| at many temperatures and compare with Onsager's exact curve
//   #viz-denoise    – clean a noisy black-and-white picture with an Ising energy
// The model code lives in `Ising` (no DOM), so scripts can run it in Node to check the numbers in the lesson.
// All randomness comes from a seeded mulberry32 generator.
const Ising = (function () {
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = Math.imul(a ^ (a >>> 15), a | 1);
      t = (t + Math.imul(t ^ (t >>> 7), t | 61)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Critical temperature of the square lattice (J = 1, k_B = 1) and Onsager's exact magnetisation
  const TC = 2 / Math.log(1 + Math.SQRT2);
  function onsager(T) {
    if (T >= TC) return 0;
    const s = Math.sinh(2 / T);
    return Math.pow(1 - 1 / (s * s * s * s), 0.125);
  }

  // ---------------------------------------------------------------- L×L lattice, periodic boundaries
  function Lattice(L, seed) {
    this.L = L;
    this.N = L * L;
    this.s = new Int8Array(this.N);
    this.rand = mulberry32(seed);
    this.s.fill(1);
  }
  Lattice.prototype.allUp = function () { this.s.fill(1); };
  Lattice.prototype.randomize = function () {
    for (let i = 0; i < this.N; i++) this.s[i] = this.rand() < 0.5 ? 1 : -1;
  };
  // Sum of the four neighbours of spin i
  Lattice.prototype.neighbours = function (i) {
    const L = this.L, s = this.s, x = i % L, y = (i - x) / L;
    return s[y * L + (x + 1) % L] + s[y * L + (x + L - 1) % L] + s[((y + 1) % L) * L + x] + s[((y + L - 1) % L) * L + x];
  };
  // One Monte Carlo sweep = N attempted flips with the Metropolis rule
  Lattice.prototype.sweep = function (T, h) {
    const s = this.s, N = this.N, rand = this.rand;
    for (let k = 0; k < N; k++) {
      const i = Math.floor(rand() * N);
      const dE = 2 * s[i] * (this.neighbours(i) + h);
      if (dE <= 0 || rand() < Math.exp(-dE / T)) s[i] = -s[i];
    }
  };
  Lattice.prototype.m = function () {
    let sum = 0;
    for (let i = 0; i < this.N; i++) sum += this.s[i];
    return sum / this.N;
  };
  Lattice.prototype.energyPerSpin = function (h) {
    const L = this.L, s = this.s;
    let e = 0;
    for (let y = 0; y < L; y++) for (let x = 0; x < L; x++) {
      const v = s[y * L + x];
      e -= v * (s[y * L + (x + 1) % L] + s[((y + 1) % L) * L + x]) + h * v;
    }
    return e / this.N;
  };

  // Average |m| at each temperature, warming one lattice up from all-up (used by #viz-sweep)
  const SWEEP_TS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0];
  const EQUIL = 200, MEASURE = 400;
  function measurePoint(lat, T) {
    for (let k = 0; k < EQUIL; k++) lat.sweep(T, 0);
    let acc = 0;
    for (let k = 0; k < MEASURE; k++) { lat.sweep(T, 0); acc += Math.abs(lat.m()); }
    return acc / MEASURE;
  }

  // ---------------------------------------------------------------- Small grids with open edges
  // Energy of an n×n grid (array of ±1) with J = 1 and no field: −Σ s_i s_j over neighbouring pairs
  function gridBonds(s, n) {
    let agree = 0, disagree = 0;
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const v = s[y * n + x];
      if (x + 1 < n) { if (s[y * n + x + 1] === v) agree++; else disagree++; }
      if (y + 1 < n) { if (s[(y + 1) * n + x] === v) agree++; else disagree++; }
    }
    return { agree, disagree, energy: disagree - agree };
  }
  // All 16 states of a 2×2 grid, sorted by energy
  function states2x2() {
    const out = [];
    for (let code = 0; code < 16; code++) {
      const s = [0, 1, 2, 3].map((b) => ((code >> b) & 1 ? 1 : -1));
      out.push({ s, energy: gridBonds(s, 2).energy });
    }
    out.sort((a, b) => a.energy - b.energy);
    return out;
  }
  function boltzmann(states, T) {
    const w = states.map((st) => Math.exp(-st.energy / T));
    const Z = w.reduce((a, b) => a + b, 0);
    return { p: w.map((v) => v / Z), Z };
  }

  // ---------------------------------------------------------------- Picture denoising
  const PIC = 40;
  function picture() {
    const img = new Int8Array(PIC * PIC), c = (PIC - 1) / 2;
    for (let y = 0; y < PIC; y++) for (let x = 0; x < PIC; x++) {
      const r = Math.sqrt((x - c) * (x - c) + (y - c) * (y - c));
      img[y * PIC + x] = (r >= 9 && r <= 15) || r <= 4 ? 1 : -1;
    }
    return img;
  }
  function addNoise(img, p, seed) {
    const rand = mulberry32(seed), out = new Int8Array(img);
    for (let i = 0; i < out.length; i++) if (rand() < p) out[i] = -out[i];
    return out;
  }
  // Greedy energy descent (iterated conditional modes) on E = −J Σ s_i s_j − Σ s_i y_i, starting from s = y
  function restore(noisy, J) {
    const n = PIC, s = new Int8Array(noisy);
    let sweeps = 0;
    for (; sweeps < 20; sweeps++) {
      let changed = 0;
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const i = y * n + x;
        const nb = (x > 0 ? s[i - 1] : 0) + (x < n - 1 ? s[i + 1] : 0) + (y > 0 ? s[i - n] : 0) + (y < n - 1 ? s[i + n] : 0);
        const f = J * nb + noisy[i];
        const v = f > 0 ? 1 : f < 0 ? -1 : s[i];
        if (v !== s[i]) { s[i] = v; changed++; }
      }
      if (!changed) break;
    }
    return { s, sweeps };
  }
  function countWrong(a, b) {
    let k = 0;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) k++;
    return k;
  }

  return { mulberry32, TC, onsager, Lattice, SWEEP_TS, EQUIL, MEASURE, measurePoint, gridBonds, states2x2, boltzmann, PIC, picture, addNoise, restore, countWrong };
})();

document.addEventListener("DOMContentLoaded", function () {
  const { colors, onThemeChange, linear, el, svg, axes, ticks, fmt } = window.Viz;
  const textStyle = "fill: var(--viz-text)";

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
    parent.appendChild(div);
    return div;
  }
  function hint(parent, text) {
    const div = document.createElement("div");
    div.className = "viz-hint";
    div.textContent = text;
    parent.appendChild(div);
    return div;
  }
  function plotsRow(parent) {
    const div = document.createElement("div");
    div.className = "viz-plots";
    div.style.alignItems = "flex-start";
    parent.appendChild(div);
    return div;
  }
  // Signed number with a real minus sign, never "−0"
  function sg(v, d = 0) {
    const a = Math.abs(v).toFixed(d);
    if (+a === 0) return a;
    return (v > 0 ? "+" : "−") + a;
  }
  const pct = (v, d = 1) => fmt(100 * v, d) + "%";
  // Re-run draw() when the widget crosses the phone-width threshold
  function watchNarrow(root, draw) {
    let narrow = root.clientWidth < 480;
    if ("ResizeObserver" in window) {
      new ResizeObserver(() => {
        const now = root.clientWidth < 480;
        if (now !== narrow) { narrow = now; draw(); }
      }).observe(root);
    }
    return () => narrow;
  }
  // One spin drawn as a rounded square with an arrow
  function spinCell(parent, x, y, size, v, c, attrs) {
    const r = el("rect", Object.assign({ x, y, width: size, height: size, rx: size * 0.16, fill: v > 0 ? c.model : c.data }, attrs || {}), parent);
    const t = el("text", { x: x + size / 2, y: y + size * 0.68, "text-anchor": "middle", style: `fill: ${c.bg}; font-size: ${Math.round(size * 0.52)}px; font-weight: 700; pointer-events: none` }, parent);
    t.textContent = v > 0 ? "↑" : "↓";
    return r;
  }

  // ---------------------------------------------------------------- Widget 1: click spins, watch the energy
  function spinGrid(root) {
    const n = 4, W = 320, H = 320, SIZE = 56, PITCH = 72, X0 = 24;
    const plots = plotsRow(root);
    const left = svg(W, H), right = svg(W, H);
    left.style.cursor = "pointer";
    plots.append(left, right);
    hint(root, "Click any spin to flip it. Green links join neighbours that agree, red links join neighbours that disagree.");
    const ctrl = control(root, `Try: <button data-p="up">All up</button><button data-p="halves">Two halves</button><button data-p="island">One odd spin</button><button data-p="checker">Checkerboard</button><button data-p="random">Random</button>`);
    const out = readout(root);
    const rand = Ising.mulberry32(11);
    let s = new Array(n * n).fill(1);

    const presets = {
      up: () => 1,
      halves: (x) => (x < 2 ? 1 : -1),
      island: (x, y) => (x === 1 && y === 1 ? -1 : 1),
      checker: (x, y) => ((x + y) % 2 ? -1 : 1),
      random: () => (rand() < 0.5 ? 1 : -1),
    };
    const yE = linear(-24, 24, H - 34, 34);

    function draw() {
      const c = colors();
      left.innerHTML = right.innerHTML = "";
      const pos = (k) => X0 + k * PITCH;
      // Bonds first, so the spins sit on top of them
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const v = s[y * n + x], cx = pos(x) + SIZE / 2, cy = pos(y) + SIZE / 2;
        if (x + 1 < n) el("line", { x1: cx, y1: cy, x2: cx + PITCH, y2: cy, stroke: s[y * n + x + 1] === v ? c.opt : c.grad, "stroke-width": 7 }, left);
        if (y + 1 < n) el("line", { x1: cx, y1: cy, x2: cx, y2: cy + PITCH, stroke: s[(y + 1) * n + x] === v ? c.opt : c.grad, "stroke-width": 7 }, left);
      }
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) spinCell(left, pos(x), pos(y), SIZE, s[y * n + x], c, { "data-i": y * n + x });

      // Energy ladder
      const b = Ising.gridBonds(s, n), AX = 150;
      el("line", { x1: AX, x2: AX, y1: yE(-24), y2: yE(24), stroke: c.muted, "stroke-width": 2 }, right);
      const labels = { "-24": "all agree  −24", "-12": "−12", "0": "0", "12": "+12", "24": "checkerboard  +24" };
      for (const t of [-24, -12, 0, 12, 24]) {
        el("line", { x1: AX - 6, x2: AX, y1: yE(t), y2: yE(t), stroke: c.muted, "stroke-width": 2 }, right);
        el("text", { x: AX - 10, y: yE(t) + 4, "text-anchor": "end" }, right).textContent = labels[t];
      }
      el("text", { x: AX, y: 16, "text-anchor": "middle" }, right).textContent = "energy E (lower = more stable)";
      el("line", { x1: AX, x2: AX + 70, y1: yE(b.energy), y2: yE(b.energy), stroke: c.loss, "stroke-width": 4 }, right);
      el("circle", { cx: AX, cy: yE(b.energy), r: 6, fill: c.loss }, right);
      el("text", { x: AX + 78, y: yE(b.energy) + 5, style: `${textStyle}; font-size: 15px; font-weight: 700` }, right).textContent = "E = " + sg(b.energy);

      const m = s.reduce((a, v) => a + v, 0) / (n * n);
      out.textContent = `links that agree: ${b.agree} · disagree: ${b.disagree} · energy E = ${b.disagree} − ${b.agree} = ${sg(b.energy)} · magnetisation m = ${sg(m, 2)}`;
    }

    left.addEventListener("click", (e) => {
      const i = e.target.dataset && e.target.dataset.i;
      if (i === undefined) return;
      s[+i] = -s[+i];
      draw();
    });
    ctrl.addEventListener("click", (e) => {
      const p = e.target.dataset.p;
      if (!p) return;
      s = s.map((_, i) => presets[p](i % n, Math.floor(i / n)));
      draw();
    });
    onThemeChange(draw);
    draw();
  }

  // ---------------------------------------------------------------- Widget 2: Boltzmann probabilities of a 2×2 grid
  function boltzmannTiles(root) {
    const view = svg(640, 180);
    root.appendChild(view);
    const ctrl = control(root, `
      <label>temperature <b>T</b> <input type="range" min="0.2" max="10" step="0.1" value="2"> <span class="val"></span></label>
      Try: <button data-t="0.5">Cold (0.5)</button><button data-t="2">Warm (2)</button><button data-t="10">Hot (10)</button>`);
    const slider = ctrl.querySelector("input"), val = ctrl.querySelector(".val");
    const out = readout(root);
    const states = Ising.states2x2();
    const CELL = 12, GAP = 2, BAR = 80;

    function draw() {
      const c = colors(), T = +slider.value, narrow = isNarrow();
      val.textContent = T.toFixed(1);
      const cols = narrow ? 8 : 16, W = narrow ? 340 : 640, ROW = 172, H = ROW * (16 / cols) + 8;
      view.setAttribute("viewBox", `0 0 ${W} ${H}`);
      view.innerHTML = "";
      const { p } = Ising.boltzmann(states, T), cw = W / cols;
      const colour = (e) => (e < 0 ? c.opt : e > 0 ? c.grad : c.loss);
      states.forEach((st, k) => {
        const cx = (k % cols + 0.5) * cw, top = Math.floor(k / cols) * ROW + 8;
        el("text", { x: cx, y: top + 10, "text-anchor": "middle", style: `fill: ${colour(st.energy)}; font-weight: 700` }, view).textContent = "E " + sg(st.energy);
        const g0 = cx - CELL - GAP / 2;
        st.s.forEach((v, j) => el("rect", { x: g0 + (j % 2) * (CELL + GAP), y: top + 18 + Math.floor(j / 2) * (CELL + GAP), width: CELL, height: CELL, rx: 2, fill: v > 0 ? c.model : c.data }, view));
        const base = top + 52 + BAR, hgt = Math.max(1, (p[k] / 0.5) * BAR);
        el("rect", { x: cx - 9, y: top + 52, width: 18, height: BAR, fill: c.grid, rx: 2 }, view);
        el("rect", { x: cx - 9, y: base - hgt, width: 18, height: hgt, fill: colour(st.energy), rx: 2 }, view);
        el("text", { x: cx, y: base + 14, "text-anchor": "middle", style: `${textStyle}; font-size: 10px` }, view).textContent = pct(p[k]);
      });
      const sum = (lo, hi) => p.slice(lo, hi).reduce((a, b) => a + b, 0);
      out.textContent = `T = ${T.toFixed(1)} · the 2 aligned states together: ${pct(sum(0, 2))} · the 12 mixed states: ${pct(sum(2, 14))} · the 2 checkerboards: ${pct(sum(14, 16))}`;
    }
    const isNarrow = watchNarrow(root, draw);
    slider.addEventListener("input", draw);
    ctrl.addEventListener("click", (e) => {
      if (!e.target.dataset.t) return;
      slider.value = e.target.dataset.t;
      draw();
    });
    onThemeChange(draw);
    draw();
  }

  // ---------------------------------------------------------------- Widget 3: should this spin flip?
  function flipRule(root) {
    const W = 320, H = 250, SIZE = 56, PITCH = 72;
    const plots = plotsRow(root);
    const left = svg(W, H), right = svg(W, H);
    left.style.cursor = "pointer";
    plots.append(left, right);
    hint(root, "The spin in the middle (thick border) is the one that wants to flip. Click it or any neighbour to change the situation.");
    const ctrl = control(root, `
      <label>temperature <b>T</b> <input type="range" min="0.2" max="5" step="0.1" value="2"> <span class="val"></span></label>
      <button data-a="try">Try this flip 100 times</button>`);
    const slider = ctrl.querySelector("input"), val = ctrl.querySelector(".val");
    const out = readout(root);
    const rand = Ising.mulberry32(7);
    // index 0 = centre, 1..4 = up, right, down, left
    let s = [1, 1, 1, 1, -1], tried = "";
    const OFF = [[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]];
    const yL = linear(-4, 4, 150, 50);

    function draw() {
      const c = colors(), T = +slider.value;
      val.textContent = T.toFixed(1);
      left.innerHTML = right.innerHTML = "";
      const cx = W / 2, cy = H / 2;
      for (let k = 1; k <= 4; k++) {
        el("line", { x1: cx, y1: cy, x2: cx + OFF[k][0] * PITCH, y2: cy + OFF[k][1] * PITCH, stroke: s[k] === s[0] ? c.opt : c.grad, "stroke-width": 7 }, left);
      }
      OFF.forEach(([dx, dy], k) => {
        spinCell(left, cx + dx * PITCH - SIZE / 2, cy + dy * PITCH - SIZE / 2, SIZE, s[k], c,
          k === 0 ? { "data-i": k, stroke: c.text, "stroke-width": 4 } : { "data-i": k });
      });

      const nb = s[1] + s[2] + s[3] + s[4], before = -s[0] * nb, dE = -2 * before, after = -before;
      const p = dE <= 0 ? 1 : Math.exp(-dE / T), col = dE <= 0 ? c.opt : c.grad;
      el("text", { x: W / 2, y: 22, "text-anchor": "middle", style: `${textStyle}; font-size: 15px; font-weight: 700` }, right).textContent =
        `ΔE = ${sg(dE)}  (${dE < 0 ? "downhill" : dE > 0 ? "uphill" : "flat"})`;
      el("line", { x1: 40, x2: 130, y1: yL(before), y2: yL(before), stroke: c.loss, "stroke-width": 4 }, right);
      el("line", { x1: 190, x2: 280, y1: yL(after), y2: yL(after), stroke: c.loss, "stroke-width": 4 }, right);
      el("text", { x: 85, y: yL(before) - 8, "text-anchor": "middle" }, right).textContent = "now: " + sg(before);
      el("text", { x: 235, y: yL(after) - 8, "text-anchor": "middle" }, right).textContent = "if flipped: " + sg(after);
      el("line", { x1: 136, y1: yL(before), x2: 184, y2: yL(after), stroke: col, "stroke-width": 2.5, "stroke-dasharray": "5 4" }, right);
      el("circle", { cx: 184, cy: yL(after), r: 4.5, fill: col }, right);
      el("text", { x: 40, y: 190, style: textStyle }, right).textContent = `chance of flipping: ${pct(p)}`;
      el("rect", { x: 40, y: 198, width: 240, height: 18, rx: 4, fill: c.grid }, right);
      el("rect", { x: 40, y: 198, width: Math.max(2, 240 * p), height: 18, rx: 4, fill: col }, right);

      const rule = dE <= 0 ? "energy does not go up → always flip" : `energy goes up → flip with probability e^(−${dE}/${T.toFixed(1)}) = ${pct(p)}`;
      out.textContent = `neighbours add up to ${sg(nb)} · ΔE = ${sg(dE)} · ${rule}${tried}`;
    }
    left.addEventListener("click", (e) => {
      const i = e.target.dataset && e.target.dataset.i;
      if (i === undefined) return;
      s[+i] = -s[+i]; tried = "";
      draw();
    });
    ctrl.addEventListener("click", (e) => {
      if (e.target.dataset.a !== "try") return;
      const nb = s[1] + s[2] + s[3] + s[4], dE = 2 * s[0] * nb, T = +slider.value;
      let n = 0;
      for (let k = 0; k < 100; k++) if (dE <= 0 || rand() < Math.exp(-dE / T)) n++;
      tried = ` · 100 tries: flipped ${n} times`;
      draw();
    });
    slider.addEventListener("input", () => { tried = ""; draw(); });
    onThemeChange(draw);
    draw();
  }

  // ---------------------------------------------------------------- Widget 4: live lattice
  function liveLattice(root) {
    const L = 64, PX = 5, SIZE = L * PX, SEED = 6, KEEP = 300;
    const W = 320, H = 320, M = { l: 40, r: 12, t: 12, b: 38 };
    const plots = plotsRow(root);
    const canvas = document.createElement("canvas");
    canvas.width = SIZE; canvas.height = SIZE;
    const chart = svg(W, H);
    plots.append(canvas, chart);
    const ctx = canvas.getContext("2d");
    const ctrl = control(root, `
      <label>temperature <b>T</b> <input type="range" name="T" min="0.5" max="5" step="0.01" value="1.5"> <span class="val"></span></label>
      <label>field <b>h</b> <input type="range" name="h" min="-1" max="1" step="0.05" value="0"> <span class="val"></span></label>`);
    const ctrl2 = control(root, `
      <button data-a="play">▶ Play</button><button data-a="random">Random start</button><button data-a="up">All up</button>
      <label>speed <select><option value="1">1 sweep / frame</option><option value="5">5 sweeps / frame</option><option value="20">20 sweeps / frame</option></select></label>`);
    const presets = control(root, `Try: <button data-t="1.5">Cold (T = 1.5)</button><button data-t="2.27">Critical (T = 2.27)</button><button data-t="3.5">Hot (T = 3.5)</button>`);
    const tIn = ctrl.querySelector('[name="T"]'), hIn = ctrl.querySelector('[name="h"]');
    const playBtn = ctrl2.querySelector('[data-a="play"]'), speed = ctrl2.querySelector("select");
    const out = readout(root);
    hint(root, "Orange = spin up, blue = spin down. In the chart, m = +1 means all up and −1 all down. One sweep = 4,096 attempted flips.");

    let lat, sweeps, history, timer = null;
    const xs = linear(0, KEEP, M.l, W - M.r), ys = linear(-1, 1, H - M.b, M.t);

    function start(kind) {
      lat = new Ising.Lattice(L, SEED);
      if (kind === "random") lat.randomize();
      sweeps = 0; history = [lat.m()];
      draw();
    }
    function draw() {
      const c = colors(), T = +tIn.value, h = +hIn.value;
      tIn.parentElement.querySelector(".val").textContent = T.toFixed(2);
      hIn.parentElement.querySelector(".val").textContent = sg(h, 2);
      ctx.fillStyle = c.data; ctx.fillRect(0, 0, SIZE, SIZE);
      ctx.fillStyle = c.model;
      for (let i = 0; i < lat.N; i++) if (lat.s[i] > 0) ctx.fillRect((i % L) * PX, Math.floor(i / L) * PX, PX, PX);

      chart.innerHTML = "";
      axes(chart, xs, ys, { xDomain: [0, KEEP], yDomain: [-1, 1], xTicks: [], yTicks: ticks(-1, 1, 0.5), xLabel: `sweeps (latest ${KEEP})`, yLabel: "magnetisation m" });
      el("line", { x1: xs(0), x2: xs(KEEP), y1: ys(0), y2: ys(0), stroke: c.muted, "stroke-dasharray": "4 4" }, chart);
      let d = "";
      history.forEach((v, i) => { d += (d ? "L" : "M") + xs(i).toFixed(1) + "," + ys(v).toFixed(1); });
      el("path", { d, fill: "none", stroke: c.loss, "stroke-width": 2 }, chart);
      el("circle", { cx: xs(history.length - 1), cy: ys(history[history.length - 1]), r: 4, fill: c.loss }, chart);

      out.textContent = `sweep ${sweeps} · T = ${T.toFixed(2)} · h = ${sg(h, 2)} · m = ${sg(lat.m(), 2)} · energy per spin = ${sg(lat.energyPerSpin(h), 2)}`;
    }
    function tick() {
      const T = +tIn.value, h = +hIn.value, k = +speed.value;
      for (let j = 0; j < k; j++) {
        lat.sweep(T, h); sweeps++;
        history.push(lat.m());
        if (history.length > KEEP + 1) history.shift();
      }
      draw();
    }
    function setPlaying(on) {
      if (on && !timer) timer = setInterval(tick, 40);
      if (!on && timer) { clearInterval(timer); timer = null; }
      playBtn.textContent = timer ? "⏸ Pause" : "▶ Play";
    }
    ctrl2.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (a === "play") setPlaying(!timer);
      if (a === "random") start("random");
      if (a === "up") start("up");
    });
    presets.addEventListener("click", (e) => {
      if (!e.target.dataset.t) return;
      tIn.value = e.target.dataset.t;
      draw();
      setPlaying(true);
    });
    tIn.addEventListener("input", draw);
    hIn.addEventListener("input", draw);
    // Stop burning CPU when the widget is off screen
    if ("IntersectionObserver" in window) {
      new IntersectionObserver((entries) => { if (!entries[0].isIntersecting) setPlaying(false); }).observe(root);
    }
    onThemeChange(draw);
    start("random");
  }

  // ---------------------------------------------------------------- Widget 5: the phase-transition experiment
  function sweepExperiment(root) {
    const view = svg(600, 300);
    root.appendChild(view);
    const ctrl = control(root, `
      <label>grid size <select><option value="8">8 × 8</option><option value="16">16 × 16</option><option value="32" selected>32 × 32</option></select></label>
      <button data-a="run">▶ Run the experiment</button>`);
    const size = ctrl.querySelector("select");
    const out = readout(root);
    hint(root, `Each dot: ${Ising.EQUIL} sweeps to settle at that temperature, then the average of |m| over ${Ising.MEASURE} more sweeps.`);
    let results = [], timer = null;

    function draw() {
      const c = colors(), narrow = isNarrow();
      const W = narrow ? 340 : 600, H = narrow ? 270 : 300, M = { l: 44, r: 14, t: 14, b: 38 };
      view.setAttribute("viewBox", `0 0 ${W} ${H}`);
      view.innerHTML = "";
      const x = linear(1, 4, M.l, W - M.r), y = linear(0, 1, H - M.b, M.t);
      axes(view, x, y, { xDomain: [1, 4], yDomain: [0, 1], xTicks: ticks(1, 4, 0.5), yTicks: ticks(0, 1, 0.25), xLabel: "temperature T", yLabel: "average |m|" });
      el("line", { x1: x(Ising.TC), x2: x(Ising.TC), y1: y(0), y2: y(1), stroke: c.grad, "stroke-dasharray": "5 4", "stroke-width": 1.5 }, view);
      el("text", { x: x(Ising.TC) + 6, y: M.t + 12, style: `fill: ${c.grad}` }, view).textContent = "T_c ≈ 2.27";
      let d = "";
      for (let T = 1; T <= Ising.TC; T += 0.005) d += (d ? "L" : "M") + x(T).toFixed(1) + "," + y(Ising.onsager(T)).toFixed(1);
      d += `L${x(Ising.TC).toFixed(1)},${y(0)}L${x(4)},${y(0)}`;
      el("path", { d, fill: "none", stroke: c.opt, "stroke-width": 2.5 }, view);
      el("text", { x: W - M.r - 6, y: y(0.62), "text-anchor": "end", style: `fill: ${c.opt}` }, view).textContent = "green: exact, infinite grid";
      el("text", { x: W - M.r - 6, y: y(0.62) + 16, "text-anchor": "end", style: `fill: ${c.model}` }, view).textContent = "orange: our simulation";
      results.forEach((r) => el("circle", { cx: x(r.T), cy: y(r.m), r: 5, fill: c.model, stroke: c.bg, "stroke-width": 1.5 }, view));

      if (!results.length) { out.textContent = "Press Run to measure the magnetisation at 16 temperatures."; return; }
      const last = results[results.length - 1];
      const done = results.length === Ising.SWEEP_TS.length ? " · done" : "";
      out.textContent = `${size.value} × ${size.value} grid · T = ${last.T.toFixed(2)} · measured |m| = ${fmt(last.m, 3)} · exact for an infinite grid = ${fmt(Ising.onsager(last.T), 3)}${done}`;
    }
    function run() {
      clearTimeout(timer);
      results = [];
      const lat = new Ising.Lattice(+size.value, 5);
      let k = 0;
      const next = () => {
        const T = Ising.SWEEP_TS[k];
        results.push({ T, m: Ising.measurePoint(lat, T) });
        draw();
        k++;
        timer = k < Ising.SWEEP_TS.length ? setTimeout(next, 60) : null;
      };
      next();
    }
    const isNarrow = watchNarrow(root, draw);
    ctrl.addEventListener("click", (e) => { if (e.target.dataset.a === "run") run(); });
    size.addEventListener("change", () => { clearTimeout(timer); timer = null; results = []; draw(); });
    onThemeChange(draw);
    draw();
  }

  // ---------------------------------------------------------------- Widget 6: denoise a picture
  function denoise(root) {
    const n = Ising.PIC, PX = 4, SIZE = n * PX;
    const plots = plotsRow(root);
    const panels = ["original", "noisy (what we observe)", "restored"].map((label) => {
      const box = document.createElement("div");
      box.style.flex = "1 1 90px";
      const canvas = document.createElement("canvas");
      canvas.width = SIZE; canvas.height = SIZE;
      const cap = document.createElement("div");
      cap.className = "viz-hint";
      cap.style.textAlign = "center";
      cap.textContent = label;
      box.append(canvas, cap);
      plots.appendChild(box);
      return canvas.getContext("2d");
    });
    const ctrl = control(root, `
      <label>noise <input type="range" min="5" max="30" step="5" value="10"> <span class="val"></span></label>
      <label>neighbour strength <b>J</b> <select>
        <option value="0">0: ignore neighbours</option>
        <option value="0.3">0.3: flip if all 4 disagree</option>
        <option value="1" selected>1: follow the majority</option></select></label>
      <button data-a="noise">New noise</button>`);
    const slider = ctrl.querySelector("input"), val = ctrl.querySelector(".val"), jSel = ctrl.querySelector("select");
    const out = readout(root);
    hint(root, "Red pixels are wrong (different from the original).");
    const clean = Ising.picture();
    let seed = 3;

    function paint(ctx, img, c) {
      ctx.fillStyle = c.bg; ctx.fillRect(0, 0, SIZE, SIZE);
      for (let i = 0; i < img.length; i++) {
        const wrong = img[i] !== clean[i];
        if (img[i] < 0 && !wrong) continue;
        ctx.fillStyle = wrong ? c.grad : c.text;
        ctx.fillRect((i % n) * PX, Math.floor(i / n) * PX, PX, PX);
      }
    }
    function draw() {
      const c = colors(), p = +slider.value / 100, J = +jSel.value;
      val.textContent = slider.value + "% of pixels flipped";
      const noisy = Ising.addNoise(clean, p, seed), res = Ising.restore(noisy, J);
      paint(panels[0], clean, c); paint(panels[1], noisy, c); paint(panels[2], res.s, c);
      const a = Ising.countWrong(noisy, clean), b = Ising.countWrong(res.s, clean), N = n * n;
      out.textContent = `wrong pixels · noisy: ${a} of ${N} (${pct(a / N)}) · restored: ${b} (${pct(b / N)}) · sweeps until nothing changed: ${res.sweeps}`;
    }
    slider.addEventListener("input", draw);
    jSel.addEventListener("change", draw);
    ctrl.addEventListener("click", (e) => { if (e.target.dataset.a === "noise") { seed++; draw(); } });
    onThemeChange(draw);
    draw();
  }

  const mounts = {
    "viz-spins": spinGrid, "viz-boltzmann": boltzmannTiles, "viz-flip": flipRule,
    "viz-lattice": liveLattice, "viz-sweep": sweepExperiment, "viz-denoise": denoise,
  };
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
