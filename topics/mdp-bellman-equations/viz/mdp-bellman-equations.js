// Interactive widgets for the "Markov Decision Processes and the Bellman Equations" lesson.
// Three widgets, each mounted on a <div> with a matching id:
//   #viz-robot  – the recycling-robot MDP: step through transitions and average sampled returns
//   #viz-backup – Bellman backup explorer on the cliff world: click a cell, see its one-step lookahead
//   #viz-qstar  – optimal action-values Q* on the cliff world, four triangles per cell
// Runs after DOMContentLoaded so the site-wide helpers (assets/js/viz-common.js) are available.
document.addEventListener("DOMContentLoaded", function () {
  const { colors, onThemeChange, linear, el, svg, axes, ticks, fmt } = window.Viz;
  const textStyle = "fill: var(--viz-text)";

  // ---------------------------------------------------------------- small helpers
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
    div.style.whiteSpace = "pre";
    div.style.overflowX = "auto";
    div.style.lineHeight = "1.5";
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

  // "−3.06" with a real minus sign, "+2" for positive rewards when asked
  const num = (v, d = 2, plus = false) => (v < 0 ? "−" : plus && v > 0 ? "+" : "") + fmt(Math.abs(v), d);
  const paren = (v, d = 2) => (v < 0 ? `(${num(v, d)})` : num(v, d));

  // Solve A x = b with Gaussian elimination (partial pivoting). Small systems only.
  function solveLinear(A, b) {
    const n = b.length;
    const M = A.map((row, i) => [...row, b[i]]);
    for (let k = 0; k < n; k++) {
      let p = k;
      for (let i = k + 1; i < n; i++) if (Math.abs(M[i][k]) > Math.abs(M[p][k])) p = i;
      [M[k], M[p]] = [M[p], M[k]];
      for (let i = k + 1; i < n; i++) {
        const f = M[i][k] / M[k][k];
        for (let j = k; j <= n; j++) M[i][j] -= f * M[k][j];
      }
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      let s = M[i][n];
      for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
      x[i] = s / M[i][i];
    }
    return x;
  }

  // Draw a (possibly curved) arrow from p to q. bend > 0 bends to the left of travel.
  // Returns the midpoint of the curve (for labels).
  function arrow(root, p, q, opts) {
    const { color, width = 1.6, bend = 0, dashed = false, opacity = 1, gapStart = 0, gapEnd = 0 } = opts;
    const dx = q[0] - p[0], dy = q[1] - p[1], L = Math.hypot(dx, dy);
    const mx = (p[0] + q[0]) / 2 - bend * dy, my = (p[1] + q[1]) / 2 + bend * dx;
    // move the ends back along the curve's tangents so they stop at the circles' edges
    const trim = (a, c, g) => { const ux = c[0] - a[0], uy = c[1] - a[1], l = Math.hypot(ux, uy) || 1; return [a[0] + (ux / l) * g, a[1] + (uy / l) * g]; };
    const a = trim(p, [mx, my], gapStart), b = trim(q, [mx, my], gapEnd);
    el("path", { d: `M${a[0]},${a[1]} Q${mx},${my} ${b[0]},${b[1]}`, fill: "none", stroke: color, "stroke-width": width, "stroke-dasharray": dashed ? "4 3" : "none", opacity }, root);
    // arrow head along the tangent at the end (from the control point to the end)
    const tx = b[0] - mx, ty = b[1] - my, tl = Math.hypot(tx, ty) || 1, ux = tx / tl, uy = ty / tl, h = 5 + width;
    el("path", { d: `M${b[0]},${b[1]} L${b[0] - ux * h - uy * h * 0.55},${b[1] - uy * h + ux * h * 0.55} L${b[0] - ux * h + uy * h * 0.55},${b[1] - uy * h - ux * h * 0.55} Z`, fill: color, opacity }, root);
    return [0.25 * a[0] + 0.5 * mx + 0.25 * b[0], 0.25 * a[1] + 0.5 * my + 0.25 * b[1], L];
  }

  // A text label with a background halo so it stays readable on top of lines
  function label(root, x, y, text, opts = {}) {
    const t = el("text", { x, y, "text-anchor": opts.anchor || "middle", "dominant-baseline": "middle", style: (opts.strong ? textStyle + ";" : "") + `font-size:${opts.size || 11}px; paint-order: stroke; stroke: ${colors().bg}; stroke-width: 4px; stroke-linejoin: round` }, root);
    t.textContent = text;
    return t;
  }

  // ================================================================ Widget 1: the recycling robot
  const RS = ["high", "low"];
  const RDYN = {
    high: { search: [[0.7, "high", 2], [0.3, "low", 2]], wait: [[1, "high", 1]] },
    low: { search: [[0.6, "low", 2], [0.4, "high", -3]], wait: [[1, "low", 1]], recharge: [[1, "high", 0]] },
  };
  const RGAMMA = 0.9;
  const RPOLICIES = {
    coin: { name: "Coin flip", pi: { high: { search: 1 }, low: { search: 0.5, recharge: 0.5 } } },
    search: { name: "Always search", pi: { high: { search: 1 }, low: { search: 1 } } },
    cautious: { name: "Cautious", pi: { high: { search: 1 }, low: { recharge: 1 } } },
    random: { name: "Random", pi: { high: { search: 0.5, wait: 0.5 }, low: { search: 1 / 3, wait: 1 / 3, recharge: 1 / 3 } } },
  };

  function robotValues(pi) {
    const P = [[0, 0], [0, 0]], r = [0, 0];
    RS.forEach((s, i) => {
      for (const a in pi[s]) for (const [p, s2, rw] of RDYN[s][a]) { P[i][RS.indexOf(s2)] += pi[s][a] * p; r[i] += pi[s][a] * p * rw; }
    });
    const A = P.map((row, i) => row.map((v, j) => (i === j ? 1 : 0) - RGAMMA * v));
    const V = solveLinear(A, r);
    return { high: V[0], low: V[1] };
  }

  function sample(pairs) {   // pairs: [[item, prob], ...]
    let u = Math.random();
    for (const [it, p] of pairs) if ((u -= p) <= 0) return it;
    return pairs[pairs.length - 1][0];
  }

  function robotStep(pi, s) {
    const a = sample(Object.entries(pi[s]));
    const [p, s2, rw] = sample(RDYN[s][a].map((o) => [o, o[0]]));
    return { a, pa: pi[s][a], p, s2, rw };
  }

  function robotWidget(root) {
    const W = 440, H = 320;
    const graph = svg(W, H);
    graph.style.maxWidth = "520px";
    graph.style.margin = "0 auto";
    root.appendChild(graph);
    const presets = control(root, `Policy π: ${Object.entries(RPOLICIES).map(([k, v]) => `<button data-p="${k}">${v.name}</button>`).join("")}`);
    const ctrl = control(root, `<button data-a="step">Take one step</button><button data-a="one">Sample 1 return</button><button data-a="many">Sample 1000 returns</button><button data-a="reset">Reset</button>`);
    const out = readout(root);
    const PW = 440, PH = 210, M = { l: 40, r: 12, t: 26, b: 36 };
    const plot = svg(PW, PH);
    plot.style.maxWidth = "520px";
    plot.style.margin = "10px auto 0";
    root.appendChild(plot);
    hint(root, "Each dot is the discounted return of one sampled run starting in “high” (200 steps; γ = 0.9). The orange line is their running average.");

    // layout of the graph: two state circles, one small dot per (state, action)
    const S = { high: [125, 165], low: [315, 165] }, SR = 36, DR = 7;
    const DOT = {
      "high:search": [220, 45], "high:wait": [40, 80],
      "low:search": [220, 285], "low:wait": [400, 80], "low:recharge": [220, 165],
    };
    // policy arrows (state → action) bend outwards, outcome arrows back to the same state bend inwards
    const POL = { "high:search": -0.15, "high:wait": -0.25, "low:search": -0.15, "low:wait": -0.25, "low:recharge": 0 };
    // how each outcome arrow bends, and where its "p · r" label sits (dx, dy from the curve midpoint)
    const OUT = {
      "high:search:high": { bend: -0.15, off: [14, 8] }, "high:search:low": { bend: 0, off: [16, -8] },
      "high:wait:high": { bend: -0.25, off: [12, -6] },
      "low:search:low": { bend: -0.15, off: [-14, -8] }, "low:search:high": { bend: 0, off: [-16, 8] },
      "low:wait:low": { bend: -0.25, off: [14, 4] },
      "low:recharge:high": { bend: 0, off: [0, -10] },
    };

    let key = "coin", pi, V, state, t, log, hl, samples, lastRun;

    function setPolicy(k) {
      key = k; pi = RPOLICIES[k].pi; V = robotValues(pi);
      reset();
    }

    function reset() {
      state = "high"; t = 0; log = []; hl = null; samples = []; lastRun = null;
      draw();
    }

    function drawGraph() {
      const c = colors();
      graph.innerHTML = "";
      // policy arrows (state → action dot) and outcome arrows (action dot → next state)
      for (const k in DOT) {
        const [s, a] = k.split(":"), d = DOT[k], pa = pi[s][a] || 0;
        const active = hl && hl.s === s && hl.a === a;
        arrow(graph, S[s], d, { color: c.model, width: active ? 3 : 1.4, bend: POL[k], dashed: pa === 0, opacity: pa === 0 ? 0.35 : active || !hl ? 1 : 0.45, gapStart: SR + 2, gapEnd: DR + 2 });
        for (const [p, s2, rw] of RDYN[s][a]) {
          const o = OUT[`${k}:${s2}`];
          const taken = active && hl.s2 === s2;
          const [lx, ly] = arrow(graph, d, S[s2], { color: taken ? c.model : c.data, width: taken ? 3 : 1.4, bend: o.bend, opacity: pa === 0 ? 0.3 : taken || !hl ? 1 : 0.4, gapStart: DR + 1, gapEnd: SR + 3 });
          label(graph, lx + o.off[0], ly + o.off[1], `${p.toFixed(1)} · ${rw > 0 ? "+" : rw < 0 ? "−" : ""}${Math.abs(rw)}`, { size: 10.5, strong: taken });
        }
      }
      for (const k in DOT) {
        const [s, a] = k.split(":"), d = DOT[k], pa = pi[s][a] || 0;
        const active = hl && hl.s === s && hl.a === a;
        el("circle", { cx: d[0], cy: d[1], r: active ? DR + 2 : DR, fill: pa === 0 ? c.bg : c.model, stroke: c.model, "stroke-width": 1.5, opacity: pa === 0 ? 0.5 : 1 }, graph);
        const above = d[1] < 150;
        const ty = d[1] + (above ? -15 : 17);
        label(graph, d[0], ty, a, { size: 11.5, strong: true });
        label(graph, d[0], ty + (above ? -13 : 13), `π=${pa === 1 ? "1" : pa === 0 ? "0" : fmt(pa, 2)}`, { size: 10 });
      }
      // state circles
      for (const s of RS) {
        const [x, y] = S[s], here = s === state;
        el("circle", { cx: x, cy: y, r: SR, fill: here ? c.model : c.data, "fill-opacity": here ? 0.22 : 0.14, stroke: here ? c.model : c.data, "stroke-width": here ? 4 : 1.8 }, graph);
        el("text", { x, y: y - 9, "text-anchor": "middle", style: textStyle + "; font-size: 13px; font-weight: 700" }, graph).textContent = s;
        el("text", { x, y: y + 6, "text-anchor": "middle", style: "font-size: 10px" }, graph).textContent = "battery";
        el("text", { x, y: y + 21, "text-anchor": "middle", style: textStyle + "; font-size: 10.5px" }, graph).textContent = `V=${fmt(V[s], 2)}`;
      }
      el("text", { x: 8, y: 314, style: "font-size: 10px" }, graph).textContent = "labels: probability · reward";
      el("text", { x: 432, y: 314, "text-anchor": "end", style: "font-size: 10px" }, graph).textContent = "orange ring = robot";
    }

    function drawPlot() {
      const c = colors();
      plot.innerHTML = "";
      const n = Math.max(samples.length, 10);
      const vals = samples.concat([V.high]);
      const lo = Math.floor(Math.min(...vals) - 1), hi = Math.ceil(Math.max(...vals) + 1);
      const x = linear(0, n, M.l, PW - M.r), y = linear(lo, hi, PH - M.b, M.t);
      const xstep = n <= 10 ? 2 : n <= 100 ? 20 : n <= 1000 ? 200 : n <= 3000 ? 500 : 1000;
      const ystep = hi - lo <= 8 ? 1 : hi - lo <= 20 ? 2 : 5;
      axes(plot, x, y, { xDomain: [0, n], yDomain: [lo, hi], xTicks: ticks(0, n, xstep), yTicks: ticks(lo, hi, ystep), xLabel: "number of sampled returns" });
      samples.forEach((g, i) => el("circle", { cx: x(i + 1), cy: y(g), r: samples.length > 200 ? 1.4 : 2.5, fill: c.data, "fill-opacity": 0.45 }, plot));
      let sum = 0, d = "";
      samples.forEach((g, i) => { sum += g; d += (i ? "L" : "M") + x(i + 1) + "," + y(sum / (i + 1)); });
      if (d) el("path", { d, fill: "none", stroke: c.model, "stroke-width": 2.2 }, plot);
      el("line", { x1: x(0), x2: x(n), y1: y(V.high), y2: y(V.high), stroke: c.loss, "stroke-width": 2, "stroke-dasharray": "6 4" }, plot);
      el("text", { x: M.l + 4, y: 14, style: textStyle }, plot).textContent = `dashed: exact V(high) = ${fmt(V.high, 2)}`;
    }

    function draw() {
      drawGraph();
      drawPlot();
      const lines = [];
      lines.push(`policy “${RPOLICIES[key].name}”, γ = 0.9`, `exact: V(high) = ${fmt(V.high, 2)}, V(low) = ${fmt(V.low, 2)}`);
      if (log.length) lines.push("", ...log.slice(-4));
      else lines.push("", "Press “Take one step” to watch it move.");
      if (lastRun) {
        const r = lastRun.rewards, g = RGAMMA;
        lines.push("", `last run: rewards ${r.slice(0, 5).map((v) => num(v, 0, true)).join(", ")}, …`,
          `  G₀ = ${num(r[0], 0)} + 0.9·${paren(r[1], 0)} + 0.81·${paren(r[2], 0)} + … = ${num(lastRun.G0)}`,
          `  G₁ = ${num(lastRun.G1)}`,
          `  r₁ + γG₁ = ${num(r[0], 0)} + 0.9·${paren(lastRun.G1)} = ${num(r[0] + g * lastRun.G1)} ✓`);
      }
      if (samples.length) {
        const avg = samples.reduce((a, b) => a + b, 0) / samples.length;
        lines.push("", `${samples.length} sampled return${samples.length > 1 ? "s" : ""} from high:`, `  average ${num(avg)} vs exact V(high) ${num(V.high)}`);
      }
      out.textContent = lines.join("\n");
    }

    function takeStep() {
      const s = state, o = robotStep(pi, s);
      t += 1;
      log.push(`t=${t}  ${s}, ${o.a} → ${o.s2}, r = ${num(o.rw, 0, true)}`);
      hl = { s, a: o.a, s2: o.s2 };
      state = o.s2;
      draw();
    }

    function run() {   // one 200-step run from "high"; returns rewards and G0, G1
      let s = "high";
      const rewards = [];
      for (let k = 0; k < 200; k++) { const o = robotStep(pi, s); rewards.push(o.rw); s = o.s2; }
      let G = 0;
      const Gs = new Array(rewards.length + 1).fill(0);
      for (let k = rewards.length - 1; k >= 0; k--) { G = rewards[k] + RGAMMA * G; Gs[k] = G; }   // G_t = r_{t+1} + γ G_{t+1}
      return { rewards, G0: Gs[0], G1: Gs[1] };
    }

    presets.addEventListener("click", (e) => { if (e.target.dataset.p) setPolicy(e.target.dataset.p); });
    ctrl.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (a === "step") takeStep();
      if (a === "reset") reset();
      if (a === "one") { lastRun = run(); samples.push(lastRun.G0); draw(); }
      if (a === "many") { for (let k = 0; k < 1000; k++) samples.push(run().G0); draw(); }
    });
    onThemeChange(draw);
    setPolicy("coin");
  }

  // ================================================================ The cliff world (from lesson 1)
  const MAP = ["......", "..#...", "......", "SCCCCG"];
  const ROWS = MAP.length, COLS = MAP[0].length;
  const START = [3, 0];
  const GAMMA = 0.95;
  const ACTIONS = [[-1, 0], [0, 1], [1, 0], [0, -1]];   // up, right, down, left
  const ARROWS = ["↑", "→", "↓", "←"];
  const tile = (r, c) => MAP[r][c];
  const isTerminal = (r, c) => tile(r, c) === "G" || tile(r, c) === "C";
  const isFloor = (r, c) => tile(r, c) === "." || tile(r, c) === "S";
  const FLOOR = [];
  for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) if (isFloor(r, c)) FLOOR.push([r, c]);
  const IDX = {};
  FLOOR.forEach(([r, c], i) => { IDX[r * COLS + c] = i; });

  function move(r, c, a) {
    const nr = r + ACTIONS[a][0], nc = c + ACTIONS[a][1];
    if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS || tile(nr, nc) === "#") return [r, c];
    return [nr, nc];
  }
  const reward = (r, c) => (tile(r, c) === "G" ? 10 : tile(r, c) === "C" ? -20 : -1);
  // Every result of trying action a: [probability, row, col, reward, done]
  function outcomes(r, c, a, slip) {
    return [[1 - slip, a], [slip / 2, (a + 1) % 4], [slip / 2, (a + 3) % 4]]
      .filter(([p]) => p > 1e-12)
      .map(([p, d]) => { const [nr, nc] = move(r, c, d); return [p, nr, nc, reward(nr, nc), isTerminal(nr, nc)]; });
  }
  const zeros = () => MAP.map((row) => [...row].map(() => 0));

  // Q(s, a) = Σ p · (r + γ V(s′)) — the one-step lookahead
  const qValue = (V, r, c, a, slip) => outcomes(r, c, a, slip).reduce((sum, [p, nr, nc, rw, end]) => sum + p * (rw + (end ? 0 : GAMMA * V[nr][nc])), 0);

  // Bellman expectation equation as a linear system: (I − γ P^π) V = r^π.  probs[r][c] = [π(↑), π(→), π(↓), π(←)]
  function evaluate(probs, slip) {
    const n = FLOOR.length;
    const A = FLOOR.map((_, i) => FLOOR.map((__, j) => (i === j ? 1 : 0))), b = new Array(n).fill(0);
    FLOOR.forEach(([r, c], i) => {
      probs[r][c].forEach((pa, a) => {
        if (!pa) return;
        for (const [p, nr, nc, rw, end] of outcomes(r, c, a, slip)) {
          b[i] += pa * p * rw;
          if (!end) A[i][IDX[nr * COLS + nc]] -= GAMMA * pa * p;
        }
      });
    });
    const x = solveLinear(A, b), V = zeros();
    FLOOR.forEach(([r, c], i) => { V[r][c] = x[i]; });
    return V;
  }

  // Bellman optimality equation, solved by repeating V(s) ← max_a Q(s, a) (value iteration, next lesson)
  function optimal(slip) {
    let V = zeros();
    for (let k = 0; k < 5000; k++) {
      let delta = 0;
      const Vn = zeros();
      for (const [r, c] of FLOOR) {
        Vn[r][c] = Math.max(...[0, 1, 2, 3].map((a) => qValue(V, r, c, a, slip)));
        delta = Math.max(delta, Math.abs(Vn[r][c] - V[r][c]));
      }
      V = Vn;
      if (delta < 1e-11) break;
    }
    const Q = zeros().map((row, r) => row.map((_, c) => (isFloor(r, c) ? [0, 1, 2, 3].map((a) => qValue(V, r, c, a, slip)) : null)));
    return { V, Q };
  }

  const CELL = 56, PAD = 6;
  const GW = COLS * CELL + 2 * PAD, GH = ROWS * CELL + 2 * PAD;
  const px = (c) => PAD + c * CELL, py = (r) => PAD + r * CELL;
  const cellName = (r, c) => (r === START[0] && c === START[1] ? "start" : `${r},${c}`);

  function drawBoard(root, opts = {}) {
    const col = colors();
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const t = tile(r, c), x = px(c), y = py(r);
        el("rect", { x, y, width: CELL, height: CELL, fill: col.bg, stroke: col.border }, root);
        if (t === "#") el("rect", { x: x + 1, y: y + 1, width: CELL - 2, height: CELL - 2, fill: col.muted, "fill-opacity": 0.55 }, root);
        if (t === "G" || t === "C") {
          el("rect", { x: x + 1, y: y + 1, width: CELL - 2, height: CELL - 2, fill: t === "G" ? col.opt : col.grad, "fill-opacity": t === "G" ? 0.45 : 0.6 }, root);
          el("text", { x: x + CELL / 2, y: y + CELL / 2 - 3, "text-anchor": "middle", style: textStyle }, root).textContent = t === "G" ? "GOAL" : "cliff";
          el("text", { x: x + CELL / 2, y: y + CELL / 2 + 12, "text-anchor": "middle" }, root).textContent = t === "G" ? "+10" : "−20";
        }
        if (isFloor(r, c) && opts.tint) {
          const [fill, op] = opts.tint(r, c);
          el("rect", { x: x + 1, y: y + 1, width: CELL - 2, height: CELL - 2, fill, "fill-opacity": op }, root);
        }
        if (t === "S" && !opts.noStart) el("text", { x: x + 4, y: y + 12, style: "font-size: 9.5px" }, root).textContent = "start";
      }
    }
  }

  function boardSvg(root) {
    const s = svg(GW, GH);
    s.style.maxWidth = "520px";
    s.style.margin = "0 auto";
    s.style.cursor = "pointer";
    root.appendChild(s);
    return s;
  }

  function cellAt(board, e) {
    const b = board.getBoundingClientRect();
    const c = Math.floor((((e.clientX - b.left) / b.width) * GW - PAD) / CELL);
    const r = Math.floor((((e.clientY - b.top) / b.height) * GH - PAD) / CELL);
    return r >= 0 && r < ROWS && c >= 0 && c < COLS && isFloor(r, c) ? [r, c] : null;
  }

  // One outcome line of a backup: p × (r + γ V(s′)) = …
  function term([p, nr, nc, rw, end], V) {
    if (end) return ` ${fmt(p, 2)} × (${num(rw, 0)} + 0 [${tile(nr, nc) === "G" ? "goal" : "cliff"}])`.padEnd(31) + `= ${num(p * rw, 3)}`;
    return ` ${fmt(p, 2)} × (${num(rw, 0)} + 0.95 × ${paren(V[nr][nc])})`.padEnd(31) + `= ${num(p * (rw + GAMMA * V[nr][nc]), 3)}`;
  }

  const sel = (board, r, c, color) => el("rect", { x: px(c) + 2, y: py(r) + 2, width: CELL - 4, height: CELL - 4, fill: "none", stroke: color, "stroke-width": 3, rx: 3 }, board);

  // ================================================================ Widget 2: Bellman backup explorer
  const PRESETS = {
    edge: ["↓↓→↓↓↓", "↓↓.↓↓↓", "→→→→→↓", "↑....."],
    safe: ["→→→→→↓", "↑↑.↑↑↓", "↑↑←↑↑↓", "↑....."],
  };

  function backupWidget(root) {
    const board = boardSvg(root);
    hint(root, "Click any floor cell to write out its Bellman equation. Outlined cells are where you can land next, with the chance of landing there.");
    const presets = control(root, `Policy π: <button data-p="random">Random</button><button data-p="edge">Edge of the cliff</button><button data-p="safe">Safe detour</button><button data-p="optimal">Optimal</button>`);
    const ctrl = control(root, `<label>slip chance <input type="range" min="0" max="0.4" step="0.05" value="0"> <span></span></label><span>γ = 0.95</span>`);
    const slipIn = ctrl.querySelector("input");
    const out = readout(root);

    let preset = "edge", probs, V, cur = [2, 1];

    function refresh() {
      const slip = +slipIn.value;
      if (preset === "random") probs = zeros().map((row) => row.map(() => [0.25, 0.25, 0.25, 0.25]));
      else if (preset === "optimal") {
        const { Q } = optimal(slip);
        probs = zeros().map((row, r) => row.map((_, c) => {
          if (!isFloor(r, c)) return [0, 0, 0, 0];
          const best = Q[r][c].indexOf(Math.max(...Q[r][c]));
          return [0, 1, 2, 3].map((a) => (a === best ? 1 : 0));
        }));
      } else probs = zeros().map((row, r) => row.map((_, c) => [0, 1, 2, 3].map((a) => (isFloor(r, c) && ARROWS[a] === PRESETS[preset][r][c] ? 1 : 0))));
      V = evaluate(probs, slip);
      draw();
    }

    function draw() {
      const col = colors(), slip = +slipIn.value;
      slipIn.nextElementSibling.textContent = slip.toFixed(2);
      board.innerHTML = "";
      drawBoard(board, { tint: (r, c) => [V[r][c] >= 0 ? col.opt : col.grad, Math.min(1, Math.abs(V[r][c]) / 12) * 0.4] });
      // where can we land from the selected cell? total probability per next cell
      const [sr, sc] = cur, land = {};
      probs[sr][sc].forEach((pa, a) => {
        if (pa) for (const [p, nr, nc] of outcomes(sr, sc, a, slip)) land[nr * COLS + nc] = (land[nr * COLS + nc] || 0) + pa * p;
      });
      for (const r0 in land) {
        const r = Math.floor(r0 / COLS), c = r0 % COLS;
        el("rect", { x: px(c) + 3, y: py(r) + 3, width: CELL - 6, height: CELL - 6, fill: "none", stroke: col.data, "stroke-width": 2, "stroke-dasharray": "4 3", rx: 3 }, board);
        label(board, px(c) + CELL - 5, py(r) + 11, fmt(land[r0], 2), { anchor: "end", size: 9.5, strong: true });
      }
      for (const [r, c] of FLOOR) {
        const cx = px(c) + CELL / 2, cy = py(r) + CELL / 2, pr = probs[r][c];
        const k = pr.filter((p) => p > 0).length;
        el("text", { x: cx, y: cy + 4, "text-anchor": "middle", style: textStyle + "; font-size: 18px" }, board).textContent = k > 1 ? "✱" : ARROWS[pr.indexOf(1)];
        el("text", { x: cx, y: py(r) + CELL - 5, "text-anchor": "middle", style: textStyle + "; font-size: 10px" }, board).textContent = fmt(V[r][c], 2);
      }
      sel(board, sr, sc, col.model);
      writeOut();
    }

    function writeOut() {
      const slip = +slipIn.value, [r, c] = cur, name = cellName(r, c);
      const lines = [`V(${name}) = Σₐ π(a|s) Σ p·[r + γ·V(s′)]`, ""];
      let total = 0;
      probs[r][c].forEach((pa, a) => {
        if (!pa) return;
        const q = qValue(V, r, c, a, slip);
        total += pa * q;
        lines.push(`action ${ARROWS[a]}  π = ${fmt(pa, 2)}   → Q = ${num(q)}`);
        for (const o of outcomes(r, c, a, slip)) lines.push(term(o, V));
      });
      const stored = V[r][c], res = Math.abs(stored - total);
      lines.push("", `one-step lookahead = ${num(total, 4)}`, `stored V(${name})`.padEnd(19) + `= ${num(stored, 4)}`,
        `residual = ${res === 0 ? "0" : res.toExponential(0)}  ${res < 1e-9 ? "✓ the equation holds" : ""}`);
      out.textContent = lines.join("\n");
    }

    board.addEventListener("click", (e) => { const rc = cellAt(board, e); if (rc) { cur = rc; draw(); } });
    presets.addEventListener("click", (e) => { if (e.target.dataset.p) { preset = e.target.dataset.p; refresh(); } });
    slipIn.addEventListener("input", refresh);
    onThemeChange(draw);
    refresh();
  }

  // ================================================================ Widget 3: Q* as four triangles per cell
  function qstarWidget(root) {
    const board = boardSvg(root);
    hint(root, "Each cell has four triangles, one per action, coloured by Q*(s, a): green = good, red = bad. Arrows mark the greedy action. Click a cell for the numbers.");
    const ctrl = control(root, `<label>slip chance <input type="range" min="0" max="0.4" step="0.05" value="0"> <span></span></label><span>γ = 0.95</span><label><input type="checkbox" checked> show greedy path from start</label>`);
    const [slipIn, pathIn] = ctrl.querySelectorAll("input");
    const out = readout(root);
    let opt, cur = [3, 0];

    function refresh() { opt = optimal(+slipIn.value); draw(); }

    const best = (q) => Math.max(...q);
    const greedy = (q) => [0, 1, 2, 3].filter((a) => q[a] > best(q) - 1e-6);

    function draw() {
      const col = colors(), slip = +slipIn.value, { V, Q } = opt;
      slipIn.nextElementSibling.textContent = slip.toFixed(2);
      board.innerHTML = "";
      drawBoard(board, { noStart: true });
      for (const [r, c] of FLOOR) {
        const x = px(c), y = py(r), cx = x + CELL / 2, cy = y + CELL / 2, q = Q[r][c], g = greedy(q);
        const corners = [[x, y], [x + CELL, y], [x + CELL, y + CELL], [x, y + CELL]];
        // triangle for action a: centre + the two corners on that side (up = top edge, right = right edge, …)
        [0, 1, 2, 3].forEach((a) => {
          const p1 = corners[a], p2 = corners[(a + 1) % 4];
          el("path", { d: `M${cx},${cy} L${p1[0]},${p1[1]} L${p2[0]},${p2[1]} Z`, fill: q[a] >= 0 ? col.opt : col.grad, "fill-opacity": 0.08 + Math.min(1, Math.abs(q[a]) / 15) * 0.55, stroke: col.bg, "stroke-width": 1.5 }, board);
        });
      }
      if (pathIn.checked) {   // follow the greedy action from the start, ignoring slips
        let [r, c] = START;
        const pts = [[r, c]];
        for (let k = 0; k < 30 && isFloor(r, c); k++) { [r, c] = move(r, c, greedy(Q[r][c])[0]); if (pts.some(([a, b]) => a === r && b === c)) break; pts.push([r, c]); }
        const d = pts.map(([pr, pc], i) => (i ? "L" : "M") + (px(pc) + CELL / 2) + "," + (py(pr) + CELL / 2)).join("");
        el("path", { d, fill: "none", stroke: col.model, "stroke-width": 3.5, "stroke-opacity": 0.5, "stroke-linejoin": "round", "stroke-linecap": "round" }, board);
        el("circle", { cx: px(START[1]) + CELL / 2, cy: py(START[0]) + CELL / 2, r: 6, fill: col.model, stroke: col.bg, "stroke-width": 2 }, board);
      }
      for (const [r, c] of FLOOR) {   // greedy arrows on top of the path
        const cx = px(c) + CELL / 2, cy = py(r) + CELL / 2, g = greedy(Q[r][c]);
        g.forEach((a) => {
          const d = ACTIONS[a];
          el("text", { x: cx + d[1] * 15, y: cy + d[0] * 15 + 5, "text-anchor": "middle", style: textStyle + "; font-size: 15px; font-weight: 700" }, board).textContent = ARROWS[a];
        });
      }
      sel(board, cur[0], cur[1], col.model);

      const [r, c] = cur, q = Q[r][c], g = greedy(q), name = cellName(r, c);
      const lines = [`Q*(${name}, a) for each action a:`];
      q.forEach((v, a) => lines.push(`  ${ARROWS[a]}  ${num(v).padStart(7)}${g.includes(a) ? "   ← best" : ""}`));
      lines.push(`V*(${name}) = maxₐ Q*(${name}, a) = ${num(V[r][c])}`, "", `Q*(${name}, ${ARROWS[g[0]]}) = Σ p·[r + γ·V*(s′)]`, `  where V*(s′) = maxₐ′ Q*(s′, a′)`);
      for (const o of outcomes(r, c, g[0], slip)) lines.push(term(o, V));
      lines.push("", `V*(start) = ${num(V[START[0]][START[1]])} at slip ${slip.toFixed(2)}`);
      out.textContent = lines.join("\n");
    }

    board.addEventListener("click", (e) => { const rc = cellAt(board, e); if (rc) { cur = rc; draw(); } });
    slipIn.addEventListener("input", refresh);
    pathIn.addEventListener("change", draw);
    onThemeChange(draw);
    refresh();
  }

  const mounts = { "viz-robot": robotWidget, "viz-backup": backupWidget, "viz-qstar": qstarWidget };
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
