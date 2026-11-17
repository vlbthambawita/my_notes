// Interactive widgets for the "Actor-critic methods" lesson.
// Three widgets, each mounted on a <div> with a matching id:
//   #viz-ac-trace – a tabular one-step actor-critic learning the cliff world, one transition at a time
//   #viz-ppo-clip – the PPO clipped objective as a function of the probability ratio r, for A > 0 and A < 0
//   #viz-entropy  – a softmax policy with a temperature α: probabilities and entropy
// Runs after DOMContentLoaded so the site-wide helpers (assets/js/viz-common.js) are available.
document.addEventListener("DOMContentLoaded", function () {
  const { colors, onThemeChange, linear, el, svg, axes, ticks, fmt } = window.Viz;
  const textStyle = "fill: var(--viz-text)";

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
    div.textContent = text;
    parent.appendChild(div);
    return div;
  }

  // x position (in viewBox units) of a pointer event over an SVG of width W
  function pointerX(svgNode, e, W) {
    const b = svgNode.getBoundingClientRect();
    return ((e.clientX - b.left) / b.width) * W;
  }

  // ---------------------------------------------------------------- The cliff world (same as lesson 1)
  const MAP = ["......", "..#...", "......", "SCCCCG"];
  const ROWS = MAP.length, COLS = MAP[0].length;
  const START = [3, 0];
  const ACTIONS = [[-1, 0], [0, 1], [1, 0], [0, -1]];   // up, right, down, left
  const ARROWS = ["↑", "→", "↓", "←"];
  const GAMMA = 0.95;
  const tile = (r, c) => MAP[r][c];
  const isFloor = (r, c) => tile(r, c) === "." || tile(r, c) === "S";

  function envStep(r, c, a) {
    let nr = r + ACTIONS[a][0], nc = c + ACTIONS[a][1];
    if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS || tile(nr, nc) === "#") { nr = r; nc = c; }
    const t = tile(nr, nc);
    if (t === "G") return [nr, nc, 10, true];
    if (t === "C") return [nr, nc, -20, true];
    return [nr, nc, -1, false];
  }

  const CELL = 64, PAD = 6;
  const GW = COLS * CELL + 2 * PAD, GH = ROWS * CELL + 2 * PAD;
  const px = (c) => PAD + c * CELL, py = (r) => PAD + r * CELL;

  // ---------------------------------------------------------------- Widget 1: actor-critic trace
  function traceWidget(root) {
    const board = svg(GW, GH);
    board.style.maxWidth = "540px";
    board.style.margin = "0 auto";
    root.appendChild(board);
    hint(root, "Each cell: four triangles = the actor's probabilities π(a|s) (darker = more likely), number = the critic's V(s). Cell colour: green = good, red = bad.");
    const ctrl = control(root, `
      <button data-a="step">1 step</button><button data-a="ep">1 episode</button><button data-a="many">20 episodes</button><button data-a="reset">Reset</button>`);
    const sliders = control(root, `
      <label><span>critic step α<sub>w</sub></span> <input type="range" min="0.05" max="0.5" step="0.05" value="0.1"> <span></span></label>
      <label><span>actor step α<sub>θ</sub></span> <input type="range" min="0.05" max="0.5" step="0.05" value="0.1"> <span></span></label>`);
    const [awIn, athIn] = sliders.querySelectorAll("input");
    const out = readout(root);

    let V, H, pos, t, episodes, history, trail, last;

    function policy(r, c) {
      const h = H[r][c], m = Math.max(...h);
      const e = h.map((v) => Math.exp(v - m));
      const s = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / s);
    }

    function sample(p) {
      let u = Math.random();
      for (let i = 0; i < p.length; i++) if ((u -= p[i]) <= 0) return i;
      return p.length - 1;
    }

    function reset() {
      V = MAP.map((row) => [...row].map(() => 0));
      H = MAP.map((row) => [...row].map(() => [0, 0, 0, 0]));
      episodes = 0; history = []; last = null;
      newEpisode();
      draw();
    }

    function newEpisode() { pos = START.slice(); t = 0; trail = [pos.slice()]; }

    // One transition: act, compute δ, update critic and actor. Returns a description.
    function oneStep() {
      if (t === 0) trail = [pos.slice()];
      const aw = +awIn.value, ath = +athIn.value;
      const [r, c] = pos;
      const p = policy(r, c);
      const a = sample(p);
      const [nr, nc, rw, done] = envStep(r, c, a);
      const vS = V[r][c], vNext = done ? 0 : V[nr][nc];
      const delta = rw + GAMMA * vNext - vS;
      V[r][c] = vS + aw * delta;                                   // critic update
      for (let b = 0; b < 4; b++) H[r][c][b] += ath * delta * ((b === a ? 1 : 0) - p[b]);   // actor update
      const pNew = policy(r, c)[a];
      t += 1; pos = [nr, nc]; trail.push(pos.slice());
      last = { r, c, a, p: p[a], pNew, rw, nr, nc, done, vS, vNext, delta, vNew: V[r][c], aw };
      if (done || t >= 100) {
        episodes += 1;
        history.push(done ? (rw > 0 ? "goal" : "cliff") : "timeout");
        if (history.length > 20) history.shift();
        last.end = done ? (rw > 0 ? "reached the goal" : "fell off the cliff") : "gave up after 100 steps";
        last.steps = t;
        newEpisode();
        return true;
      }
      return false;
    }

    function runEpisode() { while (!oneStep()); }

    function draw() {
      const col = colors();
      awIn.nextElementSibling.textContent = (+awIn.value).toFixed(2);
      athIn.nextElementSibling.textContent = (+athIn.value).toFixed(2);
      board.innerHTML = "";
      const overlay = [];   // triangles and numbers, drawn on top of the trail
      for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
          const tl = tile(r, c), x = px(c), y = py(r);
          el("rect", { x, y, width: CELL, height: CELL, fill: col.bg, stroke: col.border }, board);
          if (tl === "#") el("rect", { x: x + 1, y: y + 1, width: CELL - 2, height: CELL - 2, fill: col.muted, "fill-opacity": 0.55 }, board);
          if (tl === "G" || tl === "C") {
            el("rect", { x: x + 1, y: y + 1, width: CELL - 2, height: CELL - 2, fill: tl === "G" ? col.opt : col.grad, "fill-opacity": tl === "G" ? 0.45 : 0.6 }, board);
            el("text", { x: x + CELL / 2, y: y + CELL / 2 - 3, "text-anchor": "middle", style: textStyle }, board).textContent = tl === "G" ? "GOAL" : "cliff";
            el("text", { x: x + CELL / 2, y: y + CELL / 2 + 12, "text-anchor": "middle" }, board).textContent = tl === "G" ? "+10" : "−20";
          }
          if (!isFloor(r, c)) continue;
          const v = V[r][c];
          el("rect", { x: x + 1, y: y + 1, width: CELL - 2, height: CELL - 2, fill: v >= 0 ? col.opt : col.grad, "fill-opacity": Math.min(1, Math.abs(v) / 12) * 0.4 }, board);
          if (tl === "S") el("text", { x: x + 3, y: y + 11, style: "font-size: 9px" }, board).textContent = "start";
          // actor: one triangle per direction, opacity = probability
          overlay.push(() => {
            const p = policy(r, c), cx = x + CELL / 2, cy = y + CELL / 2, near = 13, far = 27, half = 8;
            for (let a = 0; a < 4; a++) {
              const [dr, dc] = ACTIONS[a];
              const tip = [cx + dc * far, cy + dr * far];
              const b1 = [cx + dc * near - dr * half, cy + dr * near - dc * half];
              const b2 = [cx + dc * near + dr * half, cy + dr * near + dc * half];
              el("polygon", { points: [tip, b1, b2].map((q) => q.join(",")).join(" "), fill: col.model, "fill-opacity": 0.12 + 0.88 * p[a] }, board);
            }
            el("text", { x: cx, y: cy + 4, "text-anchor": "middle", style: textStyle + "; font-size: 10px" }, board).textContent = fmt(v, 1);
          });
        }
      }
      if (trail.length > 1) {
        const d = trail.map(([r, c], i) => (i ? "L" : "M") + (px(c) + CELL / 2) + "," + (py(r) + CELL / 2)).join("");
        el("path", { d, fill: "none", stroke: col.data, "stroke-width": 3, "stroke-opacity": 0.35, "stroke-linejoin": "round" }, board);
      }
      overlay.forEach((f) => f());
      // the agent: a blue frame around its cell (keeps the numbers readable)
      el("rect", { x: px(pos[1]) + 2.5, y: py(pos[0]) + 2.5, width: CELL - 5, height: CELL - 5, rx: 4, fill: "none", stroke: col.data, "stroke-width": 4 }, board);

      let text;
      if (!last) {
        text = "The actor starts uniform (every direction 0.25) and the critic knows nothing (V = 0 everywhere). Press “1 step”.";
      } else {
        const L = last;
        const num = (v) => (v < 0 ? "(" + fmt(v, 2) + ")" : fmt(v, 2));
        text = `in (${L.r},${L.c}) the actor sampled ${ARROWS[L.a]} (π = ${fmt(L.p, 2)})  →  reward ${L.rw}, landed in (${L.nr},${L.nc})${L.done ? "  [episode over, so V(s′) = 0]" : ""}\n` +
          `TD error: δ = r + γ·V(s′) − V(s) = ${L.rw} + 0.95 × ${num(L.vNext)} − ${num(L.vS)} = ${fmt(L.delta, 2)}\n` +
          `critic:   V(s) ← V(s) + α_w·δ = ${fmt(L.vS, 2)} + ${L.aw.toFixed(2)} × ${num(L.delta)} = ${fmt(L.vNew, 2)}\n` +
          `actor:    π(${ARROWS[L.a]} | s) ${fmt(L.p, 2)} → ${fmt(L.pNew, 2)}  (${L.delta >= 0 ? "δ > 0: better than expected, do it more" : "δ < 0: worse than expected, do it less"})`;
        if (L.end) text += `\nepisode ${episodes} ${L.end} after ${L.steps} steps`;
      }
      if (history.length) {
        const n = (k) => history.filter((h) => h === k).length;
        text += `\nepisodes so far: ${episodes}   last ${history.length}: goal ${n("goal")}, cliff ${n("cliff")}, timeout ${n("timeout")}`;
      }
      out.textContent = text;
    }

    ctrl.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (a === "step") oneStep();
      else if (a === "ep") runEpisode();
      else if (a === "many") for (let k = 0; k < 20; k++) runEpisode();
      else if (a === "reset") { reset(); return; }
      else return;
      draw();
    });
    awIn.addEventListener("input", draw);
    athIn.addEventListener("input", draw);
    onThemeChange(draw);
    reset();
  }

  // ---------------------------------------------------------------- Widget 2: PPO clip explorer
  function clipWidget(root) {
    const W = 320, H = 240, M = { l: 40, r: 12, t: 30, b: 38 };
    const plots = document.createElement("div");
    plots.className = "viz-plots";
    root.appendChild(plots);
    const CASES = [
      { A: 1, name: "A > 0 (good action, A = +1)", y0: 0, y1: 2 },
      { A: -1, name: "A < 0 (bad action, A = −1)", y0: -2, y1: 0 },
    ];
    const charts = CASES.map(() => { const s = svg(W, H); s.style.touchAction = "none"; s.style.cursor = "ew-resize"; plots.appendChild(s); return s; });
    hint(root, "Drag on either chart (or use the slider) to move the ratio r. Red band: the objective is flat there, so this sample gives zero gradient.");
    const ctrl = control(root, `
      <label><span>ratio r = π<sub>θ</sub> / π<sub>old</sub></span> <input type="range" min="0" max="2" step="0.01" value="1.3"> <span></span></label>
      <label>clip range ε <input type="range" min="0.05" max="0.5" step="0.05" value="0.2"> <span></span></label>`);
    const [rIn, eIn] = ctrl.querySelectorAll("input");
    const out = readout(root);
    const x = linear(0, 2, M.l, W - M.r);
    const clip = (r, e) => Math.min(Math.max(r, 1 - e), 1 + e);
    const objective = (r, A, e) => Math.min(r * A, clip(r, e) * A);
    const gradZero = (r, A, e) => (A > 0 ? r > 1 + e : r < 1 - e);

    function draw() {
      const col = colors(), r = +rIn.value, e = +eIn.value;
      rIn.nextElementSibling.textContent = r.toFixed(2);
      eIn.nextElementSibling.textContent = e.toFixed(2);
      const lines = [];
      CASES.forEach((cs, i) => {
        const g = charts[i];
        g.innerHTML = "";
        const y = linear(cs.y0, cs.y1, H - M.b, M.t);
        // zero-gradient band
        const [b0, b1] = cs.A > 0 ? [1 + e, 2] : [0, 1 - e];
        el("rect", { x: x(b0), y: M.t, width: x(b1) - x(b0), height: H - M.t - M.b, fill: col.grad, "fill-opacity": 0.12 }, g);
        axes(g, x, y, { xDomain: [0, 2], yDomain: [cs.y0, cs.y1], xTicks: ticks(0, 2, 0.5), yTicks: ticks(cs.y0, cs.y1, 0.5), xLabel: "ratio r" });
        el("text", { x: (x(b0) + x(b1)) / 2, y: cs.A > 0 ? H - M.b - 8 : M.t + 14, "text-anchor": "middle", style: `fill: ${col.grad}` }, g).textContent = "gradient = 0";
        // old-policy line
        el("line", { x1: x(1), x2: x(1), y1: M.t, y2: H - M.b, stroke: col.muted, "stroke-dasharray": "3 3" }, g);
        el("text", { x: x(1) + 4, y: cs.A > 0 ? M.t + 12 : H - M.b - 8 }, g).textContent = "r = 1 (no change)";
        // unclipped r·A
        el("line", { x1: x(0), y1: y(0), x2: x(2), y2: y(2 * cs.A), stroke: col.muted, "stroke-width": 1.5, "stroke-dasharray": "6 4" }, g);
        // clipped objective
        const pts = [];
        for (let k = 0; k <= 200; k++) { const rv = (2 * k) / 200; pts.push(`${x(rv)},${y(objective(rv, cs.A, e))}`); }
        el("polyline", { points: pts.join(" "), fill: "none", stroke: col.model, "stroke-width": 3 }, g);
        el("text", { x: M.l + 4, y: 16, style: textStyle }, g).textContent = cs.name;
        const z = gradZero(r, cs.A, e), o = objective(r, cs.A, e);
        el("line", { x1: x(r), x2: x(r), y1: M.t, y2: H - M.b, stroke: z ? col.grad : col.model, "stroke-opacity": 0.5 }, g);
        el("circle", { cx: x(r), cy: y(o), r: 7, fill: z ? col.grad : col.model, stroke: col.bg, "stroke-width": 2 }, g);
        lines.push(`A = ${cs.A > 0 ? "+1" : "−1"}:  r·A = ${fmt(r * cs.A, 2)}   clip(r)·A = ${fmt(clip(r, e) * cs.A, 2)}   min = ${fmt(o, 2)}   →  ${z ? "gradient 0 (clipped: this sample stops pushing)" : "gradient flows (slope " + (cs.A > 0 ? "+1" : "−1") + ")"}`);
      });
      out.textContent = `dashed grey = unclipped r·A, orange = PPO objective min(r·A, clip(r, 1−ε, 1+ε)·A)\n` + lines.join("\n");
    }

    charts.forEach((s) => {
      const move = (ev) => {
        if (ev.type === "pointermove" && !ev.buttons) return;
        const v = Math.min(2, Math.max(0, x.invert(pointerX(s, ev, W))));
        rIn.value = v.toFixed(2);
        draw();
      };
      s.addEventListener("pointerdown", (ev) => { s.setPointerCapture(ev.pointerId); move(ev); });
      s.addEventListener("pointermove", move);
    });
    rIn.addEventListener("input", draw);
    eIn.addEventListener("input", draw);
    onThemeChange(draw);
    draw();
  }

  // ---------------------------------------------------------------- Widget 3: temperature and entropy
  function entropyWidget(root) {
    const W = 360, H = 250, M = { l: 40, r: 12, t: 34, b: 44 };
    const Q = [3, 2.5, 1, 0];
    const NAMES = ["a₁", "a₂", "a₃", "a₄"];
    const chart = svg(W, H);
    chart.style.maxWidth = "520px";
    chart.style.margin = "0 auto";
    root.appendChild(chart);
    const ctrl = control(root, `
      <label>temperature <b>α</b> <input type="range" min="-1.3" max="1" step="0.01" value="0"> <span></span></label>`);
    const presets = control(root, `Try: <button data-v="0.1">α = 0.1</button><button data-v="0.5">0.5</button><button data-v="1">1</button><button data-v="2">2</button><button data-v="10">10</button>`);
    const slider = ctrl.querySelector("input");
    const out = readout(root);
    const LN4 = Math.log(4);
    const x = linear(-0.5, 3.5, M.l, W - M.r - 70), y = linear(0, 1, H - M.b, M.t);
    const ey = linear(0, LN4, H - M.b, M.t);

    function draw() {
      const col = colors(), alpha = 10 ** +slider.value;
      slider.nextElementSibling.textContent = alpha < 1 ? alpha.toFixed(2) : alpha.toFixed(1);
      const m = Math.max(...Q.map((q) => q / alpha));
      const e = Q.map((q) => Math.exp(q / alpha - m)), s = e.reduce((a, b) => a + b, 0);
      const p = e.map((v) => v / s);
      const ent = -p.reduce((acc, v) => acc + (v > 0 ? v * Math.log(v) : 0), 0);
      chart.innerHTML = "";
      axes(chart, x, y, { xDomain: [-0.5, 3.5], yDomain: [0, 1], xTicks: [], yTicks: ticks(0, 1, 0.25), yLabel: "π(a | s)" });
      const bw = (x(1) - x(0)) * 0.6;
      p.forEach((v, i) => {
        el("rect", { x: x(i) - bw / 2, y: y(v), width: bw, height: y(0) - y(v), fill: col.model, rx: 3 }, chart);
        el("text", { x: x(i), y: y(v) - 5, "text-anchor": "middle", style: textStyle }, chart).textContent = fmt(v, 2);
        el("text", { x: x(i), y: H - M.b + 15, "text-anchor": "middle", style: textStyle }, chart).textContent = NAMES[i];
        el("text", { x: x(i), y: H - M.b + 30, "text-anchor": "middle" }, chart).textContent = `Q = ${Q[i]}`;
      });
      // entropy meter on the right
      const mx = W - M.r - 34;
      el("rect", { x: mx, y: ey(LN4), width: 22, height: ey(0) - ey(LN4), fill: "none", stroke: col.border }, chart);
      el("rect", { x: mx, y: ey(ent), width: 22, height: ey(0) - ey(ent), fill: col.loss, "fill-opacity": 0.8 }, chart);
      el("text", { x: mx + 11, y: M.t - 8, "text-anchor": "middle", style: textStyle }, chart).textContent = "entropy";
      el("text", { x: mx + 11, y: H - M.b + 15, "text-anchor": "middle", style: textStyle }, chart).textContent = fmt(ent, 2);
      el("text", { x: mx - 4, y: ey(LN4) + 4, "text-anchor": "end" }, chart).textContent = "max";
      const expQ = p.reduce((acc, v, i) => acc + v * Q[i], 0);
      const look = ent < 0.2 ? "almost greedy: nearly always the best action" : ent > 1.3 ? "almost uniform: close to random exploration" : "in between: prefers good actions but keeps trying others";
      out.textContent = `π(a|s) ∝ exp(Q(s,a) / α)   →   π = [${p.map((v) => fmt(v, 3)).join(", ")}]\n` +
        `entropy H = ${fmt(ent, 3)} nats  (max ln 4 = ${fmt(LN4, 3)})   expected Q = ${fmt(expQ, 2)}\n${look}`;
    }

    slider.addEventListener("input", draw);
    presets.addEventListener("click", (e) => {
      if (e.target.dataset.v === undefined) return;
      slider.value = Math.log10(+e.target.dataset.v);
      draw();
    });
    onThemeChange(draw);
    draw();
  }

  const mounts = { "viz-ac-trace": traceWidget, "viz-ppo-clip": clipWidget, "viz-entropy": entropyWidget };
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
