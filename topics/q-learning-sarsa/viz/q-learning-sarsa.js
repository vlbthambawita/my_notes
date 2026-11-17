// Interactive widgets for the "Q-Learning and SARSA" lesson.
// Three widgets, each mounted on a <div> with a matching id:
//   #viz-qvalues – the optimal action values Q*(s, a) of the cliff world, four numbers per cell
//   #viz-train   – watch SARSA or Q-learning learn the cliff world live, then run the greedy policy
//   #viz-curves  – reward per episode while learning, SARSA vs Q-learning, averaged over many seeded runs
// Runs after DOMContentLoaded so the site-wide helpers (assets/js/viz-common.js) are available.
document.addEventListener("DOMContentLoaded", function () {
  const { colors, onThemeChange, linear, el, svg, axes, ticks, fmt } = window.Viz;

  // ---------------------------------------------------------------- The grid world (same as lesson 1)
  // S = start, G = goal (+10), C = cliff (−20), # = wall, . = floor (−1 per step)
  const MAP = [
    "......",
    "..#...",
    "......",
    "SCCCCG",
  ];
  const ROWS = MAP.length, COLS = MAP[0].length;
  const START = [3, 0];
  const ACTIONS = [[-1, 0], [0, 1], [1, 0], [0, -1]];   // up, right, down, left
  const ARROWS = ["↑", "→", "↓", "←"];
  const GAMMA = 1;                                      // undiscounted, as in Sutton & Barto's cliff walk
  const MAX_STEPS = 200;                                // safety cap on one episode

  const tile = (r, c) => MAP[r][c];
  const isTerminal = (r, c) => tile(r, c) === "G" || tile(r, c) === "C";
  const isFloor = (r, c) => tile(r, c) === "." || tile(r, c) === "S";
  const reward = (r, c) => (tile(r, c) === "G" ? 10 : tile(r, c) === "C" ? -20 : -1);

  function move(r, c, a) {
    const nr = r + ACTIONS[a][0], nc = c + ACTIONS[a][1];
    if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS || tile(nr, nc) === "#") return [r, c];
    return [nr, nc];
  }

  // Small seeded random number generator, so the learning curves are reproducible
  function mulberry32(seed) {
    let t = seed >>> 0;
    return function () {
      t = (t + 0x6d2b79f5) >>> 0;
      let x = Math.imul(t ^ (t >>> 15), 1 | t);
      x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
      return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
  }

  // One step of the environment: with probability `slip` the agent slides sideways
  function envStep(r, c, a, slip, rand) {
    if (slip > 0 && rand() < slip) a = (a + (rand() < 0.5 ? 1 : 3)) % 4;
    const [nr, nc] = move(r, c, a);
    return [nr, nc, reward(nr, nc), isTerminal(nr, nc)];
  }

  // Q is a flat array: Q[(r * COLS + c) * 4 + a]
  const qi = (r, c, a) => (r * COLS + c) * 4 + a;
  const newQ = () => new Float64Array(ROWS * COLS * 4);
  const maxQ = (Q, r, c) => Math.max(Q[qi(r, c, 0)], Q[qi(r, c, 1)], Q[qi(r, c, 2)], Q[qi(r, c, 3)]);
  function greedy(Q, r, c) {                            // first best action (for drawing)
    let best = 0;
    for (let a = 1; a < 4; a++) if (Q[qi(r, c, a)] > Q[qi(r, c, best)]) best = a;
    return best;
  }
  function epsGreedy(Q, r, c, eps, rand) {
    if (rand() < eps) return Math.floor(rand() * 4);    // explore
    const m = maxQ(Q, r, c), ties = [];
    for (let a = 0; a < 4; a++) if (Q[qi(r, c, a)] === m) ties.push(a);
    return ties[Math.floor(rand() * ties.length)];      // exploit, ties broken at random
  }

  // Optimal action values by value iteration: Q*(s,a) = Σ p(s'|s,a) [r + γ max_a' Q*(s',a')]
  function qStar(slip) {
    const Q = newQ();
    for (let it = 0; it < 2000; it++) {
      let delta = 0;
      for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
        if (!isFloor(r, c)) continue;
        for (let a = 0; a < 4; a++) {
          let v = 0;
          for (const [p, d] of [[1 - slip, a], [slip / 2, (a + 1) % 4], [slip / 2, (a + 3) % 4]]) {
            if (p <= 0) continue;
            const [nr, nc] = move(r, c, d);
            v += p * (reward(nr, nc) + (isTerminal(nr, nc) ? 0 : GAMMA * maxQ(Q, nr, nc)));
          }
          delta = Math.max(delta, Math.abs(v - Q[qi(r, c, a)]));
          Q[qi(r, c, a)] = v;
        }
      }
      if (delta < 1e-9) break;
    }
    return Q;
  }

  // A SARSA or Q-learning agent that learns one step at a time
  class Learner {
    constructor(algo, rand) {
      this.algo = algo; this.rand = rand;
      this.alpha = 0.2; this.eps = 0.3; this.slip = 0;
      this.Q = newQ(); this.history = []; this.episodes = 0;
      this.startEpisode();
    }
    startEpisode() {
      [this.r, this.c] = START;
      this.a = epsGreedy(this.Q, this.r, this.c, this.eps, this.rand);
      this.total = 0; this.t = 0; this.trail = [START.slice()];
    }
    // One time step: act, observe, update. Returns true when the episode just ended.
    step() {
      const Q = this.Q, { r, c, a } = this;
      const [nr, nc, rew, done] = envStep(r, c, a, this.slip, this.rand);
      this.total += rew; this.t++;
      const a2 = epsGreedy(Q, nr, nc, this.eps, this.rand);          // the action we will really take next
      let target;
      if (done) target = rew;                                        // no future after goal or cliff
      else if (this.algo === "sarsa") target = rew + GAMMA * Q[qi(nr, nc, a2)];   // on-policy
      else target = rew + GAMMA * maxQ(Q, nr, nc);                   // off-policy
      Q[qi(r, c, a)] += this.alpha * (target - Q[qi(r, c, a)]);
      this.trail.push([nr, nc]);
      if (done || this.t >= MAX_STEPS) {
        this.history.push({ total: this.total, fell: tile(nr, nc) === "C" });
        this.episodes++;
        this.startEpisode();
        return true;
      }
      this.r = nr; this.c = nc; this.a = a2;
      return false;
    }
    episode() { while (!this.step()); }
  }

  // Follow the greedy policy from the start without slipping: the route the agent has "decided on"
  function greedyRoute(Q) {
    let [r, c] = START;
    const path = [[r, c]], seen = new Set([r * COLS + c]);
    let total = 0;
    while (path.length < 40) {
      const allZero = [0, 1, 2, 3].every((a) => Q[qi(r, c, a)] === 0);
      if (allZero) return { path, total, kind: "unknown" };
      [r, c] = move(r, c, greedy(Q, r, c));
      total += reward(r, c);
      path.push([r, c]);
      if (isTerminal(r, c)) return { path, total, kind: tile(r, c) === "G" ? "goal" : "cliff" };
      if (seen.has(r * COLS + c)) return { path, total, kind: "loop" };
      seen.add(r * COLS + c);
    }
    return { path, total, kind: "loop" };
  }

  function describeRoute(g) {
    const n = g.path.length - 1;
    if (g.kind === "goal") return `${n} steps, return ${g.total > 0 ? "+" : ""}${g.total}` + (n === 7 ? " (along the cliff edge)" : " (a detour away from the edge)");
    if (g.kind === "cliff") return "straight into the cliff";
    return "none yet (it goes round in circles or stops at an unexplored cell)";
  }

  // ---------------------------------------------------------------- Drawing
  const CELL = 56, PAD = 6;
  const GW = COLS * CELL + 2 * PAD, GH = ROWS * CELL + 2 * PAD;
  const px = (c) => PAD + c * CELL, py = (r) => PAD + r * CELL;
  const textStyle = "fill: var(--viz-text)";

  function drawBoard(root) {
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
      }
    }
  }

  // Each floor cell is split into four triangles, one per action, shaded by Q(s, a):
  // green = good, red = bad. opts.numbers writes the values, opts.arrows draws the greedy action.
  function drawQ(root, Q, opts = {}) {
    const col = colors(), scale = opts.scale || 12;
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        if (!isFloor(r, c)) continue;
        const x = px(c), y = py(r), cx = x + CELL / 2, cy = y + CELL / 2;
        const corners = [[x, y], [x + CELL, y], [x + CELL, y + CELL], [x, y + CELL]];
        const visited = [0, 1, 2, 3].some((a) => Q[qi(r, c, a)] !== 0);
        for (let a = 0; a < 4; a++) {
          const q = Q[qi(r, c, a)];
          const [p1, p2] = [corners[a], corners[(a + 1) % 4]];
          el("path", {
            d: `M${p1[0]},${p1[1]}L${p2[0]},${p2[1]}L${cx},${cy}Z`,
            fill: q >= 0 ? col.opt : col.grad, "fill-opacity": Math.min(1, Math.abs(q) / scale) * 0.6,
            stroke: col.border, "stroke-width": 0.6,
          }, root);
        }
        if (opts.numbers) {
          const pos = [[cx, y + 12], [x + CELL - 10, cy + 3], [cx, y + CELL - 4], [x + 10, cy + 3]];
          for (let a = 0; a < 4; a++) {
            const q = Q[qi(r, c, a)];
            el("text", { x: pos[a][0], y: pos[a][1], "text-anchor": "middle", style: textStyle + "; font-size: 9px" }, root)
              .textContent = Math.abs(q - Math.round(q)) < 0.05 ? Math.round(q) : q.toFixed(1);
          }
        }
        if (opts.arrows && visited) {
          el("text", { x: cx, y: cy + 5, "text-anchor": "middle", style: textStyle + `; font-size: ${opts.numbers ? 13 : 17}px; font-weight: 600` }, root)
            .textContent = ARROWS[greedy(Q, r, c)];
        }
        if (tile(r, c) === "S" && !opts.numbers) el("text", { x: x + 3, y: y + 10, style: "font-size: 9px" }, root).textContent = "start";
      }
    }
  }

  function drawPath(root, path, color, dashed, width = 3) {
    if (path.length < 2) return;
    const d = path.map(([r, c], i) => (i ? "L" : "M") + (px(c) + CELL / 2) + "," + (py(r) + CELL / 2)).join("");
    el("path", { d, fill: "none", stroke: color, "stroke-width": width, "stroke-opacity": dashed ? 0.95 : 0.45, "stroke-linejoin": "round", "stroke-dasharray": dashed ? "7 5" : "none" }, root);
  }

  function drawAgent(root, r, c) {
    const col = colors();
    el("circle", { cx: px(c) + CELL / 2, cy: py(r) + CELL / 2, r: 11, fill: col.model, stroke: col.bg, "stroke-width": 2 }, root);
  }

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

  function boardSvg(root) {
    const s = svg(GW, GH);
    s.style.maxWidth = "520px";
    s.style.margin = "0 auto";
    root.appendChild(s);
    return s;
  }

  function cellAt(board, e) {
    const b = board.getBoundingClientRect();
    const c = Math.floor((((e.clientX - b.left) / b.width) * GW - PAD) / CELL);
    const r = Math.floor((((e.clientY - b.top) / b.height) * GH - PAD) / CELL);
    return [r, c];
  }

  // ---------------------------------------------------------------- Widget 1: four numbers per cell
  function qValuesWidget(root) {
    const board = boardSvg(root);
    board.style.cursor = "pointer";
    hint(root, "Each cell holds four numbers, one per action: the total reward you can expect if you take that action and then play perfectly. Click a cell.");
    const ctrl = control(root, `<label>slip chance <input type="range" min="0" max="0.3" step="0.05" value="0"> <span class="val"></span></label>`);
    const slipIn = ctrl.querySelector("input"), val = ctrl.querySelector(".val");
    const out = readout(root);
    let Q, sel = [2, 1];

    function draw() {
      const col = colors();
      val.textContent = (+slipIn.value).toFixed(2);
      board.innerHTML = "";
      drawBoard(board);
      drawQ(board, Q, { numbers: true, arrows: true });
      const [r, c] = sel;
      el("rect", { x: px(c) + 1.5, y: py(r) + 1.5, width: CELL - 3, height: CELL - 3, fill: "none", stroke: col.model, "stroke-width": 3 }, board);
      const names = ["up", "right", "down", "left"];
      const best = greedy(Q, r, c);
      const parts = [0, 1, 2, 3].map((a) => `Q*(s, ${ARROWS[a]}) = ${fmt(Q[qi(r, c, a)], 2)}`);
      const lands = [0, 1, 2, 3].map((a) => {
        const [nr, nc] = move(r, c, a);
        return tile(nr, nc) === "C" ? "the cliff" : tile(nr, nc) === "G" ? "the goal" : nr === r && nc === c ? "nowhere (bump)" : `(${nr}, ${nc})`;
      });
      out.textContent =
        `cell s = (row ${r}, col ${c}): ${parts.join(" · ")}\n` +
        `best action = ${ARROWS[best]} ${names[best]}, simply the biggest of the four numbers. No need to know where each move leads.\n` +
        `With only V you would also need the model: ↑ leads to ${lands[0]}, → to ${lands[1]}, ↓ to ${lands[2]}, ← to ${lands[3]}.`;
    }

    function refresh() { Q = qStar(+slipIn.value); draw(); }
    board.addEventListener("click", (e) => {
      const [r, c] = cellAt(board, e);
      if (r < 0 || r >= ROWS || c < 0 || c >= COLS || !isFloor(r, c)) return;
      sel = [r, c];
      draw();
    });
    slipIn.addEventListener("input", refresh);
    onThemeChange(draw);
    refresh();
  }

  // ---------------------------------------------------------------- Widget 2: live training
  function trainWidget(root) {
    const board = boardSvg(root);
    hint(root, "Triangles = the agent's current Q(s, a) for each action (green good, red bad). Bold arrows = greedy action. Green dashed line = the route the greedy policy would take now. Orange = the agent exploring with ε-greedy.");
    const ctrl1 = control(root, `
      <label>algorithm <select class="algo"><option value="sarsa">SARSA</option><option value="q">Q-learning</option></select></label>
      <label>speed <select class="speed"><option value="slow">slow</option><option value="fast" selected>fast</option><option value="turbo">turbo</option></select></label>`);
    const ctrl2 = control(root, `
      <label>α <input type="range" class="alpha" min="0.05" max="1" step="0.05" value="0.2"> <span></span></label>
      <label>ε <input type="range" class="eps" min="0" max="0.5" step="0.05" value="0.3"> <span></span></label>
      <label>slip <input type="range" class="slip" min="0" max="0.3" step="0.05" value="0"> <span></span></label>`);
    const ctrl3 = control(root, `
      <button data-a="play">▶ Train</button><button data-a="100">+100 episodes</button>
      <button data-a="frozen">Run greedy policy (ε = 0)</button><button data-a="reset">Reset</button>`);
    const algoIn = ctrl1.querySelector(".algo"), speedIn = ctrl1.querySelector(".speed");
    const alphaIn = ctrl2.querySelector(".alpha"), epsIn = ctrl2.querySelector(".eps"), slipIn = ctrl2.querySelector(".slip");
    const playBtn = ctrl3.querySelector('[data-a="play"]');
    const out = readout(root);
    const rand = mulberry32(Date.now() % 100000);

    let L, timer = null, frozen = null, note = "";

    function sync() {
      L.alpha = +alphaIn.value; L.eps = +epsIn.value; L.slip = +slipIn.value;
      for (const inp of [alphaIn, epsIn, slipIn]) inp.nextElementSibling.textContent = (+inp.value).toFixed(2);
    }

    function reset() {
      stop(); frozen = null; note = "";
      L = new Learner(algoIn.value, rand);
      sync(); draw();
    }

    function stop() {
      clearInterval(timer); timer = null;
      playBtn.textContent = "▶ Train";
    }

    function tick() {
      const sp = speedIn.value;
      if (sp === "slow") L.step();
      else if (sp === "fast") { for (let k = 0; k < 4; k++) L.step(); }
      else for (let k = 0; k < 20; k++) L.episode();
      draw();
    }

    function play() {
      if (timer) { stop(); return; }
      frozen = null; note = "";
      playBtn.textContent = "⏸ Pause";
      timer = setInterval(tick, speedIn.value === "slow" ? 150 : 40);
    }

    function runFrozen() {
      stop();
      // play one episode with the greedy policy, no exploration (the floor may still slip)
      let [r, c] = START, total = 0, done = false, t = 0;
      const path = [[r, c]];
      while (!done && t < 50) {
        let rew;
        [r, c, rew, done] = envStep(r, c, greedy(L.Q, r, c), L.slip, rand);
        total += rew; t++;
        path.push([r, c]);
      }
      let i = 0;
      frozen = { path: [path[0]], pos: path[0] };
      note = "";
      timer = setInterval(() => {
        i++;
        frozen = { path: path.slice(0, i + 1), pos: path[Math.min(i, path.length - 1)] };
        if (i >= path.length - 1) {
          clearInterval(timer); timer = null;
          note = `\ngreedy run (ε = 0): ${done ? (tile(r, c) === "G" ? "reached the goal" : "fell off the cliff") : "gave up after 50 steps"} in ${t} steps, total reward ${total > 0 ? "+" : ""}${total}`;
        }
        draw();
      }, 160);
      draw();
    }

    function draw() {
      const col = colors();
      board.innerHTML = "";
      drawBoard(board);
      drawQ(board, L.Q, { arrows: true, scale: 14 });
      const g = greedyRoute(L.Q);
      if (g.kind === "goal") drawPath(board, g.path, col.opt, true);
      if (frozen) {
        drawPath(board, frozen.path, col.model, false, 4);
        drawAgent(board, frozen.pos[0], frozen.pos[1]);
      } else {
        if (speedIn.value !== "turbo" || !timer) drawPath(board, L.trail, col.model, false);
        drawAgent(board, L.r, L.c);
      }
      const h = L.history, last = h.slice(-50);
      const avg = last.length ? last.reduce((s, e) => s + e.total, 0) / last.length : NaN;
      const fell = last.filter((e) => e.fell).length;
      out.textContent =
        `${L.algo === "sarsa" ? "SARSA" : "Q-learning"} · episodes finished: ${L.episodes} · current episode: step ${L.t}, reward so far ${L.total}\n` +
        (last.length ? `last ${last.length} episodes: average reward ${fmt(avg, 2)}, fell off the cliff ${fell} times\n` : "no finished episodes yet\n") +
        `greedy route now: ${describeRoute(g)}` + note;
    }

    algoIn.addEventListener("change", reset);
    for (const inp of [alphaIn, epsIn, slipIn]) inp.addEventListener("input", () => { sync(); if (!timer) draw(); });
    speedIn.addEventListener("change", () => { if (timer && !frozen) { stop(); play(); } });
    ctrl3.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (a === "play") play();
      if (a === "100") { stop(); frozen = null; note = ""; for (let k = 0; k < 100; k++) L.episode(); draw(); }
      if (a === "frozen") runFrozen();
      if (a === "reset") reset();
    });
    onThemeChange(draw);
    reset();
  }

  // ---------------------------------------------------------------- Widget 3: learning curves
  function curvesWidget(root) {
    const W = 400, H = 260, M = { l: 40, r: 86, t: 14, b: 40 };
    const EPISODES = 500, RUNS = 200, SMOOTH = 10;
    const YMIN = -45, YMAX = 6;
    const chart = svg(W, H);
    chart.style.maxWidth = "640px";
    chart.style.margin = "0 auto";
    root.appendChild(chart);
    const ctrl = control(root, `
      <label>ε <input type="range" min="0.05" max="0.4" step="0.05" value="0.3"> <span class="val"></span></label>
      <label>α <select><option>0.1</option><option selected>0.2</option><option>0.5</option></select></label>`);
    const presets = control(root, `Try: <button data-e="0.3">ε = 0.3 (the classic result)</button><button data-e="0.1">ε = 0.1</button>`);
    const epsIn = ctrl.querySelector("input"), alphaIn = ctrl.querySelector("select"), val = ctrl.querySelector(".val");
    const out = readout(root);
    const x = linear(0, EPISODES, M.l, W - M.r), y = linear(YMIN, YMAX, H - M.b, M.t);
    const ALGOS = [{ key: "sarsa", name: "SARSA", color: "data" }, { key: "q", name: "Q-learning", color: "model" }];
    let res = null, hover = null;

    function compute() {
      const eps = +epsIn.value, alpha = +alphaIn.value;
      res = ALGOS.map((A, ai) => {
        const sum = new Float64Array(EPISODES);
        let last = 0, falls = 0, edge = 0, detour = 0;
        for (let run = 0; run < RUNS; run++) {
          const L = new Learner(A.key, mulberry32(1000 * ai + run + 1));
          L.alpha = alpha; L.eps = eps; L.startEpisode();
          for (let e = 0; e < EPISODES; e++) {
            L.episode();
            const ep = L.history[e];
            sum[e] += ep.total;
            if (e >= EPISODES - 100) { last += ep.total; falls += ep.fell ? 1 : 0; }
          }
          const g = greedyRoute(L.Q);
          if (g.kind === "goal" && g.path.length - 1 === 7) edge++;
          else if (g.kind === "goal") detour++;
        }
        const mean = Array.from(sum, (s) => s / RUNS);
        const smooth = mean.map((_, i) => {
          const lo = Math.max(0, i - SMOOTH + 1);
          let s = 0;
          for (let j = lo; j <= i; j++) s += mean[j];
          return s / (i - lo + 1);
        });
        return { ...A, smooth, last: last / (100 * RUNS), falls: falls / (100 * RUNS), edge, detour };
      });
    }

    function draw() {
      const col = colors();
      val.textContent = (+epsIn.value).toFixed(2);
      chart.innerHTML = "";
      axes(chart, x, y, { xDomain: [0, EPISODES], yDomain: [YMIN, YMAX], xTicks: ticks(0, EPISODES, 100), yTicks: ticks(-40, 0, 10), xLabel: "episode" });
      el("text", { x: M.l + 4, y: M.t + 12, style: "font-size: 11px" }, chart).textContent = "reward per episode (avg of 200 runs)";
      if (!res) return;
      const ends = [];
      for (const A of res) {
        const d = A.smooth.map((v, i) => (i ? "L" : "M") + x(i + 1).toFixed(1) + "," + y(Math.max(YMIN, v)).toFixed(1)).join("");
        el("path", { d, fill: "none", stroke: col[A.color], "stroke-width": 2, "stroke-linejoin": "round" }, chart);
        ends.push({ A, yy: y(A.smooth[EPISODES - 1]) });
      }
      // direct labels at the right end, nudged apart if they would overlap
      ends.sort((a, b) => a.yy - b.yy);
      if (ends[1].yy - ends[0].yy < 26) { const mid = (ends[0].yy + ends[1].yy) / 2; ends[0].yy = mid - 13; ends[1].yy = mid + 13; }
      for (const { A, yy } of ends) {
        el("circle", { cx: W - M.r + 6, cy: yy, r: 3, fill: col[A.color] }, chart);
        el("text", { x: W - M.r + 12, y: yy - 1, style: textStyle + "; font-size: 11px" }, chart).textContent = A.name;
        el("text", { x: W - M.r + 12, y: yy + 11, style: "font-size: 10px" }, chart).textContent = `last 100: ${fmt(A.last, 1)}`;
      }
      if (hover !== null) {
        el("line", { x1: x(hover), x2: x(hover), y1: y(YMIN), y2: y(YMAX), stroke: col.muted, "stroke-dasharray": "3 3" }, chart);
        for (const A of res) el("circle", { cx: x(hover), cy: y(Math.max(YMIN, A.smooth[hover - 1])), r: 4, fill: col[A.color], stroke: col.bg, "stroke-width": 2 }, chart);
        const txt = `episode ${hover}: ` + res.map((A) => `${A.name} ${fmt(A.smooth[hover - 1], 1)}`).join(" · ");
        const tx = Math.min(Math.max(x(hover), M.l + 120), W - M.r - 120);
        el("rect", { x: tx - 118, y: H - M.b - 26, width: 236, height: 18, rx: 4, fill: col.bg, stroke: col.border }, chart);
        el("text", { x: tx, y: H - M.b - 13, "text-anchor": "middle", style: textStyle + "; font-size: 10px" }, chart).textContent = txt;
      }
      const pct = (v) => `${Math.round(100 * v)}%`;
      const [S, Qd] = res;
      out.textContent =
        `average reward per episode over the last 100 episodes:  SARSA ${fmt(S.last, 2)} · Q-learning ${fmt(Qd.last, 2)}\n` +
        `episodes that end in the cliff (last 100):  SARSA ${pct(S.falls)} · Q-learning ${pct(Qd.falls)}\n` +
        `greedy route after training:  SARSA edge ${S.edge}/${RUNS}, detour ${S.detour}/${RUNS}   ·   Q-learning edge ${Qd.edge}/${RUNS}, detour ${Qd.detour}/${RUNS}`;
    }

    function recompute() {
      res = null; hover = null;
      out.textContent = "training 2 × 200 agents…";
      draw();
      setTimeout(() => { compute(); draw(); }, 20);
    }

    chart.addEventListener("mousemove", (e) => {
      if (!res) return;
      const b = chart.getBoundingClientRect();
      const ep = Math.round(x.invert(((e.clientX - b.left) / b.width) * W));
      hover = ep >= 1 && ep <= EPISODES ? ep : null;
      draw();
    });
    chart.addEventListener("mouseleave", () => { hover = null; draw(); });
    epsIn.addEventListener("input", () => { val.textContent = (+epsIn.value).toFixed(2); });
    epsIn.addEventListener("change", recompute);
    alphaIn.addEventListener("change", recompute);
    presets.addEventListener("click", (e) => {
      if (e.target.dataset.e === undefined) return;
      epsIn.value = e.target.dataset.e;
      recompute();
    });
    onThemeChange(draw);
    recompute();
  }

  const mounts = { "viz-qvalues": qValuesWidget, "viz-train": trainWidget, "viz-curves": curvesWidget };
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
