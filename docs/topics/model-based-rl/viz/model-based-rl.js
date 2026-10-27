// Interactive widgets for the "Model-based RL and planning" lesson.
// Three widgets, each mounted on a <div> with a matching id:
//   #viz-dyna        – watch one Dyna-Q agent on the cliff world: its Q-values, its learned model, its real path
//   #viz-dyna-curves – how fast Dyna-Q finds the best route with n = 0, 5, 50 (or your own n) planning steps
//   #viz-mcts        – Monte Carlo Tree Search on a tic-tac-toe position, one phase or many simulations at a time
// Runs after DOMContentLoaded so the site-wide helpers (assets/js/viz-common.js) are available.
document.addEventListener("DOMContentLoaded", function () {
  const { colors, onThemeChange, linear, el, svg, axes, ticks, fmt } = window.Viz;
  const textStyle = "fill: var(--viz-text)";

  // Small seeded random number generator (mulberry32), so every run can be repeated exactly
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const randInt = (rand, n) => Math.floor(rand() * n);

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

  // Mark one button in a group as selected
  function markActive(group, attr, value) {
    for (const b of group.querySelectorAll(`button[${attr}]`)) {
      const on = b.getAttribute(attr) === String(value);
      b.style.borderColor = on ? "var(--viz-model)" : "";
      b.style.fontWeight = on ? "700" : "";
    }
  }

  // ================================================================= The cliff world (from lesson 1)
  const MAP = ["......", "..#...", "......", "SCCCCG"];
  const ROWS = 4, COLS = 6, NS = ROWS * COLS;
  const START = 3 * COLS + 0;
  const MOVES = [[-1, 0], [0, 1], [1, 0], [0, -1]];      // up, right, down, left
  const ARROWS = ["↑", "→", "↓", "←"];
  const tile = (s) => MAP[Math.floor(s / COLS)][s % COLS];
  const isFloor = (s) => tile(s) === "." || tile(s) === "S";
  const FLOOR_PAIRS = [...Array(NS).keys()].filter(isFloor).length * 4;

  // One real step: returns [next state, reward, done]
  function envStep(s, a) {
    const r0 = Math.floor(s / COLS), c0 = s % COLS;
    let r = r0 + MOVES[a][0], c = c0 + MOVES[a][1];
    if (r < 0 || r >= ROWS || c < 0 || c >= COLS || MAP[r][c] === "#") { r = r0; c = c0; }
    const s2 = r * COLS + c, t = MAP[r][c];
    if (t === "G") return [s2, 10, true];
    if (t === "C") return [s2, -20, true];
    return [s2, -1, false];
  }

  // ---------------------------------------------------------------- Dyna-Q (same algorithm as the Python in the lesson)
  const ALPHA = 0.1, GAMMA = 0.95, EPS = 0.1;

  function makeAgent(n, seed) {
    return { n, rand: rng(seed), Q: new Float64Array(NS * 4), model: new Map(), seen: [], episodes: 0, realSteps: 0, planSteps: 0 };
  }

  const maxQ = (ag, s) => Math.max(ag.Q[4 * s], ag.Q[4 * s + 1], ag.Q[4 * s + 2], ag.Q[4 * s + 3]);

  function greedy(ag, s) {
    const m = maxQ(ag, s), best = [];
    for (let a = 0; a < 4; a++) if (ag.Q[4 * s + a] === m) best.push(a);
    return best[randInt(ag.rand, best.length)];
  }

  function qUpdate(ag, s, a, r, s2, done) {
    const target = done ? r : r + GAMMA * maxQ(ag, s2);
    ag.Q[4 * s + a] += ALPHA * (target - ag.Q[4 * s + a]);
  }

  // Play one episode: act for real, learn the model, plan n times per real step
  function runEpisode(ag) {
    let s = START, done = false, t = 0, G = 0, end = "timeout";
    const path = [s];
    while (!done && t < 1000) {
      const a = ag.rand() < EPS ? randInt(ag.rand, 4) : greedy(ag, s);
      const [s2, r, d] = envStep(s, a);
      done = d; G += r;
      qUpdate(ag, s, a, r, s2, done);
      const key = 4 * s + a;
      if (!ag.model.has(key)) { ag.model.set(key, new Map()); ag.seen.push(key); }
      const counts = ag.model.get(key), ok = `${s2},${r},${done ? 1 : 0}`;
      counts.set(ok, (counts.get(ok) || 0) + 1);
      for (let i = 0; i < ag.n; i++) {
        const pk = ag.seen[randInt(ag.rand, ag.seen.length)];
        const outs = ag.model.get(pk);
        let total = 0;
        for (const v of outs.values()) total += v;
        let u = ag.rand() * total, pick = null;
        for (const [k, v] of outs) { pick = k; if ((u -= v) < 0) break; }
        const [ps2, pr, pd] = pick.split(",").map(Number);
        qUpdate(ag, pk >> 2, pk & 3, pr, ps2, pd === 1);
      }
      ag.planSteps += ag.n;
      s = s2; t++; path.push(s);
      if (done) end = tile(s) === "G" ? "goal" : "cliff";
    }
    ag.episodes++; ag.realSteps += t;
    return { path, steps: t, G, end };
  }

  // Follow the greedy policy (first best action, no exploration); number of steps to the goal or null
  function greedyRoute(ag) {
    let s = START;
    const path = [s];
    for (let t = 1; t < 30; t++) {
      const m = maxQ(ag, s);
      let a = 0;
      while (ag.Q[4 * s + a] !== m) a++;
      const [s2, r, done] = envStep(s, a);
      s = s2; path.push(s);
      if (done) return { len: r > 0 ? t : null, path };
    }
    return { len: null, path };
  }

  // ---------------------------------------------------------------- Board drawing
  const CELL = 56, PAD = 6;
  const GW = COLS * CELL + 2 * PAD, GH = ROWS * CELL + 2 * PAD;
  const px = (s) => PAD + (s % COLS) * CELL, py = (s) => PAD + Math.floor(s / COLS) * CELL;

  function drawBoard(root, tint) {
    const col = colors();
    for (let s = 0; s < NS; s++) {
      const t = tile(s), x = px(s), y = py(s);
      el("rect", { x, y, width: CELL, height: CELL, fill: col.bg, stroke: col.border }, root);
      if (t === "#") el("rect", { x: x + 1, y: y + 1, width: CELL - 2, height: CELL - 2, fill: col.muted, "fill-opacity": 0.55 }, root);
      if (t === "G" || t === "C") {
        el("rect", { x: x + 1, y: y + 1, width: CELL - 2, height: CELL - 2, fill: t === "G" ? col.opt : col.grad, "fill-opacity": t === "G" ? 0.45 : 0.6 }, root);
        el("text", { x: x + CELL / 2, y: y + CELL / 2 - 3, "text-anchor": "middle", style: textStyle }, root).textContent = t === "G" ? "GOAL" : "cliff";
        el("text", { x: x + CELL / 2, y: y + CELL / 2 + 12, "text-anchor": "middle" }, root).textContent = t === "G" ? "+10" : "−20";
      }
      if (isFloor(s) && tint) {
        const [fill, op] = tint(s);
        el("rect", { x: x + 1, y: y + 1, width: CELL - 2, height: CELL - 2, fill, "fill-opacity": op }, root);
      }
      if (t === "S") el("text", { x: x + 4, y: y + 12 }, root).textContent = "start";
    }
  }

  function drawPath(root, path, colour, width, opacity, dash) {
    if (path.length < 2) return;
    const d = path.map((s, i) => (i ? "L" : "M") + (px(s) + CELL / 2) + "," + (py(s) + CELL / 2)).join("");
    const attrs = { d, fill: "none", stroke: colour, "stroke-width": width, "stroke-opacity": opacity, "stroke-linejoin": "round", "stroke-linecap": "round" };
    if (dash) attrs["stroke-dasharray"] = dash;
    el("path", attrs, root);
  }

  // ================================================================= Widget 1: one Dyna-Q agent, step by step
  function dynaWidget(root) {
    const board = svg(GW, GH);
    board.style.maxWidth = "520px";
    board.style.margin = "0 auto";
    root.appendChild(board);
    hint(root, "Orange ticks = what the model knows: one tick on a cell's side for each (state, action) the agent has tried. Arrows = greedy action from Q, number = max Q. Blue line = the last real episode; dashed green = the current greedy route.");
    const nCtrl = control(root, `planning steps per real step: <button data-n="0">n = 0</button><button data-n="5">n = 5</button><button data-n="50">n = 50</button>`);
    const ctrl = control(root, `<button data-a="one">Run 1 episode</button><button data-a="ten">Run 10 episodes</button><button data-a="new">New run</button>`);
    const out = readout(root);

    let n = 5, seed = 1, agent, last = null;

    function reset() {
      agent = makeAgent(n, seed);
      last = null;
      markActive(nCtrl, "data-n", n);
      draw();
    }

    function draw() {
      const col = colors();
      board.innerHTML = "";
      const known = (s) => [0, 1, 2, 3].some((a) => agent.model.has(4 * s + a));
      drawBoard(board, (s) => {
        if (!known(s)) return [col.bg, 0];
        const v = maxQ(agent, s);
        return [v >= 0 ? col.opt : col.grad, Math.min(1, Math.abs(v) / 12) * 0.4];
      });
      // paths underneath everything else: last real episode (blue), current greedy route (dashed green)
      const route = greedyRoute(agent);
      if (last) drawPath(board, last.path, col.data, 3, 0.35);
      if (route.len) drawPath(board, route.path, col.opt, 2.5, 0.75, "6 5");
      // the model table, drawn as ticks on the side of each cell
      for (const key of agent.seen) {
        const s = key >> 2, a = key & 3, x = px(s), y = py(s);
        const L = 18, T = 4, m = 3;
        const r = [
          { x: x + (CELL - L) / 2, y: y + m, width: L, height: T },
          { x: x + CELL - m - T, y: y + (CELL - L) / 2, width: T, height: L },
          { x: x + (CELL - L) / 2, y: y + CELL - m - T, width: L, height: T },
          { x: x + m, y: y + (CELL - L) / 2, width: T, height: L },
        ][a];
        el("rect", Object.assign({ fill: col.model, rx: 1.5 }, r), board);
      }
      for (let s = 0; s < NS; s++) {
        if (!isFloor(s) || !known(s)) continue;
        const cx = px(s) + CELL / 2, cy = py(s) + CELL / 2;
        const m = maxQ(agent, s);
        let a = 0;
        while (agent.Q[4 * s + a] !== m) a++;
        el("text", { x: cx, y: cy + 4, "text-anchor": "middle", style: textStyle + "; font-size: 18px" }, board).textContent = ARROWS[a];
        el("text", { x: cx, y: py(s) + CELL - 9, "text-anchor": "middle", style: textStyle + "; font-size: 10px" }, board).textContent = fmt(m, 1).replace(/^-0\.0$/, "0.0");
      }

      let text = `n = ${n} · episodes ${agent.episodes} · real steps ${agent.realSteps} · imagined (planning) updates ${agent.planSteps}`;
      text += `\nmodel knows ${agent.model.size} of ${FLOOR_PAIRS} state-action pairs`;
      if (last) text += `\nlast episode: ${last.steps} steps, return ${last.G}, ${last.end === "goal" ? "reached the goal" : last.end === "cliff" ? "fell off the cliff" : "gave up after 1000 steps"}`;
      text += `\ngreedy route now: ${route.len === null ? "does not reach the goal yet" : route.len + " steps" + (route.len === 7 ? " (optimal)" : "")}`;
      out.textContent = text;
    }

    nCtrl.addEventListener("click", (e) => {
      if (e.target.dataset.n === undefined) return;
      n = +e.target.dataset.n;
      reset();
    });
    ctrl.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (a === "one") { last = runEpisode(agent); draw(); }
      if (a === "ten") { for (let i = 0; i < 10; i++) last = runEpisode(agent); draw(); }
      if (a === "new") { seed += 1; reset(); }
    });
    onThemeChange(draw);
    reset();
  }

  // ================================================================= Widget 2: learning curves for several n
  function curvesWidget(root) {
    const W = 560, H = 300, M = { l: 46, r: 14, t: 16, b: 40 }, E = 40, RUNS = 50;
    const chart = svg(W, H);
    root.appendChild(chart);
    const ctrl = control(root, `
      <label>your own <b>n</b> <input type="range" min="0" max="100" step="1" value="20"> <span class="val"></span></label>
      <button data-a="seeds">New random seeds</button>`);
    const slider = ctrl.querySelector("input"), val = ctrl.querySelector(".val");
    const out = readout(root);
    hint(root, `Each curve averages ${RUNS} independent agents (α = ${ALPHA}, γ = ${GAMMA}, ε = ${EPS}). A run counts as "solved" after an episode if its greedy route is the 7-step optimal one.`);

    let seedBase = 1000;
    const cache = {};

    function curve(n) {
      const key = n + ":" + seedBase;
      if (cache[key]) return cache[key];
      const solved = new Array(E).fill(0), first = [];
      for (let k = 0; k < RUNS; k++) {
        const ag = makeAgent(n, seedBase + k);
        let f = null;
        for (let e = 0; e < E; e++) {
          runEpisode(ag);
          if (greedyRoute(ag).len === 7) { solved[e]++; if (f === null) f = e + 1; }
        }
        first.push(f === null ? Infinity : f);
      }
      first.sort((a, b) => a - b);
      const med = (first[RUNS / 2 - 1] + first[RUNS / 2]) / 2;
      return (cache[key] = { pct: solved.map((v) => (100 * v) / RUNS), med });
    }

    const x = linear(1, E, M.l, W - M.r), y = linear(0, 100, H - M.b, M.t);

    function draw() {
      const col = colors();
      const mine = +slider.value;
      val.textContent = mine;
      chart.innerHTML = "";
      axes(chart, x, y, { xDomain: [1, E], yDomain: [0, 100], xTicks: [1, ...ticks(5, E, 5)], yTicks: ticks(0, 100, 25), xLabel: "episode", yLabel: "% of runs on the optimal route" });
      const series = [
        { n: 0, colour: col.muted, width: 2 },
        { n: 5, colour: col.data, width: 2 },
        { n: 50, colour: col.model, width: 2 },
      ];
      if (![0, 5, 50].includes(mine)) series.push({ n: mine, colour: col.loss, width: 3, dash: "6 4", mine: true });
      const lines = [];
      series.forEach((sr, i) => {
        const c = curve(sr.n);
        const d = c.pct.map((v, e) => (e ? "L" : "M") + x(e + 1) + "," + y(v)).join("");
        const attrs = { d, fill: "none", stroke: sr.colour, "stroke-width": sr.width, "stroke-linejoin": "round" };
        if (sr.dash) attrs["stroke-dasharray"] = sr.dash;
        el("path", attrs, chart);
        // legend, right-hand side between the curves (they sit near 0% and near 100% there)
        const ly = y(70) + i * 17, lx = W - M.r - 128;
        el("line", { x1: lx, x2: lx + 22, y1: ly - 4, y2: ly - 4, stroke: sr.colour, "stroke-width": 3, "stroke-dasharray": sr.dash || "" }, chart);
        el("text", { x: lx + 28, y: ly, style: textStyle }, chart).textContent = `n = ${sr.n}${sr.mine ? " (yours)" : ""}`;
        lines.push(`n = ${String(sr.n).padStart(3)}: median first solved at episode ${Number.isFinite(c.med) ? fmt(c.med, 1) : "> " + E}` +
          `   ·   solved after 10 episodes: ${Math.round(c.pct[9])}%`);
      });
      out.textContent = lines.join("\n");
    }

    slider.addEventListener("change", draw);
    slider.addEventListener("input", () => { val.textContent = slider.value; });
    ctrl.addEventListener("click", (e) => {
      if (e.target.dataset.a === "seeds") { seedBase += RUNS; draw(); }
    });
    onThemeChange(draw);
    draw();
  }

  // ================================================================= Widget 3: Monte Carlo Tree Search on tic-tac-toe
  function mctsWidget(root) {
    const LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];
    const NAMES = ["top-left", "top", "top-right", "left", "centre", "right", "bottom-left", "bottom", "bottom-right"];
    const START_BOARD = "..O..XX.O";              // X to move; "left" (cell 3) wins by a fork
    const C = Math.SQRT2;

    const winner = (b) => { for (const [i, j, k] of LINES) if (b[i] !== "." && b[i] === b[j] && b[j] === b[k]) return b[i]; return null; };
    const toMove = (b) => ((b.match(/X/g) || []).length === (b.match(/O/g) || []).length ? "X" : "O");
    const legal = (b) => (winner(b) ? [] : [...b].map((v, i) => (v === "." ? i : -1)).filter((i) => i >= 0));
    const play = (b, i) => b.slice(0, i) + toMove(b) + b.slice(i + 1);
    const newNode = (b, parent, move) => ({ b, parent, move, children: [], untried: legal(b), N: 0, W: 0 });
    const uct = (ch) => ch.W / ch.N + C * Math.sqrt(Math.log(ch.parent.N) / ch.N);

    const W = 440, H = 290;
    const tree = svg(W, H);
    root.appendChild(tree);
    const caption = document.createElement("div");
    caption.className = "viz-readout";
    caption.style.whiteSpace = "pre-line";
    caption.style.color = "var(--viz-text)";
    root.appendChild(caption);
    const ctrl = control(root, `
      <button data-a="phase">Next phase</button><button data-a="one">1 simulation</button>
      <button data-a="ten">+10</button><button data-a="hundred">+100</button>
      <button data-a="reset">Reset</button><button data-a="seed">New seed</button>`);
    const table = document.createElement("div");
    table.style.overflowX = "auto";
    table.style.marginTop = "10px";
    root.appendChild(table);
    const out = readout(root);

    let seed = 1, rand, rootNode, K, sim;

    function reset() {
      rand = rng(seed);
      rootNode = newNode(START_BOARD, null, null);
      K = 0;
      sim = null;
      caption.textContent = "X to move. Press Next phase to watch one simulation, one phase at a time.";
      draw();
    }

    // One phase of the current simulation. Returns true when the simulation finished.
    function phase() {
      if (!sim) {
        // 1. Selection
        let node = rootNode;
        const path = [node];
        while (!node.untried.length && node.children.length) {
          let best = null, bv = -Infinity;
          for (const ch of node.children) { const v = uct(ch); if (v > bv) { bv = v; best = ch; } }
          node = best; path.push(node);
        }
        sim = { stage: 1, path, node, added: null, playout: [], result: null };
        const route = path.slice(1).map((n) => `${toMove(n.parent.b)} ${NAMES[n.move]}`).join(" → ");
        caption.textContent = `Simulation ${K + 1}, phase 1 · Selection: start at the root and keep taking the child with the highest UCT score` +
          (route ? `: ${route}.` : ". The root still has untried moves, so we stop right there.");
        return false;
      }
      if (sim.stage === 1) {
        // 2. Expansion
        sim.stage = 2;
        if (sim.node.untried.length) {
          const m = sim.node.untried.splice(randInt(rand, sim.node.untried.length), 1)[0];
          const ch = newNode(play(sim.node.b, m), sim.node, m);
          sim.node.children.push(ch);
          sim.node = ch; sim.path.push(ch); sim.added = ch;
          caption.textContent = `Simulation ${K + 1}, phase 2 · Expansion: add one new node to the tree for a move never tried here: ${toMove(sim.node.parent.b)} plays ${NAMES[m]}.`;
        } else {
          caption.textContent = `Simulation ${K + 1}, phase 2 · Expansion: this node is a finished game, so there is nothing to add.`;
        }
        return false;
      }
      if (sim.stage === 2) {
        // 3. Simulation (random play-out)
        sim.stage = 3;
        let b = sim.node.b;
        while (legal(b).length) {
          const ms = legal(b), m = ms[randInt(rand, ms.length)];
          sim.playout.push(`${toMove(b)} ${NAMES[m]}`);
          b = play(b, m);
        }
        sim.result = winner(b);
        const res = sim.result ? `${sim.result} wins` : "draw";
        caption.textContent = `Simulation ${K + 1}, phase 3 · Simulation: from the new node, play random moves to the end` +
          (sim.playout.length ? `: ${sim.playout.join(", ")}` : " (the game is already over)") + `. Result: ${res}.`;
        return false;
      }
      // 4. Backup
      for (const n of sim.path) {
        const mover = toMove(n.b) === "X" ? "O" : "X";
        n.N += 1;
        n.W += sim.result === null ? 0 : sim.result === mover ? 1 : -1;
      }
      const res = sim.result === "X" ? "+1 for every X move on the path, −1 for every O move" : sim.result === "O" ? "+1 for every O move on the path, −1 for every X move" : "0 for every node on the path (a draw)";
      caption.textContent = `Simulation ${K + 1}, phase 4 · Backup: add 1 to N of every node on the highlighted path, and add the result to W: ${res}.`;
      K += 1;
      sim = { done: true, path: sim.path, added: sim.added };
      return true;
    }

    function simulate(count) {
      for (let i = 0; i < count; i++) {
        if (sim && sim.done) sim = null;
        while (!phase());
      }
      caption.textContent = `${K} simulations done. The highlighted path is the last one. Thick lines = visited often.`;
      draw();
    }

    // ---- drawing
    function miniBoard(g, b, cx, cy, size, highlight) {
      const col = colors(), c = size / 3, x0 = cx - size / 2, y0 = cy - size / 2;
      el("rect", { x: x0, y: y0, width: size, height: size, fill: col.bg, stroke: col.border, rx: 3 }, g);
      for (let k = 1; k < 3; k++) {
        el("line", { x1: x0 + k * c, x2: x0 + k * c, y1: y0 + 2, y2: y0 + size - 2, stroke: col.border }, g);
        el("line", { y1: y0 + k * c, y2: y0 + k * c, x1: x0 + 2, x2: x0 + size - 2, stroke: col.border }, g);
      }
      for (let i = 0; i < 9; i++) {
        if (b[i] === ".") continue;
        const t = el("text", { x: x0 + (i % 3 + 0.5) * c, y: y0 + (Math.floor(i / 3) + 0.5) * c + c * 0.3, "text-anchor": "middle" }, g);
        t.setAttribute("style", `font-size: ${c * 0.85}px; font-weight: 700; fill: ${i === highlight ? col.model : col.text}`);
        t.textContent = b[i];
      }
    }

    function draw() {
      const col = colors();
      tree.innerHTML = "";
      const onPath = new Set(sim ? sim.path : []);
      const rootX = W / 2, rootY = 36, childY = 128, gY = 228;
      const rootMoves = legal(START_BOARD);
      const slot = (m) => 52 + rootMoves.indexOf(m) * 84;
      const deeper = (n) => n.children.reduce((s, ch) => s + 1 + deeper(ch), 0);
      const lineW = (n) => 1 + 7 * Math.sqrt(n.N / Math.max(1, rootNode.N));

      // edges first
      for (const m of rootMoves) {
        const ch = rootNode.children.find((c) => c.move === m);
        const x = slot(m);
        el("line", { x1: rootX, y1: rootY + 27, x2: x, y2: childY - 21, stroke: ch && onPath.has(ch) ? col.model : col.border,
          "stroke-width": ch ? lineW(ch) : 1, "stroke-dasharray": ch ? "" : "3 4" }, tree);
        if (!ch) continue;
        const kids = ch.children.slice().sort((a, b) => a.move - b.move);
        kids.forEach((g, j) => {
          const gx = x + (j - (kids.length - 1) / 2) * 20;
          el("line", { x1: x, y1: childY + 60, x2: gx, y2: gY, stroke: onPath.has(g) ? col.model : col.border, "stroke-width": lineW(g) }, tree);
        });
      }
      miniBoard(tree, rootNode.b, rootX, rootY, 54, -1);
      el("text", { x: rootX + 36, y: rootY - 12 }, tree).textContent = "root";
      el("text", { x: rootX + 36, y: rootY + 4, style: textStyle }, tree).textContent = `N = ${rootNode.N}`;
      el("text", { x: rootX - 36, y: rootY - 12, "text-anchor": "end" }, tree).textContent = "X to move";

      const best = rootNode.children.length ? rootNode.children.reduce((a, b) => (b.N > a.N ? b : a)) : null;
      for (const m of rootMoves) {
        const ch = rootNode.children.find((c) => c.move === m);
        const x = slot(m);
        if (!ch) {
          const g = el("g", { opacity: 0.35 }, tree);
          miniBoard(g, play(START_BOARD, m), x, childY, 42, m);
          el("text", { x, y: childY + 36, "text-anchor": "middle" }, tree).textContent = "not tried";
          continue;
        }
        if (onPath.has(ch) || (best === ch && K > 0)) {
          el("rect", { x: x - 25, y: childY - 25, width: 50, height: 50, rx: 6, fill: "none", stroke: onPath.has(ch) ? col.model : col.opt, "stroke-width": 2.5 }, tree);
        }
        miniBoard(tree, ch.b, x, childY, 42, m);
        el("text", { x, y: childY + 39, "text-anchor": "middle", style: textStyle + "; font-size: 12px" }, tree).textContent = `N ${ch.N}`;
        el("text", { x, y: childY + 53, "text-anchor": "middle", style: "font-size: 12px" }, tree).textContent = ch.N ? `Q ${ch.W / ch.N >= 0 ? "+" : ""}${fmt(ch.W / ch.N, 2)}` : "";
        const kids = ch.children.slice().sort((a, b) => a.move - b.move);
        kids.forEach((g, j) => {
          const gx = x + (j - (kids.length - 1) / 2) * 20;
          el("circle", { cx: gx, cy: gY, r: Math.min(8.5, 3 + 1.2 * Math.sqrt(g.N)), fill: onPath.has(g) ? col.model : col.bg, stroke: onPath.has(g) ? col.model : col.muted, "stroke-width": 1.5 }, tree);
        });
        const d = deeper(ch) - kids.length;
        if (d > 0) el("text", { x, y: gY + 26, "text-anchor": "middle" }, tree).textContent = `+${d} deeper`;
      }
      el("text", { x: 3, y: gY + 4 }, tree).textContent = "O";
      el("text", { x: 3, y: childY + 4 }, tree).textContent = "X";
      el("text", { x: 8, y: H - 8 }, tree).textContent = "circle size = visits of O's replies; Q is X's average result";

      // table of root children
      const rows = rootMoves.map((m) => {
        const ch = rootNode.children.find((c) => c.move === m);
        if (!ch) return `<tr><td>${NAMES[m]}</td><td>0</td><td>–</td><td>–</td><td>not tried</td></tr>`;
        if (!ch.N) return `<tr><td>${NAMES[m]}</td><td>0</td><td>–</td><td>∞</td><td>just added</td></tr>`;
        const share = rootNode.N ? ch.N / rootNode.N : 0;
        const bar = `<span style="display:inline-block;height:9px;width:${Math.round(share * 60)}px;background:var(${ch === best ? "--viz-opt" : "--viz-data"});border-radius:2px;vertical-align:middle"></span> ${Math.round(share * 100)}%`;
        return `<tr><td>${NAMES[m]}</td><td>${ch.N}</td><td>${ch.W >= 0 ? "+" : ""}${fmt(ch.W / ch.N, 2)}</td><td>${fmt(uct(ch), 2)}</td><td style="white-space:nowrap">${bar}</td></tr>`;
      });
      table.innerHTML = `<table class="mbrl-mcts-table" style="font-size:0.72rem;border-collapse:collapse;width:100%">
        <thead><tr style="text-align:left"><th>X plays</th><th>N</th><th>Q = W/N</th><th>UCT</th><th>visits</th></tr></thead>
        <tbody>${rows.join("")}</tbody></table>`;
      for (const c of table.querySelectorAll("td, th")) { c.style.padding = "3px 6px 3px 0"; c.style.borderBottom = "1px solid var(--viz-border)"; }
      out.textContent = best && K > 0
        ? `K = ${K} simulations → MCTS would play ${NAMES[best.move]} (most visited, ${Math.round((100 * best.N) / rootNode.N)}% of visits).`
        : `K = ${K} simulations. Seed ${seed}.`;
    }

    ctrl.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (a === "phase") {
        if (sim && sim.done) sim = null;
        phase();
        draw();
      }
      if (a === "one") simulate(1);
      if (a === "ten") simulate(10);
      if (a === "hundred") simulate(100);
      if (a === "reset") reset();
      if (a === "seed") { seed += 1; reset(); }
    });
    onThemeChange(draw);
    reset();
  }

  const mounts = { "viz-dyna": dynaWidget, "viz-dyna-curves": curvesWidget, "viz-mcts": mctsWidget };
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
