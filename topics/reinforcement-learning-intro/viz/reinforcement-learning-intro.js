// Interactive widgets for the "What is reinforcement learning?" lesson.
// Three widgets, each mounted on a <div> with a matching id:
//   #viz-play     – be the agent: walk through a grid world and collect rewards
//   #viz-discount – drag the discount factor γ and compare two reward streams
//   #viz-policy   – edit a policy, see the value of every cell, and test it with real episodes
// Runs after DOMContentLoaded so the site-wide helpers (assets/js/viz-common.js),
// which load at the end of the page, are available.
document.addEventListener("DOMContentLoaded", function () {
  const { colors, onThemeChange, linear, el, svg, axes, ticks, fmt } = window.Viz;

  // ---------------------------------------------------------------- The grid world
  // S = start, G = goal (+10), C = cliff (−20), # = wall, . = floor (−1 per step)
  const MAP = [
    "......",
    "..#...",
    "......",
    "SCCCCG",
  ];
  const ROWS = MAP.length, COLS = MAP[0].length;
  const START = [3, 0];
  const STEP_R = -1, GOAL_R = 10, CLIFF_R = -20;
  const ACTIONS = [[-1, 0], [0, 1], [1, 0], [0, -1]];   // up, right, down, left
  const ARROWS = ["↑", "→", "↓", "←"];
  const RANDOM = -1;                                    // policy entry meaning "pick any direction"

  const tile = (r, c) => MAP[r][c];
  const isTerminal = (r, c) => tile(r, c) === "G" || tile(r, c) === "C";
  const isFloor = (r, c) => tile(r, c) === "." || tile(r, c) === "S";

  // Where the agent lands if it moves in direction a (walls and edges: it stays put)
  function move(r, c, a) {
    const nr = r + ACTIONS[a][0], nc = c + ACTIONS[a][1];
    if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS || tile(nr, nc) === "#") return [r, c];
    return [nr, nc];
  }

  const reward = (r, c) => (tile(r, c) === "G" ? GOAL_R : tile(r, c) === "C" ? CLIFF_R : STEP_R);

  // Every possible result of trying action a: [probability, row, col, reward, done].
  // On a slippery floor the agent slides sideways (left or right of where it aimed) with probability `slip`.
  function outcomes(r, c, a, slip) {
    return [[1 - slip, a], [slip / 2, (a + 1) % 4], [slip / 2, (a + 3) % 4]]
      .filter(([p]) => p > 0)
      .map(([p, d]) => {
        const [nr, nc] = move(r, c, d);
        return [p, nr, nc, reward(nr, nc), isTerminal(nr, nc)];
      });
  }

  // Sample one step of the environment
  function envStep(r, c, a, slip) {
    const outs = outcomes(r, c, a, slip);
    let u = Math.random();
    for (const o of outs) if ((u -= o[0]) <= 0) return o;
    return outs[outs.length - 1];
  }

  const CELL = 56, PAD = 6;
  const GW = COLS * CELL + 2 * PAD, GH = ROWS * CELL + 2 * PAD;
  const px = (c) => PAD + c * CELL, py = (r) => PAD + r * CELL;
  const textStyle = "fill: var(--viz-text)";

  // Draw the board. opts.tint(r, c) may return [colour, opacity] to shade a floor cell.
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
          el("text", { x: x + CELL / 2, y: y + CELL / 2 + 12, "text-anchor": "middle" }, root).textContent = t === "G" ? `+${GOAL_R}` : `${CLIFF_R}`;
        }
        if (isFloor(r, c) && opts.tint) {
          const [fill, op] = opts.tint(r, c);
          el("rect", { x: x + 1, y: y + 1, width: CELL - 2, height: CELL - 2, fill, "fill-opacity": op }, root);
        }
        if (t === "S") el("text", { x: x + 4, y: y + 12 }, root).textContent = "start";
      }
    }
  }

  function drawAgent(root, r, c) {
    const col = colors();
    el("circle", { cx: px(c) + CELL / 2, cy: py(r) + CELL / 2, r: 13, fill: col.model, stroke: col.bg, "stroke-width": 2 }, root);
  }

  function drawTrail(root, trail) {
    if (trail.length < 2) return;
    const d = trail.map(([r, c], i) => (i ? "L" : "M") + (px(c) + CELL / 2) + "," + (py(r) + CELL / 2)).join("");
    el("path", { d, fill: "none", stroke: colors().model, "stroke-width": 3, "stroke-opacity": 0.45, "stroke-linejoin": "round" }, root);
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

  function boardSvg(root) {
    const s = svg(GW, GH);
    s.style.maxWidth = "520px";
    s.style.margin = "0 auto";
    root.appendChild(s);
    return s;
  }

  // ---------------------------------------------------------------- Widget 1: be the agent
  function playWidget(root) {
    const board = boardSvg(root);
    const hint = document.createElement("div");
    hint.className = "viz-hint";
    hint.textContent = "Use the buttons, or click the board and use the arrow keys (R starts a new episode).";
    root.appendChild(hint);
    const ctrl = control(root, `
      <button data-a="3">← left</button><button data-a="0">↑ up</button><button data-a="2">↓ down</button><button data-a="1">→ right</button>
      <button data-a="reset">New episode</button>
      <label><input type="checkbox"> slippery floor (20% chance to slide sideways)</label>`);
    const slippery = ctrl.querySelector("input");
    const out = readout(root);

    let pos, t, total, done, trail, last, history = [];

    function reset() {
      pos = START.slice(); t = 0; total = 0; done = false; trail = [pos.slice()];
      last = "You are the agent (orange). Reach the goal; every step costs −1.";
      draw();
    }

    function act(a) {
      if (done) return;
      const [, nr, nc, rew, end] = envStep(pos[0], pos[1], a, slippery.checked ? 0.2 : 0);
      const [ir, ic] = move(pos[0], pos[1], a);
      const slipped = ir !== nr || ic !== nc;
      t += 1; total += rew; pos = [nr, nc]; done = end; trail.push(pos.slice());
      last = `t=${t}: action ${ARROWS[a]}${slipped ? " … you slipped!" : ""}  →  reward ${rew > 0 ? "+" : ""}${rew}`;
      if (end) {
        last += tile(nr, nc) === "G" ? "   🎉 reached the goal, episode over." : "   💥 fell off the cliff, episode over.";
        history.push(total);
        if (history.length > 8) history.shift();
      }
      draw();
    }

    function draw() {
      board.innerHTML = "";
      drawBoard(board);
      drawTrail(board, trail);
      drawAgent(board, pos[0], pos[1]);
      const past = history.length ? `\npast episodes (total reward): ${history.join(", ")}` : "";
      out.textContent = `state = (row ${pos[0]}, col ${pos[1]})   steps = ${t}   total reward = ${total}\n${last}${past}`;
    }

    ctrl.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (a === "reset") reset();
      else if (a !== undefined) act(+a);
    });
    slippery.addEventListener("change", reset);
    board.tabIndex = 0;
    board.style.outline = "none";
    board.addEventListener("click", () => board.focus());
    board.addEventListener("keydown", (e) => {
      const keys = { ArrowUp: 0, ArrowRight: 1, ArrowDown: 2, ArrowLeft: 3 };
      if (e.key in keys) { e.preventDefault(); act(keys[e.key]); }
      if (e.key === "r" || e.key === "R") reset();
    });
    onThemeChange(draw);
    reset();
  }

  // ---------------------------------------------------------------- Widget 2: discounting
  function discountWidget(root) {
    const W = 340, H = 230, M = { l: 36, r: 10, t: 38, b: 36 }, T = 12;
    const STREAMS = [
      { name: "A: +4 right now", rewards: Array.from({ length: T }, (_, t) => (t === 0 ? 4 : 0)) },
      { name: "B: +10 after 6 steps", rewards: Array.from({ length: T }, (_, t) => (t === 6 ? 10 : 0)) },
    ];
    const plots = document.createElement("div");
    plots.className = "viz-plots";
    root.appendChild(plots);
    const charts = STREAMS.map(() => { const s = svg(W, H); plots.appendChild(s); return s; });

    const ctrl = control(root, `
      <label>discount factor <b>γ</b> <input type="range" min="0" max="1" step="0.01" value="0.9"> <span class="val"></span></label>`);
    const presets = control(root, `Try: <button data-g="0">γ = 0</button><button data-g="0.5">0.5</button><button data-g="0.8">0.8</button><button data-g="0.9">0.9</button><button data-g="0.99">0.99</button>`);
    const slider = ctrl.querySelector("input"), val = ctrl.querySelector(".val");
    const out = readout(root);

    const x = linear(-0.6, T - 0.4, M.l, W - M.r), y = linear(0, 10, H - M.b, M.t);
    const bw = x(0.35) - x(0);

    function draw() {
      const c = colors(), g = +slider.value;
      val.textContent = g.toFixed(2);
      const returns = STREAMS.map((s, i) => {
        const root = charts[i];
        root.innerHTML = "";
        axes(root, x, y, { xDomain: [-0.6, T - 0.4], yDomain: [0, 10], xTicks: ticks(0, T - 1, 1), yTicks: ticks(0, 10, 2), xLabel: "time step t" });
        let G = 0;
        s.rewards.forEach((rw, t) => {
          if (!rw) return;
          const disc = rw * g ** t;
          G += disc;
          el("rect", { x: x(t) - bw, y: y(rw), width: 2 * bw, height: y(0) - y(rw), fill: "none", stroke: c.data, "stroke-dasharray": "4 3", "stroke-width": 1.5 }, root);
          el("rect", { x: x(t) - bw, y: y(disc), width: 2 * bw, height: y(0) - y(disc), fill: c.data }, root);
          el("text", { x: x(t), y: y(Math.max(rw, disc)) - 5, "text-anchor": "middle", style: textStyle }, root).textContent = `${rw} × ${fmt(g ** t, 2)} = ${fmt(disc, 2)}`;
        });
        el("text", { x: M.l + 4, y: 14, style: textStyle }, root).textContent = `${s.name}   →   G = ${fmt(G, 2)}`;
        return G;
      });
      const [ga, gb] = returns;
      const verdict = Math.abs(ga - gb) < 0.01 ? "both are worth the same" : ga > gb ? "the agent prefers A (the reward now)" : "the agent prefers B (the bigger reward later)";
      out.textContent = `dashed = raw reward, solid = what it is worth today (reward × γ^t)\nG(A) = ${fmt(ga, 2)}   G(B) = ${fmt(gb, 2)}   →  ${verdict}`;
    }

    slider.addEventListener("input", draw);
    presets.addEventListener("click", (e) => {
      if (e.target.dataset.g === undefined) return;
      slider.value = e.target.dataset.g;
      draw();
    });
    onThemeChange(draw);
    draw();
  }

  // ---------------------------------------------------------------- Widget 3: policy and value
  function policyWidget(root) {
    const board = boardSvg(root);
    board.style.cursor = "pointer";
    const hint = document.createElement("div");
    hint.className = "viz-hint";
    hint.textContent = "Click a floor cell to change its arrow (↑ → ↓ ← then ✱ = random). Green cells are good places to be, red cells are bad.";
    root.appendChild(hint);
    const presets = control(root, `Policy: <button data-p="random">Random</button><button data-p="edge">Edge of the cliff</button><button data-p="safe">Safe detour</button>`);
    const ctrl = control(root, `
      <label>slip chance <input type="range" min="0" max="0.4" step="0.05" value="0"> <span class="slip"></span></label>
      <label>γ <input type="range" min="0.5" max="0.99" step="0.01" value="0.95"> <span class="gam"></span></label>
      <button data-a="one">Run 1 episode</button><button data-a="many">Run 1000 episodes</button>`);
    const [slipIn, gamIn] = ctrl.querySelectorAll("input");
    const out = readout(root);

    // Hand-written policies: one action per cell (rows top to bottom), . = no choice needed
    const PRESETS = {
      edge: ["↓↓→↓↓↓",
             "↓↓.↓↓↓",
             "→→→→→↓",
             "↑....."],
      safe: ["→→→→→↓",
             "↑↑.↑↑↓",
             "↑↑←↑↑↓",
             "↑....."],
    };

    let policy, values, trail = [], agent = null, timer = null, mc = null;

    function setPreset(name) {
      policy = MAP.map((row, r) => [...row].map((_, c) =>
        name === "random" ? RANDOM : ARROWS.indexOf(PRESETS[name][r][c])));
      refresh();
    }

    // Policy evaluation: repeat V(s) ← Σ_a π(a|s) Σ_s' p(s'|s,a) [r + γ V(s')] until nothing changes.
    // (How this works is the topic of a later lesson; here we only use the result.)
    function evaluate() {
      const slip = +slipIn.value, gamma = +gamIn.value;
      const V = MAP.map((row) => [...row].map(() => 0));
      for (let iter = 0; iter < 5000; iter++) {
        let delta = 0;
        for (let r = 0; r < ROWS; r++) {
          for (let c = 0; c < COLS; c++) {
            if (!isFloor(r, c)) continue;
            const acts = policy[r][c] === RANDOM ? [0, 1, 2, 3] : [policy[r][c]];
            let v = 0;
            for (const a of acts)
              for (const [p, nr, nc, rw, end] of outcomes(r, c, a, slip))
                v += (p / acts.length) * (rw + (end ? 0 : gamma * V[nr][nc]));
            delta = Math.max(delta, Math.abs(v - V[r][c]));
            V[r][c] = v;
          }
        }
        if (delta < 1e-6) break;
      }
      return V;
    }

    const pickAction = (r, c) => (policy[r][c] === RANDOM ? Math.floor(Math.random() * 4) : policy[r][c]);

    // Play one episode with the current policy; returns the path and the discounted return
    function episode() {
      const slip = +slipIn.value, gamma = +gamIn.value;
      let [r, c] = START, G = 0, disc = 1, end = false, t = 0;
      const path = [[r, c]];
      while (!end && t < 100) {
        let rw;
        [, r, c, rw, end] = envStep(r, c, pickAction(r, c), slip);
        G += disc * rw; disc *= gamma; t++;
        path.push([r, c]);
      }
      return { path, G, result: end ? (tile(r, c) === "G" ? "goal" : "cliff") : "timeout" };
    }

    function stop() { clearInterval(timer); timer = null; }

    function refresh() {
      stop(); trail = []; agent = null; mc = null;
      values = evaluate();
      draw();
    }

    function draw() {
      const col = colors();
      slipIn.nextElementSibling.textContent = (+slipIn.value).toFixed(2);
      gamIn.nextElementSibling.textContent = (+gamIn.value).toFixed(2);
      board.innerHTML = "";
      drawBoard(board, {
        tint: (r, c) => {
          const v = values[r][c];
          return [v >= 0 ? col.opt : col.grad, Math.min(1, Math.abs(v) / 12) * 0.4];
        },
      });
      for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
          if (!isFloor(r, c)) continue;
          const cx = px(c) + CELL / 2, cy = py(r) + CELL / 2;
          const a = policy[r][c];
          el("text", { x: cx, y: cy + 4, "text-anchor": "middle", style: textStyle + "; font-size: 20px" }, board).textContent = a === RANDOM ? "✱" : ARROWS[a];
          el("text", { x: cx, y: py(r) + CELL - 5, "text-anchor": "middle", style: textStyle + "; font-size: 10px" }, board).textContent = fmt(values[r][c], 1);
        }
      }
      drawTrail(board, trail);
      if (agent) drawAgent(board, agent[0], agent[1]);

      const vs = values[START[0]][START[1]];
      let text = `value of the start cell under this policy:  V(start) = ${fmt(vs, 2)}`;
      if (mc) text += `\naverage return over ${mc.n} real episodes = ${fmt(mc.avg, 2)}   (goal ${mc.goal}% · cliff ${mc.cliff}% · gave up after 100 steps ${mc.timeout}%)`;
      out.textContent = text;
    }

    function runOne() {
      stop(); mc = null;
      const ep = episode();
      let i = 0;
      timer = setInterval(() => {
        trail = ep.path.slice(0, i + 1); agent = ep.path[i];
        draw();
        if (++i >= ep.path.length) {
          stop();
          const msg = { goal: "reached the goal 🎉", cliff: "fell off the cliff 💥", timeout: "wandered for 100 steps and gave up" }[ep.result];
          out.textContent += `\nthis episode: ${msg}, discounted return G = ${fmt(ep.G, 2)}`;
        }
      }, 140);
    }

    function runMany() {
      stop(); trail = []; agent = null;
      const n = 1000, count = { goal: 0, cliff: 0, timeout: 0 };
      let sum = 0;
      for (let k = 0; k < n; k++) { const ep = episode(); sum += ep.G; count[ep.result]++; }
      const pct = (k) => Math.round((100 * count[k]) / n);
      mc = { n, avg: sum / n, goal: pct("goal"), cliff: pct("cliff"), timeout: pct("timeout") };
      draw();
    }

    board.addEventListener("click", (e) => {
      const b = board.getBoundingClientRect();
      const c = Math.floor((((e.clientX - b.left) / b.width) * GW - PAD) / CELL);
      const r = Math.floor((((e.clientY - b.top) / b.height) * GH - PAD) / CELL);
      if (r < 0 || r >= ROWS || c < 0 || c >= COLS || !isFloor(r, c)) return;
      const a = policy[r][c];
      policy[r][c] = a === RANDOM ? 0 : a === 3 ? RANDOM : a + 1;
      refresh();
    });
    presets.addEventListener("click", (e) => { if (e.target.dataset.p) setPreset(e.target.dataset.p); });
    ctrl.addEventListener("click", (e) => {
      if (e.target.dataset.a === "one") runOne();
      if (e.target.dataset.a === "many") runMany();
    });
    slipIn.addEventListener("input", refresh);
    gamIn.addEventListener("input", refresh);
    onThemeChange(draw);
    setPreset("edge");
  }

  const mounts = { "viz-play": playWidget, "viz-discount": discountWidget, "viz-policy": policyWidget };
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
