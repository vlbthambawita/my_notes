// Interactive widgets for the "Dynamic Programming: Policy and Value Iteration" lesson.
// Three widgets, each mounted on a <div> with a matching id:
//   #viz-eval – iterative policy evaluation, sweep by sweep (+ the greedy policy and Q-values)
//   #viz-pi   – policy iteration: alternate "evaluate" and "improve" until the policy stops changing
//   #viz-vi   – value iteration: Bellman optimality sweeps, greedy arrows and the max change Δ per sweep
// All three use the cliff grid world from lesson 1 and a known model p(s', r | s, a).
// Sweeps are synchronous (every state is updated from the previous sweep's values), like the numpy code in the lesson.
document.addEventListener("DOMContentLoaded", function () {
  const { colors, onThemeChange, linear, el, svg, fmt } = window.Viz;

  // ---------------------------------------------------------------- The grid world (same as lesson 1)
  const MAP = [
    "......",
    "..#...",
    "......",
    "SCCCCG",
  ];
  const ROWS = MAP.length, COLS = MAP[0].length;
  const ACTIONS = [[-1, 0], [0, 1], [1, 0], [0, -1]];   // up, right, down, left
  const ARROWS = ["↑", "→", "↓", "←"];
  const NAMES = ["up", "right", "down", "left"];
  const RANDOM = -1;                                    // policy entry meaning "each direction with probability 1/4"
  const TOL = 1e-6;                                     // stop sweeping when the biggest change Δ is below this

  const tile = (r, c) => MAP[r][c];
  const isFloor = (r, c) => tile(r, c) === "." || tile(r, c) === "S";

  // Non-terminal states, numbered 0..17
  const STATES = [];
  for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) if (isFloor(r, c)) STATES.push([r, c]);
  const N = STATES.length;
  const sid = (r, c) => STATES.findIndex(([a, b]) => a === r && b === c);
  const START = sid(3, 0);

  function move(r, c, a) {
    const nr = r + ACTIONS[a][0], nc = c + ACTIONS[a][1];
    if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS || tile(nr, nc) === "#") return [r, c];
    return [nr, nc];
  }

  // The model p(s', r | s, a): model[s][a] = list of [probability, next state id (-1 = terminal), reward]
  function buildModel(slip) {
    return STATES.map(([r, c]) => [0, 1, 2, 3].map((a) =>
      [[1 - slip, a], [slip / 2, (a + 1) % 4], [slip / 2, (a + 3) % 4]]
        .filter(([p]) => p > 0)
        .map(([p, d]) => {
          const [nr, nc] = move(r, c, d), t = tile(nr, nc);
          const rew = t === "G" ? 10 : t === "C" ? -20 : -1;
          return [p, t === "G" || t === "C" ? -1 : sid(nr, nc), rew];
        })));
  }

  // One-step lookahead: Q(s, a) = Σ p · [r + γ V(s')]
  function qValue(model, V, s, a, gamma) {
    let q = 0;
    for (const [p, ns, rw] of model[s][a]) q += p * (rw + (ns < 0 ? 0 : gamma * V[ns]));
    return q;
  }

  // One sweep of policy evaluation. Returns the new values and the biggest change Δ.
  function evalSweep(model, pi, V, gamma) {
    const out = V.map((_, s) => {
      const acts = pi[s] === RANDOM ? [0, 1, 2, 3] : [pi[s]];
      return acts.reduce((sum, a) => sum + qValue(model, V, s, a, gamma) / acts.length, 0);
    });
    return { V: out, delta: Math.max(...out.map((v, s) => Math.abs(v - V[s]))) };
  }

  function evaluateFull(model, pi, gamma) {
    let V = new Array(N).fill(0), sweeps = 0, delta = Infinity;
    while (delta >= TOL && sweeps < 20000) ({ V, delta } = evalSweep(model, pi, V, gamma)), sweeps++;
    return { V, sweeps };
  }

  // One sweep of value iteration: V(s) ← max_a Σ p · [r + γ V(s')]
  function viSweep(model, V, gamma) {
    const out = V.map((_, s) => Math.max(...[0, 1, 2, 3].map((a) => qValue(model, V, s, a, gamma))));
    return { V: out, delta: Math.max(...out.map((v, s) => Math.abs(v - V[s]))) };
  }

  // Greedy policy. With `keep`, a state keeps its old action when that action is (within 1e-9) as good as the best.
  function greedy(model, V, gamma, keep) {
    return V.map((_, s) => {
      const q = [0, 1, 2, 3].map((a) => qValue(model, V, s, a, gamma));
      let best = 0;
      for (let a = 1; a < 4; a++) if (q[a] > q[best]) best = a;
      if (keep && keep[s] !== RANDOM && q[keep[s]] >= q[best] - 1e-9) return keep[s];
      return best;
    });
  }

  // Where does the policy take you from the start when nothing slips?
  function route(pi) {
    let [r, c] = STATES[START];
    const path = [[r, c]], seen = new Set([r * COLS + c]);
    for (let k = 0; k < 40; k++) {
      [r, c] = move(r, c, pi[sid(r, c)]);
      path.push([r, c]);
      if (!isFloor(r, c) || seen.has(r * COLS + c)) break;
      seen.add(r * COLS + c);
    }
    const [lr, lc] = path[path.length - 1];
    const minRow = Math.min(...path.map(([r]) => r));
    let kind;
    if (tile(lr, lc) === "G") kind = minRow === 0 ? `safe detour over the top (${path.length - 1} steps)` : `along the cliff edge (${path.length - 1} steps)`;
    else if (tile(lr, lc) === "C") kind = "straight into the cliff";
    else if (pi[START] === 3) kind = "start cell points ← into the wall on purpose";
    else kind = "goes round in a loop (values not ready yet)";
    return { path, kind };
  }

  // ---------------------------------------------------------------- Drawing
  const CELL = 56, PAD = 6;
  const GW = COLS * CELL + 2 * PAD, GH = ROWS * CELL + 2 * PAD;
  const px = (c) => PAD + c * CELL, py = (r) => PAD + r * CELL;
  const textStyle = "fill: var(--viz-text)";

  // opts: V (values or null), pi (actions or null), arrowRole (colour key), mark (Set of state ids to outline), path
  function drawBoard(root, opts) {
    const col = colors();
    root.innerHTML = "";
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const t = tile(r, c), x = px(c), y = py(r);
        el("rect", { x, y, width: CELL, height: CELL, fill: col.bg, stroke: col.border }, root);
        if (t === "#") el("rect", { x: x + 1, y: y + 1, width: CELL - 2, height: CELL - 2, fill: col.muted, "fill-opacity": 0.55 }, root);
        if (t === "G" || t === "C") {
          el("rect", { x: x + 1, y: y + 1, width: CELL - 2, height: CELL - 2, fill: t === "G" ? col.opt : col.grad, "fill-opacity": t === "G" ? 0.45 : 0.6 }, root);
          el("text", { x: x + CELL / 2, y: y + CELL / 2 - 3, "text-anchor": "middle", style: textStyle }, root).textContent = t === "G" ? "GOAL" : "cliff";
          el("text", { x: x + CELL / 2, y: y + CELL / 2 + 12, "text-anchor": "middle" }, root).textContent = t === "G" ? "+10" : "-20";
        }
        if (t === "S") el("text", { x: x + 4, y: y + 12 }, root).textContent = "start";
      }
    }
    STATES.forEach(([r, c], s) => {
      const x = px(c), y = py(r), cx = x + CELL / 2;
      if (opts.V) {
        const v = opts.V[s];
        el("rect", { x: x + 1, y: y + 1, width: CELL - 2, height: CELL - 2, fill: v >= 0 ? col.opt : col.grad, "fill-opacity": Math.min(1, Math.abs(v) / 12) * 0.4 }, root);
        el("text", { x: cx, y: y + CELL - 5, "text-anchor": "middle", style: textStyle + "; font-size: 10px" }, root).textContent = fmt(v, 1);
      }
      if (opts.mark && opts.mark.has(s))
        el("rect", { x: x + 3, y: y + 3, width: CELL - 6, height: CELL - 6, fill: "none", stroke: col.model, "stroke-width": 2.5, rx: 4 }, root);
      if (opts.selected === s)
        el("rect", { x: x + 2, y: y + 2, width: CELL - 4, height: CELL - 4, fill: "none", stroke: col.text, "stroke-width": 2, "stroke-dasharray": "4 3", rx: 4 }, root);
      if (opts.pi) {
        const a = opts.pi[s];
        const style = (opts.arrowRole ? `fill: ${col[opts.arrowRole]}` : textStyle) + "; font-size: 20px; font-weight: 700";
        el("text", { x: cx, y: y + CELL / 2 + 5, "text-anchor": "middle", style }, root).textContent = a === RANDOM ? "✱" : ARROWS[a];
      }
    });
    if (opts.path && opts.path.length > 1) {
      const d = opts.path.map(([r, c], i) => (i ? "L" : "M") + (px(c) + CELL / 2) + "," + (py(r) + CELL / 2)).join("");
      el("path", { d, fill: "none", stroke: col.opt, "stroke-width": 4, "stroke-opacity": 0.45, "stroke-linejoin": "round", "stroke-linecap": "round" }, root);
    }
  }

  function boardSvg(parent) {
    const s = svg(GW, GH);
    parent.appendChild(s);
    return s;
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

  const SLIDERS = `
      <label>slip chance <input type="range" min="0" max="0.4" step="0.01" value="0"> <span></span></label>
      <label>γ <input type="range" min="0.5" max="0.99" step="0.01" value="0.95"> <span></span></label>`;

  function sliderValues(ctrl) {
    const [slipIn, gamIn] = ctrl.querySelectorAll("input[type=range]");
    slipIn.nextElementSibling.textContent = (+slipIn.value).toFixed(2);
    gamIn.nextElementSibling.textContent = (+gamIn.value).toFixed(2);
    return { slip: +slipIn.value, gamma: +gamIn.value };
  }

  // Hand-written policies from lesson 1 (rows top to bottom, . = no choice needed)
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
  const preset = (name) => STATES.map(([r, c]) => (name === "random" ? RANDOM : ARROWS.indexOf(PRESETS[name][r][c])));

  // ---------------------------------------------------------------- Widget 1: policy evaluation, sweep by sweep
  function evalWidget(root) {
    const board = boardSvg(root);
    board.style.maxWidth = "520px";
    board.style.margin = "0 auto";
    board.style.cursor = "pointer";
    hint(root, "Every cell starts at V = 0. Each sweep updates all 18 cells at once from the previous values. Click a cell to see its four Q-values.");
    const presets = control(root, `Policy: <button data-p="random">Random</button><button data-p="edge">Edge of the cliff</button><button data-p="safe">Safe detour</button>`);
    const ctrl = control(root, `
      <button data-a="step">1 sweep</button><button data-a="run">Run to Δ &lt; 10⁻⁶</button><button data-a="reset">Reset V = 0</button>
      <label><input type="checkbox"> show the greedy policy (one improvement step)</label>` + SLIDERS);
    const showGreedy = ctrl.querySelector("input[type=checkbox]");
    const out = readout(root);

    let model, gamma, pi, V, k, delta, timer = null, selected = START, policyName = "random";

    function reset() {
      stop();
      const p = sliderValues(ctrl);
      model = buildModel(p.slip); gamma = p.gamma;
      V = new Array(N).fill(0); k = 0; delta = NaN;
      draw();
    }
    function stop() { clearInterval(timer); timer = null; }
    function sweep() {
      if (delta < TOL) return false;
      ({ V, delta } = evalSweep(model, pi, V, gamma)); k++;
      return delta >= TOL;
    }

    function draw() {
      const g = showGreedy.checked ? greedy(model, V, gamma) : null;
      drawBoard(board, { V, pi: g || pi, arrowRole: g ? "opt" : null, selected });
      let text = `policy: ${policyName} · sweep k = ${k} · Δ = ${k ? fmt(delta, k && delta < 1e-3 ? 7 : 3) : "–"}${delta < TOL ? "  ✓ converged" : ""}\n`;
      text += `V(start) = ${fmt(V[START], 2)}`;
      if (g) {
        const changed = pi.reduce((n, a, s) => n + (a !== g[s] ? 1 : 0), 0);
        text += ` · greedy policy (green) differs from π in ${changed} of ${N} cells`;
      }
      const [r, c] = STATES[selected];
      const q = [0, 1, 2, 3].map((a) => qValue(model, V, selected, a, gamma));
      const best = q.indexOf(Math.max(...q));
      text += `\ncell (row ${r}, col ${c}): V = ${fmt(V[selected], 2)} · Q: ` + q.map((v, a) => `${ARROWS[a]} ${fmt(v, 2)}${a === best ? "★" : ""}`).join("  ");
      out.textContent = text;
    }

    presets.addEventListener("click", (e) => {
      const p = e.target.dataset.p;
      if (!p) return;
      policyName = e.target.textContent; pi = preset(p);
      reset();
    });
    ctrl.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (a === "step") { stop(); sweep(); draw(); }
      if (a === "reset") reset();
      if (a === "run") {
        stop();
        timer = setInterval(() => {
          let more = true;
          for (let i = 0; i < (k < 20 ? 1 : 4) && more; i++) more = sweep();
          draw();
          if (!more) stop();
        }, 60);
      }
    });
    showGreedy.addEventListener("change", draw);
    ctrl.querySelectorAll("input[type=range]").forEach((s) => s.addEventListener("input", reset));
    board.addEventListener("click", (e) => {
      const b = board.getBoundingClientRect();
      const c = Math.floor((((e.clientX - b.left) / b.width) * GW - PAD) / CELL);
      const r = Math.floor((((e.clientY - b.top) / b.height) * GH - PAD) / CELL);
      const s = sid(r, c);
      if (s >= 0) { selected = s; draw(); }
    });
    onThemeChange(draw);
    pi = preset("random"); policyName = "Random";
    reset();
  }

  // ---------------------------------------------------------------- Widget 2: policy iteration
  function piWidget(root) {
    const board = boardSvg(root);
    board.style.maxWidth = "520px";
    board.style.margin = "0 auto";
    hint(root, "Starts from the random policy (✱). Orange outlines mark the arrows the last improvement changed.");
    const ctrl = control(root, `<button data-a="next"></button><button data-a="run">Run to the end</button><button data-a="reset">Reset to random</button>` + SLIDERS);
    const nextBtn = ctrl.querySelector("[data-a=next]");
    const out = readout(root);

    // phase "evaluate": the next click evaluates pi; phase "improve": the next click improves it; "done": stable
    let model, gamma, pi, V, phase, round, history, changed, lastSweeps, timer = null;

    function reset() {
      stop();
      const p = sliderValues(ctrl);
      model = buildModel(p.slip); gamma = p.gamma;
      pi = preset("random"); V = null; phase = "evaluate"; round = 1; history = []; changed = new Set(); lastSweeps = 0;
      draw();
    }
    function stop() { clearInterval(timer); timer = null; }

    function next() {
      if (phase === "evaluate") {
        const res = evaluateFull(model, pi, gamma);
        V = res.V; lastSweeps = res.sweeps; history.push(fmt(V[START], 2));
        changed = new Set(); phase = "improve";
      } else if (phase === "improve") {
        const g = greedy(model, V, gamma, pi);
        changed = new Set(g.map((a, s) => (a !== pi[s] ? s : -1)).filter((s) => s >= 0));
        pi = g;
        phase = changed.size ? "evaluate" : "done";
        if (changed.size) round++;
      }
      draw();
      return phase !== "done";
    }

    function draw() {
      drawBoard(board, { V, pi, mark: changed, path: phase === "done" ? route(pi).path : null });
      nextBtn.textContent = phase === "evaluate" ? `Evaluate π${sub(round - 1)}` : phase === "improve" ? `Improve π${sub(round - 1)}` : "Done ✓";
      nextBtn.disabled = phase === "done";
      let text = `policy π${sub(round - 1)} · `;
      if (!V) text += "not evaluated yet: press Evaluate.";
      else if (phase === "improve") text += `evaluated in ${lastSweeps} sweeps: V(start) = ${fmt(V[START], 2)}. Next: act greedily on these values.`;
      else if (phase === "evaluate") text += `improvement changed ${changed.size} arrow${changed.size === 1 ? "" : "s"} (values shown are still the old policy's).`;
      else text += `improvement changed nothing: the policy is stable, so it is optimal.\n${route(pi).kind}`;
      if (history.length) text += `\nV(start) after each evaluation: ${history.join(" → ")}`;
      out.textContent = text;
    }

    const sub = (n) => String(n).split("").map((d) => "₀₁₂₃₄₅₆₇₈₉"[+d]).join("");

    ctrl.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (a === "next") { stop(); next(); }
      if (a === "reset") reset();
      if (a === "run") {
        stop();
        timer = setInterval(() => { if (!next()) stop(); }, 650);
      }
    });
    ctrl.querySelectorAll("input[type=range]").forEach((s) => s.addEventListener("input", reset));
    onThemeChange(draw);
    reset();
  }

  // ---------------------------------------------------------------- Widget 3: value iteration + Δ chart
  function viWidget(root) {
    const plots = document.createElement("div");
    plots.className = "viz-plots";
    root.appendChild(plots);
    const board = boardSvg(plots);
    const CW = 340, CH = GH, M = { l: 46, r: 12, t: 16, b: 34 };
    const chart = svg(CW, CH);
    plots.appendChild(chart);
    hint(root, "Arrows are the greedy policy for the current values; the green line is where it takes you from the start if nothing slips. Moving a slider restarts from V = 0.");
    const ctrl = control(root, `<button data-a="step">1 sweep</button><button data-a="run">Run</button><button data-a="reset">Reset V = 0</button>` + SLIDERS);
    const out = readout(root);

    let model, gamma, V, k, delta, deltas, timer = null;

    function reset(autorun) {
      stop();
      const p = sliderValues(ctrl);
      model = buildModel(p.slip); gamma = p.gamma;
      V = new Array(N).fill(0); k = 0; delta = NaN; deltas = [];
      draw();
      if (autorun) run();
    }
    function stop() { clearInterval(timer); timer = null; }
    function sweep() {
      if (delta < TOL) return false;
      ({ V, delta } = viSweep(model, V, gamma)); k++; deltas.push(delta);
      return delta >= TOL;
    }
    function run() {
      stop();
      timer = setInterval(() => { const more = sweep(); draw(); if (!more) stop(); }, 45);
    }

    function drawChart() {
      const col = colors();
      chart.innerHTML = "";
      const xmax = Math.max(40, Math.ceil(deltas.length / 20) * 20);
      const x = linear(0, xmax, M.l, CW - M.r), y = linear(-7, 2, CH - M.b, M.t);
      const g = el("g", {}, chart);
      for (let t = 0; t <= xmax; t += xmax > 60 ? 20 : 10) {
        el("line", { x1: x(t), x2: x(t), y1: y(-7), y2: y(2), stroke: col.grid }, g);
        el("text", { x: x(t), y: y(-7) + 14, "text-anchor": "middle" }, g).textContent = t;
      }
      for (let e = -6; e <= 2; e += 2) {
        el("line", { x1: x(0), x2: x(xmax), y1: y(e), y2: y(e), stroke: col.grid }, g);
        el("text", { x: x(0) - 6, y: y(e) + 4, "text-anchor": "end" }, g).textContent = e === 0 ? "1" : e === 2 ? "100" : `1e${e}`;
      }
      el("text", { x: (x(0) + x(xmax)) / 2, y: CH - 4, "text-anchor": "middle" }, g).textContent = "sweep k";
      el("text", { x: 12, y: y(-2.5), "text-anchor": "middle", transform: `rotate(-90 12 ${y(-2.5)})` }, g).textContent = "max change Δ (log scale)";
      el("line", { x1: x(0), x2: x(xmax), y1: y(-6), y2: y(-6), stroke: col.opt, "stroke-dasharray": "5 4", "stroke-width": 1.5 }, chart);
      el("text", { x: x(0) + 6, y: y(-6) - 5, "text-anchor": "start", style: `fill: ${col.opt}` }, chart).textContent = "stop: Δ < 10⁻⁶";
      const ly = (d) => y(Math.max(-7, Math.log10(Math.max(d, 1e-12))));
      if (deltas.length) {
        const d = deltas.map((v, i) => (i ? "L" : "M") + x(i + 1) + "," + ly(v)).join("");
        el("path", { d, fill: "none", stroke: col.loss, "stroke-width": 2 }, chart);
        deltas.forEach((v, i) => el("circle", { cx: x(i + 1), cy: ly(v), r: 2.6, fill: col.loss }, chart));
      }
    }

    function draw() {
      const pi = greedy(model, V, gamma);
      const rt = route(pi);
      drawBoard(board, { V, pi, path: k ? rt.path : null });
      drawChart();
      let text = `sweep k = ${k} · Δ = ${k ? (delta < 1e-3 ? delta.toExponential(1) : fmt(delta, 3)) : "–"}${delta < TOL ? " ✓ converged" : ""} · V(start) = ${fmt(V[START], 2)}`;
      if (k) text += `\ngreedy route from the start: ${rt.kind}`;
      out.textContent = text;
    }

    ctrl.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (a === "step") { stop(); sweep(); draw(); }
      if (a === "run") run();
      if (a === "reset") reset(false);
    });
    ctrl.querySelectorAll("input[type=range]").forEach((s) => s.addEventListener("input", () => reset(true)));
    onThemeChange(draw);
    reset(false);
  }

  const mounts = { "viz-eval": evalWidget, "viz-pi": piWidget, "viz-vi": viWidget };
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
