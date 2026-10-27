// Interactive widgets for the "Monte Carlo and Temporal-Difference Learning" lesson.
// Widgets, each mounted on a <div> with a matching id:
//   #viz-walk-mc – random-walk episode stepper, Monte Carlo only
//   #viz-walk    – the same stepper with MC and TD(0) side by side
//   #viz-race    – RMS error over 100 episodes for MC vs TD, averaged over 100 seeded runs
//   #viz-cliff   – TD(0) estimating V of a fixed policy on the cliff world, compared with the DP values
// Runs after DOMContentLoaded so the site-wide helpers (assets/js/viz-common.js) are available.
document.addEventListener("DOMContentLoaded", function () {
  const { colors, onThemeChange, linear, el, svg, axes, ticks, fmt } = window.Viz;
  const textStyle = "fill: var(--viz-text)";

  // Small seeded random-number generator (mulberry32), so the learning curves are reproducible.
  function mulberry32(seed) {
    let a = seed | 0;
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
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
    div.innerHTML = text;
    parent.appendChild(div);
    return div;
  }

  // ---------------------------------------------------------------- The random walk
  // States 0..6: 0 and 6 are the two ends (terminal, value 0), 1..5 are A..E. Start in C (3).
  // Each step goes left or right with probability 1/2; leaving on the right gives +1, everything else 0.
  const NAMES = ["left end", "A", "B", "C", "D", "E", "right end"];
  const TRUE_V = [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 0];
  const rmsError = (V) => Math.sqrt([1, 2, 3, 4, 5].reduce((s, i) => s + (V[i] - TRUE_V[i]) ** 2, 0) / 5);
  const freshV = () => [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0];
  const nm = (i) => (i === 0 || i === 6 ? "end" : NAMES[i]);

  // ---------------------------------------------------------------- Widget 1/2: episode stepper
  function walkWidget(root, opts) {
    const showMC = opts.mc, showTD = opts.td;
    const W = 400, H = 330;
    const s = svg(W, H);
    s.style.maxWidth = "560px";
    s.style.margin = "0 auto";
    root.appendChild(s);
    hint(root, showTD
      ? "Press <b>Step</b> to move the agent one step. TD updates the state it just left, right away. MC waits until the episode ends, then updates every state it visited."
      : "Press <b>Step</b> to move the agent one step. Monte Carlo can't update anything until the episode ends and the return G is known.");
    const ctrl = control(root, `
      <button data-a="step">Step</button><button data-a="finish">Finish episode</button>
      <button data-a="ten">Run 10 episodes</button><button data-a="reset">Reset values</button>
      <label>step size <b>α</b> <input type="range" min="0.05" max="0.5" step="0.05" value="0.1"> <span class="val"></span></label>`);
    const alphaIn = ctrl.querySelector("input"), alphaVal = ctrl.querySelector(".val");
    const out = readout(root);

    const x = linear(0, 6, 30, W - 30);
    const y = linear(0, 1, H - 34, 116);
    let Vmc, Vtd, pos, path, done, episodes, changed, lines;

    function resetValues() {
      Vmc = freshV(); Vtd = freshV(); episodes = 0;
      newEpisode();
      lines = ["Values start at 0.5 for every state (a guess). The green dashes are the true values 1/6 … 5/6."];
      draw();
    }

    function newEpisode() {
      pos = 3; path = [3]; done = false; changed = { mc: new Set(), td: new Set() };
    }

    // Move one step; returns a description of the updates
    function step(quiet) {
      if (done) { newEpisode(); if (!quiet) lines = [`Episode ${episodes + 1} starts in C.`]; }
      const alpha = +alphaIn.value;
      const from = pos, to = pos + (Math.random() < 0.5 ? -1 : 1);
      const r = to === 6 ? 1 : 0;
      changed = { mc: new Set(), td: new Set() };
      const msg = [`Episode ${episodes + 1}, step ${path.length}: ${NAMES[from]} → ${NAMES[to]}, reward ${r}`];
      if (showTD) {
        const old = Vtd[from], delta = r + Vtd[to] - old;
        Vtd[from] = old + alpha * delta;
        changed.td.add(from);
        msg.push(`TD: δ = r + V(${nm(to)}) − V(${NAMES[from]}) = ${r} + ${fmt(Vtd[to], 3)} − ${fmt(old, 3)} = ${delta >= 0 ? "+" : ""}${fmt(delta, 3)}  →  V(${NAMES[from]}) = ${fmt(Vtd[from], 3)}`);
      }
      pos = to; path.push(to);
      if (to === 0 || to === 6) {
        done = true; episodes++;
        if (showMC) {
          const G = r;                       // γ = 1 and every earlier reward is 0
          const firstVisits = [...new Set(path.slice(0, -1))];
          for (const st of firstVisits) { Vmc[st] += alpha * (G - Vmc[st]); changed.mc.add(st); }
          msg.push(`MC: episode over, return G = ${G}. Every visited state (${firstVisits.map((i) => NAMES[i]).sort().join(", ")}) moves ${Math.round(alpha * 100)}% of the way toward ${G}.`);
        }
        msg.push(`Path: ${path.map(nm).join(" → ")}`);
      } else if (showMC) {
        msg.push("MC: still waiting. The return G isn't known until the episode ends.");
      }
      if (!quiet) lines = msg;
      return done;
    }

    function draw() {
      const c = colors();
      alphaVal.textContent = (+alphaIn.value).toFixed(2);
      s.innerHTML = "";
      // the track
      const ty = 44;
      el("line", { x1: x(0), x2: x(6), y1: ty, y2: ty, stroke: c.border, "stroke-width": 3 }, s);
      for (let i = 0; i <= 6; i++) {
        const end = i === 0 || i === 6;
        el("circle", { cx: x(i), cy: ty, r: 17, fill: end ? (i === 6 ? c.opt : c.bg) : c.bg, "fill-opacity": end && i === 6 ? 0.45 : 1, stroke: c.border, "stroke-width": 1.5 }, s);
        el("text", { x: x(i), y: ty + 4, "text-anchor": "middle", style: textStyle + "; font-weight: 600" }, s).textContent = end ? (i === 6 ? "+1" : "0") : NAMES[i];
        if (end) el("text", { x: x(i), y: ty - 24, "text-anchor": "middle" }, s).textContent = "end";
      }
      el("circle", { cx: x(pos), cy: ty + 27, r: 6, fill: c.model }, s);
      el("text", { x: x(3), y: ty - 24, "text-anchor": "middle" }, s).textContent = "start";

      // value bars
      axes(s, x, y, { xDomain: [0.4, 5.6], yDomain: [0, 1], xTicks: [], yTicks: [0, 0.5, 1] });
      const both = showMC && showTD, bw = both ? 15 : 22;
      const series = [];
      if (showMC) series.push({ V: Vmc, col: c.data, ch: changed.mc, dx: both ? -bw - 1 : -bw / 2 });
      if (showTD) series.push({ V: Vtd, col: c.loss, ch: changed.td, dx: both ? 1 : -bw / 2 });
      for (let i = 1; i <= 5; i++) {
        for (const sr of series) {
          const v = Math.max(0, Math.min(1, sr.V[i]));
          el("rect", { x: x(i) + sr.dx, y: y(v), width: bw, height: y(0) - y(v), fill: sr.col, "fill-opacity": sr.ch.has(i) ? 1 : 0.6, stroke: sr.ch.has(i) ? c.text : "none", "stroke-width": 1.5 }, s);
        }
        el("line", { x1: x(i) - 24, x2: x(i) + 24, y1: y(TRUE_V[i]), y2: y(TRUE_V[i]), stroke: c.opt, "stroke-width": 2.5, "stroke-dasharray": "5 3" }, s);
        el("text", { x: x(i), y: y(0) + 16, "text-anchor": "middle", style: textStyle }, s).textContent = NAMES[i];
      }
      // legend
      let lx = x(0.4) + 4;
      const ly = 100;
      const item = (fill, label, dashed) => {
        if (dashed) el("line", { x1: lx, x2: lx + 16, y1: ly - 4, y2: ly - 4, stroke: fill, "stroke-width": 2.5, "stroke-dasharray": "5 3" }, s);
        else el("rect", { x: lx, y: ly - 10, width: 12, height: 12, fill }, s);
        el("text", { x: lx + 20, y: ly }, s).textContent = label;
        lx += 20 + label.length * 6.2 + 14;
      };
      if (showMC) item(c.data, "MC estimate");
      if (showTD) item(c.loss, "TD estimate");
      item(c.opt, "true value", true);

      const err = [];
      if (showMC) err.push(`MC RMS error ${fmt(rmsError(Vmc), 3)}`);
      if (showTD) err.push(`TD RMS error ${fmt(rmsError(Vtd), 3)}`);
      out.textContent = `episodes finished: ${episodes}   ·   ${err.join("   ·   ")}\n` + lines.join("\n");
    }

    ctrl.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (!a) return;
      if (a === "reset") return resetValues();
      if (a === "step") step(false);
      if (a === "finish") {
        if (done) step(false);
        const tdChanged = new Set(changed.td);
        while (!done) { step(false); changed.td.forEach((i) => tdChanged.add(i)); }
        changed.td = tdChanged;
        lines = [`Episode ${episodes} finished after ${path.length - 1} steps.`]
          .concat(showTD ? [`TD: updated after every step along the way (${[...tdChanged].sort().map((i) => NAMES[i]).join(", ")}).`] : [])
          .concat(lines.filter((l) => !l.startsWith("TD") && !l.startsWith("Episode")));
      }
      if (a === "ten") {
        for (let k = 0; k < 10; k++) { if (done) newEpisode(); while (!step(true)); }
        changed = { mc: new Set(), td: new Set() };
        lines = ["Ran 10 more episodes. Compare the bars with the green dashes."];
      }
      draw();
    });
    alphaIn.addEventListener("input", draw);
    onThemeChange(draw);
    resetValues();
  }

  // ---------------------------------------------------------------- Widget 3: the race (learning curves)
  const RUNS = 100, EPISODES = 100;

  // Average RMS error after each episode, over RUNS seeded runs (seeds 1..100 for TD, 100001..100100 for MC)
  function learningCurve(method, alpha) {
    const curve = new Array(EPISODES + 1).fill(0);
    for (let k = 0; k < RUNS; k++) {
      const rng = mulberry32(1 + k + (method === "td" ? 0 : 100000));
      const V = freshV();
      curve[0] += rmsError(V) / RUNS;
      for (let ep = 1; ep <= EPISODES; ep++) {
        let st = 3;
        const states = [3];
        let z;
        for (;;) {
          st += rng() < 0.5 ? -1 : 1;
          if (st === 0) { z = 0; break; }
          if (st === 6) { z = 1; break; }
          states.push(st);
        }
        if (method === "td") {
          for (let i = 0; i < states.length; i++) {
            const nxt = i + 1 < states.length ? states[i + 1] : null;
            const target = nxt === null ? z : V[nxt];
            V[states[i]] += alpha * (target - V[states[i]]);
          }
        } else {
          const seen = new Set();
          for (const s of states) { if (seen.has(s)) continue; seen.add(s); V[s] += alpha * (z - V[s]); }
        }
        curve[ep] += rmsError(V) / RUNS;
      }
    }
    return curve;
  }

  function raceWidget(root) {
    const W = 440, H = 290, M = { l: 48, r: 14, t: 16, b: 42 };
    const s = svg(W, H);
    s.style.maxWidth = "560px";
    s.style.margin = "0 auto";
    root.appendChild(s);
    const ctrl = control(root, `
      <label>MC step size <b>α</b> <input type="range" min="0.01" max="0.1" step="0.01" value="0.02"> <span class="mc"></span></label>
      <label>TD step size <b>α</b> <input type="range" min="0.01" max="0.3" step="0.01" value="0.1"> <span class="td"></span></label>`);
    const presets = control(root, `Try: <button data-p="0.02,0.1">MC 0.02 · TD 0.1</button><button data-p="0.03,0.05">MC 0.03 · TD 0.05</button><button data-p="0.1,0.3">both large</button>`);
    const [mcIn, tdIn] = ctrl.querySelectorAll("input");
    const out = readout(root);
    hint(root, `Each curve is the RMS error of the five values, averaged over ${RUNS} runs of ${EPISODES} episodes (fixed random seeds, so the same α always gives the same curve).`);

    const x = linear(0, EPISODES, M.l, W - M.r), y = linear(0, 0.25, H - M.b, M.t);
    let cache = {};
    const curve = (m, a) => (cache[m + a] ||= learningCurve(m, a));

    function draw() {
      const c = colors(), am = +mcIn.value, at = +tdIn.value;
      ctrl.querySelector(".mc").textContent = am.toFixed(2);
      ctrl.querySelector(".td").textContent = at.toFixed(2);
      s.innerHTML = "";
      axes(s, x, y, { xDomain: [0, EPISODES], yDomain: [0, 0.25], xTicks: ticks(0, EPISODES, 25), yTicks: ticks(0, 0.25, 0.05), xLabel: "episodes", yLabel: "RMS error (avg of 100 runs)" });
      const cm = curve("mc", am), ct = curve("td", at);
      const path = (cv) => cv.map((v, i) => (i ? "L" : "M") + x(i).toFixed(1) + "," + y(Math.min(v, 0.25)).toFixed(1)).join("");
      el("path", { d: path(cm), fill: "none", stroke: c.data, "stroke-width": 2.5 }, s);
      el("path", { d: path(ct), fill: "none", stroke: c.loss, "stroke-width": 2.5 }, s);
      const lab = (cv, txt, col, dy) => el("text", { x: x(EPISODES) - 4, y: y(cv[EPISODES]) + dy, "text-anchor": "end", style: `fill: ${col}; font-weight: 600` }, s).textContent = txt;
      const mcAbove = cm[EPISODES] >= ct[EPISODES];
      lab(cm, `MC α=${am.toFixed(2)}`, c.data, mcAbove ? -8 : 16);
      lab(ct, `TD α=${at.toFixed(2)}`, c.loss, mcAbove ? 16 : -8);
      const at3 = (cv) => [10, 25, 100].map((e) => fmt(cv[e], 3)).join(" / ");
      out.textContent = `RMS error after 10 / 25 / 100 episodes\nMC: ${at3(cm)}\nTD: ${at3(ct)}`;
    }

    mcIn.addEventListener("input", draw);
    tdIn.addEventListener("input", draw);
    presets.addEventListener("click", (e) => {
      if (!e.target.dataset.p) return;
      const [a, b] = e.target.dataset.p.split(",");
      mcIn.value = a; tdIn.value = b; draw();
    });
    onThemeChange(draw);
    draw();
  }

  // ---------------------------------------------------------------- The cliff world (same as lesson 1)
  const MAP = ["......", "..#...", "......", "SCCCCG"];
  const ROWS = MAP.length, COLS = MAP[0].length;
  const ACTIONS = [[-1, 0], [0, 1], [1, 0], [0, -1]];   // up, right, down, left
  const ARROWS = ["↑", "→", "↓", "←"];
  const tile = (r, c) => MAP[r][c];
  const isTerminal = (r, c) => tile(r, c) === "G" || tile(r, c) === "C";
  const isFloor = (r, c) => tile(r, c) === "." || tile(r, c) === "S";
  const FLOORS = [];
  for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) if (isFloor(r, c)) FLOORS.push([r, c]);
  function move(r, c, a) {
    const nr = r + ACTIONS[a][0], nc = c + ACTIONS[a][1];
    if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS || tile(nr, nc) === "#") return [r, c];
    return [nr, nc];
  }
  const reward = (r, c) => (tile(r, c) === "G" ? 10 : tile(r, c) === "C" ? -20 : -1);
  function outcomes(r, c, a, slip) {
    return [[1 - slip, a], [slip / 2, (a + 1) % 4], [slip / 2, (a + 3) % 4]]
      .filter(([p]) => p > 0)
      .map(([p, d]) => { const [nr, nc] = move(r, c, d); return [p, nr, nc, reward(nr, nc), isTerminal(nr, nc)]; });
  }
  const POLICIES = {
    edge: ["↓↓→↓↓↓", "↓↓.↓↓↓", "→→→→→↓", "↑....."],
    safe: ["→→→→→↓", "↑↑.↑↑↓", "↑↑←↑↑↓", "↑....."],
  };
  const GAMMA = 0.95, SEED = 1;

  function cliffWidget(root) {
    const CELL = 56, PAD = 6, GW = COLS * CELL + 2 * PAD, GH = ROWS * CELL + 2 * PAD;
    const px = (c) => PAD + c * CELL, py = (r) => PAD + r * CELL;
    const board = svg(GW, GH);
    board.style.maxWidth = "520px";
    board.style.margin = "0 auto";
    root.appendChild(board);
    hint(root, "Big number: TD's estimate V(s). Small number: the true value from dynamic programming. Every episode starts in a random floor cell so that all cells get visited.");
    const pol = control(root, `Policy: <button data-p="edge">Edge of the cliff</button><button data-p="safe">Safe detour</button>
      <label><input type="checkbox"> slippery floor (slip 0.2)</label>`);
    const ctrl = control(root, `
      <label>step size <b>α</b> <input type="range" min="0.05" max="0.5" step="0.05" value="0.1"> <span class="val"></span></label>
      <button data-a="one">Play 1 episode</button><button data-a="100">+100 episodes</button><button data-a="1000">+1000 episodes</button><button data-a="reset">Reset</button>`);
    const slipIn = pol.querySelector("input"), alphaIn = ctrl.querySelector("input");
    const out = readout(root);

    let name = "edge", policy, Vdp, V, rng, episodes, trail = [], agent = null, timer = null, last = "";

    function evaluate(slip) {
      const Vt = MAP.map((row) => [...row].map(() => 0));
      for (let it = 0; it < 5000; it++) {
        let d = 0;
        for (const [r, c] of FLOORS) {
          let v = 0;
          for (const [p, nr, nc, rw, end] of outcomes(r, c, policy[r][c], slip)) v += p * (rw + (end ? 0 : GAMMA * Vt[nr][nc]));
          d = Math.max(d, Math.abs(v - Vt[r][c]));
          Vt[r][c] = v;
        }
        if (d < 1e-9) break;
      }
      return Vt;
    }

    function reset() {
      clearInterval(timer); timer = null;
      policy = POLICIES[name].map((row) => [...row].map((ch) => Math.max(0, ARROWS.indexOf(ch))));
      Vdp = evaluate(slip());
      V = MAP.map((row) => [...row].map(() => 0));
      rng = mulberry32(SEED); episodes = 0; trail = []; agent = null;
      last = "All estimates start at 0. Play some episodes and watch them move toward the DP values.";
      draw();
    }
    const slip = () => (slipIn.checked ? 0.2 : 0);

    // One TD(0) update for the step from (r, c); returns [nr, nc, done, δ]
    function tdStep(r, c, alpha) {
      const outs = outcomes(r, c, policy[r][c], slip());
      let u = rng(), pick = outs[outs.length - 1];
      for (const o of outs) if ((u -= o[0]) <= 0) { pick = o; break; }
      const [, nr, nc, rw, end] = pick;
      const delta = rw + (end ? 0 : GAMMA * V[nr][nc]) - V[r][c];
      V[r][c] += alpha * delta;
      return [nr, nc, end, delta, rw];
    }

    function startCell() { return FLOORS[Math.floor(rng() * FLOORS.length)]; }

    function runEpisodes(n) {
      clearInterval(timer); timer = null; trail = []; agent = null;
      const alpha = +alphaIn.value;
      for (let k = 0; k < n; k++) {
        let [r, c] = startCell(), end = false;
        for (let t = 0; t < 100 && !end; t++) [r, c, end] = tdStep(r, c, alpha);
        episodes++;
      }
      last = `Ran ${n} episode${n > 1 ? "s" : ""} (α = ${alphaIn.value}).`;
      draw();
    }

    function playOne() {
      clearInterval(timer);
      const alpha = +alphaIn.value;
      let [r, c] = startCell(), t = 0, end = false;
      trail = [[r, c]]; agent = [r, c];
      last = "New episode.";
      draw();
      timer = setInterval(() => {
        const [fr, fc] = [r, c];
        let delta, rw;
        [r, c, end, delta, rw] = tdStep(r, c, alpha);
        t++;
        trail.push([r, c]); agent = [r, c];
        last = `step ${t}: reward ${rw}, δ = ${fmt(delta, 2)}  →  V(row ${fr}, col ${fc}) = ${fmt(V[fr][fc], 2)}`;
        if (end || t >= 100) {
          clearInterval(timer); timer = null; episodes++;
          last += end ? (tile(r, c) === "G" ? "   reached the goal." : "   fell off the cliff.") : "   gave up after 100 steps.";
        }
        draw();
      }, 160);
    }

    function draw() {
      const col = colors();
      ctrl.querySelector(".val").textContent = (+alphaIn.value).toFixed(2);
      board.innerHTML = "";
      for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
          const t = tile(r, c), x0 = px(c), y0 = py(r);
          el("rect", { x: x0, y: y0, width: CELL, height: CELL, fill: col.bg, stroke: col.border }, board);
          if (t === "#") el("rect", { x: x0 + 1, y: y0 + 1, width: CELL - 2, height: CELL - 2, fill: col.muted, "fill-opacity": 0.55 }, board);
          if (t === "G" || t === "C") {
            el("rect", { x: x0 + 1, y: y0 + 1, width: CELL - 2, height: CELL - 2, fill: t === "G" ? col.opt : col.grad, "fill-opacity": t === "G" ? 0.45 : 0.6 }, board);
            el("text", { x: x0 + CELL / 2, y: y0 + CELL / 2 - 3, "text-anchor": "middle", style: textStyle }, board).textContent = t === "G" ? "GOAL" : "cliff";
            el("text", { x: x0 + CELL / 2, y: y0 + CELL / 2 + 12, "text-anchor": "middle" }, board).textContent = t === "G" ? "+10" : "−20";
          }
          if (!isFloor(r, c)) continue;
          const v = V[r][c];
          el("rect", { x: x0 + 1, y: y0 + 1, width: CELL - 2, height: CELL - 2, fill: v >= 0 ? col.opt : col.grad, "fill-opacity": Math.min(1, Math.abs(v) / 12) * 0.4 }, board);
          el("text", { x: x0 + 5, y: y0 + 13, style: "font-size: 10px" }, board).textContent = ARROWS[policy[r][c]];
          el("text", { x: x0 + CELL / 2, y: y0 + CELL / 2 + 5, "text-anchor": "middle", style: textStyle + "; font-size: 15px; font-weight: 600" }, board).textContent = fmt(v, 1);
          el("text", { x: x0 + CELL / 2, y: y0 + CELL - 6, "text-anchor": "middle", style: "font-size: 10px" }, board).textContent = "DP " + fmt(Vdp[r][c], 1);
        }
      }
      if (trail.length > 1) {
        const d = trail.map(([r, c], i) => (i ? "L" : "M") + (px(c) + CELL / 2) + "," + (py(r) + CELL / 2)).join("");
        el("path", { d, fill: "none", stroke: col.model, "stroke-width": 3, "stroke-opacity": 0.45, "stroke-linejoin": "round" }, board);
      }
      if (agent) el("circle", { cx: px(agent[1]) + CELL / 2, cy: py(agent[0]) + CELL / 2, r: 9, fill: col.model, stroke: col.bg, "stroke-width": 2 }, board);
      const rms = Math.sqrt(FLOORS.reduce((s, [r, c]) => s + (V[r][c] - Vdp[r][c]) ** 2, 0) / FLOORS.length);
      out.textContent = `episodes: ${episodes}   ·   V(start): TD ${fmt(V[3][0], 2)} vs DP ${fmt(Vdp[3][0], 2)}   ·   RMS error over all cells: ${fmt(rms, 2)}\n${last}`;
    }

    pol.addEventListener("click", (e) => { if (e.target.dataset.p) { name = e.target.dataset.p; reset(); } });
    slipIn.addEventListener("change", reset);
    alphaIn.addEventListener("input", reset);
    ctrl.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (a === "reset") reset();
      else if (a === "one") playOne();
      else if (a) runEpisodes(+a);
    });
    onThemeChange(draw);
    reset();
  }

  const mounts = {
    "viz-walk-mc": (n) => walkWidget(n, { mc: true, td: false }),
    "viz-walk": (n) => walkWidget(n, { mc: true, td: true }),
    "viz-race": raceWidget,
    "viz-cliff": cliffWidget,
  };
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
