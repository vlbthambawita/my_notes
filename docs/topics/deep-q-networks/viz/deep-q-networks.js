// Interactive widgets for the "Deep Q-Networks (DQN)" lesson.
// Three widgets, each mounted on a <div> with a matching id:
//   #viz-replay  – a replay buffer filling with CartPole transitions; sample a mini-batch in order or at random
//   #viz-target  – a tiny linear Q-function trained toward a live target or a frozen target (slider C)
//   #viz-overest – why max over noisy estimates is biased upwards, and how Double DQN fixes it
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

  // Width (in SVG units) for a full-width chart: the panel's own width, so text stays ~11px on phones too
  function fitWidth(root, max) {
    return Math.max(300, Math.min(max, Math.round(root.clientWidth - 30)));
  }

  // Standard normal random number (Box–Muller)
  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // ---------------------------------------------------------------- Widget 1: experience replay
  // CartPole physics (same as the Python code in the lesson), driven by a random policy.
  function cartPole() {
    let s, t, ep = 0;
    function reset() {
      s = [0, 0, 0, 0].map(() => (Math.random() - 0.5) * 0.1);
      t = 0; ep += 1;
    }
    function step(action) {
      let [x, xd, th, thd] = s;
      const force = action === 1 ? 10 : -10;
      const cos = Math.cos(th), sin = Math.sin(th);
      const temp = (force + 0.05 * thd * thd * sin) / 1.1;
      const thAcc = (9.8 * sin - cos * temp) / (0.5 * (4 / 3 - (0.1 * cos * cos) / 1.1));
      const xAcc = temp - (0.05 * thAcc * cos) / 1.1;
      x += 0.02 * xd; xd += 0.02 * xAcc; th += 0.02 * thd; thd += 0.02 * thAcc;
      s = [x, xd, th, thd]; t += 1;
      return Math.abs(x) > 2.4 || Math.abs(th) > (12 * Math.PI) / 180;
    }
    reset();
    // One transition: returns {ep, t, th, thd}; resets automatically when the pole falls
    return function next() {
      const rec = { ep, t, th: s[2], thd: s[3] };
      if (step(Math.random() < 0.5 ? 0 : 1)) reset();
      return rec;
    };
  }

  function replayWidget(root) {
    const CAP = 120, GC = 20, GR = CAP / GC, B = 12;
    const plots = document.createElement("div");
    plots.className = "viz-plots";
    root.appendChild(plots);
    const W1 = 340, H1 = 200, CELL = 15, X0 = (W1 - GC * CELL) / 2, Y0 = 48;
    const W2 = 340, H2 = 200, M = { l: 42, r: 10, t: 24, b: 34 };
    const grid = svg(W1, H1), phase = svg(W2, H2);
    plots.appendChild(grid); plots.appendChild(phase);
    hint(root, "Each square is one transition (s, a, r, s′). Shades alternate per episode. Orange = the mini-batch the network would learn from next.");
    const ctrl = control(root, `
      <button data-a="play">▶ Stream</button>
      <button data-a="seq">Sample 12 in a row</button>
      <button data-a="rand">Sample 12 at random</button>
      <button data-a="clear">Empty the buffer</button>`);
    const out = readout(root);
    const playBtn = ctrl.querySelector('[data-a="play"]');

    let env, buf, head, count, batch, timer = null;
    const x = linear(-0.24, 0.24, M.l, W2 - M.r), y = linear(-2.4, 2.4, H2 - M.b, M.t);

    function reset() {
      env = cartPole(); buf = new Array(CAP).fill(null); head = 0; count = 0; batch = [];
      for (let i = 0; i < 90; i++) push();
      draw();
    }

    function push() {
      buf[head] = env(); head = (head + 1) % CAP; count = Math.min(CAP, count + 1);
    }

    // Slots in age order (oldest first)
    const ordered = () => Array.from({ length: count }, (_, i) => (head - count + i + CAP) % CAP);

    function sample(kind) {
      const slots = ordered();
      if (slots.length < B) return;
      if (kind === "seq") {
        const start = slots.length - B;                       // the 12 newest, in order
        batch = slots.slice(start);
      } else {
        const pool = slots.slice();
        batch = [];
        for (let i = 0; i < B; i++) batch.push(pool.splice(Math.floor(Math.random() * pool.length), 1)[0]);
      }
      draw();
    }

    function draw() {
      const c = colors();
      grid.innerHTML = "";
      el("text", { x: X0, y: 24, style: textStyle }, grid).textContent = `Replay buffer (${count}/${CAP} slots)`;
      const inBatch = new Set(batch);
      for (let i = 0; i < CAP; i++) {
        const gx = X0 + (i % GC) * CELL, gy = Y0 + Math.floor(i / GC) * CELL;
        el("rect", { x: gx + 1, y: gy + 1, width: CELL - 2, height: CELL - 2, fill: c.bg, stroke: c.border }, grid);
        const rec = buf[i];
        if (rec) el("rect", { x: gx + 2, y: gy + 2, width: CELL - 4, height: CELL - 4, fill: c.data, "fill-opacity": rec.ep % 2 ? 0.85 : 0.35 }, grid);
        if (inBatch.has(i)) el("rect", { x: gx + 0.5, y: gy + 0.5, width: CELL - 1, height: CELL - 1, fill: "none", stroke: c.model, "stroke-width": 2.5 }, grid);
      }
      if (count) {
        const last = (head - 1 + CAP) % CAP;
        const gx = X0 + (last % GC) * CELL + CELL / 2, gy = Y0 + Math.floor(last / GC) * CELL;
        el("path", { d: `M${gx},${gy - 1} l-4,-6 h8 z`, fill: c.text }, grid);
      }
      el("text", { x: X0, y: Y0 + GR * CELL + 24 }, grid).textContent = "fills row by row; when full, the oldest";
      el("text", { x: X0, y: Y0 + GR * CELL + 39 }, grid).textContent = "transition is overwritten (▾ = newest)";

      phase.innerHTML = "";
      axes(phase, x, y, { xDomain: [-0.24, 0.24], yDomain: [-2.4, 2.4], xTicks: [-0.2, -0.1, 0, 0.1, 0.2], yTicks: [-2, -1, 0, 1, 2], xLabel: "pole angle θ (rad)", yLabel: "angular speed" });
      el("text", { x: M.l + 4, y: 14, style: textStyle }, phase).textContent = "States in the buffer and in the batch";
      for (const i of ordered()) {
        const r = buf[i];
        el("circle", { cx: x(r.th), cy: y(Math.max(-2.4, Math.min(2.4, r.thd))), r: 2.6, fill: c.data, "fill-opacity": 0.45 }, phase);
      }
      if (batch.length) {
        const age = new Map(ordered().map((slot, k) => [slot, k]));
        const inTime = batch.slice().sort((a, b) => age.get(a) - age.get(b));
        const pts = inTime.map((i) => [x(buf[i].th), y(Math.max(-2.4, Math.min(2.4, buf[i].thd)))]);
        el("path", { d: pts.map(([px, py], k) => (k ? "L" : "M") + px.toFixed(1) + "," + py.toFixed(1)).join(""), fill: "none", stroke: c.model, "stroke-width": 1.2, "stroke-opacity": 0.6 }, phase);
        for (const [px, py] of pts) el("circle", { cx: px, cy: py, r: 4.5, fill: c.model, stroke: c.bg, "stroke-width": 1 }, phase);
      }
      out.textContent = describe();
    }

    function describe() {
      if (!batch.length) return "Press a sample button to draw a mini-batch of 12 transitions.";
      const recs = batch.map((i) => buf[i]);
      const eps = new Set(recs.map((r) => r.ep)).size;
      const order = ordered(), ages = batch.map((i) => order.indexOf(i));
      const span = Math.max(...ages) - Math.min(...ages) + 1;
      return `this batch comes from ${eps} episode${eps > 1 ? "s" : ""}, spread over ${span} of the ${count} stored time steps\n` +
        "(orange line = the batch in time order)";
    }

    function toggle() {
      if (timer) { clearInterval(timer); timer = null; playBtn.textContent = "▶ Stream"; return; }
      playBtn.textContent = "⏸ Pause";
      timer = setInterval(() => { push(); batch = []; draw(); }, 90);
    }

    ctrl.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (a === "play") toggle();
      if (a === "seq" || a === "rand") { if (timer) toggle(); sample(a); }   // pause the stream so the batch stays visible
      if (a === "clear") { if (timer) toggle(); env = cartPole(); buf.fill(null); head = 0; count = 0; batch = []; draw(); }
    });
    onThemeChange(draw);
    reset();
  }

  // ---------------------------------------------------------------- Widget 2: live vs frozen target
  // Three states. A and B each give reward +1 and lead to C; C loops to itself with reward 0.
  // True values (γ = 0.9): Q(A) = Q(B) = 1, Q(C) = 0. The "network" is linear with 2 weights:
  // Q(A) = w1, Q(B) = 2·w1 + w2, Q(C) = 2·w1 + 2·w2. The buffer holds A, B, C in the ratio 3 : 3 : 1.
  // Each step is one gradient step on the average squared TD error over the buffer.
  function targetWidget(root) {
    const PHI = [[1, 0], [2, 1], [2, 2]], R = [1, 1, 0], D = [3 / 7, 3 / 7, 1 / 7], NEXT = 2, GAMMA = 0.9;
    const CS = [1, 2, 3, 5, 10, 20, 50, 100], STEPS = 300;
    let W = 680;
    const H = 300, M = { l: 40, r: 64, t: 26, b: 36 };
    const chart = svg(W, H);
    root.appendChild(chart);
    const ctrl = control(root, `
      <label>sync every <b>C</b> <input type="range" min="0" max="${CS.length - 1}" step="1" value="0"> <span class="cv"></span></label>
      <label>step size <b>α</b> <input type="range" min="0.05" max="0.5" step="0.05" value="0.3"> <span class="av"></span></label>`);
    const presets = control(root, `Try: <button data-c="0">live target (C = 1)</button><button data-c="2">C = 3</button><button data-c="4">C = 10</button><button data-c="6">C = 50</button>`);
    const [cIn, aIn] = ctrl.querySelectorAll("input");
    const out = readout(root);
    const YMIN = -2, YMAX = 4;
    const y = linear(YMIN, YMAX, H - M.b, M.t);
    let x;
    const q = (w, i) => PHI[i][0] * w[0] + PHI[i][1] * w[1];

    function simulate(C, alpha) {
      let w = [0, 0], wt = [0, 0];
      const traj = [[0, 0, 0]], syncs = [];
      for (let k = 0; k < STEPS; k++) {
        const target = R.map((r) => r + GAMMA * q(wt, NEXT));          // y = r + γ Q(s′; θ⁻)
        const g = [0, 0];
        for (let i = 0; i < 3; i++) {
          const err = target[i] - q(w, i);
          g[0] += D[i] * err * PHI[i][0]; g[1] += D[i] * err * PHI[i][1];
        }
        w = [w[0] + alpha * g[0], w[1] + alpha * g[1]];
        if ((k + 1) % C === 0) { wt = w.slice(); if (C > 1) syncs.push(k + 1); }
        traj.push([0, 1, 2].map((i) => q(w, i)));
      }
      return { traj, syncs };
    }

    function draw() {
      const c = colors(), C = CS[+cIn.value], alpha = +aIn.value;
      cIn.nextElementSibling.textContent = C === 1 ? "1 step (live)" : `${C} steps`;
      aIn.nextElementSibling.textContent = alpha.toFixed(2);
      const { traj, syncs } = simulate(C, alpha);
      W = fitWidth(root, 680);
      chart.setAttribute("viewBox", `0 0 ${W} ${H}`);
      x = linear(0, STEPS, M.l, W - M.r);
      chart.innerHTML = "";
      axes(chart, x, y, { xDomain: [0, STEPS], yDomain: [YMIN, YMAX], xTicks: ticks(0, STEPS, 50), yTicks: ticks(YMIN, YMAX, 1), xLabel: "gradient step" });
      if (syncs.length <= 60) for (const s of syncs) el("line", { x1: x(s), x2: x(s), y1: y(YMIN), y2: y(YMAX), stroke: c.model, "stroke-opacity": 0.25, "stroke-dasharray": "2 3" }, chart);
      for (const v of [1, 0]) el("line", { x1: x(0), x2: x(STEPS), y1: y(v), y2: y(v), stroke: c.opt, "stroke-width": 2, "stroke-dasharray": "6 4" }, chart);
      el("text", { x: x(STEPS) + 6, y: y(1) + 4, style: `fill: ${c.opt}` }, chart).textContent = "true A, B";
      el("text", { x: x(STEPS) + 6, y: y(0) + 4, style: `fill: ${c.opt}` }, chart).textContent = "true C";
      const lineCol = [c.data, c.loss, c.model], names = ["Q(A)", "Q(B)", "Q(C)"];
      const clip = (v) => Math.max(YMIN - 0.6, Math.min(YMAX + 0.6, v));
      // clip to the plot area so a diverging run doesn't paint over the labels
      const cid = "dqn-clip-" + Math.random().toString(36).slice(2);
      const defs = el("defs", {}, chart);
      el("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b }, el("clipPath", { id: cid }, defs));
      const g = el("g", { "clip-path": `url(#${cid})` }, chart);
      for (let i = 0; i < 3; i++) {
        const d = traj.map((v, k) => (k ? "L" : "M") + x(k).toFixed(1) + "," + y(clip(v[i])).toFixed(1)).join("");
        el("path", { d, fill: "none", stroke: lineCol[i], "stroke-width": 2.2 }, g);
      }
      names.forEach((n, i) => {
        el("rect", { x: M.l + 8 + i * 62, y: 8, width: 14, height: 4, fill: lineCol[i] }, chart);
        el("text", { x: M.l + 26 + i * 62, y: 14, style: textStyle }, chart).textContent = n;
      });
      if (C > 1 && syncs.length <= 60) el("text", { x: W - M.r, y: 14, "text-anchor": "end" }, chart).textContent = W < 520 ? "dotted = sync" : "orange dotted = target synced";
      const last = traj[traj.length - 1];
      const err = Math.max(Math.abs(last[0] - 1), Math.abs(last[1] - 1), Math.abs(last[2]));
      const big = Math.max(...last.map(Math.abs));
      const verdict = !Number.isFinite(big) || big > 1e3 ? "💥 diverged: the estimates blow up"
        : err < 0.01 ? "✅ converged to the true values" : err < 0.5 ? "≈ close, still settling" : "⚠️ not converging (yet)";
      const f = (v) => (Math.abs(v) >= 1e4 ? v.toExponential(1) : fmt(v, 2));
      out.textContent = `after ${STEPS} steps: Q(A) = ${f(last[0])}, Q(B) = ${f(last[1])}, Q(C) = ${f(last[2])}   →  ${verdict}`;
    }

    cIn.addEventListener("input", draw);
    aIn.addEventListener("input", draw);
    window.addEventListener("resize", draw);
    presets.addEventListener("click", (e) => {
      if (e.target.dataset.c === undefined) return;
      cIn.value = e.target.dataset.c;
      draw();
    });
    onThemeChange(draw);
    draw();
  }

  // ---------------------------------------------------------------- Widget 3: overestimation
  // Every action is truly worth 0. Our estimates are the truth plus noise.
  // Q-learning's target uses max_a Q̂(a); Double DQN picks the action with one estimate and scores it with another.
  function overestWidget(root) {
    let W = 680;
    const H = 270, M = { l: 40, r: 12, t: 50, b: 34 };
    const chart = svg(W, H);
    root.appendChild(chart);
    const ctrl = control(root, `
      <label>number of actions <input type="range" min="2" max="10" step="1" value="4"> <span class="nv"></span></label>
      <label>noise σ <input type="range" min="0" max="2" step="0.1" value="1"> <span class="sv"></span></label>`);
    const btns = control(root, `<button data-a="one">Draw new estimates</button><button data-a="many">Draw 1000 times</button>`);
    const [nIn, sIn] = ctrl.querySelectorAll("input");
    const out = readout(root);
    let est1 = [], est2 = [], stats;
    const y = linear(-3, 3, H - M.b, M.t);

    function resetStats() { stats = { n: 0, single: 0, dbl: 0 }; }

    function drawOnce() {
      const n = +nIn.value, s = +sIn.value;
      est1 = Array.from({ length: n }, () => s * randn());
      est2 = Array.from({ length: n }, () => s * randn());
      const a = est1.indexOf(Math.max(...est1));
      stats.n += 1; stats.single += est1[a]; stats.dbl += est2[a];
    }

    function draw() {
      const c = colors(), n = +nIn.value;
      nIn.nextElementSibling.textContent = n;
      sIn.nextElementSibling.textContent = (+sIn.value).toFixed(1);
      W = fitWidth(root, 680);
      chart.setAttribute("viewBox", `0 0 ${W} ${H}`);
      chart.innerHTML = "";
      const x = linear(-0.5, n - 0.5, M.l, W - M.r), bw = Math.min(26, (x(1) - x(0)) * 0.3);
      axes(chart, x, y, { xDomain: [-0.5, n - 0.5], yDomain: [-3, 3], xTicks: [], yTicks: ticks(-3, 3, 1) });
      for (let i = 0; i < n; i++) el("text", { x: x(i), y: H - M.b + 16, "text-anchor": "middle" }, chart).textContent = `a${i + 1}`;
      el("text", { x: M.l + 4, y: 16, style: textStyle }, chart).textContent = W < 520 ? "Estimated Q per action (truth = 0)" : "Estimated Q for each action (true value of every action = 0)";
      el("line", { x1: x(-0.5), x2: x(n - 0.5), y1: y(0), y2: y(0), stroke: c.opt, "stroke-width": 2, "stroke-dasharray": "6 4" }, chart);
      if (est1.length === n) {
        const a = est1.indexOf(Math.max(...est1));
        for (let i = 0; i < n; i++) {
          const v1 = Math.max(-3, Math.min(3, est1[i])), v2 = Math.max(-3, Math.min(3, est2[i]));
          el("rect", { x: x(i) - bw - 2, y: Math.min(y(v1), y(0)), width: bw, height: Math.abs(y(v1) - y(0)), fill: c.data, "fill-opacity": i === a ? 1 : 0.45 }, chart);
          el("rect", { x: x(i) + 2, y: Math.min(y(v2), y(0)), width: bw, height: Math.abs(y(v2) - y(0)), fill: c.loss, "fill-opacity": i === a ? 1 : 0.45 }, chart);
        }
        el("text", { x: x(a), y: y(Math.min(3, Math.max(est1[a], est2[a], 0))) - 6, "text-anchor": "middle", style: textStyle }, chart).textContent = "picked";
      }
      el("rect", { x: M.l + 4, y: 26, width: 12, height: 10, fill: c.data }, chart);
      el("text", { x: M.l + 20, y: 35 }, chart).textContent = "estimate 1 (net θ)";
      el("rect", { x: M.l + 140, y: 26, width: 12, height: 10, fill: c.loss }, chart);
      el("text", { x: M.l + 156, y: 35 }, chart).textContent = "estimate 2 (net θ⁻)";
      out.textContent = stats.n
        ? `average over ${stats.n} draws (truth: 0)\nDQN: max of estimate 1 = ${fmt(stats.single / stats.n, 3)}\nDouble DQN: estimate 2 of that action = ${fmt(stats.dbl / stats.n, 3)}`
        : "Press a button to draw noisy estimates.";
    }

    btns.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (a === "one") drawOnce();
      if (a === "many") for (let k = 0; k < 1000; k++) drawOnce();
      draw();
    });
    const changed = () => { resetStats(); est1 = []; est2 = []; draw(); };
    nIn.addEventListener("input", changed);
    window.addEventListener("resize", draw);
    sIn.addEventListener("input", changed);
    onThemeChange(draw);
    resetStats();
    draw();
  }

  const mounts = { "viz-replay": replayWidget, "viz-target": targetWidget, "viz-overest": overestWidget };
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
