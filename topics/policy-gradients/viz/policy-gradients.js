// Interactive widgets for the "Policy Gradients and REINFORCE" lesson.
// Four widgets, each mounted on a <div> with a matching id:
//   #viz-corridor – the short corridor: why the best policy can be random (drag P(right), see the value J)
//   #viz-softmax  – a softmax policy over 3 actions; sample, get a reward, watch a REINFORCE update
//   #viz-learn    – REINFORCE learning curves on the short corridor, with and without a baseline
//   #viz-variance – histograms of single-episode gradient estimates, with and without a baseline
// Everything random is seeded, so the same settings always give the same pictures.
document.addEventListener("DOMContentLoaded", function () {
  const { colors, onThemeChange, linear, el, svg, axes, ticks, fmt } = window.Viz;

  // CORE START (pure functions; the lesson's numbers were checked by running this block in Node)
  // Small seeded random-number generator (mulberry32)
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  // exp(x) from + − × ÷ only. Math.exp may differ in the last bit between browsers, and these
  // simulations are chaotic enough that one bit can change a run, so we compute it ourselves:
  // every browser then shows exactly the same numbers.
  function exp(x) {
    if (x > 700) return Infinity;
    if (x < -700) return 0;
    const k = Math.round(x / 0.6931471805599453), r = x - k * 0.6931471805599453;
    let term = 1, sum = 1;
    for (let n = 1; n < 20; n++) { term = (term * r) / n; sum += term; }
    for (let i = 0; i < k; i++) sum *= 2;
    for (let i = 0; i < -k; i++) sum /= 2;
    return sum;
  }
  const pow2 = (k) => { let v = 1; for (let i = 0; i < -k; i++) v /= 2; return v; };   // 2^k for k ≤ 0
  // Approximately standard-normal noise: the sum of 12 uniform numbers, minus 6
  function gauss(rnd) {
    let s = -6;
    for (let i = 0; i < 12; i++) s += rnd();
    return s;
  }
  function softmax(h) {
    const m = Math.max(...h), e = h.map((v) => exp(v - m)), z = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / z);
  }

  // The short corridor (Sutton & Barto, Example 13.1). States 0, 1, 2 and the goal 3; −1 per step.
  // In state 1 the actions are reversed. The agent cannot tell the states apart, so its policy is one
  // number: p = P(right). The value of the start state has a closed form:
  const corridorJ = (p) => -2 * (2 - p) / (p * (1 - p));
  const P_BEST = 2 - Math.SQRT2;                       // ≈ 0.586, where J is largest (≈ −11.66)
  function corridorNext(s, right) {
    if (s === 1) return right ? 0 : 2;                 // the reversed state
    if (right) return s + 1;
    return s === 2 ? 1 : 0;                            // left from the start bumps into the wall
  }
  // Play one episode; returns the list of actions (true = right) and the visited states
  function corridorEpisode(p, rnd, maxSteps) {
    const acts = [], states = [];
    let s = 0;
    while (s !== 3 && acts.length < maxSteps) {
      const right = rnd() < p;
      acts.push(right); states.push(s);
      s = corridorNext(s, right);
    }
    return { acts, states, done: s === 3 };
  }

  // One run of REINFORCE on the short corridor with a softmax policy over two preferences
  // (h_right, h_left), starting at P(right) = 0.05. With a baseline, b is a single learned number
  // (every state looks the same) that tracks the returns: b ← b + αw (G_t − b).
  function corridorRun(alpha, useBaseline, seed, episodes, maxSteps) {
    const rnd = rng(seed);
    let hR = 0, hL = 2.9444389791664403, b = 0;          // ln 19, so P(right) = 0.05
    const returns = new Float64Array(episodes);
    for (let e = 0; e < episodes; e++) {
      const p = 1 / (1 + exp(hL - hR));
      const { acts } = corridorEpisode(p, rnd, maxSteps);
      const T = acts.length;
      returns[e] = -T;
      let step = 0;
      for (let t = 0; t < T; t++) {
        const G = -(T - t);                            // return from step t: −1 for every step left
        let weight = G;
        if (useBaseline) { weight = G - b; b += (G - b) / 64; }
        step += alpha * weight * (acts[t] ? 1 - p : -p);   // ∂ ln π(a_t) / ∂ h_right
      }
      hR += step; hL -= step;                          // ∂ ln π / ∂ h_left is the negative
    }
    return returns;
  }

  // Single-episode estimates of ∂J/∂h_right at a fixed policy p, without and with the baseline b = J(p)
  function gradientSamples(p, n, seed) {
    const rnd = rng(seed), b = corridorJ(p), plain = [], based = [];
    for (let k = 0; k < n; k++) {
      const { acts } = corridorEpisode(p, rnd, 5000);
      const T = acts.length;
      let g0 = 0, g1 = 0;
      for (let t = 0; t < T; t++) {
        const G = -(T - t), d = acts[t] ? 1 - p : -p;
        g0 += G * d; g1 += (G - b) * d;
      }
      plain.push(g0); based.push(g1);
    }
    return { plain, based, truth: corridorTrueGrad(p) };
  }
  // dJ/dh_right = dJ/dp · dp/dh_right = J'(p) · p(1 − p)
  function corridorTrueGrad(p) {
    const e = 1e-6;
    return ((corridorJ(p + e) - corridorJ(p - e)) / (2 * e)) * p * (1 - p);
  }
  // A 3-armed bandit with a softmax policy: `steps` REINFORCE updates from equal preferences.
  // Rewards are the arm's mean plus Gaussian noise; the baseline is the average reward so far.
  const BANDIT_MEANS = [4, 6, 5];
  function banditStep(state, useBaseline, alpha) {
    const pi = softmax(state.h);
    let u = state.rnd(), a = 0;
    while (a < 2 && (u -= pi[a]) > 0) a++;
    const r = BANDIT_MEANS[a] + gauss(state.rnd);
    const b = useBaseline ? state.avgR : 0, weight = r - b;
    const dh = pi.map((p, i) => alpha * weight * ((i === a ? 1 : 0) - p));   // α (r − b) ∇ ln π(a)
    state.h = state.h.map((v, i) => Math.max(-8, Math.min(8, v + dh[i])));
    state.n += 1; state.avgR += (r - state.avgR) / state.n;
    return { pi, a, r, b, weight, dh };
  }
  const banditStart = (seed) => ({ h: [0, 0, 0], rnd: rng(seed), n: 0, avgR: 0 });
  const mean = (v) => v.reduce((a, b) => a + b, 0) / v.length;
  const std = (v) => { const m = mean(v); return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1)); };
  // CORE END

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
  function plots(parent, n, W, H) {
    const wrap = document.createElement("div");
    wrap.className = "viz-plots";
    parent.appendChild(wrap);
    return Array.from({ length: n }, () => { const s = svg(W, H); wrap.appendChild(s); return s; });
  }
  const signed = (v, d = 2) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);

  // ---------------------------------------------------------------- Widget 1: the short corridor
  function corridorWidget(root) {
    const [cor, plot] = plots(root, 2, 340, 230);
    const ctrl = control(root, `
      <label><span>P(right) = <b>p</b></span> <input type="range" min="0.02" max="0.98" step="0.01" value="0.05"> <span class="val"></span></label>
      <button data-a="play">Play an episode</button>`);
    const presets = control(root, `Try: <button data-p="0.05">ε-greedy, greedy = left</button><button data-p="0.95">ε-greedy, greedy = right</button><button data-p="0.59">p = 0.59</button>`);
    const slider = ctrl.querySelector("input"), val = ctrl.querySelector(".val");
    const out = readout(root);
    const CW = 70, CX = 30, CY = 70;
    let agent = 0, steps = null, timer = null, seed = 1;

    function drawCorridor() {
      const c = colors();
      cor.innerHTML = "";
      el("text", { x: 170, y: 22, "text-anchor": "middle", style: textStyle }, cor).textContent = "The agent can't tell the grey cells apart";
      const names = ["start", "reversed!", "", "GOAL"];
      for (let i = 0; i < 4; i++) {
        const x = CX + i * CW;
        el("rect", { x, y: CY, width: CW, height: 60, fill: i === 3 ? c.opt : c.muted, "fill-opacity": i === 3 ? 0.45 : 0.18, stroke: c.border }, cor);
        el("text", { x: x + CW / 2, y: CY + 76, "text-anchor": "middle", style: i === 1 ? `fill: ${c.grad}` : "" }, cor).textContent = names[i];
      }
      el("text", { x: CX + 1.5 * CW, y: CY + 22, "text-anchor": "middle", style: `fill: ${c.grad}; font-size: 10px` }, cor).textContent = "→ goes ←";
      el("text", { x: 170, y: 210, "text-anchor": "middle" }, cor).textContent = "every step: reward −1";
      if (agent !== null && agent < 4) el("circle", { cx: CX + agent * CW + CW / 2, cy: CY + 38, r: 12, fill: c.model, stroke: c.bg, "stroke-width": 2 }, cor);
      if (steps !== null) el("text", { x: 170, y: 48, "text-anchor": "middle", style: textStyle }, cor).textContent = `steps so far: ${steps}`;
    }

    const W = 340, H = 230, M = { l: 52, r: 12, t: 22, b: 38 };
    const x = linear(0, 1, M.l, W - M.r), y = linear(-100, 0, H - M.b, M.t);
    function drawPlot() {
      const c = colors(), p = +slider.value;
      plot.innerHTML = "";
      axes(plot, x, y, { xDomain: [0, 1], yDomain: [-100, 0], xTicks: ticks(0, 1, 0.2), yTicks: ticks(-100, 0, 20), xLabel: "p = P(right)", yLabel: "value J(p)" });
      let d = "";
      for (let i = 0; i <= 400; i++) {
        const q = 0.01 + (i / 400) * 0.98, v = corridorJ(q);
        if (v < -100) { continue; }
        d += (d ? "L" : "M") + x(q).toFixed(1) + "," + y(v).toFixed(1);
      }
      el("path", { d, fill: "none", stroke: c.loss, "stroke-width": 2.5 }, plot);
      el("circle", { cx: x(P_BEST), cy: y(corridorJ(P_BEST)), r: 5, fill: c.opt }, plot);
      el("text", { x: x(P_BEST), y: y(corridorJ(P_BEST)) - 10, "text-anchor": "middle", style: textStyle }, plot).textContent = `best: p = 0.59`;
      const v = corridorJ(p);
      if (v >= -100) el("circle", { cx: x(p), cy: y(v), r: 6, fill: c.model, stroke: c.bg, "stroke-width": 2 }, plot);
      else el("text", { x: x(p), y: y(-100) - 6, "text-anchor": "middle", style: `fill: ${c.model}` }, plot).textContent = "▼ off the chart";
      val.textContent = p.toFixed(2);
      out.textContent = `J(p) = average total reward from the start = ${fmt(v, 2)}   (best possible: ${fmt(corridorJ(P_BEST), 2)} at p = ${fmt(P_BEST, 3)})\n` +
        `p = 0 or p = 1 (a deterministic policy) never reaches the goal: J = −∞`;
    }

    function stop() { clearInterval(timer); timer = null; }
    function play() {
      stop();
      const ep = corridorEpisode(+slider.value, rng(seed++), 300);
      const path = ep.states.concat(ep.done ? [3] : []);
      let i = 0;
      timer = setInterval(() => {
        agent = path[i]; steps = i; drawCorridor();
        if (++i >= path.length) {
          stop();
          out.textContent += ep.done ? `\nthis episode: reached the goal after ${ep.acts.length} steps (return ${-ep.acts.length})` : `\nthis episode: still wandering after 300 steps, gave up`;
        }
      }, Math.max(25, 160 - path.length * 2));
    }
    function draw() { drawCorridor(); drawPlot(); }
    slider.addEventListener("input", () => { stop(); agent = 0; steps = null; draw(); });
    ctrl.addEventListener("click", (e) => { if (e.target.dataset.a === "play") play(); });
    presets.addEventListener("click", (e) => {
      if (!e.target.dataset.p) return;
      slider.value = e.target.dataset.p; stop(); agent = 0; steps = null; draw();
    });
    onThemeChange(draw);
    draw();
  }

  // ---------------------------------------------------------------- Widget 2: softmax policy playground
  function softmaxWidget(root) {
    const NAMES = ["A", "B", "C"], MEANS = BANDIT_MEANS;
    const [bars, hist] = plots(root, 2, 340, 230);
    const sliders = control(root, NAMES.map((n, i) =>
      `<label>h(${n}) <input type="range" min="-4" max="4" step="0.1" value="0" data-i="${i}"> <span class="h${i}"></span></label>`).join(""));
    const ctrl = control(root, `
      <label><span>step size <b>α</b></span> <input type="range" min="0.02" max="0.3" step="0.01" value="0.1" class="alpha"> <span class="aval"></span></label>
      <label><input type="checkbox" class="base"> subtract a baseline (average reward so far)</label>`);
    const btns = control(root, `<button data-a="one">Sample an action &amp; update</button><button data-a="many">Run 100 updates</button><button data-a="reset">New run</button><button data-a="test">Test 1000 runs</button>`);
    const hs = [...sliders.querySelectorAll("input")];
    const alphaIn = ctrl.querySelector(".alpha"), baseIn = ctrl.querySelector(".base");
    const out = readout(root);

    let st, prev, trace, ghost = null, last, run = 0, test = null;
    function reset(keepGhost) {
      if (keepGhost && trace && trace.length > 1) ghost = { trace, base: trace.base };
      run += 1; st = banditStart(run); prev = null; last = null; test = null;
      trace = [softmax(st.h)[1]]; trace.base = baseIn.checked;
      draw();
    }
    function update() {
      last = banditStep(st, baseIn.checked, +alphaIn.value);
      prev = last.pi;
      trace.push(softmax(st.h)[1]);
    }
    function testRuns() {
      const alpha = +alphaIn.value, res = {};
      for (const base of [false, true]) {
        let good = 0;
        for (let seed = 0; seed < 1000; seed++) {
          const s = banditStart(seed);
          for (let k = 0; k < 100; k++) banditStep(s, base, alpha);
          if (softmax(s.h)[1] > 0.5) good++;
        }
        res[base] = good / 10;
      }
      test = `1000 runs of 100 updates at α = ${alpha.toFixed(2)}: π(B) ends above 0.5 in ${res[false].toFixed(1)}% of runs without a baseline, ${res[true].toFixed(1)}% with one`;
    }
    function draw() {
      const c = colors(), pi = softmax(st.h);
      hs.forEach((s, i) => { s.value = st.h[i]; sliders.querySelector(`.h${i}`).textContent = signed(st.h[i], 1); });
      ctrl.querySelector(".aval").textContent = (+alphaIn.value).toFixed(2);

      // bar chart of π(a)
      const W = 340, H = 230, M = { l: 50, r: 12, t: 26, b: 44 };
      const x = linear(-0.5, 2.5, M.l, W - M.r), y = linear(0, 1, H - M.b, M.t), bw = 34;
      bars.innerHTML = "";
      axes(bars, x, y, { xDomain: [-0.5, 2.5], yDomain: [0, 1], xTicks: [], yTicks: ticks(0, 1, 0.25), yLabel: "probability π(a)" });
      pi.forEach((p, i) => {
        const hot = last && last.a === i;
        el("rect", { x: x(i) - bw, y: y(p), width: 2 * bw, height: y(0) - y(p), fill: c.model, "fill-opacity": hot ? 1 : 0.65 }, bars);
        if (prev) el("rect", { x: x(i) - bw, y: y(prev[i]), width: 2 * bw, height: y(0) - y(prev[i]), fill: "none", stroke: c.text, "stroke-dasharray": "4 3" }, bars);
        el("text", { x: x(i), y: y(Math.max(p, prev ? prev[i] : 0)) - 6, "text-anchor": "middle", style: textStyle }, bars).textContent = p.toFixed(2);
        el("text", { x: x(i), y: H - M.b + 16, "text-anchor": "middle", style: textStyle }, bars).textContent = `action ${NAMES[i]}`;
        el("text", { x: x(i), y: H - M.b + 31, "text-anchor": "middle" }, bars).textContent = `mean reward ${MEANS[i]}`;
      });
      el("text", { x: M.l + 4, y: 14, style: textStyle }, bars).textContent = prev ? "dashed = before the last update" : "softmax of the preferences h";

      // history of π(B)
      const N = Math.max(100, trace.length - 1, ghost ? ghost.trace.length - 1 : 0);
      const hx = linear(0, N, M.l, W - M.r);
      hist.innerHTML = "";
      axes(hist, hx, y, { xDomain: [0, N], yDomain: [0, 1], xTicks: ticks(0, N, N > 400 ? 200 : N > 200 ? 100 : 25), yTicks: ticks(0, 1, 0.25), xLabel: "number of updates", yLabel: "π(B), the best action" });
      const line = (tr, stroke, op) => el("path", { d: tr.map((v, i) => (i ? "L" : "M") + hx(i).toFixed(1) + "," + y(v).toFixed(1)).join(""), fill: "none", stroke, "stroke-width": 2, "stroke-opacity": op }, hist);
      if (ghost) line(ghost.trace, ghost.base ? c.opt : c.grad, 0.35);
      line(trace, baseIn.checked ? c.opt : c.grad, 1);
      el("text", { x: M.l + 4, y: 14, style: textStyle }, hist).textContent = `run ${run}: ` + (baseIn.checked ? "with baseline (green)" : "no baseline (red)") + (ghost ? ", faint = last run" : "");

      let txt = `run ${run}   updates: ${st.n}   average reward so far: ${fmt(st.avgR, 2)}`;
      if (last) {
        const nm = NAMES[last.a];
        txt += `\nsampled ${nm}, reward r = ${fmt(last.r, 2)}` + (baseIn.checked ? `, baseline b = ${fmt(last.b, 2)}` : "") +
          `  →  weight ${baseIn.checked ? "r − b" : "r"} = ${signed(last.weight)}` +
          `\nΔh = α · weight · (1[a] − π) = (${last.dh.map((v) => signed(v, 3)).join(", ")})` +
          `\nπ(${nm}) went ${last.dh[last.a] >= 0 ? "UP" : "DOWN"}, the other two went ${last.dh[last.a] >= 0 ? "down" : "up"}`;
      } else txt += "\nPress “Sample an action & update”.";
      if (test) txt += "\n" + test;
      out.textContent = txt;
    }
    hs.forEach((s) => s.addEventListener("input", () => { st.h[+s.dataset.i] = +s.value; prev = null; last = null; draw(); }));
    alphaIn.addEventListener("input", () => { test = null; draw(); });
    baseIn.addEventListener("change", () => { run -= 1; reset(true); });   // same seed, so the two runs compare fairly
    btns.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (a === "one") { update(); draw(); }
      if (a === "many") { for (let k = 0; k < 100; k++) update(); draw(); }
      if (a === "reset") reset(true);
      if (a === "test") { testRuns(); draw(); }
    });
    onThemeChange(draw);
    reset(false);
  }

  // ---------------------------------------------------------------- Widget 3: learning curves on the short corridor
  function learnWidget(root) {
    const RUNS = 50, EPISODES = 1000, MAXSTEPS = 1000, BIN = 20;
    const [curve, strip] = plots(root, 2, 340, 240);
    const ctrl = control(root, `
      <label><span>step size <b>α</b> = 2<sup class="k"></sup></span> <input type="range" min="-14" max="-6" step="1" value="-10"></label>`);
    const presets = control(root, `Try: <button data-k="-13">2<sup>−13</sup></button><button data-k="-10">2<sup>−10</sup></button><button data-k="-8">2<sup>−8</sup></button><button data-k="-7">2<sup>−7</sup></button>`);
    const slider = ctrl.querySelector("input");
    hint(root, `Each curve is the average of ${RUNS} runs of ${EPISODES} episodes (seeded). Episodes are cut off after ${MAXSTEPS} steps.`);
    const out = readout(root);
    const cache = {};

    function compute(k) {
      if (cache[k]) return cache[k];
      const res = {};
      for (const base of [false, true]) {
        const avg = new Float64Array(EPISODES / BIN), finals = [];
        for (let r = 0; r < RUNS; r++) {
          const ret = corridorRun(pow2(k), base, 1000 + r, EPISODES, MAXSTEPS);
          for (let e = 0; e < EPISODES; e++) avg[Math.floor(e / BIN)] += ret[e] / (RUNS * BIN);
          let s = 0; for (let e = EPISODES - 100; e < EPISODES; e++) s += ret[e];
          finals.push(s / 100);
        }
        res[base ? "base" : "plain"] = { avg, finals };
      }
      return (cache[k] = res);
    }

    function draw() {
      const c = colors(), k = +slider.value;
      ctrl.querySelector(".k").textContent = "−" + -k;
      const res = compute(k);
      const W = 340, H = 240, M = { l: 52, r: 12, t: 26, b: 38 }, LO = -100;
      const x = linear(0, EPISODES, M.l, W - M.r), y = linear(LO, 0, H - M.b, M.t);
      const clip = (v) => Math.max(LO, v);
      curve.innerHTML = "";
      axes(curve, x, y, { xDomain: [0, EPISODES], yDomain: [LO, 0], xTicks: ticks(0, EPISODES, 250), yTicks: ticks(LO, 0, 20), xLabel: "episode", yLabel: "return per episode" });
      el("line", { x1: x(0), x2: x(EPISODES), y1: y(corridorJ(P_BEST)), y2: y(corridorJ(P_BEST)), stroke: c.text, "stroke-dasharray": "5 4", "stroke-opacity": 0.6 }, curve);
      el("text", { x: x(EPISODES) - 2, y: y(corridorJ(P_BEST)) - 5, "text-anchor": "end" }, curve).textContent = "best possible −11.66";
      for (const [key, col] of [["plain", c.grad], ["base", c.opt]]) {
        const d = Array.from(res[key].avg, (v, i) => (i ? "L" : "M") + x((i + 0.5) * BIN).toFixed(1) + "," + y(clip(v)).toFixed(1)).join("");
        el("path", { d, fill: "none", stroke: col, "stroke-width": 2.2 }, curve);
      }
      el("text", { x: M.l + 4, y: 14, style: `fill: ${c.grad}` }, curve).textContent = "REINFORCE";
      el("text", { x: M.l + 84, y: 14, style: `fill: ${c.opt}` }, curve).textContent = "with baseline";

      // strip plot: final performance of every run
      const sx = linear(-0.5, 1.5, M.l, W - M.r);
      strip.innerHTML = "";
      axes(strip, sx, y, { xDomain: [-0.5, 1.5], yDomain: [LO, 0], xTicks: [], yTicks: ticks(LO, 0, 20), yLabel: "last 100 episodes, per run" });
      el("line", { x1: sx(-0.5), x2: sx(1.5), y1: y(corridorJ(P_BEST)), y2: y(corridorJ(P_BEST)), stroke: c.text, "stroke-dasharray": "5 4", "stroke-opacity": 0.6 }, strip);
      const summary = [];
      [["plain", c.grad, "REINFORCE"], ["base", c.opt, "with baseline"]].forEach(([key, col, name], j) => {
        const f = res[key].finals, jit = rng(7);
        f.forEach((v) => el("circle", { cx: sx(j) + (jit() - 0.5) * 60, cy: y(clip(v)), r: 3.2, fill: col, "fill-opacity": 0.7 }, strip));
        el("text", { x: sx(j), y: H - M.b + 16, "text-anchor": "middle", style: `fill: ${col}` }, strip).textContent = name;
        const stuck = f.filter((v) => v < -100).length;
        summary.push(`${name.padEnd(14)} last 100 episodes: average ${fmt(mean(f), 1).padStart(6)}   worst run ${fmt(Math.min(...f), 1).padStart(7)}   runs off the chart (< −100): ${stuck}/${RUNS}`);
      });
      el("text", { x: M.l + 4, y: 14, style: textStyle }, strip).textContent = "one dot per run";
      out.textContent = summary.join("\n");
    }
    slider.addEventListener("input", draw);
    presets.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      slider.value = b.dataset.k; draw();
    });
    onThemeChange(draw);
    draw();
  }

  // ---------------------------------------------------------------- Widget 4: variance of the gradient estimate
  function varianceWidget(root) {
    const N = 2000;
    const [left, right] = plots(root, 2, 340, 220);
    const ctrl = control(root, `
      <label><span>policy P(right) = <b>p</b></span> <input type="range" min="0.1" max="0.9" step="0.05" value="0.3"> <span class="val"></span></label>`);
    const slider = ctrl.querySelector("input");
    hint(root, `${N} episodes, each giving one estimate of the gradient. The purple line is the true gradient (from the formula for J).`);
    const out = readout(root);

    function histogram(root, data, lo, hi, col, title, truth, ymax) {
      const c = colors(), W = 340, H = 220, M = { l: 40, r: 12, t: 26, b: 38 }, NB = 40;
      const x = linear(lo, hi, M.l, W - M.r);
      const counts = new Array(NB).fill(0);
      let outside = 0;
      for (const v of data) {
        const i = Math.floor(((v - lo) / (hi - lo)) * NB);
        if (i < 0 || i >= NB) outside++; else counts[i]++;
      }
      const y = linear(0, ymax, H - M.b, M.t), step = (hi - lo) / NB;
      root.innerHTML = "";
      axes(root, x, y, { xDomain: [lo, hi], yDomain: [0, ymax], xTicks: ticks(lo, hi, [10, 20, 25, 50, 100, 200, 250, 500, 1000].find((t) => (hi - lo) / t <= 6) || 2000), yTicks: [], xLabel: "estimate of ∂J/∂h(right)" });
      counts.forEach((n, i) => el("rect", { x: x(lo + i * step) + 0.5, y: y(Math.min(n, ymax)), width: Math.max(0, x(lo + step) - x(lo) - 1), height: y(0) - y(Math.min(n, ymax)), fill: col, "fill-opacity": 0.8 }, root));
      el("line", { x1: x(truth), x2: x(truth), y1: y(0), y2: M.t, stroke: c.loss, "stroke-width": 2.5 }, root);
      el("text", { x: M.l + 4, y: 14, style: `fill: ${col}` }, root).textContent = title;
      el("text", { x: W - M.r, y: 14, "text-anchor": "end" }, root).textContent = outside ? `${outside} off the chart` : "";
      return counts;
    }

    function draw() {
      const c = colors(), p = +slider.value;
      ctrl.querySelector(".val").textContent = p.toFixed(2);
      const g = gradientSamples(p, N, 11);
      const m0 = mean(g.plain), s0 = std(g.plain), m1 = mean(g.based), s1 = std(g.based);
      const lo = g.truth - 3 * s0, hi = g.truth + 3 * s0;
      const span = hi - lo, nice = span > 400 ? 100 : span > 200 ? 50 : span > 100 ? 25 : 10;
      const L = Math.floor(lo / nice) * nice, R = Math.ceil(hi / nice) * nice;
      // shared y scale: the tallest bar of either histogram
      const peak = (data) => { const cs = new Array(40).fill(0); data.forEach((v) => { const i = Math.floor(((v - L) / (R - L)) * 40); if (i >= 0 && i < 40) cs[i]++; }); return Math.max(...cs); };
      const ymax = Math.max(peak(g.plain), peak(g.based)) * 1.05;
      histogram(left, g.plain, L, R, c.grad, "weight = return G (no baseline)", g.truth, ymax);
      histogram(right, g.based, L, R, c.opt, `weight = G − b, with b = ${fmt(corridorJ(p), 1)}`, g.truth, ymax);
      out.textContent = `true gradient: ${fmt(g.truth, 2)}\n` +
        `no baseline:   average ${fmt(m0, 2).padStart(7)}   spread (std) ${fmt(s0, 1)}\n` +
        `with baseline: average ${fmt(m1, 2).padStart(7)}   spread (std) ${fmt(s1, 1)}   (${Math.round(100 * (1 - s1 / s0))}% less spread)`;
    }
    slider.addEventListener("input", draw);
    onThemeChange(draw);
    draw();
  }

  const mounts = { "viz-corridor": corridorWidget, "viz-softmax": softmaxWidget, "viz-learn": learnWidget, "viz-variance": varianceWidget };
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
