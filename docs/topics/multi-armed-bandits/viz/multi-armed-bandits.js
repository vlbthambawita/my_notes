// Interactive widgets for the "Multi-armed bandits" lesson.
// Widgets, each mounted on a <div> with a matching id:
//   #viz-play          – play a 5-armed bandit yourself, then reveal the true values
//   #viz-testbed-eps   – 10-armed testbed: greedy vs ε-greedy
//   #viz-testbed-opt   – 10-armed testbed: optimistic initial values
//   #viz-ucb           – watch UCB pick arms: estimates with confidence bonuses that shrink
//   #viz-testbed-all   – 10-armed testbed: every method, measured by regret
// Runs after DOMContentLoaded so the site-wide helpers (assets/js/viz-common.js) are available.
document.addEventListener("DOMContentLoaded", function () {
  const { colors, onThemeChange, linear, el, svg, axes, ticks, fmt } = window.Viz;
  const textStyle = "fill: var(--viz-text)";

  // ---------------------------------------------------------------- The 10-armed testbed simulation
  // Seeded random numbers, so every reader sees the same curves (and so they can be checked in Python).
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = Math.imul(a ^ (a >>> 15), a | 1);
      t = (t + Math.imul(t ^ (t >>> 7), t | 61)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const runSeed = (seed, run) => (Math.imul(run + 1, 0x9e3779b1) ^ seed) >>> 0;
  // Standard normal from two uniforms (Box–Muller)
  const gauss = (rand) => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());

  // Index of the largest value; ties broken at random with the uniform number u
  function argmaxTie(vals, u) {
    let best = -Infinity, count = 0;
    for (const v of vals) { if (v > best) { best = v; count = 1; } else if (v === best) count++; }
    let pick = Math.floor(u * count);
    for (let i = 0; i < vals.length; i++) if (vals[i] === best && pick-- === 0) return i;
    return 0;
  }

  // Play one method on runs run0 … run1−1 (each a fresh random 10-armed bandit) for `steps` pulls,
  // adding per-step totals into acc = {reward, optimal, regret}.
  // method: {kind: "eps" | "ucb" | "ts", eps, q0, alpha (missing = sample average 1/n), c}
  function simulate(method, run0, run1, steps, seed, acc, k = 10) {
    const Q = new Float64Array(k), N = new Float64Array(k), S = new Float64Array(k), vals = new Float64Array(k);
    for (let run = run0; run < run1; run++) {
      const prob = mulberry32(runSeed(seed, run));          // the bandit problem: true values q*(a)
      const qstar = Array.from({ length: k }, () => gauss(prob));
      const best = argmaxTie(qstar, 0), qbest = qstar[best];
      const rand = mulberry32(runSeed(seed + 1, run));      // the agent's choices and the reward noise
      Q.fill(method.q0 || 0); N.fill(0); S.fill(0);
      let lost = 0;
      for (let t = 0; t < steps; t++) {
        let a;
        if (method.kind === "eps") {
          const explore = rand() < method.eps, uArm = rand(), uTie = rand();
          a = explore ? Math.floor(uArm * k) : argmaxTie(Q, uTie);
        } else if (method.kind === "ucb") {
          for (let i = 0; i < k; i++) vals[i] = N[i] === 0 ? Infinity : Q[i] + method.c * Math.sqrt(Math.log(t + 1) / N[i]);
          a = argmaxTie(vals, rand());
        } else {                                              // Thompson sampling, prior N(0, 1), noise variance 1
          for (let i = 0; i < k; i++) vals[i] = S[i] / (N[i] + 1) + gauss(rand) / Math.sqrt(N[i] + 1);
          a = argmaxTie(vals, 0);
        }
        const r = qstar[a] + gauss(rand);
        N[a] += 1; S[a] += r;
        Q[a] += (method.alpha || 1 / N[a]) * (r - Q[a]);     // new estimate = old + step × error
        lost += qbest - qstar[a];
        acc.reward[t] += r; acc.optimal[t] += a === best ? 1 : 0; acc.regret[t] += lost;
      }
    }
  }

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

  // ---------------------------------------------------------------- Widget 1: play a 5-armed bandit
  function playWidget(root) {
    const K = 5, BUDGET = 25;
    const W = root.clientWidth < 480 ? 330 : 440, H = 280, M = { l: 34, r: 8, t: 26, b: 52 };
    const chart = svg(W, H);
    chart.style.maxWidth = "560px";
    chart.style.margin = "0 auto";
    chart.style.cursor = "pointer";
    root.appendChild(chart);
    hint(root, `Click a machine (or a button) to pull its arm. Each pull pays a random reward around that arm's hidden value. You have ${BUDGET} pulls: try to collect as much as you can.`);
    const ctrl = control(root, Array.from({ length: K }, (_, i) => `<button data-a="${i}">Pull ${i + 1}</button>`).join("") +
      `<button data-a="reveal">Reveal true values</button><button data-a="new">New game</button>`);
    const out = readout(root);

    const x = linear(0, K, M.l, W - M.r), y = linear(-4, 4, H - M.b, M.t);
    const colW = x(1) - x(0);
    let qstar, rewards, revealed, total, regret;

    function newGame() {
      qstar = Array.from({ length: K }, () => gauss(Math.random));
      rewards = Array.from({ length: K }, () => []);
      revealed = false; total = 0; regret = 0;
      draw();
    }

    const pulls = () => rewards.reduce((s, r) => s + r.length, 0);
    const mean = (r) => r.reduce((s, v) => s + v, 0) / r.length;

    function pull(a) {
      if (pulls() >= BUDGET) return;
      const r = qstar[a] + gauss(Math.random);
      rewards[a].push(r);
      total += r; regret += Math.max(...qstar) - qstar[a];
      if (pulls() >= BUDGET) revealed = true;
      draw();
    }

    function draw() {
      const c = colors();
      chart.innerHTML = "";
      axes(chart, x, y, { xDomain: [0, K], yDomain: [-4, 4], xTicks: [], yTicks: ticks(-4, 4, 2) });
      el("text", { x: M.l, y: 14 }, chart).textContent = "reward";
      el("line", { x1: x(0), x2: x(K), y1: y(0), y2: y(0), stroke: c.muted, "stroke-dasharray": "3 3" }, chart);
      const best = qstar.indexOf(Math.max(...qstar));
      for (let a = 0; a < K; a++) {
        const cx = x(a + 0.5), rs = rewards[a];
        // column background: click target
        el("rect", { x: x(a) + 3, y: M.t, width: colW - 6, height: y(-4) - M.t, fill: c.data, "fill-opacity": a % 2 ? 0.05 : 0.02, rx: 6 }, chart);
        rs.forEach((r, i) => {
          const jitter = ((i * 37) % 11 - 5) * (colW / 40);
          el("circle", { cx: cx + jitter, cy: y(Math.max(-4, Math.min(4, r))), r: 3.2, fill: c.data, "fill-opacity": 0.6 }, chart);
        });
        if (rs.length) {
          const q = mean(rs);
          el("line", { x1: cx - colW * 0.32, x2: cx + colW * 0.32, y1: y(q), y2: y(q), stroke: c.model, "stroke-width": 3.5, "stroke-linecap": "round" }, chart);
          el("text", { x: cx + colW * 0.36, y: y(q) + 4, style: "fill: var(--viz-model); font-weight: 700" }, chart).textContent = "Q";
        }
        if (revealed) {
          el("line", { x1: cx - colW * 0.42, x2: cx + colW * 0.42, y1: y(qstar[a]), y2: y(qstar[a]), stroke: c.opt, "stroke-width": 2, "stroke-dasharray": "5 3" }, chart);
        }
        el("text", { x: cx, y: H - M.b + 16, "text-anchor": "middle", style: textStyle + (revealed && a === best ? "; font-weight: 700" : "") }, chart).textContent = `arm ${a + 1}${revealed && a === best ? " ★" : ""}`;
        el("text", { x: cx, y: H - M.b + 30, "text-anchor": "middle" }, chart).textContent = rs.length ? `Q = ${fmt(mean(rs), 2)}` : "Q = ?";
        el("text", { x: cx, y: H - M.b + 44, "text-anchor": "middle" }, chart).textContent = revealed ? `q* = ${fmt(qstar[a], 2)}` : `pulled ${rs.length}×`;
      }
      const n = pulls();
      let text = `pulls ${n} / ${BUDGET}   total reward ${fmt(total, 1)}   average per pull ${n ? fmt(total / n, 2) : "–"}`;
      text += "\nblue dots = rewards you got · orange bar = your estimate Q (the average of the dots)";
      if (revealed) {
        text += `\ngreen dashes = true values q*. The best arm was arm ${best + 1} (q* = ${fmt(qstar[best], 2)}); you pulled it ${rewards[best].length} of ${n} times.`;
        text += `\nregret = expected reward lost by not always pulling arm ${best + 1} = ${fmt(regret, 1)}`;
      }
      out.textContent = text;
    }

    chart.addEventListener("click", (e) => {
      const b = chart.getBoundingClientRect();
      const a = Math.floor(x.invert(((e.clientX - b.left) / b.width) * W));
      if (a >= 0 && a < K) pull(a);
    });
    ctrl.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (a === "new") newGame();
      else if (a === "reveal") { revealed = true; draw(); }
      else if (a !== undefined) pull(+a);
    });
    onThemeChange(draw);
    newGame();
  }

  // ---------------------------------------------------------------- Widgets 2–4: the 10-armed testbed
  const STEPS = 1000;
  const cache = new Map();                    // results shared between the testbed widgets on this page

  // Line styles per method: colour role + optional dash, the same in every widget
  const METHOD_STYLE = {
    greedy: { role: "grad" }, eps001: { role: "data" }, eps: { role: "model" },
    opt: { role: "loss" }, ucb: { role: "opt" }, ts: { role: "text", dash: "6 4" },
  };
  const METRICS = {
    reward: { label: "average reward", y: [-0.5, 1.75], step: 0.5, pick: (r) => r.reward, end: (v) => fmt(v, 2) },
    optimal: { label: "% optimal action", y: [0, 100], step: 20, pick: (r) => r.optimal, end: (v) => fmt(v, 0) + "%" },
    regret: { label: "total regret so far", y: null, step: null, pick: (r) => r.regret, end: (v) => fmt(v, 0) },
  };

  // Run (or fetch from cache) one method, in chunks so the page stays responsive
  function runMethod(spec, runs, seed, onDone) {
    const key = JSON.stringify([spec, runs, seed]);
    if (cache.has(key)) { onDone(cache.get(key)); return () => {}; }
    const acc = { reward: new Float64Array(STEPS), optimal: new Float64Array(STEPS), regret: new Float64Array(STEPS) };
    let run = 0, cancelled = false;
    const CHUNK = 100;
    (function next() {
      if (cancelled) return;
      simulate(spec, run, Math.min(runs, run + CHUNK), STEPS, seed, acc);
      run += CHUNK;
      if (run < runs) { setTimeout(next, 0); return; }
      for (let t = 0; t < STEPS; t++) { acc.reward[t] /= runs; acc.optimal[t] *= 100 / runs; acc.regret[t] /= runs; }
      cache.set(key, acc);
      onDone(acc);
    })();
    return () => { cancelled = true; };
  }

  // config: { methods(values) → [{id, label, spec}], metrics: [name, name], controls: html, read(ctrl) → values }
  function testbedWidget(root, config) {
    const legend = document.createElement("div");
    legend.className = "viz-controls";
    legend.style.marginTop = "0";
    legend.style.marginBottom = "8px";
    root.appendChild(legend);
    const plots = document.createElement("div");
    plots.className = "viz-plots";
    root.appendChild(plots);
    const W = 340, H = 250, M = { l: 40, r: 12, t: 24, b: 36 };
    const charts = config.metrics.map(() => { const s = svg(W, H); plots.appendChild(s); return s; });
    const ctrl = control(root, config.controls);
    const out = readout(root);
    hint(root, "Each curve is an average over many random 10-armed bandits (2000 by default). Click a legend entry to hide or show a curve.");

    let methods = [], results = {}, hidden = new Set(), cancel = [], status = "";

    function legendHtml() {
      legend.innerHTML = methods.map((m) => {
        const st = METHOD_STYLE[m.id];
        const off = hidden.has(m.id);
        return `<button data-m="${m.id}" style="opacity:${off ? 0.4 : 1}"><span style="display:inline-block;width:18px;height:0;border-top:3px ${st.dash ? "dashed" : "solid"} var(--viz-${st.role});vertical-align:middle;margin-right:6px"></span>${m.label}</button>`;
      }).join("");
    }

    function rerun() {
      cancel.forEach((f) => f());
      const v = config.read(ctrl);
      methods = config.methods(v);
      results = {};
      legendHtml();
      let done = 0;
      status = `computing… 0 / ${methods.length} methods`;
      draw();
      methods.forEach((m, i) => {
        // queue the methods one after another so the first curves appear quickly
        setTimeout(() => {
          cancel.push(runMethod(m.spec, v.runs || 2000, v.seed || 1, (res) => {
            results[m.id] = res;
            done++;
            status = done < methods.length ? `computing… ${done} / ${methods.length} methods` : "";
            draw();
          }));
        }, i * 5);
      });
    }

    function draw() {
      const c = colors();
      config.metrics.forEach((name, ci) => {
        const metric = METRICS[name], root = charts[ci];
        root.innerHTML = "";
        let [y0, y1] = metric.y || [0, 0], step = metric.step;
        if (!metric.y) {                                  // regret: scale to the largest visible curve
          for (const m of methods) if (results[m.id] && !hidden.has(m.id)) y1 = Math.max(y1, results[m.id].regret[STEPS - 1]);
          y1 = Math.max(100, Math.ceil(y1 / 100) * 100);
          step = y1 <= 400 ? 100 : y1 <= 800 ? 200 : 250 * Math.ceil(y1 / 1250);
        }
        const x = linear(0, STEPS, M.l, W - M.r), y = linear(y0, y1, H - M.b, M.t);
        axes(root, x, y, { xDomain: [0, STEPS], yDomain: [y0, y1], xTicks: ticks(0, STEPS, 250), yTicks: ticks(name === "reward" ? 0 : y0, y1, step), xLabel: "steps (pulls)" });
        el("text", { x: M.l, y: 15, style: textStyle + "; font-size: 12px" }, root).textContent = metric.label;
        for (const m of methods) {
          const res = results[m.id];
          if (!res || hidden.has(m.id)) continue;
          const st = METHOD_STYLE[m.id], vals = metric.pick(res);
          // smooth the noisy per-step averages a little (window of 10 steps) for readability
          // (but not during the first 50 steps, where the interesting spikes are)
          let d = "";
          for (let t = 0; t < STEPS; t += t < 50 ? 1 : 2) {
            let s = 0, n = 0;
            const w = t < 50 || name === "regret" ? 0 : 5;
            for (let j = Math.max(0, t - w); j <= Math.min(STEPS - 1, t + w); j++) { s += vals[j]; n++; }
            const v = s / n;
            d += (t ? "L" : "M") + x(t + 1).toFixed(1) + "," + y(Math.max(y0, Math.min(y1, v))).toFixed(1);
          }
          el("path", { d, fill: "none", stroke: c[st.role], "stroke-width": 1.8, "stroke-dasharray": st.dash || "none", "stroke-linejoin": "round" }, root);
        }
        if (status) el("text", { x: (M.l + W - M.r) / 2, y: (M.t + H - M.b) / 2, "text-anchor": "middle", style: textStyle }, root).textContent = status;
      });
      const lines = methods.filter((m) => results[m.id]).map((m) => {
        const r = results[m.id];
        let s = 0, o = 0;
        for (let t = STEPS - 100; t < STEPS; t++) { s += r.reward[t]; o += r.optimal[t]; }
        return `${m.label}:  reward ${fmt(s / 100, 3)} · optimal ${fmt(o / 100, 1)}% · regret ${fmt(r.regret[STEPS - 1], 1)}`;
      });
      out.textContent = lines.length ? "last 100 steps: average reward and % optimal action · total regret after 1000 pulls\n" + lines.join("\n") : status;
    }

    legend.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      const id = b.dataset.m;
      hidden.has(id) ? hidden.delete(id) : hidden.add(id);
      legendHtml();
      draw();
    });
    ctrl.querySelectorAll("input[type=range]").forEach((inp) => {
      const show = () => { const s = inp.parentElement.querySelector(".val"); if (s) s.textContent = inp.value; };
      inp.addEventListener("input", show);
      inp.addEventListener("change", rerun);
      show();
    });
    ctrl.querySelectorAll("select, input[type=number]").forEach((inp) => inp.addEventListener("change", rerun));
    ctrl.addEventListener("click", (e) => { if (e.target.dataset.a === "rerun") rerun(); });
    onThemeChange(draw);

    // Start computing only when the widget scrolls near the screen
    status = "scroll here to run the simulation";
    methods = config.methods(config.read(ctrl));
    legendHtml();
    draw();
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((entries) => {
        if (entries.some((e) => e.isIntersecting)) { io.disconnect(); rerun(); }
      }, { rootMargin: "300px" });
      io.observe(root);
    } else rerun();
  }

  const FIELD = "font: inherit; padding: 2px 6px; border-radius: 6px; border: 1px solid var(--viz-border); background: var(--viz-bg); color: var(--viz-text)";
  const val = (ctrl, name) => +ctrl.querySelector(`[name=${name}]`).value;
  const TESTBEDS = {
    "viz-testbed-eps": {
      metrics: ["reward", "optimal"],
      controls: `<label>your <b>ε</b> <input type="range" name="eps" min="0" max="0.5" step="0.01" value="0.1"> <span class="val"></span></label>`,
      read: (ctrl) => ({ eps: val(ctrl, "eps") }),
      methods: (v) => [
        { id: "greedy", label: "greedy (ε = 0)", spec: { kind: "eps", eps: 0 } },
        { id: "eps001", label: "ε = 0.01", spec: { kind: "eps", eps: 0.01 } },
        { id: "eps", label: `ε = ${v.eps}`, spec: { kind: "eps", eps: v.eps } },
      ],
    },
    "viz-testbed-opt": {
      metrics: ["reward", "optimal"],
      controls: `<label>optimistic start <b>Q<sub>1</sub></b> <input type="range" name="q0" min="0" max="10" step="0.5" value="5"> <span class="val"></span></label>`,
      read: (ctrl) => ({ q0: val(ctrl, "q0") }),
      methods: (v) => [
        { id: "greedy", label: "greedy, Q₁ = 0", spec: { kind: "eps", eps: 0 } },
        { id: "eps", label: "ε = 0.1, Q₁ = 0", spec: { kind: "eps", eps: 0.1 } },
        { id: "opt", label: `greedy, Q₁ = ${v.q0}`, spec: { kind: "eps", eps: 0, q0: v.q0, alpha: 0.1 } },
      ],
    },
    "viz-testbed-all": {
      metrics: ["optimal", "regret"],
      controls: `<label><b>ε</b> <input type="range" name="eps" min="0" max="0.5" step="0.01" value="0.1"> <span class="val"></span></label>
        <label>UCB <b>c</b> <input type="range" name="c" min="0" max="4" step="0.1" value="2"> <span class="val"></span></label>
        <label>runs <select name="runs" style="${FIELD}"><option>200</option><option>500</option><option selected>2000</option></select></label>
        <label>seed <input type="number" name="seed" value="1" min="1" max="999" style="width:4.5em; ${FIELD}"></label>
        <button data-a="rerun">Rerun</button>`,
      read: (ctrl) => ({ eps: val(ctrl, "eps"), c: val(ctrl, "c"), runs: val(ctrl, "runs"), seed: Math.max(1, Math.round(val(ctrl, "seed")) || 1) }),
      methods: (v) => [
        { id: "greedy", label: "greedy", spec: { kind: "eps", eps: 0 } },
        { id: "eps", label: `ε-greedy ${v.eps}`, spec: { kind: "eps", eps: v.eps } },
        { id: "opt", label: "optimistic Q₁ = 5", spec: { kind: "eps", eps: 0, q0: 5, alpha: 0.1 } },
        { id: "ucb", label: `UCB c = ${v.c}`, spec: { kind: "ucb", c: v.c } },
        { id: "ts", label: "Thompson", spec: { kind: "ts" } },
      ],
    },
  };

  // ---------------------------------------------------------------- Widget 5: UCB, step by step
  function ucbWidget(root) {
    const K = 5;
    const W = root.clientWidth < 480 ? 330 : 440, H = 290, M = { l: 34, r: 8, t: 26, b: 46 };
    const Y0 = -3, Y1 = 5;
    const chart = svg(W, H);
    chart.style.maxWidth = "560px";
    chart.style.margin = "0 auto";
    root.appendChild(chart);
    const ctrl = control(root, `
      <button data-a="step">Step</button><button data-a="play">▶ Play</button><button data-a="reset">New bandit</button>
      <label>exploration <b>c</b> <input type="range" min="0" max="3" step="0.1" value="2"> <span class="val"></span></label>
      <label><input type="checkbox"> show true values</label>`);
    const cIn = ctrl.querySelector("input[type=range]"), truth = ctrl.querySelector("input[type=checkbox]");
    const playBtn = ctrl.querySelector("[data-a=play]");
    const out = readout(root);
    hint(root, "Orange dot = estimate Q(a). Purple bar = Q(a) ± bonus. UCB pulls the arm whose bar reaches highest (★). Arms never pulled have an infinite bonus.");

    const x = linear(0, K, M.l, W - M.r), y = linear(Y0, Y1, H - M.b, M.t);
    const colW = x(1) - x(0);
    let qstar, Q, N, t, last, timer = null;

    function reset() {
      stop();
      qstar = Array.from({ length: K }, () => gauss(Math.random));
      Q = new Array(K).fill(0); N = new Array(K).fill(0); t = 0; last = null;
      draw();
    }

    const bonus = (a) => (N[a] === 0 ? Infinity : +cIn.value * Math.sqrt(Math.log(t + 1) / N[a]));
    const upper = (a) => Q[a] + bonus(a);
    function choice() {
      let best = 0;
      for (let a = 1; a < K; a++) if (upper(a) > upper(best)) best = a;
      return best;
    }

    function step() {
      const a = choice(), r = qstar[a] + gauss(Math.random);
      N[a] += 1; Q[a] += (r - Q[a]) / N[a]; t += 1;
      last = { a, r };
      draw();
    }

    function stop() { clearInterval(timer); timer = null; if (playBtn) playBtn.textContent = "▶ Play"; }

    function draw() {
      const c = colors();
      cIn.nextElementSibling.textContent = (+cIn.value).toFixed(1);
      chart.innerHTML = "";
      axes(chart, x, y, { xDomain: [0, K], yDomain: [Y0, Y1], xTicks: [], yTicks: ticks(-2, 4, 2) });
      el("text", { x: M.l, y: 14 }, chart).textContent = "value";
      const next = choice(), bestArm = qstar.indexOf(Math.max(...qstar));
      for (let a = 0; a < K; a++) {
        const cx = x(a + 0.5), b = bonus(a);
        if (a === next) el("rect", { x: x(a) + 4, y: M.t, width: colW - 8, height: y(Y0) - M.t, fill: c.opt, "fill-opacity": 0.1, rx: 6 }, chart);
        if (N[a] === 0) {
          el("line", { x1: cx, x2: cx, y1: y(Y1), y2: y(Y0), stroke: c.loss, "stroke-width": 6, "stroke-opacity": 0.35 }, chart);
          el("text", { x: cx, y: y(0) + 4, "text-anchor": "middle", style: textStyle + "; font-size: 13px" }, chart).textContent = "∞";
        } else {
          const top = Math.min(Y1, Q[a] + b), bot = Math.max(Y0, Q[a] - b);
          el("rect", { x: cx - 5, y: y(top), width: 10, height: y(bot) - y(top), fill: c.loss, "fill-opacity": 0.35, rx: 3 }, chart);
          el("line", { x1: cx - 12, x2: cx + 12, y1: y(top), y2: y(top), stroke: c.loss, "stroke-width": 2.5 }, chart);
          el("circle", { cx, cy: y(Q[a]), r: 5.5, fill: c.model, stroke: c.bg, "stroke-width": 1.5 }, chart);
        }
        if (truth.checked) el("line", { x1: cx - colW * 0.35, x2: cx + colW * 0.35, y1: y(qstar[a]), y2: y(qstar[a]), stroke: c.opt, "stroke-width": 2, "stroke-dasharray": "5 3" }, chart);
        if (a === next) el("text", { x: cx, y: M.t - 8, "text-anchor": "middle", style: "fill: var(--viz-opt); font-size: 14px" }, chart).textContent = "★";
        el("text", { x: cx, y: H - M.b + 16, "text-anchor": "middle", style: textStyle + (truth.checked && a === bestArm ? "; font-weight: 700" : "") }, chart).textContent = `arm ${a + 1}`;
        el("text", { x: cx, y: H - M.b + 31, "text-anchor": "middle" }, chart).textContent = `N = ${N[a]}`;
      }
      let text = `t = ${t}`;
      if (last) text += `   last pull: arm ${last.a + 1}, reward ${fmt(last.r, 2)}`;
      const b = bonus(next);
      text += `\nnext pick: arm ${next + 1}  ` + (N[next] === 0 ? "(never tried, so its bonus is infinite)" : `(Q = ${fmt(Q[next], 2)} + bonus ${fmt(b, 2)} = ${fmt(Q[next] + b, 2)})`);
      if (truth.checked) text += `\ngreen dashes = true values q*; the best arm is arm ${bestArm + 1}`;
      out.textContent = text;
    }

    ctrl.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (a === "step") { stop(); step(); }
      if (a === "reset") reset();
      if (a === "play") {
        if (timer) { stop(); return; }
        playBtn.textContent = "⏸ Pause";
        timer = setInterval(() => { step(); if (t >= 500) stop(); }, 120);
      }
    });
    cIn.addEventListener("input", draw);
    truth.addEventListener("change", draw);
    onThemeChange(draw);
    reset();
  }

  const mounts = { "viz-play": playWidget, "viz-ucb": ucbWidget };
  for (const id in TESTBEDS) mounts[id] = (node) => testbedWidget(node, TESTBEDS[id]);
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
