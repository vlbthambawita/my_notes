// Interactive widgets for the "Offline RL and Imitation Learning" lesson.
// Two widgets, each mounted twice with different options:
//   #viz-drift-bc, #viz-drift-dagger   – lane following: behaviour cloning drifts off the road, DAgger stays on
//   #viz-offline-naive, #viz-offline-fix – offline Q-learning on the cliff world from a fixed, logged dataset
// Both use a small seeded random-number generator, so the numbers on screen are the same every time
// (and match the Python scripts used to write the lesson).
// Runs after DOMContentLoaded so the site-wide helpers (assets/js/viz-common.js) are available.
document.addEventListener("DOMContentLoaded", function () {
  const { colors, onThemeChange, linear, el, svg, axes, ticks, fmt } = window.Viz;
  const textStyle = "fill: var(--viz-text)";

  // ---------------------------------------------------------------- shared helpers
  function mulberry32(a) {                       // tiny seeded RNG: returns numbers in [0, 1)
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rnd) {                          // standard normal sample (Box–Muller)
    const u1 = 1 - rnd(), u2 = rnd();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
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
  const num = (v, d = 2) => fmt(v, d).replace("-", "−");      // typographic minus sign

  // ================================================================ Widget 1: compounding errors
  // State: e = distance from the lane centre, h = heading error. The road bends by kappa[t] each step.
  // Expert: a = kappa − 0.1·e − 0.5·h. Learner: linear policy a = w·(kappa, e, h), fitted by least squares.
  const T = 100, K1 = 0.1, K2 = 0.5, LANE = 1;
  const KAPPA = Array.from({ length: T }, (_, t) => 0.03 * Math.sin((2 * Math.PI * t) / 60));
  const expert = (k, e, h) => k - K1 * e - K2 * h;
  const linPolicy = (w) => (k, e, h) => w[0] * k + w[1] * e + w[2] * h;

  // Road centre in world coordinates (for drawing only)
  const ROAD = (() => {
    let phi = 0, y = 0;
    const ys = [0];
    for (let t = 0; t < T; t++) { phi += KAPPA[t]; y += phi; ys.push(y); }
    return ys;
  })();

  function rollout(policy, sigma, rnd) {
    let e = 0, h = 0;
    const states = [], es = [0];
    for (let t = 0; t < T; t++) {
      const k = KAPPA[t];
      states.push([k, e, h]);
      const a = policy(k, e, h);
      h = h + a - k + sigma * gauss(rnd);
      e = e + h;
      es.push(e);
    }
    return { states, es };
  }

  // Least squares via the normal equations (+ a tiny ridge), solved by Gaussian elimination
  function fit(data) {
    const A = [0, 1, 2].map((i) => [0, 1, 2].map((j) => (i === j ? 1e-6 : 0)));
    const b = [0, 0, 0];
    for (const [x, y] of data) {
      for (let i = 0; i < 3; i++) {
        b[i] += x[i] * y;
        for (let j = 0; j < 3; j++) A[i][j] += x[i] * x[j];
      }
    }
    const M = A.map((row, i) => row.concat([b[i]]));
    for (let c = 0; c < 3; c++) {
      let p = c;
      for (let r = c + 1; r < 3; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = 0; r < 3; r++) {
        if (r === c) continue;
        const f = M[r][c] / M[c][c];
        for (let k = c; k < 4; k++) M[r][k] -= f * M[c][k];
      }
    }
    return [0, 1, 2].map((i) => M[i][3] / M[i][i]);
  }

  // rounds = 0 is plain behaviour cloning; each DAgger round adds 5 drives by the learner, labelled by the expert
  function train(rounds, sigma) {
    const demo = rollout(expert, 0, mulberry32(1)).states;     // one perfect, noise-free demonstration
    const data = demo.map((s) => [s, expert(...s)]);
    let w = fit(data);
    const rnd = mulberry32(777);
    for (let i = 0; i < rounds; i++) {
      for (let j = 0; j < 5; j++) {
        for (const s of rollout(linPolicy(w), sigma, rnd).states) data.push([s, expert(...s)]);
      }
      w = fit(data);
    }
    return { w, n: data.length };
  }

  function evaluate(policy, sigma, n = 200) {
    const rnd = mulberry32(12345);                               // same noise for every policy: a fair test
    const mean = new Array(T + 1).fill(0);
    let off = 0;
    for (let i = 0; i < n; i++) {
      const { es } = rollout(policy, sigma, rnd);
      es.forEach((e, t) => (mean[t] += Math.abs(e) / n));
      if (es.some((e) => Math.abs(e) > LANE)) off++;
    }
    return { mean, off: (100 * off) / n };
  }

  function driftWidget(root, opts) {
    // Two sizes: a wide one, and a narrow one for phones (so the text stays readable)
    let W = 640, H = 230, CW = 640, CH = 220, CM = { l: 46, r: 12, t: 26, b: 40 };
    const M = { l: 8, r: 8, t: 26, b: 8 };
    const road = svg(W, H);
    root.appendChild(road);
    const chart = svg(CW, CH);
    root.appendChild(chart);
    const ctrl = control(root, `
      <label>steering noise σ <input type="range" min="0" max="0.03" step="0.005" value="0.01"> <span class="sig"></span></label>
      ${opts.dagger ? '<label>DAgger rounds <input type="range" min="0" max="3" step="1" value="1"> <span class="rnd"></span></label>' : ""}
      <button data-a="new">New drive</button>`);
    const [sigIn, rndIn] = ctrl.querySelectorAll("input");
    hint(root, opts.dagger
      ? "Top: one drive per policy, all with the same random steering noise. Bottom: average distance from the lane centre over 200 drives. DAgger rounds = 0 is plain behaviour cloning."
      : "Top: one drive per policy, both with the same random steering noise. Bottom: average distance from the lane centre over 200 drives.");
    const out = readout(root);
    let seed = 2024;

    function draw() {
      const c = colors(), sigma = +sigIn.value;
      ctrl.querySelector(".sig").textContent = sigma.toFixed(3);
      const rounds = opts.dagger ? +rndIn.value : 0;
      if (opts.dagger) ctrl.querySelector(".rnd").textContent = rounds;

      const bc = train(0, sigma);
      const pols = [
        { name: "expert", pol: expert, col: c.opt },
        { name: "behaviour cloning", pol: linPolicy(bc.w), col: c.grad, w: bc.w, n: bc.n },
      ];
      if (opts.dagger) {
        const dg = train(rounds, sigma);
        pols.push({ name: `DAgger (${rounds} round${rounds === 1 ? "" : "s"})`, pol: linPolicy(dg.w), col: c.model, w: dg.w, n: dg.n });
      }

      // --- top: the road and one sample drive per policy (same noise for each)
      const narrow = root.clientWidth < 520;
      W = CW = narrow ? 360 : 640;
      H = narrow ? 200 : 230;
      CH = narrow ? 250 : 220;
      CM.t = narrow ? 44 : 26;
      road.setAttribute("viewBox", `0 0 ${W} ${H}`);
      chart.setAttribute("viewBox", `0 0 ${CW} ${CH}`);
      road.innerHTML = "";
      const yMin = Math.min(...ROAD) - 5, yMax = Math.max(...ROAD) + 5;
      const x = linear(0, T, M.l, W - M.r), y = linear(yMin, yMax, H - M.b, M.t);
      const clipId = root.id + "-clip";
      const defs = el("defs", {}, road);
      const cp = el("clipPath", { id: clipId }, defs);
      el("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b }, cp);
      el("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b, fill: "none", stroke: c.border }, road);
      const edge = (off) => ROAD.map((v, t) => (t ? "L" : "M") + x(t) + "," + y(v + off)).join("");
      const band = edge(LANE) + ROAD.slice().reverse().map((v, i) => "L" + x(T - i) + "," + y(v - LANE)).join("") + "Z";
      el("path", { d: band, fill: c.muted, "fill-opacity": 0.18 }, road);
      el("path", { d: edge(LANE), fill: "none", stroke: c.muted, "stroke-width": 1.5 }, road);
      el("path", { d: edge(-LANE), fill: "none", stroke: c.muted, "stroke-width": 1.5 }, road);
      el("path", { d: edge(0), fill: "none", stroke: c.muted, "stroke-dasharray": "6 5" }, road);
      const g = el("g", { "clip-path": `url(#${clipId})` }, road);
      // draw the learners first and the expert last (dashed), so overlapping lines stay visible
      for (const p of pols.slice(1).concat(pols.slice(0, 1))) {
        const { es } = rollout(p.pol, sigma, mulberry32(seed));
        const d = es.map((e, t) => (t ? "L" : "M") + x(t) + "," + y(ROAD[t] + e)).join("");
        const dash = p.name === "expert" ? { "stroke-dasharray": "7 5" } : {};
        el("path", { d, fill: "none", stroke: p.col, "stroke-width": 2.5, "stroke-linejoin": "round", ...dash }, g);
      }
      el("text", { x: M.l + 4, y: 16, style: textStyle }, road).textContent = "the road (grey band = lane) and one drive per policy →";

      // --- bottom: average |offset| over 200 drives
      chart.innerHTML = "";
      const evals = pols.map((p) => evaluate(p.pol, sigma));
      const top = Math.max(2, Math.ceil(Math.max(...evals.map((e) => e.mean[T])) / 2) * 2);
      const cx = linear(0, T, CM.l, CW - CM.r), cy = linear(0, top, CH - CM.b, CM.t);
      axes(chart, cx, cy, { xDomain: [0, T], yDomain: [0, top], xTicks: ticks(0, T, 20), yTicks: ticks(0, top, top > 8 ? 4 : top > 4 ? 2 : 1), xLabel: "time step t" });
      el("line", { x1: cx(0), x2: cx(T), y1: cy(LANE), y2: cy(LANE), stroke: c.grad, "stroke-dasharray": "5 4" }, chart);
      el("text", { x: cx(2), y: cy(LANE) - 5, style: "fill: var(--viz-grad)" }, chart).textContent = "edge of the lane";
      evals.forEach((ev, i) => {
        const d = ev.mean.map((v, t) => (t ? "L" : "M") + cx(t) + "," + cy(Math.min(v, top))).join("");
        el("path", { d, fill: "none", stroke: pols[i].col, "stroke-width": 2.5 }, chart);
      });
      el("text", { x: CM.l, y: 16, style: textStyle }, chart).textContent = "average distance from the lane centre |e|";
      // legend
      let lx = CW - CM.r;
      const ly = narrow ? 34 : 16;
      for (let i = pols.length - 1; i >= 0; i--) {
        const label = pols[i].name.replace(/ \(.*\)/, "");
        const t = el("text", { x: lx, y: ly, "text-anchor": "end", style: `fill: ${pols[i].col}` }, chart);
        t.textContent = "● " + label;
        lx -= label.length * 6.4 + 22;
      }

      const lines = pols.map((p, i) => {
        const ev = evals[i];
        const w = p.w ? `   weights (bend, offset, heading) = (${num(p.w[0] + 0)}, ${num(Math.abs(p.w[1]) < 0.005 ? 0 : p.w[1])}, ${num(Math.abs(p.w[2]) < 0.005 ? 0 : p.w[2])})` : "   steers with (1, −0.10, −0.50)";
        return `${p.name}: |e| at t=25: ${fmt(ev.mean[25], 2)}, t=50: ${fmt(ev.mean[50], 2)}, t=100: ${fmt(ev.mean[100], 2)} · left the lane in ${Math.round(ev.off)}% of drives\n${w}`;
      });
      out.textContent = lines.join("\n");
    }

    sigIn.addEventListener("input", draw);
    if (rndIn) rndIn.addEventListener("input", draw);
    ctrl.addEventListener("click", (e) => { if (e.target.dataset.a === "new") { seed += 1; draw(); } });
    let wasNarrow = root.clientWidth < 520;
    window.addEventListener("resize", () => {
      const isNarrow = root.clientWidth < 520;
      if (isNarrow !== wasNarrow) { wasNarrow = isNarrow; draw(); }
    });
    onThemeChange(draw);
    draw();
  }

  // ================================================================ Widget 2: offline RL on the cliff world
  const MAP = ["......", "..#...", "......", "SCCCCG"];
  const ROWS = 4, COLS = 6, GAMMA = 0.95, START = [3, 0];
  const ACT = [[-1, 0], [0, 1], [1, 0], [0, -1]];
  const ARROWS = ["↑", "→", "↓", "←"];
  const SAFE = ["→→→→→↓", "↑↑.↑↑↓", "↑↑←↑↑↓", "↑....."];
  const BETA = 0.5;
  const isFloor = (r, c) => MAP[r][c] === "." || MAP[r][c] === "S";
  const STATES = [];
  for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) if (isFloor(r, c)) STATES.push([r, c]);
  const sid = (r, c) => r * COLS + c;

  function envStep(r, c, a) {
    let nr = r + ACT[a][0], nc = c + ACT[a][1];
    if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS || MAP[nr][nc] === "#") { nr = r; nc = c; }
    const t = MAP[nr][nc];
    if (t === "G") return [nr, nc, 10, true];
    if (t === "C") return [nr, nc, -20, true];
    return [nr, nc, -1, false];
  }

  const newQ = (v) => Array.from({ length: ROWS * COLS }, () => [v, v, v, v]);
  const qmax = (q) => Math.max(q[0], q[1], q[2], q[3]);
  const argmax = (q) => { let b = 0; for (let a = 1; a < 4; a++) if (q[a] > q[b]) b = a; return b; };

  const QSTAR = (() => {
    const Q = newQ(0);
    for (let it = 0; it < 2000; it++) {
      for (const [r, c] of STATES) {
        for (let a = 0; a < 4; a++) {
          const [nr, nc, rw, d] = envStep(r, c, a);
          Q[sid(r, c)][a] = rw + (d ? 0 : GAMMA * qmax(Q[sid(nr, nc)]));
        }
      }
    }
    return Q;
  })();

  function behaviourValue(eps) {                 // exact value of the logging policy at the start cell
    const V = new Array(ROWS * COLS).fill(0);
    for (let it = 0; it < 3000; it++) {
      for (const [r, c] of STATES) {
        const safeA = ARROWS.indexOf(SAFE[r][c]);
        let v = 0;
        for (let a = 0; a < 4; a++) {
          const p = eps / 4 + (1 - eps) * (a === safeA ? 1 : 0);
          if (!p) continue;
          const [nr, nc, rw, d] = envStep(r, c, a);
          v += p * (rw + (d ? 0 : GAMMA * V[sid(nr, nc)]));
        }
        V[sid(r, c)] = v;
      }
    }
    return V[sid(...START)];
  }

  function collect(eps, episodes, seed) {
    const rnd = mulberry32(seed), data = [];
    for (let i = 0; i < episodes; i++) {
      let [r, c] = START;
      for (let t = 0; t < 100; t++) {
        const a = rnd() < eps ? Math.floor(rnd() * 4) : ARROWS.indexOf(SAFE[r][c]);
        const [nr, nc, rw, d] = envStep(r, c, a);
        data.push([r, c, a, rw, nr, nc, d]);
        r = nr; c = nc;
        if (d) break;
      }
    }
    return data;
  }

  function counts(data) {
    const n = newQ(0);
    for (const [r, c, a] of data) n[sid(r, c)][a] += 1;
    return n;
  }

  // pessimistic = false: naive Q-learning, unseen actions keep their initial guess 0.
  // pessimistic = true: unseen actions are −∞ and rarely seen ones pay β/√n.
  function offlineQ(data, n, pessimistic) {
    const Q = n.map((row) => row.map((k) => (pessimistic && k === 0 ? -Infinity : 0)));
    const lr = 0.5;
    for (let ep = 0; ep < 200; ep++) {
      for (const [r, c, a, rw, nr, nc, d] of data) {
        let nxt = d ? 0 : qmax(Q[sid(nr, nc)]);
        if (nxt === -Infinity) nxt = 0;
        const pen = pessimistic ? BETA / Math.sqrt(n[sid(r, c)][a]) : 0;
        const q = Q[sid(r, c)];
        q[a] += lr * (rw - pen + GAMMA * nxt - q[a]);
      }
    }
    return Q;
  }

  // Follow a greedy policy from the start; stop at a terminal cell, or when a cell repeats (it loops forever)
  function followPolicy(policy) {
    let [r, c] = START, G = 0, disc = 1, end = "loops";
    const path = [[r, c]];
    for (let t = 0; t < 100; t++) {
      const [nr, nc, rw, d] = envStep(r, c, policy[sid(r, c)]);
      G += disc * rw; disc *= GAMMA; r = nr; c = nc;
      if (path.length < 40) path.push([r, c]);
      if (d) { end = rw > 0 ? "goal" : "cliff"; break; }
    }
    return { G, end, path };
  }

  function offlineWidget(root, opts) {
    const CELL = 64, PAD = 6;
    const GW = COLS * CELL + 2 * PAD, GH = ROWS * CELL + 2 * PAD;
    const px = (c) => PAD + c * CELL, py = (r) => PAD + r * CELL;
    const board = svg(GW, GH);
    board.style.maxWidth = "520px";
    board.style.margin = "0 auto";
    root.appendChild(board);
    hint(root, "Blue wedges = actions that appear in the logged data (darker = more often). Arrow = what the learned policy does; red arrow = an action it has never seen. Number = the value it believes in. Orange line = what really happens when it runs.");
    const methods = opts.methods;
    let method = opts.start;
    const mctrl = methods.length > 1
      ? control(root, "Learner: " + methods.map((m) => `<button data-m="${m}">${{ naive: "Naive Q-learning", pess: "Pessimistic Q-learning", bc: "Behaviour cloning" }[m]}</button>`).join(""))
      : null;
    const ctrl = control(root, `
      <label>exploration in the logged data ε <input type="range" min="0" max="1" step="0.1" value="${opts.eps}"> <span class="eps"></span></label>
      <label>episodes logged <input type="range" min="5" max="100" step="5" value="${opts.episodes}"> <span class="eps-n"></span></label>
      <button data-a="new">New dataset</button>`);
    const [epsIn, nIn] = ctrl.querySelectorAll("input");
    const out = readout(root);
    let seed = 2024;

    function draw() {
      const col = colors(), eps = +epsIn.value, episodes = +nIn.value;
      ctrl.querySelector(".eps").textContent = eps.toFixed(1);
      ctrl.querySelector(".eps-n").textContent = episodes;
      if (mctrl) mctrl.querySelectorAll("button").forEach((b) => (b.style.borderColor = b.dataset.m === method ? col.model : ""));

      const data = collect(eps, episodes, seed), n = counts(data);
      const visited = new Set(data.map(([r, c]) => sid(r, c)));
      let Q = null, policy;
      if (method === "bc") policy = n.map((row) => (qmax(row) > 0 ? argmax(row) : 0));
      else { Q = offlineQ(data, n, method === "pess"); policy = Q.map(argmax); }
      const run = followPolicy(policy);

      board.innerHTML = "";
      const maxN = Math.max(...n.map(qmax));
      for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
          const t = MAP[r][c], x = px(c), y = py(r);
          el("rect", { x, y, width: CELL, height: CELL, fill: col.bg, stroke: col.border }, board);
          if (t === "#") el("rect", { x: x + 1, y: y + 1, width: CELL - 2, height: CELL - 2, fill: col.muted, "fill-opacity": 0.55 }, board);
          if (t === "G" || t === "C") {
            el("rect", { x: x + 1, y: y + 1, width: CELL - 2, height: CELL - 2, fill: t === "G" ? col.opt : col.grad, "fill-opacity": t === "G" ? 0.45 : 0.6 }, board);
            el("text", { x: x + CELL / 2, y: y + 18, "text-anchor": "middle", style: textStyle }, board).textContent = t === "G" ? "GOAL" : "cliff";
          }
          if (!isFloor(r, c)) continue;
          const s = sid(r, c), cx = x + CELL / 2, cy = y + CELL / 2;
          // wedges: one small triangle per action, pointing that way
          for (let a = 0; a < 4; a++) {
            const [dr, dc] = ACT[a];
            const tipX = cx + dc * 29, tipY = cy + dr * 29, baseX = cx + dc * 21, baseY = cy + dr * 21;
            const pts = [[tipX, tipY], [baseX + dr * 7, baseY + dc * 7], [baseX - dr * 7, baseY - dc * 7]].map((p) => p.join(",")).join(" ");
            const k = n[s][a];
            if (k > 0) el("polygon", { points: pts, fill: col.data, "fill-opacity": 0.3 + 0.7 * Math.log(1 + k) / Math.log(1 + maxN) }, board);
            else el("polygon", { points: pts, fill: "none", stroke: col.muted, "stroke-width": 0.8, "stroke-dasharray": "2 2" }, board);
          }
          const a = policy[s];
          const known = n[s][a] > 0;
          const noData = !visited.has(s);
          el("text", { x: cx, y: cy + 3, "text-anchor": "middle", style: `font-size: 20px; fill: ${noData && method !== "naive" ? "var(--viz-muted)" : known ? "var(--viz-text)" : "var(--viz-grad)"}` }, board).textContent = noData && method !== "naive" ? "?" : ARROWS[a];
          if (Q && !(noData && method === "pess")) {
            const v = qmax(Q[s]);
            el("text", { x: cx, y: cy + 17, "text-anchor": "middle", style: `font-size: 10px; fill: ${known ? "var(--viz-text)" : "var(--viz-grad)"}` }, board).textContent = Math.abs(v) < 0.05 ? "0.0" : num(v, 1);
          }
        }
      }
      // what really happens
      const pathD = run.path.map(([r, c], i) => (i ? "L" : "M") + (px(c) + CELL / 2 + 6) + "," + (py(r) + CELL / 2 + 6)).join("");
      el("path", { d: pathD, fill: "none", stroke: col.model, "stroke-width": 3, "stroke-opacity": 0.7, "stroke-linejoin": "round" }, board);
      const [er, ec] = run.path[run.path.length - 1];
      el("circle", { cx: px(ec) + CELL / 2 + 6, cy: py(er) + CELL / 2 + 6, r: 6, fill: run.end === "goal" ? col.opt : col.grad, stroke: col.bg, "stroke-width": 1.5 }, board);
      el("text", { x: px(0) + 4, y: py(3) + 12 }, board).textContent = "start";

      // read-out
      const seen = n.reduce((acc, row) => acc + row.filter((k) => k > 0).length, 0);
      const name = { naive: "Naive Q-learning", pess: "Pessimistic Q-learning (β = 0.5)", bc: "Behaviour cloning (most common action)" }[method];
      const fate = { goal: "reaches the goal", cliff: "falls off the cliff", loops: "gets stuck walking into a wall" }[run.end];
      let txt = `data: ${data.length} transitions · ${seen} of ${STATES.length * 4} state-action pairs seen · the logging policy itself is worth ${num(behaviourValue(eps))}`;
      if (Q) {
        const sS = sid(...START);
        txt += `\n${name}: believes V(start) = ${num(qmax(Q[sS]))} · really gets ${num(run.G)} (${fate})`;
        let over = 0, phantom = 0;
        for (const s of visited) {
          for (let a = 0; a < 4; a++) if (Q[s][a] > QSTAR[s][a] + 0.01) over++;
          if (n[s][policy[s]] === 0) phantom++;
        }
        txt += `\nvalues higher than even the optimal Q*: ${over} · cells in the data where it picks an unseen action: ${phantom}`;
      } else {
        txt += `\n${name}: really gets ${num(run.G)} (${fate})`;
      }
      txt += `\nfor reference: the best possible policy gets ${fmt(qmax(QSTAR[sid(...START)]), 2)}`;
      out.textContent = txt;
    }

    epsIn.addEventListener("input", draw);
    nIn.addEventListener("input", draw);
    ctrl.addEventListener("click", (e) => { if (e.target.dataset.a === "new") { seed += 1; draw(); } });
    if (mctrl) mctrl.addEventListener("click", (e) => { if (e.target.dataset.m) { method = e.target.dataset.m; draw(); } });
    onThemeChange(draw);
    draw();
  }

  const mounts = {
    "viz-drift-bc": (n) => driftWidget(n, { dagger: false }),
    "viz-drift-dagger": (n) => driftWidget(n, { dagger: true }),
    "viz-offline-naive": (n) => offlineWidget(n, { methods: ["naive"], start: "naive", eps: 0.1, episodes: 20 }),
    "viz-offline-fix": (n) => offlineWidget(n, { methods: ["naive", "pess", "bc"], start: "pess", eps: 0.1, episodes: 20 }),
  };
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
