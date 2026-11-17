// Interactive widgets for the "Learning Rules: Recognize, Erase and Reject Feedback" lesson.
//   #viz-walkthrough – the book's worked example: one Car rule meets vehicles #1–#4, step by step
//   #viz-train       – train one rule on the vehicle or breast-cancer data, with each feedback type on or off
//   #viz-game        – two literals as players in a game: the 2×2 outcome table and the long-run learning outcome
// Rule memory follows Granmo, "An Introduction to Tsetlin Machines", Ch. 1 §1.3–1.4 (positions 1–10, start at 5,
// memorized from 6) and Ch. 2 §2.2 (two literals, memory depth 8, Memorize Value 1.0, Forget Value 1/s).
document.addEventListener("DOMContentLoaded", function () {
  const { colors, onThemeChange, el, svg } = window.Viz;

  // ---------------------------------------------------------------- shared helpers
  // Seeded RNG (mulberry32), ported bit-for-bit to the Python check script.
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
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

  function box(parent, cls) {
    const div = document.createElement("div");
    div.className = cls;
    parent.appendChild(div);
    return div;
  }

  // Small lesson-specific styles (colours come from the shared viz tokens).
  const style = document.createElement("style");
  style.textContent = `
    .tmf-vehicle { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; font-size: 0.75rem; margin-bottom: 8px; min-height: 1.8em; }
    .tmf-chip { border: 1px solid var(--viz-border); border-radius: 10px; padding: 0 8px; color: var(--viz-muted); background: var(--viz-bg); }
    .tmf-chip.on { border-color: var(--viz-data); color: var(--viz-data); font-weight: 600; }
    .tmf-fb { border: 1.5px solid var(--viz-model); color: var(--viz-model); border-radius: 6px; padding: 0 8px; font-weight: 700; }
    .tmf-caption { font-size: 0.74rem; color: var(--viz-text); min-height: 4.6em; margin-top: 6px; }
    .tmf-step { color: var(--viz-muted); }
    .viz table.tmf-grid { border-collapse: collapse; width: 100%; font-size: 0.7rem; margin-top: 8px; display: table; }
    .viz table.tmf-grid th, .viz table.tmf-grid td { border: 1px solid var(--viz-border); padding: 6px 4px; text-align: center; background: var(--viz-bg); color: var(--viz-text); font-weight: 400; }
    .viz table.tmf-grid th { background: var(--viz-panel); color: var(--viz-muted); }
    .tmf-cond { color: var(--viz-muted); }
    .tmf-fbname { font-weight: 700; color: var(--viz-model); margin: 2px 0; }
    .tmf-arrows { white-space: nowrap; }
    .tmf-up { color: var(--viz-opt); font-weight: 700; }
    .tmf-down { color: var(--viz-grad); font-weight: 700; }
    .tmf-none { color: var(--viz-muted); }
    .viz-controls select { font: inherit; padding: 2px 4px; border-radius: 6px; border: 1px solid var(--viz-border); background: var(--viz-bg); color: var(--viz-text); max-width: 100%; }
    .viz-controls input[type="number"] { font: inherit; padding: 2px 4px; border-radius: 6px; border: 1px solid var(--viz-border); background: var(--viz-bg); color: var(--viz-text); }
  `;
  document.head.appendChild(style);

  const DEPTH = 10, HALF = 5;              // positions 1..10; 6..10 = Memorized
  const isMem = (p) => p > HALF;
  const textStyle = "fill: var(--viz-text)";

  // Literals come in pairs: [x1, ¬x1, x2, ¬x2, ...], as in the book's memory figures.
  function literalsOf(x) {
    const out = [];
    for (const v of x) out.push(v, 1 - v);
    return out;
  }
  function namesOf(features) {
    const out = [];
    for (const f of features) out.push(f, "¬" + f);
    return out;
  }
  const evaluate = (pos, lits) => pos.every((p, i) => !isMem(p) || lits[i] === 1);
  function ruleText(pos, names, cls) {
    const c = names.filter((_, i) => isMem(pos[i]));
    return c.length ? `IF ${c.join(" AND ")} THEN ${cls}` : `IF (empty, always True) THEN ${cls}`;
  }

  // ---------------------------------------------------------------- data
  // Granmo, Ch. 1, Table 1.1: three cars and three planes.
  const VEHICLES = {
    features: ["Four Wheels", "Transports People", "Wings", "Yellow", "Blue"],
    short: ["4 Wheels", "Transports", "Wings", "Yellow", "Blue"],
    X: [[1, 1, 0, 0, 1], [1, 1, 0, 1, 0], [1, 1, 0, 1, 0], [1, 1, 1, 0, 1], [1, 0, 1, 1, 0], [0, 1, 1, 0, 1]],
    Y: [1, 1, 1, 0, 0, 0],
  };

  // UCI Breast Cancer dataset (Zwitter & Soklic, Ljubljana), all 286 patients, Booleanized like the book's Table 1.4.
  // Four digits per patient: menopause (lt40, ge40, premeno), inv-nodes (0-2, 3-5, 6 or more),
  // deg-malig (1, 2, 3), recurrence (1 = yes).
  const BC_CODE =
    "2020201020101010201010102010100020102010202010100000102020200000201020201020100010101010201020102010201020001020201010101020201020201000100020101010100020102000100010101000201010001000201020002000200000102010200010201000100010001010100010002010200010002010200020002000102010201010100020101010201020101000101020002010201010101000101010102000102010101000102020002010102010101010200020101010101020002010102010202010201020102000201020102020100010001000201020002010101020001020200010001000101010201000100010102210221020102210221021202210102010102110212020002010201011102110100020101220201010102110202000201220112012201020002011101120200010002000202021101010110011002110211012102010101021102000121011102110201010101010211020202020202020001220122022101000101020101010222020101210201020101010210020101010200010202011200110112011201110212011202120012011101120212001101110111011201110212001202120011021102121211121222120211121100122112111212112011021212120112011202121111111222112211121222122112221122112212111202120112001102120212221200121111211112111111221102100010001222121111221202120111221202120211011111122212221122122211221122120112021100111211121";
  function bcData() {
    const X = [], Y = [];
    for (let i = 0; i < BC_CODE.length; i += 4) {
      const m = +BC_CODE[i], n = +BC_CODE[i + 1], d = +BC_CODE[i + 2];
      const x = new Array(9).fill(0);
      x[m] = 1; x[3 + n] = 1; x[6 + d] = 1;
      X.push(x);
      Y.push(+BC_CODE[i + 3]);
    }
    return {
      features: ["Menopause lt40", "Menopause ge40", "Menopause premeno", "Inv-nodes 0-2", "Inv-nodes 3-5",
        "Inv-nodes 6+", "Deg-malig 1", "Deg-malig 2", "Deg-malig 3"],
      short: ["lt40", "ge40", "premeno", "nodes 0-2", "nodes 3-5", "nodes 6+", "malig 1", "malig 2", "malig 3"],
      X, Y,
    };
  }

  // ---------------------------------------------------------------- memory chart (shared by widgets 1 and 2)
  // One row per literal; x = memory position 1..10. Optional arrows and a history strip on the right.
  function memoryChart(root, names, pos, opts) {
    const c = colors();
    const narrow = opts.narrow;
    const rowH = opts.rowH || 22;
    const nameW = narrow ? 92 : 140;
    const cellW = narrow ? (opts.strip ? 14 : 24) : (opts.strip ? 22 : 34);
    const stripW = opts.strip ? (narrow ? 96 : 190) : 0;
    const x0 = nameW + 8, barW = cellW * DEPTH;
    const sx0 = x0 + barW + 14;
    const W = sx0 + stripW + (opts.strip ? 4 : 0) - (opts.strip ? 0 : 6);
    const top = 30, H = top + rowH * names.length + 8;
    root.setAttribute("viewBox", `0 0 ${W} ${H}`);
    root.innerHTML = "";
    const xc = (p) => x0 + (p - 0.5) * cellW;

    // header: forgotten / memorized halves
    el("rect", { x: x0, y: top - 4, width: cellW * HALF, height: rowH * names.length + 6, fill: c.muted, "fill-opacity": 0.07 }, root);
    el("rect", { x: x0 + cellW * HALF, y: top - 4, width: cellW * HALF, height: rowH * names.length + 6, fill: c.opt, "fill-opacity": 0.09 }, root);
    el("text", { x: x0 + (cellW * HALF) / 2, y: 12, "text-anchor": "middle" }, root).textContent = "Forgotten";
    el("text", { x: x0 + cellW * HALF * 1.5, y: 12, "text-anchor": "middle", style: `fill: ${c.opt}` }, root).textContent = "Memorized";
    for (let p = 1; p <= DEPTH; p++) {
      if (!narrow || p === 1 || p === 5 || p === 6 || p === 10)
        el("text", { x: xc(p), y: top - 8, "text-anchor": "middle", style: "font-size: 9px" }, root).textContent = p;
    }
    el("line", { x1: x0 + cellW * HALF, x2: x0 + cellW * HALF, y1: top - 4, y2: top + rowH * names.length + 2, stroke: c.text, "stroke-width": 1.5, "stroke-dasharray": "4 3" }, root);
    if (opts.strip) {
      el("text", { x: sx0 + stripW / 2, y: 12, "text-anchor": "middle" }, root).textContent = narrow ? "history" : "history (older → newer)";
    }

    names.forEach((name, i) => {
      const y = top + i * rowH + rowH / 2;
      const p = pos[i];
      el("text", { x: nameW, y: y + 4, "text-anchor": "end", style: isMem(p) ? `fill: ${c.opt}; font-weight: 700` : "" }, root).textContent = name;
      el("line", { x1: x0, x2: x0 + barW, y1: y, y2: y, stroke: c.grid }, root);
      const arrow = opts.arrows && opts.arrows[i];
      if (arrow) {
        const col = arrow.skipped ? c.muted : arrow.dir > 0 ? c.opt : c.grad;
        const from = xc(arrow.from), to = arrow.skipped ? xc(arrow.from) + arrow.dir * cellW * 0.85 : xc(p);
        if (Math.abs(to - from) > 1) {
          el("line", { x1: from, x2: to - Math.sign(to - from) * 5, y1: y, y2: y, stroke: col, "stroke-width": 2.5, "stroke-dasharray": arrow.skipped ? "3 3" : "", "stroke-opacity": arrow.skipped ? 0.8 : 1 }, root);
          const s = Math.sign(to - from);
          el("path", { d: `M${to},${y} L${to - 7 * s},${y - 4.5} L${to - 7 * s},${y + 4.5} Z`, fill: col, "fill-opacity": arrow.skipped ? 0.8 : 1 }, root);
        }
        if (arrow.skipped) el("circle", { cx: xc(arrow.from), cy: y, r: 6, fill: c.bg, stroke: c.muted, "stroke-width": 2 }, root);
      }
      if (!arrow || !arrow.skipped) {
        el("circle", { cx: xc(p), cy: y, r: 6, fill: isMem(p) ? c.opt : c.bg, stroke: isMem(p) ? c.opt : c.muted, "stroke-width": 2 }, root);
      }
      if (opts.strip && opts.history && opts.history.length) {
        const h = opts.history, n = opts.historyLen;
        const w = stripW / n;
        for (let t = 0; t < h.length; t++) {
          const q = h[t][i];
          const xx = sx0 + stripW - (h.length - t) * w;
          if (isMem(q)) el("rect", { x: xx, y: y - rowH / 2 + 3, width: w + 0.3, height: rowH - 6, fill: c.opt, "fill-opacity": 0.35 + (0.65 * (q - 6)) / 4 }, root);
          else if (q >= 4) el("rect", { x: xx, y: y - rowH / 2 + 3, width: w + 0.3, height: rowH - 6, fill: c.muted, "fill-opacity": 0.12 + 0.1 * (q - 4) }, root);
        }
      }
      if (opts.strip) el("rect", { x: sx0, y: y - rowH / 2 + 3, width: stripW, height: rowH - 6, fill: "none", stroke: c.grid }, root);
    });
    return { W, H };
  }

  // ---------------------------------------------------------------- Widget 1: the book's worked example
  function walkthroughWidget(root) {
    const names = namesOf(VEHICLES.features);
    const shortNames = namesOf(VEHICLES.short);
    // The book's choices for which random updates happen (Granmo, Ch. 1, Figs 1.6–1.11).
    // skip: literal indices whose randomized update is skipped in that step.
    const SCRIPT = [
      { v: 0, skip: [5, 7], mem: 0.1, forget: 0.9 },      // car #1: Recognize; ¬Wings, ¬Yellow skip their increment
      { v: 1, skip: [0, 5], mem: 0.1, forget: 0.9 },      // car #2: Erase; Four Wheels, ¬Wings skip their decrement
      { v: 2, skip: [5, 9], mem: 0.1, forget: 0.9 },      // car #3: Recognize; ¬Wings, ¬Blue skip their increment
      { v: 3, skip: [], mem: 0.1, forget: 0.9 },          // plane #4: Reject (never random)
    ];
    // Build every frame: positions after the step, and one arrow per literal that the feedback touched.
    const frames = [{ pos: new Array(10).fill(5), arrows: null, fb: null }];
    for (const s of SCRIPT) {
      const prev = frames[frames.length - 1].pos, pos = prev.slice(), arrows = {};
      const lits = literalsOf(VEHICLES.X[s.v]);
      const own = VEHICLES.Y[s.v] === 1, fires = evaluate(prev, lits);
      let fb = null;
      const move = (i, dir, random) => {
        const skipped = random && s.skip.includes(i);
        const blocked = (dir > 0 && prev[i] >= DEPTH) || (dir < 0 && prev[i] <= 1);
        if (!skipped && !blocked) pos[i] += dir;
        arrows[i] = { from: prev[i], dir, skipped };
      };
      if (own && fires) { fb = "Recognize"; lits.forEach((v, i) => move(i, v ? +1 : -1, true)); }
      else if (own) { fb = "Erase"; lits.forEach((v, i) => move(i, -1, true)); }
      else if (fires) { fb = "Reject"; lits.forEach((v, i) => { if (!v && !isMem(prev[i])) move(i, +1, false); }); }
      frames.push({ pos, arrows, fb, v: s.v, lits, prevRule: ruleText(prev, names, "Car") });
    }
    const CAPTION = [
      "A new rule for <b>Car</b>. Every literal starts at position 5: almost memorized, but still forgotten. The condition is empty, so the rule is True for every vehicle.",
      "Vehicle #1 is a <b>Car</b> and the empty rule is True → <b>Recognize Feedback</b>. True literals step up with probability 0.1 (Memorize Value); the book lets Four Wheels, Transports People and Blue succeed, while ¬Wings and ¬Yellow are skipped (dashed). False literals step down with probability 0.9 (Forget Value); here all of them do.",
      "Vehicle #2 is a yellow <b>Car</b>, but the rule now demands Blue, so it is False → <b>Erase Feedback</b>: every literal steps down with probability 0.9. Four Wheels and ¬Wings happen to be skipped. Only Four Wheels stays memorized.",
      "Vehicle #3 is another yellow <b>Car</b>. The rule 'Four Wheels' matches → <b>Recognize Feedback</b>. Four Wheels, Transports People and Yellow step up (¬Wings and ¬Blue are skipped); every False literal steps down. Colour literals keep getting pushed around: no colour is frequent enough to stick.",
      "Vehicle #4 is a <b>Plane</b> with four wheels that transports people, so the Car rule wrongly matches → <b>Reject Feedback</b>. Every False literal on the Forgotten side steps up, with no randomness. ¬Wings crosses from 5 to 6, and the rule now rejects this plane.",
    ];

    const top = box(root, "tmf-vehicle");
    const chart = svg(600, 300);
    chart.style.maxWidth = "640px";
    chart.style.margin = "0 auto";
    root.appendChild(chart);
    const ctrl = control(root, `<button data-a="prev">◀ Prev</button><button data-a="next">Next ▶</button><button data-a="reset">Reset</button><span class="tmf-step"></span>`);
    const cap = box(root, "viz-hint tmf-caption");
    const out = box(root, "viz-readout");
    out.style.whiteSpace = "pre-line";
    const legend = box(root, "viz-hint");
    legend.innerHTML = "Arrows: <span style=\"color: var(--viz-opt)\">green = one step towards memorized</span> · <span style=\"color: var(--viz-grad)\">red = one step towards forgotten</span> · dashed grey = the random draw skipped this update.";
    let k = 0;

    function draw() {
      const f = frames[k];
      const narrow = root.clientWidth < 480;
      memoryChart(chart, narrow ? shortNames : names, f.pos, { narrow, arrows: f.arrows, rowH: 24 });
      if (f.v === undefined) {
        top.innerHTML = "<b>Start:</b> no vehicle seen yet.";
      } else {
        const own = VEHICLES.Y[f.v] === 1;
        const chips = VEHICLES.features.map((n, j) => `<span class="tmf-chip ${VEHICLES.X[f.v][j] ? "on" : ""}">${VEHICLES.X[f.v][j] ? "" : "¬"}${n}</span>`).join(" ");
        top.innerHTML = `<b>Vehicle #${f.v + 1}: ${own ? "Car" : "Plane"}</b> ${chips} <span class="tmf-fb tmf-${f.fb.toLowerCase()}">${f.fb} Feedback</span>`;
      }
      ctrl.querySelector(".tmf-step").textContent = `step ${k} of ${frames.length - 1}`;
      cap.innerHTML = CAPTION[k];
      out.textContent = (f.prevRule ? `before: ${f.prevRule}\n` : "") + `rule:   ${ruleText(f.pos, names, "Car")}`;
    }
    ctrl.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (a === "next") k = Math.min(frames.length - 1, k + 1);
      if (a === "prev") k = Math.max(0, k - 1);
      if (a === "reset") k = 0;
      if (a) draw();
    });
    onThemeChange(draw);
    window.addEventListener("resize", draw);
    draw();
  }

  // ---------------------------------------------------------------- Widget 2: train one rule
  function trainWidget(root) {
    const DATA = {
      bc: Object.assign(bcData(), { target: 1, own: "Recurrence", other: "no recurrence", unit: "patient" }),
      car: Object.assign({}, VEHICLES, { target: 1, own: "Car", other: "Plane", unit: "vehicle" }),
      plane: Object.assign({}, VEHICLES, { target: 0, own: "Plane", other: "Car", unit: "vehicle" }),
    };
    const HIST = 200;
    const chart = svg(600, 300);
    chart.style.maxWidth = "680px";
    chart.style.margin = "0 auto";
    root.appendChild(chart);
    const c1 = control(root, `
      <label>data <select data-k="data">
        <option value="bc">breast cancer: Recurrence rule</option>
        <option value="car">vehicles: Car rule</option>
        <option value="plane">vehicles: Plane rule</option></select></label>
      <label>seed <input type="number" data-k="seed" value="13" min="1" max="999" style="width: 4.5em"></label>`);
    const c2 = control(root, `
      <label><input type="checkbox" data-k="rec" checked> Recognize</label>
      <label><input type="checkbox" data-k="erase" checked> Erase</label>
      <label><input type="checkbox" data-k="rej" checked> Reject</label>
      <label>Forget Value <input type="range" data-k="forget" min="0.05" max="0.95" step="0.05" value="0.8"> <span class="val"></span></label>`);
    const c3 = control(root, `<button data-a="1">+1 step</button><button data-a="100">+100</button><button data-a="1000">+1000</button><button data-a="reset">Reset</button>`);
    const out = box(root, "viz-readout");
    out.style.whiteSpace = "pre-line";
    const hint = box(root, "viz-hint");
    hint.textContent = "Ticks and the slider act from the next step on, so you can switch Reject on in the middle of a run. Changing the data or the seed starts again.";
    const q = (k) => root.querySelector(`[data-k="${k}"]`);

    let D, L, own, other, names, pos, t, rnd, hist, counts, last;

    function reset() {
      D = DATA[q("data").value];
      L = D.X.map(literalsOf);
      own = []; other = [];
      D.Y.forEach((y, i) => (y === D.target ? own : other).push(i));
      names = namesOf(D.features);
      pos = new Array(L[0].length).fill(5);
      t = 0;
      rnd = mulberry32(Math.max(1, Math.floor(+q("seed").value) || 1));
      hist = [pos.slice()];
      counts = { Recognize: 0, Erase: 0, Reject: 0, none: 0 };
      last = null;
      draw();
    }

    function step() {
      const forget = +q("forget").value, mem = 1 - forget;
      const pool = rnd() < 0.5 ? own : other;
      const k = pool[Math.floor(rnd() * pool.length)];
      const lits = L[k], isOwn = D.Y[k] === D.target, fires = evaluate(pos, lits);
      let fb = "none";
      if (isOwn) {
        if (fires && q("rec").checked) {
          fb = "Recognize";
          for (let i = 0; i < lits.length; i++) {
            if (lits[i]) { if (rnd() <= mem && pos[i] < DEPTH) pos[i] += 1; }
            else if (rnd() <= forget && pos[i] > 1) pos[i] -= 1;
          }
        } else if (!fires && q("erase").checked) {
          fb = "Erase";
          for (let i = 0; i < lits.length; i++) if (rnd() <= forget && pos[i] > 1) pos[i] -= 1;
        }
      } else if (fires && q("rej").checked) {
        fb = "Reject";
        for (let i = 0; i < lits.length; i++) if (!lits[i] && !isMem(pos[i])) pos[i] += 1;
      }
      counts[fb] += 1;
      t += 1;
      last = { k, isOwn, fb };
      hist.push(pos.slice());
      if (hist.length > HIST) hist.shift();
    }

    function stats() {
      let match = 0, tp = 0;
      L.forEach((lits, i) => { if (evaluate(pos, lits)) { match += 1; if (D.Y[i] === D.target) tp += 1; } });
      return { match, tp };
    }

    function draw() {
      const narrow = root.clientWidth < 480;
      memoryChart(chart, narrow ? namesOf(D.short) : names, pos, { narrow, strip: true, history: hist, historyLen: HIST, rowH: L[0].length > 10 ? 18 : 22 });
      q("forget").parentNode.querySelector(".val").textContent = `${(+q("forget").value).toFixed(2)} (Memorize ${(1 - q("forget").value).toFixed(2)})`;
      const { match, tp } = stats();
      const nOwn = own.length;
      const prec = match ? (tp / match).toFixed(2) : "–";
      const lastTxt = last ? `${D.unit} #${last.k + 1} (${last.isOwn ? D.own : D.other}) → ${last.fb === "none" ? "no feedback" : last.fb + " Feedback"}` : "none yet";
      out.textContent =
        `step ${t} · last: ${lastTxt}\n` +
        `${ruleText(pos, names, D.own)}\n` +
        `matches ${match} of ${L.length} · coverage ${tp}/${nOwn} = ${(tp / nOwn).toFixed(2)} · precision ${match ? `${tp}/${match} = ${prec}` : "– (matches nothing)"}\n` +
        `feedback so far: Recognize ${counts.Recognize} · Erase ${counts.Erase} · Reject ${counts.Reject} · none ${counts.none}`;
    }

    c1.addEventListener("change", reset);
    c2.addEventListener("input", draw);
    c3.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (!a) return;
      if (a === "reset") { reset(); return; }
      for (let i = 0; i < +a; i++) step();
      draw();
    });
    onThemeChange(draw);
    window.addEventListener("resize", draw);
    reset();
  }

  // ---------------------------------------------------------------- Widget 3: the literal game
  const N8 = 8, H8 = 4;   // Ch. 2: memory depth 8 per literal, memorized from position 5
  // Next-position distribution of one literal (Ch. 2: Memorize Value 1.0, Forget Value 1/s).
  function litStep(p, lit, ruleTrue, own, s) {
    if (own) {
      if (ruleTrue && lit) return [[Math.min(p + 1, N8), 1]];
      if (p === 1) return [[1, 1]];
      return [[p - 1, 1 / s], [p, 1 - 1 / s]];
    }
    if (ruleTrue && !lit && p <= H8) return [[p + 1, 1]];
    return [[p, 1]];
  }
  function stationary(pY, s) {
    const n = N8 * N8, M = [];
    for (let a = 1; a <= N8; a++) for (let b = 1; b <= N8; b++) {
      const row = new Float64Array(n);
      for (const l1 of [0, 1]) for (const l2 of [0, 1]) for (const own of [1, 0]) {
        const py = pY[`${l1}${l2}`];
        const pr = 0.25 * (own ? py : 1 - py);
        const ruleTrue = (l1 || a <= H8) && (l2 || b <= H8);
        for (const [na, q1] of litStep(a, l1, ruleTrue, own, s))
          for (const [nb, q2] of litStep(b, l2, ruleTrue, own, s)) row[(na - 1) * N8 + (nb - 1)] += pr * q1 * q2;
      }
      M.push(row);
    }
    let pi = new Float64Array(n).fill(1 / n);
    for (let it = 0; it < 20000; it++) {
      const nx = new Float64Array(n);
      for (let i = 0; i < n; i++) { const w = pi[i]; if (w === 0) continue; const r = M[i]; for (let j = 0; j < n; j++) nx[j] += w * r[j]; }
      let diff = 0;
      for (let j = 0; j < n; j++) diff = Math.max(diff, Math.abs(nx[j] - pi[j]));
      pi = nx;
      if (diff < 1e-13) break;
    }
    return pi;
  }

  function gameWidget(root) {
    const SCEN = [
      { label: "1: one useful literal", pY: { "11": 0.9, "01": 0.9, "10": 0.5, "00": 0.2 }, s: 1 },
      { label: "2: two good literals", pY: { "11": 0.99, "01": 0.9, "10": 0.9, "00": 0.1 }, s: 1 },
      { label: "3: pick the better one", pY: { "11": 0.97, "01": 0.9, "10": 0.7, "00": 0.1 }, s: 1 },
      { label: "4: combine them (s = 4)", pY: { "11": 0.99, "01": 0.9, "10": 0.9, "00": 0.1 }, s: 4 },
    ];
    const plots = box(root, "viz-plots");
    const left = box(plots, "tmf-game");
    const right = box(plots, "tmf-heat");

    // --- left: the game for one observation
    const obsCtrl = control(left, `
      <span>Observation:</span>
      <button data-o="l1">L1 = True</button><button data-o="l2">L2 = True</button><button data-o="y">class Y</button>`);
    const table = box(left, "tmf-gametable");
    const gameHint = box(left, "viz-hint");
    gameHint.innerHTML = "Each cell is one choice of the two literals. <span style=\"color: var(--viz-opt)\">⇑ memorize</span> · <span style=\"color: var(--viz-grad)\">↓ forget</span> · – no change.";
    const obs = { l1: 0, l2: 1, y: 1 };

    function outcome(m1, m2) {
      const ruleTrue = (obs.l1 || !m1) && (obs.l2 || !m2);
      const cond = m1 && m2 ? "L1 ∧ L2" : m1 ? "L1" : m2 ? "L2" : "∅ (empty)";
      let fb, a1, a2;
      if (obs.y) {
        if (ruleTrue) { fb = "Recognize"; a1 = obs.l1 ? "up" : "down"; a2 = obs.l2 ? "up" : "down"; }
        else { fb = "Erase"; a1 = "down"; a2 = "down"; }
      } else if (ruleTrue) {
        fb = "Reject"; a1 = !obs.l1 && !m1 ? "up" : "none"; a2 = !obs.l2 && !m2 ? "up" : "none";
      } else { fb = "no feedback"; a1 = "none"; a2 = "none"; }
      return { cond, ruleTrue, fb, a1, a2 };
    }
    const sym = (a) => (a === "up" ? '<span class="tmf-up">⇑</span>' : a === "down" ? '<span class="tmf-down">↓</span>' : '<span class="tmf-none">–</span>');

    function drawGame() {
      obsCtrl.querySelector('[data-o="l1"]').textContent = `L1 = ${obs.l1 ? "True" : "False"}`;
      obsCtrl.querySelector('[data-o="l2"]').textContent = `L2 = ${obs.l2 ? "True" : "False"}`;
      obsCtrl.querySelector('[data-o="y"]').textContent = obs.y ? "class Y (own)" : "class not-Y (other)";
      let h = '<table class="tmf-grid"><tr><th></th><th>L2 Forgotten</th><th>L2 Memorized</th></tr>';
      for (const m1 of [0, 1]) {
        h += `<tr><th>L1 ${m1 ? "Memorized" : "Forgotten"}</th>`;
        for (const m2 of [0, 1]) {
          const o = outcome(m1, m2);
          h += `<td><div class="tmf-cond">${o.cond} = ${o.ruleTrue ? "True" : "False"}</div><div class="tmf-fbname">${o.fb}</div><div class="tmf-arrows">L1 ${sym(o.a1)} &nbsp; L2 ${sym(o.a2)}</div></td>`;
        }
        h += "</tr>";
      }
      table.innerHTML = h + "</table>";
    }
    obsCtrl.addEventListener("click", (e) => {
      const o = e.target.dataset.o;
      if (o) { obs[o] = 1 - obs[o]; drawGame(); }
    });

    // --- right: the long-run outcome (stationary distribution over both memory positions)
    const sc = control(right, SCEN.map((s, i) => `<button data-s="${i}">${s.label}</button>`).join(""));
    const heat = svg(300, 300);
    heat.style.maxWidth = "360px";
    heat.style.margin = "8px auto 0";
    right.appendChild(heat);
    const sCtrl = control(right, `<label>s <input type="range" min="1" max="8" step="0.5" value="1"> <span class="val"></span></label>`);
    const sIn = sCtrl.querySelector("input");
    const out = box(right, "viz-readout");
    out.style.whiteSpace = "pre-line";
    let cur = 0;

    function drawHeat() {
      const c = colors();
      const scn = SCEN[cur], s = +sIn.value;
      sCtrl.querySelector(".val").textContent = `${s.toFixed(1)} (Forget Value 1/s = ${(1 / s).toFixed(2)})`;
      sc.querySelectorAll("button").forEach((b, i) => { b.style.borderColor = i === cur ? c.model : ""; });
      const pi = stationary(scn.pY, s);
      heat.innerHTML = "";
      const x0 = 46, y0 = 40, cs = 30;
      el("text", { x: x0 + 2 * cs, y: 12, "text-anchor": "middle" }, heat).textContent = "L2 Forgotten";
      el("text", { x: x0 + 6 * cs, y: 12, "text-anchor": "middle", style: `fill: ${c.opt}` }, heat).textContent = "L2 Memorized";
      el("text", { x: 12, y: y0 + 2 * cs, "text-anchor": "middle", transform: `rotate(-90 12 ${y0 + 2 * cs})` }, heat).textContent = "L1 Forg.";
      el("text", { x: 12, y: y0 + 6 * cs, "text-anchor": "middle", transform: `rotate(-90 12 ${y0 + 6 * cs})`, style: `fill: ${c.opt}` }, heat).textContent = "L1 Mem.";
      let mx = 0;
      for (const v of pi) mx = Math.max(mx, v);
      for (let a = 0; a < N8; a++) {
        el("text", { x: x0 - 6, y: y0 + a * cs + cs / 2 + 4, "text-anchor": "end", style: "font-size: 9px" }, heat).textContent = a + 1;
        el("text", { x: x0 + a * cs + cs / 2, y: y0 - 6, "text-anchor": "middle", style: "font-size: 9px" }, heat).textContent = a + 1;
        for (let b = 0; b < N8; b++) {
          const v = pi[a * N8 + b];
          el("rect", { x: x0 + b * cs, y: y0 + a * cs, width: cs, height: cs, fill: c.loss, "fill-opacity": 0.04 + 0.9 * (v / mx), stroke: c.bg }, heat);
          if (v >= 0.005) el("text", { x: x0 + b * cs + cs / 2, y: y0 + a * cs + cs / 2 + 4, "text-anchor": "middle", style: `font-size: 9px; ${v / mx > 0.5 ? "fill: " + c.bg : textStyle}` }, heat).textContent = v.toFixed(2).replace(/^0/, "");
        }
      }
      el("line", { x1: x0 + 4 * cs, x2: x0 + 4 * cs, y1: y0 - 2, y2: y0 + 8 * cs + 2, stroke: c.text, "stroke-width": 2 }, heat);
      el("line", { x1: x0 - 2, x2: x0 + 8 * cs + 2, y1: y0 + 4 * cs, y2: y0 + 4 * cs, stroke: c.text, "stroke-width": 2 }, heat);
      heat.setAttribute("viewBox", `0 0 ${x0 + 8 * cs + 8} ${y0 + 8 * cs + 8}`);
      const quad = (r0, c0) => { let t = 0; for (let a = r0; a < r0 + H8; a++) for (let b = c0; b < c0 + H8; b++) t += pi[a * N8 + b]; return t; };
      const p = scn.pY;
      out.textContent =
        `P(Y | L1 L2) = ${p["11"]} · P(Y | ¬L1 L2) = ${p["01"]} · P(Y | L1 ¬L2) = ${p["10"]} · P(Y | ¬L1 ¬L2) = ${p["00"]}\n` +
        `time spent with rule: empty ${quad(0, 0).toFixed(3)} · L2 only ${quad(0, H8).toFixed(3)} · L1 only ${quad(H8, 0).toFixed(3)} · L1 ∧ L2 ${quad(H8, H8).toFixed(3)}`;
    }
    sc.addEventListener("click", (e) => {
      const i = e.target.dataset.s;
      if (i === undefined) return;
      cur = +i;
      sIn.value = SCEN[cur].s;
      drawHeat();
    });
    sIn.addEventListener("input", drawHeat);
    onThemeChange(() => { drawGame(); drawHeat(); });
    drawGame();
    drawHeat();
  }

  const mounts = { "viz-walkthrough": walkthroughWidget, "viz-train": trainWidget, "viz-game": gameWidget };
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
