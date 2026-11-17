// Interactive widgets for the "What is a Tsetlin machine?" lesson.
// Three widgets, each mounted on a <div> with a matching id:
//   #viz-rule   – build a rule from literals and see which vehicles it matches (Granmo, Ch. 1, Table 1.1)
//   #viz-bool   – turn a numeric feature (tumour size) into Boolean features with thresholds
//   #viz-rules  – read a small rule set learned on the UCI Breast Cancer data and watch it vote
// Everything here is deterministic: the learned rules were produced offline by a seeded numpy run.
document.addEventListener("DOMContentLoaded", function () {
  const { colors, onThemeChange, linear, el, svg } = window.Viz;
  const textStyle = "fill: var(--viz-text)";

  // Lesson-scoped styles (chips and small tables inside widgets)
  const css = document.createElement("style");
  css.textContent = `
    .tm-chips { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 6px; margin-top: 8px; }
    .tm-chip { font: inherit; font-size: 0.72rem; padding: 5px 8px; border-radius: 8px; cursor: pointer; text-align: left;
      border: 1.5px solid var(--viz-data); background: var(--viz-bg); color: var(--viz-text); }
    .tm-chip.on { border-color: var(--viz-opt); background: color-mix(in srgb, var(--viz-opt) 22%, var(--viz-bg)); font-weight: 700; }
    .tm-chip.off-feature { border-style: dashed; border-color: var(--viz-border); color: var(--viz-muted); }
    .tm-rule { margin-top: 10px; padding: 8px 10px; border-left: 3px solid var(--viz-model); background: var(--viz-bg);
      border-radius: 4px; font-size: 0.8rem; line-height: 1.5; }
    .tm-rule b.kw { color: var(--viz-model); }
    .tm-wrap { overflow-x: auto; margin-top: 10px; }
    table.tm-table { border-collapse: collapse; width: 100%; font-size: 0.72rem; background: var(--viz-bg); }
    table.tm-table th, table.tm-table td { padding: 4px 5px; text-align: center; border-bottom: 1px solid var(--viz-border); }
    table.tm-table th { color: var(--viz-muted); font-weight: 600; }
    @media (max-width: 480px) { table.tm-table { font-size: 0.66rem; } table.tm-table th, table.tm-table td { padding: 4px 3px; } }
    table.tm-table td.lft, table.tm-table th.lft { text-align: left; }
    table.tm-table tr.tp td { background: color-mix(in srgb, var(--viz-opt) 16%, transparent); }
    table.tm-table tr.fp td { background: color-mix(in srgb, var(--viz-grad) 16%, transparent); }
    table.tm-table td.bad { box-shadow: inset 0 0 0 2px var(--viz-grad); border-radius: 4px; }
    table.tm-table td.used { font-weight: 700; color: var(--viz-text); }
    .tm-status-tp { color: var(--viz-opt); font-weight: 700; }
    .tm-status-fp { color: var(--viz-grad); font-weight: 700; }
    .tm-status-miss { color: var(--viz-muted); }
    .tm-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 8px; margin-top: 10px; }
    .tm-card { background: var(--viz-bg); border: 1.5px solid var(--viz-border); border-radius: 8px; padding: 8px 10px; font-size: 0.74rem; line-height: 1.45; }
    .tm-card.fire { border-color: var(--viz-model); box-shadow: 0 0 0 1px var(--viz-model); }
    .tm-card .tag { font-weight: 700; }
    .tm-card .tag.for { color: var(--viz-loss); }
    .tm-card .tag.against { color: var(--viz-data); }
    .tm-card .meta { color: var(--viz-muted); font-size: 0.68rem; margin-top: 3px; }
    .tm-card .state { float: right; font-weight: 700; color: var(--viz-muted); }
    .tm-card.fire .state { color: var(--viz-model); }
    .tm-verdict { margin-top: 10px; font-size: 0.8rem; }
    .tm-legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 0.7rem; color: var(--viz-muted); margin: 2px 0 6px; }
    .tm-legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
    .tm-verdict b { color: var(--viz-loss); }
  `;
  document.head.appendChild(css);

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

  // ---------------------------------------------------------------- Widget 1: rule builder
  const VF = ["Four Wheels", "Transports People", "Wings", "Yellow", "Blue"];
  const VSHORT = ["wheels", "people", "wings", "yellow", "blue"];
  // Granmo, Ch. 1, Table 1.1 (1 = True). Last entry: type.
  const VEHICLES = [
    [1, 1, 0, 0, 1, "Car"], [1, 1, 0, 1, 0, "Car"], [1, 1, 0, 1, 0, "Car"],
    [1, 1, 1, 0, 1, "Plane"], [1, 0, 1, 1, 0, "Plane"], [0, 1, 1, 0, 1, "Plane"],
  ];
  // literal id: 2k = feature k, 2k+1 = NOT feature k
  const litName = (l) => (l % 2 ? "NOT " : "") + VF[l >> 1];
  const litValue = (v, l) => (l % 2 ? 1 - v[l >> 1] : v[l >> 1]);

  function ruleWidget(root) {
    const cls = control(root, `Rule predicts: <button data-c="Car">🚗 Car</button><button data-c="Plane">✈️ Plane</button>`);
    const chips = document.createElement("div");
    chips.className = "tm-chips";
    root.appendChild(chips);
    const chipEls = [];
    for (let k = 0; k < VF.length; k++) {
      for (const neg of [0, 1]) {
        const b = document.createElement("button");
        b.className = "tm-chip";
        b.dataset.l = 2 * k + neg;
        b.textContent = litName(2 * k + neg);
        chips.appendChild(b);
        chipEls.push(b);
      }
    }
    hint(root, "Click literals to add them to the rule (green = in the rule). The rule is an AND: every chosen literal must be True.");
    const presets = control(root, `Try: <button data-p="empty">empty rule</button><button data-p="fw">Four Wheels</button><button data-p="fwtp">+ Transports People</button><button data-p="car">+ NOT Wings</button><button data-p="blue">… AND Blue</button><button data-p="plane">Wings → Plane</button><button data-p="contra">contradiction</button>`);
    const sentence = document.createElement("div");
    sentence.className = "tm-rule";
    root.appendChild(sentence);
    const wrap = document.createElement("div");
    wrap.className = "tm-wrap";
    root.appendChild(wrap);
    const out = readout(root);

    let rule = new Set([0, 2, 5]), target = "Car";
    const PRESETS = {
      empty: [[], "Car"], fw: [[0], "Car"], fwtp: [[0, 2], "Car"], car: [[0, 2, 5], "Car"],
      blue: [[0, 2, 8], "Car"], plane: [[4], "Plane"], contra: [[0, 1], "Car"],
    };

    function draw() {
      const lits = [...rule].sort((a, b) => a - b);
      chipEls.forEach((b) => b.classList.toggle("on", rule.has(+b.dataset.l)));
      cls.querySelectorAll("button").forEach((b) => (b.style.borderColor = b.dataset.c === target ? "var(--viz-model)" : ""));
      const cond = lits.length ? lits.map((l) => `<b>${litName(l)}</b>`).join(` <b class="kw">AND</b> `) : "<i>(no literals: the empty rule)</i>";
      sentence.innerHTML = `<b class="kw">IF</b> ${cond} <b class="kw">THEN</b> <b>${target}</b>`;

      let html = `<table class="tm-table"><thead><tr><th class="lft">vehicle</th>${VSHORT.map((s) => `<th>${s}</th>`).join("")}<th>rule</th></tr></thead><tbody>`;
      let matches = 0, tp = 0, fp = 0;
      const nClass = VEHICLES.filter((v) => v[5] === target).length;
      VEHICLES.forEach((v, i) => {
        const failing = lits.filter((l) => !litValue(v, l));
        const match = failing.length === 0;
        const mine = v[5] === target;
        if (match) { matches++; if (mine) tp++; else fp++; }
        const rowCls = match ? (mine ? "tp" : "fp") : "";
        const status = match
          ? (mine ? `<span class="tm-status-tp">✓</span>` : `<span class="tm-status-fp">✗</span>`)
          : (mine ? `<span class="tm-status-miss">miss</span>` : `<span class="tm-status-miss">–</span>`);
        const cells = VF.map((_, k) => {
          const used = lits.some((l) => l >> 1 === k);
          const bad = failing.some((l) => l >> 1 === k);
          return `<td class="${used ? "used" : ""} ${bad ? "bad" : ""}">${v[k] ? "●" : "·"}</td>`;
        }).join("");
        html += `<tr class="${rowCls}"><td class="lft">${i + 1}. ${v[5]}</td>${cells}<td>${status}</td></tr>`;
      });
      html += "</tbody></table>";
      wrap.innerHTML = html;
      const prec = matches ? `${tp}/${matches}` : "n/a (matches nothing)";
      out.textContent = `matches ${matches} vehicle${matches === 1 ? "" : "s"} · true positives ${tp} · false positives ${fp}\n` +
        `coverage ${tp}/${nClass} ${target}s · precision ${prec}\n` +
        `● = True, · = False. ✓ green row: matches its own class · ✗ red row: matches the other class · miss: a ${target} it doesn't match.\nA red box marks a literal that stops the rule from matching that row.`;
    }

    chips.addEventListener("click", (e) => {
      const l = e.target.dataset.l;
      if (l === undefined) return;
      rule.has(+l) ? rule.delete(+l) : rule.add(+l);
      draw();
    });
    cls.addEventListener("click", (e) => { if (e.target.dataset.c) { target = e.target.dataset.c; draw(); } });
    presets.addEventListener("click", (e) => {
      const p = PRESETS[e.target.dataset.p];
      if (!p) return;
      rule = new Set(p[0]); target = p[1];
      draw();
    });
    draw();
  }

  // ---------------------------------------------------------------- Widget 2: booleanisation playground
  // UCI Breast Cancer (Ljubljana) data: tumour size bins [lower edge in mm, recurrence, no recurrence]
  const SIZE_BINS = [
    [0, 1, 7], [5, 0, 4], [10, 1, 27], [15, 7, 23], [20, 16, 34], [25, 18, 36],
    [30, 25, 35], [35, 7, 12], [40, 6, 16], [45, 1, 2], [50, 3, 5],
  ];

  function boolWidget(root) {
    const legend = document.createElement("div");
    legend.className = "tm-legend";
    legend.innerHTML = `<span><i style="background: var(--viz-data)"></i>no recurrence</span><span><i style="background: var(--viz-grad)"></i>recurrence</span><span><i style="background: var(--viz-model)"></i>threshold</span><span>▲ the example patient</span>`;
    root.appendChild(legend);
    const holder = document.createElement("div");
    root.appendChild(holder);
    const ctrl = control(root, `
      <label>threshold 1 <input type="range" min="5" max="50" step="5" value="20" data-i="0"> <span></span></label>
      <label>threshold 2 <input type="range" min="5" max="50" step="5" value="30" data-i="1"> <span></span></label>
      <label>threshold 3 <input type="range" min="5" max="50" step="5" value="40" data-i="2"> <span></span></label>`);
    const mode = control(root, `Encoding: <button data-m="thermo">thermometer (size ≥ t)</button><button data-m="onehot">one interval each</button>
      <label>use <select data-k><option value="1">1</option><option value="2">2</option><option value="3" selected>3</option></select> thresholds</label>`);
    const patient = control(root, `<label>a patient with tumour size <input type="range" min="0" max="50" step="5" value="25"> <span></span></label>`);
    const wrap = document.createElement("div");
    wrap.className = "tm-wrap";
    root.appendChild(wrap);
    const out = readout(root);
    const sliders = [...ctrl.querySelectorAll("input")];
    const kSel = mode.querySelector("select");
    const pSlider = patient.querySelector("input");
    let enc = "thermo";

    function thresholds() {
      const k = +kSel.value;
      sliders.forEach((s, i) => (s.closest("label").style.opacity = i < k ? 1 : 0.35));
      return [...new Set(sliders.slice(0, k).map((s) => +s.value))].sort((a, b) => a - b);
    }

    // Boolean features as [name, test(lowerEdge)]
    function features(ts) {
      if (enc === "thermo") return ts.map((t) => [`size ≥ ${t} mm`, (s) => s >= t]);
      const edges = [0, ...ts, 60];
      return edges.slice(0, -1).map((a, i) => {
        const b = edges[i + 1];
        const name = i === 0 ? `size < ${b} mm` : b === 60 ? `size ≥ ${a} mm` : `${a}–${b - 1} mm`;
        return [name, (s) => s >= a && s < b];
      });
    }

    function draw() {
      const c = colors(), ts = thresholds(), feats = features(ts);
      sliders.forEach((s) => (s.nextElementSibling.textContent = s.value + " mm"));
      mode.querySelectorAll("button").forEach((b) => (b.style.borderColor = b.dataset.m === enc ? "var(--viz-model)" : ""));
      const narrow = root.clientWidth < 480;
      const W = narrow ? 340 : 600, H = narrow ? 240 : 250, M = { l: 34, r: 14, t: 28, b: 40 };
      holder.innerHTML = "";
      const s = svg(W, H);
      holder.appendChild(s);
      const x = linear(0, 55, M.l, W - M.r), y = linear(0, 60, H - M.b, M.t);
      for (let g = 0; g <= 60; g += 20) {
        el("line", { x1: M.l, x2: W - M.r, y1: y(g), y2: y(g), stroke: c.grid }, s);
        el("text", { x: M.l - 6, y: y(g) + 4, "text-anchor": "end" }, s).textContent = g;
      }
      const bw = x(5) - x(0) - 3;
      for (const [lo, rec, no] of SIZE_BINS) {
        const x0 = x(lo) + 1.5;
        el("rect", { x: x0, y: y(no), width: bw, height: y(0) - y(no), fill: c.data, "fill-opacity": 0.75 }, s);
        el("rect", { x: x0, y: y(no + rec), width: bw, height: y(no) - y(no + rec), fill: c.grad, "fill-opacity": 0.85 }, s);
      }
      for (let t = 0; t <= 55; t += narrow ? 10 : 5) el("text", { x: x(t), y: H - M.b + 14, "text-anchor": "middle" }, s).textContent = t;
      el("text", { x: (M.l + W - M.r) / 2, y: H - 8, "text-anchor": "middle" }, s).textContent = "tumour size (mm, lower edge of the recorded 5 mm range)";
      // thresholds
      ts.forEach((t) => {
        el("line", { x1: x(t), x2: x(t), y1: M.t - 6, y2: H - M.b, stroke: c.model, "stroke-width": 2, "stroke-dasharray": "5 3" }, s);
        el("text", { x: x(t), y: M.t - 10, "text-anchor": "middle", style: `fill: ${c.model}; font-weight: 700` }, s).textContent = `${t}`;
      });
      // example patient
      const pv = +pSlider.value;
      pSlider.nextElementSibling.textContent = `${pv}–${pv + 4} mm`;
      el("path", { d: `M${x(pv + 2.5)},${H - M.b + 18} l-6,8 h12 z`, fill: c.model }, s);

      // feature table
      let rows = "", total = 0, totalRec = 0;
      SIZE_BINS.forEach((b) => { total += b[1] + b[2]; totalRec += b[1]; });
      for (const [name, test] of feats) {
        let n = 0, r = 0;
        for (const [lo, rec, no] of SIZE_BINS) if (test(lo)) { n += rec + no; r += rec; }
        const rT = n ? ((100 * r) / n).toFixed(1) + "%" : "n/a";
        const rF = total - n ? ((100 * (totalRec - r)) / (total - n)).toFixed(1) + "%" : "n/a";
        const bit = test(pv) ? 1 : 0;
        rows += `<tr><td class="lft">${name}</td><td class="used">${bit ? "● True" : "· False"}</td><td>${n}</td><td>${rT}</td><td>${rF}</td></tr>`;
      }
      wrap.innerHTML = `<table class="tm-table"><thead><tr><th class="lft">Boolean feature</th><th>this patient</th><th>patients True</th><th>recurrence if True</th><th>if False</th></tr></thead><tbody>${rows}</tbody></table>`;
      const bits = feats.map(([, t]) => (t(pv) ? 1 : 0)).join(" ");
      out.textContent = `patient with a ${pv}–${pv + 4} mm tumour → bits [ ${bits} ]\n` +
        (enc === "thermo" ? "Thermometer: the bits fill up from the left as the size grows, like mercury in a thermometer." : "One interval each: exactly one bit is True (this is how the book booleanises each value).");
    }

    ctrl.addEventListener("input", draw);
    patient.addEventListener("input", draw);
    kSel.addEventListener("change", draw);
    mode.addEventListener("click", (e) => { if (e.target.dataset.m) { enc = e.target.dataset.m; draw(); } });
    window.addEventListener("resize", draw);
    onThemeChange(draw);
    draw();
  }

  // ---------------------------------------------------------------- Widget 3: a learned rule set
  const BC_NAMES = ["malignancy grade 3", "malignancy grade 1", "3 or more involved nodes", "6 or more involved nodes",
    "tumour through node capsule", "tumour 20 mm or larger", "tumour 30 mm or larger", "tumour 40 mm or larger",
    "had radiotherapy", "premenopausal", "aged 50 or over"];
  // Rules learned by a seeded numpy Tsetlin machine (3 rules per side, Memorize Value 0.75, Voting Margin 3, 60 epochs).
  // Literal k < 11 is feature k; literal k >= 11 is NOT feature k-11. side +1 votes Recurrence, -1 votes No recurrence.
  const BC_RULES = [
    { side: 1, lits: [4, 16] }, { side: 1, lits: [0, 4, 9, 19] }, { side: 1, lits: [3, 4] },
    { side: -1, lits: [20] }, { side: -1, lits: [11, 17, 18, 19] }, { side: -1, lits: [1, 14, 16] },
  ];
  // 286 patients: 11 feature bits, then recurrence (1/0), then "in the test set" (1/0)
  const PATIENTS = [
    "1000011001000", "0000010001001", "0000010001000", "0000000000100", "0000000001000", "0000000000100", "0000010001101", "0100010000100",
    "0000011101000", "0000010001000", "1000000001001", "0000010000101", "0100000000100", "1000010000100", "1000011001001", "0100011000101",
    "0000000001001", "1000011001101", "1000011000100", "0100011000100", "0000011100100", "0000000000101", "0000010001001", "0000011101101",
    "0000011001100", "0000010001000", "0100010001100", "1000010000100", "0000011101000", "0000011000100", "1000011100100", "0000000001101",
    "1000000001100", "0100000000100", "0100000000101", "0000011001001", "0000000000101", "0100000000100", "0000000001000", "0100011001000",
    "0100010000101", "0000010000101", "0100000000101", "0000000001001", "0100011100100", "0100011000100", "0000010001000", "0100010001100",
    "0100010001000", "0100010001001", "0000000000100", "0000010001000", "0100000001100", "1000010000101", "0100011100100", "0100011100100",
    "0100000000100", "0000000000101", "0100011000100", "0100000000100", "0000010001000", "0100000001001", "0100000000100", "0000011001001",
    "0100010001000", "0100000001001", "0100010001000", "1000010000100", "1000011000100", "0000011100101", "0100000000100", "0000010001001",
    "0000010000100", "0000000001100", "0000000001001", "0100000000100", "0000000000101", "0100010001000", "0000010001100", "0000010001000",
    "0000000000100", "0100000000100", "0000000000100", "0000000000100", "0100010001000", "1000011000100", "0000010000100", "0100000000100",
    "1000010000100", "0100011101000", "0000000001000", "1000010000100", "0000010000100", "0000010000101", "0100000001100", "0000011001000",
    "0000011000100", "0000000000101", "0100011001000", "0000010001001", "1000010000000", "1000011000100", "0000010001101", "0000010001000",
    "0000000001000", "0100011001001", "0000010001000", "0000011101000", "1000011001000", "0100011000101", "0100010000101", "0100000000100",
    "0000010001000", "0100000001001", "0000011001000", "0000010000100", "0100000001001", "1000010000001", "0100000001000", "0100000000100",
    "0100010000100", "0000000000100", "1000011100101", "0100011000100", "0100000000100", "0000000000101", "0011111001000", "0011110011000",
    "0000110001100", "0011111011001", "0011111011000", "1010111111000", "0011011001000", "1000011100100", "0000011010100", "0010010001000",
    "1010011111001", "0100000011001", "0000011111000", "0000011001001", "0010111100100", "0010110001101", "0100000000101", "0000011111000",
    "1011111100100", "0000010011000", "0000011110100", "0010111011100", "1000010001000", "1000011000100", "1011110000100", "1010111000101",
    "1011011000100", "1000000010100", "1000000000001", "0010011110100", "1010110000100", "0100011001101", "0100011000100", "0100011001000",
    "1000010011001", "0010111001000", "0000000010100", "0110010010101", "0110010010100", "0010010001000", "0010010001000", "0011111110001",
    "0000000001100", "0000011010000", "0010110011000", "0100000001000", "0011111000100", "0010110000100", "0010110011100", "0000011011001",
    "0000010000000", "0000000000101", "0010010011100", "1000010011000", "1000111011000", "1000111011000", "0100010011001", "1011011010101",
    "1011011010100", "0011010011000", "0100010000100", "0000000010101", "0000000001000", "0000110000100", "1011000011001", "0000010011000",
    "0011111000100", "0000000001000", "0000111111101", "0000011000101", "0110000001100", "0000000011000", "0000100010100", "0100010001100",
    "1000010000100", "0000000001110", "0100011101010", "0000011000110", "0000010001110", "0000000001010", "1000011000110", "0000010001110",
    "1000011001111", "0100011001010", "0000010001011", "0000010000111", "1000011001011", "0100010001110", "0000011100110", "0000010000011",
    "0000010000110", "0000000001010", "1000011000110", "0100000001010", "1000010001010", "0100011001011", "1000010010110", "1000010000111",
    "1010110011010", "1010010010010", "1011111001010", "1000011011110", "1010111100110", "0100011110110", "0011111101110", "0010011001010",
    "1010011011010", "0111000010111", "1000011010111", "1010110011110", "0000010001010", "0000010001011", "1000011001010", "0010110011011",
    "0010010010111", "1011100001010", "1011010010110", "1010110000110", "1011111011011", "0011011011010", "1011100011010", "1011111010110",
    "1011111000110", "0010110001010", "1000010001010", "0000011111011", "0100011101010", "1000011100110", "1000111001011", "1011111001011",
    "0100011011011", "0010110011010", "0011111010111", "1010011000110", "0010010000110", "1011110010010", "1000010000110", "0100010000110",
    "0100010000111", "1011111001011", "0010111001010", "1011110010110", "1000011001011", "0000010011010", "1011111000110", "1000010011110",
    "1000100001010", "0000111010111", "0010111010111", "1011110001011", "1011110011011", "1011100010110", "1011111001110", "1011111110110",
    "1011111110110", "0000011001010", "1000010011010", "0100010000110", "1010011000010", "1010011000110",
  ];
  const NF = BC_NAMES.length;

  function rulesWidget(root) {
    const pts = PATIENTS.map((s) => ({ x: [...s.slice(0, NF)].map(Number), y: +s[NF], test: +s[NF + 1] }));
    const testIdx = pts.map((p, i) => (p.test ? i : -1)).filter((i) => i >= 0);
    const lv = (x, l) => (l < NF ? x[l] : 1 - x[l - NF]);
    const fires = (r, x) => r.lits.length > 0 && r.lits.every((l) => lv(x, l) === 1);
    const voteSum = (x) => BC_RULES.reduce((s, r) => s + (fires(r, x) ? r.side : 0), 0);
    const english = (l) => (l < NF ? BC_NAMES[l] : `<u>not</u> ${BC_NAMES[l - NF]}`);

    const nav = control(root, `<button data-a="prev">◀ previous</button><button data-a="next">next test patient ▶</button><span class="pos"></span>`);
    const chips = document.createElement("div");
    chips.className = "tm-chips";
    root.appendChild(chips);
    const chipEls = BC_NAMES.map((n, k) => {
      const b = document.createElement("button");
      b.className = "tm-chip";
      b.dataset.k = k;
      chips.appendChild(b);
      return b;
    });
    hint(root, "Green chips are this patient's True features. Click a chip to flip it and see which rules react.");
    const cards = document.createElement("div");
    cards.className = "tm-cards";
    root.appendChild(cards);
    const verdict = document.createElement("div");
    verdict.className = "tm-verdict";
    root.appendChild(verdict);
    const out = readout(root);

    // stats over all 286 patients, computed from the data
    const ruleStats = BC_RULES.map((r) => {
      let n = 0, rec = 0;
      pts.forEach((p) => { if (fires(r, p.x)) { n++; rec += p.y; } });
      return [n, rec];
    });
    let correct = 0;
    testIdx.forEach((i) => { if ((voteSum(pts[i].x) > 0 ? 1 : 0) === pts[i].y) correct++; });

    let cur = 0, x = pts[testIdx[0]].x.slice(), edited = false;

    function load(k) {
      cur = (k + testIdx.length) % testIdx.length;
      x = pts[testIdx[cur]].x.slice();
      edited = false;
      draw();
    }

    function draw() {
      const p = pts[testIdx[cur]];
      nav.querySelector(".pos").textContent = `test patient ${cur + 1} of ${testIdx.length}${edited ? " (edited)" : ""}`;
      chipEls.forEach((b, k) => {
        b.classList.toggle("on", x[k] === 1);
        b.classList.toggle("off-feature", x[k] === 0);
        b.textContent = (x[k] ? "● " : "· ") + BC_NAMES[k];
      });
      cards.innerHTML = BC_RULES.map((r, j) => {
        const f = fires(r, x);
        const cond = r.lits.map(english).join(" <b>and</b> ");
        const tag = r.side > 0 ? `<span class="tag for">vote Recurrence</span>` : `<span class="tag against">vote No recurrence</span>`;
        const [n, rec] = ruleStats[j];
        return `<div class="tm-card ${f ? "fire" : ""}"><span class="state">${f ? "matches ✓" : "silent"}</span>` +
          `<b>Rule ${j + 1}.</b> If ${cond}, then ${tag}.` +
          `<div class="meta">matches ${n} of 286 patients; ${rec} of them had a recurrence</div></div>`;
      }).join("");
      const v = voteSum(x);
      const forV = BC_RULES.filter((r) => r.side > 0 && fires(r, x)).length;
      const agV = BC_RULES.filter((r) => r.side < 0 && fires(r, x)).length;
      const pred = v > 0 ? "Recurrence" : "No recurrence";
      const truth = edited ? "" : ` · actual: <b>${p.y ? "Recurrence" : "No recurrence"}</b> ${(v > 0 ? 1 : 0) === p.y ? "✓" : "✗"}`;
      verdict.innerHTML = `Votes: ${forV} for Recurrence − ${agV} for No recurrence = <b>${v > 0 ? "+" : ""}${v}</b> → prediction: <b>${pred}</b>${truth}`;
      out.textContent = `This rule set gets ${correct} of ${testIdx.length} test patients right (${((100 * correct) / testIdx.length).toFixed(1)}%). ` +
        `Always saying "No recurrence" would get ${testIdx.filter((i) => !pts[i].y).length}.\nA tie (vote sum 0) counts as No recurrence.`;
    }

    nav.addEventListener("click", (e) => {
      if (e.target.dataset.a === "next") load(cur + 1);
      if (e.target.dataset.a === "prev") load(cur - 1);
    });
    chips.addEventListener("click", (e) => {
      const k = e.target.dataset.k;
      if (k === undefined) return;
      x[+k] = 1 - x[+k];
      edited = true;
      draw();
    });
    load(0);
  }

  const mounts = { "viz-rule": ruleWidget, "viz-bool": boolWidget, "viz-rules": rulesWidget };
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
