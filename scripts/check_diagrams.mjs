// Layout checker for the HTML diagrams (docs/assets/js/diagram.js).
//
//   mkdocs build -d /tmp/site && node scripts/check_diagrams.mjs /tmp/site [page-filter]
//
// Opens every built page that contains a diagram in headless Chromium at several screen widths and
// measures each diagram for: overlapping / touching boxes, boxes outside the stage, edge labels sitting on
// boxes or on other labels, arrows passing through boxes or under other labels, arrows too short to see,
// words broken mid-word, text overflowing its box, and boxes or arrows that are *almost* aligned
// (a few pixels off a row/column, or a few degrees off horizontal/vertical).
// Exits with code 1 if anything is found. Needs Node 22+ (built-in WebSocket) and Playwright's Chromium,
// or set CHROME=/path/to/chrome.
import { spawn } from "node:child_process";
import { readdirSync, readFileSync, existsSync, mkdtempSync } from "node:fs";
import { join, relative } from "node:path";
import { homedir, tmpdir } from "node:os";

const WIDTHS = [1440, 1100, 820, 390];       // desktop with both sidebars, laptop, tablet, phone
const [siteDir, filter = ""] = process.argv.slice(2);
if (!siteDir) { console.error("usage: node scripts/check_diagrams.mjs <built-site-dir> [page-filter]"); process.exit(2); }

function findChrome() {
  if (process.env.CHROME) return process.env.CHROME;
  const base = join(homedir(), ".cache/ms-playwright");
  for (const d of existsSync(base) ? readdirSync(base).sort().reverse() : []) {
    for (const sub of existsSync(join(base, d)) ? readdirSync(join(base, d)) : []) {
      for (const exe of ["chrome-headless-shell", "chrome"]) {
        const p = join(base, d, sub, exe);
        if (existsSync(p)) return p;
      }
    }
  }
  throw new Error("No Chromium found: install Playwright's chromium or set CHROME");
}

function pages(dir) {
  const out = [];
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) out.push(...pages(p));
    else if (e.name === "index.html" && readFileSync(p, "utf8").includes('class="viz diagram"')) out.push(p);
  }
  return out;
}

// Runs inside the page. Returns a list of human-readable problems.
function audit() {
  const out = [];
  const R = (el) => el.getBoundingClientRect();
  const grow = (r, g) => ({ left: r.left - g, right: r.right + g, top: r.top - g, bottom: r.bottom + g });
  const hit = (a, b) => a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
  const inside = (x, y, r) => x > r.left && x < r.right && y > r.top && y < r.bottom;
  const text = (el) => el.textContent.replace(/\s+/g, " ").trim();

  document.querySelectorAll(".diagram").forEach((d, di) => {
    const title = text(d.querySelector(".viz-title") || d).slice(0, 50) || `diagram ${di + 1}`;
    const say = (msg) => out.push(`[${title}] ${msg}`);
    const stage = d.querySelector(".dg-stage"), sr = R(stage);
    const nodes = [...stage.querySelectorAll(".dg-node")].map((el) => ({ el, id: el.dataset.id, r: R(el), name: text(el.querySelector(".dg-label")) }));
    const labels = [...stage.querySelectorAll(".dg-edge-label")].map((el) => ({ el, r: R(el), name: text(el) }));
    const paths = [...stage.querySelectorAll(".dg-edge")];

    // boxes: overlap / too close / outside the stage / text overflow / broken words
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j];
        if (hit(a.r, b.r)) say(`boxes overlap: "${a.name}" and "${b.name}"`);
        else if (hit(grow(a.r, 7), b.r)) say(`boxes almost touch (<14px apart): "${a.name}" and "${b.name}"`);
      }
      if (a.r.left < sr.left - 1 || a.r.right > sr.right + 1 || a.r.top < sr.top - 1 || a.r.bottom > sr.bottom + 1)
        say(`box sticks out of the diagram: "${a.name}"`);
      if (a.el.scrollWidth > a.el.clientWidth + 1) say(`text overflows its box: "${a.name}"`);
    }
    for (const el of [...nodes.map((n) => n.el), ...labels.map((l) => l.el)]) {
      const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
      for (let t = walker.nextNode(); t; t = walker.nextNode()) {
        for (const m of t.textContent.matchAll(/[^\s-]{2,}/g)) {     // a break right after a hyphen is fine
          const range = document.createRange();
          range.setStart(t, m.index); range.setEnd(t, m.index + m[0].length);
          const tops = new Set([...range.getClientRects()].map((r) => Math.round(r.top)));
          if (tops.size > 1) say(`word broken across lines: "${m[0]}"`);
        }
      }
    }

    // edge labels: on a box, or on another label
    labels.forEach((l, i) => {
      for (const n of nodes) if (hit(grow(l.r, 2), n.r)) say(`label "${l.name}" touches box "${n.name}"`);
      for (let j = i + 1; j < labels.length; j++) if (hit(grow(l.r, 2), labels[j].r)) say(`labels overlap: "${l.name}" and "${labels[j].name}"`);
    });

    // arrows: through other boxes, under other labels, too short, nearly straight
    const nodeById = Object.fromEntries(nodes.map((n) => [n.id, n]));
    for (const p of paths) {
      const from = p.dataset.from, to = p.dataset.to;
      const len = p.getTotalLength(), name = `${from} → ${to}`;
      if (len < 22) say(`arrow too short to see (${Math.round(len)}px): ${name}`);
      const own = labels.find((l) => {
        const mid = p.getPointAtLength(len / 2);
        return inside(mid.x + sr.left, mid.y + sr.top, grow(l.r, 4));
      });
      const crossed = new Set(), under = new Set();
      for (let s = 0; s <= len; s += 3) {
        const pt = p.getPointAtLength(s), x = pt.x + sr.left, y = pt.y + sr.top;
        for (const n of nodes) if (n.id !== from && n.id !== to && inside(x, y, grow(n.r, -1))) crossed.add(n.name);
        for (const l of labels) if (l !== own && inside(x, y, l.r)) under.add(l.name);
      }
      for (const c of crossed) say(`arrow ${name} passes through box "${c}"`);
      for (const u of under) say(`arrow ${name} runs under label "${u}"`);
      const a = p.getPointAtLength(0), b = p.getPointAtLength(len);
      const straight = Math.abs(len - Math.hypot(b.x - a.x, b.y - a.y)) < 1;
      if (straight) {
        const deg = (Math.atan2(Math.abs(b.y - a.y), Math.abs(b.x - a.x)) * 180) / Math.PI;
        const off = Math.min(deg, 90 - deg);
        if (off > 0.5 && off < 8) say(`arrow ${name} is ${off.toFixed(1)}° off ${deg < 45 ? "horizontal" : "vertical"}: align its boxes`);
      }
      if (!nodeById[from] && !stage.querySelector(`.dg-group[data-id="${from}"]`)) say(`arrow from unknown id "${from}"`);
      if (!nodeById[to] && !stage.querySelector(`.dg-group[data-id="${to}"]`)) say(`arrow to unknown id "${to}"`);
    }

    // boxes that are almost (but not exactly) in the same row or column
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i].r, b = nodes[j].r;
        const dy = Math.abs((a.top + a.bottom) / 2 - (b.top + b.bottom) / 2);
        const dx = Math.abs((a.left + a.right) / 2 - (b.left + b.right) / 2);
        if (dy > 1 && dy < 12) say(`boxes nearly in the same row (${dy.toFixed(0)}px off): "${nodes[i].name}" and "${nodes[j].name}"`);
        if (dx > 1 && dx < 12) say(`boxes nearly in the same column (${dx.toFixed(0)}px off): "${nodes[i].name}" and "${nodes[j].name}"`);
      }
    }
  });
  return [...new Set(out)];
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const chrome = spawn(findChrome(), ["--headless", "--no-sandbox", "--remote-debugging-port=0",
  `--user-data-dir=${mkdtempSync(join(tmpdir(), "dgcheck-"))}`, "about:blank"], { stdio: ["ignore", "ignore", "pipe"] });
const wsUrl = await new Promise((res, rej) => {
  let buf = "";
  chrome.stderr.on("data", (c) => { buf += c; const m = buf.match(/ws:\/\/\S+/); if (m) res(m[0]); });
  setTimeout(() => rej(new Error("Chromium did not start")), 15000);
});
const port = new URL(wsUrl).port;
const target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t) => t.type === "page");
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let seq = 0; const waiting = {};
ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && waiting[d.id]) { waiting[d.id](d); delete waiting[d.id]; } };
const send = (method, params = {}) => new Promise((r) => { const id = ++seq; waiting[id] = r; ws.send(JSON.stringify({ id, method, params })); });
await send("Page.enable");

let problems = 0;
for (const file of pages(siteDir).filter((p) => relative(siteDir, p).includes(filter))) {
  for (const width of WIDTHS) {
    await send("Emulation.setDeviceMetricsOverride", { width, height: 900, deviceScaleFactor: 1, mobile: width < 600 });
    await send("Page.navigate", { url: "file://" + file });
    await sleep(1800);                                    // fonts + layout
    const r = await send("Runtime.evaluate", { expression: `(${audit})()`, returnByValue: true });
    const found = r.result?.result?.value ?? [`checker crashed: ${r.result?.exceptionDetails?.exception?.description}`];
    for (const f of found) console.log(`${relative(siteDir, file)} @${width}px  ${f}`);
    problems += found.length;
  }
}
ws.close(); chrome.kill();
console.log(problems ? `\n${problems} diagram layout problem(s) found.` : "All diagrams look clean.");
process.exit(problems ? 1 : 0);
