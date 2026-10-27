---
name: write-lesson
description: Write or update a lesson (docs/topics/...) or paper walkthrough (docs/papers/...) for the "Learning AI, Visually" MkDocs site, including its interactive widgets and diagrams. Use whenever adding a topic, converting or adding a diagram/flowchart, or building a visualization on this site. Diagrams are rich HTML (assets/js/diagram.js), never Mermaid.
---

# Writing a lesson for "Learning AI, Visually"

The site is MkDocs + Material, published from `main` to https://vajira.info/my_notes/ by GitHub Actions.
`PLAN.md` is the project plan; `docs/templates/topic-template.md` and `paper-template.md` are the page skeletons.

## Hard rules

1. **No Mermaid. Ever.** Every flowchart, loop, pipeline, tree or architecture diagram is an **HTML diagram**
   rendered by `docs/assets/js/diagram.js`. See [diagrams.md](diagrams.md) for the spec format and layout recipes.
   Mermaid support has been removed from `mkdocs.yml` on purpose; don't add it back.
2. **Visualizations are rich HTML/JS**: interactive widgets (SVG or canvas + controls) or HTML diagrams with a
   step-through. No static image when an interactive one teaches better, and no ASCII art.
3. **Every number in the prose is real.** Values quoted in tips, answers or code output must come from actually
   running the code (put scratch scripts in the scratchpad, not the repo).
4. **One idea per step, one visual per idea.** Intuition first, then the math, then code.

## Lesson structure (topics)

Follow `docs/templates/topic-template.md`: TL;DR → Level/Prerequisites → Why should I care? → Step 1…N
(explanation, visual, math, "What to notice" tip) → Putting it all together (HTML diagram + code) →
Common misconceptions → Check yourself (`??? question` blocks) → Further reading → "Next lessons".
Paper walkthroughs follow `docs/templates/paper-template.md` with the same visual rules.

Write in plain, short sentences. Use admonitions: `!!! abstract "TL;DR"`, `!!! tip "What to notice"`,
`!!! note`, `!!! warning "\"misconception\""`, `??? question`.

## Files to touch for a new lesson

- `docs/topics/<slug>/index.md`: the lesson. It ends with `<script src="viz/<slug>.js"></script>` if it has widgets.
- `docs/topics/<slug>/viz/<slug>.js`: the lesson's widgets (see below).
- `mkdocs.yml` → `nav:` under `Topics:`, in the right section.
- `docs/roadmap.md`: change the row to a link and set ✅. Keep `PLAN.md`'s suggested roadmap in sync when adding topics.
- `docs/glossary.md`: add every new term, alphabetically, as `**Term** (symbol): one sentence. ([Lesson](path#anchor))`.
  Get anchors from the built HTML (`grep -o 'id="step-[^"]*"' site/.../index.html`); slugs drop punctuation.

## Interactive widgets

Copy the pattern from `docs/topics/gradient-descent/viz/gradient-descent.js` or
`docs/topics/reinforcement-learning-intro/viz/reinforcement-learning-intro.js`:

- Wrap everything in `document.addEventListener("DOMContentLoaded", ...)` and use the helpers from
  `window.Viz` (`colors, onThemeChange, linear, el, svg, axes, ticks, fmt`, in `docs/assets/js/viz-common.js`).
- Mount on `<div id="viz-<name>" class="viz"><div class="viz-title">🎚️ Interactive: …</div></div>` through a
  `mounts` map, guarded with `dataset.ready`.
- Controls go in `.viz-controls`, live numbers in `.viz-readout`, hints in `.viz-hint` (all styled in `docs/assets/css/viz.css`).
- **Colour roles** (CSS vars, same everywhere): data = blue `--viz-data`, model/prediction/agent = orange `--viz-model`,
  loss/value = purple `--viz-loss`, gradient/bad = red `--viz-grad`, optimum/goal/good = green `--viz-opt`.
  Never hard-code colours; read them with `colors()` and redraw on `onThemeChange`.
- SVG text is muted by default (`.viz svg text`); use `style: "fill: var(--viz-text)"` for emphasised text.
- Must work in dark mode and at phone width (~390px).

### Widget pitfalls (found while writing the RL series)

- **Simulations must be seeded and reproducible.** Use a small seeded RNG (e.g. mulberry32) and port it to the
  Python check script, so the widget readout and the numbers in the prose agree exactly. Long simulations can
  still differ between browsers because `Math.exp`/`Math.log` aren't bit-identical everywhere. If a quoted number
  depends on a long run, either build those functions from + − × ÷ or quote the Python number and say the widget
  may differ slightly.
- **Heavy simulations** (thousands of runs) run in chunks with `setTimeout`, and only start when the widget scrolls into
  view (`IntersectionObserver`), so the page stays responsive.
- **SVG text shrinks on phones.** A 600px-wide viewBox scaled to 360px makes 11px text about 7px. Below ~480px switch to a
  narrower viewBox or layout instead of just scaling.
- **Tables inside widgets** pick up Material's table styles; give them their own class.
- **Display math inside admonitions or list items** needs blank lines around `$$ … $$`, or it won't render.
- **Long inline equations overflow phones.** Split them into aligned display lines.
- **Readouts:** separate items with " · ". Collapsed whitespace otherwise runs them together.
- **Avoid values like "-0.0" and "+∞" in readouts.** Format them as 0.0 and "new".

## Verify before finishing

1. `mkdocs build --strict -d <scratchpad>/site` must have no warnings.
2. `node scripts/check_diagrams.mjs <scratchpad>/site [topics/<slug>/]` must print "All diagrams look clean." It measures every
   diagram at 1440/1100/820/390px for overlaps, arrows through boxes, broken words and misalignment.
   Fix the layout (see the rules in [diagrams.md](diagrams.md)) rather than ignoring a finding.
3. Look at the page in headless Chromium at **1440px** (both sidebars, narrowest desktop content) **and 390px**, in **light and dark** mode
   (`data-md-color-scheme="slate"` on `<body>`). Click through every widget and every diagram step.
   Playwright's Chromium is in `~/.cache/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell`;
   run it with `--remote-debugging-port` and drive it over CDP with Node's built-in `WebSocket`
   (open pages with `file://`, or serve the built site with `python3 -m http.server`). Emoji show as boxes in headless
   screenshots; that's only a missing font. A `search_index.js` error under `file://` is harmless.
4. Check by eye what the script can't: nothing cramped or lopsided, no big empty bands, boxes on a clean grid,
   arrows straight (diagonal only across an empty cell), labels short and clear of boxes.

## Git

One commit per lesson (`Add lesson: <name>`). Commit and push only when the user asks.
