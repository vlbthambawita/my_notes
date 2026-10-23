# Learning AI, Visually — Project Plan

A personal learning site for AI topics and research papers. Each topic is explained **step by step, in simple language, with visualizations**, and published as a GitHub Pages site at:

> **https://vlbthambawita.github.io/my_notes/**

---

## 1. Goals

1. **Understand, not memorize.** Each lesson builds intuition first, then adds the math, then the code.
2. **See it.** Every core idea gets a picture: a diagram, an animation, or an interactive widget you can play with.
3. **Step by step.** Lessons are split into small steps; each step introduces one new idea.
4. **Papers made approachable.** Research papers are broken down into the problem, the key idea, how it works, and the results, with a visual for each.
5. **One place to revisit.** Everything is browsable topic by topic on the website, with search.

---

## 2. Technology choices

| Need | Tool | Why |
|---|---|---|
| Static site generator | **MkDocs + Material for MkDocs** | You write lessons in Markdown and get navigation, search, dark mode and a mobile layout for free |
| Math | **MathJax** (via `pymdownx.arithmatex`) | Write LaTeX like `$\mathrm{softmax}(x)$` |
| Diagrams | **Mermaid** (built into Material) | Flowcharts and architecture diagrams written as text in Markdown |
| Interactive visualizations | **Plain JavaScript + D3.js / Plotly.js**, embedded in pages | Sliders, animations, live plots (e.g. "drag the learning rate and watch gradient descent") |
| Static figures | **Python (matplotlib) scripts** saved as SVG/PNG | Reproducible plots kept in the repo |
| Code examples | Python snippets (NumPy / PyTorch), optional Colab links | Run the idea yourself |
| Hosting | **GitHub Pages**, deployed by **GitHub Actions** | Every push to `main` rebuilds and publishes the site automatically |

Why not plain HTML pages? MkDocs keeps writing fast (Markdown) while still allowing full HTML/JS where a visualization needs it.

---

## 3. Repository structure

```
my_notes/
├── PLAN.md                    ← this file
├── README.md                  ← short intro + link to the site
├── mkdocs.yml                 ← site config & navigation
├── requirements.txt           ← mkdocs-material, plugins
├── .github/workflows/
│   └── deploy.yml             ← build & publish to GitHub Pages
├── docs/                      ← everything the website shows
│   ├── index.md               ← home page: what this is, how to use it
│   ├── roadmap.md             ← list of topics, status of each
│   ├── topics/                ← one folder per topic
│   │   └── <topic-slug>/
│   │       ├── index.md       ← the lesson
│   │       ├── viz/           ← interactive JS widgets for this lesson
│   │       └── img/           ← generated / static figures
│   ├── papers/                ← one folder per paper
│   │   └── <year>-<short-name>/
│   │       ├── index.md       ← the paper walkthrough
│   │       ├── viz/
│   │       └── img/
│   ├── glossary.md            ← every term defined in one sentence
│   ├── assets/
│   │   ├── js/                ← shared visualization helpers
│   │   └── css/               ← shared styles for widgets
│   └── templates/             ← lesson & paper templates (copied for new pages)
└── scripts/                   ← Python scripts that generate figures
```

---

## 4. Lesson format (topics)

Every topic page follows the same structure, so lessons feel familiar:

1. **TL;DR** — the whole idea in 2–3 sentences.
2. **Why should I care?** — the problem this solves, with a real example.
3. **Prerequisites** — links to earlier lessons it builds on.
4. **Step 1 … Step N** — one idea per step. Each step has:
   - a plain-language explanation,
   - a **visual** (diagram, figure, or interactive widget),
   - the math, only once the intuition is there,
   - a small code snippet where it helps.
5. **Putting it together** — one diagram showing the full picture.
6. **Common misconceptions** — what people often get wrong.
7. **Check yourself** — 3–5 quick questions (answers in collapsible blocks).
8. **Further reading** — papers, videos, blog posts.

## 5. Paper walkthrough format

1. **Paper card** — title, authors, year, venue, link, one-sentence summary.
2. **The problem** — what was broken or missing before this paper.
3. **The key idea** — the main contribution, in plain words + one visual.
4. **How it works, step by step** — architecture or method, with diagrams for each part.
5. **Experiments & results** — what they tested, the main numbers, re-plotted simply.
6. **Limitations & open questions** — what the paper doesn't solve.
7. **Why it matters** — its influence and what came after.
8. **Connections** — links to related topic lessons and other papers on this site.
9. **Glossary additions** — new terms, also added to `glossary.md`.

---

## 6. Visualization principles

- **One visual per idea.** If a step has no visual, ask whether it should be split or merged.
- **Interactive where it builds intuition**: parameters you can change (learning rate, temperature, number of heads, etc.) and see the effect immediately.
- **Animate processes** (forward pass, backpropagation, attention, diffusion denoising) step by step with "Next / Previous" buttons.
- **Consistent colors** across the site (e.g. inputs always one color, weights another, outputs a third) so diagrams are easy to read.
- **Works on phone and in dark mode.**
- **No heavy builds:** widgets are small, self-contained JS files loaded only on the page that uses them.

---

## 7. Workflow: how we'll work together

**When you give a topic:**
1. I propose a short outline (steps + planned visuals) for you to confirm or adjust.
2. I write the lesson in `docs/topics/<topic>/`, build its visualizations, and add it to the navigation and the roadmap.
3. You preview it locally (`mkdocs serve`) or on the site after pushing.
4. You ask follow-up questions; I refine the page or add a "Deeper dive" section.

**When you give a paper (PDF or link):**
1. I read the paper and list the prerequisite concepts; any that are missing get a short topic lesson first.
2. I write the walkthrough in `docs/papers/<year>-<name>/` using the paper format above.
3. Figures from the paper are **redrawn / simplified** rather than copied (keeps the site clean and avoids copyright issues).

**Git:** one commit per lesson or paper (e.g. `Add lesson: attention mechanism`), pushed to `main` → site auto-deploys.

---

## 8. Implementation phases

### Phase 0 — Set up the site (first step)
- [ ] Add `mkdocs.yml` with Material theme, MathJax, Mermaid, search, dark mode
- [ ] Add `requirements.txt`
- [ ] Add GitHub Actions workflow to deploy to GitHub Pages
- [ ] Create home page, roadmap page, glossary page
- [ ] Create lesson and paper templates
- [ ] Add shared JS/CSS helpers for visualizations (color palette, step-through animation, slider controls)
- [ ] Build one **demo lesson** (e.g. "Gradient descent") to test every feature: math, Mermaid, interactive widget, code
- [ ] Enable GitHub Pages in repo settings (Settings → Pages → Source: GitHub Actions)

### Phase 1 — Topics and papers (ongoing, driven by you)
- Add lessons and paper walkthroughs as you request them.
- Keep `roadmap.md` and `glossary.md` up to date.

### Phase 2 — Make the site more connected (later)
- Concept map page: a clickable graph showing how topics and papers relate.
- Tags (e.g. `vision`, `NLP`, `generative`, `medical-AI`) to filter pages.
- "Learning paths" — suggested reading orders for a goal (e.g. "Understand LLMs from scratch").

---

## 9. Suggested topic roadmap (a starting point — you choose the order)

**Foundations**
- What is machine learning? (supervised / unsupervised / reinforcement)
- Linear regression and loss functions
- Gradient descent
- Logistic regression and classification
- Overfitting, regularization, train/val/test splits

**Neural networks**
- The perceptron and multilayer networks
- Activation functions
- Backpropagation
- Optimizers (SGD, momentum, Adam)
- Normalization and dropout

**Architectures**
- Convolutional neural networks (CNNs)
- Recurrent networks (RNN, LSTM)
- Attention and the Transformer
- Embeddings and tokenization
- Vision Transformers

**Generative AI**
- Autoencoders and VAEs
- GANs
- Diffusion models
- Large language models: pre-training, fine-tuning, RLHF
- Retrieval-augmented generation (RAG) and agents

**Evaluation and trust**
- Metrics (accuracy, precision/recall, AUC, F1)
- Explainability (saliency maps, SHAP)
- Bias, fairness, and robustness

---

## 10. Local preview

```bash
pip install -r requirements.txt
mkdocs serve          # open http://127.0.0.1:8000
```
