# The Ising model

!!! abstract "TL;DR"
    The Ising model is a grid of tiny arrows (**spins**) that each point up or down. Neighbours **prefer to agree**, and **temperature** adds random flips that fight this preference. Below a critical temperature the whole grid lines up by itself; above it, the order melts. The same three ingredients (binary units, pairwise links, an energy to minimise) are the basis of Hopfield networks, Boltzmann machines and every "temperature" knob in AI.

**Level:** beginner · **Prerequisites:** none. Knowing what $e^{x}$ looks like helps from Step 3 on.

---

## Why should I care?

A piece of iron is a magnet at room temperature. Heat it above 770 °C and the magnetism disappears. Cool it and it comes back. No atom "decides" this. Each atom only feels its nearest neighbours. So how do billions of atoms agree on one direction, and why does that agreement break at one sharp temperature?

The Ising model (also called the **Lenz–Ising model**: Wilhelm Lenz proposed it in 1920, his student Ernst Ising solved the first case in 1924) is the simplest model that answers this. It is worth learning for AI because:

- **Hopfield networks** and **Boltzmann machines** are Ising models whose links are *learned*. They earned John Hopfield and Geoffrey Hinton the 2024 Nobel Prize in Physics.
- The **Metropolis algorithm**, invented to simulate models like this one, is the ancestor of the sampling methods (MCMC) used across statistics and machine learning.
- The **temperature** setting of a language model comes from the same formula you will meet in Step 3.

---

## Step 1 — Tiny magnets on a grid

Take a grid. Put one arrow on every square. Each arrow is called a **spin** and has only two options:

- up, written $s_i = +1$
- down, written $s_i = -1$

That is the whole state of the model: a list of $+1$s and $-1$s. A 4×4 grid has 16 spins and therefore $2^{16} = 65{,}536$ possible patterns.

One number summarises a pattern: the **magnetisation** $m$, the average of all spins.

$$
m = \frac{1}{N}\sum_{i=1}^{N} s_i
$$

$m = +1$ means every spin is up, $m = -1$ means every spin is down, and $m = 0$ means half and half. A real magnet is a piece of material with $m$ far from 0.

**Try it:** click spins to flip them and watch $m$ in the readout. Ignore the right-hand panel for a moment; Step 2 explains it.

<div id="viz-spins" class="viz"><div class="viz-title">🧲 Interactive: flip spins, watch the energy</div></div>

!!! tip "What to notice"
    - **All up** gives $m = +1$. **Two halves** and **Checkerboard** both give $m = 0$, although they look very different. Magnetisation alone doesn't tell them apart. Energy will.
    - The model has no idea what "up" means physically. A spin can stand for an atom's magnetic moment, a pixel (black or white), a neuron (firing or silent), or a voter (yes or no).

---

## Step 2 — Energy: neighbours want to agree

Every pair of neighbouring spins shares a **link** (physicists say *bond*). The rule is:

- the two spins **agree** → the link contributes $-1$ to the energy (good, stable)
- the two spins **disagree** → the link contributes $+1$ (bad, tense)

The smallest possible example is two spins with one link:

| Pattern | $s_1 s_2$ | Energy |
|---|---|---|
| ↑ ↑ | $+1$ | $-1$ |
| ↓ ↓ | $+1$ | $-1$ |
| ↑ ↓ | $-1$ | $+1$ |
| ↓ ↑ | $-1$ | $+1$ |

The product $s_1 s_2$ is $+1$ when the spins agree and $-1$ when they don't, so the energy of one link is simply $-s_1 s_2$. For a whole grid, add up all links:

$$
E(s) = -J \sum_{\langle i,j \rangle} s_i s_j \;-\; h \sum_i s_i
$$

- $\langle i,j \rangle$ means "every pair of neighbours, counted once".
- $J$ is the **coupling**: how strongly neighbours want to agree. We use $J = 1$ throughout.
- $h$ is an **external field**: a push that makes every spin prefer up ($h > 0$) or down ($h < 0$), like holding a big magnet nearby. It is 0 until Step 5.

Systems drift towards **low energy**, like a ball rolling downhill. So this formula says: *patterns in which neighbours agree are preferred.*

Go back to the widget above. The 4×4 grid has 24 links. Green links agree, red links disagree, and the purple marker shows the total energy.

!!! tip "What to notice"
    - **All up** (and all down) is the lowest possible energy, $-24$: every link agrees. **Checkerboard** is the highest, $+24$.
    - **Two halves** has $m = 0$ but energy $-16$. Only the 4 links on the border between the halves disagree. **Energy lives on the borders between regions.**
    - Start from **All up** and flip one spin in the middle: 4 links turn red and the energy rises by 8. Flip a 2×2 block: 8 links are red, the energy rises by 16. An island of flipped spins costs energy in proportion to its **perimeter**. Remember this for Step 6.

!!! note "What if neighbours want to disagree?"
    With $J < 0$ the signs swap: the checkerboard becomes the lowest-energy pattern. That is an *antiferromagnet*. With a different random $J$ on every link you get a *spin glass*, which is the starting point for Hopfield networks (Step 8).

---

## Step 3 — Temperature: order against chance

If energy were the whole story, every grid would end up all up or all down. But a warm material jiggles. Heat keeps knocking spins into random directions.

Physics gives an exact rule for how often each pattern $s$ shows up at temperature $T$, the **Boltzmann distribution**:

$$
P(s) = \frac{e^{-E(s)/T}}{Z},
\qquad
Z = \sum_{\text{all patterns } s'} e^{-E(s')/T}
$$

- Lower energy → larger $e^{-E/T}$ → more likely.
- $T$ decides **how much** energy matters. Small $T$: energy differences are huge, so only the lowest-energy patterns appear. Large $T$: every pattern is almost equally likely.
- $Z$, the **partition function**, is just the sum that makes the probabilities add up to 1.

(We measure $T$ in units where Boltzmann's constant $k_B$ is 1, so $T$ and $E$ can be divided directly.)

A 2×2 grid is small enough to list everything. It has 4 links and $2^4 = 16$ patterns:

| Patterns | How many | Energy |
|---|---|---|
| all four spins agree | 2 | $-4$ |
| mixed (one or two spins differ) | 12 | $0$ |
| checkerboard | 2 | $+4$ |

So $Z = 2e^{4/T} + 12 + 2e^{-4/T}$. At $T = 2$ this is $27.05$, and each all-agree pattern has probability $e^{2}/27.05 = 27.3\%$.

**Try it:** drag the temperature. Each column is one of the 16 patterns with its probability bar.

<div id="viz-boltzmann" class="viz"><div class="viz-title">🌡️ Interactive: all 16 patterns of a 2×2 grid</div></div>

!!! tip "What to notice"
    - **Cold (T = 0.5):** the two aligned patterns take 99.8% of the probability. Energy wins.
    - **Warm (T = 2):** each aligned pattern (27.3%) is still far more likely than each mixed pattern (3.7%). But there are **12** mixed patterns and only 2 aligned ones, so together the mixed ones already hold 44.4%.
    - **Hot (T = 10):** the bars are nearly level and the mixed patterns hold 73.5%, close to their share by pure counting ($12/16 = 75\%$). Numbers win.
    - This is the key trade-off of the whole lesson: **low energy** (few ordered patterns) against **large numbers** (many disordered patterns, which physicists call *entropy*). Temperature sets the exchange rate.

---

## Step 4 — One flip at a time: the Metropolis rule

Listing all patterns worked for 4 spins. A 64×64 grid has $2^{4096}$ patterns, a number with 1,234 digits. We can never compute $Z$.

The **Metropolis algorithm** (1953) avoids $Z$ completely. It walks from pattern to pattern by trying to flip one spin at a time, and it only needs the **energy change** $\Delta E$ of that single flip:

$$
\Delta E = 2\, s_i \Big( J \sum_{j \,\in\, \text{neighbours of } i} s_j + h \Big)
$$

Only the spin's own links change, so only its four neighbours matter. The rule is then:

- $\Delta E \le 0$ (downhill or flat): **always flip**.
- $\Delta E > 0$ (uphill): flip anyway **with probability** $e^{-\Delta E / T}$.

The second line is where temperature comes in. Uphill moves are how heat creates disorder: rare when it's cold, common when it's hot.

**Try it:** build a situation by clicking the spins, then change $T$.

<div id="viz-flip" class="viz"><div class="viz-title">🎲 Interactive: should this spin flip?</div></div>

!!! tip "What to notice"
    - The starting situation (3 neighbours agree, 1 disagrees) is uphill by $\Delta E = +4$. At $T = 2$ the flip still happens 13.5% of the time.
    - Make all four neighbours agree with the centre: $\Delta E = +8$ and the chance drops to 1.8%. A spin surrounded by friends almost never flips.
    - Make most neighbours disagree: $\Delta E$ is negative and the flip is certain, whatever the temperature.
    - With $h = 0$ only five values are possible: $\Delta E \in \{-8, -4, 0, +4, +8\}$.

Repeating this rule many times makes the grid visit patterns exactly as often as the Boltzmann distribution says. One **sweep** means as many attempted flips as there are spins.

<div class="viz diagram">
<script type="application/json">
{
  "title": "🔁 The Metropolis loop",
  "cols": 3,
  "rows": 2,
  "rowHeight": 110,
  "nodes": [
    {"id": "start", "col": 0, "row": 0, "icon": "🎲", "label": "Start with any pattern", "shape": "pill", "note": "Random spins or all up. After enough sweeps the start is forgotten."},
    {"id": "pick", "col": 1, "row": 0, "icon": "👆", "label": "Pick a random spin", "role": "data", "note": "Every spin has the same chance of being picked."},
    {"id": "de", "col": 2, "row": 0, "icon": "⚖️", "label": "Compute ΔE", "sub": "if this spin flipped", "role": "loss", "note": "ΔE = 2 · s · (J · sum of the 4 neighbours + h). No other spin is needed."},
    {"id": "acc", "col": 2, "row": 1, "icon": "🌡️", "label": "Downhill? Flip. Uphill? Roll a die", "sub": "chance e<sup>−ΔE/T</sup>", "role": "model", "note": "ΔE ≤ 0: always flip. ΔE > 0: flip with probability e^(−ΔE/T)."},
    {"id": "flip", "col": 1, "row": 1, "icon": "🔄", "label": "Flip or keep the spin", "role": "opt", "note": "This new pattern is the next sample, even if nothing changed."},
    {"id": "meas", "col": 0, "row": 1, "icon": "📏", "label": "Measure m and E", "sub": "once per sweep", "note": "Average the measurements over many sweeps to get properties at temperature T."}
  ],
  "edges": [
    {"from": "start", "to": "pick"},
    {"from": "pick", "to": "de"},
    {"from": "de", "to": "acc"},
    {"from": "acc", "to": "flip"},
    {"from": "flip", "to": "pick", "label": "repeat"},
    {"from": "flip", "to": "meas"}
  ],
  "steps": [
    {"show": ["start"], "text": "Begin with any pattern of spins. It doesn't have to be a likely one."},
    {"show": ["start>pick", "pick"], "text": "Choose one spin at random."},
    {"show": ["pick>de", "de"], "text": "Work out how much the energy would change if this one spin flipped. Only its four neighbours matter."},
    {"show": ["de>acc", "acc"], "text": "If the energy would not rise, accept the flip. If it would rise, accept it only with probability e<sup>−ΔE/T</sup>."},
    {"show": ["acc>flip", "flip"], "text": "Flip the spin if the move was accepted. Otherwise leave it."},
    {"show": ["flip>pick", "pick"], "text": "Go round again. As many attempts as there are spins is called one sweep."},
    {"show": ["flip>meas", "meas"], "text": "After each sweep, record the magnetisation and the energy. Their averages are what we compare with experiments."}
  ],
  "narrow": {"cols": 2, "rows": 4, "pos": {"start": [0, 0], "pick": [0, 1], "de": [1, 1], "acc": [1, 2], "flip": [0, 2], "meas": [0, 3]}}
}
</script>
</div>

---

## Step 5 — Watch 4,096 spins at once

Here is the Metropolis loop running on a 64×64 grid. The edges wrap around (the left column is a neighbour of the right column), so no spin sits on a border. The grid starts with random spins.

**Try it:** press **▶ Play** and watch for about 20 seconds. Then try the three temperature buttons, and **Random start** to begin again.

<div id="viz-lattice" class="viz"><div class="viz-title">🔥 Interactive: a live Ising model</div></div>

!!! tip "What to notice"
    - **Cold (T = 1.5):** small patches form at once and then slowly eat each other. From this random start, $m$ climbs to $+0.37$ after 100 sweeps, $+0.89$ after 300 and $+0.98$ after 500. Nobody told the grid to choose "up". It was a tie, and chance broke it.
    - **Hot (T = 3.5):** fine-grained noise. $m$ stays within a few hundredths of 0, and no patch survives.
    - **Critical (T = 2.27):** patches of *every* size, from single spins to blobs as wide as the grid, and $m$ wanders slowly without settling. This is the most interesting temperature, and Step 6 is about it.
    - **The field breaks the tie.** At $T = 3.5$, slide $h$ to $+0.50$: $m$ jumps to about $+0.6$ within ten sweeps. Slide it back to 0 and the magnetisation is forgotten. Now do the same at $T = 1.5$: the grid stays magnetised after you remove the field. That memory is what makes a permanent magnet.
    - A cold grid sometimes gets stuck in two broad **stripes**. Their straight borders have no corners to shrink from, so the stripes can last a long time. It is an early taste of "stuck in a local minimum".

---

## Step 6 — The phase transition

Let's turn what you just watched into a measurement. For each temperature, run the simulation, wait for it to settle, and record the average of $|m|$ (the absolute value, because up and down are equally good).

**Try it:** press **Run**. The green curve is the exact answer for an infinitely large grid.

<div id="viz-sweep" class="viz"><div class="viz-title">📉 Interactive: magnetisation against temperature</div></div>

!!! tip "What to notice"
    - On the 32×32 grid the dots follow the exact curve closely while the grid is ordered: 0.986 at $T = 1.5$ (exact: 0.986) and 0.908 at $T = 2.0$ (exact: 0.911).
    - Between $T = 2.2$ and $T = 2.5$ the magnetisation collapses from 0.815 to 0.252. The order does not fade gradually; it **breaks** over a narrow range. This is a **phase transition**, like ice melting.
    - Switch to the **8 × 8** grid: the drop is smeared out, and at $T = 4$ you still measure 0.191 (against 0.044 on the 32×32 grid). A small grid can't average its random spins away. The transition is only perfectly sharp for an infinite grid.

The exact position of the break is the **critical temperature**, found by Lars Onsager in 1944:

$$
T_c = \frac{2J}{\ln\!\left(1 + \sqrt{2}\right)} \approx 2.269
$$

Below it, the exact magnetisation of an infinite grid is

$$
m = \left[\,1 - \sinh^{-4}\!\left(\frac{2J}{T}\right)\right]^{1/8}
$$

and above it, $m = 0$. That is the green curve.

**Why is there a sharp temperature?** Go back to Step 2: a flipped island costs energy in proportion to its perimeter. When it's cold, large islands are too expensive, so one colour dominates. But the number of possible island shapes also grows with the perimeter. At $T_c$ the two effects balance, islands of every size appear, and the dominant colour is lost.

!!! note "Who solved what: Lenz, Ising, Onsager"
    Ernst Ising solved the **one-dimensional** model, a single chain of spins, in his 1924 thesis. He found **no** phase transition: a chain is disordered at every temperature above zero. The reason is the island argument again. In a chain, flipping a block of any length breaks just 2 links (its two ends), so large islands are cheap and they always destroy the order. Ising guessed that the same would happen in higher dimensions. He was wrong: Rudolf Peierls showed in 1936 that the 2-D model must order, and Onsager's 1944 solution gave the exact numbers above. The 3-D model still has no exact solution, only simulations like yours.

---

## Step 7 — An AI example: cleaning a noisy picture

The Ising model is also a tool. Take a black-and-white picture where some pixels have been flipped by noise. Two facts help us repair it:

1. Most pixels are correct, so the answer should **stay close to what we observe**.
2. Real pictures are smooth, so a pixel usually **agrees with its neighbours**.

Fact 2 is the Ising link term. Fact 1 is a field, but a different one for every pixel: it pulls pixel $i$ towards its observed value $y_i$.

$$
E(s) = -J \sum_{\langle i,j \rangle} s_i s_j \;-\; \sum_i s_i\, y_i
$$

Here $s$ is our guess for the clean picture and $y$ is the noisy one ($+1$ = ink, $-1$ = paper). We look for a low-energy $s$ in the simplest way: start from $s = y$ and flip any pixel whose flip lowers the energy, until nothing changes. This is the Metropolis rule at $T = 0$ (no uphill moves).

<div id="viz-denoise" class="viz"><div class="viz-title">🧹 Interactive: denoising with an Ising energy</div></div>

!!! tip "What to notice"
    - With 10% noise, 152 of the 1,600 pixels are wrong. With **J = 1** only 15 remain wrong (0.9%).
    - **J = 0** ignores the neighbours, so the energy is lowest at $s = y$ and nothing is repaired.
    - **J = 0.3** is cautious: a pixel flips only when all four neighbours disagree with it. That leaves 76 wrong pixels, mostly where two noisy pixels touch.
    - Raise the noise to 30%: J = 1 still more than halves the errors (478 → 224), but clumps of noise now look like real shapes, and the model keeps them.

!!! note "The general idea"
    A model like this, where each variable interacts only with its neighbours, is called a **Markov random field**. Geman and Geman introduced this picture-restoration recipe in 1984, using exactly the Ising analogy.

---

## Step 8 — From magnets to neural networks

So far $J$ was the same on every link and fixed in advance. The step to machine learning is one sentence long: **let every pair of units have its own weight $w_{ij}$, and learn the weights.**

$$
E(s) = -\sum_{i<j} w_{ij}\, s_i s_j \;-\; \sum_i b_i\, s_i
$$

| Ising model | Machine learning |
|---|---|
| spin $s_i = \pm 1$ | a binary neuron, pixel or other variable |
| coupling $J$ | a **weight** $w_{ij}$ between two units |
| field $h$ | a **bias** $b_i$ |
| energy $E(s)$ | the **energy function** of an energy-based model |
| Boltzmann distribution | the model's probability of a pattern; a **softmax** is the same formula |
| temperature $T$ | sampling temperature (for example in a language model) |
| Metropolis sweeps | **MCMC** sampling (Gibbs sampling is a close relative) |
| lowest-energy patterns | stored memories (Hopfield) or likely data (Boltzmann machine) |

<div class="viz diagram">
<script type="application/json">
{
  "title": "🧬 The family tree",
  "cols": 4,
  "rows": 1,
  "rowHeight": 130,
  "nodes": [
    {"id": "ising", "col": 0, "row": 0, "icon": "🧲", "label": "Ising model", "sub": "1920–25", "role": "data", "note": "Lenz and Ising: one fixed coupling J between neighbours on a grid. Explains magnetism."},
    {"id": "hop", "col": 1, "row": 0, "icon": "🧠", "label": "Hopfield network", "sub": "1982", "role": "model", "note": "Hopfield: every pair of units has its own weight, chosen so that stored patterns are energy minima. Rolling downhill from a damaged pattern recalls the memory."},
    {"id": "bm", "col": 2, "row": 0, "icon": "🌡️", "label": "Boltzmann machine", "sub": "1985", "role": "loss", "note": "Ackley, Hinton and Sejnowski: sample at T > 0, add hidden units, and learn the weights so the Boltzmann distribution matches the data."},
    {"id": "ebm", "col": 3, "row": 0, "icon": "✨", "label": "Energy-based models", "sub": "today", "role": "opt", "note": "Restricted Boltzmann machines started deep learning's revival in 2006. Modern energy-based and diffusion models keep the idea: learn an energy, then sample."}
  ],
  "edges": [
    {"from": "ising", "to": "hop"},
    {"from": "hop", "to": "bm"},
    {"from": "bm", "to": "ebm"}
  ],
  "steps": [
    {"show": ["ising"], "text": "<b>Ising model:</b> one coupling J, fixed by nature, between neighbours on a grid. The low-energy patterns are all up and all down."},
    {"show": ["ising>hop", "hop"], "text": "<b>Hopfield network:</b> give every pair of units its own weight and choose the weights so that the patterns you want to remember are the low-energy ones. Start from a damaged pattern, flip spins downhill, and the memory comes back."},
    {"show": ["hop>bm", "bm"], "text": "<b>Boltzmann machine:</b> turn the temperature on, add hidden units, and <i>learn</i> the weights so that the Boltzmann distribution puts high probability on the training data."},
    {"show": ["bm>ebm", "ebm"], "text": "<b>Today:</b> restricted Boltzmann machines helped restart deep learning in 2006, and energy-based and diffusion models still follow the plan \"learn an energy, then sample from it\"."}
  ],
  "narrow": {"cols": 2, "rows": 2, "rowHeight": 110, "pos": {"ising": [0, 0], "hop": [1, 0], "bm": [1, 1], "ebm": [0, 1]}}
}
</script>
</div>

!!! tip "What to notice: the temperature of a language model"
    A language model gives every possible next word a score $z_k$ (a *logit*) and samples from

    $$
    P(k) = \frac{e^{z_k / T}}{\sum_{k'} e^{z_{k'}/T}}
    $$

    This is the Boltzmann distribution with energy $E_k = -z_k$. Low $T$: the model almost always picks the top word (ordered, repetitive). High $T$: all words become similar in probability (disordered, random). It is the same trade-off as in Step 3.

---

## Putting it all together

<div class="viz diagram">
<script type="application/json">
{
  "title": "🧩 The Ising model in one picture",
  "cols": 3,
  "rows": 2,
  "rowHeight": 110,
  "nodes": [
    {"id": "spins", "col": 0, "row": 0, "icon": "🧲", "label": "Spins on a grid", "sub": "s<sub>i</sub> = +1 or −1", "role": "data", "note": "The state of the model: one up-or-down value per site."},
    {"id": "energy", "col": 1, "row": 0, "icon": "⛰️", "label": "Energy E(s)", "sub": "low when neighbours agree", "role": "loss", "note": "E = −J Σ s_i s_j − h Σ s_i. Low when neighbours agree and spins follow the field."},
    {"id": "boltz", "col": 2, "row": 0, "icon": "🌡️", "label": "Boltzmann distribution", "sub": "P(s) ∝ e<sup>−E(s)/T</sup>", "role": "model", "note": "Temperature T sets how strongly low energy is preferred."},
    {"id": "metro", "col": 2, "row": 1, "icon": "🎲", "label": "Metropolis sampling", "sub": "one flip at a time", "role": "model", "note": "Flip a spin if ΔE ≤ 0, otherwise with probability e^(−ΔE/T). No partition function needed."},
    {"id": "meas", "col": 1, "row": 1, "icon": "📏", "label": "Measure magnetisation", "sub": "average of |m|", "note": "Average over many sweeps after the grid has settled."},
    {"id": "phase", "col": 0, "row": 1, "icon": "❄️", "label": "Phase transition", "sub": "at T<sub>c</sub> ≈ 2.27", "role": "opt", "note": "Ordered below T_c, disordered above it."}
  ],
  "edges": [
    {"from": "spins", "to": "energy"},
    {"from": "energy", "to": "boltz"},
    {"from": "boltz", "to": "metro"},
    {"from": "metro", "to": "meas"},
    {"from": "meas", "to": "phase"}
  ],
  "steps": [
    {"show": ["spins"], "text": "The model is a grid of spins, each +1 (up) or −1 (down)."},
    {"show": ["spins>energy", "energy"], "text": "Every pattern gets an energy. Each pair of agreeing neighbours lowers it by J, each disagreeing pair raises it by J."},
    {"show": ["energy>boltz", "boltz"], "text": "At temperature T, a pattern appears with probability proportional to e<sup>−E/T</sup>. Cold favours low energy. Hot makes all patterns alike."},
    {"show": ["boltz>metro", "metro"], "text": "We can't list all patterns, so we sample them: propose single flips and accept them with the Metropolis rule."},
    {"show": ["metro>meas", "meas"], "text": "Once the grid has settled, average the magnetisation over many sweeps."},
    {"show": ["meas>phase", "phase"], "text": "Repeat at many temperatures: the magnetisation collapses near T<sub>c</sub> ≈ 2.27. Local rules have produced a global, sudden change."}
  ],
  "narrow": {"cols": 2, "rows": 3, "pos": {"spins": [0, 0], "energy": [1, 0], "boltz": [1, 1], "metro": [0, 1], "meas": [0, 2], "phase": [1, 2]}}
}
</script>
</div>

The whole simulation fits in a few lines of NumPy:

```python
import numpy as np

def simulate(T, L=32, sweeps=600, seed=0):
    rng = np.random.default_rng(seed)
    s = np.ones((L, L), dtype=int)                 # start with every spin up
    m = []
    for sweep in range(sweeps):
        for _ in range(L * L):                     # one sweep = L*L attempted flips
            i, j = rng.integers(L, size=2)         # 1. pick a random spin
            nb = (s[(i + 1) % L, j] + s[(i - 1) % L, j] +
                  s[i, (j + 1) % L] + s[i, (j - 1) % L])
            dE = 2 * s[i, j] * nb                  # 2. energy change if it flipped (J = 1, h = 0)
            if dE <= 0 or rng.random() < np.exp(-dE / T):
                s[i, j] *= -1                      # 3. Metropolis rule: flip
        if sweep >= 200:                           # skip the first 200 sweeps (settling in)
            m.append(abs(s.mean()))                # 4. measure
    return np.mean(m)

for T in [1.5, 2.27, 3.5]:
    print(f"T = {T}: average |m| = {simulate(T):.3f}")
```

```text
T = 1.5: average |m| = 0.987
T = 2.27: average |m| = 0.573
T = 3.5: average |m| = 0.061
```

Ordered, in between, disordered. The plain Python loop takes about 25 seconds; real codes update half of the spins at once in a checkerboard pattern.

---

## Common misconceptions

!!! warning "\"The Ising model is only about magnets.\""
    It is a model of **any** set of two-state things that influence their neighbours: atoms in an alloy, molecules in a gas (occupied / empty), neurons, pixels, opinions. Magnetism is just where it was born.

!!! warning "\"When it's hot, high-energy patterns are more likely than low-energy ones.\""
    Never. At every temperature a single low-energy pattern is more likely than a single high-energy one (27.3% against 3.7% in the 2×2 grid at $T = 2$). Disorder wins when it's hot only because there are so **many more** disordered patterns.

!!! warning "\"Below the critical temperature the spins point up.\""
    Up and down have exactly the same energy when $h = 0$. The grid picks one of the two at random and then sticks with it. This is called *spontaneous symmetry breaking*. Only a field $h \ne 0$ makes one direction better.

!!! warning "\"The Metropolis animation shows how a real magnet moves in time.\""
    Metropolis is a **sampling** method. It is built so that the patterns it visits have the right probabilities, not so that the path between them is physically realistic. The pictures at each temperature are trustworthy; the "movie" is only a rough guide.

!!! warning "\"Ising discovered the phase transition.\""
    Ising solved the 1-D chain, found no transition, and wrongly concluded that the model could not explain magnets. The 2-D transition was proved by Peierls (1936) and solved exactly by Onsager (1944).

---

## Check yourself

??? question "1. A chain of three spins is ↑ ↑ ↓ (two links, J = 1, h = 0). What is its energy?"
    The first link agrees ($-1$), the second disagrees ($+1$), so $E = -1 + 1 = 0$.

??? question "2. A spin points up and all four of its neighbours point up. At T = 1.5, how likely is Metropolis to flip it?"
    $\Delta E = 2 \cdot (+1) \cdot (+4) = +8$, so the probability is $e^{-8/1.5} = 0.48\%$. About one attempt in 200.

??? question "3. In the 2×2 grid at T = 2, which is more likely: finding the grid all up, or finding it in some mixed pattern?"
    **Some mixed pattern.** All up has probability 27.3%. Each mixed pattern has only 3.7%, but there are 12 of them, so together they have 44.4%.

??? question "4. Why does a chain of spins (1-D) never order, while a grid (2-D) does?"
    In a chain, flipping a block of any length breaks only 2 links, so big disordering islands are cheap at any temperature above zero. In a grid, an island costs energy in proportion to its perimeter, so big islands are rare when it's cold.

??? question "5. A model scores two words with logits 2 and 0. What is the probability of the first word at T = 1 and at T = 0.5?"
    At $T = 1$: $e^{2}/(e^{2} + e^{0}) = 88.1\%$. At $T = 0.5$: $e^{4}/(e^{4} + 1) = 98.2\%$. Lowering the temperature sharpens the choice, exactly as it orders the spins.

??? question "6. In the denoising energy, what plays the role of the field h, and why is it different for every pixel?"
    The observed pixel $y_i$. It pulls each pixel of the answer towards the value that was actually seen at that position, so it must be local. A single global $h$ would only make the whole picture darker or lighter.

---

## Further reading

- David MacKay, [*Information Theory, Inference, and Learning Algorithms*](https://www.inference.org.uk/mackay/itila/) (free online), Chapter 31 "Ising Models" and Chapters 42–43 on Hopfield networks and Boltzmann machines.
- Stephen G. Brush, [*History of the Lenz-Ising Model*](https://doi.org/10.1103/RevModPhys.39.883), Reviews of Modern Physics 39 (1967), 883–893. Who did what, and why both names belong on the model.
- Ernst Ising, [*Beitrag zur Theorie des Ferromagnetismus*](https://doi.org/10.1007/BF02980577), Zeitschrift für Physik 31 (1925), 253–258. The 1-D solution.
- Lars Onsager, [*Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition*](https://doi.org/10.1103/PhysRev.65.117), Physical Review 65 (1944), 117–149. The exact 2-D solution and $T_c$.
- Nicholas Metropolis, Arianna Rosenbluth, Marshall Rosenbluth, Augusta Teller and Edward Teller, [*Equation of State Calculations by Fast Computing Machines*](https://doi.org/10.1063/1.1699114), Journal of Chemical Physics 21 (1953), 1087–1092. The Metropolis algorithm.
- Stuart Geman and Donald Geman, [*Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images*](https://doi.org/10.1109/TPAMI.1984.4767596), IEEE TPAMI 6 (1984), 721–741. Image restoration with Ising-like models (Step 7).
- John Hopfield, [*Neural networks and physical systems with emergent collective computational abilities*](https://doi.org/10.1073/pnas.79.8.2554), PNAS 79 (1982), 2554–2558.
- David Ackley, Geoffrey Hinton and Terrence Sejnowski, [*A Learning Algorithm for Boltzmann Machines*](https://doi.org/10.1207/s15516709cog0901_7), Cognitive Science 9 (1985), 147–169.
- The Nobel Prize in Physics 2024, [summary and popular background](https://www.nobelprize.org/prizes/physics/2024/summary/): how Hopfield's and Hinton's networks grew out of spin models.

**Next lessons:** Hopfield networks and associative memory · Boltzmann machines

<script src="viz/ising-model.js"></script>
