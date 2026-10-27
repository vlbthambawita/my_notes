# Convolutional Tsetlin Machines

!!! abstract "TL;DR"
    A standard Tsetlin machine ties every literal to one fixed pixel, so a pattern seen in one place means nothing in another. A **convolutional Tsetlin machine** fixes this by looking at the image through a small **window** that slides over every position. Each rule is checked against every **patch** the window sees, and the rule's output for the whole image is the **OR** of those checks: it fires if the pattern is there *anywhere*. Learning is the same Recognize, Erase and Reject Feedback as before, with one new choice: when several patches match, the rule learns from **one of them, picked at random**. To let rules also say *where* a pattern may appear, each patch gets its window position as extra features, **thermometer-coded** so a rule can express a range like "in the upper half". The idea is the same sliding filter as a convolutional neural network, but each filter is a small picture you can read.

**Level:** intermediate · **Prerequisites:** [Classification by voting](../tsetlin-machine-classification/index.md) (rules voting for and against a class, the vote sum $v$, the Voting Margin $T$, and how feedback is allocated). The math needs nothing beyond AND, OR and counting.

---

## Why should I care?

Many interesting patterns can turn up anywhere:

- **Heart disease in an ECG.** A recording holds many heartbeats. One abnormal waveform may reveal disease, but you don't know *which* beat it is (Granmo, *An Introduction to Tsetlin Machines*, Ch. 4, §4.1). The model must find the waveform and learn what it looks like at the same time.
- **Objects in images.** A flower, a lesion on a skin photo, a nodule on an X-ray: the same thing can sit in the corner or the centre.
- **Two problems in one.** To learn what the pattern looks like you need to know where it is, and to find it you need to know what it looks like. Convolution solves both together.

A standard Tsetlin machine (the previous lessons) would have to learn one rule for every place the pattern can appear. A convolutional one learns the pattern **once** and recognises it everywhere. And unlike the filters of a convolutional neural network, every learned filter is a small black-and-white picture that you can read and check.

---

## Step 1 — One pattern, many places

The chapter's running example is tiny on purpose: eight black-and-white images of $4 \times 4$ pixels, from two classes, **Circle** (four black pixels around a white centre) and **Cross** (the same four plus a black centre). Each shape appears in four different places (Granmo, Ch. 4, §4.2, Table 4.1).

As in [lesson 1](../tsetlin-machine-intro/index.md), each pixel becomes one Boolean feature: True for black, False for white. Name them $x_{i,j}$, with $i$ the row and $j$ the column, so a $4 \times 4$ image has 16 features (32 literals with their negations).

A standard Tsetlin machine needs a separate rule for each position, because "the pixel at row 2, column 1 is black" is a different literal from "the pixel at row 3, column 2 is black". The chapter's rule for a Circle in the top-left corner is

$$
R_1: \quad \text{if } x_{1,2} \wedge x_{2,1} \wedge \lnot x_{2,2} \wedge x_{2,3} \wedge x_{3,2} \text{ then Circle,}
$$

and a Circle one pixel to the right needs a different rule, $R_2$, with every index shifted. Four positions times two shapes gives **eight rules**, and a bigger image would need even more.

**Try it:** switch between the two views.

<div id="viz-shapes" class="viz"><div class="viz-title">🔍 Interactive: eight position-specific rules, or two window rules</div></div>

!!! tip "What to notice"
    - **Without a window**, the eight rule pictures are the same shape moved around. The machine would have to discover the same pattern four times, from a quarter of the examples each time.
    - **With a 3×3 window**, two rules cover all eight images. The window is placed wherever the shape is, and the rule only describes what is *inside* the window.
    - The window rules use the same five literals as $R_1$ ($R_5$ for Circle), with $x_{i,j}$ now meaning "row $i$, column $j$ **of the window**", not of the image.

---

## Step 2 — Looking through a window: patches and patch features

The fix is to give the Tsetlin machine a **local view**. Put a rectangular **convolution window** on the image, say $3 \times 3$ pixels, and use only the pixels inside it as features. The pixels the window covers form an **image patch**, and their truth values are the **patch features** $x_{1,1}, \dots, x_{3,3}$ (Granmo, Ch. 4, §4.2).

Now slide the window to every position where it fits. Name each position by the column $x$ and row $y$ of the window's top-left corner, counting from 0. A $4 \times 4$ image has room for a $3 \times 3$ window at four positions: $(0,0)$, $(1,0)$, $(0,1)$ and $(1,1)$. In general, a window of $w \times w$ on an image of $W \times H$ pixels has

$$
(W - w + 1) \times (H - w + 1)
$$

positions, so a $6 \times 6$ image gives 16 patches.

The key property: **every patch has the same nine features**, whatever its position. Patch feature $x_{2,2}$ always means "the centre of the window", whether the window is in the top-left corner or the bottom-right. So a rule written in patch features is automatically position-free, and the chapter's two rules

$$
\begin{aligned}
R_5 &: \text{ if } x_{1,2} \wedge x_{2,1} \wedge \lnot x_{2,2} \wedge x_{2,3} \wedge x_{3,2} \text{ then Circle} \\
R_6 &: \text{ if } x_{1,2} \wedge x_{2,1} \wedge x_{2,2} \wedge x_{2,3} \wedge x_{3,2} \text{ then Cross}
\end{aligned}
$$

replace all eight. Making the image bigger only adds window positions; the rules stay the same.

!!! note "Time series: 1-D convolution"
    An ECG is a sequence, not an image. Describe each moment in time by a few features (one column per time step), and slide the window **only along time**. That is 1-D convolution: the same idea, with the window moving in one direction (Granmo, Ch. 4, §4.2). The next widget has a "1-D sequence" image of 3 features × 8 time steps, where the window can only move sideways.

!!! tip "What to notice"
    - The window gives the machine a way to **direct its attention**: it looks at one piece of the image at a time.
    - A $3 \times 3$ window has 9 patch features, so every rule chooses from 18 literals, however large the image is. A standard Tsetlin machine on a $6 \times 6$ image would have 72 literals per rule.

---

## Step 3 — Evaluating a rule: match every patch, then OR

With several patches per image, a rule no longer gives one truth value directly. The chapter's procedure (Granmo, Ch. 4, §4.3, Figure 4.3) is:

<div class="viz diagram">
<script type="application/json">
{
  "title": "🪟 Evaluating one rule on one image",
  "cols": 3,
  "rows": 2,
  "rowHeight": 110,
  "nodes": [
    {"id": "img", "col": 0, "row": 0, "icon": "🖼️", "label": "Input image", "sub": "one Boolean feature per pixel", "role": "data", "note": "Black pixel = True, white pixel = False."},
    {"id": "win", "col": 1, "row": 0, "icon": "🪟", "label": "Slide the window", "sub": "every position (x, y)", "role": "data", "note": "A 3×3 window on a 4×4 image fits at (0,0), (1,0), (0,1) and (1,1)."},
    {"id": "feat", "col": 2, "row": 0, "icon": "🔢", "label": "Patch features", "sub": "x<sub>1,1</sub> … x<sub>3,3</sub> per patch", "role": "data", "note": "The same nine features for every patch, whatever its position."},
    {"id": "match", "col": 2, "row": 1, "icon": "✅", "label": "Match the rule", "sub": "True or False per patch", "role": "model", "note": "A patch matches if every memorized literal of the rule is True in that patch."},
    {"id": "or", "col": 1, "row": 1, "icon": "∨", "label": "OR over patches", "sub": "True if any patch matches", "role": "model", "note": "False OR False OR True OR False = True."},
    {"id": "vote", "col": 0, "row": 1, "icon": "🗳️", "label": "One output per rule", "sub": "then vote as before", "role": "loss", "note": "From here on, classification is exactly the voting from the previous lesson."}
  ],
  "edges": [
    {"from": "img", "to": "win"},
    {"from": "win", "to": "feat"},
    {"from": "feat", "to": "match"},
    {"from": "match", "to": "or"},
    {"from": "or", "to": "vote"}
  ],
  "steps": [
    {"show": ["img"], "text": "Start from the image as Boolean features: True for black, False for white."},
    {"show": ["img>win", "win"], "text": "Place the <b>convolution window</b> at every position where it fits. Each position cuts out one <b>image patch</b>."},
    {"show": ["win>feat", "feat"], "text": "Turn each patch into <b>patch features</b> x<sub>1,1</sub> … x<sub>3,3</sub>. They mean the same thing for every patch."},
    {"show": ["feat>match", "match"], "text": "<b>Match</b> the rule against each patch separately: one True or False per patch."},
    {"show": ["match>or", "or"], "text": "Combine the matches with <b>OR</b>: the rule is True for the image if it matches <b>at least one</b> patch."},
    {"show": ["or>vote", "vote"], "text": "Each rule now has <b>one</b> truth value for the image, so the vote sum and the classification work exactly as in a standard Tsetlin machine."}
  ],
  "narrow": {"cols": 2, "rows": 3, "pos": {"img": [0, 0], "win": [1, 0], "feat": [1, 1], "match": [0, 1], "or": [0, 2], "vote": [1, 2]}}
}
</script>
</div>

Write $I_j$ for the set of literals that rule $C_j$ has memorized, and $l_k^{(p)}$ for the truth value of literal $l_k$ in patch $p$. Then the rule's output for the image is

$$
C_j(\text{image}) \;=\; \bigvee_{p=1}^{P} \;\bigwedge_{k \in I_j} l_k^{(p)} ,
$$

an AND inside each patch (as before) and an OR across the $P$ patches (new). An empty rule (nothing memorized) matches every patch, so it is True, just as an empty rule was always True in a standard Tsetlin machine.

**Try it:** the widget starts with the chapter's example image and rule $R_5$. Step the window through the four positions, then paint your own images and rules.

<div id="viz-slide" class="viz"><div class="viz-title">🪟 Interactive: slide the window, match the rule, OR the results</div></div>

!!! tip "What to notice"
    - **The chapter's example.** $R_5$ fails at $(0,0)$, $(1,0)$ and $(1,1)$ and matches at $(0,1)$, where the window sits exactly on the Circle. So the rule is False OR False OR True OR False = **True** for the image (we checked this in Python too).
    - **Many matches still give one vote.** Load the bigger image: $R_5$ matches once and $R_6$ matches once. Paint a second Circle somewhere: the match map shows two Ts, but the rule's output is still a single True. A rule says "the pattern is present", not "how many times".
    - **The match map is a feature map.** The grid of Ts and Fs, one cell per window position, is exactly what a convolutional neural network calls a *feature map*. The OR squeezes it into one bit.
    - **1-D sequence:** the window can only move sideways, and the Circle (left) and Cross (right) are found at different time steps by the same rules.

---

## Step 4 — Learning: which patch does the rule learn from?

Learning uses the same three kinds of feedback as before ([Recognize, Erase & Reject feedback](../tsetlin-machine-feedback/index.md)), and the same rule for *how often*: with vote sum $v$ for the image's true class, clipped to $[-T, T]$, each rule is selected for feedback with probability

$$
\frac{T - v}{2T}.
$$

What changes is one detail. Recognize and Reject Feedback update the memory **using the truth values of the literals**: memorize the True ones, forget (or, for Reject, memorize) the False ones. With convolution, the rule saw many patches, and their literals disagree. Which patch's literals should it use?

The chapter's answer (Granmo, Ch. 4, §4.4, Figure 4.5) is **random selection among the patches the rule matched**:

| Rule's output on this image | Rule belongs to the image's class | Rule belongs to the other class |
|---|---|---|
| **True** (matches at least one patch) | **Recognize Feedback**, using the literals of **one matching patch picked at random** | **Reject Feedback**, using **one matching patch picked at random** |
| **False** (matches no patch) | **Erase Feedback**: forget literals at random; no patch needed | no feedback |

Why a *matching* patch? Recognize Feedback should make the rule more like what it already recognised, so it must learn from a patch where it fired. Reject Feedback should stop the rule firing on the wrong class, so it must learn from a patch where it fired wrongly. Why *random*? Every matching patch is equally good evidence, and picking at random lets the rule, over many images, drift towards what all its matches have in common. Erase Feedback never looks at truth values, so it needs no patch at all.

!!! note "How a pattern gets found from nothing"
    At the very start every rule is empty, so it matches **every** patch, including empty background. Recognize Feedback then learns from a random patch, often background. But Erase and Reject Feedback keep knocking out literals that don't help, and only patches that contain the shape show up again and again in images of one class. In the next widget, the first step of seed 1 shows it: all 8 rules fire on the first image, so $v = 4 - 4 = 0$, each rule is selected with probability $\tfrac{4 - 0}{8} = 0.5$, and the selected rules each learn from 1 of the 16 patches, picked at random.

!!! tip "What to notice"
    - The **only** new ingredient in learning is "pick one matching patch at random". Everything else (the vote sum, the Voting Margin, the selection probability, the three feedback types) is unchanged.
    - Because rules can now specialise in different kinds of patches, it helps to let **more rules work together** on each image. The chapter does this with a Voting Margin $T$ larger than 2: the vote sum must reach $T$, so more rules are recruited (Granmo, Ch. 4, §4.4, Remark).

---

## Step 5 — Training a convolutional Tsetlin machine

Let's make the chapter's task harder. Each image is now $6 \times 6$ with **one Circle or Cross anywhere** (16 possible places) plus **two random black noise pixels** outside the shape. There are 300 training images and 300 test images. The machine has 4 rules per class, Voting Margin $T = 4$, Forget Value 0.7 (so Memorize Value 0.3), and the book's memory depth of 10 positions per literal (Memorized = positions 6 to 10). Everything is seeded, and the widget runs exactly the same code as our Python check, so it shows the numbers below.

**Try it:** press *Train to 20 epochs*, then click the rule pictures. Use *Step: one image* (after *Reset*) to watch individual updates.

<div id="viz-learn" class="viz"><div class="viz-title">🧠 Interactive: train a convolutional Tsetlin machine on Circle vs Cross</div></div>

!!! tip "What to notice"
    - **It learns fast.** With seed 1, test accuracy is **90.7%** after one epoch and **100%** after two. It wobbles slightly on the way (98.7% at epoch 10) and ends at **100%**. Seeds 1 to 10 all end at 100% test accuracy.
    - **The rules are readable.** All four Circle rules learned the same picture: the four black pixels of a Circle, a white centre, and **white corners**. That is the chapter's $R_5$ plus four extra "must be white" literals. Two Cross rules learned $R_6$ plus white corners. Each of these fires on all 147 test Circles (or all 153 test Crosses) and on none of the other class.
    - **Not every rule is useful.** One Cross rule ended up as a single black pixel: it fires on all 300 test images, so it adds the same vote to every image and changes nothing. Another, "two black pixels side by side", fires on 117 of the 153 Crosses but also on 13 Circles (where a noise pixel lands next to the shape). The clean rules outvote it.
    - **Switch to a 6×6 window.** A 6×6 window on a 6×6 image has one position, so this *is* a standard Tsetlin machine with 72 literals per rule. With the same settings it reaches only **54.7%** test accuracy (seed 1), and **51.8%** on average over seeds 1 to 10: barely better than guessing. Even with 32 rules per class, $T = 16$, Forget Value 0.1 and 30 epochs, our Python runs only reached 62.7% to 65.7% (seeds 1 to 3). It would need a rule per shape *and* per position, and about 9 training images per shape and place is too few to learn them.
    - **Train on the left half only.** Now training shapes appear only in the left half, and the test shapes only in the right half, in places the machine has never seen a shape. The convolutional machine still scores **100%** (all 10 seeds): its rules never mention where. The standard one drops to **49.0%** (seed 1; 48.6% on average), a coin flip.

In the chapter's example of a truck photo (Granmo, Ch. 4, Figure 4.4), six different rules match different parts of the image: one captures the dark parts of the truck, another the ground beneath it. That is the same effect at a larger scale: each rule specialises in one kind of patch, and the vote combines them.

---

## Step 6 — Position encoding: letting a rule say *where*

Sometimes position **does** matter. In a handwritten 7, a horizontal stroke at the **top** is a strong clue, while the same stroke at the bottom belongs to a 2 (Granmo, Ch. 4, §4.5, Figure 4.6). A pure convolutional rule can't tell them apart: it fires wherever the stroke is. We want rules that *may* restrict a pattern to a region, without being forced to.

The fix: add the **window's position** to each patch's features. The chapter uses **thermometer encoding**. An integer from 0 to 3 is written with 3 Boolean features, by switching on features from the bottom up, like the mercury in a thermometer:

| value | $X_{\ge 1}$ | $X_{\ge 2}$ | $X_{\ge 3}$ |
|---|---|---|---|
| 0 | F | F | F |
| 1 | **T** | F | F |
| 2 | **T** | **T** | F |
| 3 | **T** | **T** | **T** |

Feature $X_{\ge t}$ simply means "the value is at least $t$". That makes ranges easy to write with literals:

- $X_{\ge 1}$ alone means "1 or more".
- $\lnot X_{\ge 3}$ means "less than 3".
- $X_{\ge 1} \wedge \lnot X_{\ge 3}$ means "1 or 2": a range, in a single rule.

Ordinary binary encoding would need only 2 bits for 4 values, and could say things like "even" in one literal. But a range like "1 to 2" would no longer be a single AND, and ranges are what you need to describe a **rectangular region** of an image (Granmo, Ch. 4, §4.5).

Each patch now gets two thermometers, one for its window column $x$ and one for its row $y$ (Granmo, Ch. 4, Figure 4.8). On a $6 \times 6$ image with a $3 \times 3$ window, $x$ and $y$ run from 0 to 3, so that is 3 + 3 = 6 position features, and each patch has 9 + 6 = 15 features, or 30 literals. A rule that forgets all position literals is still position-free; a rule that memorizes some of them only matches patches **in a rectangle** of window positions.

**Try it:** set the allowed region, then click pixels to move the Circle in and out of it. Then press *Learn* to train a machine on a task where position is the whole point.

<div id="viz-position" class="viz"><div class="viz-title">🌡️ Interactive: thermometer-coded positions make a rule location-sensitive</div></div>

!!! tip "What to notice"
    - **Same pattern, different answer.** With *Upper half* ($\lnot Y_{\ge 2}$, so window row 0 or 1), the rule fires for a Circle in the top rows and stays False when you move the same Circle down. The pattern part matched both times (✓ or ✗ in the grid); only the position literals changed.
    - **The learning task.** "Is the Circle in the upper or the lower half?" (6×6 images, 2 noise pixels, same settings as Step 5). Without position literals, the machine can only reach **73.3%** test accuracy (seed 1; **61.0%** on average over seeds 1 to 10). It does a little better than a coin flip only through quirks of the data, such as where the noise pixels can fall. With position literals it reaches **100%** (seed 1; **98.9%** on average).
    - **The learned rules say exactly what we'd hope.** With seed 1, two "upper" rules learned the full Circle picture plus $\lnot Y_{\ge 2} \wedge \lnot Y_{\ge 3}$, i.e. "a Circle with its window in row 0 or 1". Each fires on all 152 upper-half test images and on none of the 148 lower-half ones. Some other rules learned partial Circles with looser regions, which also work.

---

## Step 7 — The link to convolutional neural networks

If you have met **convolutional neural networks** (CNNs), all of this should look familiar. The two share the sliding-window idea; they differ in what slides.

| | Convolutional neural network | Convolutional Tsetlin machine |
|---|---|---|
| What slides over the image | a **filter**: a small grid of real-valued weights | a **rule**: an AND of literals on the patch features |
| Result at one position | a weighted sum, then a non-linearity (a number) | match or no match (one bit) |
| Result over all positions | a feature map | a match map (Step 3 widget) |
| Squeezing the map to one value | (global) **max pooling** | **OR**, which is exactly the max of a map of 0s and 1s |
| Learning signal goes to | the position that won the max (back-propagation through max pooling) | **one matching patch, picked at random** |
| Reading the learned filter | a grid of numbers; needs visualisation tools | a picture of must-be-black, must-be-white and don't-care pixels |
| Position | mostly discarded by pooling; can be added (e.g. extra coordinate channels) | thermometer-coded position literals the rule may use or ignore |

So a convolutional Tsetlin machine is roughly "one convolutional layer plus global max pooling, with logic rules as filters". A CNN stacks many layers, so it can build large patterns from small ones; the chapter's Tsetlin machine has one layer of rules over the patches. What you gain is that every filter is a statement you can check by eye: "a Circle is four black pixels around a white centre, with white corners".

!!! tip "What to notice"
    - **OR is max pooling for Booleans.** Both throw away *where* the pattern was found and keep *whether* it was found. That is what makes both of them position-free.
    - Both also send the learning signal through **one** position. A CNN chooses the position with the highest response; the Tsetlin machine picks at random among the positions that matched, because all matches are equally "high" (they are all True).

---

## Putting it all together

The complete learning loop (Granmo, Ch. 4, §4.4) is the standard Tsetlin machine loop from the [previous lesson](../tsetlin-machine-classification/index.md) with two convolution steps added.

<div class="viz diagram">
<script type="application/json">
{
  "title": "🔁 How a convolutional Tsetlin machine learns",
  "cols": 3,
  "rows": 2,
  "rowHeight": 115,
  "nodes": [
    {"id": "obs", "col": 0, "row": 0, "icon": "🖼️", "label": "Observe an image and its class", "role": "data"},
    {"id": "patch", "col": 1, "row": 0, "icon": "🪟", "label": "Cut it into patches", "sub": "Booleanize; add position", "role": "data", "note": "Slide the window over the image; each patch gives its pixel features and (optionally) thermometer-coded x and y."},
    {"id": "eval", "col": 2, "row": 0, "icon": "∨", "label": "Evaluate every rule", "sub": "match each patch, then OR", "role": "model", "note": "A rule is True for the image if it matches at least one patch."},
    {"id": "vote", "col": 2, "row": 1, "icon": "🗳️", "label": "Vote sum v", "sub": "clipped to [−T, T]", "role": "loss", "note": "Rules of the image's class that are True add 1; rules of the other class that are True subtract 1."},
    {"id": "sel", "col": 1, "row": 1, "icon": "🎲", "label": "Select rules", "sub": "probability (T − v) / 2T", "role": "loss", "note": "Far from the Voting Margin: many rules learn. At or beyond it: none do."},
    {"id": "fb", "col": 0, "row": 1, "icon": "✏️", "label": "Feedback", "sub": "using 1 random matching patch", "role": "opt", "note": "Own class, True: Recognize. Own class, False: Erase (no patch). Other class, True: Reject. Other class, False: nothing."}
  ],
  "edges": [
    {"from": "obs", "to": "patch"},
    {"from": "patch", "to": "eval"},
    {"from": "eval", "to": "vote"},
    {"from": "vote", "to": "sel"},
    {"from": "sel", "to": "fb"},
    {"from": "fb", "to": "obs", "label": "next image"}
  ],
  "steps": [
    {"show": ["obs"], "text": "Take a training image and its class."},
    {"show": ["obs>patch", "patch"], "text": "<b>New:</b> slide the window over it. Each position gives a patch with the same patch features, plus thermometer-coded x and y if you use position encoding."},
    {"show": ["patch>eval", "eval"], "text": "<b>New:</b> each rule is matched against every patch, and the matches are OR-ed into one truth value."},
    {"show": ["eval>vote", "vote"], "text": "As before: add the votes for the image's class, subtract the votes for the other class, clip to the Voting Margin T."},
    {"show": ["vote>sel", "sel"], "text": "As before: each rule is selected for feedback with probability (T − v) / 2T."},
    {"show": ["sel>fb", "fb"], "text": "<b>New detail:</b> Recognize and Reject Feedback use the literals of <b>one matching patch, picked at random</b>. Erase Feedback needs no patch."},
    {"show": ["fb>obs", "obs"], "text": "Repeat for the next image, epoch after epoch."}
  ],
  "narrow": {"cols": 2, "rows": 3, "pos": {"obs": [0, 0], "patch": [1, 0], "eval": [1, 1], "vote": [1, 2], "sel": [0, 2], "fb": [0, 1]}}
}
</script>
</div>

Here is a complete convolutional Tsetlin machine in numpy, on the Step 5 task. It is shorter than the widget's code, and uses numpy's random numbers, so its exact numbers differ from the widget's.

```python
import numpy as np

rng = np.random.default_rng(0)
CIRCLE = [(-1, 0), (0, -1), (0, 1), (1, 0)]          # the chapter's two shapes (offsets from the centre)
CROSS = CIRCLE + [(0, 0)]


def make_image(cls):
    """A 6x6 image: a Circle (0) or Cross (1) at a random place, plus 2 random noise pixels."""
    img = np.zeros((6, 6), bool)
    r, c = rng.integers(1, 5, size=2)
    for dr, dc in (CIRCLE, CROSS)[cls]:
        img[r + dr, c + dc] = True
    while img.sum() < len((CIRCLE, CROSS)[cls]) + 2:
        rr, cc = rng.integers(0, 6, size=2)
        if max(abs(rr - r), abs(cc - c)) > 1:
            img[rr, cc] = True
    return img


def patches(img, w=3):
    """Slide a w x w window over the image; each patch -> its literals [x_1..x_9, NOT x_1..NOT x_9]."""
    P = np.array([img[y:y + w, x:x + w].ravel()
                  for y in range(img.shape[0] - w + 1) for x in range(img.shape[1] - w + 1)])
    return np.hstack([P, ~P])


class ConvTM:
    def __init__(self, n_literals, rules_per_class=4, T=4, forget=0.7, depth=10):
        self.half = depth // 2                                            # memory positions 1 .. depth;
        self.state = np.full((2 * rules_per_class, n_literals), self.half)  # start just below Memorized
        self.cls = np.repeat([0, 1], rules_per_class)                     # which class each rule votes for
        self.T, self.F, self.depth = T, forget, depth

    def match(self, P):
        """(rules x patches): a rule matches a patch if none of its memorized literals is False there."""
        memorized = self.state > self.half
        return ~(memorized[:, None, :] & ~P[None, :, :]).any(axis=2)

    def predict(self, P):
        fires = self.match(P).any(axis=1)                                 # OR over all patches
        return 0 if fires[self.cls == 0].sum() >= fires[self.cls == 1].sum() else 1

    def update(self, P, y):
        m = self.match(P)
        fires = m.any(axis=1)
        v = np.clip(fires[self.cls == y].sum() - fires[self.cls != y].sum(), -self.T, self.T)
        for j in np.flatnonzero(rng.random(len(self.cls)) <= (self.T - v) / (2 * self.T)):
            s = self.state[j]
            if fires[j]:
                p = P[rng.choice(np.flatnonzero(m[j]))]                  # ONE matching patch, at random
                if self.cls[j] == y:                                      # Recognize Feedback
                    up = p & (rng.random(len(s)) <= 1 - self.F)
                    down = ~p & (rng.random(len(s)) <= self.F)
                    s += up & (s < self.depth)
                    s -= down & (s > 1)
                else:                                                     # Reject Feedback
                    s += ~p & (s <= self.half)
            elif self.cls[j] == y:                                        # Erase Feedback (no patch needed)
                s -= (rng.random(len(s)) <= self.F) & (s > 1)


ytr, yte = rng.integers(0, 2, 300), rng.integers(0, 2, 300)
Ptr = [patches(make_image(c)) for c in ytr]
Pte = [patches(make_image(c)) for c in yte]
tm = ConvTM(n_literals=18)
for epoch in range(1, 21):
    for i in rng.permutation(len(Ptr)):
        tm.update(Ptr[i], ytr[i])
    if epoch in (1, 5, 20):
        acc = np.mean([tm.predict(P) == y for P, y in zip(Pte, yte)])
        print(f"epoch {epoch:2d}: test accuracy {acc:.3f}")

for j in range(len(tm.cls)):                                              # draw each rule as a 3x3 picture
    on, off = tm.state[j, :9] > tm.half, tm.state[j, 9:] > tm.half
    pic = np.where(on, "#", np.where(off, "o", "."))
    print(("Circle" if tm.cls[j] == 0 else "Cross "), " | ".join("".join(row) for row in pic.reshape(3, 3)))
```

```text
epoch  1: test accuracy 0.990
epoch  5: test accuracy 1.000
epoch 20: test accuracy 1.000
Circle o#o | #o# | o#o
Circle ... | ... | #..
Circle o#o | #o# | o#o
Circle ... | ... | ...
Cross  o#o | ### | o#o
Cross  o#o | ### | o#o
Cross  o#o | ### | o#o
Cross  o#o | ### | o#o
```

`#` = must be black, `o` = must be white, `.` = don't care. Again the Circle and Cross pictures appear, each with white corners, plus two Circle rules that ended up (nearly) empty and so fire on (nearly) every image. To add position encoding, append the thermometer features of `x` and `y` to each patch in `patches` before the negations.

---

## Common misconceptions

!!! warning "\"The window makes the machine see less of the image.\""
    The window is placed at **every** position, so every pixel is seen, in every patch that covers it. What the window removes is the pixel's *absolute* position, which is exactly what makes a rule reusable everywhere.

!!! warning "\"A rule that matches in three places casts three votes.\""
    No. The matches are combined with OR, so a rule gives **one** truth value per image and at most one vote, however many patches match. In Step 3, painting a second Circle doesn't change $R_5$'s output.

!!! warning "\"Feedback updates the rule with every matching patch.\""
    Recognize and Reject Feedback use **one** matching patch, chosen at random. Over many images, the random choices average out to what the matching patches have in common.

!!! warning "\"Position encoding turns it back into a standard Tsetlin machine.\""
    Position literals are optional for each rule. A rule that forgets all of them is as position-free as before; a rule only memorizes them when restricting the region helps (like the upper-half rules in Step 6). And a region is a whole rectangle of window positions, not one fixed pixel.

!!! warning "\"Binary encoding would be just as good for positions, and shorter.\""
    Two bits can hold the four positions 0 to 3, but then "rows 1 to 2" is not a single AND of literals. Thermometer encoding spends one feature per step so that every range, and so every rectangle, is one AND (Granmo, Ch. 4, §4.5).

---

## Check yourself

??? question "1. How many window positions does a 3×3 window have on an 8×8 image? On a 28×28 image with a 10×10 window?"
    $(8 - 3 + 1)^2 = 6^2 = $ **36** positions. $(28 - 10 + 1)^2 = 19^2 = $ **361** positions. Each rule is matched against every one of them, and its output is the OR of all the matches.

??? question "2. A Circle rule matches no patch of a Circle image, and is selected for feedback. What happens, and which patch is used?"
    The rule belongs to the image's class and its output is False, so it gets **Erase Feedback**: every literal moves one step towards Forgotten with probability equal to the Forget Value. No patch is used, because Erase Feedback doesn't look at truth values.

??? question "3. A Circle rule matches 3 of the 16 patches of a Cross image, and is selected. What happens?"
    The rule fired on the wrong class, so it gets **Reject Feedback**. One of the 3 matching patches is picked at random, and every *forgotten* literal that is False in that patch moves one step towards Memorized. Once one of them crosses into Memorized, the rule no longer matches that kind of patch.

??? question "4. Window column x runs from 0 to 5. How many thermometer features do you need, and how do you write \"x is 2, 3 or 4\"?"
    Five features $X_{\ge 1}, \dots, X_{\ge 5}$ (one per step above 0). "At least 2" is $X_{\ge 2}$; "less than 5" is $\lnot X_{\ge 5}$. So the range is $X_{\ge 2} \wedge \lnot X_{\ge 5}$.

??? question "5. Why does the standard Tsetlin machine fail when it is trained on shapes in the left half and tested on shapes in the right half?"
    Its literals refer to fixed pixels. In training, the pixels of the right half were only ever background or noise, so no rule learned anything about shapes there. In our run it scored 49.0% (seed 1), a coin flip. The convolutional machine's rules talk about **window** pixels, which are the same wherever the window is, so it scored 100%.

??? question "6. In the Step 3 widget, the empty rule is True for every image. Why, and why is such a rule harmless in a trained machine?"
    With nothing memorized, no literal can be False, so it matches every patch, and the OR is True. An always-True rule adds the same vote to every image, so it carries no information about the class: the other rules simply have to outvote it, and the Voting Margin pushes learning to make them do so. In Step 5, the Cross rule that became a single black pixel behaves almost the same way: it fired on all 300 test images, and the machine still scored 100%.

---

## Further reading

- Ole-Christoffer Granmo, [*An Introduction to Tsetlin Machines*](https://tsetlinmachine.org/) (free online), Ch. 4 "Convolution": §4.1 recognizing patterns in time and space, §4.2 input and patterns (the Circle/Cross images, convolution window, 1-D convolution), §4.3 rule evaluation, §4.4 rule learning (random matching patch, Voting Margin above 2), §4.5 position encoding (thermometer encoding). This lesson's diagrams and widgets are redrawn from the chapter's figures; the 6×6 noisy Circle/Cross task is our own extension of its example.
- Ole-Christoffer Granmo, Sondre Glimsdal, Lei Jiao, Morten Goodwin, Christian W. Omlin and Geir Thore Berge, [*The Convolutional Tsetlin Machine*](https://arxiv.org/abs/1905.09688) (2019). The research paper, in the literature's terms (clauses, Type I/II feedback), with results on MNIST-style image benchmarks.

**Next lessons:** [Confidence & composites](../tsetlin-machine-confidence/index.md): does a Tsetlin machine know when it is wrong, and how to combine several specialised machines.

<script src="viz/convolutional-tsetlin-machine.js"></script>
