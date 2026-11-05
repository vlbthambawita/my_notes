# Weighted Clauses and Clause Sharing

!!! abstract "TL;DR"
    A trained Tsetlin machine often holds several **copies of the same rule**, and each copy is just one more vote. A **weighted** Tsetlin machine keeps one copy and gives it a **weight** $w$: when the rule fires, it casts $w$ votes. The weights are learned alongside the rules with two tiny updates: **+1** when a firing rule gets Recognize Feedback, **−1** when it gets Reject Feedback. A rule that is right most of the time it fires climbs to a large weight; one that is right only half the time sinks to 0 and is switched off. **Clause sharing** goes one step further: instead of a separate team of rules for every class, all classes share **one pool of rules**, and each rule has a **signed weight for every class** (positive: it votes for that class, negative: against). On a toy task with ten classes and a budget of only 20 rules, our runs give 74.4% for the plain machine, 81.1% with weights and 85.8% with a shared pool, close to the 87.3% that is the best you can do on this data.

**Level:** intermediate · **Prerequisites:** [Classification by voting](../tsetlin-machine-classification/index.md) (rules voting for and against a class, the vote sum $v$, the Voting Margin $T$, and multi-class teams). [Convolutional Tsetlin machines](../convolutional-tsetlin-machine/index.md) is useful background but not needed.

!!! note "Beyond the book"
    Chapter 5 of Ole-Christoffer Granmo's book, *Pattern Weighing and Sharing*, is not published yet. This lesson is therefore based on the research papers listed under [Further reading](#further-reading): the real-valued weighted Tsetlin machine (Phoulady et al., 2020), the integer-weighted Tsetlin machine (Abeyrathna, Granmo and Goodwin, 2021), the coalesced Tsetlin machine with clause sharing (Glimsdal and Granmo, 2021), and the open-source TMU library that implements them. It uses the book's vocabulary from the earlier lessons. All the experiments are our own small runs, made for this lesson.

---

## Why should I care?

In the [classification lesson](../tsetlin-machine-classification/index.md), two things kept happening:

- **Rules came in copies.** The book's example used two identical rules, R1 and R2, so that their two votes could outvote R3. The trained breast-cancer machine had four identical rules memorizing *"Inv-nodes 24-26"*. Every copy costs memory and computing time, and makes the model longer to read, without saying anything new.
- **Each class had its own team.** A three-class machine trains three separate teams. If two classes share a useful pattern, each team has to discover it on its own.

Clinical reasoning works differently on both counts:

- **Findings carry different weights.** Many clinical scores give some findings two points and others one. A weight says "this rule counts more" directly, instead of through copies.
- **Findings are shared between diagnoses.** Fever is evidence *for* several infections and *against* others. One rule, *"if fever"*, can serve all of those diagnoses at once, with a different weight for each.

The payoff reported in the papers is large. Phoulady et al. (2020) report that a weighted machine matched the accuracy of a standard Tsetlin machine with only 1/4 of the rules on MNIST, 1/3 on IMDb and 1/50 on Connect-4. Abeyrathna et al. (2021) report that integer weights needed, on average over their datasets, 6.5 times fewer literals than the standard machine. Glimsdal and Granmo (2021) report that a shared pool helps most when rules are scarce.

---

## Step 1 — Copies are weights

Look at the book's example again: R1 and R2 are the same rule, *"Deg-malig 3 and not Menopause lt40"*. When it fires, the vote sum gets +2. You could just as well keep **one** copy and say that it is worth **two votes**. That number is the rule's **weight** $w_j$.

With weights, the vote sum becomes a **weighted** vote sum (Phoulady et al., 2020):

$$
v \;=\; \sum_{j \,\in\, \text{for}} w_j \, C_j(x) \;-\; \sum_{j \,\in\, \text{against}} w_j \, C_j(x),
$$

where $C_j(x)$ is 1 if rule $j$ fires on example $x$ and 0 otherwise. (In the papers the rules are *clauses* $C^+_j$ and $C^-_j$ of positive and negative polarity, and the machine is a *Weighted Tsetlin Machine*.) Everything else stays as it was:

- **Prediction:** the sign of $v$ for two classes, the class with the highest $v$ for several.
- **Training:** $v$ is clipped to $[-T, T]$ and every rule is picked for feedback with the same probability $(T - v)/2T$ as before. Recognize, Erase and Reject Feedback change the rules' memories exactly as in the [feedback lesson](../tsetlin-machine-feedback/index.md).

<div class="viz diagram">
<script type="application/json">
{
  "title": "🧮 Three copies of a rule, or one rule with weight 3",
  "cols": 3,
  "rows": 4.3,
  "rowHeight": 80,
  "groups": [
    {"id": "copies", "label": "3 identical rules", "col": 0, "row": 0, "cols": 3, "rows": 1.5}
  ],
  "nodes": [
    {"id": "c1", "col": 0, "row": 0.7, "label": "Deg-malig 3 ∧ ¬lt40", "sub": "+1 vote", "role": "model", "note": "Copy 1 of the rule. When it fires, it adds one vote."},
    {"id": "c2", "col": 1, "row": 0.7, "label": "Deg-malig 3 ∧ ¬lt40", "sub": "+1 vote", "role": "model", "note": "Copy 2: the same literals, so it always fires together with copy 1."},
    {"id": "c3", "col": 2, "row": 0.7, "label": "Deg-malig 3 ∧ ¬lt40", "sub": "+1 vote", "role": "model", "note": "Copy 3: again the same rule."},
    {"id": "one", "col": 1, "row": 2.2, "icon": "⚖️", "label": "One rule, weight 3", "sub": "Deg-malig 3 ∧ ¬lt40", "role": "loss", "note": "One copy of the literals plus one small integer: the weight."},
    {"id": "vote", "col": 1, "row": 3.35, "icon": "🗳️", "label": "Vote sum v", "sub": "+3 when it fires", "shape": "pill", "role": "loss", "note": "The weighted vote sum: each firing rule adds its weight."}
  ],
  "edges": [
    {"from": "copies", "to": "one", "label": "merge"},
    {"from": "one", "to": "vote"}
  ],
  "steps": [
    {"show": ["copies", "c1", "c2", "c3"], "text": "A trained Tsetlin machine often holds identical rules. They always fire together, so three copies simply add <b>three votes</b>."},
    {"show": ["copies>one", "one"], "text": "Keep one copy and give it <b>weight 3</b>. The model now stores the literals once, and evaluates the rule once."},
    {"show": ["one>vote", "vote"], "text": "When the rule fires it adds its weight to the vote sum: v gets +3, exactly as with three copies."},
    {"show": ["one"], "text": "A weight is more flexible than copies: it is <b>learned</b> (Step 2), it can grow far beyond the number of rules you could afford to copy, and it can drop to <b>0</b>, which switches the rule off."}
  ],
  "narrow": {"cols": 2, "rows": 5.5, "rowHeight": 70, "pos": {"copies": [0, 0, 2, 3.0], "c1": [0.5, 0.55], "c2": [0.5, 1.4], "c3": [0.5, 2.25], "one": [0.5, 3.55], "vote": [0.5, 4.55]}}
}
</script>
</div>

The papers differ in what a weight can be:

| | Real-valued weights (Phoulady et al., 2020) | Integer weights (Abeyrathna et al., 2021) |
|---|---|---|
| Start value | 1.0 | 1 |
| Rule fires and gets Recognize Feedback | multiply by $1 + \gamma$ | **+1** |
| Rule fires and gets Reject Feedback | divide by $1 + \gamma$ | **−1**, never below 0 |
| Extra hyperparameter | learning rate $\gamma$ | none |
| Reading a weight | "this rule counts 2.37 times" | "this rule counts as 3 copies" |

This lesson uses **integer weights**: they need no extra hyperparameter, they read as a number of copies, and a weight of 0 switches a rule off. (The TMU library's weighted classifier also uses integer weights but keeps them at 1 or more.)

!!! tip "What to notice"
    - A weight changes **how much** a rule's vote counts, never **when** it fires. The rule's literals decide when; the weight decides how much.
    - The Voting Margin still works on the vote sum, which is now weighted. A rule with weight 8 can reach a margin of $T = 8$ on its own.

---

## Step 2 — How a rule earns its weight

The integer-weight update is two lines (Abeyrathna et al., 2021). A weight only changes when its rule **fires** and is **picked** for feedback:

- **Recognize Feedback** (the rule fired on an example of the class it votes for, a true positive): $w \leftarrow w + 1$.
- **Reject Feedback** (the rule fired on an example of the other class, a false positive): $w \leftarrow w - 1$, but not below 0.
- **Erase Feedback** leaves the weight alone: the rule did not fire, so it cast no vote.

(In the literature: Type Ia feedback raises the weight and Type II lowers it.) Which weight does a rule end up with? Take one rule on its own and let it fire on a stream of examples. Call $q$ the rule's **precision**: the share of the examples it fires on that belong to the class it votes for.

- On an own-class example, the rule's $w$ votes are on the right side, so $v = w$ (clipped at $T$) and the rule is picked with probability $(T - w)/2T$. Then $w$ goes up.
- On an other-class example, those same $w$ votes are on the wrong side, so $v = -w$ and the rule is picked with probability $(T + w)/2T$. Then $w$ goes down.

On average the weight drifts by

$$
\frac{q\,(T - w) - (1 - q)\,(T + w)}{2T} \;=\; \frac{(2q - 1)\,T - w}{2T}
$$

per example. That is positive below $w = (2q - 1)T$ and negative above it, so the weight is pulled towards the **resting point**

$$
w^\ast \;=\; T\,(2q - 1).
$$

<div id="viz-weight" class="viz"><div class="viz-title">🎚️ Interactive: how a rule earns its weight</div></div>

!!! tip "What to notice"
    - **Precise rules get big weights.** With $q = 0.9$ and $T = 10$ the formula says 8, and the weight averages **7.72** over examples 1001–2000 of the run. A rule that is nearly always right ends up close to $T$ and can almost reach the margin alone. Abeyrathna et al. describe this as high-precision rules "operating more independently".
    - **Coin-flip rules are switched off.** At $q = 0.6$ the resting point is 2.0 (the run averages **2.25**) and the weight sits at 0 for **20%** of those examples. At $q = 0.4$, below one half, the resting point is negative, so the weight hugs the floor: it is 0 for **50%** of the examples and averages **0.80**. Rules like that only add votes when, now and then, a lucky streak lifts them off 0.
    - **$T$ sets the unit.** With $q = 0.9$ and $T = 20$ the weight averages **15.28** (formula: 16). Doubling the margin doubles every weight, so weights only mean something compared with $T$ and with each other.
    - This is a simplification: a real rule shares the vote sum with other rules, so $v$ is not just its own weight. The direction is the same: precise rules gain weight until the examples they cover reach the margin.

---

## Step 3 — One shared pool of rules

Weights also make a second idea possible. In the classification lesson, a machine for $m$ classes trains **$m$ separate teams**; a rule in the team for Car votes for or against Car and nothing else. But many patterns are useful for several classes at once. In a seven-segment display, *"the top-right segment is off"* is true for exactly two digits, 5 and 6: it is evidence **for** both, and **against** the other eight.

The **coalesced Tsetlin machine** (Glimsdal and Granmo, 2021) therefore keeps **one pool of rules** for all classes, and gives every rule $j$ a signed integer **weight for every class** $i$, $w_{ij}$. Together they form a weight matrix $W$ with one row per class and one column per rule. Each class reads the same rule outputs through its own row:

$$
v_i \;=\; \sum_j w_{ij} \, C_j(x), \qquad \text{prediction} = \text{the class with the largest } v_i .
$$

A positive $w_{ij}$ means rule $j$ votes **for** class $i$ with $w_{ij}$ votes; a negative one means it votes **against** class $i$ with $|w_{ij}|$ votes. Sharing rules this way is called **clause sharing**, and the machine is *coalesced* because the separate teams have merged into one.

<div class="viz diagram">
<script type="application/json">
{
  "title": "🧩 One pool of rules, a row of weights for every class",
  "cols": 3,
  "rows": 4,
  "rowHeight": 90,
  "nodes": [
    {"id": "in", "col": 1, "row": 0, "icon": "🧾", "label": "Input literals", "sub": "x<sub>k</sub> and ¬x<sub>k</sub>", "role": "data", "note": "Booleanize as before: each feature and its negation."},
    {"id": "pool", "col": 1, "row": 1, "icon": "🧩", "label": "Shared pool of rules", "sub": "rule j fires: c<sub>j</sub> = 1", "role": "model", "note": "Every rule is evaluated once per example, however many classes there are."},
    {"id": "v0", "col": 0, "row": 2, "label": "Digit 0", "sub": "v<sub>0</sub> = Σ<sub>j</sub> w<sub>0j</sub> c<sub>j</sub>", "role": "loss", "note": "Row 0 of the weight matrix: how much each rule's firing counts for or against digit 0."},
    {"id": "v1", "col": 1, "row": 2, "label": "Digit 1", "sub": "v<sub>1</sub> = Σ<sub>j</sub> w<sub>1j</sub> c<sub>j</sub>", "role": "loss", "note": "Row 1: the same rule outputs, read with digit 1's weights."},
    {"id": "v9", "col": 2, "row": 2, "label": "… Digit 9", "sub": "v<sub>9</sub> = Σ<sub>j</sub> w<sub>9j</sub> c<sub>j</sub>", "role": "loss", "note": "One row of weights per class."},
    {"id": "max", "col": 1, "row": 3, "icon": "🏆", "label": "Highest v<sub>i</sub> wins", "shape": "pill", "role": "opt"}
  ],
  "edges": [
    {"from": "in", "to": "pool"},
    {"from": "pool", "to": "v0"},
    {"from": "pool", "to": "v1"},
    {"from": "pool", "to": "v9"},
    {"from": "v0", "to": "max"},
    {"from": "v1", "to": "max"},
    {"from": "v9", "to": "max"}
  ],
  "steps": [
    {"show": ["in"], "text": "The example is Booleanized as usual: one literal per feature and one per negated feature."},
    {"show": ["in>pool", "pool"], "text": "Every rule in the <b>shared pool</b> checks its condition once. The result is a column of 0s and 1s, the rule outputs c<sub>j</sub>."},
    {"show": ["pool>v0", "pool>v1", "pool>v9", "v0", "v1", "v9"], "text": "Each class multiplies the same rule outputs by <b>its own row of weights</b>. A positive weight is a vote for that class, a negative weight a vote against it. One rule can vote for some classes and against others."},
    {"show": ["v0>max", "v1>max", "v9>max", "max"], "text": "The class with the highest vote sum is the prediction, exactly as in a multi-class Tsetlin machine."},
    {"show": ["pool", "v0", "v1", "v9"], "text": "The multi-class machine of the classification lesson is a special case: there, each rule has weight ±1 for <b>its own class</b> and weight 0 for every other class, so rules can never be reused."}
  ],
  "narrow": {"cols": 3, "rows": 4, "rowHeight": 80, "hideSub": true}
}
</script>
</div>

!!! tip "What to notice"
    - The pool is evaluated **once** per example, not once per class. With many classes that is a large saving in computation.
    - A shared rule needs a weight for every class. The weights are small integers, so they are cheap compared with a rule's literal memories, but the matrix grows with classes × rules.

---

## Step 4 — How a shared pool learns

Training a coalesced machine combines the multi-class training of the classification lesson with the weight updates of Step 2. Take a training example of class $y$:

1. **Its own class, as "yes".** Compute $v_y$, clip it to $[-T, T]$, and pick each rule with probability $(T - v_y)/2T$. A picked rule with $w_{yj} \ge 0$ is on the example's side: Recognize Feedback if it fired, Erase if not. A picked rule with $w_{yj} < 0$ that fired voted wrongly: Reject Feedback. Every picked rule that **fired** gets $w_{yj} \leftarrow w_{yj} + 1$.
2. **One other class $k$, chosen at random, as "no".** Pick each rule with probability $(T + v_k)/2T$. Now the rules with $w_{kj} < 0$ are on the example's side (Recognize or Erase), and those with $w_{kj} \ge 0$ that fired get Reject. Every picked rule that fired gets $w_{kj} \leftarrow w_{kj} - 1$.

The weights start at random values of $+1$ or $-1$, so at first every rule is randomly for or against each class. There is **no floor at 0**: a weight can pass through 0 and change sign, and the rule then switches sides for that class. Glimsdal and Granmo describe this as rules being **repurposed**.

<div class="viz diagram">
<script type="application/json">
{
  "title": "🔁 One learning step of a coalesced Tsetlin machine",
  "cols": 3,
  "rows": 2,
  "rowHeight": 125,
  "nodes": [
    {"id": "obs", "col": 0, "row": 0, "icon": "🧾", "label": "Example x of class y", "role": "data"},
    {"id": "eval", "col": 1, "row": 0, "icon": "🧩", "label": "Evaluate the shared pool", "sub": "empty rules count as True", "role": "model", "note": "While training, an empty rule fires, so fresh rules have something to learn from."},
    {"id": "yes", "col": 2, "row": 0, "icon": "✅", "label": "Class y as 'yes'", "sub": "probability (T − v<sub>y</sub>) / 2T", "role": "loss", "note": "v_y uses class y's row of weights, clipped to [−T, T]."},
    {"id": "fyes", "col": 2, "row": 1, "icon": "✏️", "label": "Feedback for y", "sub": "fired and picked: w<sub>yj</sub> + 1", "role": "opt", "note": "w ≥ 0: Recognize (fired) or Erase (did not fire). w < 0 and fired: Reject."},
    {"id": "no", "col": 1, "row": 1, "icon": "🎲", "label": "Random class k as 'no'", "sub": "probability (T + v<sub>k</sub>) / 2T", "role": "loss", "note": "One other class, chosen at random, with its own row of weights."},
    {"id": "fno", "col": 0, "row": 1, "icon": "✏️", "label": "Feedback for k", "sub": "fired and picked: w<sub>kj</sub> − 1", "role": "grad", "note": "w < 0: Recognize or Erase. w ≥ 0 and fired: Reject."}
  ],
  "edges": [
    {"from": "obs", "to": "eval"},
    {"from": "eval", "to": "yes"},
    {"from": "yes", "to": "fyes"},
    {"from": "fyes", "to": "no"},
    {"from": "no", "to": "fno"},
    {"from": "fno", "to": "obs", "label": "next"}
  ],
  "steps": [
    {"show": ["obs"], "text": "Take the next training example and its class y."},
    {"show": ["obs>eval", "eval"], "text": "Evaluate every rule of the pool once. During training an empty rule counts as True."},
    {"show": ["eval>yes", "yes"], "text": "Treat class y as <b>yes</b>: its weighted vote sum v<sub>y</sub> should reach +T. Each rule is picked with probability (T − v<sub>y</sub>)/2T."},
    {"show": ["yes>fyes", "fyes"], "text": "The sign of w<sub>yj</sub> says which side a rule is on. Rules on y's side get <b>Recognize</b> or <b>Erase</b>; rules against y that fired get <b>Reject</b>. Every picked rule that fired moves its weight for y up by 1."},
    {"show": ["fyes>no", "no"], "text": "Then pick <b>one other class k</b> at random and treat it as <b>no</b>: v<sub>k</sub> should reach −T. Each rule is picked with probability (T + v<sub>k</sub>)/2T."},
    {"show": ["no>fno", "fno"], "text": "Now the rules with negative w<sub>kj</sub> are on the example's side. Every picked rule that fired moves its weight for k down by 1, and may cross 0 and change sides."},
    {"show": ["fno>obs", "obs"], "text": "On to the next example. One rule's memory can be pulled by two classes in one step; the Voting Margin of each class keeps the pulling in balance."}
  ],
  "narrow": {"cols": 2, "rows": 3, "rowHeight": 120, "pos": {"obs": [0, 0], "eval": [1, 0], "yes": [1, 1], "fyes": [1, 2], "no": [0, 2], "fno": [0, 1]}}
}
</script>
</div>

!!! note "Versions of the algorithm"
    The coalesced paper writes the update for **every** output at once, for problems where an example can have several outputs (multi-output, or multi-label), and scales Reject Feedback by $1/(m - 1)$ for one-hot classes. The TMU library's `TMCoalescedClassifier`, and this lesson, instead update the true class and **one randomly chosen other class**, like the multi-class machine of the classification lesson. Also note that $w = 0$ counts as "for" ($w \ge 0$), as in the paper and in TMU.

!!! tip "What to notice"
    - Nothing in the rules says which class they belong to. A rule ends up "about" digit 5 only because its weight for 5 grew.
    - The feedback a rule's memory gets now depends on the **sign of its weight** for the class being trained. That is how one rule can be refined by digit 5's examples and rejected by digit 3's.

---

## Step 5 — Inside a trained pool

Time for an example. The data is a toy task we made for this lesson: **noisy seven-segment digits**. Each digit 0–9 is drawn with the seven segments of a calculator display, named *a* (top), *b* (top right), *c* (bottom right), *d* (bottom), *e* (bottom left), *f* (top left) and *g* (middle), so there are 7 features and 14 literals. Each segment is flipped with probability 5%, so about 70% of the examples are noise-free (0.95⁷ = 0.698) and the rest have one or more wrong segments. There are 500 training and 1000 test examples.

Even a perfect model makes mistakes here: a 6 whose segment *e* flipped off looks exactly like a clean 5, and an 8 that lost segment *e* looks exactly like a clean 9. The best you can do is to pick the digit whose clean pattern is nearest (fewest differing segments), which scores **87.3%** on our test set.

The widget below shows a coalesced machine with a pool of **20 rules** trained on this data ($T = 128$, $s = 3$, 40 epochs, seed 1). Each row is one rule; the ten cells next to it are its weights for digits 0–9.

<div id="viz-pool" class="viz"><div class="viz-title">🧩 Interactive: one shared pool of rules, a weight for every digit</div></div>

!!! tip "What to notice"
    - **Every rule is shared.** All 20 rules have a positive weight for at least two digits. R1, *"e ∧ g"* (bottom-left and middle on), has weights +46 for 2, +29 for 6 and +15 for 8, and negative weights for everything else. Those are exactly the three digits whose clean pattern has both segments.
    - **Rules found the digits' defining gaps.** Four rules contain *¬b* (top-right segment off): R11, R19 and R20 are just *"¬b"*, and R18 is *"f ∧ ¬b"*. All four vote for 5 and 6, the only digits without segment *b* (R19 also gives 3 a small +4). R4 and R14 are both *"¬a"*, and vote for 1 and 4, the only digits without a top segment. Duplicates still happen in a pool, but the weights now tell you how much each copy counts for which digit.
    - **Click "5".** The clean 5 fires five rules (R11, R13, R18, R19, R20). Digit 5's column adds up to **+154**, and the runner-up is 6 with **+35**: the *¬b* rules push both 5 and 6 up, and R13, *"d ∧ f ∧ ¬e"*, separates them because a 6 has segment *e*.
    - **Edit the input.** Switch segment *e* on in the 5 and it becomes a clean 6: the *¬e* rule stops firing, R1 *"e ∧ g"* and R10 *"a ∧ e ∧ f"* start, and the prediction flips. Then try "next test example" to step through noisy test digits and see which ones the pool gets wrong.
    - The weights are large (up to 89 in size) because the margin is large, $T = 128$. As Step 2 showed, weights only mean something compared with $T$ and with each other.

---

## Step 6 — The same budget, three machines

Now the comparison the papers make: give three machines **the same number of rules** and see what they do with them.

- **Plain:** one team per digit, half the rules for and half against, every weight fixed at 1 (the classification lesson's machine).
- **Weighted:** the same teams, with integer weights (Step 2).
- **Coalesced:** one pool shared by all ten digits, with signed weights (Steps 3–4).

With a budget of 20 rules, the plain and weighted machines get **one rule for and one against** per digit; the coalesced machine gets a pool of 20. With 60 rules: three for and three against per digit, or a pool of 60. All use $s = 3$ and memory depth 100 and train for 40 epochs.

<div id="viz-budget" class="viz"><div class="viz-title">⚖️ Interactive: the same rule budget, three machines</div></div>

!!! tip "What to notice"
    - **With 20 rules (seed 1)**, the plain machine is stuck at **73.6%** from early on. The coalesced machine ends at **86.9%** (mean of epochs 31–40: **86.7%**), within half a point of the 87.3% ceiling. The weighted machine is in between but restless: it ends this run at **75.1%**, while its mean over epochs 31–40 is **81.4%**, with single epochs anywhere from 71.5% to 85.1%.
    - **Why the plain machine is stuck.** Show the plain rules: each digit's single "for" rule memorized the digit's complete clean pattern, like *"a ∧ b ∧ c ∧ d ∧ e ∧ f ∧ ¬g"* for 0. Such a rule only fires on noise-free examples, about 71% of the test set. With one vote per rule there is no way to say "most of these segments".
    - **Weighted rules are short.** Show the weighted rules: digit 6's rule *"e ∧ ¬b"* (bottom-left on, top-right off) with weight 13 is exactly what separates a 6 from every other digit. Digit 0 has *"e ∧ f ∧ ¬g"* with weight 16. The weighted machine memorized **39** literals in total, the plain one **108**.
    - **Switch to 60 rules.** Now the plain machine has enough rules to cover noisy variants, and all three end within a few points of each other (seed 1: plain 82.9%, weighted 87.4%, coalesced 85.2% in the last epoch). Weights and sharing pay off most when rules are **scarce**.

One run is not enough, so we repeated it for machine seeds 1–5 on the same data. To keep the comparison fair, each machine got its own Voting Margin, chosen from $T \in \{2, 4, 8, 16, 32, 64, 128\}$ on a separate validation set of 1000 examples. Test accuracy, averaged over epochs 31–40 and then over the 5 seeds:

| Machine | 20 rules | 60 rules |
|---|---|---|
| Plain ($T$ = 32 / 2) | 74.4% ± 1.1 | 83.2% ± 1.6 |
| Weighted ($T$ = 8 / 64) | 81.1% ± 1.1 | 86.5% ± 0.3 |
| Coalesced ($T$ = 128 / 128) | **85.8% ± 0.7** | **87.2% ± 0.3** |
| Nearest clean pattern (the best possible on this data) | 87.3% | 87.3% |

The pattern matches what the papers report on real benchmarks. Glimsdal and Granmo (2021) report, for convolutional machines on MNIST with 50 rules per class, 97.86% for the weighted machine against 98.33% for the coalesced one, and similar accuracy for both (99.33% and 99.29%) at 8000 rules per class. Their largest gap is on Kuzushiji-MNIST with 50 rules per class: 71.99% weighted against 89.66% coalesced (their abstract attributes these two numbers to Fashion-MNIST; we quote their Table 4).

!!! warning "A toy, not a benchmark"
    The seven-segment task is tiny and was made to fit this lesson: its classes really do share sub-patterns, which is exactly where sharing should help. The ranking of the three machines can be different on other data, and the best Voting Margin was at the edge of our grid for two of the machines, so they might do slightly better with an even larger $T$.

---

## Step 7 — Reading weighted and shared rules

Weights change how you read a trained machine, mostly for the better.

| Seed 1, 20 rules | literals memorized | what you read |
|---|---|---|
| Plain | 108 | 20 long rules, mostly complete digit pictures |
| Weighted | 39 | 20 short rules, each with a weight (some 0) |
| Coalesced | 40 | 20 short rules, each with a row of 10 signed weights |

Some habits for reading them:

- **Read an integer weight as copies.** *"e ∧ ¬b, weight 13"* means the same as thirteen copies of that rule in a plain machine. Abeyrathna et al. (2021) chose integer weights for exactly this reason.
- **Compare weights with $T$, and within one machine.** By Step 2, a weight close to $T$ marks a precise rule; a weight near 0 marks a rule that is right about half the time it fires. A weight from a machine with $T = 8$ can't be compared with one from $T = 128$.
- **Ignore rules with weight 0.** They cast no vote. In their bankruptcy example, Abeyrathna et al. (2021) report that integer weights switched one rule off, leaving the single readable rule *"Bankruptcy if Negative Competitiveness"*.
- **Read a shared rule column by column.** In a coalesced machine, ask "which rules vote for digit 5, and with how much?" by reading digit 5's column of weights, and "what does rule R1 mean?" by reading its row: *e ∧ g* is evidence for 2, 6 and 8.
- **Watch for empty rules.** In the weighted run, several "against" rules ended empty, with weights of 6 or 7. During training an empty rule fires on every example, so it acts as a fixed offset; when classifying it stays silent, as in the earlier lessons. It is a reminder that the weights were learned with the training-time behaviour.

!!! note "Interpretable is still not the same as correct"
    The weighted machine above reads beautifully and still had epochs at 71.5%. The weights make the model **shorter** to audit, not automatically right. Check accuracy over several runs, as in Step 6.

---

## Putting it all together

<div class="viz diagram">
<script type="application/json">
{
  "title": "🧭 Three ways to spend a budget of 20 rules",
  "cols": 3,
  "rows": 2,
  "rowHeight": 115,
  "nodes": [
    {"id": "plain", "col": 0, "row": 0, "icon": "📋", "label": "Plain teams", "sub": "one team per class, every vote = 1", "role": "model", "note": "The classification lesson's machine: duplicates are the only way to weigh a rule."},
    {"id": "weighted", "col": 1, "row": 0, "icon": "⚖️", "label": "Weighted teams", "sub": "+1 on Recognize, −1 on Reject", "role": "model", "note": "Integer-weighted Tsetlin machine (Abeyrathna et al., 2021)."},
    {"id": "coal", "col": 2, "row": 0, "icon": "🧩", "label": "Shared pool", "sub": "signed weight per rule and class", "role": "model", "note": "Coalesced Tsetlin machine with clause sharing (Glimsdal and Granmo, 2021)."},
    {"id": "r0", "col": 0, "row": 1, "label": "74.4%", "sub": "rules = full digit pictures", "role": "loss", "note": "Mean test accuracy over 5 seeds, epochs 31–40."},
    {"id": "r1", "col": 1, "row": 1, "label": "81.1%", "sub": "short rules, restless", "role": "loss", "note": "Mean test accuracy over 5 seeds, epochs 31–40."},
    {"id": "r2", "col": 2, "row": 1, "label": "85.8%", "sub": "best possible: 87.3%", "role": "opt", "note": "Mean test accuracy over 5 seeds, epochs 31–40."}
  ],
  "edges": [
    {"from": "plain", "to": "weighted"},
    {"from": "weighted", "to": "coal"},
    {"from": "plain", "to": "r0"},
    {"from": "weighted", "to": "r1"},
    {"from": "coal", "to": "r2"}
  ],
  "steps": [
    {"show": ["plain", "plain>r0", "r0"], "text": "<b>Plain:</b> with one rule for and one against per digit, each rule can only say one thing with one vote. The 'for' rules memorize complete clean digits and miss the noisy ones."},
    {"show": ["plain>weighted", "weighted", "weighted>r1", "r1"], "text": "<b>Add weights:</b> a rule's weight climbs towards T(2q − 1), so precise rules count more. Short rules with big weights replace long exact ones."},
    {"show": ["weighted>coal", "coal", "coal>r2", "r2"], "text": "<b>Share rules:</b> one pool for all ten digits, a signed weight for every pair. A rule like 'top-right segment off' serves digits 5 and 6 at once and votes against the rest."},
    {"show": ["r0", "r1", "r2"], "text": "With 60 rules the three come within a few points of each other: weights and sharing matter most when rules are <b>scarce</b>."}
  ],
  "narrow": {"cols": 2, "rows": 3, "rowHeight": 110, "pos": {"plain": [0, 0], "r0": [1, 0], "weighted": [0, 1], "r1": [1, 1], "coal": [0, 2], "r2": [1, 2]}}
}
</script>
</div>

Here is a complete coalesced Tsetlin machine in numpy, about 70 lines including the data. It follows Step 4 line by line, vectorised over the rules and literals. It uses numpy's random numbers and its own data, so its numbers differ a little from the widgets'.

```python
import numpy as np

# ---- 1. Data: noisy 7-segment digits (segments a-g; 5% of segments flipped)
SEGMENTS = {0: "abcdef", 1: "bc", 2: "abdeg", 3: "abcdg", 4: "bcfg", 5: "acdfg", 6: "acdefg", 7: "abc", 8: "abcdefg", 9: "abcdfg"}
PATTERN = np.array([[s in SEGMENTS[d] for s in "abcdefg"] for d in range(10)], dtype=np.int8)

def make_data(n, rng, noise=0.05):
    y = rng.integers(0, 10, n)
    X = PATTERN[y] ^ (rng.random((n, 7)) < noise)
    return X.astype(np.int8), y

rng = np.random.default_rng(42)
X_train, y_train = make_data(500, rng)
X_test, y_test = make_data(1000, rng)


class CoalescedTsetlinMachine:
    """One shared pool of rules. memory[j, k] = position (1 … depth) of literal k in rule j.
    W[i, j] = signed integer weight of rule j for class i (> 0: votes for i, < 0: votes against i)."""

    def __init__(self, n_classes, n_features, n_rules=20, T=128, s=3.0, depth=100, seed=1):
        self.T, self.s, self.depth, self.half = T, s, depth, depth // 2
        self.rng = np.random.default_rng(seed)
        self.memory = np.full((n_rules, 2 * n_features), depth // 2)          # top of Forgotten
        self.W = self.rng.choice([-1, 1], size=(n_classes, n_rules))          # random sides to start

    def fires(self, lits, training=False):
        memorized = self.memory > self.half
        ok = ~np.any(memorized & (lits == 0), axis=1)
        return ok if training else ok & memorized.any(axis=1)                 # empty rules stay silent

    def predict(self, x):
        return int(np.argmax(self.W @ self.fires(np.concatenate([x, 1 - x]))))   # class sums v_i = W_i · c

    def update_class(self, lits, c, i, is_class):
        v = np.clip(self.W[i] @ c, -self.T, self.T)
        p = (self.T - v) / (2 * self.T) if is_class else (self.T + v) / (2 * self.T)
        picked = self.rng.random(len(c)) <= p
        own = (self.W[i] >= 0) == is_class                                    # rule is on the example's side
        u = self.rng.random(self.memory.shape)
        mem = self.memory
        rec = (picked & own & c)[:, None]                                     # Recognize
        mem += rec & (lits == 1) & (u <= (self.s - 1) / self.s) & (mem < self.depth)
        mem -= rec & (lits == 0) & (u <= 1 / self.s) & (mem > 1)
        era = (picked & own & ~c)[:, None]                                    # Erase
        mem -= era & (u <= 1 / self.s) & (mem > 1)
        rej = (picked & ~own & c)[:, None]                                    # Reject
        mem += rej & (lits == 0) & (mem <= self.half)
        self.W[i] += np.where(picked & c, 1 if is_class else -1, 0)          # weights: toward this example's side

    def fit(self, X, y, epochs):
        n_classes = self.W.shape[0]
        for _ in range(epochs):
            for e in self.rng.permutation(len(X)):
                lits = np.concatenate([X[e], 1 - X[e]])
                other = (y[e] + self.rng.integers(1, n_classes)) % n_classes   # one random other class
                for i, is_class in ((y[e], True), (other, False)):
                    self.update_class(lits, self.fires(lits, training=True), i, is_class)
        return self

    def rules(self):
        names = list("abcdefg") + ["NOT " + s for s in "abcdefg"]
        return [" AND ".join(n for n, m in zip(names, r) if m > self.half) or "(empty)" for r in self.memory]


tm = CoalescedTsetlinMachine(n_classes=10, n_features=7).fit(X_train, y_train, epochs=40)
acc = np.mean([tm.predict(x) == t for x, t in zip(X_test, y_test)])
nearest = np.mean(np.argmin((X_test[:, None, :] != PATTERN[None]).sum(2), axis=1) == y_test)
print(f"test accuracy {acc:.3f} · nearest clean pattern {nearest:.3f}")
for j, rule in enumerate(tm.rules()[:5]):
    print(f"rule {j + 1}: IF {rule:<16} weights per digit 0-9: {tm.W[:, j].tolist()}")
```

```text
test accuracy 0.835 · nearest clean pattern 0.874
rule 1: IF c AND e AND f    weights per digit 0-9: [29, -48, -50, -52, -23, -81, 20, -66, 52, -15]
rule 2: IF a AND b AND NOT e weights per digit 0-9: [-8, -81, -62, 35, -70, -39, -32, 48, -63, 67]
rule 3: IF NOT a            weights per digit 0-9: [-18, 41, -13, -23, 34, -21, -14, -49, -23, -18]
rule 4: IF NOT a            weights per digit 0-9: [-23, 41, -12, -29, 34, -10, -10, -53, -23, -26]
rule 5: IF b AND c AND g    weights per digit 0-9: [-43, -31, -82, 27, -64, -11, -30, -39, 56, 17]
```

It takes about 7 seconds. Rule 1, *"c ∧ e ∧ f"*, votes for 0, 6 and 8, the three digits that have all three segments; rule 2, *"a ∧ b ∧ ¬e"*, votes for 3, 7 and 9. Turning it into the **weighted** machine of Step 2 takes three changes: keep a separate pool per class, fix each rule's side by its position instead of the sign of its weight, and replace the last line of `update_class` with $+1$ on Recognize and $-1$ (not below 0) on Reject.

| Piece | What it does | Book term (research term) |
|---|---|---|
| Weight $w_j$ | how many votes a firing rule casts | weight (clause weight) |
| Weighted vote sum $v$ | sum of the weights of the firing rules, for minus against | vote sum ($\sum_j w^+_j C^+_j - \sum_j w^-_j C^-_j$) |
| Weight learning | +1 on Recognize, −1 on Reject (not below 0); rests near $T(2q - 1)$ | (integer-weighted TM; SSL automaton with step 1) |
| Shared pool | one set of rules for all classes | (clause sharing; coalesced TM) |
| Weight matrix $W$ | signed weight $w_{ij}$ for every class and rule; may change sign | (clause-output weight matrix) |
| Class sum $v_i$ | $\sum_j w_{ij} C_j$; the highest wins | (class sum, $v = Wc$) |

---

## Common misconceptions

!!! warning "\"A rule's weight is the probability that it is right.\""
    No. In Step 2 the weight rests near $T(2q - 1)$, so it depends on the Voting Margin as much as on the rule. A rule with precision 0.9 rested near 7.7 with $T = 10$ and near 15.3 with $T = 20$. Weights rank rules within one machine; they are not probabilities.

!!! warning "\"Weights make the model harder to read.\""
    Integer weights are just copy counts: *"weight 13"* reads as thirteen copies of the rule. And with the same budget, the weighted machine in Step 6 memorized 39 literals against the plain machine's 108, because short rules with big weights replaced long exact ones.

!!! warning "\"A weight of 0 deletes the rule.\""
    It silences the rule's vote, but the rule still fires, still gets feedback and can earn its weight back. In the Step 2 run with $q = 0.6$, the weight sat at 0 for 20% of examples 1001–2000 and kept coming back.

!!! warning "\"In a shared pool, every rule votes for every class.\""
    Every rule has a weight for every class, but most of them are negative. In the trained pool of Step 5, each rule voted for between two and three digits and against all the others. To see what a rule is for, read its row; to see what votes for a class, read the class's column.

!!! warning "\"Clause sharing is always better.\""
    In our runs it helped most with 20 rules (85.8% against 74.4% for the plain machine) and much less with 60 (87.2% against 83.2%, with the weighted machine at 86.5%). Glimsdal and Granmo report the same: large gains with few rules per class, similar accuracy with thousands.

---

## Check yourself

??? question "1. A weighted machine has the rule \"Deg-malig 3\" voting for Recurrence with weight 3, and \"Inv-nodes 0-2\" voting against with weight 2. A patient matches both. What is v, and what is the feedback probability with T = 4 if the patient did recur?"
    $v = 3 - 2 = +1$, so the machine predicts Recurrence. For a patient with recurrence, $v = +1$ is already from the patient's own side, so each rule is picked with probability $(4 - 1)/8 = $ **0.375**.

??? question "2. A rule fires on 400 examples of its own class and 100 of the other class. Where does its integer weight settle with T = 10?"
    Its precision is $q = 400/500 = 0.8$. The resting point is $T(2q - 1) = 10 \times 0.6 = $ **6**: high enough to matter, but well short of reaching the margin alone.

??? question "3. Why does Erase Feedback leave the weight unchanged?"
    Erase Feedback goes to a rule that **did not fire** on an example of its own class. The weight measures how much the rule's vote is worth, and here it cast no vote, right or wrong. Only firings (true positives and false positives) carry information about the vote's value.

??? question "4. In a coalesced machine, rule j has weight w = −1 for digit 3. It fires on a training example of digit 3 and is picked. What happens to the rule and to the weight?"
    With $w_{3j} < 0$ the rule is *against* 3, and it fired on a 3, so it voted wrongly: it gets **Reject Feedback**, and its weight goes up by 1 to **0**. Since $w \ge 0$ counts as "for", the next time it fires on a 3 and is picked it gets **Recognize Feedback** and its weight rises to 1. The rule has been repurposed to vote for digit 3.

??? question "5. With one rule for and one against per digit, the plain machine scored about 74% while the best possible is 87.3%. Why couldn't it do better, and how do weights fix it?"
    Each "for" rule memorized its digit's complete clean pattern, which fires only on noise-free examples (about 70% of them, $0.95^7 = 0.698$), and every rule has exactly one vote, so it can't express "most of these segments". With weights, short rules that capture one distinguishing feature (like *"e ∧ ¬b"* for 6) earn large weights, and a few of them together tolerate a flipped segment.

---

## Further reading

- Adrian Phoulady, Ole-Christoffer Granmo, Saeed Rahimi Gorji and Hady Ahmady Phoulady, [*The Weighted Tsetlin Machine: Compressed Representations with Weighted Clauses*](https://arxiv.org/abs/1911.12607) (2020; Ninth International Workshop on Statistical Relational AI). Real-valued clause weights learned with a rate $\gamma$; the MNIST, IMDb and Connect-4 compression results quoted in this lesson.
- K. Darshana Abeyrathna, Ole-Christoffer Granmo and Morten Goodwin, [*Extending the Tsetlin Machine With Integer-Weighted Clauses for Increased Interpretability*](https://doi.org/10.1109/ACCESS.2021.3049569), IEEE Access 9, 8233–8248 (2021); preprint [arXiv:2005.05131](https://arxiv.org/abs/2005.05131). Integer weights (+1 / −1) learned by a stochastic-searching-on-the-line automaton; the weight update used in Step 2, and results on breast-cancer, liver-disorder, heart-disease and bankruptcy data.
- Sondre Glimsdal and Ole-Christoffer Granmo, [*Coalesced Multi-Output Tsetlin Machines with Clause Sharing*](https://arxiv.org/abs/2108.07594) (arXiv preprint, 2021). The shared pool with a signed weight matrix used in Steps 3–6, with results on MNIST, Fashion-MNIST, Kuzushiji-MNIST and imbalanced IMDb and CIFAR-10.
- CAIR, [Tsetlin Machine Unified (TMU)](https://github.com/cair/tmu): the open-source library with `TMClassifier(weighted_clauses=True)` and `TMCoalescedClassifier`, which we read to check the details of the update rules.
- Ole-Christoffer Granmo, [*An Introduction to Tsetlin Machines*](https://tsetlinmachine.org/) (free online). Chapter 5, *Pattern Weighing and Sharing*, is announced; the vocabulary of this lesson follows the published chapters.

**Next lessons:** [Autoencoders & embeddings](../tsetlin-machine-autoencoder/index.md): shared rules that predict one part of the input from the rest.

<script src="viz/weighted-tsetlin-machine.js"></script>
