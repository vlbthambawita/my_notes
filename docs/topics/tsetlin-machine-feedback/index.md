# Learning Rules: Recognize, Erase and Reject Feedback

!!! abstract "TL;DR"
    A Tsetlin-machine rule learns by nudging each literal's memory position up (memorize) or down (forget), one example at a time. When the example belongs to the rule's own class and the rule **matches**, **Recognize Feedback** memorizes the True literals and forgets the False ones. When the rule **doesn't match** an example of its own class, **Erase Feedback** forgets everything a little. Together these two find **frequent patterns** ("most cars have four wheels"), and the **Forget Value** decides how frequent a literal must be to stay. Frequent is not the same as useful for telling classes apart, so a third type, **Reject Feedback**, fires when the rule wrongly matches an example of another class: it memorizes a literal that is False for that example, so the rule stops matching it.

**Level:** beginner · **Prerequisites:** [What is a Tsetlin machine?](../tsetlin-machine-intro/index.md) (literals, AND rules) and [The Tsetlin automaton](../tsetlin-automaton/index.md) (a literal's graded memory).

---

## Why should I care?

Most machine-learning models learn by gradient descent: compute an error, take derivatives, shift millions of real-valued weights. A Tsetlin machine does something much simpler. Each rule keeps a small counter for every literal, and three plain if-then feedback rules move those counters up or down by one. There are no derivatives and no floating-point weights in the model. The result is still a readable rule, such as *IF Deg-malig 3 THEN Recurrence*.

This lesson is about **how one rule learns**. It matters for two reasons:

- **It is the whole learning algorithm in miniature.** A full Tsetlin machine (next lesson) is many rules learning side by side with exactly these three feedback types. Only *how often* each rule gets feedback is added.
- **It separates two goals you meet everywhere in medicine and science.** A rule can describe what is *typical* of a group ("almost no recurrence patients are under 40") or what *distinguishes* it from other groups ("tumour malignancy grade 3 is more than twice as common in recurrence patients"). The first is easy to learn and often useless for diagnosis. You will see exactly which feedback gives you which.

We follow Chapter 1, §1.3–1.4 and Chapter 2, §2.2 of Ole-Christoffer Granmo's book *An Introduction to Tsetlin Machines*, with its two running examples: the six vehicles and the breast-cancer recurrence data.

---

## Step 1 — A rule is its memory

From the [previous lesson](../tsetlin-automaton/index.md): each rule has a **memory** with one slot per literal. With 5 vehicle features there are 10 literals ($x_k$ and $\lnot x_k$ for each feature). Each literal sits at a position from 1 to 10:

- **1 to 5 = Forgotten.** 1 is *Maximally Forgotten*, 5 is "almost memorized". Forgotten literals are not part of the rule.
- **6 to 10 = Memorized.** 6 is lightly held, 10 is *Maximally Memorized*. Memorized literals **are** the rule.

The rule's condition is the AND of its memorized literals. A rule $C_j$ for class Car with Four Wheels and Transports People memorized reads *IF Four Wheels AND Transports People THEN Car*. Two details matter for everything below:

- **A new rule starts with every literal at position 5.** All forgotten, but each one a single step away from being memorized.
- **An empty condition is True.** A rule with nothing memorized makes no demands, so during learning it matches every example.

!!! note "Book terms and paper terms"
    The book's terms are used throughout. In research papers, the memory of one literal is a **Tsetlin automaton** with actions *include* / *exclude*, and the three feedback types below are called **Type Ia** (Recognize), **Type Ib** (Erase) and **Type II** (Reject) feedback. Recognize and Erase together are **Type I** feedback.

---

## Step 2 — Which feedback, when?

Every training example comes with its class. For one rule, only two questions decide what happens:

1. Is the example of the **rule's own class** (a car, for a Car rule)?
2. Does the rule **match** it (is its condition True)?

<div class="viz diagram">
<script type="application/json">
{
  "title": "🧭 Choosing the feedback for one rule",
  "cols": 3,
  "rows": 4,
  "rowHeight": 95,
  "nodes": [
    {"id": "ex", "col": 0, "row": 0, "icon": "🚗", "label": "Example with its class", "shape": "pill", "role": "data", "note": "The rule sees the literals of one example and its class."},
    {"id": "own", "col": 0, "row": 1, "label": "Rule's own class?", "shape": "decision", "note": "A Car rule asks: is this a car?"},
    {"id": "fit1", "col": 0, "row": 2, "label": "Rule True?", "shape": "decision", "note": "Are all memorized literals True for this example?"},
    {"id": "rec", "col": 2, "row": 2, "icon": "✅", "label": "Recognize", "sub": "True ↑, False ↓", "role": "opt", "note": "Memorize the True literals (Memorize Value), forget the False ones (Forget Value)."},
    {"id": "fit2", "col": 2, "row": 1, "label": "Rule True?", "shape": "decision", "note": "Does the rule wrongly match an example of another class?"},
    {"id": "era", "col": 0, "row": 3, "icon": "🧽", "label": "Erase", "sub": "all ↓", "role": "grad", "note": "Forget every literal, each with probability = Forget Value."},
    {"id": "rej", "col": 2, "row": 0, "icon": "⛔", "label": "Reject", "sub": "forgotten False ↑", "role": "model", "note": "Memorize every Forgotten literal that is False for this example, always (no randomness)."}
  ],
  "edges": [
    {"from": "ex", "to": "own"},
    {"from": "own", "to": "fit1", "label": "yes"},
    {"from": "fit1", "to": "rec", "label": "yes"},
    {"from": "fit1", "to": "era", "label": "no"},
    {"from": "own", "to": "fit2", "label": "no"},
    {"from": "fit2", "to": "rej", "label": "yes"}
  ],
  "steps": [
    {"show": ["ex"], "text": "An example arrives with its class label, e.g. vehicle #1, a Car."},
    {"show": ["ex>own", "own"], "text": "First question: is it of the rule's <b>own class</b>?"},
    {"show": ["own>fit1", "fit1", "fit1>rec", "rec"], "text": "Own class and the rule matches → <b>Recognize Feedback</b>: make the rule look more like this example."},
    {"show": ["fit1>era", "era"], "text": "Own class but the rule does not match → <b>Erase Feedback</b>: forget a little of everything, so the rule becomes more general."},
    {"show": ["own>fit2", "fit2", "fit2>rej", "rej"], "text": "Another class and the rule matches → a mistake → <b>Reject Feedback</b>: memorize a literal that is False here, so the rule stops matching."},
    {"show": ["fit2"], "text": "Another class and the rule does not match → nothing to fix: <b>no feedback</b>."}
  ],
  "narrow": {"cols": 2.4, "rows": 4, "rowHeight": 100, "hideSub": true, "pos": {"ex": [0, 0], "own": [0, 1], "fit1": [0, 2], "rec": [1.4, 2], "era": [0, 3], "fit2": [1.4, 1], "rej": [1.4, 0]}}
}
</script>
</div>

The same thing as a table:

| | rule **matches** (True) | rule **doesn't match** (False) |
|---|---|---|
| example of the **rule's own class** | **Recognize Feedback** | **Erase Feedback** |
| example of **another class** | **Reject Feedback** | no feedback |

!!! tip "What to notice"
    - Two of the three feedback types fire on the rule's **own** class. Only Reject looks at the other classes.
    - A rule that matches nothing at all only ever gets Erase Feedback (on its own class), which pulls it back towards matching.

---

## Step 3 — Recognize Feedback: become more like this example

**When:** the example is of the rule's own class, and the rule matches it.

**What:** for every literal,

- if the literal is **True** for the example, **memorize** it: move it one position up (unless it is already at 10), with probability equal to the **Memorize Value**;
- if the literal is **False** for the example, **forget** it: move it one position down (unless it is already at 1), with probability equal to the **Forget Value**.

The name says what it does: the rule is pulled towards *recognizing* this kind of example. Literals that keep showing up in its class climb into memory, and literals that keep being absent sink.

**Why random?** Every single step is a coin flip. Before each move you draw a random number between 0 and 1, and you only make the move if the number is at most the Memorize Value (for an increment) or the Forget Value (for a decrement). One odd example then cannot push the rule far, and different rules trained on the same data end up different, which helps when many rules work together.

**The two values set the speed.** A small Memorize Value (say 0.1) means memorizing is slow; a large Forget Value (0.9) means forgetting is fast. The book usually keeps

$$
\text{Memorize Value} + \text{Forget Value} = 1,
$$

so a single knob is enough. In the research papers the knob is the **specificity** $s \ge 1$: Memorize Value $= \frac{s-1}{s}$ and Forget Value $= \frac{1}{s}$. A Forget Value of 0.8 is $s = 1.25$; a Forget Value of 0.1 is $s = 10$.

!!! note "A Memorize Value of 1"
    The book also allows the Memorize Value to be fixed at 1.0, so True literals always step up, with only the Forget Value ($\frac1s$) left as the knob. Chapter 2 of the book (and Step 8 below) uses that version.

---

## Step 4 — Erase Feedback: loosen a rule that is too picky

**When:** the example is of the rule's own class, but the rule does **not** match it.

**What:** **forget every literal**, each one moving down a position with probability equal to the Forget Value. True or False doesn't matter.

A rule that fails on an example of its own class is too demanding: something in its condition is not typical of the class. Erase Feedback can't tell which literal is to blame, so it weakens all of them a little. The memorized literals that are really frequent get pushed back up by the next Recognize Feedback; the one that caused the mismatch does not, and drops out.

!!! tip "What to notice"
    Without Erase Feedback, a rule that memorized too much early on is stuck: it stops matching most of its class, and Recognize never fires on those examples again. In our Python check (1000 runs of 200 steps, Forget Value 0.9), a Car rule trained **without** Erase covered on average 81% of the cars, and 47% of runs got stuck with a colour in the rule (such as *… AND Yellow AND NOT Blue*). **With** Erase, the rule covered all three cars in 999 of the 1000 runs.

---

## Step 5 — Watch it happen: the book's worked example

The book walks through a Car rule meeting the first four vehicles of its table, with Memorize Value 0.1 and Forget Value 0.9. Because the updates are random, the book *chooses* which coin flips succeed. The widget replays exactly those choices. Click **Next**.

| # | Four Wheels | Transports People | Wings | Yellow | Blue | Class |
|---|---|---|---|---|---|---|
| 1 | ● | ● | · | · | ● | Car |
| 2 | ● | ● | · | ● | · | Car |
| 3 | ● | ● | · | ● | · | Car |
| 4 | ● | ● | ● | · | ● | Plane |
| 5 | ● | · | ● | ● | · | Plane |
| 6 | · | ● | ● | · | ● | Plane |

*The vehicle table from Granmo, Ch. 1, Table 1.1 (● True, · False).*

<div id="viz-walkthrough" class="viz"><div class="viz-title">🎬 Interactive: one Car rule meets vehicles #1–#4</div></div>

!!! tip "What to notice"
    - **Step 1.** The empty rule matches anything, so the very first car gives Recognize Feedback. After it the rule is *Four Wheels AND Transports People AND Blue*: it has copied the first car, colour included.
    - **Step 2.** A yellow car fails the "Blue" demand → Erase. Blue drops to 5 and Transports People drops to 5 with it. Only Four Wheels survives because its coin flip happened to skip.
    - **Step 3.** Now the rule is just *Four Wheels*, which matches the next yellow car → Recognize. Four Wheels climbs to 7, Transports People comes back to 6, and the colour literals sink lower: Yellow is True in two of the three cars and Blue in one, so neither is frequent enough to stick at these settings.
    - **Step 4.** The resulting rule *Four Wheels AND Transports People* also fits plane #4. Reject Feedback (Step 7) moves ¬Wings from 5 to 6, giving *Four Wheels AND Transports People AND ¬Wings*, which no longer matches the plane.

---

## Step 6 — Recognize + Erase find frequent patterns

Why do these two feedback types end up with "what is typical"? Look at one literal while the rule matches its class. Say the literal is True in a fraction $f$ of those examples. Each time, it moves:

- **up** with probability $f \times \text{Memorize Value}$ (it is True, and the coin flip succeeds),
- **down** with probability $(1-f) \times \text{Forget Value}$ (it is False, and the coin flip succeeds).

It drifts into memory when up beats down. With Memorize Value $= 1 - F$ and Forget Value $= F$:

$$
f\,(1-F) > (1-f)\,F \quad\Longleftrightarrow\quad f > F .
$$

So **the Forget Value is a frequency threshold.** With Forget Value 0.8, a literal must be True in more than 80% of the matched examples of its class to stay in the rule. With 0.5, more than half is enough. Low Forget Value → more literals pass → **longer, more specific rules**. That is why the papers call $s$ the *specificity*. (Erase Feedback pulls all literals down a bit more, so the threshold is approximate, but the idea holds.)

The book's breast-cancer example shows the same arithmetic. Among its three recurrence patients, Deg-malig 3 is True for all three; Menopause ge40 for two of three. With Memorize Value 0.2 and Forget Value 0.8, ge40 expects $2 \times 0.2 = 0.4$ steps up and $1 \times 0.8 = 0.8$ steps down per pass through those patients, so it is forgotten. With 0.5 and 0.5 it expects 1.0 up and 0.5 down, so it is memorized. In our run of the code at the end of this lesson, 771 of 1000 training runs with Forget Value 0.8 learned exactly the book's rule, *Deg-malig 3 AND NOT Menopause lt40* (plus the redundant ¬Deg-malig 1 and ¬Deg-malig 2).

On the vehicles, Yellow (and ¬Blue) are True for 2 of the 3 cars, $f \approx 0.67$. We trained a Car rule 500 times at each Forget Value and measured how often some colour literal was in the rule:

| Forget Value | 0.9 | 0.8 | 0.7 | 0.6 | 0.5 |
|---|---|---|---|---|---|
| time a colour is memorized | 0.1% | 5.4% | 45.9% | 91.3% | 99.4% |

The switch happens right around the predicted $F \approx 0.67$. (It is a bit above: once *Yellow* is in the rule, the rule only matches yellow cars, so Yellow is then True in 100% of the matched examples and holds on.)

!!! tip "What to notice"
    - High Forget Value (small $s$) → short, general rules that cover much of the class.
    - Low Forget Value (large $s$) → long, specific rules for sub-groups (only the two yellow cars).
    - Neither of these used a single **plane**. Recognize and Erase only ever look at the rule's own class.

---

## Step 7 — Frequent is not enough: Reject Feedback

A pattern can be typical of a class and *also* typical of the other classes. *Four Wheels AND Transports People* describes every car, and plane #4 too. On the full breast-cancer data the problem is real:

- Almost no patient is under 40 (Menopause lt40). So **¬Menopause lt40** is True for 83 of the 85 recurrence patients, a perfect frequent pattern.
- But it is also True for most patients without recurrence. The rule *IF ¬Menopause lt40 THEN Recurrence* matches 279 of the 286 patients, and only 30% of those it matches have a recurrence: no better than the base rate of 85 / 286.

**Reject Feedback** fixes this.

**When:** the example is of **another class**, and the rule **matches** it: the rule is making a mistake.

**What:** take every literal that is **False** for this example and currently **Forgotten**, and move it **one position up, always** (no coin flip).

Why exactly those literals? The rule matches, so all its memorized literals are True for this example. To stop matching, the rule needs just one memorized literal that is False here. Every Forgotten False literal is a candidate, and Reject Feedback nudges all of them. The first to cross from 5 to 6 makes the rule reject the example. In the walkthrough above, plane #4 pushes up ¬Four Wheels, ¬Transports People, ¬Wings, Yellow and ¬Blue, and ¬Wings, already at 5, crosses first.

Which candidate wins in the long run is decided by Recognize and Erase. Pushing up ¬Four Wheels is useless: it is False for every car, so Recognize Feedback immediately pushes it down again. ¬Wings is True for every car, so Recognize keeps it. The survivor is a literal that is **frequent in the rule's class and absent in the wrongly matched example**: frequent *and* discriminative.

!!! warning "Reject alone is not enough either"
    The book's notebook demonstrates this: give a rule only Reject Feedback and it keeps adding literals until it matches no plane, but it ends with contradictions like *Four Wheels AND NOT Four Wheels* that match nothing at all. In our check, a Car rule trained with only Reject matched **none** of the cars in all 1000 runs. The three feedback types work as a team.

---

## Step 8 — Train one rule yourself

Now run the real thing. Each step picks the rule's own class or the other class with equal chance (as the book's notebook does), then a random example from it, and applies the feedback from Step 2. The bars show each literal's position now, and the strip on the right its last 200 positions (green = memorized). The readout shows the current rule and how good it is:

- **coverage** = the fraction of the rule's own class it matches;
- **precision** = among all examples it matches, the fraction that really are of its class.

The data is the full **UCI Breast Cancer** dataset (286 patients from the University Medical Centre, Ljubljana, the same public source the book uses), with the book's nine features: Menopause lt40 / ge40 / premeno, Inv-nodes 0-2 / 3-5 / 6 or more, and Deg-malig 1 / 2 / 3.

<div id="viz-train" class="viz"><div class="viz-title">🏋️ Interactive: train one rule with Recognize, Erase and Reject</div></div>

**Try this** (seed 13, Forget Value 0.8, Reject switched off at first):

1. **Untick Reject** and press **+1000** twice. The rule becomes *IF ¬Menopause lt40 THEN Recurrence*: it matches 279 of 286 patients, coverage 83/85 = 0.98, precision 0.30. A beautifully frequent, useless rule.
2. **Tick Reject** and press **+1000** once. The rule switches to *IF Deg-malig 3 THEN Recurrence*: coverage 45/85 = 0.53, precision 45/85 = 0.53. It catches fewer recurrences, but a patient it flags is now 1.8 times as likely to have one.
3. **Drag the Forget Value to 0.2** (more specific) and press **+1000** a few times. The rules get longer and precision tends to rise further, at the cost of coverage.
4. Switch the data to **vehicles: Car rule**. Set the Forget Value to 0.9, press **Reset** and **+100**: *Four Wheels AND Transports People AND ¬Wings*, covering all 3 cars. Now set 0.5, press **Reset** and **+100** twice: the yellow-car rule *Four Wheels AND Transports People AND ¬Wings AND Yellow AND ¬Blue*, which covers only 2 of the 3 cars.

!!! tip "What to notice"
    - **A single rule never stops moving.** Keep pressing +1000 and the rule sometimes falls back to an empty or frequent-only condition, then finds a precise one again. The book stresses this: the outcome of learning is not one rule but a probability of being in each rule.
    - **Averaged over time it is clear.** We ran 50 seeds for 5000 steps each and averaged over steps 1000–5000. Forget Value 0.8: precision 0.33 without Reject, 0.44 with it. Forget Value 0.2: 0.37 versus 0.66. Forget Value 0.1: 0.42 versus 0.74, but then the rule covers only about 12% of recurrence patients.
    - **Coverage and precision trade off.** That is why a Tsetlin machine uses **many** rules: several precise rules together can cover the class (next lesson).

---

## Step 9 — Two literals, one rule: a game without talking

So far we followed literals one at a time. But a literal's feedback depends on the **other** literals in the same rule. Chapter 2, §2.2 of the book studies the smallest case, a rule with two literals $L_1$ and $L_2$, as a **game**.

**The Prisoner's Dilemma in one paragraph.** Two partners in crime are questioned in separate cells and can't talk. Each can stay silent or betray the other. If both stay silent, both get one year. If one betrays and the other stays silent, the traitor walks free and the silent one gets twenty years. If both betray, both get five. What is best for each prisoner depends on what the other one does, and they must decide without coordinating.

| | partner stays silent | partner betrays |
|---|---|---|
| **stay silent** | 1 year each | you 20 years, partner free |
| **betray** | you free, partner 20 years | 5 years each |

**The literal game.** Each literal is a "player" that, without talking to the other, is either Memorized or Forgotten. The four combinations give four rules: empty, $L_1$, $L_2$, or $L_1 \wedge L_2$. For a given observation, each combination decides whether the rule matches and therefore which feedback each literal gets. Look at observation ($L_1$ False, $L_2$ True, own class $Y$):

- If $L_1$ is Forgotten, the rule matches, so Recognize Feedback **pulls $L_2$ up**.
- If $L_1$ is Memorized, the rule fails, so Erase Feedback **pushes $L_2$ down**, although $L_2$ is True.

Same observation, same $L_2$, opposite feedback, all because of the other player's choice. No literal sees the others, yet their moves are coupled through the rule's truth value. That is how a rule gets coordinated without any communication.

Use the left panel to step through all eight observations (2 truth values × 2 truth values × 2 classes). The right panel solves the long-run behaviour exactly: with a memory of depth 8 per literal (positions 1–4 forgotten, 5–8 memorized), the pair of positions is a Markov chain with 64 states, and the heat map is its **stationary distribution**, the fraction of time spent at each pair of positions. Like the book, it uses Memorize Value 1.0 and Forget Value $1/s$, and all four input patterns are equally likely.

<div id="viz-game" class="viz"><div class="viz-title">🎲 Interactive: the literal game and where learning settles</div></div>

!!! tip "What to notice"
    - **Scenario 1: one useful literal.** $L_2$ predicts the class (precision 0.9), $L_1$ adds nothing. The rule is *L₂ only* **80.4%** of the time.
    - **Scenario 2: two equally good literals.** Each alone has precision 0.9. The time splits: *L₂ only* 42.5%, *L₁ only* 42.5%. Learning keeps visiting both alternatives.
    - **Scenario 3: pick the better one.** Lower $L_1$'s precision to 0.7. Now *L₂ only* gets 67.5% of the time against 13.4% for *L₁ only*.
    - **Scenario 4: combine.** Same data as Scenario 2 but $s = 4$ (Forget Value 0.25). Patterns that match only a quarter of the data are now frequent enough, and the rule is *L₁ ∧ L₂* (precision 0.99) **92.9%** of the time, with the single most likely state, both literals at position 8, at 0.36.
    - These are exactly the numbers in the book's Tables 2.5–2.8, which our 64-state calculation reproduces.

---

## Putting it all together

One rule's whole learning loop:

<div class="viz diagram">
<script type="application/json">
{
  "title": "🔁 How one rule learns",
  "cols": 3,
  "rows": 2,
  "rowHeight": 110,
  "nodes": [
    {"id": "init", "col": 0, "row": 0, "icon": "🆕", "label": "All literals at position 5", "shape": "pill", "sub": "empty rule", "note": "Every literal starts forgotten but one step from memorized."},
    {"id": "pick", "col": 1, "row": 0, "icon": "🎲", "label": "Pick an example", "sub": "own class or other", "role": "data"},
    {"id": "eval", "col": 2, "row": 0, "icon": "🔍", "label": "Does the rule match?", "role": "model", "note": "AND of the memorized literals."},
    {"id": "fb", "col": 2, "row": 1, "icon": "🧭", "label": "Recognize, Erase or Reject", "role": "loss", "note": "Own class + match: Recognize. Own class + no match: Erase. Other class + match: Reject. Otherwise nothing."},
    {"id": "upd", "col": 1, "row": 1, "icon": "↕️", "label": "Move literals ±1", "sub": "Memorize / Forget Value", "role": "opt", "note": "Recognize and Erase move literals at random with the Memorize and Forget Values; Reject always moves its literals."}
  ],
  "edges": [
    {"from": "init", "to": "pick"},
    {"from": "pick", "to": "eval"},
    {"from": "eval", "to": "fb"},
    {"from": "fb", "to": "upd"},
    {"from": "upd", "to": "pick", "label": "next"}
  ],
  "steps": [
    {"show": ["init"], "text": "A fresh rule: all literals at position 5, so the condition is empty and matches everything."},
    {"show": ["init>pick", "pick"], "text": "Pick a training example and look at its class."},
    {"show": ["pick>eval", "eval"], "text": "Evaluate the rule: are all its memorized literals True for this example?"},
    {"show": ["eval>fb", "fb"], "text": "Pick the feedback. <b>Recognize</b> and <b>Erase</b> (own class) find what is frequent; <b>Reject</b> (other class, wrong match) adds what discriminates."},
    {"show": ["fb>upd", "upd"], "text": "Move literals one position up or down. The Forget Value sets how frequent a literal must be to stay."},
    {"show": ["upd>pick", "pick"], "text": "Repeat. The memorized literals, read off at any time, are the rule."}
  ],
  "narrow": {"cols": 2, "rows": 3, "pos": {"init": [0, 0], "pick": [0, 1], "eval": [1, 1], "fb": [1, 2], "upd": [0, 2]}}
}
</script>
</div>

The whole algorithm for one rule in plain Python. It trains a Car rule on the vehicles at two Forget Values, then the book's six breast-cancer patients 1000 times:

```python
import random
from collections import Counter

DEPTH = 10                                   # memory positions 1..10; 6..10 = Memorized


def literals(x):
    """[x1, NOT x1, x2, NOT x2, ...] as 1/0 truth values."""
    return [v for xi in x for v in (xi, 1 - xi)]


def matches(pos, lits):
    """The rule is the AND of its memorized literals (empty rule = True)."""
    return all(lits[i] for i, p in enumerate(pos) if p > DEPTH // 2)


def learn_step(pos, lits, own_class, forget, rng):
    memorize = 1 - forget                    # Memorize Value + Forget Value = 1
    fires = matches(pos, lits)
    if own_class and fires:                  # Recognize: memorize True literals, forget False ones
        for i, v in enumerate(lits):
            if v and rng.random() <= memorize:
                pos[i] = min(pos[i] + 1, DEPTH)
            elif not v and rng.random() <= forget:
                pos[i] = max(pos[i] - 1, 1)
    elif own_class:                          # Erase: forget every literal
        for i in range(len(pos)):
            if rng.random() <= forget:
                pos[i] = max(pos[i] - 1, 1)
    elif fires:                              # Reject: memorize the Forgotten False literals (no randomness)
        for i, v in enumerate(lits):
            if not v and pos[i] <= DEPTH // 2:
                pos[i] += 1


def train_rule(X, y, target, forget, steps=1000, seed=0):
    rng = random.Random(seed)
    own = [i for i in range(len(X)) if y[i] == target]
    other = [i for i in range(len(X)) if y[i] != target]
    pos = [DEPTH // 2] * (2 * len(X[0]))     # every literal starts at position 5
    for _ in range(steps):
        k = rng.choice(own if rng.random() < 0.5 else other)
        learn_step(pos, literals(X[k]), y[k] == target, forget, rng)
    return pos


def rule_text(pos, features):
    names = [n for f in features for n in (f, "NOT " + f)]
    return " AND ".join(n for n, p in zip(names, pos) if p > DEPTH // 2) or "(empty)"


# Granmo, Ch. 1, Table 1.1: three cars (1) and three planes (0)
vehicles = ["Four Wheels", "Transports People", "Wings", "Yellow", "Blue"]
X_veh = [[1, 1, 0, 0, 1], [1, 1, 0, 1, 0], [1, 1, 0, 1, 0], [1, 1, 1, 0, 1], [1, 0, 1, 1, 0], [0, 1, 1, 0, 1]]
y_veh = [1, 1, 1, 0, 0, 0]
for forget in (0.9, 0.5):
    pos = train_rule(X_veh, y_veh, target=1, forget=forget, seed=1)
    print(f"Forget {forget}: IF {rule_text(pos, vehicles)} THEN Car")

# Granmo, Ch. 1, Table 1.4: six patients from the UCI breast-cancer data (1 = recurrence)
patients = ["lt40", "ge40", "premeno", "nodes 0-2", "nodes 3-5", "nodes 6-8", "malig 1", "malig 2", "malig 3"]
X_bc = [[0, 1, 0, 0, 1, 0, 0, 0, 1], [1, 0, 0, 1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 1, 0, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0, 1, 0, 0]]
y_bc = [1, 0, 1, 0, 1, 0]
rules = Counter(rule_text(train_rule(X_bc, y_bc, 1, forget=0.8, seed=s), patients) for s in range(1000))
for text, n in rules.most_common(2):
    print(f"{n:4d} of 1000 runs: IF {text} THEN Recurrence")
```

```text
Forget 0.9: IF Four Wheels AND Transports People AND NOT Wings THEN Car
Forget 0.5: IF Four Wheels AND Transports People AND NOT Wings AND Yellow AND NOT Blue THEN Car
 771 of 1000 runs: IF NOT lt40 AND NOT malig 1 AND NOT malig 2 AND malig 3 THEN Recurrence
  25 of 1000 runs: IF NOT lt40 AND NOT nodes 3-5 AND NOT malig 1 AND NOT malig 2 AND malig 3 THEN Recurrence
```

The whole learner is three `if` branches and some ±1 steps. Note that NOT malig 1 and NOT malig 2 follow from malig 3 (a tumour has one grade), so the most common rule is the book's *Deg-malig 3 AND NOT Menopause lt40*. (This script uses Python's own random generator, so its counts differ a little from the widgets, which use a small seeded generator of their own.)

| Feedback | Fires when | True literals | False literals | Random? | Finds |
|---|---|---|---|---|---|
| **Recognize** | own class, rule matches | up (Memorize Value) | down (Forget Value) | yes | frequent patterns |
| **Erase** | own class, rule fails | down (Forget Value) | down (Forget Value) | yes | keeps rules general |
| **Reject** | other class, rule matches | unchanged | Forgotten ones up | no | discriminating literals |

---

## Common misconceptions

!!! warning "\"Recognize Feedback only memorizes.\""
    It does both: True literals move up and False literals move down. The downward half is what removes details that don't belong to the pattern, like the colour of a particular car.

!!! warning "\"Reject Feedback forgets the literals that caused the wrong match.\""
    The opposite. Reject never forgets anything. It **memorizes** literals that are False for the wrongly matched example, which makes the rule's AND fail on it. Adding a literal can only make an AND rule match fewer examples.

!!! warning "\"A frequent pattern is a good classifier.\""
    *¬Menopause lt40* is True for 83 of the 85 recurrence patients, yet as a rule it has a precision of only 0.30, the base rate. Frequency within a class says nothing about the other classes. That is the job of Reject Feedback.

!!! warning "\"After training, the rule is fixed.\""
    The memory keeps moving as long as feedback continues. A rule learned from data with no perfect pattern keeps visiting several rules, spending more time in the better ones. In the widget, keep pressing +1000 and watch.

!!! warning "\"A lower Forget Value is always better because the rules get more precise.\""
    Lower Forget Value means more specific rules that cover less of their class. At 0.1, our single rule's precision rose to 0.74 but it covered only about 12% of recurrence patients. The right value depends on the data and on how many rules share the work.

---

## Check yourself

??? question "1. A Car rule is currently *Four Wheels AND Blue*. It sees vehicle #2 (a yellow car). Which feedback fires, and what happens to Four Wheels?"
    Vehicle #2 is a car (own class) but it is not Blue, so the rule is False → **Erase Feedback**. Every literal, Four Wheels included, moves down one position with probability equal to the Forget Value. Four Wheels is True for this car, but Erase doesn't look at truth values.

??? question "2. A Car rule is *Transports People*. It sees plane #6 (no four wheels, transports people, wings, not yellow, blue). Which literals can Reject Feedback move?"
    The rule matches the plane (it transports people) → **Reject Feedback**. It moves up every **Forgotten** literal that is **False** for this plane: Four Wheels, ¬Transports People, ¬Wings, Yellow and ¬Blue. Later, Recognize Feedback on the cars knocks ¬Transports People back down (False for every car), lets Yellow and ¬Blue fade at a high Forget Value (True for only two of the three cars), and keeps Four Wheels and ¬Wings (True for all cars).

??? question "3. With Memorize Value 0.3 and Forget Value 0.7, will a literal that is True for 60% of the matched examples of its class end up memorized?"
    Expected move up per example: $0.6 \times 0.3 = 0.18$. Down: $0.4 \times 0.7 = 0.28$. Down wins, so it is mostly **forgotten**. The threshold is $f > F = 0.7$, and 0.6 is below it. Lower the Forget Value below 0.6 (e.g. 0.5) and it would stay.

??? question "4. Why is Reject Feedback not randomized, while Recognize and Erase are?"
    Reject only fires on a clear **mistake**: the rule claimed an example of another class. One step up for every candidate literal fixes it fast. Recognize and Erase work on statistics (how often is this literal True in my class?), so random, small steps let the memory average over many examples instead of copying the last one.

??? question "5. In the literal game, observation (L₁ True, L₂ False, other class). What happens in each of the four cells?"
    The class is not the rule's own, so only Reject Feedback can fire, and only if the rule matches. **Empty rule**: matches → Reject: L₂ is False and Forgotten, so L₂ goes up; L₁ is True, so nothing. **L₁ only**: matches (L₁ True) → Reject: L₂ goes up. **L₂ only**: L₂ is False → rule doesn't match → no feedback. **L₁ ∧ L₂**: doesn't match → no feedback.

---

## Further reading

- Ole-Christoffer Granmo, [*An Introduction to Tsetlin Machines*](https://tsetlinmachine.org/) (free book). Chapter 1, §1.3 (Recognize and Erase Feedback) and §1.4 (Reject Feedback) are the source of this lesson's algorithm and vehicle example; Chapter 2, §2.2 (Multi-Literal Learning) is the source of the literal game. The book's Chapter 1 notebook implements one rule in about 50 lines.
- Ole-Christoffer Granmo, [*The Tsetlin Machine – A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic*](https://arxiv.org/abs/1804.01508) (2018). The original paper, with Type I and Type II feedback as tables.
- M. Zwitter & M. Soklic, [Breast Cancer dataset](https://archive.ics.uci.edu/dataset/14/breast+cancer), UCI Machine Learning Repository. The 286-patient data used in the widget.

**Next lesson:** [Many rules voting: classification with a Tsetlin machine](../tsetlin-machine-classification/index.md)

<script src="viz/tsetlin-machine-feedback.js"></script>
