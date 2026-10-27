# The Tsetlin Automaton: A Memory That Learns

!!! abstract "TL;DR"
    A **Tsetlin automaton** is a learner whose entire memory is **one integer**: a position on a short ladder. Good news (a reward) pushes it deeper into what it is doing; bad news (a penalty) pushes it toward the middle, and crossing the middle means **changing its mind**. Tsetlin invented it in the 1960s to play bandit problems. A Tsetlin machine gives every literal of every rule one of these memories: the position goes **up** (memorize) or **down** (forget) at random, depending on the training examples. Because each update is a small random step, the position does a **random walk**, and we can compute exactly where it spends its time. A literal that is True often enough for its class ends up **Memorized**, and the **Forget Value** sets what "often enough" means. A **deeper memory** turns this into a sharper, more decisive yes-or-no.

**Level:** beginner–intermediate · **Prerequisites:** [What is a Tsetlin machine?](../tsetlin-machine-intro/index.md) (literals and rules) and [Multi-armed bandits](../multi-armed-bandits/index.md) (arms, rewards, ε-greedy). The math needs only fractions and powers.

---

## Why should I care?

- **It is the only moving part.** A Tsetlin machine is built from thousands of these tiny memories, one per literal per rule. Every rule it learns is just a list of which memories are in their upper half. Understand one memory and you understand what a whole Tsetlin machine is made of.
- **You can predict the result exactly.** For a neural network, "what will this weight end up as?" rarely has a clean answer. For one literal of a Tsetlin machine, it has a short formula (Granmo, *An Introduction to Tsetlin Machines*, Ch. 2, §2.1). You will derive it, check it by simulation, and see why the **Forget Value** and the **memory depth** do what they do.
- **It is extremely cheap.** A memory that is one small integer, updated by adding or subtracting 1, is ideal for low-power chips. That is a large part of why Tsetlin machines are interesting for hardware, including medical devices that run on a battery.

---

## Step 1 — Tsetlin's idea: a memory that is a single integer

Michael Lvovitch Tsetlin (1924–1966) was a Soviet mathematician and physicist. Around 1961 he asked a simple question: how can a very small machine **learn to act well in a random environment**, when it doesn't know the rules of that environment and can't store much? His answer, now called the **Tsetlin automaton**, keeps track of all its experience with **one integer**. The Tsetlin machine is named after him (Granmo, Ch. 1, "Tsetlin Machine Origins").

The setting is exactly the [two-armed bandit](../multi-armed-bandits/index.md#step-1-the-k-armed-bandit-one-state-k-actions) from the RL series: two actions (arms), each giving a **reward** with some unknown probability and a **penalty** otherwise. The automaton has $2N$ states in a row:

- the left $N$ states mean "play arm A", the right $N$ states mean "play arm B";
- states far from the centre are **deep** (confident), states next to the centre are **shallow** (about to change its mind).

The learning rule is just two sentences:

1. **Reward:** move one step **deeper** into the current half (or stay put if already at the deepest state).
2. **Penalty:** move one step **toward the centre**. If you were already next to the centre, you cross over, so the next pull uses the **other arm**.

<div class="viz diagram">
<script type="application/json">
{
  "title": "🔁 One step of a Tsetlin automaton",
  "cols": 2,
  "rows": 2,
  "rowHeight": 105,
  "nodes": [
    {"id": "play", "col": 0, "row": 0, "icon": "🎰", "label": "Play the arm of your half", "sub": "state k is the whole memory", "role": "model", "note": "States 1..N play arm A, states N+1..2N play arm B."},
    {"id": "env", "col": 1, "row": 0, "icon": "🎲", "label": "Environment answers", "sub": "reward or penalty", "role": "data", "note": "Each arm rewards with its own hidden probability."},
    {"id": "move", "col": 1, "row": 1, "icon": "↔️", "label": "Move one step", "sub": "reward: deeper · penalty: inward", "role": "loss", "note": "A reward makes the automaton more sure of its current arm. A penalty makes it less sure."},
    {"id": "half", "col": 0, "row": 1, "icon": "🧭", "label": "Which half am I in now?", "sub": "crossed the centre = switch", "shape": "decision", "role": "grad", "note": "Changing its mind needs several penalties in a row when the automaton is deep in one half."}
  ],
  "edges": [
    {"from": "play", "to": "env"},
    {"from": "env", "to": "move"},
    {"from": "move", "to": "half"},
    {"from": "half", "to": "play", "label": "next pull"}
  ],
  "steps": [
    {"show": ["play"], "text": "The automaton's state is a single integer. Its half (left or right of the centre) decides which arm it plays."},
    {"show": ["play>env", "env"], "text": "The environment pays a <b>reward</b> or gives a <b>penalty</b>, at random, with the arm's hidden probability."},
    {"show": ["env>move", "move"], "text": "Reward: one step <b>deeper</b> (more confident). Penalty: one step <b>toward the centre</b> (less confident)."},
    {"show": ["move>half", "half"], "text": "If a penalty pushed it across the centre, the automaton has <b>changed its mind</b>: it now plays the other arm."},
    {"show": ["half>play", "play"], "text": "Repeat. No averages, no counts, no table: one integer is all it remembers."}
  ]
}
</script>
</div>

Compare this with the bandit agents you already know. ε-greedy keeps an **average reward** and a **count** for every arm and does arithmetic with them. The Tsetlin automaton keeps no estimate of anything. Its position is a kind of running "confidence score", and the depth $N$ decides how many penalties in a row it takes to talk it out of its current choice.

!!! tip "What to notice"
    - **Win-stay, lose-shift, with inertia.** With $N = 1$ the rule is "keep the arm after a reward, switch after a penalty". Larger $N$ adds inertia: a deep automaton shrugs off a few unlucky penalties.
    - **It never stops moving.** Even on the better arm, penalties keep coming now and then, so the state keeps wandering. Learning is a **balance** between being pushed out and pulled in, not a final answer.

---

## Step 2 — Watch it play a two-armed bandit

Arm A pays a reward with probability 0.4, arm B with probability 0.7, and the automaton starts in the shallowest state of arm A (the worse arm). Next to it, on the same two arms, an ε-greedy agent from the bandits lesson ($\varepsilon = 0.1$, sample averages) plays its own game. Press **+1000 pulls** right after the page loads to get the run quoted below.

<div id="viz-ta-bandit" class="viz"><div class="viz-title">🎰 Interactive: a Tsetlin automaton plays a two-armed bandit</div></div>

!!! tip "What to notice"
    - **It finds the better arm with one integer.** In the first run (run #1, $N = 3$), after 1000 pulls the automaton had played arm B on **92.7%** of pulls and the ε-greedy agent on **81.3%**. That run is a lucky one for the automaton. Averaged over 500 runs (in Python), the automaton reaches **88.8%** over the first 1000 pulls and ε-greedy **93.4%**. So with $N = 3$ the automaton is slightly worse than ε-greedy on average, with far less memory. With $N = 5$ it reaches **96.7%** and overtakes.
    - **Depth is the dial.** The green dashed line is the exact long-run share (computed in Step 5's way). For these arms it is **66.7%** with $N = 1$, **80.0%** with $N = 2$, **89.3%** with $N = 3$, **97.5%** with $N = 5$ and **99.96%** with $N = 10$. Slide $N$ and watch the dot settle deeper.
    - **It keeps changing its mind, a little.** In run #1 the automaton switched arms **45** times in 1000 pulls. Each switch is a short excursion to arm A that ends after a penalty or two.
    - **Now press "Swap the arms", then +1000 pulls again.** The world changes: arm A becomes the good one. In run #1, over those 1000 pulls the automaton spent **87.6%** of pulls on the new best arm, the ε-greedy agent only **5.2%** (the readout's last-100 shares end at 94% and 7%). Its sample averages are built from hundreds of old rewards, so they take a very long time to move. The automaton only needs a short run of penalties to cross the centre. That is what Granmo means by an algorithm that can **"change its mind"**.

!!! note "When does the automaton fail?"
    A penalty pushes inward and a reward pushes outward, so the automaton can only settle on an arm if that arm gives **more rewards than penalties**, i.e. pays with probability above 0.5. With arms paying 0.2 and 0.4, both push the automaton toward the centre more often than away from it. Then even 20 states per arm give only **75.0%** on the better arm, and more states don't help. The same "which push is stronger?" question decides what a Tsetlin machine's memory learns, as Steps 4–7 show.

---

## Step 3 — From actions to memory: one integer per literal

A Tsetlin machine reuses the automaton, with new names. Each **literal** in each rule (lesson 1) gets its own little ladder of memory positions:

| Tsetlin automaton | The book's literal memory |
|---|---|
| $2N$ states | memory positions $1, 2, \dots, D$ (the book's Chapter 1 uses $D = 10$) |
| the two actions | **Forgotten** (lower half, positions $1 \dots D/2$) or **Memorized** (upper half) |
| deepest state of each action | position 1 = **Maximally Forgotten**, position $D$ = **Maximally Memorized** |
| reward / penalty | feedback that pushes the literal **up** (memorize) or **down** (forget) |

A rule's condition is the AND of its **Memorized** literals. The book calls the number of positions $D$ the **memory depth**; in research papers you'll read "a Tsetlin automaton with $2N$ states", so $D = 2N$, and the two actions are called *include* and *exclude*. A new rule starts with every literal just below the line (position $D/2$, "about to be memorized but still Forgotten").

Two details matter for everything that follows:

- **Updates are random.** Before moving a literal up, the machine draws a random number between 0 and 1 and skips the move if it is above the **Memorize Value**. Moving down is skipped if the number is above the **Forget Value**. Randomness stops a coincidence from locking the rule in, and it keeps rules exploring (Granmo, Ch. 1, §1.3).
- **Which way to push.** For one literal and a training example of the rule's own class $Y$, the rule is simple: if the literal is **True**, push it **up** (memorize); if it is **False**, push it **down** (forget). Lesson 3 explains the full feedback rules, which also handle the whole rule and the other classes.

!!! tip "What to notice"
    - **"Memorized" is a region, not a single value.** A literal at position 6 and one at position 10 are both in the rule. The deeper one would need many more forgetting steps to leave.
    - **Memorize and Forget Value are two speed dials.** The book's Chapter 1 examples use pairs that add to 1, like Memorize 0.1 and Forget 0.9. From Chapter 2 on, the book (like the research papers) uses **Memorize Value 1** and **Forget Value $1/s$**, where $s$ is called the *specificity*. This lesson uses that version.

---

## Step 4 — One literal, one class: a random walk

Take the simplest case. A rule for class $Y$ sees only examples of $Y$, and one of its literals, $L$, is True in a fraction $p = P(L \mid Y)$ of them. Each example then moves the literal:

- **up one position** with probability $p$ (literal True; Memorize Value 1, so the move always happens), unless already at the top;
- **down one position** with probability $(1-p)\cdot\frac{1}{s}$ (literal False, and the random draw allows the forget step), unless already at the bottom;
- otherwise it stays where it is.

The examples come in random order, so the position wanders up and down: a **random walk**. Its next position depends only on where it is now and on the next random example. Mathematicians call such a process a **Markov chain** (the same "Markov" as in [MDPs](../mdp-bellman-equations/index.md)).

<div id="viz-literal" class="viz"><div class="viz-title">🧠 Interactive: one literal's memory, example by example</div></div>

!!! tip "What to notice"
    - **The bars settle onto the orange ticks.** With the defaults ($p = 0.3$, $s = 3$, depth 10), press **+10,000 examples** once: the literal has spent **75.9%** of its time Memorized, against an exact long-run value of **77.8%**. A million examples in Python give 77.7%. The orange ticks are that exact answer, derived in the next step.
    - **It never stops wandering.** Press Play and watch the triangle. Even when it is usually Memorized, a run of False examples can push it below the line for a while.
    - **Find the tipping point.** Slide $p$ down to 0.25 with $s = 3$: the bars spread out evenly over all positions. Just above 0.25, the literal leans toward Memorized; just below, toward Forgotten. The readout's ratio **ρ** crosses 1 exactly there.

---

## Step 5 — The exact learning outcome

Run the walk for a long time and record the share of time it spends at each position. That share settles down to a fixed distribution, the **stationary distribution** $\pi_1, \dots, \pi_D$. The book finds it by solving $\pi = \pi \mathbf{M}$, where $\mathbf{M}$ is the matrix of transition probabilities (or by multiplying a starting distribution by $\mathbf{M}^{10000}$). For a walk that only moves one step at a time there is a shortcut:

> **In the long run, the walk crosses each gap upward exactly as often as downward.** (It can't go up across the gap between positions $i$ and $i+1$ twice without coming back down in between.)

The rate of upward crossings is "time at $i$" × "probability of moving up", and the rate of downward crossings is "time at $i+1$" × "probability of moving down". So

$$
\pi_i \cdot p \;=\; \pi_{i+1} \cdot \frac{1-p}{s}
\qquad\Longrightarrow\qquad
\frac{\pi_{i+1}}{\pi_i} \;=\; \rho \;=\; \frac{s\,p}{1-p}.
$$

Every step up the ladder multiplies the time spent there by the same ratio $\rho$. So $\pi_i \propto \rho^{\,i-1}$: a geometric staircase, rising if $\rho > 1$ and falling if $\rho < 1$. Add up the upper half (with $D = 2n$ positions) and almost everything cancels:

$$
P(\text{Memorized}) \;=\; \frac{\rho^{\,n}}{1+\rho^{\,n}}.
$$

Read off what it says:

- **The tipping point is $\rho = 1$**, i.e. $p = p^* = \dfrac{1}{s+1}$. A literal that is True in more than a fraction $\frac{1}{s+1}$ of its class's examples is more likely Memorized than not. With $s = 3$ that is $p^* = 0.25$, which is why the widget balanced there.
- **The Forget Value sets the tipping point.** A large $s$ (small Forget Value $1/s$) means slow forgetting, so even rarer literals stay: with $s = 15$, $p^* = 0.0625$. A small $s$ keeps only very common literals. The book puts it as "the Forget Value allows you to control the frequency of the patterns learnt" (Ch. 2, §2.1).
- **The defaults again:** $p = 0.3$, $s = 3$ gives $\rho = 1.286$, and with depth 10 ($n = 5$), $P(\text{Memorized}) = 1.286^5/(1 + 1.286^5) = $ **0.778**, the orange ticks above.

!!! note "Chapter 1's Memorize Value + Forget Value = 1"
    With a general Memorize Value $m$ and Forget Value $f$, the ratio is $\rho = \frac{m\,p}{f\,(1-p)}$. If $m + f = 1$, then $\rho = 1$ exactly when $p = f$: **a literal survives if it is True in more than a fraction "Forget Value" of the class's examples.** Take the three cars of the book's vehicle table (Granmo, Ch. 1, Table 1.1): *Four Wheels* is True for all three, *Yellow* for two, *Blue* for one. With depth 10 and Memorize 0.5 / Forget 0.5, *Yellow* (2/3 > 0.5) is Memorized with probability 0.970 and *Blue* (1/3) only 0.030. With Chapter 1's Memorize 0.1 / Forget 0.9, even *Yellow* drops to 0.001, and only literals that hold for every car, like *Four Wheels*, stay. (This treats each literal on its own. In a real rule the literals interact, which is the subject of lesson 3.)

!!! tip "What to notice"
    - **The answer is a probability, not a verdict.** The Tsetlin machine never "decides once and for all". The book: "the outcome of Tsetlin machine learning is not a single solution", it is a distribution over solutions that the machine keeps visiting.
    - **The start doesn't matter.** The stationary distribution is the same wherever the literal starts. That is why the book says the initialization "does not affect the end result" (Ch. 1, §1.3).

---

## Step 6 — Deeper memory, sharper decisions

The formula $\rho^n/(1+\rho^n)$ has a familiar shape. Write it as $1/(1+\rho^{-n})$ = $\sigma(n \ln\rho)$, where $\sigma$ is the logistic **sigmoid**. The depth $n$ multiplies the "evidence" $\ln\rho$ before the sigmoid. Doubling the depth makes the switch from Forgotten to Memorized twice as steep.

<div id="viz-sharpen" class="viz"><div class="viz-title">📈 Interactive: deeper memory sharpens the decision</div></div>

!!! tip "What to notice"
    - **Every curve passes through 0.5 at $p^*$.** Depth doesn't move the tipping point; only $s$ does.
    - **Depth decides how decisive the literal is.** With $s = 3$, take $p = 0.3$, just above $p^* = 0.25$: depths 2, 4, 8 and 20 give $P(\text{Memorized}) = $ **0.563, 0.623, 0.732, 0.925**. At $p = 0.2$, just below: **0.429, 0.360, 0.240, 0.053**. A shallow memory hedges; a deep one commits.
    - **Depth 2 is a coin flip with a bias.** With one Forgotten and one Memorized position, the curve is very flat: the literal flips between in and out of the rule all the time.
    - **The price of depth is speed.** Back in the bandit widget, a deep automaton needs more penalties to cross the centre. Averaged over 500 runs, in the first 100 pulls after the arms swap, the automaton played the new best arm **82.3%** of the time with $N = 3$ but only **20.5%** with $N = 20$.

---

## Step 7 — The other class joins: the book's complete single-literal chain

A real rule also sees examples of **other** classes, written $\bar{Y}$ ("not $Y$"). The book (Ch. 2, §2.1) builds the full chain from the four kinds of example a single literal can meet. With Memorize Value 1 and Forget Value $1/s$:

| Example | What happens to the literal | Feedback (lesson 3) |
|---|---|---|
| class $Y$, literal True: $(L, Y)$ | one step **up** | Recognize |
| class $Y$, literal False: $(\lnot L, Y)$ | one step **down**, with probability $1/s$ | Erase |
| other class, literal False: $(\lnot L, \bar{Y})$ | one step **up**, but **only while Forgotten** | Reject |
| other class, literal True: $(L, \bar{Y})$ | nothing | none |

The third row is new. If the literal is Forgotten, the rule ignores it, and the rule wrongly fires for an example of another class. Memorizing a literal that is False for that example would make the rule reject it. Once the literal is Memorized, the job is done and Reject Feedback stops pushing. Multiply each kind of move by how often its example occurs and you get three numbers per example:

$$
\begin{aligned}
a &= P(L \mid Y)\,P(Y) + P(\lnot L \mid \bar{Y})\,P(\bar{Y}) && \text{(up, while Forgotten)}\\
r &= P(L \mid Y)\,P(Y) && \text{(up, while Memorized)}\\
e &= \tfrac{1}{s}\,P(\lnot L \mid Y)\,P(Y) && \text{(down, anywhere above 1)}
\end{aligned}
$$

<div class="viz diagram">
<script type="application/json">
{
  "title": "🪜 The four-position memory of one literal (book Fig. 2.8, redrawn)",
  "cols": 4,
  "rows": 1.4,
  "rowHeight": 80,
  "nodes": [
    {"id": "s1", "col": 0, "row": 0.2, "label": "<b>1</b>", "sub": "Max. Forgotten", "maxWidth": 105, "note": "Maximally Forgotten: a forget step can't go lower."},
    {"id": "s2", "col": 1, "row": 0.2, "label": "<b>2</b>", "sub": "Forgotten", "maxWidth": 105, "note": "Forgotten, but one step from the rule."},
    {"id": "s3", "col": 2, "row": 0.2, "label": "<b>3</b>", "sub": "Memorized", "role": "opt", "maxWidth": 105, "note": "Memorized: the literal is part of the rule's condition."},
    {"id": "s4", "col": 3, "row": 0.2, "label": "<b>4</b>", "sub": "Max. Memorized", "role": "opt", "maxWidth": 105, "note": "Maximally Memorized: a memorize step can't go higher."}
  ],
  "edges": [
    {"from": "s1", "to": "s2", "label": "a", "curve": 0.4},
    {"from": "s2", "to": "s3", "label": "a", "curve": 0.4},
    {"from": "s3", "to": "s4", "label": "r", "curve": 0.4},
    {"from": "s2", "to": "s1", "label": "e", "curve": 0.4},
    {"from": "s3", "to": "s2", "label": "e", "curve": 0.4},
    {"from": "s4", "to": "s3", "label": "e", "curve": 0.4}
  ],
  "steps": [
    {"show": ["s1", "s2", "s3", "s4"], "text": "Four positions: 1–2 Forgotten, 3–4 Memorized. Each training example moves the literal at most one step."},
    {"show": ["s1>s2", "s2>s3", "s1", "s2", "s3"], "text": "While Forgotten, the literal climbs with probability <b>a</b>: Recognize (class Y, literal True) plus Reject (other class, literal False)."},
    {"show": ["s3>s4", "s3", "s4"], "text": "Once Memorized, only Recognize pushes it up: probability <b>r</b>. Reject Feedback has done its job."},
    {"show": ["s2>s1", "s3>s2", "s4>s3", "s1", "s2", "s3", "s4"], "text": "From every position above 1, Erase (class Y, literal False) pulls it down with probability <b>e</b>, which includes the Forget Value 1/s."},
    {"show": ["s1", "s2", "s3", "s4"], "text": "With the remaining probability it stays where it is. The stationary distribution balances the up and down flow across every gap."}
  ],
  "narrow": {"cols": 1, "rows": 4, "rowHeight": 95, "pos": {"s4": [0, 0], "s3": [0, 1], "s2": [0, 2], "s1": [0, 3]}}
}
</script>
</div>

The same balance argument as in Step 5 works, with ratio $a/e$ for the steps inside the Forgotten half (and across the line) and $r/e$ inside the Memorized half. For depth 4 it gives

$$
\pi_1 : \pi_2 : \pi_3 : \pi_4 \;=\; e^3 \;:\; a\,e^2 \;:\; a^2 e \;:\; a^2 r ,
$$

which is the book's Table 2.1 after multiplying everything by $s^3/P(Y)$. Larger depths just add more factors of $a/e$ and $r/e$. The four scenarios of the book make the factors concrete. We computed each one exactly and by simulating a million examples (code below):

| Book scenario | Settings | Exact $\pi_1 \dots \pi_D$ | $P(\text{Memorized})$ | Book says |
|---|---|---|---|---|
| 1. Infrequent literal, fast forgetting | $s = 1$, $P(Y) = 0.9$, $P(L \mid Y) = 0.1$, $P(\lnot L \mid \bar{Y}) = 0.9$, depth 4 | 0.783, 0.174, 0.039, 0.004 | **0.043** | 0.04 |
| 2. Same literal, slow forgetting | as 1, but $s = 15$ | 0.029, 0.098, 0.327, 0.545 | **0.872** | 0.87 |
| 3. Discriminative literal | $s = 15$, $P(Y) = 0.82$, $P(L \mid Y) = 0.99$, $P(\lnot L \mid \bar{Y}) = 0.5$, depth 4 | 0.000, 0.000, 0.001, 0.999 | $1 - 4.1 \times 10^{-7}$ | 0.9999 |
| 4. Deeper memory | as 3, but depth 8 | almost all at 8 | $1 - 1.9 \times 10^{-13}$ | ≈ 1.0 |

<div id="viz-literal-full" class="viz"><div class="viz-title">🧪 Interactive: the complete single-literal chain, with the book's scenarios</div></div>

!!! tip "What to notice"
    - **Scenario 1 vs 2: the Forget Value decides.** In scenario 1 the literal is True for only 10% of class $Y$ and carries no information about the class (the book checks with Bayes' rule that $P(Y \mid L) = 0.9 = P(Y)$). With fast forgetting ($s = 1$) it is almost always Forgotten. Slow the forgetting to $s = 15$ and the same literal is kept 87% of the time: it is now above the frequency threshold the Forget Value sets. The widget starts in scenario 2; its first 10,000 examples spend **87.9%** of the time Memorized.
    - **Scenario 3: discriminative literals are kept firmly.** A literal that is True for 99% of class $Y$ **and** False for half the other class gets pushed up by both Recognize and Reject. It is Memorized all but $4 \times 10^{-7}$ of the time.
    - **Scenario 4: depth again.** Going from 4 to 8 positions shrinks the chance of being Forgotten from $4.1 \times 10^{-7}$ to $1.9 \times 10^{-13}$. Each extra position multiplies the odds by another factor $a/e$ or $r/e$: "increasing the number of Literal Automaton states improves learning accuracy exponentially" (Ch. 2, §2.1).

---

## Putting it all together

<div class="viz diagram">
<script type="application/json">
{
  "title": "🧩 How one literal learns",
  "cols": 3,
  "rows": 2,
  "rowHeight": 110,
  "nodes": [
    {"id": "start", "col": 0, "row": 0, "icon": "🆕", "label": "Start just below the line", "sub": "position D/2", "shape": "pill", "note": "Any start gives the same long-run result."},
    {"id": "draw", "col": 1, "row": 0, "icon": "📋", "label": "Next training example", "sub": "class, literal True/False", "role": "data"},
    {"id": "case", "col": 2, "row": 0, "label": "Which of the 4 cases?", "shape": "decision", "role": "grad", "note": "(L, Y): up · (¬L, Y): down with prob. 1/s · (¬L, other): up while Forgotten · (L, other): nothing."},
    {"id": "move", "col": 2, "row": 1, "icon": "🎲", "label": "Move one step at random", "sub": "Memorize / Forget Value", "role": "model"},
    {"id": "rule", "col": 1, "row": 1, "icon": "📏", "label": "In the rule if in the upper half", "role": "opt"},
    {"id": "end", "col": 0, "row": 1, "icon": "📊", "label": "Long run: P(Memorized)", "sub": "ρⁿ / (1 + ρⁿ)", "shape": "pill", "role": "loss"}
  ],
  "edges": [
    {"from": "start", "to": "draw"},
    {"from": "draw", "to": "case"},
    {"from": "case", "to": "move"},
    {"from": "move", "to": "rule"},
    {"from": "rule", "to": "draw", "label": "next"},
    {"from": "rule", "to": "end"}
  ],
  "steps": [
    {"show": ["start"], "text": "Every literal of a new rule starts just below the Memorized line."},
    {"show": ["start>draw", "draw"], "text": "A training example arrives: its class, and whether this literal is True for it."},
    {"show": ["draw>case", "case"], "text": "Four cases: own class and True (memorize), own class and False (forget), other class and False (memorize while Forgotten), other class and True (nothing)."},
    {"show": ["case>move", "move"], "text": "Move one position, if the random draw allows it (Memorize Value 1, Forget Value 1/s)."},
    {"show": ["move>rule", "rule"], "text": "The literal is part of the rule's AND exactly when it sits in the upper half."},
    {"show": ["rule>draw", "draw"], "text": "Repeat for the next example. The position never stops moving: a random walk."},
    {"show": ["rule>end", "end"], "text": "Over many examples, the time spent in the upper half approaches the exact stationary probability. Deeper memory makes it closer to 0 or 1."}
  ],
  "narrow": {"cols": 2, "rows": 4, "rowHeight": 95, "pos": {"start": [0, 0], "draw": [0, 1], "case": [1, 1], "move": [1, 2], "rule": [0, 2], "end": [0, 3]}}
}
</script>
</div>

| Knob | What it does | Effect on one literal |
|---|---|---|
| $p = P(L \mid Y)$ | how often the literal is True in its class | more often → pushed up more often |
| Forget Value $1/s$ | how often a forget step is carried out | sets the tipping point $p^* = 1/(s+1)$ (only class $Y$) |
| memory depth $D$ | number of memory positions | same tipping point, but a sharper, more decisive outcome; slower to change its mind |
| other class: $P(\lnot L \mid \bar{Y})$ | how often the literal rules out other classes | extra push up while Forgotten (Reject): discriminative literals are kept |

Here is the whole analysis in numpy: the exact stationary distribution from the "balanced flow across each gap" rule, and a simulation to check it, on the book's four scenarios.

```python
import numpy as np


def literal_chain(p_Y, p_L_Y, p_notL_notY, s, depth):
    """Up/down probabilities of one literal's memory position (positions 1..depth)."""
    a = p_L_Y * p_Y + p_notL_notY * (1 - p_Y)    # up while forgotten: Recognize + Reject
    r = p_L_Y * p_Y                              # up while memorized: Recognize only
    e = (1 - p_L_Y) * p_Y / s                    # down: Erase with Forget Value 1/s
    up = [a if i <= depth // 2 else r for i in range(1, depth)]   # from position i to i+1
    return up, e


def stationary(p_Y, p_L_Y, p_notL_notY, s, depth):
    """Exact long-run distribution: flow up across each gap = flow down across it."""
    up, e = literal_chain(p_Y, p_L_Y, p_notL_notY, s, depth)
    w = np.cumprod([1.0] + [u / e for u in up])
    return w / w.sum()


def simulate(p_Y, p_L_Y, p_notL_notY, s, depth, steps=1_000_000, seed=0):
    rng = np.random.default_rng(seed)
    pos, time_at = depth // 2, np.zeros(depth)
    for u1, u2, u3 in rng.random((steps, 3)):
        if u1 < p_Y:                                         # an example of class Y
            if u2 < p_L_Y:
                pos = min(depth, pos + 1)                    # literal True: memorize
            elif u3 < 1 / s:
                pos = max(1, pos - 1)                        # literal False: forget, with prob 1/s
        elif u2 < p_notL_notY and pos <= depth // 2:         # other class, literal False
            pos += 1                                         # memorize, but only while forgotten
        time_at[pos - 1] += 1
    return time_at / steps


scenarios = {"1": (0.9, 0.1, 0.9, 1, 4), "2": (0.9, 0.1, 0.9, 15, 4),
             "3": (0.82, 0.99, 0.5, 15, 4), "4": (0.82, 0.99, 0.5, 15, 8)}
for name, args in scenarios.items():
    exact, sim = stationary(*args), simulate(*args)
    half = args[-1] // 2
    print(f"scenario {name}: P(memorized) exact {exact[half:].sum():.4f}, simulated {sim[half:].sum():.4f}")
print("scenario 2, exact per position:", np.round(stationary(*scenarios["2"]), 3))
```

```text
scenario 1: P(memorized) exact 0.0430, simulated 0.0426
scenario 2: P(memorized) exact 0.8724, simulated 0.8719
scenario 3: P(memorized) exact 1.0000, simulated 1.0000
scenario 4: P(memorized) exact 1.0000, simulated 1.0000
scenario 2, exact per position: [0.029 0.098 0.327 0.545]
```

The exact values match the book's figures (0.04, 0.87, 0.9999 and ≈ 1.0) and the simulation agrees to within a few thousandths. We also checked the exact answer against the book's own method, $\pi \mathbf{M}^{10000}$, and against its Table 2.1.

---

## Common misconceptions

!!! warning "\"A Memorized literal stays memorized.\""
    The position keeps moving forever. In scenario 2 the literal is Memorized 87.2% of the time, so 12.8% of the time it is out of the rule. Learning gives a **distribution**, and the rule you read off at any moment is one sample from it. Deeper memory makes the sample far more reliable.

!!! warning "\"Deeper memory is simply better.\""
    Deeper memory makes the long-run outcome sharper, but it also makes the automaton slower to change its mind. After the arms swapped, an automaton with 20 states per arm played the new best arm only 20.5% of the time in the next 100 pulls, against 82.3% with 3 states (500 runs each). Depth trades **decisiveness** against **adaptivity**.

!!! warning "\"The automaton estimates how good each action is.\""
    It stores no estimate, average or count, only its position. The position is not a probability either: two automata at the same position can face very different environments. Only the **long-run distribution** of the position tells you about the environment.

!!! warning "\"A literal is memorized if it is True for most examples of the class.\""
    The tipping point is $p^* = 1/(s+1)$, set by the Forget Value, not 50%. With $s = 15$ a literal that is True for only 7% of the class can still be Memorized more often than not. Choosing $s$ is choosing how frequent a pattern must be to enter a rule.

---

## Check yourself

??? question "1. A Tsetlin automaton with N = 1 (one state per arm) plays arms paying 0.4 and 0.7. What share of pulls goes to the better arm in the long run?"
    With $N = 1$ it is win-stay, lose-shift: it leaves an arm after every penalty. It leaves arm A with probability 0.6 and arm B with probability 0.3, so it stays on B twice as long: share on B $= \frac{0.6}{0.6 + 0.3} = $ **66.7%**, the value the widget shows for $N = 1$.

??? question "2. Only class-Y examples, s = 4, and the literal is True in 25% of them. Is it more likely Memorized or Forgotten? What is P(Memorized) with depth 4?"
    The tipping point is $p^* = 1/(4+1) = 0.2$, and $0.25 > 0.2$, so **Memorized** is more likely. $\rho = \frac{4 \times 0.25}{0.75} = 1.333$, depth 4 means $n = 2$: $P = \frac{1.333^2}{1 + 1.333^2} = \frac{1.778}{2.778} = $ **0.64**.

??? question "3. You want rules to keep literals that are True in at least 10% of a class's examples. Which s (and Forget Value) should you choose?"
    Set the tipping point $\frac{1}{s+1} = 0.1$, so $s = 9$ and the Forget Value is $1/9 \approx 0.111$. (This uses the single-literal picture; in a real rule, literals interact, as lesson 3 shows.)

??? question "4. Why does Reject Feedback push the literal up only while it is Forgotten?"
    Reject Feedback is there to stop the rule from firing on another class. If the literal is already Memorized and False for that example, the rule's AND is already False: the rule doesn't fire, and there is nothing to fix. Only a Forgotten literal that is False for the other class could still help, by joining the rule.

??? question "5. In the bandit widget, swapping the arms barely bothers the automaton but ruins ε-greedy. Why?"
    ε-greedy's sample average for the old best arm is built from hundreds of rewards, and each new reward moves it by only 1/n. It takes a very long time to fall below the other arm's estimate (which is rarely updated, since that arm is mostly explored at random). The automaton has no such long memory: a few penalties in a row move it across the centre. In run #1 that meant 87.6% vs 5.2% on the new best arm over the next 1000 pulls.

---

## Further reading

- Ole-Christoffer Granmo, [*An Introduction to Tsetlin Machines*](https://tsetlinmachine.org/) (free online). Ch. 1, §1.3 (graded memory, Memorize and Forget Value, randomized updates; the "Tsetlin Machine Origins" box) and Ch. 2, §2.1 (the single-literal Markov chain, Table 2.1 and the four scenarios). This lesson's figures are redrawn from these sections.
- Ole-Christoffer Granmo, [*The Tsetlin Machine: A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic*](https://arxiv.org/abs/1804.01508) (2018). The original Tsetlin machine paper, in research-paper terms (Type I/II feedback, include/exclude, specificity $s$).
- M. L. Tsetlin, *Automaton Theory and Modeling of Biological Systems* (Academic Press, 1973). English translation of Tsetlin's work on automata in random environments.
- Kumpati Narendra & Mandayam Thathachar, *Learning Automata: An Introduction* (1989). The textbook treatment of the Tsetlin automaton and its relatives, including when they are guaranteed to find the better action.

**Next lessons:** [Recognize, Erase & Reject feedback](../tsetlin-machine-feedback/index.md)

<script src="viz/tsetlin-automaton.js"></script>
