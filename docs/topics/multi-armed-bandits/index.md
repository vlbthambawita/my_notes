# Multi-Armed Bandits

!!! abstract "TL;DR"
    A **multi-armed bandit** is the simplest reinforcement-learning problem: there is only one situation, and you repeatedly choose one of $k$ actions ("arms"), each paying a random reward from its own hidden distribution. To do well you must **explore** (try arms to learn what they are worth) and **exploit** (pull the arm that looks best). This lesson builds the tools for that trade-off: running **value estimates**, the update rule *new estimate = old + step × error*, and four ways to explore (**ε-greedy**, **optimistic initial values**, **UCB** and **Thompson sampling**), compared by their **regret**.

**Level:** beginner · **Prerequisites:** [What is RL?](../reinforcement-learning-intro/index.md) (agent, action, reward, ε-greedy). Knowing what an average and a square root are is enough for the math.

---

## Why should I care?

Many real decisions are bandit problems:

- **A/B testing on websites.** Which of three headlines gets the most clicks? Every visitor you show a weak headline to is a lost click.
- **Clinical trials.** Which of several treatments works best? Every patient given the worse treatment pays a real price, so you want to shift patients to the better one as soon as the evidence allows.
- **Recommendations and ads.** Which song, article or ad should you show next? You learn from every click, while still trying new items.

Bandits are also the cleanest place to study **exploration versus exploitation**, which lesson 1 only teased. In a bandit there are no states that change and no delayed rewards, so exploration is the *only* hard part. The tools you build here (value estimates, the incremental update, ε-greedy, optimism) are reused in every later lesson of this series.

---

## Step 1 — The k-armed bandit: one state, k actions

Picture a row of slot machines (old slang: "one-armed bandits", because they take your money). Each machine pays out a random amount when you pull its arm. Some machines are better on average than others, but you don't know which. You have a fixed number of pulls. How do you collect the most money?

That is the **k-armed bandit** problem:

- You have $k$ possible actions (arms).
- Each time step $t$, you pick one arm $a_t$ and receive a reward $r_{t+1}$.
- Each arm's reward is random, drawn from that arm's own distribution, which never changes and which you cannot see.
- Your goal: the largest total reward over many pulls.

In the language of [lesson 1](../reinforcement-learning-intro/index.md#step-2-the-loop-agent-and-environment), the agent–environment loop still runs, but the **state never changes**. Your action affects only the reward you get right now, not the situation you will be in next. That removes the credit-assignment problem completely and leaves the pure question: *which arm is best, and how much should I spend finding out?*

| Problem | Arms | Reward |
|---|---|---|
| Slot machines | the machines | the payout |
| A/B test | page variants A, B, C | 1 if the visitor clicks, else 0 |
| Clinical trial | treatments | 1 if the patient recovers, else 0 |
| News site | candidate headlines | 1 if clicked, else 0 |

**Try it:** five slot machines, 25 pulls. Each pull pays the arm's hidden value plus random noise, so one lucky or unlucky pull can mislead you. When your pulls run out, the true values are revealed.

<div id="viz-play" class="viz"><div class="viz-title">🎰 Interactive: play a 5-armed bandit</div></div>

!!! tip "What to notice"
    - **One pull tells you little.** The noise is as large as the differences between arms, so a single good reward can make a mediocre arm look great.
    - **You felt the dilemma.** Every pull spent testing a new arm is a pull not spent on your current favourite. Stick with your favourite too early and you may never find the real best arm.
    - **Play a few games.** Did you always find the best arm (★ after the reveal)? The *regret* in the readout counts how much expected reward your choices cost you compared with always pulling the best arm. Step 7 makes this the yardstick for every method.

---

## Step 2 — Action values: what is each arm worth?

The **true value** of an arm is the reward it pays **on average**:

$$
q_*(a) = \mathbb{E}\left[\, r_{t+1} \mid a_t = a \,\right]
$$

This is the bandit version of the Q-value from lesson 1. With only one state, "how good is action $a$" doesn't depend on where you are. If you knew $q_*(a)$ for every arm, the problem would be trivial: always pull the arm with the largest $q_*$. But you don't, so you **estimate** it.

The most natural estimate is the **sample average**: add up the rewards you have received from arm $a$ and divide by the number of times you pulled it.

$$
Q_t(a) = \frac{\text{sum of rewards from } a \text{ so far}}{N_t(a)}
$$

where $N_t(a)$ is how many times arm $a$ has been pulled before step $t$. This is exactly the orange bar in the widget above. By the law of large numbers, $Q_t(a) \to q_*(a)$ as $N_t(a)$ grows: more pulls, better estimate.

The **greedy action** is the arm with the highest estimate right now, $\arg\max_a Q_t(a)$. Choosing it is **exploiting**; choosing anything else is **exploring**.

!!! tip "What to notice"
    - The estimate of an arm only improves **when you pull that arm**. An arm you ignore keeps whatever estimate (good or bad) it had.
    - So a greedy agent can get stuck: if the best arm happens to give a poor first reward, its estimate stays low, and the agent never comes back to correct it.

---

## Step 3 — Learning incrementally: new estimate = old estimate + step × error

Storing every reward to recompute the average is wasteful. There is a neater way. Say arm $a$ has given rewards $r_1, \dots, r_n$ (numbered by pulls *of this arm*), and $Q_n$ is the average of the first $n-1$ of them. Then the average of all $n$ is

$$
Q_{n+1} = \frac{1}{n}\sum_{i=1}^{n} r_i = \frac{1}{n}\Big(r_n + (n-1)\,Q_n\Big) = Q_n + \frac{1}{n}\Big(r_n - Q_n\Big)
$$

Read the last form out loud:

$$
\text{new estimate} \;=\; \text{old estimate} \;+\; \text{step size} \times \underbrace{(\text{target} - \text{old estimate})}_{\text{error}}
$$

You only need to remember $Q$ and $N$ for each arm. Each new reward nudges the estimate towards itself, by a step that shrinks as evidence accumulates.

<div class="viz diagram">
<script type="application/json">
{
  "title": "🔁 Updating an estimate, one reward at a time",
  "cols": 3,
  "rows": 2,
  "rowHeight": 100,
  "nodes": [
    {"id": "init", "col": 0, "row": 0, "icon": "🆕", "label": "Start: Q(a) = 0, N(a) = 0", "shape": "pill", "note": "Before any pulls we know nothing. The starting value matters more than you might think (Step 5)."},
    {"id": "pull", "col": 1, "row": 0, "icon": "🎰", "label": "Pull arm a", "sub": "get reward r", "role": "data", "note": "The environment pays a random reward around the arm's hidden value q*(a)."},
    {"id": "err", "col": 2, "row": 0, "icon": "😮", "label": "Error: r − Q(a)", "sub": "how surprised am I?", "role": "grad", "note": "Positive: better than expected. Negative: worse than expected."},
    {"id": "size", "col": 2, "row": 1, "icon": "📐", "label": "Step size 1 / N(a)", "sub": "after N(a) ← N(a) + 1", "role": "model", "note": "Early rewards move the estimate a lot; later ones only a little. A constant α instead tracks a changing world."},
    {"id": "upd", "col": 1, "row": 1, "icon": "✏️", "label": "Q(a) ← Q(a) + step × error", "role": "loss", "note": "The new estimate moves part of the way towards the reward it just saw."}
  ],
  "edges": [
    {"from": "init", "to": "pull"},
    {"from": "pull", "to": "err"},
    {"from": "err", "to": "size"},
    {"from": "size", "to": "upd"},
    {"from": "upd", "to": "pull", "label": "next pull"}
  ],
  "steps": [
    {"show": ["init"], "text": "Every arm starts with an estimate Q(a) and a count N(a) of zero."},
    {"show": ["init>pull", "pull"], "text": "Pull an arm and receive a reward r: the <b>target</b> the estimate should move towards."},
    {"show": ["pull>err", "err"], "text": "Compare it with what you expected. The difference r − Q(a) is the <b>error</b>."},
    {"show": ["err>size", "size"], "text": "Count the pull and choose the <b>step size</b>: 1/N(a) gives the exact running average."},
    {"show": ["size>upd", "upd"], "text": "Move the estimate a fraction of the way towards the reward: <b>new = old + step × error</b>."},
    {"show": ["upd>pull", "pull"], "text": "Repeat with the next pull. The same pattern returns in TD learning, Q-learning and DQN."}
  ],
  "narrow": {"cols": 2, "rows": 3, "pos": {"init": [0, 0], "pull": [0, 1], "err": [1, 1], "size": [1, 2], "upd": [0, 2]}}
}
</script>
</div>

This pattern is everywhere in machine learning. Gradient descent ([lesson](../gradient-descent/index.md)) has the same shape, $w \leftarrow w - \eta \nabla L$: old value plus a step in the direction that reduces the error. In fact, $Q \leftarrow Q + \alpha (r - Q)$ *is* one gradient-descent step on the squared error $\tfrac{1}{2}(r - Q)^2$. In later lessons the target will change (it becomes "reward plus the value of the next state"), but the update keeps this form.

!!! note "When the world changes: a constant step size"
    With step $\tfrac{1}{n}$, the 1000th reward counts as much as the first. That is right if the arms never change. If they drift (a headline goes stale, a market moves), use a **constant step size** $\alpha$, say 0.1: $Q \leftarrow Q + \alpha (r - Q)$. Then recent rewards count more and old ones fade away exponentially, so the estimate keeps tracking the moving target. Most of deep RL uses a constant step size (the learning rate).

!!! tip "What to notice"
    - The update needs only two numbers per arm, $Q(a)$ and $N(a)$, however many pulls you make.
    - The **error** $r - Q$ is the learning signal. If rewards match expectations, the estimate stops moving.

---

## Step 4 — Greedy vs ε-greedy

Now we can build agents. The simplest is **greedy**: always pull the arm with the highest $Q$ (break ties at random). **ε-greedy** (from lesson 1) adds a little exploration: with probability $\varepsilon$ pick a random arm, otherwise act greedily.

To compare them fairly we use the classic **10-armed testbed** from Sutton & Barto's textbook:

1. Make a random bandit with 10 arms. Each arm's true value $q_*(a)$ is drawn from a normal distribution with mean 0 and standard deviation 1.
2. Each pull of arm $a$ pays $q_*(a)$ plus normal noise (standard deviation 1).
3. Let the agent play 1000 pulls, learning with sample averages.
4. Repeat on **2000 different random bandits** and average, so the curves show typical behaviour rather than luck.

The widget runs all of this in your browser (with a fixed random seed, so everyone sees the same curves). Move the slider to try your own $\varepsilon$.

<div id="viz-testbed-eps" class="viz"><div class="viz-title">📈 Interactive: greedy vs ε-greedy on the 10-armed testbed</div></div>

!!! tip "What to notice"
    - **Greedy gets stuck.** It improves fast at first, then flattens: over the last 100 steps it picks the best arm only **34.4%** of the time (average reward 1.004). In most runs it locked onto a decent-looking arm early and never checked the others again.
    - **A little exploration pays off.** ε = 0.1 reaches **80.0%** optimal (reward 1.361), and ε = 0.01 reaches **59.1%** (reward 1.299). For comparison, always pulling the best arm would earn 1.532 on average on these bandits.
    - **ε = 0.1 has a ceiling.** Even with perfect estimates it still explores 10% of the time, so it can pick the best arm at most $90\% + 10\% \times \tfrac{1}{10} = 91\%$ of the time.
    - **ε = 0.01 is slower but ends higher.** Run both for 10,000 steps (we did, in Python) and ε = 0.01 overtakes: average reward 1.501 vs 1.378 over the last 100 steps. Try ε = 0.3 or 0.5 with the slider: too much exploration wastes pulls on arms you already know are bad.

---

## Step 5 — Optimistic initial values

Here is a trick that makes even a greedy agent explore. Instead of starting every estimate at 0, start it **way too high**, say $Q_1(a) = 5$, when real values are around 0 to 2.

Now every arm looks wonderful until it is tried. The greedy agent pulls the arm with the highest estimate, gets a reward around 1, and its estimate drops: *disappointing!* So the agent moves on to another arm that still looks wonderful. It ends up trying every arm, several times, before settling down. The exploration comes from **optimism**, not randomness.

Like Sutton & Barto, this agent uses a constant step size $\alpha = 0.1$. (With step $\tfrac{1}{n}$, the very first reward would replace the 5 completely, so each arm would be tried only once. In our Python check that version reached only 69.2% optimal.)

<div id="viz-testbed-opt" class="viz"><div class="viz-title">🌈 Interactive: optimistic initial values</div></div>

!!! tip "What to notice"
    - **The early spike.** For the first 10 pulls the optimistic agent tries the 10 arms one after another, so it picks the best arm only about 10% of the time. At step 11 it returns to the arm whose estimate fell the least, often the best one, and jumps to **41.3%** optimal. The next pulls wobble as the optimism wears off.
    - **Then it beats ε-greedy.** Over the last 100 steps it picks the best arm **85.9%** of the time (reward 1.495), without ever acting randomly.
    - **Turn the dial.** With $Q_1 = 0$ it is just the greedy agent again (with α = 0.1). Bigger values explore for longer before settling.
    - **The catch:** optimism only drives exploration **at the start**. If the arms change later, the agent has no reason to explore again.

---

## Step 6 — Upper confidence bounds (UCB): explore where you are unsure

ε-greedy explores **blindly**: when it explores, a hopeless arm it has tried 300 times is as likely as a promising arm it has tried twice. Smarter: explore the arms you are **unsure** about, because they might be better than they look.

**Upper Confidence Bound** (UCB) action selection gives each arm a bonus that is large when the arm has been tried rarely:

$$
a_t = \arg\max_a \left[\, Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \,\right]
$$

- $Q_t(a)$: what the arm seems to be worth (exploitation).
- $c\sqrt{\ln t / N_t(a)}$: an **uncertainty bonus** (exploration). It shrinks as you pull the arm more ($N_t(a)$ grows) and grows slowly for arms you neglect ($\ln t$ keeps rising).
- $c$: how optimistic to be. $c = 0$ is plain greedy.
- An arm never tried has $N_t(a) = 0$ and an infinite bonus, so UCB tries every arm once first.

The idea is called **optimism in the face of uncertainty**: act as if each arm is as good as it *plausibly* could be. Either the arm really is good (you win), or it isn't and pulling it shrinks its bonus (you learn).

<div id="viz-ucb" class="viz"><div class="viz-title">🎯 Interactive: watch UCB choose</div></div>

!!! tip "What to notice"
    - **Bars shrink with pulls.** After a few dozen steps, the arm you pull most has a short bar (you are sure about it), while rarely pulled arms keep tall bars.
    - **Neglected arms come back.** Watch an arm with a low estimate: its top slowly rises (the $\ln t$ term) until it is checked again. UCB never stops exploring completely, but it explores less and less often.
    - **c sets the balance.** With $c = 0$ the bars vanish and UCB is greedy. With $c = 3$ it keeps re-checking bad arms for a long time.

On the testbed, UCB with $c = 2$ picks the best arm **84.9%** of the time over the last 100 steps (reward 1.479), and like the optimistic agent it jumps to **43.6%** at step 11, right after its round of trying each arm once.

!!! note "Thompson sampling: explore by sampling your beliefs"
    A third idea, from 1933 and very popular in industry: keep a **belief** about each arm's value, a probability distribution that is wide when you have few pulls and narrow when you have many. Each step, **draw one random sample** from each arm's belief and pull the arm with the highest sample. An uncertain arm sometimes draws a high sample and gets tried; a well-known bad arm almost never does. As the beliefs narrow, the agent naturally shifts to exploiting.

    In our testbed the belief for an arm is a normal distribution with mean $\frac{\text{sum of rewards}}{N(a) + 1}$ and standard deviation $\frac{1}{\sqrt{N(a)+1}}$ (Bayes' rule, starting from the belief that values look like $\mathcal{N}(0, 1)$). It reaches **91.3%** optimal over the last 100 steps. It has a head start here, because that starting belief matches exactly how the testbed creates its arms.

---

## Step 7 — Regret: the yardstick for exploration

How should we compare these agents? "Average reward at the end" ignores how much the agent lost **while learning**. The standard yardstick is **regret**: the reward you expect to lose compared with someone who knew the best arm $a^*$ from the start.

$$
\text{Regret}_T = \sum_{t=1}^{T} \Big( q_*(a^*) - q_*(a_t) \Big)
$$

Each step you pick the best arm costs nothing; each step you pick another arm costs the gap between their true values. Lower is better, and zero is impossible (you have to try arms to learn which is best).

How regret **grows** tells you the most:

- **Greedy:** if it locks onto the wrong arm, it pays the same gap every step forever. Regret grows **linearly** in $T$.
- **ε-greedy (fixed ε):** it keeps exploring at the same rate forever, so it also grows **linearly**, just with a smaller slope.
- **UCB and Thompson sampling:** they explore less and less as they become sure, so regret grows only like $\log T$. A classic result by Lai and Robbins (1985) shows that, up to a constant, $\log T$ is the best any method can do.

<div id="viz-testbed-all" class="viz"><div class="viz-title">🏁 Interactive: every method, measured by regret</div></div>

!!! tip "What to notice"
    - **Total regret after 1000 pulls** (2000 runs, seed 1): greedy **535.1**, ε-greedy 0.1 **229.8**, optimistic **242.3**, UCB (c = 2) **148.2**, Thompson **69.7**. For scale, a perfect agent would collect about 1532 in total.
    - **Look at the slopes on the right.** Greedy and ε-greedy keep climbing in a straight line: over the last 100 steps they still lose 0.527 and 0.169 per step. UCB and Thompson bend over and flatten: by then they lose only 0.050 and 0.017 per step.
    - **Optimistic vs ε-greedy:** the optimistic agent's early exploration is expensive (it loses 0.581 per step during steps 100–200, against 0.287 for ε-greedy), so its total regret ends slightly higher. But its curve flattens (0.036 per step at the end), so it would pull ahead in a longer run.
    - **Tune c.** Set UCB's $c$ to 1: its regret drops to **59.1** and it picks the best arm 91.7% of the time. The best knob setting depends on the problem, which is why theory-backed defaults matter.
    - **Change the seed or the number of runs.** With 200 runs the curves get noisier, but the ranking stays the same (we checked seeds 2, 3 and 7 in Python).

---

## Putting it all together

Every bandit agent is the same loop. They differ only in one box: **how to choose the arm**.

<div class="viz diagram">
<script type="application/json">
{
  "title": "🧩 A bandit agent, start to finish",
  "cols": 3,
  "rows": 2,
  "rowHeight": 110,
  "nodes": [
    {"id": "start", "col": 0, "row": 0, "icon": "🆕", "label": "Set Q(a), N(a) for every arm", "shape": "pill", "sub": "0, or optimistic"},
    {"id": "choose", "col": 1, "row": 0, "icon": "🧭", "label": "Choose an arm", "sub": "the exploration rule", "role": "model", "note": "Greedy, ε-greedy, optimistic greedy, UCB or Thompson sampling."},
    {"id": "pull", "col": 2, "row": 0, "icon": "🎰", "label": "Pull it, get reward r", "role": "data"},
    {"id": "upd", "col": 2, "row": 1, "icon": "✏️", "label": "N(a) += 1, Q(a) += step × (r − Q(a))", "role": "loss"},
    {"id": "done", "col": 1, "row": 1, "label": "Out of pulls?", "shape": "decision", "role": "grad"},
    {"id": "end", "col": 0, "row": 1, "icon": "🏁", "label": "Total reward, regret", "shape": "pill", "role": "opt"}
  ],
  "edges": [
    {"from": "start", "to": "choose"},
    {"from": "choose", "to": "pull"},
    {"from": "pull", "to": "upd"},
    {"from": "upd", "to": "done"},
    {"from": "done", "to": "choose", "label": "no"},
    {"from": "done", "to": "end", "label": "yes"}
  ],
  "steps": [
    {"show": ["start"], "text": "Give every arm a starting estimate and a count. Starting high (optimistic values) already builds in exploration."},
    {"show": ["start>choose", "choose"], "text": "<b>Greedy:</b> pick the highest Q. <b>ε-greedy:</b> the same, but a random arm with probability ε."},
    {"show": ["choose"], "text": "<b>UCB:</b> pick the highest Q + c√(ln t / N). <b>Thompson:</b> sample a guess from each arm's belief and pick the highest guess."},
    {"show": ["choose>pull", "pull"], "text": "Pull the chosen arm. The environment pays a noisy reward."},
    {"show": ["pull>upd", "upd"], "text": "Update only that arm: new estimate = old + step × error."},
    {"show": ["upd>done", "done", "done>choose", "choose"], "text": "Still have pulls left? Choose again with the improved estimates."},
    {"show": ["done>end", "end"], "text": "At the end, judge the agent by its total reward, or equivalently its <b>regret</b>."}
  ],
  "narrow": {"cols": 2, "rows": 4, "rowHeight": 95, "pos": {"start": [0, 0], "choose": [0, 1], "pull": [1, 1], "upd": [1, 2], "done": [0, 2], "end": [0, 3]}}
}
</script>
</div>

| Method | How it explores | Knob | Regret on our testbed (1000 pulls) |
|---|---|---|---|
| Greedy | never | none | 535.1 |
| ε-greedy | at random, at a fixed rate | ε | 229.8 (ε = 0.1) |
| Optimistic greedy | every arm looks great until tried | starting value $Q_1$ | 242.3 ($Q_1 = 5$) |
| UCB | arms with few pulls get a bonus | c | 148.2 (c = 2) |
| Thompson sampling | sample from beliefs, pick the best sample | prior belief | 69.7 |

Here is the whole testbed in numpy. To keep it fast it plays all 2000 bandits at once: each row of `Q` is one bandit.

```python
import numpy as np


def run_testbed(choose, runs=2000, steps=1000, k=10, q0=0.0, alpha=None, seed=0):
    """Play `runs` random k-armed bandits at once (one per row). Returns % optimal and total regret."""
    rng = np.random.default_rng(seed)
    q_star = rng.normal(0, 1, (runs, k))            # true action values, hidden from the agent
    best = q_star.argmax(axis=1)
    Q = np.full((runs, k), q0)                       # estimates
    N = np.zeros((runs, k))                          # how often each arm was pulled
    rows = np.arange(runs)
    optimal, regret = [], 0.0
    for t in range(1, steps + 1):
        a = choose(Q, N, t, rng)                     # 1. pick an arm in every run
        r = rng.normal(q_star[rows, a], 1)           # 2. pull it: noisy reward
        N[rows, a] += 1                              # 3. update the estimate:
        step = alpha if alpha else 1 / N[rows, a]    #    new = old + step × (reward − old)
        Q[rows, a] += step * (r - Q[rows, a])
        optimal.append(np.mean(a == best))
        regret += np.mean(q_star[rows, best] - q_star[rows, a])
    return 100 * np.mean(optimal[-100:]), regret


def argmax_random_ties(values, rng):
    noise = rng.random(values.shape) * 1e-9          # tiny noise breaks ties at random
    return np.argmax(values + noise, axis=1)


def epsilon_greedy(eps):
    def choose(Q, N, t, rng):
        greedy = argmax_random_ties(Q, rng)
        random = rng.integers(0, Q.shape[1], len(Q))
        return np.where(rng.random(len(Q)) < eps, random, greedy)
    return choose


def ucb(c):
    def choose(Q, N, t, rng):
        tried = N > 0
        bonus = np.where(tried, c * np.sqrt(np.log(t) / np.maximum(N, 1)), np.inf)  # untried arms first
        return argmax_random_ties(Q + bonus, rng)
    return choose


for name, kwargs in [("greedy", dict(choose=epsilon_greedy(0.0))),
                     ("ε-greedy 0.01", dict(choose=epsilon_greedy(0.01))),
                     ("ε-greedy 0.1", dict(choose=epsilon_greedy(0.1))),
                     ("optimistic Q0=5", dict(choose=epsilon_greedy(0.0), q0=5.0, alpha=0.1)),
                     ("UCB c=2", dict(choose=ucb(2.0)))]:
    pct, regret = run_testbed(**kwargs)
    print(f"{name:16s} optimal action (last 100 steps) {pct:5.1f}%   total regret {regret:6.1f}")
```

```text
greedy           optimal action (last 100 steps)  34.7%   total regret  521.5
ε-greedy 0.01    optimal action (last 100 steps)  58.4%   total regret  357.8
ε-greedy 0.1     optimal action (last 100 steps)  80.0%   total regret  233.4
optimistic Q0=5  optimal action (last 100 steps)  85.6%   total regret  240.9
UCB c=2          optimal action (last 100 steps)  85.4%   total regret  148.3
```

The numbers differ a little from the widgets because numpy draws different random bandits, but the ranking and the sizes are the same.

!!! note "From bandits to full reinforcement learning"
    - In a **contextual bandit**, the agent also sees a *context* before choosing (the user's profile, the patient's symptoms), and the best arm depends on it. This is how most real recommendation and ad systems work.
    - In **full RL**, the action also changes **what happens next**: the next state, and so the future rewards. Then "how good is this action" must include everything that follows. Describing that properly is the job of the next lesson, Markov decision processes and the Bellman equations.

---

## Common misconceptions

!!! warning "\"Exploration is wasted reward.\""
    Exploration is an **investment**. Greedy never "wastes" a pull, yet it had by far the worst regret on the testbed (535.1, against 229.8 for ε = 0.1), because it never found the best arm in most runs.

!!! warning "\"More exploration is always safer.\""
    Exploring costs reward every time. On the testbed, ε = 0.2 picked the best arm less often than ε = 0.1 (73.8% vs 80.0% over the last 100 steps) and had higher regret (352.6 vs 229.8), because it kept pulling arms it already knew were bad.

!!! warning "\"The arm with the highest average so far is the best arm.\""
    Only if you have pulled every arm often enough. After a few pulls, averages are dominated by noise. That is exactly why UCB adds a bonus that depends on the **number of pulls**, not just the average.

!!! warning "\"UCB prefers risky arms.\""
    UCB's bonus is about **how little you know** (few pulls), not about how variable an arm's payouts are. An arm with wildly varying rewards that you have pulled 500 times gets a small bonus.

---

## Check yourself

??? question "1. An arm's estimate is Q = 2.0 after 4 pulls. The 5th pull pays 4.5. What is the new sample-average estimate?"
    $Q \leftarrow 2.0 + \tfrac{1}{5}(4.5 - 2.0) = 2.0 + 0.5 = $ **2.5**. (Check: the 4 earlier rewards sum to 8.0; $(8.0 + 4.5)/5 = 2.5$.)

??? question "2. With ε = 0.1 and 10 arms, what is the highest possible % of optimal actions, even with perfect estimates?"
    With probability 0.9 it acts greedily (the best arm, once the estimates are right). With probability 0.1 it picks a random arm, which is the best one with probability $\tfrac{1}{10}$. Total: $0.9 + 0.1 \times 0.1 = 0.91$, so **91%**.

??? question "3. UCB with c = 2 at step t = 200. Arm A: Q = 1.0 after 100 pulls. Arm B: Q = 0.8 after 4 pulls. Which does it pick?"
    $\ln 200 \approx 5.30$. Arm A: $1.0 + 2\sqrt{5.30/100} \approx 1.0 + 0.46 = 1.46$. Arm B: $0.8 + 2\sqrt{5.30/4} \approx 0.8 + 2.30 = 3.10$. UCB picks **arm B**: its estimate is lower, but it is so uncertain that it might well be better.

??? question "4. Why do optimistic initial values not help when the arms' values change over time?"
    Optimism makes the agent explore only **at the beginning**, until the inflated estimates have come down. After that the agent is plain greedy. If an arm later becomes better, nothing pushes the agent to try it again. ε-greedy (with a constant step size) or UCB keep exploring and can notice the change.

??? question "5. A clinical trial gives half the patients treatment A and half treatment B, then picks the winner. A bandit algorithm shifts patients towards the better-looking treatment as results arrive. Which has lower regret, and why?"
    Usually the **bandit**. The fixed 50/50 split keeps giving the worse treatment to half the patients until the trial ends, so its regret grows linearly during the trial. A bandit (UCB, Thompson) gives the worse treatment less and less often as the evidence accumulates. The price is that the final comparison is less balanced, which is why medical trials use bandits only with careful statistical design.

---

## Further reading

- Richard Sutton & Andrew Barto, [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) (2nd ed., free online), Chapter 2. The 10-armed testbed comes from here.
- Tor Lattimore & Csaba Szepesvári, [*Bandit Algorithms*](https://tor-lattimore.com/downloads/book/book.pdf) (free online). The thorough, mathematical treatment, including regret bounds.
- Daniel Russo et al., [*A Tutorial on Thompson Sampling*](https://arxiv.org/abs/1707.02038) (2018). Clear and practical.
- Peter Auer, Nicolò Cesa-Bianchi & Paul Fischer, [*Finite-time Analysis of the Multiarmed Bandit Problem*](https://link.springer.com/article/10.1023/A:1013689704352) (2002). The paper that introduced UCB1.
- David Silver, [*UCL Course on RL*](https://www.davidsilver.uk/teaching/), Lecture 9: Exploration and Exploitation.

**Next lessons:** [MDPs & Bellman equations](../mdp-bellman-equations/index.md) · [Dynamic programming](../dynamic-programming/index.md)

<script src="viz/multi-armed-bandits.js"></script>
