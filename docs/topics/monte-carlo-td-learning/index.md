# Monte Carlo and Temporal-Difference Learning

!!! abstract "TL;DR"
    Dynamic programming needs a **model** of the world. Real agents usually don't have one, so they learn values **from experience**. **Monte Carlo (MC)** plays a whole episode, then moves each state's value toward the **return it actually got**. **Temporal-difference (TD) learning** doesn't wait: after every single step it moves $V(s)$ toward $r + \gamma V(s')$, a target that uses its *own guess* for the next state. TD learns "a guess from a guess", and that usually makes it learn faster.

**Level:** beginner–intermediate · **Prerequisites:** [What is RL?](../reinforcement-learning-intro/index.md) (returns, policies, values) and [Dynamic programming](../dynamic-programming/index.md) (policy evaluation with the Bellman equation)

---

## Why should I care?

In the [dynamic programming lesson](../dynamic-programming/index.md) we computed the value of every state by sweeping over the Bellman equation. That worked because we knew the rules: for every state and action, the exact probability of every next state and reward, $p(s', r \mid s, a)$.

Almost no real problem hands you that table. Nobody can write down the transition probabilities of a robot arm, a game of Go against an unknown opponent, or a user chatting with an assistant. What you *can* do is **try things and watch what happens**.

This lesson is about learning values from that raw experience. The two ideas in it, Monte Carlo and TD learning, are the foundation of almost everything that follows: Q-learning, SARSA and DQN are TD methods, and REINFORCE is a Monte Carlo method. In this lesson we only **predict** (estimate $V^\pi$ for a fixed policy $\pi$). Using these estimates to *improve* the policy (control) is the next lesson.

---

## Step 1 — Learning without a model

Dynamic programming and the methods in this lesson want the same thing: the value function $V^\pi(s)$ from [lesson 1](../reinforcement-learning-intro/index.md#step-5-how-good-is-a-state-the-value-function). They differ in **what they need**.

<div class="viz diagram">
<script type="application/json">
{
  "title": "🧭 Two roads to the value function",
  "cols": 3,
  "rows": 2,
  "rowHeight": 100,
  "nodes": [
    {"id": "model", "col": 0, "row": 0, "icon": "📖", "label": "A model", "sub": "p(s′, r | s, a)", "note": "The full rulebook: every possible next state and reward, with its probability."},
    {"id": "dp", "col": 1, "row": 0, "icon": "🧮", "label": "Dynamic programming", "sub": "sweep over all states", "note": "Policy evaluation: repeat V(s) ← Σ p · [r + γV(s′)] for every state until nothing changes."},
    {"id": "v", "col": 2, "row": 0, "icon": "📊", "label": "<b>Values V<sup>π</sup>(s)</b>", "role": "loss"},
    {"id": "env", "col": 0, "row": 1, "icon": "🌍", "label": "<b>Environment</b>", "sub": "a black box you can only try", "role": "data"},
    {"id": "eps", "col": 1, "row": 1, "icon": "🎬", "label": "<b>Episodes</b>", "sub": "s, r, s′, r, s″, …", "role": "data", "note": "Experience: what actually happened when the agent followed its policy."},
    {"id": "mctd", "col": 2, "row": 1, "icon": "🧠", "label": "<b>Monte Carlo / TD</b>", "sub": "average what happened", "role": "model"}
  ],
  "edges": [
    {"from": "model", "to": "dp"},
    {"from": "dp", "to": "v"},
    {"from": "env", "to": "eps", "label": "play"},
    {"from": "eps", "to": "mctd"},
    {"from": "mctd", "to": "v"}
  ],
  "steps": [
    {"show": ["model"], "text": "Dynamic programming starts from a <b>model</b>: the exact probability of every next state and reward."},
    {"show": ["model>dp", "dp", "dp>v", "v"], "text": "With the model, it computes the values by sweeping the Bellman equation over every state. No experience needed, but no model, no DP."},
    {"show": ["env", "env>eps", "eps"], "text": "Model-free methods only need to <b>interact</b>: follow the policy, and record what happens."},
    {"show": ["eps>mctd", "mctd", "mctd>v", "v"], "text": "Monte Carlo and TD turn that experience into the <b>same</b> value function, by averaging over what actually happened."}
  ],
  "narrow": {"cols": 2, "rows": 3, "pos": {"model": [0, 0], "dp": [1, 0], "v": [1, 1], "env": [0, 1], "eps": [0, 2], "mctd": [1, 2]}}
}
</script>
</div>

Remember the definition of value: $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$, the **expected return** from $s$. An expectation is just an average over many tries. So if we can't compute the average from a model, we can **estimate** it from samples, exactly like the "Run 1000 episodes" button in lesson 1 did. This is what *model-free* means: the agent never learns or uses $p(s', r \mid s, a)$.

!!! tip "What to notice"
    Both roads end at the same box. Model-free methods are not "worse DP". They answer the same question with a different ingredient: **samples** instead of **probabilities**.

---

## Step 2 — Monte Carlo: average the returns you actually got

**Monte Carlo** methods are named after the casino: they estimate an average by sampling. The recipe for estimating $V^\pi(s)$:

1. Follow $\pi$ until the episode ends.
2. For each state $s$ you visited, compute the return $G_t$ that followed (the discounted sum of rewards from that point to the end).
3. $V(s)$ is the average of all the returns you have seen from $s$.

If a state is visited twice in one episode, **first-visit MC** only uses the return after the first visit. (*Every-visit MC* uses all of them; both work.)

Keeping a list of every return is wasteful. The same average can be updated one return at a time. If $N(s)$ counts the returns seen so far from $s$:

$$
V(s) \leftarrow V(s) + \frac{1}{N(s)}\,\big(G_t - V(s)\big)
$$

Read it as: *nudge the estimate toward the new sample, by a fraction of the gap*. In practice we often use a **constant step size** $\alpha$ (say 0.1) instead of $1/N(s)$:

$$
V(s) \leftarrow V(s) + \alpha\,\big(G_t - V(s)\big)
$$

A constant $\alpha$ forgets old returns slowly, which helps when the policy (and so the true values) keep changing, as they will in the next lesson.

**The test problem: a random walk.** Five states A–E sit in a row. Every episode starts in C and steps left or right with probability ½ each. Leaving on the right gives reward +1; leaving on the left gives 0; every other step gives 0. With $\gamma = 1$, the value of a state is just *the probability of leaving on the right*, which works out to $V^\pi(A), \dots, V^\pi(E) = \tfrac16, \tfrac26, \tfrac36, \tfrac46, \tfrac56$. Knowing the true answer lets us measure exactly how well each method learns.

<div id="viz-walk-mc" class="viz"><div class="viz-title">🎲 Interactive: Monte Carlo on the random walk</div></div>

!!! tip "What to notice"
    - While the episode is running, **nothing changes**. MC cannot update until it knows the return, and it only knows the return at the end.
    - When the episode ends, **every** state visited moves toward the same number (0 or 1 here). One lucky episode that leaves on the right drags A up toward 1, even though A's true value is only $\tfrac16$.
    - Press **Run 10 episodes** a few times. The bars do approach the green dashes, but they keep jumping around: each return is a noisy sample.

---

## Step 3 — TD(0): learn a guess from a guess

Here is the key observation. The return obeys a one-step relation (the Bellman equation from the [MDP lesson](../mdp-bellman-equations/index.md)):

$$
G_t = r_{t+1} + \gamma\, G_{t+1}
\qquad\text{so}\qquad
V^\pi(s_t) = \mathbb{E}_\pi\big[\,r_{t+1} + \gamma\, V^\pi(s_{t+1})\,\big]
$$

So instead of waiting for the real $G_t$, we can plug in our **current estimate** of what comes after: $r_{t+1} + \gamma V(s_{t+1})$. That gives **TD(0)**, the simplest temporal-difference method:

$$
V(s_t) \leftarrow V(s_t) + \alpha\,\big[\underbrace{r_{t+1} + \gamma V(s_{t+1})}_{\text{TD target}} - V(s_t)\big]
$$

The bracket has its own name, the **TD error**:

$$
\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
$$

$\delta_t$ is the *surprise* of one step: how much better or worse things look one step later than they did a moment ago. Positive $\delta_t$ → $s_t$ was better than we thought, so raise $V(s_t)$. A terminal state has value 0, so on the final step the target is just $r_{t+1}$.

Updating an estimate from another estimate is called **bootstrapping** ("pulling yourself up by your own bootstraps"). DP bootstraps too: it also uses $V(s')$ in its target. TD combines the two worlds: it **samples** like Monte Carlo and **bootstraps** like DP.

<div id="viz-walk" class="viz"><div class="viz-title">⏱️ Interactive: TD updates every step, MC waits for the end</div></div>

!!! tip "What to notice"
    - **Reset values**, then step through the first episode. Every TD error is $0 + 0.5 - 0.5 = 0$ until the very last step: all states look equally good, so there is no surprise. Only the state next to the exit learns something in episode 1. MC, in contrast, moves *every* visited state.
    - In later episodes the information **spreads backwards** one step at a time: once E has a high value, a step from D into E produces a positive $\delta$ and raises D.
    - TD updates use $V(s')$, which is itself only a guess. Early on that guess is wrong, so TD's target is **biased**. But it depends on just one random step, so it jumps around much less than MC's target.

---

## Step 4 — MC vs TD: waiting for the truth vs trusting your guess

A small example makes the difference concrete. You drive home from work and keep predicting your **total travel time**:

| Moment | Minutes so far | Predicted time still to go | Predicted total |
|---|---|---|---|
| Leave the office | 0 | 30 | 30 |
| Reach the car, it's raining | 5 | 35 | 40 |
| Leave the highway, traffic was light | 20 | 15 | 35 |
| Stuck behind a slow truck | 30 | 10 | 40 |
| Turn into your street | 40 | 3 | 43 |
| Arrive home | 43 | 0 | 43 |

- **Monte Carlo** says: wait until you are home. The real total was 43, so every earlier prediction (30, 40, 35, 40) should move toward 43.
- **TD** says: don't wait. When you see the rain, your new prediction (40) already tells you that "30" was too optimistic, so update it toward 40 *right then*. Each prediction moves toward the **next** prediction.

TD is how people actually learn: you don't need to crash the car to learn that skidding was a bad sign. The trade-off, in one table:

| | Monte Carlo | TD(0) |
|---|---|---|
| Target | the real return $G_t$ | $r_{t+1} + \gamma V(s_{t+1})$ |
| Bias | **none**: $G_t$ is a true sample of $V^\pi(s_t)$ | **biased** while $V$ is still wrong |
| Variance | **high**: $G_t$ depends on every random step until the end | **low**: depends on one random step |
| When it can update | only at the end of an episode | after every step (online) |
| Needs episodes to end? | yes | no, works for never-ending tasks |
| Uses the Markov property? | no | yes (it trusts $V(s')$ to summarise the future) |

Neither is always better. But because TD's targets are much less noisy, it usually learns faster in practice, as the next step shows.

!!! tip "What to notice"
    "Bias" and "variance" here mean the same thing as in supervised learning. MC is an unbiased but noisy estimator; TD trades a little bias (which disappears as $V$ improves) for a big cut in noise.

---

## Step 5 — The race: MC vs TD on the random walk

Let's measure. Both methods start every value at 0.5 and see the same kind of episodes. After each episode we compute the **RMS error** between the estimates and the true values $\tfrac16, \dots, \tfrac56$ (the square root of the average squared error over the five states). One run is noisy, so the curves below average **100 runs** with fixed random seeds. At the start the RMS error is 0.236.

<div id="viz-race" class="viz"><div class="viz-title">🏁 Interactive: learning curves for MC and TD</div></div>

!!! tip "What to notice"
    - **TD learns much faster at first.** With the default settings (MC $\alpha = 0.02$, TD $\alpha = 0.1$), after 25 episodes TD's error is 0.056 while MC's is still 0.174.
    - **Click "MC 0.03 · TD 0.05".** TD with a small step size ends lowest of all: 0.035 after 100 episodes, against 0.056 for MC.
    - **The step size is a trade-off.** A large $\alpha$ learns fast but never settles: TD with $\alpha = 0.15$ reaches its lowest error, 0.055, around episode 25, then drifts back up to 0.069. With "both large" (MC 0.1, TD 0.3) TD's error is lowest early (0.073 after 10 episodes) and then *rises* to 0.108. With a constant $\alpha$, the estimates keep chasing each new noisy sample.
    - Compare the two at the **same** step size, $\alpha = 0.1$: TD's error bottoms out at 0.047 (around episode 45), MC's only at 0.081 (around episode 54). MC's targets are noisier, so its floor is higher.

---

## Step 6 — TD on the cliff world

The random walk is tiny. Now let TD estimate the values of a policy on the cliff world from [lesson 1](../reinforcement-learning-intro/index.md#step-2-the-loop-agent-and-environment), with $\gamma = 0.95$. The policy is fixed (the arrows); TD only *evaluates* it. In the [DP lesson](../dynamic-programming/index.md) we computed these values with the model. Here TD never sees the model, only episodes, and we can check whether it lands on the same numbers.

<div id="viz-cliff" class="viz"><div class="viz-title">🧗 Interactive: TD(0) evaluates a policy on the cliff world</div></div>

!!! tip "What to notice"
    - **TD finds the DP values without the model.** With *Edge of the cliff* and no slip, after 1000 episodes ($\alpha = 0.1$) TD's $V(\text{start})$ is 2.03, against the DP value of 2.05. After 2000 episodes the two agree to two decimals.
    - **Values spread backwards from the goal.** Press **Reset**, then **+100 episodes**: the two cells nearest the goal already match DP (10.0 and 8.5), while the top-left cells are still stuck near 0 or below. Each update only passes information one step back.
    - **On a slippery floor, a constant $\alpha$ keeps jittering.** Tick *slippery floor* (DP: $V(\text{start}) = -9.08$). After 1000 episodes TD says −9.12, but after 2000 it says −9.84: each slip is a random surprise, and with a fixed step size the estimate keeps reacting to it. Checked every 250 episodes between 1000 and 5000, the RMS error over all cells bounces between 0.59 and 2.05 and never settles to 0. The usual fix is a step size that shrinks over time.
    - The random walk and the cliff world tell the same story: TD learns quickly, and $\alpha$ decides how fast versus how precisely.

---

## Step 7 — Between the two: n-step TD and backup diagrams

MC and TD(0) are two ends of a dial. **n-step TD** looks $n$ real rewards ahead, then bootstraps:

$$
G_{t:t+n} = r_{t+1} + \gamma r_{t+2} + \dots + \gamma^{n-1} r_{t+n} + \gamma^n V(s_{t+n})
$$

$$
V(s_t) \leftarrow V(s_t) + \alpha\,\big[G_{t:t+n} - V(s_t)\big]
$$

With $n = 1$ this is TD(0). If $n$ reaches past the end of the episode, there is nothing left to bootstrap from and it becomes Monte Carlo. Values in between often work best. (**TD(λ)** averages over all $n$ at once; Sutton & Barto, Chapter 12, covers it.)

A **backup diagram** shows *what an update looks at*: how deep (one step, or all the way to the end) and how wide (every possible next state, or one sample).

<div class="viz diagram">
<script type="application/json">
{
  "title": "🌲 Backup diagrams: what each update looks at",
  "cols": 4,
  "rows": 4,
  "rowHeight": 80,
  "nodes": [
    {"id": "dp0", "col": 0, "row": 0, "label": "<b>DP</b>", "sub": "update V(s)", "role": "loss"},
    {"id": "dp1", "col": 0, "row": 1, "label": "every s′", "sub": "weighted by the model", "note": "DP looks one step ahead at ALL possible next states, using the model's probabilities, then uses V(s′)."},
    {"id": "td0", "col": 1, "row": 0, "label": "<b>TD(0)</b>", "sub": "update V(s)", "role": "loss"},
    {"id": "td1", "col": 1, "row": 1, "label": "one sampled s′", "sub": "target r + γV(s′)", "role": "data"},
    {"id": "n0", "col": 2, "row": 0, "label": "<b>2-step TD</b>", "sub": "update V(s)", "role": "loss"},
    {"id": "n1", "col": 2, "row": 1, "label": "sampled s′", "sub": "reward r<sub>1</sub>", "role": "data"},
    {"id": "n2", "col": 2, "row": 2, "label": "sampled s″", "sub": "r<sub>1</sub> + γr<sub>2</sub> + γ²V(s″)", "role": "data"},
    {"id": "mc0", "col": 3, "row": 0, "label": "<b>Monte Carlo</b>", "sub": "update V(s)", "role": "loss"},
    {"id": "mc1", "col": 3, "row": 1, "label": "sampled s′", "role": "data"},
    {"id": "mc2", "col": 3, "row": 2, "label": "⋯", "sub": "every step", "role": "data"},
    {"id": "mc3", "col": 3, "row": 3, "label": "episode end", "sub": "the real return G", "role": "opt", "shape": "pill"}
  ],
  "edges": [
    {"from": "dp0", "to": "dp1", "label": "all"},
    {"from": "td0", "to": "td1", "label": "1 sample"},
    {"from": "n0", "to": "n1"},
    {"from": "n1", "to": "n2"},
    {"from": "mc0", "to": "mc1"},
    {"from": "mc1", "to": "mc2"},
    {"from": "mc2", "to": "mc3"}
  ],
  "steps": [
    {"show": ["dp0", "dp0>dp1", "dp1"], "text": "<b>DP</b>: one step deep, but <b>full width</b>. It averages over every possible next state using the model, then bootstraps from V(s′)."},
    {"show": ["td0", "td0>td1", "td1"], "text": "<b>TD(0)</b>: one step deep and <b>one sample</b> wide. No model needed; it bootstraps from V(s′) after a single real step."},
    {"show": ["n0", "n0>n1", "n1", "n1>n2", "n2"], "text": "<b>n-step TD</b> (here n = 2): uses n real rewards, then bootstraps. Bigger n means less bias, more variance."},
    {"show": ["mc0", "mc0>mc1", "mc1", "mc1>mc2", "mc2", "mc2>mc3", "mc3"], "text": "<b>Monte Carlo</b>: one sample, <b>all the way to the end</b>. No bootstrapping at all: the target is the real return."},
    {"show": ["dp0", "td0", "n0", "mc0", "dp1", "td1", "n2", "mc3"], "text": "Left to right: from <b>bootstrapping with a model</b> to <b>pure sampling</b>. TD and n-step TD sit in the useful middle."}
  ],
  "narrow": {"cols": 2, "rows": 6.2, "rowHeight": 74, "pos": {"dp0": [0, 0], "dp1": [0, 1], "td0": [1, 0], "td1": [1, 1], "n0": [0, 2.2], "n1": [0, 3.2], "n2": [0, 4.2], "mc0": [1, 2.2], "mc1": [1, 3.2], "mc2": [1, 4.2], "mc3": [1, 5.2]}}
}
</script>
</div>

!!! tip "What to notice"
    Two questions classify every method in this course: **does it bootstrap** (use $V(s')$ in its target)? and **does it sample** (use real experience instead of a model)? DP: bootstraps, doesn't sample. MC: samples, doesn't bootstrap. TD: both.

---

## Putting it all together

Here is TD(0) prediction as a loop. Compare it with the agent loop from lesson 1: the "Learn" box is now one concrete line.

<div class="viz diagram">
<script type="application/json">
{
  "title": "🔁 TD(0) prediction, step by step",
  "cols": 3,
  "rows": 2,
  "rowHeight": 110,
  "nodes": [
    {"id": "reset", "col": 0, "row": 0, "icon": "🔁", "label": "Start an episode", "shape": "pill", "sub": "keep V from before"},
    {"id": "obs", "col": 1, "row": 0, "icon": "👀", "label": "In state s", "role": "data"},
    {"id": "pol", "col": 2, "row": 0, "icon": "🧭", "label": "Act with the fixed policy π", "role": "model"},
    {"id": "env", "col": 2, "row": 1, "icon": "🌍", "label": "Observe r and s′", "role": "data"},
    {"id": "learn", "col": 1, "row": 1, "icon": "🧠", "label": "<b>TD update</b>", "sub": "δ = r + γV(s′) − V(s), V(s) ← V(s) + αδ", "role": "loss", "note": "The whole TD(0) update. V(s′) = 0 if s′ is terminal."},
    {"id": "end", "col": 0, "row": 1, "label": "Episode over?", "shape": "decision", "role": "grad"}
  ],
  "edges": [
    {"from": "reset", "to": "obs"},
    {"from": "obs", "to": "pol"},
    {"from": "pol", "to": "env"},
    {"from": "env", "to": "learn"},
    {"from": "learn", "to": "end"},
    {"from": "end", "to": "obs", "label": "no"},
    {"from": "end", "to": "reset", "label": "yes"}
  ],
  "steps": [
    {"show": ["reset"], "text": "Start an episode. The value table V is kept from earlier episodes (at the very beginning, any guess, e.g. 0)."},
    {"show": ["reset>obs", "obs", "obs>pol", "pol"], "text": "In the current state, act with the policy we want to evaluate."},
    {"show": ["pol>env", "env"], "text": "The environment answers with a reward and the next state. No model needed: we just watch."},
    {"show": ["env>learn", "learn"], "text": "Compute the TD error δ and nudge V(s) toward r + γV(s′). This happens <b>every step</b>, not at the end."},
    {"show": ["learn>end", "end", "end>obs", "obs"], "text": "Not finished? Carry on from s′."},
    {"show": ["end>reset", "reset"], "text": "Finished? Start another episode. (Monte Carlo would do <b>all</b> its updates only here.)"}
  ],
  "narrow": {"cols": 2, "rows": 3, "pos": {"reset": [0, 0], "obs": [1, 0], "pol": [1, 1], "env": [1, 2], "learn": [0, 2], "end": [0, 1]}}
}
</script>
</div>

Here are both methods on the random walk in plain Python, with the same experiment as the learning-curve widget (100 runs of 100 episodes, values starting at 0.5):

```python
import random

# The random walk: states 1..5 are A..E, 0 and 6 are the two ends (terminal, V = 0).
TRUE_V = [None, 1/6, 2/6, 3/6, 4/6, 5/6]


def play_episode():
    """Start in C, step left or right at random. Returns a list of (s, r, s_next)."""
    s, steps = 3, []
    while 0 < s < 6:
        s_next = s + random.choice([-1, +1])
        r = 1.0 if s_next == 6 else 0.0          # +1 only for leaving on the right
        steps.append((s, r, s_next))
        s = s_next
    return steps


def mc_update(V, steps, alpha, gamma=1.0):
    """First-visit Monte Carlo: wait for the end, then move V(s) toward the real return G."""
    G, returns = 0.0, []
    for s, r, _ in reversed(steps):              # walk backwards: G_t = r_{t+1} + γ G_{t+1}
        G = r + gamma * G
        returns.append((s, G))
    returns.reverse()
    seen = set()
    for s, G in returns:
        if s not in seen:                        # only the first visit to s counts
            seen.add(s)
            V[s] += alpha * (G - V[s])


def td_update(V, steps, alpha, gamma=1.0):
    """TD(0): update after every step, toward r + γ V(s')."""
    for s, r, s_next in steps:                   # (in a live agent this runs inside the episode)
        delta = r + gamma * V[s_next] - V[s]     # the TD error
        V[s] += alpha * delta


def rms_error(V):
    return (sum((V[s] - TRUE_V[s]) ** 2 for s in range(1, 6)) / 5) ** 0.5


def experiment(update, alpha, runs=100, episodes=100):
    curve = [0.0] * (episodes + 1)               # average RMS error after each episode
    for _ in range(runs):
        V = [0.0] + [0.5] * 5 + [0.0]            # guesses start at 0.5; the ends stay 0
        curve[0] += rms_error(V) / runs
        for ep in range(1, episodes + 1):
            update(V, play_episode(), alpha)
            curve[ep] += rms_error(V) / runs
    return curve


random.seed(0)
for name, update, alpha in [("MC", mc_update, 0.02), ("MC", mc_update, 0.03),
                            ("TD", td_update, 0.05), ("TD", td_update, 0.1)]:
    c = experiment(update, alpha)
    print(f"{name} α={alpha:<4}  RMS error after 10 episodes: {c[10]:.3f}   25: {c[25]:.3f}   100: {c[100]:.3f}")
```

```text
MC α=0.02  RMS error after 10 episodes: 0.210   25: 0.176   100: 0.079
MC α=0.03  RMS error after 10 episodes: 0.197   25: 0.151   100: 0.059
TD α=0.05  RMS error after 10 episodes: 0.174   25: 0.110   100: 0.038
TD α=0.1   RMS error after 10 episodes: 0.128   25: 0.060   100: 0.056
```

Python's random numbers differ from the widget's, so the numbers differ slightly in the third decimal, but the story is the same: TD gets ahead early and ends lower. Notice how little separates the two functions: `mc_update` uses the return `G`, `td_update` uses `r + gamma * V[s_next]`. Everything else in this course (SARSA, Q-learning, DQN, actor-critic) keeps that `delta` line and changes what goes around it.

---

## Common misconceptions

!!! warning "\"TD uses its own guesses, so it converges to the wrong answer.\""
    The guesses are wrong at first, but they are corrected by real rewards at every step. For a fixed policy, tabular TD(0) converges to $V^\pi$ (with a step size that shrinks appropriately; with a small constant $\alpha$ it hovers close to it). The bias is temporary; the lower variance is what you keep.

!!! warning "\"Monte Carlo is unbiased, so it must be more accurate.\""
    Unbiased only means *right on average*. Each MC target carries the randomness of an entire episode, so with limited experience MC estimates are often further from the truth than TD's, as the learning curves show.

!!! warning "\"Model-free means the agent doesn't need an environment.\""
    The opposite: model-free methods need *lots* of interaction with the environment, because experience is their only source of information. What they don't need is a written-down **model** $p(s', r \mid s, a)$.

!!! warning "\"MC and TD need the agent to know the transition probabilities to compute the expectation.\""
    They never compute an expectation explicitly. Averaging many sampled updates does the same job: states that are reached more often simply get more updates.

---

## Check yourself

??? question "1. $V(C) = 0.5$, you run an episode from C that ends on the right (return $G = 1$), and $\alpha = 0.1$. What is the Monte Carlo update for $V(C)$?"
    $V(C) \leftarrow 0.5 + 0.1 \cdot (1 - 0.5) = $ **0.55**.

??? question "2. $V(C) = 0.5$, $V(D) = 0.6$. The agent steps from C to D with reward 0, $\gamma = 1$, $\alpha = 0.1$. What are the TD error and the new $V(C)$?"
    $\delta = 0 + 1 \cdot 0.6 - 0.5 = 0.1$, so $V(C) \leftarrow 0.5 + 0.1 \cdot 0.1 = $ **0.51**.

??? question "3. In the random walk with every value starting at 0.5, why does TD change only one state during the first episode?"
    Every step that doesn't end the episode has reward 0 and goes between two states that both have value 0.5, so $\delta = 0 + 0.5 - 0.5 = 0$. Only the final step has a target that differs from 0.5: the real reward (0 or 1) with $V(\text{end}) = 0$. So only the state next to the exit (A or E) moves.

??? question "4. Your task never ends (a thermostat controlling a building forever). Can you use Monte Carlo? Can you use TD?"
    Plain Monte Carlo needs complete episodes to compute returns, so it can't be used directly. TD only needs one step $(s, r, s')$ at a time, so it works on continuing tasks. (Discounting with $\gamma < 1$ keeps the values finite.)

??? question "5. In the learning-curve widget, TD with a large $\alpha$ gets lower error early but higher error later. Why?"
    A large step size moves each estimate a long way toward each new target. That learns fast, but every target is noisy, so the estimates keep bouncing around the true values instead of settling. A small $\alpha$ averages over more experience: slower at first, more precise in the end.

---

## Further reading

- Richard Sutton & Andrew Barto, [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) (2nd ed.), Chapter 5 (Monte Carlo), Chapter 6 (TD learning, including the random walk and driving-home examples) and Chapter 7 (n-step methods)
- Richard Sutton, [*Learning to Predict by the Methods of Temporal Differences*](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf) (1988), the paper that introduced TD learning
- David Silver, [*UCL Course on RL*](https://www.davidsilver.uk/teaching/), Lecture 4: Model-Free Prediction
- Gerald Tesauro, [*Temporal Difference Learning and TD-Gammon*](https://dl.acm.org/doi/10.1145/203330.203343) (1995), TD learning that reached world-class backgammon

**Next lessons:** [Q-learning & SARSA](../q-learning-sarsa/index.md)

<script src="viz/monte-carlo-td-learning.js"></script>
