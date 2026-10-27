# Dynamic Programming: Policy and Value Iteration

!!! abstract "TL;DR"
    If you **know the model** of the environment, the rules $p(s', r \mid s, a)$, you don't need trial and error. You can **compute** the best policy. Dynamic programming (DP) turns the Bellman equations into **update rules** and applies them to every state, over and over, until nothing changes. **Policy iteration** alternates "how good is my policy?" (evaluation) with "act greedily on that" (improvement). **Value iteration** folds both into one update. Both find the optimal policy for the cliff world in a fraction of a second.

**Level:** intermediate · **Prerequisites:** [What is RL?](../reinforcement-learning-intro/index.md) (the cliff world, policies and values) and [MDPs & Bellman equations](../mdp-bellman-equations/index.md) (the model $p(s', r \mid s, a)$ and the Bellman equations).

---

## Why should I care?

In the [MDP lesson](../mdp-bellman-equations/index.md) we wrote down the Bellman equations: what the values of a policy, and the optimal values, **must** satisfy. That tells us what the answer looks like. It doesn't tell us how to find it.

Dynamic programming is the "how", for the case where we know the rules of the world:

- **Planning problems with known rules:** inventory and stock control, maintenance schedules, queues, small robot navigation tasks, board-game endgames. If you can write down the transition probabilities, DP gives you the provably optimal policy.
- **The blueprint for everything that follows.** Monte Carlo, TD learning, Q-learning, DQN and actor-critic are all, at heart, DP with the exact sums replaced by **samples** from experience. Q-learning is "value iteration from samples". Actor-critic is "policy iteration with a neural network". If you understand this lesson, the rest of the course is variations on a theme.

The name is historical (Richard Bellman, 1950s). "Programming" here means planning, as in "a TV programme", not writing code. "Dynamic" means it unfolds over time steps.

---

## Step 1 — Turn the Bellman equation into an update rule: policy evaluation

The first job: given a policy $\pi$, compute its values $V^\pi(s)$ for every state. The Bellman expectation equation from the last lesson says the value of a state is the expected reward of one step plus the discounted value of where you land:

$$
V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\,\big[\, r + \gamma\, V^\pi(s') \,\big]
$$

That is one equation per state, and $V^\pi$ appears on both sides. The trick of **iterative policy evaluation** is almost embarrassingly simple: **use the equation as an assignment**. Start with any guess, say $V_0(s) = 0$ everywhere, and repeatedly overwrite every state with the right-hand side, computed from the previous guess:

$$
V_{k+1}(s) \;\leftarrow\; \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\,\big[\, r + \gamma\, V_k(s') \,\big] \qquad \text{for every state } s
$$

One pass over all states is called a **sweep**. Each single update is called a **backup**, because it copies ("backs up") value information from the next states into the current one. We stop when the **biggest change** in a sweep, $\Delta = \max_s |V_{k+1}(s) - V_k(s)|$, falls below a small threshold $\theta$ (we use $10^{-6}$).

Why does this converge? Each sweep shrinks the distance to the true $V^\pi$ by at least a factor $\gamma$. With $\gamma < 1$, errors die out geometrically, wherever you start. (In maths language, the update is a *contraction*.)

You have already seen this algorithm: the policy widget in [lesson 1](../reinforcement-learning-intro/index.md#step-5-how-good-is-a-state-the-value-function) secretly ran it every time you changed an arrow. Here you can watch it, sweep by sweep.

<div id="viz-eval" class="viz"><div class="viz-title">🧮 Interactive: policy evaluation, one sweep at a time</div></div>

!!! tip "What to notice"
    - **Information flows backwards from the end.** After sweep 1, every cell knows only about the −1 step cost, except the cells next to the goal or the cliff, which have seen the +10 or −20. Each sweep pushes that news one more step away. That is the backup, made visible.
    - **The random policy is slow to evaluate.** With $\gamma = 0.95$ it takes **155 sweeps** to reach $\Delta < 10^{-6}$, ending at $V(\text{start}) = -19.88$ (the same number as lesson 1). Lower $\gamma$ to 0.5 and it needs only 21; at $\gamma = 0.99$ it needs 272. A larger $\gamma$ means a longer memory, so errors fade more slowly.
    - **Deterministic paths settle exactly.** *Edge of the cliff* at slip 0 needs only 9 sweeps ($V(\text{start}) = 2.05$): once the goal's value has travelled back along every path, nothing changes any more. At slip 0.2 it needs 32 sweeps and ends at $-9.08$, as in lesson 1.

---

## Step 2 — Policy improvement: act greedily on the values

Knowing $V^\pi$ is useful only if it helps us do better. Here's how. In any state $s$, look **one step ahead** at each action, using the values of $\pi$ for whatever comes after:

$$
Q^\pi(s, a) = \sum_{s', r} p(s', r \mid s, a)\,\big[\, r + \gamma\, V^\pi(s') \,\big]
$$

$Q^\pi(s, a)$ is the return you expect if you take action $a$ **once**, then go back to following $\pi$. Now build a new policy $\pi'$ that simply picks the best-looking action everywhere. It is **greedy** with respect to $V^\pi$:

$$
\pi'(s) = \arg\max_a \; Q^\pi(s, a)
$$

Here is one real example: the cell just above the last cliff square, under the random policy (slip 0, $\gamma = 0.95$).

<div class="viz diagram">
<script type="application/json">
{
  "title": "🔍 One-step lookahead in a single cell",
  "cols": 4,
  "rows": 2,
  "rowHeight": 105,
  "nodes": [
    {"id": "s", "col": 1.5, "row": 0, "icon": "📍", "label": "<b>Cell above the last cliff square</b>", "sub": "random policy: V<sup>π</sup>(s) = −15.34", "span": 1.6, "maxWidth": 250, "role": "loss", "note": "Row 2, column 4 of the grid. Under the random policy you wander off and usually fall into the cliff, so the value is low."},
    {"id": "up", "col": 0, "row": 1, "label": "↑ up", "sub": "Q = −15.97", "note": "Move up once, then wander randomly."},
    {"id": "right", "col": 1, "row": 1, "label": "<b>→ right</b>", "sub": "Q = −7.21", "role": "opt", "note": "Move right (next to the goal), then wander randomly. The best of the four."},
    {"id": "down", "col": 2, "row": 1, "label": "<b>↓ down</b>", "sub": "Q = −20.00", "role": "grad", "note": "Step straight into the cliff: −20 and the episode ends."},
    {"id": "left", "col": 3, "row": 1, "label": "← left", "sub": "Q = −18.19", "note": "Move left, away from the goal, then wander randomly."}
  ],
  "edges": [
    {"from": "s", "to": "up"},
    {"from": "s", "to": "right"},
    {"from": "s", "to": "down"},
    {"from": "s", "to": "left"}
  ],
  "steps": [
    {"show": ["s"], "text": "Under the random policy this cell is worth −15.34: wandering from here often ends in the cliff."},
    {"show": ["s", "s>up", "s>right", "s>down", "s>left", "up", "right", "down", "left"], "text": "Look one step ahead: for each action, the reward of that step plus γ × the value (under the old policy) of where you land. That is Q<sup>π</sup>(s, a)."},
    {"show": ["s>down", "down"], "text": "Down walks into the cliff: Q = −20, the worst option."},
    {"show": ["s>right", "right"], "text": "Right is the best: Q = −7.21. So the improved policy π′ points <b>right</b> here."},
    {"show": ["s", "right"], "text": "−7.21 ≥ −15.34: taking the greedy action once already beats following π. The policy improvement theorem says that doing it <b>everywhere</b> gives a policy at least as good in every state."}
  ],
  "narrow": {"cols": 2.4, "rows": 4, "rowHeight": 80, "hideSub": false, "pos": {"s": [0, 1.5], "up": [1.4, 0], "right": [1.4, 1], "down": [1.4, 2], "left": [1.4, 3]}}
}
</script>
</div>

Why is $\pi'$ never worse than $\pi$? In every state, the greedy action is at least as good as what $\pi$ would do on average: $Q^\pi(s, \pi'(s)) \ge V^\pi(s)$. So "follow $\pi'$ for one step, then $\pi$" is at least as good as $\pi$. Apply the same argument at the next step, and the next, and you get "follow $\pi'$ forever" is at least as good too. This is the **policy improvement theorem**:

$$
Q^\pi\big(s, \pi'(s)\big) \ge V^\pi(s) \;\text{ for all } s \quad\Longrightarrow\quad V^{\pi'}(s) \ge V^\pi(s) \;\text{ for all } s
$$

And if the greedy policy is **the same** as $\pi$, then $\pi$ already satisfies the Bellman **optimality** equation, so it is optimal. No better policy exists.

!!! tip "What to notice"
    Go back to the widget in Step 1, evaluate the **Random** policy to convergence, and tick **show the greedy policy**. The green arrows are $\pi'$. Even though the random policy is terrible ($-19.88$), one greedy step on its values already gives a sensible route around the top (it is the *safe detour*, worth $-2.04$ at slip 0). Click cells to see the four Q-values behind each arrow.

---

## Step 3 — Policy iteration: evaluate, improve, repeat

Put Steps 1 and 2 in a loop and you have **policy iteration**:

1. **Evaluate** the current policy $\pi_k$ to get $V^{\pi_k}$ (many sweeps, Step 1).
2. **Improve**: $\pi_{k+1} = $ greedy with respect to $V^{\pi_k}$ (Step 2).
3. If $\pi_{k+1} = \pi_k$, stop: it is optimal. Otherwise go back to 1.

Each new policy is at least as good as the last. There are only finitely many deterministic policies ($4^{18}$ here, about 69 billion), so the loop must end. In practice it ends after a **handful** of rounds.

<div id="viz-pi" class="viz"><div class="viz-title">🔁 Interactive: policy iteration on the cliff world</div></div>

!!! tip "What to notice"
    - **Very few rounds.** At slip 0 the values of the start cell go $-19.88 \to -2.04 \to 2.05 \to 2.05$: random, then the safe detour, then the cliff-edge route. The fourth improvement changes nothing, so it stops. At slip 0.2 it goes $-19.88 \to -6.49 \to -6.30$ and stops after the third.
    - **The expensive part is evaluation.** Evaluating the random policy takes 155 sweeps; later policies need far fewer (12, 10 and 9 at slip 0). Improvement is a single cheap pass.
    - **The answer depends on the world.** Slip 0 gives the cliff-edge route ($V^* = 2.05$). Slip 0.2 gives a cautious policy worth $-6.30$, better than both hand-written policies from lesson 1 ($-9.08$ for the edge, $-6.97$ for the detour).

---

## Step 4 — Value iteration: one sweep, then improve

Policy iteration waits for evaluation to converge before improving. Does it have to? What if we improve after **every single sweep**? Then evaluation and improvement merge into one update, which is the Bellman **optimality** equation used as an assignment:

$$
V_{k+1}(s) \;\leftarrow\; \max_a \sum_{s', r} p(s', r \mid s, a)\,\big[\, r + \gamma\, V_k(s') \,\big] \qquad \text{for every state } s
$$

Compare it with policy evaluation in Step 1: the average over the policy's actions, $\sum_a \pi(a \mid s)$, has become a $\max_a$. That's all. This is **value iteration**. It converges to $V^*$ for the same reason as before (each sweep shrinks the error by at least $\gamma$). When it stops, read off the optimal policy greedily: $\pi^*(s) = \arg\max_a Q(s, a)$.

<div id="viz-vi" class="viz"><div class="viz-title">⚡ Interactive: value iteration, sweep by sweep</div></div>

!!! tip "What to notice"
    - **Slip 0:** value iteration needs **9 sweeps** ($\Delta$ drops to exactly 0 on the 9th) and finds $V^*(\text{start}) = 2.05$, the cliff-edge route. The greedy arrows are already final after sweep 7, before the values have finished.
    - **Slip 0.2:** it needs **37 sweeps** and gives $V^*(\text{start}) = -6.30$, the same answer as policy iteration. The arrows stop changing after sweep 11. Policies often converge long before values do.
    - **The route switches at slip ≈ 0.112.** Drag the slip slider slowly: at 0.11 the optimal route still hugs the cliff ($V^* = -4.17$); at 0.12 it climbs over the top ($V^* = -4.46$). Nobody told the agent "be careful"; it falls out of the maths.
    - **Above slip ≈ 0.327 the start cell points ← into the wall.** Bumping into the wall keeps you in place, and a slip from "left" only goes up or down (down is also a wall). It is the one action from the start that can never slide you into the cliff.
    - **The chart:** $\Delta$ falls roughly in a straight line on the log scale, i.e. geometrically. Changing $\gamma$ at slip 0.2 moves the sweep count only a little (19 at $\gamma = 0.5$, 34 at 0.9, 39 at 0.99), because most paths end at the goal or the cliff long before discounting matters.

---

## Step 5 — The big picture: generalized policy iteration

Policy iteration and value iteration are two ends of one idea. There are two processes, and they push each other along:

- **Evaluation** makes the values consistent with the current policy.
- **Improvement** makes the policy greedy with respect to the current values.

Each one, by doing its job, spoils the other a little: a new policy makes the old values out of date, and new values make the old policy no longer greedy. Yet the back-and-forth only stops when **both** are satisfied at once. A policy that is greedy with respect to its own values satisfies the Bellman optimality equation. So the only resting point is $\pi^*$ and $V^*$.

<div class="viz diagram">
<script type="application/json">
{
  "title": "♻️ Generalized policy iteration (GPI)",
  "cols": 3,
  "rows": 1.9,
  "rowHeight": 100,
  "nodes": [
    {"id": "pi", "col": 0, "row": 0.45, "icon": "🧭", "label": "<b>Policy π</b>", "sub": "what to do in each state", "role": "model", "note": "A table of arrows here; a neural network in deep RL (the actor)."},
    {"id": "v", "col": 2, "row": 0.45, "icon": "📊", "label": "<b>Values V</b>", "sub": "how good each state is", "role": "loss", "note": "A table of numbers here; a neural network in deep RL (the critic)."}
  ],
  "edges": [
    {"from": "pi", "to": "v", "label": "evaluate", "curve": 0.3},
    {"from": "v", "to": "pi", "label": "improve", "curve": 0.3}
  ],
  "steps": [
    {"show": ["pi", "pi>v", "v"], "text": "<b>Evaluate:</b> move V towards V<sup>π</sup>, the values of the current policy."},
    {"show": ["v", "v>pi", "pi"], "text": "<b>Improve:</b> make π greedy with respect to the current V."},
    {"show": ["pi", "pi>v", "v", "v>pi"], "text": "<b>Policy iteration</b> runs evaluation all the way to convergence before each improvement. <b>Value iteration</b> does just one sweep of evaluation before each improvement."},
    {"show": ["pi", "v"], "text": "The loop only stops when π is greedy for its own values. That is the Bellman optimality equation, so π = π* and V = V*."}
  ],
  "narrow": {"cols": 2.4, "rows": 1.9, "hideSub": true, "pos": {"pi": [0, 0.45], "v": [1.4, 0.45]}}
}
</script>
</div>

Sutton and Barto call this **generalized policy iteration (GPI)**. Almost every RL method fits it. You can evaluate fully or partially, update all states or just one, compute exact sums or use samples. As long as both processes keep running, things tend towards the optimum. Later lessons are all GPI:

| Method | Evaluation | Improvement |
|---|---|---|
| Policy iteration (this lesson) | exact, all sweeps | greedy, all states |
| Value iteration (this lesson) | one sweep | greedy, built into the max |
| Monte Carlo control, SARSA, Q-learning | from **sampled** episodes or steps | ε-greedy |
| Actor-critic (PPO, SAC) | a critic network learns $V$ or $Q$ | a gradient step on the actor network |

---

## Step 6 — The catch: you need the model, and you need every state

DP is exact and fast on the cliff world. Two things stop it from solving everything:

1. **It needs the model.** Every backup sums over $p(s', r \mid s, a)$. For a robot, a stock market or a conversation with a person, nobody can write those probabilities down. We only get to **try things and see what happens**.
2. **It sweeps over every state.** The cliff world has 18 states. Backgammon has about $10^{20}$. Even at a billion backups per second, **one** sweep over $10^{20}$ states would take about 3,200 years. Go has around $10^{170}$ positions; an image-based game has astronomically more. The number of states grows exponentially with the number of variables that describe a state. Bellman called this the **curse of dimensionality**.

The way out is to keep the GPI idea but change how the backups are done:

- **Sample instead of summing.** Replace $\sum_{s',r} p(s', r \mid s, a)[\dots]$ with what actually happened on one step or one episode. No model needed. This is [Monte Carlo and TD learning](../monte-carlo-td-learning/index.md), the next lesson.
- **Only visit states that matter.** An agent following its policy spends its time in relevant states, not in all $10^{20}$.
- **Generalize between states.** Replace the table of values with a function, e.g. a neural network, so that learning about one state teaches you about similar ones (the DQN lesson).

---

## Putting it all together

Value iteration, the shortest route from a known model to an optimal policy:

<div class="viz diagram">
<script type="application/json">
{
  "title": "⚡ Value iteration from start to finish",
  "cols": 3,
  "rows": 2,
  "rowHeight": 110,
  "nodes": [
    {"id": "init", "col": 0, "row": 0, "icon": "🌱", "label": "Start with V(s) = 0", "shape": "pill", "sub": "any guess works"},
    {"id": "sweep", "col": 1, "row": 0, "icon": "🗺️", "label": "Sweep all states", "sub": "uses the model p(s′, r | s, a)", "role": "data", "note": "DP needs the full model: every next state and reward, with its probability."},
    {"id": "backup", "col": 2, "row": 0, "icon": "🔙", "label": "<b>Bellman backup</b>", "sub": "V(s) ← max<sub>a</sub> Σ p · [r + γV(s′)]", "role": "loss", "note": "The Bellman optimality equation, used as an assignment."},
    {"id": "delta", "col": 2, "row": 1, "icon": "📉", "label": "Track Δ", "sub": "biggest change this sweep", "role": "loss"},
    {"id": "check", "col": 1, "row": 1, "label": "Δ < θ ?", "shape": "decision", "role": "grad"},
    {"id": "done", "col": 0, "row": 1, "icon": "🏁", "label": "<b>Act greedily: π*</b>", "sub": "π*(s) = argmax<sub>a</sub> Q(s, a)", "shape": "pill", "role": "opt"}
  ],
  "edges": [
    {"from": "init", "to": "sweep"},
    {"from": "sweep", "to": "backup"},
    {"from": "backup", "to": "delta"},
    {"from": "delta", "to": "check"},
    {"from": "check", "to": "sweep", "label": "no"},
    {"from": "check", "to": "done", "label": "yes"}
  ],
  "steps": [
    {"show": ["init"], "text": "Start from any guess. Zero everywhere is the usual choice."},
    {"show": ["init>sweep", "sweep"], "text": "Go through every state. For each one we need the model: which next states and rewards each action can lead to, and how likely they are."},
    {"show": ["sweep>backup", "backup"], "text": "Replace V(s) with the best one-step lookahead: the Bellman optimality backup."},
    {"show": ["backup>delta", "delta"], "text": "Remember the largest change any state made in this sweep."},
    {"show": ["delta>check", "check", "check>sweep", "sweep"], "text": "If something still changed noticeably, do another sweep. Each sweep shrinks the error by at least a factor γ."},
    {"show": ["check>done", "done"], "text": "Once the values have settled, the optimal policy is just the greedy action in every state."}
  ],
  "narrow": {"cols": 2, "rows": 4, "rowHeight": 95, "pos": {"init": [0, 0], "sweep": [0, 1], "backup": [1, 1], "delta": [1, 2], "check": [0, 2], "done": [0, 3]}}
}
</script>
</div>

Here are both algorithms in numpy. The model is stored as two arrays: `P[s, a, s']`, the probability of landing in non-terminal state `s'`, and `R[s, a]`, the expected immediate reward. Terminal states (goal, cliff) have value 0, so they don't need a column in `P`. Policies are tables `pi[s, a]` of action probabilities, so the random policy and a deterministic one use the same code.

```python
import numpy as np

MAP = ["......",
       "..#...",
       "......",
       "SCCCCG"]
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]          # up, right, down, left
ARROWS = "↑→↓←"
STATES = [(r, c) for r in range(4) for c in range(6) if MAP[r][c] in ".S"]
INDEX = {s: i for i, s in enumerate(STATES)}         # 18 non-terminal states
nS, nA = len(STATES), 4


def build_model(slip):
    """The model p(s', r | s, a) as two arrays:
    P[s, a, s'] = probability of landing in non-terminal state s'
    R[s, a]     = expected immediate reward."""
    P, R = np.zeros((nS, nA, nS)), np.zeros((nS, nA))
    for (r, c), s in INDEX.items():
        for a in range(nA):
            for p, d in [(1 - slip, a), (slip / 2, (a + 1) % 4), (slip / 2, (a + 3) % 4)]:
                nr, nc = r + MOVES[d][0], c + MOVES[d][1]
                if not (0 <= nr < 4 and 0 <= nc < 6) or MAP[nr][nc] == "#":
                    nr, nc = r, c                                    # bump: stay put
                tile = MAP[nr][nc]
                R[s, a] += p * (10 if tile == "G" else -20 if tile == "C" else -1)
                if tile in ".S":                                     # terminal states have value 0
                    P[s, a, INDEX[(nr, nc)]] += p
    return P, R


def policy_evaluation(P, R, pi, gamma, tol=1e-6):
    """Sweep V(s) <- sum_a pi(a|s) [R(s,a) + gamma * sum_s' P(s'|s,a) V(s')] until it stops changing."""
    V, sweeps = np.zeros(nS), 0
    while True:
        V_new = (pi * (R + gamma * P @ V)).sum(axis=1)  # pi[s, a] = probability of a in s
        delta, V, sweeps = np.abs(V_new - V).max(), V_new, sweeps + 1
        if delta < tol:
            return V, sweeps


def greedy(P, R, V, gamma):
    """One-step lookahead: the best action in every state, as a one-hot policy table."""
    Q = R + gamma * P @ V
    return np.eye(nA)[Q.argmax(axis=1)]


def policy_iteration(P, R, gamma):
    pi = np.full((nS, nA), 1 / nA)                       # start from the random policy
    for it in range(1, 100):
        V, sweeps = policy_evaluation(P, R, pi, gamma)   # 1. evaluate
        new_pi = greedy(P, R, V, gamma)                  # 2. improve
        print(f"  iteration {it}: evaluation took {sweeps:3d} sweeps, V(start) = {V[INDEX[(3, 0)]]:6.2f}")
        if np.array_equal(new_pi, pi):                   # 3. stable? then it is optimal
            return pi, V
        pi = new_pi


def value_iteration(P, R, gamma, tol=1e-6):
    V, sweeps = np.zeros(nS), 0
    while True:
        V_new = (R + gamma * P @ V).max(axis=1)          # Bellman optimality backup, all states at once
        delta, V, sweeps = np.abs(V_new - V).max(), V_new, sweeps + 1
        if delta < tol:
            return greedy(P, R, V, gamma), V, sweeps


def show(pi):
    grid = [list(row) for row in MAP]
    for (r, c), s in INDEX.items():
        grid[r][c] = ARROWS[pi[s].argmax()]
    return "\n".join("".join(row) for row in grid)


gamma = 0.95
for slip in [0.0, 0.2]:
    P, R = build_model(slip)
    print(f"slip = {slip}")
    print(" policy iteration:")
    pi_pi, V_pi = policy_iteration(P, R, gamma)
    pi_vi, V_vi, sweeps = value_iteration(P, R, gamma)
    print(f" value iteration: {sweeps} sweeps, V*(start) = {V_vi[INDEX[(3, 0)]]:.2f}")
    print(" same policy from both:", np.array_equal(pi_pi, pi_vi))
    print(show(pi_vi), "\n")
```

```text
slip = 0.0
 policy iteration:
  iteration 1: evaluation took 155 sweeps, V(start) = -19.88
  iteration 2: evaluation took  12 sweeps, V(start) =  -2.04
  iteration 3: evaluation took  10 sweeps, V(start) =   2.05
  iteration 4: evaluation took   9 sweeps, V(start) =   2.05
 value iteration: 9 sweeps, V*(start) = 2.05
 same policy from both: True
→→→→→↓
→↓#→→↓
→→→→→↓
↑CCCCG

slip = 0.2
 policy iteration:
  iteration 1: evaluation took 155 sweeps, V(start) = -19.88
  iteration 2: evaluation took  37 sweeps, V(start) =  -6.49
  iteration 3: evaluation took  37 sweeps, V(start) =  -6.30
 value iteration: 37 sweeps, V*(start) = -6.30
 same policy from both: True
→→→→→↓
↑↑#→→↓
↑↑→↑→↓
↑CCCCG
```

The whole thing runs in well under a second. Notice the line `R + gamma * P @ V`: it computes $Q(s, a)$ for every state and action at once. Policy evaluation averages it over the policy, value iteration takes the max, and improvement takes the argmax. Three algorithms, one backup.

Two variations you will meet in other code:

- **In-place sweeps** (Gauss–Seidel): overwrite `V[s]` immediately, so later states in the same sweep already use the new values. Often converges in fewer sweeps; lesson 1's widget did this.
- **Q-value iteration**: keep a table $Q(s, a)$ instead of $V(s)$ and back up $Q(s, a) \leftarrow \sum p\,[r + \gamma \max_{a'} Q(s', a')]$. Replace that sum by a single sampled transition and you get Q-learning.

---

## Common misconceptions

!!! warning "\"Dynamic programming is a learning algorithm.\""
    DP is **planning**, not learning. It never interacts with the environment: it reads the rules $p(s', r \mid s, a)$ and computes. RL methods proper (from the next lesson on) learn from experience because they don't have the rules.

!!! warning "\"You have to evaluate a policy exactly before you can improve it.\""
    No. Improvement works with any values, and value iteration improves after a single sweep. Policy iteration's full evaluation buys fewer, bigger improvement steps; value iteration buys cheap steps. Both reach the same $\pi^*$, and the arrows often settle long before the values (sweep 11 of 37 at slip 0.2).

!!! warning "\"The greedy policy only maximises the next reward.\""
    Greedy here means greedy with respect to $Q$, which includes $\gamma V(s')$, the value of everything that comes after. Acting greedily on **good values** is far-sighted. Acting greedily on **rewards alone** would be short-sighted.

!!! warning "\"Value iteration tells you how many steps to the goal.\""
    It tells you the expected discounted return. On a slippery floor the optimal policy can deliberately take a longer route, or even walk into a wall (slip above about 0.33), because that maximises the *expected* return, not the path length.

---

## Check yourself

??? question "1. What is the only difference between the policy evaluation update and the value iteration update?"
    Policy evaluation **averages** over the policy's actions, $\sum_a \pi(a \mid s)[\dots]$. Value iteration takes the **maximum** over actions, $\max_a [\dots]$. Everything inside the brackets, the one-step lookahead $\sum_{s',r} p(s',r \mid s,a)[r + \gamma V(s')]$, is the same.

??? question "2. Policy iteration produced the same policy twice in a row. Why can we stop and call it optimal?"
    If $\pi$ is greedy with respect to its own values $V^\pi$, then $V^\pi(s) = \max_a \sum p\,[r + \gamma V^\pi(s')]$ for every state. That is exactly the Bellman optimality equation, whose only solution is $V^*$. So $V^\pi = V^*$ and $\pi$ is optimal.

??? question "3. At slip 0.2, why is the optimal value of the start (−6.30) better than both hand-written policies from lesson 1 (−9.08 and −6.97)?"
    The optimal policy is free to choose cell by cell. From the start it climbs over the top like the safe detour, but it differs from the detour in 4 cells: two cells in row 1 (columns 3 and 4) point → towards the goal column instead of ↑, and two cells in row 2 (columns 2 and 4) point →. Neither hand-written policy is greedy with respect to its own values, so policy improvement can still make each of them better, and DP keeps improving until no cell can.

??? question "4. Your problem has 30 binary sensors, and the state is all 30 readings. How many states are there, and is DP practical?"
    $2^{30} \approx 10^9$ states. A table of that size fits in memory, and a sweep at a billion backups per second takes about a second, so it is borderline feasible **if** you know the model. With 60 sensors it would be $2^{60} \approx 10^{18}$ states, about 37 years per sweep. That exponential growth is the curse of dimensionality.

??? question "5. Value iteration on the cliff world at slip 0 stops after 9 sweeps no matter what γ is. Why doesn't a larger γ slow it down here?"
    With no slip everything is deterministic and every optimal path ends at the goal within a few steps. The longest optimal path from any cell is 8 steps. After 8 sweeps every state's value is exactly right and the next sweep changes nothing ($\Delta = 0$). The "shrink by $\gamma$ per sweep" rule is only a worst-case guarantee; it bites when episodes can last a long time, as with the random policy (155 sweeps at $\gamma = 0.95$, 272 at $0.99$).

---

## Further reading

- Richard Sutton & Andrew Barto, [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) (2nd ed., free online), Chapter 4: Dynamic Programming. Section 4.6 introduces generalized policy iteration.
- David Silver, [*UCL Course on RL*](https://www.davidsilver.uk/teaching/), Lecture 3: Planning by Dynamic Programming (video + slides)
- Richard Bellman, *Dynamic Programming* (1957), the original book, where the "curse of dimensionality" was named
- Andrej Karpathy, [*REINFORCEjs: GridWorld with Dynamic Programming*](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html), another interactive grid world with policy and value iteration

**Next lessons:** [Monte Carlo & TD learning](../monte-carlo-td-learning/index.md): the same evaluate-and-improve loop, learned from sampled experience instead of a known model.

<script src="viz/dynamic-programming.js"></script>
