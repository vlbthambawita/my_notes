# Markov Decision Processes and the Bellman Equations

!!! abstract "TL;DR"
    A **Markov decision process (MDP)** is the precise maths of an RL problem: states, actions, a table of **transition probabilities** $p(s', r \mid s, a)$, and a discount $\gamma$. Because the return is recursive ($G_t = r_{t+1} + \gamma G_{t+1}$), the value of a state equals the **expected reward of one step plus the discounted value of where you land**. That is the **Bellman equation**. Every RL method in this course is a way of solving it, exactly or approximately.

**Level:** beginner–intermediate · **Prerequisites:** [What is RL?](../reinforcement-learning-intro/index.md) (states, rewards, return, policy, value), [Multi-armed bandits](../multi-armed-bandits/index.md). Knowing what a matrix and a linear system ($Ax = b$) are helps for Step 7.

---

## Why should I care?

In the [first lesson](../reinforcement-learning-intro/index.md) the cliff-world widget showed a value $V^\pi(s)$ in every cell. It got those numbers without playing a single episode. How? It used the **Bellman equation**: a rule that ties the value of each cell to the values of its neighbours.

In the [bandits lesson](../multi-armed-bandits/index.md) every action was a one-shot decision: pull an arm, get a reward, done. Real problems have **consequences**: the action you take now changes the situation you face next. MDPs are the language for that, and the Bellman equations are the one idea behind:

- **Dynamic programming** (next lesson): solve the Bellman equation when you know the rules of the world.
- **Q-learning, SARSA, DQN:** solve it from samples, without knowing the rules.
- **Actor-critic methods (PPO, used in RLHF):** a value network trained on a Bellman target judges the policy.

If you understand this lesson, every later update rule will look familiar.

---

## Step 1 — The Markov property: the state is all you need

A **state** should contain everything about the past that matters for the future. If it does, we say the state has the **Markov property**:

$$
\Pr(s_{t+1}, r_{t+1} \mid s_t, a_t) \;=\; \Pr(s_{t+1}, r_{t+1} \mid s_0, a_0, r_1, \dots, s_t, a_t)
$$

In words: once you know the current state and action, the rest of the history adds **nothing** to your prediction of what happens next. "The future is independent of the past, given the present."

The cliff world is Markov: if you know which cell you are in, how you got there does not change where your next step can take you. Here is a state that is **not** Markov:

<div class="viz diagram">
<script type="application/json">
{
  "title": "🎾 Is one snapshot enough? A ball in a game of Pong",
  "cols": 3,
  "rows": 2,
  "rowHeight": 105,
  "nodes": [
    {"id": "one", "col": 0, "row": 0, "icon": "📷", "label": "<b>One frame</b>", "sub": "ball at x = 5", "role": "data", "note": "The state is a single screenshot: only the ball's position."},
    {"id": "q1", "col": 1, "row": 0, "label": "Where next?", "sub": "left or right?", "shape": "decision", "role": "grad"},
    {"id": "bad", "col": 2, "row": 0, "icon": "❌", "label": "<b>Not Markov</b>", "sub": "the past still matters", "role": "grad", "note": "To predict the ball you need an older frame too. The state is missing information that the history has."},
    {"id": "two", "col": 0, "row": 1, "icon": "🎞️", "label": "<b>Two frames</b>", "sub": "x = 4, then x = 5", "role": "data", "note": "The state is the last two screenshots stacked together, as DQN does with Atari games."},
    {"id": "vel", "col": 1, "row": 1, "label": "Moving right", "sub": "+1 per step", "role": "model"},
    {"id": "good", "col": 2, "row": 1, "icon": "✅", "label": "<b>Markov</b>", "sub": "next: x = 6", "role": "opt", "note": "Position and velocity together are enough. Older frames add nothing."}
  ],
  "edges": [
    {"from": "one", "to": "q1"},
    {"from": "q1", "to": "bad"},
    {"from": "two", "to": "vel"},
    {"from": "vel", "to": "good"}
  ],
  "steps": [
    {"show": ["one"], "text": "Suppose the state is a single frame of the game: the ball is at x = 5."},
    {"show": ["one>q1", "q1"], "text": "Where will the ball be next? From one frame you cannot say: it looks the same whether it moves left or right."},
    {"show": ["q1>bad", "bad"], "text": "Knowing the <b>previous</b> frame would help, so this state is <b>not Markov</b>: the history carries information the state lacks."},
    {"show": ["two", "two>vel", "vel"], "text": "Fix it by putting more into the state: two frames (x = 4, then x = 5) reveal the velocity."},
    {"show": ["vel>good", "good"], "text": "Now the state is <b>Markov</b>: position and velocity predict the next frame, and older frames add nothing. DQN stacks 4 Atari frames for exactly this reason."}
  ],
  "narrow": {"cols": 2, "rows": 3, "rowHeight": 95, "pos": {"one": [0, 0], "q1": [0, 1], "bad": [0, 2], "two": [1, 0], "vel": [1, 1], "good": [1, 2]}}
}
</script>
</div>

Real states are often only *approximately* Markov. A robot's camera image doesn't show what is behind it; a chatbot's "state" is the whole conversation so far precisely so that nothing important is lost. When the state misses something important, the problem is called **partially observable** (a POMDP), and the usual fix is the one above: put more history into the state.

!!! tip "What to notice"
    - The Markov property is a property of **how you define the state**, not of the world. The same game is non-Markov with one frame and (close to) Markov with two.
    - It is what makes everything else in this lesson possible: if the state is all you need, then the value of a state can depend only on that state, not on how you got there.

---

## Step 2 — The Markov decision process

A **Markov decision process** packs the whole RL problem into five ingredients:

| Ingredient | Symbol | Recycling robot (below) |
|---|---|---|
| States | $\mathcal{S}$ | battery **high**, battery **low** |
| Actions | $\mathcal{A}(s)$ | **search** for cans, **wait** for someone to bring one, **recharge** (only when low) |
| Dynamics | $p(s', r \mid s, a)$ | the probabilities on the arrows |
| Rewards | part of $p$ | +2 per search, +1 per wait, −3 if the battery dies and a human must rescue it |
| Discount | $\gamma$ | 0.9 |

The heart of an MDP is the **dynamics function**

$$
p(s', r \mid s, a) = \Pr(s_{t+1} = s',\ r_{t+1} = r \mid s_t = s,\ a_t = a),
$$

the probability that taking action $a$ in state $s$ lands you in $s'$ with reward $r$. For each $(s, a)$ these probabilities sum to 1. It is the "rules of the game": in the bandits lesson it was one reward distribution per arm; here it also says **where you go next**.

Our example is a little robot that collects empty cans (a classic from Sutton & Barto's textbook, with our own numbers). Searching finds cans (+2) but drains the battery: from **high** it stays high with probability 0.7. From **low**, searching is risky: with probability 0.4 the battery dies, a human carries the robot back to the charger (−3), and it starts again at **high**.

**Try it:** the orange ring marks the state the robot is in. Press **Take one step** a few times. The policy (orange arrows) picks an action, then the environment (blue arrows) rolls the dice.

<div id="viz-robot" class="viz"><div class="viz-title">🤖 Interactive: the recycling-robot MDP</div></div>

!!! tip "What to notice"
    - **Two kinds of randomness.** The **agent** chooses the action, possibly at random ($\pi(a \mid s)$, orange). The **environment** chooses the outcome ($p(s', r \mid s, a)$, blue). The agent controls only the first.
    - The default *Coin flip* policy always searches when the battery is high, and flips a coin between searching and recharging when it is low.
    - Change the policy and the values $V$ in the circles change. The MDP (blue arrows) stays the same. A value always belongs to a **policy**.
    - We will come back to the **Sample** buttons in the next step.

---

## Step 3 — The return is recursive

In lesson 1 the **return** was the discounted sum of future rewards. Look at what happens when we pull the first reward out of the sum:

$$
\begin{aligned}
G_t &= r_{t+1} + \gamma\, r_{t+2} + \gamma^2\, r_{t+3} + \dots \\
    &= r_{t+1} + \gamma \left( r_{t+2} + \gamma\, r_{t+3} + \dots \right) \\
    &= r_{t+1} + \gamma\, G_{t+1}
\end{aligned}
$$

**The return from now is the next reward plus $\gamma$ times the return from the next step.** This one line is the seed of the whole lesson.

The value of a state is the expected return: $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$. Taking the expectation of both sides of the recursion gives

$$
V^\pi(s) = \mathbb{E}_\pi\left[\, r_{t+1} + \gamma\, V^\pi(s_{t+1}) \;\middle|\; s_t = s \,\right].
$$

(The step from $G_{t+1}$ to $V^\pi(s_{t+1})$ uses the Markov property: from $s_{t+1}$ on, the future depends only on $s_{t+1}$, so the expected value of $G_{t+1}$ is just $V^\pi(s_{t+1})$.)

**Try it:** in the robot widget above, press **Sample 1 return**. The readout shows one sampled run's rewards, its return $G_0$, and checks that $G_0 = r_1 + 0.9\, G_1$. Then press **Sample 1000 returns** and watch the running average (orange) settle onto the dashed line.

!!! tip "What to notice"
    - Single returns are noisy. For the *Coin flip* policy they spread out with a standard deviation of about 2.2 around their mean.
    - Their average still lands on the exact value $V^\pi(\text{high}) = 14.60$ (usually within about 0.15 after 1000 samples). The robot task never ends, so each sample is cut off after 200 steps; the rewards after that are weighted by $0.9^{200} \approx 7 \times 10^{-10}$, which is nothing.
    - So where does the exact 14.60 come from, if not from averaging? From the Bellman equation, next.

---

## Step 4 — The Bellman expectation equation for $V^\pi$

Write out the expectation in Step 3. Two random things happen in one step: the policy picks $a$ with probability $\pi(a \mid s)$, then the environment picks $(s', r)$ with probability $p(s', r \mid s, a)$. Average over both:

$$
\boxed{\;V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\,\big[\, r + \gamma\, V^\pi(s') \,\big]\;}
$$

This is the **Bellman expectation equation**. Read it as a **one-step lookahead**: list everything that can happen in one step, score each outcome by "reward now + discounted value of where I land", and take the probability-weighted average. Computing this right-hand side for a state is called a **backup**, because it carries value information back from the next states to the current one. Drawn as a tree it is a **backup diagram**:

<div class="viz diagram">
<script type="application/json">
{
  "title": "🌳 Backup diagram: V(low) for the coin-flip policy (γ = 0.9)",
  "cols": 3,
  "rows": 3,
  "rowHeight": 110,
  "nodes": [
    {"id": "s", "col": 1.25, "row": 0, "icon": "🔋", "label": "<b>low</b>", "sub": "V = 12.6", "role": "data", "note": "The state we are backing up. Its value should equal the weighted average of everything below it."},
    {"id": "search", "col": 0.5, "row": 1, "label": "<b>search</b>", "sub": "Q = 12.06", "role": "model", "note": "0.6 × 13.34 + 0.4 × 10.14 = 12.06"},
    {"id": "rech", "col": 2, "row": 1, "label": "<b>recharge</b>", "sub": "Q = 13.14", "role": "model", "note": "1.0 × 13.14 = 13.14"},
    {"id": "l1", "col": 0, "row": 2, "label": "p 0.6 → <b>low</b>", "sub": "2 + 0.9 × 12.6 = 13.34", "role": "data"},
    {"id": "l2", "col": 1, "row": 2, "label": "p 0.4 → <b>high</b>", "sub": "−3 + 0.9 × 14.6 = 10.14", "role": "data"},
    {"id": "l3", "col": 2, "row": 2, "label": "p 1 → <b>high</b>", "sub": "0 + 0.9 × 14.6 = 13.14", "role": "data"}
  ],
  "edges": [
    {"from": "s", "to": "search", "label": "π = 0.5"},
    {"from": "s", "to": "rech", "label": "π = 0.5"},
    {"from": "search", "to": "l1"},
    {"from": "search", "to": "l2"},
    {"from": "rech", "to": "l3"}
  ],
  "steps": [
    {"show": ["s"], "text": "Start from the state whose value we want: battery <b>low</b>, under the coin-flip policy."},
    {"show": ["s", "s>search", "s>rech", "search", "rech"], "text": "<b>Agent's branch:</b> the policy picks <i>search</i> or <i>recharge</i>, each with probability 0.5."},
    {"show": ["search", "rech", "search>l1", "search>l2", "rech>l3", "l1", "l2", "l3"], "text": "<b>Environment's branch:</b> each action has its own outcomes with probabilities p(s′, r | s, a): searching keeps the battery low with p = 0.6 (+2) or kills it with p = 0.4 (−3, rescued to high)."},
    {"show": ["l1", "l2", "l3"], "text": "Score each outcome as <b>r + γ V(s′)</b>, using the values of the next states: V(low) = 12.6, V(high) = 14.6."},
    {"show": ["l1", "l2", "search>l1", "search>l2", "search"], "text": "Average the outcomes of <i>search</i>: 0.6 × 13.34 + 0.4 × 10.14 = <b>12.06</b>. This is Q(low, search)."},
    {"show": ["l3", "rech>l3", "rech"], "text": "<i>Recharge</i> has a single outcome: Q(low, recharge) = <b>13.14</b>."},
    {"show": ["search", "rech", "s>search", "s>rech", "s"], "text": "Average over the policy: 0.5 × 12.06 + 0.5 × 13.14 = <b>12.6</b>, exactly V(low). The equation holds."}
  ],
  "narrow": {"cols": 3, "rows": 3, "rowHeight": 120, "pos": {"s": [1.25, 0], "search": [0.5, 1], "rech": [2, 1], "l1": [0, 2], "l2": [1, 2], "l3": [2, 2]}}
}
</script>
</div>

The values in that tree are consistent: plug $V(\text{low}) = 12.6$ and $V(\text{high}) = 14.6$ into the right-hand side and you get $12.6$ back. That is what it means for $V^\pi$ to **satisfy** the Bellman equation. A wrong guess (say all zeros) would not.

Now the same thing on the cliff world from lesson 1, with real numbers for every cell.

**Try it:** click any floor cell. The readout writes out its Bellman equation: one line per possible outcome, $p \times (r + \gamma V(s'))$, summed up and compared with the stored value. Then raise the slip chance.

<div id="viz-backup" class="viz"><div class="viz-title">🔍 Interactive: Bellman backup explorer</div></div>

!!! tip "What to notice"
    - **Edge policy, slip 0, cell (2,1)** (just above the cliff): only one outcome, so the equation is a single line: $V = -1 + 0.95 \times 5.72 = 4.44$.
    - **Raise the slip chance to 0.2.** The same cell now has three outcomes: $0.8 \times (-1 + 0.95 \times (-2.97)) + 0.1 \times (-20) + 0.1 \times (-1 + 0.95 \times (-6.78)) = -3.055 - 2.000 - 0.744 = -5.80$. The cliff line costs a full 2 points on its own, even though it only happens one time in ten.
    - **Every cell's residual is 0 or about $10^{-15}$** (floating-point rounding): the value table satisfies all 18 Bellman equations at once. That is the defining property of $V^\pi$.
    - **Random policy, start cell:** four actions with $\pi = 0.25$ each. One of them (→) walks straight into the cliff, which drags $V(\text{start})$ down to $-19.88$, the number you saw in lesson 1.

---

## Step 5 — The same idea for $Q^\pi$

In lesson 1 we met the **action-value** $Q^\pi(s, a)$: the expected return if you take action $a$ in $s$ **first**, then follow $\pi$. It is the middle layer of the backup diagram: the value of a node *after* the agent has chosen but *before* the environment has rolled the dice. Two short equations link $V$ and $Q$:

$$
Q^\pi(s, a) = \sum_{s', r} p(s', r \mid s, a)\,\big[\, r + \gamma\, V^\pi(s') \,\big]
\qquad\qquad
V^\pi(s) = \sum_a \pi(a \mid s)\, Q^\pi(s, a)
$$

The first is the environment's half of the tree, the second the agent's half. Substituting one into the other gives the **Bellman expectation equation for $Q^\pi$**:

$$
Q^\pi(s, a) = \sum_{s', r} p(s', r \mid s, a)\,\Big[\, r + \gamma \sum_{a'} \pi(a' \mid s')\, Q^\pi(s', a') \,\Big]
$$

Why bother with $Q$ when we have $V$? Because $Q$ **compares actions**. For the robot's coin-flip policy in state **low**:

| action | $Q^\pi(\text{low}, a)$ |
|---|---|
| search | 12.06 |
| wait | 12.34 |
| recharge | **13.14** |

The policy is averaging a 12.06 action with a 13.14 action and gets $V^\pi(\text{low}) = 12.6$. Recharging every time would clearly do better. Reading $Q$ and switching to the best action is exactly how **policy improvement** works in the next lesson. In the backup explorer above, the "→ Q = …" header of each action is this $Q^\pi(s, a)$.

!!! tip "What to notice"
    - $V$ needs the **model** $p$ to compare actions: you must look one step ahead. $Q$ has that lookahead built in, so you can pick the best action with a simple $\arg\max_a Q(s, a)$. That is why model-free methods like Q-learning learn $Q$, not $V$.
    - $Q^\pi(\text{low, wait}) = 12.34$ even though this policy never waits. $Q^\pi$ answers "what if I tried $a$ just once, then went back to $\pi$?"

---

## Step 6 — The Bellman optimality equations

So far the policy was given. Now we want the **best** one. Define the **optimal value functions** as the best value any policy can achieve:

$$
V^*(s) = \max_\pi V^\pi(s), \qquad Q^*(s, a) = \max_\pi Q^\pi(s, a)
$$

An optimal agent doesn't average over its actions; it takes the best one. So in the backup diagram, the **agent's branch becomes a max** while the environment's branch stays an average (you can't choose the dice):

$$
\boxed{\;V^*(s) = \max_a \sum_{s', r} p(s', r \mid s, a)\,\big[\, r + \gamma\, V^*(s') \,\big]\;}
$$

$$
\boxed{\;Q^*(s, a) = \sum_{s', r} p(s', r \mid s, a)\,\Big[\, r + \gamma \max_{a'} Q^*(s', a') \,\Big]\;}
$$

These are the **Bellman optimality equations**. Once you have $Q^*$, acting optimally is trivial: **be greedy**,

$$
\pi^*(s) = \arg\max_a Q^*(s, a).
$$

No lookahead and no model are needed. A one-step greedy choice is globally optimal because $Q^*$ already accounts for acting optimally ever after. For the robot, $Q^*(\text{low}, \cdot)$ is 13.32 (search), 13.76 (wait) and **14.17** (recharge), so $\pi^*$ recharges when low and searches when high, with $V^*(\text{high}) = 15.75$ and $V^*(\text{low}) = 14.17$. That is the *Cautious* preset in the robot widget.

**Try it:** here is $Q^*$ for the cliff world. Each cell is split into four triangles, one per action, coloured by $Q^*(s, a)$. The arrows mark the greedy action (two arrows mean a tie), and the orange line follows them from the start. Slide the slip chance slowly from 0 to 0.4.

<div id="viz-qstar" class="viz"><div class="viz-title">🧭 Interactive: optimal action-values Q* and the greedy policy</div></div>

!!! tip "What to notice"
    - **Slip 0:** the greedy path hugs the cliff and $V^*(\text{start}) = 2.05$, the same as the hand-written *Edge* policy from lesson 1. That policy was already optimal.
    - **The route flips between slip 0.10 and 0.15.** At 0.10 the path still hugs the edge ($V^* = -3.68$); at 0.15 it takes the top row ($V^* = -5.13$).
    - **Slip 0.2:** $V^*(\text{start}) = -6.30$, better than both lesson-1 policies (edge $-9.08$, safe $-6.97$). The main route is the safe detour, but $Q^*$ also knows the best move in cells you only reach by slipping.
    - **Slip 0.35 or more:** the greedy action at the start is **←**, into the wall! At slip 0.4, $Q^*(\text{start}, ←) = -10.34$ beats $Q^*(\text{start}, ↑) = -11.21$. Bumping the wall keeps you put, and a sideways slip then goes up (good) or down (also a wall), never into the cliff. Trying ↑ risks slipping right, into the cliff. Nobody wrote that trick in; it falls out of the equations.
    - The cliff triangles (pointing down from the edge row) stay deep red at every slip: $Q^*$ knows that action is a disaster, while $V^*$ would hide it behind the max.

---

## Step 7 — A system of equations you can solve

Look again at the Bellman expectation equation. There is one equation per state, and the unknowns are the values $V^\pi(s)$. The robot has two states, so two equations (coin-flip policy, $\gamma = 0.9$):

$$
\begin{aligned}
V(\text{high}) &= 0.7\,[2 + 0.9\,V(\text{high})] + 0.3\,[2 + 0.9\,V(\text{low})] \\
V(\text{low})  &= 0.5\,\big(0.6\,[2 + 0.9\,V(\text{low})] + 0.4\,[-3 + 0.9\,V(\text{high})]\big) + 0.5\,[0 + 0.9\,V(\text{high})]
\end{aligned}
$$

Every equation is **linear** in the unknowns. Collect the policy's average reward per state in a vector $r^\pi$ and its state-to-state transition probabilities in a matrix $P^\pi$, and all of them become one matrix equation:

$$
V^\pi = r^\pi + \gamma P^\pi V^\pi
\quad\Longleftrightarrow\quad
(I - \gamma P^\pi)\, V^\pi = r^\pi
\quad\Longleftrightarrow\quad
V^\pi = (I - \gamma P^\pi)^{-1} r^\pi.
$$

For $\gamma < 1$ the matrix $I - \gamma P^\pi$ is always invertible, so there is **exactly one** solution. NumPy solves it in one line:

```python
import itertools
import numpy as np

gamma = 0.9
states = ["high", "low"]
actions = {"high": ["search", "wait"], "low": ["search", "wait", "recharge"]}
# p(s', r | s, a) written as a list of (probability, next state, reward)
dynamics = {
    ("high", "search"):   [(0.7, "high", 2), (0.3, "low", 2)],
    ("high", "wait"):     [(1.0, "high", 1)],
    ("low", "search"):    [(0.6, "low", 2), (0.4, "high", -3)],   # battery dies: rescued
    ("low", "wait"):      [(1.0, "low", 1)],
    ("low", "recharge"):  [(1.0, "high", 0)],
}


def evaluate(policy):
    """Solve V = r_pi + gamma * P_pi V exactly. policy[s] = {action: probability}."""
    n = len(states)
    P = np.zeros((n, n))   # P[s, s'] = probability of moving s -> s' under the policy
    r = np.zeros(n)        # r[s]     = expected immediate reward in s under the policy
    for i, s in enumerate(states):
        for a, pa in policy[s].items():
            for p, s2, reward in dynamics[(s, a)]:
                P[i, states.index(s2)] += pa * p
                r[i] += pa * p * reward
    return np.linalg.solve(np.eye(n) - gamma * P, r)   # V = (I - gamma P)^-1 r


coin_flip = {"high": {"search": 1.0}, "low": {"search": 0.5, "recharge": 0.5}}
print("coin-flip policy: V =", evaluate(coin_flip).round(3))

# Bellman optimality by brute force: try every deterministic policy (2 x 3 = 6 of them)
for a_high, a_low in itertools.product(actions["high"], actions["low"]):
    V = evaluate({"high": {a_high: 1.0}, "low": {a_low: 1.0}})
    print(f"high: {a_high:6s}  low: {a_low:8s}  V = {V.round(3)}")
```

```text
coin-flip policy: V = [14.6 12.6]
high: search  low: search    V = [12.603  9.863]
high: search  low: wait      V = [12.703 10.   ]
high: search  low: recharge  V = [15.748 14.173]
high: wait    low: search    V = [10.     7.826]
high: wait    low: wait      V = [10. 10.]
high: wait    low: recharge  V = [10.  9.]
```

The first line reproduces the values from the backup diagram. The loop finds the optimal policy (search when high, recharge when low) by **brute force**: its values $[15.748, 14.173]$ are the highest in **both** states at once, and they match $V^*$ from Step 6.

So why do we need the rest of the course?

- **The optimality equation is not linear.** The $\max$ breaks the matrix trick, so there is no single `np.linalg.solve` for $V^*$.
- **Brute force explodes.** The robot has 6 deterministic policies. The tiny cliff world has 18 floor cells with 4 actions each: $4^{18} \approx 6.9 \times 10^{10}$ policies.
- **Solving is expensive for big problems.** A linear solve costs about $n^3$ operations for $n$ states. Backgammon has around $10^{20}$ states, Go more than $10^{170}$.
- **Usually we don't even know $p$.** The equations need the dynamics. A real agent only sees samples.

The next lesson, **dynamic programming**, attacks the first three problems: it solves both Bellman equations by repeating the backup over and over (*policy iteration*, *value iteration*). The lessons after that attack the last one by replacing the sum over $p$ with samples.

!!! tip "What to notice"
    - The Bellman expectation equation is a **linear system** with exactly one solution, $V^\pi$. Evaluating a policy, when you know the MDP, is "just" linear algebra.
    - The Bellman optimality equation also has exactly one solution, $V^*$, but you need an iterative method (or a search over policies) to find it.

---

## Putting it all together

<div class="viz diagram">
<script type="application/json">
{
  "title": "🗺️ From an MDP to optimal behaviour",
  "cols": 2,
  "rows": 5,
  "rowHeight": 92,
  "nodes": [
    {"id": "mdp", "col": 0.5, "row": 0, "icon": "🎲", "label": "<b>MDP</b>", "sub": "S, A, p(s′, r | s, a), γ", "role": "data", "span": 1.2, "maxWidth": 220, "note": "The rules of the world. Markov: the next state and reward depend only on the current state and action."},
    {"id": "pol", "col": 0, "row": 1, "icon": "🧭", "label": "Fix a policy π", "sub": "average over actions", "role": "model"},
    {"id": "best", "col": 1, "row": 1, "icon": "🏆", "label": "Ask for the best", "sub": "max over actions", "role": "opt"},
    {"id": "bexp", "col": 0, "row": 2, "label": "<b>Bellman expectation</b>", "sub": "V<sup>π</sup>, Q<sup>π</sup>", "role": "loss", "note": "V(s) = Σ π(a|s) Σ p [r + γ V(s′)]"},
    {"id": "bopt", "col": 1, "row": 2, "label": "<b>Bellman optimality</b>", "sub": "V*, Q*", "role": "loss", "note": "V*(s) = max_a Σ p [r + γ V*(s′)]"},
    {"id": "lin", "col": 0, "row": 3, "icon": "🧮", "label": "Linear system", "sub": "(I − γP<sup>π</sup>) V = r<sup>π</sup>", "role": "model", "note": "One np.linalg.solve when the MDP is small and known."},
    {"id": "dp", "col": 1, "row": 3, "icon": "🔁", "label": "Nonlinear (max)", "sub": "iterate: dynamic programming", "role": "model", "note": "Value iteration and policy iteration, the next lesson."},
    {"id": "cmp", "col": 0, "row": 4, "icon": "⚖️", "label": "Compare policies", "sub": "brute force: tiny MDPs only", "note": "Solve the linear system for every deterministic policy and keep the best, as in Step 7. Fine for 6 policies, hopeless for 4<sup>18</sup>."},
    {"id": "greedy", "col": 1, "row": 4, "icon": "✅", "label": "<b>Act greedily</b>", "sub": "π*(s) = argmax<sub>a</sub> Q*(s, a)", "role": "opt"}
  ],
  "edges": [
    {"from": "mdp", "to": "pol"},
    {"from": "mdp", "to": "best"},
    {"from": "pol", "to": "bexp"},
    {"from": "best", "to": "bopt"},
    {"from": "bexp", "to": "lin"},
    {"from": "bopt", "to": "dp"},
    {"from": "dp", "to": "greedy"},
    {"from": "lin", "to": "cmp"},
    {"from": "cmp", "to": "greedy"}
  ],
  "steps": [
    {"show": ["mdp"], "text": "Everything starts from the MDP: states, actions, dynamics p(s′, r | s, a) and the discount γ. The state must be Markov."},
    {"show": ["mdp", "mdp>pol", "pol", "pol>bexp", "bexp"], "text": "Fix a policy and the recursion G<sub>t</sub> = r<sub>t+1</sub> + γG<sub>t+1</sub> becomes the <b>Bellman expectation equation</b>: each value is a policy-weighted, probability-weighted one-step lookahead."},
    {"show": ["bexp", "bexp>lin", "lin"], "text": "It is linear, one equation per state, so for small known MDPs you can solve it exactly with linear algebra."},
    {"show": ["lin", "lin>cmp", "cmp", "cmp>greedy", "greedy"], "text": "For a tiny MDP you could even solve it for <b>every</b> deterministic policy and keep the best. The robot has 6; the cliff world already has about 7 × 10<sup>10</sup>."},
    {"show": ["mdp", "mdp>best", "best", "best>bopt", "bopt"], "text": "Ask for the best policy instead and the average over actions becomes a <b>max</b>: the <b>Bellman optimality equation</b> for V* and Q*."},
    {"show": ["bopt", "bopt>dp", "dp"], "text": "The max makes it nonlinear. Dynamic programming (next lesson) solves it by repeating backups until nothing changes."},
    {"show": ["dp", "dp>greedy", "greedy"], "text": "With Q* in hand, acting optimally is easy: in each state pick the action with the highest Q*."}
  ],
  "narrow": {"cols": 2, "rows": 5, "rowHeight": 88}
}
</script>
</div>

Here is the cliff world as an MDP in NumPy. `outcomes()` **is** the dynamics function $p(s', r \mid s, a)$, `evaluate()` solves the Bellman expectation equation as a linear system, and the residual check confirms that the solution satisfies the equation in every cell:

```python
import numpy as np

MAP = ["......",
       "..#...",
       "......",
       "SCCCCG"]
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]                 # up, right, down, left
cells = [(r, c) for r in range(4) for c in range(6) if MAP[r][c] in ".S"]   # non-terminal states
index = {s: i for i, s in enumerate(cells)}


def outcomes(s, a, slip):
    """p(s', r | s, a): list of (probability, next cell, reward, episode over?)."""
    result = []
    for p, d in [(1 - slip, a), (slip / 2, (a + 1) % 4), (slip / 2, (a + 3) % 4)]:
        r, c = s[0] + MOVES[d][0], s[1] + MOVES[d][1]
        if not (0 <= r < 4 and 0 <= c < 6) or MAP[r][c] == "#":
            r, c = s                                         # bump: stay put
        tile = MAP[r][c]
        reward = 10 if tile == "G" else -20 if tile == "C" else -1
        if p > 0:
            result.append((p, (r, c), reward, tile in "GC"))
    return result


def evaluate(policy, slip, gamma=0.95):
    """Bellman expectation equation as a linear system: (I - gamma P) V = r."""
    n = len(cells)
    P, r = np.zeros((n, n)), np.zeros(n)
    for s in cells:
        for a, pa in policy(s).items():
            for p, s2, reward, done in outcomes(s, a, slip):
                r[index[s]] += pa * p * reward
                if not done:                                 # terminal states have value 0
                    P[index[s], index[s2]] += pa * p
    return np.linalg.solve(np.eye(n) - gamma * P, r)


def q_values(V, s, slip, gamma=0.95):
    """One-step lookahead: Q(s, a) = sum over outcomes of p * (r + gamma * V(s'))."""
    return [sum(p * (reward + (0 if done else gamma * V[index[s2]]))
                for p, s2, reward, done in outcomes(s, a, slip)) for a in range(4)]


EDGE = ["↓↓→↓↓↓", "↓↓.↓↓↓", "→→→→→↓", "↑....."]
edge = lambda s: {"↑→↓←".index(EDGE[s[0]][s[1]]): 1.0}
random_policy = lambda s: {a: 0.25 for a in range(4)}

start = index[(3, 0)]
for slip in [0.0, 0.2]:
    for name, policy in [("edge", edge), ("random", random_policy)]:
        V = evaluate(policy, slip)
        # Bellman residual: does V really satisfy V(s) = sum_a pi(a|s) Q(s, a) in every cell?
        residual = max(abs(V[index[s]] - sum(pa * q_values(V, s, slip)[a] for a, pa in policy(s).items()))
                       for s in cells)
        print(f"slip={slip}  {name:6s}  V(start) = {V[start]:6.2f}   max Bellman residual = {residual:.1e}")

V = evaluate(edge, 0.2)
print("Q_edge(start, ·) at slip 0.2 [up, right, down, left]:", np.round(q_values(V, (3, 0), 0.2), 2))
```

```text
slip=0.0  edge    V(start) =   2.05   max Bellman residual = 4.4e-16
slip=0.0  random  V(start) = -19.88   max Bellman residual = 3.6e-15
slip=0.2  edge    V(start) =  -9.08   max Bellman residual = 1.8e-15
slip=0.2  random  V(start) = -19.88   max Bellman residual = 3.6e-15
Q_edge(start, ·) at slip 0.2 [up, right, down, left]: [ -9.08 -17.73 -10.67  -9.43]
```

These are the same values as the Monte Carlo averages in lesson 1 (2.05, −9.08, −19.88), now computed **exactly**, without playing a single episode. The last line is a hint of what comes next: at the start, the edge policy's own action "up" ($-9.08$) is already the best of the four. But run `q_values` on every cell and in 7 of the 18 cells another action beats the policy's choice (at (1,4), for example, → scores 4.71 against the policy's ↓ at 2.83). Switching to those better actions and repeating is *policy iteration*, in the next lesson.

---

## Common misconceptions

!!! warning "\"The Bellman equation is an algorithm.\""
    It is a **condition** that the true values must satisfy, like "$x^2 = 2$" is a condition that $\sqrt 2$ satisfies. Algorithms (dynamic programming, TD learning, Q-learning) are different ways of *finding* values that satisfy it.

!!! warning "\"Markov means the agent has no memory.\""
    Markov is a property of the **state**, not of the agent. If memory matters, put it into the state (stack frames, add velocity, keep the conversation history). Then the state is Markov again.

!!! warning "\"V*(s) is the value of the best action, so V* tells me what to do.\""
    $V^*$ tells you *how good* a state is, not *which action* gets you there. To act on $V^*$ you still need a one-step lookahead with the model $p$. $Q^*$ has the action built in: $\arg\max_a Q^*(s, a)$ needs no model.

!!! warning "\"The optimal policy maximises the next reward.\""
    The greedy policy maximises $Q^*$, which already includes all future rewards. In the cliff world at high slip the best first move is to walk into a wall: the worst-looking action locally, but the safest long-run choice.

---

## Check yourself

??? question "1. A state in a card game is \"the cards in my hand\". Why might this not be Markov, and how would you fix it?"
    The cards already played matter: they change which cards are left in the deck and so what can happen next. A state with only your hand forgets them. Fix it by adding the history that matters (the cards already played, or counts of what remains) to the state.

??? question "2. An episode gives rewards r₁ = 1, r₂ = 0, r₃ = 4 and then ends. With γ = 0.5, compute G₂, G₁ and G₀ using the recursion."
    Work backwards. $G_3 = 0$ (the episode is over). $G_2 = r_3 + \gamma G_3 = 4$. $G_1 = r_2 + \gamma G_2 = 0 + 0.5 \times 4 = 2$. $G_0 = r_1 + \gamma G_1 = 1 + 0.5 \times 2 = $ **2**. Check: $1 + 0.5 \cdot 0 + 0.25 \cdot 4 = 2$.

??? question "3. In a state s, a policy picks action A with probability 0.25 and B with probability 0.75. Q(s, A) = 8 and Q(s, B) = 4. What is V^π(s)? What is the best possible value of s if these Q-values were Q* values?"
    $V^\pi(s) = 0.25 \times 8 + 0.75 \times 4 = 2 + 3 = $ **5**. If they were $Q^*$ values, $V^*(s) = \max(8, 4) = $ **8**, achieved by always taking A.

??? question "4. What changes between the Bellman expectation equation and the Bellman optimality equation, and why does that stop us from solving the second one with a single linear solve?"
    The average over the agent's actions, $\sum_a \pi(a \mid s)\,\cdot$, is replaced by a $\max_a$. The environment's average over outcomes stays. A max is not linear, so the equations can no longer be written as $(I - \gamma P)V = r$ with a fixed $P$; you need an iterative method such as value iteration (next lesson).

??? question "5. In the Q* widget at slip 0.4, why is walking into the wall the best first move from the start?"
    Moving left from the start bumps the wall, so the intended move keeps you put. The sideways slips go up (towards safety) or down (another wall, stay put). No outcome ends in the cliff. Moving up instead has a 0.2 chance of slipping right, straight into the cliff for −20. Paying −1 per step to wait for a safe slip is worth more: $Q^*(\text{start}, ←) = -10.34$ vs $Q^*(\text{start}, ↑) = -11.21$.

---

## Further reading

- Richard Sutton & Andrew Barto, [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) (2nd ed., free online), Chapter 3: Finite Markov Decision Processes. The recycling robot is Example 3.3.
- David Silver, [*UCL Course on RL*](https://www.davidsilver.uk/teaching/), Lecture 2: Markov Decision Processes (video + slides), with very clear backup diagrams.
- OpenAI, [*Spinning Up: Key Concepts in RL*](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#bellman-equations), the section on Bellman equations.
- Richard Bellman, *Dynamic Programming* (1957), where the equations (and the name) come from.

**Next lessons:** [Dynamic programming](../dynamic-programming/index.md) · [Monte Carlo & TD learning](../monte-carlo-td-learning/index.md)

<script src="viz/mdp-bellman-equations.js"></script>
