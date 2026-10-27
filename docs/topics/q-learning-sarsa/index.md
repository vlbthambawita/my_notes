# Q-Learning and SARSA

!!! abstract "TL;DR"
    With no model of the world, an agent can still learn to act well. It learns **action values** $Q(s,a)$ from its own experience, one step at a time, and acts by picking the action with the biggest $Q$. **SARSA** updates $Q$ towards the value of the action it will *actually* take next (**on-policy**). **Q-learning** updates towards the value of the *best* next action, whatever it actually does (**off-policy**). On the cliff world from lesson 1 this small difference matters: Q-learning learns the short route along the cliff edge but keeps falling off while it explores; SARSA learns a safer detour and collects more reward while learning.

**Level:** beginner–intermediate · **Prerequisites:** [What is RL?](../reinforcement-learning-intro/index.md) and [Monte Carlo & TD learning](../monte-carlo-td-learning/index.md) (the TD update)

---

## Why should I care?

Until now our agents have either been *told* their policy (lesson 1) or *evaluated* a fixed one. This lesson is the payoff: an agent that starts knowing nothing about the cliff world, not where the goal is and not where the cliff is, and **finds a good route by itself**, only by trying.

- **Q-learning** (Watkins, 1989) is one of the most famous algorithms in RL. Put a neural network in place of the table and you get **Deep Q-Networks**, the method that learned to play Atari games from pixels (next lesson).
- The difference between **on-policy** and **off-policy** learning that you meet here comes back everywhere: PPO (used to fine-tune chatbots) is on-policy, DQN and SAC are off-policy. Knowing which one you have tells you whether you can reuse old data, and how your agent behaves while it learns.

---

## Step 1 — From V to Q: without a model, learn action values

The previous lesson learned the value $V(s)$ of each state from experience with TD learning. That tells you how good it is to **be** somewhere. But to **act**, you need to compare actions. With $V$ alone, choosing an action means looking one step ahead:

$$
\pi(s) = \arg\max_a \; \sum_{s'} p(s' \mid s, a)\,\big[\, r + \gamma V(s') \,\big]
$$

That needs $p(s' \mid s, a)$: a **model** of where each action leads. A model-free agent doesn't have one. The fix is to learn the **action value** $Q(s,a)$ directly, "how good is it to take action $a$ in state $s$, and carry on from there". Then acting needs no model at all:

$$
\pi(s) = \arg\max_a Q(s, a)
$$

The price is four numbers per cell instead of one. The board below shows the **optimal** action values $Q^*(s,a)$ of the cliff world, as four triangles per cell.

!!! note "No discount in this lesson"
    We use $\gamma = 1$ here, as Sutton & Barto do for their cliff-walking example. Every episode ends at the goal or the cliff, so the sums stay finite, and $Q$ is then simply the **expected total reward** from that point. That makes the numbers easy to read: the 7-step route along the edge is worth $6 \times (-1) + 10 = +4$.

<div id="viz-qvalues" class="viz"><div class="viz-title">🔢 Interactive: four numbers per cell, Q*(s, a)</div></div>

!!! tip "What to notice"
    - At the start cell, $Q^*(s, \uparrow) = 4$ (the edge route) and $Q^*(s, \rightarrow) = -20$ (straight into the cliff). The best action is just the biggest number.
    - Just above the cliff at row 2, col 1, the four values are $\uparrow 4$, $\rightarrow 6$, $\downarrow -20$, $\leftarrow 4$. The greedy policy hugs the edge, because with a perfect, non-exploring agent the edge is safe.
    - $V^*(s)$ is just the largest of the four: $V^*(s) = \max_a Q^*(s,a)$.
    - Raise the slip chance: the values near the cliff drop, and from slip 0.1 on the arrows lead away from the edge (the optimal route becomes the 11-step detour). $Q^*$ already includes the risks of the **environment**. Remember this; it will matter in Step 5.

---

## Step 2 — SARSA: learn from the move you actually make next

The TD update from the previous lesson nudged $V(s)$ towards a one-step target, $r + \gamma V(s')$. Do exactly the same with $Q$. After taking action $a$ in state $s$, getting reward $r$ and landing in $s'$, the agent **picks its next action $a'$** with its usual ε-greedy policy, and uses $Q(s', a')$ as its guess of the future:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \,\big[\, \underbrace{r + \gamma\, Q(s', a')}_{\text{TD target}} - Q(s,a) \,\big]
$$

The bracket is the **TD error** $\delta_t$; $\alpha$ is the step size. Then the agent really does take $a'$, and the loop repeats. The update uses five things, $(s, a, r, s', a')$, which is where the name comes from: **S**tate, **A**ction, **R**eward, **S**tate, **A**ction.

<div class="viz diagram">
<script type="application/json">
{
  "title": "🔁 One SARSA step: (s, a, r, s′, a′)",
  "cols": 3,
  "rows": 2,
  "rowHeight": 110,
  "nodes": [
    {"id": "s", "col": 0, "row": 0, "icon": "📍", "label": "<b>S</b>tate s", "sub": "where the agent is", "role": "data"},
    {"id": "a", "col": 1, "row": 0, "icon": "🎲", "label": "<b>A</b>ction a", "sub": "already chosen, ε-greedy", "role": "model"},
    {"id": "r", "col": 2, "row": 0, "icon": "🪙", "label": "<b>R</b>eward r", "sub": "−1, +10 or −20", "role": "data"},
    {"id": "s2", "col": 2, "row": 1, "icon": "📍", "label": "next <b>S</b>tate s′", "role": "data"},
    {"id": "a2", "col": 1, "row": 1, "icon": "🎲", "label": "next <b>A</b>ction a′", "sub": "chosen ε-greedy, before updating", "role": "model", "note": "This is the key: SARSA picks its real next action first, then uses Q(s′, a′) in the target. Sometimes a′ is a random exploratory move."},
    {"id": "upd", "col": 0, "row": 1, "icon": "🧠", "label": "Update Q(s, a)", "sub": "towards r + γ Q(s′, a′)", "role": "loss"}
  ],
  "edges": [
    {"from": "s", "to": "a"},
    {"from": "a", "to": "r"},
    {"from": "r", "to": "s2"},
    {"from": "s2", "to": "a2"},
    {"from": "a2", "to": "upd"},
    {"from": "upd", "to": "s", "label": "s←s′, a←a′"}
  ],
  "steps": [
    {"show": ["s"], "text": "The agent is in state s."},
    {"show": ["s>a", "a"], "text": "It takes action a, which it chose with its ε-greedy policy: usually the best-known action, sometimes a random one."},
    {"show": ["a>r", "r", "r>s2", "s2"], "text": "The environment returns a reward r and the next state s′."},
    {"show": ["s2>a2", "a2"], "text": "Now the agent <b>chooses its next action a′</b>, with the same ε-greedy policy. It will really take it."},
    {"show": ["a2>upd", "upd"], "text": "Update: Q(s, a) moves a step α towards r + γ Q(s′, a′). If a′ was a risky random move, the target is low."},
    {"show": ["upd>s", "s"], "text": "Shift along one step (s ← s′, a ← a′) and repeat until the episode ends. At the goal or the cliff, the target is just r."}
  ],
  "narrow": {"cols": 2, "rows": 3, "pos": {"s": [0, 0], "a": [1, 0], "r": [1, 1], "s2": [1, 2], "a2": [0, 2], "upd": [0, 1]}}
}
</script>
</div>

!!! tip "What to notice"
    - SARSA is the TD update from the previous lesson, with $Q(s,a)$ in place of $V(s)$. Nothing else is new.
    - The target uses $a'$, the action the agent **really takes next**, including random exploratory moves. So SARSA learns the value of the policy it is actually following, exploration and all. That is what **on-policy** means.
    - At a terminal state (goal or cliff) there is no future, so the target is just $r$.

---

## Step 3 — Q-learning: learn about the best move, whatever you do next

Q-learning changes one thing. Instead of the action the agent will take next, it uses the **best** action in $s'$:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \,\big[\, r + \gamma \max_{a'} Q(s', a') - Q(s,a) \,\big]
$$

The agent still **behaves** ε-greedily, so it keeps exploring. But its target always assumes "from the next state on, I will play greedily". So Q-learning learns $Q^*$, the values of the **greedy** policy, even though it never follows that policy exactly. Learning about one policy (greedy) while following another (ε-greedy) is called **off-policy** learning.

A worked example shows the difference. Suppose the agent stands just above the cliff (row 2, col 1), steps right, gets $r = -1$ and lands on (row 2, col 2). Say its $Q$ already equals the $Q^*$ values from Step 1, where $Q^*(s', \cdot)$ is $\uparrow 6$, $\rightarrow 7$, $\downarrow -20$, $\leftarrow 5$. With $\gamma = 1$ and $\varepsilon = 0.3$:

| Method | Next action used in the target | Target |
|---|---|---|
| Q-learning | the best one, $\rightarrow$ (whatever the agent does next) | $-1 + 7 = 6$ |
| SARSA, if $a' = \rightarrow$ (greedy) | $\rightarrow$ | $-1 + 7 = 6$ |
| SARSA, if $a' = \downarrow$ (random move into the cliff) | $\downarrow$ | $-1 - 20 = -21$ |
| SARSA, averaged over ε-greedy $a'$ | | $3.75$ |

Q-learning's target ignores the agent's own clumsiness. SARSA's target includes it: on average, standing next to the cliff is worth less to an agent that sometimes steps off at random. Over many updates, SARSA's values for the edge cells end up lower, and that is what pushes it away from the edge.

!!! tip "What to notice"
    - The two updates differ in one term: $Q(s', a')$ versus $\max_{a'} Q(s', a')$.
    - With $\varepsilon = 0$ they are identical, because then $a'$ *is* the greedy action.
    - Neither needs a model: both learn from single transitions $(s, a, r, s')$ as they happen.

---

## Step 4 — Watch them learn the cliff

Here is lesson 1's cliff world again, but now nobody tells the agent anything. All $Q$ values start at 0. Press **Train** and watch an agent learn by trial and error. Then switch the algorithm and compare.

<div id="viz-train" class="viz"><div class="viz-title">🧗 Interactive: SARSA and Q-learning learn the cliff world</div></div>

!!! tip "What to notice"
    1. **The first episodes are chaos.** The agent wanders, bumps into walls and falls off the cliff. Each fall turns the triangle pointing at the cliff red, so the cliff is learned one fall at a time.
    2. **Q-learning settles on the edge route.** With the defaults (α = 0.2, ε = 0.3), give it a few hundred episodes: the green dashed line runs along row 2, right next to the cliff. That is the optimal route (7 steps, +4).
    3. **SARSA settles on a detour.** Reset with SARSA: its dashed line usually goes up to the top row, the long way round (11 steps, 0). Its values next to the cliff stay redder, because they include its own random steps into the cliff.
    4. **Training versus testing.** Press **Run greedy policy (ε = 0)** after training both. Q-learning's route now scores +4, SARSA's only 0. But while *training*, the readout's "fell off the cliff" count is much higher for Q-learning. Which one is "better" depends on whether you care about the reward collected while learning.
    5. Try **slip = 0.2**. Now even a greedy agent can slide off the edge, so Q-learning's $Q^*$ also learns to keep away. Q-learning accounts for the environment's randomness, just not for its own exploration.

---

## Step 5 — Who collects more reward while learning?

One run of one agent is noisy. To compare fairly, the widget below trains **200 SARSA agents and 200 Q-learning agents** (each with its own random seed) for 500 episodes and plots the **average total reward per episode** as they learn. This is the classic cliff-walking experiment from Sutton & Barto's textbook, on our smaller map.

<div id="viz-curves" class="viz"><div class="viz-title">📈 Interactive: reward per episode while learning (200 runs each)</div></div>

!!! tip "What to notice"
    - With $\varepsilon = 0.3$, **SARSA's curve ends above Q-learning's.** In our Python check (200 runs, α = 0.2, ε = 0.3, γ = 1, 500 episodes) the average reward per episode over the last 100 episodes was **−6.80 for SARSA and −8.63 for Q-learning**. In those episodes 39% of Q-learning's episodes ended in the cliff, against 14% for SARSA (SARSA still falls sometimes: the start cell itself is next to the cliff). Your widget numbers will differ slightly, since it runs its own seeds.
    - **Yet Q-learning learned the better policy.** All 200 Q-learning agents ended with the 7-step edge route (+4 when run greedily). 196 of the 200 SARSA agents ended with the 11-step detour (0 when run greedily).
    - **Now press ε = 0.1.** On this small map the result flips: Q-learning −0.05, SARSA −2.35. Our cliff is only 4 cells long, so with 10% random moves Q-learning falls in only about 1 episode in 8 (12.5%), and the short edge route pays even while exploring. Working out the exact average for a fixed route (a policy evaluation, as in the dynamic programming lesson) gives −0.12 for the ε-greedy edge route and −1.97 for the detour at ε = 0.1, but −8.76 and −7.46 at ε = 0.3. The two cross near ε = 0.2. Sutton & Barto's original map has a 10-cell cliff, which is why their famous figure shows SARSA winning already at ε = 0.1.
    - SARSA at ε = 0.1 is also a lesson in noise: the edge is the better ε-greedy route there, but one unlucky fall makes a big update with α = 0.2, and 112 of the 200 SARSA agents still ended on the detour.

---

## Step 6 — On-policy versus off-policy

Every RL agent has two roles for a policy:

- The **behaviour policy** is the one that actually picks the actions and generates the experience (here: ε-greedy).
- The **target policy** is the one whose values the agent is learning (the one the TD target "assumes" for the future).

**On-policy** methods like SARSA use the same policy for both. **Off-policy** methods like Q-learning learn about one policy (greedy) from data generated by another (ε-greedy).

<div class="viz diagram">
<script type="application/json">
{
  "title": "🔀 Same experience, two different targets",
  "cols": 3,
  "rows": 4,
  "rowHeight": 100,
  "nodes": [
    {"id": "beh", "col": 1, "row": 0, "icon": "🎲", "label": "<b>Behaviour policy</b>", "sub": "ε-greedy: acts and explores", "role": "model"},
    {"id": "exp", "col": 1, "row": 1, "icon": "📼", "label": "Experience", "sub": "s, a, r, s′", "role": "data"},
    {"id": "sar", "col": 0, "row": 2, "icon": "🟦", "label": "<b>SARSA</b> target", "sub": "r + γ Q(s′, a′)", "role": "loss", "note": "a′ is the next action the ε-greedy policy really takes. The target policy is the behaviour policy itself: on-policy."},
    {"id": "ql", "col": 2, "row": 2, "icon": "🟧", "label": "<b>Q-learning</b> target", "sub": "r + γ max Q(s′, ·)", "role": "loss", "note": "The target policy is greedy, but the behaviour policy is ε-greedy: off-policy."},
    {"id": "sres", "col": 0, "row": 3, "icon": "🛡️", "label": "Q of the ε-greedy policy", "sub": "safe detour, fewer falls"},
    {"id": "qres", "col": 2, "row": 3, "icon": "🏁", "label": "Q* of the greedy policy", "sub": "edge route, more falls"}
  ],
  "edges": [
    {"from": "beh", "to": "exp"},
    {"from": "exp", "to": "sar"},
    {"from": "exp", "to": "ql"},
    {"from": "sar", "to": "sres", "label": "learns"},
    {"from": "ql", "to": "qres", "label": "learns"}
  ],
  "steps": [
    {"show": ["beh", "beh>exp", "exp"], "text": "Both agents behave the same way: ε-greedy. They generate the same kind of experience (s, a, r, s′)."},
    {"show": ["exp", "exp>sar", "sar"], "text": "SARSA's target uses the next action its ε-greedy policy really takes. Target policy = behaviour policy: <b>on-policy</b>."},
    {"show": ["exp", "exp>ql", "ql"], "text": "Q-learning's target uses the best next action, as if it will act greedily from now on. Target policy ≠ behaviour policy: <b>off-policy</b>."},
    {"show": ["sar>sres", "sres"], "text": "SARSA learns the values of the policy it follows, exploration included. Near the cliff that policy is risky, so it prefers the detour."},
    {"show": ["ql>qres", "qres"], "text": "Q-learning learns Q*, the values of the greedy policy. Its greedy route is optimal, but while it still explores, it falls more often."}
  ],
  "narrow": {"cols": 2, "rows": 4, "rowHeight": 95, "pos": {"beh": [0.5, 0], "exp": [0.5, 1], "sar": [0, 2], "ql": [1, 2], "sres": [0, 3], "qres": [1, 3]}}
}
</script>
</div>

Why would you want off-policy learning? Because the data can come from **anywhere**: an exploring agent, an old version of the agent, a human demonstrator. That is what lets DQN reuse old experience from a replay buffer (next lesson). On-policy methods have to learn from data that their *current* policy generated.

Two loose ends:

- **Expected SARSA** replaces the sampled $Q(s', a')$ with its average over the policy: $r + \gamma \sum_{a'} \pi(a' \mid s')\, Q(s', a')$. It learns the same thing as SARSA but with less noise per update, since it doesn't depend on which $a'$ happened to be drawn. (That is the 3.75 in the table of Step 3.) In our check, with ε = 0.3, α = 0.1 and 3000 episodes, all 20 Expected SARSA runs ended on the same 11-step detour as SARSA. With a greedy $\pi$ in the sum, Expected SARSA *is* Q-learning.
- **If ε shrinks to zero, they meet.** SARSA's caution only exists because of exploration. Let ε decay over time and SARSA's target becomes Q-learning's. In our check (ε = 0.3/(1 + episode/20), α = 0.2, 2000 episodes), all 100 Q-learning runs and 86 of 100 SARSA runs ended with the edge route. (The rest of SARSA's runs had stopped exploring before they had re-checked the edge: decaying ε too fast has its own cost.)

!!! tip "What to notice"
    - "On-policy" and "off-policy" are about **which policy the target describes**, not about how the agent behaves. Both of our agents behave identically.
    - Neither is simply better. Off-policy finds the optimal greedy policy and can reuse any data. On-policy optimises the reward **you actually get while acting the way you act**, which matters when mistakes during learning are costly (a real robot near a real cliff).

---

## Putting it all together

<div class="viz diagram">
<script type="application/json">
{
  "title": "🧩 TD control: SARSA and Q-learning in one loop",
  "cols": 3,
  "rows": 2,
  "rowHeight": 110,
  "nodes": [
    {"id": "reset", "col": 0, "row": 0, "icon": "🔁", "label": "New episode", "shape": "pill", "sub": "s ← start"},
    {"id": "act", "col": 1, "row": 0, "icon": "🎲", "label": "Take action a", "sub": "ε-greedy on Q(s, ·)", "role": "model"},
    {"id": "env", "col": 2, "row": 0, "icon": "🌍", "label": "Environment", "sub": "returns r and s′", "role": "data"},
    {"id": "tgt", "col": 2, "row": 1, "icon": "🎯", "label": "TD target", "sub": "SARSA or Q-learning", "role": "loss", "note": "SARSA: r + γ Q(s′, a′) with the real next action a′. Q-learning: r + γ max Q(s′, ·). Terminal s′: just r."},
    {"id": "upd", "col": 1, "row": 1, "icon": "🧠", "label": "Update Q(s, a)", "sub": "Q += α (target − Q)", "role": "loss"},
    {"id": "end", "col": 0, "row": 1, "label": "Episode over?", "shape": "decision", "role": "grad"}
  ],
  "edges": [
    {"from": "reset", "to": "act"},
    {"from": "act", "to": "env"},
    {"from": "env", "to": "tgt"},
    {"from": "tgt", "to": "upd"},
    {"from": "upd", "to": "end"},
    {"from": "end", "to": "act", "label": "no"},
    {"from": "end", "to": "reset", "label": "yes"}
  ],
  "steps": [
    {"show": ["reset"], "text": "Start an episode. Q is a table with one number per (state, action), all 0 at the very beginning."},
    {"show": ["reset>act", "act"], "text": "Pick an action ε-greedily from the current Q: usually the biggest Q, sometimes a random one."},
    {"show": ["act>env", "env"], "text": "The environment answers with a reward r and the next state s′. No model needed: we only use what actually happened."},
    {"show": ["env>tgt", "tgt"], "text": "Build the TD target. <b>SARSA</b>: r + γ Q(s′, a′), with the next action it will really take. <b>Q-learning</b>: r + γ max Q(s′, ·)."},
    {"show": ["tgt>upd", "upd"], "text": "Move Q(s, a) a step of size α towards the target."},
    {"show": ["upd>end", "end", "end>act", "act"], "text": "Not at the goal or the cliff yet? Carry on from s′."},
    {"show": ["end>reset", "reset"], "text": "Episode over: start again, keeping the Q table. After enough episodes, acting greedily on Q is a good policy."}
  ],
  "narrow": {"cols": 2, "rows": 3, "pos": {"reset": [0, 0], "act": [1, 0], "env": [1, 1], "tgt": [1, 2], "upd": [0, 2], "end": [0, 1]}}
}
</script>
</div>

Both algorithms in plain Python, on lesson 1's cliff world. The only difference between them is one line, the target:

```python
import random
from collections import Counter

MAP = ["......",
       "..#...",
       "......",
       "SCCCCG"]
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]          # up, right, down, left


class CliffWorld:
    """Lesson 1's grid world: −1 per step, +10 at the goal, −20 for the cliff."""

    def __init__(self, slip=0.0, seed=0):
        self.slip, self.rng = slip, random.Random(seed)

    def reset(self):
        self.pos = (3, 0)
        return self.pos

    def step(self, a):
        if self.rng.random() < self.slip:              # slide sideways
            a = (a + self.rng.choice([1, 3])) % 4
        r, c = self.pos[0] + MOVES[a][0], self.pos[1] + MOVES[a][1]
        if 0 <= r < 4 and 0 <= c < 6 and MAP[r][c] != "#":
            self.pos = (r, c)
        tile = MAP[self.pos[0]][self.pos[1]]
        if tile == "G":
            return self.pos, +10, True
        if tile == "C":
            return self.pos, -20, True
        return self.pos, -1, False


def epsilon_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return rng.randrange(4)                        # explore
    best = max(Q[s])
    return rng.choice([a for a in range(4) if Q[s][a] == best])   # exploit (break ties randomly)


def train(algo, episodes=500, alpha=0.2, eps=0.3, gamma=1.0, seed=0):
    rng = random.Random(seed)
    env = CliffWorld(seed=seed)
    Q = {(r, c): [0.0] * 4 for r in range(4) for c in range(6)}
    rewards = []                                       # total reward of every training episode
    for _ in range(episodes):
        s = env.reset()
        a = epsilon_greedy(Q, s, eps, rng)
        total, done = 0, False
        while not done:
            s2, r, done = env.step(a)
            total += r
            a2 = epsilon_greedy(Q, s2, eps, rng)       # the action we will really take next
            if done:
                target = r                             # no future after the goal or the cliff
            elif algo == "sarsa":
                target = r + gamma * Q[s2][a2]         # on-policy: value of the action we WILL take
            else:
                target = r + gamma * max(Q[s2])        # off-policy: value of the BEST action
            Q[s][a] += alpha * (target - Q[s][a])      # TD update
            s, a = s2, a2
        rewards.append(total)
    return Q, rewards


def greedy_route(Q):
    """Follow the learned greedy policy (ε = 0) from the start, without slipping."""
    env, s, done, total, path = CliffWorld(), None, False, 0, []
    s = env.reset()
    while not done and len(path) < 50:
        s, r, done = env.step(Q[s].index(max(Q[s])))
        total += r
        path.append(s)
    return f"{len(path)} steps, return {total:+d}"


for eps in [0.3, 0.1]:
    print(f"ε = {eps}")
    for algo in ["sarsa", "q-learning"]:
        last100, routes = [], Counter()
        for seed in range(200):
            Q, rewards = train(algo, eps=eps, seed=seed)
            last100.append(sum(rewards[-100:]) / 100)
            routes[greedy_route(Q)] += 1
        print(f"  {algo:10s}  reward per episode (last 100) = {sum(last100) / 200:6.2f}"
              f"   greedy route: {routes.most_common(1)[0][0]} in {routes.most_common(1)[0][1]}/200 runs")
```

```text
ε = 0.3
  sarsa       reward per episode (last 100) =  -6.80   greedy route: 11 steps, return +0 in 196/200 runs
  q-learning  reward per episode (last 100) =  -8.63   greedy route: 7 steps, return +4 in 200/200 runs
ε = 0.1
  sarsa       reward per episode (last 100) =  -2.35   greedy route: 11 steps, return +0 in 112/200 runs
  q-learning  reward per episode (last 100) =  -0.05   greedy route: 7 steps, return +4 in 200/200 runs
```

(It takes about 20 seconds in plain Python.) The table `Q` has $4 \times 6 \times 4 = 96$ entries. That is fine here, but a game screen has far too many states for a table. Replacing the table with a neural network that outputs $Q(s, \cdot)$ is the idea behind Deep Q-Networks, the next lesson.

---

## Common misconceptions

!!! warning "\"Q-learning is better than SARSA because it finds the optimal policy.\""
    Q-learning finds the optimal **greedy** policy, the best one if you stop exploring. If the agent keeps exploring (as many real systems do while they learn), SARSA can collect more reward, as the ε = 0.3 curves show. "Better" depends on whether you score the learned policy or the learning itself.

!!! warning "\"SARSA avoids the cliff because it knows the cliff is dangerous.\""
    Both algorithms learn that the cliff is bad: both have $Q(s, \downarrow) \approx -20$ next to it. SARSA avoids the *edge* because its values include its **own future random moves**. Set ε to 0 and that reason disappears.

!!! warning "\"Off-policy means the agent doesn't need to explore.\""
    Q-learning still needs exploration to see the states and actions it learns about. Off-policy only means the **target** assumes a different policy (greedy) from the one that **behaves** (ε-greedy). No exploration, no data about the alternatives.

!!! warning "\"SARSA learns the optimal policy, just more slowly.\""
    With a fixed ε, SARSA learns the best **ε-greedy** policy, which is not the optimal policy when exploration is risky. It only reaches the optimal policy if ε decays towards zero over time.

---

## Check yourself

??? question "1. Why do model-free agents learn Q(s, a) instead of V(s)?"
    To act with $V$ you have to look one step ahead: "which action leads to the best next state?" That needs a model $p(s' \mid s, a)$. With $Q$ you just pick $\arg\max_a Q(s,a)$, no model required.

??? question "2. Write down the SARSA and Q-learning updates. Which single term differs?"
    SARSA: $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]$, where $a'$ is the next action the agent really takes. Q-learning: the same, with $\max_{a'} Q(s',a')$ in place of $Q(s',a')$.

??? question "3. An agent is next to the cliff. Its next action a′ happens to be a random step into the cliff, with Q(s′, a′) = −20, while max Q(s′, ·) = 7. With r = −1 and γ = 1, what are the SARSA and Q-learning targets?"
    SARSA: $-1 + (-20) = -21$. Q-learning: $-1 + 7 = 6$. SARSA "feels" its own random mistake; Q-learning assumes it will act greedily next.

??? question "4. In the learning-curves widget at ε = 0.3, Q-learning earns less reward per episode than SARSA, yet its greedy route scores +4 against SARSA's 0. How can both be true?"
    The curves measure reward **while exploring** (ε = 0.3). Q-learning's route runs along the cliff edge, where random moves often fall off (39% of its late training episodes ended in the cliff in our Python check). Its greedy route is optimal **once exploration stops**. SARSA's detour is worse without exploration, but safer with it.

??? question "5. Is Q-learning on-policy or off-policy? What practical advantage does that give?"
    Off-policy: it learns the values of the greedy policy while following an ε-greedy one. Because the target doesn't depend on how the data was generated, it can learn from old experience, other agents or demonstrations. DQN uses this to learn from a replay buffer of past transitions.

---

## Further reading

- Richard Sutton & Andrew Barto, [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) (2nd ed.), Sections 6.4–6.6: SARSA, Q-learning, Expected SARSA, and Example 6.6 *Cliff Walking*, the experiment reproduced here.
- Christopher Watkins & Peter Dayan, [*Q-learning*](https://link.springer.com/article/10.1007/BF00992698), Machine Learning 8, 1992. The convergence proof for Q-learning.
- Gavin Rummery & Mahesan Niranjan, *On-line Q-learning using connectionist systems*, Cambridge tech. report, 1994. The paper that introduced SARSA (under another name).
- Harm van Seijen et al., *A theoretical and empirical analysis of Expected Sarsa*, IEEE ADPRL 2009.
- David Silver, [*UCL Course on RL*](https://www.davidsilver.uk/teaching/), Lecture 5: Model-Free Control.

**Next lessons:** [Deep Q-Networks](../deep-q-networks/index.md)

<script src="viz/q-learning-sarsa.js"></script>
