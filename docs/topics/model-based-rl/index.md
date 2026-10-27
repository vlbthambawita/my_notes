# Model-Based RL and Planning: Dyna, MCTS and AlphaZero

!!! abstract "TL;DR"
    A **model** of the environment predicts what happens next: given a state and an action, it returns the next state and the reward. An agent that has a model can **plan**: it can try things out *in its head* instead of in the real world. **Dyna-Q** learns a model from its own experience and uses it to replay imagined steps, so it needs far fewer real steps. **Monte Carlo Tree Search** (MCTS) plans at the moment of choosing a move, by growing a search tree of possible futures. **AlphaZero** guides that search with a neural network trained by self-play, and **MuZero** even learns the model itself.

**Level:** intermediate · **Prerequisites:** [Q-learning & SARSA](../q-learning-sarsa/index.md), [Actor-critic](../actor-critic/index.md). The UCB rule from [Multi-armed bandits](../multi-armed-bandits/index.md) comes back in Step 5.

---

## Why should I care?

Every method so far in this series was **model-free**. Q-learning, DQN, policy gradients and actor-critic all learn *what to do* straight from experience, and they need a lot of it. The original DQN trained on 50 million frames per Atari game, roughly 38 days of non-stop play for a human.

Real experience is often expensive or slow: a robot wears out, a chemistry experiment takes a day, a patient cannot be used for trial and error. People don't learn like that either. Before you cross a busy road, you *imagine* what would happen if you stepped out now. That imagined experience is free.

Model-based RL gives agents the same ability. It is behind some of RL's biggest successes:

- **AlphaGo** beat Go world champion Lee Sedol in 2016, and **AlphaZero** (2017) beat the strongest chess, shogi and Go programs. Both search ahead, using the game's rules as the model.
- **MuZero** (2019) reached the same level *without being told the rules*, by learning its own model, and also mastered Atari.
- Planning with learned models is how many robot-control systems squeeze good behaviour out of a few hours of real data.

---

## Step 1 — What is a model?

In the [first lesson](../reinforcement-learning-intro/index.md) the environment was a black box: you send an action, it sends back a reward and a next state. A **model** is the agent's own copy of that black box. It answers the question *"if I am in state $s$ and take action $a$, what happens?"*

$$
\text{model}(s, a) \;\to\; \hat{p}(s', r \mid s, a)
$$

The hat on $\hat p$ means "estimated": the model is the agent's *belief* about the true dynamics $p(s', r \mid s, a)$ from the [MDP lesson](../mdp-bellman-equations/index.md). There are two kinds:

| | **Distribution model** | **Sample model** |
|---|---|---|
| What it gives | all outcomes with their probabilities | one random outcome, like a dice roll |
| Example | "80% → next cell, 10% → slide up, 10% → slide into the cliff" | "this time you slid up" |
| Used by | dynamic programming (it sums over all outcomes) | Dyna, MCTS (they just need samples) |

And two places a model can come from:

- **Given:** the rules are known exactly. Board games, physics simulators, the cliff world when we read its code.
- **Learned:** the agent builds it from its own experience. That is what Dyna (Step 3) and MuZero (Step 6) do.

With a model, the agent can do **planning**: compute or improve a policy using imagined experience instead of real experience. [Dynamic programming](../dynamic-programming/index.md) was already planning: value iteration loops over the model's probabilities and never touches the real environment. The new question in this lesson is how to plan when the model has to be **learned**, or when the state space is far too big to sweep through.

!!! tip "What to notice"
    **Model-free** agents learn *what to do* (Q-values or a policy). **Model-based** agents also learn *how the world works*, and then work out what to do. The second kind of knowledge can be reused: if the goal moves, a model of the walls and the cliff is still correct.

---

## Step 2 — Learning a model by counting

In a small, discrete world, learning a model is just **counting**. Every time the agent takes action $a$ in state $s$ and sees reward $r$ and next state $s'$, it adds one to a counter $n(s, a, s', r)$. The estimated probability of an outcome is how often it happened:

$$
\hat{p}(s', r \mid s, a) = \frac{n(s, a, s', r)}{n(s, a)}
$$

where $n(s, a)$ is how often the agent tried $a$ in $s$ at all. This is the same "average of what you saw" idea as the Monte Carlo value estimates, applied to *outcomes* instead of returns.

Here is what that looks like on the slippery cliff world (slip chance 0.2), for one state-action pair: standing at row 2, column 1 (just above the cliff) and moving **right**. The true answer is 80% to the next cell, 10% sliding up and 10% sliding down into the cliff, an expected reward of $0.9 \cdot (-1) + 0.1 \cdot (-20) = -2.9$.

| Tries $n(s, a)$ | → next cell (−1) | ↑ slide up (−1) | ↓ into the cliff (−20) | estimated expected reward |
|---|---|---|---|---|
| 10 | 10 (1.00) | 0 (0.00) | 0 (0.00) | −1.00 |
| 100 | 86 (0.86) | 6 (0.06) | 8 (0.08) | −2.52 |
| 1000 | 779 (0.78) | 104 (0.10) | 117 (0.12) | −3.22 |
| true | 0.80 | 0.10 | 0.10 | −2.90 |

!!! tip "What to notice"
    - After 10 tries the model has never seen a slip, so it believes that walking along the cliff edge is **perfectly safe**. A planner that trusts this model will happily plan routes along the edge. Small amounts of data give confident, wrong models.
    - With more data the counts approach the true probabilities. In a deterministic world (slip 0), a single try per state-action pair gives the exact model.
    - For big or continuous state spaces (images, robot joints) you can't count. You train a neural network $\hat f_\phi(s, a) \to (\hat s', \hat r)$ by supervised learning on the agent's transitions instead. The idea stays the same.

---

## Step 3 — Dyna-Q: learn, model, plan

**Dyna** (Sutton, 1990) is the simplest way to mix the two worlds. It is plain [Q-learning](../q-learning-sarsa/index.md) with one addition. After every real step, the agent:

1. updates $Q$ from the real transition, exactly as Q-learning does (**direct RL**),
2. records the transition in its model (**model learning**),
3. then performs $n$ extra Q-learning updates on transitions *imagined* by the model (**planning**).

Each planning update picks a state-action pair $(s, a)$ the agent has already tried, asks the model for an outcome $(r, s')$, and applies the usual update:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \big[\, r + \gamma \max_{a'} Q(s', a') - Q(s, a) \,\big]
$$

That's all. Real and imagined experience go through the **same** update rule; the model is just a free source of extra experience.

<div class="viz diagram">
<script type="application/json">
{
  "title": "🔁 One real step of Dyna-Q",
  "cols": 3,
  "rows": 2,
  "rowHeight": 110,
  "nodes": [
    {"id": "act", "col": 0, "row": 0, "icon": "🧭", "label": "Pick action a", "sub": "ε-greedy on Q(s, ·)", "role": "model"},
    {"id": "real", "col": 1, "row": 0, "icon": "🌍", "label": "<b>Real world</b> returns r, s′", "sub": "costly: one real step", "role": "data", "note": "The only box that touches the real environment. Everything else happens inside the agent."},
    {"id": "direct", "col": 2, "row": 0, "icon": "📈", "label": "Direct RL", "sub": "Q-learning update on (s, a, r, s′)", "role": "loss"},
    {"id": "model", "col": 2, "row": 1, "icon": "📒", "label": "<b>Model learning</b>", "sub": "count (s, a) → (r, s′)", "role": "model", "note": "In a tabular world the model is just a table of counts. With images or continuous states it would be a neural network."},
    {"id": "plan", "col": 1, "row": 1, "icon": "💭", "label": "<b>Planning</b> × n", "sub": "Q-updates on imagined steps", "role": "loss", "note": "Repeat n times: pick an (s, a) seen before, ask the model for (r, s′), apply the Q-learning update. These steps are free: no real environment needed."},
    {"id": "next", "col": 0, "row": 1, "icon": "⏭️", "label": "s ← s′", "sub": "next real step", "shape": "pill"}
  ],
  "edges": [
    {"from": "act", "to": "real"},
    {"from": "real", "to": "direct"},
    {"from": "direct", "to": "model"},
    {"from": "model", "to": "plan"},
    {"from": "plan", "to": "next"},
    {"from": "next", "to": "act", "label": "repeat"}
  ],
  "steps": [
    {"show": ["act"], "text": "In state s, pick an action ε-greedily from the current Q-values, as in Q-learning."},
    {"show": ["act>real", "real"], "text": "Take it in the <b>real</b> environment. This is the expensive part: one real step of experience."},
    {"show": ["real>direct", "direct"], "text": "<b>Direct RL:</b> the usual Q-learning update from this one real transition."},
    {"show": ["direct>model", "model"], "text": "<b>Model learning:</b> remember what happened. In a table, add 1 to the count of (s, a) → (r, s′)."},
    {"show": ["model>plan", "plan"], "text": "<b>Planning:</b> n times, pick a state-action pair you have tried before, let the model imagine its outcome, and do a Q-learning update on it. No real step needed."},
    {"show": ["plan>next", "next", "next>act", "act"], "text": "Move on to the next real state and repeat. With n = 0 this is exactly Q-learning."}
  ],
  "narrow": {"cols": 2, "rows": 3, "pos": {"act": [0, 0], "real": [1, 0], "direct": [1, 1], "model": [1, 2], "plan": [0, 2], "next": [0, 1]}}
}
</script>
</div>

**Try it:** below is one Dyna-Q agent on the (non-slippery) cliff world from lesson 1, with $\alpha = 0.1$, $\gamma = 0.95$, $\varepsilon = 0.1$ and all Q-values starting at 0. Pick $n$, then run one episode at a time. The orange ticks are the model: one tick on the side of a cell for each action the agent has tried there.

<div id="viz-dyna" class="viz"><div class="viz-title">🎚️ Interactive: inside a Dyna-Q agent</div></div>

!!! tip "What to notice"
    - **The first episode is a stumble for every n.** The model starts empty and the agent has no idea where the goal is, so the first episode usually ends in the cliff after a few random steps.
    - **Planning spreads news backwards.** Each real step changes one Q-value. Each planning step can update *any* pair the model knows, so the +10 of the goal travels back along known paths without the agent walking them again. Compare the green cells after 10 episodes with n = 0 and with n = 50: over 100 Python runs, the median number of cells with a positive value was 1 for n = 0, 3 for n = 5 and 8 for n = 50.
    - **The model fills up fast.** After a few episodes most cells have ticks on all four sides. In this deterministic world one visit is enough to know an outcome exactly.
    - **Watch the dashed green line.** With n = 50 the greedy route usually becomes the optimal 7-step path along the cliff edge within a handful of episodes. With n = 0 it takes dozens.

---

## Step 4 — How much does planning buy?

One episode is noisy. To compare fairly, we run many independent agents for each value of $n$ and ask: **after how many episodes is the greedy route the optimal 7-step route?** The widget runs 50 agents per curve right in your browser.

<div id="viz-dyna-curves" class="viz"><div class="viz-title">📈 Interactive: Dyna-Q learning curves for different n</div></div>

The Python version of the same experiment (the code is in [Putting it all together](#putting-it-all-together)), with 100 runs per setting, gives:

| planning steps $n$ | median episodes until the optimal route | runs on it after 10 episodes | median **real steps** until the optimal route |
|---|---|---|---|
| 0 (plain Q-learning) | 32 | 0% | about 840 |
| 5 | 10 | 34% | 211 |
| 50 | 7 | 84% | 129 |

!!! tip "What to notice"
    - **Planning is a trade.** With $n = 50$ the agent needs about **6× fewer real steps** than plain Q-learning, but it does 50 extra updates per real step. Dyna pays with computation (cheap) to save experience (expensive).
    - **Returns diminish.** Going from 0 to 5 helps more than going from 5 to 50. Once the model has been replayed enough for the Q-values to settle, more planning adds little. Try your own $n$ with the slider.
    - **Look closely at n = 50 late on.** Its curve flattens a little below 100%. In the Python runs, 5 of the 100 agents with $n = 50$ were still on a 9-step route after 60 episodes. That's a real weakness of planning with a learned model, and Step 7 explains it.

!!! note "Dyna and the replay buffer"
    The [DQN](../deep-q-networks/index.md) replay buffer is close to Dyna: it stores real transitions and replays them for extra updates. A replay buffer is a *non-parametric* model that can only replay exactly what happened. A real model can also answer questions about state-action pairs in new combinations, and (if it is a network) generalise to states it has never seen.

---

## Step 5 — Planning at decision time: Monte Carlo Tree Search

Dyna plans **in the background**: it improves a Q-table for all states, whenever it has spare time. A different style is to plan **at decision time**: when you have to move *now*, focus all your thinking on the current state and the futures that can follow from it. That is what chess players do.

The obvious way is a **search tree**: the current state is the root, each legal move is a branch, each reply is a branch below that, and so on. For tic-tac-toe you could search the whole tree. For Go there are about $10^{170}$ positions, so you must be selective. **Monte Carlo Tree Search** (MCTS) grows the tree one path at a time, spending more effort on moves that look promising. Every node stores two numbers: a **visit count** $N$ and a **total value** $W$, so its average value is $Q = W / N$.

Each **simulation** has four phases:

<div class="viz diagram">
<script type="application/json">
{
  "title": "🌳 One MCTS simulation, repeated K times",
  "cols": 3,
  "rows": 2,
  "rowHeight": 115,
  "nodes": [
    {"id": "root", "col": 0, "row": 0, "icon": "📍", "label": "Root = the current state", "shape": "pill", "role": "data"},
    {"id": "sel", "col": 1, "row": 0, "icon": "🔽", "label": "<b>1. Selection</b>", "sub": "walk down: highest UCT score", "role": "model", "note": "UCT = Q + c·√(ln N_parent / N_child): good average value, or not tried much yet."},
    {"id": "exp", "col": 2, "row": 0, "icon": "🌱", "label": "<b>2. Expansion</b>", "sub": "add one new child node", "role": "opt"},
    {"id": "sim", "col": 2, "row": 1, "icon": "🎲", "label": "<b>3. Simulation</b>", "sub": "random play-out to the end", "role": "data", "note": "Also called a rollout. It uses the model (the game rules) to play random moves until the game ends. AlphaZero replaces it with a value network."},
    {"id": "back", "col": 1, "row": 1, "icon": "🔼", "label": "<b>4. Backup</b>", "sub": "N += 1, W += result on the path", "role": "loss", "maxWidth": 150},
    {"id": "move", "col": 0, "row": 1, "icon": "✅", "label": "Play the most-visited move", "shape": "pill", "role": "opt", "maxWidth": 150}
  ],
  "edges": [
    {"from": "root", "to": "sel"},
    {"from": "sel", "to": "exp"},
    {"from": "exp", "to": "sim"},
    {"from": "sim", "to": "back"},
    {"from": "back", "to": "sel", "label": "repeat"},
    {"from": "back", "to": "move", "label": "after K"}
  ],
  "steps": [
    {"show": ["root"], "text": "Start from the current position. At first the tree is just this one node."},
    {"show": ["root>sel", "sel"], "text": "<b>Selection:</b> from the root, keep choosing the child with the highest UCT score until you reach a node that still has untried moves."},
    {"show": ["sel>exp", "exp"], "text": "<b>Expansion:</b> add one new child for an untried move. The tree grows by one node per simulation."},
    {"show": ["exp>sim", "sim"], "text": "<b>Simulation:</b> from the new node, play random moves to the end of the game, using the rules as the model. The result is +1, 0 or −1."},
    {"show": ["sim>back", "back"], "text": "<b>Backup:</b> walk back up the path, adding 1 to N and the result to W of every node, each from the view of the player who made that move."},
    {"show": ["back>sel", "sel"], "text": "Repeat. Good moves get a high Q and are selected more, so the tree grows deepest where it matters."},
    {"show": ["back>move", "move"], "text": "After K simulations, play the child of the root with the <b>most visits</b>. Visit counts are steadier than averages based on a few samples."}
  ],
  "narrow": {"cols": 2, "rows": 4, "pos": {"root": [0, 0], "sel": [0, 1], "exp": [1, 1], "sim": [1, 2], "back": [0, 2], "move": [0, 3]}}
}
</script>
</div>

The **selection** rule is the heart of MCTS. At each node it treats the children like the arms of a [multi-armed bandit](../multi-armed-bandits/index.md) and uses the UCB rule, which in a tree is called **UCT** (Upper Confidence bounds applied to Trees):

$$
\text{UCT}(\text{child}) = \underbrace{\frac{W}{N}}_{\text{exploit: average result}} + \underbrace{c \sqrt{\frac{\ln N_{\text{parent}}}{N}}}_{\text{explore: bonus for few visits}}
$$

The first term favours moves that have worked well. The second is large for moves that have hardly been tried, and shrinks as $N$ grows. With $c = \sqrt 2$ this balances exploring and exploiting in the same way as UCB does for bandits.

**Try it:** X to move in the position at the top. One move wins by force; the others don't. Press **Next phase** a few times to watch single simulations, then **+10** and **+100** to see where the visits go.

<div id="viz-mcts" class="viz"><div class="viz-title">🌳 Interactive: Monte Carlo Tree Search, step by step</div></div>

!!! tip "What to notice"
    - **The winning move is "left".** It makes two threats at once (the left column and the middle row), and O can only block one. Exact minimax search agrees: *left* is a win (+1), *top-left*, *top* and *centre* are draws (0), and *bottom* loses (−1).
    - **Early on, random play-outs mislead.** In our Python run with seed 0, after 25 simulations MCTS preferred *top-left* (7 visits) and even rated the losing move *bottom* at +0.25: random opponents don't punish mistakes. After 100 simulations it preferred *centre*.
    - **Then the tree takes over.** As the tree grows deeper below *left*, the search sees that every O reply loses. In the same run, *left* had 80% of the visits after 400 simulations ($Q = +0.91$) and 91% after 1000.
    - **More thinking, better moves.** Over 100 Python runs with different seeds, MCTS chose *left* in 23 runs with K = 25, 36 with K = 50, 67 with K = 100, 97 with K = 200 and all 100 with K = 400.
    - **Watch the UCT column.** Moves with few visits get a large bonus, so even bad moves are re-checked now and then. The bonus is why the search keeps looking at *left* long enough to discover it is winning.

---

## Step 6 — AlphaGo, AlphaZero and MuZero: search + deep learning

Random play-outs are a weak judge of a position, and in Go a game lasts 200+ moves, so a random play-out tells you almost nothing. The **AlphaGo** family made MCTS strong by adding a neural network $f_\theta(s) = (\mathbf{p}, v)$ with two heads:

- a **policy head** $\mathbf{p}$: prior probabilities for each move, "which moves would a strong player consider?"
- a **value head** $v$: a prediction of the final result from this position, like the critic in [actor-critic](../actor-critic/index.md).

The network changes two phases of MCTS:

1. **Selection** uses the prior $P(s, a)$ from the policy head, so the search mostly looks at moves the network likes (the **PUCT** rule):
   $$
   \text{score}(s, a) = Q(s, a) + c\, P(s, a)\, \frac{\sqrt{N(s)}}{1 + N(s, a)}
   $$
2. **Simulation is dropped.** Instead of a random play-out, the new leaf is scored by the value head $v$. One network call replaces hundreds of random moves.

How is the network trained? By **self-play**: the agent plays games against itself, choosing each move with MCTS. The search makes the agent stronger than its raw network, so the search results are better training targets than the network's own outputs:

<div class="viz diagram">
<script type="application/json">
{
  "title": "♟️ AlphaZero: self-play, search and learning",
  "cols": 3,
  "rows": 2,
  "rowHeight": 115,
  "nodes": [
    {"id": "net", "col": 0, "row": 0, "icon": "🧠", "label": "<b>Network</b> f<sub>θ</sub>(s)", "sub": "move priors p, value v", "role": "model"},
    {"id": "mcts", "col": 1, "row": 0, "icon": "🌳", "label": "<b>MCTS</b> guided by p, v", "sub": "e.g. 800 simulations", "role": "loss", "note": "Selection uses the PUCT rule with the network's priors; leaves are scored by the value head instead of a random play-out."},
    {"id": "playmove", "col": 2, "row": 0, "icon": "♟️", "label": "Play a move", "sub": "∝ visit counts π", "role": "model", "note": "The visit counts of the root's children, normalised, form an improved policy π. The move is sampled from it (early in the game) or the most-visited move is played."},
    {"id": "result", "col": 2, "row": 1, "icon": "🏁", "label": "Game ends", "sub": "result z = +1, 0 or −1", "role": "data"},
    {"id": "store", "col": 1, "row": 1, "icon": "🗄️", "label": "Store (s, π, z)", "sub": "for every position", "role": "data"},
    {"id": "train", "col": 0, "row": 1, "icon": "🎯", "label": "<b>Train θ</b>", "sub": "p → π, v → z", "role": "loss", "note": "Loss = (z − v)² − π·log p + weight decay. The policy head learns to predict the search's choices, the value head the game's result."}
  ],
  "edges": [
    {"from": "net", "to": "mcts"},
    {"from": "mcts", "to": "playmove"},
    {"from": "playmove", "to": "result", "label": "until end"},
    {"from": "result", "to": "store"},
    {"from": "store", "to": "train"},
    {"from": "train", "to": "net", "label": "new θ"}
  ],
  "steps": [
    {"show": ["net"], "text": "Start with a network f<sub>θ</sub> that outputs move priors p and a value v. At first it knows nothing: random weights."},
    {"show": ["net>mcts", "mcts"], "text": "At every move, run MCTS. The priors p steer the selection; the value v replaces the random play-out."},
    {"show": ["mcts>playmove", "playmove"], "text": "The root's visit counts give an improved policy π. Play a move from it."},
    {"show": ["playmove>result", "result"], "text": "Keep going, the agent playing both sides, until the game ends with a result z."},
    {"show": ["result>store", "store"], "text": "Every position s of the game becomes a training example (s, π, z): what the search preferred, and who won."},
    {"show": ["store>train", "train"], "text": "Train the network: the policy head imitates the search (p → π), the value head predicts the result (v → z)."},
    {"show": ["train>net", "net"], "text": "The better network makes the next searches stronger, which gives better targets. Search and learning pull each other up."}
  ],
  "narrow": {"cols": 2, "rows": 3, "pos": {"net": [0, 0], "mcts": [1, 0], "playmove": [1, 1], "result": [1, 2], "store": [0, 2], "train": [0, 1]}}
}
</script>
</div>

This is a form of **policy iteration** (from the [dynamic programming lesson](../dynamic-programming/index.md)): MCTS is the *policy improvement* step, and training the network to imitate it makes the improvement permanent.

The family in short:

| System | Model used for search | Network trained from | Games |
|---|---|---|---|
| **AlphaGo** (2016) | the game rules | human expert games, then self-play | Go |
| **AlphaGo Zero** (2017) | the game rules | self-play only, no human data | Go |
| **AlphaZero** (2017) | the game rules | self-play only | Go, chess, shogi |
| **MuZero** (2019) | a **learned** model | self-play and real experience | Go, chess, shogi, Atari |

**MuZero** goes one step further. It doesn't get the rules. It learns three networks: a *representation* that turns the observation into a hidden state, a *dynamics* network that predicts the next hidden state and reward from a hidden state and an action (the learned model), and the same *prediction* network for $\mathbf{p}$ and $v$. MCTS then searches entirely inside the learned hidden space. The learned model never has to reconstruct the screen or the board; it only has to predict what matters for planning: rewards, values and policies.

!!! tip "What to notice"
    The network alone is like fast intuition; MCTS is slow deliberate thinking that checks the intuition. Training on the search results turns slow thinking into better intuition, and the loop repeats.

---

## Step 7 — When the model is wrong

A model is only useful if it is right where the planner looks. Three things go wrong in practice:

- **Errors compound.** A one-step model that is a little off feeds its own prediction back in as the next state, so mistakes add up over a long imagined trajectory. If each predicted step is right with probability 0.95, a 10-step imagined rollout is entirely right with probability $0.95^{10} \approx 0.60$, and a 50-step one only $0.95^{50} \approx 0.08$. This is why methods such as MBPO (model-based policy optimisation) only imagine **short** rollouts branching off real states.
- **The planner exploits model errors.** A planner searches for the action with the best *predicted* outcome. If the model is wrong somewhere in a flattering way, that is exactly where the planner will go, like the 10-try model in Step 2 that thought the cliff edge was perfectly safe.
- **The model only knows what you tried.** Dyna-Q's planning only replays state-action pairs it has seen. In all 5 stuck runs from Step 4, the agent had never tried moving right from one or two cells near the end of the cliff edge, so its model knew nothing there and the planner kept polishing a 9-step detour that it *did* know. **Dyna-Q+** fixes this with a small exploration bonus for state-action pairs not tried for a long time.

!!! note "Model-based or model-free?"
    Model-based methods use data far more efficiently and can reuse a model for new goals. Model-free methods are simpler and don't suffer from model bias: their errors come only from noise. Many modern systems combine both, as Dyna does: plan where the model is trusted, learn directly from real experience everywhere.

---

## Putting it all together

<div class="viz diagram">
<script type="application/json">
{
  "title": "🗺️ The methods in this lesson",
  "cols": 4,
  "rows": 3,
  "rowHeight": 105,
  "nodes": [
    {"id": "mb", "col": 1.5, "row": 0, "icon": "🧠", "label": "<b>Model-based RL</b>", "sub": "plan with p̂(s′, r | s, a)", "span": 1.6, "maxWidth": 240, "role": "loss"},
    {"id": "given", "col": 0.5, "row": 1, "label": "<b>Model given</b>", "sub": "rules, simulator", "role": "data"},
    {"id": "learned", "col": 2.5, "row": 1, "label": "<b>Model learned</b>", "sub": "from experience", "role": "model"},
    {"id": "mcts", "col": 0, "row": 2, "label": "MCTS", "sub": "plan at decision time", "role": "data"},
    {"id": "az", "col": 1, "row": 2, "label": "AlphaZero", "sub": "MCTS + network", "role": "data"},
    {"id": "mu", "col": 2, "row": 2, "label": "MuZero", "sub": "MCTS in a learned model", "role": "model"},
    {"id": "dyna", "col": 3, "row": 2, "label": "Dyna-Q", "sub": "plan in the background", "role": "model"}
  ],
  "edges": [
    {"from": "mb", "to": "given"},
    {"from": "mb", "to": "learned"},
    {"from": "given", "to": "mcts"},
    {"from": "mcts", "to": "az"},
    {"from": "az", "to": "mu"},
    {"from": "learned", "to": "dyna"}
  ],
  "steps": [
    {"show": ["mb", "mb>given", "mb>learned", "given", "learned"], "text": "First question: is the model <b>given</b> (game rules, a physics simulator) or must it be <b>learned</b> from experience?"},
    {"show": ["learned", "learned>dyna", "dyna"], "text": "<b>Dyna-Q</b> learns a table model and plans <b>in the background</b>: extra Q-updates on imagined steps after every real step."},
    {"show": ["given", "given>mcts", "mcts"], "text": "<b>MCTS</b> uses a given model to plan <b>at decision time</b>: grow a search tree from the current state, then play the most-visited move."},
    {"show": ["mcts>az", "az"], "text": "<b>AlphaZero</b> adds a policy/value network that guides the search, and trains it by self-play on the search's results."},
    {"show": ["az>mu", "mu", "learned"], "text": "<b>MuZero</b> keeps AlphaZero's search but drops the rules: it searches inside a model it has learned itself."}
  ],
  "narrow": {"cols": 2, "rows": 4, "rowHeight": 90, "pos": {"mb": [0.5, 0], "given": [0, 1], "learned": [1, 1], "mcts": [0, 2], "dyna": [1, 2], "az": [0, 3], "mu": [1, 3]}}
}
</script>
</div>

Here is Dyna-Q on the cliff world in plain Python, with the model as a table of counts. Set `n_planning = 0` and it is ordinary Q-learning:

```python
import random
from statistics import mean, median

MAP = ["......",
       "..#...",
       "......",
       "SCCCCG"]
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]          # up, right, down, left
START = (3, 0)


def env_step(s, a):
    """The real environment (the agent does NOT get to read this code)."""
    r, c = s[0] + MOVES[a][0], s[1] + MOVES[a][1]
    if not (0 <= r < 4 and 0 <= c < 6) or MAP[r][c] == "#":
        r, c = s                                      # bumped a wall or edge: stay put
    tile = MAP[r][c]
    if tile == "G":
        return (r, c), +10, True
    if tile == "C":
        return (r, c), -20, True
    return (r, c), -1, False


def greedy(Q, s, rng):
    best = max(Q[s])
    return rng.choice([a for a in range(4) if Q[s][a] == best])   # break ties at random


def dyna_q(n_planning, episodes=60, alpha=0.1, gamma=0.95, eps=0.1, seed=0):
    rng = random.Random(seed)
    Q = {(r, c): [0.0] * 4 for r in range(4) for c in range(6)}
    model = {}                       # (s, a) -> {(s', r, done): how often we saw it}
    history = []                     # after each episode: is the greedy route the 7-step one?

    def q_update(s, a, r, s2, done):
        target = r if done else r + gamma * max(Q[s2])
        Q[s][a] += alpha * (target - Q[s][a])

    for _ in range(episodes):
        s, done, t = START, False, 0
        while not done and t < 1000:
            # 1. act in the REAL world (ε-greedy) and learn from it: plain Q-learning
            a = rng.randrange(4) if rng.random() < eps else greedy(Q, s, rng)
            s2, r, done = env_step(s, a)
            q_update(s, a, r, s2, done)
            # 2. learn the MODEL: count what happened
            counts = model.setdefault((s, a), {})
            counts[(s2, r, done)] = counts.get((s2, r, done), 0) + 1
            # 3. PLAN: n extra Q-learning updates on experience imagined by the model
            seen = list(model)
            for _ in range(n_planning):
                ps, pa = rng.choice(seen)                       # a state-action we have tried
                outcomes = model[(ps, pa)]
                (ps2, pr, pdone), = rng.choices(list(outcomes), weights=list(outcomes.values()))
                q_update(ps, pa, pr, ps2, pdone)
            s, t = s2, t + 1
        history.append(greedy_route_length(Q) == 7)
    return history


def greedy_route_length(Q):
    """Follow the greedy policy (no exploration) from the start; return its number of steps."""
    s = START
    for t in range(1, 30):
        s, r, done = env_step(s, Q[s].index(max(Q[s])))
        if done:
            return t if r > 0 else None
    return None


for n in [0, 5, 50]:
    runs = [dyna_q(n, seed=seed) for seed in range(100)]
    first = [h.index(True) + 1 if True in h else 61 for h in runs]
    print(f"n = {n:2d}: optimal route found after {median(first):4.1f} episodes (median of 100 runs); "
          f"runs on it after 10 episodes: {mean(h[9] for h in runs):.0%}, after 60: {mean(h[-1] for h in runs):.0%}")
```

```text
n =  0: optimal route found after 32.0 episodes (median of 100 runs); runs on it after 10 episodes: 0%, after 60: 100%
n =  5: optimal route found after 10.0 episodes (median of 100 runs); runs on it after 10 episodes: 34%, after 60: 100%
n = 50: optimal route found after  7.0 episodes (median of 100 runs); runs on it after 10 episodes: 84%, after 60: 95%
```

And a compact MCTS for tic-tac-toe. The "model" is just the rules (`play`, `moves`, `winner`); `minimax` checks the answer by searching the whole game tree:

```python
import math
import random

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
NAMES = ["top-left", "top", "top-right", "left", "centre", "right", "bottom-left", "bottom", "bottom-right"]


# ---- The model: the rules of tic-tac-toe. A board is a 9-character string like "..O..XX.O".
def winner(b):
    for i, j, k in LINES:
        if b[i] != "." and b[i] == b[j] == b[k]:
            return b[i]
    return None

def to_move(b):
    return "X" if b.count("X") == b.count("O") else "O"

def moves(b):
    return [] if winner(b) else [i for i in range(9) if b[i] == "."]

def play(b, i):
    return b[:i] + to_move(b) + b[i + 1:]


class Node:
    def __init__(self, board, parent=None, move=None):
        self.board, self.parent, self.move = board, parent, move
        self.children, self.untried = [], moves(board)
        self.N, self.W = 0, 0.0          # visit count, total value (for the player who moved INTO this node)


def uct(child, c=math.sqrt(2)):
    return child.W / child.N + c * math.sqrt(math.log(child.parent.N) / child.N)


def mcts(board, simulations, rng):
    root = Node(board)
    for _ in range(simulations):
        node = root
        # 1. Selection: walk down the tree, always taking the child with the highest UCT score
        while not node.untried and node.children:
            node = max(node.children, key=uct)
        # 2. Expansion: add one new child for a move we have never tried here
        if node.untried:
            m = node.untried.pop(rng.randrange(len(node.untried)))
            child = Node(play(node.board, m), node, m)
            node.children.append(child)
            node = child
        # 3. Simulation: play random moves until the game ends
        b = node.board
        while moves(b):
            b = play(b, rng.choice(moves(b)))
        w = winner(b)
        # 4. Backup: every node on the path gets +1 / 0 / -1 from the view of the player who moved into it
        while node:
            mover = "O" if to_move(node.board) == "X" else "X"
            node.N += 1
            node.W += 0 if w is None else (1 if w == mover else -1)
            node = node.parent
    return root


def minimax(b):
    """Exact value for the player to move (+1 win, 0 draw, -1 loss). Fine for a game this small."""
    if winner(b):
        return -1                        # the previous player just won
    if not moves(b):
        return 0
    return max(-minimax(play(b, m)) for m in moves(b))


board = "..O..XX.O"                      # X to move
print("minimax:", {NAMES[m]: -minimax(play(board, m)) for m in moves(board)})
for K in [25, 100, 400, 1000]:
    root = mcts(board, K, random.Random(0))
    best = max(root.children, key=lambda ch: ch.N)
    stats = ", ".join(f"{NAMES[ch.move]} {ch.N} ({ch.W / ch.N:+.2f})"
                      for ch in sorted(root.children, key=lambda ch: ch.move))
    print(f"K={K:4d}: plays {NAMES[best.move]:7s} share {best.N / K:4.0%} | N (W/N): {stats}")

for K in [25, 50, 100, 200, 400]:
    right = sum(max(mcts(board, K, random.Random(s)).children, key=lambda ch: ch.N).move == 3
                for s in range(100))
    print(f"K={K:3d}: picks the winning move in {right}/100 runs")
```

```text
minimax: {'top-left': 0, 'top': 0, 'left': 1, 'centre': 0, 'bottom': -1}
K=  25: plays top-left share  28% | N (W/N): top-left 7 (+0.43), top 6 (+0.33), left 4 (+0.00), centre 4 (+0.00), bottom 4 (+0.25)
K= 100: plays centre  share  31% | N (W/N): top-left 21 (+0.33), top 16 (+0.25), left 20 (+0.40), centre 31 (+0.45), bottom 12 (+0.08)
K= 400: plays left    share  80% | N (W/N): top-left 21 (+0.33), top 17 (+0.24), left 319 (+0.91), centre 31 (+0.45), bottom 12 (+0.08)
K=1000: plays left    share  91% | N (W/N): top-left 23 (+0.30), top 23 (+0.26), left 907 (+0.96), centre 34 (+0.44), bottom 13 (+0.00)
K= 25: picks the winning move in 23/100 runs
K= 50: picks the winning move in 36/100 runs
K=100: picks the winning move in 67/100 runs
K=200: picks the winning move in 97/100 runs
K=400: picks the winning move in 100/100 runs
```

To go from here to AlphaZero, replace the random play-out with a call to a value network, add the network's prior $P(s, a)$ to the selection score (the PUCT rule), and train the network on the visit counts and game results from self-play. Open-source versions to read: [`alpha-zero-general`](https://github.com/suragnair/alpha-zero-general) (small and readable) and DeepMind's [`mctx`](https://github.com/google-deepmind/mctx) (MCTS for AlphaZero and MuZero in JAX).

---

## Common misconceptions

!!! warning "\"Model-based means the agent is given the rules.\""
    Sometimes it is (MCTS and AlphaZero use the game rules), but often the model is **learned** from experience, as in Dyna and MuZero. "Model-based" only means the agent has *some* model of $p(s', r \mid s, a)$ and uses it to plan.

!!! warning "\"More planning is always better.\""
    Planning is only as good as the model. With a learned model, extra planning can make the agent *more* confident in the model's mistakes, as in the stuck Dyna runs and the "safe" cliff edge of Step 2. The gain from more planning also shrinks fast (0 → 5 steps helped more than 5 → 50).

!!! warning "\"MCTS picks the move with the highest average value.\""
    The usual choice is the **most-visited** child. Visits are a more robust signal: a move can have a high average from only a few lucky play-outs, but it only collects many visits if it kept looking good as the search looked deeper.

!!! warning "\"AlphaZero is just a very good neural network.\""
    The raw network is much weaker than network + search. The search improves on the network at every move, and the network is trained to imitate those improved choices. Take the search away and both the play and the training signal get worse.

---

## Check yourself

??? question "1. What does a model of the environment predict, and how would you learn one in a small grid world?"
    It predicts what happens after taking action $a$ in state $s$: the next state $s'$ and the reward $r$ (or their probabilities $\hat p(s', r \mid s, a)$). In a small grid world you count: every real transition adds 1 to $n(s, a, s', r)$, and $\hat p = n(s, a, s', r) / n(s, a)$.

??? question "2. Dyna-Q with n = 0 is which algorithm? What does n > 0 add, and what does it cost?"
    With $n = 0$ it is plain **Q-learning**. With $n > 0$ every real step is followed by $n$ Q-learning updates on transitions imagined by the model. It costs extra computation per step, but it saves real experience: on the cliff world, $n = 50$ found the optimal route after a median of 129 real steps versus about 840 for $n = 0$.

??? question "3. In the UCT formula, what happens to a child that has been visited very few times? Why is that useful?"
    Its exploration bonus $c\sqrt{\ln N_{\text{parent}} / N}$ is large, so it gets selected again even if its average looks poor. That stops the search from writing off a move after a few unlucky random play-outs. In the tic-tac-toe example, the winning move *left* had an average of 0.00 after 25 simulations and only became the favourite once the search kept revisiting it.

??? question "4. In AlphaZero, what are the training targets for the policy head and the value head?"
    The **policy head** is trained towards the MCTS visit-count distribution $\pi$ at each position (what the search preferred). The **value head** is trained towards the final game result $z$ (+1, 0 or −1 from the view of the player to move).

??? question "5. Why do errors in a learned model hurt more on long imagined rollouts than on short ones?"
    Each imagined step starts from the model's previous prediction, so errors compound. If each step is right with probability 0.95, a 10-step rollout is all correct with probability about 0.60 and a 50-step one only about 0.08. That's why methods like MBPO use short model rollouts that start from real states.

---

## Further reading

- Richard Sutton & Andrew Barto, [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) (2nd ed.), Chapter 8: *Planning and Learning with Tabular Methods* (Dyna, Dyna-Q+, MCTS).
- Browne et al., [*A Survey of Monte Carlo Tree Search Methods*](https://ieeexplore.ieee.org/document/6145622) (2012), the standard overview of MCTS and UCT.
- Silver et al., [*Mastering the game of Go without human knowledge*](https://www.nature.com/articles/nature24270) (AlphaGo Zero, Nature 2017) and [*A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*](https://www.science.org/doi/10.1126/science.aar6404) (AlphaZero, Science 2018).
- Schrittwieser et al., [*Mastering Atari, Go, chess and shogi by planning with a learned model*](https://arxiv.org/abs/1911.08265) (MuZero, 2019).
- Janner et al., [*When to Trust Your Model: Model-Based Policy Optimization*](https://arxiv.org/abs/1906.08253) (MBPO, 2019), on short model rollouts and compounding errors.
- David Silver, [*UCL Course on RL*](https://www.davidsilver.uk/teaching/), Lecture 8: Integrating Learning and Planning.

**Next lessons:** [Offline RL & imitation](../offline-rl-imitation/index.md)

<script src="viz/model-based-rl.js"></script>
