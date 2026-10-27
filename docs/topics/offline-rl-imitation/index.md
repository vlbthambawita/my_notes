# Offline RL and Imitation Learning

!!! abstract "TL;DR"
    Sometimes trial and error is too dangerous or too expensive: you can't let a robot crash a car or let an algorithm experiment on patients. Instead you learn from a **fixed dataset** of what others did. **Imitation learning** copies an expert (**behaviour cloning**), but small mistakes snowball because the copy ends up in situations the expert never showed it; **DAgger** fixes this by asking the expert about the learner's own mistakes. **Offline RL** uses the rewards in the data to do *better* than the data, but plain Q-learning breaks offline: it trusts made-up values for actions nobody ever tried. The fix is **pessimism**: only trust what the data supports (BCQ, CQL, IQL).

**Level:** intermediate · **Prerequisites:** [Q-learning & SARSA](../q-learning-sarsa/index.md), [Deep Q-Networks](../deep-q-networks/index.md)

---

## Why should I care?

Every RL method so far learns by **trying things**: an ε-greedy agent in the cliff world falls off the cliff hundreds of times before it learns not to. That is fine in a simulator. It is not fine when:

- **Healthcare:** you want a policy for dosing fluids and vasopressors in sepsis. You cannot give random doses to see what happens. But hospitals have years of electronic health records showing what clinicians did and how patients fared.
- **Driving:** you have millions of kilometres of human driving logs, but you cannot let a learning car explore by swerving into traffic.
- **Robotics:** real robots are slow, break, and need a person to reset them. Logs of past runs and human tele-operation are cheap by comparison.

In all three cases there is a lot of **logged data** and very little room for **new trial and error**. This lesson is about squeezing a good policy out of that data, and about the traps that make it much harder than it looks.

!!! note "A word of caution for clinical data"
    Learning treatment policies from records is an active research area (for example the "AI Clinician" for sepsis, see Further reading), and also a heavily debated one. Records hide things: doctors act on information that never made it into the data, so "patients who got drug X did worse" can simply mean "sicker patients got drug X". Everything in this lesson assumes the state you see contains what the decision-maker saw. In medicine that is rarely fully true, so treat any learned policy as a hypothesis to test carefully, not a treatment recommendation.

---

## Step 1 — Three ways to get experience: online, off-policy, offline

So far the agent has always been able to **collect its own data**. RL methods differ in how they use that data:

- **Online, on-policy** (SARSA, policy gradients, PPO): act with the current policy, learn from what just happened, throw the experience away, repeat.
- **Off-policy** (Q-learning, DQN): keep experience in a **replay buffer** and reuse it, even data collected by older versions of the policy. But the agent still keeps acting and adding fresh data.
- **Offline** (also called *batch RL*): someone else, the **behaviour policy** $\pi_\beta$, collected a dataset $\mathcal{D} = \{(s, a, r, s')\}$ in the past. The learner gets that dataset and **nothing else**. It never acts in the environment until training is over.

<div class="viz diagram">
<script type="application/json">
{
  "title": "🔀 Where does the experience come from?",
  "cols": 3,
  "rows": 2,
  "rowHeight": 110,
  "nodes": [
    {"id": "beh", "maxWidth": 150, "col": 0, "row": 0, "icon": "🧑‍⚕️", "label": "Behaviour policy π<sub>β</sub>", "sub": "clinicians, drivers, an old controller", "note": "Whoever made the decisions in the logged data. We may not even know exactly how they decided."},
    {"id": "data", "maxWidth": 150, "col": 0, "row": 1, "icon": "🗄️", "label": "<b>Fixed dataset D</b>", "sub": "logged once, never grows", "role": "data", "note": "Tuples (s, a, r, s′): what the situation was, what was done, what happened next."},
    {"id": "pol", "maxWidth": 150, "col": 1, "row": 0, "icon": "🤖", "label": "<b>Current policy π</b>", "role": "model"},
    {"id": "env", "maxWidth": 150, "col": 2, "row": 0, "icon": "🌍", "label": "<b>Environment</b>", "sub": "the real world or a simulator", "role": "data"},
    {"id": "buf", "maxWidth": 150, "col": 2, "row": 1, "icon": "📦", "label": "<b>Replay buffer</b>", "sub": "old + new experience", "role": "data", "note": "Off-policy methods such as DQN keep experience here and reuse it many times."},
    {"id": "learn", "maxWidth": 150, "col": 1, "row": 1, "icon": "🧠", "label": "<b>Update π</b>", "sub": "Q-learning, BC, CQL…", "role": "loss"}
  ],
  "edges": [
    {"from": "beh", "to": "data", "label": "logged"},
    {"from": "data", "to": "learn", "label": "only this"},
    {"from": "pol", "to": "env", "label": "action"},
    {"from": "env", "to": "buf", "label": "s′, r"},
    {"from": "buf", "to": "learn", "label": "sample"},
    {"from": "learn", "to": "pol", "label": "new π"}
  ],
  "steps": [
    {"show": ["pol", "pol>env", "env", "env>buf", "buf", "buf>learn", "learn", "learn>pol"], "text": "<b>Online, on-policy</b> (SARSA, PPO): the agent acts, learns from what just happened, and discards it. The “buffer” holds only the latest experience. Every improvement costs fresh trial and error."},
    {"show": ["env>buf", "buf", "buf>learn", "learn"], "text": "<b>Off-policy</b> (Q-learning, DQN): experience is <b>kept</b> in a replay buffer and reused, even when an older policy collected it. The agent still keeps acting, so it can try out its own ideas."},
    {"show": ["beh", "beh>data", "data", "data>learn", "learn"], "text": "<b>Offline</b>: someone else collected the data in the past. The learner sees only D, and <b>never acts</b> while it trains. No new data, no way to test its guesses."},
    {"show": ["learn", "learn>pol", "pol"], "text": "Only at the very end is the learned policy deployed, after careful checks. If its guesses about unseen actions were wrong, this is where it hurts."}
  ],
  "narrow": {"cols": 2.4, "rows": 3, "hideSub": true, "pos": {"pol": [0, 0], "env": [1.4, 0], "buf": [1.4, 1], "learn": [0, 1], "data": [0, 2], "beh": [1.4, 2]}}
}
</script>
</div>

There are two very different things you can do with a fixed dataset:

| | **Imitation learning** | **Offline RL** |
|---|---|---|
| Data needed | states and the expert's actions $(s, a)$ | states, actions **and rewards** $(s, a, r, s')$ |
| Goal | act like the expert | act **better** than the data, if possible |
| Main tool | supervised learning | Q-learning, made pessimistic |

!!! tip "What to notice"
    Off-policy and offline sound alike, but the difference is huge. Q-learning *is* off-policy, yet in Step 5 you will see it fail badly offline. Off-policy learners can still try an action to find out that it is bad. Offline learners can't.

---

## Step 2 — Imitation learning: behaviour cloning is supervised learning

The simplest idea: you have recordings of an expert. In each recorded state $s_i$ the expert took action $a_i$. Treat this as an ordinary **supervised learning** problem: the state is the input, the expert's action is the label. This is **behaviour cloning** (BC).

For discrete actions, train a classifier $\pi_\theta(a \mid s)$ with the usual cross-entropy loss. For continuous actions (a steering angle, a drug dose), train a regressor with squared error:

$$
\theta^* = \arg\max_\theta \sum_i \log \pi_\theta(a_i \mid s_i)
\qquad\text{or}\qquad
\theta^* = \arg\min_\theta \sum_i \big\| \pi_\theta(s_i) - a_i \big\|^2
$$

No rewards, no Bellman equation, no exploration. That is why BC is the first thing to try when you have expert data: it is just a neural network trained on labelled data, which you already know how to do. It has worked surprisingly well in practice, from NVIDIA's 2016 end-to-end car that learned to steer from 72 hours of human driving to modern robot manipulation policies trained on tele-operation data.

The running example for the next two steps is a car following a winding lane. The state is three numbers: how much the road bends here ($\kappa$), how far the car is from the lane centre ($e$), and how far its heading points away from the road ($h$). The expert steers with the bend and gently back towards the centre:

$$
a^{\text{expert}} = \kappa - 0.1\, e - 0.5\, h
$$

We record one perfect drive by this expert and fit a linear policy $a = w_1 \kappa + w_2 e + w_3 h$ to it by least squares. The fit is perfect: on the recorded states, the clone's action differs from the expert's by less than $10^{-6}$. The training loss is essentially zero.

!!! tip "What to notice"
    "Zero training error" sounds like success. Hold on to that thought: in the next step this perfect student drives off the road in 94% of drives.

---

## Step 3 — Why copying drifts: compounding errors

Here is the catch. Look at the expert's recording: the expert drove perfectly, so $e$ and $h$ were **always zero**. The data shows how to follow the bend. It never shows what to do when you are off-centre, because the expert never was. The clone learns $w = (1, 0, 0)$: "steer with the bend" and nothing else.

Now let the clone drive with a tiny bit of steering noise (a gust of wind, a bump, a small modelling error). After the first small mistake it is in a state the expert never visited. It has no idea how to recover, so it makes another error, which takes it even further from familiar territory, and so on.

**Try it:** the red line is the behaviour-cloned car, the green line is the expert, both with exactly the same random noise. The chart averages 200 drives.

<div id="viz-drift-bc" class="viz"><div class="viz-title">🚗 Interactive: a perfect copy that drives off the road</div></div>

!!! tip "What to notice"
    - With the default noise ($\sigma = 0.01$), the expert stays within about 0.03 of the centre. The clone, with **zero training error**, is on average 0.55 off at $t = 25$, 1.60 at $t = 50$ and 4.69 at $t = 100$, and leaves the lane in **94%** of drives.
    - The error doesn't just grow, it **speeds up**: each doubling of time multiplies the offset by about 2.9. Mistakes compound.
    - Set $\sigma = 0$. Now the clone is perfect too: with no mistakes, it never leaves the expert's states. The problem is not the copy itself but the **distribution shift**: at test time the learner visits states that were not in its training data.
    - Double the noise to $\sigma = 0.02$ and the clone's offset at $t = 100$ doubles to 9.37.

Why "errors grow like $T^2$"? A simple counting argument (Ross & Bagnell, 2010): suppose the learner makes a mistake with a small probability $\varepsilon$ per step **on the expert's states**, and once it is off the expert's path it never gets back. The chance of being off-track by step $t$ is $1 - (1-\varepsilon)^t \approx \varepsilon t$. Adding these up over an episode of length $T$:

$$
\text{expected steps off-track} \;=\; \sum_{t=1}^{T} \big(1 - (1-\varepsilon)^t\big) \;\approx\; \frac{\varepsilon\, T^2}{2}
$$

With $\varepsilon = 0.01$ and $T = 100$ that is 37.2 bad steps; with $T = 200$ it is 114.3. A learner that could **recover** after each mistake would only lose about $\varepsilon T$ steps: 1.0 and 2.0. Supervised learning assumes that test inputs come from the same distribution as training inputs. In sequential decision-making, **your own actions choose your next input**, so that assumption breaks.

---

## Step 4 — DAgger: ask the expert about your own mistakes

If the problem is "the training data has no off-track states", the fix is to put them in. **DAgger** (*Dataset Aggregation*, Ross, Gordon & Bagnell, 2011) does exactly this:

1. Train a policy on the expert's demonstrations (plain BC).
2. Let **the learner** drive, and record the states **it** visits, including its mistakes.
3. Ask the expert: "what would you have done here?" Label those states with the expert's actions.
4. Add them to the dataset (keep everything old), retrain, and repeat.

<div class="viz diagram">
<script type="application/json">
{
  "title": "🔁 The DAgger loop",
  "cols": 3,
  "rows": 2,
  "rowHeight": 105,
  "nodes": [
    {"id": "start", "col": 0, "row": 0, "icon": "🎬", "label": "Expert demos D", "shape": "pill", "role": "data", "note": "The usual behaviour-cloning dataset: the expert's own states and actions."},
    {"id": "train", "col": 1, "row": 0, "icon": "🎓", "label": "<b>Train π on D</b>", "sub": "supervised learning", "role": "model"},
    {"id": "run", "col": 2, "row": 0, "icon": "🚗", "label": "<b>Let π drive</b>", "sub": "record the states it visits", "role": "model", "note": "These include the learner's own mistakes: exactly the states the expert's data never covered."},
    {"id": "ask", "col": 2, "row": 1, "icon": "🧑‍🏫", "label": "<b>Expert labels them</b>", "sub": "“what would you do here?”", "role": "opt", "note": "The expert only labels; the learner is the one driving. In practice the expert can be a human, a planner, or a slow but safe controller."},
    {"id": "agg", "col": 1, "row": 1, "icon": "➕", "label": "<b>Add to D</b>", "sub": "keep all old data", "role": "data"}
  ],
  "edges": [
    {"from": "start", "to": "train"},
    {"from": "train", "to": "run"},
    {"from": "run", "to": "ask", "label": "its states"},
    {"from": "ask", "to": "agg"},
    {"from": "agg", "to": "train", "label": "retrain"}
  ],
  "steps": [
    {"show": ["start", "start>train", "train"], "text": "Round 0 is ordinary behaviour cloning on the expert's demonstrations."},
    {"show": ["train>run", "run"], "text": "Now <b>the learner</b> drives. It makes mistakes and ends up in states the expert never visited."},
    {"show": ["run>ask", "ask"], "text": "The expert labels every state the learner visited with the action it would have taken there: often a <b>correction</b>."},
    {"show": ["ask>agg", "agg"], "text": "Those labelled states are added to the dataset, old data included."},
    {"show": ["agg>train", "train"], "text": "Retrain on everything and repeat. The training data now looks like the states the learner will really see."}
  ],
  "narrow": {"cols": 2, "rows": 3, "pos": {"start": [0, 0], "train": [0, 1], "run": [1, 1], "ask": [1, 2], "agg": [0, 2]}}
}
</script>
</div>

The key change: the training states now come from **the learner's own distribution**, not the expert's. That gives the guarantee the counting argument wanted: mistakes cost about $\varepsilon T$, not $\varepsilon T^2$.

**Try it:** same road, same noise, now with a DAgger learner (orange). Each DAgger round adds 5 drives by the learner, labelled by the expert.

<div id="viz-drift-dagger" class="viz"><div class="viz-title">🧑‍🏫 Interactive: behaviour cloning vs DAgger</div></div>

!!! tip "What to notice"
    - After **one** round, DAgger learns the weights $(1.00, -0.10, -0.50)$: exactly the expert's rule, including how to steer back. Its average offset at $t = 100$ is 0.03 and it leaves the lane in 0% of drives, like the expert.
    - Set DAgger rounds to 0: the orange line becomes the red one. DAgger *starts* as behaviour cloning.
    - One round is enough here because the learner (a linear rule) can represent the expert exactly. Real policies are neural networks that can't, and usually need several rounds.
    - Set $\sigma = 0$ and look at the DAgger weights: with no mistakes there are no off-track states to learn from, so the offset and heading weights are poorly pinned down. DAgger needs the learner's mistakes to learn from them.

The catch: DAgger needs an **expert you can query** on new states. A driving instructor in the passenger seat, a slow planning algorithm, or a simulator with a known-good controller can do this. A ten-year-old hospital database cannot: nobody can go back and ask what the doctor would have done for a patient state that never happened. Practical workarounds include adding noise to the expert's demonstrations so they contain small recoveries (the *DART* idea) and training on lots of diverse data.

---

## Step 5 — Offline RL, and why plain Q-learning breaks

Imitation learning has a ceiling: at best it is as good as the expert. Often the data is **not** expert data. Hospital records mix good and bad decisions; driving logs mix careful and careless drivers. If the data also contains **rewards** (patient survived, trip completed safely), we could hope to do *better* than the average decision in the data: keep the good choices, avoid the bad ones. That is **offline RL**.

The obvious first attempt: Q-learning is off-policy, so just run it on the fixed dataset. Sweep over the logged transitions again and again with the familiar update:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \Big[\, r + \gamma \max_{a'} Q(s', a') - Q(s, a) \Big]
$$

Look at the $\max_{a'}$. It considers **every** action in the next state, including actions that **never appear in the data**. Their $Q$-values were never trained. In a table they keep whatever value they started with; in a neural network they are whatever the network happens to extrapolate. Either way, nothing in the data ever corrects them. And because the max *seeks out* the largest value, it picks exactly these made-up numbers whenever they are too high. This is called **extrapolation error**, and the result is **overestimation**: the agent believes unseen actions are wonderful.

Online, this would fix itself: the agent would try the "wonderful" action, get a bad reward, and update. Offline, it never gets the chance.

**Try it:** this is the cliff world from lesson 1 (no slipping, $\gamma = 0.95$). The data was logged by someone following the *safe detour* policy, who takes a random action a fraction $\varepsilon$ of the time. Naive Q-learning sweeps that dataset 200 times. The blue wedges show which actions appear in the data.

<div id="viz-offline-naive" class="viz"><div class="viz-title">🗄️ Interactive: naive Q-learning on a fixed dataset</div></div>

!!! tip "What to notice"
    - **The agent is confidently wrong.** With the default data (ε = 0.1, 20 episodes, 239 transitions), the data covers only 26 of the 72 state-action pairs. In the start cell the data shows only "up" and "left", each worth $-1.0$ (one step cost). "Right" and "down" were never tried and keep their starting guess of 0, which looks better. So the agent **believes** $V(\text{start}) = 0.00$, picks "right", and **really** gets $-20.00$: straight off the cliff.
    - **Red arrows** are actions the agent has never seen in the data, chosen *because* nothing ever corrected their values.
    - Set ε = 0 (the logger always follows the detour): only 11 of 72 pairs are covered, and the result is the same cliff dive. The logging policy itself is worth $-2.04$, so the "learned" policy is far worse than simply copying the data.
    - Now give it **coverage**: ε = 0.5 and 100 episodes (71 of 72 pairs). Naive Q-learning now finds the best route along the cliff edge and gets $2.05$, the optimal value. Offline Q-learning is fine **when the data covers the actions it wants to take**. The trouble is that real datasets almost never do.

!!! note "Why this is worse with neural networks"
    In a table, an unseen action keeps one stale value. A deep Q-network (lesson 7) is trained on the dataset's actions, and its outputs for other actions are an unconstrained extrapolation. Some of those extrapolations are too high by chance, the max picks them, and bootstrapping copies the inflated value backwards to earlier states. In practice offline DQN's value estimates can grow without limit while the actual policy gets worse.

---

## Step 6 — The fix: be pessimistic about what you have not seen

All modern offline RL methods share one principle: **don't trust values for actions the data doesn't support.** They differ in how they enforce it.

**1. Stay close to the data (policy constraints, e.g. BCQ).** *Batch-Constrained Q-learning* (Fujimoto, Meger & Precup, 2019) only allows the max to look at actions that the behaviour policy would plausibly take. In a table this is easy: take the max only over actions that appear in the data,

$$
r + \gamma \max_{a' \,:\, (s', a') \in \mathcal{D}} Q(s', a').
$$

With neural networks, BCQ trains a generative model of the data's actions and only considers actions it would produce. Related methods (BEAR, TD3+BC) add a penalty that keeps the learned policy close to the data's actions.

**2. Push unseen values down (pessimism, e.g. CQL).** *Conservative Q-Learning* (Kumar et al., 2020) keeps the network free, but adds a term to the loss that pushes $Q$ **down** on all actions (mostly on the ones it currently rates highly) and **up** on the actions that are actually in the data:

$$
\mathcal{L}_{\text{CQL}}(\theta) = \underbrace{\mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\Big[\big(Q_\theta(s,a) - y\big)^2\Big]}_{\text{the usual TD loss}}
\;+\; \lambda \, \mathbb{E}_{s \sim \mathcal{D}}\Big[\, \underbrace{\log \sum_{a} e^{Q_\theta(s,a)}}_{\text{push all actions down}} \;-\; \underbrace{\mathbb{E}_{a \sim \mathcal{D}}\, Q_\theta(s,a)}_{\text{push data actions up}} \Big]
$$

The net effect: an action the data never shows gets pushed down until it no longer wins the max. The learned values become a **lower bound** on the true values, so when the agent is wrong, it errs on the side of caution.

**3. Never ask about unseen actions at all (IQL).** *Implicit Q-Learning* (Kostrikov, Nair & Levine, 2021) avoids the max entirely. It learns $Q$ only on the dataset's own actions, estimates "how good is the best action the data shows here" with a trick called expectile regression, and then extracts a policy by **weighted behaviour cloning**: copy the data's actions, but copy the ones with a high advantage $Q(s,a) - V(s)$ much more strongly. It is simple, stable, and a common default today.

Our widget uses a tabular version of the same idea, a **pessimistic Q-learning**: actions that never appear in the data get value $-\infty$ (so they are never chosen), and actions seen only a few times pay a penalty $\beta / \sqrt{n(s,a)}$, where $n(s,a)$ is how often they appear. The fewer times we have seen something, the less we trust it. (The $-\infty$ part is BCQ's constraint; the penalty is CQL's spirit, in its simplest form.)

**Try it:** switch between the three learners on the same dataset.

<div id="viz-offline-fix" class="viz"><div class="viz-title">🛡️ Interactive: naive vs pessimistic offline learning</div></div>

!!! tip "What to notice"
    - On the default data (ε = 0.1, 20 episodes), **pessimistic Q-learning** follows the detour to the goal and really gets $-2.04$. It believes $V(\text{start}) = -3.00$: a little lower than the truth, which is exactly the point of pessimism. No red arrows: it never picks an action it hasn't seen.
    - Naive Q-learning on the **same** data believes $0.00$ and gets $-20.00$. Same data, same algorithm family; the only change is refusing to trust the unknown.
    - Pessimism also helps when the data is good. With ε = 0.5 and 100 episodes, pessimistic Q-learning finds the optimal edge route ($2.05$), even though the logging policy itself is only worth $-12.57$. It **stitches** together the good pieces of many imperfect episodes.
    - Pessimism is not magic. With ε = 1 (purely random logging) and only 20 episodes, the data is mostly aimless wandering and every learner fails: naive $-19.88$, pessimistic $-20.00$, behaviour cloning $-19.88$. No algorithm can find a good route the data never shows.

---

## Step 7 — Behaviour cloning or offline RL?

Use the third button in the widget above, **Behaviour cloning** (in each cell, copy the most common action in the data):

| Logged data | Logging policy is worth | Behaviour cloning | Pessimistic Q-learning |
|---|---|---|---|
| expert-like: ε = 0, 20 episodes | $-2.04$ | $-2.04$ | $-2.04$ |
| mostly good: ε = 0.1, 20 episodes | $-3.70$ | $-2.04$ | $-2.04$ |
| mixed: ε = 0.5, 100 episodes | $-12.57$ | $-2.04$ | $2.05$ |
| random: ε = 1, 100 episodes | $-19.88$ | $-19.88$ | $2.05$ |

The pattern matches what researchers find on real benchmarks:

- **Expert (or near-expert) data → behaviour cloning.** It is simple and stable, and needs no rewards. Offline RL can't beat the expert's actions if those are the only actions in the data, and its extra machinery adds ways to go wrong. In the clean-data rows, BC already gets everything there is to get.
- **Mixed-quality data with rewards → offline RL.** BC faithfully copies the average behaviour, mistakes included (it even filters out some noise by taking the most common action, which is why it beats the logging policy). Offline RL uses the rewards to tell good decisions from bad ones and can **beat every policy in the data**, as in the last two rows.
- **In between** there are hybrids: *filtered BC* (clone only the best episodes), *advantage-weighted BC* (IQL's last step), and TD3+BC (Q-learning plus a BC term).

And whatever you train, **evaluate before you deploy**. Offline, you can't simply run the policy to see how good it is. *Off-policy evaluation* (OPE) methods estimate a new policy's value from the logged data, using importance sampling or a learned Q-function, but they are noisy and share the same blind spots as the data. In healthcare, the realistic path is: offline RL → off-policy evaluation → clinician review → a prospective, carefully monitored study.

---

## Putting it all together

<div class="viz diagram">
<script type="application/json">
{
  "title": "🧭 Learning from data without trial and error: which tool?",
  "cols": 3,
  "rows": 3,
  "rowHeight": 105,
  "nodes": [
    {"id": "q1", "maxWidth": 160, "col": 0, "row": 0, "label": "Safe and cheap to try things?", "shape": "decision", "role": "grad"},
    {"id": "online", "maxWidth": 160, "col": 1, "row": 0, "icon": "🎮", "label": "<b>Online RL</b>", "sub": "Q-learning, DQN, PPO", "role": "model", "note": "Simulators, games, and anything where mistakes are cheap: learn by trial and error."},
    {"id": "q2", "maxWidth": 160, "col": 0, "row": 1, "label": "Expert-quality actions in the data?", "shape": "decision", "role": "grad"},
    {"id": "bc", "maxWidth": 140, "col": 1, "row": 1, "icon": "🎓", "label": "<b>Behaviour cloning</b>", "sub": "supervised: state → action", "role": "model", "note": "Simple and stable. Watch out for compounding errors on long tasks."},
    {"id": "dagger", "maxWidth": 140, "col": 2, "row": 1, "icon": "🧑‍🏫", "label": "<b>DAgger</b>", "sub": "expert labels the learner's states", "role": "opt", "note": "Only possible if an expert (human, planner, safe controller) can be asked about new states."},
    {"id": "off", "maxWidth": 160, "col": 0, "row": 2, "icon": "🛡️", "label": "<b>Offline RL</b>", "sub": "CQL, IQL: rewards, mixed data", "role": "loss", "note": "Needs rewards in the data. Pessimism keeps it from trusting actions the data never shows."},
    {"id": "ope", "maxWidth": 160, "col": 1, "row": 2, "icon": "🔬", "label": "<b>Evaluate offline, deploy slowly</b>", "role": "opt", "note": "Off-policy evaluation, expert review, then a small monitored trial. Essential in healthcare."}
  ],
  "edges": [
    {"from": "q1", "to": "online", "label": "yes"},
    {"from": "q1", "to": "q2", "label": "no"},
    {"from": "q2", "to": "bc", "label": "yes"},
    {"from": "bc", "to": "dagger", "label": "can ask"},
    {"from": "q2", "to": "off", "label": "mixed"},
    {"from": "off", "to": "ope", "label": "then"}
  ],
  "steps": [
    {"show": ["q1", "q1>online", "online"], "text": "If exploring is safe and cheap (a simulator, a game), use the online methods from the earlier lessons."},
    {"show": ["q1", "q1>q2", "q2"], "text": "If not, you must learn from logged data. First question: how good are the decisions in it?"},
    {"show": ["q2", "q2>bc", "bc"], "text": "Expert data: <b>behaviour cloning</b>, plain supervised learning from states to actions."},
    {"show": ["bc", "bc>dagger", "dagger"], "text": "If small errors compound on long tasks and you can query an expert, use <b>DAgger</b> to add the learner's own states to the data."},
    {"show": ["q2", "q2>off", "off"], "text": "Mixed-quality data with rewards: <b>offline RL</b> with pessimism (CQL, IQL) can beat the data, without trusting actions it never saw."},
    {"show": ["off", "off>ope", "ope"], "text": "Either way, estimate the policy's value from data, have experts review it, and deploy step by step."}
  ],
  "narrow": {"cols": 2.4, "rows": 4, "hideSub": true, "pos": {"q1": [0, 0], "online": [1.4, 0], "q2": [0, 1], "bc": [1.4, 1], "off": [0, 2], "dagger": [1.4, 2], "ope": [0, 3]}}
}
</script>
</div>

Here is behaviour cloning and DAgger on the lane-following task, in NumPy. The learner is a linear policy fitted with least squares:

```python
import numpy as np

T, K1, K2 = 100, 0.1, 0.5
kappa = 0.03 * np.sin(2 * np.pi * np.arange(T) / 60)     # how much the road bends at each step


def expert(k, e, h):                     # steer with the bend, and back towards the lane centre
    return k - K1 * e - K2 * h


def drive(policy, sigma, rng):
    """Drive one lap. e = distance from the lane centre, h = heading error."""
    e = h = 0.0
    states, offsets = [], []
    for t in range(T):
        states.append((kappa[t], e, h))
        a = policy(kappa[t], e, h)
        h += a - kappa[t] + sigma * rng.normal()          # steering noise: small mistakes
        e += h
        offsets.append(abs(e))
    return states, offsets


def fit(states):                                           # supervised learning: state -> expert action
    X = np.array(states)
    y = np.array([expert(*s) for s in states])
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    return lambda k, e, h: w @ (k, e, h), w


rng = np.random.default_rng(0)
sigma = 0.01

# Behaviour cloning: copy a perfect demonstration
demo, _ = drive(expert, 0.0, rng)
bc, w_bc = fit(demo)

# DAgger: let the learner drive, ask the expert what it should have done, retrain on everything
data, dagger, w_dag = list(demo), bc, w_bc
for round_ in range(3):
    for _ in range(5):
        visited, _ = drive(dagger, sigma, rng)
        data += visited
    dagger, w_dag = fit(data)

for name, pol, w in [("BC", bc, w_bc), ("DAgger", dagger, w_dag)]:
    runs = np.array([drive(pol, sigma, rng)[1] for _ in range(200)])
    print(f"{name:6s} weights (bend, offset, heading) = {np.round(w, 3)}   "
          f"mean |offset| at t=100: {runs[:, -1].mean():.2f}   left the lane: {(runs.max(1) > 1).mean():.0%}")
```

```text
BC     weights (bend, offset, heading) = [1. 0. 0.]   mean |offset| at t=100: 4.82   left the lane: 95%
DAgger weights (bend, offset, heading) = [ 1.  -0.1 -0.5]   mean |offset| at t=100: 0.03   left the lane: 0%
```

(The random noise differs from the widget's, so the numbers differ slightly: 4.82 and 95% here, 4.69 and 94% in the widget.)

And here is offline learning on the cliff world: a logged dataset, naive Q-learning, the pessimistic version from Step 6, and behaviour cloning, each evaluated by actually running its greedy policy:

```python
import numpy as np

MAP = ["......",
       "..#...",
       "......",
       "SCCCCG"]
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]                  # up, right, down, left
SAFE = ["→→→→→↓",                                           # the "safe detour" policy from lesson 1
        "↑↑.↑↑↓",
        "↑↑←↑↑↓",
        "↑....."]
GAMMA = 0.95


def step(s, a):
    r, c = s[0] + MOVES[a][0], s[1] + MOVES[a][1]
    if not (0 <= r < 4 and 0 <= c < 6) or MAP[r][c] == "#":
        r, c = s                                            # bumped into a wall or edge
    tile = MAP[r][c]
    return (r, c), {"G": 10, "C": -20}.get(tile, -1), tile in "GC"


def log_data(eps, episodes, rng):
    """Someone else's behaviour: the safe detour, with a random action a fraction eps of the time."""
    data = []
    for _ in range(episodes):
        s = (3, 0)
        for t in range(100):
            a = rng.integers(4) if rng.random() < eps else "↑→↓←".index(SAFE[s[0]][s[1]])
            s2, r, done = step(s, a)
            data.append((s, a, r, s2, done))
            s = s2
            if done:
                break
    return data


def offline_q_learning(data, pessimism=None, epochs=200, lr=0.5):
    n = np.zeros((4, 6, 4))                                 # how often each (s, a) appears in the data
    for s, a, *_ in data:
        n[s][a] += 1
    if pessimism is None:
        Q = np.zeros((4, 6, 4))                             # naive: unseen actions keep the guess 0
    else:
        Q = np.where(n > 0, 0.0, -np.inf)                   # pessimistic: never trust an unseen action
    for _ in range(epochs):                                 # sweep the SAME fixed dataset again and again
        for s, a, r, s2, done in data:
            nxt = 0.0 if done else Q[s2].max()
            if nxt == -np.inf:
                nxt = 0.0                                   # s2 never left in the data (cut-off episode)
            penalty = 0.0 if pessimism is None else pessimism / np.sqrt(n[s][a])
            Q[s][a] += lr * (r - penalty + GAMMA * nxt - Q[s][a])
    return Q


def behaviour_cloning(data):
    counts = np.zeros((4, 6, 4))
    for s, a, *_ in data:
        counts[s][a] += 1
    return counts                                           # act = most common action in each state


def evaluate(table):                                        # run the greedy policy once in the real world
    s, G, disc = (3, 0), 0.0, 1.0
    for t in range(100):
        s, r, done = step(s, int(np.argmax(table[s])))
        G += disc * r
        disc *= GAMMA
        if done:
            break
    return G


rng = np.random.default_rng(0)
for eps, episodes in [(0.1, 20), (0.5, 100)]:
    data = log_data(eps, episodes, rng)
    naive = offline_q_learning(data)
    pess = offline_q_learning(data, pessimism=0.5)
    print(f"data: eps={eps}, {episodes} episodes, {len(data)} transitions")
    print(f"  naive Q-learning   believes {naive[3, 0].max():6.2f}   really gets {evaluate(naive):6.2f}")
    print(f"  pessimistic        believes {pess[3, 0].max():6.2f}   really gets {evaluate(pess):6.2f}")
    print(f"  behaviour cloning                     really gets {evaluate(behaviour_cloning(data)):6.2f}")
```

```text
data: eps=0.1, 20 episodes, 261 transitions
  naive Q-learning   believes   0.00   really gets -20.00
  pessimistic        believes  -2.99   really gets  -2.04
  behaviour cloning                     really gets  -2.04
data: eps=0.5, 100 episodes, 2049 transitions
  naive Q-learning   believes   2.05   really gets   2.05
  pessimistic        believes   0.53   really gets   2.05
  behaviour cloning                     really gets  -2.04
```

The pessimistic learner's beliefs are always **below** what it really gets ($-2.99$ vs $-2.04$, $0.53$ vs $2.05$). The naive learner's belief of $0.00$ hides a cliff dive. When you can't test a policy before deploying it, an under-estimate is a much safer mistake than an over-estimate.

For real problems, libraries such as [d3rlpy](https://github.com/takuseno/d3rlpy) and [CORL](https://github.com/tinkoff-ai/CORL) implement BC, CQL, IQL and friends, and [Minari](https://minari.farama.org/) (the successor of D4RL) provides standard offline datasets.

---

## Common misconceptions

!!! warning "\"Q-learning is off-policy, so it works on any fixed dataset.\""
    Off-policy means it can learn from data collected by another policy, **as long as it can keep collecting**. Offline, the max in the target picks up values for actions the data never shows, and nothing ever corrects them. In the widget, naive Q-learning believes the start is worth 0.00 and walks off the cliff.

!!! warning "\"If behaviour cloning fits the expert data perfectly, it will behave like the expert.\""
    The fit is only measured on the **expert's** states. The clone's own small mistakes take it to states the expert never visited, where it has no idea what to do, and the errors compound. Our clone had essentially zero training error and still left the lane in 94% of drives.

!!! warning "\"Offline RL can learn any policy if you have enough data.\""
    It can only learn about actions the data contains. A million records of a single treatment protocol say nothing about the alternative. Coverage matters more than size: in the widget, 100 episodes of the plain detour cover just 11 of 72 state-action pairs.

!!! warning "\"Pessimism just makes the agent copy the data.\""
    It only stops the agent from relying on actions the data doesn't support. Among the supported ones it still picks the best, so it can combine good pieces of different episodes and **beat** the logging policy ($2.05$ vs $-12.57$ in the mixed-data example).

!!! warning "\"A policy learned from hospital records tells us the best treatment.\""
    Only if the records capture everything the clinicians based their decisions on, and the reward really measures what matters to patients. Hidden confounders, missing data and rare actions can make an offline policy look good on paper and be harmful in practice. It is a hypothesis to test, not a recommendation.

---

## Check yourself

??? question "1. What is the difference between off-policy and offline RL?"
    An **off-policy** learner (Q-learning, DQN) can learn from data collected by a different policy, but it keeps interacting with the environment and adding new data. An **offline** learner gets one fixed dataset and never interacts during training, so it can never test what happens when it tries an action the data doesn't show.

??? question "2. Why does behaviour cloning suffer from compounding errors, and why does DAgger help?"
    BC is trained only on the states the expert visited. A small mistake moves the learner to a state outside that distribution, where its actions are unreliable, so it makes more mistakes and drifts further: errors can add up to roughly $\varepsilon T^2$ over an episode. DAgger runs the **learner**, has the expert label the states the learner actually visits, and retrains on them, so the learner learns how to recover. Errors then grow only like $\varepsilon T$.

??? question "3. In the offline cliff-world widget, why does naive Q-learning pick \"right\" from the start cell?"
    In the default dataset, the start cell only shows "up" (the detour) and, once, "left". Both cost a −1 step, so their learned values are negative. "Right" and "down" never appear, so their values stay at the starting guess of 0, which is higher. The max picks "right" for a value that nothing ever checked, and "right" leads straight into the cliff (−20).

??? question "4. How does CQL prevent the problem in question 3?"
    It adds a term to the loss that pushes Q-values **down** for all actions (weighted towards the ones currently rated highest) and **up** for actions that actually appear in the data. Unseen actions only ever get pushed down, so they stop winning the max, and the learned values become a lower bound on the true ones. The widget's pessimistic learner gets the same effect in a table by giving unseen actions $-\infty$ and penalising rarely seen ones.

??? question "5. You have ICU records from many clinicians with very different habits, and a recorded outcome for each patient. Behaviour cloning or offline RL? What would make you cautious?"
    The data is **mixed quality** and has rewards (outcomes), so offline RL with pessimism (CQL, IQL) is the natural choice: BC would copy the average habit, good and bad. Reasons for caution: unrecorded information that influenced treatment (confounding), treatments that were rarely or never given in some situations (poor coverage), whether the outcome is the right reward, and the difficulty of evaluating the policy without trying it. Any result needs off-policy evaluation, clinical review and a careful prospective study before it touches patients.

---

## Further reading

- Sergey Levine, Aviral Kumar, George Tucker & Justin Fu, [*Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems*](https://arxiv.org/abs/2005.01643) (2020). The standard overview.
- Stéphane Ross, Geoffrey Gordon & Drew Bagnell, [*A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*](https://arxiv.org/abs/1011.0686) (2011). The DAgger paper.
- Scott Fujimoto, David Meger & Doina Precup, [*Off-Policy Deep Reinforcement Learning without Exploration*](https://arxiv.org/abs/1812.02900) (2019). Extrapolation error and BCQ.
- Aviral Kumar, Aurick Zhou, George Tucker & Sergey Levine, [*Conservative Q-Learning for Offline Reinforcement Learning*](https://arxiv.org/abs/2006.04779) (2020).
- Ilya Kostrikov, Ashvin Nair & Sergey Levine, [*Offline Reinforcement Learning with Implicit Q-Learning*](https://arxiv.org/abs/2110.06169) (2021).
- Matthieu Komorowski et al., [*The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care*](https://www.nature.com/articles/s41591-018-0213-5), Nature Medicine (2018), and Omer Gottesman et al., [*Guidelines for reinforcement learning in healthcare*](https://www.nature.com/articles/s41591-018-0310-5), Nature Medicine (2019): a clinical application and a careful look at its pitfalls.
- Sergey Levine, [CS 285 lectures on imitation learning and offline RL](https://rail.eecs.berkeley.edu/deeprlcourse/) (UC Berkeley, videos and slides).

**Next lessons:** [RLHF & DPO](../rlhf-dpo/index.md): imitation learning (supervised fine-tuning) and RL from human preferences are how chatbots like Claude and ChatGPT are trained.

<script src="viz/offline-rl-imitation.js"></script>
