# Project 2: how it works

The goal: from a fixed batch of logged transitions `(s, a, r, sp)`, produce a
policy, meaning one action for every state. We never interact with the
environment, so this is **offline** (batch) reinforcement learning. The policy is
graded by simulating it on the true, secret MDP. All code is in `project2.py`;
`plot_policy.py` only draws.

## 1. The approach: estimate a model, then plan

One method handles all three datasets:

1. **Maximum-likelihood model.** Count how often each `(s, a)` led to each
   `sp`: T̂(sp | s, a) = N(s, a, sp) / N(s, a). Take R̂(s, a) as the mean
   observed reward. The course says rewards are deterministic given
   `(s, a, sp)`, so the mean is the expected reward under T̂. Everything is
   stored as a sparse list of edges, so even 302,020 states × 9 actions is
   cheap.
2. **Value iteration** on that model:
   `Q(s, a) = R̂(s, a) + γ Σ_sp T̂(sp | s, a) V(sp)`, `V(s) = max_a Q(s, a)`,
   repeated until no value changes by more than the tolerance. Only actions
   that were tried count, because an untried action has no estimate.
3. **Greedy policy**: `π(s) = argmax_a Q(s, a)`.
4. **Fill unseen states**: give an action to every state that never appears
   in the data (see each dataset below).

The alternative is model-free Q-learning, replaying the data many times. It
converges to the same answer as planning on the maximum-likelihood model
(certainty equivalence), but more slowly and with a learning rate to tune. With
at most 100k samples, building the model is the simpler and more direct choice.

## 2. What the data look like, and per-dataset choices

These were found by inspecting the CSVs (counts of moves `sp − s` and rewards
per action):

**small**: 10×10 grid, γ = 0.95. Actions move −1, +1, +10, −10
(left, right, up, down). The intended move happens about 70% of the time and each
other direction about 10%. The reward is +10 in state 47 at (7, 5) and +3 in
state 82 at (2, 9). All 100 states are observed, so nothing needs filling.

**medium**: mountain car, `s = 1 + pos + 500·vel`, γ = 1 (undiscounted).
The reward is −25·(a − 4)², an acceleration cost (action 4 is free), plus about
+100,000 for reaching the flag at position ≥ 456. After that the log restarts
a new episode. Two special rules:

- *Terminal transitions.* A transition with reward ≥ 50,000 ends the
  episode: its next state contributes no future value. Without this rule
  the logged "next state" (a restart) would let the model collect the goal
  reward again and again, and with γ = 1 the values would grow without bound.
  `test_goal_reward_ends_the_episode` checks this.
- *Unseen states.* Only 22,530 of the 50,000 states appear. When a logged
  next state was never seen as a state, it is redirected to the nearest
  observed state in (position, velocity), with both axes scaled to [0, 1].
  Without that, it would be a dead end worth 0, which with negative costs
  looks attractive. After planning, each unseen state copies the action of
  its nearest observed state.

**large**: secret, 302,020 states, 9 actions, γ = 0.95. The state number
has the form `PPRRCC`: a 2-digit prefix P ∈ {15, 23, 27, 29, 30} and a
position on a grid, so the data contain only 5 × 100 = 500 distinct states.
Actions 1–4 move on the grid (±1 or ±100; each 2-digit coordinate takes
the values 01–04, 10–14, 20, so a step across a gap shows up as ±6 or ±600). Actions 5–9
usually do nothing, but from corner cells action 5 sometimes jumps to a
different grid. Rewards (−10 … +100) come from entering particular cells.
All the reachable states are in the data, so the 301,520 others can never
occur. They get the most common learned action, which is harmless.

## 3. Results

From `make policy-all` (runtimes measured in Docker on a laptop; they include
model estimation, planning and filling, but not CSV loading):

| Dataset | Observed states | VI iterations | Runtime | Model value: learned | Model value: random |
|---|---|---|---|---|---|
| small | 100 / 100 | 393 | 0.03 s | 36.9 | 2.6 |
| medium | 22,530 / 50,000 | 1,735 | 10.6 s | 97,929 | 87,744 |
| large | 500 / 302,020 | 448 | 3.7 s | 583.5 | 5.1 |

"Model value" is the mean of U^π(s) over observed states, computed by policy
evaluation **on our own estimated model**. It is a sanity check that the
learned policy is far better than random. It is *not* the course score: that
comes from the true MDP and a secret start distribution, and is available
only through Gradescope.

Plots:

- `output/small.png`: V(s) with the policy's arrows. Every arrow leads toward
  the +10 cell; the +3 cell isn't worth the detour.
- `output/medium.png`: the chosen acceleration over (position, velocity).
  Mostly "accelerate the way you are already moving" (action 1 at low
  velocity bins, 7 at high), which pumps energy to climb out of the valley.
  It is noisy in the middle, where data are sparse and the problem is
  not quite Markov.

## 4. Try it

    make verify                                  # lint, types, tests
    make plot-small                              # small.policy, small.json, small.png
    make policy-medium SOLVE_medium="--gamma 0.999 --max-iter 5000"

Ideas to explore next:

- **Smooth medium.** Fit Q with a function approximator over (pos, vel), or
  pool counts from neighbouring cells. Either would remove the speckle in
  the middle of the plot.
- **Exploit large's structure.** If every grid shares one movement model,
  pooling transitions across cells would estimate T far better from the
  same data.
- **Be pessimistic about rare transitions**, e.g. by penalising rarely tried
  `(s, a)`, so the policy relies less on transitions seen once or twice.

## A note on course policy

The course requires students to write and type their own code. This
implementation is for understanding while auditing, and should not be
submitted as a student's own work.
