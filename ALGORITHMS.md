# CISAT mechanisms and implementation choices

## Quality and acceptance

Every solution returns a finite scalar `q`; larger is better. An improving or equal-quality candidate is accepted. A worsening candidate is accepted with probability `exp((q_candidate - q_current) / T)` when `T > 0`. At zero temperature, worsening moves are rejected. Nonfinite candidates are rejected and penalized by learning. The initial and shared solutions must have finite quality.

The best-so-far solution is retained after both search and communication. Learning evaluates the proposed move before acceptance, so accepting a worsening move does not reinforce it as an improvement. `TemperatureSchedule::None` means greedy acceptance with unit move scale; other schedules pass the effective temperature into the problem's move operator.

## Operational learning

The implementation follows the multiplicative rules in [CISAT-cpp agent.cpp, revision 45b6864](https://github.com/cmudrc/CISAT-cpp/blob/45b6864fc8c83eb1b8eee56f0f74912fecd3973d/src/agents_and_teams/agent.cpp).

For learning rate `r`, multiply the selected operator's weight by `1+r` for an improvement or `1-r` for a worsening move; leave it unchanged on a tie. Normalize the affected row after every step. The supported range is `0 <= r < 1`, preventing an entire row from disappearing after an unsuccessful move. Zero initial weights remain zero. Empty matrices initialize uniformly.

- **None:** draw uniformly among every declared move operator.
- **Multinomial:** select and reinforce a single row over operators.
- **Markov:** select and reinforce the row indexed by the previous attempted operator; then remember the newly attempted operator, regardless of acceptance. The initial previous operator is uniform.
- **Hidden Markov:** transition from the current hidden state, draw an operator from the destination state's emission row, and reinforce that emission row. Transition weights stay fixed, matching the reference implementation. This is not Baum–Welch training. The initial hidden state is uniform. Transition matrices are state-by-state; emission matrices are state-by-operator. Empty matrices use one state per operator.

## Cooling and satisficing

Let `k = floor(completed_steps / dwell)`. The first dwell uses the initial temperature.

- **Geometric:** `T_anneal = T_initial * 0.95^k`. The factor 0.95 is a documented Rust implementation choice; the existing public variant has no rate field.
- **Cauchy:** `T_anneal = T_initial / (1 + delta*k)`.
- **Triki:** at each dwell boundary, calculate the population variance `v` of candidate qualities from the completed dwell. For `v > 0`, let `u = delta*T_anneal/v`. If `u > 1`, halve the agent's delta and `u` once. If `u` is still greater than one or nonfinite, hold the temperature; otherwise set `T_anneal *= 1-u`. Hold for zero variance or fewer than two samples. Clear the window after each update. This follows the adaptive-delta/hold behavior of the C++ implementation. Use `dwell >= 2` to enable adaptation; `dwell = 1` has no measurable variance and holds the initial annealing temperature.

For a problem-provided unmet-goal penalty `p` in `[0,1]` and satisficing fraction `s`, the effective temperature is `(1-s)*T_anneal + s*T_initial*p`. The problem owns the goal and its normalization. If it returns `None`, use `T_anneal` unchanged. Satisficing is applied on every search step, including after communication. A fully met goal with `s=1` yields zero temperature and may stop continuous moves; this is intentional. `None` cooling bypasses satisficing.

## Sharing and timing

Normalize qualities relative to the team's worst quality, so larger qualities receive larger nonnegative weights. If all qualities tie, use uniform weights. Add `quality_bias` uniformly and `self_bias` to the agent's own entry, then normalize again. This adapts the C++ minimization rule to Rust's maximization convention and explicitly handles ties, negative qualities, and large finite magnitudes.

Communication happens before search using a snapshot of every agent's current solution. All agents see the same snapshot. Meeting steps are one-based. Frequency zero never communicates and frequency one always communicates. Independent cohorts can use Rayon; communication stays within a team.

## Built-in problems

**Ackley** uses the standard dimension-normalized objective:

`f(x) = -20 exp(-0.2 sqrt(mean(x_i^2))) - exp(mean(cos(2*pi*x_i))) + e + 20`.

The reported quality is `20+e-f(x)`. Initial coordinates are uniform in `[-10,10]`; Cauchy perturbations are clipped to `[-32.768,32.768]`. The satisficing penalty is `clamp(f/(20+e), 0, 1)`, targeting the zero objective. This bounded search domain and penalty normalization are explicit demonstration choices.

**Structure** stores a cloneable geometry/topology description and builds a fresh published TrussX model for each evaluation. It fixes five bottom locations across a 10 m span: a pinned left support, a right roller, and two 20 kN downward loads at x=-2 m and x=3 m. All joints have restrained out-of-plane motion. The seed is a triangulated planar bridge. Free joints stay in x=[-5,5] m and y=[0.25,4] m; designs have at most 30 joints. Fixed support/load locations are never moved or removed.

The seven operators add an attached joint, remove a free joint, add a missing member, remove a member, resize one member, resize all members, and move a free joint. Moves with no eligible target leave the design unchanged. New joints connect to three nearby joints. Duplicate members, self-loops, and coincident joints are excluded.

Members are circular pipes with outer radii 5–50 mm and wall thickness `radius/7.5`, density 7850 kg/m³, Young's modulus 209 GPa, and yield strength 344 MPa. Each member's safety is the smaller of axial yielding and, in compression, ideal pinned-end Euler buckling. An unsolvable model receives quality zero. For solvable models, `quality = min(FOS/2,1)/(1+mass/250)`. Satisficing uses the larger normalized shortfall from a 250 kg mass goal and safety factor 2. These are demonstration goals, not a claim of equivalence to a published truss experiment. Connection design, imperfection effects, nonlinear response, local buckling, and design-code checks are outside this example.

## Validation boundaries

Tests cover known Ackley values, deterministic learning and cooling rules, sharing extremes, schedule timing, serial/parallel step counts, configuration errors, CLI behavior, truss mass, geometry invariants, and invalid-design penalties. They establish software behavior. They do not establish convergence guarantees, an optimization performance advantage, empirical human-team validity, or reproduction of the cited papers. Thread-local random generators currently prevent exact seeded replay; parallel and sequential stochastic trajectories can differ.
