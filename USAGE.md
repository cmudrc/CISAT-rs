# Usage and development

Usage notes for CISAT 0.3.0. See [the README](README.md) for a quick start and [ALGORITHMS.md](ALGORITHMS.md) for model definitions.

## Library

```rust
use cisat::{problems::Ackley, Cohort, CommunicationStyle, Parameters};

let parameters = Parameters {
    communication: CommunicationStyle::RegularInterval { interval: 5 },
    ..Parameters::default().with_teams(10).with_iters(100)
};
let mut cohort = Cohort::<Ackley<5>>::new(parameters);
cohort.solve();
println!("Best quality: {}", cohort.get_best_solution_so_far());
```

CISAT **maximizes** a finite quality scalar. Convert minimization objectives before returning them. `Ackley::objective()` exposes the untransformed objective (minimum zero); cohort results report the transformed quality.

Implement `Solution` to add a problem. Its ordering and subtraction must agree with `get_quality_scalar()`. The optional `satisficing_penalty()` method returns a value in `[0, 1]`: zero means the goal is met, one means a fully unmet goal. Its default is `None`, which disables satisficing for custom problems without a defined goal. `AgentMethods` and `TeamMethods` allow custom agent and team behavior; [custom_implementation.rs](examples/custom_implementation.rs) is a runnable example of all three extension points.

`solve()` adds the configured number of steps on each call; `iterate()` adds one step. A zero-step run reports the best initial solution. Teams and agents must be nonempty. Use `validate()` or `validate_for_moves()` for recoverable configuration errors; constructors assert these preconditions. Empty learning matrices mean uniform initial weights, and a learning rate of zero freezes those weights.

## Command line

Run these commands from a checkout:

```sh
cargo run --release -- --problem ackley --teams 10 --agents 3 --iter 1000 --parallel
cargo run --release -- --problem structure --learning markov --schedule triki --dwell 20 --communication-interval 5
cargo run -- --help
cargo run --example custom_implementation
```

`--learning` accepts `none`, `multinomial`, `markov`, and `hidden-markov` (also `HiddenMarkov`). `--schedule` accepts `none`, `geometric`, `cauchy`, and `triki`; values are case insensitive. Communication defaults to disabled. Choose one of `--communication-frequency 0.1`, `--communication-interval 5`, or `--meetings 1,10,50`. `--verbose` prints the final cohort state. Invalid arguments produce an error and a nonzero exit code.

## Configuration files

`Parameters::from_file(path)` reads JSON; `load_from_file` replaces an existing configuration only after parsing and validation succeed. Omitted top-level fields use defaults, and unknown fields are rejected. For example:

```json
{
  "number_of_teams": 5,
  "number_of_agents": 3,
  "number_of_iterations": 1000,
  "temperature_schedule": {"Triki": {"initial_temperature": 1.0, "delta": 0.05, "dwell": 20}},
  "operational_learning": {"Markov": {"learning_rate": 0.05, "initial_learning_matrix": []}},
  "communication": {"RegularInterval": {"interval": 5}}
}
```

The CLI accepts individual flags; JSON loading is a library API. Serialize `Parameters` with `serde_json` to capture the full configuration. Random draws currently use thread-local RNGs: runs are stochastic and serial/parallel runs are not guaranteed to follow identical trajectories.

## Development

```sh
cargo fmt --check
cargo test --locked --all-targets
cargo test --locked --doc
cargo clippy --locked --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps
cargo package --locked --allow-dirty
```

The repository lockfile supports repeatable CLI builds. TrussX comes from crates.io, with no moving Git dependency. The truss example uses linear elastic analysis, an ideal pinned-end Euler buckling check, a fixed material, a bounded design domain, and explicit mass/safety goals; it is not a design-code check or a certified structural optimizer.

## References

1. McComb, C., Cagan, J., & Kotovsky, K. (2015). Lifting the Veil: Drawing insights about design teams from a cognitively-inspired computational model. Design Studies, 40, 119-142. doi:[10.1016/j.destud.2015.06.005](https://doi.org/10.1016/j.destud.2015.06.005). [PDF](https://github.com/cmudrc/CISAT-rs/blob/master/literature/2015_DesignStudies_LiftingTheVeil.pdf)
1. McComb, C., Cagan, J., & Kotovsky, K. (2016). Drawing inspiration from human design teams for better search and optimization: The heterogeneous simulated annealing teams algorithm. Journal of Mechanical Design, 138(4). doi:[10.1115/1.4032810](https://doi.org/10.1115/1.4032810). [PDF](https://github.com/cmudrc/CISAT-rs/blob/master/literature/2016_JMD_HSAT.pdf)
2. McComb, C., Cagan, J., & Kotovsky, K. (2017). Capturing human sequence-learning abilities in configuration design tasks through markov chains. Journal of Mechanical Design, 139(9). doi:[10.1115/1.4037185](https://doi.org/10.1115/1.4037185). [PDF](https://github.com/cmudrc/CISAT-rs/blob/master/literature/2017_JMD_MarkovChain.pdf)
1. McComb, C., Cagan, J., & Kotovsky, K. (2017). Optimizing design teams based on problem properties: computational team simulations and an applied empirical test. Journal of Mechanical Design, 139(4). doi:[10.1115/1.4035793](https://doi.org/10.1115/1.4035793). [PDF](https://github.com/cmudrc/CISAT-rs/blob/master/literature/2017_JMD_OptimizingTeams.pdf)
