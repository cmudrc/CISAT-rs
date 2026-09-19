[![CI](https://github.com/cmudrc/CISAT-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/cmudrc/CISAT-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/cisat.svg)](https://crates.io/crates/cisat)
[![docs.rs](https://docs.rs/cisat/badge.svg)](https://docs.rs/cisat)

# CISAT in Rust

Cognitively-Inspired Simulated Annealing Teams (CISAT) models interacting search agents with operational learning, adaptive cooling, and satisficing. Includes Ackley and planar truss problems, custom problem support, and parallel execution.

Requires Rust 1.87 or later.

## Quick start

```rust
use cisat::{problems::Ackley, Cohort, Parameters};

let mut cohort = Cohort::<Ackley<5>>::new(Parameters::default());
cohort.solve();
println!("Best quality: {}", cohort.get_best_solution_so_far());
```

CISAT maximizes quality; `Ackley::objective()` returns the original minimization objective.

From a checkout:

```sh
cargo run --release -- --problem ackley --teams 10 --iter 1000 --parallel
cargo run -- --help
```

[Usage, configuration, and references](USAGE.md) · [Algorithms and assumptions](ALGORITHMS.md) · [Custom example](examples/custom_implementation.rs) · [Changelog](CHANGELOG.md)
