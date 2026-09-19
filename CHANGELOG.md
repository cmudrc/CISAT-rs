# Changelog

## 0.3.0 — 2026-09-19

- Complete multinomial, Markov, and hidden Markov operator learning.
- Implement geometric, Cauchy, and Triki schedules, dwell intervals, and goal-based satisficing.
- Apply temperature to candidate generation and remember shared best solutions immediately.
- Normalize sharing weights for signed/equal qualities and prevent overflow in weight sums.
- Validate configuration, learning dimensions, and numerical boundaries; load JSON atomically.
- Correct the n-dimensional Ackley formula and implement all seven structural moves.
- Rebuild the truss example against published TrussX 0.2, including mass, yield, and buckling evaluation.
- Replace the incomplete custom example and obsolete CLI dependencies; fix iteration counts and add communication flags.
- Add regression tests, algorithm documentation, and GitHub Actions checks.
- Declare Rust 1.87 as the minimum version required by the current TrussX numerical dependency.

### Migration from 0.2.2

- Rust 1.87 or later is required. The crate uses edition 2021; downstream crates may retain their own edition.
- `Parameters::load_from_file` now returns `Result<(), Box<dyn std::error::Error>>`. Handle errors with `?` or an explicit match. `Parameters::from_file` constructs validated parameters directly.
- Ackley is now dimension-generic. `Cohort::<Ackley>` retains the five-dimensional default; use `Ackley::<5>::new()` for construction without a type annotation, or select another positive dimension explicitly.
- Custom solutions must return finite quality, with larger values preferred, and declare at least one move operator. Invalid settings now fail at construction; use `validate()` or `validate_for_moves()` first for recoverable configuration errors.
- The new `Solution::satisficing_penalty()` method defaults to `None`, so existing custom solutions need no additional method. Override it to provide an unmet-goal penalty in `[0, 1]`.
- Search results change because learning, cooling, satisficing, sharing, Ackley normalization, and structural moves now execute their documented behavior. Existing numerical baselines should be regenerated.
- The CLI adds communication controls and validates options. Serial execution now performs exactly the requested number of iterations; `--delta` controls both Cauchy and Triki schedules. See `cisat --help` for defaults.
