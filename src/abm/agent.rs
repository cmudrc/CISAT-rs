//! Individual search, operational learning, and solution sharing.
use crate::utilities::randomness::{multinomial_draw, normalize, random_unit_draw};
use crate::{OperationalLearning, Parameters, Solution, TemperatureSchedule};
use rand::Rng;

/// One CISAT search agent. Each agent owns its learning and temperature state.
#[derive(Clone, Debug)]
pub struct Agent<S: Solution> {
    /// Index in the team communication snapshot.
    id: usize,
    /// Number of completed search steps.
    iteration_number: usize,
    /// Previous attempted operator, used as the Markov context.
    last_operation: usize,
    /// Current state of the hidden Markov model.
    hidden_state: usize,
    /// Effective temperature including the unmet-goal penalty.
    temperature: f64,
    /// Cooling schedule temperature before satisficing.
    annealing_temperature: f64,
    /// Agent-local Triki coefficient, reduced when an update is too large.
    triki_delta: f64,
    /// Candidate qualities collected within the current Triki dwell.
    quality_history: Vec<f64>,
    /// Normalized operator probabilities by learning context.
    learning_weights: Vec<Vec<f64>>,
    /// Fixed normalized hidden-state transition probabilities.
    transitions: Vec<Vec<f64>>,
    /// Accepted search state.
    current_solution: S,
    /// Highest-quality state seen by this agent.
    best_solution_so_far: S,
    /// Validated model settings.
    parameters: Parameters,
}

/// Extension points for alternative agent implementations.
pub trait AgentMethods<S: Solution>: Send {
    /// Generate an agent with its index within the team.
    fn new(id: usize, parameters: Parameters) -> Self;
    /// Perform one search step.
    fn iterate(&mut self);
    /// Return the best solution encountered, including shared solutions.
    fn get_best_solution_so_far(&mut self) -> S;
    /// Return the current solution.
    fn get_current_solution(&mut self) -> S;
    /// Consider a snapshot of all team solutions in agent-index order.
    fn communicate(&mut self, solutions: Vec<S>);
}

impl<S: Solution> AgentMethods<S> for Agent<S> {
    fn new(id: usize, parameters: Parameters) -> Self {
        parameters
            .validate_for_moves(S::NUMBER_OF_MOVE_OPERATORS)
            .expect("invalid CISAT parameters");
        assert!(
            id < parameters.number_of_agents,
            "agent id must be inside its team"
        );
        let n = S::NUMBER_OF_MOVE_OPERATORS;
        let uniform = || vec![1.0 / n as f64; n];
        let (mut learning_weights, mut transitions) = match &parameters.operational_learning {
            OperationalLearning::None => (vec![uniform()], vec![]),
            OperationalLearning::Multinomial {
                initial_learning_matrix,
                ..
            } => (
                vec![if initial_learning_matrix.is_empty() {
                    uniform()
                } else {
                    initial_learning_matrix.clone()
                }],
                vec![],
            ),
            OperationalLearning::Markov {
                initial_learning_matrix,
                ..
            } => (
                if initial_learning_matrix.is_empty() {
                    vec![uniform(); n]
                } else {
                    initial_learning_matrix.clone()
                },
                vec![],
            ),
            OperationalLearning::HiddenMarkov {
                initial_transition_matrix,
                initial_emission_matrix,
                ..
            } => {
                if initial_transition_matrix.is_empty() {
                    (vec![uniform(); n], vec![uniform(); n])
                } else {
                    (
                        initial_emission_matrix.clone(),
                        initial_transition_matrix.clone(),
                    )
                }
            }
        };
        learning_weights.iter_mut().for_each(|row| normalize(row));
        transitions.iter_mut().for_each(|row| normalize(row));
        let temperature = initial_temperature(&parameters.temperature_schedule);
        let triki_delta = match parameters.temperature_schedule {
            TemperatureSchedule::Triki { delta, .. } => delta,
            _ => 0.0,
        };
        let solution = S::new();
        assert!(
            solution.get_quality_scalar().is_finite(),
            "initial solution quality must be finite"
        );
        let mut agent = Self {
            id,
            iteration_number: 0,
            last_operation: rand::thread_rng().gen_range(0..n),
            hidden_state: rand::thread_rng().gen_range(0..learning_weights.len()),
            temperature,
            annealing_temperature: temperature,
            triki_delta,
            quality_history: vec![],
            learning_weights,
            transitions,
            best_solution_so_far: solution.clone(),
            current_solution: solution,
            parameters,
        };
        agent.update_temperature();
        agent
    }

    fn iterate(&mut self) {
        self.update_temperature();
        let row = match self.parameters.operational_learning {
            OperationalLearning::Markov { .. } => self.last_operation,
            OperationalLearning::HiddenMarkov { .. } => {
                self.hidden_state = multinomial_draw(&self.transitions[self.hidden_state]);
                self.hidden_state
            }
            _ => 0,
        };
        let operation = multinomial_draw(&self.learning_weights[row]);
        let mut candidate = self.current_solution.clone();
        let step = if matches!(
            self.parameters.temperature_schedule,
            TemperatureSchedule::None
        ) {
            1.0
        } else {
            self.temperature
        };
        candidate.apply_move_operator(operation, step);
        let old_quality = self.current_solution.get_quality_scalar();
        let new_quality = candidate.get_quality_scalar();
        let valid = new_quality.is_finite();
        let comparison = if valid {
            new_quality.total_cmp(&old_quality)
        } else {
            std::cmp::Ordering::Less
        };
        let rate = match self.parameters.operational_learning {
            OperationalLearning::None => 0.0,
            OperationalLearning::Multinomial { learning_rate, .. }
            | OperationalLearning::Markov { learning_rate, .. }
            | OperationalLearning::HiddenMarkov { learning_rate, .. } => learning_rate,
        };
        reinforce(&mut self.learning_weights[row], operation, comparison, rate);
        self.last_operation = operation;
        // Equal-quality moves are accepted even at zero temperature.
        if valid
            && (new_quality >= old_quality
                || (self.temperature > 0.0
                    && random_unit_draw() < ((new_quality - old_quality) / self.temperature).exp()))
        {
            self.current_solution = candidate;
        }
        self.remember_best();
        if matches!(
            self.parameters.temperature_schedule,
            TemperatureSchedule::Triki { .. }
        ) {
            self.quality_history
                .push(if valid { new_quality } else { old_quality });
        }
        self.iteration_number += 1;
    }

    fn get_best_solution_so_far(&mut self) -> S {
        self.best_solution_so_far.clone()
    }
    fn get_current_solution(&mut self) -> S {
        self.current_solution.clone()
    }

    fn communicate(&mut self, mut solutions: Vec<S>) {
        assert_eq!(
            solutions.len(),
            self.parameters.number_of_agents,
            "sharing requires one solution per agent"
        );
        let qualities: Vec<_> = solutions.iter().map(Solution::get_quality_scalar).collect();
        assert!(
            qualities.iter().all(|q| q.is_finite()),
            "shared quality must be finite"
        );
        let weights = sharing_weights(
            &qualities,
            self.id,
            self.parameters.self_bias,
            self.parameters.quality_bias,
        );
        self.current_solution = solutions.remove(multinomial_draw(&weights));
        self.remember_best();
    }
}

impl<S: Solution> Agent<S> {
    /// Temperature used for the current (or most recent) search step.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }
    /// Current normalized operator weights: one row, previous-move rows, or hidden-state rows.
    pub fn learning_weights(&self) -> &[Vec<f64>] {
        &self.learning_weights
    }

    /// Retain an improvement after either search or communication.
    fn remember_best(&mut self) {
        if self.current_solution.get_quality_scalar()
            > self.best_solution_so_far.get_quality_scalar()
        {
            self.best_solution_so_far = self.current_solution.clone();
        }
    }

    /// Apply the schedule at dwell boundaries and blend the current goal penalty.
    fn update_temperature(&mut self) {
        let initial = initial_temperature(&self.parameters.temperature_schedule);
        match self.parameters.temperature_schedule {
            TemperatureSchedule::None => {
                self.temperature = 0.0;
                return;
            }
            TemperatureSchedule::Geometric { dwell, .. } => {
                self.annealing_temperature =
                    initial * 0.95_f64.powf((self.iteration_number / dwell) as f64);
            }
            TemperatureSchedule::Cauchy { delta, dwell, .. } => {
                self.annealing_temperature =
                    initial / (1.0 + delta * (self.iteration_number / dwell) as f64);
            }
            TemperatureSchedule::Triki { dwell, .. } => {
                if self.iteration_number > 0 && self.iteration_number.is_multiple_of(dwell) {
                    self.annealing_temperature = triki_update(
                        self.annealing_temperature,
                        &mut self.triki_delta,
                        &self.quality_history,
                    );
                    self.quality_history.clear();
                }
            }
        }
        self.temperature = match self.current_solution.satisficing_penalty() {
            Some(penalty) => {
                assert!(
                    penalty.is_finite() && (0.0..=1.0).contains(&penalty),
                    "satisficing penalty must be in [0, 1]"
                );
                let fraction = self.parameters.satisficing_fraction;
                (1.0 - fraction) * self.annealing_temperature + fraction * initial * penalty
            }
            None => self.annealing_temperature,
        };
    }
}

/// Extract the schedule initial value; greedy search uses zero.
fn initial_temperature(schedule: &TemperatureSchedule) -> f64 {
    match *schedule {
        TemperatureSchedule::Geometric {
            initial_temperature,
            ..
        }
        | TemperatureSchedule::Cauchy {
            initial_temperature,
            ..
        }
        | TemperatureSchedule::Triki {
            initial_temperature,
            ..
        } => initial_temperature,
        TemperatureSchedule::None => 0.0,
    }
}

/// Multiply the attempted operator weight by the outcome factor and renormalize.
fn reinforce(weights: &mut [f64], operation: usize, comparison: std::cmp::Ordering, rate: f64) {
    use std::cmp::Ordering;
    weights[operation] *= match comparison {
        Ordering::Greater => 1.0 + rate,
        Ordering::Less => 1.0 - rate,
        Ordering::Equal => 1.0,
    };
    normalize(weights);
}

/// Combine relative quality, uniform bias, and self-bias without overflowing.
fn sharing_weights(qualities: &[f64], id: usize, self_bias: f64, quality_bias: f64) -> Vec<f64> {
    // Scale before subtracting the worst quality, allowing signed objective values.
    let scale = qualities.iter().map(|q| q.abs()).fold(1.0_f64, f64::max);
    let worst = qualities
        .iter()
        .map(|q| q / scale)
        .fold(f64::INFINITY, f64::min);
    let mut weights: Vec<_> = qualities.iter().map(|q| q / scale - worst).collect();
    normalize(&mut weights); // Ties become a uniform distribution.
    let bias_scale = self_bias.max(quality_bias).max(1.0);
    weights
        .iter_mut()
        .for_each(|w| *w = *w / bias_scale + quality_bias / bias_scale);
    weights[id] += self_bias / bias_scale;
    normalize(&mut weights);
    weights
}

/// Cool from population variance, adapting delta or holding when necessary.
fn triki_update(temperature: f64, delta: &mut f64, history: &[f64]) -> f64 {
    if history.len() < 2 {
        return temperature;
    }
    // Scaling protects the variance calculation for very large finite objectives.
    let scale = history.iter().map(|q| q.abs()).fold(1.0_f64, f64::max);
    let mean = history.iter().map(|q| q / scale).sum::<f64>() / history.len() as f64;
    let variance = history
        .iter()
        .map(|q| (q / scale - mean).powi(2))
        .sum::<f64>()
        / history.len() as f64;
    if variance <= 0.0 {
        return temperature;
    }
    let mut factor = (*delta / scale) * (temperature / scale) / variance;
    // Match the reference implementation's adaptive delta and hold behavior.
    if factor > 1.0 {
        *delta /= 2.0;
        factor /= 2.0;
    }
    if !factor.is_finite() || factor > 1.0 {
        temperature
    } else {
        temperature * (1.0 - factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cmp::Ordering, ops::Sub};

    #[derive(Clone, Debug)]
    struct Toy {
        quality: i32,
        step: f64,
    }
    impl Solution for Toy {
        const NUMBER_OF_MOVE_OPERATORS: usize = 2;
        const NUMBER_OF_OBJECTIVES: usize = 1;
        fn new() -> Self {
            Self {
                quality: 0,
                step: 0.0,
            }
        }
        fn apply_move_operator(&mut self, index: usize, temperature: f64) {
            self.quality += if index == 0 { 1 } else { -1 };
            self.step = temperature;
        }
        fn get_quality_scalar(&self) -> f64 {
            self.quality as f64
        }
        fn satisficing_penalty(&self) -> Option<f64> {
            Some(0.25)
        }
    }
    impl PartialEq for Toy {
        fn eq(&self, b: &Self) -> bool {
            self.quality == b.quality
        }
    }
    impl Eq for Toy {}
    impl PartialOrd for Toy {
        fn partial_cmp(&self, b: &Self) -> Option<Ordering> {
            Some(self.cmp(b))
        }
    }
    impl Ord for Toy {
        fn cmp(&self, b: &Self) -> Ordering {
            self.quality.cmp(&b.quality)
        }
    }
    impl Sub for Toy {
        type Output = f64;
        fn sub(self, b: Self) -> f64 {
            (self.quality - b.quality) as f64
        }
    }
    fn params() -> Parameters {
        Parameters {
            satisficing_fraction: 0.0,
            ..Parameters::default()
        }
    }

    #[test]
    fn reinforcement_rewards_penalizes_and_preserves_ties() {
        for (outcome, expected) in [
            (Ordering::Greater, 0.6),
            (Ordering::Less, 1.0 / 3.0),
            (Ordering::Equal, 0.5),
        ] {
            let mut weights = [0.5, 0.5];
            reinforce(&mut weights, 0, outcome, 0.5);
            assert!((weights[0] - expected).abs() < 1e-12);
        }
    }
    #[test]
    fn multinomial_controls_the_move_and_greedy_rejects_worsening() {
        let mut a = Agent::<Toy>::new(
            0,
            Parameters {
                temperature_schedule: TemperatureSchedule::None,
                operational_learning: OperationalLearning::Multinomial {
                    learning_rate: 0.1,
                    initial_learning_matrix: vec![0.0, 1.0],
                },
                ..params()
            },
        );
        a.iterate();
        assert_eq!(a.current_solution.quality, 0);
        assert_eq!(a.last_operation, 1);
        a.learning_weights[0] = vec![1.0, 0.0];
        a.iterate();
        assert_eq!(a.current_solution.quality, 1);
        assert_eq!(a.current_solution.step, 1.0);
    }
    #[test]
    fn markov_selects_and_updates_only_the_previous_operator_row() {
        let mut a = Agent::<Toy>::new(
            0,
            Parameters {
                operational_learning: OperationalLearning::Markov {
                    learning_rate: 0.5,
                    initial_learning_matrix: vec![vec![0.5, 0.5], vec![1.0, 0.0]],
                },
                ..params()
            },
        );
        a.last_operation = 1;
        a.iterate();
        assert_eq!(a.last_operation, 0);
        assert_eq!(a.current_solution.quality, 1);
        assert_eq!(a.learning_weights[0], vec![0.5, 0.5]);
        a.iterate();
        assert_eq!(a.learning_weights[1], vec![1.0, 0.0]);
        assert_ne!(a.learning_weights[0], vec![0.5, 0.5]);
    }
    #[test]
    fn hidden_markov_transitions_then_emits_and_learns() {
        let transitions = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let mut a = Agent::<Toy>::new(
            0,
            Parameters {
                operational_learning: OperationalLearning::HiddenMarkov {
                    learning_rate: 0.5,
                    initial_transition_matrix: transitions.clone(),
                    initial_emission_matrix: vec![vec![1.0, 0.0], vec![0.5, 0.5]],
                },
                ..params()
            },
        );
        a.hidden_state = 0;
        a.iterate();
        assert_eq!(a.hidden_state, 1);
        assert_eq!(a.learning_weights[0], vec![1.0, 0.0]);
        assert_ne!(a.learning_weights[1], vec![0.5, 0.5]);
        assert_eq!(a.transitions, transitions);
        a.iterate();
        assert_eq!(a.hidden_state, 0);
        assert_eq!(a.last_operation, 0);
    }
    #[test]
    fn geometric_and_cauchy_obey_dwell_and_reach_move_operator() {
        for (schedule, expected) in [
            (
                TemperatureSchedule::Geometric {
                    initial_temperature: 10.0,
                    dwell: 2,
                },
                9.5,
            ),
            (
                TemperatureSchedule::Cauchy {
                    initial_temperature: 10.0,
                    delta: 0.5,
                    dwell: 2,
                },
                10.0 / 1.5,
            ),
        ] {
            let mut a = Agent::<Toy>::new(
                0,
                Parameters {
                    temperature_schedule: schedule,
                    operational_learning: OperationalLearning::Multinomial {
                        learning_rate: 0.0,
                        initial_learning_matrix: vec![1.0, 0.0],
                    },
                    ..params()
                },
            );
            for _ in 0..2 {
                a.iterate();
                assert_eq!(a.current_solution.step, 10.0);
            }
            a.iterate();
            assert!((a.current_solution.step - expected).abs() < 1e-12);
        }
    }
    #[test]
    fn triki_uses_variance_and_holds_for_constant_history() {
        let mut delta = 0.01;
        assert!((triki_update(10.0, &mut delta, &[1.0, 3.0]) - 9.0).abs() < 1e-12);
        assert_eq!(triki_update(10.0, &mut delta, &[2.0, 2.0]), 10.0);
        delta = 1.0;
        assert_eq!(triki_update(10.0, &mut delta, &[1.0, 3.0]), 10.0);
        assert_eq!(delta, 0.5);
    }
    #[test]
    fn triki_updates_after_a_complete_dwell() {
        let mut a = Agent::<Toy>::new(
            0,
            Parameters {
                temperature_schedule: TemperatureSchedule::Triki {
                    initial_temperature: 1.0,
                    delta: 0.01,
                    dwell: 2,
                },
                operational_learning: OperationalLearning::Multinomial {
                    learning_rate: 0.0,
                    initial_learning_matrix: vec![1.0, 0.0],
                },
                ..params()
            },
        );
        a.iterate();
        a.iterate();
        assert_eq!(a.temperature(), 1.0);
        a.iterate();
        assert!((a.temperature() - 0.96).abs() < 1e-12);
        assert_eq!(a.quality_history.len(), 1);
    }
    #[test]
    fn satisficing_blends_problem_goal_with_annealing() {
        let a = Agent::<Toy>::new(
            0,
            Parameters {
                satisficing_fraction: 0.5,
                temperature_schedule: TemperatureSchedule::Geometric {
                    initial_temperature: 10.0,
                    dwell: 1,
                },
                ..params()
            },
        );
        assert_eq!(a.temperature(), 6.25);
    }
    #[test]
    fn sharing_handles_negative_equal_and_extreme_qualities() {
        assert_eq!(
            sharing_weights(&[-5.0, -3.0, -1.0], 0, 0.0, 0.0),
            vec![0.0, 1.0 / 3.0, 2.0 / 3.0]
        );
        assert_eq!(sharing_weights(&[-5.0, -5.0], 0, 0.0, 0.0), vec![0.5, 0.5]);
        let weights = sharing_weights(&[-f64::MAX, f64::MAX], 0, f64::MAX, f64::MAX);
        assert!(weights.iter().all(|w| w.is_finite()));
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let biased = sharing_weights(&[0.0, 1.0], 0, 2.0, 0.0);
        assert!(biased[0] > biased[1]);
    }
    #[test]
    fn shared_best_is_remembered_immediately() {
        let mut a = Agent::<Toy>::new(
            0,
            Parameters {
                number_of_agents: 2,
                self_bias: 0.0,
                quality_bias: 0.0,
                ..params()
            },
        );
        a.communicate(vec![
            Toy {
                quality: -1,
                step: 0.0,
            },
            Toy {
                quality: 5,
                step: 0.0,
            },
        ]);
        assert_eq!(a.get_best_solution_so_far().quality, 5);
    }
}
