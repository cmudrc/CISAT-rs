//! This module contains the Parameters struct and a number of enums
use serde::{Deserialize, Serialize};
use std::fmt;

/// This enum carries temperature schedule options
#[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum TemperatureSchedule {
    /// This is the Triki temperature schedule
    Triki {
        /// Initial temperature
        initial_temperature: f64,
        /// Delta updating parameter
        delta: f64,
        /// Number of search steps between annealing updates.
        dwell: usize,
    },
    /// This is the Cauchy temperature schedule
    Cauchy {
        /// Initial temperature
        initial_temperature: f64,
        /// Delta updating parameter
        delta: f64,
        /// Number of search steps between annealing updates.
        dwell: usize,
    },
    /// Geometric cooling with a fixed factor of 0.95 per dwell.
    Geometric {
        /// Initial temperature
        initial_temperature: f64,
        /// Number of search steps between annealing updates.
        dwell: usize,
    },
    /// Greedy acceptance with unit move scale; satisficing is disabled.
    None,
}

/// This enum contains options for how the agent learns
#[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum OperationalLearning {
    /// Reinforcement over the action set as a multinomial distribution
    Multinomial {
        /// Learning rate
        learning_rate: f64,
        /// Initial move weights
        initial_learning_matrix: Vec<f64>,
    },
    /// Reinforcement over a sequence-based action set
    Markov {
        /// Learning rate
        learning_rate: f64,
        /// Initial transition matrix
        initial_learning_matrix: Vec<Vec<f64>>,
    },
    /// Reinforcement over an HMM-based action set
    HiddenMarkov {
        /// Learning rate
        learning_rate: f64,
        /// Initial transition matrix
        initial_transition_matrix: Vec<Vec<f64>>,
        /// State-by-operator emission weights (learned); empty means uniform.
        initial_emission_matrix: Vec<Vec<f64>>,
    },
    /// Uniform operator selection without reinforcement.
    None,
}

/// This enum contains options for agent interaction
#[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum CommunicationStyle {
    /// Interaction based on a constant frequency value
    ConstantFrequency {
        /// Frequency value
        frequency: f64,
    },
    /// Interaction at regular intervals
    RegularInterval {
        /// Interval value
        interval: usize,
    },
    /// Interaction at scheduled times
    ScheduledMeetings {
        /// Scheduled times
        times: Vec<usize>,
    },
    /// Independent agents with no communication.
    None,
}

/// This parameters struct. This tells CISAT what to do
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(default)]
pub struct Parameters {
    /// Number of teams
    pub number_of_teams: usize,
    /// Number of agents on each team
    pub number_of_agents: usize,
    /// The number of iterations the team will run for
    pub number_of_iterations: usize,
    /// The temperature schedule to use
    pub temperature_schedule: TemperatureSchedule,
    /// The operational learning style to use
    pub operational_learning: OperationalLearning,
    /// The communication style to use
    pub communication: CommunicationStyle,
    /// The self bias value to use
    pub self_bias: f64,
    /// The quality bias value to use
    pub quality_bias: f64,
    /// The satisficing fraction to use
    pub satisficing_fraction: f64,
}

impl Parameters {
    /// Read and validate a JSON configuration. Missing fields use defaults.
    /// The receiver is unchanged if parsing or validation fails.
    pub fn load_from_file(&mut self, file_name: String) -> Result<(), Box<dyn std::error::Error>> {
        *self = Self::from_file(file_name)?;
        Ok(())
    }

    /// Construct parameters from a JSON file, returning I/O, syntax, or validation errors.
    pub fn from_file(
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let parameters: Self = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        parameters.validate().map_err(std::io::Error::other)?;
        Ok(parameters)
    }

    /// Validate configuration independently of the problem's move count.
    pub fn validate(&self) -> Result<(), String> {
        if self.number_of_teams == 0 || self.number_of_agents == 0 {
            return Err("teams and agents must both be positive".into());
        }
        for (name, value) in [
            ("self_bias", self.self_bias),
            ("quality_bias", self.quality_bias),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("{name} must be finite and nonnegative"));
            }
        }
        if !self.satisficing_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.satisficing_fraction)
        {
            return Err("satisficing_fraction must be in [0, 1]".into());
        }
        match self.temperature_schedule {
            TemperatureSchedule::None => (),
            TemperatureSchedule::Geometric {
                initial_temperature,
                dwell,
            } => {
                validate_temperature(initial_temperature, dwell)?;
            }
            TemperatureSchedule::Cauchy {
                initial_temperature,
                delta,
                dwell,
            }
            | TemperatureSchedule::Triki {
                initial_temperature,
                delta,
                dwell,
            } => {
                validate_temperature(initial_temperature, dwell)?;
                if !delta.is_finite() || delta <= 0.0 {
                    return Err("temperature delta must be finite and positive".into());
                }
            }
        }
        match &self.communication {
            CommunicationStyle::ConstantFrequency { frequency }
                if !frequency.is_finite() || !(0.0..=1.0).contains(frequency) =>
            {
                return Err("communication frequency must be in [0, 1]".into());
            }
            CommunicationStyle::RegularInterval { interval: 0 } => {
                return Err("communication interval must be positive".into());
            }
            CommunicationStyle::ScheduledMeetings { times } if times.contains(&0) => {
                return Err("scheduled meetings use one-based iteration numbers".into());
            }
            _ => (),
        }
        match &self.operational_learning {
            OperationalLearning::None => (),
            OperationalLearning::Multinomial {
                learning_rate,
                initial_learning_matrix,
            } => {
                validate_rate(*learning_rate)?;
                if !initial_learning_matrix.is_empty() {
                    validate_weights(initial_learning_matrix)?;
                }
            }
            OperationalLearning::Markov {
                learning_rate,
                initial_learning_matrix,
            } => {
                validate_rate(*learning_rate)?;
                if initial_learning_matrix
                    .iter()
                    .any(|row| row.len() != initial_learning_matrix.len())
                {
                    return Err("Markov learning requires a square matrix".into());
                }
                for row in initial_learning_matrix {
                    validate_weights(row)?;
                }
            }
            OperationalLearning::HiddenMarkov {
                learning_rate,
                initial_transition_matrix,
                initial_emission_matrix,
            } => {
                validate_rate(*learning_rate)?;
                if initial_transition_matrix.is_empty() != initial_emission_matrix.is_empty() {
                    return Err(
                        "HMM transition and emission matrices must both be provided or both empty"
                            .into(),
                    );
                }
                let states = initial_transition_matrix.len();
                if initial_transition_matrix.iter().any(|r| r.len() != states)
                    || initial_emission_matrix.len() != states
                {
                    return Err(
                        "HMM requires a square transition matrix and one emission row per state"
                            .into(),
                    );
                }
                if let Some(first) = initial_emission_matrix.first() {
                    if initial_emission_matrix
                        .iter()
                        .any(|row| row.len() != first.len())
                    {
                        return Err(
                            "HMM emission rows must have the same number of operators".into()
                        );
                    }
                }
                for row in initial_transition_matrix
                    .iter()
                    .chain(initial_emission_matrix)
                {
                    validate_weights(row)?;
                }
            }
        }
        Ok(())
    }

    /// Validate the learning matrices against a concrete problem's number of moves.
    pub fn validate_for_moves(&self, moves: usize) -> Result<(), String> {
        self.validate()?;
        if moves == 0 {
            return Err("a solution must define at least one move operator".into());
        }
        let valid = match &self.operational_learning {
            OperationalLearning::Multinomial {
                initial_learning_matrix: m,
                ..
            } => m.is_empty() || m.len() == moves,
            OperationalLearning::Markov {
                initial_learning_matrix: m,
                ..
            } => m.is_empty() || (m.len() == moves && m.iter().all(|r| r.len() == moves)),
            OperationalLearning::HiddenMarkov {
                initial_emission_matrix: m,
                ..
            } => m.iter().all(|r| r.len() == moves),
            OperationalLearning::None => true,
        };
        if !valid {
            return Err(format!(
                "learning matrix dimensions do not match {moves} move operators"
            ));
        }
        Ok(())
    }

    /// Assert that parameters are valid. Use `validate` for recoverable errors.
    pub fn verify(&self) {
        self.validate().expect("invalid CISAT parameters");
    }
    /// Returns values necessary to run HSAT
    pub fn hsat() -> Self {
        Parameters {
            temperature_schedule: TemperatureSchedule::Triki {
                initial_temperature: 100.0,
                delta: 0.05,
                dwell: 50,
            },
            communication: CommunicationStyle::RegularInterval { interval: 1 },
            self_bias: 0.0,
            quality_bias: 0.0,
            satisficing_fraction: 0.0,
            ..Default::default()
        }
    }
    /// Set teams value
    pub fn with_teams(mut self, number_of_teams: usize) -> Self {
        self.number_of_teams = number_of_teams;
        self
    }
    /// Set the number of agents per team.
    pub fn with_agents(mut self, number_of_agents: usize) -> Self {
        self.number_of_agents = number_of_agents;
        self
    }
    /// Set the number of search steps per solve call.
    pub fn with_iters(mut self, number_of_iterations: usize) -> Self {
        self.number_of_iterations = number_of_iterations;
        self
    }
}

impl Default for Parameters {
    fn default() -> Self {
        Parameters {
            number_of_teams: 1,
            number_of_agents: 3,
            number_of_iterations: 100,
            temperature_schedule: TemperatureSchedule::Geometric {
                initial_temperature: 1.0,
                dwell: 1,
            },
            operational_learning: OperationalLearning::None,
            communication: CommunicationStyle::None,
            self_bias: 1.0,
            quality_bias: 1.0,
            satisficing_fraction: 0.5,
        }
    }
}

impl fmt::Display for Parameters {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            " - {} teams, {} agents, {} iterations",
            self.number_of_teams, self.number_of_agents, self.number_of_iterations
        )?;
        writeln!(f, " - temperature: {:?}", self.temperature_schedule)?;
        writeln!(f, " - learning: {:?}", self.operational_learning)?;
        writeln!(f, " - communication: {:?}", self.communication)?;
        writeln!(
            f,
            " - self bias: {}, quality bias: {}, satisficing: {}",
            self.self_bias, self.quality_bias, self.satisficing_fraction
        )
    }
}

/// Require a finite positive starting temperature and a nonzero dwell.
fn validate_temperature(initial: f64, dwell: usize) -> Result<(), String> {
    if !initial.is_finite() || initial <= 0.0 || dwell == 0 {
        return Err(
            "initial temperature must be finite and positive; dwell must be positive".into(),
        );
    }
    Ok(())
}

/// Restrict reinforcement rates so a penalized row retains positive mass.
fn validate_rate(rate: f64) -> Result<(), String> {
    if !rate.is_finite() || !(0.0..1.0).contains(&rate) {
        return Err("learning rate must be in [0, 1)".into());
    }
    Ok(())
}

/// Validate a probability-weight row without requiring prior normalization.
fn validate_weights(weights: &[f64]) -> Result<(), String> {
    if weights.is_empty()
        || weights.iter().any(|w| !w.is_finite() || *w < 0.0)
        || !weights.iter().any(|w| *w > 0.0)
    {
        return Err(
            "weight rows must be nonempty, finite, nonnegative, and contain a positive weight"
                .into(),
        );
    }
    Ok(())
}
