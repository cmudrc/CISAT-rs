#![warn(clippy::all)]
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

//! This crate contains an implementation of the Cognitively-Inspired Simulated Annealing Teams \
//! (CISAT) framework.

mod utilities;
pub use utilities::{
    parameters::{CommunicationStyle, OperationalLearning, Parameters, TemperatureSchedule},
    Solution,
};

mod abm;
pub use abm::agent::{Agent, AgentMethods};
pub use abm::cohort::Cohort;
pub use abm::team::{Team, TeamMethods};

pub mod problems;
