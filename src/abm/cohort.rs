//! This module contains the Cohort class, a container for multiple Teams.

use super::{
    super::utilities::{parameters::Parameters, Solution},
    agent::{Agent, AgentMethods},
    team::{Team, TeamMethods},
};
use crate::problems::Ackley;
use rayon::prelude::*;
use std::marker::PhantomData;

/// This is the Cohort class, a container for multiple teams
///
/// The cohort allows you to run a set of teams easily to achieve statistical significance:
/// ```
/// use cisat::{Parameters, Cohort, problems::Ackley};
/// let mut x = Cohort::<Ackley>::new(Parameters::default());
/// x.solve();
/// ```
///Above, we specify one generic parameters `Ackley`, which tells the `Cohort` which problem to solve.
/// However, we can get more complciated than that! For instance, we feed in a struct that implements
/// `AgentMethods` to specific which type of agent to use:
/// ```
/// use cisat::{Parameters, Cohort, problems::Ackley, Agent};
/// let mut x = Cohort::<Ackley, Agent<Ackley>>::new(Parameters::default());
/// x.solve();
/// ```
/// Here, we just fed in the in-build `Agent` class which already implements `AgentMethods`. You can
/// implement the trait yourself to define a new agent though. The same applies for 'Team' and
/// 'TeamMethods':
/// ```
/// use cisat::{Parameters, Cohort, problems::Ackley, Agent, Team};
/// let mut x = Cohort::<Ackley, Agent<Ackley>, Team<Ackley, Agent<Ackley>>>::new(Parameters::default());
/// x.solve();
/// ```

#[derive(Clone, Debug)]
pub struct Cohort<S = Ackley, A = Agent<S>, T = Team<S, A>>
where
    S: Solution,
    A: AgentMethods<S>,
    T: TeamMethods<S, A>,
{
    /// This contains the parameters for the class
    parameters: Parameters,
    /// This contains the teams in the cohort
    team_list: Vec<T>,
    /// Bookkeeping the solution type
    solution_type: PhantomData<S>,
    /// Bookkeeping the agent-type
    agent_type: PhantomData<A>,
}

// /// A trait for cohort methods
// pub trait CohortMethods<
//     S: Solution,
//     A: AgentMethods<S> = Agent<S>,
//     T: TeamMethods<S, A> = Team<S, A>,
// >
// {
//     /// This generates a new cohort
//     fn new(parameters: Parameters) -> Self;
//     /// This runs the cohort using parallelism
//     fn solve(&mut self);
//     /// This runs a single iteration
//     fn iterate(&mut self);
//     /// Get the current best solution
//     fn get_best_solution_so_far(&mut self) -> f64;
// }

impl<S, A, T> Cohort<S, A, T>
where
    S: Solution,
    A: AgentMethods<S>,
    T: TeamMethods<S, A>,
{
    /// This generates a new cohort
    pub fn new(parameters: Parameters) -> Cohort<S, A, T> {
        Cohort {
            team_list: (0..parameters.number_of_teams)
                .map(|_| T::new(parameters.clone()))
                .collect(),
            solution_type: Default::default(),
            parameters,
            agent_type: Default::default(),
        }
    }

    /// This runs the cohort using parallelism
    pub fn solve(&mut self) {
        self.team_list.par_iter_mut().for_each(|x| x.solve());
    }

    /// This runs a single iteration
    pub fn iterate(&mut self) {
        self.team_list.iter_mut().for_each(|x| x.iterate());
    }

    /// Get the current best solution
    pub fn get_best_solution_so_far(&mut self) -> f64 {
        (0..self.parameters.number_of_teams)
            .map(|i| self.team_list[i].get_best_solution_so_far())
            .collect::<Vec<S>>()
            .into_iter()
            .max()
            .unwrap()
            .get_quality_scalar()
    }
}

impl Default for Cohort {
    fn default() -> Self {
        Cohort::new(Default::default())
    }
}
