use super::super::utilities::{parameters::Parameters, Solution};
use super::agent::{build_agent, Agent};

#[derive(Clone, Debug)]
pub struct Team<T> {
    pub parameters: Parameters,
    agent_list: Vec<Agent<T>>,
}

impl<T> Team<T> {
    fn solve(&mut self) {}

    fn iterate(&mut self) {}

    fn pull_best_solution(&mut self) {}
}

pub fn build_team<T: Clone + Solution<T>>(parameters: Parameters) -> Team<T> {
    let agents = vec![build_agent(0, parameters.clone()); 1];
    Team {
        parameters,
        agent_list: agents,
    }
}
