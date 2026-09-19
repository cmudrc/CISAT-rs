//! A custom problem, agent wrapper, and team wrapper, all runnable.
use cisat::{
    Agent, AgentMethods, Cohort, Parameters, Solution, Team, TeamMethods, TemperatureSchedule,
};
use std::ops::Sub;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CustomProblem(i32);
impl Solution for CustomProblem {
    const NUMBER_OF_MOVE_OPERATORS: usize = 2;
    const NUMBER_OF_OBJECTIVES: usize = 1;
    fn new() -> Self {
        Self(0)
    }
    fn apply_move_operator(&mut self, move_index: usize, _temperature: f64) {
        self.0 = (self.0 + if move_index == 0 { 1 } else { -1 }).clamp(0, 10);
    }
    fn get_quality_scalar(&self) -> f64 {
        self.0 as f64
    }
}
impl Sub for CustomProblem {
    type Output = f64;
    fn sub(self, rhs: Self) -> f64 {
        (self.0 - rhs.0) as f64
    }
}

struct CustomAgent(Agent<CustomProblem>);
impl AgentMethods<CustomProblem> for CustomAgent {
    fn new(id: usize, parameters: Parameters) -> Self {
        Self(Agent::new(id, parameters))
    }
    fn iterate(&mut self) {
        self.0.iterate();
    }
    fn get_best_solution_so_far(&mut self) -> CustomProblem {
        self.0.get_best_solution_so_far()
    }
    fn get_current_solution(&mut self) -> CustomProblem {
        self.0.get_current_solution()
    }
    fn communicate(&mut self, solutions: Vec<CustomProblem>) {
        self.0.communicate(solutions);
    }
}
struct CustomTeam(Team<CustomProblem, CustomAgent>);
impl TeamMethods<CustomProblem, CustomAgent> for CustomTeam {
    fn new(parameters: Parameters) -> Self {
        Self(Team::new(parameters))
    }
    fn iterate(&mut self) {
        self.0.iterate();
    }
    fn communicate(&mut self) {
        self.0.communicate();
    }
    fn solve(&mut self) {
        self.0.solve();
    }
    fn get_best_solution_so_far(&mut self) -> CustomProblem {
        self.0.get_best_solution_so_far()
    }
}
fn main() {
    let mut cohort = Cohort::<CustomProblem, CustomAgent, CustomTeam>::new(Parameters {
        temperature_schedule: TemperatureSchedule::None,
        ..Parameters::default()
    });
    cohort.solve();
    println!("Best custom quality: {}", cohort.get_best_solution_so_far());
}
