use super::super::utilities::{
    parameters::{OperationalLearning, Parameters, TemperatureSchedule},
    Solution,
};

#[derive(Clone, Debug)]
pub struct Agent<T> {
    /// Agent_id
    agent_id: u64,
    /// The iteration number as tracked by the agent
    iteration_number: u64,
    /// The lst operation performed by the agent
    last_operation: u64,
    /// The temperature used by the agent
    temperature: f64,
    current_solution_quality: f64,
    best_quality_so_far: f64,
    current_solution: T,
    parameters: Parameters,
}

impl<T: Solution<T>> Agent<T> {
    fn reset(&mut self) {}

    fn generate_candidate_solution(&mut self) -> T {
        T::generate_initial_solution()
    }

    fn iterate(&mut self) {}

    fn update_temperature(&mut self) {
        match self.parameters.temperature_schedule {
            TemperatureSchedule::Triki {
                initial_temperature,
                delta,
            } => {
                self.temperature = initial_temperature / (self.iteration_number as f64);
            }
            TemperatureSchedule::Cauchy {
                initial_temperature,
            } => {
                self.temperature = initial_temperature / (self.iteration_number as f64);
            }
            TemperatureSchedule::Geometric {
                initial_temperature,
            } => {
                self.temperature = initial_temperature / (self.iteration_number as f64);
            }
        }
    }
}

pub fn build_agent<T: Solution<T>>(agent_id: u64, parameters: Parameters) -> Agent<T> {
    Agent {
        agent_id,
        iteration_number: 0,
        last_operation: 0,
        temperature: 0.0,
        current_solution_quality: 0.0,
        best_quality_so_far: 0.0,
        current_solution: T::generate_initial_solution(),
        parameters,
    }
}
