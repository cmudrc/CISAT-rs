use cisat::{
    problems::Ackley, Agent, AgentMethods, Cohort, CommunicationStyle, OperationalLearning,
    Parameters, Solution, Team, TeamMethods, TemperatureSchedule,
};
use std::{
    ops::Sub,
    sync::{Arc, Mutex},
};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct Counter(i64);
impl Solution for Counter {
    const NUMBER_OF_MOVE_OPERATORS: usize = 1;
    const NUMBER_OF_OBJECTIVES: usize = 1;
    fn new() -> Self {
        Self(0)
    }
    fn apply_move_operator(&mut self, _: usize, _: f64) {
        self.0 += 1;
    }
    fn get_quality_scalar(&self) -> f64 {
        self.0 as f64
    }
}
impl Sub for Counter {
    type Output = f64;
    fn sub(self, other: Self) -> f64 {
        (self.0 - other.0) as f64
    }
}

#[test]
fn serial_and_parallel_run_exactly_the_requested_steps() {
    for steps in [0, 1, 7] {
        let params = Parameters::default().with_teams(2).with_iters(steps);
        let mut serial = Cohort::<Counter>::new(params.clone());
        let mut parallel = Cohort::<Counter>::new(params);
        for _ in 0..steps {
            serial.iterate();
        }
        parallel.solve();
        assert_eq!(serial.get_best_solution_so_far(), steps as f64);
        assert_eq!(parallel.get_best_solution_so_far(), steps as f64);
    }
}

#[derive(Clone)]
struct ObservedAgent {
    steps: i64,
    meetings: Arc<Mutex<Vec<i64>>>,
}
thread_local! { static MEETINGS: Arc<Mutex<Vec<i64>>> = Arc::default(); }
impl AgentMethods<Counter> for ObservedAgent {
    fn new(_: usize, _: Parameters) -> Self {
        Self {
            steps: 0,
            meetings: MEETINGS.with(Arc::clone),
        }
    }
    fn iterate(&mut self) {
        self.steps += 1;
    }
    fn get_best_solution_so_far(&mut self) -> Counter {
        Counter(self.steps)
    }
    fn get_current_solution(&mut self) -> Counter {
        Counter(self.steps)
    }
    fn communicate(&mut self, solutions: Vec<Counter>) {
        assert!(
            solutions.iter().all(|s| s.0 == self.steps),
            "communication must use a synchronous snapshot"
        );
        self.meetings.lock().unwrap().push(self.steps + 1);
    }
}
#[test]
fn every_communication_style_uses_the_expected_steps() {
    for (style, expected) in [
        (CommunicationStyle::None, vec![]),
        (
            CommunicationStyle::RegularInterval { interval: 2 },
            vec![2, 4],
        ),
        (
            CommunicationStyle::ScheduledMeetings {
                times: vec![1, 3, 5],
            },
            vec![1, 3, 5],
        ),
        (
            CommunicationStyle::ConstantFrequency { frequency: 0.0 },
            vec![],
        ),
        (
            CommunicationStyle::ConstantFrequency { frequency: 1.0 },
            vec![1, 2, 3, 4, 5],
        ),
    ] {
        MEETINGS.with(|m| m.lock().unwrap().clear());
        let mut team = Team::<Counter, ObservedAgent>::new(Parameters {
            number_of_agents: 1,
            number_of_iterations: 5,
            communication: style,
            ..Parameters::default()
        });
        team.solve();
        MEETINGS.with(|m| assert_eq!(*m.lock().unwrap(), expected));
    }
}
#[test]
fn sharing_uses_a_snapshot_for_multiple_agents() {
    let mut team = Team::<Counter, ObservedAgent>::new(Parameters {
        number_of_agents: 3,
        number_of_iterations: 3,
        communication: CommunicationStyle::RegularInterval { interval: 1 },
        ..Parameters::default()
    });
    team.solve();
    assert_eq!(team.get_best_solution_so_far().0, 3);
}
#[test]
fn invalid_parameter_boundaries_and_learning_shapes_are_rejected() {
    let invalid = [
        Parameters::default().with_agents(0),
        Parameters::default().with_teams(0),
        Parameters {
            self_bias: f64::NAN,
            ..Parameters::default()
        },
        Parameters {
            quality_bias: -1.0,
            ..Parameters::default()
        },
        Parameters {
            satisficing_fraction: f64::INFINITY,
            ..Parameters::default()
        },
        Parameters {
            communication: CommunicationStyle::RegularInterval { interval: 0 },
            ..Parameters::default()
        },
        Parameters {
            communication: CommunicationStyle::ConstantFrequency {
                frequency: f64::NAN,
            },
            ..Parameters::default()
        },
        Parameters {
            communication: CommunicationStyle::ScheduledMeetings { times: vec![0] },
            ..Parameters::default()
        },
        Parameters {
            temperature_schedule: TemperatureSchedule::Geometric {
                initial_temperature: 1.0,
                dwell: 0,
            },
            ..Parameters::default()
        },
        Parameters {
            temperature_schedule: TemperatureSchedule::Cauchy {
                initial_temperature: 1.0,
                dwell: 1,
                delta: -1.0,
            },
            ..Parameters::default()
        },
        Parameters {
            operational_learning: OperationalLearning::Multinomial {
                learning_rate: 1.0,
                initial_learning_matrix: vec![],
            },
            ..Parameters::default()
        },
        Parameters {
            operational_learning: OperationalLearning::Multinomial {
                learning_rate: 0.1,
                initial_learning_matrix: vec![0.0, 0.0],
            },
            ..Parameters::default()
        },
        Parameters {
            operational_learning: OperationalLearning::Markov {
                learning_rate: 0.1,
                initial_learning_matrix: vec![vec![1.0]],
            },
            ..Parameters::default()
        },
        Parameters {
            operational_learning: OperationalLearning::HiddenMarkov {
                learning_rate: 0.1,
                initial_transition_matrix: vec![vec![1.0]],
                initial_emission_matrix: vec![],
            },
            ..Parameters::default()
        },
        Parameters {
            operational_learning: OperationalLearning::HiddenMarkov {
                learning_rate: 0.1,
                initial_transition_matrix: vec![vec![1.0]],
                initial_emission_matrix: vec![vec![1.0]],
            },
            ..Parameters::default()
        },
    ];
    for params in invalid {
        assert!(params.validate_for_moves(2).is_err(), "{params:?}");
    }
    assert!(Parameters::default().validate_for_moves(0).is_err());
}
#[test]
fn config_loading_is_validated_and_atomic() {
    let path = std::env::temp_dir().join(format!("cisat-parameters-{}.json", std::process::id()));
    std::fs::write(&path, r#"{"number_of_teams":2,"number_of_iterations":7}"#).unwrap();
    let mut p = Parameters::from_file(&path).unwrap();
    assert_eq!(p.number_of_teams, 2);
    assert_eq!(p.number_of_agents, 3);
    let serialized = serde_json::to_string(&p).unwrap();
    std::fs::write(&path, serialized).unwrap();
    assert_eq!(
        Parameters::from_file(&path).unwrap().number_of_iterations,
        7
    );
    for bad in [r#"{"number_of_teams":0}"#, r#"{"unknown":3}"#, "garbage"] {
        std::fs::write(&path, bad).unwrap();
        assert!(p
            .load_from_file(path.to_string_lossy().into_owned())
            .is_err());
        assert_eq!(p.number_of_teams, 2);
    }
    std::fs::remove_file(path).unwrap();
}
#[test]
fn ackley_matches_known_points_in_multiple_dimensions() {
    assert!(Ackley::<5>::from_position([0.0; 5]).objective() < 1e-12);
    let expected = 20.0 * (1.0 - (-0.2_f64).exp());
    for actual in [
        Ackley::<1>::from_position([1.0]).objective(),
        Ackley::<2>::from_position([1.0; 2]).objective(),
        Ackley::<5>::from_position([1.0; 5]).objective(),
    ] {
        assert!((actual - expected).abs() < 1e-12);
    }
    let mut point = Ackley::<5>::from_position([0.0; 5]);
    point.apply_move_operator(0, 0.0);
    assert_eq!(point.position(), &[0.0; 5]);
    for _ in 0..100 {
        point.apply_move_operator(0, 100.0);
        assert!(point.objective().is_finite());
        assert!(point.position().iter().all(|x| x.abs() <= 32.768));
    }
}
#[test]
fn all_learning_and_schedule_combinations_run_with_monotonic_best() {
    for learning in [
        OperationalLearning::None,
        OperationalLearning::Multinomial {
            learning_rate: 0.1,
            initial_learning_matrix: vec![],
        },
        OperationalLearning::Markov {
            learning_rate: 0.1,
            initial_learning_matrix: vec![],
        },
        OperationalLearning::HiddenMarkov {
            learning_rate: 0.1,
            initial_transition_matrix: vec![],
            initial_emission_matrix: vec![],
        },
    ] {
        for schedule in [
            TemperatureSchedule::None,
            TemperatureSchedule::Geometric {
                initial_temperature: 1.0,
                dwell: 2,
            },
            TemperatureSchedule::Cauchy {
                initial_temperature: 1.0,
                delta: 0.5,
                dwell: 2,
            },
            TemperatureSchedule::Triki {
                initial_temperature: 1.0,
                delta: 0.05,
                dwell: 3,
            },
        ] {
            let mut a = Agent::<Ackley<5>>::new(
                0,
                Parameters {
                    operational_learning: learning.clone(),
                    temperature_schedule: schedule,
                    ..Parameters::default()
                },
            );
            let mut best = a.get_best_solution_so_far().get_quality_scalar();
            for _ in 0..20 {
                a.iterate();
                let next = a.get_best_solution_so_far().get_quality_scalar();
                assert!(next >= best);
                assert!(a.temperature().is_finite() && a.temperature() >= 0.0);
                best = next;
            }
        }
    }
}
