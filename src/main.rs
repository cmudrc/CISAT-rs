use cisat::{
    problems::{Ackley, Structure},
    Cohort, CommunicationStyle, OperationalLearning, Parameters, Solution, TemperatureSchedule,
};
use clap::{Parser, ValueEnum};
use std::time::Instant;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Problem {
    Ackley,
    Structure,
}
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Schedule {
    Geometric,
    Cauchy,
    Triki,
    None,
}
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Learning {
    Multinomial,
    Markov,
    #[value(alias = "hiddenmarkov")]
    HiddenMarkov,
    None,
}

/// Simulate teams using Cognitively-Inspired Simulated Annealing Teams.
#[derive(Parser, Debug)]
#[command(author, version, name = "cisat")]
struct Cli {
    /// Print final cohort state as well as the result
    #[arg(short, long)]
    verbose: bool,
    /// Run independent teams in parallel
    #[arg(short = 'R', long)]
    parallel: bool,
    /// Number of teams
    #[arg(short = 'T', long, default_value_t = 10)]
    teams: usize,
    /// Built-in search problem
    #[arg(
        short = 'P',
        long,
        value_enum,
        ignore_case = true,
        default_value = "ackley"
    )]
    problem: Problem,
    /// Agents per team
    #[arg(short = 'A', long, default_value_t = 3)]
    agents: usize,
    /// Search steps per agent
    #[arg(short = 'I', long, default_value_t = 100)]
    iter: usize,
    /// Cooling schedule; none uses greedy acceptance and unit move scale
    #[arg(
        short = 'S',
        long,
        value_enum,
        ignore_case = true,
        default_value = "geometric"
    )]
    schedule: Schedule,
    /// Initial temperature
    #[arg(short = 't', long, default_value_t = 10.0)]
    initial_temperature: f64,
    /// Cooling coefficient for Cauchy or Triki
    #[arg(short = 'd', long, default_value_t = 0.1)]
    delta: f64,
    /// Steps between cooling updates (Triki needs at least two for nonzero variance)
    #[arg(long, default_value_t = 10)]
    dwell: usize,
    /// Operational learning mode
    #[arg(
        short = 'L',
        long,
        value_enum,
        ignore_case = true,
        default_value = "markov"
    )]
    learning: Learning,
    /// Multiplicative reinforcement rate in [0, 1)
    #[arg(short = 'r', long, default_value_t = 0.05)]
    learning_rate: f64,
    /// Weight added to an agent's own solution during sharing
    #[arg(short = 'b', long, default_value_t = 1.0)]
    self_bias: f64,
    /// Uniform weight added to reduce quality bias during sharing
    #[arg(short = 'q', long, default_value_t = 1.0)]
    quality_bias: f64,
    /// Fraction of temperature controlled by the problem's unmet goal
    #[arg(short = 's', long, default_value_t = 0.5)]
    satisficing: f64,
    /// Probability of team communication on each step
    #[arg(long, conflicts_with_all = ["communication_interval", "meetings"])]
    communication_frequency: Option<f64>,
    /// Communicate every N steps
    #[arg(long, conflicts_with = "meetings")]
    communication_interval: Option<usize>,
    /// Comma-separated one-based meeting steps
    #[arg(long, value_delimiter = ',')]
    meetings: Option<Vec<usize>>,
}

fn main() {
    if let Err(message) = run(Cli::parse()) {
        eprintln!("error: {message}");
        std::process::exit(2);
    }
}
fn run(args: Cli) -> Result<(), String> {
    let temperature_schedule = match args.schedule {
        Schedule::Geometric => TemperatureSchedule::Geometric {
            initial_temperature: args.initial_temperature,
            dwell: args.dwell,
        },
        Schedule::Cauchy => TemperatureSchedule::Cauchy {
            initial_temperature: args.initial_temperature,
            delta: args.delta,
            dwell: args.dwell,
        },
        Schedule::Triki => TemperatureSchedule::Triki {
            initial_temperature: args.initial_temperature,
            delta: args.delta,
            dwell: args.dwell,
        },
        Schedule::None => TemperatureSchedule::None,
    };
    let operational_learning = match args.learning {
        Learning::Multinomial => OperationalLearning::Multinomial {
            learning_rate: args.learning_rate,
            initial_learning_matrix: vec![],
        },
        Learning::Markov => OperationalLearning::Markov {
            learning_rate: args.learning_rate,
            initial_learning_matrix: vec![],
        },
        Learning::HiddenMarkov => OperationalLearning::HiddenMarkov {
            learning_rate: args.learning_rate,
            initial_transition_matrix: vec![],
            initial_emission_matrix: vec![],
        },
        Learning::None => OperationalLearning::None,
    };
    let communication = if let Some(frequency) = args.communication_frequency {
        CommunicationStyle::ConstantFrequency { frequency }
    } else if let Some(interval) = args.communication_interval {
        CommunicationStyle::RegularInterval { interval }
    } else if let Some(times) = &args.meetings {
        CommunicationStyle::ScheduledMeetings {
            times: times.clone(),
        }
    } else {
        CommunicationStyle::None
    };
    let parameters = Parameters {
        number_of_teams: args.teams,
        number_of_agents: args.agents,
        number_of_iterations: args.iter,
        temperature_schedule,
        operational_learning,
        communication,
        self_bias: args.self_bias,
        quality_bias: args.quality_bias,
        satisficing_fraction: args.satisficing,
    };
    parameters.validate()?;
    println!("Solving {:?} with:\n{parameters}", args.problem);
    match args.problem {
        Problem::Ackley => run_all::<Ackley<5>>(parameters, &args),
        Problem::Structure => run_all::<Structure>(parameters, &args),
    }
    Ok(())
}
fn run_all<S: Solution>(parameters: Parameters, args: &Cli) {
    let started = Instant::now();
    let mut cohort = Cohort::<S>::new(parameters);
    if args.parallel {
        cohort.solve();
    } else {
        for _ in 0..args.iter {
            cohort.iterate();
        }
    }
    println!(
        "Done! {} iterations per agent in {:.3}s; best quality: {:.8}.",
        args.iter,
        started.elapsed().as_secs_f64(),
        cohort.get_best_solution_so_far()
    );
    if args.verbose {
        println!("{cohort:#?}");
    }
}
