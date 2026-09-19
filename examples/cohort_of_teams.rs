use cisat::{problems::Ackley, Cohort, CommunicationStyle, Parameters};

fn main() {
    type Problem = Ackley<5>;
    let mut x = Cohort::<Problem>::new(Parameters {
        number_of_teams: 1,
        communication: CommunicationStyle::RegularInterval { interval: 5 },
        ..Default::default()
    });

    x.solve();

    println!("{:?}", x);
}
