use std::process::Command;
fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_cisat"))
        .args(args)
        .output()
        .unwrap()
}
#[test]
fn cli_runs_both_problems_serial_and_parallel() {
    for problem in ["ackley", "structure"] {
        for parallel in [false, true] {
            let mut args = vec![
                "--problem",
                problem,
                "--teams",
                "2",
                "--agents",
                "2",
                "--iter",
                "3",
                "--learning",
                "HiddenMarkov",
                "--schedule",
                "Triki",
                "--dwell",
                "2",
                "--communication-interval",
                "1",
            ];
            if parallel {
                args.push("--parallel");
            }
            let output = run(&args);
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("3 iterations per agent"));
        }
    }
}
#[test]
fn cli_reports_invalid_input_without_panicking() {
    for args in [
        vec!["--problem", "missing"],
        vec!["--agents", "0"],
        vec!["--schedule", "wrong"],
        vec!["--communication-interval", "0"],
        vec!["--learning-rate", "1"],
        vec!["--communication-frequency", "0.5", "--meetings", "1,2"],
    ] {
        let output = run(&args);
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains("error:"));
        assert!(!error.contains("panicked"));
    }
}
