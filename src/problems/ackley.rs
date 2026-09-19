//! The standard n-dimensional Ackley minimization benchmark.
use crate::{utilities::randomness::random_uniform_vector, Solution};
use std::{cmp::Ordering, ops::Sub};

/// Ackley search on [-32.768, 32.768]^N, represented as a maximization quality.
#[derive(Clone, Debug)]
pub struct Ackley<const NUMBER_OF_DIMENSIONS: usize = 5> {
    /// Raw minimization objective.
    objective_function_value: f64,
    /// Transformed objective to maximize.
    quality_scalar: f64,
    /// Current coordinates in the bounded domain.
    x: Vec<f64>,
}

impl<const N: usize> Ackley<N> {
    /// Construct and evaluate a point inside the benchmark domain.
    /// Panics for zero dimensions, nonfinite coordinates, or coordinates outside the domain.
    pub fn from_position(position: [f64; N]) -> Self {
        assert!(N > 0, "Ackley requires at least one dimension");
        assert!(
            position
                .iter()
                .all(|x| x.is_finite() && (-32.768..=32.768).contains(x)),
            "Ackley coordinates must be finite and in [-32.768, 32.768]"
        );
        let mut solution = Self {
            x: position.to_vec(),
            objective_function_value: 0.0,
            quality_scalar: 0.0,
        };
        solution.evaluate();
        solution
    }
    /// Raw Ackley objective to minimize (zero at the origin).
    pub fn objective(&self) -> f64 {
        self.objective_function_value
    }
    /// Current search coordinates.
    pub fn position(&self) -> &[f64] {
        &self.x
    }
    /// Evaluate the standard dimension-normalized Ackley expression.
    fn evaluate(&mut self) {
        let squares = self.x.iter().map(|x| x * x).sum::<f64>() / N as f64;
        let cosines = self
            .x
            .iter()
            .map(|x| (std::f64::consts::TAU * x).cos())
            .sum::<f64>()
            / N as f64;
        self.objective_function_value =
            (-20.0 * (-0.2 * squares.sqrt()).exp() - cosines.exp() + std::f64::consts::E + 20.0)
                .max(0.0);
        self.quality_scalar = 20.0 + std::f64::consts::E - self.objective_function_value;
    }
}

impl<const N: usize> Solution for Ackley<N> {
    const NUMBER_OF_MOVE_OPERATORS: usize = 1;
    const NUMBER_OF_OBJECTIVES: usize = 1;
    fn new() -> Self {
        assert!(N > 0, "Ackley requires at least one dimension");
        let mut solution = Self {
            x: random_uniform_vector(N, -10.0, 10.0),
            objective_function_value: 0.0,
            quality_scalar: 0.0,
        };
        solution.evaluate();
        solution
    }
    fn apply_move_operator(&mut self, move_index: usize, temperature: f64) {
        assert_eq!(move_index, 0, "Ackley has one move operator");
        assert!(
            temperature.is_finite() && temperature >= 0.0,
            "step temperature must be finite and nonnegative"
        );
        let angles =
            random_uniform_vector(N, -std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
        for (x, angle) in self.x.iter_mut().zip(angles) {
            *x = (*x + angle.tan() * temperature).clamp(-32.768, 32.768);
        }
        self.evaluate();
    }
    fn get_quality_scalar(&self) -> f64 {
        self.quality_scalar
    }
    fn satisficing_penalty(&self) -> Option<f64> {
        Some((self.objective_function_value / (20.0 + std::f64::consts::E)).clamp(0.0, 1.0))
    }
}
impl<const N: usize> PartialEq for Ackley<N> {
    fn eq(&self, rhs: &Self) -> bool {
        self.cmp(rhs) == Ordering::Equal
    }
}
impl<const N: usize> Eq for Ackley<N> {}
impl<const N: usize> PartialOrd for Ackley<N> {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        Some(self.cmp(rhs))
    }
}
impl<const N: usize> Ord for Ackley<N> {
    fn cmp(&self, rhs: &Self) -> Ordering {
        self.quality_scalar.total_cmp(&rhs.quality_scalar)
    }
}
impl<const N: usize> Sub for Ackley<N> {
    type Output = f64;
    fn sub(self, rhs: Self) -> f64 {
        self.quality_scalar - rhs.quality_scalar
    }
}
