//! This module contains the Solution trait, which can be used to implement new Solution types

use std::fmt::Debug;
use std::ops::Sub;

/// This trait is the Solution trait, which provides the necessary pieces for a problem to
/// interface with CISAT
pub trait Solution: PartialOrd + Sub<Output = f64> + Sized + Send + Ord + Clone + Debug {
    /// A problem must have a number of move operators specified
    const NUMBER_OF_MOVE_OPERATORS: usize;
    /// A problem must have a number of objectives specified
    const NUMBER_OF_OBJECTIVES: usize;
    /// A problem must have a means for generating an new solution
    fn new() -> Self;
    /// A problem must have a way to apply move operators to itself
    fn apply_move_operator(&mut self, move_index: usize, temperature: f64);
    /// Optional unmet-goal penalty in [0, 1], where zero means satisfied.
    /// Returning `None` disables satisficing for problems without a defined goal.
    fn satisficing_penalty(&self) -> Option<f64> {
        None
    }

    /// A finite quality scalar, with larger values preferred.
    /// Ordering and subtraction must agree with this scalar.
    /// A problem must have a mapping to a quality scalar
    fn get_quality_scalar(&self) -> f64;
}
