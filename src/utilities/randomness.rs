//! Sampling and normalization shared by the model.
use rand::{thread_rng, Rng};
use rand_distr::{Distribution, WeightedIndex};

/// Normalize nonnegative finite weights; all-zero weights become uniform.
pub(crate) fn normalize(weights: &mut [f64]) {
    let max = weights.iter().copied().fold(0.0_f64, f64::max);
    if max == 0.0 {
        let uniform = 1.0 / weights.len() as f64;
        weights.fill(uniform);
    } else {
        // Scale first so summing large finite weights cannot overflow.
        weights.iter_mut().for_each(|w| *w /= max);
        let sum: f64 = weights.iter().sum();
        weights.iter_mut().for_each(|w| *w /= sum);
    }
}

/// Sample an index from nonempty, finite, nonnegative weights.
pub(crate) fn multinomial_draw(weights: &[f64]) -> usize {
    let mut weights = weights.to_vec();
    normalize(&mut weights);
    WeightedIndex::new(weights)
        .expect("valid sampling weights")
        .sample(&mut thread_rng())
}

/// Sample independent coordinates from the half-open interval [low, high).
pub(crate) fn random_uniform_vector(length: usize, low: f64, high: f64) -> Vec<f64> {
    let mut rng = thread_rng();
    (0..length).map(|_| rng.gen_range(low..high)).collect()
}

/// Sample a probability threshold in [0, 1).
pub(crate) fn random_unit_draw() -> f64 {
    thread_rng().gen_range(0.0..1.0)
}
