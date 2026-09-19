//! Bounded planar truss demonstration with topology, geometry, and sizing moves.
use crate::Solution;
use rand::{seq::SliceRandom, Rng};
use std::{cmp::Ordering, ops::Sub};
use trussx::{force, point, Truss};

/// Available outer pipe radii in metres.
const RADII: [f64; 10] = [
    0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050,
];
/// Elastic modulus in pascals.
const YOUNG: f64 = 209e9;
/// Yield strength in pascals.
const YIELD: f64 = 344e6;
/// Material density in kilograms per cubic metre.
const DENSITY: f64 = 7850.0;
/// Demonstration mass goal in kilograms.
const MASS_GOAL: f64 = 250.0;
/// Demonstration minimum safety factor.
const SAFETY_GOAL: f64 = 2.0;

#[derive(Clone, Debug)]
/// Cloneable member connectivity and section choice.
struct Member {
    /// Index of the first joint.
    start: usize,
    /// Index of the second joint.
    end: usize,
    /// Index into the discrete radius catalogue.
    size: usize,
}

/// A 10 m planar bridge with two supports and two fixed 20 kN load locations.
/// Unsolvable designs receive zero quality. Valid designs balance mass with
/// yielding and pinned-end Euler buckling safety. This is a research demonstration.
#[derive(Clone, Debug)]
pub struct Structure {
    /// Planar coordinates in metres; the first five locations are fixed.
    joints: Vec<[f64; 2]>,
    /// Connectivity and section choices for the design.
    members: Vec<Member>,
    /// Combined mass and safety score to maximize.
    quality_scalar: f64,
    /// Evaluated total member mass in kilograms.
    mass: f64,
    /// Smallest yielding or buckling safety factor, or zero for failed analysis.
    safety: f64,
}

impl Solution for Structure {
    const NUMBER_OF_MOVE_OPERATORS: usize = 7;
    const NUMBER_OF_OBJECTIVES: usize = 1;
    fn new() -> Self {
        let joints = vec![
            [-5.0, 0.0],
            [-2.0, 0.0],
            [1.0, 0.0],
            [3.0, 0.0],
            [5.0, 0.0],
            [-3.5, 2.0],
            [-0.5, 2.0],
            [2.0, 2.0],
            [4.0, 2.0],
        ];
        let edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (5, 6),
            (6, 7),
            (7, 8),
            (0, 5),
            (5, 1),
            (1, 6),
            (6, 2),
            (2, 7),
            (7, 3),
            (3, 8),
            (8, 4),
        ];
        let mut solution = Self {
            joints,
            members: edges
                .into_iter()
                .map(|(start, end)| Member {
                    start,
                    end,
                    size: 4,
                })
                .collect(),
            quality_scalar: 0.0,
            mass: 0.0,
            safety: 0.0,
        };
        solution.evaluate();
        solution
    }
    fn apply_move_operator(&mut self, move_index: usize, temperature: f64) {
        assert!(
            temperature.is_finite() && temperature >= 0.0,
            "step temperature must be finite and nonnegative"
        );
        match move_index {
            0 => self.add_joint_and_attach(),
            1 => self.remove_joint(),
            2 => self.add_member(),
            3 => self.remove_member(),
            4 => self.change_size_single(),
            5 => self.change_size_all(),
            6 => self.move_joint(temperature),
            _ => panic!("Structure move index must be in 0..7"),
        }
        self.evaluate();
    }
    fn get_quality_scalar(&self) -> f64 {
        self.quality_scalar
    }
    fn satisficing_penalty(&self) -> Option<f64> {
        let mass = if self.mass > MASS_GOAL {
            1.0 - MASS_GOAL / self.mass
        } else {
            0.0
        };
        let safety = (1.0 - self.safety / SAFETY_GOAL).clamp(0.0, 1.0);
        Some(mass.max(safety))
    }
}

impl Structure {
    /// Current mass in kilograms.
    pub fn mass(&self) -> f64 {
        self.mass
    }
    /// Minimum factor of safety against yielding and Euler buckling; zero if unsolvable.
    pub fn factor_of_safety(&self) -> f64 {
        self.safety
    }
    /// Number of joints, including the fixed support and load locations.
    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }
    /// Number of members.
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Place a free joint in the design domain and connect its three nearest neighbors.
    fn add_joint_and_attach(&mut self) {
        if self.joints.len() >= 30 {
            return;
        }
        let mut rng = rand::thread_rng();
        let position = [rng.gen_range(-5.0..5.0), rng.gen_range(0.25..4.0)];
        if self.joints.iter().any(|p| distance(*p, position) < 0.1) {
            return;
        }
        let mut neighbors: Vec<_> = (0..self.joints.len()).collect();
        neighbors.sort_by(|a, b| {
            distance(self.joints[*a], position).total_cmp(&distance(self.joints[*b], position))
        });
        let end = self.joints.len();
        for start in neighbors.into_iter().take(3) {
            self.members.push(Member {
                start,
                end,
                size: 4,
            });
        }
        self.joints.push(position);
    }
    /// Remove a free joint and its incident members, then repair remaining indices.
    fn remove_joint(&mut self) {
        if self.joints.len() <= 5 {
            return;
        }
        let index = rand::thread_rng().gen_range(5..self.joints.len());
        self.joints.remove(index);
        self.members.retain(|m| m.start != index && m.end != index);
        for m in &mut self.members {
            if m.start > index {
                m.start -= 1;
            }
            if m.end > index {
                m.end -= 1;
            }
        }
    }
    /// Connect a uniformly chosen pair of currently unconnected joints.
    fn add_member(&mut self) {
        let mut choices = vec![];
        for start in 0..self.joints.len() {
            for end in start + 1..self.joints.len() {
                if !self.members.iter().any(|m| {
                    (m.start == start && m.end == end) || (m.start == end && m.end == start)
                }) {
                    choices.push((start, end));
                }
            }
        }
        if let Some(&(start, end)) = choices.choose(&mut rand::thread_rng()) {
            self.members.push(Member {
                start,
                end,
                size: 4,
            });
        }
    }
    /// Remove a uniformly selected existing member.
    fn remove_member(&mut self) {
        if !self.members.is_empty() {
            self.members
                .remove(rand::thread_rng().gen_range(0..self.members.len()));
        }
    }
    /// Increase or decrease one member by one catalogue size.
    fn change_size_single(&mut self) {
        if let Some(m) = self.members.choose_mut(&mut rand::thread_rng()) {
            m.size = next_size(m.size, rand::random());
        }
    }
    /// Increase or decrease all members by one catalogue size.
    fn change_size_all(&mut self) {
        let increase = rand::random();
        for m in &mut self.members {
            m.size = next_size(m.size, increase);
        }
    }
    /// Perturb one free joint within the domain, excluding near-coincident positions.
    fn move_joint(&mut self, temperature: f64) {
        if self.joints.len() <= 5 {
            return;
        }
        let mut rng = rand::thread_rng();
        let index = rng.gen_range(5..self.joints.len());
        let step = temperature.min(1.0);
        let old = self.joints[index];
        let position = [
            (old[0] + rng.gen_range(-1.0..1.0) * step).clamp(-5.0, 5.0),
            (old[1] + rng.gen_range(-1.0..1.0) * step).clamp(0.25, 4.0),
        ];
        if self
            .joints
            .iter()
            .enumerate()
            .all(|(i, p)| i == index || distance(*p, position) >= 0.1)
        {
            self.joints[index] = position;
        }
    }
    /// Rebuild the analysis model and calculate mass, axial safety, and quality.
    fn evaluate(&mut self) {
        self.mass = 0.0;
        self.safety = 0.0;
        self.quality_scalar = 0.0;
        let mut truss = Truss::new();
        let nodes: Vec<_> = self
            .joints
            .iter()
            .map(|p| truss.add_joint(point(p[0], p[1], 0.0)))
            .collect();
        for (i, node) in nodes.iter().enumerate() {
            truss
                .set_support(*node, [i == 0, i == 0 || i == 4, true])
                .expect("existing node");
        }
        for i in [1, 3] {
            truss
                .set_load(nodes[i], force(0.0, -20_000.0, 0.0))
                .expect("existing load node");
        }
        let mut edges = vec![];
        for m in &self.members {
            let outer = RADII[m.size];
            let inner = outer * (1.0 - 1.0 / 7.5);
            let area = std::f64::consts::PI * (outer.powi(2) - inner.powi(2));
            let inertia = std::f64::consts::FRAC_PI_4 * (outer.powi(4) - inner.powi(4));
            let length = distance(self.joints[m.start], self.joints[m.end]);
            self.mass += DENSITY * area * length;
            let edge = truss.add_member(nodes[m.start], nodes[m.end]);
            truss
                .set_member_properties(edge, area, YOUNG)
                .expect("positive member properties");
            truss
                .set_member_yield_strength(edge, YIELD)
                .expect("existing member");
            edges.push((edge, area, inertia, length));
        }
        if truss.evaluate().is_err() || self.mass <= 0.0 {
            return;
        }
        let mut safety = f64::INFINITY;
        for (edge, area, inertia, length) in edges {
            let axial = truss.member_axial_force(edge).expect("evaluated member");
            if !axial.is_finite() {
                return;
            }
            if axial != 0.0 {
                safety = safety.min(YIELD * area / axial.abs());
            }
            if axial < 0.0 {
                safety = safety.min(
                    std::f64::consts::PI.powi(2) * YOUNG * inertia / (length.powi(2) * -axial),
                );
            }
        }
        if !safety.is_finite() {
            return;
        }
        self.safety = safety.max(0.0);
        self.quality_scalar = (self.safety / SAFETY_GOAL).min(1.0) / (1.0 + self.mass / MASS_GOAL);
    }
}
/// Euclidean distance between two planar points.
fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}
/// Move one size up or down without leaving the catalogue.
fn next_size(size: usize, increase: bool) -> usize {
    if increase {
        (size + 1).min(RADII.len() - 1)
    } else {
        size.saturating_sub(1)
    }
}
impl PartialEq for Structure {
    fn eq(&self, rhs: &Self) -> bool {
        self.cmp(rhs) == Ordering::Equal
    }
}
impl Eq for Structure {}
impl PartialOrd for Structure {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        Some(self.cmp(rhs))
    }
}
impl Ord for Structure {
    fn cmp(&self, rhs: &Self) -> Ordering {
        self.quality_scalar.total_cmp(&rhs.quality_scalar)
    }
}
impl Sub for Structure {
    type Output = f64;
    fn sub(self, rhs: Self) -> f64 {
        self.quality_scalar - rhs.quality_scalar
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn invariants(s: &Structure) {
        assert_eq!(
            &s.joints[..5],
            &[[-5.0, 0.0], [-2.0, 0.0], [1.0, 0.0], [3.0, 0.0], [5.0, 0.0]]
        );
        assert!(s.joints.len() <= 30);
        for (i, m) in s.members.iter().enumerate() {
            assert!(m.start < s.joints.len() && m.end < s.joints.len() && m.start != m.end);
            assert!(m.size < RADII.len());
            assert!(distance(s.joints[m.start], s.joints[m.end]) >= 0.1);
            assert!(!s.members[..i]
                .iter()
                .any(|other| (m.start == other.start && m.end == other.end)
                    || (m.start == other.end && m.end == other.start)));
        }
        assert!(s.mass.is_finite() && s.mass >= 0.0);
        assert!(s.safety.is_finite() && s.safety >= 0.0);
        assert!(s.quality_scalar.is_finite() && (0.0..=1.0).contains(&s.quality_scalar));
    }
    #[test]
    fn seed_has_nonzero_safety_and_correct_mass() {
        let s = Structure::new();
        let area =
            std::f64::consts::PI * (0.025_f64.powi(2) - (0.025_f64 * (1.0 - 1.0 / 7.5)).powi(2));
        // Bottom chord = 10 m; upper chord = 7.5 m; four 2.5 m and four sqrt(5) m diagonals.
        let expected_mass = area * DENSITY * (17.5 + 10.0 + 4.0 * 5.0_f64.sqrt());
        assert!((s.mass - expected_mass).abs() < 1e-9);
        assert!(s.safety > 0.0 && s.quality_scalar > 0.0);
        invariants(&s);
    }
    #[test]
    fn topology_and_sizing_moves_change_the_design() {
        let s = Structure::new();
        let mut candidate = s.clone();
        candidate.apply_move_operator(1, 1.0);
        assert_eq!(candidate.joints.len(), s.joints.len() - 1);
        invariants(&candidate);
        let mut candidate = s.clone();
        candidate.apply_move_operator(2, 1.0);
        assert_eq!(candidate.members.len(), s.members.len() + 1);
        invariants(&candidate);
        let mut candidate = s.clone();
        candidate.apply_move_operator(3, 1.0);
        assert_eq!(candidate.members.len(), s.members.len() - 1);
        invariants(&candidate);
        let mut candidate = s.clone();
        candidate.apply_move_operator(4, 1.0);
        assert_eq!(
            candidate
                .members
                .iter()
                .zip(&s.members)
                .filter(|(a, b)| a.size != b.size)
                .count(),
            1
        );
        assert_ne!(candidate.mass, s.mass);
        invariants(&candidate);
        let mut candidate = s.clone();
        candidate.apply_move_operator(5, 1.0);
        assert!(candidate.members.iter().all(|m| m.size != 4));
        invariants(&candidate);
    }
    #[test]
    fn all_moves_preserve_geometry_indices_and_finite_scores() {
        let mut s = Structure::new();
        for _ in 0..30 {
            for operation in 0..7 {
                s.apply_move_operator(operation, 0.1);
                invariants(&s);
            }
        }
    }
    #[test]
    fn unstable_design_clears_previous_results() {
        let mut s = Structure::new();
        s.members.clear();
        s.evaluate();
        assert_eq!(s.quality_scalar, 0.0);
        assert_eq!(s.safety, 0.0);
        assert_eq!(s.mass, 0.0);
        assert_eq!(s.satisficing_penalty(), Some(1.0));
    }
}
