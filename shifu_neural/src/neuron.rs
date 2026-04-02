//! NeuronInner — the fundamental processing unit.
//!
//! Each word IS a neuron. Receives signals, integrates, fires.
//! Summation + inhibition at the soma. Hebbian strengthening after fire.
//! Myelinated connections transmit faster (saltatory conduction).
//!
//! All indices are arena u32 — no strings in the hot path.

use rustc_hash::FxHashSet;

pub struct NeuronInner {
    pub potential: f64,
    pub threshold: f64,
    pub axon_targets: Vec<u32>,
    pub axon_weights: Vec<f64>,
    pub myelinated: FxHashSet<u32>, // indices within axon_targets
    pub refractory: bool,
    pub fire_count: u32,
    pub last_fired: i64,
}

/// Outgoing signal from a fired neuron.
pub struct Signal {
    pub target: u32,
    pub strength: f64,
}

impl NeuronInner {
    pub fn new() -> Self {
        NeuronInner {
            potential: 0.0,
            threshold: 0.3,
            axon_targets: Vec::new(),
            axon_weights: Vec::new(),
            myelinated: FxHashSet::default(),
            refractory: false,
            fire_count: 0,
            last_fired: -1,
        }
    }

    /// Dendrite receives a signal. Returns true if threshold crossed.
    #[inline]
    pub fn receive(&mut self, signal: f64, excitatory: bool) -> bool {
        if self.refractory {
            return false;
        }
        if excitatory {
            self.potential += signal;
        } else {
            self.potential -= signal * 0.5;
            if self.potential < 0.0 {
                self.potential = 0.0;
            }
        }
        self.potential >= self.threshold
    }

    /// Fire the neuron. Returns signals to send.
    pub fn fire(&mut self, epoch: i64) -> Vec<Signal> {
        if self.refractory || self.potential < self.threshold {
            return Vec::new();
        }

        self.fire_count += 1;
        self.last_fired = epoch;

        let mut signals = Vec::with_capacity(self.axon_targets.len());
        for (i, (&target, &weight)) in self
            .axon_targets
            .iter()
            .zip(self.axon_weights.iter())
            .enumerate()
        {
            let mult = if self.myelinated.contains(&(i as u32)) {
                1.5
            } else {
                1.0
            };
            signals.push(Signal {
                target,
                strength: self.potential * weight * mult,
            });
        }

        // Hebbian: strengthen all connections that just carried signal
        self.strengthen();

        // Reset
        self.potential = 0.0;
        self.refractory = true;
        signals
    }

    /// Hebbian strengthening after firing.
    fn strengthen(&mut self) {
        for i in 0..self.axon_weights.len() {
            self.axon_weights[i] += 0.1;
            // Auto-myelinate when weight crosses threshold through usage
            if !self.myelinated.contains(&(i as u32)) && self.axon_weights[i] > 0.75 {
                self.myelinated.insert(i as u32);
            }
        }
    }

    /// End absolute refractory. Enter relative refractory (higher threshold).
    #[inline]
    pub fn reset_refractory(&mut self) {
        self.refractory = false;
        self.threshold = (self.threshold + 0.1).min(0.8);
    }

    /// Gradual recovery toward resting threshold.
    #[inline]
    pub fn recover(&mut self) {
        if self.threshold > 0.3 {
            self.threshold = (self.threshold - 0.02).max(0.3);
        }
    }

    /// Grow an axon terminal. Cap at 50 connections.
    pub fn add_connection(&mut self, target: u32, weight: f64, is_myelinated: bool) {
        // Check if already connected
        for (i, &existing_target) in self.axon_targets.iter().enumerate() {
            if existing_target == target {
                if self.axon_weights[i] < weight {
                    self.axon_weights[i] = weight;
                }
                if is_myelinated {
                    self.myelinated.insert(i as u32);
                }
                return;
            }
        }
        // New connection — cap at 50
        if self.axon_targets.len() < 50 {
            let idx = self.axon_targets.len() as u32;
            self.axon_targets.push(target);
            self.axon_weights.push(weight);
            if is_myelinated {
                self.myelinated.insert(idx);
            }
        }
    }

    /// Remove a connection by index. Shifts myelinated indices.
    pub fn remove_connection(&mut self, index: usize) -> bool {
        if index >= self.axon_targets.len() {
            return false;
        }
        self.axon_targets.remove(index);
        self.axon_weights.remove(index);
        self.myelinated.remove(&(index as u32));
        // Shift myelinated indices above the removed one
        let mut new_myel = FxHashSet::default();
        for &m in &self.myelinated {
            if m > index as u32 {
                new_myel.insert(m - 1);
            } else {
                new_myel.insert(m);
            }
        }
        self.myelinated = new_myel;
        true
    }
}
