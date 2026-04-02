//! NeuralField — wave propagation engine with PyO3 bindings.

use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::seq::SliceRandom;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::arena::Arena;

pub struct NeuralFieldInner {
    pub arena: Arena,
    pub max_steps: usize,
    pub epoch: i64,
    pub last_activated: Vec<u32>,
}

impl NeuralFieldInner {
    pub fn new(max_steps: usize) -> Self {
        NeuralFieldInner {
            arena: Arena::new(),
            max_steps,
            epoch: 0,
            last_activated: Vec::new(),
        }
    }

    pub fn activate(&mut self, seed_id: u32, energy: f64) -> FxHashMap<u32, f64> {
        self.epoch += 1;

        for &id in &self.last_activated {
            if (id as usize) < self.arena.neurons.len() {
                let n = &mut self.arena.neurons[id as usize];
                n.refractory = false;
                n.potential = 0.0;
            }
        }
        self.last_activated.clear();

        self.arena.neurons[seed_id as usize].potential = energy;
        self.last_activated.push(seed_id);

        let mut pending = vec![seed_id];
        let mut activated: FxHashMap<u32, f64> = FxHashMap::default();
        activated.insert(seed_id, energy);

        let max_fired = self.arena.len().min(200);
        let mut total_fired = 0usize;
        let mut steps = 0;
        let epoch = self.epoch;

        while !pending.is_empty() && steps < self.max_steps && total_fired < max_fired {
            steps += 1;
            let mut next_pending = Vec::new();

            for &neuron_id in &pending {
                let neuron = &self.arena.neurons[neuron_id as usize];
                if neuron.potential < neuron.threshold {
                    continue;
                }

                let signals = self.arena.neurons[neuron_id as usize].fire(epoch);
                total_fired += 1;

                for sig in &signals {
                    if (sig.target as usize) >= self.arena.neurons.len() {
                        continue;
                    }

                    let is_myel = {
                        let src = &self.arena.neurons[neuron_id as usize];
                        src.axon_targets
                            .iter()
                            .position(|&t| t == sig.target)
                            .map(|i| src.myelinated.contains(&(i as u32)))
                            .unwrap_or(false)
                    };

                    let decay = if is_myel { 0.85 } else { 0.65 };
                    let decayed = sig.strength * decay;
                    if decayed < 0.01 {
                        continue;
                    }

                    let target = &mut self.arena.neurons[sig.target as usize];
                    let fired = target.receive(decayed, true);
                    *activated.entry(sig.target).or_insert(0.0) += decayed;
                    self.last_activated.push(sig.target);

                    if fired && total_fired < max_fired {
                        next_pending.push(sig.target);
                    }
                }
            }

            for &nw in &pending {
                if (nw as usize) < self.arena.neurons.len() {
                    self.arena.neurons[nw as usize].reset_refractory();
                }
            }

            pending = next_pending;
        }

        activated
    }

    pub fn heartbeat(&mut self) -> (i32, i32, Option<String>) {
        if self.arena.is_empty() {
            return (0, 0, None);
        }
        let mut rng = rand::thread_rng();
        let ids: Vec<u32> = (0..self.arena.len() as u32).collect();
        let &seed_id = ids.choose(&mut rng).unwrap();
        let seed_word = self.arena.get_word(seed_id).to_string();

        let myel_before = self.arena.total_myelinated();
        let field = self.activate(seed_id, 0.5);
        let myel_after = self.arena.total_myelinated();

        ((myel_after as i32 - myel_before as i32), field.len() as i32, Some(seed_word))
    }

    pub fn practice(&mut self, word: &str) -> (i32, i32) {
        let id = match self.arena.get_id(word) {
            Some(id) => id,
            None => return (0, 0),
        };
        let myel_before = self.arena.total_myelinated();
        let field = self.activate(id, 2.0);
        let myel_after = self.arena.total_myelinated();
        ((myel_after as i32 - myel_before as i32), field.len() as i32)
    }

    pub fn prune(&mut self, decay: f64, min_weight: f64) -> i32 {
        let mut pruned = 0i32;
        for neuron in &mut self.arena.neurons {
            let mut dead = Vec::new();
            for i in 0..neuron.axon_weights.len() {
                if neuron.myelinated.contains(&(i as u32)) {
                    neuron.axon_weights[i] *= 0.999;
                } else {
                    neuron.axon_weights[i] *= decay;
                }
                if neuron.axon_weights[i] < min_weight {
                    dead.push(i);
                }
            }
            for &i in dead.iter().rev() {
                neuron.remove_connection(i);
                pruned += 1;
            }
        }
        pruned
    }

    pub fn score_sequence(&mut self, token_ids: &[u32]) -> (f64, Vec<f64>) {
        if token_ids.len() < 2 {
            return (0.0, vec![]);
        }
        let mut scores = vec![1.0];
        let mut running: FxHashMap<u32, f64> = FxHashMap::default();

        for (i, &token_id) in token_ids.iter().enumerate() {
            if i > 0 {
                let score = running.get(&token_id).copied().unwrap_or(0.0).min(1.0);
                scores.push(score);
            }
            let field = self.activate(token_id, 0.5);
            for (w, e) in field {
                *running.entry(w).or_insert(0.0) += e;
            }
        }

        let coherence = scores.iter().sum::<f64>() / scores.len() as f64;
        (coherence, scores)
    }
}

// ═══════════════════════════════════════════════════════════
//  PyO3 Bindings
// ═══════════════════════════════════════════════════════════

#[pyclass]
pub struct NeuralField {
    pub inner: NeuralFieldInner,
}

#[pymethods]
impl NeuralField {
    #[new]
    #[pyo3(signature = (max_propagation_steps=5))]
    fn new(max_propagation_steps: usize) -> Self {
        NeuralField {
            inner: NeuralFieldInner::new(max_propagation_steps),
        }
    }

    fn ensure_neuron(&mut self, word: &str) -> NeuronProxy {
        let id = self.inner.arena.ensure(word);
        NeuronProxy { id, word: word.to_string() }
    }

    /// Add a connection between two neurons by word.
    /// This is the primary way Python code wires the field.
    #[pyo3(signature = (source, target, weight, myelinated=false))]
    fn connect(&mut self, source: &str, target: &str, weight: f64, myelinated: bool) {
        let src_id = self.inner.arena.ensure(source);
        let tgt_id = self.inner.arena.ensure(target);
        self.inner.arena.neurons[src_id as usize].add_connection(tgt_id, weight, myelinated);
    }

    #[pyo3(signature = (word, energy=1.0))]
    fn activate<'py>(&mut self, py: Python<'py>, word: &str, energy: f64) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let id = match self.inner.arena.get_id(word) {
            Some(id) => id,
            None => {
                dict.set_item(word, energy)?;
                return Ok(dict);
            }
        };

        let field = self.inner.activate(id, energy);
        for (id, e) in field {
            let w = self.inner.arena.get_word(id);
            dict.set_item(w, e)?;
        }
        Ok(dict)
    }

    #[pyo3(signature = (words, energy=1.0))]
    fn activate_multi<'py>(&mut self, py: Python<'py>, words: Vec<String>, energy: f64) -> PyResult<Bound<'py, PyDict>> {
        let mut combined: FxHashMap<u32, f64> = FxHashMap::default();
        for word in &words {
            if let Some(id) = self.inner.arena.get_id(word) {
                let field = self.inner.activate(id, energy);
                for (k, v) in field {
                    *combined.entry(k).or_insert(0.0) += v;
                }
            }
        }
        let dict = PyDict::new(py);
        for (id, e) in combined {
            dict.set_item(self.inner.arena.get_word(id), e)?;
        }
        Ok(dict)
    }

    fn score_sequence<'py>(&mut self, py: Python<'py>, tokens: Vec<String>) -> PyResult<Bound<'py, PyDict>> {
        let ids: Vec<u32> = tokens.iter().filter_map(|t| self.inner.arena.get_id(t)).collect();
        let (coherence, scores) = self.inner.score_sequence(&ids);
        let dict = PyDict::new(py);
        dict.set_item("coherence", coherence)?;
        dict.set_item("scores", scores)?;
        Ok(dict)
    }

    fn heartbeat<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (myel_new, fired, seed) = self.inner.heartbeat();
        let dict = PyDict::new(py);
        dict.set_item("myelinated_new", myel_new)?;
        dict.set_item("fired", fired)?;
        dict.set_item("shortcuts", 0)?;
        if let Some(s) = seed {
            dict.set_item("seed", s)?;
        }
        Ok(dict)
    }

    fn practice<'py>(&mut self, py: Python<'py>, focus_word: &str) -> PyResult<Bound<'py, PyDict>> {
        let (improved, fired) = self.inner.practice(focus_word);
        let dict = PyDict::new(py);
        dict.set_item("improved", improved)?;
        dict.set_item("fired", fired)?;
        Ok(dict)
    }

    #[pyo3(signature = (decay=0.99, min_weight=0.05))]
    fn prune(&mut self, decay: f64, min_weight: f64) -> i32 {
        self.inner.prune(decay, min_weight)
    }

    fn total_connections(&self) -> usize {
        self.inner.arena.total_connections()
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("neurons", self.inner.arena.len())?;
        dict.set_item("connections", self.inner.arena.total_connections())?;
        dict.set_item("myelinated", self.inner.arena.total_myelinated())?;
        dict.set_item("epoch", self.inner.epoch)?;
        Ok(dict)
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let neurons_dict = PyDict::new(py);
        let limit = self.inner.arena.len().min(5000);
        for i in 0..limit {
            let n = &self.inner.arena.neurons[i];
            let word = &self.inner.arena.id_to_word[i];
            let entry = PyDict::new(py);
            let targets: Vec<&str> = n.axon_targets.iter().take(50)
                .map(|&id| self.inner.arena.get_word(id)).collect();
            entry.set_item("targets", targets)?;
            let weights: Vec<f64> = n.axon_weights.iter().take(50)
                .map(|w| (w * 1000.0).round() / 1000.0).collect();
            entry.set_item("weights", weights)?;
            let myel: Vec<u32> = n.myelinated.iter().copied().collect();
            entry.set_item("myel", myel)?;
            entry.set_item("fires", n.fire_count)?;
            neurons_dict.set_item(word.as_str(), entry)?;
        }
        let dict = PyDict::new(py);
        dict.set_item("neurons", neurons_dict)?;
        dict.set_item("epoch", self.inner.epoch)?;
        Ok(dict)
    }

    #[classmethod]
    fn from_dict(_cls: &Bound<'_, pyo3::types::PyType>, d: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut field = NeuralFieldInner::new(5);
        if let Some(epoch) = d.get_item("epoch")? {
            field.epoch = epoch.extract()?;
        }
        if let Some(neurons_obj) = d.get_item("neurons")? {
            let neurons_dict = neurons_obj.cast::<PyDict>()?;
            // First pass: create all neurons
            for (key, _) in neurons_dict.iter() {
                let word: String = key.extract()?;
                field.arena.ensure(&word);
            }
            // Second pass: wire connections
            for (key, value) in neurons_dict.iter() {
                let word: String = key.extract()?;
                let id = field.arena.get_id(&word).unwrap();
                let entry = value.cast::<PyDict>()?;

                if let Some(targets_obj) = entry.get_item("targets")? {
                    let targets: Vec<String> = targets_obj.extract()?;
                    if let Some(weights_obj) = entry.get_item("weights")? {
                        let weights: Vec<f64> = weights_obj.extract()?;
                        let myel: Vec<u32> = entry.get_item("myel")?
                            .map(|m| m.extract().unwrap_or_default())
                            .unwrap_or_default();
                        let myel_set: FxHashSet<u32> = myel.into_iter().collect();

                        // Resolve target IDs first (may add new neurons)
                        let target_ids: Vec<u32> = targets.iter()
                            .map(|tw| field.arena.ensure(tw))
                            .collect();

                        let neuron = &mut field.arena.neurons[id as usize];
                        for (i, (tid, w)) in target_ids.iter().zip(weights.iter()).enumerate() {
                            neuron.axon_targets.push(*tid);
                            neuron.axon_weights.push(*w);
                            if myel_set.contains(&(i as u32)) {
                                neuron.myelinated.insert(i as u32);
                            }
                        }
                    }
                }
                if let Some(fires_obj) = entry.get_item("fires")? {
                    field.arena.neurons[id as usize].fire_count = fires_obj.extract()?;
                }
            }
        }
        Ok(NeuralField { inner: field })
    }

    #[pyo3(signature = (co_graph, cortex_connections=None, myelinated_pairs=None, golgi=None))]
    fn build_from_graphs(
        &mut self,
        py: Python,
        co_graph: HashMap<String, HashMap<String, f64>>,
        cortex_connections: Option<HashMap<String, HashMap<String, f64>>>,
        myelinated_pairs: Option<FxHashSet<(String, String)>>,
        golgi: Option<Py<PyAny>>,
    ) -> PyResult<i64> {
        let mut built: i64 = 0;
        let myel_set = myelinated_pairs.unwrap_or_default();

        for (word, neighbors) in &co_graph {
            if word.len() <= 2 { continue; }
            let src_id = self.inner.arena.ensure(word);

            let mut sorted_n: Vec<(&String, &f64)> = neighbors.iter().collect();
            sorted_n.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            let max_w = sorted_n.first().map(|(_, w)| **w).unwrap_or(1.0);

            for (target, weight) in sorted_n.iter().take(30) {
                if target.len() <= 2 { continue; }
                let target_id = self.inner.arena.ensure(target);
                let mut norm_w = *weight / max_w.max(1.0);

                if let Some(ref golgi_obj) = golgi {
                    if let Ok(stdp) = golgi_obj.call_method1(py, "stdp_affinity", (word.as_str(), target.as_str())) {
                        let v: f64 = stdp.extract(py).unwrap_or(0.5);
                        norm_w *= 0.5 + v * 0.5;
                    }
                }

                let is_myel = myel_set.contains(&(word.clone(), target.to_string()));
                self.inner.arena.neurons[src_id as usize].add_connection(target_id, norm_w, is_myel);
                built += 1;
            }
        }

        if let Some(cortex) = cortex_connections {
            for (source, targets) in &cortex {
                if let Some(src_id) = self.inner.arena.get_id(source) {
                    for (target, weight) in targets {
                        if let Some(tgt_id) = self.inner.arena.get_id(target) {
                            let w = (*weight / 10.0).min(1.0);
                            self.inner.arena.neurons[src_id as usize].add_connection(tgt_id, w, false);
                        }
                    }
                }
            }
        }

        Ok(built)
    }

    /// Access neurons dict-like. For Python compatibility: `nf.neurons`
    /// Returns a proxy that supports __bool__, __len__, __contains__, keys(), etc.
    #[getter]
    fn neurons(&self) -> NeuronsProxyOwned {
        // Snapshot the word list for iteration
        NeuronsProxyOwned {
            words: self.inner.arena.id_to_word.clone(),
        }
    }
}

// ═══ NeuronProxy ═══

#[pyclass]
#[derive(Clone)]
pub struct NeuronProxy {
    pub id: u32,
    pub word: String,
}

#[pymethods]
impl NeuronProxy {
    #[getter]
    fn word(&self) -> &str { &self.word }

    #[getter]
    fn fire_count(&self) -> u32 { 0 }

    fn add_connection(&self, _target: &str, _weight: f64, _myelinated: Option<bool>) {
        // Mutations go through NeuralField.ensure_neuron() + field methods
    }
}

// ═══ NeuronsProxy — dict-like snapshot ═══

#[pyclass]
pub struct NeuronsProxyOwned {
    words: Vec<String>,
}

#[pymethods]
impl NeuronsProxyOwned {
    fn __bool__(&self) -> bool { !self.words.is_empty() }
    fn __len__(&self) -> usize { self.words.len() }
    fn __contains__(&self, word: &str) -> bool { self.words.iter().any(|w| w == word) }

    fn keys(&self) -> Vec<String> { self.words.clone() }
    fn values(&self) -> Vec<NeuronProxy> {
        self.words.iter().enumerate().map(|(i, w)| NeuronProxy { id: i as u32, word: w.clone() }).collect()
    }
    fn items(&self) -> Vec<(String, NeuronProxy)> {
        self.words.iter().enumerate().map(|(i, w)| (w.clone(), NeuronProxy { id: i as u32, word: w.clone() })).collect()
    }
    fn get(&self, word: &str) -> Option<NeuronProxy> {
        self.words.iter().position(|w| w == word).map(|i| NeuronProxy { id: i as u32, word: word.to_string() })
    }
}
