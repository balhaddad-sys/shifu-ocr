//! Arena — word-indexed neuron storage.
//!
//! Words are interned into a HashMap<String, u32>.
//! Neurons live in a Vec<NeuronInner> indexed by that u32.
//! During wave propagation: zero string hashing, just array indexing.

use rustc_hash::FxHashMap;
use crate::neuron::NeuronInner;

pub struct Arena {
    pub word_to_id: FxHashMap<String, u32>,
    pub id_to_word: Vec<String>,
    pub neurons: Vec<NeuronInner>,
}

impl Arena {
    pub fn new() -> Self {
        Arena {
            word_to_id: FxHashMap::default(),
            id_to_word: Vec::new(),
            neurons: Vec::new(),
        }
    }

    /// Get or create a neuron for a word. Returns the arena index.
    pub fn ensure(&mut self, word: &str) -> u32 {
        if let Some(&id) = self.word_to_id.get(word) {
            return id;
        }
        let id = self.neurons.len() as u32;
        self.word_to_id.insert(word.to_string(), id);
        self.id_to_word.push(word.to_string());
        self.neurons.push(NeuronInner::new());
        id
    }

    /// Lookup by word. Returns None if not found.
    #[inline]
    pub fn get_id(&self, word: &str) -> Option<u32> {
        self.word_to_id.get(word).copied()
    }

    /// Get the word for an arena index.
    #[inline]
    pub fn get_word(&self, id: u32) -> &str {
        &self.id_to_word[id as usize]
    }

    /// Get neuron by index.
    #[inline]
    pub fn get(&self, id: u32) -> &NeuronInner {
        &self.neurons[id as usize]
    }

    /// Get mutable neuron by index.
    #[inline]
    pub fn get_mut(&mut self, id: u32) -> &mut NeuronInner {
        &mut self.neurons[id as usize]
    }

    /// Number of neurons.
    #[inline]
    pub fn len(&self) -> usize {
        self.neurons.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.neurons.is_empty()
    }

    /// Remove a neuron by word. Swap-removes from the vec.
    pub fn remove(&mut self, word: &str) -> bool {
        let id = match self.word_to_id.remove(word) {
            Some(id) => id,
            None => return false,
        };
        let last = self.neurons.len() as u32 - 1;
        if id != last {
            // Swap with last
            self.neurons.swap(id as usize, last as usize);
            let moved_word = self.id_to_word[last as usize].clone();
            self.id_to_word.swap(id as usize, last as usize);
            // Update the moved word's id
            self.word_to_id.insert(moved_word, id);
            // Update all axon_targets that pointed to `last` → now point to `id`
            let len = self.neurons.len() - 1;
            for neuron in &mut self.neurons[..len] {
                for target in &mut neuron.axon_targets {
                    if *target == last {
                        *target = id;
                    }
                }
            }
        }
        self.neurons.pop();
        self.id_to_word.pop();
        true
    }

    /// Total connections across all neurons.
    pub fn total_connections(&self) -> usize {
        self.neurons.iter().map(|n| n.axon_targets.len()).sum()
    }

    /// Total myelinated connections.
    pub fn total_myelinated(&self) -> usize {
        self.neurons.iter().map(|n| n.myelinated.len()).sum()
    }
}
