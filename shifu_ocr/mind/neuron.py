"""
Neuron — the fundamental processing unit.

Each word in the cortex IS a neuron.
It doesn't store data. It IS the data.
It receives signals, integrates them, fires when threshold is reached,
and sends signals to its neighbors. That's it.

No central controller. No thalamus routing activation.
The wave emerges from individual neurons talking to each other.

    dendrites: incoming connections (who sends signals TO me)
    soma: integration (sum inputs, apply threshold)
    axon: outgoing connections (who I send signals TO)
    refractory: can't fire again immediately (prevents loops)

A neuron is ~200 bytes. 60,000 neurons = 12MB. Not 500MB.
"""

from __future__ import annotations
from typing import Dict, List, Set, Optional, Callable


class Neuron:
    """One word. One processing unit. Fires independently."""

    __slots__ = (
        'word', 'potential', 'threshold', 'axon_targets',
        'axon_weights', 'refractory', 'fire_count',
        'myelinated_targets', '_last_fired',
    )

    def __init__(self, word: str, threshold: float = 0.3):
        self.word = word
        self.potential = 0.0          # Current membrane potential
        self.threshold = threshold     # Firing threshold
        self.axon_targets: List[str] = []   # Who I send to (words)
        self.axon_weights: List[float] = []  # How strongly
        self.myelinated_targets: Set[int] = set()  # Indices of myelinated connections
        self.refractory = False        # Can't fire if True
        self.fire_count = 0
        self._last_fired = -1

    def receive(self, signal: float) -> bool:
        """
        Dendrite receives a signal. Adds to membrane potential.
        Returns True if the neuron fired.
        """
        if self.refractory:
            return False
        self.potential += signal
        if self.potential >= self.threshold:
            return True  # Will fire
        return False

    def fire(self, epoch: int) -> List[tuple]:
        """
        Axon fires. Returns list of (target_word, signal_strength).
        Resets potential. Enters refractory period.
        """
        if self.refractory or self.potential < self.threshold:
            return []

        self.fire_count += 1
        self._last_fired = epoch

        # Signals to send
        signals = []
        for i, (target, weight) in enumerate(zip(self.axon_targets, self.axon_weights)):
            # Myelinated connections transmit stronger (saltatory)
            mult = 1.5 if i in self.myelinated_targets else 1.0
            signals.append((target, self.potential * weight * mult))

        # Reset
        self.potential = 0.0
        self.refractory = True
        return signals

    def reset_refractory(self):
        """End refractory period. Neuron can fire again."""
        self.refractory = False

    def add_connection(self, target: str, weight: float, myelinated: bool = False):
        """Grow an axon terminal to another neuron."""
        # Check if already connected
        for i, t in enumerate(self.axon_targets):
            if t == target:
                self.axon_weights[i] = max(self.axon_weights[i], weight)
                if myelinated:
                    self.myelinated_targets.add(i)
                return
        # New connection — cap at 50
        if len(self.axon_targets) < 50:
            idx = len(self.axon_targets)
            self.axon_targets.append(target)
            self.axon_weights.append(weight)
            if myelinated:
                self.myelinated_targets.add(idx)


class NeuralField:
    """
    A field of neurons that propagate signals to each other.

    No central loop. Seed a neuron, let the wave propagate.
    Collect the activation pattern when the wave dies.

    activate(word) → fire seed → propagate → collect settled state
    """

    def __init__(self, max_propagation_steps: int = 5):
        self.neurons: Dict[str, Neuron] = {}
        self._max_steps = max_propagation_steps
        self._epoch = 0

    def ensure_neuron(self, word: str) -> Neuron:
        if word not in self.neurons:
            self.neurons[word] = Neuron(word)
        return self.neurons[word]

    def build_from_graphs(self, co_graph: Dict[str, Dict[str, float]],
                          cortex_connections: Optional[Dict[str, Dict[str, float]]] = None,
                          myelinated_pairs: Optional[Set[tuple]] = None,
                          golgi=None) -> int:
        """
        Build neurons from the co-graph and cortex.
        Each word becomes a neuron.
        Co-graph edges become axon connections.
        Cortex synapses strengthen existing connections.
        """
        built = 0
        myel_set = myelinated_pairs or set()

        for word, neighbors in co_graph.items():
            if len(word) <= 2:
                continue
            neuron = self.ensure_neuron(word)
            # Top neighbors by weight
            top = sorted(neighbors.items(), key=lambda x: -x[1])[:30]
            max_w = top[0][1] if top else 1
            for target, weight in top:
                if len(target) <= 2:
                    continue
                self.ensure_neuron(target)
                norm_w = weight / max(max_w, 1)
                # STDP: boost connections between temporally-close words
                if golgi:
                    stdp = golgi.stdp_affinity(word, target)
                    norm_w *= (0.5 + stdp * 0.5)  # STDP boosts up to 1.5×
                is_myel = (word, target) in myel_set
                neuron.add_connection(target, norm_w, is_myel)
                built += 1

        # Strengthen from cortex if available
        if cortex_connections:
            for source, targets in cortex_connections.items():
                if source not in self.neurons:
                    continue
                neuron = self.neurons[source]
                for target, weight in targets.items():
                    if target in self.neurons:
                        neuron.add_connection(target, min(weight / 10, 1.0))

        return built

    def activate(self, word: str, energy: float = 1.0) -> Dict[str, float]:
        """
        Seed one neuron. Let the wave propagate.
        Returns the activation pattern: {word: final_potential}

        No central loop. Each neuron fires independently.
        The wave dies when no more neurons cross threshold.
        """
        if word not in self.neurons:
            return {word: energy}

        self._epoch += 1
        # Reset all refractory states
        for n in self.neurons.values():
            n.refractory = False
            n.potential = 0.0

        # Seed
        seed = self.neurons[word]
        seed.potential = energy

        # Propagation: BFS-like but each neuron fires independently
        # The queue holds neurons that MIGHT fire
        pending = [word]
        activated: Dict[str, float] = {word: energy}
        steps = 0

        while pending and steps < self._max_steps:
            steps += 1
            next_pending = []

            for neuron_word in pending:
                neuron = self.neurons.get(neuron_word)
                if not neuron or neuron.potential < neuron.threshold:
                    continue

                # Fire!
                signals = neuron.fire(self._epoch)

                for target_word, signal in signals:
                    target = self.neurons.get(target_word)
                    if not target:
                        continue

                    # Decay signal with distance
                    decayed = signal * (0.7 ** steps)
                    if decayed < 0.01:
                        continue

                    # Receive
                    fired = target.receive(decayed)
                    # Record activation
                    activated[target_word] = activated.get(target_word, 0) + decayed

                    if fired:
                        next_pending.append(target_word)

            # Reset refractory for next wave step
            for nw in pending:
                n = self.neurons.get(nw)
                if n:
                    n.reset_refractory()

            pending = next_pending

        return activated

    def activate_multi(self, words: List[str], energy: float = 1.0) -> Dict[str, float]:
        """
        Activate multiple seeds simultaneously.
        Like hearing a sentence — all words fire at once.
        The interference pattern IS the meaning.
        """
        combined: Dict[str, float] = {}
        for word in words:
            field = self.activate(word, energy)
            for w, e in field.items():
                combined[w] = combined.get(w, 0) + e
        return combined

    def score_sequence(self, tokens: List[str]) -> dict:
        """Score coherence by measuring how well each word predicts the next."""
        if len(tokens) < 2:
            return {'coherence': 0.0, 'scores': []}

        scores = [1.0]  # First word always scores 1
        running_field: Dict[str, float] = {}

        for i, token in enumerate(tokens):
            if i > 0:
                # How well did the running field predict this word?
                score = min(running_field.get(token, 0), 1.0)
                scores.append(score)

            # Add this word's activation to the running field
            field = self.activate(token, 0.5)
            for w, e in field.items():
                running_field[w] = running_field.get(w, 0) + e

        coherence = sum(scores) / len(scores)
        return {'coherence': coherence, 'scores': scores}

    def stats(self) -> dict:
        total_connections = sum(len(n.axon_targets) for n in self.neurons.values())
        total_myel = sum(len(n.myelinated_targets) for n in self.neurons.values())
        return {
            'neurons': len(self.neurons),
            'connections': total_connections,
            'myelinated': total_myel,
            'epoch': self._epoch,
        }

    def to_dict(self) -> dict:
        # Only save topology, not state
        neurons = {}
        for word, n in list(self.neurons.items())[:5000]:
            neurons[word] = {
                'targets': n.axon_targets[:50],
                'weights': [round(w, 3) for w in n.axon_weights[:50]],
                'myel': sorted(n.myelinated_targets),
                'fires': n.fire_count,
            }
        return {'neurons': neurons, 'epoch': self._epoch}

    @classmethod
    def from_dict(cls, d: dict) -> NeuralField:
        nf = cls()
        nf._epoch = d.get('epoch', 0)
        for word, data in d.get('neurons', {}).items():
            n = nf.ensure_neuron(word)
            n.axon_targets = data.get('targets', [])
            n.axon_weights = data.get('weights', [])
            n.myelinated_targets = set(data.get('myel', []))
            n.fire_count = data.get('fires', 0)
        return nf
