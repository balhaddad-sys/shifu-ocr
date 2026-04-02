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

When the Rust extension (shifu_neural) is available, NeuralField
and Neuron are replaced with the Rust implementation — 20-50x faster
wave propagation through arena-indexed neurons with zero string
hashing in the hot path.
"""

from __future__ import annotations
from typing import Dict, List, Set, Optional, Callable

# ═══ RUST CORE — if available, replaces Python NeuralField ═══
try:
    from shifu_neural import NeuralField as _RustNeuralField
    _RUST_AVAILABLE = True
except ImportError:
    _RustNeuralField = None
    _RUST_AVAILABLE = False


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

    def receive(self, signal: float, excitatory: bool = True) -> bool:
        """
        Dendrite receives a signal.

        SUMMATION: multiple weak signals add up to fire threshold.
        INHIBITION: negative signals subtract from potential.
        REFRACTORY: can't fire during refractory period.

        Like real neurons: EPSPs and IPSPs sum at the axon hillock.
        """
        if self.refractory:
            return False
        if excitatory:
            self.potential += signal
        else:
            self.potential -= signal * 0.5  # Inhibition is weaker than excitation
            self.potential = max(0.0, self.potential)  # Can't go below 0
        if self.potential >= self.threshold:
            return True
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

        # Hebbian strengthening: firing strengthens the connections that carried signal
        self.strengthen_last_fired()

        # Reset
        self.potential = 0.0
        self.refractory = True
        return signals

    def strengthen_last_fired(self):
        """
        After firing, the connections that JUST carried signal get stronger.
        Hebbian: neurons that fire together wire together.
        Called automatically after fire(). No scanning needed.
        """
        if self._last_fired < 0:
            return
        for i in range(len(self.axon_weights)):
            self.axon_weights[i] += 0.1  # Every firing strengthens
            # Myelinate when weight crosses threshold through USAGE
            # Baby brains myelinate faster (lower threshold)
            if i not in self.myelinated_targets and self.axon_weights[i] > 0.75:
                self.myelinated_targets.add(i)

    def reset_refractory(self):
        """
        End absolute refractory. Enter relative refractory.
        Threshold temporarily raised — harder to fire again immediately.
        Like the sodium channel recovery in real neurons.
        """
        self.refractory = False
        # Relative refractory: threshold raised after firing
        self.threshold = min(self.threshold + 0.1, 0.8)

    def recover(self):
        """
        Gradual recovery toward resting threshold.
        Called during diastolic flow. The neuron slowly becomes
        as excitable as before.
        """
        if self.threshold > 0.3:
            self.threshold -= 0.02
            self.threshold = max(self.threshold, 0.3)

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

        The denser and more compact the web, the easier vibration
        transmits. A single pulse in a dense web reaches everything.
        In a sparse web, vibrations die quickly.

        - Only reset neurons that PARTICIPATED (not all N neurons)
        - Myelinated paths: signal decays less (saltatory = faster)
        - Propagation depth adapts: dense web = fewer steps needed
          because each step reaches more neurons
        - Cap total fired neurons to prevent runaway
        """
        if word not in self.neurons:
            return {word: energy}

        self._epoch += 1

        # Only reset neurons touched in the PREVIOUS activation
        # Not all N neurons — that's O(N) every call = death at scale
        if not hasattr(self, '_last_activated'):
            self._last_activated: set = set()
        for w in self._last_activated:
            n = self.neurons.get(w)
            if n:
                n.refractory = False
                n.potential = 0.0
        self._last_activated = set()

        # Seed
        seed = self.neurons[word]
        seed.potential = energy

        pending = [word]
        activated: Dict[str, float] = {word: energy}
        self._last_activated.add(word)

        # Max fired: cap to prevent runaway in dense webs
        max_fired = min(len(self.neurons), 200)
        total_fired = 0
        steps = 0

        while pending and steps < self._max_steps and total_fired < max_fired:
            steps += 1
            next_pending = []

            for neuron_word in pending:
                neuron = self.neurons.get(neuron_word)
                if not neuron or neuron.potential < neuron.threshold:
                    continue

                signals = neuron.fire(self._epoch)
                total_fired += 1

                for target_word, signal in signals:
                    target = self.neurons.get(target_word)
                    if not target:
                        continue

                    # Myelinated paths decay LESS — saltatory conduction
                    # Dense web = signals travel further naturally
                    # Check if this specific connection is myelinated
                    idx = None
                    for i, t in enumerate(neuron.axon_targets):
                        if t == target_word:
                            idx = i
                            break
                    is_myel = idx is not None and idx in neuron.myelinated_targets
                    decay = 0.85 if is_myel else 0.65  # Myel: 15% loss vs 35%
                    decayed = signal * decay
                    if decayed < 0.01:
                        continue

                    fired = target.receive(decayed)
                    activated[target_word] = activated.get(target_word, 0) + decayed
                    self._last_activated.add(target_word)

                    if fired and total_fired < max_fired:
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

    # ═══ SPIDER WEB — everything through vibration ═══

    def heartbeat(self) -> dict:
        """
        The spider sits in the center and FEELS the web.
        Don't scan dictionaries. Send a vibration through the web.
        Every neuron that vibrates gets its connections strengthened.
        Myelination happens through USAGE (Hebbian), not through scanning.

        Pick a random seed neuron. Activate it. The wave propagates.
        Every neuron that fires strengthens its own connections.
        Connections that strengthen past threshold myelinate automatically.
        """
        if not self.neurons:
            return {'myelinated_new': 0, 'shortcuts': 0, 'fired': 0}

        import random
        # Pick a random neuron as the heartbeat pulse
        seed_word = random.choice(list(self.neurons.keys()))
        # Send a gentle vibration (not full activation — just a pulse)
        field = self.activate(seed_word, energy=0.5)

        # Count what happened from the vibration
        fired = len(field)
        myel_before = sum(len(n.myelinated_targets) for n in self.neurons.values())

        # Every neuron that fired already strengthened its connections
        # (Hebbian in fire() → strengthen_last_fired())
        # Count new myelinations
        myel_after = sum(len(n.myelinated_targets) for n in self.neurons.values())

        return {
            'myelinated_new': myel_after - myel_before,
            'shortcuts': 0,
            'fired': fired,
            'seed': seed_word,
        }

    def practice(self, focus_word: str) -> dict:
        """
        Practice = activate the focus word STRONGLY.
        The strong activation propagates further, reaching more neurons.
        More neurons fire = more Hebbian strengthening = faster myelination.
        Like a spider plucking a specific thread hard.
        """
        if focus_word not in self.neurons:
            return {'improved': 0, 'fired': 0}

        myel_before = sum(len(n.myelinated_targets) for n in self.neurons.values())
        field = self.activate(focus_word, energy=2.0)  # Strong activation
        myel_after = sum(len(n.myelinated_targets) for n in self.neurons.values())

        return {
            'improved': myel_after - myel_before,
            'fired': len(field),
        }

    def prune(self, decay: float = 0.99, min_weight: float = 0.05) -> int:
        """
        Delta-state maintenance: decay unused connections.
        Connections that haven't fired weaken.
        Myelinated connections resist decay (they barely weaken).
        Weak connections below min_weight get removed.
        """
        pruned = 0
        for neuron in self.neurons.values():
            dead = []
            for i in range(len(neuron.axon_weights)):
                if i in neuron.myelinated_targets:
                    neuron.axon_weights[i] *= 0.999  # Myelinated barely decay
                else:
                    neuron.axon_weights[i] *= decay
                if neuron.axon_weights[i] < min_weight:
                    dead.append(i)
            # Remove dead connections (reverse order to preserve indices)
            for i in sorted(dead, reverse=True):
                neuron.axon_targets.pop(i)
                neuron.axon_weights.pop(i)
                neuron.myelinated_targets.discard(i)
                # Shift myelinated indices
                new_myel = set()
                for m in neuron.myelinated_targets:
                    if m > i:
                        new_myel.add(m - 1)
                    elif m < i:
                        new_myel.add(m)
                neuron.myelinated_targets = new_myel
                pruned += 1
        return pruned

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


# ═══════════════════════════════════════════════════════════
#  RUST SWAP — if shifu_neural is installed, use the Rust core
# ═══════════════════════════════════════════════════════════
if _RUST_AVAILABLE:
    # Keep Python classes as _PyNeuralField / _PyNeuron for fallback
    _PyNeuralField = NeuralField
    _PyNeuron = Neuron
    # Replace with Rust
    NeuralField = _RustNeuralField
