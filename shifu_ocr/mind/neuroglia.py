"""
Neuroglia — the OTHER half of the brain.

Neurons get all the glory but glia outnumber them 10:1.
They don't just pipe blood. They REGULATE, GENERATE, HEAL, and COOL.

ASTROCYTES — thermal regulation + metabolic support
    - Detect firing rate per region (heat)
    - When a region overheats: release inhibitory signals (cool it)
    - When a region is cold: increase blood flow (warm it up)
    - Recycle neurotransmitters (reset neurons faster)
    - Calcium waves: slow, wide signals that modulate entire regions

MICROGLIA — immune system + healing + pruning
    - Patrol the neural field for damaged/dead neurons
    - Prune weak synapses (complement-tagged)
    - Phagocytose debris (remove dead connections)
    - Release cytokines: pro-inflammatory (increase vigilance)
      or anti-inflammatory (promote healing/sleep)
    - After injury (bad data, contradictions): activate and clean up

OLIGODENDROCYTE PRECURSOR CELLS (OPCs) — neurogenesis
    - Generate NEW neurons in response to demand
    - Active regions that need more capacity get new neurons
    - Differentiate into oligodendrocytes (myelination)
    - Track which regions are growing vs shrinking

Each glial type operates on a different timescale:
    - Astrocytes: fast (every cycle) — immediate thermal regulation
    - Microglia: medium (every 10 cycles) — patrol and clean
    - OPCs: slow (every 50 cycles) — grow and differentiate
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set
import time


class Astrocyte:
    """
    Thermal regulation for a brain region.

    Tracks firing rate. When too hot: inhibit. When cold: warm up.
    Each cortex layer gets its own astrocyte — different regions
    have different thermal tolerances.
    """

    __slots__ = ('region', 'heat', 'tolerance', 'coolant',
                 '_fire_history', '_max_history')

    def __init__(self, region: str, tolerance: float = 10.0):
        self.region = region
        self.heat = 0.0           # Current thermal load
        self.tolerance = tolerance # Max heat before throttling
        self.coolant = 0.0        # Inhibitory signal to release
        self._fire_history: List[float] = []
        self._max_history = 20

    def observe_firing(self, fire_count: int) -> None:
        """Record firing activity in this region."""
        self._fire_history.append(float(fire_count))
        if len(self._fire_history) > self._max_history:
            self._fire_history = self._fire_history[-self._max_history:]
        # Heat = exponential moving average of firing rate
        alpha = 0.3
        self.heat = self.heat * (1 - alpha) + fire_count * alpha

    def regulate(self) -> dict:
        """
        Thermal regulation cycle.
        Returns: {throttle: 0-1, boost: 0-1, signal: str}
        throttle > 0: reduce processing in this region
        boost > 0: increase processing in this region
        """
        if self.heat > self.tolerance:
            # OVERHEATING — release inhibitory coolant
            overheat = (self.heat - self.tolerance) / self.tolerance
            self.coolant = min(overheat, 1.0)
            # Gradual cooling
            self.heat *= 0.8
            return {
                'throttle': self.coolant,
                'boost': 0.0,
                'signal': 'cooling',
                'heat': round(self.heat, 2),
            }
        elif self.heat < self.tolerance * 0.2:
            # COLD — this region needs more activity
            cold = 1.0 - (self.heat / max(self.tolerance * 0.2, 0.01))
            self.coolant = 0.0
            return {
                'throttle': 0.0,
                'boost': min(cold * 0.5, 0.5),
                'signal': 'warming',
                'heat': round(self.heat, 2),
            }
        else:
            # Normal range
            self.coolant = max(0, self.coolant - 0.1)
            return {
                'throttle': 0.0,
                'boost': 0.0,
                'signal': 'normal',
                'heat': round(self.heat, 2),
            }

    def calcium_wave(self) -> float:
        """
        Slow, wide signal that modulates processing depth.
        High calcium = deeper processing. Low = shallow.
        Based on recent trend: rising heat = more engagement = deeper.
        """
        if len(self._fire_history) < 3:
            return 0.5  # Default medium depth
        recent = sum(self._fire_history[-3:]) / 3
        older = sum(self._fire_history[:3]) / max(len(self._fire_history[:3]), 1)
        trend = recent - older
        # Rising: deeper (up to 1.0). Falling: shallower (down to 0.1).
        return max(0.1, min(1.0, 0.5 + trend * 0.1))


class Microglia:
    """
    Immune system of the neural field.

    Patrols for damage. Prunes weak synapses. Heals after injury.
    Operates on medium timescale (every ~10 cycles).
    """

    def __init__(self):
        self._patrol_count = 0
        self._pruned_total = 0
        self._healed_total = 0
        self._inflammatory = 0.0  # 0=calm, 1=inflamed (high vigilance)
        self._damage_log: List[dict] = []

    def patrol(self, neural_field) -> dict:
        """
        Patrol the neural field. For Rust backend: use prune() method.
        For Python backend: iterate directly.
        """
        self._patrol_count += 1
        pruned = 0
        healed = 0
        removed_neurons = []

        if hasattr(neural_field, 'connect'):
            # Rust backend — use the built-in prune method
            # Microglia's job in Rust: just prune weak connections
            pruned = neural_field.prune(0.99, 0.02)
        else:
            # Python backend — iterate neurons directly
            for word, neuron in list(neural_field.neurons.items()):
                if len(neuron.axon_targets) == 0 and neuron.fire_count == 0:
                    removed_neurons.append(word)
                    continue
                dead_indices = []
                for i, weight in enumerate(neuron.axon_weights):
                    if weight < 0.02 and i not in neuron.myelinated_targets:
                        dead_indices.append(i)
                for i in sorted(dead_indices, reverse=True):
                    neuron.axon_targets.pop(i)
                    neuron.axon_weights.pop(i)
                    neuron.myelinated_targets.discard(i)
                    new_myel = set()
                    for m in neuron.myelinated_targets:
                        if m > i: new_myel.add(m - 1)
                        elif m < i: new_myel.add(m)
                    neuron.myelinated_targets = new_myel
                    pruned += 1
                if neuron.fire_count > 1000:
                    neuron.fire_count = neuron.fire_count // 2
                    healed += 1

        # Remove dead neurons
        for word in removed_neurons:
            del neural_field.neurons[word]

        self._pruned_total += pruned
        self._healed_total += healed

        # Inflammatory state: rises with damage, decays naturally
        if pruned > 10 or len(removed_neurons) > 5:
            self._inflammatory = min(1.0, self._inflammatory + 0.2)
        else:
            self._inflammatory = max(0.0, self._inflammatory - 0.05)

        return {
            'pruned': pruned,
            'removed_neurons': len(removed_neurons),
            'healed': healed,
            'inflammatory': round(self._inflammatory, 2),
        }

    def detect_damage(self, contradiction: dict) -> None:
        """Record detected contradiction/damage for healing."""
        self._damage_log.append(contradiction)
        self._inflammatory = min(1.0, self._inflammatory + 0.3)
        if len(self._damage_log) > 50:
            self._damage_log = self._damage_log[-25:]

    def is_inflamed(self) -> bool:
        return self._inflammatory > 0.5


class OPC:
    """
    Oligodendrocyte Precursor Cells — neurogenesis + myelination support.

    Track which regions are growing vs shrinking.
    Generate new neurons where demand is high.
    Differentiate into myelinating cells where usage is intense.
    Operates on slow timescale (every ~50 cycles).
    """

    def __init__(self):
        self._generation_count = 0
        self._differentiation_count = 0
        self._demand: Dict[str, float] = {}  # region -> demand score

    def assess_demand(self, neural_field, co_graph: dict) -> dict:
        """
        Where does the brain need more capacity?
        High demand: many words in co-graph but few neurons.
        Low demand: many neurons but sparse co-graph.
        """
        co_words = set(co_graph.keys())
        neuron_words = set(neural_field.neurons.keys())

        # Words that appear in co-graph but have no neuron = unmet demand
        unmet = co_words - neuron_words
        # Words with neurons but not in co-graph = over-provisioned
        excess = neuron_words - co_words

        return {
            'unmet': len(unmet),
            'excess': len(excess),
            'demand_ratio': len(unmet) / max(len(co_words), 1),
            'unmet_samples': list(unmet)[:10],
        }

    def generate(self, neural_field, co_graph: dict, max_new: int = 20) -> dict:
        """
        Neurogenesis: birth new neurons where demand exists.
        Only for words that have co-graph connections but no neuron yet.
        """
        co_words = set(co_graph.keys())
        neuron_words = set(neural_field.neurons.keys())
        unmet = co_words - neuron_words

        # Prioritize by co-graph connection count (busier = more needed)
        ranked = sorted(
            [(w, len(co_graph.get(w, {}))) for w in unmet if len(w) > 2],
            key=lambda x: -x[1],
        )[:max_new]

        born = 0
        for word, _ in ranked:
            neuron = neural_field.ensure_neuron(word)
            # Wire to co-graph neighbors that already have neurons
            neighbors = co_graph.get(word, {})
            for neighbor, weight in sorted(neighbors.items(), key=lambda x: -x[1])[:10]:
                if neighbor in neural_field.neurons and neighbor != word:
                    norm_w = min(weight / 10.0, 1.0)
                    neuron.add_connection(neighbor, norm_w)
            born += 1

        self._generation_count += born
        return {'born': born}

    def differentiate(self, neural_field) -> dict:
        """
        OPCs differentiate into oligodendrocytes near active axons.
        For Rust backend: use stats-based approach (can't iterate individual weights).
        For Python backend: iterate directly.
        """
        myelinated = 0
        # Check if this is the Rust backend (has 'connect' method but no direct neuron access)
        if hasattr(neural_field, 'connect'):
            # Rust backend — differentiation happens inside Rust heartbeat
            # via Hebbian strengthening (weight > 0.75 → auto-myelinate).
            # Nothing to do here — Rust handles it.
            pass
        else:
            # Python backend — iterate neurons directly
            for neuron in neural_field.neurons.values():
                for i, weight in enumerate(neuron.axon_weights):
                    if i not in neuron.myelinated_targets and weight > 0.6:
                        neuron.myelinated_targets.add(i)
                        myelinated += 1
        self._differentiation_count += myelinated
        return {'myelinated': myelinated}


class Neuroglia:
    """
    The complete glial system. One per brain.

    Manages astrocytes (per region), microglia (global), OPCs (global).
    Runs on different timescales:
      - Astrocytes: every cycle (thermal regulation)
      - Microglia: every 10 cycles (immune patrol)
      - OPCs: every 50 cycles (neurogenesis)
    """

    def __init__(self):
        self.astrocytes: Dict[str, Astrocyte] = {}
        self.microglia = Microglia()
        self.opc = OPC()
        self._cycle = 0
        # Global thermal state
        self._global_heat = 0.0
        self._max_heat = 50.0  # System-wide thermal limit

    def ensure_astrocyte(self, region: str, tolerance: float = 10.0) -> Astrocyte:
        """Each cortex layer / brain region gets its own astrocyte."""
        if region not in self.astrocytes:
            self.astrocytes[region] = Astrocyte(region, tolerance)
        return self.astrocytes[region]

    def cycle(self, neural_field, co_graph: dict) -> dict:
        """
        One neuroglia cycle. Different components fire at different rates.
        Returns regulation signals for the brain to act on.
        """
        self._cycle += 1
        result = {'cycle': self._cycle}

        # ═══ ASTROCYTES: every cycle (fast — thermal regulation) ═══
        thermal = {}
        total_heat = 0.0
        for region, astro in self.astrocytes.items():
            reg = astro.regulate()
            thermal[region] = reg
            total_heat += reg['heat']
        self._global_heat = total_heat
        result['thermal'] = thermal
        result['global_heat'] = round(total_heat, 2)

        # Global thermal limit: if total heat exceeds max, throttle everything
        if total_heat > self._max_heat:
            result['global_throttle'] = min((total_heat - self._max_heat) / self._max_heat, 1.0)
        else:
            result['global_throttle'] = 0.0

        # ═══ MICROGLIA: every 10 cycles (medium — immune patrol) ═══
        if self._cycle % 10 == 0:
            patrol = self.microglia.patrol(neural_field)
            result['patrol'] = patrol
        else:
            result['patrol'] = None

        # ═══ OPCs: every 50 cycles (slow — neurogenesis + myelination) ═══
        if self._cycle % 50 == 0:
            demand = self.opc.assess_demand(neural_field, co_graph)
            if demand['demand_ratio'] > 0.1:
                gen = self.opc.generate(neural_field, co_graph)
                result['neurogenesis'] = gen
            diff = self.opc.differentiate(neural_field)
            result['differentiation'] = diff
        else:
            result['neurogenesis'] = None
            result['differentiation'] = None

        return result

    def observe_region(self, region: str, fire_count: int) -> None:
        """Report firing activity to the region's astrocyte."""
        astro = self.ensure_astrocyte(region)
        astro.observe_firing(fire_count)

    def is_overheated(self) -> bool:
        """Is the brain globally overheated?"""
        return self._global_heat > self._max_heat

    def processing_depth(self, region: str) -> float:
        """
        How deep should processing be in this region?
        Calcium wave modulates depth: 0.1 (shallow) to 1.0 (deep).
        Used by replay/consolidation to decide how much work to do.
        """
        astro = self.astrocytes.get(region)
        if not astro:
            return 0.5
        return astro.calcium_wave()

    def stats(self) -> dict:
        return {
            'cycle': self._cycle,
            'global_heat': round(self._global_heat, 2),
            'regions': {r: {'heat': round(a.heat, 2), 'coolant': round(a.coolant, 2)}
                        for r, a in self.astrocytes.items()},
            'microglia': {
                'patrols': self.microglia._patrol_count,
                'pruned': self.microglia._pruned_total,
                'healed': self.microglia._healed_total,
                'inflammatory': round(self.microglia._inflammatory, 2),
            },
            'opc': {
                'generated': self.opc._generation_count,
                'differentiated': self.opc._differentiation_count,
            },
        }

    def to_dict(self) -> dict:
        return {
            'cycle': self._cycle,
            'global_heat': self._global_heat,
            'microglia_inflammatory': self.microglia._inflammatory,
            'opc_generated': self.opc._generation_count,
            'opc_differentiated': self.opc._differentiation_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Neuroglia:
        ng = cls()
        ng._cycle = d.get('cycle', 0)
        ng._global_heat = d.get('global_heat', 0)
        ng.microglia._inflammatory = d.get('microglia_inflammatory', 0)
        ng.opc._generation_count = d.get('opc_generated', 0)
        ng.opc._differentiation_count = d.get('opc_differentiated', 0)
        return ng
