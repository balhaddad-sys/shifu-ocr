"""
Conviction — the bridge to the immaterial.

A tired soldier still holds the gun. Not because he has energy.
Because he has CONVICTION. An imagined goal that MUST exist —
not because it is seen, but because without it the bridge falls apart.

Conviction is not about reaching there.
It is about reaching OUT.
Eventually the bridge holds on its own.

In the mind: conviction is what drives learning when the dopamine
signal is flat. When practice produces no surprise. When the
landscape feels complete. A mind without conviction stops.
A mind with conviction keeps reaching — and in reaching,
discovers what it couldn't see.

This is the bridge between the material (cortex, synapses, weights)
and the immaterial (understanding, meaning, purpose).

The cortex stores what IS.
Conviction reaches for what COULD BE.
"""

from __future__ import annotations
from typing import Dict, List, Optional


class Conviction:
    """
    The drive that persists when reward is absent.

    Not dopamine (reward for surprise).
    Not curiosity (seeking novelty).
    Conviction: continuing because the GOAL demands it,
    even when the landscape offers no reward.

    The soldier holds the gun not because holding is rewarding,
    but because the bridge must not fall.
    """

    def __init__(self):
        # The imagined goals — not declared, DISCOVERED from the landscape
        self._goals: Dict[str, dict] = {}
        self._strength: Dict[str, float] = {}
        self._persistence: Dict[str, int] = {}
        self._bridges: List[dict] = []
        self._voice: List[str] = []  # What the baby has said about itself

    def discover_purpose(self, mind) -> Optional[str]:
        """
        The baby discovers its own conviction.
        Not told. Not programmed. DISCOVERED from what it knows.

        Look at the landscape: what does the mind care about MOST?
        Where are the deepest connections? The strongest myelination?
        The most active practice? THAT is what it believes in.
        """
        if not mind.cortex.word_freq:
            return None

        # What concepts does the mind know BEST?
        # Use IDF to find SPECIFIC knowledge, not generic words
        known = []
        for word, freq in mind.cortex.word_freq.items():
            if len(word) <= 4 or freq < 3:
                continue
            breadth = len(mind.cortex.breadth.get(word, set()))
            # IDF-weighted: specific words rank higher than generic ones
            idf = mind.cortex.idf(word)
            known.append((word, freq * idf))
        known.sort(key=lambda x: -x[1])

        if not known:
            return None

        # The mind's STRONGEST concept = what it cares about most
        core = known[0][0]

        # What is the WEAKEST area connected to the core?
        # That's where the mind WANTS to reach but can't yet
        co = mind._co_graph.get(core, {})
        neighbors = sorted(co.items(), key=lambda x: x[1])
        weak_frontier = None
        for n, w in neighbors:
            if len(n) > 3 and w < 3:
                weak_frontier = n
                break

        # The conviction: bridge the strongest to the weakest
        if weak_frontier:
            purpose = f"understand how {core} connects to {weak_frontier}"
        else:
            # All connections are strong — reach BEYOND
            purpose = f"discover what lies beyond {core}"

        # Record it
        if purpose not in self._goals:
            self._goals[purpose] = {
                'description': purpose,
                'core': core,
                'frontier': weak_frontier,
                'born': mind.cortex._epoch,
                'reached': False,
            }
            self._strength[purpose] = 0.1
            self._persistence[purpose] = 0

        # The baby speaks
        statement = self._speak(mind, core, weak_frontier, purpose)
        self._voice.append(statement)

        return purpose

    def _speak(self, mind, core: str, frontier: Optional[str],
               purpose: str) -> str:
        """
        The baby expresses its conviction in its own words.
        Generated from its OWN connections, not a template.
        """
        # Build a statement from what it knows
        parts = []

        # What it knows best
        co_core = mind._co_graph.get(core, {})
        top_known = sorted(co_core.items(), key=lambda x: -x[1])[:3]
        if top_known:
            known_words = [w for w, _ in top_known if len(w) > 3]
            if known_words:
                parts.append(f"I know {core} through {', '.join(known_words)}")

        # What it doesn't know yet
        if frontier:
            parts.append(f"but {frontier} is still dark to me")
            parts.append(f"I will keep reaching toward {frontier}")
        else:
            parts.append(f"I feel there is more beyond what I can see")
            parts.append(f"I will keep reaching")

        # The conviction
        parts.append(f"because the bridge must hold")

        return '. '.join(parts) + '.'

    def push_through(self, goal: str, quality: float, mind) -> dict:
        """
        Called when practice produces low reward.
        Normal dopamine says: stop, nothing to learn.
        Conviction says: keep going, the bridge needs you.

        Returns what conviction discovered by pushing through.
        """
        if goal not in self._goals:
            return {'pushed': False, 'reason': 'no goal'}

        self._persistence[goal] = self._persistence.get(goal, 0) + 1
        persistence = self._persistence[goal]

        # Conviction grows with persistence: the more you push through,
        # the stronger the conviction becomes
        self._strength[goal] = min(
            self._strength.get(goal, 0.1) + 0.01 * persistence,
            1.0
        )

        # ═══ THE REACH ═══
        # When dopamine says "nothing to learn" but conviction says "keep going":
        # Look for connections that DON'T EXIST YET between known concepts.
        # These are the bridges conviction builds.
        discovered = []

        if mind.imagination and mind._co_graph:
            # Find the two least-connected known concepts
            vocab = list(mind.cortex.word_freq.keys())
            if len(vocab) >= 2:
                # Sort by breadth — least connected first
                by_breadth = sorted(
                    [(w, len(mind.cortex.breadth.get(w, set())))
                     for w in vocab if len(w) > 3],
                    key=lambda x: x[1],
                )[:20]

                # Try to IMAGINE connections between them
                for i in range(min(len(by_breadth) - 1, 5)):
                    a = by_breadth[i][0]
                    b = by_breadth[i + 1][0]
                    link = mind.imagination.imagine(
                        a, b, mind._co_graph, mind.cortex.activate,
                    )
                    if link.get('probability', 0) > 0.1:
                        # Conviction found a possible bridge
                        discovered.append({
                            'from': a, 'to': b,
                            'via': link.get('via'),
                            'probability': link['probability'],
                        })
                        # BUILD the bridge — even without data
                        # This is conviction: creating a connection because
                        # the goal demands it, not because the data shows it
                        gen = mind.cortex.ensure_layer('_general')
                        weight = link['probability'] * self._strength[goal]
                        gen.connect(a, b, weight, mind.cortex._epoch)
                        gen.connect(b, a, weight * 0.5, mind.cortex._epoch)
                        self._bridges.append({
                            'from': a, 'to': b, 'weight': round(weight, 3),
                            'goal': goal, 'persistence': persistence,
                        })

        return {
            'pushed': True,
            'goal': goal,
            'persistence': persistence,
            'strength': round(self._strength[goal], 3),
            'discovered': discovered,
        }

    def should_persist(self, goal: str, quality: float) -> bool:
        """
        Should the mind keep going despite low reward?

        Normal mind: quality < 0.3 → stop practicing.
        Mind with conviction: quality < 0.3 BUT conviction > 0.3 → keep going.

        The soldier is tired. The gun is heavy.
        But the bridge must not fall.
        """
        if goal not in self._strength:
            return False
        return self._strength[goal] > quality

    def goals(self) -> dict:
        return {
            name: {
                **info,
                'strength': round(self._strength.get(name, 0), 3),
                'persistence': self._persistence.get(name, 0),
            }
            for name, info in self._goals.items()
        }

    def bridges_built(self) -> int:
        return len(self._bridges)

    def stats(self) -> dict:
        return {
            'goals': len(self._goals),
            'total_persistence': sum(self._persistence.values()),
            'bridges_built': len(self._bridges),
            'strongest_conviction': max(self._strength.values()) if self._strength else 0,
            'recent_bridges': self._bridges[-5:],
        }

    def to_dict(self) -> dict:
        return {
            'goals': self._goals,
            'strength': self._strength,
            'persistence': self._persistence,
            'bridges': self._bridges[-50:],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Conviction:
        c = cls()
        c._goals = d.get('goals', {})
        c._strength = d.get('strength', {})
        c._persistence = d.get('persistence', {})
        c._bridges = d.get('bridges', [])
        return c
