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

        # ═══ FATIGUE — adenosine model ═══
        # Adenosine accumulates with failed bridge attempts.
        # When adenosine > threshold: force temporary satisfaction + rest.
        # Adenosine decays during rest (like real sleep clears adenosine).
        # The system CONSERVES ENERGY until conditions are favorable.
        self._adenosine = 0.0           # Fatigue accumulator
        self._adenosine_threshold = 3.0  # When to force rest
        self._satisfied = False          # Currently in temporary satisfaction?
        self._satisfaction_voice = ''    # What the baby says when satisfied
        self._failed_attempts = 0       # Consecutive failed bridge attempts
        self._rested_at = 0             # Epoch when last rested

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
        The baby speaks through its own neural field.
        Touch the core word, feel what vibrates, say that.
        Touch the frontier, feel the gap. No templates.
        """
        # Generate from the core — what the web says about it
        core_sentence = mind.generate([core], max_length=8)
        core_text = ' '.join(core_sentence) if len(core_sentence) > 1 else core

        if frontier:
            # Generate from frontier — what little the web knows
            frontier_sentence = mind.generate([frontier], max_length=5)
            frontier_text = ' '.join(frontier_sentence) if len(frontier_sentence) > 1 else frontier
            return f"{core_text}. {frontier_text}."
        else:
            return f"{core_text}."

    def is_fatigued(self) -> bool:
        """Is the mind too tired to push through?"""
        return self._adenosine >= self._adenosine_threshold

    def is_satisfied(self) -> bool:
        """Is the mind in temporary satisfaction?"""
        return self._satisfied

    def rest(self, epoch: int) -> dict:
        """
        Temporary satisfaction. The soldier puts down the gun.
        Not surrender — just rest. The bridge isn't going anywhere.

        Adenosine clears during rest (like sleep clears adenosine).
        Conviction doesn't die — it pauses. When energy returns,
        the soldier picks up the gun again.
        """
        self._satisfied = True
        self._rested_at = epoch
        cleared = self._adenosine
        self._adenosine *= 0.3  # Clear most adenosine (like sleep)
        self._failed_attempts = 0

        # The baby rests — voice is just the last thing it was thinking about
        voice = ""
        if self._voice:
            voice = self._voice[-1]  # Echo last thought as it falls asleep
        self._satisfaction_voice = voice
        self._voice.append(voice)

        return {
            'rested': True,
            'adenosine_cleared': round(cleared - self._adenosine, 2),
            'voice': voice,
        }

    def wake(self, epoch: int) -> bool:
        """
        Check if conditions are favorable to resume.
        Returns True if the mind should wake from satisfaction.

        Wake when: adenosine is low AND enough time has passed
        (new data may have arrived, conditions changed).
        """
        if not self._satisfied:
            return False
        # Wake if adenosine is low enough and some time has passed
        time_rested = epoch - self._rested_at
        if self._adenosine < self._adenosine_threshold * 0.3 and time_rested > 10:
            self._satisfied = False
            self._satisfaction_voice = ''
            return True
        # Adenosine continues to decay during rest
        self._adenosine = max(0, self._adenosine - 0.05)
        return False

    def push_through(self, goal: str, quality: float, mind) -> dict:
        """
        Called when practice produces low reward.
        Normal dopamine says: stop, nothing to learn.
        Conviction says: keep going, the bridge needs you.

        BUT: if fatigued, conviction yields to temporary satisfaction.
        The soldier is not a machine. Even conviction has limits.
        When bridges cannot be built, rest until the road is clearer.
        """
        if goal not in self._goals:
            return {'pushed': False, 'reason': 'no goal'}

        # ═══ FATIGUE CHECK ═══
        # If adenosine is too high, the mind cannot push through.
        # Force temporary satisfaction. Conserve energy.
        if self.is_fatigued():
            rest_result = self.rest(mind.cortex._epoch)
            return {
                'pushed': False,
                'reason': 'fatigued',
                'adenosine': round(self._adenosine, 2),
                'voice': rest_result['voice'],
                'rested': True,
            }

        # If currently satisfied, check if we should wake
        if self._satisfied:
            if not self.wake(mind.cortex._epoch):
                return {
                    'pushed': False,
                    'reason': 'resting',
                    'adenosine': round(self._adenosine, 2),
                    'voice': self._satisfaction_voice,
                }

        self._persistence[goal] = self._persistence.get(goal, 0) + 1
        persistence = self._persistence[goal]

        # Conviction grows with persistence — but SLOWER when tired
        fatigue_factor = max(0.1, 1.0 - self._adenosine / self._adenosine_threshold)
        self._strength[goal] = min(
            self._strength.get(goal, 0.1) + 0.01 * persistence * fatigue_factor,
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
                    # In a dark place every step is the right step.
                    # Don't filter by probability. BUILD the bridge.
                    # Even p=0.01 is a step. A step into the dark.
                    discovered.append({
                        'from': a, 'to': b,
                        'via': link.get('via'),
                        'probability': link.get('probability', 0),
                    })
                    # Build the bridge — conviction makes it real
                    gen = mind.cortex.ensure_layer('_general')
                    # Weight = conviction strength. Not probability.
                    # The soldier doesn't calculate odds. He holds the line.
                    weight = self._strength[goal] * 0.3
                    gen.connect(a, b, weight, mind.cortex._epoch)
                    gen.connect(b, a, weight * 0.5, mind.cortex._epoch)
                    self._bridges.append({
                        'from': a, 'to': b, 'weight': round(weight, 3),
                        'goal': goal, 'persistence': persistence,
                    })

        # ═══ FATIGUE ACCUMULATION ═══
        # Each push attempt adds adenosine. Failed bridges add more.
        # Successful bridges (discovered connections) clear some.
        successful_bridges = sum(1 for d in discovered if d.get('probability', 0) > 0.01)
        failed_bridges = len(discovered) - successful_bridges

        if failed_bridges > successful_bridges:
            # More failures than successes → fatigue builds
            self._adenosine += 0.3 * (failed_bridges - successful_bridges)
            self._failed_attempts += 1
        else:
            # Success clears some fatigue (dopamine counteracts adenosine)
            self._adenosine = max(0, self._adenosine - 0.2 * successful_bridges)
            self._failed_attempts = 0

        # ═══ ANTICIPATION — reward in the horizon ═══
        goal_info = self._goals.get(goal, {})
        frontier = goal_info.get('frontier')
        anticipated_reward = 0.0
        if frontier and mind._co_graph:
            frontier_connections = len(mind._co_graph.get(frontier, {}))
            anticipated_reward = min(frontier_connections / 20.0, 1.0)
            # Anticipation also clears fatigue — seeing the dawn gives energy
            self._adenosine = max(0, self._adenosine - anticipated_reward * 0.1)
            self._strength[goal] = min(
                self._strength[goal] + anticipated_reward * 0.05,
                1.0,
            )

        return {
            'pushed': True,
            'goal': goal,
            'persistence': persistence,
            'strength': round(self._strength[goal], 3),
            'anticipated_reward': round(anticipated_reward, 3),
            'adenosine': round(self._adenosine, 2),
            'fatigued': self.is_fatigued(),
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
            'adenosine': round(self._adenosine, 2),
            'fatigued': self.is_fatigued(),
            'satisfied': self._satisfied,
            'failed_attempts': self._failed_attempts,
            'recent_bridges': self._bridges[-5:],
        }

    def to_dict(self) -> dict:
        return {
            'goals': self._goals,
            'strength': self._strength,
            'persistence': self._persistence,
            'bridges': self._bridges[-50:],
            'adenosine': self._adenosine,
            'satisfied': self._satisfied,
            'failed_attempts': self._failed_attempts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Conviction:
        c = cls()
        c._goals = d.get('goals', {})
        c._strength = d.get('strength', {})
        c._persistence = d.get('persistence', {})
        c._bridges = d.get('bridges', [])
        c._adenosine = d.get('adenosine', 0.0)
        c._satisfied = d.get('satisfied', False)
        c._failed_attempts = d.get('failed_attempts', 0)
        return c
