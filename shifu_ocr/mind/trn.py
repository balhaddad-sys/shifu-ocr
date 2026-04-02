"""
TRN — Thalamic Reticular Nucleus.

The attentional spotlight. A thin shell of inhibitory neurons
that gates ALL information flowing between cortex layers.

Architecture:
    Cortex → TCR (excitatory): cortex activates relay neurons
    Cortex → TRN (excitatory): cortex activates inhibitory gate
    TRN → TCR (inhibitory): TRN SUPPRESSES relay neurons

The magic: cortical activation of TRN creates FEEDFORWARD INHIBITION.
When the cortex focuses on one modality (e.g. identity), the TRN
fires over NON-ATTENDED modalities (appearance, mechanism, relation),
suppressing their TCR targets. This reduces sensory noise.

Each spoke has its own TCR channel:
    identity_tcr    — relays "what is it" information
    appearance_tcr  — relays "how does it look" information
    function_tcr    — relays "what is it used for" information
    mechanism_tcr   — relays "how does it work" information
    relation_tcr    — relays "what is it connected to" information

The TRN sits between ALL channels. When one channel is attended,
the others are suppressed. Like a switchboard operator who connects
one caller and puts the rest on hold.

During focused attention:
    - Attended channel: TCR fires, TRN silent → signal passes
    - Non-attended channels: TRN fires → TCR suppressed → noise reduced

During broad attention (resting/exploration):
    - TRN partially active → all channels partially open
    - No single channel dominates → broad awareness
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set


class TCRChannel:
    """
    Thalamocortical Relay channel for one modality/spoke.

    Receives input from sensory processing (bottom-up)
    and cortical feedback (top-down). Relays to cortex
    UNLESS suppressed by TRN.
    """

    __slots__ = ('name', 'activation', 'suppression', '_relay_count',
                 '_suppressed_count', '_pending')

    def __init__(self, name: str):
        self.name = name
        self.activation = 0.0     # Current excitatory drive
        self.suppression = 0.0    # Current inhibition from TRN
        self._relay_count = 0     # How many signals passed through
        self._suppressed_count = 0  # How many were blocked
        self._pending: List[dict] = []  # Staged data waiting to relay

    def receive(self, signal: float, data: Optional[dict] = None) -> None:
        """Bottom-up or top-down excitation arrives."""
        self.activation += signal
        if data:
            self._pending.append(data)

    def relay(self) -> tuple:
        """
        Attempt to relay. Returns (passed: bool, strength: float).
        Signal passes only if activation > suppression.
        The NET signal = activation - suppression.
        """
        net = self.activation - self.suppression
        if net > 0:
            self._relay_count += 1
            strength = net
            pending = list(self._pending)
            self._pending.clear()
            self.activation = 0.0
            self.suppression = 0.0
            return True, strength, pending
        else:
            self._suppressed_count += 1
            # Pending data STAYS — it will try again next cycle
            # (horizontal regulation: retain data until ready)
            self.activation *= 0.5  # Decay but don't zero
            self.suppression *= 0.8  # Suppression also decays
            return False, 0.0, []


class TRN:
    """
    Thalamic Reticular Nucleus — the attentional spotlight.

    Controls which TCR channels relay and which are suppressed.
    The cortex tells the TRN what to attend to. The TRN then
    suppresses everything else.
    """

    def __init__(self):
        self.channels: Dict[str, TCRChannel] = {}
        self._attended: Optional[str] = None    # Currently attended channel
        self._attention_strength = 0.0           # How focused (0=broad, 1=laser)
        self._history: List[str] = []            # Attention history
        self._max_history = 20

    def ensure_channel(self, name: str) -> TCRChannel:
        if name not in self.channels:
            self.channels[name] = TCRChannel(name)
        return self.channels[name]

    def cortical_command(self, attend_to: str, strength: float = 0.7) -> None:
        """
        Cortex sends top-down command: "attend to this modality."

        This excites BOTH the attended TCR AND the TRN neurons
        over NON-ATTENDED channels. The TRN neurons then suppress
        their corresponding TCR targets.

        attend_to: which spoke to focus on
        strength: 0.0 = broad attention (all channels open)
                  1.0 = laser focus (only attended channel passes)
        """
        self._attended = attend_to
        self._attention_strength = min(max(strength, 0.0), 1.0)
        self._history.append(attend_to)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Excite the attended channel's TCR
        if attend_to in self.channels:
            self.channels[attend_to].activation += strength

        # TRN fires over NON-attended channels → feedforward inhibition
        for name, channel in self.channels.items():
            if name == attend_to:
                continue
            # Suppression proportional to attention strength
            # Strong focus = strong suppression of others
            channel.suppression += strength * 0.8

    def broad_attention(self) -> None:
        """
        Release focused attention. All channels partially open.
        Like the TRN relaxing — all modalities get through weakly.
        Used during resting state or exploration.
        """
        self._attended = None
        self._attention_strength = 0.0
        for channel in self.channels.values():
            channel.suppression *= 0.3  # Reduce but don't eliminate suppression
            channel.activation += 0.2    # Gentle tonic activation

    def gate_cycle(self) -> Dict[str, dict]:
        """
        Run one gating cycle. Each channel attempts to relay.
        Returns: {channel_name: {passed, strength, pending_data}}
        """
        results = {}
        for name, channel in self.channels.items():
            passed, strength, pending = channel.relay()
            results[name] = {
                'passed': passed,
                'strength': strength,
                'pending': pending,
                'activation': round(channel.activation, 3),
                'suppression': round(channel.suppression, 3),
            }
        return results

    def spotlight(self) -> dict:
        """What is the current attentional state?"""
        open_channels = []
        suppressed_channels = []
        for name, ch in self.channels.items():
            net = ch.activation - ch.suppression
            if net > 0:
                open_channels.append((name, round(net, 3)))
            else:
                suppressed_channels.append((name, round(net, 3)))

        return {
            'attended': self._attended,
            'strength': round(self._attention_strength, 3),
            'open': open_channels,
            'suppressed': suppressed_channels,
            'mode': 'focused' if self._attention_strength > 0.5 else 'broad',
        }

    def stats(self) -> dict:
        channel_stats = {}
        for name, ch in self.channels.items():
            channel_stats[name] = {
                'relayed': ch._relay_count,
                'suppressed': ch._suppressed_count,
                'pending': len(ch._pending),
            }
        return {
            'attended': self._attended,
            'strength': round(self._attention_strength, 3),
            'channels': channel_stats,
            'recent_history': self._history[-5:],
        }

    def to_dict(self) -> dict:
        return {
            'attended': self._attended,
            'strength': self._attention_strength,
            'history': self._history[-10:],
            'channels': {
                name: {
                    'relayed': ch._relay_count,
                    'suppressed': ch._suppressed_count,
                }
                for name, ch in self.channels.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> TRN:
        trn = cls()
        trn._attended = d.get('attended')
        trn._attention_strength = d.get('strength', 0.0)
        trn._history = d.get('history', [])
        for name, ch_data in d.get('channels', {}).items():
            ch = trn.ensure_channel(name)
            ch._relay_count = ch_data.get('relayed', 0)
            ch._suppressed_count = ch_data.get('suppressed', 0)
        return trn
