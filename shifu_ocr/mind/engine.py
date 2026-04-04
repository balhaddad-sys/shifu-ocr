"""
SHIFU v2 — The Embryo Engine

No dimensions. No channels. No routing weights. No vectors.
Raw observations only. Structure emerges from exposure.

feed() is the perturbation. Nodes hold what happened, what they touched,
and what they are. compare() measures directly from whatever evidence
exists. Weights grow with evidence — a word seen once gets surface-level
comparison. A word seen 1000 times gets deep structural comparison.

The engine doesn't pretend to know what it hasn't observed.
"""

import re
import json
import math
from typing import Dict, List, Optional, Any, Set

VERSION = "2.0.0"

# OCR topology: physics of ink on paper. Not learned — it's the genome.
OCR_TOPOLOGY = {
    "0,o": 0.1, "1,l": 0.2, "1,i": 0.2, "5,s": 0.3, "8,b": 0.3,
    "6,g": 0.4, "l,i": 0.2, "m,n": 0.4, "u,v": 0.5, "c,e": 0.5,
    "r,n": 0.3, "d,o": 0.3, "f,t": 0.4, "h,b": 0.4, "a,e": 0.4,
    "a,o": 0.4, "u,n": 0.4, "e,i": 0.4, "f,l": 0.4, "s,e": 0.5,
    "b,d": 0.4,
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _mean(a: List[float]) -> float:
    return sum(a) / len(a) if a else 0.0

def _sd(a: List[float]) -> float:
    if len(a) < 2:
        return 0.0
    m = _mean(a)
    return math.sqrt(sum((v - m) ** 2 for v in a) / len(a))

def _tokenize(raw: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", raw.lower()) if len(w) > 1]

def _ocr_dist(a: str, b: str) -> float:
    """Topology-weighted Levenshtein using OCR confusion costs."""
    a, b = a.lower(), b.lower()
    if len(a) < len(b):
        return _ocr_dist(b, a)
    if not b:
        return float(len(a))
    prev = list(range(len(b) + 1))
    for i in range(len(a)):
        curr = [i + 1]
        for j in range(len(b)):
            key = ",".join(sorted([a[i], b[j]]))
            sub = 0 if a[i] == b[j] else OCR_TOPOLOGY.get(key, 1.0)
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + sub))
        prev = curr
    return prev[len(b)]

def _lev_dist(a: str, b: str) -> float:
    a, b = a.lower(), b.lower()
    if len(a) < len(b):
        return _lev_dist(b, a)
    if not b:
        return float(len(a))
    prev = list(range(len(b) + 1))
    for i in range(len(a)):
        curr = [i + 1]
        for j in range(len(b)):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                            prev[j] + (0 if a[i] == b[j] else 1)))
        prev = curr
    return prev[len(b)]

def _shared_bigrams(a: str, b: str) -> float:
    """Dice coefficient of character bigrams."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    bg_a = set(a[i:i+2] for i in range(len(a) - 1))
    bg_b = set(b[i:i+2] for i in range(len(b) - 1))
    if not bg_a or not bg_b:
        return 0.0
    return 2 * len(bg_a & bg_b) / (len(bg_a) + len(bg_b))


# ---------------------------------------------------------------------------
# Node: a word's raw observation record
# ---------------------------------------------------------------------------

def _new_node(word: str) -> Dict[str, Any]:
    return {
        "chars": word,
        "freq": 0,
        "first_seen": None,
        "last_seen": None,
        "positions": [],
        "gaps": [],
        "sent_lengths": [],
        "neighbors": {},
        "next": {},
        "prev": {},
        "next2": {},
    }


# ---------------------------------------------------------------------------
# The embryo
# ---------------------------------------------------------------------------

ShifuEngine = None  # alias set below

class ShifuEmbryo:
    """One cell. No dimensions yet. Structure emerges from exposure."""

    def __init__(self) -> None:
        self.version: str = VERSION
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.sentence_count: int = 0
        self.token_count: int = 0
        self._idx_len: Dict[int, List[str]] = {}
        self._idx_bg: Dict[str, List[str]] = {}
        self._indexed: bool = True

    def _reindex(self, w: str) -> None:
        n = len(w)
        self._idx_len.setdefault(n, [])
        if w not in self._idx_len[n]:
            self._idx_len[n].append(w)
        for i in range(len(w) - 1):
            bg = w[i:i+2]
            self._idx_bg.setdefault(bg, [])
            if w not in self._idx_bg[bg]:
                self._idx_bg[bg].append(w)

    def _ensure_indexed(self) -> None:
        """Build indexes lazily — only needed for correction."""
        if self._indexed:
            return
        for w in self.nodes:
            self._reindex(w)
        self._indexed = True

    def _candidates(self, garbled: str, radius: int = 2) -> List[str]:
        self._ensure_indexed()
        g = garbled.lower()
        cands: Set[str] = set()
        for d in range(-radius, radius + 1):
            for w in self._idx_len.get(len(g) + d, []):
                cands.add(w)
        for i in range(len(g) - 1):
            for w in self._idx_bg.get(g[i:i+2], []):
                cands.add(w)
        return list(cands)

    # -------------------------------------------------------------------
    # feed() — the perturbation
    # -------------------------------------------------------------------

    def feed(self, raw: str) -> int:
        """Text hits the stone. Each word records what happened."""
        words = _tokenize(raw)
        if len(words) < 2:
            return 0

        self.sentence_count += 1
        sent_len = len(words)

        for i, w in enumerate(words):
            if w not in self.nodes:
                self.nodes[w] = _new_node(w)
                self._reindex(w)
            node = self.nodes[w]
            rel_pos = i / max(sent_len - 1, 1)

            self.token_count += 1
            node["freq"] += 1
            if node["first_seen"] is None:
                node["first_seen"] = self.sentence_count
            if node["last_seen"] is not None:
                node["gaps"].append(self.sentence_count - node["last_seen"])
                if len(node["gaps"]) > 100:
                    node["gaps"] = node["gaps"][-100:]
            node["last_seen"] = self.sentence_count
            node["positions"].append(rel_pos)
            if len(node["positions"]) > 200:
                node["positions"] = node["positions"][-200:]
            node["sent_lengths"].append(sent_len)
            if len(node["sent_lengths"]) > 100:
                node["sent_lengths"] = node["sent_lengths"][-100:]

            # Co-occurrence within window of 3
            for j in range(max(0, i - 3), min(sent_len, i + 4)):
                if j != i:
                    nb = words[j]
                    node["neighbors"][nb] = node["neighbors"].get(nb, 0) + 1

            # Directional grooves
            if i < sent_len - 1:
                nxt = words[i + 1]
                node["next"][nxt] = node["next"].get(nxt, 0) + 1
            if i > 0:
                prv = words[i - 1]
                node["prev"][prv] = node["prev"].get(prv, 0) + 1
            if i < sent_len - 2:
                b, c = words[i + 1], words[i + 2]
                node["next2"].setdefault(b, {})[c] = (
                    node["next2"].get(b, {}).get(c, 0) + 1
                )

        return len(words)

    def feed_text(self, text: str) -> Dict[str, int]:
        sentences = [s.strip() for s in re.split(r"[.!?\n]+", text) if len(s.strip()) > 5]
        tokens = sum(self.feed(s) for s in sentences)
        return {"sentences": len(sentences), "tokens": tokens}

    # -------------------------------------------------------------------
    # depth() — how much evidence supports this word
    # -------------------------------------------------------------------

    def depth(self, word: str) -> Dict[str, Any]:
        """How much the engine knows about this word."""
        w = word.lower()
        n = self.nodes.get(w)
        if not n:
            return {"level": "unborn", "evidence": 0.0}
        if n["freq"] == 1:
            return {"level": "surface", "evidence": 0.1}
        if n["freq"] < 5:
            return {"level": "shallow", "evidence": 0.2}

        nbr_count = len(n["neighbors"])
        seq_count = len(n["next"]) + len(n["prev"])
        evidence = min(
            (n["freq"] / 50) * 0.3 +
            (nbr_count / 20) * 0.3 +
            (seq_count / 10) * 0.2 +
            (len(n["positions"]) / 50) * 0.2,
            1.0
        )
        if evidence < 0.3:
            return {"level": "forming", "evidence": evidence}
        if evidence < 0.7:
            return {"level": "structured", "evidence": evidence}
        return {"level": "deep", "evidence": evidence}

    # -------------------------------------------------------------------
    # compare() — direct measurement, not cosine
    # -------------------------------------------------------------------

    def compare(self, a: str, b: str) -> Dict[str, Any]:
        """Compare two nodes. Weights grow with evidence.

        No fixed routing. Each signal contributes proportional to how
        much evidence supports it. Early in life: mostly character-level.
        After exposure: neighborhood + trajectory dominate.
        """
        a, b = a.lower(), b.lower()
        na = self.nodes.get(a)
        nb = self.nodes.get(b)
        signals: Dict[str, float] = {}
        total_weight = 0.0

        # Character-level: always available (the genome)
        signals["edit_sim"] = 1 - _lev_dist(a, b) / max(len(a), len(b), 1)
        signals["bigram_sim"] = _shared_bigrams(a, b)
        signals["ocr_sim"] = 1 - _ocr_dist(a, b) / max(len(a), len(b), 1)
        char_weight = 0.05
        total_weight += char_weight

        if not na or not nb:
            sim = signals["edit_sim"] * 0.3 + signals["bigram_sim"] * 0.3 + signals["ocr_sim"] * 0.4
            return {"similarity": sim, "signals": signals, "total_weight": total_weight, "depth": "surface"}

        # Neighborhood overlap: grows with co-occurrence evidence
        nbrs_a = list(na["neighbors"].keys())
        nbrs_b_set = set(nb["neighbors"].keys())
        shared = [x for x in nbrs_a if x in nbrs_b_set]
        union = set(nbrs_a) | nbrs_b_set
        if union:
            wt_fn = lambda w: 1 / max(math.log2(self.nodes.get(w, {}).get("freq", 0) + 1), 1)
            shared_w = sum(wt_fn(w) for w in shared)
            union_w = sum(wt_fn(w) for w in union)
            signals["neighbor_overlap"] = shared_w / union_w if union_w > 0 else 0
            nbr_weight = min(len(union) / 10, 0.35)
            total_weight += nbr_weight

        # Sequential relationship: grows with directional evidence
        tot_a = sum(na["next"].values()) if na["next"] else 0
        tot_b = sum(nb["next"].values()) if nb["next"] else 0
        if tot_a > 0 or tot_b > 0:
            signals["expects_ab"] = na["next"].get(b, 0) / max(tot_a, 1) if tot_a else 0
            signals["expects_ba"] = nb["next"].get(a, 0) / max(tot_b, 1) if tot_b else 0
            signals["directional"] = abs(signals["expects_ab"] - signals["expects_ba"])
            seq_weight = min((tot_a + tot_b) / 20, 0.25)
            total_weight += seq_weight

        # Positional similarity: grows with position observations
        if len(na["positions"]) >= 3 and len(nb["positions"]) >= 3:
            mean_a, mean_b = _mean(na["positions"]), _mean(nb["positions"])
            sd_a, sd_b = _sd(na["positions"]), _sd(nb["positions"])
            signals["positional_sim"] = 1 - min(abs(mean_a - mean_b) * 2, 1)
            signals["positional_spread_sim"] = 1 - min(abs(sd_a - sd_b) * 3, 1)
            pos_weight = min(min(len(na["positions"]), len(nb["positions"])) / 20, 0.15)
            total_weight += pos_weight

        # Frequency similarity
        if na["freq"] >= 2 and nb["freq"] >= 2:
            max_freq = max(na["freq"], nb["freq"])
            signals["freq_sim"] = 1 - abs(na["freq"] - nb["freq"]) / max_freq
            freq_weight = 0.05
            total_weight += freq_weight

        # Trajectory depth: second-order evidence
        nx2_ab = na["next2"].get(b)
        nx2_ba = nb["next2"].get(a)
        nx2_ab_n = len(nx2_ab) if nx2_ab else 0
        nx2_ba_n = len(nx2_ba) if nx2_ba else 0
        if nx2_ab_n > 0 or nx2_ba_n > 0:
            signals["trajectory_ab"] = min(nx2_ab_n / 5, 1)
            signals["trajectory_ba"] = min(nx2_ba_n / 5, 1)
            traj_weight = 0.15
            total_weight += traj_weight

        # Indirect paths: a -> ? -> b
        if len(na["next"]) >= 2:
            indirect = sum(1 for mid in na["next"] if b in self.nodes.get(mid, {}).get("next", {}))
            signals["indirect_ab"] = min(indirect / 5, 1)
            ind_weight = min(indirect / 10, 0.15)
            total_weight += ind_weight

        # Weighted combination: each signal proportional to evidence
        sim = 0.0
        if total_weight > 0:
            sim += (signals.get("edit_sim", 0) * 0.3 + signals.get("bigram_sim", 0) * 0.3 + signals.get("ocr_sim", 0) * 0.4) * char_weight
            if "neighbor_overlap" in signals:
                sim += signals["neighbor_overlap"] * min(len(union) / 10, 0.35)
            if "expects_ab" in signals:
                sim += max(signals.get("expects_ab", 0), signals.get("expects_ba", 0)) * min((tot_a + tot_b) / 20, 0.25)
            if "positional_sim" in signals:
                sim += signals["positional_sim"] * min(min(len(na["positions"]), len(nb["positions"])) / 20, 0.15)
            if "freq_sim" in signals:
                sim += signals["freq_sim"] * 0.05
            if "trajectory_ab" in signals:
                sim += max(signals["trajectory_ab"], signals.get("trajectory_ba", 0)) * 0.15
            if "indirect_ab" in signals:
                sim += signals["indirect_ab"] * min(signals["indirect_ab"] * 2 / 10, 0.15)
            sim /= total_weight

        lvl = "surface" if total_weight < 0.1 else "shallow" if total_weight < 0.3 else "forming" if total_weight < 0.6 else "deep"
        return {"similarity": sim, "signals": signals, "total_weight": total_weight, "depth": lvl}

    # -------------------------------------------------------------------
    # affinity() — pre-contact attraction, evidence-proportional
    # -------------------------------------------------------------------

    def affinity(self, a: str, b: str) -> Dict[str, Any]:
        """Pre-contact attraction. ASYMMETRIC. Weights grow with evidence."""
        a, b = a.lower(), b.lower()
        na, nb = self.nodes.get(a), self.nodes.get(b)
        if not na or not nb:
            return {"a": a, "b": b, "mutual": 0.0, "known": False}

        total_weight = 0.0
        weighted_ab = 0.0
        weighted_ba = 0.0

        # 1. Shared orbit — neighborhood Jaccard, rare-weighted
        nbrs_a = list(na["neighbors"].keys())
        nbrs_b_set = set(nb["neighbors"].keys())
        shared = [x for x in nbrs_a if x in nbrs_b_set]
        all_nbs = set(nbrs_a) | nbrs_b_set
        wt_fn = lambda w: 1 / max(math.log2(self.nodes.get(w, {}).get("freq", 0) + 1), 1)
        if all_nbs:
            sw = sum(wt_fn(w) for w in shared)
            uw = sum(wt_fn(w) for w in all_nbs)
            orbit = sw / uw if uw > 0 else 0
        else:
            orbit = 0.0
        orbit_weight = min(len(all_nbs) / 10, 0.40)
        total_weight += orbit_weight
        weighted_ab += orbit * orbit_weight
        weighted_ba += orbit * orbit_weight

        # 2. Trajectory pull — directional, asymmetric
        tot_a = sum(na["next"].values()) if na["next"] else 0
        tot_b = sum(nb["next"].values()) if nb["next"] else 0
        pull_ab = na["next"].get(b, 0) / tot_a if tot_a else 0
        pull_ba = nb["next"].get(a, 0) / tot_b if tot_b else 0
        pull_weight = min((tot_a + tot_b) / 20, 0.25)
        total_weight += pull_weight
        weighted_ab += pull_ab * pull_weight
        weighted_ba += pull_ba * pull_weight

        # 3. Indirect paths — a -> ? -> b
        ind_ab = sum(1 for mid in na["next"] if b in self.nodes.get(mid, {}).get("next", {}))
        ind_ba = sum(1 for mid in nb["next"] if a in self.nodes.get(mid, {}).get("next", {}))
        ind_ab_s = min(ind_ab / 5, 1)
        ind_ba_s = min(ind_ba / 5, 1)
        ind_weight = min(max(ind_ab, ind_ba) / 5, 0.20)
        total_weight += ind_weight
        weighted_ab += ind_ab_s * ind_weight
        weighted_ba += ind_ba_s * ind_weight

        # 4. Expectation overlap — do they predict similar futures?
        fwd_a = set(na["next"].keys())
        fwd_b = set(nb["next"].keys())
        fwd_union = fwd_a | fwd_b
        exp_overlap = len(fwd_a & fwd_b) / len(fwd_union) if fwd_union else 0
        exp_weight = min(len(fwd_union) / 10, 0.15)
        total_weight += exp_weight
        weighted_ab += exp_overlap * exp_weight
        weighted_ba += exp_overlap * exp_weight

        # 5. Character similarity — the genome (always available, weakest)
        char_sim = 1 - _lev_dist(a, b) / max(len(a), len(b), 1)
        char_weight = 0.05
        total_weight += char_weight
        weighted_ab += char_sim * char_weight
        weighted_ba += char_sim * char_weight

        # 6. Positional alignment — grows with position data
        pos_align = 0.0
        if len(na["positions"]) >= 3 and len(nb["positions"]) >= 3:
            pos_align = 1 - min(abs(_mean(na["positions"]) - _mean(nb["positions"])) * 3, 1)
        pos_weight = min(min(len(na["positions"]), len(nb["positions"])) / 30, 0.10)
        total_weight += pos_weight
        weighted_ab += pos_align * pos_weight
        weighted_ba += pos_align * pos_weight

        af_ab = weighted_ab / total_weight if total_weight > 0 else 0
        af_ba = weighted_ba / total_weight if total_weight > 0 else 0

        return {
            "a": a, "b": b, "known": True,
            "orbit": orbit, "pull_ab": pull_ab, "pull_ba": pull_ba,
            "indirect_ab": ind_ab_s, "indirect_ba": ind_ba_s,
            "char_sim": char_sim, "exp_overlap": exp_overlap, "pos_align": pos_align,
            "affinity_ab": af_ab, "affinity_ba": af_ba,
            "mutual": (af_ab + af_ba) / 2,
            "asymmetry": abs(af_ab - af_ba),
            "total_weight": total_weight,
        }

    # -------------------------------------------------------------------
    # score_sentence() — walking the graph
    # -------------------------------------------------------------------

    def score_sentence(self, raw: str) -> Dict[str, Any]:
        """Walk the sentence. Accumulate surprise from whatever evidence exists."""
        words = _tokenize(raw)
        if len(words) < 2:
            return {"words": words, "steps": [], "mean_surprise": 0, "coherence": 0}

        steps = []
        total_surprise = 0.0
        field: Dict[str, float] = {}

        for i, w in enumerate(words):
            node = self.nodes.get(w)
            step: Dict[str, Any] = {"word": w, "position": i, "known": node is not None}
            signals = 0.0
            weights = 0.0

            # Sequential: was this expected after previous?
            if i > 0:
                prev_node = self.nodes.get(words[i - 1])
                if prev_node and prev_node["next"]:
                    tot = sum(prev_node["next"].values())
                    step["seq_s"] = 1 - prev_node["next"].get(w, 0) / max(tot, 1)
                    signals += step["seq_s"] * 0.35
                    weights += 0.35

            # Trajectory: second-order
            if i >= 2:
                pp_node = self.nodes.get(words[i - 2])
                if pp_node:
                    nx2 = pp_node["next2"].get(words[i - 1])
                    if nx2:
                        tot = sum(nx2.values())
                        step["traj_s"] = 1 - nx2.get(w, 0) / max(tot, 1)
                        signals += step["traj_s"] * 0.30
                        weights += 0.30

            # Field: do all preceding words collectively expect this one?
            if i > 0 and field:
                fw = field.get(w, 0)
                max_f = max(field.values()) if field else 1
                step["field_s"] = 1 - fw / max(max_f, 1)
                signals += step["field_s"] * 0.35
                weights += 0.35

            # Affinity gate
            step["af_gate"] = 0.0
            if i > 0 and node:
                prev_node = self.nodes.get(words[i - 1])
                if prev_node and prev_node["neighbors"] and node["neighbors"]:
                    p_n = set(prev_node["neighbors"].keys())
                    w_n = set(node["neighbors"].keys())
                    union = p_n | w_n
                    step["af_gate"] = len(p_n & w_n) / len(union) if union else 0

            step["surprise"] = (
                (signals / weights) * (1 - step["af_gate"] * 0.3)
                if weights > 0
                else (0.5 if node else 1.0)
            )
            total_surprise += step["surprise"]
            step["cumulative_surprise"] = total_surprise
            steps.append(step)

            # Update field: weight by specificity (rare neighbors are more informative)
            if node and node["neighbors"]:
                total_nbr = sum(node["neighbors"].values())
                for nb, cnt in node["neighbors"].items():
                    # Specificity: how concentrated is this neighbor for this word?
                    spec = cnt / total_nbr if total_nbr > 0 else 0
                    nb_freq = self.nodes.get(nb, {}).get("freq", 1)
                    # Downweight ubiquitous words: 1/log(freq) makes rare neighbors count more
                    rarity = 1 / max(math.log2(nb_freq + 1), 1)
                    field[nb] = field.get(nb, 0) + spec * rarity
            field[w] = field.get(w, 0) + 1.0

        mean_s = total_surprise / len(steps) if steps else 0
        return {"words": words, "steps": steps, "mean_surprise": mean_s,
                "coherence": 1 - min(mean_s, 1)}

    # -------------------------------------------------------------------
    # generate() — follow the grooves
    # -------------------------------------------------------------------

    def generate(self, seeds: List[str], max_len: int = 30,
                 temperature: float = 0.7) -> Dict[str, Any]:
        """Walk forward from seed words by following the stone's grooves.

        No hardcoded word lists. No function word filters. No PMI tables.
        Just: at each step, what does the engine expect next?
        Candidates scored by: sequential expectation, trajectory,
        field support, and rarity (prefer content over function words).
        Temperature controls exploration: 0 = greedy, 1 = proportional.
        """
        words = [w.lower() for w in seeds if w.lower() in self.nodes]
        if not words:
            return {"text": "", "words": [], "steps": []}

        steps = []
        field: Dict[str, float] = {}
        used: Set[str] = set(words)

        # Prime the field from seeds
        for w in words:
            node = self.nodes.get(w)
            if node and node["neighbors"]:
                total_nbr = sum(node["neighbors"].values())
                for nb, cnt in node["neighbors"].items():
                    spec = cnt / total_nbr if total_nbr > 0 else 0
                    nb_freq = self.nodes.get(nb, {}).get("freq", 1)
                    rarity = 1 / max(math.log2(nb_freq + 1), 1)
                    field[nb] = field.get(nb, 0) + spec * rarity
            field[w] = field.get(w, 0) + 1.0

        for step_i in range(max_len):
            last = words[-1]
            prev = words[-2] if len(words) >= 2 else None
            node = self.nodes.get(last)
            if not node or not node["next"]:
                break

            # Score candidates. The key: not "how often does X follow Y"
            # but "how much MORE often does X follow Y than X follows anything."
            # A word that follows everything (like "the") scores low.
            # A word that follows specifically THIS word scores high.
            candidates: List[tuple] = []
            nx_total = sum(node["next"].values())
            recent = set(words[-4:])  # penalize very recent words

            for cand, cnt in node["next"].items():
                if cand in recent:
                    continue  # hard block on last 4 words

                score = 0.0
                weight = 0.0

                # Specificity: P(cand|last) vs P(cand|anything)
                # How much more likely is this candidate after THIS word
                # than after a random word? High = specific connection.
                p_given_last = cnt / nx_total if nx_total > 0 else 0
                cand_node = self.nodes.get(cand)
                cand_total_prev = sum(cand_node["prev"].values()) if cand_node and cand_node["prev"] else 1
                cand_freq = cand_node["freq"] if cand_node else 1
                p_baseline = cand_freq / max(self.token_count, 1)
                # Log ratio: positive = this connection is specific
                if p_baseline > 0 and p_given_last > 0:
                    log_ratio = math.log2(p_given_last / p_baseline)
                    specificity = max(0, min(log_ratio / 5, 1))
                else:
                    specificity = 0
                score += specificity * 0.35
                weight += 0.35

                # Sequential probability (tempered by specificity)
                score += p_given_last * 0.20
                weight += 0.20

                # Trajectory: prev->last->cand seen before
                if prev:
                    nx2 = self.nodes.get(prev, {}).get("next2", {}).get(last, {})
                    if nx2:
                        nx2_total = sum(nx2.values())
                        traj = nx2.get(cand, 0) / nx2_total if nx2_total > 0 else 0
                        score += traj * 0.25
                        weight += 0.25

                # Field support: does accumulated context expect this word?
                field_val = field.get(cand, 0)
                max_field = max(field.values()) if field else 1e-6
                field_sup = field_val / max(max_field, 1e-6)
                score += field_sup * 0.20
                weight += 0.20

                final = score / weight if weight > 0 else 0
                candidates.append((final, cand))

            if not candidates:
                break

            candidates.sort(key=lambda x: -x[0])

            # Temperature-based selection
            if temperature <= 0.01:
                chosen = candidates[0][1]
            else:
                # Softmax with temperature over top candidates
                top_n = min(len(candidates), 8)
                top_cands = candidates[:top_n]
                scores = [c[0] / temperature for c in top_cands]
                max_s = max(scores)
                exps = [math.exp(s - max_s) for s in scores]
                total_exp = sum(exps)
                probs = [e / total_exp for e in exps]

                # Weighted random choice (deterministic via cumulative)
                # Use step index as seed for reproducibility
                r = ((step_i * 16807 + hash(last)) % 2147483647) / 2147483647
                cumulative = 0
                chosen = top_cands[-1][1]
                for prob, (_, cand) in zip(probs, top_cands):
                    cumulative += prob
                    if r < cumulative:
                        chosen = cand
                        break

            words.append(chosen)
            used.add(chosen)
            steps.append({"word": chosen, "position": step_i,
                          "candidates": len(candidates),
                          "top_score": candidates[0][0]})

            # Update field with chosen word's neighborhood
            ch_node = self.nodes.get(chosen)
            if ch_node and ch_node["neighbors"]:
                total_nbr = sum(ch_node["neighbors"].values())
                for nb, cnt in ch_node["neighbors"].items():
                    spec = cnt / total_nbr if total_nbr > 0 else 0
                    nb_freq = self.nodes.get(nb, {}).get("freq", 1)
                    rarity_w = 1 / max(math.log2(nb_freq + 1), 1)
                    field[nb] = field.get(nb, 0) + spec * rarity_w
            field[chosen] = field.get(chosen, 0) + 1.0

        return {"text": " ".join(words), "words": words, "steps": steps}

    # -------------------------------------------------------------------
    # correct() — OCR topology is physics, not learning
    # -------------------------------------------------------------------

    def correct(self, garbled: str, k: int = 5) -> Dict[str, Any]:
        """Correct garbled text. OCR topology (physics) + evidence from exposure.

        The blend is evidence-proportional: if the engine has seen many words,
        frequency evidence gets more weight. If the engine is empty, pure
        OCR physics decides. No fixed routing — evidence grows with exposure.
        """
        cands = self._candidates(garbled)
        # How much does the engine know? More tokens = more trust in frequency
        freq_weight = min(self.token_count / 5000, 0.3)  # grows to 0.3 max
        phys_weight = 1.0 - freq_weight

        max_freq = max((self.nodes.get(w, {}).get("freq", 0) for w in cands), default=1)

        scored = []
        for w in cands:
            ocr_sim = 1 - _ocr_dist(garbled, w) / max(len(garbled), len(w), 1)
            bg_sim = _shared_bigrams(garbled, w)
            physics = ocr_sim * 0.7 + bg_sim * 0.3

            # Evidence: how well-known is this candidate?
            freq = self.nodes.get(w, {}).get("freq", 0)
            evidence = math.log2(freq + 1) / math.log2(max_freq + 1) if max_freq > 0 else 0

            score = physics * phys_weight + evidence * freq_weight
            scored.append({"word": w, "score": score, "ocr_sim": ocr_sim,
                           "bigram_sim": bg_sim, "evidence": evidence})

        scored.sort(key=lambda x: -x["score"])
        top = scored[:k]
        conf = top[0]["score"] - top[1]["score"] if len(top) >= 2 else (1.0 if top else 0.0)
        return {"candidates": top, "confidence": conf}

    def similar(self, word: str, k: int = 10) -> List[Dict[str, Any]]:
        """Find most similar words by direct comparison."""
        key = word.lower()
        out = []
        for w in self.nodes:
            if w == key:
                continue
            cmp = self.compare(word, w)
            cmp["word"] = w
            out.append(cmp)
        out.sort(key=lambda x: -x["similarity"])
        return out[:k]

    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # pressure() — the graph knows its own shape
    # -------------------------------------------------------------------

    def pressure(self) -> List[Dict[str, Any]]:
        """Where does the graph pull (vacuum) and push (surplus)?

        Negative pressure = vacuum: the graph predicts richness that
        doesn't exist yet. It PULLS — this word needs more data.
        Positive pressure = surplus: richer than surroundings expect.
        It PUSHES — this word is a hub.
        """
        result = []
        for word, node in self.nodes.items():
            # Inbound: how many times other words point to this one
            inbound = 0
            for other in self.nodes.values():
                inbound += other["next"].get(word, 0)

            actual = (len(node["neighbors"]) + len(node["next"]) + len(node["prev"]))
            p = actual - inbound

            # Closure: how connected are this word's neighbors to each other?
            nbrs = list(node["neighbors"].keys())
            internal = 0
            pairs = 0
            sample = nbrs[:15]
            for i in range(len(sample)):
                for j in range(i + 1, len(sample)):
                    pairs += 1
                    if sample[j] in self.nodes.get(sample[i], {}).get("neighbors", {}):
                        internal += 1
            closure = internal / pairs if pairs > 0 else 1.0

            result.append({
                "word": word, "pressure": p, "inbound": inbound,
                "actual": actual, "closure": closure,
                "freq": node["freq"], "depth": self.depth(word)["level"],
            })

        result.sort(key=lambda x: x["pressure"])
        return result

    def vacuums(self, k: int = 10) -> List[Dict[str, Any]]:
        """Words with negative pressure — where the graph needs more data."""
        return [p for p in self.pressure() if p["pressure"] < 0][:k]

    def surpluses(self, k: int = 10) -> List[Dict[str, Any]]:
        """Words with positive pressure — hubs that push outward."""
        return sorted([p for p in self.pressure() if p["pressure"] > 0],
                       key=lambda x: -x["pressure"])[:k]

    def bridges(self, k: int = 10) -> List[Dict[str, Any]]:
        """Words with low closure — connecting disconnected neighborhoods."""
        return [p for p in self.pressure() if p["closure"] < 0.3 and p["freq"] >= 3][:k]

    # -------------------------------------------------------------------
    # Active forgetting
    # -------------------------------------------------------------------

    def unlearn(self, a: str, b: str) -> Dict[str, Any]:
        """Weaken all edges between two words. Call twice to quarter."""
        al, bl = a.lower(), b.lower()
        na, nb = self.nodes.get(al), self.nodes.get(bl)
        if not na or not nb:
            return {"changed": False, "reason": "unknown word"}

        removed = 0
        for src, tgt in [(na, bl), (nb, al)]:
            for table in ["neighbors", "next", "prev"]:
                if tgt in src[table]:
                    src[table][tgt] = src[table][tgt] // 2
                    if src[table][tgt] <= 0:
                        del src[table][tgt]
                    removed += 1

        # Remove second-order paths through each other
        for src, tgt in [(na, bl), (nb, al)]:
            if tgt in src["next2"]:
                del src["next2"][tgt]
            for mid in list(src["next2"].keys()):
                if tgt in src["next2"].get(mid, {}):
                    del src["next2"][mid][tgt]
                    if not src["next2"][mid]:
                        del src["next2"][mid]

        return {"changed": removed > 0, "edges_weakened": removed, "a": al, "b": bl}

    def forget(self, word: str) -> Dict[str, Any]:
        """Complete removal. As if the word was never seen."""
        w = word.lower()
        if w not in self.nodes:
            return {"changed": False, "reason": "unknown word"}

        for other in self.nodes.values():
            for table in ["neighbors", "next", "prev"]:
                other[table].pop(w, None)
            other["next2"].pop(w, None)
            for mid in list(other["next2"].keys()):
                if w in other["next2"].get(mid, {}):
                    del other["next2"][mid][w]
                    if not other["next2"][mid]:
                        del other["next2"][mid]

        del self.nodes[w]
        return {"changed": True, "word": w}

    def decay(self, threshold: int = 1) -> Dict[str, int]:
        """Passive forgetting. Remove all edges at or below threshold."""
        removed = 0
        for node in self.nodes.values():
            for table in ["neighbors", "next", "prev"]:
                to_del = [k for k, v in node[table].items() if v <= threshold]
                for k in to_del:
                    del node[table][k]
                    removed += 1
            for mid in list(node["next2"].keys()):
                to_del = [k for k, v in node["next2"][mid].items() if v <= threshold]
                for k in to_del:
                    del node["next2"][mid][k]
                    removed += 1
                if not node["next2"][mid]:
                    del node["next2"][mid]
        return {"edges_removed": removed}

    # -------------------------------------------------------------------
    # Stats and serialization
    # -------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        vocab = list(self.nodes.keys())
        return {
            "version": self.version,
            "sentences": self.sentence_count,
            "tokens": self.token_count,
            "vocabulary": len(vocab),
            "mature": sum(1 for w in vocab if self.nodes[w]["freq"] >= 10),
        }

    def serialize(self) -> str:
        data = {
            "version": self.version,
            "nodes": {
                w: {
                    "freq": n["freq"],
                    "first_seen": n["first_seen"],
                    "last_seen": n["last_seen"],
                    "positions": n["positions"][-200:],
                    "gaps": n["gaps"][-100:],
                    "sent_lengths": n["sent_lengths"][-100:],
                    "neighbors": n["neighbors"],
                    "next": n["next"],
                    "prev": n["prev"],
                    "next2": n["next2"],
                }
                for w, n in self.nodes.items()
            },
            "sentence_count": self.sentence_count,
            "token_count": self.token_count,
        }
        return json.dumps(data)

    @classmethod
    def deserialize(cls, raw: str) -> "ShifuEngine":
        d = json.loads(raw)
        e = cls()
        e.sentence_count = d.get("sentence_count", 0)
        e.token_count = d.get("token_count", 0)
        for w, nd in d.get("nodes", {}).items():
            e.nodes[w] = {
                "chars": w,
                "freq": nd["freq"],
                "first_seen": nd.get("first_seen"),
                "last_seen": nd.get("last_seen"),
                "positions": nd.get("positions", []),
                "gaps": nd.get("gaps", []),
                "sent_lengths": nd.get("sent_lengths", []),
                "neighbors": nd.get("neighbors", {}),
                "next": nd.get("next", {}),
                "prev": nd.get("prev", {}),
                "next2": nd.get("next2", {}),
            }
        # Lazy reindexing: build indexes only when correction is needed
        e._indexed = False
        return e


# Alias for backward compatibility
ShifuEngine = ShifuEmbryo
