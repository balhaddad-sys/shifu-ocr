"""
SHIFU CORE ENGINE v1.4.1 — Python port

A stone etched by exposure.

The baby state is Form: 16 dimensions of structure before any data.
The stone before etching. Every word has shape before it has meaning.

feed() is the perturbation. Text hits the stone. Each sentence etches:
co-occurrence neighborhoods deepen, positional patterns wear grooves,
sequential expectations form channels, the self-model baseline shifts.

The 7 channels are faces of the same stone showing different marks
from the same perturbation:

    Form (16D)        -- the stone before etching
    Context (12D)     -- which atoms displaced together
    History (8D)      -- how deep the groove, how recently cut
    Influence (8D)    -- how the etching changed the surrounding surface
    Contrast (8D)     -- how this groove differs from the stone's average
    Expectation (8D)  -- which direction the groove runs
    Affinity          -- which unetched regions will accept the chisel easiest

60-dimensional vectors. 7 channels. No neural networks. No embeddings.
Pure deterministic feature construction from text exposure.
"""

import re
import json
import math
from typing import Dict, List, Tuple, Optional, Any, Set

VERSION = "1.4.1"

VOWELS = set("aeiou")

# Channel dimension slices
IDX = {
    "form": (0, 16),     # 16 dimensions
    "ctx":  (16, 28),    # 12 dimensions
    "hist": (28, 36),    # 8 dimensions
    "inf":  (36, 44),    # 8 dimensions
    "con":  (44, 52),    # 8 dimensions
    "exp":  (52, 60),    # 8 dimensions
    "all":  (0, 60),     # full vector
}

# Routing weights: how much each channel contributes to a purpose
ROUTING = {
    "correction": {"ocr": 0.7, "form": 0.3, "context": 0, "history": 0,
                   "influence": 0, "contrast": 0, "expectation": 0},
    "meaning":    {"form": 0.20, "context": 0.20, "history": 0.10,
                   "influence": 0.15, "contrast": 0.15, "expectation": 0.20, "ocr": 0},
}

# OCR topology: visually confusable character pairs and their costs
OCR_COSTS = {
    "0,o": 0.1, "1,l": 0.2, "1,i": 0.2, "5,s": 0.3, "8,b": 0.3,
    "6,g": 0.4, "l,i": 0.2, "m,n": 0.4, "u,v": 0.5, "c,e": 0.5,
    "r,n": 0.3, "d,o": 0.3, "f,t": 0.4, "h,b": 0.4, "a,e": 0.4,
    "a,o": 0.4, "u,n": 0.4, "e,i": 0.4, "f,l": 0.4, "s,e": 0.5,
    "b,d": 0.4,
}

CONFIDENCE_REJECT = 0.02
CONFIDENCE_LOW = 0.05


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _mean(a: List[float]) -> float:
    """Mean of a list. Returns 0 for empty lists."""
    return sum(a) / len(a) if a else 0.0


def _sd(a: List[float]) -> float:
    """Standard deviation of a list."""
    if len(a) < 2:
        return 0.0
    m = _mean(a)
    return math.sqrt(sum((v - m) ** 2 for v in a) / len(a))


def _cos(a: List[float], b: List[float], lo: int, hi: int) -> float:
    """Cosine similarity between two vectors over the slice [lo, hi)."""
    dot = sum(a[i] * b[i] for i in range(lo, hi))
    na = math.sqrt(sum(a[i] ** 2 for i in range(lo, hi)))
    nb = math.sqrt(sum(b[i] ** 2 for i in range(lo, hi)))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return dot / (na * nb)


def ocr_dist(a: str, b: str) -> float:
    """Topology-weighted Levenshtein distance using OCR confusion costs."""
    a, b = a.lower(), b.lower()
    if len(a) < len(b):
        return ocr_dist(b, a)
    if not b:
        return float(len(a))
    prev = list(range(len(b) + 1))
    for i in range(len(a)):
        curr = [i + 1]
        for j in range(len(b)):
            key = ",".join(sorted([a[i], b[j]]))
            sub_cost = 0 if a[i] == b[j] else OCR_COSTS.get(key, 1.0)
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + sub_cost))
        prev = curr
    return prev[len(b)]


def lev_dist(a: str, b: str) -> float:
    """Standard Levenshtein distance."""
    a, b = a.lower(), b.lower()
    if len(a) < len(b):
        return lev_dist(b, a)
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


def _tokenize(raw: str) -> List[str]:
    """Split text into lowercase tokens of length > 1."""
    return [w for w in re.findall(r"[a-z0-9]+", raw.lower()) if len(w) > 1]


# ---------------------------------------------------------------------------
# The stone
# ---------------------------------------------------------------------------

class ShifuEngine:
    """A stone etched by exposure.

    Born with grain (config) but no etchings. feed() is the perturbation.
    Each sentence etches co-occurrence neighborhoods, positional patterns,
    sequential expectations. The stone finds steady state through exposure.
    """

    def __init__(self) -> None:
        self.version: str = VERSION

        # -- Surface marks (etched by exposure) --
        self.wf: Dict[str, int] = {}              # word frequency
        self.co: Dict[str, Dict[str, int]] = {}    # co-occurrence neighborhoods
        self.wp: Dict[str, List[float]] = {}       # word positions in sentences
        self.bf: Dict[str, int] = {}               # bigram frequency
        self.fs: Dict[str, int] = {}               # first seen (sentence number)
        self.ls: Dict[str, int] = {}               # last seen (sentence number)
        self.eg: Dict[str, List[int]] = {}         # encounter gaps between sightings
        self.ph: Dict[str, List[float]] = {}       # position history (drift tracking)
        self.nv: Dict[str, List[float]] = {}       # neighbor volatility
        self._pn: Dict[str, Set[str]] = {}         # previous neighbors snapshot

        self.sl: Dict[str, List[int]] = {}         # sentence lengths per word
        self.sn: Dict[str, List[float]] = {}       # sentence novelty per word
        self.sd: Dict[str, List[float]] = {}       # sentence diversity per word

        # -- Self-model: the stone's knowledge of its own texture --
        self._global_pos: List[float] = []         # all positional observations
        self._global_sent_len: List[int] = []      # all sentence lengths
        self._global_freq_sum: int = 0             # total frequency mass
        self._global_word_count: int = 0           # vocabulary size

        # -- Directional grooves --
        self.nx: Dict[str, Dict[str, int]] = {}    # next-word expectations
        self.px: Dict[str, Dict[str, int]] = {}    # prev-word expectations
        self.nx2: Dict[str, Dict[str, Dict[str, int]]] = {}  # second-order trajectory

        # -- Bookkeeping --
        self.ns: int = 0                           # sentences fed
        self.nt: int = 0                           # tokens fed
        self._cache: Dict[str, List[float]] = {}   # vector cache (cleared on feed)
        self._idx_len: Dict[int, List[str]] = {}   # words indexed by length
        self._idx_bg: Dict[str, List[str]] = {}    # words indexed by bigram

    # -- Indexing for fast candidate lookup --

    def _reindex(self, w: str) -> None:
        """Add word to length and bigram indexes."""
        n = len(w)
        self._idx_len.setdefault(n, [])
        if w not in self._idx_len[n]:
            self._idx_len[n].append(w)
        for i in range(len(w) - 1):
            bg = w[i:i + 2]
            self._idx_bg.setdefault(bg, [])
            if w not in self._idx_bg[bg]:
                self._idx_bg[bg].append(w)

    def _candidates(self, garbled: str, radius: int = 2) -> List[str]:
        """Find correction candidates by length similarity and shared bigrams."""
        g = garbled.lower()
        cands: Set[str] = set()
        for d in range(-radius, radius + 1):
            ws = self._idx_len.get(len(g) + d, [])
            cands.update(ws)
        for i in range(len(g) - 1):
            ws = self._idx_bg.get(g[i:i + 2], [])
            cands.update(ws)
        return list(cands)

    # -----------------------------------------------------------------------
    # The perturbation: text hits the stone
    # -----------------------------------------------------------------------

    def feed(self, raw: str) -> int:
        """Feed one sentence. Returns token count.

        A sentence is a chisel stroke. Each word in the sentence:
          - deepens its own groove (wf)
          - displaces its neighbors (co)
          - records where on the surface it fell (wp, ph)
          - notes when it was last struck (fs, ls, eg)
          - observes how its neighborhood shifted (nv)
          - measures the width and novelty of this stroke (sl, sn, sd)
          - extends its directional channel (nx, px, nx2)
        """
        ws = _tokenize(raw)
        if len(ws) < 2:
            return 0

        self.ns += 1
        sent_len = len(ws)
        unique_in_sent = len(set(ws))

        # Update global sentence length tracker
        self._global_sent_len.append(sent_len)
        if len(self._global_sent_len) > 200:
            self._global_sent_len = self._global_sent_len[-200:]

        for i, w in enumerate(ws):
            # Relative position within sentence [0, 1]
            rp = i / max(sent_len - 1, 1)
            self.nt += 1

            # Word frequency
            is_new = w not in self.wf
            self.wf[w] = self.wf.get(w, 0) + 1
            self._global_freq_sum += 1
            self._global_word_count = len(self.wf)
            if is_new:
                self._reindex(w)

            # Word position
            self.wp.setdefault(w, []).append(rp)
            if len(self.wp[w]) > 100:
                self.wp[w] = self.wp[w][-100:]
            self._global_pos.append(rp)
            if len(self._global_pos) > 500:
                self._global_pos = self._global_pos[-500:]

            # Co-occurrence within window of 3
            self.co.setdefault(w, {})
            nb: Set[str] = set()
            for j in range(max(0, i - 3), min(sent_len, i + 4)):
                if j != i:
                    self.co[w][ws[j]] = self.co[w].get(ws[j], 0) + 1
                    nb.add(ws[j])

            # Bigram frequency
            for k in range(len(w) - 1):
                bg = w[k:k + 2]
                self.bf[bg] = self.bf.get(bg, 0) + 1

            # Temporal markers
            if w not in self.fs:
                self.fs[w] = self.ns
            if w in self.ls:
                self.eg.setdefault(w, []).append(self.ns - self.ls[w])
                if len(self.eg[w]) > 50:
                    self.eg[w] = self.eg[w][-50:]
            self.ls[w] = self.ns

            # Position history
            self.ph.setdefault(w, []).append(rp)
            if len(self.ph[w]) > 50:
                self.ph[w] = self.ph[w][-50:]

            # Neighbor volatility: how much did the neighborhood change?
            if w in self._pn and nb:
                prev = self._pn[w]
                union = prev | nb
                overlap = len(prev & nb)
                vol = 1 - overlap / max(len(union), 1)
                self.nv.setdefault(w, []).append(vol)
                if len(self.nv[w]) > 50:
                    self.nv[w] = self.nv[w][-50:]
            self._pn[w] = nb

            # Sentence-level features
            self.sl.setdefault(w, []).append(sent_len)
            if len(self.sl[w]) > 50:
                self.sl[w] = self.sl[w][-50:]

            mates = [x for x in ws if x != w]
            novelty = (sum(1 for m in mates if self.wf.get(m, 0) < 3) /
                       len(mates)) if mates else 0.0
            self.sn.setdefault(w, []).append(novelty)
            if len(self.sn[w]) > 50:
                self.sn[w] = self.sn[w][-50:]

            diversity = unique_in_sent / sent_len
            self.sd.setdefault(w, []).append(diversity)
            if len(self.sd[w]) > 50:
                self.sd[w] = self.sd[w][-50:]

            # Directional grooves: which way does the channel run?
            if i < sent_len - 1:
                nxt = ws[i + 1]
                self.nx.setdefault(w, {})[nxt] = self.nx.get(w, {}).get(nxt, 0) + 1
            if i > 0:
                prv = ws[i - 1]
                self.px.setdefault(w, {})[prv] = self.px.get(w, {}).get(prv, 0) + 1

            # Second-order trajectory: w -> b -> c
            if i < sent_len - 2:
                b, c = ws[i + 1], ws[i + 2]
                self.nx2.setdefault(w, {}).setdefault(b, {})[c] = (
                    self.nx2.get(w, {}).get(b, {}).get(c, 0) + 1
                )

        # Invalidate vector cache -- the stone's surface has changed
        self._cache = {}
        return len(ws)

    def feed_text(self, text: str) -> Dict[str, int]:
        """Split text on sentence boundaries and feed each sentence."""
        sentences = [s.strip() for s in re.split(r"[.!?\n]+", text) if len(s.strip()) > 5]
        tokens = sum(self.feed(s) for s in sentences)
        return {"sentences": len(sentences), "tokens": tokens}

    # -----------------------------------------------------------------------
    # Channel vectors: 7 faces of the same stone
    # -----------------------------------------------------------------------

    def form_vec(self, word: str) -> List[float]:
        """Form channel (16D): the stone before etching.

        Pure word shape. No corpus needed. The baby state.
        Features: CV alternation, consonant/vowel runs, onset/coda charge,
        gradient, vowel ratio, rhythm variance, transition ratios,
        trigram ratios, character uniqueness, normalized length.
        """
        w = word.lower()
        n = len(w)
        if n < 1:
            return [0.0] * 16

        f = [0.0] * 16
        # Consonant/vowel pattern: +1 = vowel, -1 = consonant
        ch = [1 if c in VOWELS else -1 for c in w]

        # [0] CV alternation rate
        if n >= 2:
            alts = sum(1 for i in range(1, n) if ch[i] != ch[i - 1])
            f[0] = alts / (n - 1)

        # [1] Max consonant run (normalized)
        mx, run = 0, 0
        for c in ch:
            if c < 0:
                run += 1
                mx = max(mx, run)
            else:
                run = 0
        f[1] = mx / n

        # [2] Max vowel run (normalized)
        mx, run = 0, 0
        for c in ch:
            if c > 0:
                run += 1
                mx = max(mx, run)
            else:
                run = 0
        f[2] = mx / n

        # [3] Onset charge (first character)
        f[3] = float(ch[0])

        # [4] Coda charge (last character)
        f[4] = float(ch[-1])

        # [5] Gradient: difference between last third and first third
        if n >= 3:
            t = n // 3
            f[5] = _mean(ch[2 * t:]) - _mean(ch[:t])

        # [6] Vowel ratio
        f[6] = sum(1 for c in ch if c > 0) / n

        # [7] Rhythm variance
        if n >= 2:
            mu = _mean(ch)
            f[7] = math.sqrt(sum((c - mu) ** 2 for c in ch) / n)

        # [8-11] Transition ratios: CV, VC, CC, VV
        if n >= 2:
            cv = vc = cc = vv = 0
            for i in range(n - 1):
                a_v = w[i] in VOWELS
                b_v = w[i + 1] in VOWELS
                if not a_v and b_v:
                    cv += 1
                elif a_v and not b_v:
                    vc += 1
                elif not a_v and not b_v:
                    cc += 1
                else:
                    vv += 1
            t = n - 1
            f[8] = cv / t
            f[9] = vc / t
            f[10] = cc / t
            f[11] = vv / t

        # [12-13] Trigram ratios: CVC, VCV
        if n >= 3:
            cvc_count = vcv_count = 0
            iv = [c in VOWELS for c in w]
            for i in range(n - 2):
                if not iv[i] and iv[i + 1] and not iv[i + 2]:
                    cvc_count += 1
                if iv[i] and not iv[i + 1] and iv[i + 2]:
                    vcv_count += 1
            f[12] = cvc_count / (n - 2)
            f[13] = vcv_count / (n - 2)

        # [14] Character uniqueness
        f[14] = len(set(w)) / n

        # [15] Normalized length
        f[15] = min(n / 15.0, 1.0)

        return f

    def context_vec(self, word: str) -> List[float]:
        """Context channel (12D): which atoms displaced together.

        Co-occurrence neighborhoods: neighbor diversity, concentration,
        frequency statistics, positional statistics, exclusivity, density.
        """
        w = word.lower()
        s = [0.0] * 12
        co = self.co.get(w)
        if not co:
            return s

        ent = sorted(co.items(), key=lambda x: -x[1])
        tot = sum(c for _, c in ent)
        n_neighbors = len(ent)

        # [0] Neighbor diversity (normalized)
        s[0] = min(n_neighbors / 30.0, 1.0)

        # [1] Top-1 concentration
        s[1] = ent[0][1] / tot if ent else 0

        # [2] Top-3 concentration
        s[2] = sum(c for _, c in ent[:3]) / max(tot, 1)

        # [3-4] Neighbor frequency mean/variance
        tf = [math.log2(self.wf.get(nb, 0) + 1) for nb, _ in ent[:5]]
        if tf:
            s[3] = _mean(tf) / 10.0
            s[4] = _sd(tf) / 5.0 if len(tf) > 1 else 0.0

        # [5-6] Neighbor position mean/variance
        np_list: List[float] = []
        for nb, _ in ent[:10]:
            p = self.wp.get(nb)
            if p:
                np_list.extend(p[-10:])
        if np_list:
            s[5] = _mean(np_list)
            s[6] = _sd(np_list) if len(np_list) > 1 else 0.0

        # [7-8] Word's own position mean/variance
        mp = self.wp.get(w)
        if mp and len(mp) > 0:
            s[7] = _mean(mp)
            s[8] = _sd(mp) if len(mp) > 1 else 0.0

        # [9] Exclusivity (Jaccard distance from top neighbor)
        if n_neighbors >= 2:
            tn = ent[0][0]
            tc = self.co.get(tn, {})
            my_keys = set(co.keys())
            top_keys = set(tc.keys())
            union = my_keys | top_keys
            if union:
                s[9] = 1 - len(my_keys & top_keys) / len(union)

        # [10] Log frequency
        s[10] = min(math.log2(self.wf.get(w, 0) + 1) / 10.0, 1.0)

        # [11] Co-occurrence density
        s[11] = tot / max(self.wf.get(w, 1), 1) / 10.0

        return s

    def history_vec(self, word: str) -> List[float]:
        """History channel (8D): how deep the groove, how recently cut.

        Temporal exposure: log frequency, age, recency, gap regularity,
        positional stability, mean position, neighbor volatility, confidence.
        """
        w = word.lower()
        s = [0.0] * 8
        cnt = self.wf.get(w, 0)
        if not cnt:
            return s

        # [0] Log frequency
        s[0] = min(math.log2(cnt + 1) / 10.0, 1.0)

        # [1] Age: fraction of engine lifetime
        s[1] = min((self.ns - self.fs.get(w, self.ns) + 1) / max(self.ns, 1), 1.0)

        # [2] Recency: how recently was it last seen
        s[2] = max(0.0, 1 - (self.ns - self.ls.get(w, 0)) / max(self.ns, 1))

        # [3] Gap regularity: consistent spacing between encounters
        g = self.eg.get(w, [])
        if len(g) >= 2:
            s[3] = 1 - min(_sd(g) / (_mean(g) + 1e-3), 2.0) / 2.0
        elif len(g) == 1:
            s[3] = 0.5

        # [4] Positional stability
        ph = self.ph.get(w, [])
        if len(ph) >= 2:
            s[4] = 1 - min(_sd(ph) * 3, 1.0)

        # [5] Mean position
        if ph:
            s[5] = _mean(ph)

        # [6] Neighbor volatility
        v = self.nv.get(w, [])
        if v:
            s[6] = _mean(v)

        # [7] Confidence (capped frequency)
        s[7] = min(cnt / 10.0, 1.0)

        return s

    def influence_vec(self, word: str) -> List[float]:
        """Influence channel (8D): how the etching changed the surrounding surface.

        Structural footprint: sentence complexity, novelty attraction,
        diversity, bridging, positional asymmetry, structural weight variance,
        influence stability.
        """
        w = word.lower()
        s = [0.0] * 8
        co = self.co.get(w)
        if not co:
            return s

        # [0] Mean sentence complexity
        sl = self.sl.get(w, [])
        if sl:
            s[0] = min(_mean(sl) / 15.0, 1.0)

        # [1] Complexity variance
        if len(sl) >= 2:
            s[1] = min(_sd(sl) / 5.0, 1.0)

        # [2] Novelty attraction
        sn = self.sn.get(w, [])
        if sn:
            s[2] = _mean(sn)

        # [3] Sentence diversity
        s_div = self.sd.get(w, [])
        if s_div:
            s[3] = _mean(s_div)

        # [4] Bridging: 1 - triangle closure among neighbors
        nbs = list(co.keys())
        if len(nbs) >= 3:
            tri = 0
            pairs = 0
            sample = nbs[:10]
            for i in range(len(sample)):
                for j in range(i + 1, len(sample)):
                    pairs += 1
                    if sample[j] in self.co.get(sample[i], {}):
                        tri += 1
            s[4] = 1 - tri / pairs if pairs else 0

        # [5] Positional asymmetry
        pos = self.wp.get(w, [])
        if pos:
            s[5] = abs(_mean(pos) - 0.5) * 2

        # [6] Structural weight variance of neighbors
        if nbs:
            nf = [math.log2(self.wf.get(n, 0) + 1) for n in nbs[:10]]
            s[6] = min(_sd(nf) / 3.0, 1.0)

        # [7] Influence stability: early vs late sentence lengths
        if len(sl) >= 4:
            h = len(sl) // 2
            e_mean = _mean(sl[:h])
            l_mean = _mean(sl[h:])
            s[7] = 1 - min(abs(l_mean - e_mean) / max(e_mean, 1), 1.0) if e_mean > 0 else 0

        return s

    def contrast_vec(self, word: str) -> List[float]:
        """Contrast channel (8D): how this groove differs from the stone's average.

        Identity through deviation from self-model: frequency contrast,
        positional contrast, sentence-length contrast, neighborhood density contrast,
        exclusivity, surprise, identity stability.
        """
        w = word.lower()
        s = [0.0] * 8
        cnt = self.wf.get(w, 0)
        if not cnt or not self._global_word_count:
            return s

        avg_freq = self._global_freq_sum / max(self._global_word_count, 1)

        # [0] Frequency contrast
        s[0] = min(abs(cnt - avg_freq) / max(avg_freq, 1), 2.0) / 2.0 if avg_freq > 0 else 0

        # [1] Positional contrast
        wp = self.wp.get(w, [])
        gp = self._global_pos
        if wp and gp:
            s[1] = min(abs(_mean(wp) - _mean(gp)) * 3, 1.0)

        # [2] Positional specificity contrast
        if len(wp) >= 2 and len(gp) >= 2:
            s[2] = min(abs(_sd(wp) - _sd(gp)) * 5, 1.0)

        # [3] Sentence-length contrast
        sl = self.sl.get(w, [])
        gsl = self._global_sent_len
        if sl and gsl:
            s[3] = min(abs(_mean(sl) - _mean(gsl)) / max(_mean(gsl), 1), 1.0)

        # [4] Neighborhood density contrast
        co = self.co.get(w, {})
        n_nbs = len(co)
        avg_nbs = (sum(len(v) for v in self.co.values()) /
                   max(self._global_word_count, 1))
        if avg_nbs > 0:
            s[4] = min(abs(n_nbs - avg_nbs) / avg_nbs, 2.0) / 2.0

        # [5] Exclusivity (Jaccard from most common word's neighborhood)
        if co:
            my_nbs = set(co.keys())
            top_word = max(self.wf, key=self.wf.get) if self.wf else None
            if top_word:
                top_nbs = set(self.co.get(top_word, {}).keys())
                union = my_nbs | top_nbs
                if union:
                    s[5] = 1 - len(my_nbs & top_nbs) / len(union)

        # [6] Surprise (inverse relative frequency)
        max_freq = max(self.wf.values()) if self.wf else 1
        s[6] = 1 - min(cnt / max(max_freq, 1), 1.0)

        # [7] Identity stability: early vs late positional sd
        if len(wp) >= 6:
            h = len(wp) // 2
            e_sd = _sd(wp[:h])
            l_sd = _sd(wp[h:])
            s[7] = max(0.0, 1 - l_sd / e_sd) if e_sd > 0 else 0

        return s

    def expectation_vec(self, word: str) -> List[float]:
        """Expectation channel (8D): which direction the groove runs.

        The brain reads 'doctor' and expects {treats, prescribed, examined...}.
        That finite set IS the directional structure.
        """
        w = word.lower()
        s = [0.0] * 8
        nx = self.nx.get(w)
        px = self.px.get(w)
        if not nx and not px:
            return s

        # [0] Forward predictability: top next-word concentration
        if nx:
            ent = sorted(nx.items(), key=lambda x: -x[1])
            tot = sum(c for _, c in ent)
            s[0] = ent[0][1] / tot if ent and tot else 0

        # [1] Forward diversity
        if nx:
            s[1] = min(len(nx) / 15.0, 1.0)

        # [2] Backward predictability
        if px:
            ent = sorted(px.items(), key=lambda x: -x[1])
            tot = sum(c for _, c in ent)
            s[2] = ent[0][1] / tot if ent and tot else 0

        # [3] Backward diversity
        if px:
            s[3] = min(len(px) / 15.0, 1.0)

        # [4] Directional asymmetry: |forward - backward| predictability
        s[4] = abs(s[0] - s[2])

        # [5] Expectation breadth: total unique transitions
        fwd_n = len(nx) if nx else 0
        bwd_n = len(px) if px else 0
        s[5] = min((fwd_n + bwd_n) / 20.0, 1.0)

        # [6] Forward-backward overlap: same words before AND after = symmetric position
        if nx and px:
            fwd = set(nx.keys())
            bwd = set(px.keys())
            union = fwd | bwd
            inter = len(fwd & bwd)
            s[6] = 1 - inter / len(union) if union else 0

        # [7] Expectation stability: top-3 concentration
        if nx:
            ent = sorted(nx.items(), key=lambda x: -x[1])
            tot = sum(c for _, c in ent)
            top3 = sum(c for _, c in ent[:3])
            s[7] = top3 / tot if tot else 0

        return s

    # -----------------------------------------------------------------------
    # Full surface reading
    # -----------------------------------------------------------------------

    def vec(self, word: str) -> List[float]:
        """Full 60D vector: all 6 channel vectors concatenated. Cached."""
        k = word.lower().strip()
        if k in self._cache:
            return self._cache[k]
        v = (self.form_vec(k) + self.context_vec(k) + self.history_vec(k) +
             self.influence_vec(k) + self.contrast_vec(k) + self.expectation_vec(k))
        self._cache[k] = v
        return v

    # -----------------------------------------------------------------------
    # Compare: ASYMMETRIC
    # -----------------------------------------------------------------------

    def compare(self, a: str, b: str, purpose: str = "meaning",
                mask: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        """Compare two etchings on the stone. ASYMMETRIC.

        compare(a, b) != compare(b, a) because grooves run in directions.
        'doctor -> treats' has a channel; 'treats -> doctor' may not.
        """
        va, vb = self.vec(a), self.vec(b)

        sc: Dict[str, Any] = {
            "form": _cos(va, vb, *IDX["form"]),
            "context": _cos(va, vb, *IDX["ctx"]),
            "history": _cos(va, vb, *IDX["hist"]),
            "influence": _cos(va, vb, *IDX["inf"]),
            "contrast": _cos(va, vb, *IDX["con"]),
            "expectation": _cos(va, vb, *IDX["exp"]),
            "ocr": max(0, 1 - ocr_dist(a, b) / max(len(a), len(b), 1)),
            "full": _cos(va, vb, *IDX["all"]),
        }

        # Directional signal
        al, bl = a.lower(), b.lower()
        nx_a = self.nx.get(al, {})
        nx_b = self.nx.get(bl, {})
        tot_a = sum(nx_a.values()) if nx_a else 0
        tot_b = sum(nx_b.values()) if nx_b else 0
        sc["expects_ab"] = nx_a.get(bl, 0) / max(tot_a, 1)
        sc["expects_ba"] = nx_b.get(al, 0) / max(tot_b, 1)
        sc["directional"] = abs(sc["expects_ab"] - sc["expects_ba"])

        # Second-order trajectory
        nx2_ab = self.nx2.get(al, {}).get(bl)
        nx2_ba = self.nx2.get(bl, {}).get(al)
        sc["trajectory_ab"] = min(len(nx2_ab) / 5.0, 1.0) if nx2_ab else 0
        sc["trajectory_ba"] = min(len(nx2_ba) / 5.0, 1.0) if nx2_ba else 0

        # Routed score
        rt = ROUTING.get(purpose, {k: 1 / 7 for k in
                         ["form", "context", "history", "influence",
                          "contrast", "expectation", "ocr"]})
        wt = dict(rt)
        if mask:
            for ch in ["form", "context", "history", "influence",
                       "contrast", "expectation", "ocr"]:
                if not mask.get(ch, True):
                    wt[ch] = 0
            s = sum(wt.values())
            if s > 0:
                wt = {k: v / s for k, v in wt.items()}

        routed = sum(wt.get(ch, 0) * sc.get(ch, 0) for ch in
                     ["form", "context", "history", "influence",
                      "contrast", "expectation", "ocr"])
        sc["routed"] = routed
        sc["weights"] = wt
        return sc

    # -----------------------------------------------------------------------
    # Correction and similarity
    # -----------------------------------------------------------------------

    def correct(self, garbled: str, k: int = 5) -> Dict[str, Any]:
        """Correct a garbled word using OCR-weighted comparison."""
        cands = self._candidates(garbled)
        out = []
        for w in cands:
            sc = self.compare(garbled, w, "correction")
            sc["word"] = w
            out.append(sc)
        out.sort(key=lambda x: (-x["routed"], -(self.wf.get(x["word"], 0))))
        top = out[:k]
        conf = top[0]["routed"] - top[1]["routed"] if len(top) >= 2 else (1.0 if top else 0.0)
        return {
            "candidates": top,
            "confidence": conf,
            "reject": conf < CONFIDENCE_REJECT,
            "low_confidence": conf < CONFIDENCE_LOW,
        }

    def similar(self, word: str, k: int = 10) -> List[Dict[str, Any]]:
        """Find most similar words by meaning routing."""
        key = word.lower()
        out = []
        for w in self.wf:
            if w == key:
                continue
            sc = self.compare(word, w, "meaning")
            sc["word"] = w
            out.append(sc)
        out.sort(key=lambda x: -x["routed"])
        return out[:k]

    # -----------------------------------------------------------------------
    # Affinity: pre-contact attraction
    # -----------------------------------------------------------------------

    def affinity(self, a: str, b: str) -> Dict[str, Any]:
        """Pre-contact attraction between two words. ASYMMETRIC.

        Predicts which words will relate BEFORE they meet in a sentence.
        Five signals weighted by discrimination power:
            Shared orbit (35%)  -- co-occurrence overlap (relational)
            Trajectory pull (20%) -- one groove runs toward the other
            Indirect paths (20%) -- groove via intermediary
            Expectation overlap (15%) -- predict similar futures
            Form resonance (5%)  -- word shape similarity
            Contrast alignment (5%) -- similar deviation from baseline
        """
        a, b = a.lower(), b.lower()
        result: Dict[str, Any] = {"a": a, "b": b}

        # 1. Form resonance (symmetric)
        fa, fb = self.form_vec(a), self.form_vec(b)
        result["form_resonance"] = _cos(fa, fb, 0, 16)

        # 2. Shared orbit (symmetric) -- weighted Jaccard
        co_a = self.co.get(a, {})
        co_b = self.co.get(b, {})
        if co_a and co_b:
            nb_a = list(co_a.keys())
            nb_b_set = set(co_b.keys())
            shared = [x for x in nb_a if x in nb_b_set]
            if shared and nb_a:
                wt_fn = lambda w: 1 / max(math.log2(self.wf.get(w, 0) + 1), 1)
                shared_wt = sum(wt_fn(w) for w in shared)
                total_wt = (sum(wt_fn(w) for w in nb_a) +
                            sum(wt_fn(w) for w in co_b) - shared_wt)
                result["shared_orbit"] = shared_wt / total_wt if total_wt > 0 else 0
            else:
                result["shared_orbit"] = 0.0
        else:
            result["shared_orbit"] = 0.0

        # 3. Trajectory pull (ASYMMETRIC)
        nx_a = self.nx.get(a, {})
        nx_b = self.nx.get(b, {})
        tot_a = sum(nx_a.values()) if nx_a else 0
        tot_b = sum(nx_b.values()) if nx_b else 0
        result["pull_ab"] = nx_a.get(b, 0) / tot_a if tot_a else 0
        result["pull_ba"] = nx_b.get(a, 0) / tot_b if tot_b else 0

        # Indirect paths
        indirect_ab = 0
        for mid in nx_a:
            if b in self.nx.get(mid, {}):
                indirect_ab += 1
        indirect_ba = 0
        for mid in nx_b:
            if a in self.nx.get(mid, {}):
                indirect_ba += 1
        result["indirect_ab"] = min(indirect_ab / 5.0, 1.0)
        result["indirect_ba"] = min(indirect_ba / 5.0, 1.0)

        # 4. Contrast alignment (symmetric)
        ca, cb = self.contrast_vec(a), self.contrast_vec(b)
        result["contrast_alignment"] = _cos(ca, cb, 0, 8)

        # 5. Expectation overlap (symmetric)
        ea, eb = self.expectation_vec(a), self.expectation_vec(b)
        result["expectation_overlap"] = _cos(ea, eb, 0, 8)

        # Combined (directional)
        result["affinity_ab"] = (
            result["form_resonance"] * 0.05 +
            result["shared_orbit"] * 0.35 +
            result["pull_ab"] * 0.20 +
            result["indirect_ab"] * 0.20 +
            result["contrast_alignment"] * 0.05 +
            result["expectation_overlap"] * 0.15
        )
        result["affinity_ba"] = (
            result["form_resonance"] * 0.05 +
            result["shared_orbit"] * 0.35 +
            result["pull_ba"] * 0.20 +
            result["indirect_ba"] * 0.20 +
            result["contrast_alignment"] * 0.05 +
            result["expectation_overlap"] * 0.15
        )
        result["mutual"] = (result["affinity_ab"] + result["affinity_ba"]) / 2
        result["asymmetry"] = abs(result["affinity_ab"] - result["affinity_ba"])

        return result

    # -----------------------------------------------------------------------
    # Sentence scoring: dynamic expectation field
    # -----------------------------------------------------------------------

    def score_sentence(self, raw: str) -> Dict[str, Any]:
        """Score a sentence by walking the stone's surface.

        At each word: sequential surprise (nx/nx2), trajectory surprise,
        and field surprise (accumulated co-occurrence field from ALL
        preceding words). The field reshapes dynamically -- each word
        deposits its neighborhood, reconfiguring what later words mean.
        """
        ws = _tokenize(raw)
        if len(ws) < 2:
            return {"words": ws, "steps": [], "total_surprise": 0,
                    "mean_surprise": 0, "coherence": 0}

        steps = []
        total_surprise = 0.0
        context_field: Dict[str, float] = {}

        for i, w in enumerate(ws):
            step: Dict[str, Any] = {"word": w, "position": i, "known": w in self.wf}

            # 1. Sequential surprise
            if i > 0:
                prev = ws[i - 1]
                nx_prev = self.nx.get(prev, {})
                total_next = sum(nx_prev.values()) if nx_prev else 0
                step["seq_expected"] = nx_prev.get(w, 0) / max(total_next, 1)
                step["seq_surprise"] = 1 - step["seq_expected"]
            else:
                step["seq_expected"] = None
                step["seq_surprise"] = 0

            # Trajectory surprise (second-order)
            if i >= 2:
                pp, prev = ws[i - 2], ws[i - 1]
                nx2_pp = self.nx2.get(pp, {}).get(prev)
                total_nx2 = sum(nx2_pp.values()) if nx2_pp else 0
                step["traj_expected"] = nx2_pp.get(w, 0) / max(total_nx2, 1) if nx2_pp else 0
                step["traj_surprise"] = 1 - step["traj_expected"]
            else:
                step["traj_expected"] = None
                step["traj_surprise"] = 0

            # 2. Field surprise
            if i > 0 and context_field:
                field_weight = context_field.get(w, 0)
                max_field = max(context_field.values()) if context_field else 1
                step["field_expected"] = field_weight / max(max_field, 1)
                step["field_surprise"] = 1 - step["field_expected"]
            else:
                step["field_expected"] = None
                step["field_surprise"] = 0

            # 3. Novelty
            co = self.co.get(w, {})
            if co:
                new_nbs = sum(1 for n in co if n not in context_field)
                step["novelty"] = new_nbs / len(co)
            else:
                step["novelty"] = 1.0

            # Combined surprise
            seq_s = step["seq_surprise"]
            traj_s = step["traj_surprise"] if i >= 2 else seq_s
            field_s = step["field_surprise"] if i > 0 else 0
            base_surprise = seq_s * 0.35 + traj_s * 0.30 + field_s * 0.35

            # Affinity gate
            if i > 0:
                prev_co = self.co.get(ws[i - 1], {})
                w_co = self.co.get(w, {})
                if prev_co and w_co:
                    p_n = set(prev_co.keys())
                    w_n = set(w_co.keys())
                    union = p_n | w_n
                    shared = len(p_n & w_n)
                    step["affinity_gate"] = shared / len(union) if union else 0
                else:
                    step["affinity_gate"] = 0
            else:
                step["affinity_gate"] = 0

            step["surprise"] = base_surprise * (1 - step["affinity_gate"] * 0.3)
            total_surprise += step["surprise"]
            step["cumulative_surprise"] = total_surprise
            steps.append(step)

            # Update field: deposit this word's neighborhood
            for nb, cnt in co.items():
                context_field[nb] = context_field.get(nb, 0) + cnt
            context_field[w] = context_field.get(w, 0) + 10

        mean_surprise = total_surprise / len(steps) if steps else 0
        coherence = 1 - min(mean_surprise, 1.0)

        return {"words": ws, "steps": steps, "total_surprise": total_surprise,
                "mean_surprise": mean_surprise, "coherence": coherence}

    # -----------------------------------------------------------------------
    # Stats and serialization
    # -----------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "sentences": self.ns,
            "tokens": self.nt,
            "vocabulary": len(self.wf),
            "bigrams": len(self.bf),
            "cooccurrences": sum(len(v) for v in self.co.values()),
            "mature": sum(1 for v in self.wf.values() if v >= 10),
        }

    def serialize(self) -> str:
        """Serialize full state to JSON."""
        def trim(d: Dict, n: int) -> Dict:
            return {k: (v[-n:] if isinstance(v, list) else v) for k, v in d.items()}

        data = {
            "version": self.version,
            "wf": self.wf, "co": self.co,
            "wp": trim(self.wp, 50), "bf": self.bf,
            "fs": self.fs, "ls": self.ls,
            "eg": trim(self.eg, 30), "ph": trim(self.ph, 30),
            "nv": trim(self.nv, 20),
            "sl": trim(self.sl, 30), "sn": trim(self.sn, 30), "sd": trim(self.sd, 30),
            "nx": self.nx, "px": self.px, "nx2": self.nx2,
            "gp": self._global_pos[-200:],
            "gsl": self._global_sent_len[-200:],
            "gfs": self._global_freq_sum,
            "gwc": self._global_word_count,
            "pn": {k: list(v) for k, v in self._pn.items()},
            "ns": self.ns, "nt": self.nt,
        }
        return json.dumps(data)

    @classmethod
    def deserialize(cls, raw: str) -> "ShifuEngine":
        """Deserialize from JSON string."""
        d = json.loads(raw)
        e = cls()
        e.wf = d.get("wf", {})
        e.co = d.get("co", {})
        e.wp = d.get("wp", {})
        e.bf = d.get("bf", {})
        e.fs = d.get("fs", {})
        e.ls = d.get("ls", {})
        e.eg = d.get("eg", {})
        e.ph = d.get("ph", {})
        e.nv = d.get("nv", {})
        e.sl = d.get("sl", {})
        e.sn = d.get("sn", {})
        e.sd = d.get("sd", {})
        e.nx = d.get("nx", {})
        e.px = d.get("px", {})
        e.nx2 = d.get("nx2", {})
        e._global_pos = d.get("gp", [])
        e._global_sent_len = d.get("gsl", [])
        e._global_freq_sum = d.get("gfs", 0)
        e._global_word_count = d.get("gwc", 0)
        if "pn" in d:
            e._pn = {k: set(v) for k, v in d["pn"].items()}
        e.ns = d.get("ns", 0)
        e.nt = d.get("nt", 0)
        for w in e.wf:
            e._reindex(w)
        return e
