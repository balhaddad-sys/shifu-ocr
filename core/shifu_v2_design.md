# SHIFU v2 — The Embryo Engine

## Design Document

**Read entirely before implementing anything.**

---

## Why v2 exists

v1.4.1 is a stone. 60 hardcoded dimensions. 7 pre-defined channels. Fixed routing weights. The hand-authored features are the observer's encoded prior distinctions — and that's the problem. The observer decided in advance what matters about a word. The data doesn't get to decide.

A stone can be etched but not reorganized. The energy cost is astronomical. What has become cannot unbecome.

v2 is the embryo. One cell. No dimensions yet. No channels. No routing weights. The first interaction creates the first distinction. The second reinforces it or creates a second. Structure emerges entirely from the sequence of exposures — like embryogenesis, where the same genome produces a neuron or a liver cell depending on which signals arrived in which order.

**Principle: Hardcode nothing. Teach everything. Let the data decide what dimensions matter.**

---

## What stays from v1

The stone taught us what questions to ask. These survive:

- `feed(sentence)` — the perturbation. Text arrives. State changes.
- `co` — co-occurrence. Which words displaced together. This is raw observation, not a feature.
- `nx / px / nx2` — sequential expectations. What follows, what precedes. Raw observation.
- `wf` — word frequency. How many times we've been struck. Raw count.
- `scoreSentence()` — walking the sequence, accumulating surprise. The method survives. The implementation changes.
- `affinity()` — pre-contact attraction. The method survives. The signals change.
- `compare()` — asymmetric. The method survives. The mechanism changes completely.

## What dies

Everything that pre-defines structure:

- `formVec()` — 16 hardcoded phonotactic features
- `contextVec()` — 12 hardcoded co-occurrence statistics
- `historyVec()` — 8 hardcoded temporal features
- `influenceVec()` — 8 hardcoded structural features
- `contrastVec()` — 8 hardcoded deviation features
- `expectationVec()` — 8 hardcoded sequential features
- `IDX` — fixed dimension boundaries
- `CONFIG.channels` — fixed channel definitions
- `CONFIG.routing` — fixed routing weights
- `vec()` — fixed 60D vector concatenation
- All cosine similarity in fixed vector space

None of these exist in v2.

---

## The Embryo Architecture

### State: Raw Observations Only

Each word is a node. The node holds only what has been directly observed:

```javascript
node[word] = {
  // ── What happened ──
  freq: 0,              // times seen
  firstSeen: null,      // sentence number
  lastSeen: null,       // sentence number
  positions: [],        // relative positions in sentences (0.0 → 1.0)
  gaps: [],             // sentences between re-encounters
  sentLengths: [],      // lengths of sentences it appeared in

  // ── What it touched ──
  neighbors: {},        // {word: count} — co-occurrence within window
  next: {},             // {word: count} — what followed (first-order)
  prev: {},             // {word: count} — what preceded (first-order)
  next2: {},            // {midWord: {word: count}} — second-order forward

  // ── What it is ──
  chars: "doctor",      // the raw character sequence (the genome)
}
```

That's it. No features. No vectors. No dimensions. Raw events.

The engine's global state is equally minimal:

```javascript
engine = {
  nodes: {},            // word → node
  sentenceCount: 0,
  tokenCount: 0,
}
```

No `_globalPos`. No `_globalSentLen`. No `_globalFreqSum`. No self-model arrays. Those emerge when you ask for them.

### feed(sentence) — The Perturbation

Identical in purpose to v1. Different in what it stores:

```javascript
feed(raw) {
  const words = tokenize(raw);
  this.sentenceCount++;

  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    const node = this.nodes[w] ??= newNode(w);
    const relPos = i / Math.max(words.length - 1, 1);

    // Record what happened
    node.freq++;
    node.firstSeen ??= this.sentenceCount;
    if (node.lastSeen) node.gaps.push(this.sentenceCount - node.lastSeen);
    node.lastSeen = this.sentenceCount;
    node.positions.push(relPos);
    node.sentLengths.push(words.length);

    // Record what it touched
    for (let j = max(0, i-3); j < min(words.length, i+4); j++) {
      if (j !== i) node.neighbors[words[j]] = (node.neighbors[words[j]] || 0) + 1;
    }
    if (i < words.length - 1) node.next[words[i+1]] = (node.next[words[i+1]] || 0) + 1;
    if (i > 0) node.prev[words[i-1]] = (node.prev[words[i-1]] || 0) + 1;
    if (i < words.length - 2) {
      node.next2[words[i+1]] ??= {};
      node.next2[words[i+1]][words[i+2]] = (node.next2[words[i+1]][words[i+2]] || 0) + 1;
    }

    // Trim histories to prevent unbounded growth
    if (node.positions.length > 200) node.positions = node.positions.slice(-200);
    if (node.gaps.length > 100) node.gaps = node.gaps.slice(-100);
    if (node.sentLengths.length > 100) node.sentLengths = node.sentLengths.slice(-100);
  }
}
```

No cache to invalidate — there's no cached vector to go stale. Every comparison is computed fresh from raw observations.

### compare(a, b) — Direct Measurement, Not Cosine

v1 compared 60D vectors via cosine similarity. v2 compares two nodes directly by measuring whatever evidence exists between them. The available evidence depends on how much each word has been observed.

```javascript
compare(a, b) {
  const na = this.nodes[a], nb = this.nodes[b];
  const signals = {};
  let totalWeight = 0;

  // ── Character-level (always available, even for unseen words) ──
  // Not 16 hardcoded features. Direct comparison.
  signals.editSim = 1 - editDistance(a, b) / Math.max(a.length, b.length, 1);
  signals.bigramSim = sharedBigrams(a, b); // Dice coefficient of character bigrams
  // Weight: low. The genome is the weakest signal.
  const charWeight = 0.05;
  totalWeight += charWeight;

  if (!na || !nb) {
    // One or both unseen. Character-level is all we have.
    return { similarity: signals.editSim * 0.5 + signals.bigramSim * 0.5, signals, depth: "surface" };
  }

  // ── Neighborhood overlap (requires corpus exposure for both) ──
  const nbrsA = Object.keys(na.neighbors), nbrsB = new Set(Object.keys(nb.neighbors));
  const shared = nbrsA.filter(x => nbrsB.has(x));
  const union = new Set([...nbrsA, ...nbrsB]);
  if (union.size > 0) {
    // Weight shared neighbors by rarity (1/log(freq)) — rare shared neighbors matter more
    const wt = w => 1 / Math.max(Math.log2((this.nodes[w]?.freq || 0) + 1), 1);
    const sharedW = shared.reduce((s, w) => s + wt(w), 0);
    const unionW = [...union].reduce((s, w) => s + wt(w), 0);
    signals.neighborOverlap = unionW > 0 ? sharedW / unionW : 0;
    const nbrWeight = Math.min(union.size / 10, 0.35); // grows with evidence
    totalWeight += nbrWeight;
  }

  // ── Sequential relationship (requires nx/px data) ──
  const totA = na.next ? Object.values(na.next).reduce((s, v) => s + v, 0) : 0;
  const totB = nb.next ? Object.values(nb.next).reduce((s, v) => s + v, 0) : 0;
  if (totA > 0 || totB > 0) {
    signals.expectsAB = totA ? (na.next[b] || 0) / totA : 0;
    signals.expectsBA = totB ? (nb.next[a] || 0) / totB : 0;
    signals.directional = Math.abs(signals.expectsAB - signals.expectsBA);
    const seqWeight = Math.min((totA + totB) / 20, 0.25);
    totalWeight += seqWeight;
  }

  // ── Positional similarity (requires enough position observations) ──
  if (na.positions.length >= 3 && nb.positions.length >= 3) {
    const meanA = mean(na.positions), meanB = mean(nb.positions);
    const sdA = stdev(na.positions), sdB = stdev(nb.positions);
    signals.positionalSim = 1 - Math.min(Math.abs(meanA - meanB) * 2, 1);
    signals.positionalSpreadSim = 1 - Math.min(Math.abs(sdA - sdB) * 3, 1);
    const posWeight = Math.min(Math.min(na.positions.length, nb.positions.length) / 20, 0.15);
    totalWeight += posWeight;
  }

  // ── Frequency similarity (requires enough exposure) ──
  if (na.freq >= 2 && nb.freq >= 2) {
    const maxFreq = Math.max(na.freq, nb.freq);
    signals.freqSim = 1 - Math.abs(na.freq - nb.freq) / maxFreq;
    const freqWeight = 0.05;
    totalWeight += freqWeight;
  }

  // ── Trajectory depth (requires nx2 data) ──
  const nx2AB = na.next2?.[b] ? Object.keys(na.next2[b]).length : 0;
  const nx2BA = nb.next2?.[a] ? Object.keys(nb.next2[a]).length : 0;
  if (nx2AB > 0 || nx2BA > 0) {
    signals.trajectoryAB = Math.min(nx2AB / 5, 1);
    signals.trajectoryBA = Math.min(nx2BA / 5, 1);
    const trajWeight = 0.15;
    totalWeight += trajWeight;
  }

  // ── Indirect paths (requires rich graph) ──
  if (Object.keys(na.next).length >= 2) {
    let indirect = 0;
    for (const mid of Object.keys(na.next)) {
      if (this.nodes[mid]?.next[b]) indirect++;
    }
    signals.indirectAB = Math.min(indirect / 5, 1);
    const indWeight = Math.min(indirect / 10, 0.15);
    totalWeight += indWeight;
  }

  // ── Weighted combination: weights grow with evidence ──
  // No fixed routing. Each signal contributes proportional to how much
  // evidence supports it. Early in life: mostly character-level.
  // After 1000 sentences: mostly neighborhood + trajectory.
  let sim = 0;
  if (totalWeight > 0) {
    if (signals.editSim !== undefined) sim += signals.editSim * charWeight;
    if (signals.neighborOverlap !== undefined) sim += signals.neighborOverlap * Math.min(union.size / 10, 0.35);
    if (signals.expectsAB !== undefined) sim += Math.max(signals.expectsAB, signals.expectsBA) * Math.min((totA + totB) / 20, 0.25);
    if (signals.positionalSim !== undefined) sim += signals.positionalSim * Math.min(Math.min(na.positions.length, nb.positions.length) / 20, 0.15);
    if (signals.freqSim !== undefined) sim += signals.freqSim * 0.05;
    if (signals.trajectoryAB !== undefined) sim += Math.max(signals.trajectoryAB, signals.trajectoryBA) * 0.15;
    if (signals.indirectAB !== undefined) sim += signals.indirectAB * Math.min(signals.indirectAB * 2 / 10, 0.15);
    sim /= totalWeight;
  }

  return {
    similarity: sim,
    signals,
    totalWeight,
    depth: totalWeight < 0.1 ? "surface" : totalWeight < 0.3 ? "shallow" : totalWeight < 0.6 ? "forming" : "deep",
  };
}
```

**The key difference from v1:** Weights are not fixed. They grow with evidence. A word seen once has mostly character-level comparison (depth: "surface"). A word seen 100 times has rich neighborhood, sequential, positional, trajectory evidence (depth: "deep"). The engine doesn't pretend to know what it hasn't observed.

### scoreSentence(raw) — Walking the Graph

Same principle as v1 but no fixed signal weights:

```javascript
scoreSentence(raw) {
  const words = tokenize(raw);
  if (words.length < 2) return { words, steps: [], coherence: 0, meanSurprise: 0 };

  const steps = [];
  let totalSurprise = 0;
  const field = new Map(); // accumulated co-occurrence field

  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    const node = this.nodes[w];
    const step = { word: w, pos: i, known: !!node };
    let signals = 0, weights = 0;

    // Sequential: was this expected after previous?
    if (i > 0) {
      const prev = this.nodes[words[i-1]];
      if (prev?.next) {
        const tot = Object.values(prev.next).reduce((a, b) => a + b, 0);
        step.seqS = 1 - ((prev.next[w] || 0) / Math.max(tot, 1));
        signals += step.seqS * 0.35;
        weights += 0.35;
      }
    }

    // Trajectory: second-order expectation
    if (i >= 2) {
      const pp = this.nodes[words[i-2]];
      const nx2 = pp?.next2?.[words[i-1]];
      if (nx2) {
        const tot = Object.values(nx2).reduce((a, b) => a + b, 0);
        step.trajS = 1 - ((nx2[w] || 0) / Math.max(tot, 1));
        signals += step.trajS * 0.30;
        weights += 0.30;
      }
    }

    // Field: do all preceding words collectively expect this one?
    if (i > 0 && field.size > 0) {
      const fw = field.get(w) || 0;
      const maxF = Math.max(...field.values(), 1);
      step.fieldS = 1 - fw / maxF;
      signals += step.fieldS * 0.35;
      weights += 0.35;
    }

    // Affinity gate: adjacent words share neighborhoods → reduce surprise
    if (i > 0 && node) {
      const prevNode = this.nodes[words[i-1]];
      if (prevNode?.neighbors && node.neighbors) {
        const pN = Object.keys(prevNode.neighbors);
        const wN = new Set(Object.keys(node.neighbors));
        const shared = pN.filter(x => wN.has(x)).length;
        const union = new Set([...pN, ...wN]).size;
        step.afGate = union ? shared / union : 0;
      }
    }

    step.surprise = weights > 0 ? (signals / weights) * (1 - (step.afGate || 0) * 0.3) : (node ? 0.5 : 1.0);
    totalSurprise += step.surprise;
    step.cumS = totalSurprise;
    steps.push(step);

    // Update field: this word's neighbors become expectations for future words
    if (node?.neighbors) {
      for (const [nb, cnt] of Object.entries(node.neighbors)) {
        field.set(nb, (field.get(nb) || 0) + cnt);
      }
    }
    field.set(w, (field.get(w) || 0) + 10);
  }

  const meanS = steps.length ? totalSurprise / steps.length : 0;
  return { words, steps, meanSurprise: meanS, coherence: 1 - Math.min(meanS, 1) };
}
```

### affinity(a, b) — Pre-Contact Attraction

Same 5-signal design as v1.4.1 but computed directly from node data, not from hardcoded channel vectors:

```javascript
affinity(a, b) {
  const na = this.nodes[a], nb = this.nodes[b];
  if (!na || !nb) return { mutual: 0, known: false };

  // 1. Shared orbit (35%) — neighborhood Jaccard, rare-weighted
  const nbrsA = Object.keys(na.neighbors), nbrsB = new Set(Object.keys(nb.neighbors));
  const shared = nbrsA.filter(x => nbrsB.has(x));
  const wt = w => 1 / Math.max(Math.log2((this.nodes[w]?.freq || 0) + 1), 1);
  const sw = shared.reduce((s, w) => s + wt(w), 0);
  const uw = [...new Set([...nbrsA, ...nbrsB])].reduce((s, w) => s + wt(w), 0);
  const orbit = uw > 0 ? sw / uw : 0;

  // 2. Trajectory pull (20%) — directional, asymmetric
  const totA = Object.values(na.next || {}).reduce((s, v) => s + v, 0);
  const totB = Object.values(nb.next || {}).reduce((s, v) => s + v, 0);
  const pullAB = totA ? (na.next[b] || 0) / totA : 0;
  const pullBA = totB ? (nb.next[a] || 0) / totB : 0;

  // 3. Indirect paths (20%) — a→?→b exists?
  let indAB = 0, indBA = 0;
  for (const mid of Object.keys(na.next || {})) if (this.nodes[mid]?.next[b]) indAB++;
  for (const mid of Object.keys(nb.next || {})) if (this.nodes[mid]?.next[a]) indBA++;
  indAB = Math.min(indAB / 5, 1);
  indBA = Math.min(indBA / 5, 1);

  // 4. Character similarity (5%) — the genome
  const charSim = 1 - editDistance(a, b) / Math.max(a.length, b.length, 1);

  // 5. Expectation overlap (15%) — do they predict similar futures?
  const fwA = new Set(Object.keys(na.next || {}));
  const fwB = new Set(Object.keys(nb.next || {}));
  const fwUnion = new Set([...fwA, ...fwB]);
  const expOvlp = fwUnion.size ? [...fwA].filter(x => fwB.has(x)).length / fwUnion.size : 0;

  // 5b. Positional alignment (5%) — do they live in similar sentence regions?
  let posAlign = 0;
  if (na.positions.length >= 3 && nb.positions.length >= 3) {
    posAlign = 1 - Math.min(Math.abs(mean(na.positions) - mean(nb.positions)) * 3, 1);
  }

  const afAB = charSim * 0.05 + orbit * 0.35 + pullAB * 0.20 + indAB * 0.20 + expOvlp * 0.15 + posAlign * 0.05;
  const afBA = charSim * 0.05 + orbit * 0.35 + pullBA * 0.20 + indBA * 0.20 + expOvlp * 0.15 + posAlign * 0.05;

  return {
    a, b, orbit, pullAB, pullBA, indAB, indBA, charSim, expOvlp, posAlign,
    afAB, afBA, mutual: (afAB + afBA) / 2, asym: Math.abs(afAB - afBA),
  };
}
```

### correct(garbled, k) — OCR Correction

The OCR topology table is the ONE hardcoded thing that stays. It's not a learned feature — it's physics. The shape of "rn" on paper is objectively similar to "m". That's not something the engine learns from exposure. It's a property of the writing medium. The embryo can have this in its genome.

```javascript
// The genome: character topology (physics of ink on paper)
const OCR_TOPOLOGY = {
  "0,o": 0.1, "1,l": 0.2, "1,i": 0.2, "5,s": 0.3, "8,b": 0.3,
  "6,g": 0.4, "l,i": 0.2, "m,n": 0.4, "u,v": 0.5, "c,e": 0.5,
  "r,n": 0.3, "d,o": 0.3, "f,t": 0.4, "h,b": 0.4, "a,e": 0.4,
  "a,o": 0.4, "u,n": 0.4, "e,i": 0.4, "f,l": 0.4, "s,e": 0.5, "b,d": 0.4,
};

correct(garbled, k = 5) {
  // Find candidates by length proximity + shared bigrams (same as v1)
  const candidates = this._findCandidates(garbled);
  // Score by topology-weighted edit distance + character bigram overlap
  // NO learned routing weights. Just: how close is this character sequence?
  const scored = candidates.map(w => {
    const ocrSim = 1 - ocrDistance(garbled, w) / Math.max(garbled.length, w.length, 1);
    const bigramSim = sharedBigrams(garbled, w);
    return { word: w, score: ocrSim * 0.7 + bigramSim * 0.3, ocrSim, bigramSim };
  });
  scored.sort((a, b) => b.score - a.score || (this.nodes[b.word]?.freq || 0) - (this.nodes[a.word]?.freq || 0));
  const top = scored.slice(0, k);
  const conf = top.length >= 2 ? top[0].score - top[1].score : top.length ? 1 : 0;
  return { candidates: top, confidence: conf };
}
```

---

## What Emerges That Wasn't Designed

In v1, the "channels" were designed. In v2, they emerge:

| v1 (designed) | v2 (emergent) | How it emerges |
|--------------|---------------|----------------|
| Form (16D) | Character-level signals | Always available from the genome (chars) |
| Context (12D) | Neighborhood overlap | Grows as `neighbors` accumulates through feeding |
| History (8D) | Frequency, recency, gap patterns | Grows as `freq`, `gaps`, `positions` accumulate |
| Influence (8D) | Sentence-length/diversity patterns | Grows as `sentLengths` accumulates |
| Contrast (8D) | Deviation from population statistics | Computed on-demand by comparing node stats to engine-wide averages |
| Expectation (8D) | Sequential predictions | Grows as `next`, `prev`, `next2` accumulate |
| Affinity | Same as v1 but from raw graph | Computed from orbit, pull, indirect paths |

The difference: in v1, these are always 60 numbers. In v2, they start as nothing and grow. A word seen once has character-level identity only. A word seen 1000 times has everything. The depth of comparison scales with experience.

---

## The Depth Metric

v2 introduces something v1 couldn't have: a measure of how much evidence supports any comparison.

```javascript
depth(word) {
  const n = this.nodes[word];
  if (!n) return { level: "unborn", evidence: 0 };
  if (n.freq === 1) return { level: "surface", evidence: 0.1 };
  if (n.freq < 5) return { level: "shallow", evidence: 0.2 };
  const nbrCount = Object.keys(n.neighbors).length;
  const seqCount = Object.keys(n.next).length + Object.keys(n.prev).length;
  const evidence = Math.min(
    (n.freq / 50) * 0.3 +
    (nbrCount / 20) * 0.3 +
    (seqCount / 10) * 0.2 +
    (n.positions.length / 50) * 0.2,
    1.0
  );
  if (evidence < 0.3) return { level: "forming", evidence };
  if (evidence < 0.7) return { level: "structured", evidence };
  return { level: "deep", evidence };
}
```

This is honest in a way v1 couldn't be. v1 always returned a 60D vector, even for a word seen once — most of those 60 dimensions were zeros, pretending to measure things that hadn't been observed. v2 says "I don't know yet" and means it.

---

## Serialization

Trivial. The entire state is the node graph + counters:

```javascript
serialize() {
  return JSON.stringify({
    nodes: Object.fromEntries(
      Object.entries(this.nodes).map(([w, n]) => [w, {
        freq: n.freq, firstSeen: n.firstSeen, lastSeen: n.lastSeen,
        positions: n.positions.slice(-200),
        gaps: n.gaps.slice(-100),
        sentLengths: n.sentLengths.slice(-100),
        neighbors: n.neighbors, next: n.next, prev: n.prev, next2: n.next2,
      }])
    ),
    sentenceCount: this.sentenceCount,
    tokenCount: this.tokenCount,
  });
}
```

No IDX. No channel config. No routing weights. Nothing to get out of sync. The serialized state IS the complete state. Nothing is lost because nothing was pre-computed.

---

## What This Cannot Do (Yet)

**No fixed-dimension vector space.** You can't do cosine similarity in v2 because there's no fixed vector. This means no off-the-shelf nearest-neighbor search, no dimensionality reduction visualization, no PCA. If you need these, you compute a snapshot vector on-demand from the raw node data — but the vector shape varies per word based on evidence depth.

**No pre-contact form comparison without character analysis.** v1's formVec gave you 16 dimensions of phonotactic structure for any word, even unseen ones. v2 gives you edit distance and bigram overlap. Less rich, but not hardcoded.

**Slower comparison.** v1 cached 60D vectors and compared via dot product. v2 recomputes from raw data every time. Cost: proportional to neighborhood size. For small vocabularies (<1000 words), negligible. For large ones, consider caching comparison results (not vectors — there are no vectors to cache).

---

## Migration Path

v2 doesn't replace v1 immediately. The path:

1. Build v2 engine with feed(), compare(), scoreSentence(), affinity(), correct()
2. Feed the same medical/legal/cooking corpus to both v1 and v2
3. Run identical tests: correction accuracy, sentence scoring, affinity discrimination
4. Compare results. Where v2 matches or beats v1, the hardcoded features weren't needed.
   Where v1 wins, ask: is it winning because the feature was right, or because 
   it had more data compressed into fewer dimensions?
5. The answer tells you what to teach and what to let emerge.

---

## The Governing Law (v2)

The embryo has no law yet. That is the point. The law emerges from what grows.

But if pressed: **Exposure creates structure. Structure enables comparison. Comparison reveals meaning. Meaning was never in the engine. It was always in the space between two words, measured by whatever evidence exists.**
