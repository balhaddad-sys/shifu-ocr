# SHIFU — A Stone Etched by Exposure

## For Claude Code: Read this ENTIRELY before writing any code.

---

## What Shifu IS

A stone. Before any data arrives, the stone has grain — 16 dimensions of structure (Form) that exist in every word before it has ever been seen in context. That is the baby state. The steady state before perturbation.

`feed()` is the perturbation. Text hits the stone. Each sentence etches: co-occurrence neighborhoods deepen, positional patterns wear grooves, sequential expectations form channels, the self-model baseline shifts. The stone doesn't compute language. Language is the pattern of etchings left by exposure.

The 7 channels are not modules. They are faces of the same stone showing different marks from the same perturbation. Convergence at 99.93% is the stone finding its steady state — same chisel strokes in different order produce the same etchings, because the grain determines where marks can form.

The engine IS the perspective. A medical engine and a legal engine are two different stones. Feed different text, develop different grain, see different meaning in the same word. A camel is transport or food depending on which stone you ask.

**60-dimensional vectors. 7 channels. No neural networks. No embeddings. No training loops. Pure deterministic feature construction from text exposure.**

---

## Architecture (v1.4.0, 60D)

### 7 Faces of the Same Stone

| Channel | Dims | What it reads from the stone |
|---------|------|------------------------------|
| **Form** | 0-15 (16D) | The stone before etching — word shape, CV alternation, consonant runs, rhythm. No corpus needed. The baby state. |
| **Context** | 16-27 (12D) | Which atoms displaced together — co-occurrence neighborhoods, concentrations, positional statistics of neighbors |
| **History** | 28-35 (8D) | How deep the groove, how recently cut — frequency, age, recency, gap regularity, positional stability |
| **Influence** | 36-43 (8D) | How the etching changed the surrounding surface — sentence complexity, novelty attraction, bridging, diversity |
| **Contrast** | 44-51 (8D) | How this groove differs from the stone's average — frequency/position/density deviation from self-model baseline |
| **Expectation** | 52-59 (8D) | Which direction the groove runs — forward/backward predictability, directional asymmetry, breadth |
| **Affinity** | computed | Which unetched regions will accept the chisel easiest — pre-contact attraction between words |
| **OCR** | scalar | Topology-weighted Levenshtein distance (22 confusion pairs like 0↔o, 1↔l, rn↔m) |

### Routing Weights

```
Correction: OCR 70% + Form 30% (all others 0%)
Meaning:    Form 20% + Context 20% + History 10% + Influence 15% + Contrast 15% + Expectation 20%
```

### Key Index Constants

```javascript
IDX = {
  form: [0, 16],    // 16 dimensions
  ctx:  [16, 28],   // 12 dimensions
  hist: [28, 36],   // 8 dimensions
  inf:  [36, 44],   // 8 dimensions
  con:  [44, 52],   // 8 dimensions
  exp:  [52, 60],   // 8 dimensions
  all:  [0, 60]     // full vector
}
```

---

## Core Engine Methods

### feed(sentence) → tokenCount
Processes one sentence. Updates: wf (word frequency), co (co-occurrence), wp (positions), bf (bigrams), fs/ls (first/last seen), eg (encounter gaps), ph (position history), nv (neighbor volatility), sl (sentence lengths), sn (sentence novelty), sd (sentence diversity), nx (next-word), px (prev-word), nx2 (second-order next-word). Invalidates vector cache.

### feedText(text) → { sentences, tokens }
Splits on `.!?\n`, feeds each sentence.

### formVec(word) → Float64Array(16)
Pure word shape. No corpus. Features: CV alternation rate, max consonant run, max vowel run, onset charge, coda charge, gradient, vowel ratio, rhythm variance, CV/VC/CC/VV transition ratios, CVC/VCV trigram ratios, character uniqueness, normalized length.

### contextVec(word) → Float64Array(12)
Corpus co-occurrence graph. Features: neighbor diversity, top-1 concentration, top-3 concentration, neighbor frequency mean/variance, neighbor position mean/variance, word position mean/variance, exclusivity (Jaccard distance from top neighbor), log frequency, co-occurrence density.

### historyVec(word) → Float64Array(8)
Temporal exposure. Features: log frequency, age (fraction of engine lifetime), recency, gap regularity, positional stability, mean position, neighbor volatility, confidence (capped frequency).

### influenceVec(word) → Float64Array(8)
Structural footprint. Features: mean sentence complexity, complexity variance, novelty attraction (fraction of rare sentence-mates), sentence diversity (unique/total ratio), bridging (1 - triangle closure among neighbors), positional asymmetry (|mean_pos - 0.5| * 2), structural weight variance (sd of neighbor frequencies), influence stability (early vs late sentence lengths).

### contrastVec(word) → Float64Array(8)
Identity through deviation from self-model. Features: frequency contrast (|word_freq - global_mean| / global_mean), positional contrast (|word_pos_mean - global_pos_mean|), positional specificity contrast, sentence-length contrast, neighborhood density contrast, exclusivity (Jaccard from most common word's neighborhood), surprise (1 - freq/max_freq), identity stability (early vs late positional sd).

### expectationVec(word) → Float64Array(8)
Sequential predictions. Features: forward predictability (top next-word concentration), forward diversity (unique next-words), backward predictability (top prev-word concentration), backward diversity, directional asymmetry (|forward - backward| predictability), breadth (total unique transitions), forward-backward overlap (1 - Jaccard of next/prev sets), expectation stability (top-3 concentration).

### vec(word) → Array(60)
Concatenation of all 7 channel vectors. Cached per word, invalidated on feed().

### compare(a, b, purpose, mask) → { routed, form, context, history, influence, contrast, expectation, ocr, full, expectsAB, expectsBA, directional, trajectoryAB, trajectoryBA, weights }
**ASYMMETRIC** — compare(a,b) ≠ compare(b,a) because:
- `expectsAB`: probability of b following a (from nx table)
- `expectsBA`: probability of a following b
- `directional`: |expectsAB - expectsBA|
- `trajectoryAB`: richness of a→b→? continuation (from nx2)
- `trajectoryBA`: richness of b→a→? continuation

### scoreSentence(text) → { words, steps, totalSurprise, meanSurprise, coherence }
**Dynamic expectation field.** Walks sentence word by word. At each step:
1. **Sequential surprise** — nx/nx2 lookup (was this word expected after previous?)
2. **Field surprise** — accumulated co-occurrence field from ALL preceding words (do preceding words collectively expect this word?)
3. **Novelty** — does this word bring new neighbors to the field?

The context field RESHAPES as it walks. Each word deposits its co-occurrence neighborhood into the field, actively reconfiguring what later words are "allowed to mean." This is the dynamic state transformation.

Combined: seqSurprise * 0.35 + trajSurprise * 0.30 + fieldSurprise * 0.35

Coherence = 1 - meanSurprise. High coherence = sentence matches engine's experience.

### correct(garbled, k) → { candidates, confidence, reject, lowConfidence }
Uses indexed candidate lookup (_idxLen + _idxBg), compares via correction routing, confidence gating at reject < 0.02, lowConfidence < 0.05.

### similar(word, k) → ranked array
Full vocabulary scan via meaning routing.

### affinity(a, b) → { a, b, formResonance, sharedOrbit, pullAB, pullBA, indirectAB, indirectBA, contrastAlignment, expectationOverlap, affinityAB, affinityBA, mutual, asymmetry }
**Pre-contact attraction.** Predicts which words will relate BEFORE they meet in a sentence. This is the only method that measures attraction between words that may never have co-occurred.

Five signals:
1. **Form resonance** (5%) — word shape similarity (symmetric)
2. **Shared orbit** (35%) — weighted Jaccard of co-occurrence neighborhoods. Rare shared neighbors count more (weighted by 1/log(freq)). DOMINANT signal.
3. **Trajectory pull** (20%) — does a appear in b's nx table? ASYMMETRIC. (pullAB ≠ pullBA)
4. **Indirect paths** (20%) — does a→?→b exist via any intermediate word? ASYMMETRIC.
5. **Contrast alignment** (5%) — do they deviate from baseline similarly?
6. **Expectation overlap** (15%) — do they predict similar futures?

ASYMMETRIC: affinityAB ≠ affinityBA because trajectory pull and indirect paths are directional. Relational signals (shared orbit 35% + indirect 20% + pull 20% = 75%) dominate over surface signals (form 5% + contrast 5% = 10%).

Use case: predict which words from a new domain will integrate easily into the engine's existing structure, before feeding the new text.

### serialize() / deserialize(json)
Full state including nx, px, nx2, self-model globals, all history arrays. Config stored with every serialization.

---

## Self-Model (Engine Baseline)

The engine maintains running statistics about itself:
- `_globalPos[]` — all positional observations (rolling window 500)
- `_globalSentLen[]` — all sentence lengths (rolling window 200)
- `_globalFreqSum` — total frequency mass
- `_globalWordCount` — vocabulary size

The Contrast channel measures each word against THIS baseline. A word that deviates from the engine's own expectations has high contrast = strong identity. A word that matches the baseline is unsurprising = weak identity.

Different engines develop different baselines. Same word, different identity. The engine IS the perspective.

---

## Performance

- **Indexed correction**: `_idxLen` (words by length ±2), `_idxBg` (words by shared bigrams). Correction does NOT scan full vocabulary.
- **Vector cache**: `_cache` per word, invalidated on every feed().
- **History trimming**: All arrays capped at 20-100 entries to prevent unbounded growth.

---

## OCR Topology Table (22 pairs)

```
0↔o: 0.1, 1↔l: 0.2, 1↔i: 0.2, 5↔s: 0.3, 8↔b: 0.3, 6↔g: 0.4,
l↔i: 0.2, m↔n: 0.4, u↔v: 0.5, c↔e: 0.5, r↔n: 0.3, d↔o: 0.3,
f↔t: 0.4, h↔b: 0.4, a↔e: 0.4, a↔o: 0.4, u↔n: 0.4, e↔i: 0.4,
f↔l: 0.4, s↔e: 0.5, b↔d: 0.4
```

---

## PWA Requirements

### The app must be a single integrated JavaScript PWA with:

1. **Offline-capable** — service worker caches all assets. Engine runs entirely client-side.
2. **IndexedDB persistence** — engine state saved/loaded via IndexedDB (not localStorage, not file system).
3. **Multi-engine** — user can create/switch between named engines (medical, legal, cooking, custom).
4. **10 tabs** matching the existing React app but updated to 60D:
   - **Feed** — paste text or upload .txt, feed to current engine
   - **Explore** — enter word, see all 7 channel vectors as bar charts, top-10 similar words
   - **Compare** — two words, see all channel scores + directional signals (expectsAB, expectsBA, trajectoryAB, trajectoryBA)
   - **Affinity** — two words, see pre-contact attraction: shared orbit, trajectory pull, indirect paths, form resonance, contrast alignment, expectation overlap. Asymmetric visualization.
   - **Correct** — paste garbled text, see corrections with confidence gating
   - **Wave** (Disturb) — enter word, see how removing it changes sentence structure
   - **Benchmark** — run correction pairs + semantic pairs + baselines
   - **Observe** — 4 experiments: separation onset, convergence, transfer (correction + meaning), channel necessity
   - **Look Back** — formed/unformed classification, discrimination test, clustering
   - **Crystallization** — track vector drift over exposure epochs
   - **Score** (NEW) — enter sentence, see step-by-step surprise/coherence with dynamic field visualization
   - **Stats** — vocabulary, tokens, sentences, mature words, engine config

### UI Style
- Dark theme: background #0a0e14, panel #0e1219, text #a8b4c4, accent #48b89a
- Font: IBM Plex Mono (monospace), Instrument Serif (headers)
- Minimal, clinical, no decorative elements
- Bar charts for vectors, line charts for crystallization/separation

### Key UI features
- **Score tab**: show each word as a step with surprise bar, field support indicator, cumulative surprise line chart, coherence score. This is the most important new visualization.
- **Compare tab**: must show asymmetric signals. "doctor → treats" should visually differ from "treats → doctor". Show expectsAB/expectsBA as directional arrows.
- **Explore tab**: 7 channel panels (not 3). Each channel has its own color and dimension labels.
- **Engine selector**: dropdown in header, create new engine, switch between engines.

---

## Test Expectations (58 tests must pass)

The test suite in `test/suite.js` covers:
- Distance functions (Levenshtein, OCR topology)
- PRNG determinism + seeded shuffle
- All 7 vector dimensions (16+12+8+8+8+8 = 60)
- Routing weights (correction vs meaning)
- Correction accuracy (seisure→seizure, bi1ateral→bilateral)
- Confidence gating
- Candidate indexing
- Serialization round-trip (including nx2, self-model)
- Convergence across training orders (>80%)
- Asymmetric compare (doctor→treats ≠ treats→doctor)
- Second-order trajectory (doctor→treats→patient exists, patient→treats→? does not)
- Sentence scoring: "doctor treats patient" more coherent than "patient treats doctor"
- Dynamic field: natural word order more coherent than scrambled
- Field support: "patient" after clinical context has low field surprise

---

## Files in this project

```
shifu/
├── CLAUDE.md          ← THIS FILE (read first)
├── package.json
├── core/
│   └── engine.js      ← Pure engine, 60D, 7 channels, all methods
├── api/
│   └── server.js      ← Express API, pluggable persistence/rate-limiting
├── test/
│   └── suite.js       ← 58 tests
├── public/
│   ├── index.html      ← PWA shell
│   ├── manifest.json   ← PWA manifest
│   └── sw.js           ← Service worker
└── docs/
    └── PHILOSOPHY.md   ← Governing law + principles
```

---

## What NOT to do

- Do NOT add "consciousness modules" or quantum vocabulary
- Do NOT rename existing methods to sound more impressive
- Do NOT add neural networks or learned weights
- Do NOT break the 60D vector structure
- Do NOT make compare() symmetric again
- Do NOT use localStorage (use IndexedDB)
- Do NOT add dependencies beyond express (backend) and vanilla JS/React (frontend)

---

## Philosophy (for UI text, not code)

- "Form can exist alone. Understanding cannot."
- "Understanding is measured not by agreement, but by relational discrimination."
- "A thing knows itself by what it is not."
- "The brain expects a finite set of next words. That expectation IS the directional structure."
- "The engine IS the perspective. Different engines see different meaning."
- "The data is what it is."
