# SHIFU v2 — The Embryo Engine

## For Claude Code: Read this ENTIRELY before writing any code.

---

## What This Is

A language engine where NOTHING is hardcoded except OCR character topology (physics of ink). No fixed dimensions. No channels. No routing weights. Structure emerges entirely from text exposure. Compare weights grow with evidence — a word seen once gets surface-level comparison, a word seen 1000 times gets deep relational analysis.

The engine knows its own shape via **pressure**: negative pressure = vacuum (the graph predicts richness that doesn't exist yet, it PULLS), positive pressure = surplus (richer than surroundings expect, it PUSHES).

## What Already Exists and Works (46/46 tests passing)

```
shifu-v2/
├── CLAUDE.md           ← THIS FILE
├── package.json        ← v2.0.0
├── tutor_run.js        ← Start a teaching session with one command
├── core/engine.js      ← The embryo (ShifuEmbryo class)
├── core/teacher.js     ← The feedback loop (Teacher class)
├── core/tutor.js       ← LLM-powered autonomous teaching (Tutor class)
├── test/suite.js       ← Engine tests (152 passing)
└── test/teacher.js     ← Teacher tests (22 passing)
```

### Engine Methods (all implemented, all tested):

| Method | What it does |
|--------|-------------|
| `feed(sentence)` | The perturbation. Records: freq, positions, gaps, neighbors, next/prev/next2. Returns token count. |
| `feedText(text)` | Splits on `.!?\n`, feeds each sentence. Returns {sentences, tokens}. |
| `depth(word)` | Evidence measurement. Returns {level, evidence}. Levels: unborn → surface → shallow → forming → structured → deep. |
| `compare(a, b)` | ASYMMETRIC. Weights grow with evidence. Returns {similarity, signals, weights, totalWeight, depth}. |
| `affinity(a, b)` | Pre-contact attraction. 6 signals. ASYMMETRIC. Returns {afAB, afBA, mutual, asym}. |
| `scoreSentence(text)` | Walks word-by-word. 3 signals + affinity gate. Returns {steps, coherence, meanSurprise}. |
| `pressure()` | The graph's shape. Returns {word, pressure, inbound, actual, closure}. Negative = vacuum. |
| `vacuums(k)` | Where the graph is pulling — needs more data. |
| `surpluses(k)` | Where the graph is pushing — hubs. |
| `bridges(k)` | Low closure — connecting disconnected neighborhoods. |
| `unlearn(a, b)` | Active forgetting. Halves edges between two words. Call repeatedly to sever. |
| `forget(word)` | Nuclear forgetting. Removes node and ALL references from entire graph. |
| `decay(threshold)` | Passive forgetting. Prunes all edges with count ≤ threshold. |
| `correct(garbled, k)` | OCR correction using topology table. |
| `similar(word, k)` | Most related words by compare(). |
| `stats()` | {version, vocab, sentences, tokens, depths}. |
| `serialize()` / `deserialize(json)` | Full round-trip. |

### Tutor Methods (LLM → Teacher → Engine autonomous loop):

| Method | What it does |
|--------|-------------|
| `round()` | One teaching round: diagnose → ask LLM to generate text for highest priority need → lesson → drill → check bridges for contamination → report. |
| `session(domain, numRounds, options)` | Full teaching session. Runs N rounds with delay between. Reports vocab growth, depth progress, remaining needs. |
| `evaluate()` | Asks LLM to generate test sentences (correct + wrong). Scores each with Shifu. Reports accuracy — does Shifu correctly distinguish valid from invalid? |

**Three providers supported:**
- `anthropic` — Claude API (needs ANTHROPIC_API_KEY)
- `openai` — GPT API (needs OPENAI_API_KEY)
- `ollama` — Local Gemma/Llama (free, unlimited, no API key)

**Run a teaching session:**
```bash
# With Claude (API key required)
ANTHROPIC_API_KEY=sk-... node tutor_run.js medical 20

# With local Gemma (free, unlimited)
# First: ollama pull gemma3:4b
node tutor_run.js medical 50 ollama gemma3:4b

# With local Llama
node tutor_run.js neurology 100 ollama llama3.2:3b
```

### Teacher Methods (closes the feedback loop):

| Method | What it does |
|--------|-------------|
| `diagnose()` | Reads pressure, finds vacuums, bridges, underexposed words, starved words. Returns plain-English needs. |
| `lesson(text, domain)` | Directed feed + immediate decay + reports what changed: new words, depth shifts, filled vacuums. |
| `drill(sentence)` | Contrastive pair: feeds sentence, scores forward AND reversed. Reports asymmetry — the direction IS the lesson. |
| `correctAssociation(a, b, rightContext)` | Unlearns wrong pair, feeds right context, measures before/after. |
| `plan()` | Returns prioritized steps: P1 fill starved, P2 reinforce underexposed, P3 investigate bridges, P4 drill contrastive, P5 decay. |
| `cycle(text, domain)` | Full iteration: diagnose → lesson → drill → diagnose. Reports before/after state. |
| `history()` | Lesson count, correction count, last 20 log entries. |

### Node Structure (each word stores only raw observations):

```javascript
{
  chars: "doctor",        // the raw character sequence
  freq: 14,              // times seen
  firstSeen: 1,          // sentence number
  lastSeen: 19,          // sentence number  
  positions: [0.0, ...], // relative position in each sentence (0→1)
  gaps: [3, 1, ...],     // sentences between re-encounters
  sentLengths: [8, ...], // length of each sentence it appeared in
  neighbors: {patient: 7, treats: 4, ...},  // co-occurrence ±3 window
  next: {treats: 4, prescribed: 5, ...},    // first-order forward
  prev: {the: 3, ...},                      // first-order backward
  next2: {treats: {patient: 1}, ...},       // second-order forward
}
```

---

## YOUR TASK: Build the PWA Chatbot

Build a single-file React chatbot (like the v1 `shifu_engine.jsx`) that embeds the v2 engine. One conversation. One input. Natural language.

### Requirements:

1. **Embed the ShifuEmbryo class** from `core/engine.js` directly in the JSX (for standalone artifact use).

2. **IndexedDB persistence** — NOT localStorage. Use this pattern:
```javascript
const IDB = {
  open() { return new Promise((res, rej) => { const r = indexedDB.open("shifu-v2", 1); r.onupgradeneeded = () => r.result.createObjectStore("state"); r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); }); },
  async get(key) { const db = await this.open(); return new Promise((res, rej) => { const tx = db.transaction("state", "readonly"); const r = tx.objectStore("state").get(key); r.onsuccess = () => res(r.result || null); r.onerror = () => rej(r.error); }); },
  async set(key, val) { const db = await this.open(); return new Promise((res, rej) => { const tx = db.transaction("state", "readwrite"); tx.objectStore("state").put(val, key); tx.oncomplete = () => res(); tx.onerror = () => rej(tx.error); }); },
  async del(key) { const db = await this.open(); return new Promise((res, rej) => { const tx = db.transaction("state", "readwrite"); tx.objectStore("state").delete(key); tx.oncomplete = () => res(); tx.onerror = () => rej(tx.error); }); },
};
```

3. **Intent parser** — detect these commands from natural language:
   - `feed: [text]` or just paste text → learn it
   - `explore [word]` or type single word → show depth, neighbors, next words, similar
   - `compare [a] and [b]` → show all signals, directional, trajectory
   - `affinity [a] and [b]` → show orbit, pull, indirect, mutual
   - `correct [text]` → OCR correction
   - `score: [sentence]` → walk with surprise bars
   - `pressure` or `vacuums` or `surpluses` → show where graph pulls/pushes
   - `bridges` → show structural bridges
   - `unlearn [a] and [b]` → weaken wrong association
   - `forget [word]` → remove a word entirely
   - `decay` or `decay [threshold]` → passive forgetting, prune weak edges
   - `diagnose` → what does the engine need? Shows vacuums, bridges, underexposed, starved words
   - `plan` → prioritized teaching steps based on pressure map
   - `lesson: [text]` → directed feed with immediate scoring and decay
   - `drill: [sentence]` → contrastive pair: scores forward AND reversed
   - `cycle: [text]` → full iteration: diagnose → lesson → drill → diagnose
   - `teach correct [a] and [b]` → unlearn wrong association, optionally feed right context
   - `history` → lesson count, correction count, recent log
   - `stats` → show vocabulary, depth distribution
   - `help` → list commands
   - `reset` → clear engine

4. **Rich inline visualizations** (in chat bubbles):
   - **Explore**: depth badge (colored by level: unborn=gray, surface=dim, shallow=blue, forming=yellow, structured=green, deep=bright green), neighbor list, next-word list
   - **Compare**: signal bars for each evidence type, directional arrows for expectsAB/BA
   - **Affinity**: 6 signal bars (orbit, pull→, ←pull, ind→, ←ind, charSim, expOvlp, posAlign)
   - **Score**: per-word surprise bars (green < 0.3, yellow < 0.7, red), ⚡ for affinity-gated words, coherence header
   - **Pressure**: colored bars (red for negative/vacuum, green for positive/surplus), word labels
   - **Bridges**: word with closure percentage

5. **UI Style**:
   - Dark theme: bg #0a0e14, panel #0e1219, border #1a2130, text #a8b4c4, accent #48b89a
   - Font: IBM Plex Mono (mono), Instrument Serif (header)
   - Chat bubble layout: user on right (blue tint), bot on left (panel bg)
   - Header: "Shifu v2" + "embryo · pressure-driven" + word count

6. **The depth indicator is critical.** Every explore response must show the word's depth level with a color badge. This is what makes v2 different from v1 — the engine KNOWS how much it knows about each word.

---

## What NOT to do

- Do NOT add fixed dimensions, channels, or routing weights
- Do NOT add `formVec()`, `contextVec()`, or any `*Vec()` methods
- Do NOT add a curiosity/boredom timer
- Do NOT add consciousness modules
- Do NOT use localStorage (use IndexedDB)
- Do NOT add neural networks or learned parameters
- Do NOT import external AI/ML libraries
- Do NOT make compare() symmetric
- Do NOT pretend to know what hasn't been observed

---

## The Governing Principle

Exposure creates structure. Structure enables comparison. Comparison reveals meaning. Pressure tells the engine where to grow. The engine doesn't pretend to know what it hasn't observed. It says "surface" and means it.

---

## Running Tests

```bash
cd shifu-v2
npm test
# Engine: 152/152, Teacher: 22/22 — total 174
```

## Key Test Results

```
doctor→treats: 25.0%, treats→doctor: 0.0%            (asymmetric ✓)
doctor→treats→?: {"patient": 1}                        (trajectory ✓)
doctor↔patient affinity: 0.1503                        (same-domain ✓)
doctor↔skillet affinity: 0.0067                        (cross-domain ✓)
"doctor treats patient" coherence: 0.5327               (natural order ✓)
"patient treats doctor" coherence: 0.3037               (reversed lower ✓)
seisure → seizure (conf 0.433)                          (OCR correction ✓)
Bridges: doctor(closure=0.27)                           (structural bridge ✓)
unlearn(doctor, skillet): direct edge severed            (active forgetting ✓)
forget(zygapophysis): all refs scrubbed from 90+ nodes  (nuclear forgetting ✓)
decay(1): 1025 weak edges removed                       (passive forgetting ✓)
drill "doctor treats patient": asymmetry 0.262          (contrastive learning ✓)
cycle: diagnose → lesson → drill → diagnose             (closed loop ✓)
plan: P1 fill starved → P2 reinforce → P3 bridges      (adaptive teaching ✓)
```
