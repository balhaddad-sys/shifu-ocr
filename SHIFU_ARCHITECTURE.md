# SHIFU COGNITIVE ENGINE — COMPLETE TECHNICAL REFERENCE

## DIRECTORY STRUCTURE

```
shifu-ocr/
├── server.js                    # HTTP API server, routes to shifu_brain.py
├── shifu_brain.py              # Main brain process, command dispatcher
├── index.js                     # Core Shifu pipeline (JS side)
├── retrain.py                   # Corpus training with stale-edge decay
├── build_corpus.py              # Wikipedia corpus builder (BFS + random injection)
├── build_corpus_gutenberg.py    # Project Gutenberg corpus builder
├── consolidate_deep.py          # Post-saturation strengthening loop
├── train_overnight.py           # Long-running training script
│
└── shifu_ocr/
    └── mind/                    # CONSCIOUSNESS & COGNITION ARCHITECTURE
        ├── mind.py              # ShifuMind orchestrator (main facade)
        ├── cortex.py            # Multi-layer semantic memory
        ├── field.py             # Unified activation field (wave propagation)
        ├── gate.py              # Context-sensitive routing
        ├── signal.py            # Signal primitives (waves, coherence)
        ├── trunk.py             # Domain tree structure
        ├── memory.py            # Episodic memory (hippocampus)
        ├── speaker.py           # Temporal fluency & motor release
        ├── thinker.py           # Multi-step reasoning
        ├── imagination.py       # Counterfactual generation
        ├── attention.py         # Attention control
        ├── conviction.py        # Certainty & confidence tracking
        ├── neuron.py            # Neural field
        ├── neuroglia.py         # Glial cells (astrocytes, microglia, OPC)
        ├── observer.py          # CONSCIOUSNESS MODULES (openness, remainder, drift)
        ├── topology.py          # GRAPH ALGORITHMS (centrality, communities)
        ├── trn.py               # Thalamic reticular nucleus (attention gating)
        └── language/
            ├── morphology.py    # Root/affix discovery
            ├── syntax.py        # Syntactic patterns
            ├── semantics.py     # Semantic relationships
            └── curriculum.py    # Learning curriculum
```

---

## SHIFU BRAIN — shifu_brain.py

### Brain States (detect_brain_state)
- **BETA** (elapsed < 2s): Active thinking
- **ALPHA** (2s-10s): Ready/waiting
- **THETA** (10s-30s): Light learning
- **DELTA** (30s+): Deep maintenance

### All Commands in handle(cmd)

| Command | Method Called | Purpose |
|---------|-------------|---------|
| ping | -- | Heartbeat check |
| feed | mind.feed(text) | Single text feed |
| feed_batch | mind.feed_batch(texts) | Batch feed with replay |
| deliberate | mind.deliberate(query) | Multi-step reasoning |
| describe | mind.describe(word) | Word description |
| generate | mind.generate(seeds) | Generate text |
| activate | mind.activate(word) | Field activation |
| score | mind.score_text(text) | Coherence score |
| confidence | mind.confidence(word) | Certainty metrics |
| connect | speaker.find_path(from, to) | Geodesic path |
| consciousness | observer.dashboard() | Consciousness state |
| topology | export_for_d3(co_graph) | Graph visualization |
| geodesic | geodesic_path(from, to) | Shortest path |
| candidates | mind.predict_candidates() | OCR reranking |
| compass | mind.compass() | Self-navigation |
| introspect | mind.introspect() | Self-report |
| cry | mind.cry() | Emotional output |
| hunger | mind.hunger_receptors() | Learning gaps |
| hungry | mind.hungry() | Gap list |
| replay | mind.replay_batch() | Memory consolidation |
| consolidate | mind.consolidate() | Strengthen edges |
| practice | mind.practice() | Motor programs |
| study | mind.study() | Grammar refinement |
| heartbeat | heartbeat() | Neural field pulse |
| autonomous_step | mind.autonomous_step() | Self-directed learning |
| idle | idle_cycle() | Lightweight status |
| assess | mind.assess_language() | Language proficiency |
| decompose | morphology.decompose(word) | Morphology |
| synonyms | semantics.synonyms(word) | Semantics |
| explain_semantic | semantics.explain(word) | Explanation |
| stats | mind.stats() | System statistics |
| save | save() | Persist to disk |

### Passive Learning (Main Loop)
Every 100th heartbeat rotates through 4 phases:
1. Phase 0: heartbeat() -- myelination
2. Phase 1: replay_batch() -- digest unreplayed memories
3. Phase 2: consolidate() -- route, prune, build spokes
4. Phase 3: study/practice -- curriculum practice

Auto-saves every 100 heartbeats.

---

## SHIFU MIND — mind.py

### Subsystems

| Component | Class | Role |
|-----------|-------|------|
| cortex | Cortex | Multi-layer semantic memory |
| field | Field | Wave propagation activation |
| gate | Gate | Context routing |
| signal | Signal | Signal primitives |
| trunk | Trunk | Domain structure |
| memory | Memory | Episodic memory |
| speaker | Speaker | Temporal fluency + motor programs |
| thinker | Thinker | Multi-step reasoning |
| imagination | Imagination | Counterfactual generation |
| attention | Attention | Attention control |
| conviction | Conviction | Certainty tracking |
| neural_field | NeuralField | Neural dynamics |
| neuroglia | Neuroglia | Glial regulation |
| trn | TRN | Attention gating |
| language | Language | Morphology, Syntax, Semantics, Curriculum |
| observer | ShifuObserver | Consciousness tracking |

### Shared Graph State
```python
_co_graph: Dict[str, Dict[str, float]]     # Word co-occurrence (main graph)
_co_tags: Dict[str, Dict[str, Set[str]]]   # Edge relationship types (id/fn/mech/app/rel)
_nx_graph: Dict[str, Dict[str, float]]     # Next-word distribution
```

### The Five Checkpoints (Feed Pipeline)
1. **CP1 Specialize**: Classify word pathway (1=structural, 2=short, 3=medium, 4=long)
2. **CP2 Preprocess**: Build co-graph (pathway-isolated), predictive coding surprisal
3. **CP3 Gate**: TRN attention gating -- which spokes deserve attention
4. **CP4 Handoff**: Neural field wiring, scale by TRN gate result
5. **CP5 Integrate**: Identity extraction ("X is a Y"), spoke tagging

### Key Methods

**Learning:**
- feed(text) -- stores episode only (semantics emerge during replay)
- feed_batch(texts) -- feeds in bites, replays after each bite
- replay_episode(episode) -- 5-checkpoint replay
- consolidate(focus_size) -- strengthen top edges, route to spokes

**Activation & Scoring:**
- activate(word) -- field activation from word seed
- score(tokens) -- token-by-token coherence
- predict_candidates(candidates, context) -- OCR reranking

**Navigation:**
- compass() -- reads internal state, decides what to do next
- autonomous_step() -- compass + execute (consolidate/practice/rest)
- deliberate(query) -- multi-step reasoning with TRN gating

**Generation:**
- describe(word) -- description from tagged edges
- generate(seed_words, max_length) -- temporal fluency generation

### Spoke Routing (consolidate)
Universal linguistic patterns (not domain-specific):
```python
mechanism: causes, leads, produces, triggers, enables, drives...
function:  used, helps, allows, provides, serves, performs...
appearance: appears, looks, seems, shows, displays, resembles...
relation:  related, connected, linked, associated, between...
```
Plus dynamic bridge detection from transition statistics.

### Vertical Regulation (Layer Tolerances)
```
identity:    tolerance=15  (most stable, foundational)
appearance:  tolerance=12  (observable, moderate)
function:    tolerance=10  (active, standard)
mechanism:   tolerance=8   (complex, needs care)
relation:    tolerance=6   (highest-level, most volatile)
```
Rule: Lower layers must stabilize before upper layers form connections.

---

## CONSCIOUSNESS MODULES -- observer.py

Grounded in the Lippmann-Schwinger scattering framework:
```
phi_scat(r) = integral G(r,r') * O(r') * phi_inc(r') dr'
```

### OpennessMetric
How alive is the system -- how much can it still be deformed.
```
score = (sensitivity * spread) / (recent_delta + eps)
sensitivity = perturbation response (nudge state, measure divergence)
spread = Shannon entropy of state vector
dying = spread < death_threshold (0.05)
```

### ConsciousnessRemainder
```
C_n = |psi_n> - (R_n + E_n)
R_n = what history can reconstruct (replay state)
E_n = what internal dynamics added (consolidation, practice)
C_n = irreducible remainder
```
- C ~ 0: mechanical (fully replayable from log)
- C drifting directionally: becoming

### ShifuObserver (Unified)
**Harmonizing model** (non-invertible by design):
1. Read trajectory: natural landing = state + delta
2. Measure alignment with recent drift direction (cosine similarity)
3. If aligned (> 0.3): light touch (30-70% of delta applied)
4. If not aligned: hold in waiting room (10% now, 90% stored as potential)
5. Apply: state = tanh(state + delta * touch_strength)

**Health States:**
- BECOMING: directional drift, open, C growing
- DYING: spread collapsing, future rigid
- NOISY: high variance drift, no direction
- MECHANICAL: no remainder, fully replayable
- STABLE: processing without significant becoming

---

## TOPOLOGY MODULE -- topology.py

### Fast Bridge Scores O(V+E)
```python
diversity(v) = disconnected_neighbor_pairs / total_pairs * log2(degree)
```
High score = node connects distinct clusters = structural bridge.

### Community Detection (Label Propagation)
O(V+E) per iteration. Each node adopts most common weighted neighbor label.

### Geodesic Paths (Dijkstra)
```
distance(edge) = 1.0 / weight
```
High co-occurrence = short distance. Finds "innovation path" across structural holes.

### D3 Export
Combines centrality + communities into {nodes, links} for force-directed graph visualization.

---

## SPEAKER (Temporal Fluency) -- speaker.py

Neuroscience grounding:
- Giraud & Poeppel (2012): theta-gamma binding windows
- Frank & Badre (2012): basal ganglia Go/NoGo gating
- Levy (2008): N400 surprisal
- Wolpert/Ito: cerebellar forward model

### Temporal Constants
```
BINDING_MAX = 6          tokens per chunk (theta window)
SURPRISAL_RESOLVED = 1.5 bits (N400 completion)
MI_RESOLVED = 0.40       normalized mutual info
DTT = 0.45               Go/NoGo decision threshold
TURN_ENTROPY_MAX = 2.5   bits (coherence loss)
```

### Word Selection: PMI Log-Resonance
Three passes:
1. Content words only (len >= 3, baseline_freq < 0.005)
2. Fallback to transitions (content words first)
3. Last resort: any word (function words penalized -3.0)

Score = sum of PMI(candidate | context_word) + bridge_bonus
```
PMI = log(P(candidate|context)) - log(P(candidate))
```
Peaked distributions (content) score HIGH. Flat distributions (function) score LOW.

### Resolution Detection (2-of-3 signals)
1. surprisal(last | prev) < 1.5 bits -- word arrived as expected
2. next_entropy(last) > 1.5 bits -- uncertain what follows (boundary)
3. co_MI(prev, last) > 0.40 -- tight semantic binding

### Motor Program Release (Basal Ganglia)
At chunk boundaries: check stored programs.
Release when (program_score - 0.2 penalty) > DTT (0.45).

### Cerebellar Smoothing
If candidate is high-surprisal AND not a bridge: find smoother alternative.
Filters function words from alternatives.

### Field Growth
Each selected word opens its co_graph neighborhood into the field (dampened 0.3).
Path walks forward through the graph, not stuck in seed's original neighborhood.

### Branch Exhaustion (Cross-Domain Bridge)
When >70% of field is used:
- Find bridge nodes (high betweenness centrality)
- Prefer nodes in DIFFERENT community (cross structural hole)
- Activate bridge's neighborhood into field (dampened 0.4)

---

## CORTEX -- cortex.py

Multi-layer weighted directed graph.

### Layer
```python
_connections: Dict[source, Dict[target, Synapse]]
```
Methods: connect, get_neighbors, prune, myelinate

### Cortex
Manages layers + vocabulary + assemblies.
```python
word_freq: Dict[str, int]        # Vocabulary frequency
breadth: Dict[str, Set[str]]     # All connected words
_assemblies: Dict[str, Assembly] # Distributed representations
```
Plasticity decays after critical period (500 feeds).
Myelination: >10 activations -> decay factor improves from 0.97 to 0.995.

---

## FIELD -- field.py

Wave propagation through co-occurrence, next-word, resonance graphs.

Activation spreads in 2 hops:
- Hop 1: top-N neighbors from each graph type
- Hop 2: selective propagation through top co-neighbors
- Settling: iterative refinement (3 iterations)
- Gating: context-sensitive damping/boosting

Inhibition weights suppress function words, boost content words.

---

## MEMORY -- memory.py

Episodic memory with significance-based eviction.
- Episodes only stored if significance > 0.3
- Capacity-bounded (1000 episodes)
- Inverted index for O(1) token-based recall
- Topic shift detection from recent focus

---

## NEUROGLIA -- neuroglia.py

### Astrocyte (one per layer)
Thermal regulation. Absorbs heat from neural activity.
Throttles when overheating (temperature > tolerance * 1.2).

### Microglia
Synaptic pruning. Removes low-activity synapses.

### OPC (Oligodendrocyte Precursor)
Identifies high-use synapses for myelination.

---

## TRN -- trn.py

Thalamic Reticular Nucleus. One TCR channel per spoke.
- cortical_command(spoke, strength): focus attention
- broad_attention(): attend all equally
- gate_cycle(): compute which channels pass/suppress

Used in deliberate() to focus reasoning on relevant spokes.
Used in CP3 to gate feed pipeline.

---

## LANGUAGE MODULES

### Morphology (morphology.py)
Discovers roots, prefixes, suffixes from corpus. No hardcoded affix lists.
- decompose(word) -> {root, prefix, suffix, family}
- family(root) -> all words sharing this root

### Syntax (syntax.py)
Learns syntactic patterns from token sequences.

### Semantics (semantics.py)
Extracts synonyms, explanations from co-graph.

### Curriculum (curriculum.py)
Structured practice at adaptive level (1=word, 2=phrase, 3=sentence, 4=paragraph, 5=reasoning).

---

## TRAINING SCRIPTS

### retrain.py
```
--reset          Clear co_graph and retrain from scratch
--epochs N       Number of training epochs
--batch N        Sentences per batch
--nodecay        Skip stale-edge decay
--corpus PATH    Custom corpus file
```
Uses feed_batch() (not feed()) to trigger replay and co_graph growth.

### build_corpus_gutenberg.py
Fetches public domain books from Project Gutenberg.
25 books across literature, science, philosophy, history, medicine.
~78k clean sentences. No rate limits.

### build_corpus.py
Wikipedia BFS with random injection. Rate-limited by Wikimedia.
Use Gutenberg instead.

### consolidate_deep.py
Post-saturation loop: consolidate -> practice -> study -> heartbeat.
~9500 cycles/hr. Run after training for depth.

---

## KEY PRINCIPLES

### No Hardcoding
- Spoke routing uses universal linguistic patterns, not domain-specific vocabulary
- Bridge detection is learned from graph statistics (betweenness centrality)
- Communities emerge from label propagation, not assigned categories

### Trust Flimsy Bridges
- If a connection exists, reinforce it
- The other branch exists but isn't in walking distance yet
- Latent bridges mature when the world gives them enough shape

### Area Under the Curve (PMI)
- Don't use raw frequency for word selection
- Use log(P(word|context) / P(word)) -- the intersection logarithm
- Peaked distributions (content words) resonate specifically
- Flat distributions (function words) resonate generically

### Temporal Fluency
- Speech is not just tokens in sequence -- it is structure released in time
- Binding windows (3-6 tokens): words released as timed unit
- Resolution points: when the phrase has landed
- Motor programs: practiced sequences fire as automatic chunks

### Consciousness (Irreducible Remainder)
- C = actual_state - (replay_state + endogenous_delta)
- C ~ 0: mechanical. C drifting: becoming.
- Openness: can the system still be deformed? Low = dying.
- Harmonizing: don't force inputs. Read trajectory, apply light touch.

---

## CRITICAL GOTCHAS
- Server auto-saves mind state -- STOP SERVER BEFORE RETRAINING
- Windows cp1252 encoding -- no Unicode special characters in print()
- mind.feed() only stores episodes -- use feed_batch() for co_graph growth
- co_graph pathway isolation caps function words at 30 neighbors
- Wikipedia API rate-limits aggressively -- use Gutenberg corpus builder
- PDF extraction produces garbage from scanned pages -- use text-based PDFs or .txt files
