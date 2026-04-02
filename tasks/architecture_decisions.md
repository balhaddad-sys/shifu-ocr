---
name: Shifu Architecture Decisions
description: All changes that produced real learning. Episodic-first, 5-checkpoint relay, Rust core, reactive plasticity, convergence-by-constraint. DO NOT lose these.
type: project
---

## Core Principle: Convergence by Constraint

Meaning is not created. It is selected by constraint.
Think globally. Converge locally.

- The graph is wide (globally open)
- The path is narrow (locally constrained)
- Structure gives truth. Resonance gives efficiency.
- Don't predict. React. Change when needed.

## The Three Pillars

1. **Structured field** (co-graph + tags) — minimal but accurate
2. **Pattern primitives** — reusable mechanisms, not memorized facts
3. **Constraint system** — goal + tags + weights collapse uncertainty into one stable path

## Danger: Wrong Inevitability

If the graph is wrong, the system converges confidently to wrong answers. Especially dangerous in medical reasoning. Guard: gate filtering, pathway isolation, noise pruning, curriculum validation.

## React, Don't Predict

The speaker doesn't construct sentences. It activates the field and reads what survives. Stimulus → field activates → output IS the activation, ordered by learned transitions. No trajectory planning. No motor programs. No caching. Pure reactive plasticity.

**Why:** Acting has infinite results (by chance). Reacting is direct and specific (by structure). Executing is not the same as knowing. Producing a desired result should be inevitable, not lucky.

## Episodic-First Memory

Feed only stores episodes (raw text + tokens + epoch). Semantic extraction happens during REPLAY in delta/theta states. The brain remembers WHERE/WHEN/WITH WHO first, semantics emerge during sleep.

## 5-Checkpoint Relay

Not everything goes to cortex raw. Each stream earns passage:
- CP1: Specialist receptors (Golgi tags, pathway classification)
- CP2: Local preprocessing (co-graph, nx-graph, breadth)
- CP3: Relay/gating (TRN decides what passes, neuroglia regulates heat)
- CP4: Task-ready handoff (Rust neural field wiring, speaker transitions)
- CP5: Higher integration (identity, spokes, syntax — only when lower layers stable)

## Rust Neural Core (shifu_neural)

PyO3 extension. Arena-indexed neurons (word→u32), zero string hashing in hot path.

## Developmental Feeding

Don't feed an elephant to a baby. Bite size scales with vocabulary:
- vocab < 50: milk (5 sentences)
- vocab < 500: puree (20)
- vocab < 5000: solid (100)
- vocab > 5000: adult (500)
Each bite immediately replayed (digested).

## Relationship-Tagged Edges

One graph, tagged edges. Not 5 separate spoke layers.
- `_co_graph` = strength (weight layer)
- `_co_tags` = meaning (semantic relation layer: id, app, fn, mech, rel)
- Goal shapes traversal through `edges_by_goal()` with soft scoring

## Neuroglia System

Astrocytes (thermal regulation), Microglia (immune/pruning), OPCs (neurogenesis). Each spoke has different thermal tolerance.

## TRN (Thalamic Reticular Nucleus)

Attentional spotlight. Each spoke has a TCR channel. Non-attended channels suppressed during deliberation.

## Fatigue (Adenosine Model)

Conviction pushes through plateaus but not forever. Adenosine builds with failed bridge attempts. At threshold → forced temporary satisfaction + rest.

## Passive Learning on CPU Time

Piggybacked on every command processed. Rust heartbeat every beat. Python work (AUC, replay, consolidation, study) every 100th beat.

## No Threading

On Windows with piped stdin, NEVER use threads. Blocking readline in main loop. Process command → learn → respond → block again.

## PYTHONUNBUFFERED

Node spawns Python with `-u` flag AND `PYTHONUNBUFFERED=1`.

## No Hardcoded Knowledge

ZERO hardcoded word lists. Everything emergent from data. Noise filter uses vowel ratio + letter distribution, not lists.

## Pattern Extraction (Curriculum)

25K words → reusable patterns ("X is a Y", "X causes Y"). Curriculum generates, scores against episodes, extracts structural templates. Transfer testing: apply pattern to new concept.

## Minimal Storage + Maximal Transformation

Don't store every fact. Store:
1. Core patterns (shared structure)
2. Anchors (ground truth nodes)
3. Relationships (tagged edges)
Resonance gives efficiency. Structure gives truth.
