# SPEC: Simulation Mode for ARC-AGI

## Problem
ARC tasks require spatial reasoning — detecting transforms from example pairs, then applying them to test inputs. A geometric detector handles simple global transforms (rotation, reflection, color mapping). But many tasks involve **local rules that propagate** — fill enclosed regions, extend patterns, complete symmetry, gravity. These need simulation, not cell-by-cell detection.

## Architecture: Interrogation + Simulation

### Interrogation (Measurement)
Given example (input, output) pairs, detect the transform rule:
- Compare input/output grids structurally
- Try each detector in order of specificity
- Return a `Rule` describing what transform was detected

### Simulation (Propagation)
Once a rule is detected, propagate it as a wave across the test grid:
- Initialize output grid from test input
- Seed the wave from anchor cells (cells determined by direct rule application)
- Propagate: each wave step determines new cells from already-determined neighbors
- Continue until grid is fully determined or wave stabilizes

**Hamiltonian analogy**: K = transform rule (detected), V = grid constraints (boundaries, existing cells). Simulate = evolve system forward under H.

## Transform Detectors (Ordered by Specificity)

1. **Identity** — output == input
2. **Color map** — bijective color substitution
3. **Rotation** — 90/180/270 degree rotation
4. **Reflection** — horizontal/vertical/diagonal flip
5. **Transpose** — matrix transpose
6. **Scaling** — integer upscale/downscale
7. **Gravity** — colored cells fall in a direction until hitting boundary/other cell
8. **Crop/Extract** — output is a subgrid of input
9. **Tile/Repeat** — output is input tiled N x M
10. **Fill enclosed** — flood fill within boundaries
11. **Pattern extension** — periodic pattern propagated
12. **Symmetry completion** — partial symmetry completed
13. **Object transform** — detect objects, apply per-object transforms
14. **Conditional cellular** — local rules based on neighborhood (cellular automaton)
15. **Compositional** — sequence of simpler transforms

## Simulation Wave Types

| Wave | Seed | Propagation |
|------|------|-------------|
| Flood fill | Boundary cells | BFS from boundary, fill color until wall |
| Pattern | Known period cells | Extend pattern by period in each direction |
| Mirror | Symmetry axis cells | Reflect across detected axis |
| Gravity | Top/bottom row | Scan column, drop/rise colored cells |
| Transform | Corner/center | Apply rotation/scale from fixed point |
| Cellular | All cells with full neighborhood | Apply local rule, expand frontier |

## Implementation

### Files
- `test_arc.py` — main: load tasks, detect, simulate, evaluate
- `data/arc/` — ARC-AGI JSON task files

### Data Structures
```python
Grid = List[List[int]]           # 2D grid of color values (0-9)
Rule = NamedTuple('Rule', [...]) # Detected transform
Wave = deque of (row, col)       # BFS frontier
```

### Pipeline
```
for each task:
    examples = task['train']      # (input, output) pairs
    test_input = task['test'][0]['input']
    
    # Phase 1: Interrogation
    rule = detect_transform(examples)
    
    # Phase 2: Simulation  
    predicted = simulate(test_input, rule)
    
    # Phase 3: Verify
    correct = predicted == expected_output
```

## Success Metric
- Baseline (geometric detector only): ~13/400
- Target (with simulation): 40+/400 (10%+)
