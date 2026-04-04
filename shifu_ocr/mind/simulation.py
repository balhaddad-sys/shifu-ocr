"""
Simulation mode for ARC-AGI: Hamiltonian forward propagation.

Interrogation = measurement (analyze training pairs, detect transform rule)
Simulation = propagation (apply rule as spatial wave, not cell-by-cell)

Instead of deciding each cell independently, detect the transform from
input->output examples and propagate it as a wave across the grid.

No hardcoded detector lists or propagator routing. Each detector
self-registers via @detector and returns a Rule that carries its
own propagation function. The solver discovers available detectors
at runtime.
"""

from typing import List, Dict, Tuple, Optional, Any, Set, Callable
from collections import deque
from itertools import product

Grid = List[List[int]]
Rule = Dict[str, Any]

# ---------------------------------------------------------------------------
# Detector registry -- detectors self-register, no hardcoded lists
# ---------------------------------------------------------------------------

_detector_registry: List[Callable] = []


def detector(fn: Callable) -> Callable:
    """Register a detection function. Detectors are tried in registration order."""
    _detector_registry.append(fn)
    return fn


# ---------------------------------------------------------------------------
# Grid utilities
# ---------------------------------------------------------------------------

def grid_eq(a: Grid, b: Grid) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        if ra != rb:
            return False
    return True


def grid_shape(g: Grid) -> Tuple[int, int]:
    rows = len(g)
    cols = len(g[0]) if rows > 0 else 0
    return (rows, cols)


def grid_copy(g: Grid) -> Grid:
    return [row[:] for row in g]


def grid_hash(g: Grid) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(row) for row in g)


# --- Geometric primitives (generators of the dihedral group + transpose) ---

def rotate_90(g: Grid) -> Grid:
    """Rotate grid 90 degrees clockwise."""
    rows, cols = grid_shape(g)
    out: Grid = [[0] * rows for _ in range(cols)]
    for r in range(rows):
        for c in range(cols):
            out[c][rows - 1 - r] = g[r][c]
    return out


def flip_h(g: Grid) -> Grid:
    """Flip horizontally (left-right)."""
    return [row[::-1] for row in g]


def flip_v(g: Grid) -> Grid:
    """Flip vertically (top-bottom)."""
    return g[::-1]


def transpose_grid(g: Grid) -> Grid:
    rows, cols = grid_shape(g)
    out: Grid = [[0] * rows for _ in range(cols)]
    for r in range(rows):
        for c in range(cols):
            out[c][r] = g[r][c]
    return out


def _compose(fns: List[Callable]) -> Callable:
    """Compose a list of grid transforms left-to-right."""
    def composed(g: Grid) -> Grid:
        result = g
        for fn in fns:
            result = fn(result)
        return result
    return composed


def _build_transform_group() -> List[Tuple[str, Callable]]:
    """Generate all distinct transforms by composing primitives.

    Instead of hardcoding [flip_h, flip_v, rot90, rot180, rot270, transpose],
    generate them by composing the 3 generators and dedup by effect on a
    probe grid. Discovers the full dihedral group + transpose compositions.
    """
    generators = [
        ("r", rotate_90),
        ("h", flip_h),
        ("v", flip_v),
        ("t", transpose_grid),
    ]

    # Probe grid: asymmetric so distinct transforms give distinct results
    probe: Grid = [
        [1, 2, 3],
        [4, 5, 6],
    ]

    seen: Set[Tuple[Tuple[int, ...], ...]] = set()
    # Identity
    seen.add(grid_hash(probe))

    transforms: List[Tuple[str, Callable]] = []

    # Try single ops, then compositions of 2, then 3
    for depth in range(1, 4):
        for combo in product(range(len(generators)), repeat=depth):
            name = "".join(generators[i][0] for i in combo)
            fns = [generators[i][1] for i in combo]
            composed = _compose(fns)
            result = composed(probe)
            h = grid_hash(result)
            if h not in seen:
                seen.add(h)
                transforms.append((name, composed))

    return transforms


# Build once at import time
_TRANSFORM_GROUP: List[Tuple[str, Callable]] = _build_transform_group()


def scale_grid(g: Grid, factor: int) -> Grid:
    """Scale grid by integer factor (each cell becomes factor x factor block)."""
    rows, cols = grid_shape(g)
    out: Grid = [[0] * (cols * factor) for _ in range(rows * factor)]
    for r in range(rows):
        for c in range(cols):
            val = g[r][c]
            for dr in range(factor):
                for dc in range(factor):
                    out[r * factor + dr][c * factor + dc] = val
    return out


def unique_colors(g: Grid) -> Set[int]:
    colors: Set[int] = set()
    for row in g:
        for c in row:
            colors.add(c)
    return colors


def color_counts(g: Grid) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for row in g:
        for c in row:
            counts[c] = counts.get(c, 0) + 1
    return counts


def background_color(g: Grid) -> int:
    """Most frequent color is background."""
    counts = color_counts(g)
    return max(counts, key=counts.get)


def connected_components(g: Grid, color: int) -> List[Set[Tuple[int, int]]]:
    """Find connected components of a given color (4-connected)."""
    rows, cols = grid_shape(g)
    visited: Set[Tuple[int, int]] = set()
    components: List[Set[Tuple[int, int]]] = []

    for r in range(rows):
        for c in range(cols):
            if g[r][c] == color and (r, c) not in visited:
                component: Set[Tuple[int, int]] = set()
                queue: deque = deque([(r, c)])
                visited.add((r, c))
                while queue:
                    cr, cc = queue.popleft()
                    component.add((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if (nr, nc) not in visited and g[nr][nc] == color:
                                visited.add((nr, nc))
                                queue.append((nr, nc))
                components.append(component)

    return components


def bounding_box(cells: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """Return (min_r, min_c, max_r+1, max_c+1) for a set of cells."""
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    return (min(rs), min(cs), max(rs) + 1, max(cs) + 1)


# ---------------------------------------------------------------------------
# DETECTORS: each registers itself and returns a Rule with a "propagate" key
#
# Rule contract:
#   rule["type"]  -- string name
#   rule["propagate"]  -- callable(Grid) -> Grid
#   rule[...]  -- any additional metadata
# ---------------------------------------------------------------------------

@detector
def detect_geometric(pairs: List[Dict]) -> Optional[Rule]:
    """Detect any geometric transform by trying all compositions of primitives."""
    for name, fn in _TRANSFORM_GROUP:
        all_match = True
        for pair in pairs:
            if not grid_eq(fn(pair["input"]), pair["output"]):
                all_match = False
                break
        if all_match:
            return {"type": "geometric", "transform": name, "propagate": fn}
    return None


@detector
def detect_scale(pairs: List[Dict]) -> Optional[Rule]:
    """Detect integer scaling -- infer factor from grid dimensions."""
    inp0, out0 = pairs[0]["input"], pairs[0]["output"]
    ir, ic = grid_shape(inp0)
    or_, oc = grid_shape(out0)

    if ir == 0 or ic == 0:
        return None

    # Infer factor from dimension ratio
    if or_ % ir != 0 or oc % ic != 0:
        return None

    fr = or_ // ir
    fc = oc // ic
    if fr != fc or fr < 2:
        return None

    factor = fr

    # Validate on all pairs
    for pair in pairs:
        if not grid_eq(scale_grid(pair["input"], factor), pair["output"]):
            return None

    def propagate(g: Grid) -> Grid:
        return scale_grid(g, factor)

    return {"type": "scale", "factor": factor, "propagate": propagate}


@detector
def detect_tile(pairs: List[Dict]) -> Optional[Rule]:
    """Detect self-tiling -- infer tile factors from grid dimensions."""
    inp0, out0 = pairs[0]["input"], pairs[0]["output"]
    ir, ic = grid_shape(inp0)
    or_, oc = grid_shape(out0)

    if ir == 0 or ic == 0:
        return None
    if or_ % ir != 0 or oc % ic != 0:
        return None

    tile_r = or_ // ir
    tile_c = oc // ic

    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        pir, pic = grid_shape(inp)
        por, poc = grid_shape(out)

        if pir == 0 or pic == 0:
            return None
        if por != pir * tile_r or poc != pic * tile_c:
            return None

        for tr in range(tile_r):
            for tc in range(tile_c):
                for r in range(pir):
                    for c in range(pic):
                        if out[tr * pir + r][tc * pic + c] != inp[r][c]:
                            return None

    def propagate(g: Grid) -> Grid:
        rows, cols = grid_shape(g)
        result: Grid = [[0] * (cols * tile_c) for _ in range(rows * tile_r)]
        for tr in range(tile_r):
            for tc in range(tile_c):
                for r in range(rows):
                    for c in range(cols):
                        result[tr * rows + r][tc * cols + c] = g[r][c]
        return result

    return {"type": "tile", "tile_r": tile_r, "tile_c": tile_c, "propagate": propagate}


@detector
def detect_gravity(pairs: List[Dict]) -> Optional[Rule]:
    """Detect directional block movement -- try all 4 cardinal directions."""
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        all_match = True
        for pair in pairs:
            if not grid_eq(_apply_gravity(pair["input"], dr, dc), pair["output"]):
                all_match = False
                break
        if all_match:
            captured_dr, captured_dc = dr, dc

            def propagate(g: Grid, _dr=captured_dr, _dc=captured_dc) -> Grid:
                return _apply_gravity(g, _dr, _dc)

            return {"type": "gravity", "dr": dr, "dc": dc, "propagate": propagate}

    return None


@detector
def detect_color_map(pairs: List[Dict]) -> Optional[Rule]:
    """Detect color substitution (permutation of colors)."""
    mapping: Dict[int, int] = {}

    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        if grid_shape(inp) != grid_shape(out):
            return None
        rows, cols = grid_shape(inp)
        for r in range(rows):
            for c in range(cols):
                ic, oc = inp[r][c], out[r][c]
                if ic in mapping:
                    if mapping[ic] != oc:
                        return None
                else:
                    mapping[ic] = oc

    # Verify consistency
    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        rows, cols = grid_shape(inp)
        for r in range(rows):
            for c in range(cols):
                if mapping.get(inp[r][c]) != out[r][c]:
                    return None

    # Skip identity mappings
    if all(k == v for k, v in mapping.items()):
        return None

    captured_map = dict(mapping)

    def propagate(g: Grid) -> Grid:
        rows, cols = grid_shape(g)
        result = grid_copy(g)
        for r in range(rows):
            for c in range(cols):
                val = g[r][c]
                if val in captured_map:
                    result[r][c] = captured_map[val]
        return result

    return {"type": "color_map", "mapping": captured_map, "propagate": propagate}


@detector
def detect_flood_fill(pairs: List[Dict]) -> Optional[Rule]:
    """Detect enclosed-region flood fill.

    Enclosed = not reachable from grid border. The wave starts from
    borders, marks exterior cells, everything left gets filled.
    """
    for pair in pairs:
        if grid_shape(pair["input"]) != grid_shape(pair["output"]):
            return None

    fill_color: Optional[int] = None
    fill_from: Optional[int] = None

    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        rows, cols = grid_shape(inp)
        for r in range(rows):
            for c in range(cols):
                if inp[r][c] != out[r][c]:
                    if fill_from is None:
                        fill_from = inp[r][c]
                    elif fill_from != inp[r][c]:
                        return None
                    if fill_color is None:
                        fill_color = out[r][c]
                    elif fill_color != out[r][c]:
                        return None

    if fill_color is None or fill_from is None:
        return None

    captured_from, captured_to = fill_from, fill_color

    def propagate(g: Grid) -> Grid:
        return _flood_wave(g, captured_from, captured_to)

    # Verify
    for pair in pairs:
        if not grid_eq(propagate(pair["input"]), pair["output"]):
            return None

    return {
        "type": "flood_fill",
        "fill_from": captured_from,
        "fill_color": captured_to,
        "propagate": propagate,
    }


@detector
def detect_mirror_completion(pairs: List[Dict]) -> Optional[Rule]:
    """Detect symmetry completion: partial pattern mirrored to complete."""
    for axis in ["vertical", "horizontal", "both"]:
        all_match = True
        for pair in pairs:
            if not grid_eq(_mirror_wave(pair["input"], axis), pair["output"]):
                all_match = False
                break
        if all_match:
            captured_axis = axis

            def propagate(g: Grid, _axis=captured_axis) -> Grid:
                return _mirror_wave(g, _axis)

            return {"type": "mirror", "axis": captured_axis, "propagate": propagate}

    return None


@detector
def detect_pattern_repeat(pairs: List[Dict]) -> Optional[Rule]:
    """Detect row or column pattern repetition."""
    for pair in pairs:
        if grid_shape(pair["input"]) != grid_shape(pair["output"]):
            return None

    # Try each row as template source
    inp0 = pairs[0]["input"]
    out0 = pairs[0]["output"]
    rows0, cols0 = grid_shape(out0)

    for source_row in range(len(inp0)):
        template = inp0[source_row]
        if all(out0[r] == template for r in range(rows0)):
            # Verify on all pairs
            all_match = True
            for pair in pairs[1:]:
                inp, out = pair["input"], pair["output"]
                if source_row >= len(inp):
                    all_match = False
                    break
                t = inp[source_row]
                rows, cols = grid_shape(out)
                if not all(out[r] == t for r in range(rows)):
                    all_match = False
                    break
            if all_match:
                captured_row = source_row

                def propagate(g: Grid, _row=captured_row) -> Grid:
                    template = g[_row][:]
                    return [template[:] for _ in range(len(g))]

                return {
                    "type": "pattern_repeat",
                    "mode": "row",
                    "source_row": captured_row,
                    "propagate": propagate,
                }

    return None


# ---------------------------------------------------------------------------
# Wave propagation helpers
# ---------------------------------------------------------------------------

def _apply_gravity(g: Grid, dr: int, dc: int) -> Grid:
    """Move all non-background cells in direction until blocked."""
    rows, cols = grid_shape(g)
    bg = background_color(g)
    out = grid_copy(g)

    if dr != 0:
        for c in range(cols):
            non_bg = [out[r][c] for r in range(rows) if out[r][c] != bg]
            for r in range(rows):
                out[r][c] = bg
            if dr > 0:
                start = rows - len(non_bg)
                for i, val in enumerate(non_bg):
                    out[start + i][c] = val
            else:
                for i, val in enumerate(non_bg):
                    out[i][c] = val
    elif dc != 0:
        for r in range(rows):
            non_bg = [out[r][c] for c in range(cols) if out[r][c] != bg]
            for c in range(cols):
                out[r][c] = bg
            if dc > 0:
                start = cols - len(non_bg)
                for i, val in enumerate(non_bg):
                    out[r][start + i] = val
            else:
                for i, val in enumerate(non_bg):
                    out[r][i] = val

    return out


def _flood_wave(g: Grid, fill_from: int, fill_color: int) -> Grid:
    """Flood fill: BFS from borders to find exterior, fill the interior."""
    rows, cols = grid_shape(g)
    out = grid_copy(g)

    exterior: Set[Tuple[int, int]] = set()
    queue: deque = deque()

    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1):
                if g[r][c] == fill_from and (r, c) not in exterior:
                    exterior.add((r, c))
                    queue.append((r, c))

    while queue:
        cr, cc = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if (nr, nc) not in exterior and g[nr][nc] == fill_from:
                    exterior.add((nr, nc))
                    queue.append((nr, nc))

    for r in range(rows):
        for c in range(cols):
            if g[r][c] == fill_from and (r, c) not in exterior:
                out[r][c] = fill_color

    return out


def _mirror_wave(g: Grid, axis: str) -> Grid:
    """Complete symmetry by mirroring non-background cells."""
    rows, cols = grid_shape(g)
    bg = background_color(g)
    out = grid_copy(g)

    if axis in ("vertical", "both"):
        for r in range(rows):
            for c in range(cols):
                mirror_c = cols - 1 - c
                if out[r][c] == bg and out[r][mirror_c] != bg:
                    out[r][c] = out[r][mirror_c]
                elif out[r][c] != bg and out[r][mirror_c] == bg:
                    out[r][mirror_c] = out[r][c]

    if axis in ("horizontal", "both"):
        for r in range(rows):
            for c in range(cols):
                mirror_r = rows - 1 - r
                if out[r][c] == bg and out[mirror_r][c] != bg:
                    out[r][c] = out[mirror_r][c]
                elif out[r][c] != bg and out[mirror_r][c] == bg:
                    out[mirror_r][c] = out[r][c]

    return out


# ---------------------------------------------------------------------------
# SOLVER
# ---------------------------------------------------------------------------

class ArcSolver:
    """ARC-AGI solver using interrogation + simulation.

    Interrogation: probe training pairs with every registered detector.
    Simulation: the matched rule carries its own propagation function.
    No routing tables, no hardcoded detector lists.
    """

    def interrogate(self, train_pairs: List[Dict]) -> Optional[Rule]:
        """Measurement phase: try every registered detector."""
        for detect_fn in _detector_registry:
            rule = detect_fn(train_pairs)
            if rule is not None:
                if self._validate(rule, train_pairs):
                    return rule
        return None

    def simulate(self, grid: Grid, rule: Rule) -> Grid:
        """Propagation phase: rule carries its own wave function."""
        propagate_fn: Callable = rule["propagate"]
        return propagate_fn(grid)

    def solve(self, task: Dict) -> Optional[Grid]:
        """Solve an ARC task: interrogate training pairs, simulate on test input."""
        rule = self.interrogate(task["train"])
        if rule is None:
            return None
        return self.simulate(task["test"][0]["input"], rule)

    def solve_all(self, task: Dict) -> List[Optional[Grid]]:
        """Solve all test cases in a task."""
        rule = self.interrogate(task["train"])
        if rule is None:
            return [None] * len(task["test"])
        return [self.simulate(tc["input"], rule) for tc in task["test"]]

    def _validate(self, rule: Rule, pairs: List[Dict]) -> bool:
        """Validate rule against all training pairs."""
        propagate_fn: Callable = rule["propagate"]
        for pair in pairs:
            result = propagate_fn(pair["input"])
            if not grid_eq(result, pair["output"]):
                return False
        return True

    def diagnose(self, task: Dict) -> Dict[str, Any]:
        """Diagnostic: which detectors fired, which validated."""
        train_pairs = task["train"]
        report: Dict[str, Any] = {
            "detectors_tried": [],
            "detectors_fired": [],
            "validated": [],
            "final_rule": None,
        }

        for detect_fn in _detector_registry:
            name = detect_fn.__name__
            report["detectors_tried"].append(name)
            rule = detect_fn(train_pairs)
            if rule is not None:
                report["detectors_fired"].append(name)
                if self._validate(rule, train_pairs):
                    report["validated"].append(name)
                    if report["final_rule"] is None:
                        report["final_rule"] = {
                            k: v for k, v in rule.items() if k != "propagate"
                        }

        return report
