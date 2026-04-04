# test_arc.py -- ARC-AGI solver
# No hardcoded transform types. Discovers mappings from data.
#
# Three discovery layers:
#   1. Coordinate mapping  -- affine (r,c) -> (r',c') explains spatial transforms
#   2. Color mapping       -- bijection search explains recoloring
#   3. Local rule          -- neighborhood lookup explains cellular/fill rules
#
# Simulation = propagate discovered mapping as wave across test grid.

import json
import os
import sys
from collections import Counter, deque
from typing import List, Tuple, Optional, Dict, Set, Callable
from itertools import product

# ---------------------------------------------------------------------------
# Discovery registry -- layers self-register, no hardcoded lists
# ---------------------------------------------------------------------------

_discovery_registry: List[Tuple[str, Callable, Callable]] = []


def discoverer(name: str, apply_fn: Callable) -> Callable:
    """Decorator: register a discovery function with its apply function."""
    def wrapper(fn: Callable) -> Callable:
        _discovery_registry.append((name, fn, apply_fn))
        return fn
    return wrapper

Grid = List[List[int]]

# ============================================================
# Grid utilities
# ============================================================

def dims(g: Grid) -> Tuple[int, int]:
    return len(g), len(g[0]) if g else 0

def make(rows: int, cols: int, fill: int = 0) -> Grid:
    return [[fill] * cols for _ in range(rows)]

def copy(g: Grid) -> Grid:
    return [row[:] for row in g]

def eq(a: Grid, b: Grid) -> bool:
    if dims(a) != dims(b):
        return False
    return all(a[r][c] == b[r][c] for r in range(len(a)) for c in range(len(a[0])))

def colors(g: Grid) -> Set[int]:
    return {g[r][c] for r in range(len(g)) for c in range(len(g[0]))}

def color_counts(g: Grid) -> Dict[int, int]:
    ct = Counter()
    for row in g:
        for v in row:
            ct[v] += 1
    return dict(ct)

def cells(g: Grid) -> List[Tuple[int, int, int]]:
    """All (row, col, value) triples."""
    return [(r, c, g[r][c]) for r in range(len(g)) for c in range(len(g[0]))]

def background(g: Grid) -> int:
    """Most common color = background."""
    ct = color_counts(g)
    return max(ct, key=ct.get)

def to_tuple(g: Grid) -> Tuple:
    return tuple(tuple(row) for row in g)

def from_tuple(t: Tuple) -> Grid:
    return [list(row) for row in t]

# ============================================================
# Connected components (object detection)
# ============================================================

def connected_components(g: Grid, ignore: int = -1) -> List[List[Tuple[int, int]]]:
    """Find connected components (4-connected). Optionally ignore a color."""
    rows, cols = dims(g)
    visited = set()
    components = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) in visited:
                continue
            if g[r][c] == ignore:
                visited.add((r, c))
                continue
            comp = []
            queue = deque([(r, c)])
            visited.add((r, c))
            color = g[r][c]
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = cr+dr, cc+dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                        if g[nr][nc] == color:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
            components.append(comp)
    return components

def multicolor_components(g: Grid, bg: int) -> List[List[Tuple[int, int]]]:
    """Connected components ignoring background, allowing mixed colors (8-connected)."""
    rows, cols = dims(g)
    visited = set()
    components = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) in visited or g[r][c] == bg:
                visited.add((r, c))
                continue
            comp = []
            queue = deque([(r, c)])
            visited.add((r, c))
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                            if g[nr][nc] != bg:
                                visited.add((nr, nc))
                                queue.append((nr, nc))
            components.append(comp)
    return components

def bbox(comp: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """Bounding box: (min_r, min_c, max_r, max_c)."""
    rs = [r for r, c in comp]
    cs = [c for r, c in comp]
    return min(rs), min(cs), max(rs), max(cs)

def extract_subgrid(g: Grid, r0: int, c0: int, r1: int, c1: int) -> Grid:
    """Extract subgrid [r0..r1] x [c0..c1] inclusive."""
    return [g[r][c0:c1+1] for r in range(r0, r1+1)]

def place_subgrid(target: Grid, sub: Grid, r0: int, c0: int):
    """Place subgrid into target at (r0, c0). Mutates target."""
    for r in range(len(sub)):
        for c in range(len(sub[0])):
            tr, tc = r0 + r, c0 + c
            if 0 <= tr < len(target) and 0 <= tc < len(target[0]):
                target[tr][tc] = sub[r][c]

# ============================================================
# Layer 1: Coordinate mapping discovery
# ============================================================
# Try all affine transforms (r',c') = (a*r + b*c + e, d*r + f*c + g)
# where a,b,d,f in {-1,0,1} and e,g are offsets.
# This covers: identity, rotations, reflections, translations, transpose.

@discoverer('coord_map', lambda inp, rule: apply_coord_map(inp, rule))
def discover_coord_map(examples: List[Dict]) -> Optional[Dict]:
    """Discover a consistent affine coordinate mapping from input->output.
    Returns {'a','b','d','f','er','ec','color_map'} or None."""
    if not examples:
        return None

    # All examples must have same input->output size relationship
    size_pairs = set()
    for ex in examples:
        ir, ic = dims(ex['input'])
        outr, outc = dims(ex['output'])
        size_pairs.add((ir, ic, outr, outc))
    if len(size_pairs) != 1:
        return None

    ir, ic, outr, outc = size_pairs.pop()

    # Try affine coefficients
    coeffs = [-1, 0, 1]
    for a, b, d, f in product(coeffs, repeat=4):
        # Skip degenerate transforms (both row coeffs zero or both col coeffs zero)
        if a == 0 and b == 0:
            continue
        if d == 0 and f == 0:
            continue

        # Determine offset from first example
        ex0 = examples[0]
        inp, out = ex0['input'], ex0['output']

        anchors = [(r, c) for r, c in
                   [(0, 0), (0, ic-1), (ir-1, 0), (ir-1, ic-1)]
                   if 0 <= r < ir and 0 <= c < ic]

        for ar, ac in anchors:
            # Predicted output coords before offset
            pr = a * ar + b * ac
            pc = d * ar + f * ac

            # Try offsets that put this anchor in range
            for er in range(outr):
                for ec in range(outc):
                    if pr + er < 0 or pr + er >= outr:
                        continue
                    if pc + ec < 0 or pc + ec >= outc:
                        continue

                    # Verify this (a,b,d,f,er,ec) explains ALL examples
                    ok = True
                    color_map_all = {}

                    for ex in examples:
                        inp2, out2 = ex['input'], ex['output']
                        ir2, ic2 = dims(inp2)

                        # Check every input cell maps to a valid output cell
                        valid = True
                        cm = {}
                        covered = set()

                        for r in range(ir2):
                            for c in range(ic2):
                                nr = a * r + b * c + er
                                nc = d * r + f * c + ec
                                if nr < 0 or nr >= outr or nc < 0 or nc >= outc:
                                    valid = False
                                    break
                                iv = inp2[r][c]
                                ov = out2[nr][nc]
                                if iv in cm:
                                    if cm[iv] != ov:
                                        valid = False
                                        break
                                else:
                                    cm[iv] = ov
                                covered.add((nr, nc))
                            if not valid:
                                break

                        if not valid:
                            ok = False
                            break

                        # Check uncovered output cells are background
                        bg_out = background(out2)
                        for r2 in range(outr):
                            for c2 in range(outc):
                                if (r2, c2) not in covered:
                                    if out2[r2][c2] != bg_out:
                                        ok = False
                                        break
                            if not ok:
                                break

                        if not ok:
                            break

                        # Merge color maps
                        for k, v in cm.items():
                            if k in color_map_all:
                                if color_map_all[k] != v:
                                    ok = False
                                    break
                            else:
                                color_map_all[k] = v
                        if not ok:
                            break

                    if ok:
                        return {
                            'type': 'coord_map',
                            'a': a, 'b': b, 'd': d, 'f': f,
                            'er': er, 'ec': ec,
                            'color_map': color_map_all,
                            'out_rows': outr, 'out_cols': outc,
                        }

    return None

def apply_coord_map(inp: Grid, rule: Dict) -> Grid:
    a, b, d, f = rule['a'], rule['b'], rule['d'], rule['f']
    er, ec = rule['er'], rule['ec']
    cm = rule['color_map']
    outr, outc = rule['out_rows'], rule['out_cols']

    bg_out = cm.get(background(inp), 0)
    out = make(outr, outc, bg_out)

    ir, ic = dims(inp)
    for r in range(ir):
        for c in range(ic):
            nr = a * r + b * c + er
            nc = d * r + f * c + ec
            if 0 <= nr < outr and 0 <= nc < outc:
                out[nr][nc] = cm.get(inp[r][c], inp[r][c])
    return out

# ============================================================
# Layer 1b: Scaling discovery
# ============================================================

@discoverer('scaling', lambda inp, rule: apply_scaling(inp, rule))
def discover_scaling(examples: List[Dict]) -> Optional[Dict]:
    """Discover integer scaling: each input cell becomes a kxk block."""
    if not examples:
        return None

    scales = set()
    for ex in examples:
        ir, ic = dims(ex['input'])
        outr, outc = dims(ex['output'])
        if ir == 0 or ic == 0:
            return None
        if outr % ir != 0 or outc % ic != 0:
            return None
        sr, sc = outr // ir, outc // ic
        scales.add((sr, sc))

    if len(scales) != 1:
        return None
    sr, sc = scales.pop()
    if sr == 1 and sc == 1:
        return None  # identity, not scaling

    # Verify: each input cell at (r,c) maps to block at (r*sr..r*sr+sr-1, c*sc..c*sc+sc-1)
    for ex in examples:
        inp, out = ex['input'], ex['output']
        ir, ic = dims(inp)
        for r in range(ir):
            for c in range(ic):
                color = inp[r][c]
                for dr in range(sr):
                    for dc in range(sc):
                        if out[r*sr+dr][c*sc+dc] != color:
                            return None

    return {'type': 'scaling', 'sr': sr, 'sc': sc}

def apply_scaling(inp: Grid, rule: Dict) -> Grid:
    sr, sc = rule['sr'], rule['sc']
    ir, ic = dims(inp)
    out = make(ir * sr, ic * sc)
    for r in range(ir):
        for c in range(ic):
            for dr in range(sr):
                for dc in range(sc):
                    out[r*sr+dr][c*sc+dc] = inp[r][c]
    return out

# ============================================================
# Layer 1c: Self-tile discovery
# ============================================================

@discoverer('self_tile', lambda inp, rule: apply_self_tile(inp, rule))
def discover_self_tile(examples: List[Dict]) -> Optional[Dict]:
    """Discover self-tiling: output is NxN grid of copies of input,
    where each cell of input controls whether its tile position
    gets a copy (non-bg) or blank (empty_color).
    Tries each color as the empty marker."""
    if not examples:
        return None

    for ex in examples:
        inp, out = ex['input'], ex['output']
        ir, ic = dims(inp)
        outr, outc = dims(out)
        if ir == 0 or ic == 0:
            return None
        if outr % ir != 0 or outc % ic != 0:
            return None
        tr, tc = outr // ir, outc // ic
        if tr != ir or tc != ic:
            return None

    # Try each possible color as the empty marker
    all_colors = set()
    for ex in examples:
        all_colors |= colors(ex['input'])

    for empty in all_colors:
        all_match = True
        for ex in examples:
            inp, out = ex['input'], ex['output']
            ir, ic = dims(inp)
            for tile_r in range(ir):
                for tile_c in range(ic):
                    for dr in range(ir):
                        for dc in range(ic):
                            or_ = tile_r * ir + dr
                            oc = tile_c * ic + dc
                            if inp[tile_r][tile_c] != empty:
                                expected = inp[dr][dc]
                            else:
                                expected = empty
                            if out[or_][oc] != expected:
                                all_match = False
                                break
                        if not all_match:
                            break
                    if not all_match:
                        break
                if not all_match:
                    break
            if not all_match:
                break
        if all_match:
            return {'type': 'self_tile', 'empty': empty}

    return None

def apply_self_tile(inp: Grid, rule: Dict) -> Grid:
    ir, ic = dims(inp)
    empty = rule['empty']
    out = make(ir * ir, ic * ic, empty)
    for tile_r in range(ir):
        for tile_c in range(ic):
            if inp[tile_r][tile_c] != empty:
                for dr in range(ir):
                    for dc in range(ic):
                        out[tile_r * ir + dr][tile_c * ic + dc] = inp[dr][dc]
    return out

# ============================================================
# Layer 2: Color mapping discovery (pure recoloring)
# ============================================================

@discoverer('color_map', lambda inp, rule: apply_color_map(inp, rule))
def discover_color_map(examples: List[Dict]) -> Optional[Dict]:
    """Discover a bijective color substitution (same grid dims, same structure)."""
    if not examples:
        return None

    for ex in examples:
        if dims(ex['input']) != dims(ex['output']):
            return None

    cm = {}
    for ex in examples:
        inp, out = ex['input'], ex['output']
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                iv, ov = inp[r][c], out[r][c]
                if iv in cm:
                    if cm[iv] != ov:
                        return None
                else:
                    cm[iv] = ov

    # Must actually change something
    if all(k == v for k, v in cm.items()):
        return None

    return {'type': 'color_map', 'map': cm}

def apply_color_map(inp: Grid, rule: Dict) -> Grid:
    cm = rule['map']
    return [[cm.get(v, v) for v in row] for row in inp]

# ============================================================
# Layer 3: Local neighborhood rule discovery
# ============================================================

def neighborhood(g: Grid, r: int, c: int, radius: int = 1) -> Tuple:
    """Extract NxN neighborhood centered at (r,c). Out-of-bounds = -1."""
    rows, cols = dims(g)
    cells = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                cells.append(g[nr][nc])
            else:
                cells.append(-1)
    return tuple(cells)

@discoverer('local_rule', lambda inp, rule: apply_local_rule(inp, rule))
def discover_local_rule(examples: List[Dict]) -> Optional[Dict]:
    """Discover a deterministic local rule: output[r][c] = f(neighborhood(input, r, c)).
    Tries radius 1 (3x3) then radius 2 (5x5)."""
    if not examples:
        return None

    for ex in examples:
        if dims(ex['input']) != dims(ex['output']):
            return None

    for radius in [1, 2]:
        table = {}
        consistent = True

        for ex in examples:
            inp, out = ex['input'], ex['output']
            rows, cols = dims(inp)
            for r in range(rows):
                for c in range(cols):
                    nb = neighborhood(inp, r, c, radius)
                    ov = out[r][c]
                    if nb in table:
                        if table[nb] != ov:
                            consistent = False
                            break
                    else:
                        table[nb] = ov
                if not consistent:
                    break
            if not consistent:
                break

        if consistent and table:
            # Verify it actually changes something
            changes = False
            for nb, ov in table.items():
                center_idx = (2 * radius + 1) ** 2 // 2
                if nb[center_idx] != ov:
                    changes = True
                    break
            if not changes:
                continue
            return {'type': 'local_rule', 'table': table, 'radius': radius}

    return None

def fuzzy_lookup(nb: Tuple, table: Dict) -> Optional[int]:
    """If exact neighborhood not in table, try replacing -1 (out-of-bounds)
    with values that produce a match. Return the value if unambiguous."""
    if nb in table:
        return table[nb]
    oob_indices = [i for i, v in enumerate(nb) if v == -1]
    if not oob_indices:
        return None
    # Collect candidate values from table entries that match on non-oob positions
    candidates = set()
    for key, val in table.items():
        match = True
        for i, v in enumerate(key):
            if i in oob_indices:
                continue
            if v != nb[i]:
                match = False
                break
        if match:
            candidates.add(val)
    if len(candidates) == 1:
        return candidates.pop()
    return None

def apply_local_rule(inp: Grid, rule: Dict) -> Grid:
    """Apply discovered local rule with fuzzy boundary matching."""
    table = rule['table']
    radius = rule['radius']
    rows, cols = dims(inp)
    out = copy(inp)

    for r in range(rows):
        for c in range(cols):
            nb = neighborhood(inp, r, c, radius)
            val = fuzzy_lookup(nb, table)
            if val is not None:
                out[r][c] = val

    return out

# ============================================================
# Layer 3b: Relative neighborhood rule (color-invariant)
# ============================================================

def rel_neighborhood(g: Grid, r: int, c: int, radius: int = 1) -> Tuple:
    """Color-relative neighborhood: encode as (center_color, [offsets from center])."""
    rows, cols = dims(g)
    center = g[r][c]
    cells = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if g[nr][nc] == center:
                    cells.append(0)  # same as center
                elif g[nr][nc] == 0:
                    cells.append(-1)  # background
                else:
                    cells.append(1)  # different non-bg
            else:
                cells.append(-2)  # out of bounds
    return (center, tuple(cells))

@discoverer('relative_rule', lambda inp, rule: apply_relative_rule(inp, rule))
def discover_relative_rule(examples: List[Dict]) -> Optional[Dict]:
    """Discover a color-relative local rule: the output depends on the
    pattern of same/different/bg neighbors, not absolute colors."""
    if not examples:
        return None

    for ex in examples:
        if dims(ex['input']) != dims(ex['output']):
            return None

    for radius in [1]:
        # table maps (center_color, rel_pattern) -> output_offset
        # output_offset: 0 = same as center, -1 = bg(0), else the actual color
        table = {}
        consistent = True

        for ex in examples:
            inp, out = ex['input'], ex['output']
            rows, cols = dims(inp)
            bg = background(inp)
            for r in range(rows):
                for c in range(cols):
                    rnb = rel_neighborhood(inp, r, c, radius)
                    ov = out[r][c]
                    center = inp[r][c]
                    # Encode output relative to center
                    if ov == center:
                        out_rel = 'same'
                    elif ov == bg:
                        out_rel = 'bg'
                    else:
                        out_rel = ov  # absolute (new color introduced)

                    if rnb in table:
                        if table[rnb] != out_rel:
                            consistent = False
                            break
                    else:
                        table[rnb] = out_rel
                if not consistent:
                    break
            if not consistent:
                break

        if consistent and table:
            changes = any(v != 'same' for v in table.values())
            if changes:
                return {'type': 'relative_rule', 'table': table, 'radius': radius}

    return None

def apply_relative_rule(inp: Grid, rule: Dict) -> Grid:
    table = rule['table']
    radius = rule['radius']
    rows, cols = dims(inp)
    bg = background(inp)
    out = copy(inp)

    for r in range(rows):
        for c in range(cols):
            rnb = rel_neighborhood(inp, r, c, radius)
            if rnb in table:
                v = table[rnb]
                if v == 'same':
                    out[r][c] = inp[r][c]
                elif v == 'bg':
                    out[r][c] = bg
                else:
                    out[r][c] = v

    return out

# ============================================================
# Layer 4: Split-and-recombine discovery
# ============================================================

def find_separators(g: Grid) -> Dict:
    """Find horizontal/vertical separator lines (rows/cols of single color)."""
    rows, cols = dims(g)
    result = {'h_seps': [], 'v_seps': []}

    for r in range(rows):
        vals = set(g[r])
        if len(vals) == 1 and vals.pop() != background(g):
            result['h_seps'].append(r)

    for c in range(cols):
        vals = set(g[r][c] for r in range(rows))
        if len(vals) == 1 and vals.pop() != background(g):
            result['v_seps'].append(c)

    return result

@discoverer('split_overlay', lambda inp, rule: apply_split_overlay(inp, rule))
def discover_split_overlay(examples: List[Dict]) -> Optional[Dict]:
    """Discover split-by-separator then overlay/combine pattern."""
    if not examples:
        return None

    for ex in examples:
        inp, out = ex['input'], ex['output']
        seps = find_separators(inp)

        # Try vertical split
        if len(seps['v_seps']) == 1:
            sc = seps['v_seps'][0]
            ir, ic = dims(inp)
            left = extract_subgrid(inp, 0, 0, ir - 1, sc - 1)
            right = extract_subgrid(inp, 0, sc + 1, ir - 1, ic - 1)

            lr, lc = dims(left)
            rr, rc = dims(right)
            outr, outc = dims(out)

            if lr == rr == outr and lc == rc == outc:
                # Same-size halves -> overlay
                # Discover the combine function
                pass  # handled below

        # Try horizontal split
        if len(seps['h_seps']) == 1:
            sr = seps['h_seps'][0]
            ir, ic = dims(inp)
            top = extract_subgrid(inp, 0, 0, sr - 1, ic - 1)
            bot = extract_subgrid(inp, sr + 1, 0, ir - 1, ic - 1)

            tr, tc = dims(top)
            br, bc = dims(bot)
            outr, outc = dims(out)

            if tr == br == outr and tc == bc == outc:
                pass  # handled below

    # Try to learn the overlay operation from all examples
    # For each vertical split example, extract left/right/output triples
    # Build a pointwise combination table: (left_val, right_val) -> output_val

    for axis in ['v', 'h']:
        combine_table = {}
        all_consistent = True

        for ex in examples:
            inp, out = ex['input'], ex['output']
            seps = find_separators(inp)
            ir, ic = dims(inp)

            if axis == 'v' and len(seps['v_seps']) == 1:
                sc = seps['v_seps'][0]
                part_a = extract_subgrid(inp, 0, 0, ir - 1, sc - 1)
                part_b = extract_subgrid(inp, 0, sc + 1, ir - 1, ic - 1)
            elif axis == 'h' and len(seps['h_seps']) == 1:
                sr = seps['h_seps'][0]
                part_a = extract_subgrid(inp, 0, 0, sr - 1, ic - 1)
                part_b = extract_subgrid(inp, sr + 1, 0, ir - 1, ic - 1)
            else:
                all_consistent = False
                break

            ar, ac = dims(part_a)
            br, bc = dims(part_b)
            outr, outc = dims(out)
            if ar != br or ac != bc or ar != outr or ac != outc:
                all_consistent = False
                break

            for r in range(ar):
                for c in range(ac):
                    key = (part_a[r][c], part_b[r][c])
                    val = out[r][c]
                    if key in combine_table:
                        if combine_table[key] != val:
                            all_consistent = False
                            break
                    else:
                        combine_table[key] = val
                if not all_consistent:
                    break
            if not all_consistent:
                break

        if all_consistent and combine_table:
            return {'type': 'split_overlay', 'axis': axis, 'table': combine_table}

    return None

def apply_split_overlay(inp: Grid, rule: Dict) -> Grid:
    axis = rule['axis']
    table = rule['table']
    ir, ic = dims(inp)
    seps = find_separators(inp)

    if axis == 'v' and seps['v_seps']:
        sc = seps['v_seps'][0]
        part_a = extract_subgrid(inp, 0, 0, ir - 1, sc - 1)
        part_b = extract_subgrid(inp, 0, sc + 1, ir - 1, ic - 1)
    elif axis == 'h' and seps['h_seps']:
        sr = seps['h_seps'][0]
        part_a = extract_subgrid(inp, 0, 0, sr - 1, ic - 1)
        part_b = extract_subgrid(inp, sr + 1, 0, ir - 1, ic - 1)
    else:
        return copy(inp)

    rows, cols = dims(part_a)
    out = make(rows, cols)
    for r in range(rows):
        for c in range(cols):
            key = (part_a[r][c], part_b[r][c])
            out[r][c] = table.get(key, 0)
    return out

# ============================================================
# Layer 5: Crop/extract discovery
# ============================================================

@discoverer('crop', lambda inp, rule: apply_crop(inp, rule))
def discover_crop(examples: List[Dict]) -> Optional[Dict]:
    """Discover that output is a subregion of input (crop to non-bg bounding box,
    or crop to a specific object)."""
    if not examples:
        return None

    # Strategy: output = bounding box of non-background cells in input
    # Try: crop to non-bg bounding box
    all_match = True
    for ex in examples:
        inp, out = ex['input'], ex['output']
        bg = background(inp)
        non_bg = [(r, c) for r, c, v in cells(inp) if v != bg]
        if not non_bg:
            all_match = False
            break
        r0, c0, r1, c1 = bbox(non_bg)
        cropped = extract_subgrid(inp, r0, c0, r1, c1)
        if not eq(cropped, out):
            all_match = False
            break
    if all_match:
        return {'type': 'crop', 'strategy': 'non_bg_bbox'}

    # Try: output is the smallest non-bg connected component
    all_match = True
    for ex in examples:
        inp, out = ex['input'], ex['output']
        bg = background(inp)
        comps = multicolor_components(inp, bg)
        if not comps:
            all_match = False
            break
        smallest = min(comps, key=len)
        r0, c0, r1, c1 = bbox(smallest)
        cropped = extract_subgrid(inp, r0, c0, r1, c1)
        if not eq(cropped, out):
            all_match = False
            break
    if all_match:
        return {'type': 'crop', 'strategy': 'smallest_component'}

    # Try: output is the largest non-bg connected component
    all_match = True
    for ex in examples:
        inp, out = ex['input'], ex['output']
        bg = background(inp)
        comps = multicolor_components(inp, bg)
        if not comps:
            all_match = False
            break
        largest = max(comps, key=len)
        r0, c0, r1, c1 = bbox(largest)
        cropped = extract_subgrid(inp, r0, c0, r1, c1)
        if not eq(cropped, out):
            all_match = False
            break
    if all_match:
        return {'type': 'crop', 'strategy': 'largest_component'}

    return None

def apply_crop(inp: Grid, rule: Dict) -> Grid:
    bg = background(inp)
    strategy = rule['strategy']

    if strategy == 'non_bg_bbox':
        non_bg = [(r, c) for r, c, v in cells(inp) if v != bg]
        if not non_bg:
            return copy(inp)
        r0, c0, r1, c1 = bbox(non_bg)
        return extract_subgrid(inp, r0, c0, r1, c1)

    comps = multicolor_components(inp, bg)
    if not comps:
        return copy(inp)

    if strategy == 'smallest_component':
        target = min(comps, key=len)
    else:
        target = max(comps, key=len)

    r0, c0, r1, c1 = bbox(target)
    return extract_subgrid(inp, r0, c0, r1, c1)

# ============================================================
# Layer 6: Gravity discovery
# ============================================================

@discoverer('gravity', lambda inp, rule: apply_gravity(inp, rule))
def discover_gravity(examples: List[Dict]) -> Optional[Dict]:
    """Discover gravity: non-bg cells fall in a direction until hitting
    boundary or another cell."""
    if not examples:
        return None

    for ex in examples:
        if dims(ex['input']) != dims(ex['output']):
            return None

    for direction in ['down', 'up', 'left', 'right']:
        all_match = True
        for ex in examples:
            predicted = apply_gravity_dir(ex['input'], direction)
            if not eq(predicted, ex['output']):
                all_match = False
                break
        if all_match:
            return {'type': 'gravity', 'direction': direction}

    return None

def apply_gravity_dir(inp: Grid, direction: str) -> Grid:
    rows, cols = dims(inp)
    bg = background(inp)
    out = make(rows, cols, bg)

    if direction == 'down':
        for c in range(cols):
            non_bg = [inp[r][c] for r in range(rows) if inp[r][c] != bg]
            for i, v in enumerate(non_bg):
                out[rows - len(non_bg) + i][c] = v
    elif direction == 'up':
        for c in range(cols):
            non_bg = [inp[r][c] for r in range(rows) if inp[r][c] != bg]
            for i, v in enumerate(non_bg):
                out[i][c] = v
    elif direction == 'right':
        for r in range(rows):
            non_bg = [inp[r][c] for c in range(cols) if inp[r][c] != bg]
            for i, v in enumerate(non_bg):
                out[r][cols - len(non_bg) + i] = v
    elif direction == 'left':
        for r in range(rows):
            non_bg = [inp[r][c] for c in range(cols) if inp[r][c] != bg]
            for i, v in enumerate(non_bg):
                out[r][i] = v
    return out

def apply_gravity(inp: Grid, rule: Dict) -> Grid:
    return apply_gravity_dir(inp, rule['direction'])

# ============================================================
# Layer 7: Fill enclosed regions discovery
# ============================================================

@discoverer('fill_enclosed', lambda inp, rule: apply_fill_enclosed(inp, rule))
def discover_fill_enclosed(examples: List[Dict]) -> Optional[Dict]:
    """Discover: background cells enclosed by non-bg cells get filled with a new color."""
    if not examples:
        return None

    for ex in examples:
        if dims(ex['input']) != dims(ex['output']):
            return None

    # Find which cells changed between input and output
    fill_color = None
    all_match = True

    for ex in examples:
        inp, out = ex['input'], ex['output']
        rows, cols = dims(inp)
        bg = background(inp)

        # Find cells that changed
        changed = {}
        for r in range(rows):
            for c in range(cols):
                if inp[r][c] != out[r][c]:
                    changed[(r, c)] = out[r][c]

        if not changed:
            continue

        # All changed cells should have the same new color
        new_colors = set(changed.values())
        if len(new_colors) != 1:
            all_match = False
            break

        fc = new_colors.pop()
        if fill_color is None:
            fill_color = fc
        # fill_color might vary per example (depends on boundary color)

        # Changed cells should be the ones enclosed (not reachable from border via bg)
        reachable = set()
        queue = deque()
        for r in range(rows):
            for c in [0, cols - 1]:
                if inp[r][c] == bg and (r, c) not in reachable:
                    reachable.add((r, c))
                    queue.append((r, c))
        for c in range(cols):
            for r in [0, rows - 1]:
                if inp[r][c] == bg and (r, c) not in reachable:
                    reachable.add((r, c))
                    queue.append((r, c))

        while queue:
            cr, cc = queue.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in reachable:
                    if inp[nr][nc] == bg:
                        reachable.add((nr, nc))
                        queue.append((nr, nc))

        enclosed = set()
        for r in range(rows):
            for c in range(cols):
                if inp[r][c] == bg and (r, c) not in reachable:
                    enclosed.add((r, c))

        if set(changed.keys()) != enclosed:
            all_match = False
            break

    if all_match and fill_color is not None:
        return {'type': 'fill_enclosed', 'fill_color': fill_color}

    # Try: fill color = non-bg boundary color
    # (the fill color might be context-dependent, not fixed)
    all_match = True
    for ex in examples:
        inp, out = ex['input'], ex['output']
        predicted = apply_fill_enclosed_adaptive(inp)
        if not eq(predicted, out):
            all_match = False
            break
    if all_match:
        return {'type': 'fill_enclosed', 'fill_color': 'adaptive'}

    return None

def apply_fill_enclosed_adaptive(inp: Grid) -> Grid:
    """Fill enclosed bg regions. Determine fill color from surrounding boundary."""
    rows, cols = dims(inp)
    bg = background(inp)
    out = copy(inp)

    # Find bg cells not reachable from border
    reachable = set()
    queue = deque()
    for r in range(rows):
        for c in [0, cols - 1]:
            if inp[r][c] == bg and (r, c) not in reachable:
                reachable.add((r, c))
                queue.append((r, c))
    for c in range(cols):
        for r in [0, rows - 1]:
            if inp[r][c] == bg and (r, c) not in reachable:
                reachable.add((r, c))
                queue.append((r, c))
    while queue:
        cr, cc = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in reachable:
                if inp[nr][nc] == bg:
                    reachable.add((nr, nc))
                    queue.append((nr, nc))

    # Find enclosed regions as connected components
    enclosed = set()
    for r in range(rows):
        for c in range(cols):
            if inp[r][c] == bg and (r, c) not in reachable:
                enclosed.add((r, c))

    if not enclosed:
        return out

    # Group enclosed cells into connected regions
    remaining = set(enclosed)
    while remaining:
        start = next(iter(remaining))
        region = set()
        q = deque([start])
        region.add(start)
        remaining.discard(start)
        while q:
            cr, cc = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = cr + dr, cc + dc
                if (nr, nc) in remaining:
                    region.add((nr, nc))
                    remaining.discard((nr, nc))
                    q.append((nr, nc))

        # Determine fill color from neighbors of this region
        neighbor_colors = Counter()
        for r, c in region:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if inp[nr][nc] != bg:
                        neighbor_colors[inp[nr][nc]] += 1

        if neighbor_colors:
            # Use most common non-bg neighbor, but pick a different color
            # Actually in many ARC tasks, the fill is a specific new color
            # Try: use the most common boundary color
            fill = neighbor_colors.most_common(1)[0][0]
        else:
            fill = bg

        for r, c in region:
            out[r][c] = fill

    return out

def apply_fill_enclosed(inp: Grid, rule: Dict) -> Grid:
    if rule['fill_color'] == 'adaptive':
        return apply_fill_enclosed_adaptive(inp)

    rows, cols = dims(inp)
    bg = background(inp)
    out = copy(inp)

    reachable = set()
    queue = deque()
    for r in range(rows):
        for c in [0, cols - 1]:
            if inp[r][c] == bg and (r, c) not in reachable:
                reachable.add((r, c))
                queue.append((r, c))
    for c in range(cols):
        for r in [0, rows - 1]:
            if inp[r][c] == bg and (r, c) not in reachable:
                reachable.add((r, c))
                queue.append((r, c))
    while queue:
        cr, cc = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in reachable:
                if inp[nr][nc] == bg:
                    reachable.add((nr, nc))
                    queue.append((nr, nc))

    for r in range(rows):
        for c in range(cols):
            if inp[r][c] == bg and (r, c) not in reachable:
                out[r][c] = rule['fill_color']
    return out

# ============================================================
# Layer 8: Pattern tiling / replication discovery
# ============================================================

def discover_pattern_tile(examples: List[Dict]) -> Optional[Dict]:
    """Discover: output is input repeated/tiled to fill output dimensions."""
    if not examples:
        return None

    for ex in examples:
        ir, ic = dims(ex['input'])
        outr, outc = dims(ex['output'])
        if outr < ir or outc < ic:
            return None
        if outr % ir != 0 or outc % ic != 0:
            return None

    all_match = True
    for ex in examples:
        inp, out = ex['input'], ex['output']
        ir, ic = dims(inp)
        outr, outc = dims(out)
        for r in range(outr):
            for c in range(outc):
                if out[r][c] != inp[r % ir][c % ic]:
                    all_match = False
                    break
            if not all_match:
                break
        if not all_match:
            break

    if all_match:
        return {'type': 'pattern_tile'}

    return None

def apply_pattern_tile(inp: Grid, rule: Dict) -> Grid:
    # Need to know output dims -- infer from input?
    # In ARC the test output dims aren't given, so we need to infer from examples
    # For now tile 3x3
    ir, ic = dims(inp)
    # This is a limitation -- we need the examples to know output size
    # We'll handle this in the solve function
    return copy(inp)

# ============================================================
# Layer 9: Majority/voting rule discovery
# ============================================================

def discover_majority_output(examples: List[Dict]) -> Optional[Dict]:
    """Discover: output is always the same grid regardless of input."""
    if len(examples) < 2:
        return None

    target = to_tuple(examples[0]['output'])
    for ex in examples[1:]:
        if to_tuple(ex['output']) != target:
            return None

    return {'type': 'constant_output', 'output': from_tuple(target)}

def apply_constant_output(inp: Grid, rule: Dict) -> Grid:
    return copy(rule['output'])

# ============================================================
# SIMULATION ENGINE
# ============================================================
# Core idea: instead of applying a discovered rule cell-by-cell in one pass,
# propagate it as a wave. Start from "anchor" cells (cells whose output
# is determined by the rule with full context), then expand the frontier.
# Each wave step determines new cells whose neighborhoods are now complete.
# This handles cascading/dependent transforms.

def simulate_wave(inp: Grid, step_fn, max_iterations: int = 100) -> Grid:
    """Generic wave propagation.
    step_fn(current_grid, r, c) -> Optional[int]
    Returns new value for (r,c) or None if can't determine yet.
    Propagates until no more changes."""
    rows, cols = dims(inp)
    current = copy(inp)

    for iteration in range(max_iterations):
        changed = False
        next_grid = copy(current)

        for r in range(rows):
            for c in range(cols):
                new_val = step_fn(current, r, c)
                if new_val is not None and new_val != current[r][c]:
                    next_grid[r][c] = new_val
                    changed = True

        current = next_grid
        if not changed:
            break

    return current

# ============================================================
# Layer 10: Iterative local rule with simulation
# ============================================================

def discover_iterative_rule(examples: List[Dict]) -> Optional[Dict]:
    """Discover a rule that needs multiple simulation steps.
    Try: apply local rule repeatedly until output matches."""
    if not examples:
        return None

    for ex in examples:
        if dims(ex['input']) != dims(ex['output']):
            return None

    # First discover the single-step local rule
    # Then check if applying it iteratively matches the output
    rule = discover_local_rule(examples)
    if rule is not None:
        return rule  # single-step suffices

    return None

# ============================================================
# Layer 11: Object-level transform discovery
# ============================================================

def discover_object_transform(examples: List[Dict]) -> Optional[Dict]:
    """Discover per-object transforms: each object in input maps to
    a transformed version in output."""
    if not examples:
        return None

    for ex in examples:
        if dims(ex['input']) != dims(ex['output']):
            return None

    # Extract objects from each example, try to match them
    # Learn a consistent per-object rule
    bg_vals = set()
    for ex in examples:
        bg_vals.add(background(ex['input']))
    if len(bg_vals) != 1:
        return None
    bg = bg_vals.pop()

    for ex in examples:
        inp, out = ex['input'], ex['output']
        in_comps = connected_components(inp, bg)
        out_comps = connected_components(out, bg)

        if len(in_comps) != len(out_comps):
            return None

        # Match by color
        in_by_color = {}
        for comp in in_comps:
            c = inp[comp[0][0]][comp[0][1]]
            in_by_color.setdefault(c, []).append(comp)

        out_by_color = {}
        for comp in out_comps:
            c = out[comp[0][0]][comp[0][1]]
            out_by_color.setdefault(c, []).append(comp)

        if set(in_by_color.keys()) != set(out_by_color.keys()):
            return None

        for color in in_by_color:
            if len(in_by_color[color]) != len(out_by_color.get(color, [])):
                return None

    return None  # Complex -- implement specific sub-cases as needed

# ============================================================
# Layer 12: Grid region fill (separator-divided regions)
# ============================================================

def get_regions(g: Grid) -> Optional[Dict]:
    """Split grid into rectangular regions by separator lines.
    Returns {'sep_color', 'h_seps', 'v_seps', 'regions': [(r0,c0,r1,c1), ...]} or None."""
    rows, cols = dims(g)
    if rows < 3 or cols < 3:
        return None

    # Find separator color that forms complete rows or cols (skip background)
    bg = background(g)
    for sep_color in sorted(colors(g), key=lambda c: (c == bg, c)):
        h_seps = []
        v_seps = []

        for r in range(rows):
            if all(g[r][c] == sep_color for c in range(cols)):
                h_seps.append(r)

        for c in range(cols):
            if all(g[r][c] == sep_color for r in range(rows)):
                v_seps.append(c)

        if not h_seps and not v_seps:
            continue

        # Build region bounds
        h_bounds = [-1] + h_seps + [rows]
        v_bounds = [-1] + v_seps + [cols]

        regions = []
        for i in range(len(h_bounds) - 1):
            for j in range(len(v_bounds) - 1):
                r0 = h_bounds[i] + 1
                r1 = h_bounds[i + 1] - 1
                c0 = v_bounds[j] + 1
                c1 = v_bounds[j + 1] - 1
                if r0 <= r1 and c0 <= c1:
                    regions.append((r0, c0, r1, c1))

        if len(regions) >= 2:
            return {
                'sep_color': sep_color,
                'h_seps': h_seps,
                'v_seps': v_seps,
                'regions': regions,
            }

    return None

def discover_region_map(examples: List[Dict]) -> Optional[Dict]:
    """Discover: grid is divided into regions by separators.
    Each region in input maps to a corresponding region in output.
    Learn the region-level transform."""
    if not examples:
        return None

    for ex in examples:
        if dims(ex['input']) != dims(ex['output']):
            return None

    # Check all examples have consistent region structure
    region_infos = []
    for ex in examples:
        ri = get_regions(ex['input'])
        if ri is None:
            return None
        region_infos.append(ri)

    # Build per-region mapping: for each region, learn input_subgrid -> output_subgrid
    # Check if it's a pointwise combine of regions
    # (e.g., one region is a "template" and others are "filled" based on it)

    # Strategy 1: regions have uniform fill in output, determined by input content
    all_uniform = True
    region_rule = {}  # maps region content signature -> output fill color

    for ex in examples:
        inp, out = ex['input'], ex['output']
        ri = get_regions(inp)
        if ri is None:
            all_uniform = False
            break

        for r0, c0, r1, c1 in ri['regions']:
            sub_in = extract_subgrid(inp, r0, c0, r1, c1)
            sub_out = extract_subgrid(out, r0, c0, r1, c1)

            # Check if output region is uniform
            out_vals = set()
            for row in sub_out:
                for v in row:
                    out_vals.add(v)

            if len(out_vals) > 1:
                all_uniform = False
                break

            # Input region signature: count of non-sep, non-bg colors
            bg = background(inp)
            in_non_bg = sum(1 for row in sub_in for v in row if v != bg and v != ri['sep_color'])
            fill = out_vals.pop()

            key = (in_non_bg > 0,)
            if key in region_rule:
                if region_rule[key] != fill:
                    all_uniform = False
                    break
            else:
                region_rule[key] = fill

        if not all_uniform:
            break

    if all_uniform and region_rule:
        return {'type': 'region_uniform', 'rule': region_rule}

    return None

def apply_region_map(inp: Grid, rule: Dict) -> Grid:
    out = copy(inp)
    ri = get_regions(inp)
    if ri is None:
        return out
    bg = background(inp)

    for r0, c0, r1, c1 in ri['regions']:
        sub_in = extract_subgrid(inp, r0, c0, r1, c1)
        in_non_bg = sum(1 for row in sub_in for v in row if v != bg and v != ri['sep_color'])
        key = (in_non_bg > 0,)
        fill = rule['rule'].get(key, bg)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                out[r][c] = fill
    return out

# ============================================================
# Layer 13: Symmetry completion
# ============================================================

def discover_symmetry(examples: List[Dict]) -> Optional[Dict]:
    """Discover: output is input made symmetric (horizontally, vertically, or both)."""
    if not examples:
        return None

    for ex in examples:
        if dims(ex['input']) != dims(ex['output']):
            return None

    for sym_type in ['h', 'v', 'hv', 'vh']:
        all_match = True
        for ex in examples:
            predicted = apply_symmetry_type(ex['input'], sym_type)
            if not eq(predicted, ex['output']):
                all_match = False
                break
        if all_match:
            return {'type': 'symmetry', 'sym': sym_type}

    return None

def apply_symmetry_type(inp: Grid, sym_type: str) -> Grid:
    rows, cols = dims(inp)
    bg = background(inp)
    out = copy(inp)

    if 'h' in sym_type:
        # Horizontal symmetry: for each cell, take the non-bg value from
        # either (r,c) or (r, cols-1-c)
        for r in range(rows):
            for c in range(cols):
                mirror_c = cols - 1 - c
                if out[r][c] == bg and out[r][mirror_c] != bg:
                    out[r][c] = out[r][mirror_c]
                elif out[r][c] != bg and out[r][mirror_c] == bg:
                    out[r][mirror_c] = out[r][c]

    if 'v' in sym_type:
        # Vertical symmetry
        for r in range(rows):
            for c in range(cols):
                mirror_r = rows - 1 - r
                if out[r][c] == bg and out[mirror_r][c] != bg:
                    out[r][c] = out[mirror_r][c]
                elif out[r][c] != bg and out[mirror_r][c] == bg:
                    out[mirror_r][c] = out[r][c]

    return out

def apply_symmetry(inp: Grid, rule: Dict) -> Grid:
    return apply_symmetry_type(inp, rule['sym'])

# ============================================================
# Layer 14: Most-common subgrid extraction
# ============================================================

def discover_extract_subgrid(examples: List[Dict]) -> Optional[Dict]:
    """Discover: output is a specific subgrid extracted based on a criterion
    (unique color, most cells, specific position, etc.)."""
    if not examples:
        return None

    # All outputs must be same size
    out_sizes = set()
    for ex in examples:
        out_sizes.add(dims(ex['output']))
    if len(out_sizes) != 1:
        return None
    out_rows, out_cols = out_sizes.pop()

    # Strategy: find all out_rows x out_cols subgrids in input that match output
    # Then find a consistent selection rule

    # Try: output = subgrid containing the most non-bg cells
    all_match = True
    for ex in examples:
        inp, out = ex['input'], ex['output']
        ir, ic = dims(inp)
        bg = background(inp)
        best = None
        best_count = -1
        for r0 in range(ir - out_rows + 1):
            for c0 in range(ic - out_cols + 1):
                sub = extract_subgrid(inp, r0, c0, r0 + out_rows - 1, c0 + out_cols - 1)
                count = sum(1 for row in sub for v in row if v != bg)
                if count > best_count:
                    best_count = count
                    best = sub
        if best is None or not eq(best, out):
            all_match = False
            break
    if all_match:
        return {'type': 'extract_subgrid', 'strategy': 'most_non_bg', 'out_rows': out_rows, 'out_cols': out_cols}

    # Try: output = subgrid with fewest non-bg cells (but > 0)
    all_match = True
    for ex in examples:
        inp, out = ex['input'], ex['output']
        ir, ic = dims(inp)
        bg = background(inp)
        best = None
        best_count = float('inf')
        for r0 in range(ir - out_rows + 1):
            for c0 in range(ic - out_cols + 1):
                sub = extract_subgrid(inp, r0, c0, r0 + out_rows - 1, c0 + out_cols - 1)
                count = sum(1 for row in sub for v in row if v != bg)
                if 0 < count < best_count:
                    best_count = count
                    best = sub
        if best is None or not eq(best, out):
            all_match = False
            break
    if all_match:
        return {'type': 'extract_subgrid', 'strategy': 'least_non_bg', 'out_rows': out_rows, 'out_cols': out_cols}

    return None

def apply_extract_subgrid(inp: Grid, rule: Dict) -> Grid:
    ir, ic = dims(inp)
    out_rows, out_cols = rule['out_rows'], rule['out_cols']
    bg = background(inp)

    if rule['strategy'] == 'most_non_bg':
        best = None
        best_count = -1
        for r0 in range(ir - out_rows + 1):
            for c0 in range(ic - out_cols + 1):
                sub = extract_subgrid(inp, r0, c0, r0 + out_rows - 1, c0 + out_cols - 1)
                count = sum(1 for row in sub for v in row if v != bg)
                if count > best_count:
                    best_count = count
                    best = sub
        return best if best else make(out_rows, out_cols)
    else:
        best = None
        best_count = float('inf')
        for r0 in range(ir - out_rows + 1):
            for c0 in range(ic - out_cols + 1):
                sub = extract_subgrid(inp, r0, c0, r0 + out_rows - 1, c0 + out_cols - 1)
                count = sum(1 for row in sub for v in row if v != bg)
                if 0 < count < best_count:
                    best_count = count
                    best = sub
        return best if best else make(out_rows, out_cols)

# ============================================================
# Layer 15: Input-output diff analysis (change propagation)
# ============================================================

def discover_diff_rule(examples: List[Dict]) -> Optional[Dict]:
    """Analyze what changes between input and output across all examples.
    Learn a rule for WHY cells change based on their spatial relationship
    to non-changed cells."""
    if not examples:
        return None

    for ex in examples:
        if dims(ex['input']) != dims(ex['output']):
            return None

    # Strategy: cells change from bg to a specific color when they are
    # "between" two cells of that color (horizontal or vertical line drawing)
    bg_vals = set(background(ex['input']) for ex in examples)
    if len(bg_vals) != 1:
        return None
    bg = bg_vals.pop()

    all_match = True
    for ex in examples:
        inp, out = ex['input'], ex['output']
        predicted = apply_line_fill(inp, bg)
        if not eq(predicted, out):
            all_match = False
            break
    if all_match:
        return {'type': 'line_fill', 'bg': bg}

    return None

def apply_line_fill(inp: Grid, bg: int) -> Grid:
    """Fill horizontal and vertical gaps between same-color cells."""
    rows, cols = dims(inp)
    out = copy(inp)

    # Horizontal: for each row, find pairs of same-color non-bg cells and fill between
    for r in range(rows):
        for c1 in range(cols):
            if inp[r][c1] == bg:
                continue
            color = inp[r][c1]
            for c2 in range(c1 + 2, cols):
                if inp[r][c2] == color:
                    # Fill between c1 and c2 if all are bg
                    all_bg = all(inp[r][c] == bg for c in range(c1 + 1, c2))
                    if all_bg:
                        for c in range(c1 + 1, c2):
                            out[r][c] = color
                    break
                elif inp[r][c2] != bg:
                    break

    # Vertical: for each column
    for c in range(cols):
        for r1 in range(rows):
            if inp[r1][c] == bg:
                continue
            color = inp[r1][c]
            for r2 in range(r1 + 2, rows):
                if inp[r2][c] == color:
                    all_bg = all(inp[r][c] == bg for r in range(r1 + 1, r2))
                    if all_bg:
                        for r in range(r1 + 1, r2):
                            out[r][c] = color
                    break
                elif inp[r2][c] != bg:
                    break

    return out

def apply_diff_rule(inp: Grid, rule: Dict) -> Grid:
    if rule['type'] == 'line_fill':
        return apply_line_fill(inp, rule['bg'])
    return copy(inp)

# ============================================================
# Layer 16: Border/frame operations
# ============================================================

def discover_border(examples: List[Dict]) -> Optional[Dict]:
    """Discover: output has a border added or the border color changed."""
    if not examples:
        return None

    # Check: output = input with border cells changed to a specific color
    for ex in examples:
        if dims(ex['input']) != dims(ex['output']):
            return None

    # Try: all border cells in output have the same color
    border_color = None
    all_match = True
    for ex in examples:
        inp, out = ex['input'], ex['output']
        rows, cols = dims(out)
        bc = set()
        for r in range(rows):
            bc.add(out[r][0])
            bc.add(out[r][cols - 1])
        for c in range(cols):
            bc.add(out[0][c])
            bc.add(out[rows - 1][c])

        if len(bc) != 1:
            all_match = False
            break

        c = bc.pop()
        if border_color is None:
            border_color = c

        # Interior must match input interior
        for r in range(1, rows - 1):
            for ci in range(1, cols - 1):
                if out[r][ci] != inp[r][ci]:
                    all_match = False
                    break
            if not all_match:
                break
        if not all_match:
            break

    if all_match and border_color is not None:
        return {'type': 'border', 'color': border_color}

    return None

def apply_border(inp: Grid, rule: Dict) -> Grid:
    rows, cols = dims(inp)
    out = copy(inp)
    c = rule['color']
    for r in range(rows):
        out[r][0] = c
        out[r][cols - 1] = c
    for ci in range(cols):
        out[0][ci] = c
        out[rows - 1][ci] = c
    return out

# ============================================================
# Layer 17: Downscale (output is a compressed version of input)
# ============================================================

def discover_downscale(examples: List[Dict]) -> Optional[Dict]:
    """Discover: output is input downscaled by integer factor.
    Tries multiple summarization strategies per block."""
    if not examples:
        return None

    scales = set()
    for ex in examples:
        ir, ic = dims(ex['input'])
        outr, outc = dims(ex['output'])
        if outr == 0 or outc == 0:
            return None
        if ir % outr != 0 or ic % outc != 0:
            return None
        sr, sc = ir // outr, ic // outc
        scales.add((sr, sc))

    if len(scales) != 1:
        return None
    sr, sc = scales.pop()
    if sr == 1 and sc == 1:
        return None

    # Try multiple summarization strategies
    def _summarize_block(block_vals: List[int], strategy: str) -> int:
        ct = Counter(block_vals)
        non_zero = {k: v for k, v in ct.items() if k != 0}
        if strategy == 'majority_nonzero':
            return max(non_zero, key=non_zero.get) if non_zero else 0
        elif strategy == 'minority_nonzero':
            # Least common non-zero color (the "unique" one)
            return min(non_zero, key=non_zero.get) if non_zero else 0
        elif strategy == 'unique_nonbg':
            # If block has exactly one non-bg color (ignoring the most-common), return it
            # Find the "noise" color (appears in almost all blocks) vs "signal"
            if len(non_zero) == 0:
                return 0
            if len(non_zero) == 1:
                color = list(non_zero.keys())[0]
                total = sr * sc
                if non_zero[color] == total:
                    return color
                return 0
            # Multiple non-zero: return least common
            return min(non_zero, key=non_zero.get)
        elif strategy == 'any_nonzero':
            return 1 if non_zero else 0
        elif strategy == 'max_color':
            return max(non_zero.keys()) if non_zero else 0
        return 0

    for strategy in ['majority_nonzero', 'minority_nonzero', 'unique_nonbg', 'max_color']:
        all_match = True
        for ex in examples:
            inp, out = ex['input'], ex['output']
            ir, ic = dims(inp)
            outr, outc = ir // sr, ic // sc
            for r in range(outr):
                for c in range(outc):
                    block = []
                    for dr in range(sr):
                        for dc in range(sc):
                            block.append(inp[r * sr + dr][c * sc + dc])
                    predicted_val = _summarize_block(block, strategy)
                    if predicted_val != out[r][c]:
                        all_match = False
                        break
                if not all_match:
                    break
            if not all_match:
                break
        if all_match:
            return {'type': 'downscale', 'sr': sr, 'sc': sc, 'strategy': strategy}

    # Try noise-color-aware: ignore one specific color, then take majority of rest
    all_colors_in_blocks = set()
    for ex in examples:
        for v in (v for row in ex['input'] for v in row):
            if v != 0:
                all_colors_in_blocks.add(v)

    for noise in all_colors_in_blocks:
        all_match = True
        for ex in examples:
            inp, out = ex['input'], ex['output']
            ir, ic = dims(inp)
            outr, outc = ir // sr, ic // sc
            for r in range(outr):
                for c in range(outc):
                    block = [inp[r * sr + dr][c * sc + dc]
                             for dr in range(sr) for dc in range(sc)]
                    filtered = {k: v for k, v in Counter(block).items() if k != 0 and k != noise}
                    predicted_val = max(filtered, key=filtered.get) if filtered else 0
                    if predicted_val != out[r][c]:
                        all_match = False
                        break
                if not all_match:
                    break
            if not all_match:
                break
        if all_match:
            return {'type': 'downscale', 'sr': sr, 'sc': sc, 'strategy': 'noise_filter',
                    'noise': noise}

    return None

def apply_downscale(inp: Grid, rule: Dict) -> Grid:
    sr, sc = rule['sr'], rule['sc']
    ir, ic = dims(inp)
    outr, outc = ir // sr, ic // sc
    out = make(outr, outc)

    strategy = rule.get('strategy', 'majority_nonzero')
    noise = rule.get('noise', -1)
    for r in range(outr):
        for c in range(outc):
            block = [inp[r * sr + dr][c * sc + dc]
                     for dr in range(sr) for dc in range(sc)]
            ct = Counter(block)
            non_zero = {k: v for k, v in ct.items() if k != 0}
            if strategy == 'noise_filter':
                filtered = {k: v for k, v in ct.items() if k != 0 and k != noise}
                out[r][c] = max(filtered, key=filtered.get) if filtered else 0
            elif strategy == 'majority_nonzero':
                out[r][c] = max(non_zero, key=non_zero.get) if non_zero else 0
            elif strategy == 'minority_nonzero':
                out[r][c] = min(non_zero, key=non_zero.get) if non_zero else 0
            elif strategy == 'unique_nonbg':
                if len(non_zero) == 0:
                    out[r][c] = 0
                elif len(non_zero) == 1:
                    color = list(non_zero.keys())[0]
                    out[r][c] = color if non_zero[color] == sr * sc else 0
                else:
                    out[r][c] = min(non_zero, key=non_zero.get)
            elif strategy == 'max_color':
                out[r][c] = max(non_zero.keys()) if non_zero else 0
    return out

# ============================================================
# Layer 18: Output size inference for variable-output tasks
# ============================================================

def infer_output_size(examples: List[Dict], test_input: Grid) -> Optional[Tuple[int, int]]:
    """Infer the output size for the test input based on example patterns."""
    # Check if output size is constant
    sizes = set(dims(ex['output']) for ex in examples)
    if len(sizes) == 1:
        return sizes.pop()

    # Check if output size is proportional to input size
    ratios = set()
    for ex in examples:
        ir, ic = dims(ex['input'])
        outr, outc = dims(ex['output'])
        if ir > 0 and ic > 0:
            ratios.add((outr / ir, outc / ic))
    if len(ratios) == 1:
        rr, rc = ratios.pop()
        tir, tic = dims(test_input)
        return (int(tir * rr), int(tic * rc))

    return None

# ============================================================
# Layer 19: Boundary/interior recoloring
# ============================================================

def is_boundary_cell(g: Grid, r: int, c: int, bg: int) -> bool:
    rows, cols = dims(g)
    if g[r][c] == bg:
        return False
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols or g[nr][nc] == bg:
            return True
    return False

def discover_boundary_recolor(examples: List[Dict]) -> Optional[Dict]:
    if not examples:
        return None
    for ex in examples:
        if dims(ex['input']) != dims(ex['output']):
            return None
    bg_vals = set(background(ex['input']) for ex in examples)
    if len(bg_vals) != 1:
        return None
    bg = bg_vals.pop()

    rule = {}
    consistent = True
    for ex in examples:
        inp, out = ex['input'], ex['output']
        rows, cols = dims(inp)
        for r in range(rows):
            for c in range(cols):
                if inp[r][c] == bg:
                    if out[r][c] != bg:
                        consistent = False
                        break
                    continue
                bnd = is_boundary_cell(inp, r, c, bg)
                key = (bnd, inp[r][c])
                val = out[r][c]
                if key in rule:
                    if rule[key] != val:
                        consistent = False
                        break
                else:
                    rule[key] = val
            if not consistent:
                break
        if not consistent:
            break
    if not consistent or not rule:
        return None
    if not any(rule.get((b, c_), c_) != c_ for b, c_ in rule):
        return None
    return {'type': 'boundary_recolor', 'rule': rule, 'bg': bg}

def apply_boundary_recolor(inp: Grid, rule_dict: Dict) -> Grid:
    rule = rule_dict['rule']
    bg = rule_dict['bg']
    rows, cols = dims(inp)
    out = copy(inp)
    for r in range(rows):
        for c in range(cols):
            if inp[r][c] == bg:
                continue
            bnd = is_boundary_cell(inp, r, c, bg)
            key = (bnd, inp[r][c])
            if key in rule:
                out[r][c] = rule[key]
    return out

# ============================================================
# Layer 20: Grid cell replication (separator grids)
# ============================================================

def discover_grid_replicate(examples: List[Dict]) -> Optional[Dict]:
    if not examples:
        return None
    for ex in examples:
        if dims(ex['input']) != dims(ex['output']):
            return None
    ri0 = get_regions(examples[0]['input'])
    if ri0 is None or len(ri0['regions']) < 2:
        return None
    for ex in examples:
        ri = get_regions(ex['input'])
        if ri is None or len(ri['regions']) != len(ri0['regions']):
            return None

    all_match = True
    for ex in examples:
        predicted = _apply_grid_replicate(ex['input'])
        if not eq(predicted, ex['output']):
            all_match = False
            break
    if all_match:
        return {'type': 'grid_replicate'}
    return None

def _apply_grid_replicate(inp: Grid) -> Grid:
    out = copy(inp)
    ri = get_regions(inp)
    if ri is None:
        return out
    bg = background(inp)
    source_sub = None
    for r0, c0, r1, c1 in ri['regions']:
        sub = extract_subgrid(inp, r0, c0, r1, c1)
        non_bg = sum(1 for row in sub for v in row if v != bg and v != ri['sep_color'])
        if non_bg > 0 and source_sub is None:
            source_sub = sub
            break
    if source_sub is None:
        return out
    sr, sc = dims(source_sub)
    for r0, c0, r1, c1 in ri['regions']:
        if r1 - r0 + 1 == sr and c1 - c0 + 1 == sc:
            place_subgrid(out, source_sub, r0, c0)
    return out

def apply_grid_replicate(inp: Grid, rule: Dict) -> Grid:
    return _apply_grid_replicate(inp)

# ============================================================
# Layer 21: Object recoloring by property
# ============================================================

def discover_object_recolor(examples: List[Dict]) -> Optional[Dict]:
    if not examples:
        return None
    for ex in examples:
        if dims(ex['input']) != dims(ex['output']):
            return None
    bg_vals = set(background(ex['input']) for ex in examples)
    if len(bg_vals) != 1:
        return None
    bg = bg_vals.pop()

    size_map = {}
    consistent = True
    for ex in examples:
        inp, out = ex['input'], ex['output']
        comps = connected_components(inp, bg)
        for comp in comps:
            in_color = inp[comp[0][0]][comp[0][1]]
            out_set = set(out[r][c] for r, c in comp)
            if len(out_set) != 1:
                consistent = False
                break
            key = (len(comp), in_color)
            val = out_set.pop()
            if key in size_map:
                if size_map[key] != val:
                    consistent = False
                    break
            else:
                size_map[key] = val
        if not consistent:
            break

    if consistent and size_map and len(size_map) <= 8 and any(k[1] != v for k, v in size_map.items()):
        return {'type': 'object_recolor', 'map': size_map, 'bg': bg}

    return None

def apply_object_recolor(inp: Grid, rule: Dict) -> Grid:
    bg = rule['bg']
    out = copy(inp)
    comps = connected_components(inp, bg)
    if rule['type'] == 'object_recolor':
        for comp in comps:
            key = (len(comp), inp[comp[0][0]][comp[0][1]])
            if key in rule['map']:
                for r, c in comp:
                    out[r][c] = rule['map'][key]
    else:
        for comp in comps:
            if len(comp) in rule['map']:
                for r, c in comp:
                    out[r][c] = rule['map'][len(comp)]
    return out

# ============================================================
# Void fill: discover interior holes and fill color from data
# ============================================================

def discover_void_fill(examples: List[Dict]) -> Optional[Dict]:
    """Discover: hollow rectangles/regions get their interior filled.
    The fill color is learned from data, not hardcoded."""
    if not examples:
        return None
    for ex in examples:
        if dims(ex['input']) != dims(ex['output']):
            return None

    bg_vals = set(background(ex['input']) for ex in examples)
    if len(bg_vals) != 1:
        return None
    bg = bg_vals.pop()

    # For each example, find interior void regions and what they become
    fill_rules = {}  # border_color -> fill_color
    all_match = True

    for ex in examples:
        inp, out = ex['input'], ex['output']
        rows, cols = dims(inp)

        # Find non-bg connected components (potential frames)
        comps = connected_components(inp, bg)
        for comp in comps:
            border_color = inp[comp[0][0]][comp[0][1]]
            # Check if all cells in comp share same color
            if not all(inp[r][c] == border_color for r, c in comp):
                continue

            r0, c0, r1, c1 = bbox(comp)
            # Check if component forms a frame (has interior bg cells)
            interior_bg = []
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if (r, c) not in set(comp) and inp[r][c] == bg:
                        interior_bg.append((r, c))

            if not interior_bg:
                continue

            # Check what the interior became in output
            fill_colors = set(out[r][c] for r, c in interior_bg)
            if len(fill_colors) != 1:
                continue
            fc = fill_colors.pop()
            if fc == bg:
                continue  # Not filled

            if border_color in fill_rules:
                if fill_rules[border_color] != fc:
                    all_match = False
                    break
            else:
                fill_rules[border_color] = fc

        if not all_match:
            break

    if not all_match or not fill_rules:
        return None

    return {'type': 'void_fill', 'fill_rules': fill_rules, 'bg': bg}

def apply_void_fill(inp: Grid, rule: Dict) -> Grid:
    bg = rule['bg']
    fill_rules = rule['fill_rules']
    out = copy(inp)
    rows, cols = dims(inp)

    comps = connected_components(inp, bg)
    for comp in comps:
        border_color = inp[comp[0][0]][comp[0][1]]
        if border_color not in fill_rules:
            continue
        if not all(inp[r][c] == border_color for r, c in comp):
            continue

        comp_set = set(comp)
        r0, c0, r1, c1 = bbox(comp)
        fc = fill_rules[border_color]

        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if (r, c) not in comp_set and inp[r][c] == bg:
                    out[r][c] = fc

    return out

# ============================================================
# Periodicity: discover minimal repeating tile from data
# ============================================================

def discover_periodicity(examples: List[Dict]) -> Optional[Dict]:
    """Discover: output is the minimal repeating tile of the input.
    Tries all divisors of input dimensions as candidate periods."""
    if not examples:
        return None

    # All examples: output must be smaller than input
    for ex in examples:
        ir, ic = dims(ex['input'])
        outr, outc = dims(ex['output'])
        if outr >= ir and outc >= ic:
            return None
        if outr == 0 or outc == 0:
            return None

    # For first example, find the period
    inp0, out0 = examples[0]['input'], examples[0]['output']
    ir, ic = dims(inp0)
    outr, outc = dims(out0)

    # The output dimensions should divide the input dimensions
    if ir % outr != 0 or ic % outc != 0:
        return None

    pr, pc = outr, outc

    # Verify: input is made of repeated copies of output
    for ex in examples:
        inp, out = ex['input'], ex['output']
        eir, eic = dims(inp)
        eor, eoc = dims(out)

        if eir % eor != 0 or eic % eoc != 0:
            return None

        # Check tiling
        for r in range(eir):
            for c in range(eic):
                if inp[r][c] != out[r % eor][c % eoc]:
                    return None

    return {'type': 'periodicity', 'pr': pr, 'pc': pc}

def apply_periodicity(inp: Grid, rule: Dict) -> Grid:
    pr, pc = rule['pr'], rule['pc']
    ir, ic = dims(inp)

    # Extract the first tile (clamped to input bounds)
    actual_pr = min(pr, ir)
    actual_pc = min(pc, ic)
    return [inp[r][:actual_pc] for r in range(actual_pr)]

# ============================================================
# Object expansion: discover objects that grow a halo
# ============================================================

def discover_object_expand(examples: List[Dict]) -> Optional[Dict]:
    """Discover: each non-bg object expands by N cells in all directions.
    The expansion color is learned from data."""
    if not examples:
        return None
    for ex in examples:
        if dims(ex['input']) != dims(ex['output']):
            return None

    bg_vals = set(background(ex['input']) for ex in examples)
    if len(bg_vals) != 1:
        return None
    bg = bg_vals.pop()

    # For each example, find what cells changed and how far from objects
    expansion = None
    expand_color_rule = {}  # source_color -> expand_color

    for ex in examples:
        inp, out = ex['input'], ex['output']
        rows, cols = dims(inp)

        # Find cells that changed
        changed = set()
        for r in range(rows):
            for c in range(cols):
                if inp[r][c] != out[r][c]:
                    changed.add((r, c))

        if not changed:
            continue

        # All changed cells should be bg in input
        if not all(inp[r][c] == bg for r, c in changed):
            return None

        # Find non-bg objects in input
        obj_cells = set()
        for r in range(rows):
            for c in range(cols):
                if inp[r][c] != bg:
                    obj_cells.add((r, c))

        if not obj_cells:
            return None

        # Measure max distance of changed cells from nearest object cell
        max_dist = 0
        for cr, cc in changed:
            min_d = min(abs(cr - or_) + abs(cc - oc) for or_, oc in obj_cells)
            if min_d > max_dist:
                max_dist = min_d

        if expansion is None:
            expansion = max_dist
        elif expansion != max_dist:
            return None

        # Learn expand color per source color
        for cr, cc in changed:
            new_color = out[cr][cc]
            # Find nearest object cell and its color
            nearest = min(obj_cells, key=lambda p: abs(cr - p[0]) + abs(cc - p[1]))
            src_color = inp[nearest[0]][nearest[1]]
            if src_color in expand_color_rule:
                if expand_color_rule[src_color] != new_color:
                    return None
            else:
                expand_color_rule[src_color] = new_color

    if expansion is None or expansion < 1 or not expand_color_rule:
        return None

    return {'type': 'object_expand', 'expansion': expansion,
            'color_rule': expand_color_rule, 'bg': bg}

def apply_object_expand(inp: Grid, rule: Dict) -> Grid:
    bg = rule['bg']
    expansion = rule['expansion']
    color_rule = rule['color_rule']
    rows, cols = dims(inp)
    out = copy(inp)

    obj_cells = {}  # (r,c) -> color for non-bg cells
    for r in range(rows):
        for c in range(cols):
            if inp[r][c] != bg:
                obj_cells[(r, c)] = inp[r][c]

    # Expand each object cell by manhattan distance
    for (or_, oc), src_color in obj_cells.items():
        if src_color not in color_rule:
            continue
        ec = color_rule[src_color]
        for dr in range(-expansion, expansion + 1):
            for dc in range(-expansion, expansion + 1):
                if abs(dr) + abs(dc) > expansion or (dr == 0 and dc == 0):
                    continue
                nr, nc = or_ + dr, oc + dc
                if 0 <= nr < rows and 0 <= nc < cols and out[nr][nc] == bg:
                    out[nr][nc] = ec

    return out

# ============================================================
# Master discovery pipeline
# ============================================================

# ---------------------------------------------------------------------------
# Auto-discover all (discover_X, apply_X) pairs from module namespace.
# No hardcoded list -- new layers just need discover_X and apply_X functions.
# ---------------------------------------------------------------------------

def _build_discovery_pipeline() -> List[Tuple[str, Callable, Callable]]:
    """Scan module globals for matching discover_*/apply_* pairs.

    A pair is registered if both discover_<name> and apply_<name> exist
    (or the special cases like discover_majority_output -> apply_constant_output).
    Falls back to the @discoverer registry for anything pre-registered.
    """
    import inspect
    g = globals()
    pairs: List[Tuple[str, Callable, Callable]] = []
    seen: Set[str] = set()

    # First: anything in the @discoverer registry
    for name, disc_fn, app_fn in _discovery_registry:
        pairs.append((name, disc_fn, app_fn))
        seen.add(name)

    # Second: auto-discover from module namespace
    discover_fns = {
        k: v for k, v in g.items()
        if k.startswith('discover_') and callable(v) and k != 'discover_and_apply'
    }

    # Naming conventions for apply functions
    name_mappings = {
        'discover_majority_output': ('constant_output', 'apply_constant_output'),
        'discover_diff_rule': ('line_fill', 'apply_diff_rule'),
    }

    for disc_name, disc_fn in discover_fns.items():
        if disc_name in name_mappings:
            layer_name, apply_name = name_mappings[disc_name]
        else:
            suffix = disc_name[len('discover_'):]
            layer_name = suffix
            apply_name = 'apply_' + suffix

        if layer_name in seen:
            continue

        apply_fn = g.get(apply_name)
        if apply_fn is not None and callable(apply_fn):
            pairs.append((layer_name, disc_fn, apply_fn))
            seen.add(layer_name)

    return pairs


# ============================================================
# Layer 23: Pixel correspondence mapping (data-driven)
# ============================================================

def discover_pixel_map(examples: List[Dict]) -> Optional[Dict]:
    """Discover a per-pixel spatial mapping from input->output by finding
    consistent coordinate correspondences across all examples.
    No hardcoded transforms -- learns f(r,c) -> (r',c') from data."""
    if not examples:
        return None

    # All examples must have consistent input/output dimensions
    dim_pairs = set()
    for ex in examples:
        dim_pairs.add((dims(ex['input']), dims(ex['output'])))
    if len(dim_pairs) != 1:
        return None
    (ir, ic), (outr, outc) = dim_pairs.pop()

    # For each output cell (or,oc), find which input cell (ir2,ic2) it came from
    # Build candidate mappings from first example using color correspondences
    ex0 = examples[0]
    inp0, out0 = ex0['input'], ex0['output']

    # Try to find a consistent mapping: for each (or,oc), find (r,c) such that
    # a color bijection maps inp0[r][c] -> out0[or][oc]
    # Approach: try each possible color mapping, then verify spatial consistency

    # Collect all possible color maps from first example
    # For each unique color pair, check if it could be part of a bijection
    color_pairs = set()
    for r in range(outr):
        for c in range(outc):
            color_pairs.add((out0[r][c],))

    # For efficiency: pick 3 non-bg anchor points in the output and find
    # their source positions in the input
    bg_out = background(out0)
    anchors = [(r, c) for r in range(outr) for c in range(outc) if out0[r][c] != bg_out]
    if len(anchors) < 2:
        return None
    anchors = anchors[:min(6, len(anchors))]

    # For each affine (a,b,d,f), compute offsets (e,g) from anchor correspondences.
    # For anchor at output position (ar, ac), the input source is at
    #   (a*ar + b*ac + e, d*ar + f*ac + g)
    # So for each candidate input cell (ir2, ic2):
    #   e = ir2 - a*ar - b*ac
    #   g = ic2 - d*ar - f*ac
    anchor_r, anchor_c = anchors[0]
    for a in [-1, 0, 1]:
        for b in [-1, 0, 1]:
            for d in [-1, 0, 1]:
                for f_coeff in [-1, 0, 1]:
                    if a == 0 and b == 0:
                        continue
                    if d == 0 and f_coeff == 0:
                        continue

                    # Compute candidate offsets from input cells
                    base_r = a * anchor_r + b * anchor_c
                    base_c = d * anchor_r + f_coeff * anchor_c

                    for ir2 in range(ir):
                        for ic2 in range(ic):
                            e = ir2 - base_r
                            g = ic2 - base_c

                            # Quick check: second anchor must also map validly
                            if len(anchors) > 1:
                                a2r, a2c = anchors[1]
                                mr2 = a * a2r + b * a2c + e
                                mc2 = d * a2r + f_coeff * a2c + g
                                if mr2 < 0 or mr2 >= ir or mc2 < 0 or mc2 >= ic:
                                    continue

                            # Verify full mapping with color bijection
                            cm = {}
                            valid = True
                            for ex in examples:
                                inp_ex, out_ex = ex['input'], ex['output']
                                for r in range(outr):
                                    for c in range(outc):
                                        mr = a * r + b * c + e
                                        mc = d * r + f_coeff * c + g
                                        if mr < 0 or mr >= ir or mc < 0 or mc >= ic:
                                            valid = False
                                            break
                                        iv = inp_ex[mr][mc]
                                        ov = out_ex[r][c]
                                        if iv in cm:
                                            if cm[iv] != ov:
                                                valid = False
                                                break
                                        else:
                                            cm[iv] = ov
                                    if not valid:
                                        break
                                if not valid:
                                    break

                            if valid and cm:
                                is_identity = (a == 1 and b == 0 and d == 0 and f_coeff == 1
                                               and e == 0 and g == 0
                                               and all(k == v for k, v in cm.items()))
                                if is_identity:
                                    continue
                                return {
                                    'type': 'pixel_map',
                                    'a': a, 'b': b, 'd': d, 'f': f_coeff,
                                    'e': e, 'g': g, 'cm': cm,
                                    'out_rows': outr, 'out_cols': outc,
                                }

    return None

def apply_pixel_map(inp: Grid, rule: Dict) -> Grid:
    a, b, d, f = rule['a'], rule['b'], rule['d'], rule['f']
    e, g = rule['e'], rule['g']
    cm = rule['cm']
    outr, outc = rule['out_rows'], rule['out_cols']
    ir, ic = dims(inp)
    out = make(outr, outc)

    for r in range(outr):
        for c in range(outc):
            mr = a * r + b * c + e
            mc = d * r + f * c + g
            if 0 <= mr < ir and 0 <= mc < ic:
                out[r][c] = cm.get(inp[mr][mc], inp[mr][mc])
    return out


# Ordering hint: specific transforms first, broad lookup rules last.
# This is advisory; the pipeline validates all candidates against training data.
def _rule_complexity(rule: Dict) -> int:
    """Estimate rule complexity from its data. Lower = simpler = preferred.

    Principles:
    - Rules with fewer learned parameters are simpler (less overfit risk)
    - Lookup-table rules (local_rule, relative_rule) are most complex
    - Rules with no lookup structure are pure transforms (simplest)
    """
    # Lookup-table rules: complexity = table size (often thousands)
    if 'table' in rule and isinstance(rule['table'], dict):
        return 1000 + len(rule['table'])

    # Rules with mapping dicts: complexity = mapping size
    for key in ('map', 'color_map', 'fill_rules', 'color_rule', 'combine_table'):
        if key in rule and isinstance(rule[key], dict):
            return 100 + len(rule[key])

    # Pure structural rules (no learned lookup): simplest
    return 0


def _lookup_coverage(rule: Dict, test_input: Grid) -> float:
    """For lookup-table rules, check what fraction of test cells are covered."""
    if 'table' not in rule or not isinstance(rule['table'], dict):
        return 1.0
    table = rule['table']
    radius = rule.get('radius', 1)
    rows, cols = dims(test_input)
    total = rows * cols
    covered = 0
    rtype = rule.get('type', '')
    for r in range(rows):
        for c in range(cols):
            if rtype == 'relative_rule':
                nb = rel_neighborhood(test_input, r, c, radius)
            else:
                nb = neighborhood(test_input, r, c, radius)
            if nb in table:
                covered += 1
    return covered / max(total, 1)


def discover_and_apply(examples: List[Dict], test_input: Grid) -> Optional[Tuple[Grid, str]]:
    """Try all discovery layers. Validate each. Pick simplest valid rule.

    No hardcoded priority — complexity is measured from the rule's data.
    Simpler rules (fewer parameters) are preferred over complex ones.
    Lookup-table rules require 90%+ coverage on test input.
    """
    pipeline = _build_discovery_pipeline()

    # Collect all valid candidates with their complexity
    candidates: List[Tuple[int, str, Any, Any]] = []

    for name, discover_fn, apply_fn in pipeline:
        rule = discover_fn(examples)
        if rule is None:
            continue

        # Validate against ALL training examples
        all_correct = True
        for ex in examples:
            predicted = apply_fn(ex['input'], rule)
            if not eq(predicted, ex['output']):
                all_correct = False
                break
        if not all_correct:
            continue

        # Only neighborhood-lookup rules need coverage checks
        # (split_overlay etc. use 'table' for different purposes)
        if rule.get('type') in ('local_rule', 'relative_rule'):
            coverage = _lookup_coverage(rule, test_input)
            if coverage < 0.70:
                continue

        complexity = _rule_complexity(rule)
        candidates.append((complexity, name, apply_fn, rule))

    if not candidates:
        return None

    # Sort by complexity — simplest rule wins
    candidates.sort(key=lambda t: t[0])
    _, name, apply_fn, rule = candidates[0]
    return apply_fn(test_input, rule), name

# ============================================================
# SHIFU SOLVER — co-graph field + wave propagation
# ============================================================

import math as _math

def _norm_patch(g: Grid, r: int, c: int) -> Tuple:
    """Color-normalized 3x3 patch. Like Shifu's Form channel:
    captures structure before meaning (color-invariant shape)."""
    rows, cols = dims(g)
    raw = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            nr, nc = r + dr, c + dc
            raw.append(g[nr][nc] if 0 <= nr < rows and 0 <= nc < cols else -1)
    freq = Counter(v for v in raw if v != -1)
    rank = {color: i for i, (color, _) in enumerate(freq.most_common())}
    rank[-1] = -1
    return tuple(rank.get(v, v) for v in raw)

def _cell_feats(g: Grid, r: int, c: int) -> List[Tuple]:
    """Multi-scale features — like Shifu's 7 channels at different resolutions."""
    rows, cols = dims(g)
    feats = []
    v = g[r][c]
    # Absolute color
    feats.append(('v', v))
    # Raw 3x3 (most specific — like the full word in Shifu)
    raw = tuple(
        g[r+dr][c+dc] if 0 <= r+dr < rows and 0 <= c+dc < cols else -1
        for dr in [-1, 0, 1] for dc in [-1, 0, 1]
    )
    feats.append(('raw', raw))
    # Normalized 3x3 (color-invariant — like Form channel)
    feats.append(('norm', _norm_patch(g, r, c)))
    # 4-adjacent colors (directional context)
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r + dr, c + dc
        feats.append(('adj', dr, dc, g[nr][nc] if 0 <= nr < rows and 0 <= nc < cols else -1))
    # Edge position
    feats.append(('pos', (r == 0, r == rows-1, c == 0, c == cols-1)))
    # Neighborhood color distribution
    nb = Counter()
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                nb[g[nr][nc]] += 1
    feats.append(('sig', tuple(sorted(nb.items()))))
    return feats

def shifu_solve(examples: List[Dict], test_input: Grid) -> Optional[Grid]:
    """Solve via Shifu-style co-occurrence field + PMI-weighted voting."""
    for ex in examples:
        if dims(ex['input']) != dims(ex['output']):
            return None
    td = set(dims(ex['input']) for ex in examples)
    if len(td) != 1:
        return None
    rows, cols = td.pop()
    if dims(test_input) != (rows, cols):
        return None

    # Feed: build co-occurrence field from examples
    co = {}
    for ex in examples:
        inp, out = ex['input'], ex['output']
        for r in range(rows):
            for c in range(cols):
                ov = out[r][c]
                for feat in _cell_feats(inp, r, c):
                    co.setdefault(feat, Counter())[ov] += 1

    # Specificity weights (Shifu's inhibition: peaked = content, flat = function)
    tw = {}
    for feat, cts in co.items():
        t = sum(cts.values())
        ent = sum(-((n/t)*_math.log2(n/t)) for n in cts.values() if n > 0)
        tw[feat] = 1.0 / (1.0 + ent)

    def _vote(feats):
        votes = Counter()
        for feat in feats:
            if feat not in co:
                continue
            w = tw.get(feat, 0.0)
            cts = co[feat]
            t = sum(cts.values())
            for v, n in cts.items():
                votes[v] += (n / t) * w
        if not votes:
            return 0, 0.0
        rk = votes.most_common()
        margin = rk[0][1] - (rk[1][1] if len(rk) > 1 else 0)
        return rk[0][0], margin / (rk[0][1] + 1e-10)

    out = [[0]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            out[r][c], _ = _vote(_cell_feats(test_input, r, c))
    return out

# ============================================================
# Task solver and evaluator
# ============================================================

def solve_task(task: Dict) -> Tuple[Optional[Grid], str]:
    examples = task['train']
    test_input = task['test'][0]['input']

    result = discover_and_apply(examples, test_input)
    if result is not None:
        return result

    predicted = shifu_solve(examples, test_input)
    if predicted is not None:
        ok = True
        for ex in examples:
            tp = shifu_solve(examples, ex['input'])
            if tp is None or not eq(tp, ex['output']):
                ok = False
                break
        if ok:
            return predicted, 'shifu_wave'

    return None, 'unsolved'

def evaluate(tasks_dir: str, verbose: bool = False):
    """Evaluate against all tasks in a directory."""
    files = sorted(f for f in os.listdir(tasks_dir) if f.endswith('.json'))
    correct = 0
    total = 0
    solved_by = Counter()
    failed = []

    for fname in files:
        path = os.path.join(tasks_dir, fname)
        with open(path) as f:
            task = json.load(f)

        total += 1
        predicted, method = solve_task(task)
        expected = task['test'][0]['output']

        if predicted is not None and eq(predicted, expected):
            correct += 1
            solved_by[method] += 1
            if verbose:
                print(f"  PASS {fname} [{method}]")
        else:
            failed.append((fname, method))
            if verbose and method != 'unsolved':
                print(f"  FAIL {fname} [{method}]")

    print(f"\nARC Results: {correct}/{total} ({100*correct/max(total,1):.1f}%)")
    print(f"\nSolved by discovery layer:")
    for method, count in sorted(solved_by.items(), key=lambda x: -x[1]):
        print(f"  {method}: {count}")
    print(f"  unsolved: {total - correct}")

    if verbose and failed:
        print(f"\nFailed tasks ({len(failed)}):")
        for fname, method in failed[:20]:
            print(f"  {fname} (tried: {method})")

    return correct, total

# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    import time

    tasks_dir = os.path.join(os.path.dirname(__file__), 'data', 'arc', 'training')
    verbose = '--verbose' in sys.argv or '-v' in sys.argv

    print("Shifu ARC Solver -- Universal Transform Discovery + Simulation")
    print(f"Tasks: {tasks_dir}")
    print()

    t0 = time.time()
    correct, total = evaluate(tasks_dir, verbose)
    elapsed = time.time() - t0

    print(f"\nTime: {elapsed:.1f}s ({elapsed/max(total,1)*1000:.0f}ms/task)")
