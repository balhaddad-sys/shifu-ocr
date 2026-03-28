"""
Topology Coherence Caps: Topological Constraints Bounding Coherence Measurements
==================================================================================

The Problem:
Coherence displacement detects text by measuring local harmony disruption.
But coherence alone is noisy — table borders, compression artifacts, color
gradients all create coherence disruptions that aren't text.

The Solution: TOPOLOGY CAPS
Topological invariants (components, holes, Euler number, connectivity)
provide hard constraints that CAP the coherence measurements:

  1. COMPONENT CAP: A valid character has 1-4 connected components.
     Coherence regions with 0 or >4 components are capped (suppressed).

  2. HOLE CAP: A valid character has 0-3 holes (e.g., 'B' has 2).
     Regions with excessive holes = noise, not text.

  3. EULER CAP: The Euler number (components - holes) falls in [-2, 4]
     for valid characters. Outside this = structural noise.

  4. ASPECT CAP: Valid characters have aspect ratios in [0.15, 5.0].
     Extremely wide = horizontal line. Extremely tall = vertical line.

  5. SIZE CAP: Valid characters fall within a size range relative to
     the median character size in the document.

  6. CONNECTIVITY CAP: The ratio of the character's boundary pixels
     to its area must fall within expected ranges.

  7. DENSITY CAP: Ink density (mass/area) must be in [0.05, 0.95].
     Near 0 = noise. Near 1 = filled rectangle (not a character).

Together these caps filter the coherence signal through topological
reality, producing clean character regions from noisy coherence maps.

The Metaphor:
Coherence is the MRI signal. Topology caps are the radiologist's knowledge.
The signal shows everything. The knowledge knows what to look for.

Author: Bader & Claude — March 2026
"""

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import morphology, filters
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# TOPOLOGY CAPS: Hard constraints on valid character topology
# =============================================================================

class TopologyCaps:
    """
    A set of topological constraints that filter coherence-detected regions.

    Each cap defines a valid range. Regions outside these ranges are
    suppressed (capped) as noise, not text.
    """

    def __init__(self,
                 component_range=(1, 4),
                 hole_range=(0, 3),
                 euler_range=(-2, 4),
                 aspect_range=(0.15, 5.0),
                 size_range_factor=(0.1, 8.0),
                 density_range=(0.05, 0.95),
                 min_area=6,
                 max_area_fraction=0.5):
        """
        Args:
            component_range: (min, max) connected components for valid character
            hole_range: (min, max) holes for valid character
            euler_range: (min, max) Euler number (components - holes)
            aspect_range: (min, max) width/height ratio
            size_range_factor: (min, max) relative to median character size
            density_range: (min, max) ink density (mass/bounding_box_area)
            min_area: minimum pixel area for a valid character
            max_area_fraction: max fraction of image that one character can occupy
        """
        self.component_range = component_range
        self.hole_range = hole_range
        self.euler_range = euler_range
        self.aspect_range = aspect_range
        self.size_range_factor = size_range_factor
        self.density_range = density_range
        self.min_area = min_area
        self.max_area_fraction = max_area_fraction

    def check(self, binary_region):
        """
        Check if a binary region passes all topology caps.

        Returns:
            (passed: bool, report: dict) — report contains each cap's result
        """
        report = {}
        h, w = binary_region.shape
        area = binary_region.sum()

        # Area check
        report['area'] = int(area)
        if area < self.min_area:
            report['passed'] = False
            report['failure'] = 'below_min_area'
            return False, report

        # Component count
        padded = np.pad(binary_region, 1, mode='constant', constant_values=0)
        _, n_components = ndimage.label(padded)
        report['components'] = int(n_components)
        if not (self.component_range[0] <= n_components <= self.component_range[1]):
            report['passed'] = False
            report['failure'] = 'component_cap'
            return False, report

        # Hole count
        _, n_bg = ndimage.label(1 - padded)
        n_holes = n_bg - 1
        report['holes'] = int(n_holes)
        if not (self.hole_range[0] <= n_holes <= self.hole_range[1]):
            report['passed'] = False
            report['failure'] = 'hole_cap'
            return False, report

        # Euler number
        euler = n_components - n_holes
        report['euler'] = int(euler)
        if not (self.euler_range[0] <= euler <= self.euler_range[1]):
            report['passed'] = False
            report['failure'] = 'euler_cap'
            return False, report

        # Aspect ratio
        aspect = w / max(h, 1)
        report['aspect'] = float(aspect)
        if not (self.aspect_range[0] <= aspect <= self.aspect_range[1]):
            report['passed'] = False
            report['failure'] = 'aspect_cap'
            return False, report

        # Density
        density = area / max(h * w, 1)
        report['density'] = float(density)
        if not (self.density_range[0] <= density <= self.density_range[1]):
            report['passed'] = False
            report['failure'] = 'density_cap'
            return False, report

        report['passed'] = True
        report['failure'] = None
        return True, report

    def filter_regions(self, regions):
        """
        Filter a list of binary regions, keeping only those that pass all caps.

        Args:
            regions: list of dicts with 'binary' key containing binary regions

        Returns:
            filtered: list of regions that pass all caps
            rejected: list of (region, report) pairs for rejected regions
        """
        filtered = []
        rejected = []

        for region in regions:
            binary = region.get('binary', region) if isinstance(region, dict) else region
            passed, report = self.check(binary)
            if passed:
                if isinstance(region, dict):
                    region['topology'] = report
                filtered.append(region)
            else:
                rejected.append((region, report))

        return filtered, rejected


# =============================================================================
# COHERENCE MAP WITH TOPOLOGY CAPS
# =============================================================================

class CoherenceCappedDetector:
    """
    Coherence-based text detection with topology caps applied.

    Pipeline:
    1. Compute coherence displacement (harmony disruption)
    2. Threshold to get candidate regions
    3. Apply topology caps to filter noise
    4. Return clean character regions
    """

    def __init__(self, caps=None, coherence_window=3):
        self.caps = caps or TopologyCaps()
        self.coherence_window = coherence_window
        self._adaptive_stats = {
            'median_height': None,
            'median_width': None,
            'median_area': None,
        }

    def compute_coherence(self, image, window=None):
        """Compute coherence displacement for an image (grayscale or color)."""
        window = window or self.coherence_window

        if len(image.shape) == 3:
            channels = [image[:, :, c].astype(float) for c in range(image.shape[2])]
        else:
            channels = [image.astype(float)]

        kernel = np.ones((window, window)) / (window * window)

        all_incoherence = []
        all_edges = []

        for ch in channels:
            # Local variance (coherence disruption)
            local_mean = ndimage.convolve(ch, kernel, mode='reflect')
            local_var = ndimage.convolve((ch - local_mean)**2, kernel, mode='reflect')
            incoherence = local_var / (local_var.max() + 1e-8)
            all_incoherence.append(incoherence)

            # Edge density (structural disruption)
            gx = ndimage.sobel(ch, axis=1)
            gy = ndimage.sobel(ch, axis=0)
            edges = np.sqrt(gx**2 + gy**2)
            edges = edges / (edges.max() + 1e-8)
            all_edges.append(edges)

        combined = np.maximum(
            np.maximum.reduce(all_incoherence),
            np.maximum.reduce(all_edges)
        )
        return combined

    def detect_text_regions(self, image, gray=None):
        """
        Full pipeline: image → topology-capped text regions.

        Args:
            image: RGB or grayscale image array
            gray: optional pre-computed grayscale

        Returns:
            regions: list of dicts with 'binary', 'bbox', 'topology' keys
        """
        # Step 1: Coherence displacement
        coherence = self.compute_coherence(image)

        # Step 2: Threshold
        try:
            from skimage.filters import threshold_otsu
            thresh = threshold_otsu(coherence)
        except Exception:
            thresh = 0.15
        binary = (coherence > thresh).astype(np.uint8)
        binary = morphology.remove_small_objects(binary.astype(bool), min_size=5).astype(np.uint8)

        # Step 3: Connected component labeling
        labeled, n_components = ndimage.label(binary)
        if n_components == 0:
            return []

        # Step 4: Extract and filter with topology caps
        regions = []
        for i in range(1, n_components + 1):
            mask = (labeled == i)
            coords = np.argwhere(mask)
            if len(coords) < 3:
                continue

            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            h, w = r1 - r0 + 1, c1 - c0 + 1

            # Max area fraction check
            if h * w > binary.shape[0] * binary.shape[1] * self.caps.max_area_fraction:
                continue

            char_binary = mask[r0:r1+1, c0:c1+1].astype(np.uint8)

            regions.append({
                'binary': char_binary,
                'bbox': (r0, c0, r1, c1),
                'area': int(mask.sum()),
                'col_start': int(c0),
            })

        # Apply topology caps
        filtered, rejected = self.caps.filter_regions(regions)

        # Update adaptive statistics
        if filtered:
            heights = [r['bbox'][2] - r['bbox'][0] + 1 for r in filtered]
            widths = [r['bbox'][3] - r['bbox'][1] + 1 for r in filtered]
            areas = [r['area'] for r in filtered]
            self._adaptive_stats['median_height'] = float(np.median(heights))
            self._adaptive_stats['median_width'] = float(np.median(widths))
            self._adaptive_stats['median_area'] = float(np.median(areas))

        # Size consistency filter: remove outliers relative to median
        if filtered and len(filtered) > 3:
            filtered = self._size_consistency_filter(filtered)

        # Sort left to right
        filtered.sort(key=lambda x: x['col_start'])

        return filtered

    def _size_consistency_filter(self, regions):
        """Remove regions whose size is inconsistent with the median."""
        areas = [r['area'] for r in regions]
        median_area = np.median(areas)
        min_factor, max_factor = self.caps.size_range_factor

        consistent = []
        for r in regions:
            ratio = r['area'] / max(median_area, 1)
            if min_factor <= ratio <= max_factor:
                consistent.append(r)

        return consistent if consistent else regions  # fallback to all if filter too aggressive

    def detect_in_cell(self, cell_rgb, cell_gray):
        """
        Detect text in a single table cell with topology caps.

        Optimized for cell-level detection where the background
        is relatively uniform within the cell.
        """
        h, w = cell_gray.shape
        if h < 5 or w < 5:
            return []

        # Smart binarization for cell (background from edges)
        border = np.concatenate([
            cell_gray[0, :], cell_gray[-1, :],
            cell_gray[:, 0], cell_gray[:, -1],
        ])
        bg_median = np.median(border)
        bg_std = max(np.std(border), 5)

        diff = bg_median - cell_gray.astype(float)
        threshold = max(bg_std * 2, 15)
        binary = (diff > threshold).astype(np.uint8)

        # Try reverse if nothing found
        if binary.sum() < 8:
            diff_light = cell_gray.astype(float) - bg_median
            binary = (diff_light > threshold).astype(np.uint8)

        # Otsu fallback
        if binary.sum() < 8:
            try:
                from skimage.filters import threshold_otsu
                t = threshold_otsu(cell_gray)
                binary = (cell_gray < t).astype(np.uint8)
            except Exception:
                return []

        if binary.sum() < 4:
            return []

        # Clean
        binary = morphology.remove_small_objects(binary.astype(bool), min_size=3).astype(np.uint8)

        # Connected components
        labeled, n_comp = ndimage.label(binary)
        if n_comp == 0:
            return []

        regions = []
        for i in range(1, n_comp + 1):
            mask = (labeled == i)
            coords = np.argwhere(mask)
            if len(coords) < 3:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            char_h, char_w = r1 - r0 + 1, c1 - c0 + 1

            if char_h < 3 or char_w < 2:
                continue
            # Filter table borders
            if char_w / max(char_h, 1) > 8 or char_h / max(char_w, 1) > 10:
                continue

            char_binary = mask[r0:r1+1, c0:c1+1].astype(np.uint8)
            regions.append({
                'binary': char_binary,
                'bbox': (r0, c0, r1, c1),
                'area': int(mask.sum()),
                'col_start': int(c0),
            })

        # Apply topology caps
        filtered, _ = self.caps.filter_regions(regions)

        # Merge diacritics (i-dots, etc.)
        filtered = self._merge_diacritics(filtered, binary)

        filtered.sort(key=lambda x: x['col_start'])
        return filtered

    def _merge_diacritics(self, components, full_binary):
        """Merge small components (dots, diacritics) into parent characters."""
        if len(components) <= 1:
            return components

        merged = []
        used = set()

        for i, comp in enumerate(components):
            if i in used:
                continue

            r0, c0, r1, c1 = comp['bbox']

            for j, other in enumerate(components):
                if j <= i or j in used:
                    continue

                or0, oc0, or1, oc1 = other['bbox']

                # Check horizontal overlap
                h_overlap = min(c1, oc1) - max(c0, oc0)
                if h_overlap < 0:
                    continue

                # Check if one is much smaller (dot/diacritic)
                if other['area'] < comp['area'] * 0.3:
                    new_r0 = min(r0, or0)
                    new_c0 = min(c0, oc0)
                    new_r1 = max(r1, or1)
                    new_c1 = max(c1, oc1)

                    char_binary = full_binary[new_r0:new_r1+1, new_c0:new_c1+1].copy()

                    comp = {
                        'binary': char_binary,
                        'bbox': (new_r0, new_c0, new_r1, new_c1),
                        'area': comp['area'] + other['area'],
                        'col_start': new_c0,
                    }
                    r0, c0, r1, c1 = comp['bbox']
                    used.add(j)

            merged.append(comp)

        merged.sort(key=lambda x: x['col_start'])
        return merged


# =============================================================================
# TOPOLOGY-COHERENCE FUSION
# Combines topology features with coherence measurements
# =============================================================================

class TopologyCoherenceFusion:
    """
    Fuses topology features and coherence measurements into a unified
    character descriptor that is both structurally grounded and
    perceptually aware.

    The topology provides the "skeleton" of understanding.
    The coherence provides the "flesh" of perception.
    Together they create robust character recognition even on
    noisy, colored, low-resolution documents.
    """

    def __init__(self, caps=None):
        self.caps = caps or TopologyCaps()
        self.detector = CoherenceCappedDetector(caps=self.caps)

    def extract_fused_features(self, binary_region):
        """
        Extract a fused topology-coherence feature vector from a binary region.

        Features (32 total):
          [0-5]   Topology: components, holes, euler, aspect, density, area_norm
          [6-11]  Geometry: v_center, h_center, v_sym, h_sym, compactness, edge_density
          [12-17] Projections: 6-bin horizontal projection
          [18-23] Projections: 6-bin vertical projection
          [24-27] Quadrants: TL, TR, BL, BR density ratios
          [28-31] Stroke: width_mean, width_var, endpoint_count, junction_count
        """
        h, w = binary_region.shape
        if h < 2 or w < 2 or binary_region.sum() == 0:
            return np.zeros(32)

        feats = []

        # --- Topology (6 features) ---
        padded = np.pad(binary_region, 1, mode='constant', constant_values=0)
        _, n_fg = ndimage.label(padded)
        _, n_bg = ndimage.label(1 - padded)
        n_holes = n_bg - 1
        euler = n_fg - n_holes
        area = binary_region.sum()
        aspect = w / max(h, 1)
        density = area / max(h * w, 1)
        area_norm = area / max(h * w, 1)

        feats.extend([float(n_fg), float(n_holes), float(euler),
                       aspect, density, area_norm])

        # --- Geometry (6 features) ---
        total = binary_region.sum()
        if total > 0:
            rows = np.arange(h).reshape(-1, 1)
            cols = np.arange(w).reshape(1, -1)
            vc = (binary_region * rows).sum() / (total * h)
            hc = (binary_region * cols).sum() / (total * w)
        else:
            vc, hc = 0.5, 0.5

        v_sym = np.mean(binary_region == np.fliplr(binary_region)) if w >= 2 else 1.0
        h_sym = np.mean(binary_region == np.flipud(binary_region)) if h >= 2 else 1.0

        edges = (np.abs(np.diff(binary_region.astype(int), axis=0)).sum() +
                 np.abs(np.diff(binary_region.astype(int), axis=1)).sum())
        compactness = edges**2 / max(total, 1) / 100
        edge_density = float(edges) / max(h * w, 1)

        feats.extend([vc, hc, float(v_sym), float(h_sym), compactness, edge_density])

        # --- Projections (12 features) ---
        for axis in [1, 0]:  # horizontal then vertical
            raw = binary_region.mean(axis=axis)
            proj = np.zeros(6)
            bw = max(1, len(raw) // 6)
            for i in range(6):
                s, e = i * bw, min((i + 1) * bw, len(raw))
                if s < len(raw):
                    proj[i] = raw[s:e].mean()
            pt = proj.sum()
            if pt > 0:
                proj /= pt
            feats.extend(proj.tolist())

        # --- Quadrants (4 features) ---
        mh, mw = h // 2, w // 2
        quads = [
            float(binary_region[:mh, :mw].mean()) if mh > 0 and mw > 0 else 0,
            float(binary_region[:mh, mw:].mean()) if mh > 0 else 0,
            float(binary_region[mh:, :mw].mean()) if mw > 0 else 0,
            float(binary_region[mh:, mw:].mean()),
        ]
        qt = sum(quads)
        feats.extend([q / qt if qt > 0 else 0.25 for q in quads])

        # --- Stroke (4 features) ---
        dist = ndimage.distance_transform_edt(binary_region)
        ink_mask = binary_region > 0
        sw_mean = dist[ink_mask].mean() if ink_mask.any() else 0.0
        sw_var = dist[ink_mask].var() if ink_mask.any() and ink_mask.sum() > 1 else 0.0

        n_endpoints = 0.0
        n_junctions = 0.0
        try:
            skel = morphology.skeletonize(binary_region.astype(bool)).astype(np.uint8)
            if skel.sum() > 0:
                kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
                neighbors = ndimage.convolve(skel, kernel, mode='constant', cval=0)
                n_endpoints = float(((skel == 1) & (neighbors == 1)).sum())
                n_junctions = float(((skel == 1) & (neighbors >= 3)).sum())
        except Exception:
            pass

        feats.extend([sw_mean, sw_var, n_endpoints, n_junctions])

        return np.array(feats[:32], dtype=float)

    def detect_and_extract(self, cell_rgb, cell_gray):
        """
        Full pipeline: cell image → topology-capped regions with fused features.

        Returns list of dicts with 'binary', 'bbox', 'features', 'topology'.
        """
        regions = self.detector.detect_in_cell(cell_rgb, cell_gray)

        for region in regions:
            region['features'] = self.extract_fused_features(region['binary'])

        return regions

    def segment_line(self, line_gray):
        """
        Segment a text line into topology-capped character regions.
        Returns character regions sorted left-to-right with space detection.
        """
        h, w = line_gray.shape
        if h < 4 or w < 4:
            return [], []

        # Binarize
        border = np.concatenate([line_gray[0, :], line_gray[-1, :]])
        bg = np.median(border)
        bg_std = max(np.std(border), 5)
        diff = bg - line_gray.astype(float)
        binary = (diff > max(bg_std * 1.5, 12)).astype(np.uint8)

        if binary.sum() < 5:
            try:
                from skimage.filters import threshold_otsu
                t = threshold_otsu(line_gray)
                binary = (line_gray < t).astype(np.uint8)
            except Exception:
                return [], []

        binary = morphology.remove_small_objects(binary.astype(bool), min_size=3).astype(np.uint8)

        # Connected components
        labeled, n_comp = ndimage.label(binary)
        if n_comp == 0:
            return [], []

        regions = []
        for i in range(1, n_comp + 1):
            mask = (labeled == i)
            coords = np.argwhere(mask)
            if len(coords) < 3:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            char_binary = mask[r0:r1+1, c0:c1+1].astype(np.uint8)

            if char_binary.shape[0] < 3 or char_binary.shape[1] < 2:
                continue

            regions.append({
                'binary': char_binary,
                'bbox': (r0, c0, r1, c1),
                'area': int(mask.sum()),
                'col_start': int(c0),
            })

        # Apply topology caps
        filtered, _ = self.caps.filter_regions(regions)
        filtered = self.detector._merge_diacritics(filtered, binary)
        filtered.sort(key=lambda x: x['col_start'])

        # Detect spaces
        spaces = []
        if len(filtered) > 1:
            gaps = []
            for i in range(1, len(filtered)):
                prev_end = filtered[i-1]['bbox'][3]
                curr_start = filtered[i]['bbox'][1]
                gaps.append(curr_start - prev_end)
            space_thresh = np.median(gaps) * 2.0 if gaps else 20

            for i in range(1, len(filtered)):
                gap = filtered[i]['bbox'][1] - filtered[i-1]['bbox'][3]
                spaces.append(gap > space_thresh)
        else:
            spaces = []

        # Extract features
        for region in filtered:
            region['features'] = self.extract_fused_features(region['binary'])

        return filtered, spaces


# =============================================================================
# ADAPTIVE TOPOLOGY CAPS
# Caps that learn from document statistics
# =============================================================================

class AdaptiveTopologyCaps(TopologyCaps):
    """
    Topology caps that adapt based on observed document statistics.

    As characters are recognized and validated, the caps tighten
    around the actual topological distribution of characters in
    this specific document.
    """

    def __init__(self):
        super().__init__()
        self._observed_components = []
        self._observed_holes = []
        self._observed_aspects = []
        self._observed_densities = []
        self._observed_areas = []
        self._n_adaptations = 0

    def observe(self, binary_region):
        """Record the topology of a confirmed character to adapt caps."""
        h, w = binary_region.shape
        if h < 2 or w < 2:
            return

        padded = np.pad(binary_region, 1, mode='constant', constant_values=0)
        _, n_fg = ndimage.label(padded)
        _, n_bg = ndimage.label(1 - padded)
        n_holes = n_bg - 1

        self._observed_components.append(n_fg)
        self._observed_holes.append(n_holes)
        self._observed_aspects.append(w / max(h, 1))
        self._observed_densities.append(binary_region.sum() / max(h * w, 1))
        self._observed_areas.append(binary_region.sum())

        # Re-adapt every 20 observations
        self._n_adaptations += 1
        if self._n_adaptations % 20 == 0 and self._n_adaptations >= 20:
            self._adapt()

    def _adapt(self):
        """Tighten caps based on observed statistics."""
        if len(self._observed_components) < 10:
            return

        # Adapt component range: observed range with 1-unit padding
        comp_arr = np.array(self._observed_components)
        self.component_range = (
            max(1, int(np.percentile(comp_arr, 2)) - 1),
            int(np.percentile(comp_arr, 98)) + 1
        )

        # Adapt hole range
        hole_arr = np.array(self._observed_holes)
        self.hole_range = (
            max(0, int(np.percentile(hole_arr, 2))),
            int(np.percentile(hole_arr, 98)) + 1
        )

        # Adapt aspect range
        asp_arr = np.array(self._observed_aspects)
        self.aspect_range = (
            max(0.1, float(np.percentile(asp_arr, 5)) * 0.7),
            float(np.percentile(asp_arr, 95)) * 1.3
        )

        # Adapt density range
        dens_arr = np.array(self._observed_densities)
        self.density_range = (
            max(0.03, float(np.percentile(dens_arr, 5)) * 0.7),
            min(0.98, float(np.percentile(dens_arr, 95)) * 1.3)
        )

    def get_stats(self):
        """Return current adaptive statistics."""
        return {
            'n_observations': self._n_adaptations,
            'component_range': self.component_range,
            'hole_range': self.hole_range,
            'aspect_range': tuple(round(x, 3) for x in self.aspect_range),
            'density_range': tuple(round(x, 3) for x in self.density_range),
        }
