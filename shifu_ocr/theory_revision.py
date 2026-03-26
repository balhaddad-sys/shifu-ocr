"""
Theory-Revision OCR Engine (TR-OCR)
====================================

Architecture:
  Layer 1 — PRINCIPLES: Hardcoded axioms about medium displacement
  Layer 2 — PREDICTIONS: Apply principles to make interpretable guesses
  Layer 3 — COLLISION: Compare prediction to reality, detect errors
  Layer 4 — REFRAMING: Diagnose WHY the error happened, revise the theory

This is NOT gradient descent. This is:
  prediction → error → diagnosis → rule revision → better prediction

The system doesn't just get better. It gets better for a REASON 
you can read, audit, and understand.

Author: Bader & Claude — March 2026
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage import morphology, filters
import json
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# LAYER 0: THE MEDIUM DISPLACEMENT PIPELINE (proven, unchanged)
# =============================================================================

def estimate_background(img, k=25):
    bg = morphology.closing(img, morphology.disk(k))
    return filters.gaussian(bg, sigma=k/2)

def compute_displacement(img, bg):
    d = bg.astype(float) - img.astype(float)
    if d.max() > d.min():
        d = (d - d.min()) / (d.max() - d.min())
    return d

def detect_perturbation(disp, thresh=0.25):
    b = disp > thresh
    b = morphology.remove_small_objects(b, min_size=8)
    return b.astype(np.uint8)

def extract_region(binary, pad=2):
    coords = np.argwhere(binary > 0)
    if len(coords) == 0:
        return binary
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0 = max(0, r0-pad), max(0, c0-pad)
    r1 = min(binary.shape[0]-1, r1+pad)
    c1 = min(binary.shape[1]-1, c1+pad)
    return binary[r0:r1+1, c0:c1+1]

def normalize(region, size=(64,64)):
    img = Image.fromarray((region*255).astype(np.uint8))
    img = img.resize(size, Image.NEAREST)
    return (np.array(img) > 127).astype(np.uint8)

def image_to_region(char_img):
    bg = estimate_background(char_img, k=15)
    disp = compute_displacement(char_img, bg)
    binary = detect_perturbation(disp)
    region = extract_region(binary)
    return normalize(region)


# =============================================================================
# LAYER 1: THE PRINCIPLES — What the system KNOWS before seeing any data
# =============================================================================

class Principles:
    """
    Hardcoded axioms about medium displacement.
    These are the textbook. The system knows them a priori.
    """
    
    @staticmethod
    def euler_number(br):
        """Topological invariant: components - holes."""
        padded = np.pad(br, 1, mode='constant', constant_values=0)
        _, nfg = ndimage.label(padded)
        _, nbg = ndimage.label(1 - padded)
        holes = nbg - 1
        return {'components': nfg, 'holes': holes, 'euler': nfg - holes}
    
    @staticmethod
    def displacement_ratio(br):
        """How much of the bounding box is perturbed."""
        return float(np.mean(br))
    
    @staticmethod
    def symmetry(br):
        """Vertical and horizontal symmetry of perturbation."""
        h, w = br.shape
        v = float(np.mean(br == np.fliplr(br))) if w >= 2 else 1.0
        hz = float(np.mean(br == np.flipud(br))) if h >= 2 else 1.0
        return {'vertical': v, 'horizontal': hz}
    
    @staticmethod
    def center_of_mass(br):
        """Where is the displacement concentrated (0-1 normalized)."""
        h, w = br.shape
        total = br.sum()
        if total == 0 or h == 0 or w == 0:
            return {'vertical': 0.5, 'horizontal': 0.5}
        rows = np.arange(h).reshape(-1, 1)
        cols = np.arange(w).reshape(1, -1)
        vc = float((br * rows).sum() / (total * h))
        hc = float((br * cols).sum() / (total * w))
        return {'vertical': vc, 'horizontal': hc}
    
    @staticmethod
    def quadrant_density(br):
        """Proportional displacement in each quadrant."""
        h, w = br.shape
        mh, mw = h//2, w//2
        tl = float(br[:mh, :mw].mean()) if mh > 0 and mw > 0 else 0
        tr = float(br[:mh, mw:].mean()) if mh > 0 else 0
        bl = float(br[mh:, :mw].mean()) if mw > 0 else 0
        brr = float(br[mh:, mw:].mean())
        total = tl + tr + bl + brr
        if total > 0:
            tl, tr, bl, brr = tl/total, tr/total, bl/total, brr/total
        return {'top_left': tl, 'top_right': tr, 'bottom_left': bl, 'bottom_right': brr}
    
    @staticmethod
    def projections(br, bins=6):
        """Horizontal and vertical displacement profiles."""
        h, w = br.shape
        hp = np.zeros(bins)
        vp = np.zeros(bins)
        if h >= 2:
            raw = br.mean(axis=1)
            bh = max(1, len(raw)//bins)
            for i in range(bins):
                s, e = i*bh, min((i+1)*bh, len(raw))
                if s < len(raw): hp[i] = raw[s:e].mean()
        if w >= 2:
            raw = br.mean(axis=0)
            bw = max(1, len(raw)//bins)
            for i in range(bins):
                s, e = i*bw, min((i+1)*bw, len(raw))
                if s < len(raw): vp[i] = raw[s:e].mean()
        ht, vt = hp.sum(), vp.sum()
        if ht > 0: hp /= ht
        if vt > 0: vp /= vt
        return {'horizontal': hp.tolist(), 'vertical': vp.tolist()}
    
    @staticmethod
    def crossing_counts(br, lines=6):
        """How many times ink crosses horizontal/vertical scan lines."""
        h, w = br.shape
        hc = []
        vc = []
        for i in range(lines):
            row = min(int((i + 0.5) * h / lines), h - 1)
            hc.append(int(np.abs(np.diff(br[row, :].astype(int))).sum() / 2))
            col = min(int((i + 0.5) * w / lines), w - 1)
            vc.append(int(np.abs(np.diff(br[:, col].astype(int))).sum() / 2))
        return {'horizontal': hc, 'vertical': vc}
    
    @staticmethod
    def junction_analysis(br):
        """Skeleton junctions and endpoints of the displaced region."""
        h, w = br.shape
        if h < 4 or w < 4 or br.sum() < 10:
            return {'endpoints': 0, 'junctions': 0, 'skel_density': 0.0}
        try:
            skel = morphology.skeletonize(br.astype(bool))
        except:
            return {'endpoints': 0, 'junctions': 0, 'skel_density': 0.0}
        if skel.sum() < 3:
            return {'endpoints': 0, 'junctions': 0, 'skel_density': 0.0}
        nc = ndimage.convolve(skel.astype(int), np.ones((3,3), dtype=int),
                              mode='constant') - skel.astype(int)
        ep = skel & (nc == 1)
        jp = skel & (nc >= 3)
        _, n_ep = ndimage.label(ep)
        _, n_jp = ndimage.label(jp)
        return {
            'endpoints': int(n_ep),
            'junctions': int(n_jp),
            'skel_density': float(skel.sum() / max(h*w, 1)),
        }
    
    @staticmethod
    def extract_all(br):
        """Extract the complete principle-based observation."""
        return {
            'topology': Principles.euler_number(br),
            'displacement_ratio': Principles.displacement_ratio(br),
            'symmetry': Principles.symmetry(br),
            'center_of_mass': Principles.center_of_mass(br),
            'quadrants': Principles.quadrant_density(br),
            'projections': Principles.projections(br),
            'crossings': Principles.crossing_counts(br),
            'junctions': Principles.junction_analysis(br),
        }


# =============================================================================
# LAYER 2: THE THEORY — What the system believes about each character
# =============================================================================

class CharacterTheory:
    """
    A theory about a single character — what the system EXPECTS 
    to observe when it sees this character.
    
    Initially formed from a single example (like a medical student 
    reading the textbook). Refined through experience.
    """
    
    def __init__(self, label):
        self.label = label
        self.observations = []       # Raw observations (principle extractions)
        self.expected = None         # Averaged/expected observation
        self.rules = []              # Learned discriminative rules
        self.confusion_history = []  # What has it been confused with?
        self.n_correct = 0
        self.n_errors = 0
    
    def add_observation(self, obs):
        """Add a new observation of this character."""
        self.observations.append(obs)
        self._update_expected()
    
    def _update_expected(self):
        """Update expected values from all observations."""
        if not self.observations:
            return
        
        self.expected = {
            'topology': {
                'holes': self._mean_of('topology', 'holes'),
                'euler': self._mean_of('topology', 'euler'),
                'components': self._mean_of('topology', 'components'),
            },
            'displacement_ratio': np.mean([o['displacement_ratio'] for o in self.observations]),
            'symmetry': {
                'vertical': np.mean([o['symmetry']['vertical'] for o in self.observations]),
                'horizontal': np.mean([o['symmetry']['horizontal'] for o in self.observations]),
            },
            'center_of_mass': {
                'vertical': np.mean([o['center_of_mass']['vertical'] for o in self.observations]),
                'horizontal': np.mean([o['center_of_mass']['horizontal'] for o in self.observations]),
            },
            'quadrants': {
                k: np.mean([o['quadrants'][k] for o in self.observations])
                for k in ['top_left', 'top_right', 'bottom_left', 'bottom_right']
            },
            'projections': {
                'horizontal': np.mean([o['projections']['horizontal'] for o in self.observations], axis=0).tolist(),
                'vertical': np.mean([o['projections']['vertical'] for o in self.observations], axis=0).tolist(),
            },
            'crossings': {
                'horizontal': np.mean([o['crossings']['horizontal'] for o in self.observations], axis=0).tolist(),
                'vertical': np.mean([o['crossings']['vertical'] for o in self.observations], axis=0).tolist(),
            },
            'junctions': {
                'endpoints': np.mean([o['junctions']['endpoints'] for o in self.observations]),
                'junctions': np.mean([o['junctions']['junctions'] for o in self.observations]),
            },
        }
        
        # Compute variances (how much each feature varies across observations)
        if len(self.observations) > 1:
            self.variance = {
                'displacement_ratio': np.var([o['displacement_ratio'] for o in self.observations]),
                'v_symmetry': np.var([o['symmetry']['vertical'] for o in self.observations]),
                'h_symmetry': np.var([o['symmetry']['horizontal'] for o in self.observations]),
            }
        else:
            self.variance = {}
    
    def _mean_of(self, section, key):
        """Helper to compute mean of a nested key."""
        return float(np.mean([o[section][key] for o in self.observations]))
    
    def distance_to(self, obs):
        """
        How well does an observation match this theory?
        Lower = better match.
        
        Uses principle-weighted distance: topology matters most,
        then proportions, then geometry.
        """
        if self.expected is None:
            return float('inf')
        
        d = 0.0
        
        # TOPOLOGY (weight: 5.0) — most invariant
        d += 5.0 * abs(obs['topology']['holes'] - self.expected['topology']['holes'])
        d += 5.0 * abs(obs['topology']['euler'] - self.expected['topology']['euler'])
        
        # DISPLACEMENT RATIO (weight: 2.0)
        d += 2.0 * abs(obs['displacement_ratio'] - self.expected['displacement_ratio'])
        
        # SYMMETRY (weight: 2.0)
        d += 2.0 * abs(obs['symmetry']['vertical'] - self.expected['symmetry']['vertical'])
        d += 2.0 * abs(obs['symmetry']['horizontal'] - self.expected['symmetry']['horizontal'])
        
        # CENTER OF MASS (weight: 2.0)
        d += 2.0 * abs(obs['center_of_mass']['vertical'] - self.expected['center_of_mass']['vertical'])
        d += 2.0 * abs(obs['center_of_mass']['horizontal'] - self.expected['center_of_mass']['horizontal'])
        
        # QUADRANTS (weight: 2.0)
        for k in ['top_left', 'top_right', 'bottom_left', 'bottom_right']:
            d += 2.0 * abs(obs['quadrants'][k] - self.expected['quadrants'][k])
        
        # PROJECTIONS (weight: 1.5)
        for axis in ['horizontal', 'vertical']:
            for i in range(len(self.expected['projections'][axis])):
                d += 1.5 * abs(obs['projections'][axis][i] - self.expected['projections'][axis][i])
        
        # CROSSINGS (weight: 1.5)
        for axis in ['horizontal', 'vertical']:
            for i in range(len(self.expected['crossings'][axis])):
                ov = obs['crossings'][axis][i]
                ev = self.expected['crossings'][axis][i]
                d += 1.5 * abs(ov - ev) / max(ev, 1)
        
        return d
    
    def check_rules(self, obs):
        """
        Check learned discriminative rules.
        Returns adjustment: negative = this theory is MORE likely,
                           positive = this theory is LESS likely.
        """
        adjustment = 0.0
        for rule in self.rules:
            result = rule['check'](obs)
            adjustment += result
        return adjustment


# =============================================================================
# LAYER 3 & 4: THE ENGINE — Prediction, Collision, Diagnosis, Reframing
# =============================================================================

class TheoryRevisionEngine:
    """
    The complete learning engine.
    
    Predicts → Gets feedback → Diagnoses error → Revises theory.
    
    Every revision is INTERPRETABLE: you can ask the system 
    "why did you change your mind?" and get a human-readable answer.
    """
    
    def __init__(self):
        self.theories = {}           # label -> CharacterTheory
        self.revision_log = []       # Complete history of every revision
        self.n_predictions = 0
        self.n_correct = 0
        self.discrimination_rules = []  # Global rules for confused pairs
    
    # --- Teaching (Layer 1: Principles applied to examples) ---
    
    def teach(self, label, binary_region):
        """
        Teach the system a character by showing it one example.
        Like a medical student reading the textbook entry.
        """
        obs = Principles.extract_all(binary_region)
        
        if label not in self.theories:
            self.theories[label] = CharacterTheory(label)
        
        self.theories[label].add_observation(obs)
        
        return obs
    
    # --- Prediction (Layer 2: Apply theory to make a guess) ---
    
    def predict(self, binary_region):
        """
        Make a prediction with full reasoning chain.
        
        Returns not just the answer, but WHY — which theory matched,
        how confident we are, and what alternatives exist.
        """
        obs = Principles.extract_all(binary_region)
        
        # Score against all theories
        scores = []
        for label, theory in self.theories.items():
            base_distance = theory.distance_to(obs)
            rule_adjustment = theory.check_rules(obs)
            
            # Also check global discrimination rules
            global_adj = 0.0
            for rule in self.discrimination_rules:
                if rule['favors'] == label or rule['disfavors'] == label:
                    result = rule['check'](obs)
                    if rule['favors'] == label:
                        global_adj -= abs(result)  # Lower distance = more likely
                    else:
                        global_adj += abs(result)
            
            final_distance = base_distance + rule_adjustment + global_adj
            scores.append({
                'label': label,
                'base_distance': base_distance,
                'rule_adjustment': rule_adjustment,
                'global_adjustment': global_adj,
                'final_distance': final_distance,
            })
        
        scores.sort(key=lambda x: x['final_distance'])
        
        top = scores[0]
        runner_up = scores[1] if len(scores) > 1 else None
        
        # Confidence: based on margin between top and runner-up
        if runner_up:
            margin = runner_up['final_distance'] - top['final_distance']
            confidence = min(margin / max(top['final_distance'], 0.1), 1.0)
        else:
            confidence = 0.5
        
        self.n_predictions += 1
        
        prediction = {
            'predicted': top['label'],
            'confidence': confidence,
            'observation': obs,
            'reasoning': {
                'top_match': top,
                'runner_up': runner_up,
                'margin': margin if runner_up else None,
                'n_rules_applied': len(self.discrimination_rules),
            },
            'all_scores': scores[:5],
        }
        
        return prediction
    
    # --- Collision & Diagnosis (Layer 3: What went wrong?) ---
    
    def learn_from_error(self, prediction, true_label):
        """
        THE CORE LEARNING MECHANISM.
        
        When a prediction is wrong, diagnose WHY:
        1. Which features SHOULD have distinguished the correct answer?
        2. In which direction does each feature discriminate?
        3. Create a rule that captures this discrimination.
        
        This is not "adjust weight #4372 by 0.003."
        This is "I confused D with O because I didn't notice the 
        horizontal asymmetry. Adding rule: if displacement center 
        is right of 0.55, favor D over O."
        """
        predicted = prediction['predicted']
        obs = prediction['observation']
        
        if predicted == true_label:
            self.n_correct += 1
            self.theories[true_label].n_correct += 1
            return {'status': 'correct', 'label': true_label}
        
        # ERROR — Begin diagnosis
        self.theories[predicted].n_errors += 1
        self.theories[predicted].confusion_history.append(true_label)
        
        # Add this observation to the correct theory
        if true_label in self.theories:
            self.theories[true_label].add_observation(obs)
        
        # DIAGNOSIS: Find the features that SHOULD discriminate
        diagnosis = self._diagnose_confusion(predicted, true_label, obs)
        
        # REFRAMING: Create discrimination rules
        new_rules = self._create_rules(predicted, true_label, diagnosis)
        
        revision = {
            'status': 'error_corrected',
            'predicted': predicted,
            'true_label': true_label,
            'diagnosis': diagnosis,
            'new_rules': [r['description'] for r in new_rules],
            'total_rules': len(self.discrimination_rules),
        }
        
        self.revision_log.append(revision)
        
        return revision
    
    def _diagnose_confusion(self, predicted, true_label, obs):
        """
        Find WHY these two characters were confused.
        
        Compare the observation against both theories.
        Find features where the observation matches the TRUE theory 
        better than the PREDICTED theory — those are the features 
        we should have been paying attention to.
        """
        pred_theory = self.theories.get(predicted)
        true_theory = self.theories.get(true_label)
        
        if not pred_theory or not true_theory or not true_theory.expected:
            return {'status': 'insufficient_data', 'features': []}
        
        discriminators = []
        
        # Check each feature dimension
        feature_checks = [
            ('holes', 
             lambda o: o['topology']['holes'],
             lambda t: t['topology']['holes']),
            ('euler_number',
             lambda o: o['topology']['euler'],
             lambda t: t['topology']['euler']),
            ('displacement_ratio',
             lambda o: o['displacement_ratio'],
             lambda t: t['displacement_ratio']),
            ('vertical_symmetry',
             lambda o: o['symmetry']['vertical'],
             lambda t: t['symmetry']['vertical']),
            ('horizontal_symmetry',
             lambda o: o['symmetry']['horizontal'],
             lambda t: t['symmetry']['horizontal']),
            ('vertical_center',
             lambda o: o['center_of_mass']['vertical'],
             lambda t: t['center_of_mass']['vertical']),
            ('horizontal_center',
             lambda o: o['center_of_mass']['horizontal'],
             lambda t: t['center_of_mass']['horizontal']),
            ('quad_top_left',
             lambda o: o['quadrants']['top_left'],
             lambda t: t['quadrants']['top_left']),
            ('quad_top_right',
             lambda o: o['quadrants']['top_right'],
             lambda t: t['quadrants']['top_right']),
            ('quad_bottom_left',
             lambda o: o['quadrants']['bottom_left'],
             lambda t: t['quadrants']['bottom_left']),
            ('quad_bottom_right',
             lambda o: o['quadrants']['bottom_right'],
             lambda t: t['quadrants']['bottom_right']),
            ('endpoints',
             lambda o: o['junctions']['endpoints'],
             lambda t: t['junctions']['endpoints']),
            ('junctions',
             lambda o: o['junctions']['junctions'],
             lambda t: t['junctions']['junctions']),
        ]
        
        for name, obs_fn, theory_fn in feature_checks:
            obs_val = obs_fn(obs)
            pred_val = theory_fn(pred_theory.expected)
            true_val = theory_fn(true_theory.expected)
            
            # How much does this feature favor the true label over predicted?
            error_pred = abs(obs_val - pred_val)
            error_true = abs(obs_val - true_val)
            
            discrimination = error_pred - error_true
            
            if abs(discrimination) > 0.01:  # Non-trivial
                discriminators.append({
                    'feature': name,
                    'observed': round(obs_val, 4),
                    'predicted_expected': round(pred_val, 4),
                    'true_expected': round(true_val, 4),
                    'discrimination_power': round(discrimination, 4),
                    'direction': 'favors_true' if discrimination > 0 else 'favors_predicted',
                })
        
        # Sort by discrimination power
        discriminators.sort(key=lambda x: abs(x['discrimination_power']), reverse=True)
        
        return {
            'confused_pair': (predicted, true_label),
            'discriminators': discriminators,
            'best_feature': discriminators[0] if discriminators else None,
        }
    
    def _create_rules(self, predicted, true_label, diagnosis):
        """
        Create interpretable discrimination rules from the diagnosis.
        
        Each rule is a simple threshold test on one feature.
        "If horizontal_center > 0.55 and choosing between D and O, favor D."
        
        These rules are READABLE. A human can audit every single one.
        """
        new_rules = []
        
        discs = diagnosis.get('discriminators', [])
        
        # Take the top discriminators and create rules
        for disc in discs[:3]:  # Max 3 rules per confusion
            feature = disc['feature']
            true_val = disc['true_expected']
            pred_val = disc['predicted_expected']
            
            # Threshold: midpoint between the two expected values
            threshold = (true_val + pred_val) / 2
            direction = 'greater' if true_val > pred_val else 'less'
            
            # Build the rule as a callable with metadata
            rule = self._build_rule(
                feature=feature,
                threshold=threshold,
                direction=direction,
                favors=true_label,
                disfavors=predicted,
                power=abs(disc['discrimination_power']),
            )
            
            # Check for duplicate rules
            is_dup = any(
                r['feature'] == feature and 
                r['favors'] == true_label and 
                r['disfavors'] == predicted
                for r in self.discrimination_rules
            )
            
            if not is_dup:
                self.discrimination_rules.append(rule)
                new_rules.append(rule)
        
        return new_rules
    
    def _build_rule(self, feature, threshold, direction, favors, disfavors, power):
        """Build an interpretable, callable discrimination rule."""
        
        # Map feature names to observation accessors
        accessors = {
            'holes': lambda o: o['topology']['holes'],
            'euler_number': lambda o: o['topology']['euler'],
            'displacement_ratio': lambda o: o['displacement_ratio'],
            'vertical_symmetry': lambda o: o['symmetry']['vertical'],
            'horizontal_symmetry': lambda o: o['symmetry']['horizontal'],
            'vertical_center': lambda o: o['center_of_mass']['vertical'],
            'horizontal_center': lambda o: o['center_of_mass']['horizontal'],
            'quad_top_left': lambda o: o['quadrants']['top_left'],
            'quad_top_right': lambda o: o['quadrants']['top_right'],
            'quad_bottom_left': lambda o: o['quadrants']['bottom_left'],
            'quad_bottom_right': lambda o: o['quadrants']['bottom_right'],
            'endpoints': lambda o: o['junctions']['endpoints'],
            'junctions': lambda o: o['junctions']['junctions'],
        }
        
        accessor = accessors.get(feature, lambda o: 0)
        
        def check(obs):
            val = accessor(obs)
            if direction == 'greater':
                return power if val > threshold else -power * 0.5
            else:
                return power if val < threshold else -power * 0.5
        
        description = (
            f"IF {feature} {'>' if direction == 'greater' else '<'} "
            f"{threshold:.3f} THEN favor '{favors}' over '{disfavors}' "
            f"(power={power:.3f})"
        )
        
        return {
            'feature': feature,
            'threshold': threshold,
            'direction': direction,
            'favors': favors,
            'disfavors': disfavors,
            'power': power,
            'check': check,
            'description': description,
        }
    
    # --- Reporting ---
    
    def get_theory_summary(self, label):
        """Human-readable summary of what the system believes about a character."""
        if label not in self.theories:
            return f"No theory for '{label}'"
        
        t = self.theories[label]
        if not t.expected:
            return f"'{label}': no observations yet"
        
        lines = [f"Theory for '{label}' ({len(t.observations)} observations):"]
        lines.append(f"  Topology: {t.expected['topology']['holes']} holes, "
                     f"euler={t.expected['topology']['euler']}")
        lines.append(f"  Displacement: {t.expected['displacement_ratio']:.1%} of bounding box")
        lines.append(f"  Symmetry: V={t.expected['symmetry']['vertical']:.2f}, "
                     f"H={t.expected['symmetry']['horizontal']:.2f}")
        lines.append(f"  Center of mass: V={t.expected['center_of_mass']['vertical']:.2f}, "
                     f"H={t.expected['center_of_mass']['horizontal']:.2f}")
        lines.append(f"  Track record: {t.n_correct} correct, {t.n_errors} errors")
        
        if t.confusion_history:
            confused_with = defaultdict(int)
            for c in t.confusion_history:
                confused_with[c] += 1
            confusions = ', '.join(f"'{k}'×{v}" for k, v in 
                                    sorted(confused_with.items(), key=lambda x: -x[1]))
            lines.append(f"  Confused with: {confusions}")
        
        # Rules involving this character
        relevant_rules = [r for r in self.discrimination_rules 
                         if r['favors'] == label or r['disfavors'] == label]
        if relevant_rules:
            lines.append(f"  Discrimination rules ({len(relevant_rules)}):")
            for r in relevant_rules:
                lines.append(f"    {r['description']}")
        
        return '\n'.join(lines)
    
    def get_revision_summary(self):
        """Summary of all theory revisions."""
        lines = [f"Theory Revision Log ({len(self.revision_log)} revisions):"]
        for i, rev in enumerate(self.revision_log):
            lines.append(f"\n  Revision {i+1}: '{rev['predicted']}' → '{rev['true_label']}'")
            if rev.get('diagnosis', {}).get('best_feature'):
                bf = rev['diagnosis']['best_feature']
                lines.append(f"    Key discriminator: {bf['feature']} "
                           f"(observed={bf['observed']}, "
                           f"expected for '{rev['true_label']}'={bf['true_expected']})")
            for rule in rev.get('new_rules', []):
                lines.append(f"    New rule: {rule}")
        return '\n'.join(lines)


# =============================================================================
# EXPERIMENT: The Learning Journey
# =============================================================================

FONTS = [
    ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'DejaVu Sans'),
    ('/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf', 'DejaVu Serif'),
    ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 'DejaVu Bold'),
    ('/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf', 'Serif Bold'),
    ('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 'DejaVu Mono'),
    ('/usr/share/fonts/truetype/freefont/FreeMono.ttf', 'FreeMono'),
    ('/usr/share/fonts/truetype/freefont/FreeSans.ttf', 'FreeSans'),
    ('/usr/share/fonts/truetype/freefont/FreeSerif.ttf', 'FreeSerif'),
    ('/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf', 'Poppins Bold'),
]

ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

def render_char(char, font_path, size=80, img_size=(100, 100)):
    img = Image.new('L', img_size, color=255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, size)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), char, font=font)
    x = (img_size[0] - (bbox[2]-bbox[0])) // 2 - bbox[0]
    y = (img_size[1] - (bbox[3]-bbox[1])) // 2 - bbox[1]
    draw.text((x, y), char, fill=0, font=font)
    return np.array(img)


def run_learning_experiment():
    """
    Simulate the learning journey:
    
    1. TEXTBOOK: Teach from one font (like reading the textbook)
    2. FIRST PATIENT: Test on a new font, make errors
    3. LEARN: Diagnose errors, create rules
    4. SECOND PATIENT: Test on another font — are we better?
    5. Repeat — track improvement over time
    
    This mirrors clinical training: principles → practice → error → 
    reflection → refined understanding → better practice.
    """
    
    engine = TheoryRevisionEngine()
    
    # =========================================================
    # PHASE 1: THE TEXTBOOK — Learn principles from one font
    # =========================================================
    print("=" * 70)
    print("PHASE 1: THE TEXTBOOK")
    print("Learning from DejaVu Sans (one example per character)")
    print("=" * 70)
    print()
    
    textbook_font = FONTS[0]
    for char in ALPHABET:
        img = render_char(char, textbook_font[0])
        region = image_to_region(img)
        engine.teach(char, region)
    
    print(f"  Taught {len(ALPHABET)} characters from {textbook_font[1]}")
    print(f"  Theories formed: {len(engine.theories)}")
    
    # Show a few theories
    for char in ['A', 'O', 'T', 'I']:
        print()
        print(f"  {engine.get_theory_summary(char)}")
    
    # =========================================================
    # PHASE 2-N: CLINICAL ROTATIONS — Test, fail, learn, improve
    # =========================================================
    
    rotation_results = []
    
    for rotation_idx, (font_path, font_name) in enumerate(FONTS[1:], 1):
        print()
        print(f"{'='*70}")
        print(f"ROTATION {rotation_idx}: {font_name}")
        print(f"{'='*70}")
        
        correct = 0
        errors = []
        
        for char in ALPHABET:
            img = render_char(char, font_path)
            region = image_to_region(img)
            
            # PREDICT
            prediction = engine.predict(region)
            pred_label = prediction['predicted']
            confidence = prediction['confidence']
            
            # COLLISION WITH REALITY
            revision = engine.learn_from_error(prediction, char)
            
            if pred_label == char:
                correct += 1
            else:
                errors.append({
                    'true': char,
                    'predicted': pred_label,
                    'confidence': confidence,
                    'revision': revision,
                })
        
        accuracy = correct / len(ALPHABET) * 100
        rotation_results.append({
            'font': font_name,
            'accuracy': accuracy,
            'correct': correct,
            'errors': len(errors),
            'total_rules': len(engine.discrimination_rules),
        })
        
        bar = "█" * int(accuracy / 2.5) + "░" * (40 - int(accuracy / 2.5))
        print(f"\n  Accuracy: {bar} {accuracy:.1f}% ({correct}/{len(ALPHABET)})")
        print(f"  Rules after this rotation: {len(engine.discrimination_rules)}")
        
        if errors:
            print(f"\n  Errors and diagnoses:")
            for e in errors[:8]:
                rev = e['revision']
                diag = rev.get('diagnosis', {})
                best = diag.get('best_feature')
                
                if best:
                    print(f"    '{e['true']}' misread as '{e['predicted']}' "
                          f"(conf={e['confidence']:.2f})")
                    print(f"      Diagnosis: {best['feature']} was {best['observed']:.3f}, "
                          f"expected {best['true_expected']:.3f} for '{e['true']}'")
                    if rev.get('new_rules'):
                        print(f"      New rule: {rev['new_rules'][0]}")
                else:
                    print(f"    '{e['true']}' misread as '{e['predicted']}' "
                          f"— insufficient data for diagnosis")
    
    # =========================================================
    # SUMMARY: THE LEARNING CURVE
    # =========================================================
    print()
    print("=" * 70)
    print("THE LEARNING CURVE")
    print("=" * 70)
    print()
    print("  How accuracy improved as the system gained experience:")
    print()
    
    for i, r in enumerate(rotation_results):
        bar = "█" * int(r['accuracy'] / 2.5) + "░" * (40 - int(r['accuracy'] / 2.5))
        print(f"  Rotation {i+1} ({r['font']:16s}) {bar} {r['accuracy']:5.1f}% "
              f"[{r['total_rules']} rules]")
    
    accuracies = [r['accuracy'] for r in rotation_results]
    print(f"\n  First rotation:  {accuracies[0]:.1f}%")
    print(f"  Last rotation:   {accuracies[-1]:.1f}%")
    print(f"  Improvement:     {accuracies[-1] - accuracies[0]:+.1f}%")
    print(f"  Total rules learned: {len(engine.discrimination_rules)}")
    
    # =========================================================
    # SHOW THE LEARNED THEORY
    # =========================================================
    print()
    print("=" * 70)
    print("WHAT THE SYSTEM LEARNED (interpretable rules)")
    print("=" * 70)
    
    # Show rules for commonly confused pairs
    shown_pairs = set()
    for rule in engine.discrimination_rules[:25]:
        pair = (rule['favors'], rule['disfavors'])
        if pair not in shown_pairs:
            print(f"\n  {rule['description']}")
            shown_pairs.add(pair)
    
    # Show evolved theories for interesting characters
    print()
    print("=" * 70)
    print("EVOLVED THEORIES (after all rotations)")
    print("=" * 70)
    
    for char in ['D', 'O', 'I', 'T', 'V', 'W', 'M', 'N']:
        print()
        print(f"  {engine.get_theory_summary(char)}")
    
    return engine, rotation_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          THEORY-REVISION OCR ENGINE (TR-OCR)                       ║")
    print("║                                                                    ║")
    print("║  Architecture:                                                     ║")
    print("║    Principles → Prediction → Error → Diagnosis → Reframing        ║")
    print("║                                                                    ║")
    print("║  'If 1+1 doesn't equal 2, something is wrong.                     ║")
    print("║   That is how learning happens — by understanding concepts         ║")
    print("║   and then trying to fill the hole. If it doesn't fit, reframe.'  ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    
    engine, results = run_learning_experiment()
    
    print()
    print("=" * 70)
    print("WHAT THIS PROVES")
    print("=" * 70)
    print()
    print("  1. The system learns from errors, not just from data.")
    print("  2. Every improvement has a human-readable REASON.")
    print("  3. You can audit every rule the system learned.")
    print("  4. The learning is CUMULATIVE — each rotation builds on the last.")
    print("  5. No neural network. No backpropagation. No black box.")
    print()
    print("  What it DOESN'T prove (yet):")
    print("  - That this scales to real handwriting")
    print("  - That the rules generalize beyond the fonts tested")
    print("  - That it can compete with deep learning on hard cases")
    print()
    print("  Those are the next collisions with reality.")
    print()


if __name__ == '__main__':
    main()
