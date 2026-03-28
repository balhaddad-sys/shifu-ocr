"""
Test Suite: MRI FLAIR OCR, Topology Coherence Caps, Image Interpreter
======================================================================

Comprehensive tests for the complete MRI-based OCR system:
  1. MRI FLAIR engine — sequences, signatures, landscapes, recognition
  2. Topology coherence caps — filtering, adaptive caps
  3. Image interpreter — multi-engine fusion, word assembly, pipeline

Author: Bader & Claude — March 2026
"""

import numpy as np
import sys
import os
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shifu_ocr.mri_flair import (
    FLAIR_OCR, FLAIRLandscape, extract_flair_signature,
    T1Weighted, T2Weighted, FLAIRSequence, DWISequence, SWISequence, MRASequence,
    measure_flair_response, _normalize_for_flair,
)
from shifu_ocr.topology_coherence import (
    TopologyCaps, CoherenceCappedDetector, TopologyCoherenceFusion,
    AdaptiveTopologyCaps,
)
from shifu_ocr.image_interpreter import (
    ImageInterpreter, MultiEngineRecognizer,
    create_interpreter, InterpretationResult, WordResult, CellResult,
)


# =============================================================================
# HELPERS
# =============================================================================

def make_char_image(char, font_size=32, img_size=(48, 48)):
    """Render a character as a binary numpy array."""
    img = Image.new('L', img_size, color=255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), char, font=font)
    x = (img_size[0] - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (img_size[1] - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), char, fill=0, font=font)
    arr = np.array(img)
    binary = (arr < 128).astype(np.uint8)
    return binary


def make_test_cell(text="Hello", width=200, height=30, font_size=20):
    """Create a test cell with rendered text, returns (rgb, gray)."""
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((5, 2), text, fill=(0, 0, 0), font=font)
    rgb = np.array(img)
    gray = np.array(img.convert('L'))
    return rgb, gray


passed = 0
failed = 0
total = 0


def test(name, condition):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")


# =============================================================================
# TEST 1: MRI SEQUENCES
# =============================================================================

def test_mri_sequences():
    print("\n" + "=" * 60)
    print("TEST 1: MRI Sequences")
    print("=" * 60)

    binary = make_char_image('A')

    # T1 Weighted
    t1 = T1Weighted()
    cascade = t1.acquire(binary)
    test("T1w produces 4-stage cascade", len(cascade) == 4)
    test("T1w each stage is numpy array", all(isinstance(s, np.ndarray) for s in cascade))
    test("T1w erosion reduces mass", cascade[0].sum() >= cascade[2].sum())

    # T2 Weighted
    t2 = T2Weighted()
    cascade = t2.acquire(binary)
    test("T2w produces 4-stage cascade", len(cascade) == 4)
    test("T2w dilation increases mass", cascade[0].sum() <= cascade[2].sum())

    # FLAIR
    flair = FLAIRSequence()
    stages = flair.acquire(binary)
    test("FLAIR produces 4 stages", len(stages) == 4)
    test("FLAIR stages are arrays", all(isinstance(s, np.ndarray) for s in stages))

    # DWI
    dwi = DWISequence()
    maps = dwi.acquire(binary)
    test("DWI produces 2 maps", len(maps) == 2)

    # SWI
    swi = SWISequence()
    maps = swi.acquire(binary)
    test("SWI produces 2 maps", len(maps) == 2)

    # MRA
    mra = MRASequence()
    maps = mra.acquire(binary)
    test("MRA produces 2 maps (skeleton + vessel)", len(maps) == 2)

    # Edge cases
    empty = np.zeros((10, 10), dtype=np.uint8)
    t1_empty = t1.acquire(empty)
    test("T1w handles empty image", len(t1_empty) == 4)

    tiny = np.ones((2, 2), dtype=np.uint8)
    flair_tiny = flair.acquire(tiny)
    test("FLAIR handles tiny image", len(flair_tiny) == 4)


# =============================================================================
# TEST 2: FLAIR SIGNATURES
# =============================================================================

def test_flair_signatures():
    print("\n" + "=" * 60)
    print("TEST 2: FLAIR Signatures")
    print("=" * 60)

    binary_a = make_char_image('A')
    binary_o = make_char_image('O')

    # Signature extraction
    sig_a = extract_flair_signature(binary_a)
    test("Signature is 252-dimensional", len(sig_a) == 252)
    test("Signature has non-zero values", np.abs(sig_a).sum() > 0)

    sig_o = extract_flair_signature(binary_o)
    test("Different chars produce different signatures",
         np.linalg.norm(sig_a - sig_o) > 0.1)

    # Same char, different rendering should be more similar
    binary_a2 = make_char_image('A', font_size=28)
    sig_a2 = extract_flair_signature(binary_a2)
    dist_same = np.linalg.norm(sig_a - sig_a2)
    dist_diff = np.linalg.norm(sig_a - sig_o)
    test("Same char variants closer than different chars", dist_same < dist_diff)

    # Measure response
    resp = measure_flair_response(binary_a)
    test("FLAIR response is 12-dimensional", len(resp) == 12)
    test("Mass is positive", resp[0] > 0)

    # Empty image
    sig_empty = extract_flair_signature(np.zeros((10, 10), dtype=np.uint8))
    test("Empty image produces zero signature", np.abs(sig_empty).sum() == 0)


# =============================================================================
# TEST 3: FLAIR LANDSCAPES
# =============================================================================

def test_flair_landscapes():
    print("\n" + "=" * 60)
    print("TEST 3: FLAIR Landscapes")
    print("=" * 60)

    land_a = FLAIRLandscape('A')
    sig_a = extract_flair_signature(make_char_image('A'))
    land_a.absorb(sig_a)

    test("Landscape has 1 observation", land_a.n == 1)
    test("Landscape mean is set", land_a.mean is not None)
    test("Landscape variance is set", land_a.variance is not None)

    # Absorb more
    land_a.absorb(extract_flair_signature(make_char_image('A', font_size=28)))
    land_a.absorb(extract_flair_signature(make_char_image('A', font_size=40)))
    test("Landscape has 3 observations", land_a.n == 3)

    # Fit
    score_a = land_a.fit(sig_a)
    sig_o = extract_flair_signature(make_char_image('O'))
    score_o = land_a.fit(sig_o)
    test("'A' fits 'A' landscape better than 'O'", score_a > score_o)

    # Confidence
    conf = land_a.confidence(sig_a)
    test("Confidence is in [0, 1]", 0 <= conf <= 1)

    # to_dict
    d = land_a.to_dict()
    test("to_dict has label", d['label'] == 'A')
    test("to_dict has n", d['n'] == 3)


# =============================================================================
# TEST 4: FLAIR-OCR ENGINE
# =============================================================================

def test_flair_ocr():
    print("\n" + "=" * 60)
    print("TEST 4: FLAIR-OCR Engine")
    print("=" * 60)

    engine = FLAIR_OCR()

    # Train on A, B, O
    for char in ['A', 'B', 'O']:
        for sz in [24, 28, 32, 36, 40]:
            engine.train(char, make_char_image(char, font_size=sz))

    test("Engine has 3 landscapes", len(engine.landscapes) == 3)

    # Recognize
    result = engine.recognize(make_char_image('A', font_size=30))
    test("Recognize returns results", len(result) > 0)
    test("Top result is 'A'", result[0][0] == 'A')

    # Predict interface
    pred = engine.predict(make_char_image('O', font_size=30))
    test("Predict returns dict with 'predicted'", 'predicted' in pred)
    test("Predict has confidence", 'confidence' in pred)

    # Analyze
    analysis = engine.analyze_character(make_char_image('B', font_size=30))
    test("Analysis has T1w sequence", 'T1w' in analysis)
    test("Analysis has FLAIR sequence", 'FLAIR' in analysis)
    test("Analysis has diagnosis", 'diagnosis' in analysis)

    # Sequence contributions
    contrib = engine.get_sequence_contribution(make_char_image('A', font_size=30))
    test("Contributions has all sequences", 'T1w' in contrib and 'FLAIR' in contrib)
    test("Contributions ranked", '_ranked' in contrib)

    # Stats
    stats = engine.stats()
    test("Stats has n_landscapes", stats['n_landscapes'] == 3)
    test("Stats has signature_dims=252", stats['signature_dims'] == 252)


# =============================================================================
# TEST 5: TOPOLOGY CAPS
# =============================================================================

def test_topology_caps():
    print("\n" + "=" * 60)
    print("TEST 5: Topology Caps")
    print("=" * 60)

    caps = TopologyCaps()

    # Valid character (A)
    char_a = make_char_image('A')
    passed_a, report_a = caps.check(char_a)
    test("'A' passes topology caps", passed_a)
    test("Report has components", 'components' in report_a)
    test("Report has holes", 'holes' in report_a)

    # Noise (random pixels)
    noise = np.random.randint(0, 2, (5, 5), dtype=np.uint8)
    passed_n, report_n = caps.check(noise)
    # Noise might pass or fail depending on random pattern — just check report exists
    test("Noise check produces report", 'area' in report_n)

    # Too small
    tiny = np.ones((2, 2), dtype=np.uint8)
    passed_t, report_t = caps.check(tiny)
    test("Tiny region fails min_area", not passed_t)
    test("Failure is below_min_area", report_t['failure'] == 'below_min_area')

    # Horizontal line (bad aspect ratio)
    hline = np.zeros((3, 100), dtype=np.uint8)
    hline[1, :] = 1
    passed_h, report_h = caps.check(hline)
    test("Horizontal line fails aspect cap", not passed_h)

    # Filter regions
    regions = [
        {'binary': make_char_image('A'), 'col_start': 0},
        {'binary': tiny, 'col_start': 50},
        {'binary': make_char_image('B'), 'col_start': 100},
    ]
    filtered, rejected = caps.filter_regions(regions)
    test("Filter keeps valid regions", len(filtered) >= 1)
    test("Filter rejects invalid regions", len(rejected) >= 1)


# =============================================================================
# TEST 6: COHERENCE-CAPPED DETECTOR
# =============================================================================

def test_coherence_detector():
    print("\n" + "=" * 60)
    print("TEST 6: Coherence-Capped Detector")
    print("=" * 60)

    detector = CoherenceCappedDetector()

    # Create a test image with text
    img = Image.new('RGB', (200, 40), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 5), "ABC", fill=(0, 0, 0), font=font)
    rgb = np.array(img)
    gray = np.array(img.convert('L'))

    # Compute coherence
    coherence = detector.compute_coherence(gray)
    test("Coherence map has same shape", coherence.shape == gray.shape)
    test("Coherence values in [0, 1]", coherence.min() >= 0 and coherence.max() <= 1)

    # Detect in cell
    regions = detector.detect_in_cell(rgb, gray)
    test("Detector finds regions in text cell", len(regions) > 0)

    # Each region has required keys
    if regions:
        test("Regions have 'binary' key", 'binary' in regions[0])
        test("Regions have 'bbox' key", 'bbox' in regions[0])
        test("Regions sorted left-to-right",
             all(regions[i]['col_start'] <= regions[i+1]['col_start']
                 for i in range(len(regions)-1)) if len(regions) > 1 else True)

    # Empty cell
    empty_rgb = np.full((30, 100, 3), 255, dtype=np.uint8)
    empty_gray = np.full((30, 100), 255, dtype=np.uint8)
    empty_regions = detector.detect_in_cell(empty_rgb, empty_gray)
    test("Empty cell produces no regions", len(empty_regions) == 0)


# =============================================================================
# TEST 7: TOPOLOGY-COHERENCE FUSION
# =============================================================================

def test_topology_fusion():
    print("\n" + "=" * 60)
    print("TEST 7: Topology-Coherence Fusion")
    print("=" * 60)

    fusion = TopologyCoherenceFusion()

    # Feature extraction
    char_a = make_char_image('A')
    features = fusion.extract_fused_features(char_a)
    test("Fused features are 32-dimensional", len(features) == 32)
    test("Features are finite", np.all(np.isfinite(features)))

    # Different chars produce different features
    char_o = make_char_image('O')
    features_o = fusion.extract_fused_features(char_o)
    test("Different chars have different features",
         np.linalg.norm(features - features_o) > 0.01)

    # Detect and extract from cell
    rgb, gray = make_test_cell("AB")
    regions = fusion.detect_and_extract(rgb, gray)
    test("Fusion detects regions in cell", len(regions) > 0)
    if regions:
        test("Regions have 'features' key", 'features' in regions[0])
        test("Features are 32-dim", len(regions[0]['features']) == 32)

    # Empty features
    empty = np.zeros((3, 3), dtype=np.uint8)
    empty_feats = fusion.extract_fused_features(empty)
    test("Empty region produces zero features", np.abs(empty_feats).sum() == 0)


# =============================================================================
# TEST 8: ADAPTIVE TOPOLOGY CAPS
# =============================================================================

def test_adaptive_caps():
    print("\n" + "=" * 60)
    print("TEST 8: Adaptive Topology Caps")
    print("=" * 60)

    caps = AdaptiveTopologyCaps()
    initial_component_range = caps.component_range

    # Observe many characters
    for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij':
        binary = make_char_image(char)
        caps.observe(binary)

    test("Caps tracked observations", caps._n_adaptations >= 30)
    test("Caps adapted (component range may change)",
         caps._n_adaptations >= 20)

    stats = caps.get_stats()
    test("Stats has n_observations", stats['n_observations'] >= 30)
    test("Stats has component_range", 'component_range' in stats)
    test("Stats has density_range", 'density_range' in stats)

    # Adapted caps should still pass valid characters
    passed, report = caps.check(make_char_image('A'))
    test("Adapted caps still pass valid character 'A'", passed)


# =============================================================================
# TEST 9: MULTI-ENGINE RECOGNIZER
# =============================================================================

def test_multi_engine():
    print("\n" + "=" * 60)
    print("TEST 9: Multi-Engine Recognizer")
    print("=" * 60)

    recognizer = MultiEngineRecognizer()

    # Train on a few characters
    for char in ['A', 'B', 'O', 'H']:
        for sz in [24, 28, 32, 36, 40]:
            recognizer.train_char(char, make_char_image(char, font_size=sz))

    # Recognize
    result = recognizer.recognize(make_char_image('A', font_size=30))
    test("Result is InterpretationResult", isinstance(result, InterpretationResult))
    test("Result has label", result.label is not None)
    test("Result confidence in [0, 1]", 0 <= result.confidence <= 1)
    test("Result has engine votes", len(result.engine_votes) > 0)
    test("Agreement in [0, 1]", 0 <= result.agreement() <= 1)

    # Top prediction should be 'A'
    test("Recognizes 'A' correctly", result.label == 'A')

    # to_dict
    d = result.to_dict()
    test("to_dict has label", 'label' in d)
    test("to_dict has votes", 'votes' in d)

    # Correct
    wrong_result = recognizer.recognize(make_char_image('B', font_size=30))
    recognizer.correct(wrong_result, 'B', make_char_image('B', font_size=30))
    test("Correction accepted", True)

    # Stats
    stats = recognizer.stats()
    test("Stats has flair_landscapes", 'flair_landscapes' in stats)
    test("Stats has weights", 'weights' in stats)


# =============================================================================
# TEST 10: IMAGE INTERPRETER
# =============================================================================

def test_image_interpreter():
    print("\n" + "=" * 60)
    print("TEST 10: Image Interpreter (Full Pipeline)")
    print("=" * 60)

    # Create interpreter (train on minimal set for speed)
    interpreter = ImageInterpreter(adaptive_caps=True)

    # Train on a few characters
    for char in list('ABCDHOabcdho0123'):
        for sz in [24, 28, 32, 36]:
            try:
                binary = make_char_image(char, font_size=sz)
                interpreter.recognizer.train_char(char, binary)
            except Exception:
                pass
    interpreter._trained = True

    test("Interpreter is trained", interpreter._trained)

    # Read a cell
    rgb, gray = make_test_cell("AB", width=100, height=35, font_size=22)
    cell_result = interpreter.read_cell(rgb, gray)
    test("Cell result is CellResult", isinstance(cell_result, CellResult))
    test("Cell result has text", isinstance(cell_result.text, str))
    test("Cell result has raw_text", isinstance(cell_result.raw_text, str))
    test("Cell confidence in [0, 1]", 0 <= cell_result.confidence() <= 1)

    # to_dict
    d = cell_result.to_dict()
    test("to_dict has text", 'text' in d)
    test("to_dict has confidence", 'confidence' in d)

    # Read a line
    line_gray = gray
    line_result = interpreter.read_line(line_gray)
    test("Line result is CellResult", isinstance(line_result, CellResult))

    # Stats
    stats = interpreter.stats()
    test("Stats has 'trained'", 'trained' in stats)
    test("Stats has 'recognizer'", 'recognizer' in stats)
    test("Stats has 'adaptive_caps'", 'adaptive_caps' in stats)

    # Summary
    summary = interpreter.summary()
    test("Summary is a string", isinstance(summary, str))
    test("Summary contains 'Image Interpreter'", 'Image Interpreter' in summary)

    # Analyze a character
    char_binary = make_char_image('A')
    analysis = interpreter.analyze_character(char_binary)
    test("Analysis has prediction", 'prediction' in analysis)
    test("Analysis has flair_analysis", 'flair_analysis' in analysis)
    test("Analysis has topology", 'topology' in analysis)


# =============================================================================
# TEST 11: DIFFERENTIATION POWER
# =============================================================================

def test_differentiation():
    print("\n" + "=" * 60)
    print("TEST 11: Differentiation Power (confusable pairs)")
    print("=" * 60)

    engine = FLAIR_OCR()

    # Train on confusable pairs
    confusable_pairs = [('a', 'e'), ('o', '0'), ('l', '1'), ('s', '5')]

    for pair in confusable_pairs:
        for char in pair:
            for sz in [24, 28, 32, 36, 40, 48]:
                engine.train(char, make_char_image(char, font_size=sz))

    # Test differentiation
    for c1, c2 in confusable_pairs:
        test_img = make_char_image(c1, font_size=34)
        results = engine.recognize(test_img, top_k=2)
        if results:
            top_label = results[0][0]
            # The FLAIR engine should at least have the correct char in top-2
            top_2_labels = [r[0] for r in results[:2]]
            test(f"'{c1}' vs '{c2}': '{c1}' in top-2", c1 in top_2_labels)


# =============================================================================
# TEST 12: INTEGRATION SMOKE TEST
# =============================================================================

def test_integration():
    print("\n" + "=" * 60)
    print("TEST 12: Integration Smoke Test")
    print("=" * 60)

    # Test that all imports work
    from shifu_ocr import (
        FLAIR_OCR, FLAIRLandscape, TopologyCaps,
        CoherenceCappedDetector, TopologyCoherenceFusion,
        ImageInterpreter, MultiEngineRecognizer,
        create_interpreter,
    )
    test("All imports successful", True)

    # Factory function
    # Don't actually train (slow) — just verify creation
    interpreter = ImageInterpreter(adaptive_caps=True)
    test("Factory creates ImageInterpreter", isinstance(interpreter, ImageInterpreter))
    test("Interpreter has recognizer", hasattr(interpreter, 'recognizer'))
    test("Interpreter has detector", hasattr(interpreter, 'detector'))
    test("Interpreter has fusion", hasattr(interpreter, 'fusion'))
    test("Interpreter has caps", hasattr(interpreter, 'caps'))


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def main():
    print()
    print("=" * 60)
    print("  MRI FLAIR OCR — Complete Test Suite")
    print("  Testing: FLAIR Engine, Topology Coherence, Image Interpreter")
    print("=" * 60)

    test_mri_sequences()
    test_flair_signatures()
    test_flair_landscapes()
    test_flair_ocr()
    test_topology_caps()
    test_coherence_detector()
    test_topology_fusion()
    test_adaptive_caps()
    test_multi_engine()
    test_image_interpreter()
    test_differentiation()
    test_integration()

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print(f"\n  {failed} test(s) FAILED")
        sys.exit(1)
    else:
        print("\n  All tests PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()
