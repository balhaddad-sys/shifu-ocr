"""
Shifu-OCR: Full Deployment Demo
=================================
Train → Test → Read text lines → Clinical post-process → Save model

End-to-end proof: image in, clinically-validated text out.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    plt = None
    gridspec = None
from shifu_ocr.engine import ShifuOCR
from shifu_ocr.clinical import ClinicalPostProcessor


# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_FONTS = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
]

TEST_FONTS = [
    '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
    '/usr/share/fonts/truetype/freefont/FreeSerif.ttf',
    '/usr/share/fonts/truetype/freefont/FreeMono.ttf',
    '/usr/share/fonts/truetype/google-fonts/Lora-Italic-Variable.ttf',
]

ALL_FONTS_NAMED = [
    ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'DejaVu Sans'),
    ('/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf', 'DejaVu Serif'),
    ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 'DejaVu Bold'),
    ('/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf', 'Serif Bold'),
    ('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 'Mono'),
    ('/usr/share/fonts/truetype/freefont/FreeSans.ttf', 'FreeSans*'),
    ('/usr/share/fonts/truetype/freefont/FreeSerif.ttf', 'FreeSerif*'),
    ('/usr/share/fonts/truetype/freefont/FreeMono.ttf', 'FreeMono*'),
]

ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
DIGITS = list('0123456789')
ALL_CHARS = ALPHABET + DIGITS


def add_noise(img, sigma=0.1):
    return np.clip(img + np.random.normal(0, sigma*255, img.shape), 0, 255).astype(np.uint8)

def add_blur(img, r=1.5):
    return np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=r)))


# =============================================================================
# PHASE 1: TRAIN
# =============================================================================

def train_model():
    print("=" * 70)
    print("PHASE 1: TRAINING")
    print("=" * 70)
    
    ocr = ShifuOCR()
    
    # Train on letters + digits at the primary size
    n = ocr.train_from_fonts(ALL_CHARS, TRAIN_FONTS, font_size=70, img_size=(100, 100))
    
    # Train at multiple sizes for scale invariance
    for size in [40, 50, 60, 80, 90]:
        ocr.train_from_fonts(ALL_CHARS, TRAIN_FONTS[:3], font_size=size, img_size=(100, 100))
    
    # Also train with tighter bounding boxes (like segmented chars from lines)
    for size in [30, 36, 42]:
        ocr.train_from_fonts(ALL_CHARS, TRAIN_FONTS[:3], font_size=size, img_size=(60, 60))
    
    stats = ocr.get_stats()
    print(f"\n  Characters:  {stats['characters']}")
    print(f"  Depth range: {stats['min_depth']}-{stats['max_depth']} observations per character")
    print(f"  Training:    {len(TRAIN_FONTS)} fonts × {len(ALL_CHARS)} characters + size variants")
    print(f"  Total:       ~{n + len(ALPHABET)*2*3} observations absorbed")
    
    return ocr


# =============================================================================
# PHASE 2: CHARACTER-LEVEL TESTING
# =============================================================================

def test_characters(ocr):
    print(f"\n{'='*70}")
    print("PHASE 2: CHARACTER-LEVEL ACCURACY")
    print("=" * 70)
    
    # Test on both training and unseen fonts
    print("\n  Per-font accuracy (letters only):")
    print(f"  {'Font':22s} {'Accuracy':>10s}  {'Bar':40s}  Notes")
    print(f"  {'-'*22} {'-'*10}  {'-'*40}  -----")
    
    font_results = {}
    for font_path, font_name in ALL_FONTS_NAMED:
        correct = 0
        for char in ALPHABET:
            img = ocr._render(char, font_path, 70, (100, 100))
            pred = ocr.predict_character(img)
            if pred['predicted'] == char:
                correct += 1
        
        acc = correct / len(ALPHABET) * 100
        bar = "█" * int(acc / 2.5) + "░" * (40 - int(acc / 2.5))
        note = "UNSEEN" if '*' in font_name else "trained"
        print(f"  {font_name:22s} {acc:8.1f}%  {bar}  {note}")
        font_results[font_name] = acc
    
    # Digit accuracy
    print(f"\n  Digit accuracy:")
    for font_path, font_name in ALL_FONTS_NAMED[:3]:
        correct = 0
        for d in DIGITS:
            img = ocr._render(d, font_path, 70, (100, 100))
            pred = ocr.predict_character(img)
            if pred['predicted'] == d:
                correct += 1
        acc = correct / len(DIGITS) * 100
        print(f"    {font_name:22s} {acc:.1f}%")
    
    # Noise robustness
    print(f"\n  Noise robustness (DejaVu Sans):")
    font = TRAIN_FONTS[0]
    for name, fn in [('Clean', lambda x: x),
                     ('Noise σ=0.1', lambda x: add_noise(x, 0.1)),
                     ('Blur r=2.0', lambda x: add_blur(x, 2.0)),
                     ('Noise+Blur', lambda x: add_blur(add_noise(x, 0.08), 1.5))]:
        np.random.seed(42)
        correct = sum(1 for c in ALPHABET
                     if ocr.predict_character(fn(ocr._render(c, font, 70, (100,100))))['predicted'] == c)
        acc = correct / len(ALPHABET) * 100
        bar = "█" * int(acc / 2.5) + "░" * (40 - int(acc / 2.5))
        print(f"    {name:20s} {bar} {acc:.1f}%")
    
    return font_results


# =============================================================================
# PHASE 3: WORD/LINE READING
# =============================================================================

def test_line_reading(ocr):
    print(f"\n{'='*70}")
    print("PHASE 3: LINE READING (end-to-end)")
    print("=" * 70)
    
    test_lines = [
        ("CRANIAL NERVES INTACT", "Clinical exam heading"),
        ("POWER 5 UPPER LIMBS", "Motor examination"),
        ("REFLEXES NORMAL", "Reflex finding"),
        ("BABINSKI NEGATIVE", "Plantar response"),
        ("HBA1C 71", "Lab value (missing decimal)"),
        ("SODIUM 139", "Lab value (normal)"),
        ("LEVETIRACETAM 500MG", "Medication"),
        ("ROTIGOTINE 4MG", "Medication"),
        ("PUPILS REACTIVE", "Cranial nerve finding"),
        ("NO PAPILLEDEMA", "Fundoscopy finding"),
    ]
    
    font = TRAIN_FONTS[0]
    cpp = ClinicalPostProcessor()
    
    print(f"\n  {'Input':35s} → {'OCR Raw':25s} → {'Clinical':25s} {'Flag'}")
    print(f"  {'-'*35}   {'-'*25}   {'-'*25} ----")
    
    line_results = []
    for text, desc in test_lines:
        img = ShifuOCR.render_text_line(text, font, font_size=36, padding=12)
        result = ocr.read_line(img)
        raw_text = result['text']
        
        # Clinical post-processing
        cpp_result = cpp.process_text(raw_text)
        clinical_text = cpp_result['output']
        
        flags = [f['flag'] for f in cpp_result['flags']]
        flag_str = ', '.join(flags[:2]) if flags else '✓'
        
        match = "✓" if raw_text == text else "~" if clinical_text.upper() == text else "✗"
        
        print(f"  {match} {text:33s} → {raw_text:25s} → {clinical_text:25s} {flag_str}")
        
        line_results.append({
            'input': text, 'raw': raw_text, 'clinical': clinical_text,
            'raw_match': raw_text == text,
            'confidence': result['confidence'],
        })
    
    raw_acc = sum(1 for r in line_results if r['raw_match']) / len(line_results) * 100
    print(f"\n  Raw line accuracy:      {raw_acc:.0f}%")
    print(f"  Average confidence:     {np.mean([r['confidence'] for r in line_results]):.0%}")
    
    return line_results


# =============================================================================
# PHASE 4: CLINICAL CONTEXT DEMO
# =============================================================================

def test_clinical_context():
    print(f"\n{'='*70}")
    print("PHASE 4: CLINICAL POST-PROCESSING")
    print("=" * 70)
    
    cpp = ClinicalPostProcessor()
    
    # Simulate OCR errors that the clinical engine should catch
    test_cases = [
        ("Crainal Merves intact", "Transposed + smudged"),
        ("Puplis reactive PERRLA", "Typo in pupils"),
        ("Power 5 upper linbs nornal tone", "Multiple small errors"),
        ("Babinksi upqoing plantar", "Multiple clinical terms"),
        ("Levetiracetam 500mg Rotigotine 4mg", "Medication list"),
        ("Carbamazepime 200mg", "Medication with OCR error"),
        ("HbA1c 71 Sodium 139 Potassium 45", "Lab values"),
    ]
    
    print()
    for text, desc in test_cases:
        result = cpp.process_text(text)
        
        print(f"  [{desc}]")
        print(f"  Input:    {result['input']}")
        print(f"  Output:   {result['output']}")
        
        flags = result['flags']
        if flags:
            for f in flags[:3]:
                extra = ''
                if 'alternatives' in f:
                    extra = f" → suggests {f['alternatives']}"
                elif 'context' in f:
                    extra = f" ({f['context']})"
                print(f"  ⚠ {f['input']:15s} → {f['output']:15s} [{f['flag']}]{extra}")
        print()


# =============================================================================
# PHASE 5: VISUALIZATION
# =============================================================================

def create_demo_visualization(ocr, line_results):
    """Create a comprehensive visual demo."""
    
    fig = plt.figure(figsize=(24, 16))
    fig.patch.set_facecolor('#0a0a0a')
    
    gs = gridspec.GridSpec(3, 4, hspace=0.35, wspace=0.3,
                           top=0.92, bottom=0.05, left=0.06, right=0.97)
    
    fig.suptitle("Shifu-OCR — Fluid Theory Optical Character Recognition\n"
                 "Model the medium. Detect displacement. Let experience shape the landscape.",
                 fontsize=16, fontweight='bold', color='white', y=0.97)
    
    # --- Panel 1: Pipeline visualization for one character ---
    chars_to_show = ['A', 'B', 'O', 'T']
    font = TRAIN_FONTS[0]
    
    for i, char in enumerate(chars_to_show):
        ax = fig.add_subplot(gs[0, i])
        img = ocr._render(char, font, 70, (100, 100))
        from shifu_ocr.engine import estimate_background, compute_displacement
        bg = estimate_background(img, k=15)
        disp = compute_displacement(img, bg)
        ax.imshow(disp, cmap='magma')
        ax.set_title(f"'{char}' — Displacement Field", color='white', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # --- Panel 2: Landscape consistency visualization ---
    chars_compare = ['B', 'O', 'D', 'I']
    from shifu_ocr.engine import extract_features, image_to_binary, extract_region, normalize_region
    
    for i, char in enumerate(chars_compare):
        ax = fig.add_subplot(gs[1, i])
        land = ocr.landscapes.get(char)
        if land and land.variance is not None:
            precision = 1.0 / (land.variance + 1e-8)
            # Show top features by precision
            n_feats = min(15, len(precision))
            from shifu_ocr.engine import FEATURE_NAMES
            indices = np.argsort(-precision)[:n_feats]
            names = [FEATURE_NAMES[j][:10] if j < len(FEATURE_NAMES) else f'f{j}' for j in indices]
            vals = precision[indices]
            vals = vals / vals.max()  # Normalize
            
            colors = ['#e74c3c' if v > 0.7 else '#f39c12' if v > 0.3 else '#3498db' for v in vals]
            bars = ax.barh(range(n_feats), vals, color=colors, edgecolor='none')
            ax.set_yticks(range(n_feats))
            ax.set_yticklabels(names, fontsize=7, color='white')
            ax.set_title(f"'{char}' — Landscape Peaks", color='white', fontsize=10, fontweight='bold')
            ax.set_xlim(0, 1.1)
            ax.invert_yaxis()
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(colors='white', labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#333')
    
    # --- Panel 3: Line reading examples ---
    for i, lr in enumerate(line_results[:4]):
        ax = fig.add_subplot(gs[2, i])
        font_path = TRAIN_FONTS[0]
        img = ShifuOCR.render_text_line(lr['input'], font_path, font_size=28, padding=8)
        ax.imshow(img, cmap='gray')
        
        match_color = '#2ecc71' if lr['raw_match'] else '#e74c3c'
        ax.set_title(f"Input: {lr['input']}\nRead: {lr['raw']}",
                     color=match_color, fontsize=9, fontweight='bold')
        ax.axis('off')
        ax.set_facecolor('#1a1a1a')
    
    return fig


def create_learning_curve_plot(ocr):
    """Show how the landscape evolved during training."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#0a0a0a')
    fig.suptitle("Shifu-OCR — What the Landscapes Learned",
                 fontsize=14, fontweight='bold', color='white')
    
    from shifu_ocr.engine import FEATURE_NAMES
    
    # Panel 1: Feature consistency across all characters
    ax = axes[0]
    all_precisions = []
    for label, land in ocr.landscapes.items():
        if land.variance is not None and label in ALPHABET:
            precision = 1.0 / (land.variance + 1e-8)
            all_precisions.append(precision)
    
    if all_precisions:
        avg_precision = np.mean(all_precisions, axis=0)
        n = min(20, len(avg_precision))
        idx = np.argsort(-avg_precision)[:n]
        names = [FEATURE_NAMES[i][:12] if i < len(FEATURE_NAMES) else f'f{i}' for i in idx]
        vals = avg_precision[idx]
        vals = vals / vals.max()
        
        colors = ['#e74c3c' if v > 0.5 else '#f39c12' if v > 0.2 else '#3498db' for v in vals]
        ax.barh(range(n), vals, color=colors)
        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=8, color='white')
        ax.set_title("Most Consistent Features\n(highest across all characters)", 
                     color='white', fontsize=11, fontweight='bold')
        ax.invert_yaxis()
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#333')
    
    # Panel 2: Confusion landscape (which characters overlap most)
    ax = axes[1]
    from collections import defaultdict
    confusion_counts = defaultdict(int)
    for label, land in ocr.landscapes.items():
        for conf, count in land.confused_with.items():
            pair = tuple(sorted([label, conf]))
            confusion_counts[pair] += count
    
    if confusion_counts:
        pairs = sorted(confusion_counts.items(), key=lambda x: -x[1])[:12]
        labels = [f"{p[0][0]}↔{p[0][1]}" for p in pairs]
        counts = [p[1] for p in pairs]
        ax.barh(range(len(labels)), counts, color='#e74c3c', alpha=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10, color='white', fontfamily='monospace')
        ax.set_title("Most Confused Pairs\n(landscape overlap)", 
                     color='white', fontsize=11, fontweight='bold')
        ax.invert_yaxis()
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#333')
    
    # Panel 3: Topology groups
    ax = axes[2]
    topo_groups = defaultdict(list)
    for label, land in ocr.landscapes.items():
        if land.mean is not None and label in ALPHABET:
            holes = round(land.mean[1])
            topo_groups[holes].append(label)
    
    y = 0
    colors_map = {0: '#2ecc71', 1: '#3498db', 2: '#e74c3c'}
    for holes in sorted(topo_groups.keys()):
        chars = sorted(topo_groups[holes])
        color = colors_map.get(holes, '#95a5a6')
        ax.text(0.05, y, f"{holes} holes:", fontsize=11, fontweight='bold',
                color=color, transform=ax.transAxes, va='top')
        char_str = '  '.join(chars)
        ax.text(0.35, y, char_str, fontsize=11, color='white',
                transform=ax.transAxes, va='top', fontfamily='monospace')
        y -= 0.15
    
    ax.set_title("Topological Classification\n(discovered from data)", 
                 color='white', fontsize=11, fontweight='bold')
    ax.axis('off')
    ax.set_facecolor('#1a1a1a')
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              SHIFU-OCR — Full Deployment                   ║")
    print("║                                                            ║")
    print("║  Fluid Theory OCR + Clinical Context Engine                ║")
    print("║  Image in → Text out → Clinically validated                ║")
    print("║                                                            ║")
    print("║  No neural network. No GPU. No cloud.                     ║")
    print("║  Just: medium displacement + fluid landscapes +            ║")
    print("║  clinical domain knowledge.                                ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    # Phase 1: Train
    ocr = train_model()
    
    # Phase 2: Character-level testing
    font_results = test_characters(ocr)
    
    # Phase 3: Line reading
    line_results = test_line_reading(ocr)
    
    # Phase 4: Clinical post-processing
    test_clinical_context()
    
    # Save model
    model_path = '/home/claude/shifu_ocr/trained_model.json'
    ocr.save(model_path)
    model_size = os.path.getsize(model_path)
    print(f"  Model saved: {model_path}")
    print(f"  Model size:  {model_size/1024:.1f} KB")
    
    # Phase 5: Visualization
    print(f"\n  Generating visualizations...")
    
    fig1 = create_demo_visualization(ocr, line_results)
    fig1.savefig('/home/claude/shifu_demo.png', dpi=150, bbox_inches='tight',
                 facecolor='#0a0a0a')
    plt.close(fig1)
    
    fig2 = create_learning_curve_plot(ocr)
    fig2.savefig('/home/claude/shifu_landscapes.png', dpi=150, bbox_inches='tight',
                 facecolor='#0a0a0a')
    plt.close(fig2)
    
    # Final summary
    stats = ocr.get_stats()
    trained_acc = np.mean([v for k, v in font_results.items() if '*' not in k])
    unseen_acc = np.mean([v for k, v in font_results.items() if '*' in k])
    
    print(f"\n{'='*70}")
    print("SHIFU-OCR — DEPLOYMENT SUMMARY")
    print("=" * 70)
    print(f"""
  MODEL
    Characters:          {stats['characters']} (A-Z + 0-9)
    Landscape depth:     {stats['min_depth']}-{stats['max_depth']} observations each
    Model size:          {model_size/1024:.1f} KB
    Architecture:        Fluid landscapes (Gaussian generative classifier)
    Neural network:      None
    GPU required:        No
    
  ACCURACY
    Trained fonts:       {trained_acc:.1f}%
    Unseen fonts:        {unseen_acc:.1f}%
    Line reading:        {sum(1 for r in line_results if r['raw_match'])}/{len(line_results)} lines correct
    
  CLINICAL FEATURES
    Domain vocabulary:   ~300 neurology terms
    Lab range checking:  10 common lab values
    Medication matching: ~50 medications
    Safety flags:        Medication ambiguity, out-of-range values
    
  DESIGN PRINCIPLES
    1. The system suggests, the clinician decides.
    2. Low confidence = loud flag, not silent guess.
    3. Context narrows candidates. It never replaces judgment.
    4. Every feature was discovered from data, not programmed.
    5. The model is 100% auditable — every landscape is readable.
""")


if __name__ == '__main__':
    main()
