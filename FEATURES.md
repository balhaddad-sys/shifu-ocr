# Shifu OCR v2.0.0 — Complete Features Reference

## Architecture: 6-Layer Neural Pipeline

```
L1: 4D MRI Spatial Encoding (Python)
L2: Line/Character Segmentation (Python)
L3: FLAIR Perturbation + Template Ensemble (Python)
L4: Document Adaptation + Somatotopic Processing (Python)
L5: Spatial Coordinate Reconstruction (Python)
L6: Clinical Correction + ATL Hub Integration (JavaScript)
```

---

## Layer 1: 4D MRI Spatial Encoding
**File:** `shifu_ocr/pipeline_worker.py`

- **Multi-sequence processing**: R, G, B, Luminance channels processed independently (like T1/T2/FLAIR/DWI in MRI)
- **Multi-scale Laplacian pyramid**: 3 levels (sigma=2, 6, 18) per channel — fine strokes, character scale, background
- **Gabor orientation contrast**: 4 orientations at freq=0.25 — text is anisotropic, backgrounds isotropic
- **Geometric mean fusion**: soft minimum across channels — tolerates variation, requires cross-channel consistency
- **Color-blind**: strips ANY colored background (white, green, yellow, red, blue) automatically

## Layer 2: Segmentation
**File:** `shifu_ocr/engine.py`

- **Line segmentation** (`segment_lines`): horizontal projection on binarized image, trusts L1 for clean input
- **Character segmentation** (`segment_characters`): vertical projection + Otsu binarization, min char width 3px
- **Line merging**: close lines (< min_gap pixels) merged to handle broken text
- **Noise filtering**: lines with < 15% alphanumeric characters = grid remnants, skipped

## Layer 3: FLAIR Perturbation + Ensemble
**File:** `shifu_ocr/engine.py`, `shifu_ocr/perturbation.py`

### FLAIR Perturbation Engine (120 dimensions)
**Principle:** Don't read the character. DISTURB it and measure the response.

**12 Perturbation Pulses:**
1. Baseline (original state)
2. Light erosion (what survives gentle thinning?)
3. Heavy erosion (what survives aggressive thinning?)
4. Dilation (what fills in and merges?)
5. Blur (how does structure dissolve?)
6. Skeleton (topological core)
7. Distance transform (deep structure)
8. Delta (rate of change under perturbation)
9. Heavy dilation (character spread at larger scale)
10. Row shift (asymmetry probe — even rows +1px, odd -1px)
11. Contrast inversion (enclosed spaces revealed)
12. Aggressive erosion (only thickest strokes survive)

**10 Measurements per Pulse:**
Mass, components, holes, vertical center, horizontal center, symmetry, compactness, edge density, stroke width, endpoint count

### Ensemble Classifier
- **FLAIR branch**: 120-dim perturbation signatures + static features + shape features = 177-dim
- **Template branch**: 32x32 normalized binary templates, cosine similarity
- **Wave interference fusion**: geometric mean of normalized scores — constructive when both agree, destructive when they disagree
- **Macula principle**: LANCZOS interpolation preserves edge detail (not NEAREST)

### Somatotopic Processing
- **Lateral connections**: each character looks at its neighbors — digit context prefers digits, letter context prefers letters
- **Two-pass read_line**: raw predictions first, then neighbor-refined predictions
- **Case disambiguation**: baseline position determines upper/lower when classifier is uncertain

## Layer 4: Document Adaptation
**File:** `shifu_ocr/engine.py`

- **2-pass OCR** (`read_page`):
  - Pass 1: OCR everything, collect high-confidence character samples (>0.5)
  - Adapt: clone model, train on document-specific samples
  - Pass 2: re-OCR with adapted model
- **Self-calibration**: the document teaches the model about its own rendering
- **Rules of presentation**: same font, same spacing = consistent responses

## Layer 5: Spatial Reconstruction
**File:** `shifu_ocr/engine.py`

- **MRI spatial encoding**: every character carries absolute (x,y) page coordinates
- **Word extraction**: gap-based word boundaries (median gap x 2.0 = space threshold)
- **Column detection**: cluster word X positions across all lines — positions that repeat = columns
- **Table reconstruction**: rows x columns from coordinate clustering
- **Relational memory**: words appearing 2+ times in same column = anchors; uncertain cells corrected by column anchors (SequenceMatcher > 0.6)

## Layer 6: Clinical Correction (ATL Hub)
**Files:** `core/pipeline.js`, `clinical/corrector.js`, `clinical/confusion.js`, `clinical/vocabulary.js`, `clinical/safety.js`

### ATL Hub (`_processLineResult`)
Integration point where all modalities converge:
- OCR text (from FLAIR engine)
- Spatial coordinates (from somatotopic map)
- Document structure (from table reconstruction)
- Clinical vocabulary (from corrector)
- Column type (from spatial position or text patterns)

### Hub-and-Spoke Scoring (7 Synaptic Signals)
1. **Confusion fit**: how well do known OCR confusions explain the difference?
2. **Column context**: does this word belong in the expected column type?
3. **Ward frequency**: how often has this word been confirmed by nurses?
4. **Context chain**: do previous words predict this word?
5. **Resonance**: structural equivalence from core engine
6. **Length compatibility**: how close in length?
7. **Proximity**: how close is the edit distance?

**Synchrony bonus**: 3+ signals firing = +0.3 (action potential), 2 signals = +0.1

### Cerebellar Multi-Pass Correction
- Pass 1: correct what's confident
- Pass 2: use corrected context to resolve uncertain words
- Pass 3: further refinement
- Stop early if no changes from previous pass

### Confusion Model
**52+ confusion pairs** with costs:
- Classic OCR: 0↔O, 1↔l, 5↔S, 8↔B
- FLAIR systematic: i↔t, 6↔g, g↔q, c↔x
- 4D MRI systematic: 4↔a/e/o, 3↔B, 0↔N, 6↔e
- **Case is free**: same letter different case = cost 0.0

### Clinical Vocabulary (600+ terms)
9 categories: ward_structure, diagnoses, examination, medications, labs, names, common_english, generated_large
- Dynamic expansion via prefix+suffix combinations (up to 100K terms)
- Column-specific vocabularies for context-aware boosting
- Medical acronyms: MRI, CT, ECG, EEG, GCS, BP, HR, RR, SPO2, etc.

### Safety System
- **Lab range validation**: 24 labs with physiological ranges, decimal repositioning
- **Medication ambiguity detection**: top-2 candidates both meds = DANGER flag
- **Dose plausibility**: >5000mg warning, insulin >200 units warning
- **Numeric canonicalization**: O→0, l→1 in digit context

### Case Normalization
- Known words with mixed OCR case → vocabulary canonical form
- ALL CAPS (<=5 chars) preserved (acronyms: CVA, AKI)
- Title Case preserved (names: Mohammed, Ward)
- Everything else → lowercase

---

## Core Language Engine
**File:** `core/engine.js`

### 60-Dimensional Word Vectors (6 Channels)
1. **Form** (16 dims): morphological shape — consonant/vowel patterns, bigrams
2. **Context** (12 dims): neighborhood structure — diversity, dominance, overlap
3. **History** (8 dims): temporal — frequency, age, recency, gap regularity
4. **Influence** (8 dims): structural impact — sentence variability, triangulation
5. **Contrast** (8 dims): deviation from norms — frequency percentile, stability
6. **Expectation** (8 dims): predictive — forward/backward dominance, directionality

### Resonance Learning
- Words that fill the same structural slots build resonance evidence
- "doctor" and "physician" never co-occur but both follow "the" and precede verbs → equivalence learned
- Dynamic discount: base 0.6 + evidence × 0.08, capped at 0.95

### Soft Trajectories
- `softNx(a)`: what follows word a? Hard path → resonance path → similarity fallback
- `softNx2(a,b)`: two-step trajectories
- Skip-gram expectations: long-range context (up to 7 words)

### Coherence Scoring
- Sequential surprise + trajectory surprise + long-range surprise + field surprise
- Weighted: 0.25 + 0.20 + 0.25 + 0.30
- Returns: coherence = 1 - mean_surprise

---

## Learning Loop
**File:** `learning/loop.js`

### Adaptive Confusion Profile
- Learns OCR confusions from nurse corrections
- Empirical cost replaces static cost as experience grows (blend over 50 corrections)

### Ward Vocabulary
- Tracks confirmed word frequencies with clinical weights
- Category associations per word
- Frequency boost for fuzzy matching: log(1 + count) / log(101)

### Context Chains
- Co-occurrence across columns: "Doctor=Saleh" → "Diagnosis=stroke"
- Bidirectional: any field predicts any other field

### Clinical Severity Weights
- Medications: weight 3.0, minConfidence 0.85, safetyOverride
- Diagnoses: weight 2.5, minConfidence 0.7
- Lab values: weight 2.5, minConfidence 0.8
- Status/Triage: weight 2.0
- Names: weight 1.5
- Room/Ward: weight 1.0

---

## Document Ingestion
**File:** `core/ingest.js`

- **PDF**: PyMuPDF → pdfplumber → PyPDF2 fallback chain, 1000+ pages, 10-minute timeout
- **CSV/TSV**: RFC-compliant parsing with quoted fields
- **Images**: PNG, JPG, TIFF, BMP → full OCR pipeline
- **Plain text**: line-by-line correction
- **Auto-detection**: magic byte detection for unknown extensions

---

## Server & UI
**File:** `server.js`

### API Endpoints (11)
POST /api/correct, /api/correct-row, /api/learn, /api/undo, /api/score, /api/compare-structure, /api/roles, /api/ingest, /api/upload
GET /api/stats, /api/history

### UI Tabs (6)
Correct Text | Ward Census | Learn | Upload File | Structure | Stats

### Security
- Body size limit: 10MB
- Path traversal validation
- Filename sanitization
- 413 for oversized requests

---

## Medical Corpus
**File:** `learning/medical_corpus.js`

300+ clinical sentences covering: Neurology, Cardiology, Emergency, Internal Medicine, Nursing, Pharmacy, Procedures, Discharge summaries

---

## Design Principles (from user)

1. **MRI-RF**: different filter scales reveal different structures
2. **FLAIR**: disturb and measure response, don't read statically
3. **Macula**: preserve detail at highest acuity
4. **Continuity**: backgrounds are homogeneous, text is discontinuous
5. **Wave interference**: constructive when signals agree, destructive when they don't
6. **Somatotopy**: neighbors inform identity, position-aware processing
7. **Hub-and-spoke**: multiple weak signals accumulate to threshold
8. **Cerebellar correction**: iterative refinement through feedback loops
9. **ATL hub**: single integration point where all modalities converge
10. **Relational memory**: never stored in isolation, always in relation
11. **Case is free**: same shape = same cost regardless of case

---

**Total: 200+ features across 15 files, 91.4% character accuracy**
