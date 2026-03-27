# Shifu OCR v2.0.0

Medical OCR engine built on fluid theory and medium displacement. No neural network. No GPU. No cloud.

## Core Design: Multi-Engine Ensemble

Shifu does not use a single OCR algorithm. Multiple engines see every character **simultaneously**, each through a different lens:

| Engine | Lens | Best At |
|--------|------|---------|
| **Topology** (`engine.py`) | Static form: components, holes, symmetry | High-resolution, clean text |
| **Fluid** (`fluid.py`) | Probability landscapes shaped by experience | Adapting to new fonts/styles |
| **Perturbation** (`perturbation.py`) | MRI-style response to disturbance | Low-resolution (6px) characters |
| **Theory-Revision** (`theory_revision.py`) | Principled prediction with auditable reasoning | Explainable corrections |

> Co-Defining (`codefining.py`) and Coherence (`coherence.py`) exist as experimental modules but are **not registered** in the active ensemble pipeline.

```
  Character Image
       │
       ├──→ Topology Engine ──→ vote (label, confidence)
       ├──→ Fluid Engine ────→ vote (label, confidence)
       ├──→ Perturbation ────→ vote (label, confidence)     ──→  FUSION  ──→  Result
       └──→ Theory-Revision ─→ vote (label, confidence)         (weighted)
                                                                    │
                                                              Nurse correction
                                                                    │
                                                    ┌───────────────┼───────────────┐
                                                    ▼               ▼               ▼
                                              reshape          revise          update
                                              landscapes       theories        constraints
                                              (all engines absorb corrections simultaneously)
```

No single lens is always right. The ensemble is. Engine weights adapt: engines that perform better on your documents automatically get more influence.

## Architecture

```
shifu-ocr/
├── index.js                 # Main API entry point
├── server.js                # Interactive web UI (localhost:3737)
├── package.json             # Node.js project config
├── requirements.txt         # Python dependencies
├── generate_pdf.py          # Reference documentation generator
│
├── core/                    # JS linguistic engine
│   ├── engine.js            # 60D vector engine with resonance learning
│   ├── feedback.js          # Three-channel feedback loop
│   ├── pipeline.js          # Image → Python OCR → JS correction
│   ├── persistence.js       # Auto-save/load session state
│   ├── ingest.js            # PDF, CSV, image, TXT document ingestion
│   ├── metrics.js           # Accuracy and performance tracking
│   ├── invariance.js        # Semantic role extraction and comparison
│   └── trainedLoader.js     # Python trained model → JS bridge
│
├── clinical/                # Medical domain knowledge
│   ├── vocabulary.js        # 200+ drugs, 40+ lab tests, ward terms
│   ├── confusion.js         # OCR character confusion cost model
│   ├── safety.js            # Lab range validation, medication ambiguity
│   ├── corrector.js         # Clinical-weighted word/line correction
│   └── numeric.js           # Dose and digit normalization
│
├── learning/                # Adaptive learning from nurse corrections
│   ├── loop.js              # Adaptive confusion, ward vocabulary, context chains
│   ├── engine.js            # Wave propagation, inhibition, settling
│   ├── corpus.js            # Medical corpus seeder (250+ sentences)
│   ├── medical_corpus.js    # Extended domain knowledge
│   ├── clinical_weights.js  # Learning rates by clinical importance
│   ├── ablation.js          # Component impact testing
│   ├── scale.js             # Scaling experiments
│   └── scale.js             # Scaling experiments
│
├── shifu_ocr/               # Python OCR engines (multi-engine ensemble)
│   ├── ensemble.py          # Multi-engine orchestrator — all engines see every character
│   ├── engine.py            # Topology engine: components, holes, symmetry, projections
│   ├── fluid.py             # Fluid engine: probability landscapes (no rules)
│   ├── perturbation.py      # Perturbation engine: MRI-style relaxation signatures
│   ├── theory_revision.py   # Theory-revision engine: auditable error diagnosis
│   ├── codefining.py        # Co-defining engine: bidirectional char↔word↔context
│   ├── coherence.py         # Coherence displacement (colored backgrounds)
│   ├── displacement.py      # Formal medium displacement theory
│   ├── photoreceptor.py     # Smart per-cell adaptive binarization
│   ├── complete.py          # Full integrated pipeline
│   ├── clinical.py          # Python-side clinical post-processing
│   ├── clinical_context.py  # Clinical context module
│   ├── pipeline_worker.py   # JS ↔ Python subprocess bridge
│   ├── deploy.py            # Deployment and model conversion
│   ├── train_medium.py      # Medium-complexity training
│   ├── train_extensive.py   # Large-scale training
│   ├── train_real.py        # Real document training
│   └── trained_model.json   # Pre-trained character data (2.4MB)
│
├── v2/                      # V2 JS modules (Firebase-ready)
│   ├── index.js             # V2 entry point
│   ├── shifuLearningLoop.js # Adaptive learning with empirical confusion
│   ├── shifuClinicalCorrector.js # Advanced clinical corrector
│   ├── shifuConfusionModel.js    # Confusion model
│   ├── shifuSafetyFlags.js       # Safety flag system
│   ├── shifuVocabulary.js        # Vocabulary module
│   └── shifuFirebase.js          # Firebase integration
│
├── training/                # Training data pipeline
│   ├── shield.py            # PII/PHI redaction (HIPAA-safe)
│   ├── harvest.py           # Seed harvesting from OCR scans
│   ├── bulk_seed.py         # Synthetic ward data generation
│   ├── prepare.py           # Training image synthesis
│   └── finetune.py          # PaddleOCR model fine-tuning
│
└── test/                    # Validation
    ├── suite.js             # Integration test suite (100+ assertions)
    └── demo.js              # Interactive demonstration
```

## Quick Start

```bash
# Install JS dependencies
npm install

# Install Python dependencies
pip install -r requirements.txt

# Start the interactive web server
npm start
# → http://localhost:3737

# Run tests
npm test

# Run demo
npm run demo
```

## Core Principles

1. **Model the medium, detect displacement** — characters are perturbations in a continuous field
2. **Fluid landscapes shaped by experience** — no fixed rules, probabilities reshape with data
3. **Structure and content co-define** — bidirectional constraints between character, word, and context
4. **Perturbation reveals identity** — MRI-style response signatures, not static form matching
5. **Clinical safety first** — never silently correct; flag dangerous ambiguities for human review
6. **Every correction teaches** — nurse confirmations reshape three fluid landscapes

## API

```javascript
const { createShifu } = require('./index');

const shifu = createShifu();

// Correct OCR text
const result = shifu.correctLine('Pt adrnitted with seizrue');

// Learn from nurse correction
shifu.learn(
  { patient: 'Moharnrned', diagnosis: 'seizrue' },
  { patient: 'Mohammed',   diagnosis: 'seizure' }
);

// Safety checks
shifu.checkLabRange('potassium', 85);  // flags: likely 8.5 or 4.5

// Structural invariance
shifu.compareStructure(
  'doctor treated patient',
  'patient was treated by doctor'
);
```

## Training Pipeline

```bash
# Generate synthetic ward data
npm run seed

# Prepare training images
npm run train:prepare

# Fine-tune PaddleOCR recognition model
npm run train:finetune
```

## License

MIT — Bader & Claude, March 2026
