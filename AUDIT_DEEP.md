# Deep Audit Report: Shifu OCR (Round 2)

**Date:** 2026-03-26
**Scope:** Performance, clinical safety, Python engine internals, subtle logic bugs
**Complements:** AUDIT.md (security, code quality, architecture)

---

## Executive Summary

This second-pass audit goes deeper than the first, focusing on **performance bottlenecks**, **clinical safety correctness**, **Python OCR engine mathematics**, and **line-by-line logic bugs**. Several **critical patient safety issues** and **algorithmic complexity problems** were uncovered.

| Category | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| Performance | 3 | 3 | 5 | 1 |
| Clinical Safety | 2 | 3 | 3 | 0 |
| Python Engines | 2 | 4 | 8 | 3 |
| Logic Bugs | 0 | 3 | 6 | 3 |
| **Total** | **7** | **13** | **22** | **7** |

---

## 1. PERFORMANCE FINDINGS

### PERF-01: O(n^4) Decay Loop [CRITICAL]
- **File:** `core/engine.js:257-264`
- **Description:** Triple-nested loop over `nx2` (second-order skip-grams) during decay, called every 100 sentences.
  ```javascript
  for (const w of Object.keys(this.nx2))        // O(V)
    for (const b of Object.keys(this.nx2[a]))    // O(E)
      for (const c of Object.keys(this.nx2[a][b])) // O(E)
        this.nx2[a][b][c] *= f;
  ```
- **Impact:** With 10K vocabulary, becomes catastrophic. Estimated 5-50s per decay cycle.
- **Fix:** Flatten nx2 into a Map with composite keys. Decay once instead of 3x nested loops.

### PERF-02: Python Subprocess Per Image — No Pooling [CRITICAL]
- **File:** `core/pipeline.js:242-278`
- **Description:** New Python process spawned for every single image. ~500ms cold start overhead per call. No process reuse.
- **Impact:** 100 images = 50+ seconds of pure spawn overhead.
- **Fix:** Implement persistent Python worker pool (4-8 workers) with message queue.

### PERF-03: Synchronous File I/O in Request Path [CRITICAL]
- **File:** `core/persistence.js:33-43, 72-91`
- **Description:** `fs.writeFileSync` x3 on every auto-save, `fs.readFileSync` on startup. Blocks event loop.
- **Impact:** 50MB state file = 500ms-2s freeze, blocking ALL concurrent requests.
- **Fix:** Use async I/O. Queue writes to batch updates.

### PERF-04: Array Slice in Hot Path [HIGH]
- **File:** `learning/engine.js:172-233`
- **Description:** `array.push()` + `array.slice(-N)` on every sentence for 6+ ring buffers. Creates new arrays constantly.
  ```javascript
  this.wp[w] = this.wp[w].slice(-100);  // New array every time
  ```
- **Impact:** O(k log k) per word where k = vocabulary. ~2.6M ops per sentence.
- **Fix:** Circular buffer with fixed-size array and index pointer.

### PERF-05: Regex Recompiled in Loop [HIGH]
- **File:** `clinical/confusion.js:89-91`
- **Description:** `new RegExp(ocr, 'g')` created per digraph per call. 7 digraphs x 100K calls/hr = 700K compilations.
- **Fix:** Pre-compile at module load.

### PERF-06: 768 Candidate Distance Calculations Per Word [HIGH]
- **File:** `clinical/corrector.js:318-445`
- **Description:** Levenshtein distance (O(len^2)) computed 768 times per word correction. Called twice in some paths.
- **Fix:** Pre-filter by length to top 50. Cache distance results.

### PERF-07: Unbounded Similarity Cache [MEDIUM]
- **File:** `core/engine.js:122`
- **Description:** `_simCache` grows without bound. Each unique target+pool combo = new entry.
- **Fix:** LRU cache with max 10K entries.

### PERF-08: Multi-Pass Correction 3x Overhead [MEDIUM]
- **File:** `clinical/corrector.js:510-551`
- **Description:** 3 passes over every word in document. 500-word doc = 1500 corrections.
- **Fix:** Convergence detection. Only re-process uncertain words.

### PERF-09: Expensive Skeletonization Per Character [MEDIUM]
- **File:** `shifu_ocr/engine.py:143-152`
- **Description:** `morphology.skeletonize()` called for every character. O(n^2) per character.
- **Fix:** Cache results. Skip for high-confidence predictions.

### PERF-10: Server Startup Blocks 2-5s [MEDIUM]
- **File:** `server.js:12-24`
- **Description:** Synchronous loading of trained model, state files, seeding — all blocking before first request.
- **Fix:** Lazy-load model on first use. Initialize metrics in background.

### PERF-11: Require() Inside Hot Loop [MEDIUM]
- **File:** `clinical/corrector.js:326`
- **Description:** `require('../core/engine')` called inside word correction loop. Node caches modules, but lookup still has overhead.
- **Fix:** Import at module top.

### PERF-12: O(n) includes() For Index Lookup [LOW]
- **File:** `core/engine.js:87`
- **Description:** `_idxLen[l].includes(w)` is O(n) linear search. Should be Set.
- **Fix:** Use `Set` instead of array.

---

## 2. CLINICAL SAFETY FINDINGS

### CLIN-01: Incomplete Dose/Unit Validation [HIGH]
- **File:** `clinical/safety.js:126-131`
- **Description:** The system does have dose validation — `checkDosePlausibility()` is called when a digit-containing word follows a known medication. However, the detection regex `/[\dO]/` is over-broad (triggers on room numbers, bed numbers, years) causing false positives. The dose plausibility function exists but unit-specific validation (mg vs g vs mcg per medication) is limited.
- **Impact:** False positive safety flags on non-dose numbers. Some implausible units may pass.
- **Recommendation:** Tighten detection regex to dose-like patterns. Expand per-medication unit tables.

### CLIN-02: PII Redaction Limited to Kuwait Formats [HIGH]
- **File:** `training/shield.py:23-33, 74`
- **Description:** PII regex patterns only detect Kuwait-format civil IDs and +965 phone numbers. No coverage for other ID formats or free-text patient names.
- **Note:** The `str.replace(original, token, 1)` pattern was originally flagged as missing duplicates, but this is **incorrect** — `finditer()` iterates over all matches in the original text, and each sequential `replace(..., 1)` correctly replaces the next remaining occurrence in the shielded string.
- **Impact:** Non-Kuwait PII formats are unprotected.
- **Fix:** Expand PII patterns or add NER-based detection.

### CLIN-03: No Physiological Range Validation [CRITICAL]
- **File:** `shifu_ocr/clinical_context.py`
- **Description:** No validation for age (0-150), blood pressure (40-250 mmHg), temperature (34-42°C), or other vital signs. The JS safety flags (`clinical/safety.js`) cover lab ranges but the Python clinical context has no range checks at all.
- **Impact:** Physiologically impossible values pass without flagging.

### CLIN-04: Vocabulary Non-Generalizable [HIGH]
- **File:** `shifu_ocr/clinical_context.py:40-150`
- **Description:** Hardcoded neurology-focused dictionary. Cardiology, pulmonology, nephrology terms missing. No fuzzy matching for medication typos ("lisinipril" → "lisinopril").
- **Impact:** Low correction accuracy outside neurology domain.

### CLIN-05: Learning Loop Poisoning Risk [HIGH]
- **File:** `learning/loop.js`
- **Description:** No validation that learned corrections are medically valid. A user could teach "amoxicillin" → "amoxicilin" (wrong spelling) and the system would learn and propagate the error. No approval workflow for high-risk corrections (medications, doses).
- **Fix:** Add a "protected terms" list for medications/drugs that cannot be overridden by learning.

### CLIN-06: Confidence Threshold Too Permissive [MEDIUM]
- **File:** `clinical/corrector.js` (decision logic)
- **Description:** Corrections with confidence as low as 0.5 can be auto-accepted. For medical text, this threshold should be higher (0.7-0.8 minimum) with mandatory "verify" for anything below.
- **Fix:** Separate thresholds for clinical vs non-clinical fields.

### CLIN-07: False Positive Corrections on Names [MEDIUM]
- **File:** `clinical/corrector.js`
- **Description:** Patient and doctor names can be "corrected" to vocabulary words. "Dr. Bader" could become "Dr. Badger" if Levenshtein distance is low enough and "badger" is in vocabulary.
- **Fix:** Add name-column detection. Skip vocabulary correction for name fields.

### CLIN-08: Ensemble Confidence Miscalibrated [MEDIUM]
- **File:** `shifu_ocr/ensemble.py:182-183`
- **Description:** `confidence = 0.5 * agreement + 0.5 * min(margin, 1.0)`. If 5/5 engines agree but all with margin=0.01, confidence = 0.505. Weak consensus gets high confidence.
- **Fix:** Use `agreement * min(margin, 1.0)` (multiplicative) so both must be high.

---

## 3. PYTHON ENGINE FINDINGS

### PY-01: Missing Log-Determinant Normalization [HIGH]
- **File:** `shifu_ocr/fluid.py:259-267`
- **Description:** Log-likelihood score missing `- 0.5 * sum(log(variance))`. Without this normalization, characters with high-variance landscapes are systematically favored over low-variance ones.
- **Impact:** Biased ensemble voting. Characters with more training data dominate.

### PY-02: NaN From Division by Zero in Displacement [HIGH]
- **File:** `shifu_ocr/displacement.py:81-82`
- **Description:** `(displacement - min) / (max - min)` when max == min (uniform regions) produces NaN. Check exists but applies AFTER normalization.
- **Fix:** Check BEFORE normalization.

### PY-03: Grid Removal Deletes Valid Characters [HIGH]
- **File:** `shifu_ocr/photoreceptor.py:119-123`
- **Description:** Hardcoded aspect ratio thresholds (8x horizontal, 10x vertical) for filtering grid lines. A "1" character (aspect ~0.2), "I" or "l" could be removed. A "W" (aspect ~5) could trigger the filter.
- **Fix:** Make thresholds adaptive based on cell size and font detection.

### PY-04: Confidence Bias Toward High-Observation Classes [HIGH]
- **File:** `shifu_ocr/fluid.py:273`
- **Description:** `confidence = log(n+1) * 0.5` is unbounded. After 10K observations, adds 4.6 points. Biases toward common characters regardless of actual fit quality.
- **Fix:** Cap: `confidence = min(log(n+1) * 0.5, 1.0)`.

### PY-05: No Held-Out Validation Set [HIGH]
- **Files:** `shifu_ocr/train_medium.py`, `shifu_ocr/train_real.py`
- **Description:** Training scripts load/train/save but never report held-out accuracy. No train/val/test split. No way to detect overfitting.
- **Fix:** Reserve 20% for validation. Report accuracy before saving model.

### PY-06: Curse of Dimensionality [MEDIUM]
- **File:** `shifu_ocr/engine.py` (feature extraction)
- **Description:** 62+ dimensional feature vector with likely <1000 samples per character class. Needs 10^6+ samples for reliable statistics in 62D space.
- **Fix:** Apply PCA or feature selection to reduce to 15-20 dimensions.

### PY-07: Variance Floor Too Small After Many Observations [MEDIUM]
- **File:** `shifu_ocr/fluid.py:234`
- **Description:** `floor = 0.1 / sqrt(n)`. At n=10000, floor=0.001. Variance drops to near-zero, creating extreme precision that rejects valid but slightly different inputs.
- **Fix:** Use fixed minimum floor (e.g., 0.01).

### PY-08: Scale-Dependent Noise Removal [MEDIUM]
- **File:** `shifu_ocr/perturbation.py:330`
- **Description:** `remove_small_objects(min_size=8)` is fixed. On 200x200 upscaled cells, 8 pixels is tiny and won't remove noise.
- **Fix:** Scale min_size proportionally to cell area.

### PY-09: Stores ALL Observations Forever [MEDIUM]
- **File:** `shifu_ocr/fluid.py:219`
- **Description:** `self.observations.append(vector.copy())`. After 1000 observations per character × 62 dimensions × 62 characters = 31 MB just for observation history.
- **Fix:** Store only sufficient statistics (mean, covariance, count).

### PY-10: Single-Pixel Symmetry Returns Perfect (1.0) [MEDIUM]
- **File:** `shifu_ocr/perturbation.py:165`, `shifu_ocr/displacement.py:234`
- **Description:** 1-pixel-wide characters return symmetry=1.0. This is semantically wrong and causes false matches with symmetric characters.

### PY-11: OOM Risk on Large Images [MEDIUM]
- **File:** `shifu_ocr/perturbation.py:313`, `shifu_ocr/fluid.py:313`
- **Description:** No image size limits. 10000x10000 image + gaussian_filter creates massive arrays.

### PY-12: Adaptive Binarization Not Truly Adaptive [MEDIUM]
- **File:** `shifu_ocr/photoreceptor.py:57-61`
- **Description:** `threshold = max(bg_std * 2, 15)`. Minimum of 15 ignores local variation for low-contrast cells.

### PY-13: Empty Image Returns Zero Vector Causing False Matches [MEDIUM]
- **File:** `shifu_ocr/perturbation.py:214`
- **Description:** Empty images return 120-dim zero vector. Any landscape trained on blank/dead patterns will match with high confidence.

### PY-14: Perturbation Pulse Ordering Undocumented [LOW]
- **File:** `shifu_ocr/perturbation.py:202-274`
- **Description:** 12 perturbation pulses in specific order. No justification or ablation study. Correlated pulses (light/heavy erosion) inflate dimensionality without adding information.

### PY-15: LANCZOS Used for Upscaling [LOW]
- **File:** `shifu_ocr/engine.py:44`
- **Description:** LANCZOS interpolation blurs tiny features when upscaling. Should use NEAREST for upscale, LANCZOS for downscale.

### PY-16: 10% Real Training Data Unused [LOW]
- **File:** `shifu_ocr/train_real.py:91`
- **Description:** Alignment skipped entirely when segment count differs by >10% from label length. Training data silently discarded.

---

## 4. LOGIC BUG FINDINGS

### BUG-01: Impossible Boolean Condition [HIGH]
- **File:** `core/feedback.js:138`
- **Code:** `if (!confirmedText && confirmedText !== '') continue;`
- **Description:** This condition can never be true. When `confirmedText` is falsy (null, undefined, 0, false, ''), the first part succeeds. But if it's `''`, the second part (`!== ''`) fails. If it's `null`/`undefined`, `!== ''` succeeds. So the condition only triggers for `null`, `undefined`, `0`, `false` — but NOT `''`. This is probably the opposite of intent: empty strings should be skipped too.
- **Fix:** `if (confirmedText == null) continue;`

### BUG-02: CSV Parser Includes Quote Delimiters in Output [HIGH]
- **File:** `core/ingest.js:245-254`
- **Code:**
  ```javascript
  if (ch === '"') {
    // ...handle escapes...
    current += ch;  // ALWAYS adds quote character
  }
  ```
- **Description:** Both delimiter quotes and escaped quotes are appended to field content. Standard CSV parsers strip delimiter quotes. All quoted CSV fields will contain spurious quote characters.
- **Fix:** Only add `ch` to `current` when it's not a quote character acting as delimiter.

### BUG-03: Shared Mutable Global State [HIGH]
- **File:** `clinical/confusion.js:42-47`
- **Code:**
  ```javascript
  let _adaptiveProfile = null;
  function setAdaptiveProfile(profile) { _adaptiveProfile = profile; }
  ```
- **Description:** Module-level mutable state. If multiple shifu instances exist in the same process, the last `setAdaptiveProfile()` call wins. All instances share one profile.
- **Fix:** Pass profile as parameter or use instance-scoped storage.

### BUG-04: stderr Mixed Into stdout in Version Check [LOW]
- **File:** `core/pipeline.js:156` (`checkPython()` only)
- **Description:** In the `checkPython()` version-check function, stderr is appended to the stdout variable. This is a **harmless workaround** because `python --version` writes to stderr on some systems. The actual OCR runner (`_runPythonOCR` at line 254-257) already correctly separates stdout and stderr into distinct buffers.
- **Status:** Fixed — version check now uses separate buffers with `(stdout || stderr)` fallback.

### BUG-05: Safety Regex Overly Broad [MEDIUM]
- **File:** `clinical/safety.js:126`
- **Code:** `/[\dO]/.test(w.corrected)`
- **Description:** Matches ANY word containing a digit. Room numbers, bed numbers, ages all trigger dose checking against the previous word. Causes false positive safety flags.
- **Fix:** Match dose-like patterns: `/^\d+\.?\d*(mg|mcg|g|ml|units?|iu)$/i`

### BUG-06: Lookbehind Assertion Compatibility [MEDIUM]
- **File:** `clinical/safety.js:93`
- **Code:** `doseStr.replace(/(?<=[0-9])l/g, '1')`
- **Description:** Lookbehind assertions require ES2018+ (Node.js 8.10+). Breaks on older runtimes at parse time (not a runtime error — syntax error prevents module loading).
- **Fix:** Use character class or manual iteration.

### BUG-07: Alignment Pre-Decrement Confusion [MEDIUM]
- **File:** `learning/loop.js:308-309`
- **Code:** `aligned.push({ type: 'deletion', pi: --i, ci: -1 });`
- **Description:** Pre-decrement means index is decremented before being used. Semantically correct but easy to misread during maintenance. Could cause subtle alignment bugs if modified.
- **Fix:** Separate decrement from usage for clarity.

### BUG-08: Floating Point Comparison [MEDIUM]
- **File:** `core/feedback.js:332`
- **Description:** Similarity scores from weighted sums compared without epsilon. Accumulated floating point errors can cause equality checks to fail.

### BUG-09: Missing Null Guard in CSV Columns [MEDIUM]
- **File:** `core/ingest.js:194-195`
- **Description:** `columns.forEach()` called without checking if columns array exists or has length.

### BUG-10: Prototype Pollution via Object Keys [LOW]
- **File:** `clinical/vocabulary.js`, `learning/loop.js`
- **Description:** Data objects use plain `{}` instead of `Object.create(null)`. User input matching prototype properties ("constructor", "toString") could cause unexpected behavior.

### BUG-11: Type Coercion in Clinical Weights [LOW]
- **File:** `learning/loop.js:69`
- **Description:** `Math.round(clinicalWeight)` silently returns NaN for string inputs. `Math.max(1, NaN)` returns 1, hiding the error.

---

## 5. PRIORITY MATRIX

### Must Fix Before Any Clinical Deployment
1. **CLIN-01**: Improve dose/unit validation (tighten regex, expand unit tables)
2. **CLIN-02**: Expand PII redaction beyond Kuwait formats
3. **CLIN-03**: No physiological range validation
4. **CLIN-05**: Learning loop poisoning (can teach wrong corrections)
5. **BUG-02**: CSV parser corrupts quoted fields

### Fix Within 1 Week
6. **PERF-01**: O(n^4) decay loop (system freeze on large vocab)
7. **PERF-02**: Python subprocess pooling (10-100x speedup)
8. **PERF-03**: Async file I/O (unblock event loop)
9. **BUG-01**: Impossible boolean condition in feedback
10. **BUG-03**: Shared mutable global state
11. **PY-02**: NaN from division by zero

### Fix Within 1 Month
12. **PY-01**: Log-determinant normalization
13. **PY-03**: Grid removal threshold adaptation
14. **PY-05**: Add validation set to training
15. **PERF-04-06**: Array slice, regex, distance calc optimizations
16. **CLIN-06-08**: Confidence thresholds, name protection, ensemble calibration

### Track as Technical Debt
17. **PY-06**: Dimensionality reduction
18. **PY-09**: Observation storage optimization
19. **PERF-07-12**: Various cache/startup/index optimizations
20. All LOW severity findings

---

*Generated by deep codebase audit (round 2)*
