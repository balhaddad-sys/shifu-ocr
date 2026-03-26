# Audit Report Round 3: Tests, Data Integrity, Dead Code, Documentation

**Date:** 2026-03-26
**Scope:** Test correctness, persistence/data integrity, dead code, documentation accuracy
**Complements:** AUDIT.md (security/quality/architecture), AUDIT_DEEP.md (performance/clinical/Python/logic)

---

## Executive Summary

This third audit pass uncovered **critical data integrity vulnerabilities** in the persistence layer, significant **dead code** bloating the codebase, and **documentation inaccuracies** that misrepresent the system's capabilities.

| Category | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| Data Integrity | 4 | 3 | 3 | 0 |
| Dead Code | 1 | 1 | 3 | 3 |
| Documentation | 1 | 0 | 0 | 3 |
| Test Gaps | 0 | 3 | 4 | 2 |
| **Total** | **6** | **7** | **10** | **8** |

---

## 1. DATA INTEGRITY & PERSISTENCE FINDINGS

### DI-01: Non-Atomic Multi-File Save [CRITICAL]
- **File:** `core/persistence.js:33-51`
- **Description:** Three separate `writeFileSync` calls for core, learning, and meta state. A crash between writes creates **split-brain state** where core and learning are from different points in time.
- **Scenario:** Process killed after core_engine.json written but before learning_engine.json → all session corrections lost on restart, but core resonance state is fresh.
- **Fix:** Write-to-temp-then-rename pattern:
  ```javascript
  fs.writeFileSync(tmpPath, data);
  fs.renameSync(tmpPath, finalPath); // Atomic on POSIX
  ```

### DI-02: Corrupt State Silently Discards ALL Learning [CRITICAL]
- **File:** `core/persistence.js:62-98`
- **Description:** If either core OR learning JSON is corrupt, `load()` returns `null` and the system silently starts fresh. Months of learned corrections permanently lost with only a `console.warn`.
- **Fix:** Back up corrupt files before discarding. Attempt partial recovery.

### DI-03: No File Locking — Multi-Process Corruption [CRITICAL]
- **File:** `core/persistence.js` (no locks)
- **Description:** No file-level or process-level locking. If server runs in cluster mode, two workers can simultaneously write state files, interleaving bytes and producing corrupt JSON.
- **Fix:** Use `proper-lockfile` or write to a database.

### DI-04: Undo Completely Broken After Restart [CRITICAL]
- **File:** `learning/loop.js:836-864`
- **Description:** During serialization, `ocrRow`, `confirmedRow`, `coreSnapshot`, and `rejected` are all stripped from history entries. After deserialize, `_history` contains only lightweight metadata — undo operations have no data to roll back to. Furthermore, `coreSnapshot` is **always** stripped (too large for disk), so even in-memory undo doesn't restore core engine vectors.
- **Impact:** Users who restart and try to undo recent corrections get silent no-ops.

### DI-05: Lossy Core Engine Serialization [HIGH]
- **File:** `core/engine.js:480-491`
- **Description:** Arrays truncated on every save: `wp` to 50, `eg` to 30, `ph` to 30, `nv` to 20, `_globalPos` to 200. Each save-load cycle permanently discards historical samples. Over time, this degrades statistical precision.
- **Impact:** Long-running systems lose positional distribution accuracy.

### DI-06: Serialize Races With Learn Mutations [HIGH]
- **File:** `learning/loop.js:476-487` vs `core/persistence.js`
- **Description:** The `_locked` boolean prevents concurrent `learn()` calls but doesn't protect `serialize()`. A save triggered during learn() reads partially-mutated state.

### DI-07: Save Failures Are Warnings-Only [HIGH]
- **File:** `core/persistence.js:52-54`
- **Description:** `catch (err) { console.warn(...) }` — application has no indication save failed. On disk-full or read-only filesystem, all subsequent corrections are in-memory only and lost on restart.

### DI-08: Metrics Events Grow Unbounded [MEDIUM]
- **File:** `core/metrics.js:60-82`
- **Description:** `this.data.events.push(entry)` with no limit. 1000 events/day × 365 days = 75MB+ JSON file that must be fully parsed on every startup.

### DI-09: Feedback Windows Grow Unbounded [MEDIUM]
- **File:** `core/feedback.js:387-397`
- **Description:** `m.windows.push({...})` appends every 50 evaluations with no pruning.

### DI-10: No Version Migration on Deserialize [MEDIUM]
- **Files:** `learning/loop.js:854`, `core/engine.js:492-503`
- **Description:** Version is saved (`"2.0.0"`) but never checked during deserialization. Code changes that alter serialization format will silently load incompatible state, causing undefined behavior.

---

## 2. DEAD CODE FINDINGS

### DC-01: Duplicate Engine File Never Imported [CRITICAL]
- **File:** `learning/engine.js` (522 LOC, v2.1.0)
- **Description:** This file is **never imported anywhere** in the codebase. The actual engine used is `core/engine.js` (v1.7.0/2.0.0). This is a complete orphan — 522 lines of dead code with a newer version number, suggesting an abandoned refactor.
- **Confidence:** 100% unused.
- **Fix:** Delete entirely.

### DC-02: Entire v2/ Directory Is Orphaned [HIGH]
- **Files:** `v2/*.js` (8 files, ~1,250 LOC)
- **Description:** No file outside `v2/` imports from it. `server.js` and `index.js` use root modules exclusively. The v2 smoke test and learning loop test are standalone scripts never run by `npm test`.
- **Confidence:** 95% unused (only accessed via `npm run test:v2`).
- **Fix:** Either integrate into main codebase or archive separately.

### DC-03: Unused Utility Exports [MEDIUM]
- **File:** `core/engine.js`
- **Description:** These exported functions are never imported elsewhere:
  - `mn` (mean), `sd` (standard deviation), `prng` (PRNG), `shuffle` (array shuffle), `topN` (object key limiter)
- **Confidence:** 100% unused outside engine.js.

### DC-04: Orphaned Python Module [MEDIUM]
- **File:** `shifu_ocr/learn_from_confusion.py`
- **Description:** Not imported by any other Python file. Not referenced in any script.
- **Confidence:** 100% unused.

### DC-05: Orphaned Training Scripts [MEDIUM]
- **Files:** `training/harvest.py`, `training/shield.py`
- **Description:** Not referenced in package.json scripts. Not imported by any module.
- **Confidence:** 80% unused (may be run manually).

### DC-06: Unused API Export [LOW]
- **File:** `index.js`
- **Description:** `generateWardSentences` is exported but never called outside tests.
- **Confidence:** 100% unused in production.

### DC-07: Standalone Test Scripts Never Run [LOW]
- **Files:** `learning/scale.js`, `learning/ablation.js`
- **Description:** Standalone scripts not referenced by test suite or package.json.

### DC-08: Coherence Engine Not Registered [LOW]
- **File:** `shifu_ocr/coherence.py`
- **Description:** File exists but is not registered in the ensemble (`ensemble.py` only registers topology, fluid, perturbation, theory_revision). Contains utility functions only.

---

## 3. DOCUMENTATION ACCURACY FINDINGS

### DOC-01: README Claims 6 Engines, Only 4 Are Active [WRONG]
- **File:** `README.md` (architecture table)
- **Claim:** 6 engines: Topology, Fluid, Perturbation, Theory-Revision, Co-Defining, Coherence
- **Reality:** `shifu_ocr/ensemble.py:326` registers only 4: `['topology', 'fluid', 'perturbation', 'theory_revision']`. Co-Defining and Coherence files exist but are NOT registered in the ensemble pipeline.
- **Impact:** Users expect 6-engine multi-lens analysis but get 4. Misleading about system capabilities.

### DOC-02: Architecture Lists Non-Existent File [OUTDATED]
- **File:** `README.md` (directory tree)
- **Claim:** `learning/suite.js` exists
- **Reality:** No such file. Tests are in `test/suite.js`.

### DOC-03: trained_model Listed as Directory [MINOR]
- **File:** `README.md`
- **Claim:** `shifu_ocr/trained_model/` (directory)
- **Reality:** `shifu_ocr/trained_model.json` (single 2.36MB JSON file)

### DOC-04: Core Engine Header Says v1.7.0 [MINOR]
- **File:** `core/engine.js:1`
- **Claim:** `// SHIFU CORE ENGINE v1.7.0`
- **Reality:** `const VERSION = "2.0.0"` and `package.json` says `"2.0.0"`

### Verified Accurate
- All 14 API endpoints documented in README exist in server.js
- All npm scripts (seed, train:prepare, train:finetune) point to real files
- Quick-start instructions (npm install, pip install, npm start) work
- Node version requirement (>=16.0.0) is correct
- MIT license attribution is accurate

---

## 4. TEST CORRECTNESS & COVERAGE GAPS

### TEST-01: Zero HTTP Endpoint Tests [HIGH]
- **Files:** `test/suite.js`, `test/demo.js`
- **Description:** No tests exercise the server.js endpoints. The complex multipart upload handler (lines 200-260), CORS handling, request body parsing, and error responses are completely untested.
- **Impact:** Server bugs can only be caught manually.

### TEST-02: No Negative/Failure Tests [HIGH]
- **File:** `test/suite.js`
- **Description:** Tests only verify correct behavior. No tests for:
  - Malformed JSON input
  - Empty strings, null values
  - Very long input (>100KB)
  - Invalid file types in upload
  - Corrupted state file loading
  - Python subprocess failures
- **Impact:** Failure modes unknown. Regressions in error handling undetected.

### TEST-03: No Persistence Roundtrip Tests [HIGH]
- **File:** `test/suite.js`
- **Description:** While there are basic serialize/deserialize tests, there are no tests that:
  - Save state, reload, and verify corrections still work identically
  - Simulate crash mid-save and verify recovery
  - Test version migration scenarios
  - Verify undo works after save/load cycle

### TEST-04: Tests Share Global State [MEDIUM]
- **File:** `test/suite.js`
- **Description:** Tests create a shared engine instance and run sequentially. Earlier tests modify the engine state (feeding sentences, learning corrections), which affects later test results. Test order matters.
- **Impact:** Moving or adding tests can cause unrelated failures.

### TEST-05: V2 Tests Minimal [MEDIUM]
- **Files:** `v2/smokeTest.js` (~150 LOC), `v2/learningLoopTest.js`
- **Description:** V2 tests are basic smoke tests that verify construction and simple operations. No equivalence testing with v1 modules.

### TEST-06: No Concurrency Tests [MEDIUM]
- **Description:** No tests for simultaneous learn() calls, concurrent save/load, or parallel request handling.

### TEST-07: Python Tests Don't Test Edge Cases [MEDIUM]
- **File:** `test/test_engines.py`
- **Description:** Tests use well-formed synthetic images. No tests for:
  - Empty/blank images
  - Single-pixel images
  - Very large images (OOM risk)
  - Corrupted image data

### TEST-08: Demo.js Not Automated [LOW]
- **File:** `test/demo.js`
- **Description:** Manual demo script with terminal output. No assertions — relies on human visual inspection.

### TEST-09: No Regression Tests for Bug Fixes [LOW]
- **Description:** Git history shows many bug fixes (e.g., "Fix split-brain on corrupt learning state", "Fix row-level corrected_verify decision") but no regression tests to prevent re-introduction.

---

## 5. PRIORITY MATRIX

### Must Fix (Data Loss Risk)
1. **DI-01**: Atomic file writes (prevent split-brain)
2. **DI-02**: Back up corrupt files before discarding
3. **DI-03**: Add file locking for concurrent access
4. **DI-04**: Document that undo doesn't survive restart, or fix it

### Should Fix (Code Health)
5. **DC-01**: Delete `learning/engine.js` (522 LOC dead code)
6. **DC-02**: Resolve v2/ directory (delete or integrate)
7. **DOC-01**: Fix README engine count (4, not 6)
8. **TEST-01**: Add HTTP endpoint tests
9. **TEST-02**: Add negative/failure case tests

### Nice to Have
10. **DI-05**: Document lossy serialization or increase truncation limits
11. **DI-08/09**: Add event pruning to metrics and feedback
12. **DC-03-05**: Remove unused exports and orphan files
13. **TEST-03**: Add persistence roundtrip tests

---

*Generated by third-pass codebase audit*
