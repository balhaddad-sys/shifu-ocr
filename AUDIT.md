# Comprehensive Audit Report: Shifu OCR

**Date:** 2026-03-26
**Scope:** Full codebase audit (security, code quality, architecture)
**Codebase:** ~15,000 LOC across 45 files (JavaScript + Python)

---

## Executive Summary

Shifu OCR is a clinical OCR correction system with a well-motivated Python/JavaScript split and a clear 5-stage pipeline. The codebase demonstrates strong domain expertise but has significant gaps in security hardening, deployment infrastructure, and code maintainability. **19 security findings**, **12 code quality issues**, and **10 architectural concerns** were identified.

| Category | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| Security | 0 | 3 | 7 | 4 |
| Code Quality | 0 | 3 | 6 | 3 |
| Architecture | 0 | 1 | 4 | 5 |

---

## 1. SECURITY FINDINGS

### SEC-01: No Authentication or Authorization [HIGH]
- **File:** `server.js` (all endpoints)
- **Description:** The HTTP server has zero authentication. All 14 API endpoints are open to any caller. Medical data can be read/modified without credentials.
- **Impact:** Critical if exposed beyond localhost. Anyone on the network can access patient data.
- **Recommendation:** Add API key or token-based authentication. Consider OAuth2 for production.

### SEC-02: Unbounded Request Body Parsing [HIGH]
- **File:** `server.js:32-41`
- **Description:** `parseBody()` accumulates chunks with no size limit. No `Content-Length` validation.
  ```javascript
  let body = '';
  req.on('data', chunk => { body += chunk; }); // No max size
  ```
- **Impact:** Memory exhaustion DoS attack via large POST bodies.
- **Recommendation:** Enforce max body size (e.g., 10MB). Check `Content-Length` header.

### SEC-03: Path Traversal Risk in File Upload [HIGH]
- **File:** `server.js:195-197, 243-245`
- **Description:** Filename sanitization uses `path.basename()` + `startsWith()` check, but this is weak against symlink attacks or race conditions. Manual multipart parsing duplicates this logic.
- **Recommendation:** Use `path.resolve()` and validate with `path.relative()`. Replace manual multipart with `busboy` or `multer`.

### SEC-04: Missing Request Stream Error Handler [MEDIUM]
- **File:** `server.js:35, 185`
- **Description:** `req.on('error')` never registered. Client disconnects during upload propagate as unhandled errors.
- **Impact:** Server crash on client network errors.

### SEC-05: Unsafe Manual Multipart Parsing [MEDIUM]
- **File:** `server.js:202-234`
- **Description:** Hand-rolled multipart/form-data parser doesn't handle all RFC 2046 edge cases. No boundary length validation. Could be confused by crafted boundaries.
- **Recommendation:** Use a battle-tested multipart library.

### SEC-06: Missing Input Length Validation [MEDIUM]
- **File:** `server.js:65-87` (all POST endpoints)
- **Description:** No maximum length on text inputs to `/api/correct`, `/api/learn`, etc. Could cause excessive processing time.
- **Recommendation:** Add max length limits per endpoint.

### SEC-07: Unvalidated JSON Deserialization [MEDIUM]
- **Files:** `core/persistence.js:80,91`, `index.js:86-88`
- **Description:** `JSON.parse` on saved state files without schema validation. Corrupted or malicious state files could cause unexpected behavior.
- **Recommendation:** Add JSON schema validation for deserialized state.

### SEC-08: Pipeline Image Path Not Validated [MEDIUM]
- **Files:** `core/pipeline.js:244-247`, `shifu_ocr/pipeline_worker.py:59`
- **Description:** Image path from upload flows to Python subprocess which opens it directly. No validation that path is within allowed directory.
- **Recommendation:** Validate resolved image path is within upload directory.

### SEC-09: Predictable Temp Directory Names [MEDIUM]
- **File:** `core/ingest.js:395`
- **Description:** `shifu_pdf_` + `Date.now()` is predictable. No explicit permissions set.

### SEC-10: CORS Hardcoded to Localhost [MEDIUM]
- **File:** `server.js:28, 49`
- **Description:** `Access-Control-Allow-Origin: http://localhost:3737` is hardcoded.

### SEC-11: Error Messages Leak System Details [LOW]
- **File:** `server.js:251, 267`
- **Description:** `err.message` returned to client may include file paths and stack info.

### SEC-12: Missing Rate Limiting [LOW]
- **File:** `server.js` (all endpoints)
- **Description:** No rate limiting. Upload endpoint is especially vulnerable to abuse.

### SEC-13: Loose Python Dependency Versions [LOW]
- **File:** `requirements.txt`
- **Description:** `>=` constraints with no upper bounds. No lock file for reproducibility.

### SEC-14: Hardcoded Port [LOW]
- **File:** `server.js:9`
- **Description:** Port 3737 is hardcoded. `Number()` on env var could return `NaN`.

### Positive Security Findings
- No hardcoded secrets or credentials
- No `eval()`, `Function()`, or dynamic code execution
- Safe subprocess spawning (argument arrays, not shell strings)
- No SQL injection (no database)
- Proper HTML escaping via `esc()` function
- Zero npm dependencies (no supply-chain risk)

---

## 2. CODE QUALITY FINDINGS

### CQ-01: Bare Except Clauses in Python [HIGH]
- **Files:** `shifu_ocr/codefining.py:135`, `shifu_ocr/clinical.py:254`, `shifu_ocr/complete.py` (multiple)
- **Description:** Bare `except:` clauses swallow all exceptions including `KeyboardInterrupt` and `SystemExit`.
- **Impact:** Makes debugging extremely difficult. Critical failures masked silently.
- **Recommendation:** Use `except Exception:` or specific exception types.

### CQ-02: Unhandled Promise Rejections [HIGH]
- **Files:** `server.js:32-41`, `core/pipeline.js`, `core/ingest.js`
- **Description:** Promise chains lack proper error handling in request stream processing. No timeout handling for streaming requests.
- **Impact:** Request processing could hang or crash the server.

### CQ-03: Missing Stream Error Handlers [HIGH]
- **File:** `server.js:35, 185`
- **Description:** `req.on('error')` never registered. If client disconnects mid-upload, error propagates unhandled.

### CQ-04: Unclosed File Descriptor [MEDIUM]
- **File:** `core/ingest.js:40-42`
- **Description:** `fs.closeSync(fd)` outside `finally` block. If `readSync` throws, file descriptor leaks.
  ```javascript
  const fd = fs.openSync(filePath, 'r');
  fs.readSync(fd, head, 0, 8, 0);
  fs.closeSync(fd);  // Not in finally block
  ```

### CQ-05: Silent Error Swallowing [MEDIUM]
- **Files:** `server.js:38`, `server.js:253,269`, `core/ingest.js:58`
- **Description:** Empty catch blocks with no logging. JSON parse failures return `{}` silently. Cleanup errors ignored.
- **Impact:** Debugging nearly impossible when parsing fails.

### CQ-06: Race Condition in Learning Lock [MEDIUM]
- **File:** `learning/loop.js:430-432`
- **Description:** Simple boolean `_locked` flag doesn't prevent concurrent async operations. Non-atomic check-and-set pattern.
- **Impact:** Potential concurrent modifications to confusion profile, vocabulary, context chains.

### CQ-07: Inefficient Ring Buffers [MEDIUM]
- **File:** `learning/engine.js:165-186`
- **Description:** Uses `array.push()` + `array.slice(-200)` which creates a new array every push beyond limit. Multiple buffers (wp, ph, nv, sl, sn, sd) all do this.
- **Impact:** O(n) per operation instead of O(1). Memory churn under heavy load.
- **Recommendation:** Use fixed-size circular buffer with index pointer.

### CQ-08: Sync Process Spawn Edge Cases [MEDIUM]
- **File:** `core/pipeline.js:247`
- **Description:** `timeout` option in `spawn()` options is not standard Node.js API. May be silently ignored.

### CQ-09: Temporary File Cleanup [MEDIUM]
- **Files:** `core/ingest.js:431,253,269`, `server.js:253,269`
- **Description:** Cleanup in `finally` blocks uses empty catch. If process crashes, temp files persist in `/tmp`.

### CQ-10: Duplicate Sanitization Logic [LOW]
- **File:** `server.js:195-197, 243-245`
- **Description:** Filename sanitization code is duplicated between direct upload and multipart handlers.

### CQ-11: Non-Standard spawn() Timeout [LOW]
- **File:** `core/pipeline.js:247`
- **Description:** `{ timeout: 60000 }` in `spawn()` options. While this works in recent Node, behavior is implementation-dependent.

### CQ-12: Magic Numbers [LOW]
- **Files:** Multiple (e.g., `core/engine.js`, `learning/engine.js`)
- **Description:** Threshold values (0.3, 0.7, 0.97, 200) scattered through code without named constants.

---

## 3. ARCHITECTURAL FINDINGS

### ARCH-01: V1/V2 Code Duplication [HIGH]
- **Files:** `clinical/*.js` vs `v2/*.js`, `learning/loop.js` vs `v2/shifuLearningLoop.js`
- **Description:** ~80-90% functional overlap between root modules and v2/ directory. V2 is a Firebase-oriented ES6 module refactor. Both are partially maintained.

  | Module | V1 (root) | V2 | LOC Delta |
  |--------|-----------|-----|-----------|
  | Clinical Corrector | 654 LOC | 244 LOC | 410 |
  | Vocabulary | 533 LOC | 225 LOC | 308 |
  | Confusion Model | 95 LOC | 142 LOC | -47 |
  | Safety Flags | 138 LOC | 227 LOC | -89 |
  | Learning Loop | 871 LOC | 707 LOC | 164 |

- **Impact:** Bug fixes in one version don't sync to the other. Feature drift is inevitable.
- **Recommendation:** Decide: deprecate v2 or fully migrate. Don't maintain parallel implementations.

### ARCH-02: No Deployment Infrastructure [MEDIUM]
- **Description:** No Dockerfile, no docker-compose, no CI/CD pipeline (`.github/workflows/`), no systemd service file, no reverse proxy config.
- **Impact:** Manual deployment only. No automated testing on push. No reproducible builds.
- **Recommendation:** Add Docker + GitHub Actions at minimum.

### ARCH-03: Hardcoded Configuration [MEDIUM]
- **Files:** `server.js:9`, `core/engine.js` (CONFIG object)
- **Description:** Only 1 environment variable (`SHIFU_VOCAB_TARGET`). Port, CORS origin, state directory, model paths all hardcoded.
- **Recommendation:** Externalize to `.env` or config files with environment-specific overrides.

### ARCH-04: No HTTP Endpoint Tests [MEDIUM]
- **Files:** `test/suite.js`, `test/demo.js`
- **Description:** Test suite covers core engine, clinical corrector, safety flags, learning loop, and Python engines (100+ assertions). But zero HTTP endpoint tests. The complex multipart upload handler (server.js:200-260) is completely untested.
- **Recommendation:** Add endpoint tests using a test HTTP client.

### ARCH-05: Fragile Python-JS Bridge [MEDIUM]
- **File:** `core/pipeline.js`
- **Description:** Subprocess bridge with JSON over stdout. Hard path dependency, no protocol versioning. 60s timeout is arbitrary with no graceful degradation.
- **Recommendation:** Add protocol version negotiation. Consider socket-based communication for production.

### ARCH-06: Monolithic Server File [LOW]
- **File:** `server.js` (600+ LOC)
- **Description:** All 14 endpoints, HTML UI, multipart parsing, and CORS handling in one file.

### ARCH-07: Relative State Paths [LOW]
- **Files:** `server.js:15`, `core/persistence.js`
- **Description:** `.state`, `.ingested`, `.tmp` are relative paths. Break if working directory changes.

### ARCH-08: No Process Manager [LOW]
- **Description:** If Node.js process crashes, nothing restarts it. No health check endpoint for load balancers.

### ARCH-09: No Structured Logging [LOW]
- **Description:** Uses `console.log`/`console.warn` only. No JSON logging, log levels, or log rotation.

### ARCH-10: Missing API Documentation [LOW]
- **Description:** README covers architecture well but has no API endpoint documentation (schemas, examples, error codes).

---

## 4. WHAT'S DONE WELL

- **Clean Python/JS boundary**: Python handles vision/character recognition, JS handles linguistic reasoning and clinical domain. Well-motivated split.
- **Zero npm dependencies**: No supply-chain attack surface. All utilities written in-house.
- **Strong test coverage for core logic**: 100+ assertions covering engine, corrector, safety, learning.
- **Good README**: Clear architecture diagrams, data flow visualization, directory structure.
- **Safe subprocess spawning**: Uses argument arrays, not shell strings.
- **Proper HTML escaping**: `esc()` function prevents XSS.
- **Clinical safety awareness**: Lab range checks, medication ambiguity detection, dose plausibility.

---

## 5. PRIORITIZED RECOMMENDATIONS

### Immediate (Security)
1. Add request body size limits to `parseBody()` and upload handler
2. Add `req.on('error')` handlers in all stream consumers
3. Add authentication layer if server is exposed beyond localhost
4. Replace manual multipart parsing with a tested library
5. Validate file paths are within allowed directories using `path.resolve()`

### Short-term (Quality)
6. Replace bare `except:` with `except Exception:` in all Python files
7. Add `finally` blocks for file descriptor cleanup in `core/ingest.js`
8. Implement proper async locking in `learning/loop.js`
9. Add HTTP endpoint tests (especially for `/api/upload`)
10. Pin Python dependency versions and add lock file

### Medium-term (Architecture)
11. Resolve v1/v2 duplication (deprecate or migrate)
12. Externalize configuration to environment variables / config files
13. Add Dockerfile and CI/CD pipeline (GitHub Actions)
14. Add health check endpoint (`/health`)
15. Replace ring buffer `slice()` pattern with fixed-size circular buffers

### Long-term (Maturity)
16. Add structured JSON logging with levels
17. Add API documentation (OpenAPI/Swagger)
18. Implement rate limiting per endpoint
19. Consider gRPC or socket-based Python-JS bridge for production
20. Split `server.js` into router + handlers

---

*Generated by comprehensive codebase audit*
