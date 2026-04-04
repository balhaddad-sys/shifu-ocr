// Shifu Calibration System
// Saves calibration data for future learning sessions.
// Every correction, every domain switch, every accuracy measurement
// becomes a calibration point that shapes future behavior.
//
// Calibrations persist across sessions — Shifu remembers what it learned.

const fs = require('fs');
const path = require('path');

const CALIBRATION_DIR = path.join(__dirname, '..', '.calibrations');
const CALIBRATION_INDEX = 'calibration_index.json';

// ─── Calibration Point ──────────────────────────────────────────────────

class CalibrationPoint {
  constructor({ domain, subdomain, input, expected, predicted, confidence, correction, timestamp }) {
    this.id = CalibrationPoint._nextId();
    this.domain = domain;
    this.subdomain = subdomain || 'general';
    this.input = input;
    this.expected = expected;
    this.predicted = predicted;
    this.confidence = confidence || 0;
    this.correction = correction || null; // { from, to, weight }
    this.timestamp = timestamp || Date.now();
    this.applied = false; // Has this calibration been applied to the model?
  }

  static _id = 0;
  static _nextId() { return `cal_${Date.now()}_${++CalibrationPoint._id}`; }

  /** How wrong was this prediction? 0 = perfect, 1 = completely wrong */
  get error() {
    if (this.expected === this.predicted) return 0;
    if (!this.expected || !this.predicted) return 1;
    const maxLen = Math.max(this.expected.length, this.predicted.length);
    if (maxLen === 0) return 0;
    let diffs = 0;
    for (let i = 0; i < maxLen; i++) {
      if ((this.expected[i] || '') !== (this.predicted[i] || '')) diffs++;
    }
    return diffs / maxLen;
  }

  /** Is this a high-value calibration point? */
  get isHighValue() {
    return this.error > 0.1 && this.confidence > 0.3;
  }
}

// ─── Calibration Store ──────────────────────────────────────────────────

class CalibrationStore {
  constructor(storeDir = CALIBRATION_DIR) {
    this.storeDir = storeDir;
    this.points = [];
    this.domainStats = {};
    this.confusionLedger = {}; // Tracks char-level confusion across all domains
    this.transferLog = [];     // Records successful cross-domain transfers
    this._dirty = false;
    this._ensureDir();
  }

  _ensureDir() {
    if (!fs.existsSync(this.storeDir)) fs.mkdirSync(this.storeDir, { recursive: true });
  }

  /** Record a new calibration point */
  record(pointData) {
    const point = pointData instanceof CalibrationPoint
      ? pointData
      : new CalibrationPoint(pointData);

    this.points.push(point);
    this._updateDomainStats(point);
    this._updateConfusionLedger(point);
    this._dirty = true;
    return point;
  }

  /** Record a batch of calibration points */
  recordBatch(points) {
    return points.map(p => this.record(p));
  }

  /** Record a correction as a calibration */
  recordCorrection(domain, ocrText, correctedText, confidence, clinicalWeight = 1) {
    return this.record({
      domain,
      input: ocrText,
      expected: correctedText,
      predicted: ocrText,
      confidence,
      correction: {
        from: ocrText,
        to: correctedText,
        weight: clinicalWeight,
      },
    });
  }

  /** Record a cross-domain transfer event */
  recordTransfer(fromDomain, toDomain, input, result, success) {
    this.transferLog.push({
      from: fromDomain,
      to: toDomain,
      input: input.slice(0, 100),
      result: result.slice(0, 100),
      success,
      timestamp: Date.now(),
    });
    // Keep last 1000 transfers
    if (this.transferLog.length > 1000) {
      this.transferLog = this.transferLog.slice(-1000);
    }
    this._dirty = true;
  }

  _updateDomainStats(point) {
    const key = point.domain;
    if (!this.domainStats[key]) {
      this.domainStats[key] = {
        total: 0,
        correct: 0,
        errors: 0,
        totalError: 0,
        highValueCount: 0,
        lastCalibration: null,
        confusionPairs: {},
        subdomainAccuracy: {},
      };
    }
    const stats = this.domainStats[key];
    stats.total++;
    if (point.error === 0) stats.correct++;
    else stats.errors++;
    stats.totalError += point.error;
    if (point.isHighValue) stats.highValueCount++;
    stats.lastCalibration = point.timestamp;

    // Subdomain tracking
    if (!stats.subdomainAccuracy[point.subdomain]) {
      stats.subdomainAccuracy[point.subdomain] = { total: 0, correct: 0 };
    }
    stats.subdomainAccuracy[point.subdomain].total++;
    if (point.error === 0) stats.subdomainAccuracy[point.subdomain].correct++;
  }

  _updateConfusionLedger(point) {
    if (!point.correction) return;
    const { from, to } = point.correction;
    const minLen = Math.min(from.length, to.length);
    for (let i = 0; i < minLen; i++) {
      if (from[i] !== to[i]) {
        const pair = [from[i].toLowerCase(), to[i].toLowerCase()].sort().join(',');
        if (!this.confusionLedger[pair]) {
          this.confusionLedger[pair] = { count: 0, domains: {}, firstSeen: Date.now() };
        }
        this.confusionLedger[pair].count++;
        this.confusionLedger[pair].domains[point.domain] =
          (this.confusionLedger[pair].domains[point.domain] || 0) + 1;
      }
    }
  }

  /** Get accuracy for a domain */
  getDomainAccuracy(domainId) {
    const stats = this.domainStats[domainId];
    if (!stats || stats.total === 0) return null;
    return {
      accuracy: stats.correct / stats.total,
      averageError: stats.totalError / stats.total,
      total: stats.total,
      highValueRatio: stats.highValueCount / stats.total,
      subdomains: stats.subdomainAccuracy,
    };
  }

  /** Get all domain accuracies sorted */
  getAllAccuracies() {
    const result = {};
    for (const domainId of Object.keys(this.domainStats)) {
      result[domainId] = this.getDomainAccuracy(domainId);
    }
    return result;
  }

  /** Get unapplied calibration points for a domain */
  getUnapplied(domainId, limit = 100) {
    return this.points
      .filter(p => p.domain === domainId && !p.applied)
      .slice(-limit);
  }

  /** Mark calibration points as applied */
  markApplied(pointIds) {
    const idSet = new Set(pointIds);
    for (const point of this.points) {
      if (idSet.has(point.id)) point.applied = true;
    }
    this._dirty = true;
  }

  /** Get high-value calibration points across all domains */
  getHighValue(limit = 50) {
    return this.points
      .filter(p => p.isHighValue && !p.applied)
      .sort((a, b) => b.error - a.error)
      .slice(0, limit);
  }

  /** Get cross-domain confusion patterns */
  getCrossDomainConfusions(minCount = 3) {
    const result = [];
    for (const [pair, data] of Object.entries(this.confusionLedger)) {
      if (data.count >= minCount) {
        const domainCount = Object.keys(data.domains).length;
        result.push({
          pair,
          count: data.count,
          domains: data.domains,
          crossDomain: domainCount > 1,
          domainCount,
        });
      }
    }
    return result.sort((a, b) => b.count - a.count);
  }

  /** Get transfer learning effectiveness */
  getTransferEffectiveness() {
    const stats = {};
    for (const t of this.transferLog) {
      const key = `${t.from}->${t.to}`;
      if (!stats[key]) stats[key] = { total: 0, success: 0 };
      stats[key].total++;
      if (t.success) stats[key].success++;
    }
    const result = {};
    for (const [key, s] of Object.entries(stats)) {
      result[key] = { ...s, rate: s.total > 0 ? s.success / s.total : 0 };
    }
    return result;
  }

  /** Compact old calibration points, keeping summaries */
  compact(maxAge = 30 * 24 * 60 * 60 * 1000) { // 30 days default
    const cutoff = Date.now() - maxAge;
    const old = this.points.filter(p => p.timestamp < cutoff && p.applied);
    const kept = this.points.filter(p => p.timestamp >= cutoff || !p.applied);

    // Summarize old points into domain stats (already tracked)
    this.points = kept;
    this._dirty = true;
    return { removed: old.length, remaining: kept.length };
  }

  // ─── Persistence ────────────────────────────────────────────────────

  /** Save calibrations to disk */
  save() {
    if (!this._dirty) return;
    this._ensureDir();

    const writeAtomic = (fileName, content) => {
      const finalPath = path.join(this.storeDir, fileName);
      const tmpPath = finalPath + '.tmp';
      fs.writeFileSync(tmpPath, content);
      fs.renameSync(tmpPath, finalPath);
    };

    // Save calibration points in domain-specific files
    const byDomain = {};
    for (const point of this.points) {
      (byDomain[point.domain] || (byDomain[point.domain] = [])).push(point);
    }

    for (const [domain, points] of Object.entries(byDomain)) {
      writeAtomic(`calibrations_${domain}.json`, JSON.stringify(points));
    }

    // Save the index
    writeAtomic(CALIBRATION_INDEX, JSON.stringify({
      savedAt: new Date().toISOString(),
      totalPoints: this.points.length,
      domainStats: this.domainStats,
      confusionLedger: this.confusionLedger,
      transferLog: this.transferLog.slice(-500),
      domains: Object.keys(byDomain),
    }, null, 2));

    this._dirty = false;
  }

  /** Load calibrations from disk */
  load() {
    const indexPath = path.join(this.storeDir, CALIBRATION_INDEX);
    if (!fs.existsSync(indexPath)) return false;

    try {
      const index = JSON.parse(fs.readFileSync(indexPath, 'utf8'));
      this.domainStats = index.domainStats || {};
      this.confusionLedger = index.confusionLedger || {};
      this.transferLog = index.transferLog || [];

      // Load domain-specific calibration files
      this.points = [];
      for (const domain of (index.domains || [])) {
        const domainFile = path.join(this.storeDir, `calibrations_${domain}.json`);
        if (fs.existsSync(domainFile)) {
          try {
            const raw = JSON.parse(fs.readFileSync(domainFile, 'utf8'));
            for (const p of raw) {
              this.points.push(Object.assign(new CalibrationPoint(p), p));
            }
          } catch (e) {
            console.warn(`Failed to load calibrations for ${domain}: ${e.message}`);
          }
        }
      }
      return true;
    } catch (e) {
      console.warn(`Failed to load calibration index: ${e.message}`);
      return false;
    }
  }

  /** Get a summary report */
  summary() {
    const accuracies = this.getAllAccuracies();
    const crossDomain = this.getCrossDomainConfusions();
    const transfers = this.getTransferEffectiveness();

    return {
      totalCalibrations: this.points.length,
      unapplied: this.points.filter(p => !p.applied).length,
      domains: accuracies,
      topConfusions: crossDomain.slice(0, 10),
      transferEffectiveness: transfers,
    };
  }
}

module.exports = { CalibrationPoint, CalibrationStore };
