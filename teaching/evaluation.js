// Shifu Cross-Domain Evaluation System
// Measures teaching effectiveness across domains with comprehensive metrics.
//
// Evaluation dimensions:
// 1. Accuracy — character-level, word-level, sentence-level
// 2. Confusion Coverage — how well the model handles known confusions
// 3. Transfer Effectiveness — does knowledge transfer between domains?
// 4. Calibration Quality — are confidence scores well-calibrated?
// 5. Temporal Stability — does accuracy hold over time?

// ─── Metric Calculators ─────────────────────────────────────────────────

/** Character Error Rate (CER) */
function charErrorRate(predicted, expected) {
  if (!expected || expected.length === 0) return predicted ? 1 : 0;
  const dp = Array.from({ length: predicted.length + 1 }, () =>
    new Array(expected.length + 1).fill(0));
  for (let i = 0; i <= predicted.length; i++) dp[i][0] = i;
  for (let j = 0; j <= expected.length; j++) dp[0][j] = j;
  for (let i = 1; i <= predicted.length; i++) {
    for (let j = 1; j <= expected.length; j++) {
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + (predicted[i - 1] === expected[j - 1] ? 0 : 1)
      );
    }
  }
  return dp[predicted.length][expected.length] / expected.length;
}

/** Word Error Rate (WER) */
function wordErrorRate(predicted, expected) {
  const predWords = predicted.trim().split(/\s+/);
  const expWords = expected.trim().split(/\s+/);
  if (expWords.length === 0) return predWords.length > 0 ? 1 : 0;

  const dp = Array.from({ length: predWords.length + 1 }, () =>
    new Array(expWords.length + 1).fill(0));
  for (let i = 0; i <= predWords.length; i++) dp[i][0] = i;
  for (let j = 0; j <= expWords.length; j++) dp[0][j] = j;
  for (let i = 1; i <= predWords.length; i++) {
    for (let j = 1; j <= expWords.length; j++) {
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + (predWords[i - 1] === expWords[j - 1] ? 0 : 1)
      );
    }
  }
  return dp[predWords.length][expWords.length] / expWords.length;
}

/** Expected Calibration Error (ECE) — measures confidence calibration */
function expectedCalibrationError(predictions, numBins = 10) {
  if (predictions.length === 0) return 0;
  const bins = Array.from({ length: numBins }, () => ({ correct: 0, total: 0, confSum: 0 }));

  for (const { confidence, correct } of predictions) {
    const binIdx = Math.min(Math.floor(confidence * numBins), numBins - 1);
    bins[binIdx].total++;
    bins[binIdx].confSum += confidence;
    if (correct) bins[binIdx].correct++;
  }

  let ece = 0;
  for (const bin of bins) {
    if (bin.total === 0) continue;
    const accuracy = bin.correct / bin.total;
    const avgConf = bin.confSum / bin.total;
    ece += (bin.total / predictions.length) * Math.abs(accuracy - avgConf);
  }
  return ece;
}

/** F1 score for confusion pair detection */
function confusionF1(predictions, confusionPairs) {
  let tp = 0, fp = 0, fn = 0;
  for (const pred of predictions) {
    const pair = [pred.predicted, pred.expected].sort().join(',');
    if (confusionPairs.has(pair)) {
      if (pred.predicted !== pred.expected) tp++;
      else fn++;
    } else {
      if (pred.predicted !== pred.expected) fp++;
    }
  }
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  return precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
}

// ─── Evaluation Run ─────────────────────────────────────────────────────

class EvaluationRun {
  constructor(domainId, phase) {
    this.id = `eval_${Date.now()}`;
    this.domainId = domainId;
    this.phase = phase || 'all';
    this.startedAt = Date.now();
    this.endedAt = null;
    this.predictions = [];
    this.metrics = null;
  }

  /** Add a prediction result */
  addPrediction(input, predicted, expected, confidence) {
    this.predictions.push({
      input,
      predicted,
      expected,
      confidence: confidence || 0,
      correct: predicted === expected,
      cer: charErrorRate(predicted, expected),
      wer: wordErrorRate(predicted, expected),
    });
  }

  /** Compute all metrics */
  compute(confusionPairs) {
    const n = this.predictions.length;
    if (n === 0) {
      this.metrics = { accuracy: 0, cer: 0, wer: 0, ece: 0, f1: 0, n: 0 };
      return this.metrics;
    }

    const accuracy = this.predictions.filter(p => p.correct).length / n;
    const avgCer = this.predictions.reduce((s, p) => s + p.cer, 0) / n;
    const avgWer = this.predictions.reduce((s, p) => s + p.wer, 0) / n;
    const ece = expectedCalibrationError(this.predictions);
    const f1 = confusionPairs ? confusionF1(this.predictions, confusionPairs) : null;

    // Confidence distribution
    const confBuckets = { low: 0, medium: 0, high: 0 };
    for (const p of this.predictions) {
      if (p.confidence < 0.3) confBuckets.low++;
      else if (p.confidence < 0.7) confBuckets.medium++;
      else confBuckets.high++;
    }

    // Error analysis — most common errors
    const errorCounts = {};
    for (const p of this.predictions) {
      if (!p.correct) {
        const minLen = Math.min(p.predicted.length, p.expected.length);
        for (let i = 0; i < minLen; i++) {
          if (p.predicted[i] !== p.expected[i]) {
            const pair = [p.predicted[i], p.expected[i]].sort().join(',');
            errorCounts[pair] = (errorCounts[pair] || 0) + 1;
          }
        }
      }
    }
    const topErrors = Object.entries(errorCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([pair, count]) => ({ pair, count }));

    this.metrics = {
      n,
      accuracy,
      cer: avgCer,
      wer: avgWer,
      ece,
      f1,
      confidenceDistribution: confBuckets,
      topErrors,
    };

    this.endedAt = Date.now();
    return this.metrics;
  }
}

// ─── Cross-Domain Evaluator ─────────────────────────────────────────────

class CrossDomainEvaluator {
  constructor(calibrationStore) {
    this.calibrations = calibrationStore;
    this.runs = [];
    this.domainBaselines = {}; // Historical baselines per domain
  }

  /** Run evaluation for a single domain using calibration data */
  evaluateDomain(domainId) {
    const run = new EvaluationRun(domainId);

    // Use calibration points as test data
    const points = this.calibrations.points
      .filter(p => p.domain === domainId && p.expected);

    for (const point of points) {
      run.addPrediction(
        point.input,
        point.predicted,
        point.expected,
        point.confidence
      );
    }

    // Get domain confusion pairs
    const domain = require('./domains').DOMAINS[domainId];
    const confusionPairs = domain
      ? new Set(Object.keys(domain.confusionProfile))
      : new Set();

    run.compute(confusionPairs);
    this.runs.push(run);

    // Update baseline
    if (run.metrics && run.metrics.n > 0) {
      if (!this.domainBaselines[domainId]) {
        this.domainBaselines[domainId] = [];
      }
      this.domainBaselines[domainId].push({
        timestamp: Date.now(),
        accuracy: run.metrics.accuracy,
        cer: run.metrics.cer,
        n: run.metrics.n,
      });
      // Keep last 50 baselines
      if (this.domainBaselines[domainId].length > 50) {
        this.domainBaselines[domainId] = this.domainBaselines[domainId].slice(-50);
      }
    }

    return run;
  }

  /** Run evaluation across all domains */
  evaluateAll() {
    const domains = Object.keys(this.calibrations.domainStats);
    const results = {};
    for (const domainId of domains) {
      results[domainId] = this.evaluateDomain(domainId);
    }
    return results;
  }

  /** Evaluate cross-domain transfer effectiveness */
  evaluateTransfer() {
    const transfers = this.calibrations.getTransferEffectiveness();
    const crossConfusions = this.calibrations.getCrossDomainConfusions(2);

    return {
      transferPaths: transfers,
      universalConfusions: crossConfusions.filter(c => c.crossDomain),
      domainSpecificConfusions: crossConfusions.filter(c => !c.crossDomain),
    };
  }

  /** Check temporal stability — is accuracy improving, stable, or degrading? */
  temporalStability(domainId) {
    const baselines = this.domainBaselines[domainId];
    if (!baselines || baselines.length < 3) return { trend: 'insufficient_data', data: baselines };

    const recent = baselines.slice(-5);
    const older = baselines.slice(-10, -5);

    if (older.length === 0) return { trend: 'insufficient_data', data: recent };

    const recentAccuracy = recent.reduce((s, b) => s + b.accuracy, 0) / recent.length;
    const olderAccuracy = older.reduce((s, b) => s + b.accuracy, 0) / older.length;
    const delta = recentAccuracy - olderAccuracy;

    let trend;
    if (delta > 0.05) trend = 'improving';
    else if (delta < -0.05) trend = 'degrading';
    else trend = 'stable';

    return {
      trend,
      recentAccuracy,
      olderAccuracy,
      delta,
      dataPoints: baselines.length,
    };
  }

  /** Generate a comprehensive evaluation report */
  generateReport() {
    const allEvals = this.evaluateAll();
    const transferEval = this.evaluateTransfer();

    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        domainsEvaluated: Object.keys(allEvals).length,
        totalPredictions: 0,
        overallAccuracy: 0,
        overallCER: 0,
      },
      domains: {},
      transfer: transferEval,
      recommendations: [],
    };

    let totalPredictions = 0;
    let weightedAccuracy = 0;
    let weightedCER = 0;

    for (const [domainId, run] of Object.entries(allEvals)) {
      if (!run.metrics || run.metrics.n === 0) continue;

      report.domains[domainId] = {
        ...run.metrics,
        stability: this.temporalStability(domainId),
      };

      totalPredictions += run.metrics.n;
      weightedAccuracy += run.metrics.accuracy * run.metrics.n;
      weightedCER += run.metrics.cer * run.metrics.n;

      // Generate domain-specific recommendations
      if (run.metrics.accuracy < 0.6) {
        report.recommendations.push({
          domain: domainId,
          severity: 'high',
          message: `${domainId} accuracy is ${(run.metrics.accuracy * 100).toFixed(1)}% — needs significant teaching`,
        });
      }
      if (run.metrics.ece > 0.15) {
        report.recommendations.push({
          domain: domainId,
          severity: 'medium',
          message: `${domainId} confidence is poorly calibrated (ECE: ${run.metrics.ece.toFixed(3)})`,
        });
      }
      if (run.metrics.topErrors.length > 0 && run.metrics.topErrors[0].count > 5) {
        const topErr = run.metrics.topErrors[0];
        report.recommendations.push({
          domain: domainId,
          severity: 'medium',
          message: `Most common confusion in ${domainId}: ${topErr.pair} (${topErr.count} occurrences)`,
        });
      }
    }

    report.summary.totalPredictions = totalPredictions;
    report.summary.overallAccuracy = totalPredictions > 0 ? weightedAccuracy / totalPredictions : 0;
    report.summary.overallCER = totalPredictions > 0 ? weightedCER / totalPredictions : 0;

    return report;
  }

  /** Serialize evaluator state */
  serialize() {
    return {
      domainBaselines: this.domainBaselines,
      runCount: this.runs.length,
    };
  }

  /** Restore evaluator state */
  restore(state) {
    if (state.domainBaselines) this.domainBaselines = state.domainBaselines;
  }
}

module.exports = {
  charErrorRate,
  wordErrorRate,
  expectedCalibrationError,
  confusionF1,
  EvaluationRun,
  CrossDomainEvaluator,
};
