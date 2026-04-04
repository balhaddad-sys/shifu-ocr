// SHIFU v2 TEACHER — Closes the feedback loop.
// The engine tells you where it's pulling. The teacher fills the voids.
// Not exposure (here's a million sentences). Instruction (your vacuums say
// you're thin on pharmacology — here's targeted text, now verify).

class Teacher {
  constructor(engine) {
    this.engine = engine;
    this.log = [];       // lesson history
    this.corrections = 0;
    this.lessons = 0;
  }

  // ─── Diagnose: what does the engine need? ─────────────────────────
  // Reads the pressure map and returns a structured diagnosis.
  diagnose() {
    const vacs = this.engine.vacuums(10);
    const surps = this.engine.surpluses(10);
    const brs = this.engine.bridges(10);
    const st = this.engine.stats();

    // Find shallow words that appear often but lack structure
    const underexposed = [];
    for (const [w, n] of Object.entries(this.engine.nodes)) {
      const d = this.engine.depth(w);
      if (n.freq >= 3 && d.level === "shallow") underexposed.push({ word: w, freq: n.freq, evidence: d.evidence });
    }
    underexposed.sort((a, b) => b.freq - a.freq);

    // Find words with high inbound expectation but low actual structure
    const starved = [];
    for (const [w, n] of Object.entries(this.engine.nodes)) {
      let inbound = 0;
      for (const [, other] of Object.entries(this.engine.nodes)) {
        if (other.next[w]) inbound += other.next[w];
      }
      const actual = Object.keys(n.neighbors).length;
      if (inbound > actual && inbound >= 3) starved.push({ word: w, inbound, actual, deficit: inbound - actual });
    }
    starved.sort((a, b) => b.deficit - a.deficit);

    return {
      vocab: st.vocab,
      depths: st.depths,
      vacuums: vacs,
      surpluses: surps,
      bridges: brs,
      underexposed: underexposed.slice(0, 10),
      starved: starved.slice(0, 10),
      // Plain English: what to do next
      needs: this._prescribe(vacs, underexposed, starved, brs, st),
    };
  }

  _prescribe(vacs, underexposed, starved, bridges, st) {
    const needs = [];
    if (st.vocab === 0) { needs.push("Empty engine. Feed any text to begin."); return needs; }
    if (st.depths.deep < 3) needs.push("Very few deep words. Feed more repeated sentences — the engine needs reinforcement, not breadth.");
    if (underexposed.length > 5) needs.push(`${underexposed.length} words seen 3+ times but still shallow. Feed sentences containing: ${underexposed.slice(0, 5).map(w => w.word).join(", ")}.`);
    if (starved.length > 0) needs.push(`${starved.length} words predicted by others but structurally thin. Feed contexts for: ${starved.slice(0, 5).map(w => w.word).join(", ")}.`);
    if (bridges.length > 0) needs.push(`${bridges.length} bridge words connecting disconnected neighborhoods. Consider whether these are real bridges or cross-contamination: ${bridges.slice(0, 3).map(b => b.word).join(", ")}.`);
    if (needs.length === 0) needs.push("Engine looks balanced. Feed new domain text to expand, or run contrastive drills to sharpen.");
    return needs;
  }

  // ─── Lesson: directed feed + immediate verification ───────────────
  // Not just feed(). Feed, then score, then report what changed.
  lesson(text, domain = "general") {
    const before = this.engine.stats();
    const beforePressure = this.engine.pressure();

    // Feed the text
    const fed = this.engine.feedText(text);

    // Decay noise immediately — single-occurrence edges from this feed
    // are probably noise unless reinforced later
    // (Don't decay on first lesson — need some noise to start)
    const decayed = before.vocab > 50 ? this.engine.decay(1) : { edgesRemoved: 0 };

    const after = this.engine.stats();
    const afterPressure = this.engine.pressure();

    // What changed?
    const newWords = after.vocab - before.vocab;
    const newDepths = {};
    for (const level of Object.keys(after.depths)) {
      newDepths[level] = after.depths[level] - before.depths[level];
    }

    // Which vacuums got filled?
    const beforeVacWords = new Set(beforePressure.filter(p => p.pressure < 0).map(p => p.word));
    const afterVacWords = new Set(afterPressure.filter(p => p.pressure < 0).map(p => p.word));
    const filledVacuums = [...beforeVacWords].filter(w => !afterVacWords.has(w));
    const newVacuums = [...afterVacWords].filter(w => !beforeVacWords.has(w));

    this.lessons++;
    const result = {
      lesson: this.lessons,
      domain,
      fed,
      newWords,
      newDepths,
      decayed: decayed.edgesRemoved,
      filledVacuums,
      newVacuums,
    };
    this.log.push(result);
    return result;
  }

  // ─── Drill: contrastive pair scoring ──────────────────────────────
  // Feed a sentence, then score it AND its reverse. The asymmetry is the lesson.
  drill(sentence) {
    // Feed it first (if not already known)
    const tokens = this.engine.feed(sentence);

    // Score the original
    const forward = this.engine.scoreSentence(sentence);

    // Reverse the content words (skip function words under 3 chars)
    const words = sentence.toLowerCase().match(/[a-z0-9]+/g) || [];
    const reversed = [...words].reverse().join(" ");
    const backward = this.engine.scoreSentence(reversed);

    const asymmetry = forward.coherence - backward.coherence;

    return {
      sentence,
      reversed,
      forwardCoherence: forward.coherence,
      backwardCoherence: backward.coherence,
      asymmetry,
      learned: asymmetry > 0.05,
      forwardSteps: forward.steps,
      backwardSteps: backward.steps,
    };
  }

  // ─── Correct: told this is wrong, fix it ──────────────────────────
  // "seizure" should NOT be associated with "skillet".
  // Unlearn the wrong pair, feed the right context, verify.
  correctAssociation(wordA, wordB, rightContext) {
    // Measure before
    const beforeAff = this.engine.affinity(wordA, wordB);

    // Unlearn the wrong connection
    const ul = this.engine.unlearn(wordA, wordB);

    // If right context provided, feed it to build correct associations
    let fed = null;
    if (rightContext) fed = this.engine.feedText(rightContext);

    // Measure after
    const afterAff = this.engine.affinity(wordA, wordB);

    this.corrections++;
    return {
      correction: this.corrections,
      wordA,
      wordB,
      before: beforeAff.mutual,
      after: afterAff.mutual,
      weakened: afterAff.mutual < beforeAff.mutual,
      unlearned: ul,
      fed,
    };
  }

  // ─── Plan: what to teach next ─────────────────────────────────────
  // Returns a concrete plan based on the engine's current state.
  plan() {
    const diag = this.diagnose();
    const steps = [];

    // Priority 1: fill starved words (high inbound expectation, low structure)
    if (diag.starved.length > 0) {
      const targets = diag.starved.slice(0, 3).map(s => s.word);
      steps.push({
        priority: 1,
        action: "feed_targeted",
        targets,
        instruction: `Feed sentences containing these words in varied contexts: ${targets.join(", ")}`,
      });
    }

    // Priority 2: reinforce underexposed words
    if (diag.underexposed.length > 0) {
      const targets = diag.underexposed.slice(0, 5).map(u => u.word);
      steps.push({
        priority: 2,
        action: "feed_reinforcement",
        targets,
        instruction: `These words appear often but lack depth. Feed more varied sentences containing: ${targets.join(", ")}`,
      });
    }

    // Priority 3: investigate bridges (are they real or contamination?)
    if (diag.bridges.length > 0) {
      const targets = diag.bridges.slice(0, 3).map(b => b.word);
      steps.push({
        priority: 3,
        action: "investigate_bridges",
        targets,
        instruction: `These words bridge disconnected neighborhoods. Check: are they legitimate cross-domain words (like "treatment") or contamination? If contamination, use unlearn().`,
      });
    }

    // Priority 4: contrastive drills for structured words
    const structured = Object.entries(this.engine.nodes)
      .filter(([w]) => this.engine.depth(w).level === "structured" || this.engine.depth(w).level === "deep")
      .map(([w]) => w).slice(0, 5);
    if (structured.length >= 2) {
      steps.push({
        priority: 4,
        action: "drill_contrastive",
        targets: structured,
        instruction: `Run contrastive drills: score "${structured[0]} ... ${structured[1]}" vs reversed. The asymmetry sharpens direction.`,
      });
    }

    // Priority 5: decay if graph is large enough
    if (diag.vocab > 100) {
      steps.push({
        priority: 5,
        action: "decay",
        instruction: "Run decay(1) to prune single-occurrence noise, then re-diagnose.",
      });
    }

    return { diagnosis: diag, steps };
  }

  // ─── Cycle: one full teaching iteration ───────────────────────────
  // diagnose → lesson → decay → drill → verify
  cycle(text, domain = "general") {
    const before = this.diagnose();
    const lessonResult = this.lesson(text, domain);
    
    // Pick a sentence from the text for contrastive drill
    const sents = text.split(/[.!?\n]+/).map(s => s.trim()).filter(s => s.length > 10);
    let drillResult = null;
    if (sents.length > 0) {
      drillResult = this.drill(sents[0]);
    }

    const after = this.diagnose();

    return {
      before: { vocab: before.vocab, depths: before.depths, needs: before.needs },
      lesson: lessonResult,
      drill: drillResult,
      after: { vocab: after.vocab, depths: after.depths, needs: after.needs },
    };
  }

  // ─── History ────────────────────────────────────────────────────
  history() {
    return {
      lessons: this.lessons,
      corrections: this.corrections,
      log: this.log.slice(-20), // last 20 lessons
    };
  }
}

module.exports = { Teacher };
