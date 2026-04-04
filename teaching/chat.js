// Shifu Chat Engine — Importable Module
// No LLM. No transformer. No hardcoded stopwords or templates.
// Everything derived from the engine's learned co-occurrence + wave propagation.

const fs = require('fs');
const path = require('path');

class KnowledgeBase {
  constructor() { this.sentences = []; this.index = {}; }

  add(sentence, domain) {
    const idx = this.sentences.length;
    const words = new Set((sentence.toLowerCase().match(/[a-z0-9]+/g) || []).filter(w => w.length > 1));
    this.sentences.push({ text: sentence, domain, words });
    for (const w of words) (this.index[w] || (this.index[w] = [])).push(idx);
  }

  retrieve(queryWords, waveField, engine, limit = 5) {
    const vocabSize = Object.keys(engine.wf).length || 1;
    const candidates = new Set();
    for (const w of queryWords) for (const idx of (this.index[w] || [])) candidates.add(idx);
    if (waveField) {
      for (const [w] of [...waveField.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12)) {
        for (const idx of (this.index[w] || []).slice(0, 8)) candidates.add(idx);
      }
    }
    const results = [];
    for (const idx of candidates) {
      const sent = this.sentences[idx];
      let score = 0;
      for (const w of queryWords) {
        if (sent.words.has(w)) {
          const freq = engine.wf[w] || 1;
          score += Math.max(Math.log(vocabSize / freq), 0.1) * 2;
        }
      }
      if (waveField) {
        for (const w of sent.words) {
          const energy = waveField.get(w) || 0;
          if (energy > 0.005) score += energy * 2;
        }
      }
      if (score > 0) results.push({ text: sent.text, domain: sent.domain, score });
    }
    return results.sort((a, b) => b.score - a.score).slice(0, limit);
  }

  get size() { return this.sentences.length; }
}

class ChatMemory {
  constructor() { this.topicField = new Map(); this.turns = []; }

  update(field) {
    for (const [w, e] of this.topicField) {
      this.topicField.set(w, e * 0.5);
      if (e < 0.003) this.topicField.delete(w);
    }
    for (const [w, e] of field) {
      this.topicField.set(w, (this.topicField.get(w) || 0) + e * 0.5);
    }
  }

  addTurn(role, text) {
    this.turns.push({ role, text, time: Date.now() });
    if (this.turns.length > 30) this.turns = this.turns.slice(-30);
  }

  getTopicWords(engine, n = 8) {
    return [...this.topicField.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, n)
      .map(([w]) => w)
      .filter(w => _isContentWord(engine, w));
  }
}

// ─── Derived from data (not hardcoded) ──────────────────────────────

function _isContentWord(engine, word) {
  const freq = engine.wf[word];
  if (!freq) return false;
  const medianFreq = engine._medianFreq || 1;
  if (freq > medianFreq * 10) {
    const coBreadth = engine.co[word] ? Object.keys(engine.co[word]).length : 0;
    const vocabSize = Object.keys(engine.wf).length;
    if (coBreadth > Math.sqrt(vocabSize) * 2) return false;
  }
  return true;
}

function _extractConcepts(engine, words) {
  const vocabSize = Object.keys(engine.wf).length || 1;
  const concepts = [];
  for (const w of words) {
    if (!engine.wf[w] || !_isContentWord(engine, w)) continue;
    const freq = engine.wf[w];
    const importance = Math.log(vocabSize / freq);
    const coWords = engine.co[w]
      ? Object.entries(engine.co[w]).sort((a, b) => b[1] - a[1])
          .filter(([cw]) => _isContentWord(engine, cw)).slice(0, 8).map(([cw]) => cw)
      : [];
    const nextWords = engine.nx[w]
      ? Object.entries(engine.nx[w]).sort((a, b) => b[1] - a[1])
          .filter(([nw]) => _isContentWord(engine, nw)).slice(0, 5).map(([nw]) => nw)
      : [];
    const resWords = engine.res[w]
      ? Object.entries(engine.res[w]).sort((a, b) => b[1] - a[1])
          .filter(([rw]) => _isContentWord(engine, rw)).slice(0, 4).map(([rw]) => rw)
      : [];
    concepts.push({ word: w, freq, importance, coWords, nextWords, resWords });
  }
  return concepts.sort((a, b) => b.importance - a.importance);
}

// ─── Chat Engine ────────────────────────────────────────────────────

class ShifuChat {
  constructor(engine) {
    this.engine = engine;
    this.kb = new KnowledgeBase();
    this.memory = new ChatMemory();
    this._loaded = false;
  }

  /** Feed a sentence into both the engine and knowledge base */
  learn(sentence, domain) {
    this.engine.feed(sentence);
    this.kb.add(sentence, domain || 'general');
  }

  /** Bulk load from the project's existing data */
  loadProjectData() {
    if (this._loaded) return;
    let count = 0;

    // Load medical corpus (already in engine if server pre-seeded, just add to KB)
    try {
      const med = require('../learning/medical_corpus');
      for (const s of med.slice(0, 150)) { this.kb.add(s, 'medical'); count++; }
    } catch (e) {}

    try {
      const { MEDICAL_CORPUS } = require('../learning/corpus');
      for (const s of MEDICAL_CORPUS.slice(0, 150)) { this.kb.add(s, 'medical'); count++; }
    } catch (e) {}

    // Load language corpus (KB only — engine already has it if server pre-seeded)
    try {
      const raw = fs.readFileSync(path.join(__dirname, '..', 'corpus_data', 'shifu_language_corpus.txt'), 'utf8');
      const lines = raw.split('\n').filter(l => l.trim().length > 15).slice(0, 300);
      for (const s of lines) { this.kb.add(s, 'general'); count++; }
    } catch (e) {}

    // Load seed output files
    const seedDir = path.join(__dirname, '..', 'seeds_out');
    if (fs.existsSync(seedDir)) {
      try {
        const seedFiles = fs.readdirSync(seedDir).filter(f => f.endsWith('_seed.json'));
        for (const file of seedFiles) {
          try {
            const data = JSON.parse(fs.readFileSync(path.join(seedDir, file), 'utf8'));
            for (const s of (data.sentences || [])) { this.learn(s, data.domain || 'general'); count++; }
          } catch (e) {}
        }
      } catch (e) {}
    }

    this._loaded = true;
    return count;
  }

  /**
   * Use the engine's OCR correction to fix misspelled words.
   * Uses _candidates (length index + bigram index) for fast lookup,
   * then picks the best match by character distance + frequency.
   */
  _correctInput(words) {
    const corrected = [];
    const corrections = [];

    for (const w of words) {
      // If the engine already knows this word, keep it
      if (this.engine.wf[w]) {
        corrected.push(w);
        continue;
      }

      // Skip very short words (not worth correcting)
      if (w.length < 3) {
        corrected.push(w);
        continue;
      }

      // Fast candidate lookup via length + bigram index (radius 3 for longer words)
      const radius = w.length > 5 ? 3 : 2;
      let candidates = this.engine._candidates(w, radius);

      // Also try with consecutive duplicates collapsed (morpphine -> morphine)
      const collapsed = w.replace(/(.)\1+/g, '$1');
      if (collapsed !== w) {
        // If collapsed form is in vocabulary directly, use it
        if (this.engine.wf[collapsed]) {
          corrected.push(collapsed);
          corrections.push({ from: w, to: collapsed });
          continue;
        }
        const extraCands = this.engine._candidates(collapsed, radius);
        for (const c of extraCands) if (!candidates.includes(c)) candidates.push(c);
      }

      if (candidates.length === 0) {
        corrected.push(w);
        continue;
      }

      // Score candidates by character distance + frequency
      let bestMatch = null;
      let bestScore = -Infinity;

      for (const cand of candidates) {
        // Character-level distance
        let dist = 0;
        const maxLen = Math.max(w.length, cand.length);
        const minLen = Math.min(w.length, cand.length);
        for (let i = 0; i < minLen; i++) {
          if (w[i] !== cand[i]) dist++;
        }
        dist += maxLen - minLen;

        // Max 2 character differences for short words, 3 for longer
        const maxDist = w.length <= 5 ? 2 : 3;
        if (dist > maxDist) continue;

        // Score: lower distance is better, higher frequency is better
        const freq = this.engine.wf[cand] || 0;
        const score = (1 - dist / maxLen) * 10 + Math.log(freq + 1);

        if (score > bestScore) {
          bestScore = score;
          bestMatch = cand;
        }
      }

      if (bestMatch) {
        corrected.push(bestMatch);
        corrections.push({ from: w, to: bestMatch });
      } else {
        corrected.push(w);
      }
    }

    return { corrected, corrections };
  }

  /** Respond to a user message */
  respond(input) {
    const rawWords = (input.toLowerCase().match(/[a-z0-9]+/g) || []).filter(w => w.length > 1);

    // Step 1: OCR-correct the input using the engine's learned vocabulary
    const { corrected: allWords, corrections } = this._correctInput(rawWords);

    const concepts = _extractConcepts(this.engine, allWords);
    const queryWords = concepts.length > 0 ? concepts.map(c => c.word) : allWords;

    // Wave propagation
    const field = new Map();
    for (const w of queryWords) {
      if (!this.engine.wf[w]) continue;
      const act = this.engine.activate(w);
      for (const [t, e] of act) field.set(t, (field.get(t) || 0) + e);
    }
    for (const [w, e] of this.memory.topicField) field.set(w, (field.get(w) || 0) + e * 0.3);

    // Retrieve
    const retrieved = this.kb.retrieve(queryWords, field, this.engine, 8);

    // Build response
    const parts = [];
    if (concepts.length === 0) {
      const anyKnown = allWords.some(w => this.engine.wf[w]);
      parts.push(anyKnown
        ? 'Those are common words in my vocabulary, but I need more specific terms.'
        : "Those words aren't in my vocabulary yet. Feed me text containing them.");
    } else {
      const main = concepts[0];
      parts.push(`"${main.word}" — seen ${main.freq} time(s).`);
      if (main.coWords.length > 0) parts.push(`Associated with: ${main.coWords.join(', ')}.`);
      if (main.nextWords.length > 0) parts.push(`Followed by: ${main.nextWords.join(', ')}.`);
      if (main.resWords.length > 0) parts.push(`Resonates with: ${main.resWords.join(', ')}.`);

      const meaningful = retrieved.filter(r => r.score > 0.5);
      if (meaningful.length > 0) {
        parts.push('');
        for (const r of meaningful.slice(0, 3)) parts.push(`> ${r.text}`);
      }

      if (concepts.length >= 2) {
        const a = concepts[0], b = concepts[1];
        const shared = a.coWords.filter(w => b.coWords.includes(w));
        if (shared.length > 0) {
          parts.push(`\n"${a.word}" and "${b.word}" connect through: ${shared.slice(0, 4).join(', ')}.`);
        } else {
          const fieldA = this.engine.activate(a.word);
          if (fieldA.has(b.word)) {
            parts.push(`\n"${a.word}" activates "${b.word}" (energy: ${fieldA.get(b.word).toFixed(4)}).`);
          }
        }
      }

      const topicWords = this.memory.getTopicWords(this.engine, 5);
      const overlap = topicWords.filter(w => concepts.some(c => c.coWords.includes(w)));
      if (overlap.length > 0 && this.memory.turns.length > 0) {
        parts.push(`\nBuilding on: ${overlap.slice(0, 4).join(', ')}.`);
      }
    }

    // Detect domain
    const domainCounts = {};
    for (const r of retrieved) {
      if (r.domain) domainCounts[r.domain] = (domainCounts[r.domain] || 0) + r.score;
    }
    const domain = Object.entries(domainCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || null;

    // Update memory
    this.memory.update(field);
    this.memory.addTurn('user', input);
    const responseText = parts.join('\n');
    this.memory.addTurn('shifu', responseText);

    return {
      text: responseText,
      domain,
      concepts: concepts.slice(0, 3).map(c => c.word),
      corrections: corrections.length > 0 ? corrections : undefined,
      retrieved: retrieved.length,
      vocabSize: Object.keys(this.engine.wf).length,
      kbSize: this.kb.size,
    };
  }

  /** Reset conversation memory (not knowledge) */
  resetMemory() {
    this.memory = new ChatMemory();
  }

  /** Get conversation history */
  getHistory() {
    return this.memory.turns;
  }

  /** Stats */
  stats() {
    return {
      vocabSize: Object.keys(this.engine.wf).length,
      knowledgeBase: this.kb.size,
      conversationTurns: this.memory.turns.length,
      topicWords: this.memory.getTopicWords(this.engine, 10),
    };
  }
}

module.exports = { ShifuChat, KnowledgeBase, ChatMemory };
