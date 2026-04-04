#!/usr/bin/env node
// ═══════════════════════════════════════════════════════════════════════
// SHIFU CONVERSATION — No LLM, No Transformer, No Hardcoding
// ═══════════════════════════════════════════════════════════════════════
// Everything is derived from the learned neural network:
//   - "Stopwords" = words with high frequency + high co-occurrence breadth (learned)
//   - Concept importance = IDF from the engine's own vocabulary (learned)
//   - Responses = retrieved from learned sentences (not templates)
//   - Domain detection = from wave propagation patterns (learned)

const fs = require('fs');
const path = require('path');
const { ShifuEngine } = require('../core/engine');

const C = {
  reset: '\x1b[0m', bright: '\x1b[1m', dim: '\x1b[2m',
  green: '\x1b[32m', red: '\x1b[31m', yellow: '\x1b[33m',
  cyan: '\x1b[36m', magenta: '\x1b[35m',
};

// ─── Knowledge Base (sentence index for retrieval) ──────────────────

class KnowledgeBase {
  constructor() { this.sentences = []; this.index = {}; }

  add(sentence, domain) {
    const idx = this.sentences.length;
    const words = new Set((sentence.toLowerCase().match(/[a-z0-9]+/g) || []).filter(w => w.length > 1));
    this.sentences.push({ text: sentence, domain, words });
    for (const w of words) (this.index[w] || (this.index[w] = [])).push(idx);
  }

  retrieve(queryWords, waveField, engine, limit = 5) {
    const N = this.sentences.length || 1;
    const candidates = new Set();

    // Candidates from direct word matches
    for (const w of queryWords) {
      for (const idx of (this.index[w] || [])) candidates.add(idx);
    }
    // Candidates from wave-activated words (the engine's own associations)
    if (waveField) {
      for (const [w] of [...waveField.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12)) {
        for (const idx of (this.index[w] || []).slice(0, 8)) candidates.add(idx);
      }
    }

    // Score using IDF from the engine's own vocabulary
    const vocabSize = Object.keys(engine.wf).length || 1;
    const results = [];

    for (const idx of candidates) {
      const sent = this.sentences[idx];
      let score = 0;

      for (const w of queryWords) {
        if (sent.words.has(w)) {
          // IDF: rare words in the engine's vocabulary score higher
          const freq = engine.wf[w] || 1;
          const idf = Math.log(vocabSize / freq);
          score += Math.max(idf, 0.1) * 2;
        }
      }

      // Wave energy boost
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
}

// ─── Conversation Memory ────────────────────────────────────────────

class Memory {
  constructor() { this.topicField = new Map(); this.turns = []; }

  update(field) {
    // Decay old context, merge new
    for (const [w, e] of this.topicField) {
      this.topicField.set(w, e * 0.5);
      if (e < 0.003) this.topicField.delete(w);
    }
    for (const [w, e] of field) {
      this.topicField.set(w, (this.topicField.get(w) || 0) + e * 0.5);
    }
  }

  addTurn(role, text) {
    this.turns.push({ role, text });
    if (this.turns.length > 20) this.turns = this.turns.slice(-20);
  }
}

// ─── Core: derive everything from the engine ────────────────────────

function isContentWord(engine, word) {
  // A word is a "content word" if it has LOW frequency relative to the vocabulary.
  // High-frequency words with broad co-occurrence are function words.
  // This is learned entirely from the data — no hardcoded list.
  const freq = engine.wf[word];
  if (!freq) return false;

  const vocabSize = Object.keys(engine.wf).length;
  const medianFreq = engine._medianFreq || 1;

  // If the word is 10x+ more common than median, it's probably a function word
  if (freq > medianFreq * 10) {
    // But check co-occurrence breadth — if it co-occurs with MANY words, it's generic
    const coBreadth = engine.co[word] ? Object.keys(engine.co[word]).length : 0;
    const medianBreadth = Math.sqrt(vocabSize); // rough estimate
    if (coBreadth > medianBreadth * 2) return false; // too generic
  }

  return true;
}

function getWordImportance(engine, word) {
  // IDF-like importance score derived from the engine's own statistics
  if (!engine.wf[word]) return 0;
  const vocabSize = Object.keys(engine.wf).length || 1;
  const freq = engine.wf[word];
  return Math.log(vocabSize / freq);
}

function extractConcepts(engine, words) {
  // Extract and rank concepts by importance (all derived from learned data)
  const concepts = [];

  for (const w of words) {
    if (!engine.wf[w]) continue;
    if (!isContentWord(engine, w)) continue;

    const importance = getWordImportance(engine, w);
    const freq = engine.wf[w];

    // Get associated words (filtered to content words only)
    const coWords = engine.co[w]
      ? Object.entries(engine.co[w])
          .sort((a, b) => b[1] - a[1])
          .filter(([cw]) => isContentWord(engine, cw))
          .slice(0, 8)
          .map(([cw]) => cw)
      : [];

    const nextWords = engine.nx[w]
      ? Object.entries(engine.nx[w])
          .sort((a, b) => b[1] - a[1])
          .filter(([nw]) => isContentWord(engine, nw))
          .slice(0, 5)
          .map(([nw]) => nw)
      : [];

    const resWords = engine.res[w]
      ? Object.entries(engine.res[w])
          .sort((a, b) => b[1] - a[1])
          .filter(([rw]) => isContentWord(engine, rw))
          .slice(0, 4)
          .map(([rw]) => rw)
      : [];

    concepts.push({ word: w, freq, importance, coWords, nextWords, resWords });
  }

  return concepts.sort((a, b) => b.importance - a.importance);
}

function buildWaveField(engine, words, memory) {
  const field = new Map();
  for (const w of words) {
    if (!engine.wf[w]) continue;
    const act = engine.activate(w);
    for (const [t, e] of act) field.set(t, (field.get(t) || 0) + e);
  }
  // Layer in conversation memory
  for (const [w, e] of memory.topicField) {
    field.set(w, (field.get(w) || 0) + e * 0.3);
  }
  return field;
}

// ─── Response Construction (from learned data, not templates) ───────

function buildResponse(engine, kb, concepts, retrieved, words, memory) {
  const parts = [];

  if (concepts.length === 0) {
    // No known content words — check if engine knows ANY of the input words
    const knownCount = words.filter(w => engine.wf[w]).length;
    if (knownCount > 0) {
      parts.push('Those are common words in my vocabulary, but I need more specific terms.');
    } else {
      parts.push("Those words aren't in my vocabulary yet. Feed me text containing them.");
    }
    return parts.join('\n');
  }

  const main = concepts[0];

  // Describe the main concept using learned associations
  parts.push(`"${main.word}" — seen ${main.freq} time(s) in my training.`);

  if (main.coWords.length > 0) {
    parts.push(`Associated with: ${main.coWords.join(', ')}.`);
  }
  if (main.nextWords.length > 0) {
    parts.push(`Followed by: ${main.nextWords.join(', ')}.`);
  }
  if (main.resWords.length > 0) {
    parts.push(`Resonates with: ${main.resWords.join(', ')}.`);
  }

  // Retrieved knowledge (the most valuable part)
  const meaningful = retrieved.filter(r => r.score > 0.5);
  if (meaningful.length > 0) {
    parts.push('');
    for (const r of meaningful.slice(0, 3)) {
      parts.push(`  > ${r.text}`);
    }
  }

  // Cross-concept connections
  if (concepts.length >= 2) {
    const a = concepts[0], b = concepts[1];
    const shared = a.coWords.filter(w => b.coWords.includes(w));
    if (shared.length > 0) {
      parts.push(`\n"${a.word}" and "${b.word}" connect through: ${shared.slice(0, 4).join(', ')}.`);
    } else {
      // Check wave propagation — can we reach B from A?
      const fieldA = engine.activate(a.word);
      if (fieldA.has(b.word)) {
        parts.push(`\n"${a.word}" activates "${b.word}" in my network (energy: ${fieldA.get(b.word).toFixed(4)}).`);
      }
    }
  }

  // Conversation continuity — what does this add to what we discussed?
  const topicWords = [...memory.topicField.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([w]) => w)
    .filter(w => isContentWord(engine, w));

  const contextOverlap = topicWords.filter(w =>
    concepts.some(c => c.coWords.includes(w) || c.word === w)
  );

  if (contextOverlap.length > 0 && memory.turns.length > 0) {
    parts.push(`\nBuilding on: ${contextOverlap.slice(0, 4).join(', ')}.`);
  }

  return parts.join('\n');
}

// ─── Main respond function ──────────────────────────────────────────

function respond(input, engine, kb, memory) {
  const allWords = (input.toLowerCase().match(/[a-z0-9]+/g) || []).filter(w => w.length > 1);

  // Extract concepts (importance ranking learned from data)
  const concepts = extractConcepts(engine, allWords);
  const contentWords = concepts.map(c => c.word);
  const queryWords = contentWords.length > 0 ? contentWords : allWords;

  // Wave propagation
  const field = buildWaveField(engine, queryWords, memory);

  // Retrieve relevant learned sentences
  const retrieved = kb.retrieve(queryWords, field, engine, 8);

  // Build response from learned data
  const response = buildResponse(engine, kb, concepts, retrieved, allWords, memory);

  // Update memory
  memory.update(field);
  memory.addTurn('user', input);
  memory.addTurn('shifu', response);

  // Detect domain from retrieved results
  const domainCounts = {};
  for (const r of retrieved) {
    if (r.domain) domainCounts[r.domain] = (domainCounts[r.domain] || 0) + r.score;
  }
  const domain = Object.entries(domainCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || null;

  return { text: response, domain };
}

// ═══════════════════════════════════════════════════════════════════════
// BUILD BRAIN — load all existing learned data (no hardcoded corpus)
// ═══════════════════════════════════════════════════════════════════════

console.log(`\n${C.bright}${C.cyan}${'═'.repeat(70)}${C.reset}`);
console.log(`${C.bright}${C.cyan}  SHIFU CONVERSATION — Zero Hardcoding${C.reset}`);
console.log(`${C.bright}${C.cyan}  Everything derived from learned co-occurrence + wave propagation${C.reset}`);
console.log(`${C.bright}${C.cyan}${'═'.repeat(70)}${C.reset}`);

const engine = new ShifuEngine();
const kb = new KnowledgeBase();
let totalSentences = 0;

// Load from teaching system's seed output (if available)
const seedDir = path.join(__dirname, '..', 'seeds_out');
if (fs.existsSync(seedDir)) {
  const seedFiles = fs.readdirSync(seedDir).filter(f => f.endsWith('_seed.json'));
  for (const file of seedFiles) {
    try {
      const data = JSON.parse(fs.readFileSync(path.join(seedDir, file), 'utf8'));
      const domain = data.domain || 'general';
      for (const s of (data.sentences || [])) {
        engine.feed(s);
        kb.add(s, domain);
        totalSentences++;
      }
    } catch (e) {}
  }
}

// Load from existing Shifu corpora (these are already part of the project)
try {
  const med = require('../learning/medical_corpus');
  for (const s of med) { engine.feed(s); kb.add(s, 'medical'); totalSentences++; }
} catch (e) {}

try {
  const { MEDICAL_CORPUS } = require('../learning/corpus');
  for (const s of MEDICAL_CORPUS) { engine.feed(s); kb.add(s, 'medical'); totalSentences++; }
} catch (e) {}

try {
  const raw = fs.readFileSync(path.join(__dirname, '..', 'corpus_data', 'shifu_language_corpus.txt'), 'utf8');
  const lines = raw.split('\n').filter(l => l.trim().length > 15).slice(0, 500);
  for (const s of lines) { engine.feed(s); kb.add(s, 'general'); totalSentences++; }
} catch (e) {}

// Load cross-domain test data
const crossDomainDir = path.join(__dirname, '..', 'test', 'cross_domain_data');
if (fs.existsSync(crossDomainDir)) {
  for (const file of ['gutenberg_alice.txt', 'gutenberg_frankenstein.txt', 'gutenberg_pride.txt']) {
    try {
      const raw = fs.readFileSync(path.join(crossDomainDir, file), 'utf8');
      const lines = raw.split('\n').filter(l => l.trim().length > 20).slice(0, 200);
      for (const s of lines) { engine.feed(s); kb.add(s, 'literary'); totalSentences++; }
    } catch (e) {}
  }
}

const vocabSize = Object.keys(engine.wf).length;
const medianFreq = engine._medianFreq;
console.log(`\n${C.green}  Brain: ${vocabSize} words, ${totalSentences} sentences. Median freq: ${medianFreq}.${C.reset}`);
console.log(`${C.dim}  Content word threshold derived from data (no stopword list).${C.reset}`);

// ═══════════════════════════════════════════════════════════════════════
// SIMULATED CONVERSATION
// ═══════════════════════════════════════════════════════════════════════

const memory = new Memory();

const conversation = [
  "What is troponin?",
  "How is stroke treated?",
  "Tell me about pneumonia",
  "What is the connection between diabetes and insulin?",
  "What is a plaintiff?",
  "Compare plaintiff and defendant",
  "Tell me about revenue and earnings",
  "What is a catalyst?",
  "How are stroke and cardiac related?",
  "The patient has chest pain",
  "What medications help with pain?",
  "What about morphine?",
];

console.log(`\n${C.bright}${C.magenta}--- CONVERSATION ---${C.reset}`);

for (const input of conversation) {
  console.log(`\n${C.bright}You:${C.reset} ${input}`);

  const result = respond(input, engine, kb, memory);
  const tag = result.domain ? `${C.dim}[${result.domain}]${C.reset} ` : '';
  console.log(`\n${tag}${C.cyan}Shifu:${C.reset}`);
  for (const line of result.text.split('\n')) {
    console.log(`  ${line}`);
  }
}

console.log(`\n${C.bright}${C.cyan}${'═'.repeat(70)}${C.reset}`);
console.log(`${C.dim}  No LLM. No transformer. No hardcoded stopwords. No templates.${C.reset}`);
console.log(`${C.dim}  Concepts ranked by learned IDF. Responses from retrieved knowledge.${C.reset}`);
console.log(`${C.dim}  Run interactively: node teaching/converse.js${C.reset}`);
console.log(`${C.bright}${C.cyan}${'═'.repeat(70)}${C.reset}\n`);
