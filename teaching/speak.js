#!/usr/bin/env node
// ═══════════════════════════════════════════════════════════════════════
// SHIFU SPEAKS — Text generation from learned neural network
// ═══════════════════════════════════════════════════════════════════════
//
// Uses the engine's next-word predictions, co-occurrence network,
// wave propagation, and resonance to generate coherent sentences.
//
// Usage:
//   node teaching/speak.js                      # Generate from all domains
//   node teaching/speak.js --seed patient       # Start from a word
//   node teaching/speak.js --domain medical     # Speak in a domain
//   node teaching/speak.js --sentences 20       # Generate N sentences
//   node teaching/speak.js --interactive        # Chat mode

const { ShifuEngine } = require('../core/engine');
const { createTeacher } = require('./index');

const C = {
  reset: '\x1b[0m', bright: '\x1b[1m', dim: '\x1b[2m',
  green: '\x1b[32m', red: '\x1b[31m', yellow: '\x1b[33m',
  cyan: '\x1b[36m', magenta: '\x1b[35m', blue: '\x1b[34m',
};

// ─── CLI args ───────────────────────────────────────────────────────
const args = process.argv.slice(2);
function getArg(name) {
  const idx = args.indexOf('--' + name);
  return idx >= 0 && idx + 1 < args.length ? args[idx + 1] : null;
}
const SEED_WORD = getArg('seed');
const DOMAIN = getArg('domain');
const NUM_SENTENCES = parseInt(getArg('sentences') || '10', 10);
const INTERACTIVE = args.includes('--interactive');

// ─── Sentence Generator ─────────────────────────────────────────────

class ShifuSpeaker {
  constructor(engine) {
    this.engine = engine;
  }

  /**
   * Pick the next word using weighted random from next-word predictions,
   * biased by wave propagation context.
   */
  _pickNext(currentWord, contextField, usedWords, temperature = 0.7) {
    const nx = this.engine.nx[currentWord];
    if (!nx) return null;

    // Get candidates from next-word predictions
    const candidates = Object.entries(nx)
      .filter(([w]) => !usedWords.has(w) || Math.random() < 0.2) // Allow some repetition
      .map(([word, count]) => {
        let score = count;

        // Boost words that are activated in the wave field
        if (contextField && contextField.has(word)) {
          score *= 1 + contextField.get(word) * 3;
        }

        // Boost words with strong co-occurrence to the current word
        const co = this.engine.co[currentWord];
        if (co && co[word]) {
          score *= 1 + co[word] * 0.5;
        }

        // Slight boost for resonant words
        const res = this.engine.res[currentWord];
        if (res && res[word]) {
          score *= 1.3;
        }

        return { word, score };
      });

    if (candidates.length === 0) return null;

    // Apply temperature (lower = more deterministic)
    const total = candidates.reduce((s, c) => s + Math.pow(c.score, 1 / temperature), 0);
    if (total === 0) return candidates[0].word;

    let rand = Math.random() * total;
    for (const c of candidates) {
      rand -= Math.pow(c.score, 1 / temperature);
      if (rand <= 0) return c.word;
    }
    return candidates[candidates.length - 1].word;
  }

  /**
   * Pick a good starting word for a sentence.
   */
  _pickStarter(domain) {
    // Domain-specific starters
    const starters = {
      medical: ['patient', 'doctor', 'nurse', 'laboratory', 'chest', 'blood', 'medication'],
      legal: ['the', 'plaintiff', 'defendant', 'court', 'counsel', 'witness'],
      financial: ['quarterly', 'revenue', 'earnings', 'the', 'total', 'net', 'operating'],
      scientific: ['the', 'results', 'experiment', 'analysis'],
      engineering: ['the', 'load', 'power', 'signal', 'database'],
      education: ['students', 'the', 'teacher', 'curriculum', 'faculty'],
      government: ['the', 'federal', 'agency', 'department', 'city'],
      retail: ['invoice', 'customer', 'product', 'inventory', 'order'],
      logistics: ['container', 'freight', 'shipment', 'fleet', 'warehouse'],
    };

    const pool = domain && starters[domain] ? starters[domain] : Object.keys(starters).flatMap(d => starters[d]);

    // Filter to words the engine actually knows
    const known = pool.filter(w => this.engine.wf[w] && this.engine.nx[w]);
    if (known.length === 0) {
      // Fall back to highest-frequency words that have next-word predictions
      const allWords = Object.entries(this.engine.wf)
        .filter(([w]) => this.engine.nx[w] && Object.keys(this.engine.nx[w]).length > 1)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 20)
        .map(([w]) => w);
      return allWords[Math.floor(Math.random() * allWords.length)] || null;
    }

    return known[Math.floor(Math.random() * known.length)];
  }

  /**
   * Generate a single sentence.
   */
  generate(seedWord, maxLen = 15, temperature = 0.7) {
    let current = (seedWord || this._pickStarter())?.toLowerCase();
    if (!current || !this.engine.wf[current]) return null;

    const words = [current];
    const usedWords = new Set([current]);

    // Build initial context field from the seed word
    let contextField = this.engine.activate(current);

    for (let i = 0; i < maxLen - 1; i++) {
      const next = this._pickNext(current, contextField, usedWords, temperature);
      if (!next) break;

      words.push(next);
      usedWords.add(next);
      current = next;

      // Update context field with wave propagation from the new word
      const newField = this.engine.activateInContext
        ? this.engine.activateInContext(current, contextField)
        : this.engine.activate(current);

      // Merge fields — newer words contribute more
      for (const [w, energy] of newField) {
        contextField.set(w, (contextField.get(w) || 0) * 0.6 + energy * 0.4);
      }
    }

    // Capitalize and format
    if (words.length < 2) return null;
    words[0] = words[0][0].toUpperCase() + words[0].slice(1);
    return words.join(' ') + '.';
  }

  /**
   * Generate a sentence that stays within a domain's vocabulary.
   */
  generateInDomain(domain, seedWord, maxLen = 15, temperature = 0.7) {
    const starter = seedWord || this._pickStarter(domain);
    return this.generate(starter, maxLen, temperature);
  }

  /**
   * Generate a "thought chain" — follow the wave propagation to see
   * what the network associates with a concept.
   */
  associationChain(word, depth = 8) {
    word = word.toLowerCase();
    if (!this.engine.wf[word]) return null;

    const chain = [word];
    const visited = new Set([word]);
    let current = word;

    for (let i = 0; i < depth; i++) {
      const field = this.engine.activate(current);
      // Find the strongest unvisited activation
      let best = null, bestEnergy = 0;
      for (const [w, energy] of field) {
        if (!visited.has(w) && energy > bestEnergy && w.length > 2) {
          best = w;
          bestEnergy = energy;
        }
      }
      if (!best) break;
      chain.push(best);
      visited.add(best);
      current = best;
    }

    return chain;
  }

  /**
   * Complete a partial sentence.
   */
  complete(partialSentence, maxWords = 10, temperature = 0.7) {
    const words = partialSentence.toLowerCase().split(/\s+/).filter(w => w.length > 1);
    if (words.length === 0) return null;

    // Feed the partial sentence into the context
    let contextField = new Map();
    for (const w of words) {
      if (this.engine.wf[w]) {
        const field = this.engine.activate(w);
        for (const [target, energy] of field) {
          contextField.set(target, (contextField.get(target) || 0) + energy);
        }
      }
    }

    // Generate from the last word
    const current = words[words.length - 1];
    const generated = [];
    const usedWords = new Set(words);
    let cur = current;

    for (let i = 0; i < maxWords; i++) {
      const next = this._pickNext(cur, contextField, usedWords, temperature);
      if (!next) break;
      generated.push(next);
      usedWords.add(next);
      cur = next;

      const newField = this.engine.activate(cur);
      for (const [w, energy] of newField) {
        contextField.set(w, (contextField.get(w) || 0) * 0.5 + energy * 0.5);
      }
    }

    return words.concat(generated).join(' ');
  }

  /**
   * "What does the network think about X?" — show all connections.
   */
  introspect(word) {
    word = word.toLowerCase();
    if (!this.engine.wf[word]) return null;

    const field = this.engine.activate(word);
    const topActivations = [...field.entries()]
      .filter(([w]) => w !== word)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);

    const nextWords = this.engine.nx[word]
      ? Object.entries(this.engine.nx[word]).sort((a, b) => b[1] - a[1]).slice(0, 5)
      : [];

    const prevWords = this.engine.px[word]
      ? Object.entries(this.engine.px[word]).sort((a, b) => b[1] - a[1]).slice(0, 5)
      : [];

    const coWords = this.engine.co[word]
      ? Object.entries(this.engine.co[word]).sort((a, b) => b[1] - a[1]).slice(0, 8)
      : [];

    const resonance = this.engine.res[word]
      ? Object.entries(this.engine.res[word]).sort((a, b) => b[1] - a[1]).slice(0, 5)
      : [];

    return {
      word,
      frequency: this.engine.wf[word],
      topActivations: topActivations.map(([w, e]) => ({ word: w, energy: e.toFixed(4) })),
      nextWords: nextWords.map(([w, c]) => ({ word: w, count: c })),
      prevWords: prevWords.map(([w, c]) => ({ word: w, count: c })),
      coOccurs: coWords.map(([w, c]) => ({ word: w, count: c })),
      resonance: resonance.map(([w, s]) => ({ word: w, strength: s.toFixed(3) })),
    };
  }
}

// ═══════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════

function main() {
  console.log('');
  console.log(`${C.bright}${C.cyan}${'═'.repeat(70)}${C.reset}`);
  console.log(`${C.bright}${C.cyan}  SHIFU SPEAKS — Generating text from learned neural network${C.reset}`);
  console.log(`${C.bright}${C.cyan}${'═'.repeat(70)}${C.reset}`);

  // Create and feed the engine
  console.log(`\n${C.dim}  Building brain...${C.reset}`);
  const engine = new ShifuEngine();
  const teacher = createTeacher({ engine }, { restore: false, seedBuiltin: false });

  // Feed massive corpus from feed_live data
  const FEED = {
    medical: [
      "Patient presents with acute onset left sided weakness and slurred speech consistent with stroke.",
      "CT head showed no evidence of hemorrhage or midline shift in the brain parenchyma.",
      "MRI brain revealed acute ischemic infarct in the left middle cerebral artery territory.",
      "Patient was started on aspirin and clopidogrel for secondary stroke prevention therapy.",
      "Neurology team reviewed the patient and recommended thrombolysis with alteplase.",
      "Doctor prescribed levetiracetam for seizure prophylaxis after the cerebrovascular accident.",
      "Troponin I elevated at 2.4 ng/mL consistent with non-ST elevation myocardial infarction.",
      "ECG showed ST elevation in leads V1 through V4 indicating anterior wall MI.",
      "Patient started on dual antiplatelet therapy with aspirin and ticagrelor loading doses.",
      "Heparin drip initiated at 18 units per kilogram per hour for anticoagulation.",
      "Echocardiogram showed ejection fraction of 35 percent with global hypokinesis.",
      "Cardiology consulted for urgent cardiac catheterization and possible intervention.",
      "Blood pressure 140 over 90 heart rate 98 beats per minute respiratory rate 22.",
      "Chest X-ray reveals bilateral pulmonary infiltrates consistent with pneumonia.",
      "Patient started on piperacillin tazobactam 4.5 grams IV every 6 hours.",
      "Nurse administered morphine sulfate 4mg IV for chest pain management.",
      "Laboratory results show hemoglobin 10.2 and platelet count 180 thousand.",
      "Vancomycin trough level checked at 15.2 within therapeutic range.",
      "Patient admitted with community acquired pneumonia and fever.",
      "Doctor prescribed ceftriaxone and azithromycin for pneumonia treatment.",
      "Patient with COPD exacerbation requiring supplemental oxygen therapy.",
      "Nebulized salbutamol and ipratropium given for acute bronchospasm.",
      "Patient developed acute kidney injury with rising creatinine level.",
      "Potassium elevated at 6.2 requiring urgent treatment with calcium gluconate.",
      "Complete blood count showed WBC 12.5 hemoglobin 10.2 platelets 180.",
      "Basic metabolic panel sodium 138 potassium 4.2 chloride 102 bicarbonate 24.",
      "Patient underwent laparoscopic cholecystectomy for symptomatic cholelithiasis.",
      "Wound care performed with normal saline irrigation and sterile dressing applied.",
      "Vital signs recorded blood pressure 135 over 82 pulse 78 temperature 37.1.",
      "Fall risk assessment score of 45 indicating high risk for the patient.",
      "Metformin 500mg twice daily prescribed for newly diagnosed type 2 diabetes.",
      "Insulin glargine 20 units subcutaneous at bedtime for basal glycemic control.",
      "Warfarin dose adjusted to achieve target INR of 2.0 to 3.0 for anticoagulation.",
      "Patient diagnosed with stage IIIA non-small cell lung cancer.",
      "Chemotherapy with carboplatin and paclitaxel scheduled every 21 days for 4 cycles.",
      "Blood cultures grew methicillin resistant Staphylococcus aureus requiring treatment.",
      "Procalcitonin level 5.8 indicating bacterial infection requiring antibiotics.",
      "Discharge summary completed with follow up in outpatient clinic next week.",
    ],
    legal: [
      "The plaintiff alleges breach of contract under section 207 of the Uniform Commercial Code.",
      "Defendant filed a motion to dismiss for failure to state a claim upon which relief can be granted.",
      "The court granted summary judgment in favor of the respondent on all remaining counts.",
      "Counsel submitted memorandum of law in support of the motion for preliminary injunction.",
      "Pursuant to Rule 12 of the Federal Rules of Civil Procedure the complaint is deficient.",
      "The arbitration clause in paragraph 14 of the agreement governs all dispute resolution.",
      "Witness testified under oath regarding the chain of custody of the physical evidence.",
      "The statute of limitations for personal injury tort claims is three years from discovery.",
      "Court ordered production of all documents responsive to interrogatories within thirty days.",
      "The jury returned a verdict of not guilty on all counts of the indictment.",
      "Defense counsel filed a motion in limine to exclude expert testimony on damages.",
      "The appellate court reversed the lower court decision on constitutional grounds.",
      "Plaintiff seeks compensatory damages in the amount of five million dollars.",
      "The settlement conference resulted in a binding agreement between all parties.",
      "Judge sustained the objection on grounds of hearsay under evidentiary rules.",
      "Attorney filed a notice of appeal within the statutory thirty day deadline.",
    ],
    financial: [
      "Quarterly revenue increased 12 percent year over year reaching 4.2 billion dollars.",
      "The Federal Reserve raised interest rates by 25 basis points at the latest meeting.",
      "Net income attributable to common shareholders was 1.8 billion for the fiscal quarter.",
      "Operating expenses decreased 8 percent due to restructuring charges of 500 million.",
      "Earnings per share came in at 2.15 dollars beating consensus estimates by 12 cents.",
      "The portfolio allocation shifted toward investment grade fixed income securities.",
      "EBITDA margin expanded 200 basis points to 28.5 percent reflecting operational efficiency.",
      "Total assets under management reached 12.8 billion dollars a record for the firm.",
      "Free cash flow generation of 2.1 billion supported share repurchase program.",
      "Capital expenditures of 800 million were directed toward new manufacturing facilities.",
      "Gross profit margin expanded to 62 percent from 58 percent in the prior year period.",
      "Return on invested capital reached 18.5 percent exceeding weighted average cost of capital.",
    ],
    scientific: [
      "The experiment demonstrated a statistically significant correlation with p-value below 0.001.",
      "Spectroscopic analysis revealed absorption peaks at 254 nanometers and 380 nanometers.",
      "The catalyst increased reaction yield from 45 percent to 92 percent under mild conditions.",
      "Genome sequencing identified three novel single nucleotide polymorphisms in the target region.",
      "The protein crystal structure was resolved at 1.9 angstroms using X-ray diffraction.",
      "Mass spectrometry confirmed the molecular weight of the synthesized compound at 342 daltons.",
      "Flow cytometry analysis showed 78 percent of cells expressed the surface marker.",
      "The solar cell achieved 24.5 percent power conversion efficiency under standard conditions.",
      "Western blot analysis confirmed upregulation of the target protein by 3.2 fold.",
      "Monte Carlo simulation converged after 10 million iterations with confidence interval.",
    ],
  };

  // Feed everything
  let totalFed = 0;
  for (const [domain, sentences] of Object.entries(FEED)) {
    teacher.activateDomain(domain);
    for (const s of sentences) {
      engine.feed(s);
      totalFed++;
    }
  }

  // Also load the existing language corpus
  const fs = require('fs');
  const path = require('path');
  try {
    const corpusPath = path.join(__dirname, '..', 'corpus_data', 'shifu_language_corpus.txt');
    const lines = fs.readFileSync(corpusPath, 'utf8').split('\n').filter(l => l.trim().length > 20);
    for (const line of lines.slice(0, 1000)) {
      engine.feed(line);
      totalFed++;
    }
  } catch (e) { /* corpus not available */ }

  try {
    const medCorpus = require('../learning/medical_corpus');
    for (const s of medCorpus.slice(0, 200)) {
      engine.feed(s);
      totalFed++;
    }
  } catch (e) { /* not available */ }

  const vocabSize = Object.keys(engine.wf).length;
  console.log(`${C.green}  Brain ready: ${vocabSize} words, ${totalFed} sentences absorbed.${C.reset}`);

  const speaker = new ShifuSpeaker(engine);

  // ─── SECTION 1: Free Generation ─────────────────────────────────

  console.log(`\n${C.bright}${C.magenta}--- SHIFU GENERATES SENTENCES ---${C.reset}`);

  const domains = DOMAIN ? [DOMAIN] : ['medical', 'legal', 'financial', 'scientific'];
  const seeds = SEED_WORD ? [SEED_WORD] : null;

  for (const domain of domains) {
    console.log(`\n  ${C.bright}${C.cyan}[${domain.toUpperCase()}]${C.reset}`);
    const count = DOMAIN ? NUM_SENTENCES : Math.min(NUM_SENTENCES, 5);
    for (let i = 0; i < count; i++) {
      const seed = seeds ? seeds[i % seeds.length] : null;
      const sentence = speaker.generateInDomain(domain, seed, 12 + Math.floor(Math.random() * 6), 0.6 + Math.random() * 0.3);
      if (sentence) {
        console.log(`    ${C.green}"${sentence}"${C.reset}`);
      }
    }
  }

  // ─── SECTION 2: Association Chains ──────────────────────────────

  console.log(`\n${C.bright}${C.magenta}--- ASSOCIATION CHAINS (wave propagation) ---${C.reset}`);

  const chainSeeds = SEED_WORD ? [SEED_WORD] : ['patient', 'court', 'revenue', 'experiment', 'morphine', 'plaintiff'];
  for (const seed of chainSeeds) {
    const chain = speaker.associationChain(seed, 8);
    if (chain) {
      console.log(`  ${C.bright}${seed}${C.reset} ${C.dim}->${C.reset} ${chain.slice(1).map(w => `${C.cyan}${w}${C.reset}`).join(` ${C.dim}->${C.reset} `)}`);
    }
  }

  // ─── SECTION 3: Sentence Completion ─────────────────────────────

  console.log(`\n${C.bright}${C.magenta}--- SENTENCE COMPLETION ---${C.reset}`);

  const prompts = [
    "patient presents with",
    "the court granted",
    "quarterly revenue",
    "the experiment demonstrated",
    "doctor prescribed",
    "defendant filed",
    "chest xray revealed",
  ];

  for (const prompt of prompts) {
    const completed = speaker.complete(prompt, 10, 0.6);
    if (completed) {
      const original = prompt.split(' ').length;
      const words = completed.split(' ');
      const highlighted = words.slice(0, original).join(' ') +
        ` ${C.green}${words.slice(original).join(' ')}${C.reset}`;
      console.log(`  ${C.dim}${prompt}${C.reset} -> ${highlighted}`);
    }
  }

  // ─── SECTION 4: Word Introspection ──────────────────────────────

  console.log(`\n${C.bright}${C.magenta}--- WHAT SHIFU KNOWS (introspection) ---${C.reset}`);

  const inspectWords = SEED_WORD ? [SEED_WORD] : ['patient', 'court', 'revenue'];
  for (const word of inspectWords) {
    const info = speaker.introspect(word);
    if (!info) continue;

    console.log(`\n  ${C.bright}${C.cyan}"${word}"${C.reset} (freq: ${info.frequency})`);
    if (info.nextWords.length) {
      console.log(`    ${C.dim}next:${C.reset} ${info.nextWords.map(w => `${w.word}(${w.count})`).join(', ')}`);
    }
    if (info.prevWords.length) {
      console.log(`    ${C.dim}prev:${C.reset} ${info.prevWords.map(w => `${w.word}(${w.count})`).join(', ')}`);
    }
    if (info.coOccurs.length) {
      console.log(`    ${C.dim}near:${C.reset} ${info.coOccurs.map(w => `${w.word}(${w.count})`).join(', ')}`);
    }
    if (info.resonance.length) {
      console.log(`    ${C.dim}resonates:${C.reset} ${info.resonance.map(w => `${w.word}(${w.strength})`).join(', ')}`);
    }
    if (info.topActivations.length) {
      console.log(`    ${C.dim}activates:${C.reset} ${info.topActivations.slice(0, 6).map(w => `${w.word}(${w.energy})`).join(', ')}`);
    }
  }

  // ─── Interactive Mode ───────────────────────────────────────────

  if (INTERACTIVE) {
    const readline = require('readline');
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

    console.log(`\n${C.bright}${C.cyan}--- INTERACTIVE MODE ---${C.reset}`);
    console.log(`${C.dim}  Commands:${C.reset}`);
    console.log(`${C.dim}    <word>           Generate a sentence starting with that word${C.reset}`);
    console.log(`${C.dim}    ? <word>         Introspect what Shifu knows about a word${C.reset}`);
    console.log(`${C.dim}    > <sentence>     Complete a partial sentence${C.reset}`);
    console.log(`${C.dim}    ~ <word>         Show association chain${C.reset}`);
    console.log(`${C.dim}    quit             Exit${C.reset}`);

    const prompt = () => {
      rl.question(`\n${C.bright}shifu>${C.reset} `, (input) => {
        input = input.trim();
        if (!input || input === 'quit' || input === 'exit') {
          console.log(`${C.dim}Goodbye.${C.reset}`);
          rl.close();
          return;
        }

        if (input.startsWith('? ')) {
          const word = input.slice(2).trim();
          const info = speaker.introspect(word);
          if (!info) { console.log(`${C.red}  Unknown word: ${word}${C.reset}`); }
          else {
            console.log(`  ${C.cyan}"${word}"${C.reset} freq=${info.frequency}`);
            if (info.nextWords.length) console.log(`  next: ${info.nextWords.map(w => w.word).join(', ')}`);
            if (info.prevWords.length) console.log(`  prev: ${info.prevWords.map(w => w.word).join(', ')}`);
            if (info.coOccurs.length) console.log(`  near: ${info.coOccurs.map(w => w.word).join(', ')}`);
            if (info.topActivations.length) console.log(`  wave: ${info.topActivations.slice(0, 8).map(w => w.word).join(', ')}`);
          }
        } else if (input.startsWith('> ')) {
          const partial = input.slice(2).trim();
          const completed = speaker.complete(partial, 12);
          if (completed) console.log(`  ${C.green}${completed}${C.reset}`);
          else console.log(`${C.red}  Could not complete.${C.reset}`);
        } else if (input.startsWith('~ ')) {
          const word = input.slice(2).trim();
          const chain = speaker.associationChain(word, 10);
          if (chain) console.log(`  ${chain.join(' -> ')}`);
          else console.log(`${C.red}  Unknown word: ${word}${C.reset}`);
        } else {
          // Generate sentences from the word
          for (let i = 0; i < 3; i++) {
            const s = speaker.generate(input.toLowerCase(), 14, 0.7);
            if (s) console.log(`  ${C.green}${s}${C.reset}`);
          }
        }

        prompt();
      });
    };
    prompt();
    return; // Don't print footer in interactive mode
  }

  console.log('');
  console.log(`${C.bright}${C.cyan}${'═'.repeat(70)}${C.reset}`);
  console.log(`${C.dim}  Shifu spoke using ${vocabSize} words of learned knowledge.${C.reset}`);
  console.log(`${C.dim}  Try: node teaching/speak.js --interactive${C.reset}`);
  console.log(`${C.dim}  Try: node teaching/speak.js --seed morphine --sentences 10${C.reset}`);
  console.log(`${C.bright}${C.cyan}${'═'.repeat(70)}${C.reset}`);
  console.log('');
}

main();
