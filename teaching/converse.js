#!/usr/bin/env node
// ═══════════════════════════════════════════════════════════════════════
// SHIFU CONVERSATION ENGINE
// ═══════════════════════════════════════════════════════════════════════
//
// A conversational system that uses Shifu's learned neural network to:
//   1. UNDERSTAND — parse input, extract key concepts via wave propagation
//   2. RETRIEVE  — find the most relevant learned knowledge
//   3. REASON    — combine retrieved knowledge into coherent responses
//   4. REMEMBER  — maintain conversation context across turns
//
// This is NOT an LLM. It's a knowledge retrieval + pattern matching
// system built on top of Shifu's co-occurrence/resonance network.

const fs = require('fs');
const path = require('path');
const readline = require('readline');
const { ShifuEngine } = require('../core/engine');

const C = {
  reset: '\x1b[0m', bright: '\x1b[1m', dim: '\x1b[2m',
  green: '\x1b[32m', red: '\x1b[31m', yellow: '\x1b[33m',
  cyan: '\x1b[36m', magenta: '\x1b[35m', blue: '\x1b[34m',
  white: '\x1b[37m',
};

// ─── Knowledge Base ─────────────────────────────────────────────────
// Sentences the engine has learned, indexed for retrieval.

class KnowledgeBase {
  constructor() {
    this.sentences = [];       // All known sentences
    this.index = {};           // word -> [sentence indices]
    this.domainSentences = {}; // domain -> [sentence indices]
  }

  add(sentence, domain) {
    const idx = this.sentences.length;
    this.sentences.push({ text: sentence, domain, words: new Set() });
    const words = sentence.toLowerCase().match(/[a-z0-9]+/g) || [];
    for (const w of words) {
      if (w.length < 2) continue;
      this.sentences[idx].words.add(w);
      (this.index[w] || (this.index[w] = [])).push(idx);
    }
    if (domain) {
      (this.domainSentences[domain] || (this.domainSentences[domain] = [])).push(idx);
    }
  }

  /**
   * Find the most relevant sentences for a set of query words,
   * using TF-IDF-like scoring boosted by wave propagation energy.
   */
  retrieve(queryWords, waveField, limit = 5, domainFilter = null) {
    const scores = {};
    const idf = {};

    // Compute IDF for each query word
    const N = this.sentences.length || 1;
    for (const w of queryWords) {
      const df = (this.index[w] || []).length;
      idf[w] = df > 0 ? Math.log(N / df) : 0;
    }

    // Score each candidate sentence
    const candidateSet = new Set();
    for (const w of queryWords) {
      for (const idx of (this.index[w] || [])) {
        candidateSet.add(idx);
      }
    }

    // Also add candidates from wave-activated words
    if (waveField) {
      const topActivated = [...waveField.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 15);
      for (const [w] of topActivated) {
        for (const idx of (this.index[w] || []).slice(0, 10)) {
          candidateSet.add(idx);
        }
      }
    }

    for (const idx of candidateSet) {
      const sent = this.sentences[idx];
      if (domainFilter && sent.domain !== domainFilter) continue;

      let score = 0;

      // Base score: word overlap with IDF weighting
      for (const w of queryWords) {
        if (sent.words.has(w)) {
          score += (idf[w] || 1) * 2;
        }
      }

      // Wave boost: if the sentence contains highly activated words
      if (waveField) {
        for (const w of sent.words) {
          const energy = waveField.get(w) || 0;
          if (energy > 0.01) score += energy * 3;
        }
      }

      // Length penalty: prefer medium-length sentences for responses
      const wordCount = sent.words.size;
      if (wordCount < 4) score *= 0.5;
      if (wordCount > 20) score *= 0.8;

      if (score > 0) scores[idx] = score;
    }

    return Object.entries(scores)
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit)
      .map(([idx, score]) => ({
        text: this.sentences[idx].text,
        domain: this.sentences[idx].domain,
        score,
      }));
  }
}

// ─── Conversation Context ───────────────────────────────────────────

class ConversationContext {
  constructor() {
    this.turns = [];
    this.topicField = new Map();  // Running wave field of the conversation
    this.activeDomain = null;
    this.entities = new Set();    // Named entities mentioned
    this.recentWords = [];        // Last N important words
  }

  addTurn(role, text, domain) {
    this.turns.push({ role, text, domain, time: Date.now() });
    if (this.turns.length > 20) this.turns = this.turns.slice(-20);
    if (domain) this.activeDomain = domain;
  }

  /** Merge a wave field into the running topic context (with decay) */
  mergeField(field) {
    // Decay existing context
    for (const [w, e] of this.topicField) {
      this.topicField.set(w, e * 0.6);
      if (e < 0.005) this.topicField.delete(w);
    }
    // Add new activations
    for (const [w, e] of field) {
      this.topicField.set(w, (this.topicField.get(w) || 0) + e * 0.4);
    }
  }

  addWords(words) {
    this.recentWords.push(...words);
    if (this.recentWords.length > 30) this.recentWords = this.recentWords.slice(-30);
  }

  getTopicWords(n = 10) {
    return [...this.topicField.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, n)
      .map(([w]) => w);
  }
}

// ─── Intent Detection ───────────────────────────────────────────────

function detectIntent(input) {
  const lower = input.toLowerCase().trim();

  if (/^(what|explain|describe|tell me about|define)\b/i.test(lower)) return 'ask_about';
  if (/^(how|why|when|where|who)\b/i.test(lower)) return 'question';
  if (/^(is|are|does|do|can|could|would|will|should)\b/i.test(lower)) return 'yes_no';
  if (/\?$/.test(lower)) return 'question';
  if (/^(list|show|give me|name)\b/i.test(lower)) return 'list';
  if (/^(compare|difference|versus|vs)\b/i.test(lower)) return 'compare';
  if (/^(help|what can you|commands)\b/i.test(lower)) return 'help';
  if (/^(hi|hello|hey|greetings)\b/i.test(lower)) return 'greeting';
  if (/^(bye|goodbye|quit|exit)\b/i.test(lower)) return 'farewell';
  if (/^(thank|thanks)\b/i.test(lower)) return 'thanks';
  return 'statement';
}

// ─── Domain Detection ───────────────────────────────────────────────

const DOMAIN_KEYWORDS = {
  medical: ['patient', 'doctor', 'nurse', 'diagnosis', 'medication', 'dose', 'mg', 'hospital', 'treatment', 'symptoms', 'blood', 'prescription', 'surgery', 'ward', 'clinical', 'disease', 'therapy', 'drug', 'lab', 'vital', 'pain', 'infection', 'cardiac', 'stroke', 'cancer', 'diabetes', 'pneumonia'],
  legal: ['court', 'plaintiff', 'defendant', 'law', 'statute', 'judge', 'verdict', 'contract', 'liability', 'attorney', 'counsel', 'motion', 'case', 'trial', 'witness', 'appeal', 'ruling', 'damages', 'lawsuit'],
  financial: ['revenue', 'profit', 'stock', 'market', 'dividend', 'earnings', 'asset', 'debt', 'investment', 'budget', 'fiscal', 'interest', 'rate', 'billion', 'million', 'shares', 'capital', 'equity'],
  scientific: ['experiment', 'hypothesis', 'molecule', 'equation', 'catalyst', 'enzyme', 'genome', 'spectrum', 'quantum', 'cell', 'protein', 'reaction', 'analysis', 'research'],
  engineering: ['circuit', 'voltage', 'torque', 'sensor', 'firmware', 'controller', 'frequency', 'load', 'stress', 'algorithm', 'database'],
};

function detectDomain(words) {
  const scores = {};
  for (const [domain, keywords] of Object.entries(DOMAIN_KEYWORDS)) {
    scores[domain] = words.filter(w => keywords.includes(w)).length;
  }
  const best = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
  return best && best[1] > 0 ? best[0] : null;
}

// ─── Response Builder ───────────────────────────────────────────────

class ResponseBuilder {
  constructor(engine, kb) {
    this.engine = engine;
    this.kb = kb;
  }

  /**
   * Build a response by:
   * 1. Extracting key concepts from input
   * 2. Activating wave propagation to find related knowledge
   * 3. Retrieving the most relevant learned sentences
   * 4. Composing a coherent response
   */
  respond(input, context) {
    const words = (input.toLowerCase().match(/[a-z0-9]+/g) || []).filter(w => w.length > 2);
    if (words.length === 0) return { text: "I need more to work with. Ask me about something I've learned.", domain: null };

    const intent = detectIntent(input);
    const domain = detectDomain(words) || context.activeDomain;

    // Handle meta intents
    if (intent === 'greeting') return { text: this._greet(domain), domain };
    if (intent === 'farewell') return { text: "Goodbye. My neural pathways will remember this conversation.", domain };
    if (intent === 'thanks') return { text: "You're welcome. Every exchange strengthens my connections.", domain };
    if (intent === 'help') return { text: this._help(), domain };

    // Build wave propagation field from input words
    const combinedField = new Map();
    for (const w of words) {
      if (!this.engine.wf[w]) continue;
      const field = this.engine.activate(w);
      for (const [target, energy] of field) {
        combinedField.set(target, (combinedField.get(target) || 0) + energy);
      }
    }

    // Merge with conversation context
    if (context.topicField.size > 0) {
      for (const [w, e] of context.topicField) {
        combinedField.set(w, (combinedField.get(w) || 0) + e * 0.3);
      }
    }

    // Retrieve relevant sentences
    const retrieved = this.kb.retrieve(words, combinedField, 8, null);

    // Find what the engine knows about key concepts
    const conceptInfo = this._analyzeConceptsDeep(words);

    // Build the response
    const response = this._compose(input, intent, words, domain, retrieved, conceptInfo, combinedField, context);

    // Update context
    context.mergeField(combinedField);
    context.addWords(words);
    context.addTurn('user', input, domain);
    context.addTurn('shifu', response, domain);

    return { text: response, domain, retrieved: retrieved.length, concepts: conceptInfo.length };
  }

  _analyzeConceptsDeep(words) {
    const concepts = [];
    for (const w of words) {
      if (!this.engine.wf[w]) continue;
      const freq = this.engine.wf[w];
      if (freq < 1) continue;

      const nx = this.engine.nx[w];
      const co = this.engine.co[w];
      const res = this.engine.res[w];

      const nextWords = nx ? Object.entries(nx).sort((a, b) => b[1] - a[1]).slice(0, 5) : [];
      const coWords = co ? Object.entries(co).sort((a, b) => b[1] - a[1]).slice(0, 6) : [];
      const resWords = res ? Object.entries(res).sort((a, b) => b[1] - a[1]).slice(0, 3) : [];

      concepts.push({
        word: w,
        frequency: freq,
        nextWords: nextWords.map(([w]) => w),
        coWords: coWords.map(([w]) => w),
        resWords: resWords.map(([w]) => w),
      });
    }
    return concepts.sort((a, b) => b.frequency - a.frequency);
  }

  _compose(input, intent, words, domain, retrieved, concepts, waveField, context) {
    const parts = [];

    // Acknowledge what the user asked about
    const keyConceptNames = concepts.slice(0, 3).map(c => c.word);

    if (intent === 'ask_about' || intent === 'question') {
      if (concepts.length === 0) {
        return `I haven't learned about ${words.join(', ')} yet. Teach me by feeding more data in that domain.`;
      }

      const main = concepts[0];

      // What I know about this concept
      parts.push(this._describeKnowledge(main, domain));

      // Related knowledge from retrieved sentences
      const relevantSentences = retrieved
        .filter(r => r.score > 1)
        .slice(0, 3);

      if (relevantSentences.length > 0) {
        parts.push('');
        parts.push('From what I\'ve learned:');
        for (const r of relevantSentences) {
          parts.push(`  "${r.text}"`);
        }
      }

      // Connections to other concepts
      if (concepts.length > 1) {
        const connections = this._findConnections(concepts);
        if (connections.length > 0) {
          parts.push('');
          parts.push(`I see connections: ${connections.join('; ')}.`);
        }
      }

    } else if (intent === 'yes_no') {
      const main = concepts[0];
      if (!main) return `I'm not sure. I haven't learned enough about that yet.`;

      const relevantSentences = retrieved.filter(r => r.score > 1.5);
      if (relevantSentences.length > 0) {
        parts.push(`Based on my knowledge of "${main.word}" (seen ${main.frequency} times):`);
        parts.push(`  "${relevantSentences[0].text}"`);
        if (relevantSentences.length > 1) {
          parts.push(`  "${relevantSentences[1].text}"`);
        }
      } else {
        parts.push(`I know "${main.word}" appears in context with: ${main.coWords.slice(0, 4).join(', ')}.`);
        parts.push(`But I need more data to give a confident answer.`);
      }

    } else if (intent === 'list') {
      if (concepts.length === 0) return `I don't have enough data to list that. Feed me more ${domain || 'general'} text.`;

      const main = concepts[0];
      parts.push(`What I know related to "${main.word}":`);

      // List co-occurring terms
      if (main.coWords.length > 0) {
        parts.push(`  Associated terms: ${main.coWords.join(', ')}`);
      }
      if (main.nextWords.length > 0) {
        parts.push(`  Often followed by: ${main.nextWords.join(', ')}`);
      }

      // Pull from retrieved knowledge
      const topRetrieved = retrieved.slice(0, 4);
      if (topRetrieved.length > 0) {
        parts.push('  Related knowledge:');
        for (const r of topRetrieved) {
          parts.push(`    - ${r.text}`);
        }
      }

    } else if (intent === 'compare') {
      if (concepts.length < 2) {
        return `Give me two concepts to compare. I need both sides.`;
      }
      parts.push(this._compareKnowledge(concepts[0], concepts[1]));

    } else {
      // Statement — acknowledge and relate
      if (concepts.length > 0) {
        const main = concepts[0];
        parts.push(this._respondToStatement(main, retrieved, domain, context));
      } else {
        // Unknown words — I can't contribute
        parts.push(`I haven't encountered those terms yet. My knowledge comes from what I've been taught.`);
        if (domain) parts.push(`Try asking about ${domain} topics I've been trained on.`);
      }
    }

    return parts.join('\n');
  }

  _describeKnowledge(concept, domain) {
    const { word, frequency, nextWords, coWords, resWords } = concept;
    const parts = [];

    if (frequency >= 50) {
      parts.push(`I know "${word}" very well (seen ${frequency} times).`);
    } else if (frequency >= 10) {
      parts.push(`I have good knowledge of "${word}" (seen ${frequency} times).`);
    } else if (frequency >= 3) {
      parts.push(`I've encountered "${word}" a few times (${frequency}).`);
    } else {
      parts.push(`I've seen "${word}" ${frequency} time(s) — my knowledge is limited.`);
    }

    if (coWords.length > 0) {
      parts.push(`It appears near: ${coWords.join(', ')}.`);
    }
    if (nextWords.length > 0) {
      parts.push(`It's typically followed by: ${nextWords.join(', ')}.`);
    }
    if (resWords.length > 0) {
      parts.push(`It resonates with: ${resWords.join(', ')}.`);
    }

    return parts.join(' ');
  }

  _findConnections(concepts) {
    const connections = [];
    for (let i = 0; i < concepts.length - 1; i++) {
      for (let j = i + 1; j < concepts.length; j++) {
        const a = concepts[i], b = concepts[j];
        // Check co-occurrence
        if (a.coWords.includes(b.word) || b.coWords.includes(a.word)) {
          connections.push(`"${a.word}" and "${b.word}" often appear together`);
        }
        // Check resonance
        if (a.resWords.includes(b.word) || b.resWords.includes(a.word)) {
          connections.push(`"${a.word}" resonates with "${b.word}"`);
        }
        // Check shared context
        const shared = a.coWords.filter(w => b.coWords.includes(w));
        if (shared.length > 0 && !connections.length) {
          connections.push(`"${a.word}" and "${b.word}" share context with: ${shared.slice(0, 3).join(', ')}`);
        }
      }
    }
    return connections.slice(0, 3);
  }

  _compareKnowledge(a, b) {
    const parts = [];
    parts.push(`Comparing "${a.word}" (freq: ${a.frequency}) vs "${b.word}" (freq: ${b.frequency}):`);

    // Context comparison
    parts.push(`  "${a.word}" appears with: ${a.coWords.slice(0, 5).join(', ')}`);
    parts.push(`  "${b.word}" appears with: ${b.coWords.slice(0, 5).join(', ')}`);

    // Shared context
    const shared = a.coWords.filter(w => b.coWords.includes(w));
    if (shared.length > 0) {
      parts.push(`  Shared context: ${shared.join(', ')}`);
    } else {
      parts.push(`  No shared context — these seem to be from different domains.`);
    }

    // Sequence comparison
    if (a.nextWords.length && b.nextWords.length) {
      parts.push(`  After "${a.word}": ${a.nextWords.slice(0, 3).join(', ')}`);
      parts.push(`  After "${b.word}": ${b.nextWords.slice(0, 3).join(', ')}`);
    }

    return parts.join('\n');
  }

  _respondToStatement(concept, retrieved, domain, context) {
    const parts = [];

    // Relate to what we know
    if (retrieved.length > 0 && retrieved[0].score > 2) {
      parts.push(`That connects to what I know:`);
      parts.push(`  "${retrieved[0].text}"`);
      if (retrieved.length > 1 && retrieved[1].score > 1.5) {
        parts.push(`  "${retrieved[1].text}"`);
      }
    } else {
      parts.push(`I understand "${concept.word}". It appears with: ${concept.coWords.slice(0, 4).join(', ')}.`);
    }

    // Add something from the topic context
    const topicWords = context.getTopicWords(5);
    const relevant = topicWords.filter(w => concept.coWords.includes(w));
    if (relevant.length > 0) {
      parts.push(`This relates to what we've been discussing: ${relevant.join(', ')}.`);
    }

    return parts.join('\n');
  }

  _greet(domain) {
    const vocabSize = Object.keys(this.engine.wf).length;
    const domainStr = domain ? ` I'm strongest in ${domain}.` : '';
    return `Hello. I'm Shifu — a neural network with ${vocabSize} learned words.${domainStr} Ask me about what I've learned, or teach me something new.`;
  }

  _help() {
    return [
      'I can:',
      '  - Answer questions about topics I\'ve learned (medical, legal, financial, scientific, engineering)',
      '  - Describe what I know about a concept: "what is troponin"',
      '  - Compare concepts: "compare plaintiff and defendant"',
      '  - List related terms: "list medications"',
      '  - Complete your thoughts if you give me context',
      '  - Learn from corrections you teach me',
      '',
      'My knowledge comes from the sentences I\'ve been fed.',
      'The more I learn, the better I get.',
    ].join('\n');
  }
}

// ═══════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════

function loadCorpora() {
  const all = {};

  // Medical
  all.medical = [
    "Patient presents with acute onset left sided weakness and slurred speech consistent with stroke.",
    "CT head showed no evidence of hemorrhage or midline shift in the brain parenchyma.",
    "MRI brain revealed acute ischemic infarct in the left middle cerebral artery territory.",
    "Patient was started on aspirin and clopidogrel for secondary stroke prevention therapy.",
    "Neurology team reviewed the patient and recommended thrombolysis with alteplase.",
    "Doctor prescribed levetiracetam for seizure prophylaxis after the cerebrovascular accident.",
    "Troponin I elevated at 2.4 consistent with non ST elevation myocardial infarction.",
    "ECG showed ST elevation in leads V1 through V4 indicating anterior wall MI.",
    "Patient started on dual antiplatelet therapy with aspirin and ticagrelor loading doses.",
    "Heparin drip initiated at 18 units per kilogram per hour for anticoagulation.",
    "Echocardiogram showed ejection fraction of 35 percent with global hypokinesis.",
    "Cardiology consulted for urgent cardiac catheterization and possible intervention.",
    "Chest X-ray reveals bilateral pulmonary infiltrates consistent with pneumonia.",
    "Patient started on piperacillin tazobactam for hospital acquired infection.",
    "Nurse administered morphine sulfate 4mg IV for chest pain management.",
    "Laboratory results show hemoglobin 10.2 and platelet count 180 thousand.",
    "Vancomycin trough level checked and found within therapeutic range.",
    "Blood cultures grew methicillin resistant Staphylococcus aureus.",
    "Patient developed acute kidney injury with rising creatinine level.",
    "Potassium elevated at 6.2 requiring urgent treatment with calcium gluconate.",
    "Patient underwent laparoscopic cholecystectomy for symptomatic cholelithiasis.",
    "Metformin 500mg twice daily prescribed for type 2 diabetes mellitus.",
    "Insulin glargine 20 units subcutaneous at bedtime for basal glycemic control.",
    "Warfarin dose adjusted to achieve target INR of 2 to 3 for anticoagulation.",
    "Complete blood count showed WBC 12.5 hemoglobin 10.2 platelets 180.",
    "Patient diagnosed with stage IIIA non small cell lung cancer.",
    "Chemotherapy with carboplatin and paclitaxel scheduled every 21 days.",
    "Procalcitonin level elevated suggesting bacterial infection requiring antibiotics.",
    "Fall risk assessment indicates high risk requiring bed alarm activation.",
    "Vital signs blood pressure 135 over 82 pulse 78 temperature 37.1 oxygen saturation 97.",
    "Wound care performed with normal saline irrigation and sterile dressing.",
    "Pain assessed at 6 out of 10 on visual analog scale.",
    "Foley catheter output 450mL clear yellow urine over 8 hours.",
    "Patient ambulated 50 feet in hallway with physical therapy assistance.",
    "Discharge summary completed with follow up in outpatient clinic.",
  ];

  all.legal = [
    "The plaintiff alleges breach of contract under the Uniform Commercial Code.",
    "Defendant filed a motion to dismiss for failure to state a claim.",
    "The court granted summary judgment in favor of the respondent.",
    "Counsel submitted memorandum of law in support of preliminary injunction.",
    "The arbitration clause governs all dispute resolution between the parties.",
    "Witness testified under oath regarding the chain of custody of evidence.",
    "The statute of limitations for personal injury is three years from discovery.",
    "Court ordered production of all documents responsive to interrogatories.",
    "The jury returned a verdict of not guilty on all counts.",
    "Defense counsel filed a motion to exclude expert testimony on damages.",
    "The appellate court reversed the lower court decision on constitutional grounds.",
    "Plaintiff seeks compensatory damages in the amount of five million dollars.",
    "The settlement conference resulted in a binding agreement between parties.",
    "Judge sustained the objection on grounds of hearsay.",
    "Attorney filed a notice of appeal within the statutory deadline.",
    "The lease agreement contains a force majeure clause for pandemic disruptions.",
    "Due diligence review revealed material undisclosed liabilities.",
    "The non compete agreement is enforceable for two years within the jurisdiction.",
    "Cross examination revealed inconsistencies in the witness testimony.",
    "The class action certification was granted for affected consumers.",
  ];

  all.financial = [
    "Quarterly revenue increased 12 percent year over year to 4.2 billion.",
    "The Federal Reserve raised interest rates by 25 basis points.",
    "Net income attributable to shareholders was 1.8 billion for the quarter.",
    "Operating expenses decreased due to restructuring charges.",
    "Earnings per share came in at 2.15 beating consensus estimates.",
    "EBITDA margin expanded 200 basis points to 28.5 percent.",
    "Total assets under management reached 12.8 billion dollars.",
    "Free cash flow generation supported share repurchase program.",
    "Capital expenditures directed toward new manufacturing facilities.",
    "Gross profit margin expanded to 62 percent from 58 percent.",
    "Return on invested capital reached 18.5 percent.",
    "The bond offering was oversubscribed by institutional investors.",
    "Inventory levels decreased reflecting improved supply chain management.",
    "The company maintained investment grade credit rating with stable outlook.",
    "Cash and cash equivalents totaled 5.6 billion at quarter end.",
  ];

  all.scientific = [
    "The experiment demonstrated a statistically significant correlation.",
    "Spectroscopic analysis revealed absorption peaks at 254 nanometers.",
    "The catalyst increased reaction yield from 45 to 92 percent.",
    "Genome sequencing identified novel single nucleotide polymorphisms.",
    "The protein crystal structure was resolved at 1.9 angstroms.",
    "Mass spectrometry confirmed the molecular weight of the compound.",
    "Flow cytometry showed 78 percent of cells expressed the marker.",
    "The solar cell achieved 24.5 percent power conversion efficiency.",
    "Western blot confirmed upregulation of the target protein.",
    "Monte Carlo simulation converged with 95 percent confidence interval.",
  ];

  all.engineering = [
    "The microcontroller operates at 3.3 volts with 168 megahertz clock.",
    "Tensile strength of the alloy exceeded 450 megapascals.",
    "The PID control loop uses gains for stable response.",
    "Load testing revealed maximum deflection of 2.3 millimeters.",
    "The firmware update resolved the communication timeout issue.",
    "Power supply delivers 12 volts at 5 amperes with low ripple.",
    "Finite element analysis showed maximum stress at the fillet.",
    "Signal to noise ratio measured at 45 decibels.",
    "Database query time reduced from 2.3 seconds to 45 milliseconds.",
    "The hydraulic actuator provides 100 kilonewtons of force.",
  ];

  // Load existing corpus file
  try {
    const corpusPath = path.join(__dirname, '..', 'corpus_data', 'shifu_language_corpus.txt');
    const lines = fs.readFileSync(corpusPath, 'utf8').split('\n').filter(l => l.trim().length > 20);
    all.general = lines.slice(0, 500);
  } catch (e) { all.general = []; }

  // Load medical corpus
  try {
    const med = require('../learning/medical_corpus');
    all.medical = all.medical.concat(med.slice(0, 200));
  } catch (e) {}

  return all;
}

function main() {
  console.log('');
  console.log(`${C.bright}${C.cyan}${'═'.repeat(70)}${C.reset}`);
  console.log(`${C.bright}${C.cyan}  SHIFU CONVERSATION ENGINE${C.reset}`);
  console.log(`${C.bright}${C.cyan}  Talk to a neural network that learned from cross-domain corpora${C.reset}`);
  console.log(`${C.bright}${C.cyan}${'═'.repeat(70)}${C.reset}`);

  // Build brain
  console.log(`\n${C.dim}  Loading knowledge...${C.reset}`);
  const engine = new ShifuEngine();
  const kb = new KnowledgeBase();
  const corpora = loadCorpora();

  let totalSentences = 0;
  for (const [domain, sentences] of Object.entries(corpora)) {
    for (const s of sentences) {
      engine.feed(s);
      kb.add(s, domain);
      totalSentences++;
    }
  }

  const vocabSize = Object.keys(engine.wf).length;
  console.log(`${C.green}  Ready: ${vocabSize} words, ${totalSentences} sentences, ${Object.keys(corpora).length} domains.${C.reset}`);

  const builder = new ResponseBuilder(engine, kb);
  const context = new ConversationContext();

  console.log(`${C.dim}  Type your message. Try: "what is troponin", "tell me about stroke", "help"${C.reset}`);
  console.log(`${C.dim}  Type "quit" to exit.${C.reset}`);

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const prompt = () => {
    const domainBadge = context.activeDomain
      ? `${C.dim}[${context.activeDomain}]${C.reset} `
      : '';

    rl.question(`\n${C.bright}you>${C.reset} `, (input) => {
      input = (input || '').trim();
      if (!input) { prompt(); return; }
      if (input === 'quit' || input === 'exit') {
        console.log(`\n${C.cyan}  Shifu: Goodbye. I'll remember what we discussed.${C.reset}\n`);
        rl.close();
        return;
      }

      const result = builder.respond(input, context);

      // Display response
      const domainTag = result.domain ? `${C.dim}[${result.domain}]${C.reset} ` : '';
      console.log(`\n${domainTag}${C.cyan}Shifu:${C.reset}`);
      for (const line of result.text.split('\n')) {
        console.log(`  ${line}`);
      }

      prompt();
    });
  };

  prompt();
}

main();
