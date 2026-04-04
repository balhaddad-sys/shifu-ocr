// Shifu Cross-Domain Registry
// Maps knowledge domains with their vocabularies, confusion patterns, and transfer rules.
// Each domain carries its own resonance landscape that Shifu can switch between or blend.

const DOMAINS = {
  // ─── MEDICAL (existing Shifu strength) ──────────────────────────────
  medical: {
    id: 'medical',
    label: 'Medical & Clinical',
    subdomains: ['neurology', 'cardiology', 'oncology', 'radiology', 'pharmacy', 'nursing', 'surgery', 'pediatrics', 'emergency'],
    seedSources: [
      { type: 'huggingface', dataset: 'bigbio/med_qa', split: 'train' },
      { type: 'huggingface', dataset: 'gamino/wiki_medical_terms', split: 'train' },
      { type: 'kaggle', dataset: 'tboyle10/medicaltranscriptions' },
      { type: 'kaggle', dataset: 'chaitanyakck/medical-text' },
    ],
    confusionProfile: {
      'O,0': 0.1, 'l,1': 0.2, 'I,1': 0.2, '5,S': 0.3, '8,B': 0.3,
      'rn,m': 0.2, 'cl,d': 0.3, 'li,h': 0.3,
    },
    transferWeights: { legal: 0.3, scientific: 0.6, general: 0.4 },
    validators: ['labRange', 'dosePlausibility', 'medicationAmbiguity'],
    priority: 10,
  },

  // ─── LEGAL ──────────────────────────────────────────────────────────
  legal: {
    id: 'legal',
    label: 'Legal & Regulatory',
    subdomains: ['contracts', 'litigation', 'compliance', 'patents', 'corporate', 'criminal', 'immigration'],
    seedSources: [
      { type: 'huggingface', dataset: 'pile-of-law/pile-of-law', split: 'train', subset: 'courtlistener_docket_entry_documents' },
      { type: 'huggingface', dataset: 'lexlms/lex_glue', split: 'train', config: 'eurlex' },
      { type: 'kaggle', dataset: 'amohankumar/legal-text-classification-dataset' },
    ],
    confusionProfile: {
      'O,0': 0.1, 'l,1': 0.2, 'I,l': 0.1, '§,S': 0.3, '¶,P': 0.4,
      'rn,m': 0.2,
    },
    transferWeights: { medical: 0.2, financial: 0.5, general: 0.5 },
    validators: ['sectionReference', 'caseNumber', 'dateFormat'],
    priority: 8,
  },

  // ─── FINANCIAL ──────────────────────────────────────────────────────
  financial: {
    id: 'financial',
    label: 'Financial & Accounting',
    subdomains: ['banking', 'accounting', 'insurance', 'trading', 'audit', 'tax', 'investment'],
    seedSources: [
      { type: 'huggingface', dataset: 'financial_phrasebank', split: 'train' },
      { type: 'huggingface', dataset: 'AdaptLLM/finance-tasks', split: 'train' },
      { type: 'kaggle', dataset: 'ankurzing/sentiment-analysis-for-financial-news' },
      { type: 'kaggle', dataset: 'jeet2016/us-financial-news-articles' },
    ],
    confusionProfile: {
      'O,0': 0.05, 'l,1': 0.1, '5,S': 0.2, '$,S': 0.3, '€,E': 0.4,
      '£,E': 0.4, '¥,Y': 0.3, ',,.': 0.1,
    },
    transferWeights: { legal: 0.5, general: 0.4, scientific: 0.2 },
    validators: ['currencyFormat', 'accountNumber', 'percentageRange'],
    priority: 8,
  },

  // ─── SCIENTIFIC ─────────────────────────────────────────────────────
  scientific: {
    id: 'scientific',
    label: 'Scientific & Research',
    subdomains: ['physics', 'chemistry', 'biology', 'mathematics', 'engineering', 'materials', 'environmental'],
    seedSources: [
      { type: 'huggingface', dataset: 'scientific_papers', split: 'train', config: 'arxiv' },
      { type: 'huggingface', dataset: 'allenai/scirepeval', split: 'train' },
      { type: 'kaggle', dataset: 'Cornell-University/arxiv' },
      { type: 'kaggle', dataset: 'benhamner/nips-papers' },
    ],
    confusionProfile: {
      'O,0': 0.1, 'l,1': 0.15, 'x,×': 0.2, '−,-': 0.1, 'α,a': 0.3,
      'β,B': 0.3, 'μ,u': 0.2, 'π,n': 0.3, 'Σ,E': 0.3, 'Δ,A': 0.3,
    },
    transferWeights: { medical: 0.5, engineering: 0.6, general: 0.3 },
    validators: ['unitConsistency', 'formulaBalance', 'significantFigures'],
    priority: 7,
  },

  // ─── ENGINEERING & TECHNICAL ────────────────────────────────────────
  engineering: {
    id: 'engineering',
    label: 'Engineering & Technical',
    subdomains: ['mechanical', 'electrical', 'civil', 'software', 'aerospace', 'industrial', 'automotive'],
    seedSources: [
      { type: 'huggingface', dataset: 'bigcode/the-stack', split: 'train', subset: 'python' },
      { type: 'kaggle', dataset: 'stackoverflow/stacksample' },
      { type: 'kaggle', dataset: 'promptcloud/product-listing-technical-specs' },
    ],
    confusionProfile: {
      'O,0': 0.1, 'l,1': 0.1, 'I,l': 0.1, '-,_': 0.2, '{,(': 0.3,
      '},)': 0.3, ':,;': 0.2, '=,≡': 0.3,
    },
    transferWeights: { scientific: 0.6, financial: 0.3, general: 0.4 },
    validators: ['measurementUnit', 'toleranceRange', 'partNumber'],
    priority: 7,
  },

  // ─── EDUCATION ──────────────────────────────────────────────────────
  education: {
    id: 'education',
    label: 'Education & Academic',
    subdomains: ['k12', 'university', 'assessment', 'curriculum', 'research', 'administration'],
    seedSources: [
      { type: 'huggingface', dataset: 'cais/mmlu', split: 'test' },
      { type: 'huggingface', dataset: 'race', split: 'train', config: 'all' },
      { type: 'kaggle', dataset: 'Cornell-University/arxiv' },
    ],
    confusionProfile: {
      'O,0': 0.1, 'l,1': 0.2, 'I,l': 0.1, 'rn,m': 0.2,
    },
    transferWeights: { scientific: 0.5, general: 0.7, medical: 0.3 },
    validators: ['gradeRange', 'dateFormat', 'referenceFormat'],
    priority: 6,
  },

  // ─── GOVERNMENT & PUBLIC SECTOR ─────────────────────────────────────
  government: {
    id: 'government',
    label: 'Government & Public Sector',
    subdomains: ['census', 'records', 'permits', 'voting', 'defense', 'social_services', 'immigration'],
    seedSources: [
      { type: 'huggingface', dataset: 'joelito/Multi_Legal_Pile', split: 'train' },
      { type: 'kaggle', dataset: 'usgs/earthquake-database' },
    ],
    confusionProfile: {
      'O,0': 0.1, 'l,1': 0.2, 'I,1': 0.2, '5,S': 0.3,
    },
    transferWeights: { legal: 0.6, financial: 0.4, general: 0.5 },
    validators: ['idFormat', 'dateFormat', 'zipCode'],
    priority: 6,
  },

  // ─── RETAIL & COMMERCE ──────────────────────────────────────────────
  retail: {
    id: 'retail',
    label: 'Retail & E-Commerce',
    subdomains: ['inventory', 'receipts', 'invoices', 'shipping', 'product_catalog', 'pricing'],
    seedSources: [
      { type: 'huggingface', dataset: 'katanaml/invoices-donut-data-v1', split: 'train' },
      { type: 'kaggle', dataset: 'olistbr/brazilian-ecommerce' },
      { type: 'kaggle', dataset: 'carrie1/ecommerce-data' },
    ],
    confusionProfile: {
      'O,0': 0.05, 'l,1': 0.1, '$,S': 0.3, ',,.': 0.1, 'I,1': 0.15,
    },
    transferWeights: { financial: 0.6, general: 0.5, engineering: 0.2 },
    validators: ['skuFormat', 'priceFormat', 'quantityRange'],
    priority: 5,
  },

  // ─── LOGISTICS & TRANSPORTATION ─────────────────────────────────────
  logistics: {
    id: 'logistics',
    label: 'Logistics & Transportation',
    subdomains: ['shipping', 'warehousing', 'fleet', 'customs', 'supply_chain', 'freight'],
    seedSources: [
      { type: 'kaggle', dataset: 'laurinbrechter/supply-chain-data' },
      { type: 'kaggle', dataset: 'zusmani/usdot-traffic-fatalities' },
    ],
    confusionProfile: {
      'O,0': 0.1, 'l,1': 0.1, 'B,8': 0.3, 'D,0': 0.3, 'I,1': 0.2,
    },
    transferWeights: { retail: 0.5, financial: 0.3, government: 0.3 },
    validators: ['trackingNumber', 'containerCode', 'hazmatClass'],
    priority: 5,
  },

  // ─── GENERAL / LITERARY ─────────────────────────────────────────────
  general: {
    id: 'general',
    label: 'General & Literary',
    subdomains: ['fiction', 'nonfiction', 'journalism', 'correspondence', 'reference', 'historical'],
    seedSources: [
      { type: 'huggingface', dataset: 'wikitext', split: 'train', config: 'wikitext-103-v1' },
      { type: 'huggingface', dataset: 'bookcorpus', split: 'train' },
      { type: 'kaggle', dataset: 'lakshmi25npathi/imdb-dataset-of-50k-movie-reviews' },
      { type: 'kaggle', dataset: 'therohk/million-headlines' },
    ],
    confusionProfile: {
      'O,0': 0.1, 'l,1': 0.2, 'I,l': 0.1, 'rn,m': 0.2,
    },
    transferWeights: { education: 0.5, legal: 0.2, scientific: 0.2 },
    validators: [],
    priority: 3,
  },

  // ─── MULTILINGUAL ───────────────────────────────────────────────────
  multilingual: {
    id: 'multilingual',
    label: 'Multilingual & Transliteration',
    subdomains: ['arabic', 'chinese', 'hindi', 'spanish', 'french', 'german', 'japanese', 'korean', 'russian'],
    seedSources: [
      { type: 'huggingface', dataset: 'facebook/flores', split: 'dev' },
      { type: 'huggingface', dataset: 'mc4', split: 'train', config: 'ar' },
      { type: 'kaggle', dataset: 'alexlwh/multilingual-sentence-embeddings' },
    ],
    confusionProfile: {
      'é,e': 0.1, 'ñ,n': 0.2, 'ü,u': 0.2, 'ö,o': 0.2, 'ç,c': 0.2,
      'ä,a': 0.2, 'à,a': 0.1, 'î,i': 0.2,
    },
    transferWeights: { general: 0.4, education: 0.3 },
    validators: ['encodingConsistency', 'scriptDetection'],
    priority: 4,
  },

  // ─── HANDWRITING & HISTORICAL DOCUMENTS ─────────────────────────────
  handwriting: {
    id: 'handwriting',
    label: 'Handwriting & Historical',
    subdomains: ['cursive', 'print', 'historical', 'forms', 'notes', 'manuscripts'],
    seedSources: [
      { type: 'huggingface', dataset: 'Teklia/IAM-line', split: 'train' },
      { type: 'kaggle', dataset: 'landlord/handwriting-recognition' },
      { type: 'kaggle', dataset: 'sachinpatel21/az-handwritten-alphabets-in-csv-format' },
    ],
    confusionProfile: {
      'a,o': 0.2, 'e,c': 0.2, 'n,u': 0.2, 'h,b': 0.3, 'r,v': 0.3,
      'f,t': 0.2, 'i,l': 0.1, 'g,q': 0.2, 'm,w': 0.3,
    },
    transferWeights: { general: 0.5, education: 0.3, medical: 0.4 },
    validators: ['wordBoundary', 'lineSegmentation'],
    priority: 6,
  },
};

// ─── Domain Registry ────────────────────────────────────────────────────

class DomainRegistry {
  constructor() {
    this.domains = {};
    this.activeDomains = new Set();
    this.domainHistory = [];

    // Load all built-in domains
    for (const [id, domain] of Object.entries(DOMAINS)) {
      this.domains[id] = { ...domain, learned: 0, accuracy: 0, lastActive: null };
    }
  }

  /** Get a domain by ID */
  get(domainId) {
    return this.domains[domainId] || null;
  }

  /** List all registered domains */
  list() {
    return Object.values(this.domains).sort((a, b) => b.priority - a.priority);
  }

  /** Register a custom domain */
  register(domain) {
    if (!domain.id || !domain.label) throw new Error('Domain must have id and label');
    this.domains[domain.id] = {
      subdomains: [],
      seedSources: [],
      confusionProfile: {},
      transferWeights: {},
      validators: [],
      priority: 5,
      learned: 0,
      accuracy: 0,
      lastActive: null,
      ...domain,
    };
    return this.domains[domain.id];
  }

  /** Activate a domain for the current session */
  activate(domainId) {
    if (!this.domains[domainId]) throw new Error(`Unknown domain: ${domainId}`);
    this.activeDomains.add(domainId);
    this.domains[domainId].lastActive = new Date().toISOString();
    this.domainHistory.push({ domain: domainId, activated: Date.now() });
    return this.domains[domainId];
  }

  /** Deactivate a domain */
  deactivate(domainId) {
    this.activeDomains.delete(domainId);
  }

  /** Get active domains sorted by priority */
  getActive() {
    return [...this.activeDomains]
      .map(id => this.domains[id])
      .filter(Boolean)
      .sort((a, b) => b.priority - a.priority);
  }

  /** Detect likely domain from text content */
  detect(text) {
    const lower = text.toLowerCase();
    const scores = {};

    // Score each domain by keyword density
    const domainKeywords = {
      medical: ['patient', 'doctor', 'diagnosis', 'medication', 'dose', 'mg', 'lab', 'nurse', 'hospital', 'ct', 'mri', 'blood', 'surgery', 'ward', 'clinical', 'treatment', 'prescription', 'vital', 'symptom', 'chronic'],
      legal: ['court', 'plaintiff', 'defendant', 'statute', 'jurisdiction', 'counsel', 'motion', 'verdict', 'appeal', 'contract', 'clause', 'liability', 'damages', 'witness', 'testimony', 'filing', 'brief', 'ruling'],
      financial: ['revenue', 'asset', 'liability', 'equity', 'balance', 'debit', 'credit', 'interest', 'dividend', 'portfolio', 'market', 'stock', 'bond', 'fiscal', 'budget', 'audit', 'depreciation', 'amortization'],
      scientific: ['hypothesis', 'experiment', 'observation', 'molecule', 'equation', 'velocity', 'spectrum', 'isotope', 'catalyst', 'theorem', 'quantum', 'entropy', 'photon', 'enzyme', 'genome', 'neuron', 'algorithm'],
      engineering: ['circuit', 'voltage', 'torque', 'load', 'stress', 'strain', 'tolerance', 'schematic', 'blueprint', 'specification', 'calibrate', 'weld', 'assembly', 'prototype', 'firmware', 'sensor', 'actuator'],
      education: ['student', 'teacher', 'curriculum', 'grade', 'exam', 'syllabus', 'semester', 'lecture', 'assignment', 'campus', 'enrollment', 'diploma', 'course', 'faculty', 'tuition'],
      government: ['citizen', 'permit', 'regulation', 'agency', 'federal', 'municipal', 'ordinance', 'census', 'election', 'legislative', 'enforcement', 'compliance', 'registration', 'department'],
      retail: ['invoice', 'receipt', 'sku', 'quantity', 'price', 'discount', 'catalog', 'inventory', 'shipping', 'order', 'customer', 'product', 'barcode', 'warehouse', 'vendor'],
      logistics: ['tracking', 'shipment', 'freight', 'container', 'warehouse', 'route', 'delivery', 'customs', 'manifest', 'dispatch', 'fleet', 'cargo', 'pallet', 'loading'],
      general: ['the', 'and', 'but', 'however', 'although', 'chapter', 'story', 'character', 'narrative', 'author', 'novel', 'article', 'report'],
      multilingual: ['translation', 'transliteration', 'locale', 'encoding', 'unicode', 'script', 'diacritical', 'accent'],
      handwriting: ['handwritten', 'cursive', 'manuscript', 'ink', 'pen', 'stroke', 'form', 'signature'],
    };

    for (const [domainId, keywords] of Object.entries(domainKeywords)) {
      let count = 0;
      for (const kw of keywords) {
        const regex = new RegExp(`\\b${kw}\\b`, 'gi');
        const matches = lower.match(regex);
        if (matches) count += matches.length;
      }
      scores[domainId] = count;
    }

    // Return sorted domains by score (top 3)
    const ranked = Object.entries(scores)
      .filter(([, s]) => s > 0)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([id, score]) => ({ domain: this.domains[id], score }));

    return ranked.length > 0 ? ranked : [{ domain: this.domains.general, score: 1 }];
  }

  /** Get blended confusion profile for active domains */
  getBlendedConfusion() {
    const active = this.getActive();
    if (active.length === 0) return {};

    const blended = {};
    const totalPriority = active.reduce((s, d) => s + d.priority, 0);

    for (const domain of active) {
      const weight = domain.priority / totalPriority;
      for (const [pair, cost] of Object.entries(domain.confusionProfile)) {
        const sorted = pair.split(',').sort().join(',');
        blended[sorted] = Math.min(blended[sorted] ?? 1, cost * weight + (blended[sorted] || 0) * (1 - weight));
      }
    }
    return blended;
  }

  /** Find transfer learning opportunities between two domains */
  getTransferWeight(fromDomain, toDomain) {
    const from = this.domains[fromDomain];
    if (!from || !from.transferWeights) return 0;
    return from.transferWeights[toDomain] || 0;
  }

  /** Serialize for persistence */
  serialize() {
    return {
      domains: this.domains,
      activeDomains: [...this.activeDomains],
      domainHistory: this.domainHistory.slice(-100),
    };
  }

  /** Restore from saved state */
  restore(state) {
    if (state.domains) {
      for (const [id, saved] of Object.entries(state.domains)) {
        if (this.domains[id]) {
          this.domains[id].learned = saved.learned || 0;
          this.domains[id].accuracy = saved.accuracy || 0;
          this.domains[id].lastActive = saved.lastActive || null;
        } else {
          this.domains[id] = saved;
        }
      }
    }
    if (state.activeDomains) {
      this.activeDomains = new Set(state.activeDomains);
    }
    if (state.domainHistory) {
      this.domainHistory = state.domainHistory;
    }
  }
}

module.exports = { DOMAINS, DomainRegistry };