// Shifu Unified Clinical Corrector
//
// Single correction pipeline that stacks ALL layers:
//   1. Digraph fix (rn→m, cl→d)
//   2. Exact vocabulary match
//   3. Fuzzy match with ADAPTIVE confusion costs (not static)
//   4. Core engine resonance boost (structurally equivalent words)
//   5. Ward vocabulary frequency boost (learned from nurse confirmations)
//   6. Context chain boost (co-occurrence patterns)
//   7. Column-type context boost
//   8. Safety flags (lab ranges, medication ambiguity, doses)
//
// SAFETY PRINCIPLE: Never silently override. Flag uncertainty.

const { ocrWeightedDistance, fixDigraphs } = require('./confusion');
const {
  getBaseWordSet,
  getBaseCandidateIndex,
  getIndexedCandidates,
  getColumnVocabulary,
  VOCABULARY,
} = require('./vocabulary');
const { runSafetyChecks } = require('./safety');
const { normalizeLineNumeric } = require('./numeric');

// ── Common English word guard ─────────────────────────────────────
// Words that are real English and should NEVER be corrected to medical terms.
// Covers function words, common verbs, nouns, adjectives, adverbs.
// This prevents "came→care", "his→hhs", "water→bader", "form→storm" etc.
let _commonEnglishSet = null;
function _isCommonEnglish(word) {
  if (!_commonEnglishSet) {
    _commonEnglishSet = new Set([
      // Determiners, pronouns, prepositions, conjunctions
      'a','an','the','this','that','these','those','my','your','his','her','its','our','their',
      'i','me','we','us','he','him','she','they','them','it','who','whom','what','which',
      'and','or','but','nor','so','yet','if','then','than','as','at','by','in','on','to',
      'of','for','with','from','into','onto','upon','about','above','below','between',
      'through','during','before','after','since','until','against','among','within',
      'without','under','over','along','across','around','behind','beside','beyond',
      'down','near','off','out','past','toward','towards','up',
      // Common verbs (present, past, participles)
      'is','am','are','was','were','be','been','being','have','has','had','having',
      'do','does','did','doing','done','will','would','shall','should','may','might',
      'can','could','must','need','dare','ought',
      'go','goes','went','gone','going','come','came','coming',
      'get','gets','got','getting','give','gave','given','giving',
      'take','took','taken','taking','make','made','making',
      'say','said','saying','tell','told','telling','ask','asked','asking',
      'know','knew','known','knowing','think','thought','thinking',
      'see','saw','seen','seeing','look','looked','looking',
      'find','found','finding','show','showed','shown','showing',
      'feel','felt','feeling','keep','kept','keeping',
      'let','leave','left','leaving','put','run','running',
      'set','sit','sat','sitting','stand','stood','standing',
      'turn','turned','turning','move','moved','moving',
      'try','tried','trying','use','used','using',
      'work','worked','working','call','called','calling',
      'help','helped','helping','start','started','starting',
      'stop','stopped','stopping','play','played','playing',
      'open','opened','opening','close','closed','closing',
      'read','write','wrote','written','writing',
      'speak','spoke','spoken','speaking',
      'bring','brought','bringing','hold','held','holding',
      'learn','learned','learning','understand','understood','understanding',
      'watch','watched','watching','follow','followed','following',
      'live','lived','living','believe','believed','believing',
      'happen','happened','happening','include','included','including',
      'allow','allowed','allowing','add','added','adding',
      'grow','grew','grown','growing','lose','lost','losing',
      'pay','paid','paying','meet','met','meeting',
      'send','sent','sending','build','built','building',
      'fall','fell','fallen','falling','cut','hit','hurt',
      'serve','served','serving','appear','appeared','appearing',
      'cover','covered','covering','offer','offered','offering',
      'raise','raised','raising','create','created','creating',
      'remember','remembered','remembering','consider','considered','considering',
      'explain','explained','explaining','expect','expected','expecting',
      'discuss','discussed','discussing','reach','reached','reaching',
      'become','became','becoming','remain','remained','remaining',
      'suggest','suggested','suggesting','ensure','ensured','ensuring',
      'require','required','requiring','provide','provided','providing',
      'develop','developed','developing','produce','produced','producing',
      'receive','received','receiving','continue','continued','continuing',
      'change','changed','changing','place','placed','placing',
      'describe','described','describing','improve','improved','improving',
      'perform','performed','performing','affect','affected','affecting',
      'practice','practiced','practising','adopt','adopted','adopting',
      'address','addressed','addressing','promote','promoted','promoting',
      'realize','realized','realizing','observe','observed','observing',
      'share','shared','sharing','encourage','encouraged','encouraging',
      'reduce','reduced','reducing','avoid','avoided','avoiding',
      'communicate','communicated','communicating',
      'demonstrate','demonstrated','demonstrating',
      'investigate','investigated','investigating',
      'maintain','maintained','maintaining',
      'acknowledge','acknowledged','acknowledging',
      'confirm','confirmed','confirming','gather','gathered','gathering',
      'direct','directed','directing','interrupt','interrupted',
      'note','noted','noting','type','typed','typing',
      'print','printed','printing','guide','guided','guiding',
      'elaborate','elaborated','elaborating',
      'worry','worried','worrying',
      // Common nouns
      'people','person','man','men','woman','women','child','children',
      'time','year','years','day','days','week','month','way','part',
      'case','cases','form','forms','line','kind','end','side',
      'life','world','place','thing','things','word','words',
      'fact','water','name','family','friend','friends',
      'problem','question','questions','answer','answers',
      'work','job','company','service','process','information',
      'experience','practice','example','structure','role',
      'sense','interest','attention','direction','approach',
      'center','school','student','students','teacher','morning',
      'history','context','condition','situation','community',
      'environment','understanding','skill','skills','perspective',
      'outcome','outcomes','impact','difference','importance',
      'knowledge','detail','details','record','records','unit',
      'table','chair','cost','note','notes','style','level',
      'risk','effect','value','type','resource','resources',
      'concern','concerns','complaint','complaints',
      'idea','ideas','expectation','expectations',
      'purpose','purposes','performance','communication',
      'consultation','consultations','conversation','observation',
      'relationship','behavior','attitude','limitation','temptation',
      'technique','protocol','guide','checklist','summary',
      'recommendation','recommendations','improvement','fulfillment',
      'attachment','engagement','balance','uncertainty','curiosity',
      'decision','control','stability','compassion','empathy',
      'teaching','reasoning','diagnosis','medicine','doctor','doctors',
      'seat','border','floor','minute','voice','stage',
      // Common adjectives
      'good','better','best','bad','worse','worst',
      'new','old','big','small','long','short','large','great',
      'high','low','first','last','next','other','same','different',
      'important','full','young','right','sure','clear',
      'free','hard','real','whole','particular','possible',
      'able','certain','true','single','close','open','obvious',
      'difficult','huge','slight','subtle','warm','brief',
      'particular','general','initial','final','formal','personal',
      'public','medical','clinical','primary','secondary',
      'necessary','appropriate','productive','meaningful',
      'comfortable','agreeable','compliant','flexible','rigid',
      'analytical','crucial','quintessential','eloquent','transformative',
      'keen','genuine','curious','concerned','worried','afraid',
      'complicated','definitive','counterproductive',
      // Common adverbs
      'not','also','very','well','just','only','still','even',
      'here','there','now','always','never','often','again',
      'already','quite','really','almost','enough','much','more',
      'most','too','away','back','however','instead','please',
      'directly','immediately','entirely','continuously','pleasantly',
      'extremely','initially','briefly','mostly','approximately',
      'particularly','intuitively','substantially','excessively',
      'genuinely','smoothly','attentively','freely','eloquently',
      // Time/sequence
      'today','yesterday','tomorrow','soon','later','early',
      // Misc common words that appear in the PDF
      'although','because','whether','while','where','when','how','why',
      'like','every','being','anything','something','everything',
      'making','having','getting','going','coming','saying','doing',
      'born','told','heard','felt','meant','kept','paid','sent',
      'less','enough','either','neither','whether','whose','wherever',
      'hello','sorry','okay','yes','no',
    ]);
  }
  return _commonEnglishSet.has(word);
}

function preserveCase(original, corrected) {
  if (!original || !corrected) return corrected;
  if (original === original.toUpperCase() && original.length > 1) return corrected.toUpperCase();
  if (original[0] === original[0].toUpperCase() && original.slice(1) === original.slice(1).toLowerCase()) {
    return corrected.charAt(0).toUpperCase() + corrected.slice(1).toLowerCase();
  }
  if (original === original.toLowerCase()) return corrected.toLowerCase();
  return corrected;
}

/**
 * Correct a single word through the full pipeline.
 *
 * @param {string} rawWord
 * @param {object} options
 * @param {string}  options.columnType     - Column context for vocabulary boosting
 * @param {object}  options.learningEngine - ShifuLearningEngine (adaptive costs + freq + context)
 * @param {object}  options.coreEngine     - ShifuEngine (resonance partners)
 * @param {object}  options.knownContext   - Already-read fields for context chain lookup
 * @param {string[]} options.previousWords - Words already corrected in this line
 */
function correctWord(rawWord, options = {}) {
  const word = rawWord.trim();
  if (!word) return { original: word, corrected: word, confidence: 0, flag: 'empty', candidates: [] };

  const wordLower = word.toLowerCase();
  const { learningEngine, coreEngine, knownContext, columnType } = options;
  const baseWords = getBaseWordSet();

  // Numbers pass through (safety checks happen at line level)
  if (/^[\d.,/\-:]+$/.test(word)) {
    return { original: word, corrected: word, confidence: 0.8, flag: 'number', candidates: [] };
  }

  // Titles like "Dr.", "Mr.", "Mrs.", "Prof." — pass through as-is
  if (/^(dr|mr|mrs|ms|prof)\.?$/i.test(word)) {
    return { original: word, corrected: word, confidence: 0.95, flag: 'title', candidates: [] };
  }

  // Dosage tokens (e.g., "5OOmg", "200mg", "4mg", "1.5ml") — pass through
  // Also handle OCR-garbled digits: O→0, l→1, I→1
  if (/^\d+\.?\d*(mg|mcg|g|ml|units?|iu|%|mmol|meq)$/i.test(word) ||
      /^[\dOoIl]+\.?[\dOoIl]*(mg|mcg|g|ml|units?|iu|%|mmol|meq)$/i.test(word)) {
    return { original: word, corrected: word, confidence: 0.8, flag: 'dosage', candidates: [] };
  }

  // Room/bed codes: short alphanumeric identifiers (max 6 chars), mostly digits+letters
  // e.g., "3O2A", "4O1", "2O8", "1CU-5" — but NOT "1eft", "A1athoub", "str0ke"
  // Must have at least 2 digits or be all-caps, to avoid false-positiving on OCR-garbled words
  if (word.length <= 6 && /\d/.test(word) && /[a-zA-Z]/.test(word) && !/^[a-zA-Z]{2,}\d[a-zA-Z]{2,}$/.test(word)) {
    const digitCount = (word.match(/\d/g) || []).length;
    const isAllCaps = word === word.toUpperCase() && word.length <= 5;
    // Need 2+ digits, or all-caps, or has dashes — single digit + letters is likely OCR garble
    if ((digitCount >= 2 && /^\d/.test(word)) || isAllCaps || word.includes('-')) {
      return { original: word, corrected: word, confidence: 0.7, flag: 'room_code', candidates: [] };
    }
  }

  // Very short words (1-3 chars) that aren't in vocabulary — don't try to correct.
  // Protects acronyms like MRI, DVT, CVA from being corrected to shorter words.
  if (wordLower.length <= 3 && !baseWords.has(wordLower)) {
    return { original: word, corrected: word, confidence: 0.5, flag: 'short', candidates: [] };
  }

  // Punctuation-heavy tokens
  if (/[^a-zA-Z]/.test(word) && word.replace(/[^a-zA-Z]/g, '').length < word.length * 0.5) {
    return { original: word, corrected: word, confidence: 0.6, flag: 'punctuation', candidates: [] };
  }

  // Build the effective word set: base vocabulary + learned words
  // Exact match
  const isKnown = learningEngine
    ? learningEngine.vocabulary.isKnown(wordLower)
    : baseWords.has(wordLower);

  if (isKnown) {
    const freqBoost = learningEngine
      ? learningEngine.vocabulary.getFrequencyBoost(wordLower) : 0;
    return {
      original: word, corrected: word,
      confidence: Math.min(0.9 + freqBoost * 0.1, 1.0),
      flag: 'exact', candidates: [],
    };
  }

  // Common English words should never be corrected to medical terms
  if (_isCommonEnglish(wordLower)) {
    return {
      original: word, corrected: word,
      confidence: 0.85, flag: 'clean',
      candidates: [],
    };
  }

  // Digraph correction
  const digraphFixed = fixDigraphs(wordLower);
  const digraphKnown = learningEngine
    ? learningEngine.vocabulary.isKnown(digraphFixed)
    : baseWords.has(digraphFixed);
  if (digraphFixed !== wordLower && digraphKnown) {
    return {
      original: word, corrected: preserveCase(word, digraphFixed),
      confidence: 0.85, flag: 'digraph_corrected',
      candidates: [{ word: digraphFixed, distance: 0.3 }],
    };
  }

  // Fuzzy matching through all layers
  const isOcr = options.ocrSource === true;
  const defaultMaxDist = wordLower.length <= 4
    ? Math.max(wordLower.length * 0.35, 1.0)   // short: 1-1.4 edits max
    : Math.max(wordLower.length * 0.45, 2.0);  // longer: proportional
  const maxDist = options.maxDistance ?? defaultMaxDist;
  const maxLenDiff = wordLower.length <= 4 ? 1 : 2;
  const contextWords = columnType ? getColumnVocabulary(columnType) : baseWords;
  const candidateWords = learningEngine
    ? learningEngine.vocabulary.getCandidateWords(wordLower, {
        maxLengthDiff: maxLenDiff,
        maxCandidates: 768,
      })
    : getIndexedCandidates(wordLower, getBaseCandidateIndex(), baseWords, {
        maxLengthDiff: maxLenDiff,
        maxCandidates: 768,
      });

  // Get resonance partners from core engine (words that fill same structural slots)
  let resonanceSet = null;
  if (coreEngine) {
    const partners = coreEngine.resonancePartners(wordLower, 20);
    if (partners.length > 0) {
      resonanceSet = new Map(partners.map(p => [p.word, p.discount]));
    }
  }

  const candidates = [];

  for (const vocabWord of candidateWords) {
    // Quick length filter — stricter for short words
    const maxLenDiff = wordLower.length <= 4 ? 1 : 2;
    if (Math.abs(vocabWord.length - wordLower.length) > maxLenDiff) continue;

    // Distance: use adaptive confusion costs if available, else static
    const dist = learningEngine
      ? learningEngine.confusion.weightedDistance(wordLower, vocabWord)
      : ocrWeightedDistance(wordLower, vocabWord);
    if (dist > maxDist) continue;

    // Layer 1: Column context boost
    const colBoost = contextWords.has(vocabWord) ? 0.4 : 0;

    // Layer 2: Ward vocabulary frequency boost
    const freqBoost = learningEngine
      ? learningEngine.vocabulary.getFrequencyBoost(vocabWord) : 0;

    // Layer 3: Context chain boost (co-occurrence patterns)
    const chainBoost = (learningEngine && knownContext)
      ? learningEngine.context.getBoost(knownContext, columnType || '', vocabWord) : 0;

    // Layer 4: Core engine resonance boost
    let resonanceBoost = 0;
    if (resonanceSet && resonanceSet.has(vocabWord)) {
      resonanceBoost = resonanceSet.get(vocabWord) * 0.3;
    }

    // Combined score
    const lenRatio = Math.min(wordLower.length, vocabWord.length) /
                     Math.max(wordLower.length, vocabWord.length);
    const totalBoost = colBoost + freqBoost * 0.5 + chainBoost + resonanceBoost;
    const score = Math.max(0,
      1.0 - (dist - totalBoost) / Math.max(wordLower.length, 3)
    ) * lenRatio;

    candidates.push({
      word: vocabWord, distance: dist, score,
      boosts: { column: colBoost, frequency: freqBoost, context: chainBoost, resonance: resonanceBoost },
    });
  }

  candidates.sort((a, b) => b.score - a.score);
  const topCandidates = candidates.slice(0, 5);

  if (topCandidates.length === 0) {
    return { original: word, corrected: word, confidence: 0, flag: 'unknown', candidates: [] };
  }

  const top = topCandidates[0];
  const margin = topCandidates.length > 1 ? top.score - topCandidates[1].score : top.score;

  // ── Topographic evidence gate ──────────────────────────────────
  // Core principle: only correct if the input shows EVIDENCE of OCR damage.
  //
  // Evidence of OCR damage:
  //   - Digits mixed with letters (1eft, str0ke, Pat1ent)
  //   - Known digraph confusions (rn→m already handled above)
  //   - OCR-weighted distance is SIGNIFICANTLY cheaper than standard edit distance
  //     (meaning the confusion model explains the difference via character topology)
  //
  // If a word is pure clean letters with no OCR signatures, the confusion model's
  // cheap costs for pairs like m/r, c/e are just topographic similarity —
  // they don't mean the word was actually garbled by a scanner.
  const hasOcrSignature = /\d/.test(word) && /[a-zA-Z]/.test(word); // digit+letter mix
  if (!hasOcrSignature) {
    // Topographic gate: no digit/letter mixing means this word came from
    // clean text, not a scanner. Only correct if the confusion model provides
    // strong evidence of topographic damage.
    const { levDist: _levDist } = require('../core/engine');
    const editDist = _levDist(wordLower, top.word);
    if (editDist > 0) {
      const confusionRatio = top.distance / editDist;
      // OCR source: allow more aggressive correction (confusion model explains more damage)
      const threshold = isOcr
        ? (wordLower.length <= 4 ? 0.5 : 0.7)
        : (wordLower.length <= 4 ? 0.35 : 0.55);

      if (confusionRatio >= threshold) {
        // High ratio = no OCR confusion explains the difference.
        // Exceptions (allow correction despite high ratio):
        // 1. Length differs + edit dist 1 = insertion/deletion (chst→chest)
        // 2. Same length + edit dist 1 + long unknown word = transposition (levetiracelam→levetiracetam)
        //    But NOT for short/known words (came→care = block)
        const isInsDel = editDist <= 1 && wordLower.length !== top.word.length;
        const isLongTransposition = editDist <= 1 && wordLower.length >= 8;
        if (!isInsDel && !isLongTransposition) {
          return { original: word, corrected: word, confidence: 0, flag: 'clean', candidates: topCandidates.slice(0, 3).map(c => ({
            word: c.word, distance: Math.round(c.distance * 100) / 100,
          })) };
        }
      }

      // Even with low ratio: if the edit distance is high relative to word length,
      // this is too much change for a word without OCR signatures. Clean text with
      // 3+ edits on a 7-letter word is not OCR damage — it's a different word.
      // Exception: if confusion ratio is very low (<0.2), the edits ARE explained
      // by known OCR confusions (e.g., I↔t from FLAIR engine), so allow them.
      const editCap = isOcr ? 4 : 3;
      const shortEditCap = isOcr ? 3 : 2;
      if ((editDist >= editCap || (editDist >= shortEditCap && wordLower.length <= 5)) && confusionRatio >= 0.2) {
        return { original: word, corrected: word, confidence: 0, flag: 'clean', candidates: topCandidates.slice(0, 3).map(c => ({
          word: c.word, distance: Math.round(c.distance * 100) / 100,
        })) };
      }
    }
  }

  // Medication ambiguity check
  const meds = new Set(VOCABULARY.medications.map(m => m.toLowerCase()));
  const topIsMed = meds.has(top.word);
  const secondIsMed = topCandidates.length > 1 && meds.has(topCandidates[1].word);

  let confidence = top.score;
  let flag;
  if (topIsMed && secondIsMed && margin < 0.2) {
    flag = 'DANGER_medication_ambiguity';
    confidence = Math.min(confidence, 0.3);
  } else if (confidence > 0.8 && margin > 0.15) {
    flag = 'high_confidence';
  } else if (confidence > 0.5) {
    flag = 'corrected_verify';
  } else {
    flag = 'low_confidence';
  }

  // Low-confidence and dangerous matches: keep the original, don't substitute
  const correctedWord = (flag === 'low_confidence' || flag === 'DANGER_medication_ambiguity')
    ? word
    : preserveCase(word, top.word);

  return {
    original: word, corrected: correctedWord,
    confidence, flag,
    candidates: topCandidates.slice(0, 3).map(c => ({
      word: c.word,
      distance: Math.round(c.distance * 100) / 100,
      boosts: {
        column: Math.round((c.boosts.column || 0) * 100) / 100,
        frequency: Math.round((c.boosts.frequency || 0) * 100) / 100,
        context: Math.round((c.boosts.context || 0) * 100) / 100,
        resonance: Math.round((c.boosts.resonance || 0) * 100) / 100,
      },
    })),
  };
}

/**
 * Correct a full line of OCR text through the unified pipeline.
 */
function correctLine(ocrText, options = {}) {
  const rawWords = ocrText.split(/\s+/).filter(w => w.length > 0);
  const results = [];
  const previousWords = [];

  for (const rawWord of rawWords) {
    // Strip leading/trailing punctuation, correct the core word, re-wrap
    const leadMatch = rawWord.match(/^([^a-zA-Z0-9]*)/);
    const trailMatch = rawWord.match(/([^a-zA-Z0-9]*)$/);
    const lead = leadMatch ? leadMatch[1] : '';
    const trail = trailMatch ? trailMatch[1] : '';
    const core = rawWord.slice(lead.length, rawWord.length - (trail.length || 0)) || rawWord;
    const coreResult = correctWord(core, { ...options, previousWords });
    const result = {
      ...coreResult,
      original: rawWord,
      corrected: lead + coreResult.corrected + trail,
    };
    results.push(result);
    previousWords.push(result.corrected);
  }

  // Numeric canonicalization: normalize OCR-garbled digits in doses
  const rawOutput = results.map(r => r.corrected).join(' ');
  const numeric = normalizeLineNumeric(rawOutput);
  // If numeric normalization changed anything, update the output and add flags
  const output = numeric.text;
  if (numeric.text !== rawOutput) {
    // Update individual word results to reflect normalized text
    const normalizedWords = output.split(/\s+/);
    for (let i = 0; i < Math.min(results.length, normalizedWords.length); i++) {
      if (results[i].corrected !== normalizedWords[i]) {
        results[i].corrected = normalizedWords[i];
        if (results[i].flag === 'dosage' || results[i].flag === 'number') {
          results[i].flag = 'dose_normalized';
        }
      }
    }
  }

  const safetyFlags = [...runSafetyChecks(results), ...numeric.flags];
  const avgConfidence = results.length > 0
    ? results.reduce((sum, r) => sum + r.confidence, 0) / results.length : 0;

  return {
    input: ocrText,
    output,
    words: results,
    safetyFlags,
    avgConfidence: Math.round(avgConfidence * 100) / 100,
    hasWarnings: safetyFlags.some(f => f.severity === 'warning' || f.severity === 'error'),
    hasDangers: safetyFlags.some(f => f.severity === 'danger'),
  };
}

/**
 * Correct a table row — each column gets its own context.
 * Columns are processed left-to-right, building context for subsequent columns.
 */
function correctTableRow(row, options = {}) {
  const corrected = {};
  const allFlags = [];
  const knownContext = { ...(options.knownContext || {}) };

  for (const [columnName, cellText] of Object.entries(row)) {
    if (!cellText || typeof cellText !== 'string') {
      corrected[columnName] = { input: cellText, output: cellText, words: [], safetyFlags: [] };
      continue;
    }

    const result = correctLine(cellText, {
      ...options,
      columnType: columnName,
      knownContext,
    });

    corrected[columnName] = result;

    // Feed this column's output as context for subsequent columns
    knownContext[columnName] = result.output;

    for (const flag of result.safetyFlags) {
      allFlags.push({ column: columnName, ...flag });
    }
  }

  return {
    corrected,
    safetyFlags: allFlags,
    hasDangers: allFlags.some(f => f.severity === 'danger'),
    hasWarnings: allFlags.some(f => f.severity === 'warning' || f.severity === 'error'),
  };
}

/**
 * Confidence assessment.
 * @returns {'accept'|'verify'|'reject'}
 */
function assessConfidence(correctionResult) {
  if (correctionResult.hasDangers) return 'reject';
  if (correctionResult.hasWarnings) return 'verify';
  // Direct check: safety flags with severity 'error' also force verify
  const flags = correctionResult.safetyFlags || [];
  if (flags.some(f => f.severity === 'error' || f.severity === 'warning')) return 'verify';
  // Any low-confidence, unknown, or verify-flagged word forces verify
  const words = correctionResult.words || [];
  if (words.some(w => w.flag === 'low_confidence' || w.flag === 'unknown' || w.flag === 'corrected_verify')) return 'verify';
  if (correctionResult.avgConfidence > 0.7) return 'accept';
  if (correctionResult.avgConfidence > 0.4) return 'verify';
  return 'reject';
}

module.exports = { correctWord, correctLine, correctTableRow, assessConfidence };
