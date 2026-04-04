// Structural Invariance Layer
//
// The gap: Shifu demonstrates emergent structural semantics but not yet
// invariant semantic abstraction. "doctor treats patient" and "patient was
// treated by doctor" score differently because the engine tracks surface
// position, not underlying role structure.
//
// This layer extracts WHO-DOES-WHAT-TO-WHOM from surface form, using the
// engine's own learned structure (not an external parser). It then compares
// sentences at the role level, where active and passive collapse to the
// same representation.
//
// Three capabilities:
//   1. Role extraction: AGENT, ACTION, PATIENT from any surface form
//   2. Structural comparison: role-level similarity ignoring surface order
//   3. Cross-domain transfer: "doctor treats patient" ≈ "teacher educates student"

// Passive voice markers
const PASSIVE_AUX = new Set(['was', 'were', 'been', 'being', 'is', 'are', 'got']);
const PASSIVE_BY = 'by';

/**
 * Extract relational roles from a sentence using the engine's learned structure.
 *
 * Uses three signals from the engine:
 *   - Action words: high bidirectional connectivity (many nx AND px entries)
 *   - Agent words: typically precede actions
 *   - Patient words: typically follow actions
 *   - Passive detection: aux + verb + "by" pattern → swap agent/patient
 *
 * @param {string} sentence
 * @param {object} engine - ShifuEngine instance
 * @returns {object} { agent, action, patient, passive, roles[], raw }
 */
function extractRoles(sentence, engine) {
  const words = (sentence.toLowerCase().match(/[a-z0-9]+/g) || []).filter(w => w.length > 1);
  if (words.length < 2) return { agent: null, action: null, patient: null, passive: false, roles: [], raw: words };

  // Detect passive voice: look for aux + content word + "by"
  let passive = false;
  let byIndex = -1;
  for (let i = 0; i < words.length; i++) {
    if (PASSIVE_AUX.has(words[i])) {
      // Look ahead for "by"
      for (let j = i + 1; j < Math.min(words.length, i + 5); j++) {
        if (words[j] === PASSIVE_BY) {
          passive = true;
          byIndex = j;
          break;
        }
      }
      if (passive) break;
    }
  }

  // ── Competitive Constraint Satisfaction ──────────────────────────
  // Instead of scoring each word independently, we evaluate complete
  // (agent, action, patient) triples and pick the one that best
  // explains the sentence's directional structure.
  //
  // Constraint: ACTION must explain directional asymmetry between
  // what precedes it (agents) and what follows it (patients).
  // AGENT and PATIENT are then the best remaining candidates.

  const FUNCTION_WORDS = new Set([...PASSIVE_AUX, PASSIVE_BY, 'the', 'with', 'for', 'and', 'to', 'in', 'from', 'of', 'on', 'at', 'an']);

  // Classify each word
  const scored = words.map((w, i) => {
    if (FUNCTION_WORDS.has(w)) {
      return { word: w, index: i, type: 'function' };
    }
    const nx = engine.nx[w] || {};
    const px = engine.px[w] || {};
    const nxKeys = new Set(Object.keys(nx));
    const pxKeys = new Set(Object.keys(px));

    // Directional asymmetry: what follows differs from what precedes
    const union = new Set([...nxKeys, ...pxKeys]);
    const overlap = [...nxKeys].filter(x => pxKeys.has(x)).length;
    const asymmetry = union.size > 0 ? 1 - overlap / union.size : 0;

    const freq = engine.wf[w] || 0;
    return { word: w, index: i, type: 'content', asymmetry, nxKeys, pxKeys, freq };
  });

  const content = scored.filter(s => s.type === 'content');
  let action = null, agent = null, patient = null;

  if (content.length >= 2) {
    // ── Step 1: ACTION wins by best explaining directional structure ──
    // For each candidate action, measure how well the words before it
    // form a coherent "agent set" and the words after form a "patient set"
    // via the engine's nx/px tables.
    const maxFreq = Math.max(...content.map(c => c.freq), 1);

    let bestTripleScore = -Infinity;

    for (const actCand of content) {
      const before = content.filter(c => c.index < actCand.index);
      const after = content.filter(c => c.index > actCand.index);

      // In active voice, action must have content words on BOTH sides
      // (agent before, patient after). This prevents boundary words from winning.
      if (!passive && (before.length === 0 || after.length === 0)) {
        // Only allow this if it's the ONLY option (2-word sentence)
        if (content.length > 2) continue;
      }
      if (before.length === 0 && after.length === 0) continue;

      // Score this candidate as action:
      // 1. Asymmetry: actions bridge different neighborhoods
      let tripleScore = actCand.asymmetry * 10;

      // 2. Directional fit: does nx[agent] expect this action?
      //    does nx[action] expect the patient?
      for (const b of before) {
        const expectsAct = (engine.nx[b.word] || {})[actCand.word] || 0;
        tripleScore += Math.min(expectsAct, 5) * 2;
      }
      for (const a of after) {
        const expectsPat = (engine.nx[actCand.word] || {})[a.word] || 0;
        tripleScore += Math.min(expectsPat, 5) * 2;
      }

      // 3. Completeness bonus: triples with both agent and patient are preferred
      if (before.length > 0 && after.length > 0) tripleScore += 3;

      // 4. Frequency: very high-frequency words are more likely arguments
      const freqRatio = actCand.freq / maxFreq;
      tripleScore -= freqRatio * 4;

      // 5. Position: actions not at sentence boundaries
      if (actCand.index === 0) tripleScore -= 5;
      if (actCand.index === words.length - 1) tripleScore -= 3;

      if (tripleScore > bestTripleScore) {
        bestTripleScore = tripleScore;
        action = actCand;

        // ── Step 2: AGENT and PATIENT from remaining words ──
        if (passive && byIndex > -1) {
          // Passive: subject is patient, "by" phrase is agent
          const beforeAct = before.length > 0 ? before[0] : null;
          const afterByWords = content.filter(c => c.index > byIndex && c.word !== actCand.word);
          patient = beforeAct;
          agent = afterByWords.length > 0 ? afterByWords[0] : null;
        } else {
          // Active: first before is agent, first after is patient
          agent = before.length > 0 ? before[0] : null;
          patient = after.length > 0 ? after[0] : null;
        }
      }
    }
  } else if (content.length === 1) {
    // Single content word — can only be action (or agent with no action)
    action = content[0];
  }

  // Build normalized role list
  const roles = [];
  if (agent) roles.push({ role: 'agent', word: agent.word, index: agent.index });
  if (action) roles.push({ role: 'action', word: action.word, index: action.index });
  if (patient) roles.push({ role: 'patient', word: patient.word, index: patient.index });

  return {
    agent: agent ? agent.word : null,
    action: action ? action.word : null,
    patient: patient ? patient.word : null,
    passive,
    roles,
    raw: words,
  };
}

/**
 * Compare two role structures using the engine's resonance system.
 *
 * This is the key invariance operation: it compares AGENT-to-AGENT,
 * ACTION-to-ACTION, PATIENT-to-PATIENT using resonance similarity,
 * not surface form. Active/passive collapse because roles are normalized.
 *
 * @param {object} rolesA - from extractRoles()
 * @param {object} rolesB - from extractRoles()
 * @param {object} engine - ShifuEngine instance
 * @returns {object} { similarity, agentSim, actionSim, patientSim, details }
 */
function compareRoles(rolesA, rolesB, engine) {
  const result = { similarity: 0, agentSim: 0, actionSim: 0, patientSim: 0, details: {} };

  // Compare each role pair
  const pairs = [
    ['agent', rolesA.agent, rolesB.agent],
    ['action', rolesA.action, rolesB.action],
    ['patient', rolesA.patient, rolesB.patient],
  ];

  let totalWeight = 0;
  let weightedSum = 0;

  for (const [role, wordA, wordB] of pairs) {
    if (!wordA || !wordB) {
      result.details[role] = { wordA, wordB, sim: 0, method: 'missing' };
      continue;
    }

    const a = wordA.toLowerCase();
    const b = wordB.toLowerCase();

    // Same word = perfect match
    if (a === b) {
      const sim = 1.0;
      result[`${role}Sim`] = sim;
      result.details[role] = { wordA: a, wordB: b, sim, method: 'exact' };
      weightedSum += sim * (role === 'action' ? 1.5 : 1.0);
      totalWeight += (role === 'action' ? 1.5 : 1.0);
      continue;
    }

    // Check resonance: have these words been learned as structural equivalents?
    const resonance = engine.res[a]?.[b] || 0;

    // Check context similarity
    const ctxA = engine.contextVec(a);
    const ctxB = engine.contextVec(b);
    let ctxSim = 0;
    if (ctxA.some(x => x !== 0) && ctxB.some(x => x !== 0)) {
      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < ctxA.length; i++) {
        dot += ctxA[i] * ctxB[i];
        na += ctxA[i] ** 2;
        nb += ctxB[i] ** 2;
      }
      na = Math.sqrt(na); nb = Math.sqrt(nb);
      ctxSim = (na < 1e-6 || nb < 1e-6) ? 0 : dot / (na * nb);
    }

    // Combine: resonance is the strongest signal, context similarity is backup
    const resSim = resonance > 0 ? Math.min(resonance / 5, 1.0) : 0;
    const sim = Math.max(resSim, ctxSim * 0.7);

    result[`${role}Sim`] = sim;
    result.details[role] = { wordA: a, wordB: b, sim, resonance, ctxSim, method: resonance > 0 ? 'resonance' : 'context' };

    const weight = role === 'action' ? 1.5 : 1.0;
    weightedSum += sim * weight;
    totalWeight += weight;
  }

  result.similarity = totalWeight > 0 ? weightedSum / totalWeight : 0;

  // ── Cross-role swap detection ──────────────────────────────────
  // Three tiers:
  //   1. Exact swap: same words in swapped roles
  //   2. Partial swap: one side exact, same action
  //   3. Cross-domain swap: A.agent resonates with B.patient more than B.agent
  //      (detects "doctor treats patient" vs "student educates teacher")
  result.swapDetected = false;

  const agA = rolesA.agent ? rolesA.agent.toLowerCase() : null;
  const patA = rolesA.patient ? rolesA.patient.toLowerCase() : null;
  const agB = rolesB.agent ? rolesB.agent.toLowerCase() : null;
  const patB = rolesB.patient ? rolesB.patient.toLowerCase() : null;

  // Tier 1: Full exact swap
  if (agA && patA && agB && patB && agA === patB && patA === agB) {
    result.swapDetected = true;
    result.similarity *= 0.5;
  }
  // Tier 2: Partial exact swap (same action)
  else if (rolesA.action && rolesB.action &&
           rolesA.action.toLowerCase() === rolesB.action.toLowerCase()) {
    const partialSwap = (agA && patB && agA === patB) || (patA && agB && patA === agB);
    if (partialSwap) {
      result.swapDetected = true;
      result.similarity *= 0.6;
    }
  }
  // Tier 3: Cross-domain swap via resonance
  // If A.agent is more similar to B.patient than to B.agent, AND
  // A.patient is more similar to B.agent than to B.patient, roles are swapped
  else if (agA && patA && agB && patB &&
           agA !== agB && patA !== patB) { // different words = cross-domain
    const _roleSim = (w1, w2) => {
      if (!w1 || !w2) return 0;
      if (w1 === w2) return 1.0;
      const res = engine.res[w1]?.[w2] || 0;
      if (res > 0) return Math.min(res / 5, 1.0);
      // Fall back to context similarity
      const c1 = engine.contextVec(w1), c2 = engine.contextVec(w2);
      let dot = 0, n1 = 0, n2 = 0;
      for (let k = 0; k < c1.length; k++) { dot += c1[k]*c2[k]; n1 += c1[k]**2; n2 += c2[k]**2; }
      n1 = Math.sqrt(n1); n2 = Math.sqrt(n2);
      return (n1 < 1e-6 || n2 < 1e-6) ? 0 : Math.max(0, dot/(n1*n2)) * 0.7;
    };

    // Correct alignment: A.agent↔B.agent, A.patient↔B.patient
    const correctAlign = _roleSim(agA, agB) + _roleSim(patA, patB);
    // Swapped alignment: A.agent↔B.patient, A.patient↔B.agent
    const swapAlign = _roleSim(agA, patB) + _roleSim(patA, agB);

    if (swapAlign > correctAlign + 0.05) {
      // Swapped alignment is stronger — this is a cross-domain reversal
      result.swapDetected = true;
      result.crossDomainSwap = true;
      result.similarity *= 0.65;
    }
  }

  return result;
}

/**
 * Score structural invariance between two sentences.
 *
 * This is the high-level API: takes two sentences, extracts roles from both,
 * and compares at the structural level. Returns a single similarity score
 * that is invariant to active/passive transformation.
 *
 * @param {string} sentA
 * @param {string} sentB
 * @param {object} engine - ShifuEngine instance
 * @returns {object} { invariance, rolesA, rolesB, comparison }
 */
function structuralInvariance(sentA, sentB, engine) {
  const rolesA = extractRoles(sentA, engine);
  const rolesB = extractRoles(sentB, engine);
  const comparison = compareRoles(rolesA, rolesB, engine);

  return {
    invariance: comparison.similarity,
    rolesA,
    rolesB,
    comparison,
  };
}

module.exports = { extractRoles, compareRoles, structuralInvariance };
