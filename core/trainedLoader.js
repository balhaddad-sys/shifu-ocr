// Trained Model Loader
// Loads the pre-trained Python model data (trained_model.json) into the JS engine.
// The Python engine trained on character images across multiple fonts.
// We extract the confusion patterns and character relationships to strengthen
// the JS resonance engine's understanding of what characters look like.

const fs = require('fs');
const path = require('path');

const MODEL_PATH = path.join(__dirname, '..', 'shifu_ocr', 'trained_model.json');

function loadTrainedModel(modelPath = MODEL_PATH) {
  if (!fs.existsSync(modelPath)) return null;
  try {
    return JSON.parse(fs.readFileSync(modelPath, 'utf-8'));
  } catch (e) {
    console.warn(`Failed to load trained model: ${e.message}`);
    return null;
  }
}

/**
 * Extract confusion pairs from the trained Python model.
 * Even without recorded confusion data, we generate knowledge from the
 * character landscape statistics (what was trained, how deeply).
 */
function extractConfusionKnowledge(model) {
  if (!model || !model.landscapes) return { pairs: [], sentences: [] };

  const pairs = [];
  const sentences = [];

  // Extract any recorded confusion data
  for (const [label, landscape] of Object.entries(model.landscapes)) {
    const confused = landscape.confused_with || {};
    for (const [confusedWith, count] of Object.entries(confused)) {
      pairs.push({ char: label, confusedWith, count, label });
    }
  }
  pairs.sort((a, b) => b.count - a.count);

  // Generate sentences from recorded confusions
  for (const { char, confusedWith } of pairs.slice(0, 50)) {
    sentences.push(`The character ${char} is not ${confusedWith}.`);
    sentences.push(`${char} looks similar to ${confusedWith} but they are different.`);
    const medSentences = generateMedicalContextForChars(char, confusedWith);
    sentences.push(...medSentences);
  }

  // Even without confusion data, generate knowledge from known OCR-confusable pairs
  // These are topology-predicted: characters that share structural features
  const KNOWN_CONFUSABLE = [
    ['O', '0'], ['l', '1'], ['I', '1'], ['I', 'l'],
    ['5', 'S'], ['8', 'B'], ['6', 'G'], ['2', 'Z'],
    ['D', 'O'], ['9', 'g'], ['9', 'q'],
  ];

  for (const [a, b] of KNOWN_CONFUSABLE) {
    if (model.landscapes[a] && model.landscapes[b]) {
      sentences.push(`${a} and ${b} are visually similar but must be distinguished in medical text.`);
      const medSentences = generateMedicalContextForChars(a, b);
      sentences.push(...medSentences);
    }
  }

  // Generate sentences about trained character groups
  const charLabels = Object.keys(model.landscapes);
  const letters = charLabels.filter(c => /^[A-Z]$/.test(c));
  const digits = charLabels.filter(c => /^[0-9]$/.test(c));

  if (letters.length > 0) {
    sentences.push(`The model recognizes letters ${letters.join(' ')} from medical documents.`);
  }
  if (digits.length > 0) {
    sentences.push(`The model recognizes digits ${digits.join(' ')} from ward census data.`);
  }

  return { pairs, sentences };
}

function generateMedicalContextForChars(char1, char2) {
  const sentences = [];

  const CONTEXT_MAP = {
    'O0': [
      'O2 saturation 98 percent on room air.',
      'Patient on 2L O2 via nasal cannula.',
      'Blood type O positive confirmed.',
      'Ward 10 bed 02 patient transferred.',
    ],
    '1Il': [
      'ICU bed 1 isolation precautions.',
      'INR 1.2 within therapeutic range.',
      'Level 1 trauma activation called.',
    ],
    '5Ss': [
      'Sodium 145 slightly elevated.',
      'GCS score 15 fully conscious.',
      'Stage 5 chronic kidney disease.',
    ],
    '8Bb': [
      'Bed 8B male ward admitted.',
      'Blood pressure 180 over 80.',
      'B12 level 800 normal range.',
    ],
    '2Zz': [
      'Zone 2 evacuation priority.',
      'Zinc 12 mg daily supplement.',
    ],
    'Dd0O': [
      'Doctor ordered CT head.',
      'Diagnosis confirmed by MRI.',
      'Dose adjusted to 200mg daily.',
    ],
  };

  for (const [key, sents] of Object.entries(CONTEXT_MAP)) {
    if (key.includes(char1) || key.includes(char2)) {
      sentences.push(...sents);
    }
  }

  return sentences;
}

/**
 * Feed the trained model knowledge into a ShifuEngine core.
 * This bridges Python training → JS understanding.
 */
function feedTrainedModelToEngine(core, modelPath = MODEL_PATH) {
  const model = loadTrainedModel(modelPath);
  if (!model) return { fed: 0, pairs: 0 };

  const { pairs, sentences } = extractConfusionKnowledge(model);

  let fed = 0;
  for (const sentence of sentences) {
    core.feed(sentence);
    fed++;
  }

  // Feed character statistics as pseudo-sentences
  const charStats = Object.entries(model.landscapes)
    .map(([label, l]) => ({ label, n: l.n, accuracy: l.n_correct / Math.max(l.n, 1) }))
    .sort((a, b) => b.n - a.n);

  for (const { label } of charStats.slice(0, 20)) {
    core.feed(`Character ${label} is commonly seen in medical documents.`);
    fed++;
  }

  return { fed, pairs: pairs.length, characters: charStats.length };
}

module.exports = { loadTrainedModel, extractConfusionKnowledge, feedTrainedModelToEngine };
