"""
Generate a PDF documenting all functions in Shifu OCR v1.7.0
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from datetime import datetime

OUTPUT = "Shifu_OCR_Functions_Reference.pdf"

# Colors
DARK_BG = HexColor("#1a1a2e")
ACCENT = HexColor("#0f3460")
HIGHLIGHT = HexColor("#e94560")
SOFT_BLUE = HexColor("#16213e")
TEXT_DARK = HexColor("#222222")
SECTION_BG = HexColor("#e8eaf6")
SUBSECTION_BG = HexColor("#f5f5f5")
BORDER_COLOR = HexColor("#3f51b5")
LIGHT_ACCENT = HexColor("#e3f2fd")

def build_pdf():
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=A4,
        topMargin=0.6*inch,
        bottomMargin=0.6*inch,
        leftMargin=0.7*inch,
        rightMargin=0.7*inch,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"],
        fontSize=26, leading=32, textColor=ACCENT,
        spaceAfter=6, fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=12, leading=16, textColor=HexColor("#555555"),
        spaceAfter=20, fontName="Helvetica",
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading1"],
        fontSize=18, leading=22, textColor=ACCENT,
        spaceBefore=18, spaceAfter=8, fontName="Helvetica-Bold",
        borderWidth=2, borderColor=BORDER_COLOR, borderPadding=4,
    )
    subsection_style = ParagraphStyle(
        "Subsection", parent=styles["Heading2"],
        fontSize=14, leading=18, textColor=HexColor("#1a237e"),
        spaceBefore=14, spaceAfter=6, fontName="Helvetica-Bold",
    )
    func_name_style = ParagraphStyle(
        "FuncName", parent=styles["Normal"],
        fontSize=11, leading=14, textColor=HIGHLIGHT,
        fontName="Courier-Bold", spaceBefore=8, spaceAfter=2,
    )
    desc_style = ParagraphStyle(
        "Desc", parent=styles["Normal"],
        fontSize=10, leading=14, textColor=TEXT_DARK,
        fontName="Helvetica", spaceAfter=2, leftIndent=12,
    )
    param_style = ParagraphStyle(
        "Param", parent=styles["Normal"],
        fontSize=9, leading=12, textColor=HexColor("#444444"),
        fontName="Courier", leftIndent=24, spaceAfter=1,
    )
    class_style = ParagraphStyle(
        "ClassName", parent=styles["Heading3"],
        fontSize=13, leading=17, textColor=HexColor("#0d47a1"),
        fontName="Helvetica-BoldOblique", spaceBefore=10, spaceAfter=4,
    )
    note_style = ParagraphStyle(
        "Note", parent=styles["Normal"],
        fontSize=9, leading=12, textColor=HexColor("#666666"),
        fontName="Helvetica-Oblique", leftIndent=12, spaceAfter=4,
    )

    story = []

    # ── TITLE PAGE ──────────────────────────────────────────────
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("SHIFU OCR", title_style))
    story.append(Paragraph("Complete Functions Reference", ParagraphStyle(
        "Sub2", parent=styles["Normal"], fontSize=16, leading=20,
        textColor=HIGHLIGHT, fontName="Helvetica-Bold", spaceAfter=8,
    )))
    story.append(Paragraph("v1.7.0 — Medical OCR Post-Processing Engine", subtitle_style))
    story.append(HRFlowable(width="80%", thickness=2, color=ACCENT, spaceAfter=16))
    story.append(Paragraph(
        "Resonance learning, clinical vocabulary, safety flags, adaptive correction, "
        "fluid-theory character recognition, and a self-improving feedback loop.",
        ParagraphStyle("Intro", parent=styles["Normal"], fontSize=11, leading=15,
                       textColor=HexColor("#333"), alignment=TA_JUSTIFY, spaceAfter=20)
    ))

    # Architecture overview table
    arch_data = [
        ["Layer", "Description"],
        ["Core Engine (JS)", "Resonance learning, soft trajectories, skip-gram expectations"],
        ["Clinical Layer", "Vocabulary, confusion model, safety flags, corrector"],
        ["Learning Loop", "Adaptive confusion, ward vocabulary, context chains"],
        ["Medical Corpus", "Pre-trained medical domain knowledge seeder"],
        ["Trained Model Loader", "Python OCR character landscapes to JS confusion awareness"],
        ["Pipeline", "Image to Python OCR to JS correction to safety-checked output"],
        ["Persistence", "Auto-save/load learning state between sessions"],
        ["Feedback Loop", "Propose, evaluate, learn cycle with metrics"],
        ["Metrics Tracker", "Raw/corrected accuracy, per-column breakdown, trends"],
        ["Document Ingestor", "PDF, CSV, image, text ingestion with auto-detection"],
        ["Clinical Weights", "Severity-weighted learning rates for critical corrections"],
        ["Python OCR Engine", "Fluid-theory character recognition (Landscape classifier)"],
        ["Clinical Post-Processor", "Python-side medical vocabulary + context correction"],
        ["Pipeline Worker", "Python subprocess bridge (ShifuOCR + PaddleOCR backends)"],
    ]
    t = Table(arch_data, colWidths=[2*inch, 4.2*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEADING", (0, 0), (-1, -1), 12),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT_ACCENT),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", note_style))
    story.append(PageBreak())

    # ── HELPER: add a function entry ────────────────────────────
    def add_func(name, desc, params=None, returns=None, note=None):
        items = []
        items.append(Paragraph(name, func_name_style))
        items.append(Paragraph(desc, desc_style))
        if params:
            for p in params:
                items.append(Paragraph(p, param_style))
        if returns:
            items.append(Paragraph(f"<b>Returns:</b> {returns}", ParagraphStyle(
                "Ret", parent=desc_style, fontSize=9, textColor=HexColor("#1b5e20"),
                fontName="Helvetica", leftIndent=24,
            )))
        if note:
            items.append(Paragraph(note, note_style))
        items.append(Spacer(1, 4))
        story.append(KeepTogether(items))

    def add_section(title):
        story.append(Paragraph(title, section_style))
        story.append(HRFlowable(width="100%", thickness=1, color=BORDER_COLOR, spaceAfter=6))

    def add_subsection(title):
        story.append(Paragraph(title, subsection_style))

    def add_class(name):
        story.append(Paragraph(name, class_style))

    # ════════════════════════════════════════════════════════════
    # 1. TOP-LEVEL API (index.js)
    # ════════════════════════════════════════════════════════════
    add_section("1. Top-Level API (index.js)")
    story.append(Paragraph(
        "The main entry point. Creates fully initialized Shifu systems with all layers wired together.",
        desc_style))
    story.append(Spacer(1, 6))

    add_func("createShifu(opts)",
        "Create a fully initialized Shifu system. Loads trained model, seeds with medical corpus, "
        "and optionally restores saved state.",
        ["opts.seed (bool) — Pre-seed with medical corpus (default: true)",
         "opts.loadTrained (bool) — Load Python trained model (default: true)",
         "opts.autoSave (bool) — Enable auto-save after corrections (default: false)",
         "opts.stateDir (string) — Directory for persistence",
         "opts.savedCoreState (object) — Serialized core engine state to restore",
         "opts.savedLearningState (object) — Serialized learning engine state to restore"],
        "Complete Shifu system object with correctLine, learn, createPipeline, etc.")

    add_func("restoreShifu(savedState, opts)",
        "Restore a Shifu system from previously serialized state.",
        ["savedState (object) — Output from shifu.serialize()",
         "opts (object) — Additional options passed to createShifu"],
        "Restored Shifu system object")

    add_func("createOrRestore(opts)",
        "Create a Shifu system that auto-loads from saved state if available. "
        "Falls back to fresh initialization if no saved state found.",
        ["opts.stateDir (string) — Directory for persistence"],
        "Shifu system (restored or fresh) with autoSave enabled")

    add_subsection("Shifu Instance Methods")

    add_func("shifu.correctLine(ocrText, options)",
        "Correct a line of OCR text through the full correction pipeline (digraph fix, vocabulary match, "
        "fuzzy match with adaptive confusion, resonance boost, safety flags).",
        ["ocrText (string) — Raw OCR output",
         "options (object) — columnType, knownContext, etc."],
        "{ input, output, words[], safetyFlags[], avgConfidence, hasWarnings, hasDangers }")

    add_func("shifu.correctTableRow(row, options)",
        "Correct a table row with column-aware context. Each column gets its own vocabulary boost.",
        ["row (object) — { columnName: cellText, ... }"],
        "{ corrected, safetyFlags, hasDangers, hasWarnings }")

    add_func("shifu.correctRowAdaptive(ocrRow, knownContext)",
        "Correct a row using the adaptive learning engine. Adds coherence scoring from the core engine.",
        ["ocrRow (object) — { columnName: cellText, ... }",
         "knownContext (object) — Already-known fields for context chain lookup"],
        "{ corrected (with coherence + meanSurprise), safetyFlags, ... }")

    add_func("shifu.learn(ocrRow, confirmedRow)",
        "Learn from a nurse/doctor confirmed correction. Updates confusion costs, vocabulary frequencies, "
        "context chains, and feeds confirmed text into the core resonance engine.",
        ["ocrRow (object) — Original OCR row",
         "confirmedRow (object) — Human-verified correct row"])

    add_func("shifu.scoreSentence(text)",
        "Score text for linguistic coherence using the core engine.",
        ["text (string) — Text to evaluate"],
        "{ coherence, meanSurprise, ... }")

    add_func("shifu.compare(a, b, profile)",
        "Compare two words/phrases across multiple dimensions (form, context, history, influence, contrast, expectation).",
        ["a, b (string) — Words to compare",
         "profile (string) — 'meaning' or 'correction'"],
        "Multi-dimensional similarity scores")

    add_func("shifu.similar(word, k)",
        "Find the k most similar words to the given word.",
        ["word (string)", "k (number) — default: 10"],
        "Array of { word, similarity } objects")

    add_func("shifu.correct(garbled, k)",
        "Find the best correction candidates for a garbled word using OCR distance.",
        ["garbled (string)", "k (number) — default: 5"],
        "Array of correction candidates")

    add_func("shifu.resonancePartners(word, k)",
        "Get words that have been learned to be structurally equivalent (fill same slots).",
        ["word (string)", "k (number) — default: 10"],
        "Array of { word, evidence, discount }")

    add_func("shifu.createPipeline(opts)",
        "Create an end-to-end pipeline: Image to Python OCR to JS correction to output.",
        ["opts (object) — python, modelPath, tempDir"],
        "ShifuPipeline instance")

    add_func("shifu.checkLabRange(labName, rawValue)",
        "Check if a lab value falls within physiological range.",
        ["labName (string) — e.g. 'sodium', 'potassium'",
         "rawValue (string|number)"],
        "{ status, message, value, alternatives? }")

    add_func("shifu.checkMedicationAmbiguity(ocrText, candidates)",
        "Check if OCR text is ambiguous between two medications.",
        returns="Safety flag or null")

    add_func("shifu.checkDosePlausibility(medName, doseStr)",
        "Check if a medication dose is physiologically plausible.",
        returns="Safety flag or null")

    add_func("shifu.assessConfidence(correctionResult)",
        "Assess overall confidence of a correction result.",
        returns="'accept' | 'verify' | 'reject'")

    add_func("shifu.stats()",
        "Get system statistics: core engine stats, learning stats, vocabulary size.",
        returns="{ core, learning, vocabulary }")

    add_func("shifu.serialize()",
        "Serialize the full system state for persistence.",
        returns="{ core, learning, version, savedAt }")

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # 2. CORE ENGINE
    # ════════════════════════════════════════════════════════════
    add_section("2. Core Engine (core/engine.js)")
    add_class("class ShifuEngine")
    story.append(Paragraph(
        "The resonance learning engine. Builds multi-dimensional word representations from raw text. "
        "v1.7.0 adds resonance — learned structural equivalence between words that fill the same slots.",
        desc_style))

    add_func("engine.feed(sentence)",
        "Feed a single sentence. Updates word frequencies, co-occurrences, bigram transitions (nx, px, nx2), "
        "skip-gram expectations (snx), and accumulates resonance evidence between structurally equivalent words.",
        ["sentence (string)"],
        "Number of tokens processed")

    add_func("engine.feedText(text)",
        "Feed multi-sentence text (splits on sentence boundaries).",
        returns="{ sentences, tokens }")

    add_func("engine.feedBatch(texts)",
        "Feed an array of text strings.",
        returns="{ texts, sentences, tokens }")

    add_func("engine.vec(word)",
        "Compute the full 60-dimensional word vector: form(16) + context(12) + history(8) + influence(8) + contrast(8) + expectation(8).",
        returns="Float64Array[60]")

    add_func("engine.formVec(word)",
        "16-dimensional form vector: vowel/consonant patterns, syllable structure, positional encoding.",
        returns="Float64Array[16]")

    add_func("engine.contextVec(word)",
        "12-dimensional context vector: neighborhood richness, frequency patterns, positional spread.",
        returns="Float64Array[12]")

    add_func("engine.historyVec(word)",
        "8-dimensional history vector: frequency, age, recency, regularity, positional stability.",
        returns="Float64Array[8]")

    add_func("engine.influenceVec(word)",
        "8-dimensional influence vector: sentence length patterns, novelty exposure, clustering, drift.",
        returns="Float64Array[8]")

    add_func("engine.contrastVec(word)",
        "8-dimensional contrast vector: deviation from global averages in frequency, position, context.",
        returns="Float64Array[8]")

    add_func("engine.expectationVec(word)",
        "8-dimensional expectation vector: forward/backward predictability, branching factor, concentration.",
        returns="Float64Array[8]")

    add_func("engine.compare(a, b, profile, mask)",
        "Multi-dimensional comparison with profile-weighted scoring. Includes soft trajectory and resonance awareness.",
        ["profile — 'meaning' or 'correction'"],
        "Object with per-channel similarities + expectsAB, trajectoryAB, resonance flags")

    add_func("engine.softNx(word)",
        "Augmented next-word prediction. Uses hard data, then resonance partners, then contextVec similarity fallback.",
        returns="{ table, soft (bool), resonance (bool) }")

    add_func("engine.softNx2(a, b)",
        "Augmented trigram prediction. Resonance-aware two-step lookahead.",
        returns="{ table, soft, resonance }")

    add_func("engine.resonancePartners(word, k)",
        "Get words with learned structural equivalence, sorted by accumulated evidence.",
        returns="Array of { word, evidence, discount }")

    add_func("engine.decay(factor)",
        "Apply temporal decay to all statistics. Resonance decays slower (sqrt of factor).")

    add_func("engine.compact(opts)",
        "Prune low-value entries to control memory. Limits per-word nx, co, snx, res entries.",
        returns="{ nx, co, snx, nx2, res, vocab } — counts of pruned entries")

    add_func("engine.serialize() / ShifuEngine.deserialize(json)",
        "Serialize to/from JSON string for persistence.")

    add_func("engine.stats()",
        "Get engine statistics: vocabulary size, total sentences, tokens, resonance pairs.",
        returns="Stats object")

    add_subsection("Utility Functions")
    add_func("ocrDist(a, b)", "OCR-weighted edit distance using topology-predicted confusion costs.", returns="number")
    add_func("levDist(a, b)", "Standard Levenshtein edit distance.", returns="number")
    add_func("cos(a, b, lo, hi)", "Cosine similarity between vector slices.", returns="number")
    add_func("cosVec(a, b)", "Full-vector cosine similarity.", returns="number")

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # 3. CLINICAL CORRECTOR
    # ════════════════════════════════════════════════════════════
    add_section("3. Clinical Corrector (clinical/corrector.js)")
    story.append(Paragraph(
        "Unified correction pipeline stacking 8 layers: digraph fix, exact match, "
        "fuzzy match with adaptive confusion, resonance boost, ward vocabulary boost, "
        "context chain boost, column-type boost, and safety flags.", desc_style))

    add_func("correctWord(rawWord, options)",
        "Correct a single word through the full pipeline. Handles numbers, titles, dosages, room codes, "
        "short words, and runs multi-layer fuzzy matching with medication ambiguity detection.",
        ["rawWord (string)",
         "options.columnType — Column context for vocabulary boosting",
         "options.learningEngine — ShifuLearningEngine instance",
         "options.coreEngine — ShifuEngine instance (for resonance partners)",
         "options.knownContext — Already-read fields for context chain lookup",
         "options.previousWords — Words already corrected in this line"],
        "{ original, corrected, confidence, flag, candidates[] }",
        "Flags: exact, digraph_corrected, high_confidence, corrected_verify, low_confidence, "
        "unknown, DANGER_medication_ambiguity, number, title, dosage, room_code, short, empty")

    add_func("correctLine(ocrText, options)",
        "Correct a full line of OCR text. Runs correctWord on each token, then safety checks.",
        returns="{ input, output, words[], safetyFlags[], avgConfidence, hasWarnings, hasDangers }")

    add_func("correctTableRow(row, options)",
        "Correct a table row with column-aware context. Columns processed left-to-right, each feeding "
        "context to subsequent columns.",
        returns="{ corrected, safetyFlags, hasDangers, hasWarnings }")

    add_func("assessConfidence(correctionResult)",
        "Determine accept/verify/reject based on dangers, warnings, unknown words, and average confidence.",
        returns="'accept' | 'verify' | 'reject'")

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # 4. CLINICAL VOCABULARY
    # ════════════════════════════════════════════════════════════
    add_section("4. Clinical Vocabulary (clinical/vocabulary.js)")
    story.append(Paragraph(
        "Organized medical vocabulary: ward structure, diagnoses (neuro, respiratory, cardiac, renal, GI, "
        "hematology, infectious, endocrine), examination terms, medications (200+), labs, and patient names.",
        desc_style))

    add_func("VOCABULARY", "Object with categories: ward_structure, diagnoses, examination, medications, labs, names.")
    add_func("buildWordSet()", "Build a flat Set of all vocabulary words (lowercased, including sub-tokens).", returns="Set<string>")
    add_func("getColumnVocabulary(columnType)",
        "Get context-specific vocabulary subset for a column type (patient, diagnosis, doctor, status, medication, lab, exam).",
        returns="Set<string>")

    # ════════════════════════════════════════════════════════════
    # 5. CONFUSION MODEL
    # ════════════════════════════════════════════════════════════
    add_section("5. Confusion Model (clinical/confusion.js)")
    story.append(Paragraph(
        "Topology-predicted character substitution costs. Lower cost = more likely OCR confusion.",
        desc_style))

    add_func("CONFUSION_PAIRS", "Object mapping character pairs to substitution costs (e.g., 'O,0': 0.1).")
    add_func("getConfusionCost(char1, char2)", "Get the OCR confusion cost between two characters.", returns="number (0.0 - 1.0)")
    add_func("ocrWeightedDistance(str1, str2)", "Confusion-weighted edit distance between two strings.", returns="number")
    add_func("DIGRAPH_CONFUSIONS", "Array of digraph confusion rules (rn->m, cl->d, vv->w, etc.).")
    add_func("fixDigraphs(text)", "Apply digraph corrections to text (e.g., 'rn' -> 'm').", returns="string")

    # ════════════════════════════════════════════════════════════
    # 6. SAFETY FLAGS
    # ════════════════════════════════════════════════════════════
    add_section("6. Safety Flags (clinical/safety.js)")
    story.append(Paragraph(
        "Clinical validation layer. Catches dangerous OCR errors. Design principle: never silently correct, always flag.",
        desc_style))

    add_func("LAB_RANGES", "Object with physiological ranges for 25 lab tests (sodium, potassium, glucose, HbA1c, INR, troponin, etc.).")
    add_func("checkLabRange(labName, rawValue)",
        "Check if a value is within physiological range. Suggests decimal-point alternatives for out-of-range values.",
        returns="{ status, severity?, message, value, alternatives? }")
    add_func("checkMedicationAmbiguity(ocrText, candidates)",
        "Flag when OCR text is ambiguous between two medications with close edit distance.",
        returns="{ status: 'MEDICATION_AMBIGUITY', severity: 'danger', ... } or null")
    add_func("checkDosePlausibility(medName, doseStr)",
        "Check if a dose is physiologically plausible (e.g., >5000mg, insulin >200 units).",
        returns="{ status: 'IMPLAUSIBLE_DOSE', severity: 'warning', ... } or null")
    add_func("runSafetyChecks(words)",
        "Run all safety checks on an array of corrected word objects. Detects medication ambiguity, "
        "out-of-range labs, and implausible doses.",
        returns="Array of safety flags")

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # 7. LEARNING LOOP
    # ════════════════════════════════════════════════════════════
    add_section("7. Learning Loop (learning/loop.js)")
    story.append(Paragraph(
        "Three fluid landscapes that reshape with every nurse correction, integrated with the core resonance engine.",
        desc_style))

    add_class("class AdaptiveConfusionProfile")
    add_func("recordCorrection(ocrText, correctedText, clinicalWeight)",
        "Record a correction event. Uses edit-distance backtrace alignment to extract character pairs. "
        "Clinical weight makes the system learn faster from critical corrections (medications 3x, diagnoses 2.5x).")
    add_func("getCost(char1, char2)",
        "Dynamic confusion cost. Starts with base cost, blends toward empirical cost as data accumulates. "
        "P(confusion) from actual correction data replaces static hardcoded costs.",
        returns="number (0.05 - 1.0)")
    add_func("getCharConfidence(char)",
        "Real confidence for a character based on observed accuracy (correct / total).",
        returns="number (0.0 - 1.0)")
    add_func("weightedDistance(str1, str2)",
        "Edit distance using adaptive confusion costs (dynamic, not static).",
        returns="number")
    add_func("getTopConfusions(n)",
        "Get the most frequently confused character pairs.",
        returns="Array of { pair, count, cost }")

    add_class("class WardVocabulary")
    add_func("confirmWord(word, category)",
        "Confirm a word was correctly read. Increases frequency count and category association.")
    add_func("confirmReading(fields)",
        "Confirm all words in a row of fields. Single vocabulary update path.")
    add_func("getFrequencyBoost(word)",
        "Logarithmic frequency boost for vocabulary matching.",
        returns="number (0.0 - 1.0)")
    add_func("isKnown(word)",
        "Check if a word is in base vocabulary or has been confirmed at least once.",
        returns="boolean")
    add_func("getColumnWords(columnType)",
        "Get words relevant to a column type with base/frequency/boost metadata.",
        returns="Map<string, { base, frequency, boost }>")
    add_func("getAllWords()",
        "Get all known words (base + learned).",
        returns="Set<string>")
    add_func("getLearnedWords(minCount)",
        "Get words learned from corrections (not in base vocabulary).",
        returns="Array of { word, count, categories }")

    add_class("class ContextChains")
    add_func("learn(fields)",
        "Learn co-occurrence patterns between all field pairs in a confirmed row.")
    add_func("getBoost(knownFields, targetColumn, candidateWord)",
        "Get context chain boost for a candidate word given known fields.",
        returns="number (boost value)")
    add_func("predict(knownFields, targetColumn, topK)",
        "Predict likely values for a target column given known fields.",
        returns="Array of { word, count }")

    add_class("class ShifuLearningEngine")
    story.append(Paragraph(
        "Integrates all three landscapes (confusion, vocabulary, context) and bridges to the core engine.",
        desc_style))
    add_func("correctRow(ocrRow, knownContext)",
        "Correct a row using adaptive confusion costs, ward vocabulary, and context chains.",
        returns="{ corrected, safetyFlags, hasDangers, hasWarnings }")
    add_func("learn(ocrRow, confirmedRow, coreEngine)",
        "Learn from a confirmed correction. Updates all three landscapes and feeds confirmed text "
        "into the core resonance engine. Uses clinical-weighted learning rates.")
    add_func("getStats()",
        "Get learning engine statistics.",
        returns="{ totalCorrections, confusionPairs, topConfusions, vocabularySize, learnedWords, contextChains }")
    add_func("toJSON() / fromJSON(data, baseVocabulary)",
        "Serialize/deserialize the full learning engine state.")

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # 8. CLINICAL WEIGHTS
    # ════════════════════════════════════════════════════════════
    add_section("8. Clinical Weights (learning/clinical_weights.js)")
    story.append(Paragraph(
        "Not all errors are equal. Medication errors can kill. Severity-weighted learning rates shape "
        "how aggressively the system learns from different types of corrections.",
        desc_style))

    add_func("CLINICAL_WEIGHTS",
        "Object with column weights (medication: 3.0, diagnosis: 2.5, lab: 2.5, status: 2.0, "
        "doctor: 1.5, patient: 1.5, room: 1.0) and category weights.")
    add_func("getColumnWeight(columnType)",
        "Get the clinical weight config for a column type.",
        returns="{ weight, minConfidence, safetyOverride }")
    add_func("getWordWeight(word, vocabulary)",
        "Get the clinical weight for a word based on vocabulary category.",
        returns="number (1.0 - 3.0)")
    add_func("getLearningRate(columnType, word, vocabulary)",
        "Compute weighted learning rate = max(columnWeight, wordWeight).",
        returns="number (1.0 - 3.0)")
    add_func("requiresSafetyReview(columnType, word, vocabulary)",
        "Check if a correction requires safety re-evaluation.",
        returns="boolean")

    # ════════════════════════════════════════════════════════════
    # 9. MEDICAL CORPUS
    # ════════════════════════════════════════════════════════════
    add_section("9. Medical Corpus (learning/corpus.js)")
    add_func("MEDICAL_CORPUS",
        "Array of 170+ clinical sentences covering neurology, respiratory, cardiac, renal, GI, infectious, "
        "endocrine, hematology, ward census patterns, clinical examination, lab values, and medication prescribing.")
    add_func("seedEngine(engine)",
        "Seed a ShifuEngine with the medical corpus.",
        returns="{ sentences, tokens, corpus_size }")
    add_func("generateWardSentences(count)",
        "Generate synthetic ward census sentences with randomized patient names, diagnoses, doctors, wards.",
        returns="Array of sentences")
    add_func("fullSeed(engine, passes)",
        "Full seeding: 3 passes over medical corpus + 100 synthetic ward sentences.",
        returns="{ totalSentences, totalTokens }")

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # 10. PIPELINE
    # ════════════════════════════════════════════════════════════
    add_section("10. Pipeline (core/pipeline.js)")
    add_class("class ShifuPipeline")
    story.append(Paragraph(
        "End-to-end pipeline: Image to Python OCR to JS correction to safety-checked output. "
        "Bridges Python fluid-theory OCR with JS correction engine.",
        desc_style))

    add_func("processImage(imagePath, options)",
        "Process a single image through the full pipeline. Supports 'line' and 'table' formats.",
        ["imagePath (string)", "options.format — 'line' | 'table'", "options.columns — Column names for table format"],
        "Promise<{ raw, lines/rows, overallDecision }>")
    add_func("processImages(imagePaths, options)",
        "Process multiple images sequentially.",
        returns="Promise<Array of results>")
    add_func("processText(rawText, options)",
        "Process raw OCR text (skip Python step). Useful when OCR is done externally.",
        returns="{ input, corrected, words, safetyFlags, coherence, decision, ... }")
    add_func("processTableRow(row, options)",
        "Process a table row with both standard and adaptive correction, merged with coherence scoring.",
        returns="{ corrected, safetyFlags, decision, hasDangers, hasWarnings }")
    add_func("learnFromCorrection(ocrRow, confirmedRow)",
        "Learn from a confirmed correction (nurse/doctor verified).")

    # ════════════════════════════════════════════════════════════
    # 11. PERSISTENCE
    # ════════════════════════════════════════════════════════════
    add_section("11. Persistence (core/persistence.js)")
    add_class("class ShifuPersistence")
    add_func("save(shifu)", "Save the full Shifu system state (core engine + learning engine + metadata).", returns="savedAt timestamp")
    add_func("load()", "Load saved state from disk.", returns="{ savedCoreState, savedLearningState, meta } or null")
    add_func("hasSavedState()", "Check if saved state exists.", returns="boolean")
    add_func("getMeta()", "Get metadata about saved state (version, timestamps, sizes).", returns="object or null")
    add_func("clear()", "Clear saved state (reset to fresh).")

    add_func("withAutoSave(shifu, options)",
        "Auto-saving wrapper. Wraps a shifu instance to auto-save after every N learn() calls.",
        ["options.stateDir", "options.saveInterval — Save every N corrections (default: 5)"],
        "Wrapped shifu with forceSave() and persistence property")

    # ════════════════════════════════════════════════════════════
    # 12. FEEDBACK LOOP
    # ════════════════════════════════════════════════════════════
    add_section("12. Feedback Loop (core/feedback.js)")
    add_class("class FeedbackLoop")
    story.append(Paragraph(
        "The system's self-awareness layer. predict() -> evaluate() -> learn() -> reshape. "
        "Three feedback channels: character-level, word-level, and line-level. "
        "All feed the metrics system so improvement is measurable.",
        desc_style))

    add_func("propose(ocrText, options)",
        "Generate a correction proposal for review. Does NOT apply — caller must confirm.",
        returns="{ id, timestamp, input, proposed, words[], safetyFlags, coherence, decision }")
    add_func("proposeRow(ocrRow, options)",
        "Generate correction proposals for a table row.",
        returns="{ id, columns, safetyFlags, decision }")
    add_func("evaluate(proposal, confirmed)",
        "Evaluate a proposal against ground truth. THE critical method — closes the feedback loop. "
        "Uses LCS alignment for accurate token matching (handles insertions/deletions). "
        "Automatically calls shifu.learn() to update all landscapes.",
        ["proposal — from propose() or proposeRow()",
         "confirmed (string | object) — ground truth text or row"],
        "{ proposalId, words[], correct, total, accuracy, corrections[], safetyHits[] }")
    add_func("getMetrics()",
        "Get current performance metrics with per-flag breakdown, improvement windows, top errors, and safety stats.",
        returns="{ totalProposals, totalEvaluations, accuracy, byFlag, windows, topErrors, safety }")
    add_func("report()",
        "Generate a human-readable performance report string.",
        returns="string")

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # 13. METRICS TRACKER
    # ════════════════════════════════════════════════════════════
    add_section("13. Metrics Tracker (core/metrics.js)")
    add_class("class MetricsTracker")
    story.append(Paragraph(
        "Tracks improvement over time. Three dimensions: raw accuracy, corrected accuracy, clinical safety.",
        desc_style))

    add_func("record(event)",
        "Record a single correction event with ground truth.",
        ["event.ocrText — Raw OCR output",
         "event.correctedText — Shifu's correction",
         "event.groundTruth — Human-verified correct text",
         "event.meta — Optional { column, source }"],
        "Entry with rawAccuracy, correctedAccuracy, improvement, fixes, regressions")
    add_func("recordWardCensus(ocrRows, correctedRows, groundTruthRows)",
        "Record a full ward census evaluation (all columns, all rows).",
        returns="Array of per-field entries")
    add_func("summary()",
        "Get full metrics summary: overall accuracy, per-column breakdown, recent trend.",
        returns="{ totalWords, rawAccuracy, correctedAccuracy, improvement, byColumn, recentTrend }")
    add_func("report()",
        "Generate a human-readable accuracy report.",
        returns="string")
    add_func("reset()", "Reset all metrics.")

    # ════════════════════════════════════════════════════════════
    # 14. DOCUMENT INGESTOR
    # ════════════════════════════════════════════════════════════
    add_section("14. Document Ingestor (core/ingest.js)")
    add_class("class DocumentIngestor")
    story.append(Paragraph(
        "Processes actual hospital documents: ward sheets, lab reports, PDFs, images, CSV/TSV, plain text. "
        "Each input becomes a standardized document flowing through the correction + feedback pipeline.",
        desc_style))

    add_func("ingestFile(filePath)",
        "Ingest a file with auto-detected format (.csv, .tsv, .txt, .png, .jpg, .pdf, etc.). "
        "Saves processed document as JSON.",
        returns="Promise<{ type, source, filename, ingestedAt, ... }>")
    add_func("ingestDirectory(dirPath, options)",
        "Ingest all supported files in a directory.",
        ["options.extensions — File extensions to include (default: csv, txt, png, jpg, pdf)"],
        "Promise<{ files, documents }>")
    add_func("ingestRawText(text, options)",
        "Ingest raw text directly (e.g., from clipboard or API). Supports line and table formats.",
        returns="Processed document")
    add_func("ingestWardCensus(rows)",
        "Ingest a ward census table (the core use case). Array of row objects.",
        returns="{ type: 'ward_census', rows, summary: { total, accepted, needsVerification, rejected, accuracy } }")

    # ════════════════════════════════════════════════════════════
    # 15. TRAINED MODEL LOADER
    # ════════════════════════════════════════════════════════════
    add_section("15. Trained Model Loader (core/trainedLoader.js)")
    add_func("loadTrainedModel(modelPath)",
        "Load the pre-trained Python model (trained_model.json).",
        returns="Parsed model object or null")
    add_func("extractConfusionKnowledge(model)",
        "Extract confusion pairs and generate training sentences from character landscape statistics. "
        "Includes recorded confusions + topology-predicted confusable pairs + medical context sentences.",
        returns="{ pairs, sentences }")
    add_func("feedTrainedModelToEngine(core, modelPath)",
        "Feed trained model knowledge into a ShifuEngine. Bridges Python training to JS understanding.",
        returns="{ fed, pairs, characters }")

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # 16. PYTHON OCR ENGINE
    # ════════════════════════════════════════════════════════════
    add_section("16. Python OCR Engine (shifu_ocr/engine.py)")
    story.append(Paragraph(
        "Fluid Theory OCR: model the medium, detect displacement, let experience shape the landscape. "
        "No neural network. No GPU. No cloud.",
        desc_style))

    add_subsection("Image Processing Pipeline")
    add_func("estimate_background(img, k=25)", "Estimate background using morphological closing + Gaussian blur.", returns="numpy array")
    add_func("compute_displacement(img, bg)", "Compute normalized displacement field (background - foreground).", returns="numpy array [0,1]")
    add_func("detect_perturbation(disp, thresh=0.25)", "Binary segmentation of displacement field.", returns="numpy uint8 array")
    add_func("extract_region(binary, pad=3)", "Extract the bounding box of a binary character region.", returns="numpy array")
    add_func("normalize_region(region, size=(64,64))", "Resize binary region to standard size.", returns="numpy array")
    add_func("image_to_binary(char_img, bg_kernel, disp_thresh)", "Full pipeline: image to binary + displacement.", returns="(binary, displacement)")

    add_subsection("Feature Extraction")
    add_func("extract_features(binary_region)",
        "Extract 38-dimensional feature vector: topology (components, holes, euler), displacement ratio, "
        "symmetry (v/h), center of mass, quadrant density, projection profiles (h/v x 6), "
        "crossing counts (h/v x 6), skeleton endpoints + junctions.",
        returns="numpy array[38]")

    add_class("class Landscape")
    add_func("absorb(fv)", "Add a feature vector observation. Updates running mean and variance.")
    add_func("fit(fv)", "Score a feature vector against this landscape (Gaussian log-likelihood + frequency prior).", returns="float score")
    add_func("to_dict() / from_dict(d)", "Serialize/deserialize landscape state.")

    add_class("class ShifuOCR")
    add_func("train_character(label, grayscale_image)", "Train on a single character image. Builds feature vector and absorbs into landscape.")
    add_func("train_from_fonts(characters, font_paths, font_size, img_size)",
        "Train on rendered characters across multiple fonts.",
        returns="Number of observations")
    add_func("predict_character(grayscale_image, top_k=5)",
        "Predict a single character. Scores all landscapes, computes confidence from margin.",
        returns="{ predicted, confidence, margin, candidates, features, binary_region }")
    add_func("correct(prediction, true_label)",
        "Learn from a correction. Updates landscape observations and confusion tracking.",
        returns="boolean (was prediction correct?)")
    add_func("segment_characters(grayscale_image, min_char_width=3)",
        "Segment a text line into individual character images using Otsu + vertical projection.",
        returns="Array of { image, bbox }")
    add_func("read_line(grayscale_image, space_threshold)",
        "Read a full line of text: segment, predict each character, detect spaces.",
        returns="{ text, characters[], confidence }")
    add_func("render_text_line(text, font_path, font_size, padding)",
        "Static method: render text as a grayscale image for testing.",
        returns="numpy array")
    add_func("save(path) / load(path)", "Save/load trained model to/from JSON.")
    add_func("get_stats()", "Get model statistics: characters, depth range, accuracy.", returns="dict")

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    # 17. PYTHON CLINICAL POST-PROCESSOR
    # ════════════════════════════════════════════════════════════
    add_section("17. Clinical Post-Processor (shifu_ocr/clinical.py)")
    add_func("levenshtein(s1, s2)", "Standard Levenshtein edit distance.", returns="int")
    add_func("ocr_distance(s1, s2)", "OCR confusion-weighted edit distance (Python version).", returns="float")
    add_func("NEURO_VOCAB", "Medical vocabulary: exam sections, cranial, motor, sensory, reflexes, medications, general.")
    add_func("LAB_RANGES", "Lab test ranges (Python version): 10 common tests.")

    add_class("class ClinicalPostProcessor")
    add_func("process_word(ocr_word, max_dist=2.5)",
        "Process a single OCR word: exact match, number handling (with lab range checking), "
        "fuzzy match with context-aware boosting, medication ambiguity detection.",
        returns="{ input, output, confidence, flag, candidates, context_used }")
    add_func("process_text(ocr_text)",
        "Process a full OCR text string through the clinical pipeline.",
        returns="{ input, output, words[], avg_confidence, flags[] }")

    # ════════════════════════════════════════════════════════════
    # 18. PIPELINE WORKER
    # ════════════════════════════════════════════════════════════
    add_section("18. Pipeline Worker (shifu_ocr/pipeline_worker.py)")
    story.append(Paragraph(
        "Python subprocess called by the JS engine via child_process. "
        "Supports two OCR backends: ShifuOCR (fluid theory) and PaddleOCR (deep learning).",
        desc_style))

    add_func("load_shifu_ocr(model_path)", "Load the fluid-theory OCR engine from a model file.", returns="ShifuOCR instance")
    add_func("load_paddle_ocr()", "Load PaddleOCR if available.", returns="PaddleOCR instance or None")
    add_func("read_image(image_path)", "Read image as grayscale numpy array.", returns="numpy array")
    add_func("ocr_with_shifu(image_path, model_path)",
        "Run fluid-theory OCR on a single image.",
        returns="{ backend, text, lines, confidence, characters }")
    add_func("ocr_with_paddle(image_path)",
        "Run PaddleOCR on an image. Supports full pages and tables. Groups detections into rows by Y-coordinate.",
        returns="{ backend, text, lines, rows, detections, avg_confidence }")
    add_func("ocr_table_with_paddle(image_path, columns)",
        "Extract a table from an image using PaddleOCR with column mapping.",
        returns="{ ..., columns, rows }")
    add_func("main()",
        "CLI entry point. Args: --image, --format (line|table), --model, --backend (auto|shifu|paddle), --columns.",
        note="Auto backend: PaddleOCR for tables, ShifuOCR for single lines.")

    # ── FOOTER ──────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*inch))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT, spaceAfter=12))
    story.append(Paragraph(
        "Shifu OCR v1.7.0 — Total: 100+ functions across 18 modules (JS + Python)<br/>"
        "The system suggests, the clinician decides. Low confidence = loud flag, not silent guess.",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=10, leading=14,
                       textColor=ACCENT, alignment=TA_CENTER, fontName="Helvetica-Bold")
    ))

    doc.build(story)
    print(f"PDF generated: {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
